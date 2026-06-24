// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// ---------------------------------------------------------------------------
// Declarative graph pattern matching engine for execution provider fusions.
//
// Describes fusable subgraph patterns as PNode trees, then searches the model
// graph for matching subgraphs.  Each PNode specifies an operation type,
// input constraints, and edges to follow upstream (toward producers) or
// downstream (toward consumers).
//
// Example — detecting a QuickGelu pattern:
//
//       input
//         |
//     Mul(alpha)       upstream of Sigmoid
//         |
//      Sigmoid  <---   anchor node (pattern starts here)
//         |
//     Mul(input)       downstream of Sigmoid
//         |
//       output
//
//     auto pattern = PNode("Sigmoid")
//         .Input(0, PCapture("x"))
//         .Upstream(0, PNode("Mul").Input(1, PScalar(1.702f)))
//         .Downstream(0, PNode("Mul").Input(0, PSameAs("x")));
//
// MatchPattern() does a single recursive descent from the anchor node,
// validating all constraints inline (including Predicate lambdas for
// semantic checks like tensor rank).  On success it returns the indices of
// matched nodes, captured tensor names, and scalar values.
//
// Key design properties:
//   - Single-consumer invariant enforced automatically on downstream edges.
//   - Greedy longest match: optional branches are tried on a snapshot and
//     the longest result wins.
//   - Rules sharing the same anchor are merged into one PNode tree by
//     EpFusionManager so shared prefixes are evaluated once.
// ---------------------------------------------------------------------------

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cmath>

#include <onnxruntime_c_api.h>

#include "fusion_utils.h"

namespace dml_ep {

// ---------------------------------------------------------------------------
// Input constraint — describes what a node's input must satisfy.
//
// Constraints are attached to PNode via .Input(index, constraint).  When
// index is SIZE_MAX, the engine scans all inputs and passes if any matches.
//
// The five kinds form a progression of expressiveness:
//   Any       — accept anything (no check)
//   Capture   — accept anything, but record the tensor name for later lookup
//   SameAs    — must be the same tensor as a previously captured name
//   Scalar    — must be a constant initializer with a specific numeric value
//   Predicate — caller-provided lambda for checks the other kinds can't express
//               (e.g., "must be an initializer with rank <= 1")
// ---------------------------------------------------------------------------
struct PInput {
    enum class Kind { Any, Scalar, SameAs, Capture, Predicate };

    // Predicate callback signature.  Receives:
    //   value_name   — name of the input tensor being checked
    //   value_info   — shape/type metadata for this input (may be null)
    //   ort_api      — ORT C API handle for querying tensor properties
    //   initializers — map of constant tensor names to their values
    //   node         — the graph node owning this input, for cross-input checks
    //                  (e.g., verifying last-dim match between bias and activation)
    // Return true to accept, false to reject.
    using PredicateFn = std::function<bool(
        const std::string&                                       value_name,
        const OrtValueInfo*                                      value_info,
        const OrtApi&                                            ort_api,
        const std::unordered_map<std::string, const OrtValue*>&  initializers,
        const OrtNode*                                           node)>;

    Kind        kind         = Kind::Any;
    std::string capture_name;          // Capture/SameAs/Predicate: key in PatternMatch::value_names
    float       scalar_value = 0.0f;   // Scalar: expected constant value
    float       scalar_tol   = 1e-4f;  // Scalar: absolute tolerance for comparison
    PredicateFn predicate;             // Predicate: the lambda to evaluate
};

// Convenience constructors
inline PInput PAny()                               { return {PInput::Kind::Any}; }
inline PInput PCapture(std::string name)           { return {PInput::Kind::Capture, std::move(name)}; }
inline PInput PSameAs(std::string name)            { return {PInput::Kind::SameAs, std::move(name)}; }
inline PInput PScalar(float v, float tol = 1e-4f)                        { return {PInput::Kind::Scalar, {}, v, tol}; }
inline PInput PScalarAs(std::string name, float v, float tol = 1e-4f)    { return {PInput::Kind::Scalar, std::move(name), v, tol}; }

// PScalarAtAny — succeeds if ANY input of the node is a scalar near `v`.
// Register with index SIZE_MAX; the engine scans all inputs.
inline PInput PScalarAtAny(float v, float tol = 1e-4f) { return {PInput::Kind::Scalar, {}, v, tol}; }

// PSameAsAtAny — succeeds if ANY input matches the previously captured value name.
// Register with index SIZE_MAX; the engine scans all inputs.
inline PInput PSameAsAtAny(std::string name) { return {PInput::Kind::SameAs, std::move(name)}; }

// Convenience: capture the value name AND require it matches an existing capture.
// Used when the same tensor must appear in two places.
inline PInput PSameAsOrCapture(std::string name)   { return {PInput::Kind::SameAs, std::move(name)}; }

// ---------------------------------------------------------------------------
// Predicate constraints — lambda-based checks for conditions that the
// built-in constraint kinds (Scalar, SameAs, etc.) cannot express.
//
// The predicate receives full context about the input being checked:
//   - value_name:   tensor name (for initializer map lookup)
//   - value_info:   OrtValueInfo* with shape/type metadata (may be null)
//   - ort_api:      ORT C API handle for querying value_info properties
//   - initializers: map of constant tensor names to their OrtValue* data
//
// PPredicate       — check a specific input slot (fixed index)
// PPredicateAtAny  — scan all inputs, accept first that passes (SIZE_MAX)
//
// PInitializerWithMaxRank is a concrete predicate that checks:
//   1. The tensor is a constant (exists in the initializer map), AND
//   2. Its number of dimensions (rank) is at most max_rank
// This is used to validate that a bias weight is 1-dimensional during
// pattern matching, rather than deferring the check to compile time.
//
// Example:
//   .Input(SIZE_MAX, PInitializerWithMaxRank(1, "BiasGelu.bias_name"))
//
//   This scans all inputs of the node.  For each input, it checks if the
//   tensor is a constant with rank <= 1.  On the first match, it captures
//   the tensor's name as "BiasGelu.bias_name" in PatternMatch::value_names
//   so Compile() can retrieve it without re-scanning the graph.
// ---------------------------------------------------------------------------
inline PInput PPredicate(PInput::PredicateFn fn) {
    PInput p;
    p.kind      = PInput::Kind::Predicate;
    p.predicate = std::move(fn);
    return p;
}

inline PInput PPredicateAtAny(PInput::PredicateFn fn) {
    return PPredicate(std::move(fn));
}

inline PInput PInitializerWithMaxRank(size_t max_rank, std::string capture_name = {}) {
    PInput p = PPredicate(
        [max_rank](const std::string& name,
                   const OrtValueInfo* vi,
                   const OrtApi& ort_api,
                   const std::unordered_map<std::string, const OrtValue*>& initializers,
                   const OrtNode* /*node*/) -> bool {
            if (!initializers.count(name)) return false;
            if (!vi) return true;  // rank unknown — allow through
            const OrtTypeInfo* ti = nullptr;
            OrtStatus* st = ort_api.GetValueInfoTypeInfo(vi, &ti);
            if (st || !ti) { if (st) ort_api.ReleaseStatus(st); return true; }
            const OrtTensorTypeAndShapeInfo* si = nullptr;
            ort_api.CastTypeInfoToTensorInfo(ti, &si);
            if (!si) return true;
            size_t rank = 0;
            ort_api.GetDimensionsCount(si, &rank);
            // rank=0 means shape info unavailable from the EP C API.
            // For scalar constraints (max_rank=0) reject unknown rank since we
            // cannot confirm the initializer is truly scalar.  For non-scalar
            // constraints allow unknown rank as a permissive fallback.
            if (rank == 0) return max_rank > 0;
            return rank <= max_rank;
        });
    p.capture_name = std::move(capture_name);
    return p;
}

// ---------------------------------------------------------------------------
// Edge direction for traversal
// ---------------------------------------------------------------------------
enum class PEdgeDir { Upstream, Downstream };

// ---------------------------------------------------------------------------
// PNode — describes one node in the pattern tree.
// ---------------------------------------------------------------------------
struct PNode {
    std::string op_type;
    std::string domain;       // empty = any domain (ai.onnx or empty both OK)

    // Per-input constraints. Multiple constraints can share the same index
    // (including SIZE_MAX for "any input" checks).
    std::vector<std::pair<size_t, PInput>> input_constraints;

    // A capture name for this node's index in the match result.
    // Defaults to "<op_type>/<occurrence_count>" if empty.
    std::string node_capture;

    // Edges to follow from this node.
    // child is heap-allocated because PNode is a recursive type.
    struct Edge {
        PEdgeDir                 dir;
        size_t                   value_index;  // output index (Downstream) or input index (Upstream)
        std::unique_ptr<PNode>   child;
        bool                     optional = false;

        Edge(PEdgeDir d, size_t vi, std::unique_ptr<PNode> c, bool opt)
            : dir(d), value_index(vi), child(std::move(c)), optional(opt) {}

        Edge(Edge&&) = default;
        Edge& operator=(Edge&&) = default;
        Edge(const Edge&) = delete;
        Edge& operator=(const Edge&) = delete;
    };
    std::vector<Edge> edges;

    explicit PNode(std::string op, std::string dom = {})
        : op_type(std::move(op)), domain(std::move(dom)) {}

    // Deep copy — required so builder chains can pass PNode by value.
    PNode(const PNode& o)
        : op_type(o.op_type), domain(o.domain),
          input_constraints(o.input_constraints),
          node_capture(o.node_capture)
    {
        for (const auto& e : o.edges) {
            edges.emplace_back(e.dir, e.value_index,
                               std::make_unique<PNode>(*e.child), e.optional);
        }
    }

    PNode(PNode&&) = default;
    PNode& operator=(PNode&&) = default;
    PNode& operator=(const PNode&) = delete;

    // Builder methods return *this by reference for chaining.
    PNode& Input(size_t idx, PInput constraint) {
        input_constraints.emplace_back(idx, std::move(constraint));
        return *this;
    }
    PNode& As(std::string name) {
        node_capture = std::move(name);
        return *this;
    }
    PNode& Downstream(size_t output_idx, PNode child, bool optional = false) {
        edges.emplace_back(PEdgeDir::Downstream, output_idx,
                           std::make_unique<PNode>(std::move(child)), optional);
        return *this;
    }
    PNode& Upstream(size_t input_idx, PNode child, bool optional = false) {
        edges.emplace_back(PEdgeDir::Upstream, input_idx,
                           std::make_unique<PNode>(std::move(child)), optional);
        return *this;
    }
};

// ---------------------------------------------------------------------------
// PatternMatch — result of a successful match.
// ---------------------------------------------------------------------------
struct PatternMatch {
    // Node indices for each captured node (capture name → gc.node_infos index).
    std::unordered_map<std::string, size_t> node_indices;
    // Captured value names (PCapture).
    std::unordered_map<std::string, std::string> value_names;
    // Scalar values read from initializers during match.
    std::unordered_map<std::string, float> scalar_values;
    // All matched node indices (for submission to AddNodesToFuse).
    std::vector<size_t> all_nodes;

    size_t NodeIdx(const std::string& capture) const {
        auto it = node_indices.find(capture);
        return (it != node_indices.end()) ? it->second : SIZE_MAX;
    }
    std::string ValueName(const std::string& capture) const {
        auto it = value_names.find(capture);
        return (it != value_names.end()) ? it->second : std::string{};
    }
    float ScalarValue(const std::string& capture, float default_val = 0.0f) const {
        auto it = scalar_values.find(capture);
        return (it != scalar_values.end()) ? it->second : default_val;
    }
};

// ---------------------------------------------------------------------------
// MatchPattern — attempt to match `pattern` rooted at `anchor_idx`.
//
// Returns true and populates `out` on success.
// Returns false if any constraint is not satisfied.
//
// Single-consumer invariant: every edge the engine traverses (upstream or
// downstream) is automatically checked for single-consumer — ensuring the
// nodes can be safely fused without breaking other graph paths.
// ---------------------------------------------------------------------------
bool MatchPattern(
    size_t                                                   anchor_idx,
    const PNode&                                             pattern,
    const fusion_utils::GraphConnectivity&                   gc,
    const OrtApi&                                            ort_api,
    const std::unordered_map<std::string, const OrtValue*>&  initializers,
    const std::unordered_set<size_t>&                        already_matched,
    PatternMatch&                                            out);

}  // namespace dml_ep
