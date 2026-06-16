// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// ---------------------------------------------------------------------------
// Declarative graph pattern matching engine for EP-level fusions.
//
// Usage:
//   auto pattern = PNode("Sigmoid")
//       .Upstream(0, PNode("Mul")
//           .Input(0, PAnyOf("x"))          // capture as "x"
//           .Input(1, PScalar(/* alpha */))) // scalar constant
//       .Downstream(0, PNode("Mul")
//           .Input(0, PSameAs("x"))         // must equal captured "x"
//           .Input(1, PAny()));
//
//   PatternMatch m;
//   if (MatchPattern(anchor_idx, pattern, gc, initializers, already_matched, m)) {
//       size_t sigmoid_idx = m.NodeIdx("Sigmoid/0");
//       float  alpha       = m.ScalarValue("alpha");
//   }
//
// Design:
//   - Each PNode describes one graph node by op type, with constraints on
//     its inputs and edges to follow upstream/downstream.
//   - The engine does recursive descent, checking single-consumer invariants
//     on every traversed edge automatically.
//   - Longest pattern wins: caller sorts patterns by descending node count
//     and tries them in order; already_matched prevents re-use of nodes.
//   - Patterns are value objects — copy freely, no heap allocation at match time.
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
// Input constraint — describes what a node input must satisfy.
// ---------------------------------------------------------------------------
struct PInput {
    enum class Kind {
        Any,        // no constraint
        Scalar,     // must be a scalar initializer with value near `scalar_value`
        SameAs,     // must equal the value name captured under `capture_name`
        Capture,    // no value constraint; captures the value name
    };

    Kind        kind         = Kind::Any;
    std::string capture_name;          // for Capture and SameAs
    float       scalar_value = 0.0f;   // for Scalar
    float       scalar_tol   = 1e-4f;  // for Scalar
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
