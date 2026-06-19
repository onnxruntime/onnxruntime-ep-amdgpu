// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "pattern_matcher.h"
#include "fusion_utils.h"

#include <cmath>

namespace dml_ep {

using namespace fusion_utils;

// ---------------------------------------------------------------------------
// Internal recursive matcher — walks the model graph from an anchor node,
// checking each visited node against its corresponding PNode in the pattern
// tree.
//
// The traversal follows edges in both directions:
//
//   Upstream:   anchor ← producer   (follow input[i] to the node that created it)
//   Downstream: anchor → consumer   (follow output[i] to the node that reads it)
//
// At each node, input constraints (Scalar, SameAs, Predicate, etc.) are
// validated before the node is committed to the match result.
//
// Optional edges implement "greedy longest match": the engine snapshots the
// current state, attempts the optional branch, and keeps whichever outcome
// matches more nodes.  Non-optional edges fail the entire match if they
// cannot be followed.
// ---------------------------------------------------------------------------
static bool MatchNode(
    size_t                                                   node_idx,
    const PNode&                                             pattern,
    const GraphConnectivity&                                 gc,
    const OrtApi&                                            ort_api,
    const std::unordered_map<std::string, const OrtValue*>&  initializers,
    const std::unordered_set<size_t>&                        already_matched,
    PatternMatch&                                            out)
{
    if (node_idx >= gc.node_infos.size()) return false;
    if (already_matched.count(node_idx)) return false;
    for (size_t idx : out.all_nodes) {
        if (idx == node_idx) return false;
    }

    const NodeInfo& ni = gc.node_infos[node_idx];

    if (ni.op_type != pattern.op_type) return false;
    if (!pattern.domain.empty() && ni.domain != pattern.domain) return false;

    std::string capture = pattern.node_capture.empty()
        ? pattern.op_type + "/" + std::to_string(out.all_nodes.size())
        : pattern.node_capture;

    // Pre-fetch OrtValueInfo* for all inputs of this node, but only when the
    // pattern has Predicate constraints that need shape/type metadata.
    // Built once per node and reused across all constraint evaluations.
    //
    // The other constraint kinds (Scalar, SameAs, Capture) only need the
    // input tensor name (available from NodeInfo::input_names), so this
    // fetch is skipped entirely when no predicates are present.
    bool has_predicates = false;
    for (const auto& [_, c] : pattern.input_constraints) {
        if (c.kind == PInput::Kind::Predicate) { has_predicates = true; break; }
    }

    std::unordered_map<std::string, const OrtValueInfo*> input_value_infos;
    if (has_predicates) {
        size_t input_count = 0;
        ort_api.Node_GetNumInputs(ni.node, &input_count);
        std::vector<const OrtValueInfo*> vis(input_count, nullptr);
        if (input_count > 0) ort_api.Node_GetInputs(ni.node, vis.data(), input_count);
        for (size_t i = 0; i < input_count; ++i) {
            if (!vis[i]) continue;
            const char* name = nullptr;
            OrtStatus* st = ort_api.GetValueInfoName(vis[i], &name);
            if (st || !name) { if (st) ort_api.ReleaseStatus(st); continue; }
            input_value_infos[name] = vis[i];
        }
    }

    // Validate per-input constraints.
    for (const auto& [idx, constraint] : pattern.input_constraints) {
        // SIZE_MAX is a sentinel meaning "check all inputs, pass if any matches".
        if (idx == SIZE_MAX) {
            bool any_matched = false;
            for (const auto& val : ni.input_names) {
                if (constraint.kind == PInput::Kind::Scalar) {
                    auto init_it = initializers.find(val);
                    if (init_it == initializers.end()) continue;
                    float v = 0.0f;
                    if (!TryReadScalarFloat(ort_api, init_it->second, v)) continue;
                    if (std::abs(v - constraint.scalar_value) > constraint.scalar_tol) continue;
                    if (!constraint.capture_name.empty()) out.scalar_values[constraint.capture_name] = v;
                    any_matched = true;
                    break;
                } else if (constraint.kind == PInput::Kind::SameAs) {
                    auto it = out.value_names.find(constraint.capture_name);
                    if (it == out.value_names.end()) {
                        // First use — capture whichever input is not an initializer
                        // (i.e. a data tensor rather than a constant).
                        for (const auto& v2 : ni.input_names) {
                            if (initializers.find(v2) == initializers.end()) {
                                out.value_names[constraint.capture_name] = v2;
                                any_matched = true;
                                break;
                            }
                        }
                    } else {
                        // Already captured — check if any input matches.
                        for (const auto& v2 : ni.input_names) {
                            if (v2 == it->second) { any_matched = true; break; }
                        }
                    }
                } else if (constraint.kind == PInput::Kind::Predicate) {
                    auto vi_it = input_value_infos.find(val);
                    const OrtValueInfo* vi = (vi_it != input_value_infos.end()) ? vi_it->second : nullptr;
                    if (constraint.predicate &&
                        constraint.predicate(val, vi, ort_api, initializers, ni.node)) {
                        if (!constraint.capture_name.empty())
                            out.value_names[constraint.capture_name] = val;
                        any_matched = true;
                        break;
                    }
                }
            }
            if (!any_matched) return false;
            continue;
        }

        if (idx >= ni.input_names.size()) return false;
        const std::string& val_name = ni.input_names[idx];

        switch (constraint.kind) {
        case PInput::Kind::Any:
            break;

        case PInput::Kind::Capture:
            out.value_names[constraint.capture_name] = val_name;
            break;

        case PInput::Kind::SameAs: {
            auto it = out.value_names.find(constraint.capture_name);
            if (it == out.value_names.end()) {
                out.value_names[constraint.capture_name] = val_name;
            } else if (it->second != val_name) {
                return false;
            }
            break;
        }

        case PInput::Kind::Scalar: {
            auto init_it = initializers.find(val_name);
            if (init_it == initializers.end()) return false;
            float v = 0.0f;
            if (!TryReadScalarFloat(ort_api, init_it->second, v)) return false;
            if (std::abs(v - constraint.scalar_value) > constraint.scalar_tol) return false;
            if (!constraint.capture_name.empty()) {
                out.scalar_values[constraint.capture_name] = v;
            }
            break;
        }

        case PInput::Kind::Predicate: {
            auto vi_it = input_value_infos.find(val_name);
            const OrtValueInfo* vi = (vi_it != input_value_infos.end()) ? vi_it->second : nullptr;
            if (!constraint.predicate ||
                !constraint.predicate(val_name, vi, ort_api, initializers, ni.node))
                return false;
            if (!constraint.capture_name.empty())
                out.value_names[constraint.capture_name] = val_name;
            break;
        }
        }
    }

    // Commit this node.
    out.node_indices[capture] = node_idx;
    out.all_nodes.push_back(node_idx);

    // Process edges.
    for (const auto& edge : pattern.edges) {
        size_t child_idx = SIZE_MAX;

        if (edge.dir == PEdgeDir::Downstream) {
            if (edge.value_index >= ni.output_names.size()) {
                if (!edge.optional) { out.all_nodes.pop_back(); out.node_indices.erase(capture); return false; }
                continue;
            }
            const std::string& out_val = ni.output_names[edge.value_index];
            if (!gc.HasSingleConsumer(out_val)) {
                if (!edge.optional) { out.all_nodes.pop_back(); out.node_indices.erase(capture); return false; }
                continue;
            }
            child_idx = gc.consumer_map.at(out_val)[0];

        } else {  // Upstream
            if (edge.value_index >= ni.input_names.size()) {
                if (!edge.optional) { out.all_nodes.pop_back(); out.node_indices.erase(capture); return false; }
                continue;
            }
            const std::string& in_val = ni.input_names[edge.value_index];
            // Upstream edges do not require single-consumer: the value connecting
            // a node to its upstream producer may legitimately have multiple consumers
            // (e.g. the bias_add output in BiasGelu feeds both Div and the final Mul).
            // Single-consumer is only enforced on downstream edges to ensure
            // intermediate outputs are not shared with nodes outside the pattern.
            auto prod_it = gc.producer_map.find(in_val);
            if (prod_it == gc.producer_map.end()) {
                if (!edge.optional) { out.all_nodes.pop_back(); out.node_indices.erase(capture); return false; }
                continue;
            }
            child_idx = prod_it->second;
        }

        if (!edge.optional) {
            if (!MatchNode(child_idx, *edge.child, gc, ort_api, initializers,
                           already_matched, out)) {
                out.all_nodes.pop_back();
                out.node_indices.erase(capture);
                return false;
            }
        } else {
            // Optional edge: try it on a snapshot; keep whichever gives the
            // longest match (greedy longest-match strategy).
            PatternMatch snapshot = out;
            if (MatchNode(child_idx, *edge.child, gc, ort_api, initializers,
                          already_matched, snapshot)
                && snapshot.all_nodes.size() > out.all_nodes.size()) {
                out = std::move(snapshot);
            }
        }
    }

    return true;
}

bool MatchPattern(
    size_t                                                   anchor_idx,
    const PNode&                                             pattern,
    const fusion_utils::GraphConnectivity&                   gc,
    const OrtApi&                                            ort_api,
    const std::unordered_map<std::string, const OrtValue*>&  initializers,
    const std::unordered_set<size_t>&                        already_matched,
    PatternMatch&                                            out)
{
    out = PatternMatch{};
    return MatchNode(anchor_idx, pattern, gc, ort_api, initializers,
                     already_matched, out);
}

}  // namespace dml_ep
