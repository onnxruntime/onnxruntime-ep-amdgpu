// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "pattern_matcher.h"
#include "fusion_utils.h"

#include <cmath>

namespace dml_ep {

using namespace fusion_utils;

// ---------------------------------------------------------------------------
// Internal recursive matcher.
//
// Optional edges implement "greedy longest match": when a branch is optional,
// we attempt it on a snapshot of the current state and keep whichever outcome
// has the most matched nodes.  Non-optional edges fail the entire match if
// they cannot be followed.
//
// Multiple optional branches at the same node are each tried; the one that
// extends the match furthest wins, ensuring the longest pattern takes priority.
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
