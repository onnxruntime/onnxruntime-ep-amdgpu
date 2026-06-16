// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "ep_fusion_manager.h"

#include "quick_gelu_ep_fusion.h"
#include "bias_gelu_ep_fusion.h"

#include "dml_execution_provider.h"  // must precede dml_abi_kernel.h
#include "dml_abi_kernel.h"          // DML_PERF_LOG
#include <algorithm>

namespace dml_ep {

// ---------------------------------------------------------------------------
// MergePatterns
//
// Merges `additional` into `base` so both patterns are encoded in one tree.
// Where edges match (same direction, value_index, child op type), recurse.
// Where `additional` has an edge `base` doesn't, add it as optional=true —
// the greedy engine picks it up when present (longest match wins) and
// silently skips it when absent (shorter match for the base rule).
// ---------------------------------------------------------------------------
static void MergePatterns(PNode& base, PNode& additional) {
    // Merge input constraints — add any from additional not in base.
    for (const auto& [idx, constraint] : additional.input_constraints) {
        bool already_present = false;
        for (const auto& [bidx, _] : base.input_constraints) {
            if (bidx == idx) { already_present = true; break; }
        }
        if (!already_present) {
            base.input_constraints.emplace_back(idx, constraint);
        }
    }

    // Merge edges.
    for (auto& add_edge : additional.edges) {
        bool found = false;
        for (auto& base_edge : base.edges) {
            if (base_edge.dir         == add_edge.dir         &&
                base_edge.value_index == add_edge.value_index &&
                base_edge.child->op_type == add_edge.child->op_type) {
                // Same edge in both trees — recurse into children.
                MergePatterns(*base_edge.child, *add_edge.child);
                found = true;
                break;
            }
        }
        if (!found) {
            // additional has an edge base doesn't — this is a divergence point.
            // Make any base edges at the same (dir, value_index) position optional
            // so the engine tries both branches rather than failing on the base branch.
            for (auto& base_edge : base.edges) {
                if (base_edge.dir == add_edge.dir &&
                    base_edge.value_index == add_edge.value_index) {
                    base_edge.optional = true;
                }
            }
            // Add the additional branch as optional.
            base.edges.emplace_back(
                add_edge.dir,
                add_edge.value_index,
                std::make_unique<PNode>(*add_edge.child),
                /*optional=*/true);
        }
    }
}

// ---------------------------------------------------------------------------
// BuildAnchorIndex
//
// Rules sharing an anchor are merged into one tree via MergePatterns:
//   - Shared edges are recursed into (common prefix evaluated once).
//   - Diverging edges from additional are added as optional branches.
//   - Base edges that diverge from additional are also made optional so the
//     engine tries both branches — the greedy match picks the longest.
//
// Ordering: longest patterns first within each anchor group so the base tree
// is the most complete pattern and shorter variants fill in optional branches.
//
// Capture name rule: if two rules share an anchor, their capture names MUST
// be unique (e.g. prefix with rule name) to prevent PSameAs cross-contamination
// across merged branches.
// ---------------------------------------------------------------------------
EpFusionManager::AnchorIndex EpFusionManager::BuildAnchorIndex() {
    std::vector<std::unique_ptr<IFusionRule>> all_rules;
    all_rules.push_back(MakeQuickGeluFusionRule());
    all_rules.push_back(MakeBiasGeluFusionRule());
    // Future rules: insert shortest pattern first within each anchor group.

    AnchorIndex index;
    for (auto& rule : all_rules) {
        std::string anchor(rule->AnchorOpType());
        auto it = index.find(anchor);
        if (it == index.end()) {
            // First rule for this anchor — becomes the base tree.
            AnchorEntry entry;
            entry.tree = std::make_unique<PNode>(rule->BuildPattern());
            entry.rules.push_back(std::move(rule));
            index.emplace(std::move(anchor), std::move(entry));
        } else {
            // Additional rule — merge its pattern into the existing base tree.
            PNode additional = rule->BuildPattern();
            MergePatterns(*it->second.tree, additional);
            it->second.rules.push_back(std::move(rule));
        }
    }
    return index;
}

// ---------------------------------------------------------------------------
// ApplyFusions — O(nodes + matches)
//
// For each matched group, stores a FusionMatch keyed by the sorted set of
// original graph node IDs.  CompileFusion uses those IDs to look up the
// pre-computed match without re-running pattern detection.
// ---------------------------------------------------------------------------
void EpFusionManager::ApplyFusions(
    const AnchorIndex&                                       index,
    const OrtApi&                                            ort_api,
    const OrtEpApi&                                          ep_api,
    const OrtGraph*                                          graph,
    OrtEpGraphSupportInfo*                                   graph_support_info,
    const std::unordered_map<std::string, const OrtValue*>&  initializers,
    const IsSupportedFn&                                     is_supported_fn,
    const std::unordered_set<size_t>&                        cpu_preferred_node_ids,
    uint32_t                                                 device_data_type_mask,
    std::unordered_set<size_t>&                              fused_node_ids,
    FusionMatchMap&                                          fusion_map)
{
    using namespace fusion_utils;

    const GraphConnectivity gc = BuildGraphConnectivity(ort_api, graph);
    if (gc.node_infos.empty()) return;
    if (index.empty()) return;

    std::unordered_set<size_t> already_consumed;

    for (size_t anchor_idx = 0; anchor_idx < gc.node_infos.size(); ++anchor_idx) {
        if (already_consumed.count(anchor_idx)) continue;

        const std::string& op = gc.node_infos[anchor_idx].op_type;
        auto it = index.find(op);
        if (it == index.end()) continue;

        PatternMatch m;
        if (!MatchPattern(anchor_idx, *it->second.tree, gc, ort_api,
                          initializers, already_consumed, m)) {
            continue;
        }

        // Verify all matched nodes are DML-capable and not CPU-preferred.
        bool all_supported = true;
        for (size_t idx : m.all_nodes) {
            const OrtNode* n = gc.node_infos[idx].node;
            size_t nid = GetNodeId(ort_api, n);
            if (cpu_preferred_node_ids.count(nid) ||
                !is_supported_fn(n, device_data_type_mask)) {
                all_supported = false;
                break;
            }
        }
        if (!all_supported) continue;

        // Collect sorted node IDs and compute hash — used as O(1) lookup key.
        std::vector<size_t> node_ids;
        node_ids.reserve(m.all_nodes.size());
        std::vector<const OrtNode*> nodes_to_fuse;
        nodes_to_fuse.reserve(m.all_nodes.size());
        for (size_t idx : m.all_nodes) {
            size_t nid = GetNodeId(ort_api, gc.node_infos[idx].node);
            node_ids.push_back(nid);
            nodes_to_fuse.push_back(gc.node_infos[idx].node);
        }
        std::sort(node_ids.begin(), node_ids.end());
        const size_t group_hash = HashNodeIds(node_ids);

        OrtNodeFusionOptions fusion_options{ORT_API_VERSION, true};
        OrtStatus* st = ep_api.EpGraphSupportInfo_AddNodesToFuse(
            graph_support_info,
            nodes_to_fuse.data(),
            nodes_to_fuse.size(),
            &fusion_options);
        if (st) { ort_api.ReleaseStatus(st); continue; }

        // Find the owning rule for this match.  Every successful MatchPattern
        // result must map to exactly one rule via MatchesResult — if none is
        // found the match was produced by a tree whose rules are misconfigured.
        const IFusionRule* owning_rule = nullptr;
        for (const auto& rule : it->second.rules) {
            if (rule->MatchesResult(m)) { owning_rule = rule.get(); break; }
        }
        if (!owning_rule) continue;  // guard: skip if no rule claims this match

        FusionMatch fm;
        fm.match = std::move(m);
        fm.rule  = owning_rule;
        fusion_map.emplace(group_hash, std::move(fm));

        for (size_t nid : node_ids) fused_node_ids.insert(nid);
        for (size_t idx : fusion_map.at(group_hash).match.all_nodes) {
            already_consumed.insert(idx);
        }
    }
}

// ---------------------------------------------------------------------------
// CompileFusion — no pattern re-detection
//
// Identifies the pre-computed FusionMatch by comparing the fused subgraph's
// node IDs against the stored groups.  Calls Compile() directly with the
// cached PatternMatch.
// ---------------------------------------------------------------------------
OrtNodeComputeInfo* EpFusionManager::CompileFusion(
    const OrtApi&                                            ort_api,
    const OrtGraph*                                          fused_subgraph,
    const std::unordered_map<std::string, const OrtValue*>&  initializers,
    PluginDmlExecutionProviderImpl*                          provider,
    FusionMatchMap&                                          fusion_map)
{
    using namespace fusion_utils;
    // No BuildAnchorIndex() or MatchPattern() here — the FusionMatchMap was
    // populated by ApplyFusions and contains pre-computed PatternMatch results.
    // CompileFusion only needs to hash the subgraph node IDs for O(1) lookup.

    // Sort subgraph node IDs and hash them — O(k log k) where k is pattern size.
    size_t num_nodes = 0;
    ort_api.Graph_GetNumNodes(fused_subgraph, &num_nodes);
    std::vector<const OrtNode*> subgraph_nodes(num_nodes, nullptr);
    if (num_nodes > 0) ort_api.Graph_GetNodes(fused_subgraph, subgraph_nodes.data(), num_nodes);

    std::vector<size_t> subgraph_ids;
    subgraph_ids.reserve(num_nodes);
    for (const OrtNode* n : subgraph_nodes) {
        if (n) subgraph_ids.push_back(GetNodeId(ort_api, n));
    }
    std::sort(subgraph_ids.begin(), subgraph_ids.end());
    const size_t group_hash = HashNodeIds(subgraph_ids);

    // O(1) lookup in the pre-built map.
    auto it = fusion_map.find(group_hash);
    if (it == fusion_map.end()) {
        DML_PERF_LOG("[CompileFusion] no pre-computed match found for subgraph");
        return nullptr;
    }

    FusionMatch fm = std::move(it->second);
    fusion_map.erase(it);  // consume so each match is used exactly once

    if (!fm.rule) {
        DML_PERF_LOG("[CompileFusion] matched group has no owning rule");
        return nullptr;
    }

    // Build combined initializer map for Compile().
    std::unordered_map<std::string, const OrtValue*> subgraph_initializers = initializers;
    {
        size_t num_init = 0;
        ort_api.Graph_GetNumInitializers(fused_subgraph, &num_init);
        if (num_init > 0) {
            std::vector<const OrtValueInfo*> init_vis(num_init, nullptr);
            ort_api.Graph_GetInitializers(fused_subgraph, init_vis.data(), num_init);
            for (const OrtValueInfo* vi : init_vis) {
                if (!vi) continue;
                const char* name = nullptr;
                OrtStatus* st = ort_api.GetValueInfoName(vi, &name);
                if (st || !name) { if (st) ort_api.ReleaseStatus(st); continue; }
                const OrtValue* val = nullptr;
                st = ort_api.ValueInfo_GetInitializerValue(vi, &val);
                if (st) { ort_api.ReleaseStatus(st); continue; }
                if (val) subgraph_initializers[name] = val;
            }
        }
    }

    return fm.rule->Compile(ort_api, fused_subgraph,
                            subgraph_initializers, provider, fm.match);
}

}  // namespace dml_ep
