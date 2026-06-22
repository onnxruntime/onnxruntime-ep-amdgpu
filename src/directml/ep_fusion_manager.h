// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <onnxruntime_c_api.h>

#include "fusion_utils.h"
#include "pattern_matcher.h"

// Forward declarations
struct OrtEpApi;
struct OrtEpGraphSupportInfo;
struct OrtNodeComputeInfo;

namespace dml_ep {

class PluginDmlExecutionProviderImpl;

// Callback used to check whether an individual node is supported by DML.
using IsSupportedFn = std::function<bool(const OrtNode*, uint32_t /*deviceDataTypeMask*/)>;

// ---------------------------------------------------------------------------
// IFusionRule
//
// Represents a single graph fusion pattern.  Each rule declares:
//   - AnchorOpType()  — the op type to anchor on (e.g. "Sigmoid", "Erf")
//   - BuildPattern()  — the PNode tree rooted at that anchor
//   - MatchesResult() — whether a given PatternMatch belongs to this rule
//   - Compile()       — compile the fused subgraph into an OrtNodeComputeInfo
//
// Rules sharing the same AnchorOpType are merged by EpFusionManager into one
// PNode tree with optional branches so the engine evaluates shared prefixes
// once and picks the longest match automatically.
//
// To add a new fusion:
//   1. Implement IFusionRule in its own header/source pair.
//   2. Include the header in ep_fusion_manager.cc.
//   3. Add the rule to EpFusionManager::BuildAnchorIndex().
// ---------------------------------------------------------------------------
struct IFusionRule {
    virtual ~IFusionRule() = default;

    // The op type this rule anchors on.  All nodes of this type in the graph
    // will be tested against this rule's pattern.
    virtual std::string_view AnchorOpType() const = 0;

    // Build the pattern tree for this rule.  Called once per session to
    // populate the anchor index — no per-node allocation.
    virtual PNode BuildPattern() const = 0;

    // Returns true if the given PatternMatch result belongs to this rule.
    // Used by CompileFusion to route to the correct Compile() when multiple
    // rules share the same anchor.
    virtual bool MatchesResult(const PatternMatch& m) const = 0;

    // Compile the fused subgraph.  `match` carries the capture results from
    // the pattern engine so Compile() can read alpha, bias names, etc.
    // without re-running pattern detection.
    virtual OrtNodeComputeInfo* Compile(
        const OrtApi&                                            ort_api,
        const OrtGraph*                                          fused_subgraph,
        const std::unordered_map<std::string, const OrtValue*>&  initializers,
        PluginDmlExecutionProviderImpl*                          provider,
        const PatternMatch&                                      match) const = 0;
};

// ---------------------------------------------------------------------------
// EpFusionManager
//
// Builds an anchor-indexed pattern tree map at session init and uses it for
// O(nodes + matches) graph traversal — one MatchPattern call per anchor node
// instead of one per (node × rule).
//
// ApplyFusions — called from GetCapabilityImpl.
// CompileFusion — called from CompileImpl for each fused subgraph.
// ---------------------------------------------------------------------------
// Stores the pattern match result and owning rule for one fused group.
// Produced by ApplyFusions and consumed by CompileFusion — eliminates
// re-detection in CompileFusion.
struct FusionMatch {
    PatternMatch          match;
    const IFusionRule*    rule = nullptr;   // non-owning; owned by AnchorIndex
};

// Hash a sorted sequence of node IDs into a single size_t.
// Nodes must be sorted before calling so the same logical group always
// produces the same hash regardless of traversal order.
inline size_t HashNodeIds(const std::vector<size_t>& sorted_ids) {
    size_t seed = sorted_ids.size();
    for (size_t id : sorted_ids) {
        seed ^= id + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

// Map from hashed node ID set → FusionMatch.
// Built in ApplyFusions; entries erased one-by-one as CompileFusion consumes them.
using FusionMatchMap = std::unordered_map<size_t, FusionMatch>;

class EpFusionManager {
public:
    // Types must be declared before the methods that use them.

    // One entry per anchor op type: the merged PNode tree and the rule list
    // for Compile() dispatch.  Stored on ExecutionProviderPlugin so raw
    // IFusionRule* pointers in FusionMatch remain valid for the session.
    struct AnchorEntry {
        std::unique_ptr<PNode>                         tree;
        std::vector<std::unique_ptr<IFusionRule>>      rules;
    };
    using AnchorIndex = std::unordered_map<std::string, AnchorEntry>;

    // Build once per session and store on ExecutionProviderPlugin — the rules
    // it owns are referenced by raw pointer in FusionMatch and must outlive it.
    static AnchorIndex BuildAnchorIndex();

    // ApplyFusions detects patterns using the pre-built index and stores one
    // FusionMatch per fused group in fusion_map.
    static void ApplyFusions(
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
        FusionMatchMap&                                          fusion_map);

    // CompileFusion does an O(1) hash lookup — no pattern re-detection.
    static OrtNodeComputeInfo* CompileFusion(
        const OrtApi&                                            ort_api,
        const OrtGraph*                                          fused_subgraph,
        const std::unordered_map<std::string, const OrtValue*>&  initializers,
        PluginDmlExecutionProviderImpl*                          provider,
        FusionMatchMap&                                          fusion_map);

private:
};

}  // namespace dml_ep
