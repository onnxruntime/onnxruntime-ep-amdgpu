// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "ep_fusion_manager.h"

namespace dml_ep {

// ---------------------------------------------------------------------------
// QuickGeluFusionRule
//
// Pattern (anchors on "Sigmoid"):
//   3-node: Mul(x, alpha) → Sigmoid → Mul(x, sigmoid_out)
//   2-node:               Sigmoid(x) → Mul(x, sigmoid_out)
//
// The 3-node variant is encoded as an optional upstream branch so the engine
// picks it greedily when present, falling back to 2-node otherwise.
// ---------------------------------------------------------------------------
class QuickGeluFusionRule final : public IFusionRule {
public:
    std::string_view AnchorOpType() const override { return "Sigmoid"; }
    PNode            BuildPattern()  const override;
    bool             MatchesResult(const PatternMatch& m) const override;

    OrtNodeComputeInfo* Compile(
        const OrtApi&                                            ort_api,
        const OrtGraph*                                          fused_subgraph,
        const std::unordered_map<std::string, const OrtValue*>&  initializers,
        PluginDmlExecutionProviderImpl*                          provider,
        const PatternMatch&                                      match) const override;
};

inline std::unique_ptr<IFusionRule> MakeQuickGeluFusionRule() {
    return std::make_unique<QuickGeluFusionRule>();
}

}  // namespace dml_ep
