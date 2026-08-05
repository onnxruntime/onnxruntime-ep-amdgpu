// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "ep_fusion_manager.h"

namespace dml_ep {

// ---------------------------------------------------------------------------
// BiasGeluFusionRule
//
// Detects the raw BiasGelu expansion (6 primitive ONNX ops) anchored on Erf:
//   Add(input, bias) → Div(√2) → Erf → Add(+1) → Mul → Mul(0.5)
//
// Two variants encoded as optional branches (greedy longest match wins):
//   Pattern 1: upstream sibling Mul(add_out, 0.5) feeds the final Mul
//   Pattern 2: downstream Mul(0.5) follows the final Mul
// ---------------------------------------------------------------------------
class BiasGeluFusionRule final : public IFusionRule {
public:
    std::string_view AnchorOpType() const override { return "Erf"; }
    PNode            BuildPattern()  const override;
    bool             MatchesResult(const PatternMatch& m) const override;

    OrtNodeComputeInfo* Compile(
        const OrtApi&                                            ort_api,
        const OrtGraph*                                          fused_subgraph,
        const std::unordered_map<std::string, const OrtValue*>&  initializers,
        PluginDmlExecutionProviderImpl*                          provider,
        const PatternMatch&                                      match) const override;
};

inline std::unique_ptr<IFusionRule> MakeBiasGeluFusionRule() {
    return std::make_unique<BiasGeluFusionRule>();
}

}  // namespace dml_ep
