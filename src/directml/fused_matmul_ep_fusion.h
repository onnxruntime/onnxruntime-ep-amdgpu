// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "ep_fusion_manager.h"

namespace dml_ep {

// ---------------------------------------------------------------------------
// FusedMatMulFusionRule
//
// Detects Transpose → MatMul → optional activation sequences:
//   Transpose(A) → MatMul(_, B)  → FusedMatMul(transA)
//   MatMul(A, _) ← Transpose(B) → FusedMatMul(transB)
//   Any of above → Softmax/Relu/... → FusedMatMulActivation
//
// Covers: com.microsoft.FusedMatMul, com.microsoft.FusedMatMulActivation,
//         com.microsoft.dml.DmlFusedMatMul
// ---------------------------------------------------------------------------
class FusedMatMulFusionRule final : public IFusionRule {
public:
    std::string_view AnchorOpType() const override { return "MatMul"; }
    PNode            BuildPattern()  const override;
    bool             MatchesResult(const PatternMatch& m) const override;

    bool ValidateCapture(
        const PatternMatch&                            match,
        const fusion_utils::GraphConnectivity&         gc,
        const OrtApi&                                  ort_api) const override;

    void CapturePreFusionData(
        PatternMatch&                            match,
        const fusion_utils::GraphConnectivity&   gc,
        const OrtApi&                            ort_api) const override;

    OrtNodeComputeInfo* Compile(
        const OrtApi&                                            ort_api,
        const OrtGraph*                                          fused_subgraph,
        const std::unordered_map<std::string, const OrtValue*>&  initializers,
        PluginDmlExecutionProviderImpl*                          provider,
        const PatternMatch&                                      match) const override;
};

inline std::unique_ptr<IFusionRule> MakeFusedMatMulFusionRule() {
    return std::make_unique<FusedMatMulFusionRule>();
}

}  // namespace dml_ep
