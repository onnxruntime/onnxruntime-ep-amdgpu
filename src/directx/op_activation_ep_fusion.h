// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ep_fusion_manager.h"
#include "fusion_activation_helper.h"

namespace dml_ep {

// ---------------------------------------------------------------------------
// OpActivationConfig
//
// Per-base-op configuration for OpActivationFusionRule.  A single rule class
// is instantiated once per base op type, parameterised by this struct.
// ---------------------------------------------------------------------------
struct OpActivationConfig {
    std::string              base_op_type;        // "Conv", "Add", "Gemm", etc.
    std::string              fused_op_type;       // "DmlFusedConv", etc. (informational)
    std::vector<std::string> allowed_activations; // op type strings; filtered for device at ctor time
    std::optional<uint32_t>  input_count_filter;  // if set, base op must have exactly this many inputs
};

// ---------------------------------------------------------------------------
// OpActivationFusionRule
//
// Generic Tier-2 rule for BaseOp → Activation 2-node fusions.
//
// Pattern (anchored on base_op_type, e.g. "Add"):
//   BaseOp(inputs...) → [Relu | LeakyRelu | Sigmoid | ...]   (activation optional)
//
// The activation branch is downstream from output[0] of the base op with
// the single-consumer invariant enforced automatically by the pattern engine.
//
// Implemented: Add, Sum (2-input), Gemm, Conv, ConvTranspose,
// BatchNormalization, InstanceNormalization, MeanVarianceNormalization.
// Norm ops are gated off on MCDM devices.
// ---------------------------------------------------------------------------
class OpActivationFusionRule final : public IFusionRule {
public:
    explicit OpActivationFusionRule(OpActivationConfig config);

    std::string_view AnchorOpType() const override { return m_config.base_op_type; }
    PNode            BuildPattern()  const override;
    bool             MatchesResult(const PatternMatch& m) const override;

    void CapturePreFusionData(
        PatternMatch&                            match,
        const fusion_utils::GraphConnectivity&   gc,
        const OrtApi&                            ort_api) const override;

    bool ValidateCapture(
        const PatternMatch&                            match,
        const fusion_utils::GraphConnectivity&         gc,
        const OrtApi&                                  ort_api) const override;

    OrtNodeComputeInfo* Compile(
        const OrtApi&                                            ort_api,
        const OrtGraph*                                          fused_subgraph,
        const std::unordered_map<std::string, const OrtValue*>&  initializers,
        PluginDmlExecutionProviderImpl*                          provider,
        const PatternMatch&                                      match) const override;

private:
    OpActivationConfig m_config;

    // Capture key prefix — unique per base op type to avoid conflicts when
    // multiple rules share the same anchor in the merged pattern tree.
    std::string CaptureKey(const std::string& suffix) const {
        return "OpAct_" + m_config.base_op_type + "." + suffix;
    }
};

// ---------------------------------------------------------------------------
// Factory functions
//
// MakeAllOpActivationFusionRules returns rules for Add, Sum, Gemm, Conv,
// ConvTranspose, BatchNorm, InstanceNorm, and MVN, with the allowed
// activation list filtered for the current device type. Norm ops are
// gated off on MCDM devices.
// ---------------------------------------------------------------------------
std::vector<std::unique_ptr<IFusionRule>> MakeAllOpActivationFusionRules(bool isMcdmDevice);

}  // namespace dml_ep
