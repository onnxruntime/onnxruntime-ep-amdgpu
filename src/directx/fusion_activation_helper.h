// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// ---------------------------------------------------------------------------
// Shared activation infrastructure for Tier-2 EP fusion rules.
//
// Extracted from fused_matmul_ep_fusion.cc to avoid duplication across
// OpActivationFusionRule and FusedMatMulFusionRule.
// ---------------------------------------------------------------------------

#include <DirectML.h>
#include <string>
#include <string_view>
#include <vector>

#include <onnxruntime_c_api.h>

namespace dml_ep {

// ---------------------------------------------------------------------------
// FusedActivationType — activation kinds that can be fused into DML operators.
// Shared by FusedMatMulFusionRule and OpActivationFusionRule.
// ---------------------------------------------------------------------------
enum class FusedActivationType {
    None,
    Relu,
    Sigmoid,
    Tanh,
    LeakyRelu,
    Elu,
    HardSigmoid,
    Selu,
    Softplus,
    Softsign,
    ThresholdedRelu,
    ScaledTanh,
    ParametricSoftplus,
};

inline const char* FusedActivationTypeName(FusedActivationType a) {
    switch (a) {
    case FusedActivationType::None:               return "None";
    case FusedActivationType::Relu:               return "Relu";
    case FusedActivationType::Sigmoid:            return "Sigmoid";
    case FusedActivationType::Tanh:               return "Tanh";
    case FusedActivationType::LeakyRelu:          return "LeakyRelu";
    case FusedActivationType::Elu:                return "Elu";
    case FusedActivationType::HardSigmoid:        return "HardSigmoid";
    case FusedActivationType::Selu:               return "Selu";
    case FusedActivationType::Softplus:           return "Softplus";
    case FusedActivationType::Softsign:           return "Softsign";
    case FusedActivationType::ThresholdedRelu:    return "ThresholdedRelu";
    case FusedActivationType::ScaledTanh:         return "ScaledTanh";
    case FusedActivationType::ParametricSoftplus: return "ParametricSoftplus";
    default:                                      return "Unknown";
    }
}

// Map ONNX activation op type string to enum.
// Returns None if not a recognised fusable activation.
inline FusedActivationType FusedActivationTypeFromOpType(std::string_view op_type) {
    if (op_type == "Relu")               return FusedActivationType::Relu;
    if (op_type == "Sigmoid")            return FusedActivationType::Sigmoid;
    if (op_type == "Tanh")               return FusedActivationType::Tanh;
    if (op_type == "LeakyRelu")          return FusedActivationType::LeakyRelu;
    if (op_type == "Elu")                return FusedActivationType::Elu;
    if (op_type == "HardSigmoid")        return FusedActivationType::HardSigmoid;
    if (op_type == "Selu")               return FusedActivationType::Selu;
    if (op_type == "Softplus")           return FusedActivationType::Softplus;
    if (op_type == "Softsign")           return FusedActivationType::Softsign;
    if (op_type == "ThresholdedRelu")    return FusedActivationType::ThresholdedRelu;
    if (op_type == "ScaledTanh")         return FusedActivationType::ScaledTanh;
    if (op_type == "ParametricSoftplus") return FusedActivationType::ParametricSoftplus;
    return FusedActivationType::None;
}

// ---------------------------------------------------------------------------
// ActivationDescStorage
//
// Holds all DML activation operator desc variants plus the unified
// DML_OPERATOR_DESC pointing into one of them.  Must remain alive for the
// lifetime of any DML operator that references FusedActivation.
// ---------------------------------------------------------------------------
struct ActivationDescStorage {
    DML_ACTIVATION_RELU_OPERATOR_DESC              relu{};
    DML_ACTIVATION_SIGMOID_OPERATOR_DESC           sigmoid{};
    DML_ACTIVATION_TANH_OPERATOR_DESC              tanh{};
    DML_ACTIVATION_LEAKY_RELU_OPERATOR_DESC        leaky{};
    DML_ACTIVATION_ELU_OPERATOR_DESC               elu{};
    DML_ACTIVATION_HARD_SIGMOID_OPERATOR_DESC      hardsig{};
    DML_ACTIVATION_SCALED_ELU_OPERATOR_DESC        selu{};
    DML_ACTIVATION_SOFTPLUS_OPERATOR_DESC          softplus{};
    DML_ACTIVATION_SOFTSIGN_OPERATOR_DESC          softsign{};
    DML_ACTIVATION_THRESHOLDED_RELU_OPERATOR_DESC  threlu{};
    DML_ACTIVATION_SCALED_TANH_OPERATOR_DESC       scaledtanh{};
    DML_ACTIVATION_PARAMETRIC_SOFTPLUS_OPERATOR_DESC paramsoftplus{};

    DML_OPERATOR_DESC desc{};
    bool valid = false;
};

// Build a DML activation descriptor from type + parameters.
// Returns valid=false for FusedActivationType::None or unknown types.
inline ActivationDescStorage BuildActivationDesc(
    FusedActivationType act,
    float act_alpha,
    float act_beta,
    float act_gamma)
{
    ActivationDescStorage s;
    switch (act) {
    case FusedActivationType::Relu:
        s.desc = { DML_OPERATOR_ACTIVATION_RELU, &s.relu };
        s.valid = true;
        break;
    case FusedActivationType::Sigmoid:
        s.desc = { DML_OPERATOR_ACTIVATION_SIGMOID, &s.sigmoid };
        s.valid = true;
        break;
    case FusedActivationType::Tanh:
        s.desc = { DML_OPERATOR_ACTIVATION_TANH, &s.tanh };
        s.valid = true;
        break;
    case FusedActivationType::LeakyRelu:
        s.leaky.Alpha = act_alpha;
        s.desc = { DML_OPERATOR_ACTIVATION_LEAKY_RELU, &s.leaky };
        s.valid = true;
        break;
    case FusedActivationType::Elu:
        s.elu.Alpha = act_alpha;
        s.desc = { DML_OPERATOR_ACTIVATION_ELU, &s.elu };
        s.valid = true;
        break;
    case FusedActivationType::HardSigmoid:
        s.hardsig.Alpha = act_alpha;
        s.hardsig.Beta  = act_beta;
        s.desc = { DML_OPERATOR_ACTIVATION_HARD_SIGMOID, &s.hardsig };
        s.valid = true;
        break;
    case FusedActivationType::Selu:
        // ORT maps ONNX Selu to DML_OPERATOR_ACTIVATION_SCALED_ELU.
        // Alpha = ONNX alpha (default 1.67326319), Gamma = ONNX gamma (default 1.05070102).
        s.selu.Alpha = act_alpha;
        s.selu.Gamma = act_gamma;
        s.desc = { DML_OPERATOR_ACTIVATION_SCALED_ELU, &s.selu };
        s.valid = true;
        break;
    case FusedActivationType::Softplus:
        s.softplus.Steepness = 1.0f;
        s.desc = { DML_OPERATOR_ACTIVATION_SOFTPLUS, &s.softplus };
        s.valid = true;
        break;
    case FusedActivationType::Softsign:
        s.desc = { DML_OPERATOR_ACTIVATION_SOFTSIGN, &s.softsign };
        s.valid = true;
        break;
    case FusedActivationType::ThresholdedRelu:
        s.threlu.Alpha = act_alpha;
        s.desc = { DML_OPERATOR_ACTIVATION_THRESHOLDED_RELU, &s.threlu };
        s.valid = true;
        break;
    case FusedActivationType::ScaledTanh:
        s.scaledtanh.Alpha = act_alpha;
        s.scaledtanh.Beta  = act_beta;
        s.desc = { DML_OPERATOR_ACTIVATION_SCALED_TANH, &s.scaledtanh };
        s.valid = true;
        break;
    case FusedActivationType::ParametricSoftplus:
        s.paramsoftplus.Alpha = act_alpha;
        s.paramsoftplus.Beta  = act_beta;
        s.desc = { DML_OPERATOR_ACTIVATION_PARAMETRIC_SOFTPLUS, &s.paramsoftplus };
        s.valid = true;
        break;
    default:
        s.valid = false;
        break;
    }
    return s;
}

// ---------------------------------------------------------------------------
// ReadRuntimeShape
//
// Read the element type, packed sizes, and byte count from an OrtValue*
// at kernel compute time.  Returns false on unsupported dtype (non fp16/fp32).
// ---------------------------------------------------------------------------
inline bool ReadRuntimeShape(
    const OrtApi&          api,
    const OrtValue*        value,
    DML_TENSOR_DATA_TYPE&  out_dml_dtype,
    std::vector<uint32_t>& out_sizes,
    uint64_t&              out_bytes)
{
    OrtTensorTypeAndShapeInfo* shape_info = nullptr;
    api.GetTensorTypeAndShape(const_cast<OrtValue*>(value), &shape_info);
    if (!shape_info) return false;

    size_t elem_count = 0;
    api.GetTensorShapeElementCount(shape_info, &elem_count);
    ONNXTensorElementDataType dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    api.GetTensorElementType(shape_info, &dtype);
    size_t rank = 0;
    api.GetDimensionsCount(shape_info, &rank);
    std::vector<int64_t> dims(rank);
    if (rank > 0) api.GetDimensions(shape_info, dims.data(), rank);
    api.ReleaseTensorTypeAndShapeInfo(shape_info);

    if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        out_dml_dtype = DML_TENSOR_DATA_TYPE_FLOAT32;
    else if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
        out_dml_dtype = DML_TENSOR_DATA_TYPE_FLOAT16;
    else
        return false;

    size_t elem_size = (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) ? 2 : 4;
    out_sizes.resize(rank);
    for (size_t i = 0; i < rank; ++i)
        // Preserve a genuine 0 dim (empty tensor) so callers can detect it and
        // no-op the way ORT's DML EP does — DML descriptors can't represent a
        // zero-sized dim, so an empty tensor must never reach compile/bind.
        // Only a symbolic/dynamic dim (< 0) is clamped to a 1 placeholder.
        out_sizes[i] = static_cast<uint32_t>(dims[i] >= 0 ? dims[i] : 1);
    // DML requires TotalTensorSizeInBytes to be DWORD-aligned (multiple of 4).
    uint64_t raw = static_cast<uint64_t>(elem_count * elem_size);
    out_bytes = (raw + 3u) & ~uint64_t(3u);
    return true;
}

}  // namespace dml_ep
