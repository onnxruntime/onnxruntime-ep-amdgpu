// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ---------------------------------------------------------------------------
// OpActivationFusionRule — Tier-2 EP-level fusion for BaseOp + Activation.
//
// MVP covers Add, Sum (2-input), and Gemm.
//
// Pattern (anchored on the base op, looking downstream for an optional
// single-input activation):
//
//   BaseOp(inputs...) ─── output[0] ──▶ [Relu | LeakyRelu | Sigmoid | ...]
//
// The pattern engine enforces the single-consumer invariant on the downstream
// edge automatically.  CapturePreFusionData reads the activation's attributes.
// Compile builds the corresponding DML operator with FusedActivation set.
//
// Naming convention for pattern captures:
//   "OpAct_<BaseOpType>.act.<ActOpType>"  — the matched activation node
//   "OpAct_<BaseOpType>.act_alpha"        — scalar alpha captured from activation attrs
//   "OpAct_<BaseOpType>.act_beta"         — scalar beta
//   "OpAct_<BaseOpType>.act_gamma"        — scalar gamma (Selu)
//   "OpAct_<BaseOpType>.act_type"         — activation op type string (in value_names)
// ---------------------------------------------------------------------------

#include "op_activation_ep_fusion.h"

#include "fusion_utils.h"
#include "pattern_matcher.h"
#include "ort_node_adapter.h"
#include "dml_execution_provider.h"
#include "DmlExecutionProvider/IExecutionProvider.h"
#include "dml_abi_kernel.h"  // DML_PERF_LOG

#include <DirectML.h>
#include <wrl/client.h>
#include <gsl/gsl>
#include <algorithm>
#include <mutex>
#include <numeric>
#include <optional>
#include <string_view>
#include <vector>

namespace dml_ep {

using namespace fusion_utils;
using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// Full list of activation op type names the rule can match downstream.
// Dropout is intentionally omitted (TryGetFusedActivationDesc returns nullopt).
// PRelu is omitted: it has 2 inputs and ORT's single-input constraint rejects it.
// ---------------------------------------------------------------------------
static const std::vector<std::string> kAllActivations = {
    "Relu", "Sigmoid", "Tanh", "LeakyRelu", "Elu",
    "HardSigmoid", "Selu", "Softplus", "Softsign",
    "ThresholdedRelu", "ScaledTanh", "ParametricSoftplus",
};

static const std::vector<std::string> kMcdmActivations = {
    "Relu", "LeakyRelu",
};

// ---------------------------------------------------------------------------
// GemmShapes — shape/stride helper for DML GEMM, mirroring ORT's logic.
// Duplicated from fused_matmul_ep_fusion.cc (file-local there).
// ---------------------------------------------------------------------------
struct GemmShapes {
    std::vector<uint32_t> a_sizes;
    std::vector<uint32_t> a_strides;
    std::vector<uint32_t> b_sizes;
    std::vector<uint32_t> b_strides;
    std::vector<uint32_t> c_sizes;
    bool                  valid = false;
};

static GemmShapes PrepareGemmShapes(
    const std::vector<uint32_t>& a_sizes_in,
    const std::vector<uint32_t>& b_sizes_in,
    bool trans_a,
    bool trans_b)
{
    GemmShapes g;

    auto packed_strides = [](const std::vector<uint32_t>& sizes) {
        std::vector<uint32_t> s(sizes.size());
        uint32_t stride = 1;
        for (int i = static_cast<int>(sizes.size()) - 1; i >= 0; --i) {
            s[i] = stride;
            stride *= sizes[i];
        }
        return s;
    };

    g.a_sizes   = a_sizes_in;
    g.b_sizes   = b_sizes_in;
    g.a_strides = packed_strides(g.a_sizes);
    g.b_strides = packed_strides(g.b_sizes);

    // Rank-1 expansion (row/col vector treatment)
    if (g.a_sizes.size() == 1) {
        g.a_sizes.insert(g.a_sizes.begin(), 1u);
        g.a_strides.insert(g.a_strides.begin(), 0u);
    }
    if (g.b_sizes.size() == 1) {
        g.b_sizes.push_back(1u);
        g.b_strides.push_back(0u);
    }
    if (g.a_sizes.size() < 2 || g.b_sizes.size() < 2) return g;

    // Equalise ranks
    size_t broadcast_rank = std::max(g.a_sizes.size(), g.b_sizes.size());
    while (g.a_sizes.size() < broadcast_rank) {
        g.a_sizes.insert(g.a_sizes.begin(), 1u);
        g.a_strides.insert(g.a_strides.begin(), 0u);
    }
    while (g.b_sizes.size() < broadcast_rank) {
        g.b_sizes.insert(g.b_sizes.begin(), 1u);
        g.b_strides.insert(g.b_strides.begin(), 0u);
    }

    // Broadcast batch dims
    for (size_t d = 0; d + 2 < broadcast_rank; ++d) {
        uint32_t da = g.a_sizes[d], db = g.b_sizes[d];
        if (da == db) continue;
        uint32_t bd = std::max(da, db);
        if (da == 1) { g.a_sizes[d] = bd; g.a_strides[d] = 0; }
        if (db == 1) { g.b_sizes[d] = bd; g.b_strides[d] = 0; }
    }

    // Pad to 4D minimum
    while (g.a_sizes.size() < 4) { g.a_sizes.insert(g.a_sizes.begin(), 1u); g.a_strides.insert(g.a_strides.begin(), 0u); }
    while (g.b_sizes.size() < 4) { g.b_sizes.insert(g.b_sizes.begin(), 1u); g.b_strides.insert(g.b_strides.begin(), 0u); }

    uint32_t M        = trans_a ? g.a_sizes[3] : g.a_sizes[2];
    uint32_t K_from_A = trans_a ? g.a_sizes[2] : g.a_sizes[3];
    uint32_t K_from_B = trans_b ? g.b_sizes[3] : g.b_sizes[2];
    uint32_t N        = trans_b ? g.b_sizes[2] : g.b_sizes[3];
    if (K_from_A != K_from_B) return g;

    g.c_sizes = { g.a_sizes[0], g.a_sizes[1], M, N };
    g.valid = true;
    return g;
}

// ---------------------------------------------------------------------------
// Logging helpers
// ---------------------------------------------------------------------------

static std::string SizesStr(const std::vector<uint32_t>& s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) {
        if (i) r += ",";
        r += std::to_string(s[i]);
    }
    return r + "]";
}

static std::string DimsStr(const std::vector<int64_t>& s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) {
        if (i) r += ",";
        r += (s[i] < 0 ? "?" : std::to_string(s[i]));
    }
    return r + "]";
}

static std::string DtypeStr(DML_TENSOR_DATA_TYPE d) {
    switch (d) {
    case DML_TENSOR_DATA_TYPE_FLOAT32: return "fp32";
    case DML_TENSOR_DATA_TYPE_FLOAT16: return "fp16";
    default:                           return "unknown";
    }
}

// ---------------------------------------------------------------------------
// OpActivationFusionRule implementation
// ---------------------------------------------------------------------------

OpActivationFusionRule::OpActivationFusionRule(OpActivationConfig config)
    : m_config(std::move(config))
{
}

// ---------------------------------------------------------------------------
// BuildPattern
//
// One optional downstream branch per allowed activation.  All branches are at
// output index 0.  The engine tries each optional in order and takes the first
// (and only) match — since it's an activation downstream there can be at most
// one matching branch.
// ---------------------------------------------------------------------------
PNode OpActivationFusionRule::BuildPattern() const {
    PNode base(m_config.base_op_type);
    base.As(CaptureKey("base"));

    for (const auto& act : m_config.allowed_activations) {
        base.Downstream(0,
            PNode(act).As(CaptureKey("act." + act)),
            /*optional=*/true);
    }
    return base;
}

// ---------------------------------------------------------------------------
// MatchesResult
//
// At least one activation capture must be populated.  The base op alone
// (no activation downstream) is not a fusion candidate.
// ---------------------------------------------------------------------------
bool OpActivationFusionRule::MatchesResult(const PatternMatch& m) const {
    for (const auto& act : m_config.allowed_activations) {
        if (m.NodeIdx(CaptureKey("act." + act)) != SIZE_MAX)
            return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// CapturePreFusionData
//
// Reads activation-specific attributes (alpha, beta, gamma) from the matched
// activation node in gc and stores them in match.scalar_values.  Also records
// the activation op type string so Compile() can dispatch without re-scanning.
// ---------------------------------------------------------------------------
void OpActivationFusionRule::CapturePreFusionData(
    PatternMatch&                          match,
    const fusion_utils::GraphConnectivity& gc,
    const OrtApi&                          ort_api) const
{
    // Find which activation was matched.
    size_t act_idx = SIZE_MAX;
    std::string act_op_type;
    for (const auto& act : m_config.allowed_activations) {
        size_t idx = match.NodeIdx(CaptureKey("act." + act));
        if (idx != SIZE_MAX) {
            act_idx = idx;
            act_op_type = act;
            break;
        }
    }
    if (act_idx == SIZE_MAX || act_idx >= gc.node_infos.size())
        return;

    // Read activation node metadata.
    int act_since_version = 0;
    {
        OrtStatus* sv_st = ort_api.Node_GetSinceVersion(gc.node_infos[act_idx].node, &act_since_version);
        if (!sv_st) {
            match.scalar_values[CaptureKey("act_since_version")] = static_cast<float>(act_since_version);
        } else {
            ort_api.ReleaseStatus(sv_st);
        }
    }

    match.value_names[CaptureKey("act_type")] = act_op_type;

    // Read activation-specific attributes.
    OrtNodeAdapter adapter(gc.node_infos[act_idx].node, ort_api);

    float alpha = 0.0f, beta = 0.0f, gamma = 0.0f;

    if (act_op_type == "LeakyRelu") {
        alpha = adapter.GetAttributeFloat("alpha", 0.01f);
    } else if (act_op_type == "Elu" || act_op_type == "ThresholdedRelu") {
        alpha = adapter.GetAttributeFloat("alpha", 1.0f);
    } else if (act_op_type == "HardSigmoid") {
        alpha = adapter.GetAttributeFloat("alpha", 0.2f);
        beta  = adapter.GetAttributeFloat("beta",  0.5f);
    } else if (act_op_type == "Selu") {
        alpha = adapter.GetAttributeFloat("alpha", 1.67326319f);
        gamma = adapter.GetAttributeFloat("gamma", 1.05070102f);
    } else if (act_op_type == "ScaledTanh") {
        alpha = adapter.GetAttributeFloat("alpha", 1.0f);
        beta  = adapter.GetAttributeFloat("beta",  1.0f);
    } else if (act_op_type == "ParametricSoftplus") {
        alpha = adapter.GetAttributeFloat("alpha", 1.0f);
        beta  = adapter.GetAttributeFloat("beta",  1.0f);
    }

    match.scalar_values[CaptureKey("act_alpha")] = alpha;
    match.scalar_values[CaptureKey("act_beta")]  = beta;
    match.scalar_values[CaptureKey("act_gamma")] = gamma;
}

// ---------------------------------------------------------------------------
// Opset versions accepted per base op type — mirrors ORT's c_fusableOps table.
// Opset versions accepted per activation type — mirrors ORT's c_activationOps.
// Both use exact sinceVersion matching, same as ORT's TryGetFusedOp().
// ---------------------------------------------------------------------------
static bool IsKnownActivationVersion(const std::string& act_type, int since_version) {
    // Versions mirror ORT's c_activationOps table exactly.
    // All "opset 7" activations actually have sc_sinceVer = 6 in ORT's OperatorVersions.h.
    if (act_type == "Relu")               return since_version == 6 || since_version == 13 || since_version == 14;
    if (act_type == "LeakyRelu")          return since_version == 6 || since_version == 16;
    if (act_type == "Sigmoid")            return since_version == 6 || since_version == 13;
    if (act_type == "Tanh")               return since_version == 6 || since_version == 13;
    if (act_type == "Elu")                return since_version == 6;
    if (act_type == "Selu")               return since_version == 6;
    if (act_type == "HardSigmoid")        return since_version == 6;
    if (act_type == "Softsign")           return since_version == 1;
    if (act_type == "Softplus")           return since_version == 1;
    if (act_type == "ThresholdedRelu")    return since_version == 1 || since_version == 10;
    if (act_type == "ScaledTanh")         return since_version == 1;
    if (act_type == "ParametricSoftplus") return since_version == 1;
    return false;
}

// kOnnxDomain is "" (empty string) in the ORT C API for the ai.onnx domain.
static bool IsOnnxDomain(std::string_view domain) {
    return domain.empty() || domain == "ai.onnx";
}

static bool IsKnownFusableVersion(const std::string& op_type, int since_version) {
    // Versions must match ORT's c_fusableOps table in OperatorUtility.cpp exactly.
    // Conv / ConvTranspose
    if (op_type == "Conv" || op_type == "ConvTranspose")
        return since_version == 1 || since_version == 11;
    // BatchNormalization
    if (op_type == "BatchNormalization")
        return since_version == 7 || since_version == 9
            || since_version == 14 || since_version == 15;
    // InstanceNormalization
    if (op_type == "InstanceNormalization")
        return since_version == 6;
    // MeanVarianceNormalization
    if (op_type == "MeanVarianceNormalization")
        return since_version == 1 || since_version == 9 || since_version == 13;
    // Gemm
    if (op_type == "Gemm")
        return since_version == 7 || since_version == 9
            || since_version == 11 || since_version == 13;
    // MatMul
    if (op_type == "MatMul")
        return since_version == 7 || since_version == 9 || since_version == 13;
    // Add
    if (op_type == "Add")
        return since_version == 7 || since_version == 13 || since_version == 14;
    // Sum
    if (op_type == "Sum")
        return since_version == 8 || since_version == 13;
    return false;
}

// ---------------------------------------------------------------------------
// ValidateCapture
//
// Rejects fusions when:
//   - Base op opset version is not in the known fusable set (mirrors ORT's
//     c_fusableOps table which uses exact sinceVersion matching).
//   - Base op has more than one output def (e.g. BatchNorm training mode).
//   - Input count filter fails (Sum requires exactly 2 inputs).
//   - Activation node has more than 1 input (mirrors ORT's single-input check
//     at GraphTransformer.cpp:110 — guards against PRelu-like activations).
//   - Data type is not fp16/fp32.
// ---------------------------------------------------------------------------
bool OpActivationFusionRule::ValidateCapture(
    const PatternMatch&                    match,
    const fusion_utils::GraphConnectivity& gc,
    const OrtApi&                          ort_api) const
{
    size_t base_idx = match.NodeIdx(CaptureKey("base"));
    if (base_idx == SIZE_MAX || base_idx >= gc.node_infos.size()) return true;
    const auto& base_info = gc.node_infos[base_idx];

    // Base op domain + opset version check — mirrors ORT's c_fusableOps exact
    // (type, domain, sinceVersion) triple lookup in TryGetFusedOp().
    {
        const char* domain_cstr = nullptr;
        OrtStatus* dst = ort_api.Node_GetDomain(base_info.node, &domain_cstr);
        if (!dst) {
            if (!IsOnnxDomain(domain_cstr ? domain_cstr : ""))
                return false;
        } else {
            ort_api.ReleaseStatus(dst);
        }

        int since_version = 0;
        OrtStatus* st = ort_api.Node_GetSinceVersion(base_info.node, &since_version);
        if (!st && !IsKnownFusableVersion(m_config.base_op_type, since_version))
            return false;
        if (st) ort_api.ReleaseStatus(st);
    }

    // Activation domain + opset version check — mirrors ORT's c_activationOps exact
    // version lookup.  Type, version, and domain were captured in CapturePreFusionData.
    {
        std::string act_type = match.ValueName(CaptureKey("act_type"));
        if (!act_type.empty()) {
            for (const auto& a : m_config.allowed_activations) {
                size_t act_idx = match.NodeIdx(CaptureKey("act." + a));
                if (act_idx == SIZE_MAX || act_idx >= gc.node_infos.size()) continue;

                const char* act_domain_cstr = nullptr;
                OrtStatus* dst = ort_api.Node_GetDomain(gc.node_infos[act_idx].node, &act_domain_cstr);
                if (!dst && !IsOnnxDomain(act_domain_cstr ? act_domain_cstr : ""))
                    return false;
                if (dst) ort_api.ReleaseStatus(dst);

                float sv_f = match.ScalarValue(CaptureKey("act_since_version"), -1.0f);
                if (sv_f >= 0.0f) {
                    int since_version = static_cast<int>(sv_f);
                    if (!IsKnownActivationVersion(act_type, since_version))
                        return false;
                }
                break;
            }
        }
    }

    // Base op must have exactly one output def — guards against multi-output ops
    // such as BatchNormalization in training mode (outputs mean/variance too).
    {
        size_t num_outputs = 0;
        ort_api.Node_GetNumOutputs(base_info.node, &num_outputs);
        if (num_outputs != 1)
            return false;
    }

    // Base op output must have exactly one consumer (the activation).
    // SiLU/Swish patterns (Conv→Sigmoid→Mul where Mul also reads Conv output)
    // would break if fused, because the fused output becomes activation(x)
    // and the original pre-activation output is lost.
    if (!base_info.output_names.empty()) {
        const std::string& base_out = base_info.output_names[0];
        auto cons_it = gc.consumer_map.find(base_out);
        if (cons_it != gc.consumer_map.end() && cons_it->second.size() > 1)
            return false;
    }

    // Input count filter (Sum requires exactly 2).
    if (m_config.input_count_filter) {
        size_t input_count = 0;
        ort_api.Node_GetNumInputs(base_info.node, &input_count);
        if (input_count != static_cast<size_t>(*m_config.input_count_filter))
            return false;
    }

    // Activation must take exactly 1 input — mirrors ORT's check at
    // GraphTransformer.cpp:110.
    {
        for (const auto& a : m_config.allowed_activations) {
            size_t act_idx = match.NodeIdx(CaptureKey("act." + a));
            if (act_idx == SIZE_MAX || act_idx >= gc.node_infos.size()) continue;
            size_t act_inputs = 0;
            ort_api.Node_GetNumInputs(gc.node_infos[act_idx].node, &act_inputs);
            if (act_inputs != 1)
                return false;
            break;
        }
    }

    // Data type: fp16 or fp32 only.  Read from first input's type info if available.
    {
        size_t ic = 0;
        ort_api.Node_GetNumInputs(base_info.node, &ic);
        if (ic > 0) {
            std::vector<const OrtValueInfo*> vis(ic, nullptr);
            ort_api.Node_GetInputs(base_info.node, vis.data(), ic);
            if (vis[0]) {
                const OrtTypeInfo* ti = nullptr;
                OrtStatus* st = ort_api.GetValueInfoTypeInfo(vis[0], &ti);
                if (!st && ti) {
                    const OrtTensorTypeAndShapeInfo* si = nullptr;
                    ort_api.CastTypeInfoToTensorInfo(ti, &si);
                    if (si) {
                        ONNXTensorElementDataType dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
                        ort_api.GetTensorElementType(si, &dtype);
                        if (dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
                            && dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                            && dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
                            return false;
                    }
                }
                if (st) ort_api.ReleaseStatus(st);
            }
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Kernel state and per-op DML compile helpers
// ---------------------------------------------------------------------------

struct OpActivationCompiledKernel {
    ComPtr<IDMLCompiledOperator>      compiled_op;
    ComPtr<ID3D12Resource>            persistent_resource;
    ComPtr<IUnknown>                  persistent_allocator;
    std::optional<DML_BUFFER_BINDING> persistent_binding;

    bool IsValid() const { return compiled_op != nullptr; }
};

// Parameters captured at Compile time — enough to (re)build the DML operator.
enum class BaseOpKind { Add, Sum, Gemm, Conv, ConvTranspose, BatchNorm, InstanceNorm, MVN };

struct OpActivationKernelState {
    BaseOpKind                      base_op_kind;
    FusedActivationType             activation;
    float                           act_alpha = 0.0f;
    float                           act_beta  = 0.0f;
    float                           act_gamma = 0.0f;

    // Gemm-specific
    bool                            trans_a = false;
    bool                            trans_b = false;
    float                           gemm_alpha = 1.0f;
    float                           gemm_beta  = 1.0f;

    // Input indices in the fused subgraph.
    // When one side is a constant initializer, its index is SIZE_MAX and the
    // GPU-resident buffer is held in initializer_gpu_resource below.
    size_t                          input_idx_a = 0;
    size_t                          input_idx_b = 1;
    // Gemm optional C tensor: SIZE_MAX if absent
    size_t                          input_idx_c = SIZE_MAX;

    // Constant initializer side: 0=A, 1=B, -1=neither.
    // Mirrors FusedMatMulFusionRule's initializer_side pattern.
    int                             initializer_side = -1;
    ComPtr<ID3D12Resource>          initializer_gpu_resource;
    ComPtr<IUnknown>                initializer_allocator_ref;
    uint64_t                        initializer_bytes = 0;
    std::vector<uint32_t>           initializer_sizes;
    DML_TENSOR_DATA_TYPE            initializer_dml_dtype = DML_TENSOR_DATA_TYPE_UNKNOWN;

    // Conv / ConvTranspose attributes (read at Compile time from ONNX node attrs).
    std::vector<uint32_t>           conv_kernel_shape;
    std::vector<uint32_t>           conv_strides;
    std::vector<uint32_t>           conv_dilations;
    std::vector<uint32_t>           conv_pads;           // [start0..startN, end0..endN]
    std::vector<uint32_t>           conv_output_padding; // ConvTranspose only
    std::vector<uint32_t>           conv_output_shape;   // ConvTranspose: explicit output_shape attr (spatial only)
    std::string                     conv_auto_pad;        // "", "NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID"
    uint32_t                        conv_group = 1;

    // BatchNorm / InstanceNorm / MVN attributes.
    float                           norm_epsilon = 1e-5f;
    bool                            bn_spatial   = true;
    bool                            mvn_normalize_variance = true;
    bool                            mvn_across_channels    = false;
    std::vector<int32_t>            mvn_axes;             // explicit ONNX axes (opset 13+), signed; negatives resolved against input rank at compile/compute

    // Multi-input resources for Conv/BatchNorm/InstanceNorm.
    // Parallel vectors indexed by ONNX input slot:
    //   Conv:         [0]=X, [1]=W,     [2]=B (or absent)
    //   BatchNorm:    [0]=X, [1]=scale, [2]=bias, [3]=mean, [4]=var
    //   InstanceNorm: [0]=X, [1]=scale, [2]=bias
    //   MVN:          [0]=X only
    // Each non-X slot is either a pure constant (dyn_input_indices == SIZE_MAX,
    // pre-uploaded into static_* below) or a runtime input (dyn_input_indices
    // holds its graph-input index; bound directly from the OrtValue at Compute,
    // matching ORT). X (slot 0) is always runtime.
    std::vector<size_t>             dyn_input_indices;    // size = num ONNX inputs
    std::vector<ComPtr<ID3D12Resource>> static_gpu_resources;
    std::vector<ComPtr<IUnknown>>   static_allocator_refs;
    std::vector<uint64_t>           static_bytes;
    std::vector<std::vector<uint32_t>> static_sizes;
    std::vector<DML_TENSOR_DATA_TYPE>  static_dml_dtypes;

    PluginDmlExecutionProviderImpl* provider = nullptr;
    const OrtApi*                  ort_api  = nullptr;

    std::mutex                      init_mutex;
    bool                            initialized = false;
    OpActivationCompiledKernel      kernel;
    std::vector<uint32_t>           compiled_a_sizes;
    std::vector<uint32_t>           compiled_b_sizes;
    // Multi-input path: per-slot sizes the kernel was last compiled for.
    // Includes dynamic (runtime) weight slots so a shape change on any input
    // triggers a recompile, not just X.
    std::vector<std::vector<uint32_t>> compiled_slot_sizes;
    // Scratch: runtime shapes/bytes read from the current Compute's OrtValues,
    // indexed by ONNX slot. Only dynamic slots (dyn_input_indices != SIZE_MAX)
    // are populated; constant slots use static_sizes/static_bytes instead.
    std::vector<std::vector<uint32_t>> runtime_slot_sizes;
    std::vector<uint64_t>              runtime_slot_bytes;
};

// ---- Compile helper for Add / Sum (DML_ELEMENT_WISE_ADD1_OPERATOR_DESC) ----

static OpActivationCompiledKernel CompileAddActivation(
    PluginDmlExecutionProviderImpl* provider,
    FusedActivationType             activation,
    float                           act_alpha,
    float                           act_beta,
    float                           act_gamma,
    DML_TENSOR_DATA_TYPE            dml_dtype,
    const std::vector<uint32_t>&    a_sizes_in,
    const std::vector<uint32_t>&    b_sizes_in,
    uint64_t                        a_bytes,
    uint64_t                        b_bytes)
{
    OpActivationCompiledKernel result;

    ComPtr<IDMLDevice> dml_device;
    if (FAILED(provider->GetDmlDevice(dml_device.GetAddressOf()))) {
        DML_PERF_LOG("[OpActFusion/Add] CompileAddActivation: GetDmlDevice FAILED");
        return result;
    }

    ActivationDescStorage act_storage = BuildActivationDesc(activation, act_alpha, act_beta, act_gamma);

    // Compute broadcast output shape: element-wise max per dim after right-aligning.
    // Pad both inputs to the same rank first (DML minimum 4D, maximum 8D for DML >= 0x3000).
    size_t rank = std::max(a_sizes_in.size(), b_sizes_in.size());
    if (rank < 4) rank = 4;
    if (rank > DML_TENSOR_DIMENSION_COUNT_MAX1) {
        DML_PERF_LOG("[OpActFusion/Add] CompileAddActivation: rank ", rank, " exceeds DML max (", DML_TENSOR_DIMENSION_COUNT_MAX1, ")");
        return result;
    }

    // Pad physical sizes to rank (right-aligned, 1-filled on the left).
    auto pad_sizes = [&](const std::vector<uint32_t>& s_in) {
        std::vector<uint32_t> out(rank, 1u);
        size_t offset = rank - s_in.size();
        for (size_t i = 0; i < s_in.size(); ++i) out[offset + i] = s_in[i];
        return out;
    };

    std::vector<uint32_t> a_phys = pad_sizes(a_sizes_in);
    std::vector<uint32_t> b_phys = pad_sizes(b_sizes_in);

    // Broadcast output sizes.
    std::vector<uint32_t> c_sizes(rank);
    for (size_t i = 0; i < rank; ++i)
        c_sizes[i] = std::max(a_phys[i], b_phys[i]);

    // DML broadcast encoding: the Sizes field must equal the broadcast output
    // shape (not the physical shape). Stride is 0 on dims where the physical
    // size is 1 (broadcast dim), and packed on dims where it matches output.
    // TotalTensorSizeInBytes reflects the actual physical allocation.
    auto make_broadcast_strides = [&](const std::vector<uint32_t>& phys) {
        std::vector<uint32_t> strides(rank, 0u);
        uint32_t stride = 1;
        for (int i = static_cast<int>(rank) - 1; i >= 0; --i) {
            if (phys[i] != 1u) {
                strides[i] = stride;
                stride *= phys[i];
            }
            // stride stays 0 for broadcast (phys==1) dims
        }
        return strides;
    };

    std::vector<uint32_t> a_strides = make_broadcast_strides(a_phys);
    std::vector<uint32_t> b_strides = make_broadcast_strides(b_phys);

    size_t elem_size = (dml_dtype == DML_TENSOR_DATA_TYPE_FLOAT32) ? 4 : 2;
    size_t c_elem_count = 1;
    for (auto d : c_sizes) c_elem_count *= static_cast<size_t>(d);
    uint64_t c_bytes = ((static_cast<uint64_t>(c_elem_count * elem_size) + 3u) & ~uint64_t(3u));

    // Input descriptors use broadcast output sizes + stride-0 on broadcast dims.
    // TotalTensorSizeInBytes = physical buffer size (a_bytes / b_bytes).
    DML_BUFFER_TENSOR_DESC a_buf{};
    a_buf.DataType               = dml_dtype;
    a_buf.DimensionCount         = static_cast<UINT>(rank);
    a_buf.Sizes                  = c_sizes.data();   // broadcast shape
    a_buf.Strides                = a_strides.data(); // 0 on broadcast dims
    a_buf.TotalTensorSizeInBytes = a_bytes;          // physical size

    DML_BUFFER_TENSOR_DESC b_buf{};
    b_buf.DataType               = dml_dtype;
    b_buf.DimensionCount         = static_cast<UINT>(rank);
    b_buf.Sizes                  = c_sizes.data();   // broadcast shape
    b_buf.Strides                = b_strides.data(); // 0 on broadcast dims
    b_buf.TotalTensorSizeInBytes = b_bytes;          // physical size

    DML_BUFFER_TENSOR_DESC c_buf{};
    c_buf.DataType               = dml_dtype;
    c_buf.DimensionCount         = static_cast<UINT>(rank);
    c_buf.Sizes                  = c_sizes.data();
    c_buf.TotalTensorSizeInBytes = c_bytes;

    DML_TENSOR_DESC a_desc{ DML_TENSOR_TYPE_BUFFER, &a_buf };
    DML_TENSOR_DESC b_desc{ DML_TENSOR_TYPE_BUFFER, &b_buf };
    DML_TENSOR_DESC c_desc{ DML_TENSOR_TYPE_BUFFER, &c_buf };

    DML_ELEMENT_WISE_ADD1_OPERATOR_DESC add_desc{};
    add_desc.ATensor          = &a_desc;
    add_desc.BTensor          = &b_desc;
    add_desc.OutputTensor     = &c_desc;
    add_desc.FusedActivation  = act_storage.valid ? &act_storage.desc : nullptr;
    DML_OPERATOR_DESC op_desc{ DML_OPERATOR_ELEMENT_WISE_ADD1, &add_desc };

    ComPtr<IDMLOperator> op;
    if (FAILED(dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op)))) {
        DML_PERF_LOG("[OpActFusion/Add] CreateOperator FAILED");
        return result;
    }
    DML_EXECUTION_FLAGS exec_flags = (dml_dtype == DML_TENSOR_DATA_TYPE_FLOAT16)
        ? DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION
        : DML_EXECUTION_FLAG_NONE;
    if (FAILED(dml_device->CompileOperator(op.Get(), exec_flags,
                                            IID_PPV_ARGS(result.compiled_op.GetAddressOf())))) {
        DML_PERF_LOG("[OpActFusion/Add] CompileOperator FAILED");
        return result;
    }

    UINT64 persistent_size = result.compiled_op->GetBindingProperties().PersistentResourceSize;
    if (persistent_size > 0) {
        if (FAILED(provider->AllocatePooledResource(
                static_cast<size_t>(persistent_size), AllocatorRoundingMode::Disabled,
                result.persistent_resource.GetAddressOf(),
                result.persistent_allocator.GetAddressOf()))) {
            result.compiled_op.Reset();
            return result;
        }
        result.persistent_binding = DML_BUFFER_BINDING{
            result.persistent_resource.Get(), 0, persistent_size };
    }
    const DML_BUFFER_BINDING* persistent_ptr =
        result.persistent_binding ? &*result.persistent_binding : nullptr;
    if (FAILED(provider->InitializeOperator(
            result.compiled_op.Get(), persistent_ptr,
            gsl::span<const DML_BUFFER_BINDING>{}))) {
        result.compiled_op.Reset();
        return result;
    }
    provider->QueueReference(result.compiled_op.Get());
    if (result.persistent_allocator)
        provider->QueueReference(result.persistent_allocator.Get());

    return result;
}

// ---- Compile helper for Gemm (DML_GEMM_OPERATOR_DESC) ----

static OpActivationCompiledKernel CompileGemmActivation(
    PluginDmlExecutionProviderImpl* provider,
    bool                            trans_a,
    bool                            trans_b,
    float                           gemm_alpha,
    float                           gemm_beta,
    FusedActivationType             activation,
    float                           act_alpha,
    float                           act_beta,
    float                           act_gamma,
    DML_TENSOR_DATA_TYPE            dml_dtype,
    const std::vector<uint32_t>&    a_sizes_in,
    const std::vector<uint32_t>&    b_sizes_in,
    uint64_t                        a_bytes,
    uint64_t                        b_bytes,
    const std::vector<uint32_t>*    c_sizes_in,
    uint64_t                        c_bias_bytes)
{
    OpActivationCompiledKernel result;

    ComPtr<IDMLDevice> dml_device;
    if (FAILED(provider->GetDmlDevice(dml_device.GetAddressOf()))) {
        DML_PERF_LOG("[OpActFusion/Gemm] GetDmlDevice FAILED");
        return result;
    }

    GemmShapes g = PrepareGemmShapes(a_sizes_in, b_sizes_in, trans_a, trans_b);
    if (!g.valid) {
        DML_PERF_LOG("[OpActFusion/Gemm] PrepareGemmShapes FAILED");
        return result;
    }

    size_t elem_size = (dml_dtype == DML_TENSOR_DATA_TYPE_FLOAT32) ? 4 : 2;
    size_t c_out_elems = 1;
    for (auto d : g.c_sizes) c_out_elems *= static_cast<size_t>(d);
    uint64_t c_out_bytes = ((static_cast<uint64_t>(c_out_elems * elem_size) + 3u) & ~uint64_t(3u));

    DML_BUFFER_TENSOR_DESC a_buf{};
    a_buf.DataType = dml_dtype; a_buf.DimensionCount = 4;
    a_buf.Sizes = g.a_sizes.data(); a_buf.Strides = g.a_strides.data();
    a_buf.TotalTensorSizeInBytes = a_bytes;
    DML_TENSOR_DESC a_desc{ DML_TENSOR_TYPE_BUFFER, &a_buf };

    DML_BUFFER_TENSOR_DESC b_buf{};
    b_buf.DataType = dml_dtype; b_buf.DimensionCount = 4;
    b_buf.Sizes = g.b_sizes.data(); b_buf.Strides = g.b_strides.data();
    b_buf.TotalTensorSizeInBytes = b_bytes;
    DML_TENSOR_DESC b_desc{ DML_TENSOR_TYPE_BUFFER, &b_buf };

    // Optional bias C: broadcast from [1,1,1,N] to match output shape [*,*,M,N].
    std::vector<uint32_t> c_bias_4d;
    std::vector<uint32_t> c_bias_strides;
    DML_BUFFER_TENSOR_DESC c_bias_buf{};
    DML_TENSOR_DESC c_bias_desc{};
    bool has_bias = (c_sizes_in != nullptr && !c_sizes_in->empty());
    if (has_bias) {
        // Broadcast C to the output rank.  C from Gemm is typically [N] or [M,N].
        const auto& csizes = *c_sizes_in;
        c_bias_4d.assign(4, 1u);
        c_bias_strides.assign(4, 0u);
        size_t offset = 4 - csizes.size();
        uint32_t stride = 1;
        for (int i = static_cast<int>(csizes.size()) - 1; i >= 0; --i) {
            c_bias_4d[offset + i] = csizes[i];
            c_bias_strides[offset + i] = stride;
            stride *= csizes[i];
        }
        // Broadcast dims where size is 1 (stride stays 0 for broadcast).
        for (size_t d = 0; d < 4; ++d) {
            if (c_bias_4d[d] == 1) c_bias_strides[d] = 0;
        }
        c_bias_buf.DataType               = dml_dtype;
        c_bias_buf.DimensionCount         = 4;
        c_bias_buf.Sizes                  = c_bias_4d.data();
        c_bias_buf.Strides                = c_bias_strides.data();
        c_bias_buf.TotalTensorSizeInBytes = c_bias_bytes;
        c_bias_desc = { DML_TENSOR_TYPE_BUFFER, &c_bias_buf };
    }

    DML_BUFFER_TENSOR_DESC out_buf{};
    out_buf.DataType               = dml_dtype;
    out_buf.DimensionCount         = 4;
    out_buf.Sizes                  = g.c_sizes.data();
    out_buf.TotalTensorSizeInBytes = c_out_bytes;
    DML_TENSOR_DESC out_desc{ DML_TENSOR_TYPE_BUFFER, &out_buf };

    ActivationDescStorage act_storage = BuildActivationDesc(activation, act_alpha, act_beta, act_gamma);

    DML_GEMM_OPERATOR_DESC gemm_desc{};
    gemm_desc.ATensor        = &a_desc;
    gemm_desc.BTensor        = &b_desc;
    gemm_desc.CTensor        = has_bias ? &c_bias_desc : nullptr;
    gemm_desc.OutputTensor   = &out_desc;
    gemm_desc.TransA         = trans_a ? DML_MATRIX_TRANSFORM_TRANSPOSE : DML_MATRIX_TRANSFORM_NONE;
    gemm_desc.TransB         = trans_b ? DML_MATRIX_TRANSFORM_TRANSPOSE : DML_MATRIX_TRANSFORM_NONE;
    gemm_desc.Alpha          = gemm_alpha;
    gemm_desc.Beta           = has_bias ? gemm_beta : 0.0f;
    gemm_desc.FusedActivation = act_storage.valid ? &act_storage.desc : nullptr;
    DML_OPERATOR_DESC op_desc{ DML_OPERATOR_GEMM, &gemm_desc };

    ComPtr<IDMLOperator> op;
    if (FAILED(dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op)))) {
        DML_PERF_LOG("[OpActFusion/Gemm] CreateOperator FAILED");
        return result;
    }
    DML_EXECUTION_FLAGS exec_flags = (dml_dtype == DML_TENSOR_DATA_TYPE_FLOAT16)
        ? DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION
        : DML_EXECUTION_FLAG_NONE;
    if (FAILED(dml_device->CompileOperator(op.Get(), exec_flags,
                                            IID_PPV_ARGS(result.compiled_op.GetAddressOf())))) {
        DML_PERF_LOG("[OpActFusion/Gemm] CompileOperator FAILED");
        return result;
    }

    UINT64 persistent_size = result.compiled_op->GetBindingProperties().PersistentResourceSize;
    if (persistent_size > 0) {
        if (FAILED(provider->AllocatePooledResource(
                static_cast<size_t>(persistent_size), AllocatorRoundingMode::Disabled,
                result.persistent_resource.GetAddressOf(),
                result.persistent_allocator.GetAddressOf()))) {
            result.compiled_op.Reset();
            return result;
        }
        result.persistent_binding = DML_BUFFER_BINDING{
            result.persistent_resource.Get(), 0, persistent_size };
    }
    const DML_BUFFER_BINDING* persistent_ptr =
        result.persistent_binding ? &*result.persistent_binding : nullptr;
    if (FAILED(provider->InitializeOperator(
            result.compiled_op.Get(), persistent_ptr,
            gsl::span<const DML_BUFFER_BINDING>{}))) {
        result.compiled_op.Reset();
        return result;
    }
    provider->QueueReference(result.compiled_op.Get());
    if (result.persistent_allocator)
        provider->QueueReference(result.persistent_allocator.Get());

    return result;
}

// ---------------------------------------------------------------------------
// Helper: pad a size vector to at least min_rank (left-fill with 1s).
// ---------------------------------------------------------------------------
static std::vector<uint32_t> PadToRank(const std::vector<uint32_t>& s, size_t min_rank) {
    if (s.size() >= min_rank) return s;
    std::vector<uint32_t> out(min_rank, 1u);
    size_t off = min_rank - s.size();
    for (size_t i = 0; i < s.size(); ++i) out[off + i] = s[i];
    return out;
}

// Resolve explicit ONNX MVN axes into DML axis indices, mirroring ORT's
// GetDmlAdjustedAxis (OperatorUtility.cpp): normalize negatives against the
// original ONNX rank, then shift into the left-padded DML tensor space.
static std::vector<uint32_t> ResolveMvnAxes(
    const std::vector<int32_t>& onnx_axes, size_t onnx_rank, size_t dml_rank)
{
    std::vector<uint32_t> dml_axes;
    dml_axes.reserve(onnx_axes.size());
    const int32_t rank = static_cast<int32_t>(onnx_rank);
    const int32_t shift = static_cast<int32_t>(dml_rank) - rank;
    for (int32_t axis : onnx_axes) {
        if (axis < 0) axis += rank;
        dml_axes.push_back(static_cast<uint32_t>(axis + shift));
    }
    return dml_axes;
}

// ---------------------------------------------------------------------------
// Helper: compile a DML operator (shared boilerplate for all Compile helpers).
// ---------------------------------------------------------------------------
static bool FinalizeCompiledKernel(
    IDMLDevice*                  dml_device,
    PluginDmlExecutionProviderImpl* provider,
    IDMLOperator*                op,
    const char*                  log_tag,
    OpActivationCompiledKernel&  out,
    DML_TENSOR_DATA_TYPE         dml_dtype = DML_TENSOR_DATA_TYPE_UNKNOWN)
{
    DML_EXECUTION_FLAGS exec_flags = (dml_dtype == DML_TENSOR_DATA_TYPE_FLOAT16)
        ? DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION
        : DML_EXECUTION_FLAG_NONE;
    if (FAILED(dml_device->CompileOperator(op, exec_flags,
                                           IID_PPV_ARGS(out.compiled_op.GetAddressOf())))) {
        DML_PERF_LOG(log_tag, " CompileOperator FAILED");
        return false;
    }
    UINT64 persistent_size = out.compiled_op->GetBindingProperties().PersistentResourceSize;
    if (persistent_size > 0) {
        if (FAILED(provider->AllocatePooledResource(
                static_cast<size_t>(persistent_size), AllocatorRoundingMode::Disabled,
                out.persistent_resource.GetAddressOf(),
                out.persistent_allocator.GetAddressOf()))) {
            out.compiled_op.Reset();
            return false;
        }
        out.persistent_binding = DML_BUFFER_BINDING{ out.persistent_resource.Get(), 0, persistent_size };
    }
    const DML_BUFFER_BINDING* persistent_ptr =
        out.persistent_binding ? &*out.persistent_binding : nullptr;
    if (FAILED(provider->InitializeOperator(
            out.compiled_op.Get(), persistent_ptr,
            gsl::span<const DML_BUFFER_BINDING>{}))) {
        out.compiled_op.Reset();
        return false;
    }
    provider->QueueReference(out.compiled_op.Get());
    if (out.persistent_allocator)
        provider->QueueReference(out.persistent_allocator.Get());
    return true;
}

// ---------------------------------------------------------------------------
// Compile helper for Conv / ConvTranspose (DML_CONVOLUTION_OPERATOR_DESC).
// W (weights) and optionally B (bias) are pre-uploaded static inputs.
// X is a runtime input whose shape is provided here.
// ---------------------------------------------------------------------------
static OpActivationCompiledKernel CompileConvActivation(
    PluginDmlExecutionProviderImpl* provider,
    bool                            is_transposed,
    const std::vector<uint32_t>&    kernel_shape,  // spatial dims only
    const std::vector<uint32_t>&    strides_in,
    const std::vector<uint32_t>&    dilations_in,
    const std::vector<uint32_t>&    pads_in,       // [s0..sN, e0..eN]
    const std::vector<uint32_t>&    output_padding_in,
    const std::vector<uint32_t>&    output_shape_in, // ConvTranspose explicit spatial output (may be empty)
    const std::string&              auto_pad,        // "NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID", ""
    uint32_t                        group_count,
    FusedActivationType             activation,
    float act_alpha, float act_beta, float act_gamma,
    DML_TENSOR_DATA_TYPE            dml_dtype,
    const std::vector<uint32_t>&    x_sizes_in,   // [N, C_in, spatial...]
    uint64_t                        x_bytes,
    const std::vector<uint32_t>&    w_sizes_in,   // [C_out, C_in/g, kH, kW] or [C_in, C_out/g, kH, kW]
    uint64_t                        w_bytes,
    const std::vector<uint32_t>*    b_sizes_in,   // nullptr if no bias; else [C_out]
    uint64_t                        b_bytes)
{
    OpActivationCompiledKernel result;

    const char* log_tag = is_transposed ? "[OpActFusion/ConvTranspose]" : "[OpActFusion/Conv]";

    if (x_sizes_in.size() < 3) {
        DML_PERF_LOG(log_tag, " CompileConvActivation: X rank < 3");
        return result;
    }

    ComPtr<IDMLDevice> dml_device;
    if (FAILED(provider->GetDmlDevice(dml_device.GetAddressOf()))) {
        DML_PERF_LOG(log_tag, " GetDmlDevice FAILED");
        return result;
    }

    // Spatial dimension count: X rank minus 2 (N and C), minimum 2.
    size_t spatial_dims = x_sizes_in.size() - 2;
    if (spatial_dims < 2) spatial_dims = 2;  // pad 1D to 2D

    // Pad X to [N, C, H, W, ...] with spatial_dims spatial dimensions.
    // For 1D input [N,C,W]: insert a 1 at dim-2 → [N,C,1,W], matching ORT's
    // "insert between C and W" convention (NOT append at the end).
    std::vector<uint32_t> x_sizes = x_sizes_in;
    if (x_sizes_in.size() == 3 && spatial_dims == 2)
        x_sizes.insert(x_sizes.begin() + 2, 1u);
    std::vector<uint32_t> w_sizes = w_sizes_in;
    if (w_sizes_in.size() == 3 && spatial_dims == 2)
        w_sizes.insert(w_sizes.begin() + 2, 1u);

    // For 1D→2D: all per-spatial attribute vectors need identity values
    // inserted at the front, matching the spatial dim inserted at position 2
    // in X/W (i.e. [N,C,1,W] — spatial dim 0 is the padded one).
    // ORT uses FillWithLeadingValues which prepends defaults at the front.
    bool pad_1d = (x_sizes_in.size() == 3 && spatial_dims == 2);
    auto pad_front = [&](const std::vector<uint32_t>& src, uint32_t def) {
        std::vector<uint32_t> v = src.empty()
            ? std::vector<uint32_t>(spatial_dims, def)
            : src;
        while (v.size() < spatial_dims) v.insert(v.begin(), def);
        return v;
    };
    std::vector<uint32_t> strides    = pad_front(strides_in,        1u);
    std::vector<uint32_t> dilations  = pad_front(dilations_in,      1u);
    std::vector<uint32_t> out_padding = pad_front(output_padding_in, 0u);
    std::vector<uint32_t> ks_padded  = pad_front(kernel_shape,      1u);

    // Compute output shape and resolve pads.
    uint32_t c_out = is_transposed
        ? w_sizes[1] * group_count    // ConvTranspose: filter is [C_in, C_out/g, ...]
        : w_sizes[0];                 // Conv:          filter is [C_out, C_in/g, ...]

    std::vector<uint32_t> start_pads(spatial_dims, 0u);
    std::vector<uint32_t> end_pads(spatial_dims, 0u);

    // Determine if auto_pad requires computed padding (SAME_UPPER / SAME_LOWER).
    bool is_same_upper = (auto_pad == "SAME_UPPER");
    bool is_same_lower = (auto_pad == "SAME_LOWER");
    bool use_auto_pad  = is_same_upper || is_same_lower;

    if (!use_auto_pad && !pads_in.empty()) {
        size_t orig_spatial = pads_in.size() / 2;
        size_t offset = spatial_dims - orig_spatial;
        for (size_t i = 0; i < orig_spatial && (i + offset) < spatial_dims; ++i) {
            start_pads[i + offset] = pads_in[i];
            end_pads[i + offset]   = pads_in[orig_spatial + i];
        }
    }
    // VALID → pads stay zero (already initialized to zero).

    std::vector<uint32_t> out_sizes;
    out_sizes.push_back(x_sizes[0]);  // N
    out_sizes.push_back(c_out);       // C_out

    for (size_t i = 0; i < spatial_dims; ++i) {
        uint32_t x_s = x_sizes[2 + i];
        uint32_t k_s = (ks_padded.size() > i) ? ks_padded[i] : w_sizes[2 + i];

        if (is_transposed) {
            uint32_t spatial_out;
            if (!output_shape_in.empty()) {
                // output_shape attr directly specifies the spatial output size.
                // Take the last spatial_dims elements (output_shape may include N/C or just spatial).
                size_t off = output_shape_in.size() - spatial_dims;
                spatial_out = output_shape_in[off + i];
                // Back-compute pads from output_shape (mirrors ORT InitializeKernelAndShapesTransposed).
                uint32_t window_size = dilations[i] * (k_s - 1) + 1;
                int total_pad = static_cast<int>((x_s - 1) * strides[i] + window_size)
                                - static_cast<int>(spatial_out);
                if (total_pad < 0) total_pad = 0;
                start_pads[i] = is_same_upper
                    ? static_cast<uint32_t>(total_pad / 2)
                    : static_cast<uint32_t>((total_pad + 1) / 2);
                end_pads[i] = static_cast<uint32_t>(total_pad) - start_pads[i];
                out_padding[i] = 0;  // output_shape overrides output_padding
            } else if (use_auto_pad) {
                // SAME_UPPER/SAME_LOWER on ConvTranspose without explicit output_shape:
                // Apply the SAME padding formula to the ConvTranspose input dims
                // (mirrors ORT's ResolveAutoPadding called with the transposed-conv input).
                uint32_t strided_out = (x_s + strides[i] - 1) / strides[i];
                uint32_t kernel_len  = 1 + (k_s - 1) * dilations[i];
                uint32_t len_needed  = strides[i] * (strided_out - 1) + kernel_len;
                uint32_t total_pad   = (len_needed <= x_s) ? 0u : (len_needed - x_s);
                start_pads[i] = is_same_upper
                    ? total_pad / 2
                    : (total_pad + 1) / 2;
                end_pads[i] = total_pad - start_pads[i];
                spatial_out = (x_s - 1) * strides[i]
                              - start_pads[i] - end_pads[i]
                              + dilations[i] * (k_s - 1) + 1
                              + out_padding[i];
            } else {
                spatial_out = (x_s - 1) * strides[i]
                              - start_pads[i] - end_pads[i]
                              + dilations[i] * (k_s - 1) + 1
                              + out_padding[i];
            }
            out_sizes.push_back(spatial_out);
        } else {
            // Conv forward.
            uint32_t spatial_out;
            if (use_auto_pad) {
                // SAME: output = ceil(input / stride).
                spatial_out = (x_s + strides[i] - 1) / strides[i];
                uint32_t effective_kernel = dilations[i] * (k_s - 1) + 1;
                uint32_t total_pad = (spatial_out > 1)
                    ? (spatial_out - 1) * strides[i] + effective_kernel - x_s
                    : 0;
                if (is_same_upper) {
                    start_pads[i] = total_pad / 2;
                } else {
                    start_pads[i] = (total_pad + 1) / 2;
                }
                end_pads[i] = total_pad - start_pads[i];
            } else {
                spatial_out = (x_s + start_pads[i] + end_pads[i] - dilations[i] * (k_s - 1) - 1)
                              / strides[i] + 1;
            }
            out_sizes.push_back(spatial_out);
        }
    }

    size_t elem_bytes = (dml_dtype == DML_TENSOR_DATA_TYPE_FLOAT32) ? 4 : 2;

    // Build tensor descriptors (packed strides, DWORD-aligned total bytes).
    // Storage for DML_TENSOR_DESCs that outlive the descriptor construction.
    struct TensorDescStorage {
        std::vector<uint32_t> sizes;
        std::vector<uint32_t> strides_vec;
        uint64_t              total_bytes = 0;
        DML_BUFFER_TENSOR_DESC buf{};
        DML_TENSOR_DESC        td{};
    };

    auto make_packed = [&](const std::vector<uint32_t>& szs, uint64_t bytes) -> TensorDescStorage {
        TensorDescStorage s;
        s.sizes = szs;
        s.total_bytes = bytes;
        s.buf.DataType           = dml_dtype;
        s.buf.Flags              = DML_TENSOR_FLAG_NONE;
        s.buf.DimensionCount     = static_cast<UINT>(szs.size());
        s.buf.Sizes              = s.sizes.data();
        s.buf.Strides            = nullptr;
        s.buf.TotalTensorSizeInBytes = bytes;
        s.buf.GuaranteedBaseOffsetAlignment = 0;
        s.td = { DML_TENSOR_TYPE_BUFFER, &s.buf };
        return s;
    };

    // Compute aligned bytes from sizes.
    auto size_to_bytes = [&](const std::vector<uint32_t>& szs) -> uint64_t {
        size_t count = 1;
        for (auto v : szs) count *= v;
        return (static_cast<uint64_t>(count * elem_bytes) + 3u) & ~uint64_t(3u);
    };

    // ORT always passes zeros to DML OutputPadding — it is used only for output size
    // computation and must be zeroed before the descriptor is built (matches ORT's
    // memset(kernelArgs.outputPadding, 0, ...) in DmlOperatorConvolution.cpp).
    std::fill(out_padding.begin(), out_padding.end(), 0u);

    TensorDescStorage x_desc  = make_packed(x_sizes,   x_bytes > 0 ? x_bytes : size_to_bytes(x_sizes));
    TensorDescStorage w_desc  = make_packed(w_sizes,   w_bytes > 0 ? w_bytes : size_to_bytes(w_sizes));
    TensorDescStorage out_desc = make_packed(out_sizes, size_to_bytes(out_sizes));

    // Bias: 1D [C_out], broadcast to [1, C_out, 1, 1, ...] for DML.
    TensorDescStorage b_desc;
    bool has_bias = (b_sizes_in != nullptr);
    if (has_bias) {
        std::vector<uint32_t> b_logical(2 + spatial_dims, 1u);
        b_logical[1] = c_out;
        b_desc = make_packed(b_logical, b_bytes > 0 ? b_bytes : size_to_bytes(*b_sizes_in));
    }

    ActivationDescStorage act_storage = BuildActivationDesc(activation, act_alpha, act_beta, act_gamma);

    DML_CONVOLUTION_OPERATOR_DESC conv_desc{};
    conv_desc.InputTensor    = &x_desc.td;
    conv_desc.FilterTensor   = &w_desc.td;
    conv_desc.BiasTensor     = has_bias ? &b_desc.td : nullptr;
    conv_desc.OutputTensor   = &out_desc.td;
    conv_desc.Mode           = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
    conv_desc.Direction      = is_transposed
        ? DML_CONVOLUTION_DIRECTION_BACKWARD
        : DML_CONVOLUTION_DIRECTION_FORWARD;
    conv_desc.DimensionCount = static_cast<UINT>(spatial_dims);
    conv_desc.Strides        = strides.data();
    conv_desc.Dilations      = dilations.data();
    conv_desc.StartPadding   = start_pads.data();
    conv_desc.EndPadding     = end_pads.data();
    conv_desc.OutputPadding  = out_padding.data();
    conv_desc.GroupCount     = group_count;
    conv_desc.FusedActivation = act_storage.valid ? &act_storage.desc : nullptr;
    DML_OPERATOR_DESC op_desc{ DML_OPERATOR_CONVOLUTION, &conv_desc };

    ComPtr<IDMLOperator> op;
    if (FAILED(dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op)))) {
        DML_PERF_LOG(log_tag, " CreateOperator FAILED");
        return result;
    }
    if (!FinalizeCompiledKernel(dml_device.Get(), provider, op.Get(), log_tag, result, dml_dtype))
        return result;

    DML_PERF_LOG(log_tag, " CompileConvActivation OK"
                 " X=", SizesStr(x_sizes), " W=", SizesStr(w_sizes),
                 " Out=", SizesStr(out_sizes),
                 " strides=", SizesStr(strides), " dilations=", SizesStr(dilations),
                 " pads=", SizesStr(start_pads), "/", SizesStr(end_pads),
                 " group=", group_count,
                 " activation=", FusedActivationTypeName(activation));
    return result;
}

// ---------------------------------------------------------------------------
// Compile helper for BatchNormalization (DML_BATCH_NORMALIZATION_OPERATOR_DESC).
// All of scale, bias, mean, var are pre-uploaded static inputs.
// X is the only runtime input.
// DML field order: {InputTensor, MeanTensor, VarianceTensor, ScaleTensor, BiasTensor}
// ONNX input order: [X, Scale, Bias, Mean, Variance]
// ---------------------------------------------------------------------------
static OpActivationCompiledKernel CompileBatchNormActivation(
    PluginDmlExecutionProviderImpl* provider,
    float                           epsilon,
    bool                            spatial,
    FusedActivationType             activation,
    float act_alpha, float act_beta, float act_gamma,
    DML_TENSOR_DATA_TYPE            dml_dtype,
    const std::vector<uint32_t>&    x_sizes,    // [N, C, spatial...]
    uint64_t                        x_bytes,
    const std::vector<uint32_t>&    scale_sizes, // [C]
    uint64_t                        scale_bytes,
    const std::vector<uint32_t>&    bias_sizes,  // [C]
    uint64_t                        bias_bytes,
    const std::vector<uint32_t>&    mean_sizes,  // [C]
    uint64_t                        mean_bytes,
    const std::vector<uint32_t>&    var_sizes,   // [C]
    uint64_t                        var_bytes)
{
    OpActivationCompiledKernel result;
    const char* log_tag = "[OpActFusion/BatchNorm]";

    if (x_sizes.size() < 2) {
        DML_PERF_LOG(log_tag, " CompileBatchNormActivation: X rank < 2");
        return result;
    }

    ComPtr<IDMLDevice> dml_device;
    if (FAILED(provider->GetDmlDevice(dml_device.GetAddressOf()))) {
        DML_PERF_LOG(log_tag, " GetDmlDevice FAILED");
        return result;
    }

    // Pad X to at least 4D.
    size_t dml_rank = std::max(x_sizes.size(), size_t(4));
    std::vector<uint32_t> x_padded = PadToRank(x_sizes, dml_rank);
    uint32_t c = x_padded[1];
    size_t elem_bytes = (dml_dtype == DML_TENSOR_DATA_TYPE_FLOAT32) ? 4 : 2;

    auto size_to_bytes = [&](const std::vector<uint32_t>& s, uint64_t hint) -> uint64_t {
        if (hint > 0) return hint;
        size_t n = 1; for (auto v : s) n *= v;
        return (static_cast<uint64_t>(n * elem_bytes) + 3u) & ~uint64_t(3u);
    };

    // Build packed X descriptor (padded to dml_rank).
    struct TDS {
        std::vector<uint32_t> sizes, strides_vec;
        uint64_t total_bytes = 0;
        DML_BUFFER_TENSOR_DESC buf{};
        DML_TENSOR_DESC td{};
    };
    auto make_packed = [&](const std::vector<uint32_t>& szs, uint64_t bytes) -> TDS {
        TDS s;
        s.sizes = szs;
        s.total_bytes = bytes;
        s.buf.DataType           = dml_dtype;
        s.buf.Flags              = DML_TENSOR_FLAG_NONE;
        s.buf.DimensionCount     = static_cast<UINT>(szs.size());
        s.buf.Sizes              = s.sizes.data();
        s.buf.Strides            = nullptr;
        s.buf.TotalTensorSizeInBytes = bytes;
        s.buf.GuaranteedBaseOffsetAlignment = 0;
        s.td = { DML_TENSOR_TYPE_BUFFER, &s.buf };
        return s;
    };

    TDS x_desc   = make_packed(x_padded, size_to_bytes(x_padded, x_bytes));
    TDS out_desc = make_packed(x_padded, size_to_bytes(x_padded, x_bytes));

    uint64_t c_aligned_bytes_scale = size_to_bytes(scale_sizes, scale_bytes);
    uint64_t c_aligned_bytes_bias  = size_to_bytes(bias_sizes,  bias_bytes);
    uint64_t c_aligned_bytes_mean  = size_to_bytes(mean_sizes,  mean_bytes);
    uint64_t c_aligned_bytes_var   = size_to_bytes(var_sizes,   var_bytes);

    auto make_1d_broadcast = [&](uint64_t bytes) -> TDS {
        TDS s;
        s.sizes = std::vector<uint32_t>(dml_rank, 1u);
        s.sizes[1] = c;
        s.strides_vec.resize(dml_rank, 0u);
        s.strides_vec[1] = 1u;
        s.total_bytes = bytes;
        s.buf.DataType           = dml_dtype;
        s.buf.Flags              = DML_TENSOR_FLAG_NONE;
        s.buf.DimensionCount     = static_cast<UINT>(dml_rank);
        s.buf.Sizes              = s.sizes.data();
        s.buf.Strides            = s.strides_vec.data();
        s.buf.TotalTensorSizeInBytes = bytes;
        s.buf.GuaranteedBaseOffsetAlignment = 0;
        s.td = { DML_TENSOR_TYPE_BUFFER, &s.buf };
        return s;
    };

    TDS scale_desc = make_1d_broadcast(c_aligned_bytes_scale);
    TDS bias_desc  = make_1d_broadcast(c_aligned_bytes_bias);
    TDS mean_desc  = make_1d_broadcast(c_aligned_bytes_mean);
    TDS var_desc   = make_1d_broadcast(c_aligned_bytes_var);

    ActivationDescStorage act_storage = BuildActivationDesc(activation, act_alpha, act_beta, act_gamma);

    DML_BATCH_NORMALIZATION_OPERATOR_DESC bn_desc{};
    bn_desc.InputTensor    = &x_desc.td;
    bn_desc.MeanTensor     = &mean_desc.td;
    bn_desc.VarianceTensor = &var_desc.td;
    bn_desc.ScaleTensor    = &scale_desc.td;
    bn_desc.BiasTensor     = &bias_desc.td;
    bn_desc.OutputTensor   = &out_desc.td;
    bn_desc.Spatial        = spatial ? TRUE : FALSE;
    bn_desc.Epsilon        = epsilon;
    bn_desc.FusedActivation = act_storage.valid ? &act_storage.desc : nullptr;
    DML_OPERATOR_DESC op_desc{ DML_OPERATOR_BATCH_NORMALIZATION, &bn_desc };

    ComPtr<IDMLOperator> op;
    if (FAILED(dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op)))) {
        DML_PERF_LOG(log_tag, " CreateOperator FAILED");
        return result;
    }
    if (!FinalizeCompiledKernel(dml_device.Get(), provider, op.Get(), log_tag, result, dml_dtype))
        return result;

    DML_PERF_LOG(log_tag, " CompileBatchNormActivation OK"
                 " X=", SizesStr(x_padded),
                 " activation=", FusedActivationTypeName(activation));
    return result;
}

// ---------------------------------------------------------------------------
// Compile helper for InstanceNormalization
// (DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC, spatial axes only).
// scale and bias are pre-uploaded static inputs.
// ---------------------------------------------------------------------------
static OpActivationCompiledKernel CompileInstanceNormActivation(
    PluginDmlExecutionProviderImpl* provider,
    float                           epsilon,
    FusedActivationType             activation,
    float act_alpha, float act_beta, float act_gamma,
    DML_TENSOR_DATA_TYPE            dml_dtype,
    const std::vector<uint32_t>&    x_sizes,
    uint64_t                        x_bytes,
    const std::vector<uint32_t>&    scale_sizes,
    uint64_t                        scale_bytes,
    const std::vector<uint32_t>&    bias_sizes,
    uint64_t                        bias_bytes)
{
    OpActivationCompiledKernel result;
    const char* log_tag = "[OpActFusion/InstanceNorm]";

    if (x_sizes.size() < 2) {
        DML_PERF_LOG(log_tag, " CompileInstanceNormActivation: X rank < 2");
        return result;
    }

    ComPtr<IDMLDevice> dml_device;
    if (FAILED(provider->GetDmlDevice(dml_device.GetAddressOf()))) {
        DML_PERF_LOG(log_tag, " GetDmlDevice FAILED");
        return result;
    }

    size_t dml_rank = std::max(x_sizes.size(), size_t(4));
    std::vector<uint32_t> x_padded = PadToRank(x_sizes, dml_rank);
    uint32_t c = x_padded[1];
    size_t elem_bytes = (dml_dtype == DML_TENSOR_DATA_TYPE_FLOAT32) ? 4 : 2;

    auto size_to_bytes = [&](const std::vector<uint32_t>& s, uint64_t hint) -> uint64_t {
        if (hint > 0) return hint;
        size_t n = 1; for (auto v : s) n *= v;
        return (static_cast<uint64_t>(n * elem_bytes) + 3u) & ~uint64_t(3u);
    };

    struct TDS {
        std::vector<uint32_t> sizes, strides_vec;
        uint64_t total_bytes = 0;
        DML_BUFFER_TENSOR_DESC buf{};
        DML_TENSOR_DESC td{};
    };
    auto make_packed = [&](const std::vector<uint32_t>& szs, uint64_t bytes) -> TDS {
        TDS s;
        s.sizes = szs;
        s.total_bytes = bytes;
        s.buf.DataType           = dml_dtype;
        s.buf.Flags              = DML_TENSOR_FLAG_NONE;
        s.buf.DimensionCount     = static_cast<UINT>(szs.size());
        s.buf.Sizes              = s.sizes.data();
        s.buf.Strides            = nullptr;
        s.buf.TotalTensorSizeInBytes = bytes;
        s.buf.GuaranteedBaseOffsetAlignment = 0;
        s.td = { DML_TENSOR_TYPE_BUFFER, &s.buf };
        return s;
    };

    TDS x_desc   = make_packed(x_padded, size_to_bytes(x_padded, x_bytes));
    TDS out_desc = make_packed(x_padded, size_to_bytes(x_padded, x_bytes));

    // Scale/Bias: [1, C, 1, 1, ...] with stride-0 broadcast on non-C dims.
    auto make_1d_broadcast = [&](uint64_t bytes) -> TDS {
        TDS s;
        s.sizes = std::vector<uint32_t>(dml_rank, 1u);
        s.sizes[1] = c;
        s.strides_vec.resize(dml_rank, 0u);
        s.strides_vec[1] = 1u;
        s.total_bytes = bytes;
        s.buf.DataType           = dml_dtype;
        s.buf.Flags              = DML_TENSOR_FLAG_NONE;
        s.buf.DimensionCount     = static_cast<UINT>(dml_rank);
        s.buf.Sizes              = s.sizes.data();
        s.buf.Strides            = s.strides_vec.data();
        s.buf.TotalTensorSizeInBytes = bytes;
        s.buf.GuaranteedBaseOffsetAlignment = 0;
        s.td = { DML_TENSOR_TYPE_BUFFER, &s.buf };
        return s;
    };

    TDS scale_desc = make_1d_broadcast(size_to_bytes(scale_sizes, scale_bytes));
    TDS bias_desc  = make_1d_broadcast(size_to_bytes(bias_sizes,  bias_bytes));

    // Axes: reduce spatial dims [2, 3, ..., dml_rank-1].
    std::vector<uint32_t> axes;
    for (uint32_t i = 2; i < static_cast<uint32_t>(dml_rank); ++i) axes.push_back(i);

    ActivationDescStorage act_storage = BuildActivationDesc(activation, act_alpha, act_beta, act_gamma);

    DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC mvn_desc{};
    mvn_desc.InputTensor       = &x_desc.td;
    mvn_desc.ScaleTensor       = &scale_desc.td;
    mvn_desc.BiasTensor        = &bias_desc.td;
    mvn_desc.OutputTensor      = &out_desc.td;
    mvn_desc.Axes              = axes.data();
    mvn_desc.AxisCount         = static_cast<UINT>(axes.size());
    mvn_desc.NormalizeVariance = TRUE;
    mvn_desc.Epsilon           = epsilon;
    mvn_desc.FusedActivation   = act_storage.valid ? &act_storage.desc : nullptr;
    DML_OPERATOR_DESC op_desc{ DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1, &mvn_desc };

    ComPtr<IDMLOperator> op;
    if (FAILED(dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op)))) {
        DML_PERF_LOG(log_tag, " CreateOperator FAILED");
        return result;
    }
    if (!FinalizeCompiledKernel(dml_device.Get(), provider, op.Get(), log_tag, result, dml_dtype))
        return result;

    DML_PERF_LOG(log_tag, " CompileInstanceNormActivation OK"
                 " X=", SizesStr(x_padded),
                 " activation=", FusedActivationTypeName(activation));
    return result;
}

// ---------------------------------------------------------------------------
// Compile helper for MeanVarianceNormalization
// (DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC, no scale/bias inputs).
// axes are resolved at Compile time from ONNX attributes.
// ---------------------------------------------------------------------------
static OpActivationCompiledKernel CompileMVNActivation(
    PluginDmlExecutionProviderImpl* provider,
    const std::vector<uint32_t>&    axes,   // already resolved (DML UINT indices)
    bool                            normalize_variance,
    FusedActivationType             activation,
    float act_alpha, float act_beta, float act_gamma,
    DML_TENSOR_DATA_TYPE            dml_dtype,
    const std::vector<uint32_t>&    x_sizes,
    uint64_t                        x_bytes)
{
    OpActivationCompiledKernel result;
    const char* log_tag = "[OpActFusion/MVN]";

    if (x_sizes.empty()) {
        DML_PERF_LOG(log_tag, " CompileMVNActivation: empty X");
        return result;
    }

    ComPtr<IDMLDevice> dml_device;
    if (FAILED(provider->GetDmlDevice(dml_device.GetAddressOf()))) {
        DML_PERF_LOG(log_tag, " GetDmlDevice FAILED");
        return result;
    }

    size_t dml_rank = std::max(x_sizes.size(), size_t(4));
    std::vector<uint32_t> x_padded = PadToRank(x_sizes, dml_rank);
    size_t elem_bytes = (dml_dtype == DML_TENSOR_DATA_TYPE_FLOAT32) ? 4 : 2;

    auto size_to_bytes = [&](const std::vector<uint32_t>& s, uint64_t hint) -> uint64_t {
        if (hint > 0) return hint;
        size_t n = 1; for (auto v : s) n *= v;
        return (static_cast<uint64_t>(n * elem_bytes) + 3u) & ~uint64_t(3u);
    };

    struct TDS {
        std::vector<uint32_t> sizes, strides_vec;
        uint64_t total_bytes = 0;
        DML_BUFFER_TENSOR_DESC buf{};
        DML_TENSOR_DESC td{};
    };
    auto make_packed = [&](const std::vector<uint32_t>& szs, uint64_t bytes) -> TDS {
        TDS s;
        s.sizes = szs;
        s.total_bytes = bytes;
        s.buf.DataType           = dml_dtype;
        s.buf.Flags              = DML_TENSOR_FLAG_NONE;
        s.buf.DimensionCount     = static_cast<UINT>(szs.size());
        s.buf.Sizes              = s.sizes.data();
        s.buf.Strides            = nullptr;
        s.buf.TotalTensorSizeInBytes = bytes;
        s.buf.GuaranteedBaseOffsetAlignment = 0;
        s.td = { DML_TENSOR_TYPE_BUFFER, &s.buf };
        return s;
    };

    TDS x_desc   = make_packed(x_padded, size_to_bytes(x_padded, x_bytes));
    TDS out_desc = make_packed(x_padded, size_to_bytes(x_padded, x_bytes));

    ActivationDescStorage act_storage = BuildActivationDesc(activation, act_alpha, act_beta, act_gamma);

    DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC mvn_desc{};
    mvn_desc.InputTensor       = &x_desc.td;
    mvn_desc.ScaleTensor       = nullptr;
    mvn_desc.BiasTensor        = nullptr;
    mvn_desc.OutputTensor      = &out_desc.td;
    mvn_desc.Axes              = axes.data();
    mvn_desc.AxisCount         = static_cast<UINT>(axes.size());
    mvn_desc.NormalizeVariance = normalize_variance ? TRUE : FALSE;
    mvn_desc.Epsilon           = 1e-5f;
    mvn_desc.FusedActivation   = act_storage.valid ? &act_storage.desc : nullptr;
    DML_OPERATOR_DESC op_desc{ DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1, &mvn_desc };

    ComPtr<IDMLOperator> op;
    if (FAILED(dml_device->CreateOperator(&op_desc, IID_PPV_ARGS(&op)))) {
        DML_PERF_LOG(log_tag, " CreateOperator FAILED");
        return result;
    }
    if (!FinalizeCompiledKernel(dml_device.Get(), provider, op.Get(), log_tag, result, dml_dtype))
        return result;

    DML_PERF_LOG(log_tag, " CompileMVNActivation OK"
                 " X=", SizesStr(x_padded),
                 " activation=", FusedActivationTypeName(activation));
    return result;
}

// ---------------------------------------------------------------------------
// Compute callback — shared for Add, Sum, Gemm (dispatched via base_op_kind).
// ---------------------------------------------------------------------------

static OrtStatus* ORT_API_CALL OpActivation_Compute(
    OrtNodeComputeInfo* /*this_ptr*/,
    void* compute_state,
    OrtKernelContext* kernel_context) noexcept
{
    auto* state = static_cast<OpActivationKernelState*>(compute_state);
    if (!state || !state->provider || !state->ort_api) return nullptr;
    const OrtApi& api = *state->ort_api;

    // Multi-input op path: Conv, ConvTranspose, BatchNorm, InstanceNorm, MVN.
    // These use dyn_input_indices / static_gpu_resources instead of the 2-input path below.
    if (state->base_op_kind == BaseOpKind::Conv        ||
        state->base_op_kind == BaseOpKind::ConvTranspose ||
        state->base_op_kind == BaseOpKind::BatchNorm   ||
        state->base_op_kind == BaseOpKind::InstanceNorm ||
        state->base_op_kind == BaseOpKind::MVN)
    {
        size_t num_slots = state->dyn_input_indices.size();

        // Fetch runtime input values (slot 0 = X is always runtime for these ops).
        std::vector<const OrtValue*> slot_values(num_slots, nullptr);
        for (size_t i = 0; i < num_slots; ++i) {
            size_t gi = state->dyn_input_indices[i];
            if (gi == SIZE_MAX) continue;  // static initializer slot
            OrtStatus* st = api.KernelContext_GetInput(kernel_context, gi, &slot_values[i]);
            if (st || !slot_values[i]) {
                if (st) api.ReleaseStatus(st);
                return api.CreateStatus(ORT_FAIL, "OpActFusion: failed to get runtime input");
            }
        }

        // Read X shape (slot 0 is always the runtime data input).
        DML_TENSOR_DATA_TYPE dml_dtype = DML_TENSOR_DATA_TYPE_UNKNOWN;
        std::vector<uint32_t> x_sizes;
        uint64_t x_bytes = 0;
        if (!slot_values[0] || !ReadRuntimeShape(api, slot_values[0], dml_dtype, x_sizes, x_bytes))
            return api.CreateStatus(ORT_FAIL, "OpActFusion: failed to read X shape");

        // Resolve per-slot shape/bytes for every input. A slot is either:
        //   dynamic  (dyn_input_indices[s] != SIZE_MAX) — shape from the runtime
        //            OrtValue in slot_values[s]; or
        //   constant (SIZE_MAX) — shape pre-computed in static_sizes/static_bytes.
        // Compilation, the shape-change key, and binding all read through these.
        auto slot_sizes = [&](size_t s) -> const std::vector<uint32_t>& {
            static const std::vector<uint32_t> empty;
            if (s >= num_slots) return empty;
            if (state->dyn_input_indices[s] != SIZE_MAX) return state->runtime_slot_sizes[s];
            return state->static_sizes[s];
        };
        auto slot_bytes = [&](size_t s) -> uint64_t {
            if (s >= num_slots) return 0;
            if (state->dyn_input_indices[s] != SIZE_MAX) return state->runtime_slot_bytes[s];
            return state->static_bytes[s];
        };

        // Read runtime shapes for all dynamic non-X slots (X already read above).
        state->runtime_slot_sizes.assign(num_slots, {});
        state->runtime_slot_bytes.assign(num_slots, 0);
        state->runtime_slot_sizes[0] = x_sizes;
        state->runtime_slot_bytes[0] = x_bytes;
        for (size_t s = 1; s < num_slots; ++s) {
            if (state->dyn_input_indices[s] == SIZE_MAX) continue;  // constant
            DML_TENSOR_DATA_TYPE sd = DML_TENSOR_DATA_TYPE_UNKNOWN;
            if (!slot_values[s] ||
                !ReadRuntimeShape(api, slot_values[s], sd,
                                  state->runtime_slot_sizes[s], state->runtime_slot_bytes[s]))
                return api.CreateStatus(ORT_FAIL, "OpActFusion: failed to read dynamic weight shape");
        }

        // Lazy-init / shape-change recompile.
        auto do_compile = [&]() -> bool {
            size_t elem_sz = (dml_dtype == DML_TENSOR_DATA_TYPE_FLOAT32) ? 4 : 2;
            auto byte_size = [&](const std::vector<uint32_t>& s) -> uint64_t {
                size_t n = 1; for (auto v : s) n *= v;
                return ((static_cast<uint64_t>(n * elem_sz) + 3u) & ~uint64_t(3u));
            };

            if (state->base_op_kind == BaseOpKind::Conv ||
                state->base_op_kind == BaseOpKind::ConvTranspose) {
                const std::vector<uint32_t>& w_sizes = slot_sizes(1);
                uint64_t w_bytes = slot_bytes(1);
                bool has_bias = (num_slots > 2 && !slot_sizes(2).empty() && slot_bytes(2) > 0);
                state->kernel = CompileConvActivation(
                    state->provider,
                    state->base_op_kind == BaseOpKind::ConvTranspose,
                    state->conv_kernel_shape, state->conv_strides,
                    state->conv_dilations, state->conv_pads,
                    state->conv_output_padding, state->conv_output_shape,
                    state->conv_auto_pad, state->conv_group,
                    state->activation, state->act_alpha, state->act_beta, state->act_gamma,
                    dml_dtype, x_sizes, x_bytes,
                    w_sizes, w_bytes,
                    has_bias ? &slot_sizes(2) : nullptr,
                    has_bias ? slot_bytes(2) : 0);
            } else if (state->base_op_kind == BaseOpKind::BatchNorm) {
                state->kernel = CompileBatchNormActivation(
                    state->provider,
                    state->norm_epsilon, state->bn_spatial,
                    state->activation, state->act_alpha, state->act_beta, state->act_gamma,
                    dml_dtype, x_sizes, x_bytes,
                    slot_sizes(1), slot_bytes(1),
                    slot_sizes(2), slot_bytes(2),
                    slot_sizes(3), slot_bytes(3),
                    slot_sizes(4), slot_bytes(4));
            } else if (state->base_op_kind == BaseOpKind::InstanceNorm) {
                state->kernel = CompileInstanceNormActivation(
                    state->provider,
                    state->norm_epsilon,
                    state->activation, state->act_alpha, state->act_beta, state->act_gamma,
                    dml_dtype, x_sizes, x_bytes,
                    slot_sizes(1), slot_bytes(1),
                    slot_sizes(2), slot_bytes(2));
            } else {  // MVN
                size_t dml_rank = std::max(x_sizes.size(), size_t(4));
                std::vector<uint32_t> resolved_axes;
                if (!state->mvn_axes.empty()) {
                    resolved_axes = ResolveMvnAxes(state->mvn_axes, x_sizes.size(), dml_rank);
                } else if (state->mvn_across_channels) {
                    for (uint32_t i = 0; i < static_cast<uint32_t>(dml_rank); ++i) resolved_axes.push_back(i);
                } else {
                    resolved_axes.push_back(0);
                    for (uint32_t i = 2; i < static_cast<uint32_t>(dml_rank); ++i) resolved_axes.push_back(i);
                }
                state->kernel = CompileMVNActivation(
                    state->provider,
                    resolved_axes,
                    state->mvn_normalize_variance,
                    state->activation, state->act_alpha, state->act_beta, state->act_gamma,
                    dml_dtype, x_sizes, x_bytes);
            }
            return state->kernel.IsValid();
        };

        // Snapshot of every input slot's shape — the recompile key. Covers X and
        // any dynamic weight slots so a shape change on any input is detected.
        auto current_slot_sizes = [&]() -> std::vector<std::vector<uint32_t>> {
            std::vector<std::vector<uint32_t>> snap(num_slots);
            for (size_t s = 0; s < num_slots; ++s) snap[s] = slot_sizes(s);
            return snap;
        };

        if (!state->initialized) {
            std::lock_guard<std::mutex> lock(state->init_mutex);
            if (!state->initialized) {
                DML_PERF_LOG("[OpActFusion/", state->base_op_kind == BaseOpKind::Conv ? "Conv" :
                             state->base_op_kind == BaseOpKind::ConvTranspose ? "ConvTranspose" :
                             state->base_op_kind == BaseOpKind::BatchNorm ? "BatchNorm" :
                             state->base_op_kind == BaseOpKind::InstanceNorm ? "InstanceNorm" : "MVN",
                             "] Compute: lazy-init X=", SizesStr(x_sizes));
                if (!do_compile())
                    return api.CreateStatus(ORT_FAIL, "OpActFusion: DML compilation failed");
                state->compiled_a_sizes = x_sizes;
                state->compiled_slot_sizes = current_slot_sizes();
                state->initialized = true;
            }
        }

        const OpActivationCompiledKernel* active = &state->kernel;
        OpActivationCompiledKernel temp_mi;
        if (current_slot_sizes() != state->compiled_slot_sizes) {
            // Shape changed on X or a dynamic weight — recompile temporary kernel.
            auto saved = state->kernel;
            if (!do_compile()) {
                state->kernel = saved;
                return api.CreateStatus(ORT_FAIL, "OpActFusion: shape-change recompile failed");
            }
            temp_mi = std::move(state->kernel);
            state->kernel = saved;
            active = &temp_mi;
        }

        // Compute output shape.
        std::vector<int64_t> out_dims;
        if (state->base_op_kind == BaseOpKind::Conv ||
            state->base_op_kind == BaseOpKind::ConvTranspose)
        {
            // Compute output shape in the ORIGINAL rank (not DML-padded).
            // DML padding to 2D spatial dims is internal to CompileConvActivation;
            // ORT expects the output shape to match the original Conv semantics.
            const std::vector<uint32_t>& w_sizes = slot_sizes(1);
            size_t spatial_dims = x_sizes.size() >= 2 ? x_sizes.size() - 2 : 0;
            if (spatial_dims == 0) spatial_dims = 1;

            uint32_t c_out = (state->base_op_kind == BaseOpKind::ConvTranspose)
                ? w_sizes[1] * state->conv_group
                : w_sizes[0];

            auto fill_def = [&](const std::vector<uint32_t>& src, uint32_t def) -> std::vector<uint32_t> {
                return src.empty() ? std::vector<uint32_t>(spatial_dims, def) : src;
            };
            auto strides   = fill_def(state->conv_strides, 1u);
            auto dilations = fill_def(state->conv_dilations, 1u);
            std::vector<uint32_t> sp(spatial_dims, 0u), ep(spatial_dims, 0u);
            bool is_same_upper_c = (state->conv_auto_pad == "SAME_UPPER");
            bool is_same_lower_c = (state->conv_auto_pad == "SAME_LOWER");
            bool use_auto_pad_c  = is_same_upper_c || is_same_lower_c;
            if (!use_auto_pad_c && state->conv_pads.size() == spatial_dims * 2) {
                for (size_t i = 0; i < spatial_dims; ++i) {
                    sp[i] = state->conv_pads[i];
                    ep[i] = state->conv_pads[spatial_dims + i];
                }
            }
            std::vector<uint32_t> op2(spatial_dims, 0u);
            if (state->conv_output_padding.size() == spatial_dims) op2 = state->conv_output_padding;
            while (strides.size() < spatial_dims) strides.push_back(1u);
            while (dilations.size() < spatial_dims) dilations.push_back(1u);

            out_dims.push_back(static_cast<int64_t>(x_sizes[0]));
            out_dims.push_back(static_cast<int64_t>(c_out));
            for (size_t i = 0; i < spatial_dims; ++i) {
                uint32_t x_s = x_sizes[2 + i];
                uint32_t k_s = (state->conv_kernel_shape.size() > i)
                    ? state->conv_kernel_shape[i]
                    : (w_sizes.size() > 2 + i ? w_sizes[2 + i] : 1u);
                if (state->base_op_kind == BaseOpKind::ConvTranspose) {
                    uint32_t spatial_out;
                    if (!state->conv_output_shape.empty()) {
                        // output_shape attr takes priority.
                        size_t off = state->conv_output_shape.size() - spatial_dims;
                        spatial_out = state->conv_output_shape[off + i];
                    } else {
                        spatial_out = (x_s - 1) * strides[i] - sp[i] - ep[i]
                            + dilations[i] * (k_s - 1) + 1 + op2[i];
                    }
                    out_dims.push_back(static_cast<int64_t>(spatial_out));
                } else {
                    uint32_t spatial_out;
                    if (use_auto_pad_c) {
                        spatial_out = (x_s + strides[i] - 1) / strides[i];
                    } else {
                        spatial_out = (x_s + sp[i] + ep[i] - dilations[i] * (k_s - 1) - 1)
                                      / strides[i] + 1;
                    }
                    out_dims.push_back(static_cast<int64_t>(spatial_out));
                }
            }
        } else {
            // BN / IN / MVN: output = X shape.
            for (auto v : x_sizes) out_dims.push_back(static_cast<int64_t>(v));
        }

        OrtValue* out_value = nullptr;
        {
            OrtStatus* st = api.KernelContext_GetOutput(
                kernel_context, 0, out_dims.data(), out_dims.size(), &out_value);
            if (st || !out_value) {
                if (st) api.ReleaseStatus(st);
                return api.CreateStatus(ORT_FAIL, "OpActFusion: failed to get output");
            }
        }

        auto get_res = [&](const OrtValue* v) -> ID3D12Resource* {
            void* raw = nullptr;
            OrtStatus* st = api.GetTensorMutableData(const_cast<OrtValue*>(v), &raw);
            if (st || !raw) { if (st) api.ReleaseStatus(st); return nullptr; }
            return state->provider->DecodeResource(raw);
        };

        // Resolve the GPU resource for an input slot: a dynamic slot binds the
        // runtime OrtValue directly (no upload — DML inputs are already GPU
        // resident); a constant slot binds its pre-uploaded static buffer.
        auto slot_res = [&](size_t s) -> ID3D12Resource* {
            if (s >= num_slots) return nullptr;
            if (state->dyn_input_indices[s] != SIZE_MAX)
                return slot_values[s] ? get_res(slot_values[s]) : nullptr;
            return state->static_gpu_resources[s].Get();
        };

        // Build DML input binding slots.
        // Ordering mirrors the DML operator descriptors:
        //   Conv:        [X, W, B]            (3 slots)
        //   BatchNorm:   [X, Mean, Var, Scale, Bias] (5 slots, DML ordering)
        //   InstanceNorm:[X, Scale, Bias]     (3 slots)
        //   MVN:         [X]                  (1 slot)
        size_t elem_size_mi = (dml_dtype == DML_TENSOR_DATA_TYPE_FLOAT32) ? 4 : 2;
        size_t out_elem_count_mi = 1;
        for (auto d : out_dims) out_elem_count_mi *= static_cast<size_t>(d);
        uint64_t out_bytes_mi = static_cast<uint64_t>(out_elem_count_mi * elem_size_mi);

        ID3D12Resource* out_res = get_res(out_value);
        if (!out_res)
            return api.CreateStatus(ORT_FAIL, "OpActFusion: failed to get output D3D12 resource");

        DML_BUFFER_BINDING out_binding{ out_res, 0, out_bytes_mi };
        DML_BINDING_DESC output_desc{ DML_BINDING_TYPE_BUFFER, &out_binding };

        HRESULT hr = E_FAIL;
        const DML_BUFFER_BINDING* persistent =
            active->persistent_binding ? &*active->persistent_binding : nullptr;

        if (state->base_op_kind == BaseOpKind::Conv ||
            state->base_op_kind == BaseOpKind::ConvTranspose) {
            // 3 inputs: [X, W, B]
            ID3D12Resource* x_res = get_res(slot_values[0]);
            ID3D12Resource* w_res = slot_res(1);
            bool has_bias_slot = (num_slots > 2 && !slot_sizes(2).empty() && slot_bytes(2) > 0);
            ID3D12Resource* b_res = has_bias_slot ? slot_res(2) : nullptr;

            DML_BUFFER_BINDING x_bind{ x_res, 0, x_bytes };
            DML_BUFFER_BINDING w_bind{ w_res, 0, slot_bytes(1) };
            DML_BUFFER_BINDING b_bind{};
            if (has_bias_slot) b_bind = { b_res, 0, slot_bytes(2) };
            DML_BINDING_DESC b_bind_desc = has_bias_slot
                ? DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, &b_bind }
                : DML_BINDING_DESC{ DML_BINDING_TYPE_NONE, nullptr };
            if (!x_bind.Buffer || !w_bind.Buffer) {
                DML_PERF_LOG("[OpActFusion/Conv] Compute: MISSING resource x=", (uintptr_t)x_res, " w=", (uintptr_t)w_res);
                return api.CreateStatus(ORT_FAIL, "OpActFusion/Conv: missing input resource");
            }
            DML_BINDING_DESC input_descs[3] = {
                { DML_BINDING_TYPE_BUFFER, &x_bind },
                { DML_BINDING_TYPE_BUFFER, &w_bind },
                b_bind_desc,
            };

            hr = state->provider->ExecuteOperator(
                active->compiled_op.Get(), persistent,
                gsl::make_span(input_descs, 3),
                gsl::make_span(&output_desc, 1));
            if (FAILED(hr))
                return api.CreateStatus(ORT_FAIL, "OpActFusion/Conv: ExecuteOperator failed");

        } else if (state->base_op_kind == BaseOpKind::BatchNorm) {
            // DML order: [InputTensor, MeanTensor, VarianceTensor, ScaleTensor, BiasTensor]
            // ONNX slots: 0=X, 1=scale, 2=bias, 3=mean, 4=var
            DML_BUFFER_BINDING x_bind     { get_res(slot_values[0]), 0, x_bytes };
            DML_BUFFER_BINDING mean_bind  { slot_res(3), 0, slot_bytes(3) };
            DML_BUFFER_BINDING var_bind   { slot_res(4), 0, slot_bytes(4) };
            DML_BUFFER_BINDING scale_bind { slot_res(1), 0, slot_bytes(1) };
            DML_BUFFER_BINDING bias_bind  { slot_res(2), 0, slot_bytes(2) };
            if (!x_bind.Buffer || !mean_bind.Buffer || !var_bind.Buffer ||
                !scale_bind.Buffer || !bias_bind.Buffer)
                return api.CreateStatus(ORT_FAIL, "OpActFusion/BatchNorm: missing input resource");
            DML_BINDING_DESC input_descs[5] = {
                { DML_BINDING_TYPE_BUFFER, &x_bind     },
                { DML_BINDING_TYPE_BUFFER, &mean_bind  },
                { DML_BINDING_TYPE_BUFFER, &var_bind   },
                { DML_BINDING_TYPE_BUFFER, &scale_bind },
                { DML_BINDING_TYPE_BUFFER, &bias_bind  },
            };
            hr = state->provider->ExecuteOperator(
                active->compiled_op.Get(), persistent,
                gsl::make_span(input_descs, 5),
                gsl::make_span(&output_desc, 1));
            if (FAILED(hr))
                return api.CreateStatus(ORT_FAIL, "OpActFusion/BatchNorm: ExecuteOperator failed");

        } else if (state->base_op_kind == BaseOpKind::InstanceNorm) {
            // 3 inputs: [X, Scale, Bias]
            DML_BUFFER_BINDING x_bind     { get_res(slot_values[0]), 0, x_bytes };
            DML_BUFFER_BINDING scale_bind { slot_res(1), 0, slot_bytes(1) };
            DML_BUFFER_BINDING bias_bind  { slot_res(2), 0, slot_bytes(2) };
            if (!x_bind.Buffer || !scale_bind.Buffer || !bias_bind.Buffer)
                return api.CreateStatus(ORT_FAIL, "OpActFusion/InstanceNorm: missing input resource");
            DML_BINDING_DESC input_descs[3] = {
                { DML_BINDING_TYPE_BUFFER, &x_bind     },
                { DML_BINDING_TYPE_BUFFER, &scale_bind },
                { DML_BINDING_TYPE_BUFFER, &bias_bind  },
            };
            hr = state->provider->ExecuteOperator(
                active->compiled_op.Get(), persistent,
                gsl::make_span(input_descs, 3),
                gsl::make_span(&output_desc, 1));
            if (FAILED(hr))
                return api.CreateStatus(ORT_FAIL, "OpActFusion/InstanceNorm: ExecuteOperator failed");

        } else {  // MVN
            // 1 input: [X]
            DML_BUFFER_BINDING x_bind{ get_res(slot_values[0]), 0, x_bytes };
            if (!x_bind.Buffer)
                return api.CreateStatus(ORT_FAIL, "OpActFusion/MVN: missing X resource");
            DML_BINDING_DESC input_descs[1] = {
                { DML_BINDING_TYPE_BUFFER, &x_bind },
            };
            hr = state->provider->ExecuteOperator(
                active->compiled_op.Get(), persistent,
                gsl::make_span(input_descs, 1),
                gsl::make_span(&output_desc, 1));
            if (FAILED(hr))
                return api.CreateStatus(ORT_FAIL, "OpActFusion/MVN: ExecuteOperator failed");
        }

        state->provider->QueueReference(active->compiled_op.Get());
        return nullptr;
    }

    // Read runtime inputs. When one side is a constant initializer its
    // input_idx is SIZE_MAX — skip GetInput for that side.
    const OrtValue* a_value = nullptr;
    if (state->initializer_side != 0) {
        OrtStatus* st = api.KernelContext_GetInput(
            kernel_context, state->input_idx_a, &a_value);
        if (st || !a_value) {
            if (st) api.ReleaseStatus(st);
            return api.CreateStatus(ORT_FAIL, "OpActFusion: failed to get input A");
        }
    }

    const OrtValue* b_value = nullptr;
    if (state->initializer_side != 1) {
        OrtStatus* st = api.KernelContext_GetInput(
            kernel_context, state->input_idx_b, &b_value);
        if (st || !b_value) {
            if (st) api.ReleaseStatus(st);
            return api.CreateStatus(ORT_FAIL, "OpActFusion: failed to get input B");
        }
    }

    // Read optional C (Gemm bias).
    const OrtValue* c_value = nullptr;
    if (state->base_op_kind == BaseOpKind::Gemm && state->input_idx_c != SIZE_MAX) {
        OrtStatus* st = api.KernelContext_GetInput(
            kernel_context, state->input_idx_c, &c_value);
        if (st) api.ReleaseStatus(st);
    }

    // Read runtime shapes. For the initializer side, use the shape captured at Compile time.
    DML_TENSOR_DATA_TYPE dml_dtype = DML_TENSOR_DATA_TYPE_UNKNOWN;
    std::vector<uint32_t> a_sizes, b_sizes;
    uint64_t a_bytes = 0, b_bytes = 0;

    DML_TENSOR_DATA_TYPE dummy_dtype = DML_TENSOR_DATA_TYPE_UNKNOWN;
    if (state->initializer_side == 0) {
        // A is an initializer — use its pre-computed shape.
        a_sizes = state->initializer_sizes;
        a_bytes = state->initializer_bytes;
        dml_dtype = state->initializer_dml_dtype;
        if (!ReadRuntimeShape(api, b_value, dummy_dtype, b_sizes, b_bytes))
            return api.CreateStatus(ORT_FAIL, "OpActFusion: failed to read shape B");
    } else if (state->initializer_side == 1) {
        // B is an initializer — use its pre-computed shape.
        b_sizes = state->initializer_sizes;
        b_bytes = state->initializer_bytes;
        if (!ReadRuntimeShape(api, a_value, dml_dtype, a_sizes, a_bytes))
            return api.CreateStatus(ORT_FAIL, "OpActFusion: failed to read shape A");
    } else {
        if (!ReadRuntimeShape(api, a_value, dml_dtype, a_sizes, a_bytes))
            return api.CreateStatus(ORT_FAIL, "OpActFusion: failed to read shape A");
        if (!ReadRuntimeShape(api, b_value, dummy_dtype, b_sizes, b_bytes))
            return api.CreateStatus(ORT_FAIL, "OpActFusion: failed to read shape B");
    }

    std::vector<uint32_t> c_sizes;
    uint64_t c_bytes = 0;
    if (c_value) {
        DML_TENSOR_DATA_TYPE c_dtype = DML_TENSOR_DATA_TYPE_UNKNOWN;
        ReadRuntimeShape(api, c_value, c_dtype, c_sizes, c_bytes);
    }

    // Lazy-init: compile on first Compute with real shapes.
    if (!state->initialized) {
        std::lock_guard<std::mutex> lock(state->init_mutex);
        if (!state->initialized) {
            DML_PERF_LOG("[OpActFusion] Compute: lazy-init"
                         " activation=", FusedActivationTypeName(state->activation),
                         " dtype=", DtypeStr(dml_dtype),
                         " A=", SizesStr(a_sizes), " B=", SizesStr(b_sizes));
            if (state->base_op_kind == BaseOpKind::Add ||
                state->base_op_kind == BaseOpKind::Sum) {
                state->kernel = CompileAddActivation(
                    state->provider, state->activation,
                    state->act_alpha, state->act_beta, state->act_gamma,
                    dml_dtype, a_sizes, b_sizes, a_bytes, b_bytes);
            } else {  // Gemm
                state->kernel = CompileGemmActivation(
                    state->provider,
                    state->trans_a, state->trans_b,
                    state->gemm_alpha, state->gemm_beta,
                    state->activation,
                    state->act_alpha, state->act_beta, state->act_gamma,
                    dml_dtype, a_sizes, b_sizes, a_bytes, b_bytes,
                    c_value ? &c_sizes : nullptr, c_bytes);
            }
            if (!state->kernel.IsValid()) {
                DML_PERF_LOG("[OpActFusion] Compute: lazy-init FAILED");
                return api.CreateStatus(ORT_FAIL, "OpActFusion: DML compilation failed");
            }
            state->compiled_a_sizes = a_sizes;
            state->compiled_b_sizes = b_sizes;
            state->initialized = true;
            DML_PERF_LOG("[OpActFusion] Compute: lazy-init SUCCESS");
        }
    }

    // Shape-change: recompile a temporary kernel if sizes differ.
    const OpActivationCompiledKernel* active = &state->kernel;
    OpActivationCompiledKernel temp;
    if (a_sizes != state->compiled_a_sizes || b_sizes != state->compiled_b_sizes) {
        DML_PERF_LOG("[OpActFusion] Compute: shape change"
                     " prev_A=", SizesStr(state->compiled_a_sizes),
                     " new_A=", SizesStr(a_sizes),
                     " prev_B=", SizesStr(state->compiled_b_sizes),
                     " new_B=", SizesStr(b_sizes));
        if (state->base_op_kind == BaseOpKind::Add ||
            state->base_op_kind == BaseOpKind::Sum) {
            temp = CompileAddActivation(
                state->provider, state->activation,
                state->act_alpha, state->act_beta, state->act_gamma,
                dml_dtype, a_sizes, b_sizes, a_bytes, b_bytes);
        } else {
            temp = CompileGemmActivation(
                state->provider,
                state->trans_a, state->trans_b,
                state->gemm_alpha, state->gemm_beta,
                state->activation,
                state->act_alpha, state->act_beta, state->act_gamma,
                dml_dtype, a_sizes, b_sizes, a_bytes, b_bytes,
                c_value ? &c_sizes : nullptr, c_bytes);
        }
        if (!temp.IsValid()) {
            DML_PERF_LOG("[OpActFusion] Compute: shape-change recompile FAILED");
            return api.CreateStatus(ORT_FAIL, "OpActFusion: temporary kernel compilation failed");
        }
        active = &temp;
    }

    // Allocate output tensor.
    std::vector<int64_t> out_dims;
    if (state->base_op_kind == BaseOpKind::Add ||
        state->base_op_kind == BaseOpKind::Sum) {
        // Broadcast output shape: right-align both inputs and take max per dim.
        size_t rank = std::max(a_sizes.size(), b_sizes.size());
        out_dims.resize(rank);
        for (size_t i = 0; i < rank; ++i) {
            // right-align: position i from the left maps to the i-th element
            // when prefixed with 1s to reach `rank`.
            size_t a_off = rank - a_sizes.size();  // how many 1s prepended to a
            size_t b_off = rank - b_sizes.size();
            uint32_t ai = (i >= a_off) ? a_sizes[i - a_off] : 1u;
            uint32_t bi = (i >= b_off) ? b_sizes[i - b_off] : 1u;
            out_dims[i] = static_cast<int64_t>(std::max(ai, bi));
        }
    } else {  // Gemm
        GemmShapes g = PrepareGemmShapes(a_sizes, b_sizes, state->trans_a, state->trans_b);
        if (!g.valid)
            return api.CreateStatus(ORT_FAIL, "OpActFusion/Gemm: invalid shapes at compute");
        // Strip 4D padding back to logical rank.
        size_t logical_rank = std::max(a_sizes.size(), b_sizes.size());
        if (logical_rank < 2) logical_rank = 2;
        size_t skip = g.c_sizes.size() > logical_rank ? g.c_sizes.size() - logical_rank : 0;
        out_dims.reserve(logical_rank);
        for (size_t i = skip; i < g.c_sizes.size(); ++i)
            out_dims.push_back(static_cast<int64_t>(g.c_sizes[i]));
    }

    OrtValue* out_value = nullptr;
    {
        OrtStatus* st = api.KernelContext_GetOutput(
            kernel_context, 0, out_dims.data(), out_dims.size(), &out_value);
        if (st || !out_value) {
            if (st) api.ReleaseStatus(st);
            return api.CreateStatus(ORT_FAIL, "OpActFusion: failed to get output");
        }
    }

    auto get_resource = [&](const OrtValue* v) -> ID3D12Resource* {
        void* raw = nullptr;
        OrtStatus* st = api.GetTensorMutableData(const_cast<OrtValue*>(v), &raw);
        if (st || !raw) { if (st) api.ReleaseStatus(st); return nullptr; }
        return state->provider->DecodeResource(raw);
    };

    // For the constant side, use the pre-uploaded GPU resource directly.
    ID3D12Resource* a_res = (state->initializer_side == 0)
        ? state->initializer_gpu_resource.Get()
        : get_resource(a_value);
    ID3D12Resource* b_res = (state->initializer_side == 1)
        ? state->initializer_gpu_resource.Get()
        : get_resource(b_value);
    ID3D12Resource* out_res = get_resource(out_value);
    if (!a_res || !b_res || !out_res)
        return api.CreateStatus(ORT_FAIL, "OpActFusion: failed to get D3D12 resources");

    // Byte size of output for binding.
    size_t out_elem_count = 1;
    for (auto d : out_dims) out_elem_count *= static_cast<size_t>(d);
    size_t elem_size = (dml_dtype == DML_TENSOR_DATA_TYPE_FLOAT32) ? 4 : 2;
    uint64_t out_bytes = static_cast<uint64_t>(out_elem_count * elem_size);

    if (state->base_op_kind == BaseOpKind::Add ||
        state->base_op_kind == BaseOpKind::Sum) {
        DML_BUFFER_BINDING a_binding{ a_res, 0, a_bytes };
        DML_BUFFER_BINDING b_binding{ b_res, 0, b_bytes };
        DML_BUFFER_BINDING o_binding{ out_res, 0, out_bytes };
        DML_BINDING_DESC input_descs[2] = {
            { DML_BINDING_TYPE_BUFFER, &a_binding },
            { DML_BINDING_TYPE_BUFFER, &b_binding },
        };
        DML_BINDING_DESC output_desc{ DML_BINDING_TYPE_BUFFER, &o_binding };

        const DML_BUFFER_BINDING* persistent =
            active->persistent_binding ? &*active->persistent_binding : nullptr;
        HRESULT hr = state->provider->ExecuteOperator(
            active->compiled_op.Get(), persistent,
            gsl::make_span(input_descs, 2),
            gsl::make_span(&output_desc, 1));
        if (FAILED(hr))
            return api.CreateStatus(ORT_FAIL, "OpActFusion/Add: ExecuteOperator failed");
    } else {
        // Gemm: 3 inputs (A, B, C) — DML GEMM always expects 3 binding slots.
        DML_BUFFER_BINDING a_binding{ a_res, 0, a_bytes };
        DML_BUFFER_BINDING b_binding{ b_res, 0, b_bytes };
        DML_BUFFER_BINDING o_binding{ out_res, 0, out_bytes };

        ID3D12Resource* c_res = c_value ? get_resource(c_value) : nullptr;
        DML_BUFFER_BINDING c_binding{};
        if (c_res) c_binding = { c_res, 0, c_bytes };

        DML_BINDING_DESC c_bind_desc = c_res
            ? DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, &c_binding }
            : DML_BINDING_DESC{ DML_BINDING_TYPE_NONE, nullptr };
        DML_BINDING_DESC input_descs[3] = {
            { DML_BINDING_TYPE_BUFFER, &a_binding },
            { DML_BINDING_TYPE_BUFFER, &b_binding },
            c_bind_desc,
        };
        DML_BINDING_DESC output_desc{ DML_BINDING_TYPE_BUFFER, &o_binding };

        const DML_BUFFER_BINDING* persistent =
            active->persistent_binding ? &*active->persistent_binding : nullptr;
        HRESULT hr = state->provider->ExecuteOperator(
            active->compiled_op.Get(), persistent,
            gsl::make_span(input_descs, 3),
            gsl::make_span(&output_desc, 1));
        if (FAILED(hr))
            return api.CreateStatus(ORT_FAIL, "OpActFusion/Gemm: ExecuteOperator failed");
    }

    state->provider->QueueReference(active->compiled_op.Get());
    return nullptr;
}

// ---------------------------------------------------------------------------
// OrtNodeComputeInfo wrapper
// ---------------------------------------------------------------------------
struct OpActivationNodeComputeInfo : OrtNodeComputeInfo {
    OpActivationKernelState* state = nullptr;

    OpActivationNodeComputeInfo() {
        ort_version_supported = ORT_API_VERSION;
        CreateState = [](OrtNodeComputeInfo* self, OrtNodeComputeContext*, void** out) noexcept -> OrtStatus* {
            *out = static_cast<OpActivationNodeComputeInfo*>(self)->state;
            return nullptr;
        };
        Compute      = OpActivation_Compute;
        ReleaseState = [](OrtNodeComputeInfo*, void*) noexcept {};
    }

    ~OpActivationNodeComputeInfo() { delete state; }
};

// ---------------------------------------------------------------------------
// Compile
// ---------------------------------------------------------------------------

OrtNodeComputeInfo* OpActivationFusionRule::Compile(
    const OrtApi&                                            ort_api,
    const OrtGraph*                                          fused_subgraph,
    const std::unordered_map<std::string, const OrtValue*>&  initializers,
    PluginDmlExecutionProviderImpl*                          provider,
    const PatternMatch&                                      match) const
{
    // Activation type and parameters captured in CapturePreFusionData.
    std::string act_type_str = match.ValueName(CaptureKey("act_type"));
    if (act_type_str.empty()) {
        DML_PERF_LOG("[OpActFusion/", m_config.base_op_type, "] Compile: no activation captured");
        return nullptr;
    }
    FusedActivationType activation = FusedActivationTypeFromOpType(act_type_str);
    if (activation == FusedActivationType::None) {
        DML_PERF_LOG("[OpActFusion/", m_config.base_op_type, "] Compile: unknown activation ", act_type_str);
        return nullptr;
    }
    float act_alpha = match.ScalarValue(CaptureKey("act_alpha"), 0.0f);
    float act_beta  = match.ScalarValue(CaptureKey("act_beta"),  0.0f);
    float act_gamma = match.ScalarValue(CaptureKey("act_gamma"), 0.0f);

    // Determine base op kind.
    BaseOpKind kind;
    if (m_config.base_op_type == "Add" || m_config.base_op_type == "Sum")
        kind = (m_config.base_op_type == "Add") ? BaseOpKind::Add : BaseOpKind::Sum;
    else if (m_config.base_op_type == "Gemm")
        kind = BaseOpKind::Gemm;
    else if (m_config.base_op_type == "Conv")
        kind = BaseOpKind::Conv;
    else if (m_config.base_op_type == "ConvTranspose")
        kind = BaseOpKind::ConvTranspose;
    else if (m_config.base_op_type == "BatchNormalization")
        kind = BaseOpKind::BatchNorm;
    else if (m_config.base_op_type == "InstanceNormalization")
        kind = BaseOpKind::InstanceNorm;
    else if (m_config.base_op_type == "MeanVarianceNormalization")
        kind = BaseOpKind::MVN;
    else {
        DML_PERF_LOG("[OpActFusion] Compile: unsupported base op ", m_config.base_op_type);
        return nullptr;
    }

    // Walk subgraph nodes to find the base op and read its attributes.
    size_t num_nodes = 0;
    ort_api.Graph_GetNumNodes(fused_subgraph, &num_nodes);
    std::vector<const OrtNode*> sg_nodes(num_nodes, nullptr);
    if (num_nodes > 0) ort_api.Graph_GetNodes(fused_subgraph, sg_nodes.data(), num_nodes);

    const OrtNode* base_node = nullptr;
    for (const OrtNode* n : sg_nodes) {
        const char* op = nullptr;
        ort_api.Node_GetOperatorType(n, &op);
        if (op && std::string_view(op) == m_config.base_op_type) {
            base_node = n;
            break;
        }
    }
    if (!base_node) {
        DML_PERF_LOG("[OpActFusion/", m_config.base_op_type, "] Compile: base node not found in subgraph");
        return nullptr;
    }

    // Read Gemm attributes if needed.
    bool  trans_a    = false;
    bool  trans_b    = false;
    float gemm_alpha = 1.0f;
    float gemm_beta  = 1.0f;
    if (kind == BaseOpKind::Gemm) {
        OrtNodeAdapter adapter(base_node, ort_api);
        trans_a    = adapter.GetAttributeInt("transA", 0) != 0;
        trans_b    = adapter.GetAttributeInt("transB", 0) != 0;
        gemm_alpha = adapter.GetAttributeFloat("alpha", 1.0f);
        gemm_beta  = adapter.GetAttributeFloat("beta",  1.0f);
    }

    // Read Conv / ConvTranspose attributes.
    std::vector<uint32_t> conv_kernel_shape, conv_strides, conv_dilations, conv_pads, conv_output_padding, conv_output_shape;
    std::string conv_auto_pad;
    uint32_t conv_group = 1;
    bool conv_has_dynamic_pads = false;
    if (kind == BaseOpKind::Conv || kind == BaseOpKind::ConvTranspose) {
        OrtNodeAdapter adapter(base_node, ort_api);
        auto to_u32 = [](const std::vector<int64_t>& v) {
            std::vector<uint32_t> out(v.size());
            for (size_t i = 0; i < v.size(); ++i) out[i] = static_cast<uint32_t>(v[i]);
            return out;
        };
        conv_kernel_shape   = to_u32(adapter.GetAttributeInts("kernel_shape"));
        conv_strides        = to_u32(adapter.GetAttributeInts("strides"));
        conv_dilations      = to_u32(adapter.GetAttributeInts("dilations"));
        conv_pads           = to_u32(adapter.GetAttributeInts("pads"));
        conv_output_padding = to_u32(adapter.GetAttributeInts("output_padding"));
        conv_output_shape   = to_u32(adapter.GetAttributeInts("output_shape")); // ConvTranspose explicit output
        conv_auto_pad       = adapter.GetAttributeString("auto_pad", "NOTSET");
        conv_group          = static_cast<uint32_t>(adapter.GetAttributeInt("group", 1));

        // Dynamic-pads Conv (opset 11): pads is input[2] instead of an attribute,
        // bias shifts to input[3]. Detect by checking for 4+ inputs with empty pads attr.
        size_t n_inputs = 0;
        ort_api.Node_GetNumInputs(base_node, &n_inputs);
        if (n_inputs >= 4 && conv_pads.empty()) {
            conv_has_dynamic_pads = true;
        }
    }

    // Read BatchNorm / InstanceNorm / MVN attributes.
    float norm_epsilon = 1e-5f;
    bool  bn_spatial   = true;
    bool  mvn_normalize_variance = true;
    bool  mvn_across_channels    = false;
    std::vector<int32_t> mvn_explicit_axes;
    if (kind == BaseOpKind::BatchNorm || kind == BaseOpKind::InstanceNorm) {
        OrtNodeAdapter adapter(base_node, ort_api);
        // ORT DML defaults: BatchNorm epsilon = 0.0f (DmlOperatorBatchNormalization.cpp),
        // InstanceNorm epsilon = 1e-5f (DmlOperatorInstanceNormalization.cpp).
        const float epsilon_default = (kind == BaseOpKind::BatchNorm) ? 0.0f : 1e-5f;
        norm_epsilon = adapter.GetAttributeFloat("epsilon", epsilon_default);
        if (kind == BaseOpKind::BatchNorm)
            bn_spatial = adapter.GetAttributeInt("spatial", 1) != 0;
    }
    if (kind == BaseOpKind::MVN) {
        OrtNodeAdapter adapter(base_node, ort_api);
        mvn_normalize_variance = adapter.GetAttributeInt("normalize_variance", 1) != 0;
        mvn_across_channels    = adapter.GetAttributeInt("across_channels", 0) != 0;
        auto raw_axes = adapter.GetAttributeInts("axes");
        for (auto v : raw_axes) mvn_explicit_axes.push_back(static_cast<int32_t>(v));
    }

    // Resolve graph input indices.  ORT may reorder fused subgraph inputs.
    // One input (typically the weight B) may be a constant initializer — it
    // won't appear as a graph input but will be in the initializers map.
    // Mirrors FusedMatMulFusionRule's initializer detection pattern.
    size_t num_inputs = 0;
    ort_api.Graph_GetNumInputs(fused_subgraph, &num_inputs);
    if (num_inputs < 1) {
        DML_PERF_LOG("[OpActFusion/", m_config.base_op_type, "] Compile: no graph inputs");
        return nullptr;
    }
    std::vector<const OrtValueInfo*> input_vis(num_inputs, nullptr);
    ort_api.Graph_GetInputs(fused_subgraph, input_vis.data(), num_inputs);

    auto read_name = [&](const OrtValueInfo* vi) -> std::string {
        if (!vi) return {};
        const char* name = nullptr;
        OrtStatus* st = ort_api.GetValueInfoName(vi, &name);
        std::string r = (st || !name) ? std::string{} : std::string(name);
        if (st) ort_api.ReleaseStatus(st);
        return r;
    };

    // Read base node input value names.
    size_t base_input_count = 0;
    ort_api.Node_GetNumInputs(base_node, &base_input_count);
    std::vector<const OrtValueInfo*> base_inputs(base_input_count, nullptr);
    if (base_input_count > 0)
        ort_api.Node_GetInputs(base_node, base_inputs.data(), base_input_count);

    auto find_graph_input = [&](const std::string& name) -> size_t {
        for (size_t gi = 0; gi < num_inputs; ++gi) {
            if (read_name(input_vis[gi]) == name) return gi;
        }
        return SIZE_MAX;
    };

    std::string name_a = (base_inputs.size() > 0) ? read_name(base_inputs[0]) : std::string{};
    std::string name_b = (base_inputs.size() > 1) ? read_name(base_inputs[1]) : std::string{};
    std::string name_c = (base_inputs.size() > 2) ? read_name(base_inputs[2]) : std::string{};

    size_t idx_a = find_graph_input(name_a);
    size_t idx_b = find_graph_input(name_b);
    size_t idx_c = name_c.empty() ? SIZE_MAX : find_graph_input(name_c);

    // Detect which side (if any) is a constant initializer.
    // An initializer won't appear as a graph input — idx will be SIZE_MAX but
    // the name will be present in the initializers map.
    int initializer_side = -1;
    size_t resolved_a_idx = 0;
    size_t resolved_b_idx = (num_inputs >= 2) ? 1 : 0;

    if (idx_a != SIZE_MAX && idx_b != SIZE_MAX) {
        resolved_a_idx = idx_a;
        resolved_b_idx = idx_b;
    } else if (idx_a != SIZE_MAX && idx_b == SIZE_MAX) {
        resolved_a_idx = idx_a;
        initializer_side = 1;  // B is a constant
    } else if (idx_a == SIZE_MAX && idx_b != SIZE_MAX) {
        resolved_b_idx = idx_b;
        initializer_side = 0;  // A is a constant
    } else {
        // Both are initializers — nothing to compute at runtime.
        DML_PERF_LOG("[OpActFusion/", m_config.base_op_type, "] Compile: both inputs are initializers");
        return nullptr;
    }

    DML_PERF_LOG("[OpActFusion/", m_config.base_op_type, "] Compile: graph inputs=", num_inputs,
                 " A=idx[", resolved_a_idx, "] B=idx[", resolved_b_idx, "]",
                 " initializer_side=", initializer_side);

    // Determine element type and static shape for eager compilation.
    auto get_dims = [&](const OrtValueInfo* vi, std::vector<int64_t>& dims_out) -> bool {
        if (!vi) return false;
        const OrtTypeInfo* ti = nullptr;
        OrtStatus* st = ort_api.GetValueInfoTypeInfo(vi, &ti);
        if (st || !ti) { if (st) ort_api.ReleaseStatus(st); return false; }
        const OrtTensorTypeAndShapeInfo* si = nullptr;
        ort_api.CastTypeInfoToTensorInfo(ti, &si);
        if (!si) return false;
        size_t r = 0; ort_api.GetDimensionsCount(si, &r);
        dims_out.assign(r, -1);
        if (r > 0) ort_api.GetDimensions(si, dims_out.data(), r);
        return true;
    };

    auto get_elem_type = [&](const OrtValueInfo* vi) -> ONNXTensorElementDataType {
        if (!vi) return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        const OrtTypeInfo* ti = nullptr;
        OrtStatus* st = ort_api.GetValueInfoTypeInfo(vi, &ti);
        if (st || !ti) { if (st) ort_api.ReleaseStatus(st); return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED; }
        const OrtTensorTypeAndShapeInfo* si = nullptr;
        ort_api.CastTypeInfoToTensorInfo(ti, &si);
        if (!si) return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        ONNXTensorElementDataType e = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        ort_api.GetTensorElementType(si, &e);
        return e;
    };

    // Determine element type from the runtime input side (not the initializer side).
    const OrtValueInfo* runtime_vi = (initializer_side == 0)
        ? input_vis[resolved_b_idx]
        : input_vis[resolved_a_idx];
    ONNXTensorElementDataType elem_type = get_elem_type(runtime_vi);
    DML_TENSOR_DATA_TYPE dml_dtype = DML_TENSOR_DATA_TYPE_UNKNOWN;
    switch (elem_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   dml_dtype = DML_TENSOR_DATA_TYPE_FLOAT32; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: dml_dtype = DML_TENSOR_DATA_TYPE_FLOAT16; break;
        default:
            DML_PERF_LOG("[OpActFusion/", m_config.base_op_type, "] Compile: unsupported dtype");
            return nullptr;
    }

    // Dims from graph inputs for runtime sides; dims from initializer for constant side.
    std::vector<int64_t> a_dims, b_dims, c_dims;
    if (initializer_side != 0) get_dims(input_vis[resolved_a_idx], a_dims);
    if (initializer_side != 1) get_dims(input_vis[resolved_b_idx], b_dims);
    if (idx_c != SIZE_MAX)     get_dims(input_vis[idx_c], c_dims);

    auto* kernel_state               = new OpActivationKernelState();
    kernel_state->base_op_kind       = kind;
    kernel_state->activation         = activation;
    kernel_state->act_alpha          = act_alpha;
    kernel_state->act_beta           = act_beta;
    kernel_state->act_gamma          = act_gamma;
    kernel_state->trans_a            = trans_a;
    kernel_state->trans_b            = trans_b;
    kernel_state->gemm_alpha         = gemm_alpha;
    kernel_state->gemm_beta          = gemm_beta;
    kernel_state->input_idx_a        = (initializer_side == 0) ? SIZE_MAX : resolved_a_idx;
    kernel_state->input_idx_b        = (initializer_side == 1) ? SIZE_MAX : resolved_b_idx;
    kernel_state->input_idx_c        = idx_c;
    kernel_state->initializer_side   = initializer_side;
    kernel_state->provider           = provider;
    kernel_state->ort_api            = &ort_api;

    // Store Conv / BatchNorm / InstanceNorm / MVN attributes.
    if (kind == BaseOpKind::Conv || kind == BaseOpKind::ConvTranspose) {
        kernel_state->conv_kernel_shape   = conv_kernel_shape;
        kernel_state->conv_strides        = conv_strides;
        kernel_state->conv_dilations      = conv_dilations;
        kernel_state->conv_pads           = conv_pads;
        kernel_state->conv_output_padding = conv_output_padding;
        kernel_state->conv_output_shape   = conv_output_shape;
        kernel_state->conv_auto_pad       = conv_auto_pad;
        kernel_state->conv_group          = conv_group;
    }
    if (kind == BaseOpKind::BatchNorm || kind == BaseOpKind::InstanceNorm) {
        kernel_state->norm_epsilon = norm_epsilon;
        kernel_state->bn_spatial   = bn_spatial;
    }
    if (kind == BaseOpKind::MVN) {
        kernel_state->mvn_normalize_variance = mvn_normalize_variance;
        kernel_state->mvn_across_channels    = mvn_across_channels;
        kernel_state->mvn_axes               = mvn_explicit_axes;
    }

    // Dynamic-pads Conv: input[2] is pads tensor, bias is input[3].
    // Extract pads from the initializer, then remap inputs to [X, W, B]
    // so the Compute path's hardcoded slot indices (W=1, B=2) still work.
    if (conv_has_dynamic_pads && base_input_count >= 4) {
        std::string pads_name = (base_inputs.size() > 2) ? read_name(base_inputs[2]) : std::string{};
        if (!pads_name.empty()) {
            auto pads_it = initializers.find(pads_name);
            if (pads_it != initializers.end() && pads_it->second) {
                void* pads_ptr = nullptr;
                OrtStatus* st = ort_api.GetTensorMutableData(
                    const_cast<OrtValue*>(pads_it->second), &pads_ptr);
                if (!st && pads_ptr) {
                    OrtTensorTypeAndShapeInfo* psi = nullptr;
                    ort_api.GetTensorTypeAndShape(const_cast<OrtValue*>(pads_it->second), &psi);
                    if (psi) {
                        size_t pads_count = 0;
                        ort_api.GetTensorShapeElementCount(psi, &pads_count);
                        ort_api.ReleaseTensorTypeAndShapeInfo(psi);
                        const int64_t* pads_data = static_cast<const int64_t*>(pads_ptr);
                        conv_pads.resize(pads_count);
                        for (size_t p = 0; p < pads_count; ++p)
                            conv_pads[p] = static_cast<uint32_t>(std::max<int64_t>(pads_data[p], 0));
                        kernel_state->conv_pads = conv_pads;
                    }
                }
                if (st) ort_api.ReleaseStatus(st);
            }
        }
        // Remove pads from base_inputs, shift bias from slot 3 to slot 2.
        if (base_inputs.size() >= 4) {
            base_inputs.erase(base_inputs.begin() + 2);
            base_input_count = base_inputs.size();
        }
        DML_PERF_LOG("[OpActFusion/Conv] Compile: dynamic pads extracted, pads=",
                     conv_pads.size(), " values, remapped to ", base_input_count, " inputs");
    }

    // For multi-input ops (Conv, BatchNorm, InstanceNorm, MVN): use the generic
    // per-slot initializer upload path instead of the legacy initializer_side approach.
    bool is_multi_input_op = (kind == BaseOpKind::Conv    || kind == BaseOpKind::ConvTranspose ||
                              kind == BaseOpKind::BatchNorm || kind == BaseOpKind::InstanceNorm ||
                              kind == BaseOpKind::MVN);

    size_t elem_sz_mi = (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) ? 2 : 4;
    if (is_multi_input_op) {
        kernel_state->dyn_input_indices.resize(base_input_count, SIZE_MAX);
        kernel_state->static_gpu_resources.resize(base_input_count);
        kernel_state->static_allocator_refs.resize(base_input_count);
        kernel_state->static_bytes.resize(base_input_count, 0);
        kernel_state->static_sizes.resize(base_input_count);
        kernel_state->static_dml_dtypes.resize(base_input_count, DML_TENSOR_DATA_TYPE_UNKNOWN);

        for (size_t slot = 0; slot < base_input_count; ++slot) {
            if (slot >= base_inputs.size() || !base_inputs[slot]) {
                kernel_state->dyn_input_indices[slot] = SIZE_MAX;
                continue;
            }
            std::string slot_name = read_name(base_inputs[slot]);
            if (slot_name.empty()) {
                kernel_state->dyn_input_indices[slot] = SIZE_MAX;
                continue;
            }
            size_t gi = find_graph_input(slot_name);
            auto init_it = initializers.find(slot_name);
            bool has_initializer = (init_it != initializers.end() && init_it->second != nullptr);

            if (gi != SIZE_MAX) {
                // Slot is a graph input: bind it directly from the runtime
                // OrtValue at Compute time (DML inputs are already GPU-resident,
                // so no upload is needed). This covers weights produced upstream
                // and overridable weights that appear as both an initializer and
                // a graph input — in the latter case, binding the runtime value
                // honors any caller-supplied override. Only a pure constant
                // (initializer, not a graph input) is uploaded below.
                kernel_state->dyn_input_indices[slot] = gi;
            } else if (has_initializer) {
                // Pure constant initializer (not a graph input): upload once to a
                // pooled GPU buffer; the Compute path binds static_gpu_resources.
                void* cpu_ptr = nullptr;
                OrtStatus* st = ort_api.GetTensorMutableData(
                    const_cast<OrtValue*>(init_it->second), &cpu_ptr);
                if (st || !cpu_ptr) {
                    if (st) ort_api.ReleaseStatus(st);
                    DML_PERF_LOG("[OpActFusion/", m_config.base_op_type,
                                 "] Compile: failed to get CPU data for slot ", slot);
                    delete kernel_state;
                    return nullptr;
                }
                OrtTensorTypeAndShapeInfo* si = nullptr;
                ort_api.GetTensorTypeAndShape(const_cast<OrtValue*>(init_it->second), &si);
                if (!si) {
                    DML_PERF_LOG("[OpActFusion/", m_config.base_op_type,
                                 "] Compile: failed to get shape for slot ", slot);
                    delete kernel_state;
                    return nullptr;
                }
                size_t elem_count = 0, rank = 0;
                ort_api.GetTensorShapeElementCount(si, &elem_count);
                ort_api.GetDimensionsCount(si, &rank);
                std::vector<int64_t> dims(rank);
                if (rank > 0) ort_api.GetDimensions(si, dims.data(), rank);
                ort_api.ReleaseTensorTypeAndShapeInfo(si);

                kernel_state->static_sizes[slot].resize(rank);
                for (size_t d = 0; d < rank; ++d)
                    kernel_state->static_sizes[slot][d] = static_cast<uint32_t>(dims[d] > 0 ? dims[d] : 1);
                kernel_state->static_bytes[slot] = ((static_cast<uint64_t>(elem_count * elem_sz_mi) + 3u) & ~uint64_t(3u));
                kernel_state->static_dml_dtypes[slot] = dml_dtype;

                HRESULT hr = provider->AllocatePooledResource(
                    static_cast<size_t>(kernel_state->static_bytes[slot]),
                    AllocatorRoundingMode::Disabled,
                    kernel_state->static_gpu_resources[slot].GetAddressOf(),
                    kernel_state->static_allocator_refs[slot].GetAddressOf());
                if (FAILED(hr)) {
                    DML_PERF_LOG("[OpActFusion/", m_config.base_op_type,
                                 "] Compile: AllocatePooledResource FAILED slot=", slot,
                                 " bytes=", kernel_state->static_bytes[slot]);
                    delete kernel_state;
                    return nullptr;
                }
                hr = provider->UploadToResource(
                    kernel_state->static_gpu_resources[slot].Get(), cpu_ptr,
                    kernel_state->static_bytes[slot]);
                if (FAILED(hr)) {
                    DML_PERF_LOG("[OpActFusion/", m_config.base_op_type,
                                 "] Compile: UploadToResource FAILED slot=", slot);
                    delete kernel_state;
                    return nullptr;
                }
                DML_PERF_LOG("[OpActFusion/", m_config.base_op_type,
                             "] Compile: uploaded slot=", slot, " name='", slot_name,
                             "' bytes=", kernel_state->static_bytes[slot]);
            } else {
                DML_PERF_LOG("[OpActFusion/", m_config.base_op_type,
                             "] Compile: slot ", slot, " name='", slot_name,
                             "' not in graph inputs or initializers");
                delete kernel_state;
                return nullptr;
            }
        }
    }

    // Upload constant initializer to GPU (mirrors FusedMatMulFusionRule).
    // Only used for the 2-input ops (Add, Sum, Gemm).
    if (!is_multi_input_op && initializer_side >= 0) {
        const std::string& init_name = (initializer_side == 0) ? name_a : name_b;
        auto init_it = initializers.find(init_name);
        if (init_it == initializers.end() || !init_it->second) {
            DML_PERF_LOG("[OpActFusion/", m_config.base_op_type, "] Compile: initializer '", init_name, "' not found");
            delete kernel_state;
            return nullptr;
        }
        void* cpu_ptr = nullptr;
        OrtStatus* st = ort_api.GetTensorMutableData(
            const_cast<OrtValue*>(init_it->second), &cpu_ptr);
        if (st || !cpu_ptr) {
            if (st) ort_api.ReleaseStatus(st);
            DML_PERF_LOG("[OpActFusion/", m_config.base_op_type, "] Compile: failed to get initializer CPU data");
            delete kernel_state;
            return nullptr;
        }
        OrtTensorTypeAndShapeInfo* init_shape = nullptr;
        ort_api.GetTensorTypeAndShape(const_cast<OrtValue*>(init_it->second), &init_shape);
        size_t init_elem_count = 0;
        size_t init_rank = 0;
        if (init_shape) {
            ort_api.GetTensorShapeElementCount(init_shape, &init_elem_count);
            ort_api.GetDimensionsCount(init_shape, &init_rank);
            std::vector<int64_t> init_dims(init_rank);
            if (init_rank > 0) ort_api.GetDimensions(init_shape, init_dims.data(), init_rank);
            kernel_state->initializer_sizes.resize(init_rank);
            for (size_t i = 0; i < init_rank; ++i)
                kernel_state->initializer_sizes[i] = static_cast<uint32_t>(init_dims[i] > 0 ? init_dims[i] : 1);
            // Populate dims for the initializer side so eager compilation can use them.
            if (initializer_side == 0) {
                a_dims.resize(init_rank);
                for (size_t i = 0; i < init_rank; ++i) a_dims[i] = init_dims[i];
            } else {
                b_dims.resize(init_rank);
                for (size_t i = 0; i < init_rank; ++i) b_dims[i] = init_dims[i];
            }
            ort_api.ReleaseTensorTypeAndShapeInfo(init_shape);
        }
        size_t elem_sz = (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) ? 2 : 4;
        kernel_state->initializer_bytes    = ((static_cast<uint64_t>(init_elem_count * elem_sz) + 3u) & ~uint64_t(3u));
        kernel_state->initializer_dml_dtype = dml_dtype;

        HRESULT hr = provider->AllocatePooledResource(
            static_cast<size_t>(kernel_state->initializer_bytes),
            AllocatorRoundingMode::Disabled,
            kernel_state->initializer_gpu_resource.GetAddressOf(),
            kernel_state->initializer_allocator_ref.GetAddressOf());
        if (FAILED(hr)) {
            DML_PERF_LOG("[OpActFusion/", m_config.base_op_type, "] Compile: failed to allocate initializer GPU buffer");
            delete kernel_state;
            return nullptr;
        }
        hr = provider->UploadToResource(
            kernel_state->initializer_gpu_resource.Get(), cpu_ptr,
            kernel_state->initializer_bytes);
        if (FAILED(hr)) {
            DML_PERF_LOG("[OpActFusion/", m_config.base_op_type, "] Compile: failed to upload initializer to GPU");
            delete kernel_state;
            return nullptr;
        }
        DML_PERF_LOG("[OpActFusion/", m_config.base_op_type, "] Compile: uploaded initializer side=", initializer_side,
                     " name=", init_name, " bytes=", kernel_state->initializer_bytes);
    }

    DML_PERF_LOG("[OpActFusion/", m_config.base_op_type, "] Compile:"
                 " activation=", FusedActivationTypeName(activation),
                 " alpha=", act_alpha, " beta=", act_beta, " gamma=", act_gamma,
                 " | dtype=", DtypeStr(dml_dtype),
                 " | A=idx[", resolved_a_idx, "] name='", name_a, "' dims=", DimsStr(a_dims),
                 " | B=idx[", resolved_b_idx, "] name='", name_b, "' dims=", DimsStr(b_dims),
                 (name_c.empty() ? "" : (" | C='" + name_c + "' dims=" + DimsStr(c_dims))),
                 " | initializer_side=", initializer_side,
                 (kind == BaseOpKind::Gemm
                     ? (" | transA=" + std::to_string(trans_a) +
                        " transB=" + std::to_string(trans_b) +
                        " gemm_alpha=" + std::to_string(gemm_alpha) +
                        " gemm_beta="  + std::to_string(gemm_beta))
                     : std::string{}));

    // Eager compilation when all dims are concrete.
    // For multi-input ops (Conv/BN/IN/MVN): X (slot 0) dims come from graph input;
    // all other slots are static and already uploaded — sizes in static_sizes.

    auto dims_to_sizes_fn = [](const std::vector<int64_t>& dims) {
        std::vector<uint32_t> s(dims.size());
        for (size_t i = 0; i < dims.size(); ++i)
            s[i] = static_cast<uint32_t>(dims[i] > 0 ? dims[i] : 1);
        return s;
    };

    if (is_multi_input_op) {
        // Get X dims from the runtime graph input (slot 0 is always X for these ops).
        size_t x_graph_idx = (kernel_state->dyn_input_indices.size() > 0)
            ? kernel_state->dyn_input_indices[0] : SIZE_MAX;
        std::vector<int64_t> x_dims_i64;
        if (x_graph_idx != SIZE_MAX && x_graph_idx < input_vis.size())
            get_dims(input_vis[x_graph_idx], x_dims_i64);

        bool x_concrete = !x_dims_i64.empty();
        for (auto d : x_dims_i64) if (d <= 0) { x_concrete = false; break; }

        // Eager compile needs every non-X input shape known now. A dynamic
        // (runtime) weight/scale/bias slot has no static shape until Compute,
        // so defer the whole kernel to first Compute in that case.
        bool all_weights_static = true;
        for (size_t s = 1; s < kernel_state->dyn_input_indices.size(); ++s) {
            if (kernel_state->dyn_input_indices[s] != SIZE_MAX) { all_weights_static = false; break; }
        }

        if (x_concrete && all_weights_static) {
            size_t elem_sz = (dml_dtype == DML_TENSOR_DATA_TYPE_FLOAT32) ? 4 : 2;
            std::vector<uint32_t> x_sizes = dims_to_sizes_fn(x_dims_i64);
            size_t x_elems = 1; for (auto v : x_sizes) x_elems *= v;
            uint64_t x_bytes_eager = ((static_cast<uint64_t>(x_elems * elem_sz) + 3u) & ~uint64_t(3u));

            DML_PERF_LOG("[OpActFusion/", m_config.base_op_type, "] Compile: eager path X=", SizesStr(x_sizes));

            if (kind == BaseOpKind::Conv || kind == BaseOpKind::ConvTranspose) {
                const std::vector<uint32_t>& w_sizes = kernel_state->static_sizes[1];
                uint64_t w_bytes_eager = kernel_state->static_bytes[1];
                bool has_bias_eager = (kernel_state->static_sizes.size() > 2 &&
                                       !kernel_state->static_sizes[2].empty() &&
                                       kernel_state->static_bytes[2] > 0);
                const std::vector<uint32_t>* b_sz_ptr = has_bias_eager ? &kernel_state->static_sizes[2] : nullptr;
                uint64_t b_bytes_eager = has_bias_eager ? kernel_state->static_bytes[2] : 0;
                kernel_state->kernel = CompileConvActivation(
                    provider, kind == BaseOpKind::ConvTranspose,
                    kernel_state->conv_kernel_shape, kernel_state->conv_strides,
                    kernel_state->conv_dilations, kernel_state->conv_pads,
                    kernel_state->conv_output_padding, kernel_state->conv_output_shape,
                    kernel_state->conv_auto_pad, kernel_state->conv_group,
                    activation, act_alpha, act_beta, act_gamma,
                    dml_dtype, x_sizes, x_bytes_eager,
                    w_sizes, w_bytes_eager, b_sz_ptr, b_bytes_eager);
            } else if (kind == BaseOpKind::BatchNorm) {
                kernel_state->kernel = CompileBatchNormActivation(
                    provider, kernel_state->norm_epsilon, kernel_state->bn_spatial,
                    activation, act_alpha, act_beta, act_gamma,
                    dml_dtype, x_sizes, x_bytes_eager,
                    kernel_state->static_sizes[1], kernel_state->static_bytes[1],
                    kernel_state->static_sizes[2], kernel_state->static_bytes[2],
                    kernel_state->static_sizes[3], kernel_state->static_bytes[3],
                    kernel_state->static_sizes[4], kernel_state->static_bytes[4]);
            } else if (kind == BaseOpKind::InstanceNorm) {
                kernel_state->kernel = CompileInstanceNormActivation(
                    provider, kernel_state->norm_epsilon,
                    activation, act_alpha, act_beta, act_gamma,
                    dml_dtype, x_sizes, x_bytes_eager,
                    kernel_state->static_sizes[1], kernel_state->static_bytes[1],
                    kernel_state->static_sizes[2], kernel_state->static_bytes[2]);
            } else {  // MVN
                size_t dml_rank = std::max(x_sizes.size(), size_t(4));
                std::vector<uint32_t> resolved_axes;
                if (!kernel_state->mvn_axes.empty()) {
                    resolved_axes = ResolveMvnAxes(kernel_state->mvn_axes, x_sizes.size(), dml_rank);
                } else if (kernel_state->mvn_across_channels) {
                    for (uint32_t i = 0; i < static_cast<uint32_t>(dml_rank); ++i) resolved_axes.push_back(i);
                } else {
                    resolved_axes.push_back(0);
                    for (uint32_t i = 2; i < static_cast<uint32_t>(dml_rank); ++i) resolved_axes.push_back(i);
                }
                kernel_state->kernel = CompileMVNActivation(
                    provider, resolved_axes, kernel_state->mvn_normalize_variance,
                    activation, act_alpha, act_beta, act_gamma,
                    dml_dtype, x_sizes, x_bytes_eager);
            }

            if (!kernel_state->kernel.IsValid()) {
                DML_PERF_LOG("[OpActFusion/", m_config.base_op_type, "] Compile: eager FAILED");
                delete kernel_state;
                return nullptr;
            }
            kernel_state->compiled_a_sizes = x_sizes;
            // Snapshot the per-slot shapes the kernel was compiled with, matching what
            // the Compute-time recompile check (current_slot_sizes) reconstructs. Slot 0
            // is X; the remaining slots are static weights. Without this, the eager path
            // leaves compiled_slot_sizes empty and every Compute sees a shape "mismatch"
            // and recompiles the kernel on every inference.
            {
                size_t nslots = kernel_state->dyn_input_indices.size();
                kernel_state->compiled_slot_sizes.assign(nslots, {});
                if (nslots > 0) kernel_state->compiled_slot_sizes[0] = x_sizes;
                for (size_t s = 1; s < nslots; ++s)
                    kernel_state->compiled_slot_sizes[s] = kernel_state->static_sizes[s];
            }
            kernel_state->initialized = true;
            DML_PERF_LOG("[OpActFusion/", m_config.base_op_type, "] Compile: eager SUCCESS");
        } else {
            DML_PERF_LOG("[OpActFusion/", m_config.base_op_type, "] Compile: DEFERRED"
                         " — dynamic X dims, will compile on first Compute");
        }
    } else {
        // 2-input ops: Add, Sum, Gemm.
        bool has_dynamic = false;
        for (auto d : a_dims) if (d <= 0) { has_dynamic = true; break; }
        for (auto d : b_dims) if (!has_dynamic && d <= 0) { has_dynamic = true; break; }

        if (!has_dynamic && !a_dims.empty() && !b_dims.empty()) {
            size_t elem_sz = (dml_dtype == DML_TENSOR_DATA_TYPE_FLOAT32) ? 4 : 2;
            std::vector<uint32_t> a_sizes = dims_to_sizes_fn(a_dims);
            std::vector<uint32_t> b_sizes = dims_to_sizes_fn(b_dims);
            size_t a_elems = 1, b_elems = 1;
            for (auto v : a_sizes) a_elems *= v;
            for (auto v : b_sizes) b_elems *= v;

            // DML requires TotalTensorSizeInBytes to be DWORD-aligned.
            auto dml_bytes = [&](size_t elems) -> uint64_t {
                return ((static_cast<uint64_t>(elems * elem_sz) + 3u) & ~uint64_t(3u));
            };

            DML_PERF_LOG("[OpActFusion/", m_config.base_op_type, "] Compile: eager path"
                         " A=", SizesStr(a_sizes), " B=", SizesStr(b_sizes));

            if (kind == BaseOpKind::Add || kind == BaseOpKind::Sum) {
                kernel_state->kernel = CompileAddActivation(
                    provider, activation, act_alpha, act_beta, act_gamma,
                    dml_dtype, a_sizes, b_sizes,
                    dml_bytes(a_elems), dml_bytes(b_elems));
            } else {  // Gemm
                std::vector<uint32_t> c_sizes;
                uint64_t c_b = 0;
                if (!c_dims.empty()) {
                    c_sizes = dims_to_sizes_fn(c_dims);
                    size_t c_elems = 1;
                    for (auto v : c_sizes) c_elems *= v;
                    c_b = dml_bytes(c_elems);
                    DML_PERF_LOG("[OpActFusion/Gemm] Compile: bias C=", SizesStr(c_sizes));
                }
                kernel_state->kernel = CompileGemmActivation(
                    provider, trans_a, trans_b, gemm_alpha, gemm_beta,
                    activation, act_alpha, act_beta, act_gamma,
                    dml_dtype, a_sizes, b_sizes,
                    dml_bytes(a_elems), dml_bytes(b_elems),
                    c_sizes.empty() ? nullptr : &c_sizes, c_b);
            }

            if (!kernel_state->kernel.IsValid()) {
                DML_PERF_LOG("[OpActFusion/", m_config.base_op_type, "] Compile: eager FAILED"
                             " — DML CreateOperator/CompileOperator returned error");
                delete kernel_state;
                return nullptr;
            }
            for (auto d : a_dims)
                kernel_state->compiled_a_sizes.push_back(static_cast<uint32_t>(d > 0 ? d : 1));
            for (auto d : b_dims)
                kernel_state->compiled_b_sizes.push_back(static_cast<uint32_t>(d > 0 ? d : 1));
            kernel_state->initialized = true;
            DML_PERF_LOG("[OpActFusion/", m_config.base_op_type, "] Compile: eager SUCCESS"
                         " — compiled DML op ready");
        } else {
            DML_PERF_LOG("[OpActFusion/", m_config.base_op_type, "] Compile: DEFERRED"
                         " — dynamic dims detected, will compile on first Compute"
                         " A=", DimsStr(a_dims), " B=", DimsStr(b_dims));
        }
    }

    DML_PERF_LOG("[OpActFusion/", m_config.base_op_type, "] Compile: OK"
                 " — OrtNodeComputeInfo returned to ORT");
    auto* info  = new OpActivationNodeComputeInfo();
    info->state = kernel_state;
    return info;
}

// ---------------------------------------------------------------------------
// Factory — build config for each base op and return rule instances.
// ---------------------------------------------------------------------------

std::vector<std::unique_ptr<IFusionRule>> MakeAllOpActivationFusionRules(bool isMcdmDevice) {
    std::vector<std::unique_ptr<IFusionRule>> rules;

    const std::vector<std::string>& full_act  = kAllActivations;
    const std::vector<std::string>& mcdm_act  = kMcdmActivations;
    const std::vector<std::string>& add_act   = kMcdmActivations;  // Add/Sum: Relu+LeakyRelu only always

    // Add: Relu+LeakyRelu only regardless of device.
    {
        OpActivationConfig cfg;
        cfg.base_op_type        = "Add";
        cfg.fused_op_type       = "DmlFusedAdd";
        cfg.allowed_activations = add_act;
        rules.push_back(std::make_unique<OpActivationFusionRule>(std::move(cfg)));
    }

    // Sum: same activations as Add, exactly 2 inputs.
    {
        OpActivationConfig cfg;
        cfg.base_op_type        = "Sum";
        cfg.fused_op_type       = "DmlFusedSum";
        cfg.allowed_activations = add_act;
        cfg.input_count_filter  = 2u;
        rules.push_back(std::make_unique<OpActivationFusionRule>(std::move(cfg)));
    }

    // Gemm: all activations on non-MCDM, Relu+LeakyRelu on MCDM.
    {
        OpActivationConfig cfg;
        cfg.base_op_type        = "Gemm";
        cfg.fused_op_type       = "DmlFusedGemm";
        cfg.allowed_activations = isMcdmDevice ? mcdm_act : full_act;
        rules.push_back(std::make_unique<OpActivationFusionRule>(std::move(cfg)));
    }

    // Conv: all activations on non-MCDM, Relu+LeakyRelu on MCDM.
    {
        OpActivationConfig cfg;
        cfg.base_op_type        = "Conv";
        cfg.fused_op_type       = "DmlFusedConv";
        cfg.allowed_activations = isMcdmDevice ? mcdm_act : full_act;
        rules.push_back(std::make_unique<OpActivationFusionRule>(std::move(cfg)));
    }

    // ConvTranspose: same policy as Conv.
    {
        OpActivationConfig cfg;
        cfg.base_op_type        = "ConvTranspose";
        cfg.fused_op_type       = "DmlFusedConvTranspose";
        cfg.allowed_activations = isMcdmDevice ? mcdm_act : full_act;
        rules.push_back(std::make_unique<OpActivationFusionRule>(std::move(cfg)));
    }

    // BatchNormalization: all activations on non-MCDM; DISABLED on MCDM.
    if (!isMcdmDevice) {
        OpActivationConfig cfg;
        cfg.base_op_type        = "BatchNormalization";
        cfg.fused_op_type       = "DmlFusedBatchNormalization";
        cfg.allowed_activations = full_act;
        rules.push_back(std::make_unique<OpActivationFusionRule>(std::move(cfg)));
    }

    // InstanceNormalization: all activations on non-MCDM; DISABLED on MCDM.
    if (!isMcdmDevice) {
        OpActivationConfig cfg;
        cfg.base_op_type        = "InstanceNormalization";
        cfg.fused_op_type       = "DmlFusedInstanceNormalization";
        cfg.allowed_activations = full_act;
        rules.push_back(std::make_unique<OpActivationFusionRule>(std::move(cfg)));
    }

    // MeanVarianceNormalization: all activations on non-MCDM; DISABLED on MCDM.
    if (!isMcdmDevice) {
        OpActivationConfig cfg;
        cfg.base_op_type        = "MeanVarianceNormalization";
        cfg.fused_op_type       = "DmlFusedMeanVarianceNormalization";
        cfg.allowed_activations = full_act;
        rules.push_back(std::make_unique<OpActivationFusionRule>(std::move(cfg)));
    }

    return rules;
}

}  // namespace dml_ep
