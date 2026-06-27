// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ---------------------------------------------------------------------------
// FusedMatMulFusionRule — Tier-2 EP-level fusion for Transpose + MatMul + Scale
//
// Detects raw ONNX operator sequences and fuses them into a single DML GEMM
// operator, eliminating intermediate kernel launches and memory traffic.
//
// Pattern detected (all optional branches independent):
//
//     [Transpose(A)]  [Transpose(B)]      upstream, optional per side
//           |               |
//           +---- MatMul ---+              anchor node
//                   |
//          [Mul(scalar) or Div(scalar)]    downstream, optional scale
//                   |
//             [activation]                 downstream, optional (Relu, Sigmoid, ...)
//
// Pipeline:
//
//   1. BuildPattern()        — declares the PNode tree with optional branches
//
//   2. MatchPattern()        — pattern engine traverses graph from each MatMul
//                              anchor, validates constraints (perm, scalar, etc.)
//
//   3. MatchesResult()       — confirms at least one neighbor was captured
//                              (bare MatMul without neighbors is not fused)
//
//   4. CapturePreFusionData()— reads Transpose perm attributes from the original
//                              graph and writes transA/transB/transBatch flags
//                              into match.scalar_values for use by Compile().
//                              Also captures Transpose source tensor names.
//                              Rejects rank-2 transposes (DML GEMM limitation)
//                              and transBatch when rank cannot be verified.
//
//   5. ValidateCapture()     — rejects fusions that would produce invalid GEMM:
//                              same-input MatMul, unsupported data types,
//                              initializer inputs without a Transpose to absorb,
//                              scale Mul with multi-consumer MatMul output.
//
//   6. Exclusion pass        — (in ApplyFusions) removes non-terminal nodes with
//                              external consumers from the fused subgraph.
//                              Terminal nodes (pattern leaf nodes) are kept — ORT
//                              rewires their external consumers automatically.
//
//   7. Compile()             — reads trans flags, scale, activation from the
//                              PatternMatch.  Resolves graph input indices.
//                              Uploads initializer inputs to GPU if needed.
//                              Builds the DML GEMM descriptor via
//                              PrepareGemmShapes() + CompileFusedMatMulDml().
//                              Defers to first Compute() when shapes are dynamic.
//
//   8. FusedMatMul_Compute() — reads runtime tensor shapes, compiles DML GEMM
//                              on first call (lazy-init), recompiles on shape
//                              change.  Binds A, B, C resources and executes.
//
// Shape handling (PrepareGemmShapes):
//
//   1. Compute packed strides
//   2. Expand rank-1 inputs (broadcast dim with stride 0)
//   3. Apply transBatch rotation on sizes + strides
//   4. Equalise ranks by prepending [size=1, stride=0]
//   5. Broadcast batch dims (stride 0 on size-1 side)
//   6. Pad to 4D minimum (DML requirement)
//   7. Derive M, K, N and validate K agreement
// ---------------------------------------------------------------------------

#include "fused_matmul_ep_fusion.h"
#include "fusion_utils.h"
#include "pattern_matcher.h"
#include "ort_node_adapter.h"
#include "dml_execution_provider.h"   // PluginDmlExecutionProviderImpl
#include "DmlExecutionProvider/IExecutionProvider.h"
#include "dml_abi_kernel.h"

#include <DirectML.h>
#include <wrl/client.h>
#include <gsl/gsl>
#include <algorithm>
#include <numeric>
#include <optional>
#include <vector>

namespace dml_ep {

using namespace fusion_utils;
using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// Activation type captured from the optional downstream branch.
// ---------------------------------------------------------------------------
enum class FusedMatMulActivation {
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
};

static const char* ActivationName(FusedMatMulActivation a) {
    switch (a) {
    case FusedMatMulActivation::None:             return "None";
    case FusedMatMulActivation::Relu:             return "Relu";
    case FusedMatMulActivation::Sigmoid:          return "Sigmoid";
    case FusedMatMulActivation::Tanh:             return "Tanh";
    case FusedMatMulActivation::LeakyRelu:        return "LeakyRelu";
    case FusedMatMulActivation::Elu:              return "Elu";
    case FusedMatMulActivation::HardSigmoid:      return "HardSigmoid";
    case FusedMatMulActivation::Selu:             return "Selu";
    case FusedMatMulActivation::Softplus:         return "Softplus";
    case FusedMatMulActivation::Softsign:         return "Softsign";
    case FusedMatMulActivation::ThresholdedRelu:  return "ThresholdedRelu";
    default:                                      return "Unknown";
    }
}

static std::string SizesToString(const std::vector<uint32_t>& sizes) {
    std::string s = "[";
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (i > 0) s += ",";
        s += std::to_string(sizes[i]);
    }
    s += "]";
    return s;
}

// ---------------------------------------------------------------------------
// Transpose perm analysis — determines if a Transpose can be absorbed into
// a GEMM's TransA/TransB or TransBatch flags.
// ---------------------------------------------------------------------------
struct TransposeInfo {
    bool is_trans       = false;
    bool is_trans_batch = false;
};

static bool AnalyzeTransposePerm(
    const std::vector<int64_t>& perms,
    TransposeInfo& out)
{
    out = {};
    size_t rank = perms.size();
    if (rank < 2) return false;

    int64_t last_axis = static_cast<int64_t>(rank) - 1;

    size_t last_axis_index = rank;
    if (perms[rank - 1] == last_axis) last_axis_index = rank - 1;
    else if (perms[rank - 2] == last_axis) last_axis_index = rank - 2;
    if (last_axis_index == rank) return false;

    if (rank >= 3) {
        if (perms[0] != 0 && perms[0] != 1) return false;
        for (size_t i = 0; i + 1 < rank - 2; ++i) {
            if (perms[i] + 1 != perms[i + 1]) return false;
        }
    }

    out.is_trans       = (last_axis_index == rank - 2);
    out.is_trans_batch = (rank >= 3 && perms[0] == 1);
    return true;
}

// ---------------------------------------------------------------------------
// IFusionRule implementation
//
// Patterns detected (MatMul + optional Transpose(s) + optional scale + optional activation):
//
//   [Transpose(A)] [Transpose(B)]   ← upstream, each independently optional
//         |               |
//         +──── MatMul ───+          ← anchor
//                  |
//      [Mul(scalar) or Div(scalar)]  ← downstream optional scale
//                  |
//         [activation op]            ← downstream optional (on MatMul or scale)
// ---------------------------------------------------------------------------

// Transpose perm predicate: used in BuildPattern to reject Transposes whose
// perm cannot be expressed as GEMM TransA/TransB/TransBatch flags.
// The predicate's `node` argument is the Transpose node being visited.
static bool TransposePermPredicate(
    const std::string& /*value_name*/,
    const OrtValueInfo* value_info,
    const OrtApi& ort_api,
    const std::unordered_map<std::string, const OrtValue*>& /*initializers*/,
    const OrtNode* node)
{
    OrtNodeAdapter adapter(node, ort_api);
    std::vector<int64_t> perms = adapter.GetAttributeInts("perm");
    if (perms.empty()) {
        // No perm attribute — default perm reverses all dims.
        // Mirror ORT's matmul_transpose_fusion: if static shape is unavailable
        // we cannot determine the rank and therefore cannot validate the perm,
        // so reject the fusion (same as GetTransposePerms returning false).
        if (!value_info) return false;
        const OrtTypeInfo* type_info = nullptr;
        OrtStatus* st = ort_api.GetValueInfoTypeInfo(value_info, &type_info);
        if (st || !type_info) { if (st) ort_api.ReleaseStatus(st); return false; }
        const OrtTensorTypeAndShapeInfo* shape_info = nullptr;
        ort_api.CastTypeInfoToTensorInfo(type_info, &shape_info);
        if (!shape_info) return false;
        size_t rank = 0;
        ort_api.GetDimensionsCount(shape_info, &rank);
        if (rank == 0) return false;  // unknown rank — reject like ORT does

        // Construct the default reversal perm and validate it.
        perms.resize(rank);
        std::iota(perms.rbegin(), perms.rend(), 0);
    }
    TransposeInfo info;
    return AnalyzeTransposePerm(perms, info);
}

PNode FusedMatMulFusionRule::BuildPattern() const {
    // Helper: attaches optional activation branches to a PNode.
    // Returns the node by value (PNode builder methods return *this by reference,
    // so we chain on a local and move the result).
    auto AddActivations = [](PNode n) -> PNode {
        n.Downstream(0, PNode("Relu").As("FusedMatMul.relu"),                   /*optional=*/true)
         .Downstream(0, PNode("Sigmoid").As("FusedMatMul.sigmoid"),              /*optional=*/true)
         .Downstream(0, PNode("Tanh").As("FusedMatMul.tanh"),                   /*optional=*/true)
         .Downstream(0, PNode("LeakyRelu").As("FusedMatMul.leaky"),             /*optional=*/true)
         .Downstream(0, PNode("Elu").As("FusedMatMul.elu"),                     /*optional=*/true)
         .Downstream(0, PNode("HardSigmoid").As("FusedMatMul.hsig"),            /*optional=*/true)
         .Downstream(0, PNode("Selu").As("FusedMatMul.selu"),                   /*optional=*/true)
         .Downstream(0, PNode("Softplus").As("FusedMatMul.splus"),              /*optional=*/true)
         .Downstream(0, PNode("Softsign").As("FusedMatMul.ssign"),              /*optional=*/true)
         .Downstream(0, PNode("ThresholdedRelu").As("FusedMatMul.threlu"),      /*optional=*/true);
        return n;
    };

    // Scale predicate: the initializer must be a true scalar (single element).
    // Checks both that it's in the initializer map AND that TryReadScalarFloat
    // succeeds (elem_count==1).  This mirrors ORT's GetScalarConstantInitializer
    // which validates shape.Size()==1.
    auto ScalarInitPredicate = [](const std::string& name,
                                  const OrtValueInfo* /*vi*/,
                                  const OrtApi& ort_api,
                                  const std::unordered_map<std::string, const OrtValue*>& initializers,
                                  const OrtNode* /*node*/) -> bool {
        auto it = initializers.find(name);
        if (it == initializers.end()) return false;
        float v = 0.0f;
        return fusion_utils::TryReadScalarFloat(ort_api, it->second, v);
    };

    // Scale Mul: any input must be a scalar constant initializer.
    PInput scale_mul_input = PPredicate(ScalarInitPredicate);
    scale_mul_input.capture_name = "FusedMatMul.scale_name";
    PNode scale_mul = AddActivations(
        PNode("Mul").As("FusedMatMul.scale_mul")
            .Input(SIZE_MAX, std::move(scale_mul_input)));

    // Scale Div: divisor (input[1]) must be a scalar constant initializer.
    PInput scale_div_input = PPredicate(ScalarInitPredicate);
    scale_div_input.capture_name = "FusedMatMul.scale_div_name";
    PNode scale_div = AddActivations(
        PNode("Div").As("FusedMatMul.scale_div")
            .Input(1, std::move(scale_div_input)));

    // The full pattern tree anchored at MatMul.
    //
    // Upstream Transpose edges validate that the perm is fusable into GEMM via
    // TransposePermPredicate (checked against input[0] of the Transpose node —
    // the predicate's `node` arg is the Transpose itself).
    //
    // Activation branches are added at two levels:
    //   1. Directly on MatMul output (no scale node matched).
    //   2. Inside scale_mul / scale_div (activation follows the scale node).
    // The engine tries each optional branch; the greedy longest match wins.
    PNode matmul = PNode("MatMul").As("FusedMatMul.matmul")
        .Upstream(0,
            PNode("Transpose").As("FusedMatMul.trans_a")
                .Input(0, PPredicate(TransposePermPredicate)),
            /*optional=*/true)
        .Upstream(1,
            PNode("Transpose").As("FusedMatMul.trans_b")
                .Input(0, PPredicate(TransposePermPredicate)),
            /*optional=*/true)
        .Downstream(0, std::move(scale_mul), /*optional=*/true)
        .Downstream(0, std::move(scale_div), /*optional=*/true);

    return AddActivations(std::move(matmul));
}

// ---------------------------------------------------------------------------
// CapturePreFusionData — called after pattern matching, before external-consumer
// exclusion.  Reads each matched Transpose node's perm attribute from the
// original graph and writes the resulting flags into match.scalar_values so
// Compile() can retrieve them even if the Transpose is later excluded from
// the fused subgraph (because it has external consumers).
// ---------------------------------------------------------------------------
void FusedMatMulFusionRule::CapturePreFusionData(
    PatternMatch&                          match,
    const fusion_utils::GraphConnectivity& gc,
    const OrtApi&                          ort_api) const
{
    struct Side { const char* idx_key; const char* trans_key; const char* batch_key; };
    static const Side sides[2] = {
        { "FusedMatMul.trans_a", "FusedMatMul.transA", "FusedMatMul.transBatchA" },
        { "FusedMatMul.trans_b", "FusedMatMul.transB", "FusedMatMul.transBatchB" },
    };

    for (const auto& side : sides) {
        size_t node_idx = match.NodeIdx(side.idx_key);
        if (node_idx == SIZE_MAX) continue;
        if (node_idx >= gc.node_infos.size()) continue;

        OrtNodeAdapter adapter(gc.node_infos[node_idx].node, ort_api);
        std::vector<int64_t> perms = adapter.GetAttributeInts("perm");
        TransposeInfo info;
        if (perms.empty()) {
            info.is_trans = true;
        } else if (!AnalyzeTransposePerm(perms, info)) {
            continue;
        }

        // Reject rank-2 transpose fusions.  DML GEMM cannot produce a [1,1]
        // output, and with rank-2 inputs the EP C API does not expose static
        // shapes at init time to validate whether M=1 && N=1.
        if (perms.size() == 2) {
            match.node_indices.erase(side.idx_key);
            auto it = std::find(match.all_nodes.begin(), match.all_nodes.end(), node_idx);
            if (it != match.all_nodes.end()) match.all_nodes.erase(it);
            continue;
        }

        // Reject transBatch when the other input's rank is unavailable.  ORT
        // requires both inputs to have the same rank >= 3 for transBatch; the
        // EP C API does not expose static shapes so we cannot verify this.
        if (info.is_trans_batch) {
            size_t matmul_idx = match.NodeIdx("FusedMatMul.matmul");
            if (matmul_idx == SIZE_MAX || matmul_idx >= gc.node_infos.size()) continue;
            int other_side = (side.idx_key == std::string_view("FusedMatMul.trans_a")) ? 1 : 0;
            const auto& matmul_inputs = gc.node_infos[matmul_idx].input_names;
            if (other_side < static_cast<int>(matmul_inputs.size())) {
                // Check if the other input's shape is available.
                size_t mic = 0;
                ort_api.Node_GetNumInputs(gc.node_infos[matmul_idx].node, &mic);
                std::vector<const OrtValueInfo*> vis(mic, nullptr);
                if (mic > 0) ort_api.Node_GetInputs(gc.node_infos[matmul_idx].node, vis.data(), mic);
                size_t other_rank = 0;
                if (static_cast<size_t>(other_side) < mic && vis[other_side]) {
                    const OrtTypeInfo* ti = nullptr;
                    OrtStatus* st = ort_api.GetValueInfoTypeInfo(vis[other_side], &ti);
                    if (!st && ti) {
                        const OrtTensorTypeAndShapeInfo* si = nullptr;
                        ort_api.CastTypeInfoToTensorInfo(ti, &si);
                        if (si) ort_api.GetDimensionsCount(si, &other_rank);
                    }
                    if (st) ort_api.ReleaseStatus(st);
                }
                if (other_rank == 0 || other_rank < 3 || other_rank != perms.size()) {
                    match.node_indices.erase(side.idx_key);
                    auto rm = std::find(match.all_nodes.begin(), match.all_nodes.end(), node_idx);
                    if (rm != match.all_nodes.end()) match.all_nodes.erase(rm);
                    continue;
                }
            }
        }

        match.scalar_values[side.trans_key]  = info.is_trans       ? 1.0f : 0.0f;
        match.scalar_values[side.batch_key]  = info.is_trans_batch ? 1.0f : 0.0f;

        // Capture the Transpose's source input name so Compile() can resolve the
        // correct graph input index even when the Transpose is excluded from the
        // fused subgraph (kept in the outer graph for other consumers).
        const auto& input_names = gc.node_infos[node_idx].input_names;
        if (!input_names.empty()) {
            // "FusedMatMul.trans_a" → "FusedMatMul.source_a"
            std::string source_key = std::string(side.idx_key);
            source_key.replace(source_key.find("trans_"), 6, "source_");
            match.value_names[source_key] = input_names[0];

        }
    }
}

bool FusedMatMulFusionRule::MatchesResult(const PatternMatch& m) const {
    if (m.NodeIdx("FusedMatMul.matmul") == SIZE_MAX) return false;
    // Require at least one neighbor captured — bare MatMul with no Transpose,
    // scale, or activation is not a fusion candidate.
    return m.NodeIdx("FusedMatMul.trans_a")  != SIZE_MAX
        || m.NodeIdx("FusedMatMul.trans_b")  != SIZE_MAX
        || m.NodeIdx("FusedMatMul.scale_mul") != SIZE_MAX
        || m.NodeIdx("FusedMatMul.scale_div") != SIZE_MAX
        || m.NodeIdx("FusedMatMul.relu")     != SIZE_MAX
        || m.NodeIdx("FusedMatMul.sigmoid")  != SIZE_MAX
        || m.NodeIdx("FusedMatMul.tanh")     != SIZE_MAX
        || m.NodeIdx("FusedMatMul.leaky")    != SIZE_MAX
        || m.NodeIdx("FusedMatMul.elu")      != SIZE_MAX
        || m.NodeIdx("FusedMatMul.hsig")     != SIZE_MAX
        || m.NodeIdx("FusedMatMul.selu")     != SIZE_MAX
        || m.NodeIdx("FusedMatMul.splus")    != SIZE_MAX
        || m.NodeIdx("FusedMatMul.ssign")    != SIZE_MAX
        || m.NodeIdx("FusedMatMul.threlu")   != SIZE_MAX;
}

// ---------------------------------------------------------------------------
// Compiled kernel state
// ---------------------------------------------------------------------------
struct FusedMatMulCompiledKernel {
    ComPtr<IDMLCompiledOperator>      compiled_op;
    ComPtr<ID3D12Resource>            persistent_resource;
    ComPtr<IUnknown>                  persistent_allocator;
    std::optional<DML_BUFFER_BINDING> persistent_binding;

    bool IsValid() const { return compiled_op != nullptr; }
};

struct FusedMatMulKernelState {
    bool                           trans_a       = false;
    bool                           trans_b       = false;
    bool                           trans_batch_a = false;
    bool                           trans_batch_b = false;
    float                          alpha         = 1.0f;
    FusedMatMulActivation          activation    = FusedMatMulActivation::None;
    float                          act_alpha     = 0.0f;
    float                          act_beta      = 0.0f;
    float                          act_gamma     = 0.0f;
    PluginDmlExecutionProviderImpl* provider     = nullptr;
    const OrtApi*                  ort_api      = nullptr;

    size_t                         graph_input_idx_a = 0;
    size_t                         graph_input_idx_b = 1;

    // When one MatMul input is a constant initializer, it's uploaded to GPU at
    // Compile time and bound directly at Compute time.  The other input comes
    // from KernelContext_GetInput as usual.
    // 0 = A is initializer, 1 = B is initializer, -1 = neither.
    int                            initializer_side = -1;
    ComPtr<ID3D12Resource>         initializer_gpu_resource;
    ComPtr<IUnknown>               initializer_allocator_ref;
    uint64_t                       initializer_bytes = 0;
    std::vector<uint32_t>          initializer_sizes;  // concrete shape of the initializer

    std::mutex                     init_mutex;
    bool                           initialized = false;
    FusedMatMulCompiledKernel      kernel;

    std::vector<uint32_t>          compiled_a_sizes;
    std::vector<uint32_t>          compiled_b_sizes;
};

// ---------------------------------------------------------------------------
// PrepareGemmShapes
//
// Single source of truth for all shape/stride math needed to build a DML
// GEMM descriptor and compute the output shape.  Mirrors ORT's combination
// of GetFusedMatMulSizesAndStrides + FusedMatMulShapeMapping.
//
// Steps (matching ORT exactly):
//   1. Compute packed strides for each input.
//   2. Handle rank-1 inputs by inserting a broadcast dimension (stride 0):
//        A rank-1: prepend [1, stride=0]   → treated as row vector [1 × N]
//        B rank-1: append  [1, stride=0]   → treated as column vector [N × 1]
//   3. Apply transBatch rotation on sizes+strides (before rank equalisation).
//   4. Equalise ranks by prepending [size=1, stride=0] to the shorter input.
//   5. Broadcast batch dimensions (all except last 2): where one side is 1
//      and the other is N, expand both to N and set stride=0 on the size-1
//      side — this is the stride-0 broadcast mechanism.
//   6. Pad both to at least 4 dimensions (DML minimum).
//   7. Derive output sizes from the 4D shapes + trans flags.
// ---------------------------------------------------------------------------
struct GemmShapes {
    std::vector<uint32_t> a_sizes;
    std::vector<uint32_t> a_strides;
    std::vector<uint32_t> b_sizes;
    std::vector<uint32_t> b_strides;
    std::vector<uint32_t> c_sizes;   // 4D output, also usable as logical out shape
    bool                  valid = false;
};

static GemmShapes PrepareGemmShapes(
    const std::vector<uint32_t>& a_sizes_in,
    const std::vector<uint32_t>& b_sizes_in,
    bool trans_a,
    bool trans_b,
    bool trans_batch_a,
    bool trans_batch_b)
{
    GemmShapes g;

    // Step 1 — packed strides.
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

    // Step 2 — rank-1 inputs: insert broadcast dimension with stride 0.
    // A rank-1 [N] → [1, N] with strides [0, 1]  (treated as 1×N row vector)
    // B rank-1 [N] → [N, 1] with strides [1, 0]  (treated as N×1 col vector)
    if (g.a_sizes.size() == 1) {
        g.a_sizes.insert(g.a_sizes.begin(), 1u);
        g.a_strides.insert(g.a_strides.begin(), 0u);
    }
    if (g.b_sizes.size() == 1) {
        g.b_sizes.push_back(1u);
        g.b_strides.push_back(0u);
    }

    if (g.a_sizes.size() < 2 || g.b_sizes.size() < 2) return g;  // degenerate

    // Step 3 — transBatch rotation: rotate first batch dim into last batch
    // position (second-to-last overall) on sizes and strides together.
    auto apply_trans_batch = [](std::vector<uint32_t>& sizes, std::vector<uint32_t>& strides) {
        if (sizes.size() <= 2) return;
        std::rotate(sizes.begin(),   sizes.begin()   + 1, sizes.end()   - 1);
        std::rotate(strides.begin(), strides.begin() + 1, strides.end() - 1);
    };
    if (trans_batch_a) apply_trans_batch(g.a_sizes, g.a_strides);
    if (trans_batch_b) apply_trans_batch(g.b_sizes, g.b_strides);

    // Step 4 — equalise ranks by prepending [size=1, stride=0].
    size_t broadcast_rank = std::max(g.a_sizes.size(), g.b_sizes.size());
    while (g.a_sizes.size() < broadcast_rank) {
        g.a_sizes.insert(g.a_sizes.begin(), 1u);
        g.a_strides.insert(g.a_strides.begin(), 0u);
    }
    while (g.b_sizes.size() < broadcast_rank) {
        g.b_sizes.insert(g.b_sizes.begin(), 1u);
        g.b_strides.insert(g.b_strides.begin(), 0u);
    }

    // Step 5 — broadcast batch dimensions (all except last 2).
    for (size_t d = 0; d + 2 < broadcast_rank; ++d) {
        uint32_t da = g.a_sizes[d], db = g.b_sizes[d];
        if (da == db) continue;
        uint32_t bd = std::max(da, db);
        if (da == 1) { g.a_sizes[d] = bd; g.a_strides[d] = 0; }
        if (db == 1) { g.b_sizes[d] = bd; g.b_strides[d] = 0; }
    }

    // Step 6 — pad to 4D minimum.
    while (g.a_sizes.size() < 4) { g.a_sizes.insert(g.a_sizes.begin(), 1u); g.a_strides.insert(g.a_strides.begin(), 0u); }
    while (g.b_sizes.size() < 4) { g.b_sizes.insert(g.b_sizes.begin(), 1u); g.b_strides.insert(g.b_strides.begin(), 0u); }

    // Step 7 — output sizes.  TransA/B operate on the innermost 2 dims.
    uint32_t M        = trans_a ? g.a_sizes[3] : g.a_sizes[2];
    uint32_t K_from_A = trans_a ? g.a_sizes[2] : g.a_sizes[3];
    uint32_t K_from_B = trans_b ? g.b_sizes[3] : g.b_sizes[2];
    uint32_t N        = trans_b ? g.b_sizes[2] : g.b_sizes[3];

    // K must agree between A and B — a mismatch means the inputs cannot form a
    // valid matrix multiply regardless of transposition flags.
    if (K_from_A != K_from_B) return g;  // valid stays false

    g.c_sizes = { g.a_sizes[0], g.a_sizes[1], M, N };

    g.valid = true;
    return g;
}

bool FusedMatMulFusionRule::ValidateCapture(
    const PatternMatch&                    match,
    const fusion_utils::GraphConnectivity& gc,
    const OrtApi&                          ort_api) const
{
    size_t matmul_idx = match.NodeIdx("FusedMatMul.matmul");
    if (matmul_idx == SIZE_MAX || matmul_idx >= gc.node_infos.size()) return true;
    const auto& matmul_info = gc.node_infos[matmul_idx];

    if (matmul_info.input_names.size() < 2) return true;

    // Reject when both MatMul inputs are the same tensor.
    if (matmul_info.input_names[0] == matmul_info.input_names[1]) return false;

    // Reject when both MatMul inputs come from outside the matched pattern AND
    // neither has a producer node in gc (i.e., both are graph-level initializers
    // or constants with no runtime component). A fused subgraph with 0 graph
    // inputs cannot be compiled by FusedMatMul — Compile would bail with
    // "no inputs" and CompileImpl would fail.
    {
        auto is_external_non_produced = [&](const std::string& name) {
            // Produced by a node in gc but NOT in the matched group → graph input (ok)
            auto pit = gc.producer_map.find(name);
            if (pit != gc.producer_map.end()) {
                // Has a producer — check if it's inside the matched pattern.
                bool in_match = false;
                for (size_t idx : match.all_nodes) {
                    if (idx == pit->second) { in_match = true; break; }
                }
                return !in_match;  // outside the pattern → becomes a graph input
            }
            // No producer in gc → this is an initializer/constant (no graph input slot)
            return false;  // initializer: NOT an external runtime input
        };

        // For each MatMul input: true if it will be a runtime graph input.
        bool a_is_runtime = is_external_non_produced(matmul_info.input_names[0]);
        bool b_is_runtime = is_external_non_produced(matmul_info.input_names[1]);

        if (!a_is_runtime && !b_is_runtime)
            return false;
    }

    // Reject unsupported data types at init time.  DML GEMM only supports
    // fp32 and fp16; double and bfloat16 require different DML feature levels
    // that we do not currently target.  If the type is unknown (rank=0 from
    // the EP C API), allow through — Compile will check again at runtime.
    {
        size_t ic = 0;
        ort_api.Node_GetNumInputs(matmul_info.node, &ic);
        std::vector<const OrtValueInfo*> vis(ic, nullptr);
        if (ic > 0) ort_api.Node_GetInputs(matmul_info.node, vis.data(), ic);
        if (ic >= 1 && vis[0]) {
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

    // When absorbing a downstream scale Mul/Div, the MatMul output must have
    // exactly one consumer (the scale node).  Multiple consumers would cause
    // the other consumers to receive the scaled result instead of the unscaled
    // MatMul output.
    bool has_scale = match.NodeIdx("FusedMatMul.scale_mul") != SIZE_MAX
                  || match.NodeIdx("FusedMatMul.scale_div") != SIZE_MAX;
    if (has_scale && !matmul_info.output_names.empty()) {
        if (!gc.HasSingleConsumer(matmul_info.output_names[0])) return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Build DML activation descriptor from enum + parameters.
// ---------------------------------------------------------------------------
struct ActivationDescStorage {
    DML_ACTIVATION_RELU_OPERATOR_DESC           relu{};
    DML_ACTIVATION_SIGMOID_OPERATOR_DESC        sigmoid{};
    DML_ACTIVATION_TANH_OPERATOR_DESC           tanh{};
    DML_ACTIVATION_LEAKY_RELU_OPERATOR_DESC     leaky{};
    DML_ACTIVATION_ELU_OPERATOR_DESC            elu{};
    DML_ACTIVATION_HARD_SIGMOID_OPERATOR_DESC   hardsig{};
    DML_ACTIVATION_LINEAR_OPERATOR_DESC         selu{};
    DML_ACTIVATION_SOFTPLUS_OPERATOR_DESC       softplus{};
    DML_ACTIVATION_SOFTSIGN_OPERATOR_DESC       softsign{};
    DML_ACTIVATION_THRESHOLDED_RELU_OPERATOR_DESC threlu{};

    DML_OPERATOR_DESC desc{};
    bool valid = false;
};

static ActivationDescStorage BuildActivationDesc(
    FusedMatMulActivation act, float act_alpha, float act_beta, float act_gamma)
{
    ActivationDescStorage s;
    switch (act) {
    case FusedMatMulActivation::Relu:
        s.desc = { DML_OPERATOR_ACTIVATION_RELU, &s.relu }; s.valid = true; break;
    case FusedMatMulActivation::Sigmoid:
        s.desc = { DML_OPERATOR_ACTIVATION_SIGMOID, &s.sigmoid }; s.valid = true; break;
    case FusedMatMulActivation::Tanh:
        s.desc = { DML_OPERATOR_ACTIVATION_TANH, &s.tanh }; s.valid = true; break;
    case FusedMatMulActivation::LeakyRelu:
        s.leaky.Alpha = act_alpha;
        s.desc = { DML_OPERATOR_ACTIVATION_LEAKY_RELU, &s.leaky }; s.valid = true; break;
    case FusedMatMulActivation::Elu:
        s.elu.Alpha = act_alpha;
        s.desc = { DML_OPERATOR_ACTIVATION_ELU, &s.elu }; s.valid = true; break;
    case FusedMatMulActivation::HardSigmoid:
        s.hardsig.Alpha = act_alpha; s.hardsig.Beta = act_beta;
        s.desc = { DML_OPERATOR_ACTIVATION_HARD_SIGMOID, &s.hardsig }; s.valid = true; break;
    case FusedMatMulActivation::Selu:
        s.selu.Alpha = act_alpha; s.selu.Beta = act_gamma;
        s.desc = { DML_OPERATOR_ACTIVATION_LINEAR, &s.selu }; s.valid = true; break;
    case FusedMatMulActivation::Softplus:
        s.softplus.Steepness = 1.0f;
        s.desc = { DML_OPERATOR_ACTIVATION_SOFTPLUS, &s.softplus }; s.valid = true; break;
    case FusedMatMulActivation::Softsign:
        s.desc = { DML_OPERATOR_ACTIVATION_SOFTSIGN, &s.softsign }; s.valid = true; break;
    case FusedMatMulActivation::ThresholdedRelu:
        s.threlu.Alpha = act_alpha;
        s.desc = { DML_OPERATOR_ACTIVATION_THRESHOLDED_RELU, &s.threlu }; s.valid = true; break;
    default:
        s.valid = false; break;
    }
    return s;
}

// ---------------------------------------------------------------------------
// Compile a GEMM DML operator for given concrete shapes.
// ---------------------------------------------------------------------------
static FusedMatMulCompiledKernel CompileFusedMatMulDml(
    PluginDmlExecutionProviderImpl*  provider,
    bool                             trans_a,
    bool                             trans_b,
    bool                             trans_batch_a,
    bool                             trans_batch_b,
    float                            alpha,
    FusedMatMulActivation            activation,
    float                            act_alpha,
    float                            act_beta,
    float                            act_gamma,
    DML_TENSOR_DATA_TYPE             dml_dtype,
    const std::vector<uint32_t>&     a_sizes_in,
    const std::vector<uint32_t>&     b_sizes_in,
    uint64_t                         a_bytes,
    uint64_t                         b_bytes)
{
    FusedMatMulCompiledKernel result;

    ComPtr<IDMLDevice> dml_device;
    if (FAILED(provider->GetDmlDevice(dml_device.GetAddressOf())))
        return result;

    GemmShapes g = PrepareGemmShapes(a_sizes_in, b_sizes_in,
                                     trans_a, trans_b, trans_batch_a, trans_batch_b);
    if (!g.valid)
        return result;

    // TransA/B flags are suppressed for rank-1 inputs (no matrix dims to swap).
    // PrepareGemmShapes already expanded rank-1 to rank-2, so use original rank.
    size_t rank_a = a_sizes_in.size();
    size_t rank_b = b_sizes_in.size();

    auto& a_sizes   = g.a_sizes;
    auto& a_strides = g.a_strides;
    auto& b_sizes   = g.b_sizes;
    auto& b_strides = g.b_strides;
    auto& c_sizes   = g.c_sizes;


    auto elem_count = [](const std::vector<uint32_t>& s) -> size_t {
        size_t n = 1; for (auto v : s) n *= v; return n;
    };
    size_t elem_size = (dml_dtype == DML_TENSOR_DATA_TYPE_FLOAT32) ? 4 : 2;
    uint64_t c_bytes = static_cast<uint64_t>(elem_count(c_sizes) * elem_size);

    DML_BUFFER_TENSOR_DESC a_buf{};
    a_buf.DataType = dml_dtype; a_buf.DimensionCount = 4;
    a_buf.Sizes = a_sizes.data(); a_buf.Strides = a_strides.data();
    a_buf.TotalTensorSizeInBytes = a_bytes;
    DML_TENSOR_DESC a_desc{ DML_TENSOR_TYPE_BUFFER, &a_buf };

    DML_BUFFER_TENSOR_DESC b_buf{};
    b_buf.DataType = dml_dtype; b_buf.DimensionCount = 4;
    b_buf.Sizes = b_sizes.data(); b_buf.Strides = b_strides.data();
    b_buf.TotalTensorSizeInBytes = b_bytes;
    DML_TENSOR_DESC b_desc{ DML_TENSOR_TYPE_BUFFER, &b_buf };

    DML_BUFFER_TENSOR_DESC c_buf{};
    c_buf.DataType = dml_dtype; c_buf.DimensionCount = 4;
    c_buf.Sizes = c_sizes.data();
    // DML requires TotalTensorSizeInBytes to be DWORD-aligned (minimum 4).
    c_buf.TotalTensorSizeInBytes = c_bytes;
    DML_TENSOR_DESC c_desc{ DML_TENSOR_TYPE_BUFFER, &c_buf };

    ActivationDescStorage act_storage = BuildActivationDesc(activation, act_alpha, act_beta, act_gamma);

    DML_GEMM_OPERATOR_DESC gemm_desc{};
    gemm_desc.ATensor        = &a_desc;
    gemm_desc.BTensor        = &b_desc;
    gemm_desc.CTensor        = nullptr;
    gemm_desc.OutputTensor   = &c_desc;
    gemm_desc.TransA         = (trans_a && rank_a > 1) ? DML_MATRIX_TRANSFORM_TRANSPOSE : DML_MATRIX_TRANSFORM_NONE;
    gemm_desc.TransB         = (trans_b && rank_b > 1) ? DML_MATRIX_TRANSFORM_TRANSPOSE : DML_MATRIX_TRANSFORM_NONE;
    gemm_desc.Alpha          = alpha;
    gemm_desc.Beta           = 0.0f;
    gemm_desc.FusedActivation = act_storage.valid ? &act_storage.desc : nullptr;
    DML_OPERATOR_DESC gemm_opdesc{ DML_OPERATOR_GEMM, &gemm_desc };

    ComPtr<IDMLOperator> gemm_op;
    if (FAILED(dml_device->CreateOperator(&gemm_opdesc, IID_PPV_ARGS(&gemm_op))))
        return result;

    if (FAILED(dml_device->CompileOperator(gemm_op.Get(), DML_EXECUTION_FLAG_NONE,
                                            IID_PPV_ARGS(result.compiled_op.GetAddressOf()))))
        return result;

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
// Helper: read concrete tensor sizes from an OrtValue* at runtime.
// ---------------------------------------------------------------------------
static bool ReadRuntimeShape(
    const OrtApi&             api,
    const OrtValue*           value,
    DML_TENSOR_DATA_TYPE&     out_dml_dtype,
    std::vector<uint32_t>&    out_sizes,
    uint64_t&                 out_bytes)
{
    OrtTensorTypeAndShapeInfo* shape_info = nullptr;
    api.GetTensorTypeAndShape(const_cast<OrtValue*>(value), &shape_info);
    if (!shape_info) return false;

    size_t elem_count = 0; api.GetTensorShapeElementCount(shape_info, &elem_count);
    ONNXTensorElementDataType dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    api.GetTensorElementType(shape_info, &dtype);
    size_t rank = 0; api.GetDimensionsCount(shape_info, &rank);
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
        out_sizes[i] = static_cast<uint32_t>(dims[i] > 0 ? dims[i] : 1);
    out_bytes = static_cast<uint64_t>(elem_count * elem_size);
    return true;
}

// ---------------------------------------------------------------------------
// OrtNodeComputeInfo callbacks
// ---------------------------------------------------------------------------

static OrtStatus* ORT_API_CALL FusedMatMul_Compute(
    OrtNodeComputeInfo* this_ptr,
    void* compute_state,
    OrtKernelContext* kernel_context) noexcept
{
    auto* state = static_cast<FusedMatMulKernelState*>(compute_state);
    if (!state || !state->provider || !state->ort_api) return nullptr;
    const OrtApi& api = *state->ort_api;

    // Read runtime inputs.  When one side is an initializer (uploaded to GPU at
    // Compile time), read only the non-initializer side from the kernel context.
    // The initializer side's shape is read from the GPU-resident OrtValue stored
    // during Compile, and its D3D12 resource is bound directly.
    const OrtValue* a_value = nullptr;
    const OrtValue* b_value = nullptr;
    {
        if (state->initializer_side != 0) {
            OrtStatus* st = api.KernelContext_GetInput(kernel_context,
                static_cast<size_t>(state->graph_input_idx_a), &a_value);
            if (st || !a_value) { if (st) api.ReleaseStatus(st);
                return api.CreateStatus(ORT_FAIL, "FusedMatMul: failed to get input A"); }
        }
        if (state->initializer_side != 1) {
            OrtStatus* st = api.KernelContext_GetInput(kernel_context,
                static_cast<size_t>(state->graph_input_idx_b), &b_value);
            if (st || !b_value) { if (st) api.ReleaseStatus(st);
                return api.CreateStatus(ORT_FAIL, "FusedMatMul: failed to get input B"); }
        }
    }

    DML_TENSOR_DATA_TYPE dml_dtype = DML_TENSOR_DATA_TYPE_UNKNOWN;
    std::vector<uint32_t> a_sizes, b_sizes;
    uint64_t a_bytes = 0, b_bytes = 0;

    DML_TENSOR_DATA_TYPE dummy_dtype = DML_TENSOR_DATA_TYPE_UNKNOWN;

    if (a_value) {
        if (!ReadRuntimeShape(api, a_value, dml_dtype, a_sizes, a_bytes))
            return api.CreateStatus(ORT_FAIL, "FusedMatMul: failed to read shape of A");
    }
    if (b_value) {
        if (!ReadRuntimeShape(api, b_value, dummy_dtype, b_sizes, b_bytes))
            return api.CreateStatus(ORT_FAIL, "FusedMatMul: failed to read shape of B");
    }

    // For the initializer side, use the shape captured at Compile time.
    if (state->initializer_side == 0) {
        dml_dtype = dummy_dtype;
        a_bytes = state->initializer_bytes;
        a_sizes = state->initializer_sizes;
    } else if (state->initializer_side == 1) {
        b_bytes = state->initializer_bytes;
        b_sizes = state->initializer_sizes;
    }


    if (!state->initialized) {
        std::lock_guard<std::mutex> lock(state->init_mutex);
        if (!state->initialized) {
            state->kernel = CompileFusedMatMulDml(
                state->provider, state->trans_a, state->trans_b,
                state->trans_batch_a, state->trans_batch_b, state->alpha,
                state->activation, state->act_alpha, state->act_beta, state->act_gamma,
                dml_dtype, a_sizes, b_sizes, a_bytes, b_bytes);
            if (!state->kernel.IsValid())
                return api.CreateStatus(ORT_FAIL, "FusedMatMul: DML compilation failed");
            state->compiled_a_sizes = a_sizes;
            state->compiled_b_sizes = b_sizes;
            state->initialized = true;
        }
    }

    const FusedMatMulCompiledKernel* active_kernel = &state->kernel;
    FusedMatMulCompiledKernel temp_kernel;
    if (a_sizes != state->compiled_a_sizes || b_sizes != state->compiled_b_sizes) {
        temp_kernel = CompileFusedMatMulDml(
            state->provider, state->trans_a, state->trans_b,
            state->trans_batch_a, state->trans_batch_b, state->alpha,
            state->activation, state->act_alpha, state->act_beta, state->act_gamma,
            dml_dtype, a_sizes, b_sizes, a_bytes, b_bytes);
        if (!temp_kernel.IsValid())
            return api.CreateStatus(ORT_FAIL, "FusedMatMul: temporary kernel compilation failed");
        active_kernel = &temp_kernel;
    }

    GemmShapes g = PrepareGemmShapes(a_sizes, b_sizes,
                                     state->trans_a, state->trans_b,
                                     state->trans_batch_a, state->trans_batch_b);
    if (!g.valid)
        return api.CreateStatus(ORT_FAIL, "FusedMatMul: invalid shapes");

    // PrepareGemmShapes always pads c_sizes to 4D for DML. ORT expects the
    // output at the original logical rank — max(rank_a, rank_b), min 2 —
    // because downstream nodes were shaped against the original MatMul output.
    // Strip the leading padding 1s back down to the logical rank.
    size_t logical_rank = std::max(a_sizes.size(), b_sizes.size());
    if (logical_rank < 2) logical_rank = 2;
    size_t skip = g.c_sizes.size() > logical_rank ? g.c_sizes.size() - logical_rank : 0;
    std::vector<int64_t> out_dims;
    out_dims.reserve(logical_rank);
    for (size_t i = skip; i < g.c_sizes.size(); ++i)
        out_dims.push_back(static_cast<int64_t>(g.c_sizes[i]));

    size_t c_elem_count = 1;
    for (auto d : g.c_sizes) c_elem_count *= static_cast<size_t>(d);
    size_t elem_size = (dml_dtype == DML_TENSOR_DATA_TYPE_FLOAT32) ? 4 : 2;
    uint64_t c_bytes = static_cast<uint64_t>(c_elem_count * elem_size);


    OrtValue* c_value = nullptr;
    {
        OrtStatus* st = api.KernelContext_GetOutput(
            kernel_context, 0, out_dims.data(), out_dims.size(), &c_value);
        if (st || !c_value) { if (st) api.ReleaseStatus(st);
            return api.CreateStatus(ORT_FAIL, "FusedMatMul: failed to get output"); }
    }

    auto get_resource = [&](const OrtValue* value) -> ID3D12Resource* {
        void* raw = nullptr;
        OrtStatus* st = api.GetTensorMutableData(const_cast<OrtValue*>(value), &raw);
        if (st || !raw) { if (st) api.ReleaseStatus(st); return nullptr; }
        return state->provider->DecodeResource(raw);
    };

    ID3D12Resource* a_resource = (state->initializer_side == 0)
        ? state->initializer_gpu_resource.Get()
        : get_resource(a_value);
    ID3D12Resource* b_resource = (state->initializer_side == 1)
        ? state->initializer_gpu_resource.Get()
        : get_resource(b_value);
    ID3D12Resource* c_resource = get_resource(c_value);
    if (!a_resource || !b_resource || !c_resource)
        return api.CreateStatus(ORT_FAIL, "FusedMatMul: failed to get D3D12 resources");

    DML_BUFFER_BINDING a_binding{ a_resource, 0, a_bytes };
    DML_BUFFER_BINDING b_binding{ b_resource, 0, b_bytes };
    DML_BUFFER_BINDING c_out_binding{ c_resource, 0, c_bytes };

    // DML GEMM has 3 inputs (A, B, C) — bind the third as empty even when CTensor is null.
    DML_BINDING_DESC input_descs[3] = {
        { DML_BINDING_TYPE_BUFFER, &a_binding },
        { DML_BINDING_TYPE_BUFFER, &b_binding },
        { DML_BINDING_TYPE_NONE, nullptr },
    };
    DML_BINDING_DESC output_desc{ DML_BINDING_TYPE_BUFFER, &c_out_binding };

    const DML_BUFFER_BINDING* persistent = active_kernel->persistent_binding
        ? &*active_kernel->persistent_binding : nullptr;

    HRESULT hr = state->provider->ExecuteOperator(
        active_kernel->compiled_op.Get(),
        persistent,
        gsl::make_span(input_descs, 3),
        gsl::make_span(&output_desc, 1));

    if (FAILED(hr))
        return api.CreateStatus(ORT_FAIL, "FusedMatMul: ExecuteOperator failed");

    state->provider->QueueReference(active_kernel->compiled_op.Get());
    return nullptr;
}

// ---------------------------------------------------------------------------
// OrtNodeComputeInfo wrapper
// ---------------------------------------------------------------------------
struct FusedMatMulNodeComputeInfo : OrtNodeComputeInfo {
    FusedMatMulKernelState* state = nullptr;

    FusedMatMulNodeComputeInfo() {
        ort_version_supported = ORT_API_VERSION;
        CreateState = [](OrtNodeComputeInfo* self, OrtNodeComputeContext*, void** out) noexcept -> OrtStatus* {
            *out = static_cast<FusedMatMulNodeComputeInfo*>(self)->state;
            return nullptr;
        };
        Compute      = FusedMatMul_Compute;
        ReleaseState = [](OrtNodeComputeInfo*, void*) noexcept {};
    }

    ~FusedMatMulNodeComputeInfo() { delete state; }
};

// ---------------------------------------------------------------------------
// Compile
// ---------------------------------------------------------------------------
OrtNodeComputeInfo* FusedMatMulFusionRule::Compile(
    const OrtApi&                    ort_api,
    const OrtGraph*                  fused_subgraph,
    const std::unordered_map<std::string, const OrtValue*>& initializers,
    PluginDmlExecutionProviderImpl*  provider,
    const PatternMatch&              match) const
{
    // -------------------------------------------------------------------------
    // Step 1: resolve transpose flags from the subgraph Transpose nodes.
    //
    // BuildPattern() captures Transpose nodes under "FusedMatMul.trans_a" and
    // "FusedMatMul.trans_b". The TransposePermPredicate already validated that
    // the perm is fusable; here we re-read it to get the concrete flags.
    // -------------------------------------------------------------------------
    bool trans_a       = false;
    bool trans_b       = false;
    bool trans_batch_a = false;
    bool trans_batch_b = false;

    auto read_name = [&](const OrtValueInfo* vi) -> std::string {
        if (!vi) return {};
        const char* name = nullptr;
        OrtStatus* st = ort_api.GetValueInfoName(vi, &name);
        std::string result_str = (st || !name) ? std::string{} : std::string(name);
        if (st) ort_api.ReleaseStatus(st);
        return result_str;
    };

    // Trans flags were written into match.scalar_values by CapturePreFusionData() during
    // ApplyFusions — before any Transpose nodes were potentially excluded from
    // the fused subgraph.  Read them directly so they are available even when
    // the Transpose was kept in the graph for other consumers and is absent
    // from the fused subgraph's node list.
    trans_a       = match.ScalarValue("FusedMatMul.transA",     0.0f) != 0.0f;
    trans_b       = match.ScalarValue("FusedMatMul.transB",     0.0f) != 0.0f;
    trans_batch_a = match.ScalarValue("FusedMatMul.transBatchA", 0.0f) != 0.0f;
    trans_batch_b = match.ScalarValue("FusedMatMul.transBatchB", 0.0f) != 0.0f;

    // Walk subgraph nodes for other purposes (activation attrs, input resolution).
    size_t nn = 0;
    ort_api.Graph_GetNumNodes(fused_subgraph, &nn);
    std::vector<const OrtNode*> subgraph_nodes(nn, nullptr);
    if (nn > 0) ort_api.Graph_GetNodes(fused_subgraph, subgraph_nodes.data(), nn);

    // Locate the MatMul node's input value names.
    std::string matmul_a_name, matmul_b_name;
    for (const OrtNode* n : subgraph_nodes) {
        const char* op = nullptr;
        ort_api.Node_GetOperatorType(n, &op);
        if (!op || std::string_view(op) != "MatMul") continue;
        size_t ic = 0; ort_api.Node_GetNumInputs(n, &ic);
        std::vector<const OrtValueInfo*> ins(ic, nullptr);
        if (ic > 0) ort_api.Node_GetInputs(n, ins.data(), ic);
        if (ic >= 1) matmul_a_name = read_name(ins[0]);
        if (ic >= 2) matmul_b_name = read_name(ins[1]);
        break;
    }

    // -------------------------------------------------------------------------
    // Step 2: resolve alpha (scale) from the pattern match.
    //
    // BuildPattern() captures the scale initializer name via PInitializerWithMaxRank.
    // "FusedMatMul.scale_name" = Mul(scalar) branch.
    // "FusedMatMul.scale_div_name" = Div(scalar) branch.
    // -------------------------------------------------------------------------
    float alpha = 1.0f;
    {
        std::string scale_name = match.ValueName("FusedMatMul.scale_name");
        if (!scale_name.empty()) {
            auto it = initializers.find(scale_name);
            if (it != initializers.end()) {
                float v = 0.0f;
                if (fusion_utils::TryReadScalarFloat(ort_api, it->second, v)) alpha = v;
            }
        }
        std::string div_name = match.ValueName("FusedMatMul.scale_div_name");
        if (!div_name.empty()) {
            auto it = initializers.find(div_name);
            if (it != initializers.end()) {
                float v = 0.0f;
                if (fusion_utils::TryReadScalarFloat(ort_api, it->second, v) && v != 0.0f)
                    alpha = 1.0f / v;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Step 3: detect activation from the pattern match captures.
    //
    // BuildPattern() assigns a unique capture name to each activation branch.
    // We check which one is populated.
    // -------------------------------------------------------------------------
    FusedMatMulActivation activation = FusedMatMulActivation::None;
    if      (match.NodeIdx("FusedMatMul.relu")   != SIZE_MAX) activation = FusedMatMulActivation::Relu;
    else if (match.NodeIdx("FusedMatMul.sigmoid") != SIZE_MAX) activation = FusedMatMulActivation::Sigmoid;
    else if (match.NodeIdx("FusedMatMul.tanh")   != SIZE_MAX) activation = FusedMatMulActivation::Tanh;
    else if (match.NodeIdx("FusedMatMul.leaky")  != SIZE_MAX) activation = FusedMatMulActivation::LeakyRelu;
    else if (match.NodeIdx("FusedMatMul.elu")    != SIZE_MAX) activation = FusedMatMulActivation::Elu;
    else if (match.NodeIdx("FusedMatMul.hsig")   != SIZE_MAX) activation = FusedMatMulActivation::HardSigmoid;
    else if (match.NodeIdx("FusedMatMul.selu")   != SIZE_MAX) activation = FusedMatMulActivation::Selu;
    else if (match.NodeIdx("FusedMatMul.splus")  != SIZE_MAX) activation = FusedMatMulActivation::Softplus;
    else if (match.NodeIdx("FusedMatMul.ssign")  != SIZE_MAX) activation = FusedMatMulActivation::Softsign;
    else if (match.NodeIdx("FusedMatMul.threlu") != SIZE_MAX) activation = FusedMatMulActivation::ThresholdedRelu;

    // Read activation-specific attributes from the matched activation node.
    float act_alpha = 0.0f, act_beta = 0.0f, act_gamma = 0.0f;
    for (const OrtNode* n : subgraph_nodes) {
        const char* op = nullptr;
        ort_api.Node_GetOperatorType(n, &op);
        if (!op) continue;
        std::string_view op_sv(op);
        if (op_sv == "LeakyRelu" || op_sv == "Elu" || op_sv == "ThresholdedRelu") {
            OrtNodeAdapter adapter(n, ort_api);
            act_alpha = adapter.GetAttributeFloat("alpha", op_sv == "LeakyRelu" ? 0.01f : 1.0f);
        } else if (op_sv == "HardSigmoid") {
            OrtNodeAdapter adapter(n, ort_api);
            act_alpha = adapter.GetAttributeFloat("alpha", 0.2f);
            act_beta  = adapter.GetAttributeFloat("beta", 0.5f);
        } else if (op_sv == "Selu") {
            OrtNodeAdapter adapter(n, ort_api);
            act_alpha = adapter.GetAttributeFloat("alpha", 1.67326319f);
            act_gamma = adapter.GetAttributeFloat("gamma", 1.05070102f);
        }
    }

    // -------------------------------------------------------------------------
    // Step 4: resolve graph input indices for A and B.
    //
    // ORT may reorder fused subgraph inputs arbitrarily. We trace the MatMul
    // input value names (and any Transpose source names) back to the graph
    // boundary to find the correct input indices.
    // -------------------------------------------------------------------------
    size_t num_inputs = 0;
    ort_api.Graph_GetNumInputs(fused_subgraph, &num_inputs);
    if (num_inputs < 1) {
        // The fused subgraph has no graph inputs — all data comes from initializers
        // or intermediate edges ORT did not expose. Both MatMul inputs are constants
        // (ORT constant folding should have eliminated this) or ORT subgraph formation
        // issue. Either way, we cannot compile without a runtime input.
        return nullptr;
    }

    std::vector<const OrtValueInfo*> input_vis(num_inputs, nullptr);
    ort_api.Graph_GetInputs(fused_subgraph, input_vis.data(), num_inputs);

    auto find_graph_input = [&](const std::string& name) -> size_t {
        for (size_t gi = 0; gi < num_inputs; ++gi) {
            if (!input_vis[gi]) continue;
            if (read_name(input_vis[gi]) == name) return gi;
        }
        return SIZE_MAX;
    };

    // Transpose source names — captured by CapturePreFusionData into match.value_names.
    // Used as fallback when the Transpose was excluded from the fused subgraph
    // (ORT promotes its source tensor to a subgraph input instead).
    std::string source_a = match.ValueName("FusedMatMul.source_a");
    std::string source_b = match.ValueName("FusedMatMul.source_b");

    // Resolve graph input indices for A and B.  When one input is a constant
    // initializer it won't appear as a graph input — detect this and upload
    // the initializer to GPU (same pattern as BiasGelu's bias upload).
    size_t idx_a = find_graph_input(matmul_a_name);
    if (idx_a == SIZE_MAX && !source_a.empty()) idx_a = find_graph_input(source_a);
    size_t idx_b = find_graph_input(matmul_b_name);
    if (idx_b == SIZE_MAX && !source_b.empty()) idx_b = find_graph_input(source_b);

    int initializer_side = -1;  // -1 = neither, 0 = A, 1 = B
    size_t resolved_a_idx = 0, resolved_b_idx = (num_inputs >= 2) ? 1 : 0;

    if (idx_a != SIZE_MAX && idx_b != SIZE_MAX) {
        resolved_a_idx = idx_a;
        resolved_b_idx = idx_b;
    } else if (idx_a != SIZE_MAX && idx_b == SIZE_MAX) {
        resolved_a_idx = idx_a;
        initializer_side = 1;  // B is an initializer
    } else if (idx_a == SIZE_MAX && idx_b != SIZE_MAX) {
        resolved_b_idx = idx_b;
        initializer_side = 0;  // A is an initializer
    }
    // -------------------------------------------------------------------------
    // Step 5: determine element type and static shapes for eager compilation.
    // -------------------------------------------------------------------------
    auto get_elem_type = [&](const OrtValueInfo* vi) -> ONNXTensorElementDataType {
        if (!vi) return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        const OrtTypeInfo* ti = nullptr;
        OrtStatus* st = ort_api.GetValueInfoTypeInfo(vi, &ti);
        if (st || !ti) { if (st) ort_api.ReleaseStatus(st); return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED; }
        const OrtTensorTypeAndShapeInfo* si = nullptr;
        ort_api.CastTypeInfoToTensorInfo(ti, &si);
        if (!si) return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        ONNXTensorElementDataType etype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        ort_api.GetTensorElementType(si, &etype);
        return etype;
    };

    auto get_dims = [&](const OrtValueInfo* vi, std::vector<int64_t>& dims_out) -> size_t {
        if (!vi) return 0;
        const OrtTypeInfo* ti = nullptr;
        OrtStatus* st = ort_api.GetValueInfoTypeInfo(vi, &ti);
        if (st || !ti) { if (st) ort_api.ReleaseStatus(st); return 0; }
        const OrtTensorTypeAndShapeInfo* si = nullptr;
        ort_api.CastTypeInfoToTensorInfo(ti, &si);
        if (!si) return 0;
        size_t r = 0; ort_api.GetDimensionsCount(si, &r);
        dims_out.resize(r, -1);
        if (r > 0) ort_api.GetDimensions(si, dims_out.data(), r);
        return r;
    };

    ONNXTensorElementDataType elem_type = get_elem_type(input_vis[resolved_a_idx]);
    DML_TENSOR_DATA_TYPE dml_dtype = DML_TENSOR_DATA_TYPE_UNKNOWN;
    switch (elem_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   dml_dtype = DML_TENSOR_DATA_TYPE_FLOAT32; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: dml_dtype = DML_TENSOR_DATA_TYPE_FLOAT16; break;
        default:
            return nullptr;
    }

    std::vector<int64_t> a_dims, b_dims;
    size_t a_rank = get_dims(input_vis[resolved_a_idx], a_dims);
    size_t b_rank = get_dims(input_vis[resolved_b_idx], b_dims);

    if (trans_batch_a || trans_batch_b) {
        if (a_rank < 3 || a_rank != b_rank) {
            trans_batch_a = false;
            trans_batch_b = false;
        }
    }

    auto* kernel_state           = new FusedMatMulKernelState();
    kernel_state->trans_a        = trans_a;
    kernel_state->trans_b        = trans_b;
    kernel_state->trans_batch_a  = trans_batch_a;
    kernel_state->trans_batch_b  = trans_batch_b;
    kernel_state->alpha          = alpha;
    kernel_state->activation     = activation;
    kernel_state->act_alpha      = act_alpha;
    kernel_state->act_beta       = act_beta;
    kernel_state->act_gamma      = act_gamma;
    kernel_state->graph_input_idx_a = resolved_a_idx;
    kernel_state->graph_input_idx_b = resolved_b_idx;
    kernel_state->initializer_side = initializer_side;
    kernel_state->provider       = provider;
    kernel_state->ort_api        = &ort_api;

    // Upload the initializer input to GPU if one side is a constant.
    if (initializer_side >= 0) {
        const std::string& init_name = (initializer_side == 0) ? matmul_a_name : matmul_b_name;
        auto init_it = initializers.find(init_name);
        if (init_it == initializers.end() || !init_it->second) {
            delete kernel_state;
            return nullptr;
        }
        void* cpu_ptr = nullptr;
        OrtStatus* st = ort_api.GetTensorMutableData(
            const_cast<OrtValue*>(init_it->second), &cpu_ptr);
        if (st || !cpu_ptr) {
            if (st) ort_api.ReleaseStatus(st);
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
            ort_api.ReleaseTensorTypeAndShapeInfo(init_shape);
        }
        size_t elem_sz = (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) ? 2 : 4;
        kernel_state->initializer_bytes = static_cast<uint64_t>(init_elem_count * elem_sz);

        HRESULT hr = provider->AllocatePooledResource(
            static_cast<size_t>(kernel_state->initializer_bytes),
            AllocatorRoundingMode::Disabled,
            kernel_state->initializer_gpu_resource.GetAddressOf(),
            kernel_state->initializer_allocator_ref.GetAddressOf());
        if (FAILED(hr)) {
            delete kernel_state;
            return nullptr;
        }
        hr = provider->UploadToResource(
            kernel_state->initializer_gpu_resource.Get(), cpu_ptr,
            kernel_state->initializer_bytes);
        if (FAILED(hr)) {
            delete kernel_state;
            return nullptr;
        }
    }

    bool has_dynamic = false;
    for (auto d : a_dims) if (d <= 0) { has_dynamic = true; break; }
    for (auto d : b_dims) if (!has_dynamic && d <= 0) { has_dynamic = true; break; }

    if (!has_dynamic && a_rank >= 2 && b_rank >= 2) {
        size_t elem_sz = (dml_dtype == DML_TENSOR_DATA_TYPE_FLOAT32) ? 4 : 2;

        std::vector<uint32_t> a_sizes(a_rank), b_sizes(b_rank);
        size_t a_elems = 1, b_elems = 1;
        for (size_t i = 0; i < a_rank; ++i) { a_sizes[i] = static_cast<uint32_t>(a_dims[i]); a_elems *= a_sizes[i]; }
        for (size_t i = 0; i < b_rank; ++i) { b_sizes[i] = static_cast<uint32_t>(b_dims[i]); b_elems *= b_sizes[i]; }

        kernel_state->kernel = CompileFusedMatMulDml(
            provider, trans_a, trans_b, trans_batch_a, trans_batch_b, alpha,
            activation, act_alpha, act_beta, act_gamma,
            dml_dtype, a_sizes, b_sizes,
            static_cast<uint64_t>(a_elems * elem_sz),
            static_cast<uint64_t>(b_elems * elem_sz));
        if (!kernel_state->kernel.IsValid()) {
            delete kernel_state;
            return nullptr;
        }
        kernel_state->compiled_a_sizes = a_sizes;
        kernel_state->compiled_b_sizes = b_sizes;
        kernel_state->initialized = true;
    }

    auto* info  = new FusedMatMulNodeComputeInfo();
    info->state = kernel_state;
    return info;
}

}  // namespace dml_ep
