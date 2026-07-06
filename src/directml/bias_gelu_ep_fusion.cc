// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "bias_gelu_ep_fusion.h"
#include "fusion_utils.h"
#include "pattern_matcher.h"
#include "dml_execution_provider.h"
#include "DmlExecutionProvider/IExecutionProvider.h"
#include "dml_abi_kernel.h"  // DML_PERF_LOG

#include <DirectML.h>
#include <wrl/client.h>
#include <gsl/gsl>
#include <optional>
#include <vector>
#include <mutex>

namespace dml_ep {

using namespace fusion_utils;
using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// BiasGelu fusion — detects the raw ONNX expansion of Add(x, bias) + Gelu.
//
// ONNX does not have a native "BiasGelu" operation.  Instead, model exporters
// expand it into 5-6 primitive operations.  This fusion detects that expansion
// and replaces it with a single DirectML kernel (DML_ELEMENT_WISE_ADD1 with
// fused GELU activation), which is significantly faster.
//
// The expansion follows the mathematical definition of GELU:
//   Gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
//
// When preceded by a bias addition (Add(x, bias)), the full graph is:
//
//   Pattern 1 (6 nodes, upstream half-scale):
//
//                input    bias[1D]
//                  \       /
//        +---------Add---------+        "BiasGelu.bias_add"
//        |          |          |
//        |        Div(sqrt2)   |        "BiasGelu.div"
//        |          |          |
//        |         Erf         |        "BiasGelu.erf"  (anchor node)
//        |          |          |
//        |        Add(+1)      |        "BiasGelu.add1"
//        |          |          |
//        |    Mul(x, 0.5)      |        "BiasGelu.mul_half_p1"
//        |       \  |          |
//        +--------Mul----------+        "BiasGelu.mul_x"
//                  |
//               output
//
//   Pattern 2 (6 nodes, downstream half-scale):
//
//                input    bias[1D]
//                  \       /
//        +---------Add---------+        "BiasGelu.bias_add"
//        |          |          |
//        |        Div(sqrt2)   |        "BiasGelu.div"
//        |          |          |
//        |         Erf         |        "BiasGelu.erf"  (anchor node)
//        |          |          |
//        |        Add(+1)      |        "BiasGelu.add1"
//        |          |          |
//        +--------Mul----------+        "BiasGelu.mul_x"
//                  |
//              Mul(0.5)                 "BiasGelu.mul_half_p2"
//                  |
//               output
//
// Both variants are encoded as optional branches in one PNode tree.  The
// pattern engine's greedy longest-match strategy picks whichever is present.
//
// The pattern anchors on Erf because it is the most distinctive node —
// most models have far fewer Erf nodes than Add or Mul nodes, so the
// anchor index has fewer candidates to test.
//
// KEY CONSTRAINT: the bias input of the Add node must be a rank-1 constant
// (a vector).  This is validated at pattern-match time using a predicate
// constraint (PInitializerWithMaxRank), which prevents claiming nodes that
// would fail later during compilation.  See ORT's bias_gelu_fusion.cc for
// the equivalent check.
// ---------------------------------------------------------------------------

static constexpr float kSqrt2    = 1.41421356237f;
static constexpr float kSqrt2Tol = 1e-3f;  // tolerance to cover both float32/float16 representations

// ---------------------------------------------------------------------------
// Bias input predicate — mirrors ORT's bias_gelu_fusion.cc checks:
//   1. Must be a constant (in the initializer map)
//   2. Must be rank 1 (a bias vector)
//   3. Both Add inputs must have known shapes
//   4. Last dimension of bias must match last dimension of the other input
//
// Uses the node pointer to inspect the other Add input for cross-input
// validation.  If shapes are unavailable, the check is skipped (deferred
// to runtime).
// ---------------------------------------------------------------------------
static PInput PBiasInput(std::string capture_name) {
    PInput p = PPredicate(
        [](const std::string& name,
           const OrtValueInfo* vi,
           const OrtApi& ort_api,
           const std::unordered_map<std::string, const OrtValue*>& initializers,
           const OrtNode* node) -> bool {

            if (!initializers.count(name)) return false;

            // Helper: read rank and last dimension from an OrtValueInfo.
            // Returns {rank, last_dim}.  rank == 0 means shape is unknown.
            auto get_rank_and_last_dim = [&](const OrtValueInfo* v) -> std::pair<size_t, int64_t> {
                if (!v) return {0, -1};
                const OrtTypeInfo* ti = nullptr;
                OrtStatus* st = ort_api.GetValueInfoTypeInfo(v, &ti);
                if (st || !ti) { if (st) ort_api.ReleaseStatus(st); return {0, -1}; }
                const OrtTensorTypeAndShapeInfo* si = nullptr;
                ort_api.CastTypeInfoToTensorInfo(ti, &si);
                if (!si) return {0, -1};
                size_t rank = 0;
                ort_api.GetDimensionsCount(si, &rank);
                if (rank == 0) return {0, -1};
                std::vector<int64_t> dims(rank, -1);
                ort_api.GetDimensions(si, dims.data(), rank);
                return {rank, dims[rank - 1]};
            };

            auto [bias_rank, bias_last_dim] = get_rank_and_last_dim(vi);
            if (bias_rank == 0) return false;  // shape unknown — reject (ORT also rejects null shapes)
            if (bias_rank != 1) return false;  // bias must be rank 1

            if (!node) return true;  // can't check other input without node

            // Find the other Add input and verify last-dim match.
            size_t input_count = 0;
            ort_api.Node_GetNumInputs(node, &input_count);
            std::vector<const OrtValueInfo*> inputs(input_count, nullptr);
            if (input_count > 0) ort_api.Node_GetInputs(node, inputs.data(), input_count);

            for (size_t i = 0; i < input_count; ++i) {
                if (!inputs[i]) continue;
                const char* in_name = nullptr;
                OrtStatus* st = ort_api.GetValueInfoName(inputs[i], &in_name);
                if (st || !in_name) { if (st) ort_api.ReleaseStatus(st); continue; }
                if (std::string_view(in_name) == name) continue;  // skip self

                auto [other_rank, other_last_dim] = get_rank_and_last_dim(inputs[i]);
                if (other_rank == 0) return false;  // other shape unknown — reject

                // Both shapes known — last dimensions must match.
                if (bias_last_dim >= 0 && other_last_dim >= 0 &&
                    bias_last_dim != other_last_dim) {
                    return false;
                }
                break;
            }
            return true;
        });
    p.capture_name = std::move(capture_name);
    return p;
}

// ---------------------------------------------------------------------------
// IFusionRule implementation
// ---------------------------------------------------------------------------

PNode BiasGeluFusionRule::BuildPattern() const {
    return PNode("Erf").As("BiasGelu.erf")
        .Upstream(0,
            PNode("Div").As("BiasGelu.div")
                .Input(0, PCapture("BiasGelu.add_out"))
                .Input(1, PScalar(kSqrt2, kSqrt2Tol))
                .Upstream(0,
                    PNode("Add").As("BiasGelu.bias_add")
                        .Input(SIZE_MAX, PBiasInput("BiasGelu.bias_name"))))
        .Downstream(0,
            PNode("Add").As("BiasGelu.add1")
                .Input(SIZE_MAX, PScalarAtAny(1.0f))
                .Downstream(0,
                    PNode("Mul").As("BiasGelu.mul_x")
                        .Input(SIZE_MAX, PSameAsAtAny("BiasGelu.add_out"))
                        .Upstream(1,
                            PNode("Mul").As("BiasGelu.mul_half_p1")
                                .Input(SIZE_MAX, PSameAsAtAny("BiasGelu.add_out"))
                                .Input(SIZE_MAX, PScalarAtAny(0.5f)),
                            /*optional=*/true)
                        .Downstream(0,
                            PNode("Mul").As("BiasGelu.mul_half_p2")
                                .Input(1, PScalar(0.5f)),
                            /*optional=*/true)));
}

bool BiasGeluFusionRule::MatchesResult(const PatternMatch& m) const {
    return m.NodeIdx("BiasGelu.erf")      != SIZE_MAX
        && m.NodeIdx("BiasGelu.bias_add") != SIZE_MAX
        && !m.ValueName("BiasGelu.bias_name").empty();
}

// ---------------------------------------------------------------------------
// Kernel state — same lazy-init pattern as QuickGelu
// ---------------------------------------------------------------------------
struct BiasGeluCompiledKernel {
    ComPtr<IDMLCompiledOperator>      compiled_op;
    ComPtr<ID3D12Resource>            persistent_resource;
    ComPtr<IUnknown>                  persistent_allocator;
    std::optional<DML_BUFFER_BINDING> persistent_binding;

    bool IsValid() const { return compiled_op != nullptr; }
};

struct BiasGeluKernelState {
    PluginDmlExecutionProviderImpl* provider = nullptr;
    const OrtApi*                  ort_api  = nullptr;

    std::mutex                     init_mutex;
    bool                           initialized = false;
    BiasGeluCompiledKernel         kernel;

    // Shape used to compile the primary kernel — detects shape changes.
    std::vector<uint32_t>          compiled_input_sizes;

    // Bias GPU buffer — allocated and uploaded at Compile() time.
    ComPtr<ID3D12Resource>         bias_gpu_resource;
    ComPtr<IUnknown>               bias_allocator_ref;
    uint64_t                       bias_bytes = 0;
};

// ---------------------------------------------------------------------------
// Lazy DML compilation — called on first Compute with real input shapes.
// Builds DML_ELEMENT_WISE_ADD1 with fused DML_OPERATOR_ACTIVATION_GELU,
// matching DmlOperatorBiasGelu exactly.
// ---------------------------------------------------------------------------
static BiasGeluCompiledKernel CompileBiasGeluDml(
    PluginDmlExecutionProviderImpl* provider,
    DML_TENSOR_DATA_TYPE            dml_dtype,
    const std::vector<uint32_t>&    input_sizes,
    const std::vector<uint32_t>&    bias_sizes,
    uint64_t                        input_bytes,
    uint64_t                        bias_bytes)
{
    BiasGeluCompiledKernel result;

    ComPtr<IDMLDevice> dml_device;
    if (FAILED(provider->GetDmlDevice(dml_device.GetAddressOf()))) return result;

    DML_BUFFER_TENSOR_DESC input_buf{};
    input_buf.DataType              = dml_dtype;
    input_buf.Flags                 = DML_TENSOR_FLAG_NONE;
    input_buf.DimensionCount        = static_cast<UINT>(input_sizes.size());
    input_buf.Sizes                 = input_sizes.data();
    input_buf.Strides               = nullptr;
    input_buf.TotalTensorSizeInBytes = input_bytes;
    DML_TENSOR_DESC input_desc{ DML_TENSOR_TYPE_BUFFER, &input_buf };

    std::vector<uint32_t> bias_strides(input_sizes.size());
    {
        uint32_t stride = 1;
        for (int i = static_cast<int>(input_sizes.size()) - 1; i >= 0; --i) {
            bias_strides[i] = (bias_sizes[i] == 1) ? 0 : stride;
            if (bias_sizes[i] != 1) stride *= bias_sizes[i];
        }
    }

    DML_BUFFER_TENSOR_DESC bias_buf{};
    bias_buf.DataType               = dml_dtype;
    bias_buf.Flags                  = DML_TENSOR_FLAG_NONE;
    bias_buf.DimensionCount         = static_cast<UINT>(input_sizes.size());
    bias_buf.Sizes                  = input_sizes.data();
    bias_buf.Strides                = bias_strides.data();
    bias_buf.TotalTensorSizeInBytes  = bias_bytes;
    DML_TENSOR_DESC bias_desc{ DML_TENSOR_TYPE_BUFFER, &bias_buf };

    DML_BUFFER_TENSOR_DESC output_buf = input_buf;
    DML_TENSOR_DESC output_desc{ DML_TENSOR_TYPE_BUFFER, &output_buf };

    DML_ACTIVATION_GELU_OPERATOR_DESC gelu_desc{};
    DML_OPERATOR_DESC gelu_op_desc{ DML_OPERATOR_ACTIVATION_GELU, &gelu_desc };

    DML_ELEMENT_WISE_ADD1_OPERATOR_DESC add_desc{};
    add_desc.ATensor         = &input_desc;
    add_desc.BTensor         = &bias_desc;
    add_desc.FusedActivation = &gelu_op_desc;
    add_desc.OutputTensor    = &output_desc;
    DML_OPERATOR_DESC add_op_desc{ DML_OPERATOR_ELEMENT_WISE_ADD1, &add_desc };

    ComPtr<IDMLOperator> dml_op;
    if (FAILED(dml_device->CreateOperator(&add_op_desc, IID_PPV_ARGS(&dml_op)))) return result;

    if (FAILED(dml_device->CompileOperator(dml_op.Get(), DML_EXECUTION_FLAG_NONE,
                                           IID_PPV_ARGS(result.compiled_op.GetAddressOf())))) {
        DML_PERF_LOG("[BiasGelu] CompileBiasGeluDml: CompileOperator FAILED");
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

    const DML_BUFFER_BINDING* persistent_ptr = result.persistent_binding
        ? &*result.persistent_binding : nullptr;
    if (FAILED(provider->InitializeOperator(
            result.compiled_op.Get(), persistent_ptr,
            gsl::span<const DML_BUFFER_BINDING>{}))) {
        result.compiled_op.Reset();
        return result;
    }

    provider->QueueReference(result.compiled_op.Get());
    if (result.persistent_allocator) {
        provider->QueueReference(result.persistent_allocator.Get());
    }

    return result;
}

// ---------------------------------------------------------------------------
// Compute — lazy-init on first call, then dispatch
// ---------------------------------------------------------------------------
static OrtStatus* ORT_API_CALL BiasGelu_Compute(
    OrtNodeComputeInfo* /*this_ptr*/,
    void* compute_state,
    OrtKernelContext* kernel_context) noexcept
{
    auto* state = static_cast<BiasGeluKernelState*>(compute_state);
    if (!state || !state->provider || !state->ort_api) return nullptr;

    const OrtApi& api = *state->ort_api;
    // The fused kernel receives only the main (non-initializer) input at runtime.
    // The bias is a model weight already on the GPU — its handle is stored in state.
    const OrtValue* input0 = nullptr;
    {
        OrtStatus* st = api.KernelContext_GetInput(kernel_context, 0, &input0);
        if (st || !input0) { if (st) api.ReleaseStatus(st);
            return api.CreateStatus(ORT_FAIL, "BiasGelu: failed to get main input"); }
    }

    OrtTensorTypeAndShapeInfo* shape0 = nullptr;
    api.GetTensorTypeAndShape(const_cast<OrtValue*>(input0), &shape0);
    size_t elem_count0 = 0; api.GetTensorShapeElementCount(shape0, &elem_count0);
    ONNXTensorElementDataType dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    api.GetTensorElementType(shape0, &dtype);
    size_t main_rank = 0; api.GetDimensionsCount(shape0, &main_rank);
    std::vector<int64_t> main_dims(main_rank);
    if (main_rank > 0) api.GetDimensions(shape0, main_dims.data(), main_rank);
    api.ReleaseTensorTypeAndShapeInfo(shape0);

    size_t elem_size = (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) ? 2 : 4;
    uint64_t main_bytes      = static_cast<uint64_t>(elem_count0 * elem_size);
    uint64_t bias_bytes_actual = state->bias_bytes;

    DML_TENSOR_DATA_TYPE dml_dtype = DML_TENSOR_DATA_TYPE_UNKNOWN;
    if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)   dml_dtype = DML_TENSOR_DATA_TYPE_FLOAT32;
    if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) dml_dtype = DML_TENSOR_DATA_TYPE_FLOAT16;
    if (dml_dtype == DML_TENSOR_DATA_TYPE_UNKNOWN) {
        return api.CreateStatus(ORT_FAIL, "BiasGelu: unsupported dtype");
    }

    std::vector<uint32_t> input_sizes(main_rank);
    for (size_t i = 0; i < main_rank; ++i) {
        input_sizes[i] = static_cast<uint32_t>(main_dims[i] > 0 ? main_dims[i] : 1);
    }

    size_t bias_elems = state->bias_bytes / elem_size;
    std::vector<uint32_t> bias_sizes(main_rank, 1);
    bias_sizes[main_rank - 1] = static_cast<uint32_t>(bias_elems);

    // Lazy-init: compile on first Compute with real shapes.
    if (!state->initialized) {
        std::lock_guard<std::mutex> lock(state->init_mutex);
        if (!state->initialized) {
            state->kernel = CompileBiasGeluDml(
                state->provider, dml_dtype, input_sizes, bias_sizes,
                main_bytes, state->bias_bytes);
            if (!state->kernel.IsValid()) {
                return api.CreateStatus(ORT_FAIL, "BiasGelu: DML compilation failed");
            }
            state->compiled_input_sizes = input_sizes;
            state->initialized = true;
        }
    }

    // Shape-change: compile a temporary kernel for this call only, then discard.
    const BiasGeluCompiledKernel* active_kernel = &state->kernel;
    BiasGeluCompiledKernel temp_kernel;
    if (input_sizes != state->compiled_input_sizes) {
        temp_kernel = CompileBiasGeluDml(
            state->provider, dml_dtype, input_sizes, bias_sizes,
            main_bytes, state->bias_bytes);
        if (!temp_kernel.IsValid()) {
            return api.CreateStatus(ORT_FAIL, "BiasGelu: temporary kernel compilation failed");
        }
        active_kernel = &temp_kernel;
    }

    // Allocate output (same shape as main input).
    OrtValue* output = nullptr;
    {
        OrtStatus* st = api.KernelContext_GetOutput(
            kernel_context, 0, main_dims.data(), main_rank, &output);
        if (st || !output) { if (st) api.ReleaseStatus(st);
            return api.CreateStatus(ORT_FAIL, "BiasGelu: failed to get output"); }
    }

    // Get D3D12 handle for the main input tensor.
    auto get_resource = [&](const OrtValue* v) -> ID3D12Resource* {
        void* raw = nullptr;
        OrtStatus* st = api.GetTensorMutableData(const_cast<OrtValue*>(v), &raw);
        if (st || !raw) { if (st) api.ReleaseStatus(st); return nullptr; }
        return state->provider->DecodeResource(raw);
    };

    ID3D12Resource* r_main = get_resource(input0);
    ID3D12Resource* r_bias = state->bias_gpu_resource.Get();
    ID3D12Resource* r_out  = get_resource(output);
    if (!r_main || !r_bias || !r_out) {
        return api.CreateStatus(ORT_FAIL, "BiasGelu: failed to get D3D12 resources");
    }

    DML_BUFFER_BINDING b0   { r_main, 0, main_bytes        };
    DML_BUFFER_BINDING b1   { r_bias, 0, bias_bytes_actual };
    DML_BUFFER_BINDING bout { r_out,  0, main_bytes        };

    DML_BINDING_DESC d0   { DML_BINDING_TYPE_BUFFER, &b0   };
    DML_BINDING_DESC d1   { DML_BINDING_TYPE_BUFFER, &b1   };
    DML_BINDING_DESC dout { DML_BINDING_TYPE_BUFFER, &bout };

    DML_BINDING_DESC inputs[] = { d0, d1 };

    const DML_BUFFER_BINDING* persistent = active_kernel->persistent_binding
        ? &*active_kernel->persistent_binding : nullptr;

    HRESULT hr = state->provider->ExecuteOperator(
        active_kernel->compiled_op.Get(),
        persistent,
        gsl::make_span(inputs, 2),
        gsl::make_span(&dout, 1));

    if (FAILED(hr)) {
        return api.CreateStatus(ORT_FAIL, "BiasGelu: ExecuteOperator failed");
    }

    state->provider->QueueReference(active_kernel->compiled_op.Get());
    return nullptr;
}

static void ORT_API_CALL BiasGelu_ReleaseState(
    OrtNodeComputeInfo* /*this_ptr*/,
    void* compute_state) noexcept
{
    delete static_cast<BiasGeluKernelState*>(compute_state);
}

struct BiasGeluNodeComputeInfo : OrtNodeComputeInfo {
    BiasGeluKernelState* state = nullptr;

    BiasGeluNodeComputeInfo() {
        ort_version_supported = ORT_API_VERSION;
        CreateState = [](OrtNodeComputeInfo* self, OrtNodeComputeContext*, void** out) noexcept -> OrtStatus* {
            *out = static_cast<BiasGeluNodeComputeInfo*>(self)->state;
            return nullptr;
        };
        Compute      = BiasGelu_Compute;
        ReleaseState = [](OrtNodeComputeInfo*, void*) noexcept {};
    }

    ~BiasGeluNodeComputeInfo() { delete state; }
};

// ---------------------------------------------------------------------------
// Compile — deferred (shapes required at DML operator creation time).
// `match` carries the pattern captures so we don't re-detect bias name etc.
// ---------------------------------------------------------------------------
OrtNodeComputeInfo* BiasGeluFusionRule::Compile(
    const OrtApi&                    ort_api,
    const OrtGraph*                  fused_subgraph,
    const std::unordered_map<std::string, const OrtValue*>& initializers,
    PluginDmlExecutionProviderImpl*  provider,
    const PatternMatch&              match) const
{
    // The pattern matcher captured the bias name via PInitializerWithMaxRank.
    const std::string bias_name_str = match.ValueName("BiasGelu.bias_name");
    if (bias_name_str.empty()) {
        DML_PERF_LOG("[BiasGelu] Compile: missing bias_name capture");
        return nullptr;
    }
    const char* bias_name = bias_name_str.c_str();

    // Find the main (non-bias) runtime tensor from the fused subgraph inputs.
    // The bias is an initializer; the main tensor is the only non-initializer input.
    size_t num_inputs = 0;
    ort_api.Graph_GetNumInputs(fused_subgraph, &num_inputs);
    if (num_inputs < 1) {
        DML_PERF_LOG("[BiasGelu] Compile: no graph inputs");
        return nullptr;
    }
    std::vector<const OrtValueInfo*> all_input_vis(num_inputs, nullptr);
    ort_api.Graph_GetInputs(fused_subgraph, all_input_vis.data(), num_inputs);

    const OrtValueInfo* main_vi = nullptr;
    for (const OrtValueInfo* vi : all_input_vis) {
        if (!vi) continue;
        const char* name = nullptr;
        OrtStatus* st = ort_api.GetValueInfoName(vi, &name);
        if (st || !name) { if (st) ort_api.ReleaseStatus(st); continue; }
        if (std::string_view(name) != bias_name_str) { main_vi = vi; break; }
    }
    if (!main_vi) {
        DML_PERF_LOG("[BiasGelu] Compile: could not identify main input");
        return nullptr;
    }

    // Find the bias value info from subgraph initializers by name.
    const OrtValueInfo* bias_vi = nullptr;
    {
        size_t num_init = 0;
        ort_api.Graph_GetNumInitializers(fused_subgraph, &num_init);
        std::vector<const OrtValueInfo*> init_vis(num_init, nullptr);
        if (num_init > 0) ort_api.Graph_GetInitializers(fused_subgraph, init_vis.data(), num_init);
        for (const OrtValueInfo* vi : init_vis) {
            if (!vi) continue;
            const char* name = nullptr;
            OrtStatus* st = ort_api.GetValueInfoName(vi, &name);
            if (st || !name) { if (st) ort_api.ReleaseStatus(st); continue; }
            if (std::string_view(name) == bias_name_str) { bias_vi = vi; break; }
        }
    }
    if (!bias_vi) {
        DML_PERF_LOG("[BiasGelu] Compile: bias not found in subgraph initializers");
        return nullptr;
    }

    // Read static dimensions for both inputs.
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

    std::vector<int64_t> main_dims_v, bias_dims_v;
    size_t main_rank = get_dims(main_vi, main_dims_v);
    size_t bias_rank = get_dims(bias_vi, bias_dims_v);
    std::vector<int64_t>* main_dims = &main_dims_v;
    std::vector<int64_t>* bias_dims = &bias_dims_v;

    // Get element type from the main input.
    ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    {
        const OrtTypeInfo* ti = nullptr;
        OrtStatus* st = ort_api.GetValueInfoTypeInfo(main_vi, &ti);
        if (!st && ti) {
            const OrtTensorTypeAndShapeInfo* si = nullptr;
            ort_api.CastTypeInfoToTensorInfo(ti, &si);
            if (si) ort_api.GetTensorElementType(si, &elem_type);
        }
        if (st) ort_api.ReleaseStatus(st);
    }

    auto* kernel_state     = new BiasGeluKernelState();
    kernel_state->provider = provider;
    kernel_state->ort_api  = &ort_api;

    // Allocate a GPU buffer for the bias and upload the CPU data using the upload
    // heap. UploadToResource is available during CompileImpl (session init), so
    // this happens once at startup — no per-inference cost.
    {
        auto bias_it = initializers.find(bias_name);
        if (bias_it == initializers.end() || !bias_it->second) {
            DML_PERF_LOG("[BiasGelu] Compile: bias not found in initializers, skipping");
            delete kernel_state;
            return nullptr;
        }

        void* cpu_ptr = nullptr;
        OrtStatus* st = ort_api.GetTensorMutableData(
            const_cast<OrtValue*>(bias_it->second), &cpu_ptr);
        if (st || !cpu_ptr) { if (st) ort_api.ReleaseStatus(st);
            DML_PERF_LOG("[BiasGelu] Compile: failed to get bias CPU data, skipping");
            delete kernel_state;
            return nullptr;
        }

        size_t elem_size_b = (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) ? 2 : 4;
        size_t bias_elems_b = 1;
        for (auto d : *bias_dims) if (d > 0) bias_elems_b *= static_cast<size_t>(d);
        kernel_state->bias_bytes = static_cast<uint64_t>(bias_elems_b * elem_size_b);

        // Allocate GPU buffer through the EP's pooled allocator.
        // AllocatePooledResource uses DmlBucketizedBufferAllocator which:
        //   - rounds size to a power-of-two bucket and reuses freed resources
        //   - calls DmlCommittedResourceAllocator::Alloc for new resources
        //     (CreateCommittedResource with ALLOW_UNORDERED_ACCESS + COMMON)
        //   - registers the resource in the allocator's tracking map
        // Registration is critical — bypassing it with CreateCommittedResource
        // directly corrupts the allocator's bookkeeping and crashes CopyTensorsPlugin.
        HRESULT hr = provider->AllocatePooledResource(
            static_cast<size_t>(kernel_state->bias_bytes),
            AllocatorRoundingMode::Disabled,
            kernel_state->bias_gpu_resource.GetAddressOf(),
            kernel_state->bias_allocator_ref.GetAddressOf());
        if (FAILED(hr)) {
            DML_PERF_LOG("[BiasGelu] Compile: failed to allocate bias GPU buffer, skipping");
            delete kernel_state;
            return nullptr;
        }
        // bias_allocator_ref is stored in kernel_state — it must outlive the
        // session to prevent the pool reclaiming the buffer after a GPU sync.

        // Upload CPU data to GPU.
        hr = provider->UploadToResource(
            kernel_state->bias_gpu_resource.Get(), cpu_ptr, kernel_state->bias_bytes);
        if (FAILED(hr)) {
            DML_PERF_LOG("[BiasGelu] Compile: failed to upload bias to GPU, skipping");
            delete kernel_state;
            return nullptr;
        }

    }

    // Bias is now on GPU. If all dims are also concrete, compile DML eagerly now.
    // If shapes are dynamic, defer to first Compute() when real dims are known.
    bool has_dynamic = false;
    if (main_rank == 0 || bias_rank == 0 || bias_rank > main_rank) has_dynamic = true;
    for (auto d : *main_dims) if (d <= 0) { has_dynamic = true; break; }
    for (auto d : *bias_dims) if (!has_dynamic && d <= 0) { has_dynamic = true; break; }

    if (!has_dynamic && elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
        DML_TENSOR_DATA_TYPE dml_dtype = DML_TENSOR_DATA_TYPE_UNKNOWN;
        if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)   dml_dtype = DML_TENSOR_DATA_TYPE_FLOAT32;
        if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) dml_dtype = DML_TENSOR_DATA_TYPE_FLOAT16;

        if (dml_dtype != DML_TENSOR_DATA_TYPE_UNKNOWN) {
            size_t elem_size = (dml_dtype == DML_TENSOR_DATA_TYPE_FLOAT32) ? 4 : 2;

            std::vector<uint32_t> input_sizes(main_rank);
            size_t total_elems = 1;
            for (size_t i = 0; i < main_rank; ++i) {
                input_sizes[i] = static_cast<uint32_t>((*main_dims)[i]);
                total_elems *= input_sizes[i];
            }
            uint64_t input_bytes = static_cast<uint64_t>(total_elems * elem_size);

            // Broadcast bias shape to match main input rank.
            std::vector<uint32_t> bias_sizes(main_rank, 1);
            size_t bias_offset = main_rank - bias_rank;
            size_t bias_elems = 1;
            for (size_t i = 0; i < bias_rank; ++i) {
                bias_sizes[bias_offset + i] = static_cast<uint32_t>((*bias_dims)[i]);
                bias_elems *= bias_sizes[bias_offset + i];
            }
            uint64_t bias_bytes = static_cast<uint64_t>(bias_elems * elem_size);

            kernel_state->kernel = CompileBiasGeluDml(
                provider, dml_dtype, input_sizes, bias_sizes,
                input_bytes, bias_bytes);
            if (kernel_state->kernel.IsValid()) {
                kernel_state->compiled_input_sizes = input_sizes;
                kernel_state->initialized = true;
            } else {
                DML_PERF_LOG("[BiasGelu] Compile: eager compilation failed, falling back to DEFERRED");
            }
        }
    }

    auto* info  = new BiasGeluNodeComputeInfo();
    info->state = kernel_state;
    return info;
}

}  // namespace dml_ep
