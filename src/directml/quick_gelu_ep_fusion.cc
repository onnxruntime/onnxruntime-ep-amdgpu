// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "quick_gelu_ep_fusion.h"
#include "fusion_utils.h"
#include "pattern_matcher.h"
#include "dml_execution_provider.h"   // PluginDmlExecutionProviderImpl
#include "DmlExecutionProvider/IExecutionProvider.h"
#include "dml_abi_kernel.h"           // DML_PERF_LOG

#include <DirectML.h>
#include <wrl/client.h>
#include <gsl/gsl>
#include <optional>
#include <vector>

namespace dml_ep {

using namespace fusion_utils;
using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// IFusionRule implementation
// ---------------------------------------------------------------------------

PNode QuickGeluFusionRule::BuildPattern() const {
    // Capture names prefixed "QuickGelu." to avoid conflicts when merged with
    // other rules sharing the "Sigmoid" anchor.
    // QuickGelu: x * sigmoid(alpha * x)
    //
    // 2-node variant (alpha=1):   x → Sigmoid(x) → Mul(x, sigmoid_out)
    // 3-node variant (alpha≠1):   x → Mul(x, alpha) → Sigmoid(alpha*x) → Mul(x, sigmoid_out)
    //
    // PSameAs on first use acts as a capture.  The upstream alpha_mul (processed
    // first when present) captures x from its non-scalar input.  The downstream
    // output_mul then validates that the same x appears as one of its inputs.
    // When the upstream branch is absent (2-node variant), the downstream Mul's
    // PSameAs is the first use and captures x directly.
    // Alpha predicate: must be a constant initializer with exactly 1 element.
    // Uses TryReadScalarFloat for validation (handles fp32, fp64, fp16).
    PInput alpha_input = PPredicate(
        [](const std::string& name, const OrtValueInfo*,
           const OrtApi& ort_api,
           const std::unordered_map<std::string, const OrtValue*>& initializers,
           const OrtNode*) -> bool {
            auto it = initializers.find(name);
            if (it == initializers.end()) return false;
            float v = 0.0f;
            return fusion_utils::TryReadScalarFloat(ort_api, it->second, v);
        });
    alpha_input.capture_name = "QuickGelu.alpha";

    // 2-node: PCapture on Sigmoid captures x directly. Downstream Mul validates.
    // 3-node: PCapture on Sigmoid captures alpha*x (wrong). The upstream alpha_mul
    //         predicate overwrites "QuickGelu.input" with x (the non-scalar input).
    //         Downstream Mul then validates that x appears as one of its inputs.
    PInput alpha_mul_x_input = PPredicate(
        [](const std::string& name, const OrtValueInfo*,
           const OrtApi&,
           const std::unordered_map<std::string, const OrtValue*>& initializers,
           const OrtNode*) -> bool {
            return initializers.find(name) == initializers.end();
        });
    alpha_mul_x_input.capture_name = "QuickGelu.input";

    return PNode("Sigmoid").As("QuickGelu.sigmoid")
        .Input(0, PCapture("QuickGelu.input"))
        .Upstream(0,
            PNode("Mul").As("QuickGelu.alpha_mul")
                .Input(SIZE_MAX, std::move(alpha_mul_x_input))
                .Input(SIZE_MAX, std::move(alpha_input)),
            /*optional=*/true)
        .Downstream(0,
            PNode("Mul").As("QuickGelu.output_mul")
                .Input(SIZE_MAX, PSameAs("QuickGelu.input")));
}

bool QuickGeluFusionRule::MatchesResult(const PatternMatch& m) const {
    return m.NodeIdx("QuickGelu.sigmoid") != SIZE_MAX;
}

// ---------------------------------------------------------------------------
// Kernel state — stored per session, shared across inferences
//
// Mirrors the original ORT DML EP's LAZY_INIT pattern: the DML graph is not
// compiled at CompileImpl time (shapes are unknown then), but instead on the
// first Compute call when real input shapes are available from OrtKernelContext.
// ---------------------------------------------------------------------------
// A compiled DML operator + its persistent resource, used for both the
// primary cached kernel and temporary shape-change kernels.
struct QuickGeluCompiledKernel {
    ComPtr<IDMLCompiledOperator>      compiled_op;
    ComPtr<ID3D12Resource>            persistent_resource;
    ComPtr<IUnknown>                  persistent_allocator;
    std::optional<DML_BUFFER_BINDING> persistent_binding;

    bool IsValid() const { return compiled_op != nullptr; }
};

struct QuickGeluKernelState {
    // Set at CompileImpl time (always available)
    float                          alpha = 1.0f;
    PluginDmlExecutionProviderImpl* provider = nullptr;  // non-owning, session lifetime
    const OrtApi*                  ort_api = nullptr;

    // Primary compiled kernel (lazy-initialized on first Compute).
    std::mutex                     init_mutex;
    bool                           initialized = false;
    QuickGeluCompiledKernel        kernel;

    // Shape used to compile the primary kernel — used to detect shape changes.
    std::vector<uint32_t>          compiled_sizes;
};

// ---------------------------------------------------------------------------
// Compile a QuickGelu DML graph for the given concrete shapes.
// Returns a QuickGeluCompiledKernel; caller decides whether to cache it
// (primary init) or use it temporarily (shape change) then discard.
// ---------------------------------------------------------------------------
static QuickGeluCompiledKernel CompileQuickGeluDml(
    PluginDmlExecutionProviderImpl*  provider,
    float                            alpha,
    DML_TENSOR_DATA_TYPE             dml_dtype,
    const std::vector<uint32_t>&     sizes,
    uint64_t                         total_bytes)
{
    QuickGeluCompiledKernel result;

    ComPtr<IDMLDevice> dml_device;
    if (FAILED(provider->GetDmlDevice(dml_device.GetAddressOf()))) return result;

    ComPtr<IDMLDevice1> dml_device1;
    if (FAILED(dml_device.As(&dml_device1))) return result;

    DML_BUFFER_TENSOR_DESC buf_tensor_desc{};
    buf_tensor_desc.DataType              = dml_dtype;
    buf_tensor_desc.Flags                 = DML_TENSOR_FLAG_NONE;
    buf_tensor_desc.DimensionCount        = static_cast<UINT>(sizes.size());
    buf_tensor_desc.Sizes                 = sizes.data();
    buf_tensor_desc.Strides               = nullptr;
    buf_tensor_desc.TotalTensorSizeInBytes = total_bytes;
    buf_tensor_desc.GuaranteedBaseOffsetAlignment = 0;
    DML_TENSOR_DESC tensor_desc{ DML_TENSOR_TYPE_BUFFER, &buf_tensor_desc };

    enum NodeIndex : UINT { kSigmoid = 0, kMultiply = 1, kMulAlpha = 2 };

    DML_ACTIVATION_SIGMOID_OPERATOR_DESC sigmoid_desc{};
    sigmoid_desc.InputTensor  = &tensor_desc;
    sigmoid_desc.OutputTensor = &tensor_desc;
    DML_OPERATOR_DESC dml_sigmoid_opdesc{ DML_OPERATOR_ACTIVATION_SIGMOID, &sigmoid_desc };

    DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC multiply_desc{};
    multiply_desc.ATensor      = &tensor_desc;
    multiply_desc.BTensor      = &tensor_desc;
    multiply_desc.OutputTensor = &tensor_desc;
    DML_OPERATOR_DESC dml_multiply_opdesc{ DML_OPERATOR_ELEMENT_WISE_MULTIPLY, &multiply_desc };

    DML_SCALE_BIAS scale_bias{ alpha, 0.0f };
    DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC mul_alpha_desc{};
    mul_alpha_desc.InputTensor  = &tensor_desc;
    mul_alpha_desc.OutputTensor = &tensor_desc;
    mul_alpha_desc.ScaleBias    = (alpha != 1.0f) ? &scale_bias : nullptr;
    DML_OPERATOR_DESC dml_mul_alpha_opdesc{ DML_OPERATOR_ELEMENT_WISE_IDENTITY, &mul_alpha_desc };

    ComPtr<IDMLOperator> sigmoid_op, multiply_op, mul_alpha_op;
    if (FAILED(dml_device->CreateOperator(&dml_sigmoid_opdesc,  IID_PPV_ARGS(&sigmoid_op))))  return result;
    if (FAILED(dml_device->CreateOperator(&dml_multiply_opdesc, IID_PPV_ARGS(&multiply_op)))) return result;
    if (alpha != 1.0f) {
        if (FAILED(dml_device->CreateOperator(&dml_mul_alpha_opdesc, IID_PPV_ARGS(&mul_alpha_op)))) return result;
    }

    DML_OPERATOR_GRAPH_NODE_DESC sigmoid_node_desc  { sigmoid_op.Get(),  nullptr };
    DML_OPERATOR_GRAPH_NODE_DESC multiply_node_desc { multiply_op.Get(), nullptr };
    DML_OPERATOR_GRAPH_NODE_DESC mul_alpha_node_desc{ mul_alpha_op ? mul_alpha_op.Get() : nullptr, nullptr };

    DML_GRAPH_NODE_DESC sigmoid_node  { DML_GRAPH_NODE_TYPE_OPERATOR, &sigmoid_node_desc };
    DML_GRAPH_NODE_DESC multiply_node { DML_GRAPH_NODE_TYPE_OPERATOR, &multiply_node_desc };
    DML_GRAPH_NODE_DESC mul_alpha_node{ DML_GRAPH_NODE_TYPE_OPERATOR, &mul_alpha_node_desc };

    std::vector<DML_GRAPH_NODE_DESC> graph_nodes;
    graph_nodes.push_back(sigmoid_node);
    graph_nodes.push_back(multiply_node);
    if (alpha != 1.0f) graph_nodes.push_back(mul_alpha_node);

    std::vector<DML_GRAPH_EDGE_DESC> input_edge_descs, intermediate_edge_descs, output_edge_descs;
    DML_INPUT_GRAPH_EDGE_DESC        in_to_alpha{}, in_to_sigmoid{}, in_to_multiply{};
    DML_INTERMEDIATE_GRAPH_EDGE_DESC alpha_to_sigmoid{}, sigmoid_to_multiply{};
    DML_OUTPUT_GRAPH_EDGE_DESC       multiply_to_out{};

    if (alpha != 1.0f) {
        in_to_alpha.GraphInputIndex  = 0; in_to_alpha.ToNodeIndex = kMulAlpha; in_to_alpha.ToNodeInputIndex = 0;
        input_edge_descs.push_back({ DML_GRAPH_EDGE_TYPE_INPUT, &in_to_alpha });
        alpha_to_sigmoid.FromNodeIndex = kMulAlpha; alpha_to_sigmoid.FromNodeOutputIndex = 0;
        alpha_to_sigmoid.ToNodeIndex   = kSigmoid;  alpha_to_sigmoid.ToNodeInputIndex    = 0;
        intermediate_edge_descs.push_back({ DML_GRAPH_EDGE_TYPE_INTERMEDIATE, &alpha_to_sigmoid });
    } else {
        in_to_sigmoid.GraphInputIndex  = 0; in_to_sigmoid.ToNodeIndex = kSigmoid; in_to_sigmoid.ToNodeInputIndex = 0;
        input_edge_descs.push_back({ DML_GRAPH_EDGE_TYPE_INPUT, &in_to_sigmoid });
    }

    in_to_multiply.GraphInputIndex  = 0; in_to_multiply.ToNodeIndex = kMultiply; in_to_multiply.ToNodeInputIndex = 0;
    input_edge_descs.push_back({ DML_GRAPH_EDGE_TYPE_INPUT, &in_to_multiply });

    sigmoid_to_multiply.FromNodeIndex = kSigmoid; sigmoid_to_multiply.FromNodeOutputIndex = 0;
    sigmoid_to_multiply.ToNodeIndex   = kMultiply; sigmoid_to_multiply.ToNodeInputIndex   = 1;
    intermediate_edge_descs.push_back({ DML_GRAPH_EDGE_TYPE_INTERMEDIATE, &sigmoid_to_multiply });

    multiply_to_out.FromNodeIndex = kMultiply; multiply_to_out.FromNodeOutputIndex = 0;
    multiply_to_out.GraphOutputIndex = 0;
    output_edge_descs.push_back({ DML_GRAPH_EDGE_TYPE_OUTPUT, &multiply_to_out });

    DML_GRAPH_DESC graph_desc{};
    graph_desc.InputCount            = 1;
    graph_desc.OutputCount           = 1;
    graph_desc.NodeCount             = static_cast<UINT>(graph_nodes.size());
    graph_desc.Nodes                 = graph_nodes.data();
    graph_desc.InputEdgeCount        = static_cast<UINT>(input_edge_descs.size());
    graph_desc.InputEdges            = input_edge_descs.data();
    graph_desc.OutputEdgeCount       = static_cast<UINT>(output_edge_descs.size());
    graph_desc.OutputEdges           = output_edge_descs.data();
    graph_desc.IntermediateEdgeCount = static_cast<UINT>(intermediate_edge_descs.size());
    graph_desc.IntermediateEdges     = intermediate_edge_descs.data();

    if (FAILED(dml_device1->CompileGraph(&graph_desc, DML_EXECUTION_FLAG_NONE,
                                         IID_PPV_ARGS(result.compiled_op.GetAddressOf())))) {
        DML_PERF_LOG("[QuickGelu] CompileQuickGeluDml: CompileGraph FAILED");
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
// OrtNodeComputeInfo callbacks
// ---------------------------------------------------------------------------

static OrtStatus* ORT_API_CALL QuickGelu_Compute(
    OrtNodeComputeInfo* this_ptr,
    void* compute_state,
    OrtKernelContext* kernel_context) noexcept
{
    auto* state = static_cast<QuickGeluKernelState*>(compute_state);
    if (!state || !state->provider || !state->ort_api) return nullptr;

    const OrtApi& api = *state->ort_api;
    // Get input and read its concrete runtime shape.
    const OrtValue* input_value = nullptr;
    {
        OrtStatus* st = api.KernelContext_GetInput(kernel_context, 0, &input_value);
        if (st || !input_value) { if (st) api.ReleaseStatus(st);
            return api.CreateStatus(ORT_FAIL, "QuickGelu: failed to get input"); }
    }

    OrtTensorTypeAndShapeInfo* shape_info = nullptr;
    api.GetTensorTypeAndShape(const_cast<OrtValue*>(input_value), &shape_info);
    size_t elem_count = 0; api.GetTensorShapeElementCount(shape_info, &elem_count);
    ONNXTensorElementDataType dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    api.GetTensorElementType(shape_info, &dtype);
    size_t rank = 0; api.GetDimensionsCount(shape_info, &rank);
    std::vector<int64_t> dims(rank);
    if (rank > 0) api.GetDimensions(shape_info, dims.data(), rank);
    api.ReleaseTensorTypeAndShapeInfo(shape_info);

    size_t elem_size = (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) ? 2 : 4;
    uint64_t input_size = static_cast<uint64_t>(elem_count * elem_size);

    DML_TENSOR_DATA_TYPE dml_dtype = DML_TENSOR_DATA_TYPE_UNKNOWN;
    if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)   dml_dtype = DML_TENSOR_DATA_TYPE_FLOAT32;
    if (dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) dml_dtype = DML_TENSOR_DATA_TYPE_FLOAT16;
    if (dml_dtype == DML_TENSOR_DATA_TYPE_UNKNOWN) {
        return api.CreateStatus(ORT_FAIL, "QuickGelu: unsupported dtype");
    }

    std::vector<uint32_t> sizes(rank);
    for (size_t i = 0; i < rank; ++i) {
        sizes[i] = static_cast<uint32_t>(dims[i] > 0 ? dims[i] : 1);
    }

    // Lazy-init: compile on first Compute with real shapes.
    if (!state->initialized) {
        std::lock_guard<std::mutex> lock(state->init_mutex);
        if (!state->initialized) {
            state->kernel = CompileQuickGeluDml(
                state->provider, state->alpha, dml_dtype, sizes, input_size);
            if (!state->kernel.IsValid()) {
                return api.CreateStatus(ORT_FAIL, "QuickGelu: DML compilation failed");
            }
            state->compiled_sizes = sizes;
            state->initialized = true;
        }
    }

    // Shape-change: if input shape differs from what was compiled, create a
    // temporary kernel for this call only — then discard it. Mirrors the
    // original ORT DML EP's local kernel recompile on shape change.
    const QuickGeluCompiledKernel* active_kernel = &state->kernel;
    QuickGeluCompiledKernel temp_kernel;
    if (sizes != state->compiled_sizes) {
        temp_kernel = CompileQuickGeluDml(
            state->provider, state->alpha, dml_dtype, sizes, input_size);
        if (!temp_kernel.IsValid()) {
            return api.CreateStatus(ORT_FAIL, "QuickGelu: temporary kernel compilation failed");
        }
        active_kernel = &temp_kernel;
    }

    // Allocate output (same shape as input — elementwise op).
    OrtValue* output_value = nullptr;
    {
        OrtStatus* st = api.KernelContext_GetOutput(
            kernel_context, 0, dims.data(), rank, &output_value);
        if (st || !output_value) { if (st) api.ReleaseStatus(st);
            return api.CreateStatus(ORT_FAIL, "QuickGelu: failed to get output"); }
    }

    // Helper: translate OrtValue GPU allocation to ID3D12Resource*.
    auto get_resource = [&](const OrtValue* value) -> ID3D12Resource* {
        void* raw = nullptr;
        OrtStatus* st = api.GetTensorMutableData(const_cast<OrtValue*>(value), &raw);
        if (st || !raw) { if (st) api.ReleaseStatus(st); return nullptr; }
        return state->provider->DecodeResource(raw);
    };

    ID3D12Resource* input_resource  = get_resource(input_value);
    ID3D12Resource* output_resource = get_resource(output_value);
    if (!input_resource || !output_resource) {
        return api.CreateStatus(ORT_FAIL, "QuickGelu: failed to get D3D12 resources");
    }

    DML_BUFFER_BINDING input_binding  { input_resource,  0, input_size };
    DML_BUFFER_BINDING output_binding { output_resource, 0, input_size };

    DML_BINDING_DESC input_desc  { DML_BINDING_TYPE_BUFFER, &input_binding };
    DML_BINDING_DESC output_desc { DML_BINDING_TYPE_BUFFER, &output_binding };

    const DML_BUFFER_BINDING* persistent = active_kernel->persistent_binding
        ? &*active_kernel->persistent_binding : nullptr;

    HRESULT hr = state->provider->ExecuteOperator(
        active_kernel->compiled_op.Get(),
        persistent,
        gsl::make_span(&input_desc,  1),
        gsl::make_span(&output_desc, 1));

    if (FAILED(hr)) {
        return api.CreateStatus(ORT_FAIL, "QuickGelu: ExecuteOperator failed");
    }

    state->provider->QueueReference(active_kernel->compiled_op.Get());
    return nullptr;
}

static void ORT_API_CALL QuickGelu_ReleaseState(
    OrtNodeComputeInfo* /*this_ptr*/,
    void* compute_state) noexcept
{
    delete static_cast<QuickGeluKernelState*>(compute_state);
}


// ---------------------------------------------------------------------------
// OrtNodeComputeInfo wrapper that carries the kernel state
// ---------------------------------------------------------------------------
struct QuickGeluNodeComputeInfo : OrtNodeComputeInfo {
    QuickGeluKernelState* state = nullptr;  // owned

    QuickGeluNodeComputeInfo() {
        ort_version_supported = ORT_API_VERSION;
        CreateState = [](OrtNodeComputeInfo* self, OrtNodeComputeContext*, void** out) noexcept -> OrtStatus* {
            *out = static_cast<QuickGeluNodeComputeInfo*>(self)->state;
            return nullptr;
        };
        Compute      = QuickGelu_Compute;
        ReleaseState = [](OrtNodeComputeInfo*, void*) noexcept {};  // state owned by info destructor
    }

    ~QuickGeluNodeComputeInfo() {
        delete state;
    }
};

// ---------------------------------------------------------------------------
// Compile — eager if shapes are static, deferred to first Compute if dynamic.
// Mirrors the original ORT DML EP: EAGER when InputTensorShapesDefined(),
// DEFERRED (LAZY_INIT) otherwise.
// ---------------------------------------------------------------------------
OrtNodeComputeInfo* QuickGeluFusionRule::Compile(
    const OrtApi&                    ort_api,
    const OrtGraph*                  fused_subgraph,
    const std::unordered_map<std::string, const OrtValue*>& initializers,
    PluginDmlExecutionProviderImpl*  provider,
    const PatternMatch&              match) const
{
    // Alpha: read from the initializer via the captured name.
    // Default 1.0 when the optional alpha_mul branch wasn't matched.
    float alpha = 1.0f;
    {
        std::string alpha_name = match.ValueName("QuickGelu.alpha");
        if (!alpha_name.empty()) {
            auto it = initializers.find(alpha_name);
            if (it != initializers.end()) {
                fusion_utils::TryReadScalarFloat(ort_api, it->second, alpha);
            }
        }
    }

    // Read static input shape from the fused subgraph's type info.
    size_t num_inputs = 0;
    ort_api.Graph_GetNumInputs(fused_subgraph, &num_inputs);
    if (num_inputs < 1) return nullptr;

    std::vector<const OrtValueInfo*> input_vis(num_inputs, nullptr);
    ort_api.Graph_GetInputs(fused_subgraph, input_vis.data(), num_inputs);
    if (!input_vis[0]) return nullptr;

    const OrtTypeInfo* type_info = nullptr;
    {
        OrtStatus* st = ort_api.GetValueInfoTypeInfo(input_vis[0], &type_info);
        if (st || !type_info) { if (st) ort_api.ReleaseStatus(st); return nullptr; }
    }

    const OrtTensorTypeAndShapeInfo* shape_info = nullptr;
    ort_api.CastTypeInfoToTensorInfo(type_info, &shape_info);
    if (!shape_info) return nullptr;

    ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    ort_api.GetTensorElementType(shape_info, &elem_type);

    size_t rank = 0;
    ort_api.GetDimensionsCount(shape_info, &rank);
    std::vector<int64_t> dims(rank, -1);
    if (rank > 0) ort_api.GetDimensions(shape_info, dims.data(), rank);

    // Validate element type.
    DML_TENSOR_DATA_TYPE dml_dtype = DML_TENSOR_DATA_TYPE_UNKNOWN;
    switch (elem_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   dml_dtype = DML_TENSOR_DATA_TYPE_FLOAT32; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: dml_dtype = DML_TENSOR_DATA_TYPE_FLOAT16; break;
        default: return nullptr;
    }

    // Build kernel state — holds everything needed by Compute.
    auto* kernel_state     = new QuickGeluKernelState();
    kernel_state->alpha    = alpha;
    kernel_state->provider = provider;
    kernel_state->ort_api  = &ort_api;

    // Check for dynamic dims: if any are unknown, defer to first Compute (LAZY_INIT).
    // rank=0 with unknown element count means shape is unavailable from the EP
    // C API (not a true scalar) — defer to compute time when real shapes arrive.
    size_t static_elem_count = 0;
    ort_api.GetTensorShapeElementCount(shape_info, &static_elem_count);
    bool has_dynamic_dims = (rank == 0 && static_elem_count == 0);
    for (size_t i = 0; i < rank; ++i) {
        if (dims[i] <= 0) { has_dynamic_dims = true; break; }
    }

    if (!has_dynamic_dims) {
        // All dims concrete — compile eagerly.
        size_t elem_size = (dml_dtype == DML_TENSOR_DATA_TYPE_FLOAT32) ? 4 : 2;
        size_t total_elems = 1;
        std::vector<uint32_t> sizes(rank);
        for (size_t i = 0; i < rank; ++i) {
            sizes[i] = static_cast<uint32_t>(dims[i]);
            total_elems *= sizes[i];
        }
        uint64_t total_bytes = static_cast<uint64_t>(total_elems * elem_size);

        kernel_state->kernel = CompileQuickGeluDml(
            provider, alpha, dml_dtype, sizes, total_bytes);
        if (!kernel_state->kernel.IsValid()) {
            delete kernel_state;
            return nullptr;
        }
        kernel_state->compiled_sizes = sizes;
        kernel_state->initialized = true;
    }

    auto* info  = new QuickGeluNodeComputeInfo();
    info->state = kernel_state;
    return info;
}

}  // namespace dml_ep
