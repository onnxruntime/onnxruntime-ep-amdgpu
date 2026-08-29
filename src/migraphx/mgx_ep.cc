// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_set>

#include <gsl/span>

#include <migraphx/migraphx.hpp>
#include <migraphx/version.h>

#include "common/constants.h"
#include "common/enumerate.h"
#include "common/env_var.h"
#include "common/murmurhash3.h"
#include "common/ort_graph_to_proto.h"
#include "common/telemetry.h"

#include "hip/stream_support.h"

#include "mgx_dynamic_batch.h"
#include "mgx_ep.h"
#include "mgx_ep_ctx.h"
#include "mgx_hip_graph.h"
#include "mgx_info.h"
#include "mgx_precompile.h"
#include "mgx_program_ops.h"
#include "mgx_utils.h"

namespace mgx_ep {

const char* ExecutionProvider::GetName() const noexcept {
    return ep_name_.c_str();
}

namespace {

// HIP's current device is a per-thread setting.  ORT/Triton can run Compile()
// and the compute functions on pool threads that never hit the constructor's
// hipSetDevice (so they default to device 0), which then loads code objects /
// launches kernels on the wrong GPU on a non-zero-device instance -- surfacing
// as "no kernel image is available" / "invalid resource handle".  This RAII
// guard pins the calling thread to the EP's device for the enclosing scope and
// restores the previous device on exit, so the device is set once per call (and
// only when it actually differs) instead of unconditionally on every branch.
// Mirrors the built-in EP's HipDeviceGuard.
struct HipDeviceGuard {
    int prev_{0};
    explicit HipDeviceGuard(int dev) {
        HIP_CALL_THROW(hipGetDevice(&prev_));
        if (dev != prev_) {
            HIP_CALL_THROW(hipSetDevice(dev));
        }
    }
    ~HipDeviceGuard() {
        (void)hipSetDevice(prev_);  // best-effort restore; never throw from a dtor
    }
    HipDeviceGuard(const HipDeviceGuard&) = delete;
    HipDeviceGuard& operator=(const HipDeviceGuard&) = delete;
};

// TEMPORARY A/B gate (env ORT_MIGRAPHX_LEGACY_COMPUTE_SYNC). When true, Compute
// keeps the legacy unconditional per-fused-node hipStreamSynchronize. When false
// (default), the hot path stays async and correctness comes from stream-ordered
// copies (DataTransfer issues on the compute stream ORT hands us) plus the single
// per-Run SyncStream::OnSessionRunEnd drain -- matching the built-in EP, which
// never drained per-Compute. Cached once; remove this gate (and the legacy branch)
// once the pipelined path is confirmed. See mgx_ep.h env_var::kLegacyComputeSync.
bool LegacyComputeSyncEnabled() {
    static const bool enabled{
        ParseEnvironmentVariableWithDefault<bool>(env_var::kLegacyComputeSync, false)};
    return enabled;
}

// Post-Compute synchronization for a fused subgraph. With a real compute stream
// (the common Triton path), skip the drain and let the stream-ordered D2H fetch
// plus the per-Run OnSessionRunEnd sync serialize the read; only drain when either
// the legacy gate is on or ORT gave us no stream (null == default stream), where a
// later null-stream copy would not order against a non-blocking producer.
[[nodiscard]] Ort::Status FinishComputeStream(hipStream_t hip_stream) {
    if (LegacyComputeSyncEnabled() || hip_stream == nullptr) {
        HIP_RETURN_IF_ERROR(hipStreamSynchronize(hip_stream));
    }
    return STATUS_OK;
}

ONNXTensorElementDataType GetElementType(const Ort::ConstTypeInfo& type_info) {
    switch (type_info.GetONNXType()) {
    case ONNX_TYPE_TENSOR:
    case ONNX_TYPE_SPARSETENSOR:
        return type_info.GetTensorTypeAndShapeInfo().GetElementType();
    case ONNX_TYPE_SEQUENCE:
        return GetElementType(Ort::ConstTypeInfo{type_info.GetSequenceTypeInfo().GetSequenceElementType()});
    case ONNX_TYPE_MAP:
        return GetElementType(Ort::ConstTypeInfo{type_info.GetMapTypeInfo().GetMapValueType()});
    case ONNX_TYPE_OPTIONAL:
        return GetElementType(Ort::ConstTypeInfo{type_info.GetOptionalTypeInfo().GetOptionalElementType()});
    case ONNX_TYPE_OPAQUE:
    case ONNX_TYPE_UNKNOWN:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
    THROW("invalid ONNX type");
}

bool IsTypeSupported(const Ort::ConstValueInfo& value_info) {
    switch (GetElementType(value_info.TypeInfo())) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return true;
    default:
        return false;
    }
}

bool GetMIGraphXType(ONNXTensorElementDataType type, migraphx_shape_datatype_t& mgx_type) {
    mgx_type = migraphx_shape_float_type;
    switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        mgx_type = migraphx_shape_half_type;
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        mgx_type = migraphx_shape_bf16_type;
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        mgx_type = migraphx_shape_float_type;
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        mgx_type = migraphx_shape_double_type;
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
        mgx_type = migraphx_shape_fp8e4m3fnuz_type;
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
        mgx_type = migraphx_shape_fp8e4m3fn_type;
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
        mgx_type = migraphx_shape_fp8e5m2_type;
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
        mgx_type = migraphx_shape_fp8e5m2fnuz_type;
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1:
        mgx_type = migraphx_shape_fp4x2_type;
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        mgx_type = migraphx_shape_int8_type;
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        mgx_type = migraphx_shape_int16_type;
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        mgx_type = migraphx_shape_int32_type;
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        mgx_type = migraphx_shape_int64_type;
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        mgx_type = migraphx_shape_uint8_type;
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        mgx_type = migraphx_shape_uint16_type;
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        mgx_type = migraphx_shape_uint32_type;
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        mgx_type = migraphx_shape_uint64_type;
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        mgx_type = migraphx_shape_bool_type;
        break;
    default:
    /*    LOG(WARNING) << "unsupported data type " << type
                     << ", fallback to CPU implementation"; */
        return false;
    }
    return true;
}

bool IsUnsupportedOpMode(const Ort::ConstGraph& graph, const Ort::ConstNode& node) {
    const auto op_type{node.GetOperatorType()};
    const auto inputs{GetValueInfos(node.GetInputs())};
    if (op_type == "ArgMax" || op_type == "ArgMin") {
        // we do not support select_last_index = 1 for now
        Ort::ConstOpAttr attr;
        const Ort::Status status{node.GetAttributeByName("select_last_index", attr)};
        if (status.IsOK() && attr) {
            if (int64_t value; !attr.GetValue(value).IsOK() || value != 0) {
                return true;
            }
        }
    } else if (op_type == "ConstantOfShape") {
        if (!CanEvalNodeArgument(graph, node, {0})) {
            return true;
        }
    } else if (op_type == "ConvInteger") {
        // only support int8 and uint8 type
        if (inputs.empty()) {
            return true;
        }
        for (const auto& input : inputs) {
            if (!IsInt8Tensor(input) && !IsUint8Tensor(input)) {
                return true;
            }
        }
    } else if (op_type == "Expand") {
        // only supports constant shape input values
        if (!CanEvalNodeArgument(graph, node, {1})) {
            return true;
        }
    } else if (op_type == "MaxPool") {
        // MaxPool "indices" output is not currently supported
        if (GetValueInfos(node.GetOutputs()).size() > 1) {
            return true;
        }
        // ceil_mode and dilations attrs are not supported
        Ort::ConstOpAttr attr;
        Ort::Status status{node.GetAttributeByName("dilations", attr)};
        if (status.IsOK() && attr) {
            std::vector<int64_t> dilations;
            if (!attr.GetValueArray(dilations).IsOK()) {
                return true;
            }
            if (!ranges::all_of(dilations,
                [](int64_t dilation) { return dilation == 1; })) {
                return true;
            }
        }
        // storage order 1 (column major format) is not supported
        status = node.GetAttributeByName("storage_order", attr);
        if (status.IsOK() && attr) {
            if (int64_t value; !attr.GetValue(value).IsOK() || value != 0) {
                return true;
            }
        }
    } else if (op_type == "MatMulInteger") {
        if (inputs.empty()) {
            return true;
        }
        // only support int8 and uint8 type
        for (const auto& input : inputs) {
            if (!IsInt8Tensor(input) && !IsUint8Tensor(input)) {
                return true;
            }
        }
    } else if (op_type == "OneHot") {
        if (!CanEvalNodeArgument(graph, node, {1})) {
            return true;
        }
    } else if (op_type == "Pad") {
        // if pad size is not constant, migraphx cannot support
        if (inputs.size() >= 2) {
            if (!CanEvalNodeArgument(graph, node, {1})) {
                return true;
            }
        }
        static const std::set<std::string_view> allowed_modes{
            "constant", "reflect", "edge"
        };
        Ort::ConstOpAttr attr;
        Ort::Status status{node.GetAttributeByName("mode", attr)};
        if (status.IsOK() && attr) {
            if (std::string mode; !attr.GetValue(mode).IsOK() || allowed_modes.count(mode) == 0) {
                return true;
            }
        }
    } else if (op_type == "Range") {
        std::vector<std::size_t> v(inputs.size());
        std::iota(v.begin(), v.end(), 0);
        if (!CanEvalNodeArgument(graph, node, v)) {
            return true;
        }
    } else if (op_type == "Reshape") {
        if (inputs.size() == 2) {
            if (CanEvalNodeArgument(graph, node, {1})) {
                return false;
            }
        }
    } else if (op_type == "Resize" || op_type == "Upsample") {
        Ort::ConstOpAttr attr;
        Ort::Status status{node.GetAttributeByName("coordinate_transformation_mode", attr)};
        if (status.IsOK() && attr) {
            if (std::string value; !attr.GetValue(value).IsOK() || value == "tf_crop_and_resize") {
                return true;
            }
        }
        status = node.GetAttributeByName("mode", attr);
        if (status.IsOK() && attr) {
            if (std::string value; !attr.GetValue(value).IsOK() || value == "cubic") {
                return true;
            }
        }
    } else if (op_type == "ReduceSum") {
        if (inputs.size() == 2) {
            return !CanEvalNodeArgument(graph, node, {1});
        }
    } else if (op_type == "Slice") {
        std::vector<std::size_t> v(inputs.size());
        std::iota(v.begin(), v.end(), 0);
        v.erase(v.begin());
        if (!CanEvalNodeArgument(graph, node, v)) {
            return true;
        }
        Ort::ConstOpAttr s_attr;
        Ort::Status s_status{node.GetAttributeByName("starts", s_attr)};
        Ort::ConstOpAttr e_attr;
        Ort::Status e_status{node.GetAttributeByName("ends", e_attr)};
        if (s_status.IsOK() && s_attr &&e_status.IsOK() && e_attr) {
            std::vector<int64_t> starts;
            if (!s_attr.GetValueArray(starts).IsOK()) {
                return true;
            }
            std::vector<int64_t> ends;
            if (!e_attr.GetValueArray(ends).IsOK()) {
                return true;
            }
            for (const auto& [s, e]: zip(starts, ends)) {
                if (s > e) { return true; }
            }
        }
    } else if (op_type == "Split") {
        // cannot process input dim of 0 size
        for (const auto& input : inputs) {
            const auto shape{input.TypeInfo().GetTensorTypeAndShapeInfo().GetShape()};
            if (!shape.empty() && shape == std::vector<int64_t>{0}) {
                return true;
            }
        }
        if (inputs.size() == 2) {
            return !CanEvalNodeArgument(graph, node, {1});
        }
    } else if (op_type == "Tile") {
        if (!CanEvalNodeArgument(graph, node, {1})) {
            return true;
        }
    } else if (op_type == "TopK") {
        if (!CanEvalNodeArgument(graph, node, {1})) {
            return true;
        }
    } else if (op_type == "Unsqueeze" || op_type == "Squeeze") {
        if (inputs.size() == 2) {
            return !CanEvalNodeArgument(graph, node, {1});
        }
    }
    // Op doesn't fall into known any of unsupported modes.
    return false;
}

std::unordered_set<size_t> CollectCpuControlFlowBoundaryNodes(
    const std::vector<Ort::ConstNode>& sorted_nodes)
{
    // If/Loop/Scan on CPU. Greater/Cast too so the bool condition is CPU-resident
    std::unordered_set<size_t> boundary_nodes;
    for (const auto& node : sorted_nodes) {
        const auto op_type{node.GetOperatorType()};
        if (op_type == "If" || op_type == "Loop" || op_type == "Scan" ||
            op_type == "Greater" || op_type == "Cast") {
            boundary_nodes.insert(node.GetId());
        }
    }
    return boundary_nodes;
}

std::vector<std::vector<Ort::ConstNode>> GetPartitionedSubgraphs(const std::vector<Ort::ConstNode>& nodes,
    const std::vector<Ort::ConstNode>& unsupported_nodes)
{
    std::vector<std::vector<Ort::ConstNode>> subgraphs;
    auto begin{nodes.begin()};
    for (const auto& unsupported_node : unsupported_nodes) {
        auto it{std::find_if(begin, nodes.end(),
            [&unsupported_node](const Ort::ConstNode& node) {
                return node.GetId() == unsupported_node.GetId();
            })};
        std::vector<Ort::ConstNode> subgraph{begin, it};
        if (!subgraph.empty()) {
            subgraphs.emplace_back(subgraph);
        }
        begin = ++it;
    }
    std::vector<Ort::ConstNode> subgraph{begin, nodes.end()};
    if (!subgraph.empty()) {
        subgraphs.emplace_back(subgraph);
    }
    return subgraphs;
}

Map<size_t> BuildInputNameIndices(const Ort::ConstGraph& graph, const Ort::ConstNode& fused_node) {
    Map<size_t> input_name_indices;
    size_t input_index{};
    const auto add_input = [&](const Ort::ConstValueInfo& input) {
        if (input == nullptr || input.IsConstantInitializer()) {
            return;
        }
        if (input_name_indices.count(input.GetName()) != 0) {
            return;
        }
        input_name_indices.emplace(input.GetName(), input_index++);
    };
    // KernelContext indices: fused boundary inputs, then graph inputs (branch feeds).
    for (const auto& input : GetValueInfos(fused_node.GetInputs())) {
        add_input(input);
    }
    for (const auto& input : GetValueInfos(graph.GetInputs())) {
        add_input(input);
    }
    for (const auto& input : GetValueInfos(fused_node.GetImplicitInputs())) {
        add_input(input);
    }
    return input_name_indices;
}

void* GetGpuInputData(ComputeState& cs, const Ort::KernelContext& ctx, const std::string& name,
    size_t index, const migraphx::shape& prog_shape, hipStream_t stream)
{
    const auto input_tensor{ctx.GetInput(index)};
    const auto mem{input_tensor.GetTensorMemoryInfo()};
    if (mem.GetDeviceType() != OrtMemoryInfoDeviceType_CPU) {
        return const_cast<void*>(input_tensor.GetTensorRawData());
    }

    const std::size_t bytes{prog_shape.bytes()};
    auto& buf{cs.cpu_input_upload_bufs[name]};
    if (buf.data == nullptr || buf.size_bytes < bytes) {
        if (buf.data != nullptr) {
            (void)hipFree(buf.data);
        }
        void* ptr{nullptr};
        HIP_CALL_THROW(hipMalloc(&ptr, bytes));
        buf.data = ptr;
        buf.size_bytes = bytes;
        buf.shape = prog_shape;
    }

    const void* src{input_tensor.GetTensorRawData()};
    if (bytes > 0) {
        if (stream != nullptr) {
            HIP_CALL_THROW(hipMemcpyAsync(buf.data, src, bytes, hipMemcpyHostToDevice, stream));
        } else {
            HIP_CALL_THROW(hipMemcpy(buf.data, src, bytes, hipMemcpyHostToDevice));
        }
    }
    return buf.data;
}

// Populate the direct-bind binding cache for the current compiled shape.  Walks
// the program parameters once, recording each input's (name, ORT index, shape),
// each output's (name, ORT index, shape), and whether a "scratch" parameter is
// present (plus its shape) so the hot path can rebind and allocate scratch without
// per-call name lookups or a scratch scan.  Any parameter that is not an input,
// output, or "scratch" is simply skipped, as in the built-in EP -- the direct path
// is taken whenever it is configured (no per-shape eligibility gate).  Inputs are
// recorded straight from the program's parameter shapes (name, ORT index,
// mgx_shape) with no ORT tensor introspection: under Triton's GPU-first io-binding
// they are device-resident and their dtype matches the compiled program by
// construction, mirroring the built-in EP's populate-once design.
void PopulateDirectCaches(DirectBindCache& dbc,
    const migraphx::program_parameter_shapes& param_shapes,
    const Map<std::size_t>& input_name_indices)
{
    dbc.inputs.clear();
    dbc.outputs.clear();
    dbc.inputs.reserve(input_name_indices.size());
    dbc.has_scratch = false;

    for (std::string name : param_shapes.names()) {
        if (const auto it{input_name_indices.find(name)}; it != input_name_indices.end()) {
            const auto prog_shape{param_shapes[name.c_str()]};
            dbc.inputs.push_back(CachedDirectInput{std::move(name), it->second, prog_shape});
        } else if (std::string_view{name} == "scratch") {
            // Bound separately each call (pointer-stable EP-owned buffer).  Record
            // its presence + shape once so the hot path allocates/zeros scratch
            // without re-scanning all parameter names for it every call (#4).
            dbc.has_scratch = true;
            dbc.scratch_shape = param_shapes[name.c_str()];
        } else if (const auto oi{ComputeOutputIndex(name)}; oi != -1) {
            const auto prog_shape{param_shapes[name.c_str()]};
            const auto lens{prog_shape.lengths()};
            std::vector<std::int64_t> ort_shape{lens.begin(), lens.end()};
            dbc.outputs.push_back(CachedDirectOutput{
                std::move(name), static_cast<std::size_t>(oi), prog_shape, std::move(ort_shape)});
        }
    }
}

}  // namespace

#define PARSE_ENV_VAR(__name__, __value__)                                               \
    const auto __value__##env{ParseEnvironmentVariable<decltype(__value__)>(__name__)};  \
    if (__value__##env.has_value()) {                                                    \
        __value__ = __value__##env.value();                                              \
    }

ExecutionProvider::ExecutionProvider(const ProviderFactory& factory, std::string_view ep_name, Ort::ConstSessionOptions session_options, const Ort::Logger& logger)
    : OrtEp{NegotiatedOrtApiVersion()}, ApiPtrs{factory.ort_api, factory.ep_api, factory.model_editor_api}, factory_{factory}, logger_{logger}, ep_name_{ep_name}
{
    OrtEp::GetName = [](const OrtEp* this_) noexcept {
        API_CALL_T(const ExecutionProvider, this_, GetName, "invalid object pointer");
    };
    OrtEp::GetCapability = [](OrtEp* this_, const OrtGraph* graph,
                              OrtEpGraphSupportInfo* graph_support_info) noexcept {
        API_CALL_S(ExecutionProvider, this_, GetCapability, Ort::ConstGraph{graph}, graph_support_info);
    };
    OrtEp::Compile = [](OrtEp* this_, const OrtGraph** graphs, const OrtNode** fused_nodes,
                        size_t count, OrtNodeComputeInfo** node_compute_infos, OrtNode** ep_context_nodes) noexcept {
        API_CALL_S(ExecutionProvider, this_, Compile, {graphs, graphs + count},
                   {fused_nodes, fused_nodes + count}, {node_compute_infos, count}, {ep_context_nodes, count});
    };
    OrtEp::ReleaseNodeComputeInfos = [](OrtEp* this_, OrtNodeComputeInfo** node_compute_infos,
                                        size_t num_node_compute_info) noexcept {
        API_CALL_V(ExecutionProvider, this_, ReleaseNodeComputeInfos, {node_compute_infos, num_node_compute_info});
    };
    // OrtEp::SetDynamicOptions = [](OrtEp* this_, const char* const* option_keys, const char* const* option_values,
    //         size_t num_options) noexcept {
    //     API_CALL_S(ExecutionProvider, this_, SetDynamicOptions, option_keys, option_values, num_options);
    // };
    // OrtEp::OnRunStart = [](OrtEp* this_, const OrtRunOptions* run_options) noexcept {
    //     API_CALL_S(ExecutionProvider, this_, OnRunStart, run_options);
    // };
    OrtEp::OnRunEnd = [](OrtEp* this_, const OrtRunOptions* run_options, bool sync_stream) noexcept {
        API_CALL_S(ExecutionProvider, this_, OnRunEnd, run_options, sync_stream);
    };
    OrtEp::CreateSyncStreamForDevice = [](OrtEp* this_, const OrtMemoryDevice* memory_device,
                                          OrtSyncStreamImpl** stream) noexcept {
        API_CALL_S(ExecutionProvider, this_, CreateSyncStreamForDevice, memory_device, stream);
    };
    // OrtEp::GetCompiledModelCompatibilityInfo = [](OrtEp* this_, const OrtGraph* graph) noexcept {
    //     API_CALL_T(ExecutionProvider, this_, GetCompiledModelCompatibilityInfo, "invalid object pointer", graph);
    // };
    OrtEp::GetKernelRegistry = [](OrtEp* this_, const OrtKernelRegistry** kernel_registry) noexcept {
        API_CALL_S(ExecutionProvider, this_, GetKernelRegistry, kernel_registry);
    };

    std::string lowercase{ep_name_};
    std::transform(lowercase.begin(), lowercase.end(), lowercase.begin(), ::tolower);

    OrtKeyValuePairs* ort_key_value_pairs;
    THROW_IF_ERROR(ort_api.GetSessionOptionsConfigEntries(session_options, &ort_key_value_pairs));

    const Ort::KeyValuePairs key_value_pairs{ort_key_value_pairs};
    const std::string ep_prefix{"ep." + lowercase + "."};

    ProviderOptions provider_options;
    for (const auto& [key, value] : key_value_pairs.GetKeyValuePairs()) {
        if (key.rfind(ep_prefix, 0) == 0) {
            provider_options.emplace(key.substr(ep_prefix.length()), value);
        } else if (key.rfind("ep.directml.", 0) == 0) {
            // Keys namespaced to the DirectML sibling backend (e.g. ep.directml.enable_host_accessible)
            // are not ours. The umbrella forwards every ep.* entry to whichever backend it selects, so
            // a DirectML-targeted key can reach the migraphx backend when the umbrella routes to migraphx
            // (e.g. OGA's trivial init session, backend=Auto). Skip it — ProviderInfo would otherwise
            // reject it as an unknown provider option and fail EP creation.
            continue;
        } else if (key.rfind("ep.", 0) == 0) {
            provider_options.emplace(key, value);
        }
    }

    const ProviderInfo info{provider_options};

    device_id_ = info.device_id;
    enable_fp16_ = info.enable_fp16;
    enable_bf16_ = info.enable_bf16;
    enable_fp8_ = info.enable_fp8;
    enable_int8_ = info.enable_int8;
    exhaustive_tune_ = info.exhaustive_tune;
    mlss_use_specific_ops_ = info.mlss_use_specific_ops;
    model_arch_ = info.model_arch;
    cache_dir_ = info.cache_dir;
    int8_use_native_calibration_table_ = info.int8_use_native_calibration_table;
    int8_calibration_table_name_ = info.int8_calibration_table_name;
    dump_subgraphs_ = info.dump_subgraphs;
    compute_mode_ = info.compute_mode;
    disable_compiled_model_caching_ = info.disable_caching;
    force_recompile_ = info.force_recompile;
    context_embed_mode_ = info.context_embed_mode;
    context_enable_ = info.context_enable;
    external_initializers_file_name_ = info.external_initializers_file_name;
    context_file_path_ = info.context_file_path;
    context_node_name_prefix_ = info.context_node_name_prefix;
    hip_graph_enable_ = info.hip_graph_enable;
    max_dynamic_batch_ = info.max_dynamic_batch;
    compile_batches_ = info.compile_batches;
    precompile_at_load_ = info.precompile_at_load;
    coalesce_io_enable_ = info.coalesce_io;
    cpu_control_flow_enable_ = info.cpu_control_flow;
    static_pad_seq_ = info.static_pad_seq;
    static_pad_seq_len_ = info.static_pad_seq_len;
    static_pad_inputs_ = info.static_pad_inputs;
    static_pad_outputs_ = info.static_pad_outputs;
    has_user_compute_stream_ = info.has_user_compute_stream;
    user_compute_stream_ = info.user_compute_stream;

    HIP_CALL_THROW(hipSetDevice(device_id_));
    HIP_CALL_THROW(hipGetDeviceProperties(&device_prop_, device_id_));

    compute_capability_ = device_prop_.gcnArchName;

    PARSE_ENV_VAR(env_var::kFP16Enable, enable_fp16_);
    PARSE_ENV_VAR(env_var::kBF16Enable, enable_bf16_);
    PARSE_ENV_VAR(env_var::kFP8Enable, enable_fp8_);
    PARSE_ENV_VAR(env_var::kINT8Enable, enable_int8_);
    PARSE_ENV_VAR(env_var::kDisableCompiledModelCaching, disable_compiled_model_caching_);
    PARSE_ENV_VAR(env_var::kForceRecompile, force_recompile_);
    PARSE_ENV_VAR(env_var::kINT8CalibrationTableName, int8_calibration_table_name_);
    PARSE_ENV_VAR(env_var::kINT8UseNativeCalibrationTable, int8_use_native_calibration_table_);
    PARSE_ENV_VAR(env_var::kCacheDir, cache_dir_);
    PARSE_ENV_VAR(env_var::kDumpSubgraphs, dump_subgraphs_);
    PARSE_ENV_VAR(env_var::kDumpEpContextModel, context_enable_);
    PARSE_ENV_VAR(env_var::kExhaustiveTune, exhaustive_tune_);
    PARSE_ENV_VAR(env_var::kHipGraphEnable, hip_graph_enable_);
    PARSE_ENV_VAR(env_var::kMaxDynamicBatch, max_dynamic_batch_);
    PARSE_ENV_VAR(env_var::kCompileBatches, compile_batches_);
    PARSE_ENV_VAR(env_var::kPrecompileAtLoad, precompile_at_load_);
    PARSE_ENV_VAR(env_var::kCoalesceIO, coalesce_io_enable_);
    PARSE_ENV_VAR(env_var::kStaticPadSeq, static_pad_seq_);
    // Not PARSE_ENV_VAR: we need to know whether the value was given, not just what it is.
    if (const auto pad_len_env{ParseEnvironmentVariable<std::size_t>(env_var::kStaticPadSeqLen)};
        pad_len_env.has_value()) {
        static_pad_seq_len_ = pad_len_env.value();
        static_pad_seq_len_from_env_ = true;
    }
    PARSE_ENV_VAR(env_var::kStaticPadInputs, static_pad_inputs_);
    PARSE_ENV_VAR(env_var::kStaticPadOutputs, static_pad_outputs_);
    PARSE_ENV_VAR(env_var::kMlssUseSpecificOps, mlss_use_specific_ops_);
    PARSE_ENV_VAR(env_var::kCpuControlFlow, cpu_control_flow_enable_);
    PARSE_ENV_VAR(env_var::kModelArch, model_arch_);

    // Per-architecture ops to force onto AMDMLSS.
    // Add a row here to enable specific ops on additional architectures.
    struct arch_mlss_ops {
        std::string_view arch;
        std::string_view ops;  // comma-separated op names
    };
    static constexpr std::array<arch_mlss_ops, 2> kArchMlssOps{{
        {"gfx1200", "conv"},
        {"gfx1201", "conv"},
    }};

    for (const auto& [arch, ops] : kArchMlssOps) {
        if (compute_capability_.rfind(arch, 0) == 0) {
            if (!mlss_use_specific_ops_.empty()) {
                mlss_use_specific_ops_ += ",";
            }
            mlss_use_specific_ops_ += ops;
        }
    }

    auto compute_mode{platform::GetEnvironmentVar(env_var::kComputeMode)};
    if (!compute_mode.empty()) {
        std::transform(compute_mode.begin(), compute_mode.end(), compute_mode.begin(), ::tolower);
        if (compute_mode == "eager" || compute_mode == "0") {
            compute_mode_ = ComputeMode::Eager;
        } else if (compute_mode == "balanced" || compute_mode == "50") {
            compute_mode_ = ComputeMode::Balanced;
        } else if (compute_mode == "maximum" || compute_mode == "100") {
            compute_mode_ = ComputeMode::Maximum;
        } else {
            /* TODO: log invalid value for the compute mode - do not change it. */
        }
    }

    if (cpu_control_flow_enable_ && hip_graph_enable_) {
        hip_graph_enable_ = false;
    }

    if (enable_fp16_ && enable_bf16_) {
        enable_fp16_ = enable_bf16_ = false;
        /* TODO: log fp16 and bf16 are mutually exclusive - ignore both flags. */
    }

    if (enable_fp8_ && enable_int8_) {
        enable_fp8_ = enable_int8_ = false;
        /* TODO: log fp8 and int8 are mutually exclusive - ignore both flags. */
    }

    // hipGraph capture requires single-stream MIGraphX execution and a capturable
    // (non-default, non-tracing) stream.  Disable it when the environment forces
    // configurations that are incompatible with capture.
    if (hip_graph_enable_) {
        const auto nstreams{ParseEnvironmentVariableWithDefault<int>("MIGRAPHX_NSTREAMS", 1)};
        const auto trace_eval{ParseEnvironmentVariableWithDefault<int>("MIGRAPHX_TRACE_EVAL", 0)};
        const auto null_stream{ParseEnvironmentVariableWithDefault<int>("MIGRAPHX_ENABLE_NULL_STREAM", 0)};
        if (nstreams > 1 || trace_eval != 0 || null_stream != 0) {
            // MIGRAPHX_NSTREAMS>1: multi-stream execution cannot be captured.
            // MIGRAPHX_TRACE_EVAL: inserts per-instruction hipStreamSynchronize.
            // MIGRAPHX_ENABLE_NULL_STREAM: default stream is illegal during capture.
            hip_graph_enable_ = false;
            /* TODO: log that hipGraph was disabled due to incompatible MIGraphX env. */
        }
    }

    // Coalesced input H2D rides on the staging path, which only runs when hipGraph
    // (or dynamic batching) is active.  Without one of those the flag has no effect.
    if (coalesce_io_enable_ && !hip_graph_enable_ && max_dynamic_batch_ == 0) {
        /* TODO: log warning that ORT_MIGRAPHX_COALESCE_IO requires hipGraph or
           dynamic batching (the staging path) to take effect. */
    }

    // If compile_batches is set, derive max_dynamic_batch from the spec's max value.
    if (!compile_batches_.empty()) {
        if (const auto explicit_sizes{ParseCompileBatches(compile_batches_)}; !explicit_sizes.empty()) {
            if (const auto derived_max{explicit_sizes.back()}; max_dynamic_batch_ < derived_max) {
                max_dynamic_batch_ = derived_max;
            }
        }
    }

    int8_calibration_cache_available_ =
        (enable_int8_ || enable_fp8_) && !int8_calibration_table_name_.empty();

    if (int8_calibration_cache_available_)
        try {
            if (const auto int8_calibration_cache_path{cache_dir_ / int8_calibration_table_name_};
                !ReadDynamicRange(int8_calibration_cache_path, int8_use_native_calibration_table_, dynamic_ranges_)) {
                throw std::runtime_error{"failed to read INT8 calibration table"};
            }
        }
        catch (const std::exception& e) {
            enable_int8_ = false;
            /* TODO: log error reading calibration table */
        }

    /* TODO: print configured options for the session */

    // Register only after construction succeeds so the factory count is balanced
    // by the destructor.
    if (cpu_control_flow_enable_) {
        factory_.EnableCpuControlFlow();
    }
}

ExecutionProvider::~ExecutionProvider() {
    // Best-effort teardown of EP-owned device memory and captured graphs.  Errors
    // are ignored because the HIP context may already be torn down at process exit.
    (void)hipSetDevice(device_id_);
    for (auto& [name, cs] : compute_states_) {
        for (auto& [hash, entry] : cs.hip_graph_cache) {
            if (entry.exec != nullptr) {
                (void)hipGraphExecDestroy(entry.exec);
                entry.exec = nullptr;
            }
            if (entry.graph != nullptr) {
                (void)hipGraphDestroy(entry.graph);
                entry.graph = nullptr;
            }
            entry.captured = false;
        }
        // Staging inputs/outputs and the coalesce arena come from hipMallocAsync, so
        // they are released with hipFreeAsync.  Best-effort teardown: the free is
        // queued on the default stream and reclaimed with the context at exit.
        for (auto& [param_name, buf] : cs.staging_inputs) {
            // Arena sub-views are not independent allocations; the arena is freed below.
            if (buf.data != nullptr && !buf.is_arena_view) {
                (void)hipFreeAsync(buf.data, nullptr);
            }
            buf.data = nullptr;
        }
        if (cs.in_arena_dev != nullptr) {
            (void)hipFreeAsync(cs.in_arena_dev, nullptr);
            cs.in_arena_dev = nullptr;
        }
        if (cs.in_staging_host != nullptr) {
            (void)hipHostFree(cs.in_staging_host);
            cs.in_staging_host = nullptr;
        }
        for (auto& [param_name, buf] : cs.staging_outputs) {
            if (buf.data != nullptr) {
                (void)hipFreeAsync(buf.data, nullptr);
                buf.data = nullptr;
            }
        }
        for (auto& [hash, buf] : cs.scratch_bufs) {
            if (buf.data != nullptr) {
                (void)hipFree(buf.data);
                buf.data = nullptr;
            }
        }
        for (auto& [param_name, buf] : cs.cpu_input_upload_bufs) {
            if (buf.data != nullptr) {
                (void)hipFree(buf.data);
                buf.data = nullptr;
            }
        }
    }

    if (cpu_control_flow_enable_) {
        factory_.DisableCpuControlFlow();
    }
}

Ort::Status ExecutionProvider::GetCapability(const Ort::ConstGraph& graph,
                                             OrtEpGraphSupportInfo* graph_support_info) const noexcept
try {
    if (graph.GetNodes().empty()) {
        return STATUS_OK;
    }

    // With cpu_control_flow, compile If/Loop/Scan branch bodies; ops in
    // CollectCpuControlFlowBoundaryNodes stay on CPU.
    const auto parent_node = graph.GetParentNode();
    const bool in_control_flow_subgraph = parent_node &&
        (parent_node.GetOperatorType() == "If" ||
         parent_node.GetOperatorType() == "Loop" ||
         parent_node.GetOperatorType() == "Scan");

    if (!cpu_control_flow_enable_ && in_control_flow_subgraph) {
        return STATUS_OK;
    }

    // Check if this graph contains EPContext nodes intended for this EP.
    if (EpContextNodeReader::GraphHasContextNode(graph)) {
        std::vector<Ort::ConstNode> ep_context_nodes;
        for (const auto& node : graph.GetNodes()) {
            if (node.GetOperatorType() != "EPContext") {
                continue;
            }
            Ort::ConstOpAttr source_attr;
            Ort::Status status{node.GetAttributeByName("source", source_attr)};
            if (status.IsOK() && source_attr) {
                std::string source;
                if (source_attr.GetValue(source).IsOK() && source != kEpContextSource) {
                    continue;
                }
            }
            ep_context_nodes.push_back(node);
        }
        if (!ep_context_nodes.empty()) {
            // EPContext nodes must be fused (not added as single nodes) so that
            // ORT routes them through Compile() instead of looking for a kernel
            OrtNodeFusionOptions node_fusion_options{NegotiatedOrtApiVersion(), true};
            RETURN_IF_STATUS(ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info,
                reinterpret_cast<const OrtNode* const*>(ep_context_nodes.data()),
                ep_context_nodes.size(), &node_fusion_options));
            return STATUS_OK;
        }
    }

    static const auto supported_op_types{
        []() -> std::set<std::string> {
            const auto v{migraphx::get_onnx_operators()}; return {v.begin(), v.end()};
        }()
    };
    std::vector<Ort::ConstNode> supported_nodes, unsupported_nodes;

    const auto sorted_nodes{GetKahnsVariantTopologicalSortedNodes(graph)};
    const std::unordered_set<size_t> cpu_boundary_nodes{
        cpu_control_flow_enable_ && !in_control_flow_subgraph
            ? CollectCpuControlFlowBoundaryNodes(sorted_nodes)
            : std::unordered_set<size_t>{}};

    for (const auto& node : sorted_nodes) {
        if (cpu_control_flow_enable_ && !in_control_flow_subgraph &&
            cpu_boundary_nodes.count(node.GetId()) != 0) {
            unsupported_nodes.push_back(node);
            continue;
        }

        bool are_types_supported{true};
        for (const auto& output : GetValueInfos(node.GetOutputs())) {
            are_types_supported &= IsTypeSupported(output);
        }
        for (const auto& input : GetValueInfos(node.GetInputs())) {
            are_types_supported &= IsTypeSupported(input);
        }

        if (!are_types_supported || supported_op_types.count(node.GetOperatorType()) == 0 ||
            (node.GetDomain() == kOnnxDomain && IsUnsupportedOpMode(graph, node)))
        {
            const auto op_type{node.GetOperatorType()};
            unsupported_nodes.push_back(node);
            continue;
        }
        supported_nodes.emplace_back(node);
    }
    if (unsupported_nodes.empty()) {
        OrtNodeFusionOptions node_fusion_options{NegotiatedOrtApiVersion(), true};
        RETURN_IF_STATUS(ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info,
            reinterpret_cast<const OrtNode* const*>(supported_nodes.data()), supported_nodes.size(),
            &node_fusion_options));
    } else {
        const auto subgraphs{GetPartitionedSubgraphs(sorted_nodes, unsupported_nodes)};
        /* TODO: log unsupported nodes */
        for (const auto& subgraph : subgraphs) {
            OrtNodeFusionOptions node_fusion_options{NegotiatedOrtApiVersion(), true};
            RETURN_IF_STATUS(ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info,
                reinterpret_cast<const OrtNode* const*>(subgraph.data()), subgraph.size(),
                &node_fusion_options));
        }
    }
    return STATUS_OK;
} catch (const Ort::Exception& e) {
    return Ort::Status{e};
} catch (const std::exception& e) {
    return Ort::Status{e.what(), ORT_EP_FAIL};
}

namespace {

constexpr std::uint64_t MIGraphX_Version =
    (MIGRAPHX_VERSION_MAJOR << 16) | (MIGRAPHX_VERSION_MINOR << 8) | MIGRAPHX_VERSION_PATCH;

bool get_input_output_names(const Ort::Graph& graph, std::vector<std::string>& input_names, std::vector<std::string>& output_names) {
    output_names.clear();
    ranges::for_each(graph.GetOutputs(),
        [&](const auto& output) {
            if (output != nullptr) {
                if (const auto name{output.GetName()}; !name.empty()) {
                    output_names.emplace_back(name);
                }
            }
        });
    const auto inputs{graph.GetInputs()};
    input_names.clear();
    if (inputs.empty()) {
        return false;
    }
    ranges::for_each(inputs,
        [&](const auto& input) {
            if (input != nullptr) {
                input_names.emplace_back(input.GetName());
            }
        });
    return ranges::all_of(inputs,
        [](const auto& input) {
            if (input == nullptr) {
                return true;
            }
            const auto shape{input.TypeInfo().GetTensorTypeAndShapeInfo().GetShape()};
            if (shape.empty()) {
                return false;
            }
            return ranges::all_of(shape,
                [](const auto& dim) {
                    return dim != -1;
                });
        });
}

}  // namespace

void ExecutionProvider::CollectTelemetry(telemetry::BackendData& out) const noexcept {
    out = backend_telemetry_;
}

Ort::Status ExecutionProvider::CreateNodeComputeInfoFromGraph(const Ort::ConstGraph& graph,
    const Ort::ConstNode& fused_node, const Map<size_t>& input_name_indices, const Map<size_t>& output_name_indices,
    const std::string& mxr_prefix, OrtNodeComputeInfo*& node_compute_info, OrtNode*& ep_context_node)
{
    bool loaded_from_cache = false;
    const auto sorted_nodes{GetKahnsVariantTopologicalSortedNodes(graph)};
    Ort::Graph sorted_graph{graph.GetGraphView(sorted_nodes)};
    ONNX_NAMESPACE::ModelProto model_proto{};
    RETURN_IF_ERROR(GraphToProto(sorted_graph, model_proto));
    std::string onnx_string;
    if (!model_proto.SerializeToString(&onnx_string) || onnx_string.empty()) {
        return Ort::Status{"Serializing a model proto to string failed!", ORT_EP_FAIL};
    }

    auto subgraph_name{fused_node.GetName()};
    if (dump_subgraphs_) {
        std::ofstream ofs{subgraph_name + ".onnx", std::ios::binary | std::ios::trunc};
        std::ignore = model_proto.SerializeToOstream(&ofs);
    }

    std::vector<std::string> input_names, output_names;
    const auto has_input_shape{get_input_output_names(sorted_graph, input_names, output_names)};
    fs::path model_path{graph.GetModelPath()};

    // If context_enable is set but cache_dir isn't, default to context_file_path's or the
    // model's directory so the EPContext node references an MXR file that actually gets written.
    fs::path effective_cache_dir{cache_dir_};
    if (context_enable_ && effective_cache_dir.empty()) {
        if (!context_file_path_.empty()) {
            effective_cache_dir = context_file_path_.parent_path();
        } else if (!model_path.empty()) {
            effective_cache_dir = model_path.parent_path();
        }
    }

    RETURN_IF(context_enable_ && !has_input_shape,
        "ep.context_enable is set but the graph has dynamic input shapes at compile time; "
        "EPContext currently supports one statically compiled MXR per partition and cannot be "
        "used with dynamic batching. Provide static input shapes before session creation. To use "
        "dynamic batching instead, disable ep.context_enable and configure "
        "ORT_MIGRAPHX_MAX_DYNAMIC_BATCH and, optionally, ORT_MIGRAPHX_COMPILE_BATCHES.");

    migraphx::program program;
    migraphx::onnx_options onnx_options;

    const PrecompilePlan pre_plan = (!static_pad_seq_ && !context_enable_) ?
        BuildPrecompilePlan(graph, fused_node, input_name_indices, max_dynamic_batch_, compile_batches_) :
        PrecompilePlan{false, false, {}, {}};
    const auto& [pre_eligible, pre_bucketed, pre_batch_sizes, pre_shapes_by_name] = pre_plan;
    const bool use_plan_cache{pre_eligible && !effective_cache_dir.empty()};

    std::string input_shapes_hash_hex;
    if (has_input_shape) {
        hash::Value input_shapes_hash{};
        for (const auto& input : GetValueInfos(sorted_graph.GetInputs())) {
            const auto shape{input.TypeInfo().GetTensorTypeAndShapeInfo().GetShape()};
            hash::Hash(input_shapes_hash, shape);
        }
        input_shapes_hash_hex = hash::ToHex(input_shapes_hash);

        if (!use_plan_cache) {
        fs::path mxr_path;
        if (!effective_cache_dir.empty()) {
            mxr_path = effective_cache_dir / (mxr_prefix + input_shapes_hash_hex + ".mxr");
        }
        loaded_from_cache = !force_recompile_ && load_compiled_program(program, mxr_path);
        backend_telemetry_.loaded_from_cache = loaded_from_cache;
        if (!loaded_from_cache) {
            const auto external_data_dir{external_data_dir_.empty() ?
                model_path.parent_path() : external_data_dir_};
            onnx_options.set_external_data_path(external_data_dir.string());
            program = migraphx::parse_onnx_buffer(onnx_string, onnx_options);
            migraphx::program_parameters params;
            calibrate_and_quantize(program, t_, params, enable_fp16_, enable_bf16_, enable_int8_,
                enable_fp8_, int8_calibration_cache_available_, dynamic_ranges_);
            compile_program(program, t_, exhaustive_tune_, mlss_use_specific_ops_);
            // context_enable needs this file on disk even if caching is otherwise disabled.
            if (!disable_compiled_model_caching_ || context_enable_) {
                save_compiled_program(program, mxr_path);
            }
        }
        const auto output_shapes{program.get_output_shapes()};
        for (const auto& [n, s] : zip(output_names, output_shapes)) {
            onnx_options.set_input_parameter_shape(n, s.lengths());
        }
        }
    }

    if (context_enable_) {
        // input_shapes_hash_hex is non-empty here: the RETURN_IF above requires has_input_shape.
        const fs::path ep_context_mxr_path{mxr_prefix + input_shapes_hash_hex + ".mxr"};

        EpContextNodeHelper ep_context_helper{*this, sorted_graph, fused_node};
        RETURN_IF_ERROR(ep_context_helper.CreateEpContextNode(ep_context_mxr_path, effective_cache_dir,
            context_embed_mode_, compute_capability_, model_path, context_node_name_prefix_,
            ep_context_node));
    }

    auto [state_it, inserted] = compute_states_.emplace(subgraph_name,
        ComputeState{
            mutex_,
            device_id_,
            t_,
            onnx_options,
            program,
            enable_fp16_,
            enable_bf16_,
            enable_fp8_,
            enable_int8_,
            int8_calibration_cache_available_,
            has_input_shape,
            dump_subgraphs_,
            exhaustive_tune_,
            mlss_use_specific_ops_,
            dynamic_ranges_,
            input_name_indices,
            output_name_indices,
            onnx_string,
            compute_mode_,
            model_path,
            effective_cache_dir,
            disable_compiled_model_caching_,
            force_recompile_,
            external_data_dir_,
            mxr_prefix,
        });

    // Propagate hipGraph / dynamic-batch configuration onto the compute state.
    auto& compute_state{state_it->second};
    compute_state.hip_graph_enable = hip_graph_enable_;
    compute_state.max_dynamic_batch = max_dynamic_batch_;
    compute_state.compile_batches = compile_batches_;
    compute_state.coalesce_io = coalesce_io_enable_;
    // Direct-bind zero-copy hipGraph is the default when hipGraph is enabled and
    // coalesce_io is off; coalesce_io routes inputs through the pinned staging
    // arena and is therefore mutually exclusive with the direct-bind path.
    compute_state.use_direct_hip_graph = hip_graph_enable_ && !coalesce_io_enable_;
    // Item 1 hybrid: with coalesce_io on, inputs stay coalesced but outputs are
    // bound directly to ORT tensors (drift-checked), skipping the staging->ORT D2D
    // copies.  Re-set each call so a transient-drift disable can recover; a sustained
    // drift keeps hybrid_recapture_count past the cap, holding it on the staging path.
    compute_state.hybrid_output_enable = hip_graph_enable_ && coalesce_io_enable_;
    compute_state.static_pad_seq = static_pad_seq_;
    compute_state.static_pad_seq_len = static_pad_seq_len_;
    compute_state.static_pad_seq_len_from_env = static_pad_seq_len_from_env_;
    if (static_pad_seq_) {
        // static-pad and dynamic batching both rewrite shapes; combined, the staging copies
        // collide. Unsupported together.
        if (max_dynamic_batch_ > 0) {
            return Ort::Status{"static seq-padding is not supported with dynamic batching "
                "(max_dynamic_batch > 0); disable one", ORT_EP_FAIL};
        }
        compute_state.static_pad_input_axes = ParseNameAxisSpec(static_pad_inputs_);
        // Outputs: MIGraphX program params are named "#output_N", not their ONNX
        // names, so resolve the user's "logits:1" spec to (ORT output index -> axis)
        // via output_name_indices; the runtime slice matches on index.
        compute_state.static_pad_output_axes_by_index.clear();
        for (const auto& [oname, axis] : ParseNameAxisSpec(static_pad_outputs_)) {
            if (const auto it{output_name_indices.find(oname)}; it != output_name_indices.end()) {
                compute_state.static_pad_output_axes_by_index.emplace(it->second, axis);
            }
        }
        // No named inputs -> nothing to pad; keep the feature inert.
        if (compute_state.static_pad_input_axes.empty()) {
            compute_state.static_pad_seq = false;
        }
    }

    if (max_dynamic_batch_ > 0 || !compile_batches_.empty()) {
        compute_state.has_dynamic_batch = max_dynamic_batch_ > 0;
        compute_state.compiled_batch_sizes = GenerateCompiledBatchSizes(
            max_dynamic_batch_, compile_batches_);
    }

    compute_state.defer_compilation = true;
    if (use_plan_cache) {
        RETURN_IF_ERROR(PreloadMxrPrograms(pre_plan, input_name_indices, compute_state.cached_programs,
            force_recompile_, effective_cache_dir, mxr_prefix));
        if (precompile_at_load_) {
            RETURN_IF_ERROR(CompileMissingPrograms(pre_plan, input_name_indices, onnx_string,
                compute_state.cached_programs, t_, enable_fp16_, enable_bf16_, enable_int8_, enable_fp8_,
                int8_calibration_cache_available_, dynamic_ranges_, exhaustive_tune_, mlss_use_specific_ops_,
                disable_compiled_model_caching_, model_path, external_data_dir_, effective_cache_dir, mxr_prefix));
        }
        if (!compute_state.cached_programs.empty()) {
            compute_state.program = SelectDefaultProgram(compute_state.cached_programs, pre_bucketed,
                pre_batch_sizes, pre_shapes_by_name, input_name_indices);
            if (pre_bucketed) {
                compute_state.compiled_batch_sizes = pre_batch_sizes;
                compute_state.has_dynamic_batch = true;
            } else {
                compute_state.has_input_shapes = true;
                const auto output_shapes{compute_state.program.get_output_shapes()};
                for (const auto& [n, s] : zip(output_names, output_shapes)) {
                    compute_state.onnx_options.set_input_parameter_shape(n, s.lengths());
                }
            }
            backend_telemetry_.loaded_from_cache = true;
        }
        compute_state.defer_compilation = AnyPlannedTargetMissing(pre_plan, input_name_indices,
            compute_state.cached_programs);
    } else if (has_input_shape && program.get_parameter_shapes().size() > 0) {
        compute_state.defer_compilation = false;
    }

    node_compute_info = std::make_unique<NodeComputeInfo>(*this).release();
    return STATUS_OK;
}

Ort::Status ExecutionProvider::CreateNodeComputeInfoFromCache(const Ort::ConstGraph& graph,
    const Ort::ConstNode& fused_node, const Map<size_t>& input_name_indices, const Map<size_t>& output_name_indices,
    OrtNodeComputeInfo*& node_compute_info)
{
    std::string name{fused_node.GetName()};

    fs::path ctx_cache_dir{cache_dir_};
    if (ctx_cache_dir.empty()) {
        if (!context_file_path_.empty()) {
            ctx_cache_dir = context_file_path_.parent_path();
        } else {
            const auto model_path{graph.GetModelPath()};
            if (!model_path.empty()) {
                ctx_cache_dir = fs::path{model_path}.parent_path();
            }
        }
    }

    const auto current_sdk_version{
        std::to_string(MIGRAPHX_VERSION_MAJOR) + "." +
        std::to_string(MIGRAPHX_VERSION_MINOR) + "." +
        std::to_string(MIGRAPHX_VERSION_PATCH)};

    try {
        EpContextNodeReader ep_ctx_reader{*this, graph, logger_, ctx_cache_dir,
            compute_capability_, current_sdk_version};

        ep_context_compute_states_.emplace(name,
            EpContextComputeState{
                mutex_,
                device_id_,
                t_,
                ep_ctx_reader.GetProgram(),
                input_name_indices,
                output_name_indices,
            });
    } catch (const std::exception& e) {
        return Ort::Status{e.what(), ORT_INVALID_GRAPH};
    }

    node_compute_info = std::make_unique<EpContextNodeComputeInfo>(*this).release();
    return STATUS_OK;
}

Ort::Status ExecutionProvider::Compile(const std::vector<Ort::ConstGraph>& graphs,
    const std::vector<Ort::ConstNode>& fused_nodes, gsl::span<OrtNodeComputeInfo*> node_compute_infos,
    gsl::span<OrtNode*> ep_context_nodes) noexcept
try {
    // Reset the backend-specific telemetry gathered for this compile. Generic
    // fields (model, version, ...) are collected by the wrapper; here we only
    // record what is specific to MIGraphX (gfx arch and MXR cache state).
    backend_telemetry_.backend = telemetry::Backend::MIGraphX;
    backend_telemetry_.has_gfx_arch = !compute_capability_.empty();
    std::snprintf(backend_telemetry_.gfx_arch, sizeof(backend_telemetry_.gfx_arch),
        "%s", compute_capability_.c_str());
    backend_telemetry_.has_loaded_from_cache = true;
    backend_telemetry_.loaded_from_cache = false;

    for (const auto& [graph, fused_node, node_compute_info, ep_context_node] : zip(graphs, fused_nodes, node_compute_infos, ep_context_nodes)) {
        const auto input_name_indices{BuildInputNameIndices(graph, fused_node)};
        const auto outputs{GetValueInfos(fused_node.GetOutputs())};
        std::unordered_map<std::string, size_t> output_name_indices;
        output_name_indices.reserve(outputs.size());
        for (const auto& [i, output] : enumerate(outputs)) {
            output_name_indices.emplace(output.GetName(), i);
        }

        if (EpContextNodeReader::GraphHasContextNode(graph)) {
            backend_telemetry_.loaded_from_cache = true;
            RETURN_IF_ERROR(CreateNodeComputeInfoFromCache(graph, fused_node, input_name_indices,
                output_name_indices, node_compute_info));
        } else {
            const auto mxr_prefix{hash::ToHex(MIGraphX_Version) + "-" + GenerateGraphId(graph) + "-" +
                hash::ToHex(std::string_view{device_prop_.gcnArchName}) + "-"};

            RETURN_IF_ERROR(CreateNodeComputeInfoFromGraph(graph, fused_node, input_name_indices,
                output_name_indices, mxr_prefix, node_compute_info, ep_context_node));
        }
    }
    return STATUS_OK;
} catch (const Ort::Exception& e) {
    return Ort::Status{e};
} catch (const std::exception& e) {
    return Ort::Status{e.what(), ORT_EP_FAIL};
}

Ort::Status ExecutionProvider::ReleaseNodeComputeInfos(const gsl::span<OrtNodeComputeInfo*> node_compute_info) noexcept
try {
    for (const auto& info : node_compute_info) {
        delete info;
    }
    return STATUS_OK;
} catch (const Ort::Exception& e) {
    return Ort::Status{e};
} catch (const std::exception& e) {
    return Ort::Status{e.what(), ORT_EP_FAIL};
}

Ort::Status ExecutionProvider::OnRunEnd(const OrtRunOptions* /* run_options */, bool /* sync_stream */) noexcept
try {
    HIP_RETURN_IF_ERROR(hipSetDevice(device_id_));
    return STATUS_OK;
} catch (const Ort::Exception& e) {
    return Ort::Status{e};
} catch (const std::exception& e) {
    return Ort::Status{e.what(), ORT_EP_FAIL};
}

Ort::Status ExecutionProvider::CreateSyncStreamForDevice(const OrtMemoryDevice* memory_device, OrtSyncStreamImpl** stream)
try {
    // Adopt the application-owned stream when one was supplied via provider
    // options (mirrors the built-in EP); otherwise SyncStream creates+owns its own.
    const auto external_stream{has_user_compute_stream_
        ? static_cast<hipStream_t>(user_compute_stream_)
        : static_cast<hipStream_t>(nullptr)};
    *stream = std::make_unique<hip::SyncStream>(*this, device_id_, external_stream).release();
    return STATUS_OK;
} catch (const Ort::Exception& e) {
    return Ort::Status{e};
} catch (const std::exception& e) {
    return Ort::Status{e.what(), ORT_EP_FAIL};
}

Ort::Status ExecutionProvider::GetKernelRegistry(const OrtKernelRegistry** kernel_registry) const
try {
    return factory_.GetKernelRegistry(ep_name_, *kernel_registry);
} catch (const Ort::Exception& e) {
    return Ort::Status{e};
} catch (const std::exception& e) {
    return Ort::Status{e.what(), ORT_EP_FAIL};
}

Ort::Status NodeComputeInfo::CreateState(OrtNodeComputeContext* compute_context, void*& compute_state) noexcept
try {
    const auto fused_node_name{ep_.ep_api.NodeComputeContext_NodeName(compute_context)};
    auto& cs{ep_.GetComputeState(fused_node_name)};
    compute_state = &cs;
    return STATUS_OK;
} catch (const Ort::Exception& e) {
    return Ort::Status{e};
} catch (const std::exception& e) {
    return Ort::Status{e.what(), ORT_EP_FAIL};
}

namespace {

// Resolved once per Compute call and threaded to each mechanism path so the
// direct / staging / eager branches never re-read shapes or re-derive the batch.
struct ComputeIOInfo {
    DynamicBatchContext dyn{};
    StaticSeqContext seq{};
    ShapeKey shape_key{};
    bool needs_padding{false};
    const migraphx::program_parameter_shapes* param_shapes{nullptr};
    hipStream_t     hip_stream{nullptr};  // ORT compute stream, resolved once per call
};

// Map an actual input shape to the shape the program is compiled for, in place: a
// batched input (axis-0 extent == requested batch) is bucketed up to the target
// batch, and a named seq input has its token axis padded up to the static target
// length.  Single source of truth for the transform applied by both the shape hash
// and the parse-time parameter shapes, so the two can never drift.
void ApplyEffectiveShape(std::vector<std::int64_t>& shape, const std::string& name,
    const DynamicBatchContext& dyn, const StaticSeqContext& seq) {
    if (dyn.active && !shape.empty() &&
        static_cast<std::size_t>(shape.front()) == dyn.requested_batch) {
        shape.front() = static_cast<std::int64_t>(dyn.target_batch);
    }
    if (seq.active && seq.input_axes != nullptr) {
        if (const auto it{seq.input_axes->find(name)}; it != seq.input_axes->end()) {
            const int axis{it->second};
            if (axis >= 0 && static_cast<std::size_t>(axis) < shape.size() &&
                static_cast<std::size_t>(shape[axis]) == seq.real_len) {
                shape[axis] = static_cast<std::int64_t>(seq.target_len);
            }
        }
    }
}

// Resolve static seq-padding for this call (inert when disabled).  A named input is
// padded on its token axis to the fixed target length, only for prefill: the axis
// extent must be in (1, target).  decode (extent == 1) and already-max prompts are
// left alone.  Returns an inactive context when static_pad_seq_len == 0.
StaticSeqContext ResolveSeqPadding(ComputeState& compute_state,
    const Ort::KernelContext& kernel_context) {
    StaticSeqContext seq{};
    if (!compute_state.static_pad_seq) {
        return seq;
    }
    const auto& input_name_indices{compute_state.input_name_indices};
    // Take the target from the attention mask (axis 1; axis 0 is batch, scaled by num_beams).
    // It carries the length the caller is really generating with, and is the same length it
    // sized its KV cache to.  The configured value cannot be trusted here: for OGA it is
    // config.search.max_length, read while the session is created, so a caller that sets
    // max_length on GeneratorParams sets it too late for us and we would pad to the model's
    // full context.  An explicit env value still wins; models with no mask keep the
    // configured length.
    static const std::string kAttentionMaskInput{"attention_mask"};
    std::size_t target{compute_state.static_pad_seq_len};
    if (not compute_state.static_pad_seq_len_from_env) {
        if (const auto mask_it{input_name_indices.find(kAttentionMaskInput)};
            mask_it != input_name_indices.end()) {
            const auto mask_shape{
                kernel_context.GetInput(mask_it->second).GetTensorTypeAndShapeInfo().GetShape()};
            if (mask_shape.size() > 1 && mask_shape[1] > 0) {
                target = static_cast<std::size_t>(mask_shape[1]);
            }
        }
    }
    for (const auto& [name, axis] : compute_state.static_pad_input_axes) {
        if (target == 0) {
            break;  // nothing usable to pad to -> leave shapes untouched (inert, as before)
        }
        const auto it{input_name_indices.find(name)};
        if (it == input_name_indices.end()) {
            continue;
        }
        const auto shape{kernel_context.GetInput(it->second).GetTensorTypeAndShapeInfo().GetShape()};
        if (axis < 0 || static_cast<size_t>(axis) >= shape.size()) {
            continue;
        }
        const auto extent{static_cast<size_t>(shape[axis])};
        if (extent > 1 && extent < target) {
            seq.active = true;
            seq.real_len = extent;
            seq.target_len = target;
            seq.input_axes = &compute_state.static_pad_input_axes;
            seq.output_axes_by_index = &compute_state.static_pad_output_axes_by_index;
            break;  // all named inputs share the same token length this call
        }
    }
    return seq;
}

// Single pass over the model inputs: gather the raw dims (flattened), record each
// input's rank and raw data ptr on the compute state, and resolve the dynamic-batch
// bucket from the lowest-index input's axis-0 extent.  Returns the flattened raw
// shapes so the caller can compare against the previous call and hash them.
std::vector<std::int64_t> GatherInputShapesAndBatch(ComputeState& compute_state,
    const Ort::KernelContext& kernel_context, DynamicBatchContext& dyn) {
    const auto& input_name_indices{compute_state.input_name_indices};
    std::vector<std::int64_t> current_input_shapes;
    current_input_shapes.reserve(input_name_indices.size() * 4);
    // Raw input data ptrs (by ORT index), reused by the coalesced copy; nullptr for
    // any index the scan skips.
    compute_state.cur_input_data.assign(input_name_indices.size(), nullptr);
    // Per-input dim counts, so the shape-changed rehash slices current_input_shapes by
    // offset instead of re-fetching shapes.
    compute_state.cur_input_ranks.clear();
    compute_state.cur_input_ranks.reserve(input_name_indices.size());
    // Reused per input: dims read via the C API to avoid a GetShape() vector alloc each.
    std::vector<std::int64_t> dims;
    const bool track_batch{compute_state.max_dynamic_batch > 0};
    bool have_batch_min{false};
    std::size_t batch_min_index{0};
    std::size_t requested_batch{0};
    for (const auto& [name, index] : input_name_indices) {
        const auto input_value{kernel_context.GetInput(index)};
        const auto info{input_value.GetTensorTypeAndShapeInfo()};
        const OrtTensorTypeAndShapeInfo* info_handle{info};
        const std::size_t rank{info.GetDimensionsCount()};
        dims.resize(rank);
        if (rank > 0) {
            Ort::ThrowOnError(Ort::GetApi().GetDimensions(info_handle, dims.data(), rank));
        }
        current_input_shapes.insert(current_input_shapes.end(), dims.begin(), dims.end());
        compute_state.cur_input_ranks.push_back(static_cast<std::uint32_t>(rank));
        if (index < compute_state.cur_input_data.size()) {
            compute_state.cur_input_data[index] = input_value.GetTensorRawData();
        }
        if (track_batch && rank > 0 && (!have_batch_min || index < batch_min_index)) {
            have_batch_min = true;
            batch_min_index = index;
            requested_batch = static_cast<std::size_t>(dims.front());
        }
    }
    if (track_batch && requested_batch > 0) {
        if (compute_state.compiled_batch_sizes.empty()) {
            compute_state.compiled_batch_sizes = GenerateCompiledBatchSizes(
                compute_state.max_dynamic_batch, compute_state.compile_batches);
        }
        const auto bucket{FindNearestCompiledBatchSize(requested_batch, compute_state.compiled_batch_sizes)};
        dyn.requested_batch = requested_batch;
        dyn.target_batch = bucket > 0 ? bucket : requested_batch;
        dyn.active = true;
    }
    return current_input_shapes;
}

// Hash every input's effective (batch/seq-transformed) shape, reusing the dims already
// gathered in current_input_shapes + cur_input_ranks (no GetShape() re-fetch).  Used
// for both the first-call seed and the shape-changed rehash so there is a single
// hashing code path -- the two can never produce divergent keys for identical inputs.
hash::Value HashEffectiveInputShapes(const ComputeState& compute_state,
    const std::vector<std::int64_t>& current_input_shapes,
    const DynamicBatchContext& dyn, const StaticSeqContext& seq) {
    hash::Value input_shapes_hash{};
    std::vector<std::int64_t> dims;
    std::size_t shape_off{0};
    std::size_t rank_idx{0};
    for (const auto& [name, index] : compute_state.input_name_indices) {
        const std::size_t rank{compute_state.cur_input_ranks[rank_idx++]};
        dims.assign(current_input_shapes.begin() + shape_off,
                    current_input_shapes.begin() + shape_off + rank);
        shape_off += rank;
        ApplyEffectiveShape(dims, name, dyn, seq);
        hash::Hash(input_shapes_hash,
            gsl::span<const std::int64_t>{dims.data(), dims.size()});
    }
    return input_shapes_hash;
}

// Resolve the program for shape_key: reuse the in-memory cached program if present,
// otherwise load the .mxr or (re)parse + calibrate + compile + save, then drop any
// per-shape caches bound to the previous program and record the active key.  Mirrors
// the built-in EP's handle_input_shape_mismatch + load_or_compile_model split.  Only
// called on a shape miss (the hot path skips it).  Throws on parse/compile failure.
void ResolveProgram(ComputeState& compute_state, const Ort::KernelContext& kernel_context,
    ShapeKey shape_key, const hash::Value& input_shapes_hash,
    const DynamicBatchContext& dyn, const StaticSeqContext& seq) {
    const auto& input_name_indices{compute_state.input_name_indices};
    auto& program{compute_state.program};
    auto& onnx_options{compute_state.onnx_options};

    bool loaded_from_cache{false};

    if (const auto cit{compute_state.cached_programs.find(shape_key)};
        cit != compute_state.cached_programs.end()) {
        program = cit->second;
        loaded_from_cache = true;
    } else if (compute_state.hip_graph_enable && !dyn.active) {
        // Unbounded dynamic-shape (e.g. LLM) path: keep only the current shape's
        // graph/staging, so invalidate before the program changes.
        DestroyHipGraphs(compute_state);
        FreeStaging(compute_state,
            static_cast<hipStream_t>(kernel_context.GetGPUComputeStream()));
        // Direct-bind bindings reference the graphs just destroyed; drop them
        // so they are repopulated (and re-captured) against the new program.
        // Clear param shapes too so the map does not grow per token length.
        compute_state.direct_bind_cache.clear();
        compute_state.cached_param_shapes.clear();
    }

    if (!loaded_from_cache) {
        migraphx::program_parameters compile_params{};
        fs::path mxr_path;
        if (!compute_state.cache_dir.empty()) {
            mxr_path = compute_state.cache_dir /
                (compute_state.mxr_prefix + hash::ToHex(input_shapes_hash) + ".mxr");
        }
        if (compute_state.force_recompile || !load_compiled_program(program, mxr_path)) {
            const auto external_data_dir{compute_state.external_data_dir.empty() ?
                compute_state.model_path.parent_path() : compute_state.external_data_dir};
            onnx_options.set_external_data_path(external_data_dir.string());
            // Set input parameter shapes for the parse (onnx_options' only consumer),
            // applying the same effective-shape transform used to hash so the compiled
            // program matches the key it is cached under.
            for (const auto& [pname, pindex] : input_name_indices) {
                auto pshape{kernel_context.GetInput(pindex).GetTensorTypeAndShapeInfo().GetShape()};
                ApplyEffectiveShape(pshape, pname, dyn, seq);
                onnx_options.set_input_parameter_shape(pname, {pshape.begin(), pshape.end()});
            }
            program = migraphx::parse_onnx_buffer(compute_state.onnx_string, onnx_options);
            // Pre-compile shapes, for calibration only.
            const auto calib_param_shapes{program.get_parameter_shapes()};
            if ((compute_state.enable_int8 ^ compute_state.enable_fp8) && compute_state.int8_calibration_cache_available) {
                for (const auto& name : calib_param_shapes.names()) {
                    if (input_name_indices.count(name) > 0) {
                        const auto index{input_name_indices.at(name)};
                        auto value{kernel_context.GetInput(index)};
                        auto tensor_info{value.GetTensorTypeAndShapeInfo()};
                        const auto tensor_shape{tensor_info.GetShape()};
                        const auto tensor_type{tensor_info.GetElementType()};

                        migraphx_shape_datatype_t datatype;
                        GetMIGraphXType(tensor_type, datatype);

                        if (const auto prog_shapes{calib_param_shapes[name]}; datatype != prog_shapes.type()) {
                            throw std::runtime_error{"NodeComputeInfo::Compute(): input parameter type mismatch"};
                        }

                        compile_params.add(name, migraphx::argument{calib_param_shapes[name],
                            GetGpuInputData(compute_state, kernel_context, name, index, calib_param_shapes[name], nullptr)});
                    }
                }
            }
            calibrate_and_quantize(program, compute_state.t, compile_params,
                compute_state.enable_fp16, compute_state.enable_bf16, compute_state.enable_int8,
                compute_state.enable_fp8, compute_state.int8_calibration_cache_available, compute_state.dynamic_ranges);

            compile_program(program, compute_state.t, compute_state.exhaustive_tune,
                compute_state.mlss_use_specific_ops);
            if (!compute_state.disable_compiled_model_caching) {
                save_compiled_program(program, mxr_path);
            }
        }
        // Keep the freshly compiled program so it (and its captured graph) survive
        // later shape/bucket switches.
        if (dyn.active || !compute_state.defer_compilation) {
            compute_state.cached_programs.emplace(shape_key, program);
        }
        // Freshly built -> drop anything cached against the old program.
        compute_state.staging_bind_cache.erase(shape_key);
        compute_state.direct_bind_cache.erase(shape_key);
        compute_state.cached_param_shapes.erase(shape_key);
    }
    // Program now matches this call's shapes; record its key for the next compare.
    compute_state.active_program_shape_key = shape_key;
    compute_state.has_active_program_shape_key = true;
}

// Cache-tier stage (orchestrator): resolve padding, gather input shapes once, hash +
// match against the active program (compiling on a miss), then bind param_shapes and
// record everything the mechanism paths need into `io`.  This collapses the built-in
// EP's ultra-fast / fast / standard tiers into a single deterministic pass so no
// mechanism path re-reads shapes.  Throws on failure (the Compute function-try-block
// converts it to a Status).
Ort::Status ResolveComputeIO(ComputeState& compute_state,
    const Ort::KernelContext& kernel_context, ComputeIOInfo& io) {
    // 1. Resolve the padding contexts for this call (both inert when disabled).
    const StaticSeqContext seq{ResolveSeqPadding(compute_state, kernel_context)};
    DynamicBatchContext dyn{};

    // 2. Single pass: gather input shapes + data ptrs and resolve the dynamic-batch
    //    bucket.  If the raw shapes are identical to the previous call we reuse the
    //    cached hash and skip the rehash/match work below.
    std::vector<std::int64_t> current_input_shapes{
        GatherInputShapesAndBatch(compute_state, kernel_context, dyn)};
    const bool shapes_unchanged{compute_state.has_last_input_shapes &&
                                current_input_shapes == compute_state.last_input_shapes};

    // 3. Hash the effective input shapes (unless unchanged) and decide whether the
    //    active program still matches this call.  A single hashing path serves both
    //    the first-call seed and the shape-changed rehash.
    hash::Value input_shapes_hash{};
    bool input_shapes_match{compute_state.has_input_shapes};
    if (shapes_unchanged) {
        input_shapes_hash = compute_state.last_input_shapes_hash;
        input_shapes_match = true;
    } else {
        input_shapes_hash =
            HashEffectiveInputShapes(compute_state, current_input_shapes, dyn, seq);
        if (!compute_state.has_input_shapes) {
            // First call: seed the shapes hash and leave match false so we compile below.
            compute_state.has_input_shapes = true;
            input_shapes_match = false;
        }
    }

    // Integer key for the per-shape hot caches and the cached_programs lookup.
    const ShapeKey shape_key{hash::ShapeKeyOf(input_shapes_hash)};

    // The active program matches iff its compiled key equals this call's.  (shapes_
    // unchanged already set true; the first call left it false so it compiles below.)
    if (!shapes_unchanged && input_shapes_match) {
        input_shapes_match = compute_state.has_active_program_shape_key &&
                             shape_key == compute_state.active_program_shape_key;
    }

    // 4. Shapes differ from the active program (e.g. LLMs / bucket switch): look up
    //    the program for this shape by its integer key, or (re)compile it.
    if (!input_shapes_match) {
        ResolveProgram(compute_state, kernel_context, shape_key, input_shapes_hash, dyn, seq);
    }

    // 5. Bind param_shapes for the program that will run this call (cached per hash;
    //    built once on a miss).
    auto& cached_param_shapes{compute_state.cached_param_shapes};
    auto param_shapes_it{cached_param_shapes.find(shape_key)};
    if (param_shapes_it == cached_param_shapes.end()) {
        param_shapes_it = cached_param_shapes.emplace(
            shape_key, compute_state.program.get_parameter_shapes()).first;
    }
    const auto& param_shapes{param_shapes_it->second};

    // 6. Item 4: remember these shapes so the next identical call takes the fast path.
    //    Set only after the (possible) recompile above, so the recorded hash always
    //    corresponds to a program that is compiled and ready for these shapes.
    if (!shapes_unchanged) {
        compute_state.last_input_shapes = std::move(current_input_shapes);
        compute_state.last_input_shapes_hash = input_shapes_hash;
        compute_state.has_last_input_shapes = true;
    }

    // Whether this call needs the program run at a larger shape than the request
    // (batch bucketed up, or seq padded up).  Padding requires the staging path
    // (input pad / output slice); the direct-bind path cannot be used.
    const bool needs_batch_pad{dyn.active && dyn.target_batch > dyn.requested_batch};
    const bool needs_seq_pad{seq.active && seq.target_len > seq.real_len};

    io.dyn = dyn;
    io.seq = seq;
    io.shape_key = shape_key;
    io.needs_padding = needs_batch_pad || needs_seq_pad;
    io.param_shapes = &param_shapes;
    return STATUS_OK;
}

// Mechanism (built-in EP's use_direct_hip_graph): direct-bind (zero-copy)
// hipGraph.  Bind ORT input/output tensor pointers straight into the program and
// capture/replay with no staging copies.  Returns nullopt when this path does not
// apply (fall through to staging); otherwise the terminal Status.  Inputs are
// assumed device-resident (Triton's GPU-first io-binding hands the EP device
// pointers); gated on no batch/seq padding.  Any disqualifying case falls through;
// sustained ORT pointer drift disables the direct path at runtime (see
// RunProgramOrHipGraphDirect), which also lands on the staging path.
std::optional<Ort::Status> TryDirectBind(ComputeState& compute_state,
    const Ort::KernelContext& kernel_context, const ComputeIOInfo& io) {
    const auto& input_name_indices{compute_state.input_name_indices};
    auto& program{compute_state.program};
    const auto& param_shapes{*io.param_shapes};
    const auto& dyn{io.dyn};
    const auto shape_key{io.shape_key};
    const bool needs_padding{io.needs_padding};

    if (compute_state.use_direct_hip_graph && !needs_padding && param_shapes.size() > 0) {
        const auto hip_stream{io.hip_stream};

        // Populate the binding cache once per compiled shape (keyed by hash), so
        // subsequent calls for the same shape -- and alternating dynamic-batch
        // buckets -- reuse it, skipping the per-call name lookups and dtype checks.
        auto dbc_it{compute_state.direct_bind_cache.find(shape_key)};
        if (dbc_it == compute_state.direct_bind_cache.end()) {
            DirectBindCache dbc;
            PopulateDirectCaches(dbc, param_shapes, input_name_indices);
            // ORT output indices are constant per shape -- build once here.
            dbc.prog_output_indices.reserve(dbc.outputs.size());
            for (const auto& out : dbc.outputs) {
                dbc.prog_output_indices.push_back(out.output_index);
            }
            dbc_it = compute_state.direct_bind_cache.emplace(shape_key, std::move(dbc)).first;
        }
        auto& dbc{dbc_it->second};

        // Gather the current ORT device pointers into reused flat vectors (no
        // per-call maps, string hashing, or program_parameters.add(): those are
        // deferred to the capture/eager path).  Order matches dbc.inputs/outputs
        // so RunProgramOrHipGraphDirect can compare them positionally for drift.
        dbc.cur_input_ptrs.resize(dbc.inputs.size());
        for (std::size_t i{0}; i < dbc.inputs.size(); ++i) {
            dbc.cur_input_ptrs[i] = const_cast<void*>(
                kernel_context.GetInput(dbc.inputs[i].ort_index).GetTensorRawData());
        }
        dbc.cur_output_ptrs.resize(dbc.outputs.size());
        for (std::size_t i{0}; i < dbc.outputs.size(); ++i) {
            const auto& out{dbc.outputs[i]};
            auto output_tensor{kernel_context.GetOutput(out.output_index,
                out.ort_shape.data(), out.ort_shape.size())};
            dbc.cur_output_ptrs[i] = output_tensor.GetTensorMutableRawData();
        }
        // EP-owned scratch (allocate-and-zero on first use, zero thereafter).
        // Presence + shape were resolved once in PopulateDirectCaches, and the
        // scratch_bufs slot is cached on the dbc, so this does no scan and (after
        // the first call) no map lookup.
        const auto scratch{GetOrAllocScratchCached(compute_state, dbc, shape_key, hip_stream)};

        RunProgramOrHipGraphDirect(compute_state, hip_stream, kernel_context, program,
            dbc, shape_key, scratch, dyn,
            compute_state.use_direct_hip_graph, compute_state.direct_recapture_count);
        return STATUS_OK;
    }
    return std::nullopt;
}

// Mechanism (built-in EP's staging hipGraph path): stage I/O into EP-owned
// buffers, bind scratch, then replay/capture a graph.  Required for hipGraph
// capture (pointer stability) and for dynamic batching (input padding / output
// slicing).  Returns nullopt when this path does not apply (fall through to
// eager); otherwise the terminal Status.
std::optional<Ort::Status> TryStaging(ComputeState& compute_state,
    const Ort::KernelContext& kernel_context, const ComputeIOInfo& io) {
    auto& program{compute_state.program};
    const auto& param_shapes{*io.param_shapes};
    const auto& dyn{io.dyn};
    const auto& seq{io.seq};
    const auto shape_key{io.shape_key};

    if ((compute_state.hip_graph_enable || dyn.active || seq.active) && param_shapes.size() > 0) {
        const auto hip_stream{io.hip_stream};
        AllocateStaging(compute_state, param_shapes, hip_stream, dyn);
        // Item 3: reuse a cached binding for this shape hash.  Staging buffers and
        // scratch are pointer-stable until FreeStaging, so binding once and replaying
        // avoids re-doing N program_parameters.add() calls, string work, and a
        // scratch lookup on every inference.  The binding is resolved before the input
        // copy because it also carries the flat input-copy plan the copy consumes, so
        // the per-call copy touches no parameter names, std::strings, or map lookups.
        auto bind_it{compute_state.staging_bind_cache.find(shape_key)};
        if (bind_it == compute_state.staging_bind_cache.end()) {
            bind_it = compute_state.staging_bind_cache.emplace(
                shape_key,
                BindStagingParams(compute_state, param_shapes, shape_key, hip_stream)).first;
        }
        auto& bind{bind_it->second};
        CopyInputsToStaging(compute_state, bind, kernel_context, hip_stream, dyn, seq);

        // Item 1 (hybrid direct-bound outputs): when this call needs no padding, bind
        // the program outputs directly to the ORT tensors (drift-checked) and skip the
        // per-output staging->ORT D2D copies entirely.  Inputs are already coalesced
        // into the (pointer-stable) arena, so cur_input_ptrs never drift and only the
        // ORT output pointers are gathered + drift-checked here.  Reuses the direct
        // path's capture/replay/drift machinery with the hybrid's own enable flag +
        // recapture counter; on ineligibility or sustained output drift it falls back
        // to the staging copy path below.  Padding disqualifies the hybrid because the
        // program writes target-batch/target-len rows that the staging path must slice
        // down to the requested shape.
        const bool no_padding{
            (!dyn.active || dyn.target_batch == dyn.requested_batch) &&
            (!seq.active || seq.target_len == seq.real_len)};
        if (compute_state.hybrid_output_enable && no_padding && bind.hybrid.eligible &&
            compute_state.staging_inputs_coalesced) {
            auto& hyb{bind.hybrid};
            hyb.cur_output_ptrs.resize(hyb.outputs.size());
            for (std::size_t i{0}; i < hyb.outputs.size(); ++i) {
                const auto& out{hyb.outputs[i]};
                auto output_tensor{kernel_context.GetOutput(out.output_index,
                    out.ort_shape.data(), out.ort_shape.size())};
                hyb.cur_output_ptrs[i] = output_tensor.GetTensorMutableRawData();
            }
            // cur_input_ptrs are the arena sub-views resolved once in BindStagingParams.
            const auto scratch{GetOrAllocScratchCached(compute_state, hyb, shape_key, hip_stream)};
            RunProgramOrHipGraphDirect(compute_state, hip_stream, kernel_context, program,
                hyb, shape_key, scratch, dyn,
                compute_state.hybrid_output_enable, compute_state.hybrid_recapture_count);
        } else {
            RunProgramOrHipGraph(compute_state, hip_stream, kernel_context, program,
                bind, shape_key, dyn);
            CopyStagingOutputsToOrt(bind, kernel_context, hip_stream, dyn, seq);
        }
        return STATUS_OK;
    }
    return std::nullopt;
}

// Mechanism (built-in EP's standard run): eager fallback with no hipGraph.  Build
// program_parameters against the ORT tensors, run the program, and materialize any
// extra outputs.  Terminal: the last path in the chain always runs and returns a
// Status.
Ort::Status RunEager(ComputeState& compute_state,
    const Ort::KernelContext& kernel_context, const ComputeIOInfo& io) {
    const auto& input_name_indices{compute_state.input_name_indices};
    auto& program{compute_state.program};
    const auto& param_shapes{*io.param_shapes};

    migraphx::program_parameters compute_params;
    auto output_shapes{program.get_output_shapes()};
    std::vector<size_t> output_indices;
    const auto hip_stream{io.hip_stream};
    if (param_shapes.size() > 0) {
        for (std::string name : param_shapes.names()) {
            if (input_name_indices.count(name) > 0) {
                const auto index{input_name_indices.at(name)};
                auto input_tensor{kernel_context.GetInput(index)};
                auto tensor_info{input_tensor.GetTensorTypeAndShapeInfo()};
                auto tensor_type{tensor_info.GetElementType()};

                migraphx_shape_datatype_t datatype;
                GetMIGraphXType(tensor_type, datatype);

                const auto prog_shape{param_shapes[name.c_str()]};
                if (datatype != prog_shape.type()) {
                    throw std::runtime_error{"NodeComputeInfo::Compute(): tensor parameter type mismatch"};
                }
                void* input_data{GetGpuInputData(compute_state, kernel_context, name, index, prog_shape, hip_stream)};
                compute_params.add(name.c_str(), migraphx::argument{prog_shape, input_data});
            } else {
                // it is an output argument
                constexpr std::string_view name_prefix{"#output_"};
                if (const auto pos{name.find(name_prefix)}; pos != std::string_view::npos) {
                    const auto sub{name.substr(pos + name_prefix.length())};
                    auto output_index{ToInteger<size_t>(Trim(sub, std::isdigit))};
                    output_indices.emplace_back(output_index);

                    auto output_shape{output_shapes[output_index]};
                    const auto lengths{output_shape.lengths()};
                    std::vector<int64_t> tensor_shape{lengths.begin(), lengths.end()};
                    auto output_tensor{kernel_context.GetOutput(output_index, tensor_shape.data(), tensor_shape.size())};
                    void* output_data{output_tensor.GetTensorMutableRawData()};
                    auto argument_shape{param_shapes[name.c_str()]};
                    compute_params.add(name.c_str(), migraphx::argument{argument_shape, output_data});
                } else {
                    return Ort::Status{MakeString("NodeComputeInfo::Compute(): unbound program parameter '",
                        name, "'").c_str(), ORT_EP_FAIL};
                }
            }
        }
    }
    {
        std::lock_guard lock{compute_state.mutex};

        auto prog_outputs{program.run_async(compute_params, hip_stream)};
        CopyUnboundOutputsToOrt(kernel_context, hip_stream, prog_outputs, output_indices);
    }
    return STATUS_OK;
}

// Finalize a mechanism's result: propagate any error unchanged, otherwise finalize
// the compute stream once (async no-op by default; drains only on the legacy gate /
// null stream).  A free function so no closure is constructed per Compute call.
Ort::Status FinalizeCompute(Ort::Status status, hipStream_t hip_stream) {
    RETURN_IF_ERROR(std::move(status));
    return FinishComputeStream(hip_stream);
}

}  // namespace

// Orchestrator: resolve the shape / program once, then walk the mechanism chain.
// Each Try* returns nullopt to fall through to the next mechanism; the first to
// return a value (OK or error) is this call's terminal result.  The function-try
// -block converts any exception thrown by a stage into a Status.
Ort::Status NodeComputeInfo::Compute(ComputeState& compute_state,
    const Ort::KernelContext& kernel_context) noexcept
try {
    // Pin this (possibly pool-owned) thread to the EP's GPU for the whole call:
    // (re)compilation loads code objects and every run branch launches kernels,
    // all of which must target compute_state.device_id.  Set once here (only if
    // the thread isn't already on it) rather than before each branch.
    const HipDeviceGuard dev_guard{compute_state.device_id};

    ComputeIOInfo io;
    io.hip_stream = static_cast<hipStream_t>(kernel_context.GetGPUComputeStream());
    RETURN_IF_ERROR(ResolveComputeIO(compute_state, kernel_context, io));

    // Every mechanism issues stream-ordered work and hands back a Status; the stream
    // is finalized once via FinalizeCompute (async no-op by default; drains only on the
    // legacy gate or a null stream).  An early error short-circuits before the sync,
    // matching the pre-refactor per-path behavior.
    if (auto status{TryDirectBind(compute_state, kernel_context, io)}) {
        return FinalizeCompute(std::move(*status), io.hip_stream);
    }
    if (auto status{TryStaging(compute_state, kernel_context, io)}) {
        return FinalizeCompute(std::move(*status), io.hip_stream);
    }
    return FinalizeCompute(RunEager(compute_state, kernel_context, io), io.hip_stream);
} catch (const Ort::Exception& e) {
    return Ort::Status{e};
} catch (const std::exception& e) {
    return Ort::Status{e.what(), ORT_EP_FAIL};
}

Ort::Status EpContextNodeComputeInfo::CreateState(OrtNodeComputeContext* compute_context, void*& compute_state) noexcept
try {
    const auto fused_node_name{ep_.ep_api.NodeComputeContext_NodeName(compute_context)};
    auto& cs{ep_.EpContext_GetComputeState(fused_node_name)};
    compute_state = &cs;
    return STATUS_OK;
} catch (const Ort::Exception& e) {
    return Ort::Status{e};
} catch (const std::exception& e) {
    return Ort::Status{e.what(), ORT_EP_FAIL};
}

Ort::Status EpContextNodeComputeInfo::Compute(EpContextComputeState& compute_state, const Ort::KernelContext& kernel_context) noexcept
try {
    const auto& input_name_indices{compute_state.input_name_indices};
    auto& program{compute_state.program};

    // Pin this (possibly pool-owned) thread to the EP's GPU before the kernel
    // launch below (see HipDeviceGuard); set once instead of before run_async.
    const HipDeviceGuard dev_guard{compute_state.device_id};

    auto param_shapes{program.get_parameter_shapes()};
    auto output_shapes{program.get_output_shapes()};

    migraphx::program_parameters compute_params;
    std::vector<size_t> output_indices;

    if (param_shapes.size() > 0) {
        for (std::string name : param_shapes.names()) {
            if (input_name_indices.count(name) > 0) {
                const auto index{input_name_indices.at(name)};
                auto input_tensor{kernel_context.GetInput(index)};
                auto tensor_info{input_tensor.GetTensorTypeAndShapeInfo()};
                auto tensor_type{tensor_info.GetElementType()};

                migraphx_shape_datatype_t datatype;
                GetMIGraphXType(tensor_type, datatype);

                if (auto prog_shape{param_shapes[name.c_str()]}; datatype != prog_shape.type()) {
                    throw std::runtime_error{"EpContextNodeComputeInfo::Compute(): tensor parameter type mismatch"};
                }
                compute_params.add(name.c_str(), migraphx::argument{param_shapes[name.c_str()],
                    const_cast<void*>(input_tensor.GetTensorRawData())});
            } else {
                constexpr std::string_view name_prefix{"#output_"};
                if (const auto pos{name.find(name_prefix)}; pos != std::string_view::npos) {
                    const auto sub{name.substr(pos + name_prefix.length())};
                    auto output_index{ToInteger<size_t>(Trim(sub, std::isdigit))};
                    output_indices.emplace_back(output_index);

                    auto output_shape{output_shapes[output_index]};
                    const auto lengths{output_shape.lengths()};
                    std::vector<int64_t> tensor_shape{lengths.begin(), lengths.end()};
                    auto output_tensor{kernel_context.GetOutput(output_index, tensor_shape.data(), tensor_shape.size())};
                    void* output_data{output_tensor.GetTensorMutableRawData()};
                    auto argument_shape{param_shapes[name.c_str()]};
                    compute_params.add(name.c_str(), migraphx::argument{argument_shape, output_data});
                } else {
                    return Ort::Status{MakeString("EpContextNodeComputeInfo::Compute(): unbound program parameter '",
                        name, "'").c_str(), ORT_EP_FAIL};
                }
            }
        }
    }
    {
        std::lock_guard lock{compute_state.mutex};

        auto hip_stream{static_cast<hipStream_t>(kernel_context.GetGPUComputeStream())};
        auto prog_outputs{program.run_async(compute_params, hip_stream)};

        if (auto output_size{prog_outputs.size()}; output_indices.size() < output_size) {
            for (size_t i{}; i < output_size; ++i) {
                if (ranges::find(output_indices, i) != output_indices.end()) {
                    continue;
                }
                auto gpu_resource{prog_outputs[i]};
                migraphx::shape resource_shape{gpu_resource.get_shape()};
                auto resource_lengths{resource_shape.lengths()};
                std::vector<int64_t> shapes{resource_lengths.begin(), resource_lengths.end()};
                auto output_tensor{kernel_context.GetOutput(i, shapes)};
                void* output_data{output_tensor.GetTensorMutableRawData()};
                HIP_CALL_THROW(hipMemcpyWithStream(output_data, gpu_resource.data(), resource_shape.bytes(),
                    hipMemcpyDeviceToDevice, hip_stream));
            }
        }
        // Async by default: the stream-ordered D2H fetch plus the per-Run
        // OnSessionRunEnd sync order the read. Legacy full-drain is gated (see
        // FinishComputeStream / env_var::kLegacyComputeSync).
        RETURN_IF_ERROR(FinishComputeStream(hip_stream));
    }
    return STATUS_OK;
} catch (const Ort::Exception& e) {
    return Ort::Status{e};
} catch (const std::exception& e) {
    return Ort::Status{e.what(), ORT_EP_FAIL};
}

}  // namespace mgx_ep

// Telemetry collector exported for the amdgpu wrapper. The wrapper resolves this
// by symbol (see telemetry::kGetBackendDataSymbol) and calls it with a live
// backend EP to pull MIGraphX-specific telemetry.
extern "C" bool GetBackendTelemetry(const OrtEp* ep, telemetry::BackendData* out) noexcept {
    if (ep == nullptr || out == nullptr) {
        return false;
    }
    static_cast<const mgx_ep::ExecutionProvider*>(ep)->CollectTelemetry(*out);
    return true;
}
