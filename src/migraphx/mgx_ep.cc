// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <charconv>
#include <set>
#include <string>
#include <string_view>

#include <gsl/span>

#include <migraphx/migraphx.hpp>
#include <migraphx/version.h>

#include "common/constants.h"
#include "common/enumerate.h"
#include "common/env_var.h"
#include "common/murmurhash3.h"
#include "common/ort_graph_to_proto.h"

#include "hip/stream_support.h"

#include "mgx_dynamic_batch.h"
#include "mgx_ep.h"
#include "mgx_ep_ctx.h"
#include "mgx_info.h"
#include "mgx_utils.h"

namespace mgx_ep {

const char* ExecutionProvider::GetName() const noexcept {
    return ep_name_.c_str();
}

namespace {

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
            if (!IsInt8Tensor(input) || !IsUint8Tensor(input)) {
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
            if (!IsInt8Tensor(input) || !IsUint8Tensor(input)) {
                return true;
            }
        }
    } else if (op_type == "NonZero") {
        if (!CanEvalNodeArgument(graph, node, {0})) {
            return true;
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

bool AllNodesAssignedToEp(const Ort::ConstGraph& graph, std::string_view ep_name) {
    const auto nodes{graph.GetNodes()};
    return !nodes.empty() && ranges::all_of(nodes,
        [&ep_name](const Ort::ConstNode& node) { return node.GetName() == ep_name; });
}

bool IsNodeControlFlowOp(const Ort::ConstNode& node) {
    const auto op_type{node.GetOperatorType()};
    return op_type == "If" || op_type == "Loop" || op_type == "Scan";
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

}  // namespace

#define PARSE_ENV_VAR(__name__, __value__)                                               \
    const auto __value__##env{ParseEnvironmentVariable<decltype(__value__)>(__name__)};  \
    if (__value__##env.has_value()) {                                                    \
        __value__ = __value__##env.value();                                              \
    }

ExecutionProvider::ExecutionProvider(const ProviderFactory& factory, std::string_view ep_name, Ort::ConstSessionOptions session_options, const Ort::Logger& logger)
    : OrtEp{ORT_API_VERSION}, ApiPtrs{factory.ort_api, factory.ep_api, factory.model_editor_api}, factory_{factory}, logger_{logger}, ep_name_{ep_name}
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
        }
    }

    const ProviderInfo info{provider_options};

    device_id_ = info.device_id;
    enable_fp16_ = info.enable_fp16;
    enable_bf16_ = info.enable_bf16;
    enable_fp8_ = info.enable_fp8;
    enable_int8_ = info.enable_int8;
    exhaustive_tune_ = info.exhaustive_tune;
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
        for (auto& [param_name, buf] : cs.staging_inputs) {
            if (buf.data != nullptr) {
                (void)hipFree(buf.data);
                buf.data = nullptr;
            }
        }
        for (auto& [param_name, buf] : cs.staging_outputs) {
            if (buf.data != nullptr) {
                (void)hipFree(buf.data);
                buf.data = nullptr;
            }
        }
        for (auto& [hash, buf] : cs.scratch_bufs) {
            if (buf.data != nullptr) {
                (void)hipFree(buf.data);
                buf.data = nullptr;
            }
        }
    }
}

Ort::Status ExecutionProvider::GetCapability(const Ort::ConstGraph& graph,
                                             OrtEpGraphSupportInfo* graph_support_info) const noexcept
try {
    if (graph.GetNodes().empty()) {
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
            OrtNodeFusionOptions node_fusion_options{ORT_API_VERSION, true};
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
    for (const auto& node : sorted_nodes) {
        if (IsNodeControlFlowOp(node)) {
            auto supported_control_flow = [this](const Ort::ConstNode& node) {
                return ranges::all_of(node.GetSubgraphs(),
                    [this](const Ort::AttrNameSubgraph& attr) {
                        return AllNodesAssignedToEp(attr.sub_graph, ep_name_);
                    });
            };
            if (!supported_control_flow(node)) {
                return STATUS_OK;
            }
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
        OrtNodeFusionOptions node_fusion_options{ORT_API_VERSION, true};
        RETURN_IF_STATUS(ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info,
            reinterpret_cast<const OrtNode* const*>(supported_nodes.data()), supported_nodes.size(),
            &node_fusion_options));
    } else {
        const auto subgraphs{GetPartitionedSubgraphs(sorted_nodes, unsupported_nodes)};
        /* TODO: log unsupported nodes */
        for (const auto& subgraph : subgraphs) {
            if (subgraph.size() == 1) {
                RETURN_IF_STATUS(ep_api.EpGraphSupportInfo_AddSingleNode(graph_support_info,
                    subgraph.front()));
            } else {
                OrtNodeFusionOptions node_fusion_options{ORT_API_VERSION, true};
                RETURN_IF_STATUS(ep_api.EpGraphSupportInfo_AddNodesToFuse(graph_support_info,
                    reinterpret_cast<const OrtNode* const*>(subgraph.data()), subgraph.size(),
                    &node_fusion_options));
            }
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

bool load_compiled_program(migraphx::program& prog, const fs::path& path)
try {
    if (!path.empty() && exists(path)) {
        prog = migraphx::load(path.string().c_str());
        return true;
    }
    return false;
} catch (const std::exception&) {
    return false;
}

void save_compiled_program(const migraphx::program& prog, const fs::path& path) {
    if (!path.empty()) {
        migraphx::file_options options;
        options.set_file_format("msgpack");
        migraphx::save(prog, path.string().c_str(), options);
    }
}

void calibrate_and_quantize(const migraphx::program& prog, const migraphx::target& target,
    const migraphx::program_parameters& params,
    bool fp16_enable, bool bf16_enable, bool int8_enable, bool fp8_enable, bool int8_calibration_cache_available,
    const std::unordered_map<std::string, float>& dynamic_range_map)
{
    if ((int8_enable ^ fp8_enable) && int8_calibration_cache_available) {
        const auto param_shapes{prog.get_parameter_shapes()};
        for (const auto& [key, value] : dynamic_range_map) {
            const auto shape{migraphx::shape(migraphx_shape_float_type)};
            params.add(key.c_str(), migraphx::argument(shape, const_cast<float*>(&value)));
        }
        if (int8_enable) {
            migraphx::quantize_int8_options options;
            options.add_calibration_data(params);
            options.add_op_name("convolution");
            options.add_op_name("dot");
            migraphx::quantize_int8(prog, target, options);
        } else if (fp8_enable) {
            migraphx::quantize_fp8_options options;
            options.add_calibration_data(params);
            migraphx::quantize_fp8(prog, target, options);
        }
    }
    if (fp16_enable) {
        migraphx::quantize_fp16(prog);
    }
    if (bf16_enable) {
        migraphx::quantize_bf16(prog);
    }
}

void compile_program(const migraphx::program& prog, const migraphx::target& target, bool exhaustive_tune) {
    migraphx::compile_options options;
    options.set_fast_math(false);
    options.set_exhaustive_tune_flag(exhaustive_tune);
    prog.compile(target, options);
}

}  // namespace <anonymous>

Ort::Status ExecutionProvider::CreateNodeComputeInfoFromGraph(const Ort::ConstGraph& graph,
    const Ort::ConstNode& fused_node, const Map<size_t>& input_name_indices, const Map<size_t>& output_name_indices,
    const std::string& mxr_prefix, OrtNodeComputeInfo*& node_compute_info, OrtNode*& ep_context_node)
{
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

    migraphx::program program;
    migraphx::onnx_options onnx_options;

    if (has_input_shape) {
        hash::Value input_shapes_hash{};
        for (const auto& input : GetValueInfos(sorted_graph.GetInputs())) {
            const auto shape{input.TypeInfo().GetTensorTypeAndShapeInfo().GetShape()};
            hash::Hash(input_shapes_hash, shape);
        }
        fs::path mxr_path;
        if (!cache_dir_.empty()) {
            mxr_path = cache_dir_ / (mxr_prefix + hash::ToHex(input_shapes_hash) + ".mxr");
        }
        if (force_recompile_ || !load_compiled_program(program, mxr_path)) {
            const auto external_data_dir{external_data_dir_.empty() ?
                model_path.parent_path() : external_data_dir_};
            onnx_options.set_external_data_path(external_data_dir.string());
            program = migraphx::parse_onnx_buffer(onnx_string, onnx_options);
            migraphx::program_parameters params;
            calibrate_and_quantize(program, t_, params, enable_fp16_, enable_bf16_, enable_int8_,
                enable_fp8_, int8_calibration_cache_available_, dynamic_ranges_);
            compile_program(program, t_, exhaustive_tune_);
            if (!disable_compiled_model_caching_) {
                save_compiled_program(program, mxr_path);
            }
        }
        const auto output_shapes{program.get_output_shapes()};
        for (const auto& [n, s] : zip(output_names, output_shapes)) {
            onnx_options.set_input_parameter_shape(n, s.lengths());
        }
    }

    if (context_enable_) {
        fs::path ep_context_mxr_path;
        if (has_input_shape && !cache_dir_.empty()) {
            hash::Value input_shapes_hash{};
            for (const auto& input : GetValueInfos(sorted_graph.GetInputs())) {
                const auto shape{input.TypeInfo().GetTensorTypeAndShapeInfo().GetShape()};
                hash::Hash(input_shapes_hash, shape);
            }
            ep_context_mxr_path = mxr_prefix + hash::ToHex(input_shapes_hash) + ".mxr";
        } else {
            const auto model_stem{model_path.stem().string()};
            ep_context_mxr_path = model_stem + "_migraphx.mxr";
        }

        EpContextNodeHelper ep_context_helper{*this, sorted_graph, fused_node};
        RETURN_IF_ERROR(ep_context_helper.CreateEpContextNode(ep_context_mxr_path, cache_dir_,
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
            dynamic_ranges_,
            input_name_indices,
            output_name_indices,
            onnx_string,
            compute_mode_,
            model_path,
            cache_dir_,
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
    for (const auto& [graph, fused_node, node_compute_info, ep_context_node] : zip(graphs, fused_nodes, node_compute_infos, ep_context_nodes)) {
        const auto inputs{GetValueInfos(fused_node.GetInputs())};
        std::unordered_map<std::string, size_t> input_name_indices;
        input_name_indices.reserve(inputs.size());
        const auto initializers{graph.GetInitializers()};
        for (const auto& [i, input] : enumerate(inputs)) {
            if (!input.IsConstantInitializer()) {
                input_name_indices.emplace(input.GetName(), i);
            }
        }
        const auto outputs{GetValueInfos(fused_node.GetOutputs())};
        std::unordered_map<std::string, size_t> output_name_indices;
        output_name_indices.reserve(outputs.size());
        for (const auto& [i, output] : enumerate(outputs)) {
            output_name_indices.emplace(output.GetName(), i);
        }

        if (EpContextNodeReader::GraphHasContextNode(graph)) {
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
    if (const auto status{hipStreamQuery(stream_)}; status != hipSuccess) {
        HIP_RETURN_IF_ERROR(hipStreamSynchronize(stream_));
    }
    return STATUS_OK;
} catch (const Ort::Exception& e) {
    return Ort::Status{e};
} catch (const std::exception& e) {
    return Ort::Status{e.what(), ORT_EP_FAIL};
}

Ort::Status ExecutionProvider::CreateSyncStreamForDevice(const OrtMemoryDevice* memory_device, OrtSyncStreamImpl** stream)
try {
    *stream = std::make_unique<hip::SyncStream>(*this, device_id_, nullptr).release();
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

Ort::Status NodeComputeInfo::Compute(ComputeState& compute_state, const Ort::KernelContext& kernel_context) noexcept
    try {
    const auto& input_name_indices{compute_state.input_name_indices};
    auto& program{compute_state.program};
    auto& onnx_options{compute_state.onnx_options};

    bool input_shapes_match{compute_state.has_input_shapes};
    migraphx::program_parameter_shapes param_shapes;
    hash::Value input_shapes_hash{};

    if (!compute_state.has_input_shapes) {
        for (auto& [name, index] : input_name_indices) {
            auto value{kernel_context.GetInput(index)};
            auto shape{value.GetTensorTypeAndShapeInfo().GetShape()};
            onnx_options.set_input_parameter_shape(name, {shape.begin(), shape.end()});
            hash::Hash(input_shapes_hash, shape);
        }
        compute_state.has_input_shapes = true;
    } else {
        param_shapes = program.get_parameter_shapes();

        // check whether input shapes match with shapes of program inputs
        if (param_shapes.size() > 0) {
            for (const auto& name : param_shapes.names()) {
                if (input_name_indices.count(name) > 0) {
                    auto value{kernel_context.GetInput(input_name_indices.at(name))};
                    auto shape{value.GetTensorTypeAndShapeInfo().GetShape()};
                    auto prog_shape{param_shapes[name]};
                    auto prog_lengths{prog_shape.lengths()};
                    auto prog_strides{prog_shape.strides()};
                    if (prog_lengths.size() == 1 && prog_lengths.front() == 1 &&
                        prog_strides.size() == 1 && prog_strides.front() == 0) {
                        prog_lengths.clear();
                    }
                    std::vector<size_t> lengths{shape.begin(), shape.end()};
                    if (prog_lengths != lengths) {
                        onnx_options.set_input_parameter_shape(name, lengths);
                        input_shapes_match = false;
                    }
                }
            }
        }

        // Compute hash over all model inputs, consistent with the first-call path.
        // The shape comparison above iterates param_shapes.names() which may be a
        // subset of input_name_indices if MIGraphX optimized away parameters during
        // compilation. Using input_name_indices ensures the cache key is identical
        // for identical input shapes regardless of which program is currently active.
        for (const auto& [name, index] : input_name_indices) {
            auto value{kernel_context.GetInput(index)};
            auto shape{value.GetTensorTypeAndShapeInfo().GetShape()};
            hash::Hash(input_shapes_hash, shape);
        }
    }

    // If the input shapes are different (e.g., LLMs), the EP needs to reparse and recompile the program
    if (!input_shapes_match) {
        migraphx::program_parameters compile_params{};
        fs::path mxr_path;
        if (!compute_state.cache_dir.empty()) {
            mxr_path = compute_state.cache_dir / (compute_state.mxr_prefix + hash::ToHex(input_shapes_hash) + ".mxr");
        }
        if (compute_state.force_recompile || !load_compiled_program(program, mxr_path)) {
            const auto external_data_dir{compute_state.external_data_dir.empty() ?
                compute_state.model_path.parent_path() : compute_state.external_data_dir};
            onnx_options.set_external_data_path(external_data_dir.string());
            program = migraphx::parse_onnx_buffer(compute_state.onnx_string, onnx_options);
            param_shapes = program.get_parameter_shapes();
            if ((compute_state.enable_int8 ^ compute_state.enable_fp8) && compute_state.int8_calibration_cache_available) {
                for (const auto& name : param_shapes.names()) {
                    if (input_name_indices.count(name) > 0) {
                        const auto index{input_name_indices.at(name)};
                        auto value{kernel_context.GetInput(index)};
                        auto tensor_info{value.GetTensorTypeAndShapeInfo()};
                        const auto tensor_shape{tensor_info.GetShape()};
                        const auto tensor_type{tensor_info.GetElementType()};

                        migraphx_shape_datatype_t datatype;
                        GetMIGraphXType(tensor_type, datatype);

                        if (const auto prog_shapes{param_shapes[name]}; datatype != prog_shapes.type()) {
                            throw std::runtime_error{"NodeComputeInfo::Compute(): input parameter type mismatch"};
                        }

                        compile_params.add(name, migraphx::argument{param_shapes[name],
                            const_cast<void*>(value.GetTensorRawData())});
                    }
                }
            }
            calibrate_and_quantize(program, compute_state.t, compile_params,
                compute_state.enable_fp16, compute_state.enable_bf16, compute_state.enable_int8,
                compute_state.enable_fp8, compute_state.int8_calibration_cache_available, compute_state.dynamic_ranges);

            compile_program(program, compute_state.t, compute_state.exhaustive_tune);
            if (!compute_state.disable_compiled_model_caching) {
                save_compiled_program(program, mxr_path);
            }
        }
        param_shapes = program.get_parameter_shapes();
    }

    migraphx::program_parameters compute_params;
    auto output_shapes{program.get_output_shapes()};
    std::vector<size_t> output_indices;
    if (param_shapes.size() > 0) {
        for (std::string name : param_shapes.names()) {
            if (input_name_indices.count(name) > 0) {
                const auto index{input_name_indices.at(name)};
                auto input_tensor{kernel_context.GetInput(index)};
                auto tensor_info{input_tensor.GetTensorTypeAndShapeInfo()};
                auto tensor_shape{tensor_info.GetShape()};
                auto tensor_type{tensor_info.GetElementType()};

                migraphx_shape_datatype_t datatype;
                GetMIGraphXType(tensor_type, datatype);

                if (auto prog_shape{param_shapes[name.c_str()]}; datatype != prog_shape.type()) {
                    throw std::runtime_error{"NodeComputeInfo::Compute(): tensor parameter type mismatch"};
                }
                compute_params.add(name.c_str(), migraphx::argument{param_shapes[name.c_str()],
                    const_cast<void*>(input_tensor.GetTensorRawData())});
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
                }
            }
        }
    }
    {
        std::lock_guard lock{compute_state.mutex};

        HIP_RETURN_IF_ERROR(hipSetDevice(compute_state.device_id));
        auto hip_stream{static_cast<hipStream_t>(kernel_context.GetGPUComputeStream())};
        auto prog_outputs{program.run_async(compute_params, hip_stream)};
        HIP_RETURN_IF_ERROR(hipStreamSynchronize(hip_stream));

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
            HIP_RETURN_IF_ERROR(hipStreamSynchronize(hip_stream));
        }
    }
    return STATUS_OK;
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
                }
            }
        }
    }
    {
        std::lock_guard lock{compute_state.mutex};

        HIP_RETURN_IF_ERROR(hipSetDevice(compute_state.device_id));
        auto hip_stream{static_cast<hipStream_t>(kernel_context.GetGPUComputeStream())};
        auto prog_outputs{program.run_async(compute_params, hip_stream)};
        HIP_RETURN_IF_ERROR(hipStreamSynchronize(hip_stream));

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
            HIP_RETURN_IF_ERROR(hipStreamSynchronize(hip_stream));
        }
    }
    return STATUS_OK;
} catch (const Ort::Exception& e) {
    return Ort::Status{e};
} catch (const std::exception& e) {
    return Ort::Status{e.what(), ORT_EP_FAIL};
}

}  // namespace mgx_ep
