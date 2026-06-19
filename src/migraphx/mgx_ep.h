// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <ciso646>
#include <set>
#include <mutex>

#include <hip/hip_runtime_api.h>
#include <migraphx/migraphx.hpp>

#include "common/path_string.h"
#include "common/plugin_ep_utils.h"

#include "mgx_factory.h"
#include "mgx_info.h"
#include "mgx_utils.h"

namespace mgx_ep {

namespace env_var {
constexpr auto kFP16Enable = "ORT_MIGRAPHX_FP16_ENABLE"sv;
constexpr auto kBF16Enable = "ORT_MIGRAPHX_BF16_ENABLE"sv;
constexpr auto kFP8Enable = "ORT_MIGRAPHX_FP8_ENABLE"sv;
constexpr auto kINT8Enable = "ORT_MIGRAPHX_INT8_ENABLE"sv;
constexpr auto kDisableCompiledModelCaching = "ORT_MIGRAPHX_DISABLE_COMPILED_MODEL_CACHING"sv;
constexpr auto kForceRecompile = "ORT_MIGRAPHX_FORCE_RECOMPILE"sv;
constexpr auto kDumpSubgraphs = "ORT_MIGRAPHX_DUMP_SUBGRAPHS"sv;
constexpr auto kDumpEpContextModel = "ORT_MIGRAPHX_DUMP_EP_CONTEXT_MODEL"sv;
constexpr auto kINT8CalibrationTableName = "ORT_MIGRAPHX_INT8_CALIBRATION_TABLE_NAME"sv;
constexpr auto kCacheDir = "ORT_MIGRAPHX_CACHE_DIR"sv;
constexpr auto kComputeMode = "ORT_MIGRAPHX_COMPUTE_MODE"sv;
constexpr auto kINT8UseNativeCalibrationTable = "ORT_MIGRAPHX_INT8_USE_NATIVE_CALIBRATION_TABLE"sv;
constexpr auto kExhaustiveTune = "ORT_MIGRAPHX_EXHAUSTIVE_TUNE"sv;
constexpr auto kMlssUseSpecificOps = "ORT_MIGRAPHX_MLSS_USE_SPECIFIC_OPS"sv;
}  // namespace env_vars

struct ComputeState {
    std::mutex& mutex;
    int device_id;
    const migraphx::target& t;
    migraphx::onnx_options onnx_options;
    migraphx::program program;
    bool enable_fp16{};
    bool enable_bf16{};
    bool enable_fp8{};
    bool enable_int8{};
    bool int8_calibration_cache_available{};
    bool has_input_shapes{};
    bool dump_subgraphs_{};
    bool exhaustive_tune{};
    const Map<float>& dynamic_ranges;
    Map<size_t> input_name_indices;
    Map<size_t> output_name_indices;
    std::string onnx_string;
    ComputeMode compute_mode{ComputeMode::Balanced};
    fs::path model_path;
    fs::path cache_dir;
    bool disable_compiled_model_caching{};
    bool force_recompile{};
    fs::path external_data_dir;
    std::string mxr_prefix;
};

struct EpContextComputeState {
    std::mutex& mutex;
    int device_id;
    const migraphx::target& t;
    migraphx::program program;
    Map<size_t> input_name_indices;
    Map<size_t> output_name_indices;
};

struct ExecutionProvider : OrtEp, ApiPtrs {
    ExecutionProvider(const ProviderFactory& api_ptrs, std::string_view ep_name,
        Ort::ConstSessionOptions session_options, const Ort::Logger& logger);

    ComputeState& GetComputeState(const std::string& fused_node_name) {
        const auto it{compute_states_.find(fused_node_name)};
        if (it == compute_states_.end()) {
            throw std::runtime_error{"unknown compute state for the fused node '"
                + fused_node_name + "'"};
        }
        return it->second;
    }
    EpContextComputeState& EpContext_GetComputeState(const std::string& fused_node_name) {
        const auto it{ep_context_compute_states_.find(fused_node_name)};
        if (it == ep_context_compute_states_.end()) {
            throw std::runtime_error{"unknown EPContext compute state for the fused node '"
                + fused_node_name + "'"};
        }
        return it->second;
    }

private:
    [[nodiscard]] const char* GetName() const noexcept;

    Ort::Status GetCapability(const Ort::ConstGraph& graph,
        OrtEpGraphSupportInfo* graph_support_info) const noexcept;

    Ort::Status Compile(const std::vector<Ort::ConstGraph>& graphs,
        const std::vector<Ort::ConstNode>& fused_nodes,
        gsl::span<OrtNodeComputeInfo*> node_compute_info,
        gsl::span<OrtNode*> ep_context_nodes) noexcept;

    Ort::Status ReleaseNodeComputeInfos(
        gsl::span<OrtNodeComputeInfo*> node_compute_info) noexcept;

    // Ort::Status SetDynamicOptions(const char* const* option_keys, const char* const* option_values, size_t num_options);
    // Ort::Status OnRunStart(const OrtRunOptions* run_options);

    Ort::Status OnRunEnd(const OrtRunOptions* run_options, bool sync_stream) noexcept;
    Ort::Status CreateSyncStreamForDevice(const OrtMemoryDevice* memory_device, OrtSyncStreamImpl** stream);
    // const char* GetCompiledModelCompatibilityInfo(const OrtGraph* graph) const;
    Ort::Status GetKernelRegistry(const OrtKernelRegistry** kernel_registry) const;

    Ort::Status CreateNodeComputeInfoFromGraph(const Ort::ConstGraph& graph, const Ort::ConstNode& fused_node,
        const Map<size_t>& input_name_indices, const Map<size_t>& output_name_indices, const std::string& mxr_prefix,
        OrtNodeComputeInfo*& node_compute_info, OrtNode*& ep_context_node, bool& loaded_from_cache);

    void LogTelemetry(const fs::path& model_path, bool loaded_from_cache) const noexcept;

    Ort::Status CreateNodeComputeInfoFromCache(const Ort::ConstGraph& graph, const Ort::ConstNode& fused_node,
        const Map<size_t>& input_name_indices, const Map<size_t>& output_name_indices,
        OrtNodeComputeInfo*& node_compute_info);

    const ProviderFactory& factory_;

    const Ort::Logger logger_;
    ComputeMode compute_mode_{ComputeMode::Balanced};

    migraphx::target t_{"gpu"};

    Map<float> dynamic_ranges_;
    Map<EpContextComputeState> ep_context_compute_states_;
    Map<ComputeState> compute_states_;

    hipStream_t stream_{};
    hipDeviceProp_t device_prop_{};

    int device_id_{};
    std::string ep_name_{};
    bool disable_compiled_model_caching_{};
    bool force_recompile_{};
    bool enable_fp16_{};
    bool enable_bf16_{};
    bool enable_fp8_{};
    bool enable_int8_{};
    bool exhaustive_tune_{};
    std::string mlss_use_specific_ops_{};
    bool int8_calibration_cache_available_{};
    bool int8_use_native_calibration_table_{};
    bool dump_subgraphs_{};
    fs::path cache_dir_{};
    std::string int8_calibration_table_name_{};
    fs::path external_data_dir_{};
    std::string compute_capability_{};
    bool context_embed_mode_{};
    bool context_enable_{};
    std::string context_node_name_prefix_{};
    fs::path context_file_path_{};
    fs::path external_initializers_file_name_{};

    std::mutex mutex_{};
};

struct NodeComputeInfo : OrtNodeComputeInfo {
    explicit NodeComputeInfo(ExecutionProvider& ep)
        : OrtNodeComputeInfo{ORT_API_VERSION}, ep_{ep}
    {
        OrtNodeComputeInfo::CreateState = [](OrtNodeComputeInfo* this_,
            OrtNodeComputeContext* compute_context, void** compute_state) noexcept {
            API_CALL_S(NodeComputeInfo, this_, CreateState, compute_context, *compute_state);
        };
        OrtNodeComputeInfo::Compute = [](OrtNodeComputeInfo* this_,
            void* compute_state, OrtKernelContext* kernel_context) noexcept {
            API_CALL_S(NodeComputeInfo, this_, Compute, *static_cast<ComputeState*>(compute_state),
                Ort::KernelContext{kernel_context});
        };
        OrtNodeComputeInfo::ReleaseState = [](OrtNodeComputeInfo* this_, void* compute_state) noexcept {
            API_CALL_V(NodeComputeInfo, this_, ReleaseState, compute_state);
        };
    }

private:
    ExecutionProvider& ep_;

    Ort::Status CreateState(OrtNodeComputeContext* compute_context, void*& compute_state) noexcept;
    Ort::Status Compute(ComputeState& compute_state, const Ort::KernelContext& kernel_context) noexcept;
    void ReleaseState([[maybe_unused]] void* compute_state) noexcept {}
};

struct EpContextNodeComputeInfo : OrtNodeComputeInfo {
    explicit EpContextNodeComputeInfo(ExecutionProvider& ep)
        : OrtNodeComputeInfo{ORT_API_VERSION}, ep_{ep}
    {
        OrtNodeComputeInfo::CreateState = [](OrtNodeComputeInfo* this_,
            OrtNodeComputeContext* compute_context, void** compute_state) noexcept {
            API_CALL_S(EpContextNodeComputeInfo, this_, CreateState, compute_context, *compute_state);
        };
        OrtNodeComputeInfo::Compute = [](OrtNodeComputeInfo* this_,
            void* compute_state, OrtKernelContext* kernel_context) noexcept {
            API_CALL_S(EpContextNodeComputeInfo, this_, Compute, *static_cast<EpContextComputeState*>(compute_state),
                Ort::KernelContext{kernel_context});
        };
        OrtNodeComputeInfo::ReleaseState = [](OrtNodeComputeInfo* this_, void* compute_state) noexcept {
            API_CALL_V(EpContextNodeComputeInfo, this_, ReleaseState, compute_state);
        };
    }

private:
    ExecutionProvider& ep_;

    Ort::Status CreateState(OrtNodeComputeContext* compute_context, void*& compute_state) noexcept;
    Ort::Status Compute(EpContextComputeState& compute_state, const Ort::KernelContext& kernel_context) noexcept;
    void ReleaseState([[maybe_unused]] void* compute_state) noexcept {}
};

}  // namespace mgx_ep
