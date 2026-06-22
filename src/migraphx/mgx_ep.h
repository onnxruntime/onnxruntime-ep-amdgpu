// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <ciso646>
#include <cstdint>
#include <set>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

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
constexpr auto kHipGraphEnable = "ORT_MIGRAPHX_HIP_GRAPH_ENABLE"sv;
constexpr auto kMaxDynamicBatch = "ORT_MIGRAPHX_MAX_DYNAMIC_BATCH"sv;
constexpr auto kCompileBatches = "ORT_MIGRAPHX_COMPILE_BATCHES"sv;
constexpr auto kCoalesceIO = "ORT_MIGRAPHX_COALESCE_IO"sv;
constexpr auto kMlssUseSpecificOps = "ORT_MIGRAPHX_MLSS_USE_SPECIFIC_OPS"sv;
}  // namespace env_vars

// EP-owned device staging buffer (pointer-stable across runs so it can be
// safely baked into a captured hipGraph).  Plain hipMalloc/hipFree owned;
// freed centrally in ~ExecutionProvider.
struct StagingBuffer {
    void* data{nullptr};
    std::size_t size_bytes{};
    migraphx::shape shape{};
    // When the input arena is active (coalesce_io), `data` is a sub-view into the
    // shared device arena rather than an independent allocation, so it must not be
    // freed individually.  `arena_offset` is its byte offset within the arena.
    std::size_t arena_offset{};
    bool is_arena_view{};
};

// EP-owned scratch buffer bound to a MIGraphX program's "scratch" parameter.
// Owning it (instead of letting MIGraphX use its internal arena) lets us zero
// it before every capture/replay so kernels start from a deterministic
// baseline.  One per compiled program variant (keyed by shape hash).
struct ScratchBuffer {
    void* data{nullptr};
    std::size_t size_bytes{};
    migraphx::shape shape{};
};

// A captured hipGraph plus the metadata needed to replay it correctly.
struct CapturedHipGraph {
    hipGraph_t graph{nullptr};
    hipGraphExec_t exec{nullptr};
    bool captured{false};

    // MIGraphX outputs not bound to a pre-allocated buffer; their device data
    // is stable across replays and is copied out after each launch.
    struct ExtraOutput {
        std::size_t output_index{};
        std::vector<std::int64_t> ort_shape{};
        void* gpu_data{nullptr};
        std::size_t bytes{};
    };
    std::vector<ExtraOutput> extra_outputs{};

    // Scratch pointer baked into the captured kernels; a mismatch forces re-capture.
    void* captured_scratch_ptr{nullptr};

    // Output buffers (ptr + bytes) that must be zeroed before every replay
    // because some captured kernels read-modify-write their output.
    std::vector<std::pair<void*, std::size_t>> captured_output_zeroes{};
};

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

    // ── Configuration (set at Compile time) ──────────────────────────────────
    bool hip_graph_enable{};
    std::size_t max_dynamic_batch{};
    std::string compile_batches{};

    // ── Dynamic-batch runtime state ──────────────────────────────────────────
    bool has_dynamic_batch{};
    bool defer_compilation{};
    std::vector<std::size_t> compiled_batch_sizes{};
    // Compiled program variants keyed by shape/batch hash.
    Map<migraphx::program> cached_programs{};

    // ── Coalesced input arena (ORT_MIGRAPHX_COALESCE_IO) ─────────────────────
    // When coalesce_io is set, every input staging buffer's data points into a
    // single device arena (in_arena_dev) fed by one pinned host staging buffer
    // (in_staging_host); copying gathers all inputs host-side then issues one H2D.
    bool coalesce_io{};
    bool staging_inputs_coalesced{};
    void* in_arena_dev{nullptr};
    void* in_staging_host{nullptr};
    std::size_t in_arena_bytes{};

    // ── hipGraph / staging / scratch runtime state (owned device memory) ──────
    // Staging buffers keyed by MIGraphX program parameter name.
    Map<StagingBuffer> staging_inputs{};
    Map<StagingBuffer> staging_outputs{};
    bool staging_allocated{};
    // Scratch buffers keyed by shape hash.
    Map<ScratchBuffer> scratch_bufs{};
    // Captured graphs keyed by shape hash.
    Map<CapturedHipGraph> hip_graph_cache{};
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

    ~ExecutionProvider();

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
        OrtNodeComputeInfo*& node_compute_info, OrtNode*& ep_context_node);

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
    bool hip_graph_enable_{};
    std::size_t max_dynamic_batch_{};
    std::string compile_batches_{};
    bool coalesce_io_enable_{};

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
