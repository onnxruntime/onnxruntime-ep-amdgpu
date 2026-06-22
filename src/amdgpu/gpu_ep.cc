// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gpu_info.h"
#include "gpu_ep.h"

#include "gpu_options.h"
#include "mgx_options.h"

#define EP_CALL_T(backend, fn, defval, ...)                            \
    do {                                                               \
        return (backend != nullptr &&                                  \
                backend->fn != nullptr) ?                              \
                    backend->fn(backend, __VA_ARGS__) : defval;        \
    } while (0)

#define EP_CALL_S(backend, fn, ...)                                    \
    do {                                                               \
        if (backend == nullptr) {                                      \
            return MAKE_STATUS(ORT_EP_FAIL, #fn ": invalid backend");  \
        }                                                              \
        if (backend->fn != nullptr) {                                  \
            RETURN_IF_ERROR(backend->fn(backend, __VA_ARGS__));        \
        }                                                              \
        return STATUS_OK;                                              \
    } while (0)

#define EP_CALL_V(backend, fn, ...)                                    \
    do {                                                               \
        if (backend != nullptr && backend->fn != nullptr) {            \
            backend->fn(backend, __VA_ARGS__);                         \
        }                                                              \
    } while (0)


namespace gpu_ep {

ExecutionProvider::ExecutionProvider(ProviderFactory& factory, std::string_view ep_name,
        const Ort::ConstSessionOptions& session_options, const OrtLogger* logger)
    : OrtEp{ORT_API_VERSION},
      ApiPtrs{factory.ort_api, factory.ep_api, factory.model_editor_api},
      factory_{factory}, ep_name_{ep_name}, logger_{logger}
{
    OrtEp::GetName = [](const OrtEp* this_) noexcept {
        API_CALL_T(const ExecutionProvider, this_, GetName, "");
    };
    OrtEp::GetCapability = [](OrtEp* this_, const OrtGraph* graph,
                              OrtEpGraphSupportInfo* graph_support_info) noexcept {
        API_CALL_S(ExecutionProvider, this_, GetCapability, graph, graph_support_info);
    };
    OrtEp::Compile = [](OrtEp* this_, const OrtGraph** graphs, const OrtNode** fused_nodes,
                        size_t count, OrtNodeComputeInfo** node_compute_infos, OrtNode** ep_context_nodes) noexcept {
        API_CALL_S(ExecutionProvider, this_, Compile, graphs, fused_nodes, count,
            node_compute_infos, ep_context_nodes);
    };
    OrtEp::ReleaseNodeComputeInfos = [](OrtEp* this_, OrtNodeComputeInfo** node_compute_infos,
                                        size_t num_node_compute_info) noexcept {
        API_CALL_V(ExecutionProvider, this_, ReleaseNodeComputeInfos, node_compute_infos, num_node_compute_info);
    };
    // TODO: OrtEp::GetPreferredDataLayout = []
    // TODO: OrtEp::ShouldConvertDataLayoutForOp = []
    OrtEp::SetDynamicOptions = [](OrtEp* this_, const char* const* option_keys, const char* const* option_values,
            size_t num_options) noexcept {
        API_CALL_S(ExecutionProvider, this_, SetDynamicOptions, option_keys, option_values, num_options);
    };
    OrtEp::OnRunStart = [](OrtEp* this_, const OrtRunOptions* run_options) noexcept {
        API_CALL_S(ExecutionProvider, this_, OnRunStart, run_options);
    };
    OrtEp::OnRunEnd = [](OrtEp* this_, const OrtRunOptions* run_options, bool sync_stream) noexcept {
        API_CALL_S(ExecutionProvider, this_, OnRunEnd, run_options, sync_stream);
    };
    OrtEp::CreateSyncStreamForDevice = [](OrtEp* this_, const OrtMemoryDevice* memory_device,
                                          OrtSyncStreamImpl** stream) noexcept {
        API_CALL_S(ExecutionProvider, this_, CreateSyncStreamForDevice, memory_device, stream);
    };
    OrtEp::GetCompiledModelCompatibilityInfo = [](OrtEp* this_, const OrtGraph* graph) noexcept {
        API_CALL_T(ExecutionProvider, this_, GetCompiledModelCompatibilityInfo, "", graph);
    };
    OrtEp::GetKernelRegistry = [](OrtEp* this_, const OrtKernelRegistry** kernel_registry) noexcept {
        API_CALL_S(ExecutionProvider, this_, GetKernelRegistry, kernel_registry);
    };
    // TODO: OrtEp::IsConcurrentRunSupported = []

    std::string lowercase{ep_name_};
    std::transform(lowercase.begin(), lowercase.end(), lowercase.begin(), ::tolower);

    OrtKeyValuePairs* ort_key_value_pairs;
    THROW_IF_ERROR(ort_api.GetSessionOptionsConfigEntries(session_options, &ort_key_value_pairs));

    const Ort::KeyValuePairs key_value_pairs{ort_key_value_pairs};
    const std::string ep_prefix{"ep." + lowercase + "."};

    OrtSessionOptions* local_session_options{};
    THROW_IF_ERROR(ort_api.CreateSessionOptions(&local_session_options));

    ProviderOptions provider_options;
    for (const auto& [key, value] : key_value_pairs.GetKeyValuePairs()) {
        if (key.rfind(ep_prefix, 0) == 0) {
            provider_options.emplace(key.substr(ep_prefix.length()), value);
        } else {
            THROW_IF_ERROR(ort_api.AddSessionConfigEntry(local_session_options, key.c_str(), value.c_str()));
        }
    }

    const ProviderInfo info{provider_options};

    const auto create_directml_backend = [&] {
        THROW_IF_ERROR(factory.CreateDirectMLBackend(local_session_options, logger, backend_ep_));
        // DirectML manages its own per-session GPU allocator (DmlBucketizedBufferAllocator)
        // via EP-level CreateAllocator. Wire it now that we know the backend is DirectML.
        // MIGraphX allocators are handled at factory level — leave OrtEp::CreateAllocator null
        // so ORT falls back to ep_factory_.CreateAllocator (the Allocator wrapper).
        OrtEp::CreateAllocator = [](OrtEp* this_, const OrtMemoryInfo* memory_info,
                                    OrtAllocator** allocator) noexcept {
            API_CALL_S(ExecutionProvider, this_, CreateAllocator, memory_info, allocator);
        };
    };

    const auto create_migraphx_backend = [&] {
        const auto get_name = [](const std::string_view sv) {
            return std::string{"ep."}.append(kMIGraphXBackend).append(".").append(sv);
        };
        if (info.device_id.has_value()) {
            THROW_IF_ERROR(ort_api.AddSessionConfigEntry(
                local_session_options,
                get_name(mgx_ep::provider_option::kDeviceId).c_str(),
                std::to_string(info.device_id.value()).c_str()));
        }
        if (info.cache_dir.has_value()) {
            THROW_IF_ERROR(ort_api.AddSessionConfigEntry(
                local_session_options,
                get_name(mgx_ep::provider_option::kCacheDir).c_str(),
                info.cache_dir.value().string().c_str()));
        }
        if (info.disable_caching.has_value()) {
            THROW_IF_ERROR(ort_api.AddSessionConfigEntry(
                local_session_options,
                get_name(mgx_ep::provider_option::kDisableCaching).c_str(),
                std::to_string(info.disable_caching.value()).c_str()));
        }
        if (info.force_recompile.has_value()) {
            THROW_IF_ERROR(ort_api.AddSessionConfigEntry(
                local_session_options,
                get_name(mgx_ep::provider_option::kForceRecompile).c_str(),
                std::to_string(info.force_recompile.value()).c_str()));
        }
        if (info.exhaustive_tune.has_value()) {
            THROW_IF_ERROR(ort_api.AddSessionConfigEntry(
                local_session_options,
                get_name(mgx_ep::provider_option::kExhaustiveTune).c_str(),
                std::to_string(info.exhaustive_tune.value()).c_str()));
        }
        if (info.mlss_use_specific_ops.has_value()) {
            THROW_IF_ERROR(ort_api.AddSessionConfigEntry(
                local_session_options,
                get_name(mgx_ep::provider_option::kMlssUseSpecificOps).c_str(),
                info.mlss_use_specific_ops.value().c_str()));
        }
        THROW_IF_ERROR(factory.CreateMIGraphXBackend(local_session_options, logger, backend_ep_));
    };

    if (info.profile == Profile::Eager) {
        create_directml_backend();
    } else if (info.profile == Profile::DirectML) {
        create_directml_backend();
    } else if (info.profile == Profile::MIGraphX) {
        create_migraphx_backend();
    } else {
        create_migraphx_backend();
    }
    ort_api.ReleaseSessionOptions(local_session_options);
}

ExecutionProvider::~ExecutionProvider() {
    // Release the backend EP through the backend factory that created it.
    // This frees all session-scoped resources: GPU allocator, D3D12/DML devices,
    // execution context, upload/readback heaps, and kernel registry.
    // Without this, each session leaks the entire directml EP chain.
    if (backend_ep_ != nullptr) {
        const auto backend_factory = factory_.GetBackendFactory();
        if (backend_factory != nullptr && backend_factory->ReleaseEp != nullptr) {
            backend_factory->ReleaseEp(backend_factory, backend_ep_);
        }
        backend_ep_ = nullptr;
    }
}

const char* ExecutionProvider::GetName() const noexcept {
    return ep_name_.c_str();
}

Ort::Status ExecutionProvider::GetCapability(const OrtGraph* graph,
    OrtEpGraphSupportInfo* graph_support_info) const noexcept
{
    EP_CALL_S(backend_ep_, GetCapability, graph, graph_support_info);
}

Ort::Status ExecutionProvider::Compile(const OrtGraph** graphs, const OrtNode** fused_nodes, size_t count,
    OrtNodeComputeInfo** node_compute_infos, OrtNode** ep_context_nodes) const noexcept
{
    EP_CALL_S(backend_ep_, Compile, graphs, fused_nodes, count, node_compute_infos, ep_context_nodes);
}

void ExecutionProvider::ReleaseNodeComputeInfos(OrtNodeComputeInfo** node_compute_info,
    size_t num_node_compute_info) const noexcept
{
    EP_CALL_V(backend_ep_, ReleaseNodeComputeInfos, node_compute_info, num_node_compute_info);
}

Ort::Status ExecutionProvider::SetDynamicOptions(const char* const* option_keys,
    const char* const* option_values, size_t num_options) const
{
    // TODO: check if the profile changed
    EP_CALL_S(backend_ep_, SetDynamicOptions, option_keys, option_values, num_options);
}

Ort::Status ExecutionProvider::OnRunStart(const OrtRunOptions* run_options) const noexcept {
    EP_CALL_S(backend_ep_, OnRunStart, run_options);
}

Ort::Status ExecutionProvider::CreateAllocator(const OrtMemoryInfo* memory_info,
    OrtAllocator** allocator) const noexcept {
    EP_CALL_S(backend_ep_, CreateAllocator, memory_info, allocator);
}

Ort::Status ExecutionProvider::OnRunEnd(const OrtRunOptions* run_options, bool sync_stream) const noexcept {
    EP_CALL_S(backend_ep_, OnRunEnd, run_options, sync_stream);
}

Ort::Status ExecutionProvider::CreateSyncStreamForDevice(const OrtMemoryDevice* memory_device,
    OrtSyncStreamImpl** stream) const
{
    EP_CALL_S(backend_ep_, CreateSyncStreamForDevice, memory_device, stream);
}

Ort::Status ExecutionProvider::GetKernelRegistry(const OrtKernelRegistry** kernel_registry) const {
    EP_CALL_S(backend_ep_, GetKernelRegistry, kernel_registry);
}

const char* ExecutionProvider::GetCompiledModelCompatibilityInfo(const OrtGraph* graph) const {
    EP_CALL_T(backend_ep_, GetCompiledModelCompatibilityInfo, "", graph);
}

}  // namespace gpu_ep
