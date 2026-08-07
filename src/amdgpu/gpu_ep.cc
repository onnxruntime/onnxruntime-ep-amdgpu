// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <filesystem>

#include "gpu_info.h"
#include "gpu_ep.h"

#include "gpu_options.h"
#ifdef USE_MIGRAPHX
#include "mgx_options.h"
#endif
#include "gpu_routing_policy.h"  // select_backend, model_arch_hash, is_webnn
#include "hip/utils.h"           // hipGetDeviceProperties for ASIC-based backend routing
#include "common/env_var.h"      // ParseEnvironmentVariableWithDefault (routing trace)

#include <iostream>
#include <string>
#include <string_view>

#include "common/telemetry.h"

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

namespace {

telemetry::Backend BackendForProfile(Profile profile) noexcept {
    switch (profile) {
        case Profile::Hip:
            return telemetry::Backend::Hip;
#ifdef USE_DML
        case Profile::Eager:
        case Profile::DirectML:
            return telemetry::Backend::DirectML;
#endif
        case Profile::Auto:
        case Profile::Optimized:
        case Profile::MIGraphX:
        default:
            // Without DirectML support the wrapper always runs on MIGraphX.
            return telemetry::Backend::MIGraphX;
    }
}

}  // namespace

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
    backend_ = BackendForProfile(info.profile);

    // Telemetry is an internal EP facility. It is enabled by default and uses the
    // platform default directory (LocalLow on Windows).
    telemetry::Config telemetry_config;
    telemetry_config.enabled = true;
    telemetry_config.file = true;
    telemetry_.emplace(std::move(telemetry_config), &factory_.TelemetryWriter());

    // Resolved-profile name for the routing trace (below). Covers every Profile value so the trace
    // never mislabels an explicit profile (e.g. Optimized/Hip are not DirectML). Auto is always
    // resolved to a concrete backend before this is called.
    const auto profile_name = [](Profile p) -> const char* {
        switch (p) {
            case Profile::MIGraphX:  return "MIGraphX";
            case Profile::DirectML:  return "DirectML";
            case Profile::Hip:       return "Hip";
            case Profile::Eager:     return "Eager";
            case Profile::Optimized: return "Optimized";
            case Profile::Auto:      return "Auto";
        }
        return "Auto";
    };

    // Heuristic backend selection for the Auto profile (explicit profiles honored as-is). Policy and
    // priority order live in gpu_routing_policy.h select_backend(); this just supplies the ASIC.
    // QA routing observability (ORT_AMDGPU_TRACE_ROUTING): follows the MIGRAPHX_TRACE_* convention —
    // env-gated, direct-to-stdout, greppable "[amdgpu-routing]" marker with key=value fields.
    const auto route_by_heuristic = [&]() -> Profile {
        const bool trace = ParseEnvironmentVariableWithDefault<bool>("ORT_AMDGPU_TRACE_ROUTING", false);
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, info.device_id.value_or(0)) != hipSuccess) {
            if (trace) {
                std::cout << "[amdgpu-routing] hipGetDeviceProperties failed -> MIGraphX (default)"
                          << std::endl;
            }
            return Profile::MIGraphX;  // preserve the historical default on query failure
        }
        const Profile chosen = select_backend(prop.gcnArchName, model_arch_hash(info.model_arch),
                                              is_webnn(info.model_fw), info.profile);
        if (trace) {
            std::cout << "[amdgpu-routing] arch=\"" << prop.gcnArchName << "\""
                      << " model_arch=" << (info.model_arch ? *info.model_arch : "(none)")
                      << " model_fw=" << (info.model_fw ? *info.model_fw : "(none)")
                      << " -> " << profile_name(chosen)
                      << std::endl;
        }
        return chosen;
    };

#ifdef USE_DML
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
#endif

    const auto create_hip_backend = [&] {
        // hip backend manages allocator/data-transfer at the backend factory level,
        // reached through the amdgpu Allocator/DataTransfer wrappers — leave
        // OrtEp::CreateAllocator null so ORT falls back to ep_factory_.CreateAllocator.
        THROW_IF_ERROR(factory.CreateHipBackend(local_session_options, logger, backend_ep_));
    };

    const auto create_migraphx_backend = [&] {
#ifdef USE_MIGRAPHX
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
        if (info.cpu_control_flow.has_value()) {
            THROW_IF_ERROR(ort_api.AddSessionConfigEntry(
                local_session_options,
                get_name(mgx_ep::provider_option::kCpuControlFlow).c_str(),
                info.cpu_control_flow.value() ? "1" : "0"));
        }
        if (info.model_arch.has_value()) {
            THROW_IF_ERROR(ort_api.AddSessionConfigEntry(
                local_session_options,
                get_name(mgx_ep::provider_option::kModelArch).c_str(),
                info.model_arch.value().c_str()));
        }
        if (info.hip_graph_enable.has_value()) {
            THROW_IF_ERROR(ort_api.AddSessionConfigEntry(
                local_session_options,
                get_name(mgx_ep::provider_option::kHipGraphEnable).c_str(),
                info.hip_graph_enable.value().c_str()));
        }
        // Relay static seq-padding options to the backend, which implements pad/slice.
        if (info.static_pad_seq.has_value()) {
            THROW_IF_ERROR(ort_api.AddSessionConfigEntry(
                local_session_options,
                get_name(mgx_ep::provider_option::kStaticPadSeq).c_str(),
                info.static_pad_seq.value().c_str()));
        }
        if (info.static_pad_seq_len.has_value()) {
            THROW_IF_ERROR(ort_api.AddSessionConfigEntry(
                local_session_options,
                get_name(mgx_ep::provider_option::kStaticPadSeqLen).c_str(),
                info.static_pad_seq_len.value().c_str()));
        }
        if (info.static_pad_inputs.has_value()) {
            THROW_IF_ERROR(ort_api.AddSessionConfigEntry(
                local_session_options,
                get_name(mgx_ep::provider_option::kStaticPadInputs).c_str(),
                info.static_pad_inputs.value().c_str()));
        }
        if (info.static_pad_outputs.has_value()) {
            THROW_IF_ERROR(ort_api.AddSessionConfigEntry(
                local_session_options,
                get_name(mgx_ep::provider_option::kStaticPadOutputs).c_str(),
                info.static_pad_outputs.value().c_str()));
        }
        THROW_IF_ERROR(factory.CreateMIGraphXBackend(local_session_options, logger, backend_ep_));
#endif
    };

    // Explicit profile is honored; Auto/Optimized derives from (ASIC, model_arch). select_backend()
    // (inside route_by_heuristic) applies both, so the result covers every profile value.
    const Profile effective = route_by_heuristic();

#ifdef USE_DML
    if (effective == Profile::Eager) {
        create_directml_backend();
    } else if (effective == Profile::DirectML) {
        create_directml_backend();
    } else if (effective == Profile::MIGraphX) {
        create_migraphx_backend();
    } else if (effective == Profile::Hip) {
        create_hip_backend();
    } else {
        create_migraphx_backend();
    }
#else
    // DirectML not built (e.g. Linux): DirectML/Eager profiles fall back to MIGraphX.
    if (true) {
        create_hip_backend();
    } else {
        create_migraphx_backend();
    }
#endif
    // Capture per-EP now: the shared factory_ backend field is overwritten by a later
    // umbrella EP (different backend). See PR for the cross-backend UAF details.
    backend_ep_factory_ = factory_.GetBackendFactory();
    ort_api.ReleaseSessionOptions(local_session_options);
}

ExecutionProvider::~ExecutionProvider() {
    // Release backend_ep_ via the factory that created THIS EP (backend_ep_factory_), not the
    // shared factory_ field. Frees all session-scoped resources (allocator, D3D12/DML devices,
    // execution context, heaps, kernel registry).
    if (backend_ep_ != nullptr) {
        if (backend_ep_factory_ != nullptr && backend_ep_factory_->ReleaseEp != nullptr) {
            backend_ep_factory_->ReleaseEp(backend_ep_factory_, backend_ep_);
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
    if (backend_ep_ == nullptr) {
        return MAKE_STATUS(ORT_EP_FAIL, "Compile: invalid backend");
    }
    if (backend_ep_->Compile != nullptr) {
        RETURN_IF_ERROR(backend_ep_->Compile(backend_ep_, graphs, fused_nodes, count,
            node_compute_infos, ep_context_nodes));
    }
    if (count > 0 && graphs != nullptr) {
        LogTelemetry(Ort::ConstGraph{graphs[0]});
    }
    return STATUS_OK;
}

void ExecutionProvider::LogTelemetry(const Ort::ConstGraph& graph) const noexcept try {
    if (!telemetry_ || !telemetry_->IsEnabled()) {
        return;
    }
    std::call_once(telemetry_once_, [&] {
        // Generic, backend-agnostic fields collected by the wrapper.
        telemetry::Record record;
        record.SetEpVersion(factory_.GetVersion())
              .SetBackend(backend_)
              .SetParentProcess(telemetry::ParentProcessName());
        const std::filesystem::path model_path{graph.GetModelPath()};
        if (model_path.has_filename()) {
            record.SetModelName(model_path.filename().string());
        }
        if (const telemetry::GetBackendDataFn collect = factory_.GetBackendTelemetryFn();
                collect != nullptr) {
            telemetry::BackendData data{};
            if (collect(backend_ep_, &data)) {
                record.Merge(data);
            }
        }
        telemetry_->Write(record);
    });
} catch (const std::exception&) {
    // Telemetry must never disrupt inference.
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
