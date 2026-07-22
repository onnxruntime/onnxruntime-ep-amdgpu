// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gpu_info.h"
#include "gpu_ep.h"

#include "gpu_options.h"
#include "mgx_options.h"
#include "hip/utils.h"        // hipGetDeviceProperties for ASIC-based backend routing
#include "common/env_var.h"   // ParseEnvironmentVariable (routing test override)

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <string>
#include <string_view>

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

// FNV-1a (64-bit). Used to store model_arch names as hash constants so the plaintext
// names appear in neither the source nor the shipped binary.
constexpr std::uint64_t fnv1a(std::string_view s) {
    std::uint64_t h{0xcbf29ce484222325ULL};
    for (const char c : s) {
        h ^= static_cast<std::uint8_t>(c);
        h *= 0x100000001b3ULL;
    }
    return h;
}

// Normalize a model_arch string (lowercase + trim) before hashing, so "Llama", " llama"
// all match. Must mirror the tool that generates the kLlmModelArch constants.
std::string normalize_model_arch(std::string_view value) {
    std::string s{value};
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const auto not_space = [](unsigned char c) { return std::isspace(c) == 0; };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

// Known LLM model_arch families (hashed). Presence of model_arch alone does NOT imply
// LLM (non-LLM callers may set it too); only a match here counts as an LLM.
constexpr std::array<std::uint64_t, 9> kLlmModelArch{{
    fnv1a("llama"), fnv1a("qwen2"), fnv1a("qwen3"), fnv1a("phi3"), fnv1a("gpt2"),
    fnv1a("mistral"), fnv1a("gemma"), fnv1a("gemma2"), fnv1a("lfm2"),
}};

// Parse the numeric arch from a gcnArchName, e.g. "gfx1201" or "gfx1200:xnack-" -> 1201/1200.
int parse_gfx_arch(std::string_view arch_name) {
    int arch{0};
    const auto pos = arch_name.find("gfx");
    if (pos != std::string_view::npos) {
        for (std::size_t i = pos + 3;
             i < arch_name.size() && std::isdigit(static_cast<unsigned char>(arch_name[i])); ++i) {
            arch = arch * 10 + (arch_name[i] - '0');
        }
    }
    return arch;
}

// True if a model_arch string names a known LLM family. Presence alone does NOT imply LLM
// (non-LLM callers may set model_arch too); only a match against kLlmModelArch counts.
bool is_llm_model_arch(const std::optional<std::string>& model_arch) {
    if (!model_arch.has_value()) return false;
    const std::uint64_t h = fnv1a(normalize_model_arch(model_arch.value()));
    return std::find(kLlmModelArch.begin(), kLlmModelArch.end(), h) != kLlmModelArch.end();
}

// Pure backend-selection policy (AMDGPUUmbrellaEP.md §4.3). No I/O — unit-testable.
// Only the Auto profile is heuristic-driven; every other profile (explicit backend, or Optimized)
// is honored/left as-is and dispatched unchanged. Auto derives from (arch, is_llm):
//   Medusa gfx1170/gfx1171 -> DirectML (checked first)
//   LLM:     arch < gfx1100 -> DirectML, else MIGraphX
//   non-LLM: arch < gfx1150 -> DirectML, else MIGraphX
Profile select_backend(int arch, bool is_llm, Profile profile) {
    if (profile != Profile::Auto) {
        return profile;  // explicit profile (and Optimized) honored/dispatched as-is
    }
    if (arch == 1170 || arch == 1171) return Profile::DirectML;              // Medusa (first)
    if (is_llm)  return arch < 1100 ? Profile::DirectML : Profile::MIGraphX;
    return arch < 1150 ? Profile::DirectML : Profile::MIGraphX;
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

    // Heuristic backend selection for the Auto/Optimized profile. Explicit profiles
    // (migraphx/directml/hip) are honored unchanged below. Policy (AMDGPUUmbrellaEP.md §4.3):
    //   Medusa gfx1170/gfx1171          -> DirectML (checked first)
    //   LLM      (model_arch is a known family): arch < gfx1100 -> DirectML, else MIGraphX
    //   non-LLM  (no/other model_arch):          arch < gfx1150 -> DirectML, else MIGraphX
    // ORT_AMDGPU_ROUTING_FORCE_ARCH=<gfxNNNN> overrides the queried arch for testing on a single GPU.
    const auto route_by_heuristic = [&]() -> Profile {
        std::string arch_name;
        const auto forced = ParseEnvironmentVariable<std::string>("ORT_AMDGPU_ROUTING_FORCE_ARCH");
        if (forced.has_value()) {
            arch_name = *forced;
        } else {
            hipDeviceProp_t prop{};
            if (hipGetDeviceProperties(&prop, info.device_id.value_or(0)) != hipSuccess) {
                return Profile::MIGraphX;  // preserve the historical default on query failure
            }
            arch_name = prop.gcnArchName;  // e.g. "gfx1201" or "gfx1100:xnack-"
        }
        // TODO(routing): WebNN caller -> DirectML on all ASICs once the WebNN signal is available here.
        const int arch = parse_gfx_arch(arch_name);
        const bool is_llm = is_llm_model_arch(info.model_arch);
        const Profile chosen = select_backend(arch, is_llm, info.profile);

        if (ParseEnvironmentVariableWithDefault<bool>("ORT_AMDGPU_ROUTING_DEBUG", false)) {
            fprintf(stderr, "[amdgpu-routing] arch=\"%s\"(%d)%s model_arch=%s llm=%d -> %s\n", arch_name.c_str(), arch,
                    forced.has_value() ? " [forced]" : "",
                    info.model_arch.has_value() ? info.model_arch.value().c_str() : "(none)", is_llm,
                    chosen == Profile::DirectML ? "DirectML" : "MIGraphX");
            fflush(stderr);
        }
        return chosen;
    };

    const auto create_directml_backend = [&] {
        if (ParseEnvironmentVariableWithDefault<bool>("ORT_AMDGPU_ROUTING_DEBUG", false)) {
            fprintf(stderr, "[amdgpu-routing] create DirectML backend\n"); fflush(stderr);
        }
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

    const auto create_hip_backend = [&] {
        if (ParseEnvironmentVariableWithDefault<bool>("ORT_AMDGPU_ROUTING_DEBUG", false)) {
            fprintf(stderr, "[amdgpu-routing] create Hip backend\n"); fflush(stderr);
        }
        // hip backend manages allocator/data-transfer at the backend factory level,
        // reached through the amdgpu Allocator/DataTransfer wrappers — leave
        // OrtEp::CreateAllocator null so ORT falls back to ep_factory_.CreateAllocator.
        THROW_IF_ERROR(factory.CreateHipBackend(local_session_options, logger, backend_ep_));
    };

    const auto create_migraphx_backend = [&] {
        if (ParseEnvironmentVariableWithDefault<bool>("ORT_AMDGPU_ROUTING_DEBUG", false)) {
            fprintf(stderr, "[amdgpu-routing] create MIGraphX backend\n"); fflush(stderr);
        }
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
        if (info.model_arch.has_value()) {
            THROW_IF_ERROR(ort_api.AddSessionConfigEntry(
                local_session_options,
                get_name(mgx_ep::provider_option::kModelArch).c_str(),
                info.model_arch.value().c_str()));
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
    };

    // Explicit profile is honored; Auto/Optimized derives from (ASIC, model_arch). select_backend()
    // (inside route_by_heuristic) applies both, so the result covers every profile value.
    const Profile effective = route_by_heuristic();

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
