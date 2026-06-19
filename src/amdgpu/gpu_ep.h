// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <mutex>
#include <string>

#include "gpu_info.h"
#include "gpu_factory.h"

namespace gpu_ep {

struct ExecutionProvider : OrtEp, ApiPtrs {
    ExecutionProvider(ProviderFactory& factory, std::string_view ep_name,
        const Ort::ConstSessionOptions& session_options, const OrtLogger* logger);
    ~ExecutionProvider();

private:
    [[nodiscard]] const char* GetName() const noexcept;

    Ort::Status GetCapability(const OrtGraph* graph,
        OrtEpGraphSupportInfo* graph_support_info) const noexcept;

    Ort::Status Compile(const OrtGraph** graphs, const OrtNode** fused_nodes, size_t count,
        OrtNodeComputeInfo** node_compute_infos, OrtNode** ep_context_nodes) const noexcept;

    void ReleaseNodeComputeInfos(OrtNodeComputeInfo** node_compute_info, size_t num_node_compute_info) const noexcept;

    [[nodiscard]] Ort::Status SetDynamicOptions(const char* const* option_keys,
        const char* const* option_values, size_t num_options) const;

    Ort::Status OnRunStart(const OrtRunOptions* run_options) const noexcept;
    Ort::Status OnRunEnd(const OrtRunOptions* run_options, bool sync_stream) const noexcept;
    Ort::Status CreateAllocator(const OrtMemoryInfo* memory_info, OrtAllocator** allocator) const noexcept;
    Ort::Status CreateSyncStreamForDevice(const OrtMemoryDevice* memory_device, OrtSyncStreamImpl** stream) const;
    [[nodiscard]] const char* GetCompiledModelCompatibilityInfo(const OrtGraph* graph) const;
    Ort::Status GetKernelRegistry(const OrtKernelRegistry** kernel_registry) const;

    void LogTelemetry(const OrtGraph* const* graphs, size_t count) const noexcept;

    ProviderFactory& factory_;
    OrtEp* backend_ep_{};

    std::string ep_name_;
    std::string backend_name_;
    mutable std::once_flag telemetry_once_;
    const Ort::Logger logger_{};
};

}  // namespace gpu_ep
