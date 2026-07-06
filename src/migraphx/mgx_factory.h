// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>
#include <gsl/gsl>

#include "common/plugin_ep_utils.h"

#include "hip/allocator.h"
#include "hip/data_transfer.h"

namespace mgx_ep {

struct ProviderFactory : OrtEpFactory, ApiPtrs {
    ProviderFactory(const ApiPtrs& api_ptrs, const char* ep_name, const Ort::Logger& default_logger);

    Ort::Status GetKernelRegistry(std::string_view ep_name, const OrtKernelRegistry*& kernel_registry) const;

private:
    [[nodiscard]] const char* GetName() const;
    [[nodiscard]] const char* GetVendor() const;

    Ort::Status GetSupportedDevices(const std::vector<Ort::ConstHardwareDevice>& devices,
        const gsl::span<OrtEpDevice*>& ep_devices, size_t& num_ep_devices);

    Ort::Status CreateEp(const std::vector<Ort::ConstHardwareDevice>& devices,
        const std::vector<Ort::ConstKeyValuePairs>& ep_metadata_pairs,
        const Ort::ConstSessionOptions& session_options, const Ort::Logger& logger, OrtEp*& ep);

    void ReleaseEp(OrtEp* ep) const;

    [[nodiscard]] uint32_t GetVendorId() const;
    [[nodiscard]] const char* GetVersion() const;

    // Ort::Status ValidateCompiledModelCompatibilityInfo(const std::vector<Ort::ConstHardwareDevice>& devices,
    //     std::string_view compatibility_info, OrtCompiledModelCompatibility* model_compatibility);

    Ort::Status CreateAllocator(const Ort::ConstMemoryInfo& memory_info,
        const Ort::ConstKeyValuePairs& allocator_options, OrtAllocator*& allocator);

    void ReleaseAllocator(OrtAllocator* allocator);

    Ort::Status CreateDataTransfer(OrtDataTransferImpl*& data_transfer) const;

    [[nodiscard]] bool IsStreamAware() const;

    Ort::Status CreateSyncStreamForDevice(const OrtMemoryDevice* memory_info,
        const Ort::ConstKeyValuePairs& stream_options, OrtSyncStreamImpl*& stream);

    //    Ort::Status GetHardwareDeviceIncompatibilityDetails(const OrtHardwareDevice* hw,
    //    OrtDeviceEpIncompatibilityDetails* details);

    Ort::Status CreateExternalResourceImporterForDevice(const Ort::ConstEpDevice& ep_device,
        OrtExternalResourceImporterImpl*& out_importer) const;

    //    Ort::Status GetNumCustomOpDomains(size_t* num_domains) const;
    //    Ort::Status GetCustomOpDomains(OrtCustomOpDomain** domains, size_t num_domains) const;

    const Ort::Logger default_logger_;

    std::string ep_name_;
    std::string_view version_{"0.1.0"};

    mutable OrtKernelRegistry* kernel_registry_{};

    MemoryInfoMap pinned_memory_infos_;
    MemoryInfoMap gpu_memory_infos_;

    MemoryDeviceList pinned_memory_devices_;
    MemoryDeviceList gpu_memory_devices_;

    AllocatorMap<hip::PinnedAllocator> pinned_allocators_;
    AllocatorMap<hip::Allocator> gpu_allocators_;

    std::unique_ptr<hip::DataTransfer> data_transfer_;
};

}  // namespace mgx_ep
