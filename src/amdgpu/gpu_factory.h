// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_allocator.h"
#include "gpu_data_transfer.h"

namespace gpu_ep {

struct ProviderFactory : OrtEpFactory, ApiPtrs {
    ProviderFactory(const ApiPtrs& api_ptrs, const OrtApiBase* ort_api_base, const char* ep_name, const OrtLogger* default_logger);
    ~ProviderFactory();

    Ort::Status CreateDirectMLBackend(const OrtSessionOptions* session_options, const OrtLogger* logger, OrtEp*& ep) {
        RETURN_IF_ERROR(dml_ep_factory_->CreateEp(dml_ep_factory_, nullptr, nullptr, 0, session_options, logger, &ep));
        backend_ep_factory_ = dml_ep_factory_;
        return STATUS_OK;
    }

    Ort::Status CreateMIGraphXBackend(const OrtSessionOptions* session_options, const OrtLogger* logger, OrtEp*& ep) {
        RETURN_IF_ERROR(mgx_ep_factory_->CreateEp(mgx_ep_factory_, nullptr, nullptr, 0, session_options, logger, &ep));
        backend_ep_factory_ = mgx_ep_factory_;
        return STATUS_OK;
    }

    Ort::Status CreateHipBackend(const OrtSessionOptions* session_options, const OrtLogger* logger, OrtEp*& ep) {
        RETURN_IF_ERROR(hip_ep_factory_->CreateEp(hip_ep_factory_, nullptr, nullptr, 0, session_options, logger, &ep));
        backend_ep_factory_ = hip_ep_factory_;
        return STATUS_OK;
    }

    OrtEpFactory* GetBackendFactory() const noexcept {
        return backend_ep_factory_;
    }

private:
    [[nodiscard]] const char* GetVendor() const;
    [[nodiscard]] const char* GetName() const;

    Ort::Status GetSupportedDevices(const std::vector<Ort::ConstHardwareDevice>& devices,
    const gsl::span<OrtEpDevice*>& ep_devices, size_t& num_ep_devices) noexcept;

    Ort::Status CreateEp(gsl::span<const OrtHardwareDevice* const> devices,
        gsl::span<const OrtKeyValuePairs* const> ep_metadata,
        const Ort::ConstSessionOptions& session_options, const OrtLogger* logger, OrtEp*& ep);

    void ReleaseEp(OrtEp* ep) const;

    [[nodiscard]] uint32_t GetVendorId() const;
    [[nodiscard]] const char* GetVersion() const;

    // Ort::Status ValidateCompiledModelCompatibilityInfo(const std::vector<Ort::ConstHardwareDevice>& devices,
    //      std::string_view compatibility_info, OrtCompiledModelCompatibility* model_compatibility);

    Ort::Status CreateAllocator(const OrtMemoryInfo* memory_info,
        const OrtKeyValuePairs* allocator_options, OrtAllocator** allocator);

    void ReleaseAllocator(OrtAllocator* allocator) const;

    Ort::Status CreateDataTransfer(OrtDataTransferImpl** data_transfer);

    [[nodiscard]] bool IsStreamAware() const;

    Ort::Status CreateSyncStreamForDevice(const OrtMemoryDevice* memory_device,
        const OrtKeyValuePairs* stream_options, OrtSyncStreamImpl** stream) const;

    Ort::Status GetHardwareDeviceIncompatibilityDetails(const OrtHardwareDevice* device,
        OrtDeviceEpIncompatibilityDetails* details) const;

    Ort::Status CreateExternalResourceImporterForDevice(const OrtEpDevice* ep_device,
        OrtExternalResourceImporterImpl** out_importer) const;

    Ort::Status GetNumCustomOpDomains(size_t* num_domains) const;
    Ort::Status GetCustomOpDomains(OrtCustomOpDomain** domains, size_t num_domains) const;

    OrtEpFactory* backend_ep_factory_{};
    const Ort::Logger default_logger_{};

    std::string ep_name_;
    static constexpr std::string_view version_{"0.1.0"};

    typedef OrtStatus* (*ReleaseEpFactory_t)(OrtEpFactory*);
    typedef OrtStatus* (*CreateEpFactories_t)(const char*, const OrtApiBase*, const OrtLogger*, OrtEpFactory**, size_t, size_t*);

    void* dml_backend_{};
    ReleaseEpFactory_t dml_release_ep_factory_{};

    OrtEpFactory* dml_ep_factory_{};

    void* mgx_backend_{};
    ReleaseEpFactory_t mgx_release_ep_factory_{};

    OrtEpFactory* mgx_ep_factory_{};

    void* hip_backend_{};
    ReleaseEpFactory_t hip_release_ep_factory_{};

    OrtEpFactory* hip_ep_factory_{};

    OrtHardwareDevice* virtual_device_{};

    // Owned memory infos for the GPU and pinned device slots, registered with OrtEpDevice.
    // These must outlive the EpDevice objects that reference them.
    OrtMemoryInfo* gpu_memory_info_{};
    OrtMemoryInfo* pinned_memory_info_{};

    std::unique_ptr<DataTransfer> data_transfer_{};
    std::unique_ptr<Allocator> allocator_{};
};

}  // namespace gpu_ep