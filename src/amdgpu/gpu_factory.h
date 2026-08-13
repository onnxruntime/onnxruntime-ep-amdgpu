// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_allocator.h"
#include "gpu_data_transfer.h"

#include "common/telemetry.h"

namespace gpu_ep {

struct ProviderFactory : OrtEpFactory, ApiPtrs {
    ProviderFactory(const ApiPtrs& api_ptrs, const OrtApiBase* ort_api_base, const char* ep_name, const OrtLogger* default_logger);
    ~ProviderFactory();

    Ort::Status CreateDirectXBackend(const OrtSessionOptions* session_options, const OrtLogger* logger, OrtEp*& ep) {
#ifdef USE_DML
        RETURN_IF_ERROR(dml_ep_factory_->CreateEp(dml_ep_factory_, nullptr, nullptr, 0, session_options, logger, &ep));
        backend_ep_factory_ = dml_ep_factory_;
        backend_get_telemetry_ = dml_get_telemetry_;
        return STATUS_OK;
#else
        (void)session_options;
        (void)logger;
        (void)ep;
        return MAKE_STATUS(ORT_FAIL,
            "DirectML backend requested, but the AMDGPU EP was not built with DirectML support");
#endif
    }

    Ort::Status CreateMIGraphXBackend(const OrtSessionOptions* session_options, const OrtLogger* logger, OrtEp*& ep) {
        RETURN_IF_ERROR(mgx_ep_factory_->CreateEp(mgx_ep_factory_, nullptr, nullptr, 0, session_options, logger, &ep));
        backend_ep_factory_ = mgx_ep_factory_;
        backend_get_telemetry_ = mgx_get_telemetry_;
        return STATUS_OK;
    }

    Ort::Status CreateHipBackend(const OrtSessionOptions* session_options, const OrtLogger* logger, OrtEp*& ep) {
        // null when not built with hip support.
        if (hip_ep_factory_ == nullptr) {
            return MAKE_STATUS(ORT_FAIL,
                "hip backend requested (profile=hip), but the AMDGPU EP was not built with hip support");
        }
        RETURN_IF_ERROR(hip_ep_factory_->CreateEp(hip_ep_factory_, nullptr, nullptr, 0, session_options, logger, &ep));
        backend_ep_factory_ = hip_ep_factory_;
        return STATUS_OK;
    }

    OrtEpFactory* GetBackendFactory() const noexcept {
        return backend_ep_factory_;
    }

    // Telemetry collector exported by the active backend DLL, or null if that
    // backend contributes no backend-specific telemetry.
    [[nodiscard]] telemetry::GetBackendDataFn GetBackendTelemetryFn() const noexcept {
        return backend_get_telemetry_;
    }

    [[nodiscard]] telemetry::FileWriter& TelemetryWriter() noexcept { return telemetry_writer_; }

    [[nodiscard]] const char* GetVersion() const;

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
    telemetry::GetBackendDataFn backend_get_telemetry_{};
    telemetry::FileWriter telemetry_writer_;
    const Ort::Logger default_logger_{};

    std::string ep_name_;
    static constexpr std::string_view version_{"0.1.0"};

    typedef OrtStatus* (*ReleaseEpFactory_t)(OrtEpFactory*);
    typedef OrtStatus* (*CreateEpFactories_t)(const char*, const OrtApiBase*, const OrtLogger*, OrtEpFactory**, size_t, size_t*);

#ifdef USE_DML
    void* dml_backend_{};
    ReleaseEpFactory_t dml_release_ep_factory_{};

    OrtEpFactory* dml_ep_factory_{};
    telemetry::GetBackendDataFn dml_get_telemetry_{};
#endif

    void* mgx_backend_{};
    ReleaseEpFactory_t mgx_release_ep_factory_{};

    OrtEpFactory* mgx_ep_factory_{};
    telemetry::GetBackendDataFn mgx_get_telemetry_{};

    void* hip_backend_{};
    ReleaseEpFactory_t hip_release_ep_factory_{};

    OrtEpFactory* hip_ep_factory_{};

    OrtHardwareDevice* virtual_device_{};

    // Owned memory infos for the GPU and pinned device slots, registered with OrtEpDevice.
    // These must outlive the EpDevice objects that reference them.
    OrtMemoryInfo* gpu_memory_info_{};
    OrtMemoryInfo* pinned_memory_info_{};

    // One allocator per device memory type. ORT registers a separate shared
    // allocator per OrtMemoryInfo (default vs host-accessible) and matches them
    // by the device reported from each allocator's Info(); a single shared
    // instance would report only one device, leaving the other unmatched at
    // unload (use-after-free on the factory during teardown).
    std::unique_ptr<Allocator> gpu_allocator_{};     // OrtDeviceMemoryType_DEFAULT
    std::unique_ptr<Allocator> pinned_allocator_{};  // OrtDeviceMemoryType_HOST_ACCESSIBLE
};

}  // namespace gpu_ep