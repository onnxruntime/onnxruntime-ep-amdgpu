// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_allocator.h"
#include "gpu_data_transfer.h"
#include "gpu_info.h"

#include "common/telemetry.h"

namespace gpu_ep {

using ReleaseEpFactory_t = OrtStatus* (*)(OrtEpFactory*);
using CreateEpFactories_t = OrtStatus* (*)(const char*, const OrtApiBase*, const OrtLogger*,
    OrtEpFactory**, size_t, size_t*);

// A backend EP library and the factory it created. Empty when the backend was not built or
// failed to load; load_error then holds the reason, which fails any session that routes to
// this backend.
struct BackendLibrary {
    BackendLibrary() = default;
    BackendLibrary(const BackendLibrary&) = delete;
    BackendLibrary& operator=(const BackendLibrary&) = delete;
    ~BackendLibrary() { Reset(); }

    [[nodiscard]] bool IsLoaded() const noexcept { return factory != nullptr; }

    // Releases the factory, then unloads the library.
    void Reset() noexcept;

    void* library{};
    ReleaseEpFactory_t release{};
    OrtEpFactory* factory{};
    telemetry::GetBackendDataFn telemetry{};
    std::string load_error;
};

struct ProviderFactory : OrtEpFactory, ApiPtrs {
    ProviderFactory(const ApiPtrs& api_ptrs, const OrtApiBase* ort_api_base, const char* ep_name, const OrtLogger* default_logger);
    ~ProviderFactory();

    // Fails with the stored load error if the backend is unavailable. On success also returns
    // the backend factory that owns ep and that backend's telemetry collector (null if it has
    // none), so the caller does not have to read them back from the shared slot below, which a
    // concurrently created session may already have overwritten.
    Ort::Status CreateBackend(Backend backend, const OrtSessionOptions* session_options,
        const OrtLogger* logger, OrtEp*& ep, OrtEpFactory*& ep_factory,
        telemetry::GetBackendDataFn& get_telemetry);

    // Backend factory of the most recently created EP, for the factory-level allocator and
    // data transfer fallbacks that have no session to pin to.
    OrtEpFactory* GetBackendFactory() const noexcept {
        return backend_ep_factory_;
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
    telemetry::FileWriter telemetry_writer_;
    const Ort::Logger default_logger_{};

    std::string ep_name_;
    static constexpr std::string_view version_{"0.1.0"};

    // Loads one backend into its slot. A failure is only recorded in the slot, not raised.
    void LoadBackend(BackendLibrary& backend, Backend kind, const PathString& file,
        const char* registration_name, const OrtApiBase* ort_api_base, const OrtLogger* default_logger);

    [[nodiscard]] const BackendLibrary& FindBackend(Backend backend) const noexcept;

    // Declared before the allocators below so those are destroyed first, even when the
    // constructor throws: they call into the backend factories.
    BackendLibrary dml_backend_;
    BackendLibrary mgx_backend_;
    BackendLibrary hip_backend_;

    // Backends that can contribute custom op schemas. Filled as each backend
    // factory is created, so it only holds backends that were built and loaded.
    std::vector<OrtEpFactory*> custom_op_backends_;

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