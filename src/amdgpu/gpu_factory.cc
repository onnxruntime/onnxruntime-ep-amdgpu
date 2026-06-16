// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "common/enumerate.h"
#include "common/dynamic_library.h"

#include "gpu_options.h"
#include "gpu_ep.h"
#include "gpu_factory.h"

#include "onnxruntime_ep_device_ep_metadata_keys.h"

#define BACKEND_CALL_T(backend, fn, defval, ...)                       \
    do {                                                               \
        return (backend != nullptr &&                                  \
                backend->fn != nullptr) ?                              \
                    backend->fn(backend, __VA_ARGS__) : defval;        \
    } while (0)

#define BACKEND_CALL_S(backend, fn, ...)                               \
    do {                                                               \
        if (backend != nullptr) {                                      \
            if (backend->fn == nullptr) {                              \
                return MAKE_STATUS(ORT_NOT_IMPLEMENTED, #fn ": method not implemented"); \
            }                                                          \
            RETURN_IF_ERROR(backend->fn(backend, __VA_ARGS__));        \
            return STATUS_OK;                                          \
        }                                                              \
        return MAKE_STATUS(ORT_EP_FAIL, #fn ": invalid backend factory"); \
    } while (0)

#define BACKEND_CALL_V(backend, fn, ...)                               \
    do {                                                               \
        if (backend != nullptr && backend->fn != nullptr) {            \
            backend->fn(backend, __VA_ARGS__);                         \
        }                                                              \
    } while (0)


namespace gpu_ep {

namespace {
constexpr auto directmlBackend{LIBRARY_PREFIX ORT_TSTR("directml-backend") LIBRARY_SUFFIX};
constexpr auto migraphxBackend{LIBRARY_PREFIX ORT_TSTR("migraphx-backend") LIBRARY_SUFFIX};
constexpr auto hipepBackend{LIBRARY_PREFIX ORT_TSTR("hipep-backend") LIBRARY_SUFFIX};
}

ProviderFactory::ProviderFactory(const ApiPtrs& api_ptrs, const OrtApiBase* ort_api_base, const char* ep_name, const OrtLogger* default_logger)
    : OrtEpFactory{ORT_API_VERSION}, ApiPtrs{api_ptrs}, default_logger_{default_logger}, ep_name_{ep_name}
{
    OrtEpFactory::GetName = [](const OrtEpFactory* this_) noexcept {
        API_CALL_T(const ProviderFactory, this_, GetName, "invalid object pointer");
    };
    OrtEpFactory::GetVendor = [](const OrtEpFactory* this_) noexcept {
        API_CALL_T(const ProviderFactory, this_, GetVendor, "invalid object pointer");
    };
    OrtEpFactory::GetSupportedDevices = [](OrtEpFactory* this_,
            const OrtHardwareDevice* const* devices, size_t num_devices, OrtEpDevice** ep_devices,
            size_t max_ep_devices, size_t* num_ep_devices) noexcept {
        API_CALL_S(ProviderFactory, this_, GetSupportedDevices, {devices, devices + num_devices},
            {ep_devices, ep_devices + max_ep_devices}, *num_ep_devices);
    };
    OrtEpFactory::CreateEp = [](OrtEpFactory* this_, const OrtHardwareDevice* const* devices,
            const OrtKeyValuePairs* const* ep_metadata_pairs, size_t num_devices,
            const OrtSessionOptions* session_options, const OrtLogger* logger, OrtEp** ep) noexcept
    {
        API_CALL_S(ProviderFactory, this_, CreateEp, {devices, devices + num_devices},
            {ep_metadata_pairs, ep_metadata_pairs + num_devices},
            Ort::ConstSessionOptions{session_options}, logger, *ep);
    };
    OrtEpFactory::ReleaseEp = [](OrtEpFactory* this_, OrtEp* ep) noexcept {
        API_CALL_V(ProviderFactory, this_, ReleaseEp, ep);
    };
    OrtEpFactory::GetVendorId = [](const OrtEpFactory* this_) noexcept {
        API_CALL_T(const ProviderFactory, this_, GetVendorId, static_cast<uint32_t>(-1));
    };
    OrtEpFactory::GetVersion = [](const OrtEpFactory* this_) noexcept {
        API_CALL_T(const ProviderFactory, this_, GetVersion, "invalid object pointer");
    };
    // TODO: OrtEpFactory::ValidateCompiledModelCompatibilityInfo = []
    OrtEpFactory::CreateAllocator = [](OrtEpFactory* this_, const OrtMemoryInfo* memory_info,
            const OrtKeyValuePairs* allocator_options, OrtAllocator** allocator) noexcept {
        API_CALL_S(ProviderFactory, this_, CreateAllocator, memory_info,
            allocator_options, allocator);
    };
    OrtEpFactory::ReleaseAllocator = [](OrtEpFactory* this_, OrtAllocator* allocator) noexcept {
        API_CALL_V(ProviderFactory, this_, ReleaseAllocator, allocator);
    };
    OrtEpFactory::CreateDataTransfer = [](OrtEpFactory* this_, OrtDataTransferImpl** data_transfer) noexcept {
        API_CALL_S(ProviderFactory, this_, CreateDataTransfer, data_transfer);
    };
    OrtEpFactory::IsStreamAware = [](const OrtEpFactory* this_) noexcept {
        API_CALL_T(const ProviderFactory, this_, IsStreamAware, false);
    };
    OrtEpFactory::CreateSyncStreamForDevice = [](OrtEpFactory* this_, const OrtMemoryDevice* memory_device,
            const OrtKeyValuePairs* stream_options, OrtSyncStreamImpl** stream) noexcept {
        API_CALL_S(ProviderFactory, this_, CreateSyncStreamForDevice, memory_device,
            stream_options, stream);
    };
    OrtEpFactory::GetHardwareDeviceIncompatibilityDetails = [](OrtEpFactory* this_, const OrtHardwareDevice* hw,
            OrtDeviceEpIncompatibilityDetails* details) noexcept {
        API_CALL_S(ProviderFactory, this_, GetHardwareDeviceIncompatibilityDetails, hw, details);
    };
    OrtEpFactory::CreateExternalResourceImporterForDevice = [](OrtEpFactory* this_, const OrtEpDevice* ep_device,
            OrtExternalResourceImporterImpl** out_importer) noexcept {
        API_CALL_S(ProviderFactory, this_, CreateExternalResourceImporterForDevice,
            ep_device, out_importer);
    };
    OrtEpFactory::GetNumCustomOpDomains = [](OrtEpFactory* this_, size_t* num_domains) noexcept {
        API_CALL_S(ProviderFactory, this_, GetNumCustomOpDomains, num_domains);
    };
    OrtEpFactory::GetCustomOpDomains = [](OrtEpFactory* this_, OrtCustomOpDomain** domains,
            const size_t num_domains) noexcept {
        API_CALL_S(ProviderFactory, this_, GetCustomOpDomains, domains, num_domains);
    };

    THROW_IF_ERROR(LoadDynamicLibrary(directmlBackend, &dml_backend_));
    THROW_IF_ERROR(GetSymbolFromLibrary(dml_backend_,
        "ReleaseEpFactory", reinterpret_cast<void**>(&dml_release_ep_factory_)));

    CreateEpFactories_t dml_create_ep_factories{};
    THROW_IF_ERROR(GetSymbolFromLibrary(dml_backend_,
        "CreateEpFactories", reinterpret_cast<void**>(&dml_create_ep_factories)));

    size_t factories_created{};
    // Pass ep_name_ (e.g. "amdgpu") so the directml backend registers its kernels,
    // allocators, and node assignments under the same name ORT sees for this EP.
    // Using kDirectMLBackend ("directml") would cause a provider name mismatch:
    // ORT would look up kernels under "amdgpu" but find them stamped as "directml".
    THROW_IF_ERROR(dml_create_ep_factories(ep_name_.c_str(), ort_api_base, default_logger,
        &dml_ep_factory_, 1, &factories_created));

    THROW_IF_ERROR(LoadDynamicLibrary(migraphxBackend, &mgx_backend_));
    THROW_IF_ERROR(GetSymbolFromLibrary(mgx_backend_,
        "ReleaseEpFactory", reinterpret_cast<void**>(&mgx_release_ep_factory_)));

    CreateEpFactories_t mgx_create_ep_factories{};
    THROW_IF_ERROR(GetSymbolFromLibrary(mgx_backend_,
        "CreateEpFactories", reinterpret_cast<void**>(&mgx_create_ep_factories)));

    THROW_IF_ERROR(mgx_create_ep_factories(kMIGraphXBackend, ort_api_base, default_logger,
        &mgx_ep_factory_, 1, &factories_created));

    THROW_IF_ERROR(LoadDynamicLibrary(hipepBackend, &hip_backend_));
    THROW_IF_ERROR(GetSymbolFromLibrary(hip_backend_,
        "ReleaseEpFactory", reinterpret_cast<void**>(&hip_release_ep_factory_)));

    CreateEpFactories_t hip_create_ep_factories{};
    THROW_IF_ERROR(GetSymbolFromLibrary(hip_backend_,
        "CreateEpFactories", reinterpret_cast<void**>(&hip_create_ep_factories)));

    // Pass ep_name_ so the hipep backend's EP reports the same name ORT sees.
    THROW_IF_ERROR(hip_create_ep_factories(ep_name_.c_str(), ort_api_base, default_logger,
        &hip_ep_factory_, 1, &factories_created));

    data_transfer_ = std::make_unique<DataTransfer>(*this);
}

ProviderFactory::~ProviderFactory() {
    // Destroy members that hold backend function pointers before unloading the DLLs.
    // ~Allocator calls backend_ep_factory_->ReleaseAllocator and ~DataTransfer calls
    // backend factory methods — both would crash if the DLL is already unloaded.
    // Session-owned resources (DmlBucketizedBufferAllocator etc.) are already released
    // by the time the factory is destroyed since sessions are destroyed first.
    allocator_.reset();
    data_transfer_.reset();

    if (gpu_memory_info_) {
        ort_api.ReleaseMemoryInfo(gpu_memory_info_);
    }
    if (pinned_memory_info_) {
        ort_api.ReleaseMemoryInfo(pinned_memory_info_);
    }
    if (!UnloadDynamicLibrary(dml_backend_).IsOK()) {
        /* TODO: log failure while unloading DirectML EP library */
    }
    if (!UnloadDynamicLibrary(mgx_backend_).IsOK()) {
        /* TODO: log failure while unloading MIGraphX EP library */
    }
    if (!UnloadDynamicLibrary(hip_backend_).IsOK()) {
        /* TODO: log failure while unloading hipep EP library */
    }
}

const char* ProviderFactory::GetName() const {
    return ep_name_.c_str();
}

const char* ProviderFactory::GetVendor() const {
    return amd::Vendor;
}

Ort::Status ProviderFactory::GetSupportedDevices(const std::vector<Ort::ConstHardwareDevice>& devices,
    const gsl::span<OrtEpDevice*>& ep_devices, size_t& num_ep_devices) noexcept
{
    num_ep_devices = 0;
    for(const auto& [i, device] : enumerate(devices)) {
        if (num_ep_devices >= ep_devices.size()) {
            break;
        }
        if (device.VendorId() == amd::VendorId && device.Type() == OrtHardwareDeviceType_GPU) {
            Ort::EpDevice ep_device{*this, const_cast<Ort::ConstHardwareDevice&>(device)};

            Ort::ConstKeyValuePairs metadata = ep_device.Device().Metadata();

            // Use DxgiAdapterNumber from device metadata as the device_id when available,
            // so the memory info device matches what DX12 uses to identify the adapter.
            int device_index = i;
            std::vector<const char*> keys, values;
            metadata.GetKeyValuePairs(keys, values);
            for (size_t idx = 0; idx < keys.size(); ++idx) {
                if (strcmp(keys[idx], "DxgiAdapterNumber") == 0) {
                    device_index = std::atoi(values[idx]);
                }
            }

            // amdgpu-ep owns these memory info objects for the factory lifetime.
            // The Allocator wrapper's Info() returns gpu_memory_info_, ensuring that
            // ORT's plan locations and allocator map keys resolve to the same OrtDevice.
            // Note: gpu_memory_info_ and pinned_memory_info_ are overwritten on each
            // iteration, so only the last matched device is retained. This is acceptable
            // for the common single-GPU case; multi-GPU support would require a per-device map.
            RETURN_IF_ERROR(ort_api.CreateMemoryInfo_V2("default", OrtMemoryInfoDeviceType_GPU,
                amd::VendorId, device_index, OrtDeviceMemoryType_DEFAULT, 0, OrtDeviceAllocator, &gpu_memory_info_));
            RETURN_IF_ERROR(ep_api.EpDevice_AddAllocatorInfo(ep_device, gpu_memory_info_));
            RETURN_IF_ERROR(ort_api.CreateMemoryInfo_V2("pinned", OrtMemoryInfoDeviceType_GPU,
                amd::VendorId, device_index, OrtDeviceMemoryType_HOST_ACCESSIBLE, 0, OrtDeviceAllocator, &pinned_memory_info_));
            RETURN_IF_ERROR(ep_api.EpDevice_AddAllocatorInfo(ep_device, pinned_memory_info_));
            ep_devices[num_ep_devices++] = ep_device.release();
        }
    }
    return STATUS_OK;
}

Ort::Status ProviderFactory::CreateEp(gsl::span<const OrtHardwareDevice* const> devices,
    gsl::span<const OrtKeyValuePairs* const> ep_metadata,
    const Ort::ConstSessionOptions& session_options, const OrtLogger* logger, OrtEp* &ep)
try {
    ep = nullptr;
    if (devices.size() > 1) {
        return MAKE_STATUS(ORT_INVALID_ARGUMENT, "only supports selection for a single device");
    }

    ep = std::make_unique<ExecutionProvider>(*this, ep_name_, devices, ep_metadata,
        session_options, logger).release();
    return STATUS_OK;
} catch (const Ort::Exception& e) {
    return Ort::Status{e};
} catch (const std::exception& e) {
    return Ort::Status{e.what(), ORT_EP_FAIL};
}

void ProviderFactory::ReleaseEp(OrtEp* ep) const {
    delete reinterpret_cast<ExecutionProvider*>(ep);
}

uint32_t ProviderFactory::GetVendorId() const {
    return amd::VendorId;
}

const char* ProviderFactory::GetVersion() const {
    return version_.data();
}

Ort::Status ProviderFactory::CreateAllocator(const OrtMemoryInfo* memory_info,
    const OrtKeyValuePairs* allocator_options, OrtAllocator** allocator)
{
    // The Allocator wrapper owns the amdgpu-ep memory info and lazily delegates
    // actual allocation to whichever backend (DirectML or MIGraphX) is active.
    // Its Info() returns the amdgpu-owned memory_info_ pointer, ensuring that
    // the device key in ORT's allocator map matches the plan location device.
    if (allocator_ == nullptr) {
        allocator_ = std::make_unique<Allocator>(*this, memory_info, allocator_options);
    }
    *allocator = allocator_.get();
    return STATUS_OK;
}

void ProviderFactory::ReleaseAllocator(OrtAllocator*) const {
    // no-op — the allocator is owned by allocator_ and shared across sessions.
}

Ort::Status ProviderFactory::CreateDataTransfer(OrtDataTransferImpl** data_transfer) {
    *data_transfer = data_transfer_.get();
    return STATUS_OK;
}

bool ProviderFactory::IsStreamAware() const {
    BACKEND_CALL_T(backend_ep_factory_, IsStreamAware, false);
}

Ort::Status ProviderFactory::CreateSyncStreamForDevice(const OrtMemoryDevice* memory_device,
    const OrtKeyValuePairs* stream_options, OrtSyncStreamImpl** stream) const
{
    BACKEND_CALL_S(backend_ep_factory_, CreateSyncStreamForDevice,
        memory_device, stream_options, stream);
}

Ort::Status ProviderFactory::CreateExternalResourceImporterForDevice(const OrtEpDevice* ep_device,
    OrtExternalResourceImporterImpl** out_importer) const
{
    BACKEND_CALL_S(backend_ep_factory_, CreateExternalResourceImporterForDevice,
        ep_device, out_importer);
}

Ort::Status ProviderFactory::GetHardwareDeviceIncompatibilityDetails(const OrtHardwareDevice* device,
    OrtDeviceEpIncompatibilityDetails* details) const
{
    BACKEND_CALL_S(backend_ep_factory_, GetHardwareDeviceIncompatibilityDetails,
        device, details);
}

Ort::Status ProviderFactory::GetNumCustomOpDomains(size_t* num_domains) const {
    // Forward to the directml backend factory so ORT registers com.microsoft schemas
    // (e.g. GroupNorm, SkipLayerNormalization) that the directml EP claims during GetCapability.
    // The backend factory is available at factory init time, before any session is created.
    if (dml_ep_factory_ != nullptr && dml_ep_factory_->GetNumCustomOpDomains != nullptr) {
        RETURN_IF_ERROR(dml_ep_factory_->GetNumCustomOpDomains(dml_ep_factory_, num_domains));
        return STATUS_OK;
    }
    *num_domains = 0;
    return STATUS_OK;
}

Ort::Status ProviderFactory::GetCustomOpDomains(OrtCustomOpDomain** domains, size_t num_domains) const {
    if (dml_ep_factory_ != nullptr && dml_ep_factory_->GetCustomOpDomains != nullptr) {
        RETURN_IF_ERROR(dml_ep_factory_->GetCustomOpDomains(dml_ep_factory_, domains, num_domains));
    }
    return STATUS_OK;
}

}  // namespace gpu_ep

extern "C" {

OrtStatus* CreateEpFactories(const char* registration_name, const OrtApiBase* ort_api_base,
    const OrtLogger* default_logger, OrtEpFactory** factories, size_t max_factories, size_t* num_factories)
{
    if (registration_name == nullptr || *registration_name == 0) {
        RETURN_STATUS(ORT_INVALID_ARGUMENT, "invalid or empty registration name");
    }
    if (ort_api_base == nullptr) {
        RETURN_STATUS(ORT_INVALID_ARGUMENT, "invalid base API pointer");
    }
    if (factories == nullptr || num_factories == nullptr || max_factories == 0) {
        RETURN_STATUS(ORT_INVALID_ARGUMENT, "invalid factories return buffer or capacity too small");
    }
    try {
#ifdef _WIN32
        HMODULE module = nullptr;
        if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                  GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              static_cast<LPCSTR>(static_cast<void*>(CreateEpFactories)),
                              &module) != 0) {
            std::vector<wchar_t> buffer;
            for (;;) {
                buffer.resize(buffer.size() + MAX_PATH);
                if (const auto writen{GetModuleFileNameW(module, buffer.data(),
                        static_cast<DWORD>(buffer.size()))}; writen < buffer.size()) {
                    break;
                }
            }
            fs::path path(buffer.begin(), buffer.end());
            SetDllDirectoryW(path.parent_path().native().c_str());
        }
#endif
        const OrtApi* ort_api{ort_api_base->GetApi(ORT_API_VERSION)};
        const OrtEpApi* ep_api{ort_api->GetEpApi()};
        const OrtModelEditorApi* model_editor_api{ort_api->GetModelEditorApi()};

        Ort::InitApi(ort_api);

        *num_factories = 1;
        *factories = std::make_unique<gpu_ep::ProviderFactory>(ApiPtrs{*ort_api, *ep_api, *model_editor_api},
            ort_api_base, registration_name, default_logger).release();

        return nullptr;
    }
    catch (const std::exception& e) {
        RETURN_STATUS(ORT_EP_FAIL, e.what());
    }
}

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
    delete reinterpret_cast<gpu_ep::ProviderFactory*>(factory);
    return nullptr;
}

}  // extern "C"