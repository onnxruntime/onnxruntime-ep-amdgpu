// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "common/parse_string.h"
#include "common/plugin_ep_utils.h"

#include "hip/utils.h"
#include "hip/stream_support.h"

#include "mgx_ep.h"
#include "mgx_factory.h"
#include "mgx_interop.h"

#include "mgx_kernel_reg.h"

namespace mgx_ep {

ProviderFactory::ProviderFactory(const ApiPtrs& api_ptrs, const char* ep_name, const Ort::Logger& default_logger)
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
            {ep_devices, max_ep_devices}, *num_ep_devices);
    };
    OrtEpFactory::CreateEp = [](OrtEpFactory* this_, const OrtHardwareDevice* const* devices,
            const OrtKeyValuePairs* const* ep_metadata_pairs, size_t num_devices,
            const OrtSessionOptions* session_options, const OrtLogger* logger, OrtEp** ep) noexcept
    {
        API_CALL_S(ProviderFactory, this_, CreateEp, {devices, devices + num_devices},
            {ep_metadata_pairs, ep_metadata_pairs + num_devices},
            Ort::ConstSessionOptions{session_options}, Ort::Logger{logger}, *ep);
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
    // OrtEpFactory::ValidateCompiledModelCompatibilityInfo = [](OrtEpFactory* this_,
    //         const OrtHardwareDevice* const* devices, size_t num_devices, const char* compatibility_info,
    //         OrtCompiledModelCompatibility* model_compatibility) noexcept {
    //     API_CALL_S(ProviderFactory, this_, ValidateCompiledModelCompatibilityInfo, devices, num_devices,
    //         compatibility_info, model_compatibility);
    // };
    OrtEpFactory::CreateAllocator = [](OrtEpFactory* this_, const OrtMemoryInfo* memory_info,
            const OrtKeyValuePairs* allocator_options, OrtAllocator** allocator) noexcept {
        API_CALL_S(ProviderFactory, this_, CreateAllocator, Ort::ConstMemoryInfo{memory_info},
            Ort::ConstKeyValuePairs{allocator_options}, *allocator);
    };
    OrtEpFactory::ReleaseAllocator = [](OrtEpFactory* this_, OrtAllocator* allocator) noexcept {
        API_CALL_V(ProviderFactory, this_, ReleaseAllocator, allocator);
    };
    OrtEpFactory::CreateDataTransfer = [](OrtEpFactory* this_, OrtDataTransferImpl** data_transfer) noexcept {
        API_CALL_S(ProviderFactory, this_, CreateDataTransfer, *data_transfer);
    };
    OrtEpFactory::IsStreamAware = [](const OrtEpFactory* this_) noexcept {
        API_CALL_T(const ProviderFactory, this_, IsStreamAware, false);
    };
    OrtEpFactory::CreateSyncStreamForDevice = [](OrtEpFactory* this_, const OrtMemoryDevice* memory_device,
            const OrtKeyValuePairs* stream_options, OrtSyncStreamImpl** stream) noexcept {
        API_CALL_S(ProviderFactory, this_, CreateSyncStreamForDevice, memory_device,
            Ort::ConstKeyValuePairs{stream_options}, *stream);
    };
    // OrtEpFactory::GetHardwareDeviceIncompatibilityDetails = [](OrtEpFactory* this_, const OrtHardwareDevice* hw,
    //         OrtDeviceEpIncompatibilityDetails* details) noexcept {
    //     API_CALL_S(ProviderFactory, this_, GetHardwareDeviceIncompatibilityDetails, hw, details);
    // };
    OrtEpFactory::CreateExternalResourceImporterForDevice = [](OrtEpFactory* this_, const OrtEpDevice* ep_device,
            OrtExternalResourceImporterImpl** out_importer) noexcept {
        API_CALL_S(ProviderFactory, this_, CreateExternalResourceImporterForDevice,
            Ort::ConstEpDevice{ep_device}, *out_importer);
    };
    // OrtEpFactory::GetNumCustomOpDomains = [](OrtEpFactory* this_, size_t* num_domains) noexcept {
    //     API_CALL_S(ProviderFactory, this_, GetNumCustomOpDomains, num_domains);
    // };
    // OrtEpFactory::GetCustomOpDomains = [](OrtEpFactory* this_, OrtCustomOpDomain** domains,
    //         const size_t num_domains) noexcept {
    //     API_CALL_S(ProviderFactory, this_, GetCustomOpDomains, domains, num_domains);
    // };

    data_transfer_ = std::make_unique<hip::DataTransfer>(*this);
}

const char* ProviderFactory::GetName() const {
    return ep_name_.c_str();
}

const char* ProviderFactory::GetVendor() const {
    return amd::Vendor;
}

Ort::Status ProviderFactory::GetSupportedDevices(const std::vector<Ort::ConstHardwareDevice>& devices,
    const gsl::span<OrtEpDevice*>& ep_devices, size_t& num_ep_devices)
try {
    int num_hip_devices{};
    HIP_RETURN_IF_ERROR(hipGetDeviceCount(&num_hip_devices));
    for (int i{}; i < num_hip_devices; ++i) {
        OrtMemoryInfo* memory_info{};
        RETURN_IF_ERROR(ort_api.CreateMemoryInfo_V2("Hip", OrtMemoryInfoDeviceType_GPU, amd::VendorId,
            i, OrtDeviceMemoryType_DEFAULT, 0, OrtDeviceAllocator, &memory_info));
        gpu_memory_infos_.insert_or_assign(i, MemoryInfoPtr{memory_info, ort_api.ReleaseMemoryInfo});
        RETURN_IF_ERROR(ort_api.CreateMemoryInfo_V2("HipPinned", OrtMemoryInfoDeviceType_GPU, amd::VendorId,
            i, OrtDeviceMemoryType_HOST_ACCESSIBLE, 0, OrtDeviceAllocator, &memory_info));
        pinned_memory_infos_.insert_or_assign(i, MemoryInfoPtr{memory_info, ort_api.ReleaseMemoryInfo});
    }
    num_ep_devices = 0;
    int device_id{};
    for (size_t i{}; i < devices.size(); ++i) {
        if (device_id >= num_hip_devices || num_ep_devices >= ep_devices.size()) {
            break;
        }
        Ort::ConstHardwareDevice device(devices[i]);
        if (device.VendorId() == amd::VendorId && device.Type() == OrtHardwareDeviceType_GPU) {
            const auto metadata{device.Metadata().GetKeyValuePairs()};
            hipDeviceProp_t device_prop;
            HIP_RETURN_IF_ERROR(hipGetDeviceProperties(&device_prop, device_id));
#ifdef _WIN32
            const std::string_view luid{metadata.at("LUID")};
            if (ToInteger<size_t>(luid) != *reinterpret_cast<size_t*>(&device_prop.luid[0])) {
                continue;
            }
#else
            // TODO: required metadata for unique device identification on Linux operating system.
#endif
            Ort::KeyValuePairs ep_metadata;
            ep_metadata.Add("arch", device_prop.gcnArchName);
            Ort::KeyValuePairs ep_options;
            Ort::EpDevice ep_device{*this, device, ep_metadata.GetConst(), ep_options.GetConst()};
            const OrtMemoryInfo* pinned_memory_info{pinned_memory_infos_.at(device_id).get()};
            RETURN_IF_ERROR(ep_api.EpDevice_AddAllocatorInfo(ep_device, pinned_memory_info));
            pinned_memory_devices_.emplace_back(ep_api.MemoryInfo_GetMemoryDevice(pinned_memory_info));

            const OrtMemoryInfo* gpu_memory_info{gpu_memory_infos_.at(device_id).get()};
            RETURN_IF_ERROR(ep_api.EpDevice_AddAllocatorInfo(ep_device, gpu_memory_info));
            gpu_memory_devices_.emplace_back(ep_api.MemoryInfo_GetMemoryDevice(gpu_memory_info));

            ep_devices[num_ep_devices++] = ep_device.release();
            ++device_id;
        }
    }
    return STATUS_OK;
} catch (const Ort::Exception& e) {
    return Ort::Status{e};
} catch (const std::exception& e) {
    return Ort::Status{e.what(), ORT_EP_FAIL};
}

Ort::Status ProviderFactory::CreateEp(const std::vector<Ort::ConstHardwareDevice>& devices,
    const std::vector<Ort::ConstKeyValuePairs>& /* ep_metadata_pairs */,
    const Ort::ConstSessionOptions& session_options, const Ort::Logger& logger, OrtEp* &ep)
try {
    ep = nullptr;
    if (devices.size() > 1) {
        return MAKE_STATUS(ORT_INVALID_ARGUMENT,
            "MIGraphX EP supports selection for a single or none devices");
    }
    ep = std::make_unique<ExecutionProvider>(*this, ep_name_, session_options, logger).release();
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

// Ort::Status ProviderFactory::ValidateCompiledModelCompatibilityInfo(const OrtHardwareDevice* const* devices,
//     size_t num_devices, const char* compatibility_info, OrtCompiledModelCompatibility* model_compatibility)
// try {
//     return STATUS_OK;
// } catch (const Ort::Exception& e) {
//     return Ort::Status{e};
// } catch (const std::exception& e) {
//     return Ort::Status{e.what(), ORT_EP_FAIL};
// }

Ort::Status ProviderFactory::CreateAllocator(const Ort::ConstMemoryInfo& memory_info,
    const Ort::ConstKeyValuePairs& /* allocator_options */, OrtAllocator* &allocator)
try {
    auto device_id{memory_info.GetDeviceId()};
    if (const auto memory_type{memory_info.GetDeviceMemoryType()}; memory_type == OrtDeviceMemoryType_DEFAULT) {
        if (const auto it{gpu_allocators_.find(device_id)}; it != gpu_allocators_.end()) {
            allocator = it->second.get();
        } else {
            allocator = gpu_allocators_.emplace(device_id,
                std::make_unique<hip::Allocator>(memory_info, device_id)).first->second.get();
        }
    } else if (memory_type == OrtDeviceMemoryType_HOST_ACCESSIBLE) {
        if (const auto it{pinned_allocators_.find(device_id)}; it != pinned_allocators_.end()) {
            allocator = it->second.get();
        } else {
            allocator = pinned_allocators_.emplace(device_id,
                std::make_unique<hip::PinnedAllocator>(memory_info, device_id)).first->second.get();
        }
    } else {
        return Ort::Status{"unsupported device memory type provided", ORT_ENGINE_ERROR};
    }
    return STATUS_OK;
} catch (const Ort::Exception& e) {
    return Ort::Status{e};
} catch (const std::exception& e) {
    return Ort::Status{e.what(), ORT_EP_FAIL};
}

void ProviderFactory::ReleaseAllocator(OrtAllocator* /* allocator */) {
    // no-op. The allocators are shared across sessions.
}

Ort::Status ProviderFactory::CreateDataTransfer(OrtDataTransferImpl*& data_transfer) const
try {
    data_transfer = data_transfer_.get();
    return STATUS_OK;
} catch (const Ort::Exception& e) {
    return Ort::Status{e};
} catch (const std::exception& e) {
    return Ort::Status{e.what(), ORT_EP_FAIL};
}

bool ProviderFactory::IsStreamAware() const {
    return true;
}

Ort::Status ProviderFactory::CreateSyncStreamForDevice(const OrtMemoryDevice* memory_device,
    const Ort::ConstKeyValuePairs& stream_options, OrtSyncStreamImpl*& stream)
try {
    stream = nullptr;
    if (memory_device == nullptr) {
        return ORT_MAKE_STATUS(ORT_INVALID_ARGUMENT, "memory_device cannot be nullptr");
    }
    const int device_id{static_cast<int>(ep_api.MemoryDevice_GetDeviceId(memory_device))};
    auto sync_stream{std::make_unique<hip::SyncStream>(*this, device_id, stream_options)};
    stream = sync_stream.release();
    return STATUS_OK;
} catch (const Ort::Exception& e) {
    return Ort::Status{e};
} catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ORT_EP_FAIL, e.what());
}

Ort::Status ProviderFactory::CreateExternalResourceImporterForDevice(const Ort::ConstEpDevice& ep_device,
    OrtExternalResourceImporterImpl*& out_importer) const
try {
    out_importer = nullptr;
    if (ep_device == nullptr) {
        return ORT_MAKE_STATUS(ORT_INVALID_ARGUMENT, "ep_device cannot be nullptr");
    }
    const auto memory_info{ep_device.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT)};
    if (memory_info == nullptr) {
        return ORT_MAKE_STATUS(ORT_EP_FAIL, "EP device does not expose default device memory info");
    }
    const auto* memory_device{ep_api.MemoryInfo_GetMemoryDevice(memory_info)};
    if (memory_device == nullptr) {
        return ORT_MAKE_STATUS(ORT_EP_FAIL, "EP device does not expose a memory device");
    }
    const int device_id{static_cast<int>(ep_api.MemoryDevice_GetDeviceId(memory_device))};
    auto importer{std::make_unique<ExternalResourceImporter>(*this, device_id)};
    out_importer = importer.release();
    return STATUS_OK;
} catch (const Ort::Exception& e) {
    return Ort::Status{e};
} catch (const std::exception& e) {
    return ORT_MAKE_STATUS(ORT_EP_FAIL, e.what());
}

// Ort::Status ProviderFactory::GetHardwareDeviceIncompatibilityDetails(const OrtHardwareDevice* hw,
//     OrtDeviceEpIncompatibilityDetails* details)
// try {
//     /* TODO: check for known incompatibility reasons between a hardware device and the execution provider
//      *       Use OrtEpApi::DeviceEpIncompatibilityDetails_SetDetails to update the 'details' information.
//      *       Leave it unchanged means no incompatibilities.
//      */
//     return STATUS_OK;
// } catch (const Ort::Exception& e) {
//     return Ort::Status{e};
// } catch (const std::exception& e) {
//     return Ort::Status{e.what(), ORT_EP_FAIL};
// }

// Ort::Status ProviderFactory::GetNumCustomOpDomains(size_t& num_domains) const
// try {
//     num_domains = 0;
//     return nullptr;
// } catch (const Ort::Exception& e) {
//     return Ort::Status{e};
// } catch (const std::exception& e) {
//     return Ort::Status{e.what(), ORT_EP_FAIL};
// }

// Ort::Status ProviderFactory::GetCustomOpDomains(OrtCustomOpDomain** domains, size_t num_domains) const
// try {
//     return STATUS_OK;
// } catch (const Ort::Exception& e) {
//     return Ort::Status{e};
// } catch (const std::exception& e) {
//     return Ort::Status{e.what(), ORT_EP_FAIL};
// }

Ort::Status ProviderFactory::GetKernelRegistry(std::string_view ep_name, const OrtKernelRegistry*& kernel_registry) const {
    kernel_registry = nullptr;
    if (GetNumRegisteredKernels() != 0) {
        if (kernel_registry_ == nullptr) {
            RETURN_IF_ERROR(CreateKernelRegistry(ep_name, data_transfer_.get(), kernel_registry_));
        }
        kernel_registry = kernel_registry_;
    }
    return STATUS_OK;
}

}  // namespace mgx_ep

// Public symbols

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
        ::SetEnvironmentVariable("MIGRAPHX_MLIR_USE_SPECIFIC_OPS", "dot,convolution,fused,attention");
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
        if (ort_api == nullptr) {
            RETURN_STATUS(ORT_EP_FAIL, "Ort API not available: ", ORT_API_VERSION);
        }
        const OrtEpApi* ep_api{ort_api->GetEpApi()};
        const OrtModelEditorApi* model_editor_api{ort_api->GetModelEditorApi()};

        Ort::InitApi(ort_api);

        *num_factories = 1;
        *factories = std::make_unique<mgx_ep::ProviderFactory>(ApiPtrs{*ort_api, *ep_api, *model_editor_api},
            registration_name, Ort::Logger{default_logger}).release();

        return nullptr;
    }
    catch (const std::exception& e) {
        RETURN_STATUS(ORT_EP_FAIL, e.what());
    }
}

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
    delete reinterpret_cast<mgx_ep::ProviderFactory*>(factory);
    return nullptr;
}

}  // extern "C"
