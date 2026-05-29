// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include <mutex>
#include "plugin_ep_utils.h"
#define INITGUID
#include <guiddef.h>
#include <dxcore.h>
#undef INITGUID

#include <d3d12.h>
#include <dxgi1_6.h>
#include <directx/d3dx12.h>
#include <DirectML.h>

#include <wrl/client.h>
#define IID_GRAPHICS_PPV_ARGS IID_PPV_ARGS
using Microsoft::WRL::ComPtr;

#include <wil/wrl.h>
#include <wil/result.h>
#include "dml_data_transfer.h"
#include "cpu_allocator.h"
#include "dml_ep.h"

namespace Dml {

class ProviderFactory : public OrtEpFactory, public ApiPtrs
{
public:
    ProviderFactory(const ApiPtrs& api_ptrs, std::string_view ep_name, const Ort::Logger& default_logger);
    ~ProviderFactory() = default;

    const Ort::Logger& default_logger_;

private:
    static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr) noexcept;

    static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr) noexcept;
    static uint32_t ORT_API_CALL GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept;

    static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory* this_ptr) noexcept;

    Ort::Status GetSupportedDevices(const std::vector<Ort::ConstHardwareDevice>& devices,
        const gsl::span<OrtEpDevice*>& ep_devices, size_t& num_ep_devices) noexcept;

    static OrtStatus* ORT_API_CALL CreateEpImpl(OrtEpFactory* this_ptr,
                                                const OrtHardwareDevice* const* devices,
                                                const OrtKeyValuePairs* const* ep_metadata,
                                                size_t num_devices,
                                                const OrtSessionOptions* session_options,
                                                const OrtLogger* logger,
                                                OrtEp** ep) noexcept;

    static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept;

    static OrtStatus* ORT_API_CALL CreateAllocatorImpl(OrtEpFactory* this_ptr,
                                                       const OrtMemoryInfo* memory_info,
                                                       const OrtKeyValuePairs* /*allocator_options*/,
                                                       OrtAllocator** allocator) noexcept;

    static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory* /*this*/, OrtAllocator* allocator) noexcept;

    static OrtStatus* ORT_API_CALL CreateDataTransferImpl(OrtEpFactory* this_ptr,
                                                          OrtDataTransferImpl** data_transfer) noexcept;

    static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory* this_ptr) noexcept;

    static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(OrtEpFactory* this_ptr,
                                                                 const OrtMemoryDevice* memory_device,
                                                                 const OrtKeyValuePairs* stream_options,
                                                                 OrtSyncStreamImpl** stream) noexcept;

    static OrtStatus* ORT_API_CALL CreateExternalResourceImporterForDeviceImpl(
        OrtEpFactory* this_ptr, const OrtEpDevice* ep_device, OrtExternalResourceImporterImpl** out_importer) noexcept;

    static OrtStatus* ORT_API_CALL GetHardwareDeviceIncompatibilityDetailsImpl(
        OrtEpFactory* this_ptr, const OrtHardwareDevice* hw, OrtDeviceEpIncompatibilityDetails* details) noexcept;

    static OrtStatus* ORT_API_CALL GetNumCustomOpDomainsImpl(OrtEpFactory* this_ptr,
                                                             _Out_ size_t* num_domains) noexcept;

    static OrtStatus* ORT_API_CALL GetCustomOpDomainsImpl(OrtEpFactory* this_ptr,
                                                          _Outptr_result_maybenull_ OrtCustomOpDomain** domains,
                                                          _Out_ size_t num_domains) noexcept;

    void CreateDMLAndD3DResources();
    ComPtr<ID3D12Device> CreateD3d12Device();
    ComPtr<ID3D12CommandQueue> CreateCommandQueue(const ComPtr<ID3D12Device>& device);
    ComPtr<IDMLDevice> CreateDMLDevice(const ComPtr<ID3D12Device>& device);
    bool IsCpuAllocator(const OrtMemoryInfo* memory_info);
    bool IsGpuAllocator(const OrtMemoryInfo* memory_info);
    std::vector<ComPtr<IDXCoreAdapter>> GetAdapters();
    void CreateD3DDeviceFromAdapter(IDXCoreAdapter* adapter, ComPtr<ID3D12Device>& device);

    std::string ep_name_{};
    const std::string vendor_{amd::Vendor};
    static constexpr uint32_t vendor_id_{amd::VendorId};
    const std::string ep_version_{"0.1.0"};

    Ort::MemoryInfo bucketized_buffer_memory_info_;
    Ort::MemoryInfo readonly_memory_info_;
    Ort::MemoryInfo cpu_input_allocator_;

    ComPtr<ID3D12Device> d3d12_device;
    ComPtr<ID3D12CommandQueue> cmd_queue;
    ComPtr<IDMLDevice> dml_device;

    std::unique_ptr<ExecutionProviderPlugin> m_ep;
    ExecutionProviderPlugin* m_ep_raw = nullptr; // non-owning observer pointer
    std::unique_ptr<DMLDataTransfer> dml_data_transfer_implementation; // factory-owned, shared across sessions
};


}  // namespace Dml
