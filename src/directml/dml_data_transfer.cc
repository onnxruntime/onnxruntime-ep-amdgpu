// Copyright (c) Advance Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "dml_data_transfer.h"
#include <gsl/span>


DMLDataTransfer::DMLDataTransfer(ApiPtrs api_ptrs) : ApiPtrs(api_ptrs)
{
    CanCopy = CanCopyImpl;
    CopyTensors = CopyTensorsImpl;
    Release = ReleaseImpl;

}

void DMLDataTransfer::AttachExecutionProvider(std::shared_ptr<Dml::PluginDmlExecutionProviderImpl> ep)
{
    m_executionProvider = ep;
}

void DMLDataTransfer::AttachFactoryEpRef(Dml::ExecutionProviderPlugin** ep_raw_ref)
{
    m_ep_raw_ref = ep_raw_ref;
}

/*static*/
bool ORT_API_CALL DMLDataTransfer::CanCopyImpl(const OrtDataTransferImpl* this_ptr,
                                               const OrtMemoryDevice* src_memory_device,
                                               const OrtMemoryDevice* dst_memory_device) noexcept
{
    const auto& impl = *static_cast<const DMLDataTransfer*>(this_ptr);

    OrtMemoryInfoDeviceType src_type = impl.ep_api.MemoryDevice_GetDeviceType(src_memory_device);
    OrtMemoryInfoDeviceType dst_type = impl.ep_api.MemoryDevice_GetDeviceType(dst_memory_device);

    bool is_src_device_AMD = impl.ep_api.MemoryDevice_GetVendorId(src_memory_device) == 0x1002;
    bool is_dst_device_AMD = impl.ep_api.MemoryDevice_GetVendorId(dst_memory_device) == 0x1002;

    if ((src_type == OrtMemoryInfoDeviceType_GPU && is_src_device_AMD == true) ||
        (dst_type == OrtMemoryInfoDeviceType_GPU && is_dst_device_AMD == true)) {
        return true;
    }

    return (src_type == OrtMemoryInfoDeviceType_GPU && dst_type == OrtMemoryInfoDeviceType_GPU) ||
        (src_type == OrtMemoryInfoDeviceType_GPU && dst_type == OrtMemoryInfoDeviceType_CPU) ||
        (src_type == OrtMemoryInfoDeviceType_CPU && dst_type == OrtMemoryInfoDeviceType_GPU);
}


// function to copy one or more tensors.
OrtStatus* ORT_API_CALL DMLDataTransfer::CopyTensorsImpl(OrtDataTransferImpl* this_ptr,
                                                         const OrtValue** src_tensors_ptr, OrtValue** dst_tensors_ptr,
                                                         OrtSyncStream** streams_ptr, size_t num_tensors) noexcept
{
    DMLDataTransfer& impl = *static_cast<DMLDataTransfer*>(this_ptr);

    // Lazy attach: if the EP was not attached at construction time (factory-level transfer
    // created before the EP instance), resolve it now via the pointer to the factory's m_ep_raw.
    if (!impl.m_executionProvider && impl.m_ep_raw_ref && *impl.m_ep_raw_ref) {
        impl.m_executionProvider = (*impl.m_ep_raw_ref)->GetInternetalExecutionProvider();
    }

    if (!impl.m_executionProvider) {
        return impl.ort_api.CreateStatus(ORT_FAIL, "DMLDataTransfer: execution provider not attached");
    }

    impl.m_executionProvider->CopyTensorsPlugin(src_tensors_ptr, dst_tensors_ptr, streams_ptr, num_tensors);

    return nullptr;
}

/*static*/
void ORT_API_CALL DMLDataTransfer::ReleaseImpl(OrtDataTransferImpl* this_ptr) noexcept {
    // Factory owns this object and manages its lifetime via dml_data_transfer_implementation unique_ptr.
    // ORT's Release call is intentionally ignored — the factory destructor handles cleanup.
}
