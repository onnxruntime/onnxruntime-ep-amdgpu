// Copyright (c) Advance Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "dml_data_transfer.h"

#include "dml_ep.h"
#include "dml_execution_provider.h"
#include "dml_bucketized_buffer_allocator.h"
#include "plugin_dml_AllocationInfo.h"

namespace dml_ep {

DMLDataTransfer::DMLDataTransfer(ApiPtrs api_ptrs) : ApiPtrs(api_ptrs)
{
    CanCopy = CanCopyImpl;
    CopyTensors = CopyTensorsImpl;
    Release = ReleaseImpl;
}

void DMLDataTransfer::AttachExecutionProvider(std::shared_ptr<PluginDmlExecutionProviderImpl> ep)
{
    m_executionProvider = ep;
}

void DMLDataTransfer::AttachFactoryEpRef(ExecutionProviderPlugin** ep_raw_ref)
{
    m_ep_raw_ref = ep_raw_ref;
}

void DMLDataTransfer::RegisterProvider(const std::shared_ptr<PluginDmlExecutionProviderImpl>& ep)
{
    if (!ep) return;
    std::lock_guard<std::mutex> lock(m_providersMutex);
    for (const auto& p : m_providers) {
        if (p.get() == ep.get()) return;  // already registered
    }
    m_providers.push_back(ep);
}

void DMLDataTransfer::UnregisterProvider(const PluginDmlExecutionProviderImpl* ep)
{
    if (!ep) return;
    std::lock_guard<std::mutex> lock(m_providersMutex);
    for (auto it = m_providers.begin(); it != m_providers.end(); ++it) {
        if (it->get() == ep) { m_providers.erase(it); return; }
    }
}

// Resolve the EP that owns the first GPU tensor by decoding its allocation handle to the owning
// bucketized allocator, then matching that allocator to a registered EP. A single CopyTensors call
// operates on tensors from one session, so the first GPU tensor determines the owner. Returns the
// attached provider when no GPU tensor is present or the owner can't be matched.
std::shared_ptr<PluginDmlExecutionProviderImpl> DMLDataTransfer::ResolveOwningProvider(
    const OrtValue* const* src_tensors, OrtValue* const* dst_tensors, size_t num_tensors)
{
    auto ownerOf = [this](const OrtValue* value) -> DmlBucketizedBufferAllocator* {
        if (!value) return nullptr;
        const OrtMemoryDevice* dev = ep_api.Value_GetMemoryDevice(value);
        if (!dev || ep_api.MemoryDevice_GetDeviceType(dev) != OrtMemoryInfoDeviceType_GPU) return nullptr;
        // For a GPU tensor, the data pointer IS the PluginDmlAllocationInfo* handle (DecodeDataHandle
        // only static_casts it). Its GetOwner() is the allocator that made it.
        void* data = nullptr;
        if (ort_api.GetTensorMutableData(const_cast<OrtValue*>(value), &data) != nullptr || !data) return nullptr;
        return static_cast<PluginDmlAllocationInfo*>(data)->GetOwner();
    };

    DmlBucketizedBufferAllocator* owner = nullptr;
    for (size_t i = 0; i < num_tensors && !owner; ++i) {
        owner = ownerOf(src_tensors ? src_tensors[i] : nullptr);
        if (!owner) owner = ownerOf(dst_tensors ? dst_tensors[i] : nullptr);
    }
    if (!owner) return m_executionProvider;

    std::lock_guard<std::mutex> lock(m_providersMutex);
    for (const auto& p : m_providers) {
        if (p && p->GetBucketizedAllocator() == owner) return p;
    }
    return m_executionProvider;  // fall back if not found (shouldn't happen for a live session)
}

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
        impl.m_executionProvider = (*impl.m_ep_raw_ref)->GetInternalExecutionProvider();
    }

    // Route the copy to the EP that owns the tensors, not a single cached provider. A shared
    // DMLDataTransfer services multiple sessions, each with its own ExecutionContext/queue/fence;
    // a copy must run on the owning session's context so its readback waits on the fence that
    // produced the data.
    std::shared_ptr<PluginDmlExecutionProviderImpl> provider =
        impl.ResolveOwningProvider(src_tensors_ptr, dst_tensors_ptr, num_tensors);

    if (!provider) {
        return impl.ort_api.CreateStatus(ORT_FAIL, "DMLDataTransfer: execution provider not attached");
    }

    provider->CopyTensorsPlugin(src_tensors_ptr, dst_tensors_ptr, streams_ptr, num_tensors);

    return nullptr;
}

/*static*/
void ORT_API_CALL DMLDataTransfer::ReleaseImpl(OrtDataTransferImpl* this_ptr) noexcept {
    // Factory owns this object and manages its lifetime via dml_data_transfer_implementation unique_ptr.
    // ORT's Release call is intentionally ignored — the factory destructor handles cleanup.
}

}  // namespace dml_ep