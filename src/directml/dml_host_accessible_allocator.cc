// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "dml_host_accessible_allocator.h"

#include "DmlExecutionProvider/DmlResourceWrapper.h"
#include "DmlExecutionProvider/ErrorHandling.h"

namespace dml_ep {

DmlHostAccessibleAllocator::DmlHostAccessibleAllocator(
    const OrtMemoryInfo* memory_info,
    ID3D12Device* device,
    std::unique_ptr<DmlSubAllocator>&& subAllocator)
    : m_device(device), m_subAllocator(std::move(subAllocator)), m_memoryInfo(memory_info)
{
    OrtAllocator::version = ORT_API_VERSION;
    OrtAllocator::Alloc = [](OrtAllocator* self, size_t size) -> void* {
        return static_cast<DmlHostAccessibleAllocator*>(self)->AllocImpl(size);
    };
    OrtAllocator::Free = [](OrtAllocator* self, void* p) {
        static_cast<DmlHostAccessibleAllocator*>(self)->FreeImpl(p);
    };
    OrtAllocator::Info = [](const OrtAllocator* self) -> const OrtMemoryInfo* {
        return static_cast<const DmlHostAccessibleAllocator*>(self)->m_memoryInfo;
    };
    OrtAllocator::Reserve = [](OrtAllocator* self, size_t size) -> void* {
        return static_cast<DmlHostAccessibleAllocator*>(self)->AllocImpl(size);
    };
    OrtAllocator::GetStats = nullptr;
    OrtAllocator::AllocOnStream = nullptr;
}

DmlHostAccessibleAllocator::~DmlHostAccessibleAllocator()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    // Persistent maps are released when each resource (held by the wrapper) is destroyed; explicit
    // Unmap is optional but tidy. ComPtr release of the wrappers frees the D3D12 resources.
    for (auto& kv : m_allocations) {
        if (auto* res = kv.second.resource ? kv.second.resource->GetD3D12Resource() : nullptr) {
            res->Unmap(0, nullptr);
        }
    }
    m_allocations.clear();
}

void* DmlHostAccessibleAllocator::AllocImpl(size_t size)
{
    // Create a CUSTOM/L0/WRITE_COMBINE committed resource (heap props baked into the sub-allocator).
    Microsoft::WRL::ComPtr<DmlResourceWrapper> wrapper = m_subAllocator->Alloc(size);
    ID3D12Resource* resource = wrapper->GetD3D12Resource();

    // Persistently map it — valid for a CPU-accessible heap for the resource's whole lifetime.
    void* cpu_ptr = nullptr;
    ORT_THROW_IF_FAILED(resource->Map(0, nullptr, &cpu_ptr));

    // Build a PluginDmlAllocationInfo (owner=nullptr so its dtor no-ops — we own the resource via the
    // wrapper) so the per-node kernel decode path (GetAllocationFromDataPointer -> GetABIDataInterface)
    // gets the IUnknown allocation object it expects, and GetResource() works for the fused-bind path.
    Microsoft::WRL::ComPtr<PluginDmlAllocationInfo> allocInfo;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        allocInfo = wil::MakeOrThrow<PluginDmlAllocationInfo>(
            /*owner=*/nullptr, /*id=*/m_nextId++, /*pooledResourceId=*/0, wrapper.Get(), size);
        m_allocations[cpu_ptr] = Allocation{std::move(wrapper), std::move(allocInfo), cpu_ptr};
    }
    return cpu_ptr;  // OGA writes decode inputs directly here; decodes back to the resource on bind.
}

void DmlHostAccessibleAllocator::FreeImpl(void* cpu_ptr)
{
    if (cpu_ptr == nullptr) return;
    Microsoft::WRL::ComPtr<DmlResourceWrapper> to_release;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_allocations.find(cpu_ptr);
        if (it == m_allocations.end()) return;  // not ours (shouldn't happen; ORT frees via Info())
        if (auto* res = it->second.resource ? it->second.resource->GetD3D12Resource() : nullptr) {
            res->Unmap(0, nullptr);
        }
        to_release = std::move(it->second.resource);
        m_allocations.erase(it);
    }
    // to_release drops here (outside the lock) -> D3D12 resource destroyed.
}

ID3D12Resource* DmlHostAccessibleAllocator::TryGetResource(void* cpu_ptr) const
{
    // Exact-match only: assumes callers bind the Alloc() base pointer, not a base+offset sub-view.
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_allocations.find(cpu_ptr);
    if (it == m_allocations.end()) {
        return nullptr;
    }
    return it->second.resource ? it->second.resource->GetD3D12Resource() : nullptr;
}

IUnknown* DmlHostAccessibleAllocator::TryGetAllocation(void* cpu_ptr) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_allocations.find(cpu_ptr);
    if (it == m_allocations.end() || !it->second.allocInfo) {
        return nullptr;
    }
    IUnknown* p = it->second.allocInfo.Get();
    p->AddRef();  // caller owns the reference (matches GetAllocationFromDataPointer contract)
    return p;
}

}  // namespace dml_ep
