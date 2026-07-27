// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Non-pooled allocator for HOST_ACCESSIBLE (CPU-writable, GPU-readable) decode inputs on DirectML.
// Backs each allocation with a D3D12 CUSTOM/L0/WRITE_COMBINE committed resource (ReBAR-free,
// system-memory), persistently Map()'d so Alloc() returns a CPU pointer OGA can write in place.
// The CPU pointer decodes back to its ID3D12Resource for live binding (DmlHostAccessibleAllocator
// owns the cpu_ptr -> resource map). Deliberately NOT pooled: decode inputs are tiny/few/long-lived,
// and pooling is the source of stale-handle bugs. The DEFAULT allocator (DmlBucketizedBufferAllocator)
// is untouched.

#pragma once

#include <mutex>
#include <unordered_map>

#include "dml_client.h"
#include "DmlExecutionProvider/DmlSubAllocator.h"
#include "plugin_dml_AllocationInfo.h"

namespace dml_ep {

class DmlHostAccessibleAllocator : public OrtAllocator {
public:
    DmlHostAccessibleAllocator(const OrtMemoryInfo* memory_info,
                               ID3D12Device* device,
                               std::unique_ptr<DmlSubAllocator>&& subAllocator);
    ~DmlHostAccessibleAllocator();

    // Returns the ID3D12Resource backing a CPU pointer previously returned by Alloc, or nullptr if
    // this allocator did not produce that pointer (caller then falls back to the DEFAULT allocator).
    // Used by the fused-graph bind path (DecodeResource).
    ID3D12Resource* TryGetResource(void* cpu_ptr) const;

    // Returns the PluginDmlAllocationInfo backing a CPU pointer (AddRef'd, caller releases), or
    // nullptr if not ours. Used by the per-node kernel path (GetAllocationFromDataPointer), which
    // needs the IUnknown allocation object, not just the raw resource.
    IUnknown* TryGetAllocation(void* cpu_ptr) const;

private:
    void* AllocImpl(size_t size);
    void FreeImpl(void* cpu_ptr);

    struct Allocation {
        Microsoft::WRL::ComPtr<DmlResourceWrapper> resource;      // owns the ID3D12Resource
        Microsoft::WRL::ComPtr<PluginDmlAllocationInfo> allocInfo; // owner=nullptr wrapper for decode paths
        void* mapped_cpu_ptr;
    };

    Microsoft::WRL::ComPtr<ID3D12Device> m_device;
    std::unique_ptr<DmlSubAllocator> m_subAllocator;
    const OrtMemoryInfo* m_memoryInfo;
    size_t m_nextId = 1;

    mutable std::mutex m_mutex;
    std::unordered_map<void*, Allocation> m_allocations;  // cpu_ptr -> {resource, allocInfo}
};

}  // namespace dml_ep
