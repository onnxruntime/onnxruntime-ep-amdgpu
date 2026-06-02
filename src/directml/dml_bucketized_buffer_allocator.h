// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "dml_client.h"

#include "dml_execution_context.h"
#include "plugin_dml_AllocationInfo.h"
#include "DmlExecutionProvider/DmlSubAllocator.h"
#include "DmlExecutionProvider/inc/DmlExecutionProvider.h"


namespace dml_ep {

class DmlSubAllocator;
class PluginDmlExecutionContext;

// Implements a Lotus allocator for D3D12 heap buffers, using a bucket allocation strategy. The allocator
// maintains a set of fixed-size buckets, with each bucket containing one or more D3D12 buffers of that fixed size.
// All requested allocation sizes are rounded up to the nearest bucket size, which ensures minimal fragmentation
// while providing an upper bound on the amount of memory "wasted" with each allocation.
class DmlBucketizedBufferAllocator : public OrtAllocator {
public:
    ~DmlBucketizedBufferAllocator();

    // Constructs a BucketizedBufferAllocator which allocates D3D12 committed resources with the specified heap
    // properties, resource flags, and initial resource state.
    DmlBucketizedBufferAllocator(
        ID3D12Device* device,
        PluginDmlExecutionContext* context,
        const D3D12_HEAP_PROPERTIES& heapProps,
        D3D12_HEAP_FLAGS heapFlags,
        D3D12_RESOURCE_FLAGS resourceFlags,
        D3D12_RESOURCE_STATES initialState,
        std::unique_ptr<DmlSubAllocator>&& subAllocator);

    // Returns the information associated with an opaque allocation handle returned by IAllocator::Alloc.
    const PluginDmlAllocationInfo* DecodeDataHandle(const void* opaqueHandle);

    void SetDefaultRoundingMode(AllocatorRoundingMode roundingMode);

    void* AllocImpl(size_t size);
    void* AllocImpl(size_t size, AllocatorRoundingMode roundingMode);
    void FreeImpl(void* p);


private:
    static const uint32_t c_minResourceSizeExponent = 16; // 2^16 = 64KB

    // The pool consists of a number of buckets, and each bucket contains a number of resources of the same size.
    // The resources in each bucket are always sized as a power of two, and each bucket contains resources twice
    // as large as the previous bucket.
    struct Resource {
        Microsoft::WRL::ComPtr<DmlResourceWrapper> resource;
        uint64_t resourceId;
    };

    struct Bucket {
        std::vector<Resource> resources;
    };

    static gsl::index GetBucketIndexFromSize(uint64_t size);
    static uint64_t GetBucketSizeFromIndex(gsl::index index);

    friend class PluginDmlAllocationInfo;
    void FreeResource(void* p, uint64_t resourceId);

    Microsoft::WRL::ComPtr<ID3D12Device> m_device;
    D3D12_HEAP_PROPERTIES m_heapProperties;
    D3D12_HEAP_FLAGS m_heapFlags;
    D3D12_RESOURCE_FLAGS m_resourceFlags;
    D3D12_RESOURCE_STATES m_initialState;

    std::vector<Bucket> m_pool;
    size_t m_currentAllocationId = 0;
    uint64_t m_currentResourceId = 0;

    // Unless specifically requested, allocation sizes are not rounded to enable pooling
    // until SetDefaultRoundingMode is called.  This should be done at completion of session
    // initialization.
    AllocatorRoundingMode m_defaultRoundingMode = AllocatorRoundingMode::Disabled;

    Microsoft::WRL::ComPtr<PluginDmlExecutionContext> m_context;
    std::unique_ptr<DmlSubAllocator> m_subAllocator;
    OrtMemoryInfo m_memoryInfo;

#ifndef NDEBUG
    // Useful for debugging; keeps track of all allocations that haven't been freed yet
    std::map<size_t, PluginDmlAllocationInfo*> m_outstandingAllocationsById;
#endif
};

}  // namespace dml_ep
