// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cstdint>
#include <wrl/client.h>
#include <wrl/implements.h>
#include "d3d12.h"
#include "DmlExecutionProvider/DmlResourceWrapper.h"
#include <utility>

namespace dml_ep {

    class DmlBucketizedBufferAllocator;

    class PluginDmlAllocationInfo : public Microsoft::WRL::RuntimeClass<
        Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, IUnknown>
    {
    public:
        PluginDmlAllocationInfo(
            DmlBucketizedBufferAllocator* owner,
            size_t id,
            uint64_t pooledResourceId,
            DmlResourceWrapper* resourceWrapper,
            size_t requestedSize)
            : m_owner(owner)
            , m_allocationId(id)
            , m_pooledResourceId(pooledResourceId)
            , m_resourceWrapper(resourceWrapper)
            , m_requestedSize(requestedSize)
        {}

        ~PluginDmlAllocationInfo();

        DmlBucketizedBufferAllocator* GetOwner() const
        {
            return m_owner;
        }

        ID3D12Resource* GetResource() const
        {
            return m_resourceWrapper->GetD3D12Resource();
        }

        Microsoft::WRL::ComPtr<DmlResourceWrapper> DetachResourceWrapper() const
        {
            return std::move(m_resourceWrapper);
        }

        size_t GetRequestedSize() const
        {
            return m_requestedSize;
        }

        size_t GetId() const
        {
            return m_allocationId;
        }

        uint64_t GetPooledResourceId() const
        {
            return m_pooledResourceId;
        }

    private:
        // The bucketized buffer allocator must outlive the allocation info
        DmlBucketizedBufferAllocator* m_owner;
        size_t m_allocationId; // For debugging purposes
        uint64_t m_pooledResourceId = 0;
        Microsoft::WRL::ComPtr<DmlResourceWrapper> m_resourceWrapper;

        // The size requested during Alloc(), which may be smaller than the physical resource size
        size_t m_requestedSize;
    };

}  // namespace dml_ep
