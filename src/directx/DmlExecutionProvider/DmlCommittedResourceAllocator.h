// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "DmlSubAllocator.h"

namespace dml_ep {

    struct DmlResourceWrapper;

    class DmlCommittedResourceAllocator : public DmlSubAllocator
    {
    public:
        // Defaults to DEFAULT heap (VRAM). Pass CUSTOM/L0 heap props for host-accessible buffers.
        DmlCommittedResourceAllocator(ID3D12Device* device,
                                      D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT))
            : m_device(device), m_heapProps(heapProps) {}
        Microsoft::WRL::ComPtr<DmlResourceWrapper> Alloc(size_t size) final;

    private:
        ID3D12Device* m_device = nullptr;
        D3D12_HEAP_PROPERTIES m_heapProps;
    };

}  // namespace dml_ep
