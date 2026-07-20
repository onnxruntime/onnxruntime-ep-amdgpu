// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "DmlSubAllocator.h"

namespace dml_ep {

    struct DmlResourceWrapper;

    class DmlCommittedResourceAllocator : public DmlSubAllocator
    {
    public:
        // Defaults to DEFAULT heap (GPU-exclusive VRAM). Pass CUSTOM/L0/WRITE_COMBINE heap props for
        // CPU-writable, GPU-readable buffers (host-accessible decode inputs) — works WITHOUT ReBAR,
        // unlike GPU_UPLOAD. Read cost for KB-sized inputs measured ~= VRAM (dmlmembench).
        DmlCommittedResourceAllocator(ID3D12Device* device,
                                      D3D12_HEAP_PROPERTIES heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT))
            : m_device(device), m_heapProps(heapProps) {}
        Microsoft::WRL::ComPtr<DmlResourceWrapper> Alloc(size_t size) final;

    private:
        ID3D12Device* m_device = nullptr;
        D3D12_HEAP_PROPERTIES m_heapProps;
    };

}  // namespace dml_ep
