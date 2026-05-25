// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#define INITGUID
#include <guiddef.h>
#include <dxcore.h>
#undef INITGUID

#include <d3d12.h>
#include <dxgi1_6.h>
#include <directx/d3dx12.h>
#include <DirectML.h>
#include <gsl/span>

#include <assert.h>
#include <wil/wrl.h>
#include <wil/result.h>
#include <wrl/client.h>

#define IID_GRAPHICS_PPV_ARGS IID_PPV_ARGS
using Microsoft::WRL::ComPtr;

#include "dml_execution_context.h"

namespace Dml
{
    class PluginDmlExecutionContext;

    // Because we never perform more than one readback at a time, we don't need anything fancy for managing the
    // readback heap - just maintain a single resource and reallocate it if it's not big enough.
    class PluginDmlReadbackHeap
    {
    public:
        PluginDmlReadbackHeap(ID3D12Device* device, PluginDmlExecutionContext* executionContext);

        // Copies data from the specified GPU resource into CPU memory pointed-to by the span. This method will block
        // until the copy is complete.
        void ReadbackFromGpu(
            gsl::span<std::byte> dst,
            ID3D12Resource* src,
            uint64_t srcOffset,
            D3D12_RESOURCE_STATES srcState);

        // Overload supporting batching
        void ReadbackFromGpu(
            gsl::span<void*> dst,
            gsl::span<const uint32_t > dstSizes,
            gsl::span<ID3D12Resource*> src,
            D3D12_RESOURCE_STATES srcState);

        static ComPtr<ID3D12Resource> CreateReadbackHeap(ID3D12Device* device, size_t size);
    private:
        void EnsureReadbackHeap(size_t size);

        static constexpr size_t c_initialCapacity = 1024 * 1024; // 1MB

        ComPtr<ID3D12Device> m_device;
        ComPtr<PluginDmlExecutionContext> m_executionContext;

        ComPtr<ID3D12Resource> m_readbackHeap;
        size_t m_capacity = 0;
    };

} // namespace Dml
