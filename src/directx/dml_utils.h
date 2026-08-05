// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/plugin_ep_utils.h"
#include "dml_factory.h"

namespace dml_ep {

static void ThrowIfFailed(HRESULT hr) {
    if (FAILED(hr)) {
        throw std::runtime_error("HRESULT failed");
    }
}

class D3DResourceHelper
{
public:
    static void CloseExecuteResetWait(
        ID3D12Device* d3d12Device,
        ID3D12CommandQueue* commandQueue,
        ID3D12CommandAllocator* commandAllocator,
        ID3D12GraphicsCommandList* commandList);

    static void CreateCommittedResourceHelper(
        ID3D12Device* device,
        Microsoft::WRL::ComPtr<ID3D12Resource>& buffer,
        size_t sizeInBytes);

    static void UploadDataToGpuHelper(
        ID3D12Device* device,
        ID3D12GraphicsCommandList* commandList,
        ID3D12CommandQueue* commandQueue,
        ID3D12CommandAllocator* commandAllocator,
        Microsoft::WRL::ComPtr<ID3D12Resource>& destination,
        D3D12_SUBRESOURCE_DATA bufferSubresourceData,
        size_t size);

    static Microsoft::WRL::ComPtr<ID3D12Resource> CreateDmlCommittedResourceAllocator(ID3D12Device* device, size_t size);

};


}  // namespace dml_ep
