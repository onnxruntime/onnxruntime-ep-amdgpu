// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include <windows.h>
#include "plugin_ep_utils.h"
//#include <winrt/windows.foundation.h>
//#include <winrt/windows.ai.machinelearning.h>
//
//
//
//
//
#include "dml_factory.h"
#include <stdlib.h>

//

namespace Dml {

using Microsoft::WRL::ComPtr;
    
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
        ComPtr<ID3D12Resource>& buffer,
        size_t sizeInBytes);

    static void UploadDataToGpuHelper(
        ID3D12Device* device,
        ID3D12GraphicsCommandList* commandList,
        ID3D12CommandQueue* commandQueue,
        ID3D12CommandAllocator* commandAllocator,
        ComPtr<ID3D12Resource>& destination,
        D3D12_SUBRESOURCE_DATA bufferSubresourceData,
        size_t size);

    static ComPtr<ID3D12Resource> CreateDmlCommittedResourceAllocator(ID3D12Device* device, size_t size);

};


}   // namespace Dml
