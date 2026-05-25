// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "dml_utils.h"

namespace Dml {

    void D3DResourceHelper::CloseExecuteResetWait(
        ID3D12Device* d3d12Device,
        ID3D12CommandQueue* commandQueue,
        ID3D12CommandAllocator* commandAllocator,
        ID3D12GraphicsCommandList* commandList)
    {
        ID3D12CommandList* commandlists[] = {commandList};

        ThrowIfFailed(commandList->Close());
        commandQueue->ExecuteCommandLists(_countof(commandlists), commandlists);

        // Create a fence and event to wait for the command list to finish executing
        // note: In a real application, reuse fences and events to remove creation overhead. Fix later.
        ComPtr<ID3D12Fence> d3D12Fence;
        ThrowIfFailed(d3d12Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_GRAPHICS_PPV_ARGS(d3D12Fence.GetAddressOf())));

        wil::unique_handle fenceEventHandle(::CreateEvent(nullptr, true, false, nullptr));
        THROW_LAST_ERROR_IF_NULL(fenceEventHandle);

        ThrowIfFailed(commandQueue->Signal(d3D12Fence.Get(), 1));
        ThrowIfFailed(d3D12Fence->SetEventOnCompletion(1, fenceEventHandle.get()));

        ::WaitForSingleObjectEx(fenceEventHandle.get(), INFINITE, FALSE);

        ThrowIfFailed(commandAllocator->Reset());
        ThrowIfFailed(commandList->Reset(commandAllocator, nullptr));
    }

    void D3DResourceHelper::CreateCommittedResourceHelper(ID3D12Device* device, ComPtr<ID3D12Resource>& buffer, size_t sizeInBytes)
    {
        CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_DEFAULT);
        CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        ThrowIfFailed(device->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_NONE,
            &bufferDesc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(buffer.GetAddressOf())));
    }

    void D3DResourceHelper::UploadDataToGpuHelper(
        ID3D12Device* device,
        ID3D12GraphicsCommandList* commandList,
        ID3D12CommandQueue* commandQueue,
        ID3D12CommandAllocator* commandAllocator,
        ComPtr<ID3D12Resource>& destination,
        D3D12_SUBRESOURCE_DATA bufferSubresourceData,
        size_t size)
    {
        ComPtr<ID3D12Resource> uploadBuffer;
        CD3DX12_RESOURCE_DESC uploadBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(size);
        CD3DX12_HEAP_PROPERTIES uploadHeapProps(D3D12_HEAP_TYPE_UPLOAD);

        ThrowIfFailed(device->CreateCommittedResource(
            &uploadHeapProps,
            D3D12_HEAP_FLAG_NONE,
            &uploadBufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_GRAPHICS_PPV_ARGS(uploadBuffer.GetAddressOf())));

        // Upload  to the GPU.
        ::UpdateSubresources(commandList, destination.Get(), uploadBuffer.Get(), 0, 0, 1, &bufferSubresourceData);

        CloseExecuteResetWait(device, commandQueue, commandAllocator, commandList);
    }

    ComPtr<ID3D12Resource> D3DResourceHelper::CreateDmlCommittedResourceAllocator(ID3D12Device* device, size_t size)
    {
        ComPtr<ID3D12Resource> resource;
        CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_DEFAULT);
        auto buffer = CD3DX12_RESOURCE_DESC::Buffer(size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        ThrowIfFailed(device->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_NONE,
            &buffer,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_GRAPHICS_PPV_ARGS(resource.GetAddressOf())));

        return resource;
    }


}  // namespace Dml

#include "Eigen/Core"

namespace onnxruntime::math {

namespace {
template <typename Dst, typename Src>
std::enable_if_t<
    sizeof(Src) == sizeof(Dst) &&
        std::is_trivially_copyable_v<Src> &&
        std::is_trivially_copyable_v<Dst> &&
        std::is_trivially_constructible_v<Dst>,
    Dst>
BitCast(const Src& src) {
    Dst dst;
    std::memcpy(&dst, &src, sizeof(dst));
    return dst;
}
}  // namespace

float halfToFloat(uint16_t h) {
    return Eigen::half_impl::half_to_float(Eigen::half_impl::raw_uint16_to_half(h));
}

uint16_t floatToHalf(float f) {
    return BitCast<uint16_t>(Eigen::half_impl::float_to_half_rtne(f).x);
}

}  // namespace onnxruntime::math
