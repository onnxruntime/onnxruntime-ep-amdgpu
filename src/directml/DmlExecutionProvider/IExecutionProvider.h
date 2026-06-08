// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "d3d12.h"

interface IDMLCompiledOperator;
struct DML_BUFFER_BINDING;
struct DML_BINDING_DESC;

namespace dml_ep {

    struct Binding
    {
        // Non-null if required at the stage where it is used, i.e. Initialization
        IMLOperatorTensor* tensor;

        UINT64 sizeInBytes;
    };

    // DML specific interface into the execution provider, which avoids any dependencies with
    // internal Lotus data types.
    interface __declspec(uuid("b2488edb-fad2-4704-a6d2-5b5b129d4b8e"))
    IExecutionProvider : IUnknown
    {
        virtual ~IExecutionProvider() = default;
        STDMETHOD(GetD3DDevice)(_COM_Outptr_ ID3D12Device** d3dDevice) const noexcept = 0;

        STDMETHOD(GetDmlDevice)(_COM_Outptr_ IDMLDevice** dmlDevice) const noexcept = 0;

        STDMETHOD(ExecuteCommandList)(
            ID3D12GraphicsCommandList* commandList,
            _Outptr_ ID3D12Fence** fence,
            _Out_ uint64_t* completionValue
            ) const noexcept = 0;

        STDMETHOD(AddUAVBarrier)() const noexcept = 0;

        STDMETHOD(InitializeOperator)(
            const Microsoft::WRL::ComPtr<IDMLCompiledOperator>& op,
            const std::optional<DML_BUFFER_BINDING>& persistentResourceBinding,
            const std::vector<DML_BUFFER_BINDING>& inputs
            ) const noexcept = 0;

        STDMETHOD(ExecuteOperator)(
            const Microsoft::WRL::ComPtr<IDMLCompiledOperator>& op,
            const std::optional<DML_BUFFER_BINDING>& persistentResourceBinding,
            const std::vector<Microsoft::WRL::ComPtr<IMLOperatorTensor>>& inputs,
            const std::vector<Microsoft::WRL::ComPtr<IMLOperatorTensor>>& outputs
            ) const noexcept = 0;

        STDMETHOD(ExecuteOperator)(
            const Microsoft::WRL::ComPtr<IDMLCompiledOperator>& op,
            const std::optional<DML_BUFFER_BINDING>& persistentResourceBinding,
            const std::vector<DML_BINDING_DESC>& inputs,
            const std::vector<DML_BINDING_DESC>& outputs
            ) const noexcept = 0;

        STDMETHOD(CopyTensor)(
            const Microsoft::WRL::ComPtr<IMLOperatorTensor>& dst,
            const Microsoft::WRL::ComPtr<IMLOperatorTensor>& src
            ) const noexcept = 0;

        STDMETHOD(CopyTensors)(
            const std::vector<Microsoft::WRL::ComPtr<IMLOperatorTensor>>& dst,
            const std::vector<Microsoft::WRL::ComPtr<IMLOperatorTensor>>& src
            ) const noexcept = 0;

        STDMETHOD(FillTensorWithPattern)(
            const Microsoft::WRL::ComPtr<IMLOperatorTensor>& dst,
            std::vector<std::byte>& value
            ) const noexcept = 0;

        STDMETHOD(UploadToResource)(
            ID3D12Resource* dstData, const void* srcData, uint64_t srcDataSize
            ) const noexcept = 0;

        STDMETHOD_(D3D12_COMMAND_LIST_TYPE, GetCommandListTypeForQueue)() const noexcept = 0;
        STDMETHOD_(void, Flush)() const noexcept = 0;

        STDMETHOD_(ID3D12Resource*, DecodeResource)(void* allocation) const noexcept = 0;
        STDMETHOD(AllocatePooledResource(size_t size, AllocatorRoundingMode roundingMode, ID3D12Resource **d3dResource, IUnknown* *pooledResource)) const noexcept = 0;

        STDMETHOD_(bool, IsMcdmDevice)() const noexcept = 0;
        STDMETHOD_(bool, CustomHeapsSupported)() const noexcept = 0;
        STDMETHOD_(bool, MetacommandsEnabled)() const noexcept = 0;
    };

}  // namespace dml_ep
