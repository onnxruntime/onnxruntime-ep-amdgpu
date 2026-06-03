// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/parse_string.h"

#include "dml_execution_provider.h"
#include "DmlExecutionProvider/DmlCommittedResourceAllocator.h"

namespace dml_ep {

PluginDmlExecutionProviderImpl::~PluginDmlExecutionProviderImpl() {
    if (m_cpuMemInfo != nullptr) {
        ort_api.ReleaseMemoryInfo(m_cpuMemInfo);
    }
    if (m_gpuMemInfo != nullptr) {
        ort_api.ReleaseMemoryInfo(m_gpuMemInfo);
    }
}

    void PluginDmlExecutionProviderImpl::Close()
    {
        // Release the cached command list references before closing the context
        m_capturedGraphs.clear();

        m_context->Close();
    }

    void PluginDmlExecutionProviderImpl::WaitForOutstandingWork() const
    {
        Flush();
        m_context->GetCurrentCompletionEvent().WaitForSignal(m_cpuSyncSpinningEnabled);
    }

    HRESULT __stdcall PluginDmlExecutionProviderImpl::AllocatePooledResource(
        size_t size,
        AllocatorRoundingMode roundingMode,
        ID3D12Resource **d3dResource,
        IUnknown** pooledResource
    ) const noexcept
    {
        ORT_TRY
        {
        Microsoft::WRL::ComPtr<IUnknown> allocation;
        allocation.Attach(static_cast<IUnknown* >(m_allocator->AllocImpl(size, roundingMode)));

        const auto* allocInfo = m_allocator->DecodeDataHandle(allocation.Get());

        Microsoft::WRL::ComPtr<ID3D12Resource> resource = allocInfo->GetResource();
        resource.CopyTo(d3dResource);
        *pooledResource = allocation.Detach();
        return S_OK;
        }
        ORT_CATCH_RETURN
    }

    ID3D12Resource* __stdcall PluginDmlExecutionProviderImpl::DecodeResource(void* allocation) const noexcept
    {
        ORT_TRY
        {
            const PluginDmlAllocationInfo* allocInfo = m_allocator->DecodeDataHandle(allocation);
            return allocInfo->GetResource();
        }
        ORT_CATCH_GENERIC
        {
            return nullptr;
        }
    }

    PluginDmlExecutionProviderImpl::PluginDmlExecutionProviderImpl(
        IDMLDevice* dmlDevice,
        ID3D12Device* d3d12Device,
        ExecutionContext* executionContext,
        const ApiPtrs& api_ptrs,
        bool enableMetacommands,
        bool enableGraphCapture,
        bool enableCpuSyncSpinning,
        bool disableMemoryArena)
        : ApiPtrs{api_ptrs},
          m_d3d12Device(d3d12Device),
          m_dmlDevice(dmlDevice),
          m_areMetacommandsEnabled(enableMetacommands),
          m_graphCaptureEnabled(enableGraphCapture),
          m_cpuSyncSpinningEnabled(enableCpuSyncSpinning),
          m_memoryArenaDisabled(disableMemoryArena),
          m_context(executionContext)
    {

        D3D12_FEATURE_DATA_FEATURE_LEVELS featureLevels = {};

        D3D_FEATURE_LEVEL featureLevelsList[] = {
            D3D_FEATURE_LEVEL_1_0_GENERIC,
            D3D_FEATURE_LEVEL_1_0_CORE,
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_12_0,
            D3D_FEATURE_LEVEL_12_1
        };

        featureLevels.NumFeatureLevels = ARRAYSIZE(featureLevelsList);
        featureLevels.pFeatureLevelsRequested = featureLevelsList;
        ORT_THROW_IF_FAILED(d3d12Device->CheckFeatureSupport(
            D3D12_FEATURE_FEATURE_LEVELS,
            &featureLevels,
            sizeof(featureLevels)
            ));

        D3D12_FEATURE_DATA_D3D12_OPTIONS4 featureOptions = {};
        if (SUCCEEDED(d3d12Device->CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS4,
            &featureOptions,
            sizeof(featureOptions))))
        {
            m_native16BitShaderOpsSupported = featureOptions.Native16BitShaderOpsSupported;
        }

        m_isMcdmDevice = (featureLevels.MaxSupportedFeatureLevel <= D3D_FEATURE_LEVEL_1_0_CORE);
        m_areCustomHeapsSupported = !m_isMcdmDevice;

        if (m_isMcdmDevice)
        {

            // TODO: Ingest updated header file
            typedef struct D3D12_FEATURE_DATA_D3D12_OPTIONS19
            {
                BOOL MismatchingOutputDimensionsSupported;
                UINT SupportedSampleCountsWithNoOutputs;
                BOOL PointSamplingAddressesNeverRoundUp;
                BOOL RasterizerDesc2Supported;
                BOOL NarrowQuadrilateralLinesSupported;
                BOOL AnisoFilterWithPointMipSupported;
                UINT MaxSamplerDescriptorHeapSize;
                UINT MaxSamplerDescriptorHeapSizeWithStaticSamplers;
                UINT MaxViewDescriptorHeapSize;
                _Out_  BOOL ComputeOnlyCustomHeapSupported;
            } 	D3D12_FEATURE_DATA_D3D12_OPTIONS19;

            D3D12_FEATURE_DATA_D3D12_OPTIONS19 options19 = {};

            // The call may fail in which case the default value is false
            d3d12Device->CheckFeatureSupport(static_cast<D3D12_FEATURE>(48) /*D3D12_FEATURE_D3D12_OPTIONS19*/, &options19, sizeof(options19));
            m_areCustomHeapsSupported = options19.ComputeOnlyCustomHeapSupported;
        }

        m_uploadHeap = std::make_unique<PluginDmlPooledUploadHeap>(m_d3d12Device.Get(), m_context.Get());
        m_readbackHeap = std::make_unique<PluginDmlReadbackHeap>(m_d3d12Device.Get(), m_context.Get());

        m_lastUploadFlushTime = std::chrono::steady_clock::now();

        THROW_IF_ERROR(ort_api.CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeCPU, &m_cpuMemInfo));
        THROW_IF_ERROR(ort_api.CreateMemoryInfo_V2("GPU", OrtMemoryInfoDeviceType_GPU, amd::VendorId,
            0, OrtDeviceMemoryType_DEFAULT, 0, OrtDeviceAllocator, &m_gpuMemInfo));
    }

    std::vector<OrtAllocator*> PluginDmlExecutionProviderImpl::CreatePreferredAllocators() {
        if (!m_allocator)
        {
            // Create an allocator for D3D12 buffers used to hold tensor data. The returned buffers from the allocator
            // should be DEFAULT heap buffers which can be used as UAVs, and which start in UAV state.
            m_allocator = std::make_shared<DmlBucketizedBufferAllocator>(
                m_gpuMemInfo,
                m_d3d12Device.Get(),
                m_context.Get(),  // TODO(leca): REVIEW: Will it cause memory issue when m_context is released in EP while alloc is released in sessionState?
                CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
                D3D12_HEAP_FLAG_NONE,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                std::make_unique<DmlCommittedResourceAllocator>(m_d3d12Device.Get()));
            m_context->SetAllocator(m_allocator);
            m_cpuInputAllocator = std::make_shared<CpuAllocator>(m_cpuMemInfo);
        }

        return std::vector<OrtAllocator*>{m_allocator.get(), m_cpuInputAllocator.get()};
    }

    HRESULT __stdcall PluginDmlExecutionProviderImpl::GetD3DDevice(_COM_Outptr_ ID3D12Device** d3dDevice) const noexcept
    {
        m_d3d12Device.CopyTo(d3dDevice);
        _Analysis_assume_(*d3dDevice != nullptr);
        return S_OK;
    }

    HRESULT __stdcall PluginDmlExecutionProviderImpl::GetDmlDevice(_COM_Outptr_ IDMLDevice** dmlDevice) const noexcept
    {
        m_dmlDevice.CopyTo(dmlDevice);
        _Analysis_assume_(*dmlDevice != nullptr);
        return S_OK;
    }

    HRESULT __stdcall PluginDmlExecutionProviderImpl::ExecuteCommandList(
        ID3D12GraphicsCommandList* commandList,
        _Outptr_ ID3D12Fence** fence,
        _Out_ uint64_t* completionValue
        ) const noexcept
    {
        ORT_TRY
        {
        assert(!m_closed);
        m_context->ExecuteCommandList(commandList, fence, completionValue);

        return S_OK;
        }
        ORT_CATCH_RETURN
    }

    HRESULT __stdcall PluginDmlExecutionProviderImpl::AddUAVBarrier() const noexcept
    {
        ORT_TRY
        {
        assert(!m_closed);

        m_context->AddUAVBarrier();

        return S_OK;
        }
        ORT_CATCH_RETURN
    }

    HRESULT __stdcall PluginDmlExecutionProviderImpl::InitializeOperator(
        IDMLCompiledOperator* op,
        _In_opt_ const DML_BUFFER_BINDING* persistentResourceBinding,
        gsl::span<const DML_BUFFER_BINDING> inputBindings
        ) const noexcept
    {
        ORT_TRY
        {
        assert(!m_closed);

        bool hasInputsToBind = false;
        std::vector<DML_BUFFER_BINDING> inputBufferBindings(inputBindings.size());

        for (size_t i = 0; i < inputBindings.size(); i++)
        {
            if (inputBindings[i].Buffer)
            {
                hasInputsToBind = true;
                inputBufferBindings[i] = { inputBindings[i].Buffer, inputBindings[i].Offset, inputBindings[i].SizeInBytes };
            }
        }

        DML_BINDING_DESC persistentResourceBindingDesc =
            persistentResourceBinding
            ? DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, persistentResourceBinding }
            : DML_BINDING_DESC{ DML_BINDING_TYPE_NONE, nullptr };

        DML_BUFFER_ARRAY_BINDING inputBufferArrayDesc;
        inputBufferArrayDesc.BindingCount = gsl::narrow_cast<uint32_t>(inputBufferBindings.size());
        inputBufferArrayDesc.Bindings = inputBufferBindings.data();

        DML_BINDING_DESC inputArrayBindingDesc = hasInputsToBind ?
            DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER_ARRAY, &inputBufferArrayDesc } :
            DML_BINDING_DESC{ DML_BINDING_TYPE_NONE, nullptr };

        m_context->InitializeOperator(
            op,
            persistentResourceBindingDesc,
            inputArrayBindingDesc);

        return S_OK;
        }
        ORT_CATCH_RETURN
    }

    HRESULT __stdcall PluginDmlExecutionProviderImpl::ExecuteOperator(
        IDMLCompiledOperator* op,
        _In_opt_ const DML_BUFFER_BINDING* persistentResourceBinding,
        gsl::span<IMLOperatorTensor*> inputTensors,
        gsl::span<IMLOperatorTensor*> outputTensors
        ) const noexcept
    {
        ORT_TRY
        {
        assert(!m_closed);

        std::vector<uint32_t> shape;

        for (IMLOperatorTensor* tensor : inputTensors)
        {
            if (tensor)
            {
                shape.resize(tensor->GetDimensionCount());
                ORT_THROW_IF_FAILED(tensor->GetShape(tensor->GetDimensionCount(), shape.data()));

                if (OperatorHelper::ContainsEmptyDimensions(shape))
                {
                    return S_OK;
                }
            }
        }

        for (IMLOperatorTensor* tensor : outputTensors)
        {
            if (tensor)
            {
                shape.resize(tensor->GetDimensionCount());
                ORT_THROW_IF_FAILED(tensor->GetShape(tensor->GetDimensionCount(), shape.data()));

                if (OperatorHelper::ContainsEmptyDimensions(shape))
                {
                    return S_OK;
                }
            }
        }

        auto FillBindings = [this](auto& bufferBindings, auto& bindingDescs, auto& tensors, const char* role)
        {
            uint32_t tensorIdx = 0;
            for (IMLOperatorTensor* tensor : tensors)
            {
                if (tensor)
                {
                    ORT_THROW_HR_IF(E_INVALIDARG, !tensor->IsDataInterface());

                    Microsoft::WRL::ComPtr<IUnknown> dataInterface = MLOperatorTensor(tensor).GetDataInterface();
                    ORT_THROW_HR_IF(E_INVALIDARG, !dataInterface);

                    const PluginDmlAllocationInfo* allocInfo = m_allocator->DecodeDataHandle(dataInterface.Get());
                    ORT_THROW_HR_IF(E_INVALIDARG, !allocInfo);

                    ID3D12Resource* resource = allocInfo->GetResource();
                    ORT_THROW_HR_IF(E_INVALIDARG, !resource);

                    D3D12_RESOURCE_DESC resourceDesc = resource->GetDesc();

                    bufferBindings.push_back({ resource, 0, resourceDesc.Width });
                    bindingDescs.push_back({ DML_BINDING_TYPE_BUFFER, &bufferBindings.back() });
                }
                else
                {
                    bufferBindings.push_back({ nullptr, 0, 0 });
                    bindingDescs.push_back({ DML_BINDING_TYPE_NONE, nullptr });
                }
                tensorIdx++;
            }
        };

        std::vector<DML_BUFFER_BINDING> inputBufferBindings;
        inputBufferBindings.reserve(inputTensors.size());
        std::vector<DML_BINDING_DESC> inputBindings;
        inputBindings.reserve(inputTensors.size());
        FillBindings(inputBufferBindings, inputBindings, inputTensors, "in");

        std::vector<DML_BUFFER_BINDING> outputBufferBindings;
        outputBufferBindings.reserve(outputTensors.size());
        std::vector<DML_BINDING_DESC> outputBindings;
        outputBindings.reserve(outputTensors.size());
        FillBindings(outputBufferBindings, outputBindings, outputTensors, "out");

        ORT_THROW_IF_FAILED(ExecuteOperator(op, persistentResourceBinding, inputBindings, outputBindings));

        return S_OK;
        }
        ORT_CATCH_RETURN
    }

    HRESULT __stdcall PluginDmlExecutionProviderImpl::ExecuteOperator(
        IDMLCompiledOperator* op,
        _In_opt_ const DML_BUFFER_BINDING* persistentResourceBinding,
        gsl::span<DML_BINDING_DESC> inputTensors,
        gsl::span<DML_BINDING_DESC> outputTensors
        ) const noexcept
    {
        ORT_TRY
        {
        assert(!m_closed);

        DML_BINDING_DESC persistentResourceBindingDesc =
            persistentResourceBinding
            ? DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, persistentResourceBinding }
            : DML_BINDING_DESC{ DML_BINDING_TYPE_NONE, nullptr };

        m_context->ExecuteOperator(
            op,
            persistentResourceBindingDesc,
            inputTensors,
            outputTensors);

        return S_OK;
        }
        ORT_CATCH_RETURN
    }

    static gsl::span<const std::byte> AsByteSpan(const void* data, size_t sizeInBytes)
    {
        return gsl::make_span(static_cast<const std::byte*>(data), sizeInBytes);
    }

    static gsl::span<std::byte> AsByteSpan(void* data, size_t sizeInBytes)
    {
        return gsl::make_span(static_cast<std::byte*>(data), sizeInBytes);
    }

    HRESULT __stdcall PluginDmlExecutionProviderImpl::CopyTensor(IMLOperatorTensor* dst, IMLOperatorTensor* src) const noexcept
    {
        ORT_TRY
        {
        assert(!m_closed);

        const size_t sourceSizeInBytes = ComputeByteSizeFromTensor(*src);
        const size_t dataSizeInBytes = ComputeByteSizeFromTensor(*dst);
        ORT_THROW_HR_IF(E_INVALIDARG, dataSizeInBytes != sourceSizeInBytes); // Tensors must be the same size

        if (dataSizeInBytes == 0)
        {
            return S_OK;
        }

        if (src->IsCpuData() && !dst->IsCpuData())
        {
            //
            // CPU -> GPU copy (upload)
            //
            const PluginDmlAllocationInfo* dstAllocInfo = m_allocator->DecodeDataHandle(MLOperatorTensor(dst).GetDataInterface().Get());

            ID3D12Resource* dstData = dstAllocInfo->GetResource();
            const void* srcData = src->GetData();

            constexpr uint64_t dstOffset = 0;
            const auto dstState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; // GPU resources are always kept in UAV state

            m_uploadHeap->BeginUploadToGpu(dstData, dstOffset, dstState, AsByteSpan(srcData, dataSizeInBytes));

            // Continuously upload memory located in upload heaps during session initialization to avoid running out of it
            if (!m_sessionInitialized)
            {
                FlushUploadsIfReady();
            }
        }
        else if (!src->IsCpuData() && dst->IsCpuData())
        {
            //
            // GPU -> CPU copy (readback)
            //

            void* dstData = dst->GetData();
            const PluginDmlAllocationInfo* srcAllocInfo = m_allocator->DecodeDataHandle(MLOperatorTensor(src).GetDataInterface().Get());

            ID3D12Resource* srcData = srcAllocInfo->GetResource();

            const uint64_t srcOffset = 0;
            const auto srcState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; // GPU resources are always kept in UAV state

            // Performs a blocking call to synchronize and read back data from the GPU into the destination buffer
            m_readbackHeap->ReadbackFromGpu(AsByteSpan(dstData, dataSizeInBytes), srcData, srcOffset, srcState);
        }
        else if (!src->IsCpuData() && !dst->IsCpuData())
        {
            //
            // GPU -> GPU copy
            //
            const PluginDmlAllocationInfo* srcAllocInfo = m_allocator->DecodeDataHandle(MLOperatorTensor(src).GetDataInterface().Get());
            const PluginDmlAllocationInfo* dstAllocInfo = m_allocator->DecodeDataHandle(MLOperatorTensor(dst).GetDataInterface().Get());

            ID3D12Resource* srcData = srcAllocInfo->GetResource();
            ID3D12Resource* dstData = dstAllocInfo->GetResource();
            m_context->CopyBufferRegion(dstData, 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, srcData, 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, dataSizeInBytes);
        }
        else
        {
            // CPU -> CPU copies not supported
            ORT_THROW_HR(E_INVALIDARG);
        }

        return S_OK;
        }
        ORT_CATCH_RETURN
    }

    
    OrtStatus* PluginDmlExecutionProviderImpl::CopyTensorsPlugin(
        const OrtValue** src_tensors_ptr,
        OrtValue** dst_tensors_ptr,
        OrtSyncStream** streams_ptr,
        size_t num_tensors)
    {
            //const DMLDataTransfer& impl = *static_cast<const DMLDataTransfer*>(this_ptr);
        auto src_tensors = gsl::make_span<const OrtValue*>(src_tensors_ptr, num_tensors);
        auto dst_tensors = gsl::make_span<OrtValue*>(dst_tensors_ptr, num_tensors);

        for (size_t i = 0; i < num_tensors; ++i)
        {
            const OrtMemoryDevice* src_device = ep_api.Value_GetMemoryDevice(src_tensors[i]);
            const OrtMemoryDevice* dst_device = ep_api.Value_GetMemoryDevice(dst_tensors[i]);

            OrtMemoryInfoDeviceType src_device_type = ep_api.MemoryDevice_GetDeviceType(src_device);
            OrtMemoryInfoDeviceType dst_device_type = ep_api.MemoryDevice_GetDeviceType(dst_device);

            onnxruntime::Tensor* dst_tensor = dst_tensors[i]->GetMutable<onnxruntime::Tensor>();

            TensorWrapper destInternal(
                dst_tensor, IsGpuTensor(*dst_tensor),
                this, true);

            TensorWrapper srcInternal(
                const_cast<onnxruntime::Tensor*>(&src_tensors[i]->Get<onnxruntime::Tensor>()),
                IsGpuTensor(src_tensors[i]->Get<onnxruntime::Tensor>()),
                this, true);

            const size_t dataSizeInBytes = ComputeByteSizeFromTensor(destInternal);

            if (dataSizeInBytes == 0) {
                continue;
            }

            if (dst_device_type == OrtMemoryInfoDeviceType_GPU)
            {
                if (src_device_type == OrtMemoryInfoDeviceType_GPU)
                {
                    // GPU -> GPU copy
                    GpuToGpuCopy(&srcInternal, &destInternal);
                }
                else
                {
                    // CPU -> GPU (upload)
                    CpuToGpuCopy(&srcInternal, &destInternal);
                }
            }
            else if (src_device_type == OrtMemoryInfoDeviceType_GPU)
            {
                // GPU -> CPU copy (readback)
                GpuToCpuCopy(&srcInternal, &destInternal);
            }
            else
            {
                const void* src_data = nullptr;
                void* dst_data = nullptr;
                size_t bytes;

                RETURN_IF_ERROR(ort_api.GetTensorData(src_tensors[i], &src_data));
                RETURN_IF_ERROR(ort_api.GetTensorMutableData(dst_tensors[i], &dst_data));
                RETURN_IF_ERROR(ort_api.GetTensorSizeInBytes(src_tensors[i], &bytes));
                // CPU -> CPU. may involve copy a to/from host accessible memory and a synchronize may be required first
                memcpy(dst_data, src_data, bytes);
            }
        }
        return nullptr;
    }

    
    bool PluginDmlExecutionProviderImpl::IsGpuTensor(const onnxruntime::Tensor& tensor) {
        return strcmp(tensor.Location().name.c_str(), onnxruntime::CPU) &&
            !(tensor.Location().mem_type == ::OrtMemType::OrtMemTypeCPUOutput ||
              tensor.Location().mem_type == ::OrtMemType::OrtMemTypeCPUInput);
    }

    void PluginDmlExecutionProviderImpl::CpuToGpuCopy(IMLOperatorTensor* src,
                                                      IMLOperatorTensor* dst)
    {
        const size_t dataSizeInBytes = ComputeByteSizeFromTensor(*dst);

        //// CPU -> GPU copy (upload)
        const PluginDmlAllocationInfo* dstAllocInfo = m_allocator->DecodeDataHandle(MLOperatorTensor(dst).GetDataInterface().Get());

        ID3D12Resource* dstData = dstAllocInfo->GetResource();
        const void* srcData = src->GetData();

        constexpr uint64_t dstOffset = 0;
        const auto dstState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; // GPU resources are always kept in UAV state

        m_uploadHeap->BeginUploadToGpu(dstData, dstOffset, dstState, AsByteSpan(srcData, dataSizeInBytes));

        // Continuously upload memory located in upload heaps during session initialization to avoid running out of it
        if (!m_sessionInitialized) {
            FlushUploadsIfReady();
        }
    }

    void PluginDmlExecutionProviderImpl::GpuToGpuCopy(IMLOperatorTensor* src, IMLOperatorTensor* dst)
    {
        const size_t dataSizeInBytes = ComputeByteSizeFromTensor(*dst);

        const PluginDmlAllocationInfo* srcAllocInfo = m_allocator->DecodeDataHandle(MLOperatorTensor(src).GetDataInterface().Get());
        const PluginDmlAllocationInfo* dstAllocInfo = m_allocator->DecodeDataHandle(MLOperatorTensor(dst).GetDataInterface().Get());

        ID3D12Resource* srcData = srcAllocInfo->GetResource();
        ID3D12Resource* dstData = dstAllocInfo->GetResource();

        m_context->CopyBufferRegion(dstData, 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, srcData, 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, dataSizeInBytes);
    }

    void PluginDmlExecutionProviderImpl::GpuToCpuCopy(IMLOperatorTensor* src, IMLOperatorTensor* dst)
    {
        const size_t dataSizeInBytes = ComputeByteSizeFromTensor(*dst);

        void* dstData = dst->GetData();
        const PluginDmlAllocationInfo* srcAllocInfo = m_allocator->DecodeDataHandle(MLOperatorTensor(src).GetDataInterface().Get());

        ID3D12Resource* srcData = srcAllocInfo->GetResource();

        const uint64_t srcOffset = 0;
        const auto srcState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; // GPU resources are always kept in UAV state

        // Performs a blocking call to synchronize and read back data from the GPU into the destination buffer
        m_readbackHeap->ReadbackFromGpu(AsByteSpan(dstData, dataSizeInBytes), srcData, srcOffset, srcState);
    }

    HRESULT __stdcall PluginDmlExecutionProviderImpl::CopyTensors(gsl::span<IMLOperatorTensor*> dst, gsl::span<IMLOperatorTensor*> src) const noexcept
    {
        ORT_TRY
        {
        ORT_THROW_HR_IF(E_INVALIDARG, dst.size() != src.size());

        // Source and destination for batched GPU -> CPU copies
        std::vector<ID3D12Resource*> srcDatas;
        std::vector<void*> dstDatas;
        std::vector<uint32_t> dataSizesInBytes;

        for (uint32_t i = 0; i < dst.size(); ++i)
        {
            // This batching implementation only handles GPU -> CPU copies.  Other copies do not require synchronization
            // and are batched across multiple calls to CopyTensor.
            if (src[i]->IsCpuData() || !dst[i]->IsCpuData())
            {
                ORT_THROW_IF_FAILED(CopyTensor(dst[i], src[i]));
                continue;
            }

            const size_t dstSizeInBytes = ComputeByteSizeFromTensor(*dst[i]);
            const size_t srcSizeInBytes = ComputeByteSizeFromTensor(*src[i]);

            ORT_THROW_HR_IF(E_INVALIDARG, dstSizeInBytes != srcSizeInBytes); // Tensors must be the same size
            const size_t dataSizeInBytes = dstSizeInBytes;

            if (dataSizeInBytes == 0)
            {
                continue;
            }

            dataSizesInBytes.push_back(static_cast<uint32_t>(ComputeByteSizeFromTensor(*dst[i])));
            ORT_THROW_HR_IF(E_INVALIDARG, dataSizesInBytes.back() != ComputeByteSizeFromTensor(*src[i])); // Tensors must be the same size

            dstDatas.push_back(dst[i]->GetData());

            auto srcDataInterface = MLOperatorTensor(src[i]).GetDataInterface();

            const PluginDmlAllocationInfo* srcAllocInfo =
                m_allocator->DecodeDataHandle(srcDataInterface.Get());

            if (srcAllocInfo) {
                srcDatas.push_back(srcAllocInfo->GetResource());
            }
        }

        if (!srcDatas.empty()) {
            const auto srcState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; // GPU resources are always kept in UAV state

            // Performs a blocking call to synchronize and read back data from the GPU into the destination buffer
            m_readbackHeap->ReadbackFromGpu(dstDatas, dataSizesInBytes, srcDatas, srcState);
        }

        return S_OK;
        }
        ORT_CATCH_RETURN
    }

    HRESULT STDMETHODCALLTYPE PluginDmlExecutionProviderImpl::FillTensorWithPattern(
        IMLOperatorTensor* dst,
        gsl::span<const std::byte> rawValue // Data type agnostic rawValue, treated as raw bits
        ) const noexcept
    {
        ORT_TRY
        {
        auto mlTensor = MLOperatorTensor(dst).GetDataInterface();
        if (mlTensor != nullptr)
        {
            const PluginDmlAllocationInfo* dstAllocInfo = m_allocator->DecodeDataHandle(mlTensor.Get());
            ID3D12Resource* dstData = dstAllocInfo->GetResource();
            m_context->FillBufferWithPattern(dstData, rawValue);
        }

        return S_OK;
        }
        ORT_CATCH_RETURN
    }

    HRESULT __stdcall PluginDmlExecutionProviderImpl::UploadToResource(ID3D12Resource* dstData, const void* srcData, uint64_t srcDataSize) const noexcept
    {
        ORT_TRY
        {
        assert(!m_closed);

        m_uploadHeap->BeginUploadToGpu(dstData, 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, AsByteSpan(srcData, static_cast<size_t>(srcDataSize)));
        FlushUploadsIfReady();

        return S_OK;
        }
        ORT_CATCH_RETURN
    }

    void PluginDmlExecutionProviderImpl::FlushUploadsIfReady() const
    {
        // Periodically flush uploads to make sure the GPU is not idle for too long
        if (std::chrono::steady_clock::now() - m_lastUploadFlushTime > m_batchFlushInterval)
        {
            Flush();
            m_lastUploadFlushTime = std::chrono::steady_clock::now();
        }
    }

    uint32_t PluginDmlExecutionProviderImpl::GetSupportedDeviceDataTypeMask() const
    {
        // The DML provider registers all supported kernels up-front regardless of actual device capability,
        // but this is problematic later when executing the graph because DirectML will fail to create
        // the operator, and by that late phase, it's long past too late to recover. So, this function queries
        // the actual type capabilities so the partitioner may assigns nodes to the CPU if the GPU cannot
        // handle them, similar to the fallback in CUDAExecutionProvider::GetCapability for certain RNN/GRU/Conv
        // attributes.

        return dml_ep::GetSupportedDeviceDataTypeMask(m_dmlDevice.Get());
    }


    void __stdcall PluginDmlExecutionProviderImpl::Flush() const
    {
        assert(!m_closed);
        m_context->Flush();
    }

    void PluginDmlExecutionProviderImpl::ReleaseCompletedReferences()
    {
         m_context->ReleaseCompletedReferences();
    }

    void PluginDmlExecutionProviderImpl::QueueReference(IUnknown* object)
    {
        assert(!m_closed);
        m_context->QueueReference(object);
    }

    void PluginDmlExecutionProviderImpl::GetShadowCopyIfRequired(
        bool isInternalOperator,
        IUnknown* data,
        IUnknown** dataCopy) const
    {
        assert(!m_closed);

        *dataCopy = data;
        data->AddRef();
    }

    void PluginDmlExecutionProviderImpl::GetABIDataInterface(
        bool isInternalOperator,
        IUnknown* data,
        IUnknown** abiData) const
    {
        assert(!m_closed);

        if (isInternalOperator)
        {
            *abiData = data;
            data->AddRef();
        }
        else
        {
            Microsoft::WRL::ComPtr<ID3D12Resource> resource = m_allocator->DecodeDataHandle(data)->GetResource();
            *abiData = resource.Detach();
        }
    }

    uint64_t PluginDmlExecutionProviderImpl::TryGetPooledAllocationId(
        IUnknown* data,
        bool isInternalOperator)
    {
        assert(!isInternalOperator);
        return m_allocator->DecodeDataHandle(data)->GetPooledResourceId();
    }

    void PluginDmlExecutionProviderImpl::GetABIExecutionInterfaceAndInvalidateState(
        bool isInternalOperator,
        IUnknown** abiExecutionObject) const
    {
        assert(!m_closed);

        if (isInternalOperator)
        {
            Microsoft::WRL::ComPtr<IUnknown> thisPtr = const_cast<IExecutionProvider*>(static_cast<const IExecutionProvider*>(this));
            *abiExecutionObject = thisPtr.Detach();
        }
        else
        {
            Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> commandList;
            m_context->GetCommandListForRecordingAndInvalidateState(commandList.GetAddressOf());
            *abiExecutionObject = commandList.Detach();
        }
    }

    bool PluginDmlExecutionProviderImpl::TransitionsRequiredForOperator(
        bool isInternalOperator
    )
    {
        // External operators receive resources in Common state, while internal operators receive
        // them in UAV state. Resources are otherwise kept in UAV state (or are promotable to UAV).
        return !isInternalOperator;
    }

    void PluginDmlExecutionProviderImpl::TransitionResourcesForOperator(
        bool isBeforeOp,
        uint32_t resourceCount,
        IUnknown** resources
    )
    {
        std::vector<D3D12_RESOURCE_BARRIER> barriers;
        barriers.reserve(resourceCount);

        for (uint32_t i = 0; i < resourceCount; ++i)
        {
            Microsoft::WRL::ComPtr<ID3D12Resource> resource;
            ORT_THROW_IF_FAILED(resources[i]->QueryInterface(resource.GetAddressOf()));

            // Custom operators receive resources in Common state and must return them to Common
            // state when finished.  Resources are otherwise kept in UAV state (or are promotable to UAV).
            barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(
                resource.Get(),
                isBeforeOp ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_COMMON,
                isBeforeOp ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_UNORDERED_ACCESS
            ));
        }

        if (!barriers.empty())
        {
            m_context->ResourceBarrier(barriers);
        }
    }

    D3D12_COMMAND_LIST_TYPE __stdcall PluginDmlExecutionProviderImpl::GetCommandListTypeForQueue() const
    {
        return m_context->GetCommandListTypeForQueue();
    }

    bool __stdcall PluginDmlExecutionProviderImpl::IsMcdmDevice() const noexcept
    {
        return m_isMcdmDevice;
    }

    bool __stdcall PluginDmlExecutionProviderImpl::CustomHeapsSupported() const noexcept
    {
        return m_areCustomHeapsSupported;
    }

    bool __stdcall PluginDmlExecutionProviderImpl::MetacommandsEnabled() const noexcept
    {
        return m_areMetacommandsEnabled;
    }

    bool PluginDmlExecutionProviderImpl::CpuSyncSpinningEnabled() const noexcept
    {
        return m_cpuSyncSpinningEnabled;
    }

    bool PluginDmlExecutionProviderImpl::GraphCaptureEnabled() const noexcept
    {
        return m_graphCaptureEnabled;
    }

    bool PluginDmlExecutionProviderImpl::GraphCaptured(int graph_annotation_id) const
    {
        return m_graphCapturingDone.find(graph_annotation_id) != m_graphCapturingDone.end();
    };

    std::shared_ptr<const InternalRegistrationInfoMap>
    PluginDmlExecutionProviderImpl::GetInternalRegistrationInfoMap() const
    {
        return m_internalRegInfoMap;
    }

    // Get the allocation object (IUnknown*) from a data pointer
    // This is ABI-safe because both sides use the same allocator instance - no struct crossing
    IUnknown* PluginDmlExecutionProviderImpl::GetAllocationFromDataPointer(void* data_ptr) const
    {
        if (!data_ptr || !m_allocator) {
            return nullptr;
        }

        // Use the allocator's DecodeDataHandle to get the PluginDmlAllocationInfo
        // This is safe because we're using the same allocator instance, not crossing DLL boundaries
        const PluginDmlAllocationInfo* alloc_info = m_allocator->DecodeDataHandle(data_ptr);
        if (!alloc_info) {
            return nullptr;
        }

        // PluginDmlAllocationInfo inherits from IUnknown, so we can QueryInterface safely
        IUnknown* allocation = const_cast<PluginDmlAllocationInfo*>(alloc_info);
        allocation->AddRef();  // Caller owns the reference
        return allocation;
    }

    std::shared_ptr<OrtAllocator> PluginDmlExecutionProviderImpl::GetGpuAllocator()
    {
        return m_allocator;
    }

    std::shared_ptr<OrtAllocator> PluginDmlExecutionProviderImpl::GetCpuInputAllocator()
    {
        return m_cpuInputAllocator;
    }

    Ort::Status PluginDmlExecutionProviderImpl::OnSessionInitializationEnd()
    {
        if (m_sessionInitialized == false) {
            // Flush and trim resources, including staging memory used to upload weights.
            // This reduces memory usage immediately after session creation, and avoids
            // performance impact of deallocation during first evaluation.
            Flush();
            m_context->GetCurrentCompletionEvent().WaitForSignal(m_cpuSyncSpinningEnabled);
            m_context->ReleaseCompletedReferences();
            m_uploadHeap->Trim();

            if (!m_memoryArenaDisabled) {
                // Allocations after this point are potentially transient and their sizes are
                // rounded to enable pooling.
                m_allocator->SetDefaultRoundingMode(AllocatorRoundingMode::Enabled);
            }

            m_sessionInitialized = true;
        }

        return STATUS_OK;
    }

    void PluginDmlExecutionProviderImpl::AppendCapturedGraph(int annotationId, std::unique_ptr<DmlReusedCommandListState> capturedGraph)
    {
        m_capturedGraphs[annotationId].push_back(std::move(capturedGraph));
    }

    Ort::Status PluginDmlExecutionProviderImpl::ReplayGraph(int graph_annotation_id)
    {
        for (auto& capturedGraph : m_capturedGraphs[graph_annotation_id])
        {
            ExecuteCommandList(capturedGraph->graphicsCommandList.Get(), &capturedGraph->fence, &capturedGraph->completionValue);
        }

        return STATUS_OK;
    }

    Ort::Status PluginDmlExecutionProviderImpl::OnRunStart(const OrtRunOptions& run_options)
    {
        if (GraphCaptureEnabled())
        {
            auto graphAnnotationStr = ort_api.GetRunConfigEntry(&run_options, "gpu_graph_id");
            // If graph annotation is not provided, fall back to the one dml graph per session behavior
            int dmlGraphAnnotationId = 0;
            if (graphAnnotationStr != nullptr)
            {
                ORT_ENFORCE(TryParseStringWithClassicLocale<int>(graphAnnotationStr, dmlGraphAnnotationId),
                            "Failed to parse the dml graph annotation id: ",
                            *graphAnnotationStr);
            }

            m_currentGraphAnnotationId = dmlGraphAnnotationId;
        }

        return STATUS_OK;
    }

    Ort::Status PluginDmlExecutionProviderImpl::OnRunEnd()
    {
        if (GraphCaptureEnabled() && m_currentGraphAnnotationId != -1)
        {
            m_graphCapturingDone.insert(m_currentGraphAnnotationId);
        }

        // Flush any pending work to the GPU, but don't block for completion, permitting it
        // to overlap other work.
        Flush();
        return STATUS_OK;
    }

}  // namespace dml_ep
