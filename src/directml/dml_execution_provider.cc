// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/parse_string.h"
#include "common/env_var.h"

#include "dml_execution_provider.h"
#include "DmlExecutionProvider/DmlCommittedResourceAllocator.h"
#include "dml_perf_timer.h"

namespace dml_ep {

PluginDmlExecutionProviderImpl::~PluginDmlExecutionProviderImpl() {
    if (m_cpuMemInfo != nullptr) {
        ort_api.ReleaseMemoryInfo(m_cpuMemInfo);
    }
    if (m_gpuMemInfo != nullptr) {
        ort_api.ReleaseMemoryInfo(m_gpuMemInfo);
    }
    if (m_hostAccessibleMemInfo != nullptr) {
        ort_api.ReleaseMemoryInfo(m_hostAccessibleMemInfo);
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
            // Host-accessible decode inputs return a mapped CPU pointer from Alloc (not an allocInfo
            // handle). Check that allocator first; if it owns the pointer, return its resource. Else
            // fall through to the DEFAULT allocator's handle decode (unchanged behavior).
            if (m_hostAccessibleAllocator) {
                if (ID3D12Resource* res = m_hostAccessibleAllocator->TryGetResource(allocation)) {
                    return res;
                }
            }
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
        bool disableMemoryArena,
        std::shared_ptr<DmlHostAccessibleAllocator>* factoryHostAllocHolder)
        : ApiPtrs{api_ptrs},
          m_d3d12Device(d3d12Device),
          m_dmlDevice(dmlDevice),
          m_areMetacommandsEnabled(enableMetacommands),
          m_graphCaptureEnabled(enableGraphCapture),
          m_cpuSyncSpinningEnabled(enableCpuSyncSpinning),
          m_memoryArenaDisabled(disableMemoryArena),
          m_context(executionContext),
          m_factoryHostAllocHolder(factoryHostAllocHolder)
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

        // Host-accessible (CUSTOM/L0/WRITE_COMBINE) decode inputs are OPT-IN on the DirectML backend,
        // gated by the AMDGPU_DML_HOST_ACCESSIBLE env var (default OFF).
        //
        // WHY OFF BY DEFAULT: the CUSTOM/L0 committed resource is backed by the CPU-visible VRAM aperture,
        // whose availability/budget depend on the board's Resizable-BAR configuration. On some discrete
        // GPUs a real allocation intermittently fails with E_OUTOFMEMORY, surfacing as std::bad_alloc at
        // session init on the raw-ORT EAGER-allocator path (CreatePreferredAllocators) — which, unlike
        // OGA, has NO try/catch fallback. The 256-byte probe below does NOT predict this (it can pass
        // while real decode-input allocations later fail). Building the allocator also adds a measurable
        // session-creation cost. modelbench (raw ORT) never uses host-accessible inputs, so it must never
        // pay this: with the flag off, m_hostAccessibleSupported stays false, the allocator is never built,
        // and dml_factory.cc CreateAllocatorImpl resolves any HOST_ACCESSIBLE request to a plain CPU
        // allocator (harmless — nothing binds it as a D3D12 resource on that path).
        //
        // WHEN ON (OGA sets AMDGPU_DML_HOST_ACCESSIBLE=1 at runtime): the probe runs and, if it succeeds,
        // the real CUSTOM/L0 allocator is built and served, so OGA can update decode inputs in place.
        // OGA's request path is guarded (model.cpp try/catch + null-allocator fallback), so it degrades
        // rather than hard-fails. MIGraphX/HIP host-accessible (different backend + hipHostMalloc) is
        // independent of this flag.
        const bool host_accessible_opt_in =
            ParseEnvironmentVariableWithDefault<bool>("AMDGPU_DML_HOST_ACCESSIBLE", false);
        if (host_accessible_opt_in) {
            D3D12_HEAP_PROPERTIES hp{};
            hp.Type = D3D12_HEAP_TYPE_CUSTOM;
            hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE;
            hp.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
            hp.CreationNodeMask = 1;
            hp.VisibleNodeMask = 1;
            auto probeBuf = CD3DX12_RESOURCE_DESC::Buffer(256, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            Microsoft::WRL::ComPtr<ID3D12Resource> probe;
            m_hostAccessibleSupported = SUCCEEDED(m_d3d12Device->CreateCommittedResource(
                &hp, D3D12_HEAP_FLAG_NONE, &probeBuf, D3D12_RESOURCE_STATE_COMMON, nullptr,
                IID_PPV_ARGS(probe.GetAddressOf())));
        }
        if (m_hostAccessibleSupported) {
            THROW_IF_ERROR(ort_api.CreateMemoryInfo_V2("GPUHostAccessible", OrtMemoryInfoDeviceType_GPU,
                amd::VendorId, 0, OrtDeviceMemoryType_HOST_ACCESSIBLE, 0, OrtDeviceAllocator,
                &m_hostAccessibleMemInfo));
        }
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

            // Host-accessible (CUSTOM/L0/WRITE_COMBINE) allocator for decode inputs — additive, only
            // when supported. It does NOT replace m_allocator: KV/scratch/tensors stay on the DEFAULT
            // heap. CreateAllocatorImpl routes HOST_ACCESSIBLE requests here.
            //
            // SHARED across EP instances via the factory-owned holder: GPU-KV makes OGA create 2
            // sessions -> 2 EPs; a per-EP allocator gave 2 allocation maps, so a position_ids pointer
            // allocated in one EP MISSed in the executing EP's map (pos_ids_reformat/Reshape
            // E_INVALIDARG). The first EP builds the allocator and publishes it to the holder; later
            // EPs adopt it -> one map. Requires the shared factory d3d12_device (D3D12 resources are
            // device-local), which CreateEpImpl now guarantees.
            if (m_factoryHostAllocHolder && *m_factoryHostAllocHolder) {
                // Another EP already built it — adopt the shared instance.
                m_hostAccessibleAllocator = *m_factoryHostAllocHolder;
            } else if (m_hostAccessibleSupported) {
                D3D12_HEAP_PROPERTIES hostHeap{};
                hostHeap.Type = D3D12_HEAP_TYPE_CUSTOM;
                // WRITE_COMBINE: CPU-write-fast, GPU-readable. Decode inputs are CPU-written and
                // GPU-read, so this is the correct direction. (Logits, which are GPU-written and
                // CPU-read, must NOT use this heap; they are routed off it on the OGA side.)
                hostHeap.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE;
                hostHeap.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
                hostHeap.CreationNodeMask = 1;
                hostHeap.VisibleNodeMask = 1;
                // Non-pooled: Alloc persistently Map()s the CUSTOM/L0 resource and returns a CPU
                // pointer OGA writes in place; it decodes back to the resource on bind (DecodeResource).
                m_hostAccessibleAllocator = std::make_shared<DmlHostAccessibleAllocator>(
                    m_hostAccessibleMemInfo,
                    m_d3d12Device.Get(),
                    std::make_unique<DmlCommittedResourceAllocator>(m_d3d12Device.Get(), hostHeap));
                // Publish to the factory holder so sibling EPs share this one instance.
                if (m_factoryHostAllocHolder)
                    *m_factoryHostAllocHolder = m_hostAccessibleAllocator;
            }
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

                    // The data interface may ALREADY be an ID3D12Resource* (AbiSafeTensor non-internal
                    // path: GetABIDataInterface pre-resolves the resource via DecodeDataHandle), or a
                    // PluginDmlAllocationInfo* (legacy handle). DecodeDataHandle only static_casts, so
                    // decoding a resource-as-allocation reinterprets the vtable -> garbage resource ->
                    // DML dispatch E_INVALIDARG (the host-accessible per-node path hit this; the fused
                    // DEFAULT path never calls FillBindings). Discriminate by QueryInterface: a resource
                    // answers ID3D12Resource; a PluginDmlAllocationInfo (IUnknown-only) does not.
                    Microsoft::WRL::ComPtr<ID3D12Resource> resourceCom;
                    ID3D12Resource* resource = nullptr;
                    if (SUCCEEDED(dataInterface.As(&resourceCom))) {
                        resource = resourceCom.Get();
                    } else {
                        const PluginDmlAllocationInfo* allocInfo = m_allocator->DecodeDataHandle(dataInterface.Get());
                        ORT_THROW_HR_IF(E_INVALIDARG, !allocInfo);
                        resource = allocInfo->GetResource();
                    }
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
        auto src_tensors = gsl::make_span<const OrtValue*>(src_tensors_ptr, num_tensors);
        auto dst_tensors = gsl::make_span<OrtValue*>(dst_tensors_ptr, num_tensors);

        // Process tensors in array order to preserve dependencies. GPU->CPU copies
        // are accumulated into a batch for a single flush+wait. Any non-GPU->CPU
        // operation flushes the pending batch first, so ordering is maintained.
        std::vector<ID3D12Resource*> readbackSrcResources;
        std::vector<void*> readbackDstPtrs;
        std::vector<uint32_t> readbackSizes;

        auto flushPendingReadbacks = [&]() {
            if (!readbackSrcResources.empty()) {
                const auto srcState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                m_readbackHeap->ReadbackFromGpu(readbackDstPtrs, readbackSizes, readbackSrcResources, srcState);
                readbackSrcResources.clear();
                readbackDstPtrs.clear();
                readbackSizes.clear();
            }
        };

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
                flushPendingReadbacks();
                if (src_device_type == OrtMemoryInfoDeviceType_GPU)
                    GpuToGpuCopy(&srcInternal, &destInternal);
                else
                    CpuToGpuCopy(&srcInternal, &destInternal);
            }
            else if (src_device_type == OrtMemoryInfoDeviceType_GPU)
            {
                // GPU -> CPU: accumulate for batched readback. Resolve src via the host-accessible-aware
                // resolver (CUSTOM/L0 source supported).
                void* dstData = destInternal.GetData();
                ID3D12Resource* srcRes = ResolveTensorResource(&srcInternal);
                if (srcRes) {
                    readbackSrcResources.push_back(srcRes);
                    readbackDstPtrs.push_back(dstData);
                    readbackSizes.push_back(static_cast<uint32_t>(dataSizeInBytes));
                }
            }
            else
            {
                flushPendingReadbacks();
                const void* src_data = nullptr;
                void* dst_data = nullptr;
                size_t bytes;
                RETURN_IF_ERROR(ort_api.GetTensorData(src_tensors[i], &src_data));
                RETURN_IF_ERROR(ort_api.GetTensorMutableData(dst_tensors[i], &dst_data));
                RETURN_IF_ERROR(ort_api.GetTensorSizeInBytes(src_tensors[i], &bytes));
                memcpy(dst_data, src_data, bytes);
            }
        }

        flushPendingReadbacks();
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

        //// CPU -> GPU copy (upload). Resolve dst via the host-accessible-aware resolver so a
        //// CUSTOM/L0 destination (OGA host-accessible decode input) doesn't hit GetDataInterface()'s
        //// IsDataInterface() assert (E_INVALIDARG) or the wrong DEFAULT-allocator decode.
        ID3D12Resource* dstData = ResolveTensorResource(dst);
        ORT_THROW_HR_IF(E_INVALIDARG, !dstData);
        const void* srcData = src->GetData();

        constexpr uint64_t dstOffset = 0;
        const auto dstState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; // GPU resources are always kept in UAV state

        m_uploadHeap->BeginUploadToGpu(dstData, dstOffset, dstState, AsByteSpan(srcData, dataSizeInBytes));

        // Continuously upload memory located in upload heaps during session initialization to avoid running out of it
        if (!m_sessionInitialized) {
            FlushUploadsIfReady();
        }
    }

    // Resolve the backing ID3D12Resource for a DML tensor that may live on the DEFAULT bucketized
    // allocator OR the host-accessible (CUSTOM/L0) allocator.
    //
    // READING GUIDE (why this differs from the old inline decode): host-accessible decode inputs are
    // OPT-IN (AMDGPU_DML_HOST_ACCESSIBLE=1). When the flag is OFF (the default), m_hostAccessibleAllocator
    // is null, so inside DecodeResource() the host-accessible lookup is skipped and this reduces to the
    // ORIGINAL DEFAULT path: m_allocator->DecodeDataHandle(handle)->GetResource(). Treat the
    // host-accessible cases below as non-existent unless the flag is set.
    //
    //   * DEFAULT-heap tensor: the data-interface IS a PluginDmlAllocationInfo handle -> first branch.
    //   * host-accessible tensor (flag on): the data ptr is the mapped cpu_ptr; DecodeResource() maps it
    //     back by pointer identity via the host-accessible allocator -> either branch resolves it.
    // GetDataInterface() is avoided on the raw-pointer branch because it throws when IsDataInterface()
    // is false (e.g. a host-accessible tensor classified as CPU).
    ID3D12Resource* PluginDmlExecutionProviderImpl::ResolveTensorResource(IMLOperatorTensor* tensor)
    {
        if (tensor->IsDataInterface()) {
            // DEFAULT-heap: the data interface is a PluginDmlAllocationInfo handle.
            Microsoft::WRL::ComPtr<IUnknown> di;
            tensor->GetDataInterface(di.GetAddressOf());
            if (di) {
                // Could be the allocation handle (DEFAULT) or, for host-accessible, the mapped ptr
                // reinterpreted — DecodeResource handles the host-accessible case by pointer identity.
                ID3D12Resource* r = DecodeResource(di.Get());
                if (r) return r;
            }
        }
        // Host-accessible / raw-pointer path: the tensor's data ptr is the mapped cpu_ptr.
        if (void* raw = tensor->GetData()) {
            ID3D12Resource* r = DecodeResource(raw);
            if (r) return r;
        }
        return nullptr;
    }

    void PluginDmlExecutionProviderImpl::GpuToGpuCopy(IMLOperatorTensor* src, IMLOperatorTensor* dst)
    {
        const size_t dataSizeInBytes = ComputeByteSizeFromTensor(*dst);

        ID3D12Resource* srcData = ResolveTensorResource(src);
        ID3D12Resource* dstData = ResolveTensorResource(dst);
        ORT_THROW_HR_IF(E_INVALIDARG, !srcData || !dstData);

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
        // All DML resources are buffers with D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.
        // D3D12 buffers implicitly promote COMMON→UAV on first use and decay UAV→COMMON
        // at ExecuteCommandLists boundaries. The null UAV barrier inserted after each
        // DML dispatch (DmlCommandRecorder::ExecuteOperator) handles inter-op ordering.
        // Explicit UAV↔COMMON transitions are therefore redundant for pure buffer workloads.
        return false;
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

            // Retained for IWinmlExecutionProvider interface compliance.
            // Not called in practice — TransitionsRequiredForOperator returns false.
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

        // Host-accessible decode inputs return a mapped CPU pointer, not a bucketized allocation
        // handle. Check that allocator first; it hands back an AddRef'd PluginDmlAllocationInfo.
        if (m_hostAccessibleAllocator) {
            if (IUnknown* host_alloc = m_hostAccessibleAllocator->TryGetAllocation(data_ptr)) {
                return host_alloc;  // already AddRef'd
            }
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
    std::shared_ptr<OrtAllocator> PluginDmlExecutionProviderImpl::GetHostAccessibleAllocator()
    {
        return m_hostAccessibleAllocator;  // null when CUSTOM/L0 unsupported -> caller falls back
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
#ifdef DML_PERF_PROFILE
        uint64_t _ore_t0 = PerfNowUs();
        PERF_TIMER_LOG("[PERF] OnRunEnd ENTER: ", _ore_t0, " us\n");
#endif

        if (GraphCaptureEnabled() && m_currentGraphAnnotationId != -1)
        {
            m_graphCapturingDone.insert(m_currentGraphAnnotationId);
        }

        // Flush any pending work to the GPU, but don't block for completion, permitting it
        // to overlap other work.
        Flush();

#ifdef DML_PERF_PROFILE
        { uint64_t _t = PerfNowUs(); PERF_TIMER_LOG("[PERF] OnRunEnd EXIT: ", _t, " us (+", _t - _ore_t0, " total)\n"); }
#endif

        return STATUS_OK;
    }

}  // namespace dml_ep
