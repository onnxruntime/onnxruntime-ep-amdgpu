// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "common/plugin_ep_utils.h"
#include "cpu_allocator.h"
#include "dml_readback_heap.h"
#include "dml_pooled_upload_heap.h"
#include "dml_bucketized_buffer_allocator.h"
#include "dml_common.h"
#include "DmlExecutionProvider/inc/IWinmlExecutionProvider.h"
#include "DmlExecutionProvider/inc/DmlExecutionProvider.h"
#include "DmlExecutionProvider/DmlCommittedResourceAllocator.h"
#include "DmlExecutionProvider/IExecutionProvider.h"
#include "iallocator_to_ort_allocator_adapter.h"
#include "core/common/inlined_containers.h"
#include "core/graph/ep_api_types.h"
#include "dml_abi_custom_registry.h"
#include "DmlExecutionProvider/ErrorHandling.h"
#include <wil/wrl.h>
#include <wil/result.h>
#include "OperatorAuthorHelper/OperatorHelper.h"
#include "DmlExecutionProvider/AllocationInfo.h"
#include "DmlExecutionProvider/DmlCommittedResourceWrapper.h"
#include "DmlExecutionProvider/precomp.h"
#include "DmlExecutionProvider/DmlReusedCommandListState.h"
#include "DmlExecutionProvider/ExecutionContext.h"
#include "dml_execution_context.h"
#include "core/framework/data_types.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <directx/d3dx12.h>
#include <wrl/client.h>
#include <DirectML.h>
#include <wrl/implements.h>

#define IID_GRAPHICS_PPV_ARGS IID_PPV_ARGS

namespace onnxruntime {
class IResourceAccountant;
class GraphOptimizerRegistry;
}

namespace WRL {
template <typename... TInterfaces>
using Base = Microsoft::WRL::RuntimeClass<
    Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
    TInterfaces...>;
}

namespace dml_ep {

        class PooledUploadHeap;
    class ReadbackHeap;
    class ExecutionContext;
    class BucketizedBufferAllocator;
    class ExecutionProvider;

    class PluginDmlExecutionProviderImpl 
        : public WRL::Base<IExecutionProvider, Windows::AI::MachineLearning::Adapter::IWinmlExecutionProvider>
        , public ApiPtrs
    {
    public:
        virtual ~PluginDmlExecutionProviderImpl();

        PluginDmlExecutionProviderImpl(
            IDMLDevice* dmlDevice,
            ID3D12Device* d3d12Device,
            PluginDmlExecutionContext* executionContext,
            const ApiPtrs& api_ptrs,
            bool enableMetacommands,
            bool enableGraphCapture,
            bool enableCpuSyncSpinning,
            bool disableMemoryArena);

        void ReleaseCompletedReferences();

        OrtStatus* CopyTensorsPlugin(const OrtValue** src_tensors_ptr,
                                    OrtValue** dst_tensors_ptr, OrtSyncStream** streams_ptr,
                                    size_t num_tensors);

    public: // implements IExecutionProvider
        STDMETHOD(GetD3DDevice)(_COM_Outptr_ ID3D12Device** d3dDevice) const noexcept final;

        STDMETHOD(GetDmlDevice)(_COM_Outptr_ IDMLDevice** dmlDevice) const noexcept final;

        STDMETHOD(ExecuteCommandList)(
            ID3D12GraphicsCommandList* commandList,
            _Outptr_ ID3D12Fence** fence,
            _Out_ uint64_t* completionValue
            ) const noexcept final;

        STDMETHOD(AddUAVBarrier)() const noexcept final;

        STDMETHOD(InitializeOperator)(
            IDMLCompiledOperator* op,
            _In_opt_ const DML_BUFFER_BINDING* persistentResourceBinding,
            gsl::span<const DML_BUFFER_BINDING> inputBindings
            ) const noexcept final;

        STDMETHOD(ExecuteOperator)(
            IDMLCompiledOperator* op,
            _In_opt_ const DML_BUFFER_BINDING* persistentResourceBinding,
            gsl::span<IMLOperatorTensor*> inputTensors,
            gsl::span<IMLOperatorTensor*> outputTensors
            ) const noexcept final;

        STDMETHOD(ExecuteOperator)(
            IDMLCompiledOperator* op,
            _In_opt_ const DML_BUFFER_BINDING* persistentResourceBinding,
            gsl::span<DML_BINDING_DESC> inputTensors,
            gsl::span<DML_BINDING_DESC> outputTensors
            ) const noexcept final;

        STDMETHOD(CopyTensor)(IMLOperatorTensor* dst, IMLOperatorTensor* src) const noexcept final;
        STDMETHOD(CopyTensors)(gsl::span<IMLOperatorTensor*> dst, gsl::span<IMLOperatorTensor*> src) const noexcept final;

        STDMETHOD(FillTensorWithPattern)(
            IMLOperatorTensor* dst,
            gsl::span<const std::byte> rawValue
            ) const noexcept final;

        STDMETHOD(UploadToResource)(ID3D12Resource* dstData, const void* srcData, uint64_t srcDataSize) const noexcept final;

        //std::vector<std::unique_ptr<onnxruntime::ComputeCapability>>
        //GetCapability(
        //    const onnxruntime::GraphViewer& graph,
        //    const onnxruntime::IExecutionProvider::IKernelLookup& kernel_lookup,
        //    const onnxruntime::GraphOptimizerRegistry& graph_optimizer_registry,
        //    onnxruntime::IResourceAccountant* resource_accountant,
        //    const onnxruntime::Ort::Logger& logger) const;

        uint32_t GetSupportedDeviceDataTypeMask() const;

        // IWinmlExecutionProvider methods
        void QueueReference(IUnknown* object) override;

        void GetShadowCopyIfRequired(
            bool isInternalOperator,
            IUnknown* data,
            IUnknown** dataCopy) const override;

        void GetABIDataInterface(
            bool isInternalOperator,
            IUnknown* data,
            IUnknown** abiData) const override;

       uint64_t TryGetPooledAllocationId(
            IUnknown* data,
            bool isInternalOperator) override;

        void GetABIExecutionInterfaceAndInvalidateState(
            bool isInternalOperator,
            IUnknown** abiExecutionObject) const override;

        bool TransitionsRequiredForOperator(
            bool isInternalOperator
        ) override;

        void TransitionResourcesForOperator(
            bool isBeforeOp,
            uint32_t resourceCount,
            IUnknown** resources
        ) override;

        STDMETHOD_(D3D12_COMMAND_LIST_TYPE, GetCommandListTypeForQueue)() const override;
        STDMETHOD_(void, Flush)() const override;

        // Waits for flushed work, discards unflushed work, and discards associated references to
        // prevent circular references.  Must be the last call on the object before destruction.
        void Close() override;

        void WaitForOutstandingWork();

        // Allocate a resource from pools.  Releasing pooledResource returns it to the pool.
        STDMETHOD(AllocatePooledResource)(
            size_t size,
            AllocatorRoundingMode roundingMode,
            ID3D12Resource **d3dResource,
            IUnknown* *pooledResource
        ) const noexcept final;

        STDMETHOD_(ID3D12Resource*, DecodeResource)(void* allocation) const noexcept final;

        std::shared_ptr<onnxruntime::KernelRegistry> GetKernelRegistry() const
        {
            return m_kernelRegistry;
        }

        STDMETHOD_(bool, IsMcdmDevice)() const noexcept final;
        STDMETHOD_(bool, CustomHeapsSupported)() const noexcept final;

        STDMETHOD_(bool, MetacommandsEnabled)() const noexcept final;
        bool GraphCaptureEnabled() const noexcept;
        bool GraphCaptured(int graph_annotation_id) const;
        Ort::Status ReplayGraph(int graph_annotation_id);
        Ort::Status OnRunStart(const OrtRunOptions& run_options);
        Ort::Status OnRunEnd();
        int GetCurrentGraphAnnotationId() const { return m_currentGraphAnnotationId; }
        void AppendCapturedGraph(int annotationId, std::unique_ptr<DmlReusedCommandListState> capturedGraph);
        bool CpuSyncSpinningEnabled() const noexcept;
        std::shared_ptr<OrtAllocator> GetGpuAllocator();
        std::shared_ptr<OrtAllocator> GetCpuInputAllocator();

        std::shared_ptr<const Windows::AI::MachineLearning::Adapter::InternalRegistrationInfoMap>
        GetInternalRegistrationInfoMap() const;

        // Get the allocation object (IUnknown*) from a data pointer
        // This is ABI-safe because both sides use the same allocator instance
        IUnknown* GetAllocationFromDataPointer(void* data_ptr) const;

        void IncreasePartitionKernelPrefixVal() const
        {
            m_partitionKernelPrefixVal++;
        }

        uint64_t GetPartitionKernelPrefixVal() const
        {
            return m_partitionKernelPrefixVal;
        }

        Ort::Status OnSessionInitializationEnd();
        std::vector<OrtAllocator*> CreatePreferredAllocators();

    private:
        void Initialize(ID3D12CommandQueue* queue, ExecutionProvider& executionProvider);

        //bool IsNodeSupportedByDml(
        //    const onnxruntime::Node& node,
        //    const onnxruntime::IExecutionProvider::IKernelLookup& kernel_lookup,
        //    uint32_t supportedDeviceDataTypeMask // Each bit corresponds to each DML_TENSOR_DATA_TYPE.
        //) const;

        void FlushUploadsIfReady() const;

        void CpuToGpuCopy(IMLOperatorTensor* src, IMLOperatorTensor* dst);
        void GpuToGpuCopy(IMLOperatorTensor* src, IMLOperatorTensor* dst);
        void GpuToCpuCopy(IMLOperatorTensor* src, IMLOperatorTensor* dst);
        bool IsGpuTensor(const onnxruntime::Tensor& tensor);

        Microsoft::WRL::ComPtr<ID3D12Device> m_d3d12Device;
        Microsoft::WRL::ComPtr<IDMLDevice> m_dmlDevice;
        bool m_isMcdmDevice = false;
        bool m_areCustomHeapsSupported = false;
        bool m_areMetacommandsEnabled = true;
        int m_currentGraphAnnotationId = 0;
        bool m_native16BitShaderOpsSupported = false;
        bool m_graphCaptured = false;
        bool m_graphCaptureEnabled = false;

        std::unordered_map<int, std::vector<std::unique_ptr<DmlReusedCommandListState>>> m_capturedGraphs;
        std::unordered_set<int> m_graphCapturingDone;
        bool m_sessionInitialized = false;
        bool m_cpuSyncSpinningEnabled = false;
        bool m_memoryArenaDisabled = false;

        Microsoft::WRL::ComPtr<PluginDmlExecutionContext> m_context;
        std::unique_ptr<OrtMemoryInfo> m_cpuMemInfo;
        std::unique_ptr<PluginDmlPooledUploadHeap> m_uploadHeap;
        std::unique_ptr<PluginDmlReadbackHeap> m_readbackHeap;
        std::shared_ptr<DmlBucketizedBufferAllocator> m_allocator;
        std::shared_ptr<CpuAllocator> m_cpuInputAllocator;
        std::shared_ptr<onnxruntime::KernelRegistry> m_kernelRegistry;
        std::shared_ptr<const Windows::AI::MachineLearning::Adapter::InternalRegistrationInfoMap> m_internalRegInfoMap;
        mutable uint64_t m_partitionKernelPrefixVal = 0;
        bool m_closed = false;
        mutable std::chrono::time_point<std::chrono::steady_clock> m_lastUploadFlushTime;

        static constexpr std::chrono::milliseconds m_batchFlushInterval = std::chrono::milliseconds(10);
    };

}  // namespace dml_ep
