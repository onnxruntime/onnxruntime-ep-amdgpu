// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "plugin_ep_utils.h"
#include "cpu_allocator.h"
#include "dml_bucketized_buffer_allocator.h"
#include "ort_node_adapter.h"
//#include "DmlExecutionProvider/DmlCommon.h"
#include "dml_common.h"
#include "DmlExecutionProvider/inc/IWinmlExecutionProvider.h"
#include "DmlExecutionProvider/MLOperatorAuthorImpl.h"
#include "DmlExecutionProvider/AbiCustomRegistry.h"
#include "DmlExecutionProvider/inc/DmlExecutionProvider.h"
#include "DmlExecutionProvider/DmlCommittedResourceAllocator.h"
#include "dml_execution_provider.h"
//#include "iallocator_to_ort_allocator_adapter.h"
#include "core/common/inlined_containers.h"
#include "core/graph/ep_api_types.h"
#include "dml_abi_custom_registry.h"
#include "dml_abi_kernel.h"
#include "DmlExecutionProvider/ErrorHandling.h"
#include <wil/wrl.h>
#include <wil/result.h>
#include "OperatorAuthorHelper/OperatorHelper.h"
#include "DmlExecutionProvider/DmlReusedCommandListState.h"
#include "DmlExecutionProvider/AllocationInfo.h"
#include "dml_data_transfer.h"
#include "core/graph/graph.h"
//#include "core/graph/abi_graph_types.h"
#include "core/graph/node_arg.h"
#include "core/framework/kernel_registry.h"
//#include "DmlExecutionProvider/DmlCommon.h"
//#include "dml_common.h"


#include <d3d12.h>
#include <dxgi1_6.h>
#include <directx/d3dx12.h>
#include <wrl/client.h>
#include <DirectML.h>

#include <queue>

using Microsoft::WRL::ComPtr;

class DMLDataTransfer;

namespace Dml {

class ExecutionProviderPlugin 
    : public OrtEp
    , public ApiPtrs
{
public:
    ExecutionProviderPlugin(
        const ApiPtrs& api_ptrs,
        std::string_view name,
        ID3D12Device* d3d12_device,
        IDMLDevice* dml_device,
        ComPtr<PluginDmlExecutionContext> executionContext);

            //const OrtSessionOptions *session_options, const OrtLogger *logger,
    ~ExecutionProviderPlugin();

    void ReleaseCompletedReferences();

    std::shared_ptr<Dml::PluginDmlExecutionProviderImpl> GetInternetalExecutionProvider();
    bool IsGetCapabilityCompleted();

    DMLDataTransfer* GetDataTransfer();

    void Flush() const;

    static gsl::span<const std::byte> AsByteSpan(const void* data, size_t sizeInBytes);

    static gsl::span<std::byte> AsByteSpan(void* data, size_t sizeInBytes);

    bool GraphCaptureEnabled() const noexcept;

    void Release();

private:
    // State passed as kernel_create_func_state — holds information needed for kernel creation
    // stored in a long-lived vector owned by the ExecutionProviderPlugin.
    struct KernelCreateFuncState {
        const OrtApi* ort_api_ptr = nullptr;
        std::string operator_name;  // For excluding operators from ABI-safe path

        // TRY ABI-SAFE PATH FIRST
        IMLOperatorKernelFactory* kernel_factory = nullptr;
        IMLOperatorShapeInferrer* shape_inferrer = nullptr;
        const Windows::AI::MachineLearning::Adapter::AttributeMap* default_attributes = nullptr;
        std::vector<uint32_t> required_constant_cpu_inputs;
        bool requires_input_shapes_at_creation = false;
        bool requires_output_shapes_at_creation = false;
        bool is_internal_operator = false;  // For resource state transitions (MemcpyToHost/FromHost)
        std::vector<std::string> tensor_attribute_names;  // Tensor-typed ONNX attribute names (e.g., ConstantOfShape's "value")
        const Dml::PluginDmlExecutionProviderImpl* dml_execution_provider = nullptr;

        // FALLBACK: ABI-UNSAFE PATH (when ABI-safe fails, e.g., E_UNEXPECTED)
        onnxruntime::KernelCreateFn* kernel_create_fn = nullptr;
    };

    // Adapter to wrap onnxruntime::OpKernel as OrtKernelImpl (for unsafe fallback)
    struct DmlKernelImplAdapter : OrtKernelImpl {
        std::unique_ptr<onnxruntime::OpKernel> internal_kernel;
        const OrtApi* ort_api_ptr;
    };

    static OrtStatus* ORT_API_CALL DmlKernelImplAdapter_Compute(
        OrtKernelImpl* this_ptr,
        OrtKernelContext* context) noexcept;

    static void ORT_API_CALL DmlKernelImplAdapter_Release(OrtKernelImpl* this_ptr) noexcept;

    struct OrtKernelDefDeleter {
        const OrtEpApi* ep_api = nullptr;
        void operator()(OrtKernelDef* d) const {
            if (d && ep_api)
                ep_api->ReleaseKernelDef(d);
        }
    };
    using UniqueOrtKernelDef = std::unique_ptr<OrtKernelDef, OrtKernelDefDeleter>;


    static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;

    static OrtStatus* ORT_API_CALL GetCapabilityImpl(
        OrtEp* this_ptr,
        const OrtGraph* graph,
        OrtEpGraphSupportInfo* graph_support_info) noexcept;

    static OrtStatus* ORT_API_CALL CompileImpl(
        _In_ OrtEp* this_ptr,
        _In_ const OrtGraph** graphs,
        _In_ const OrtNode** fused_nodes,
        _In_ size_t count,
        _Out_writes_all_(count) OrtNodeComputeInfo** node_compute_infos,
        _Out_writes_(count) OrtNode** ep_context_nodes) noexcept;

    static void ORT_API_CALL ReleaseNodeComputeInfosImpl(
        OrtEp* this_ptr,
        OrtNodeComputeInfo** node_compute_infos,
        size_t num_node_compute_infos) noexcept;

    static OrtStatus* ORT_API_CALL GetPreferredDataLayoutImpl(
        _In_ OrtEp* this_ptr,
        _Out_ OrtEpDataLayout* preferred_data_layout) noexcept;

    static OrtStatus* ORT_API_CALL ShouldConvertDataLayoutForOpImpl(
        _In_ OrtEp* this_ptr,
        _In_z_ const char* domain,
        _In_z_ const char* op_type,
        OrtEpDataLayout target_data_layout,
        _Out_ int* should_convert) noexcept;

    static OrtStatus* ORT_API_CALL SetDynamicOptionsImpl(
        _In_ OrtEp* this_ptr,
        const char* const* option_keys,
        const char* const* option_values,
        size_t num_options) noexcept;

    static OrtStatus* ORT_API_CALL OnRunStartImpl(
        _In_ OrtEp* this_ptr,
        const OrtRunOptions* run_options) noexcept;

    static OrtStatus* ORT_API_CALL OnRunEndImpl(
        _In_ OrtEp* this_ptr,
        const OrtRunOptions* run_options,
        bool sync_stream) noexcept;

    static OrtStatus* ORT_API_CALL CreateAllocatorImpl(
        _In_ OrtEp* this_ptr,
        const OrtMemoryInfo* memory_info,
        OrtAllocator** allocator) noexcept;

    static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(
        _In_ OrtEp* this_ptr,
        _In_ const OrtMemoryDevice* memory_device,
        _Outptr_ OrtSyncStreamImpl** stream) noexcept;

    static const char* ORT_API_CALL GetCompiledModelCompatibilityInfoImpl(
        _In_ OrtEp* this_ptr,
        _In_ const OrtGraph* graph) noexcept;

    static OrtStatus* ORT_API_CALL GetKernelRegistryImpl(
        _In_ OrtEp* this_ptr,
        _Outptr_ const OrtKernelRegistry** kernel_registry) noexcept;

    static OrtStatus* ORT_API_CALL IsConcurrentRunSupportedImpl(
        _In_ OrtEp* this_ptr,
        _Out_ bool* is_supported) noexcept;

    static OrtStatus* ORT_API_CALL OnSessionInitializationEndImpl(_In_ OrtEp* this_ptr) noexcept;

    static OrtStatus* DmlKernelCreateFuncAdapter(void* kernel_create_func_state, const OrtKernelInfo* info,
                                                              OrtKernelImpl** kernel_out) noexcept;

    uint32_t GetSupportedDeviceDataTypeMask() const;
    std::unordered_set<size_t> GetCpuPreferredNodes(const OrtGraph* graph, OrtEpGraphSupportInfo* graph_support_info, gsl::span<const OrtNode*> tentative_nodes);

    bool IsCpuAllocator(const OrtMemoryInfo* memory_info);
    bool IsGpuAllocator(const OrtMemoryInfo* memory_info);
    bool IsCustomOpShader(const Dml::OrtNodeAdapter& adapter);
    bool IsCpuOnDmlOperator(const Dml::OrtNodeAdapter& adapter);
    bool IsDmlSequenceOperator(const Dml::OrtNodeAdapter& adapter);
    bool IsSmallInitializer(const OrtGraph* graph, const OrtValueInfo* valueInfo);

    bool IsNodeSupportedByDml(
        const OrtNode* node,
        OrtEpGraphSupportInfo* graph_support_info,
        uint32_t supportedDeviceDataTypeMask);

    bool DoesNodeContainSupportedDataTypes(
        const OrtNode* node,
        const Windows::AI::MachineLearning::Adapter::InternalRegistrationInfo* regInfo,
        uint32_t supportedDeviceDataTypeMask, // Each bit corresponds to each DML_TENSOR_DATA_TYPE.
        bool native16BitShaderOpsSupported);

    bool TryGetTensorDataType(
        const onnxruntime::NodeArg& nodeArg,
        _Out_ MLOperatorEdgeType* edgeType,
        _Out_ MLOperatorTensorDataType* onnxElementType);

    OrtStatus* ConvertKernelRegistryToOrtKernelRegistry();

    ONNXTensorElementDataType GetElementTypeFromMLDataType(onnxruntime::MLDataType);

    std::string name_{};
    const OrtLogger* logger_;

    const std::string vendor_{"AMD"};
    static constexpr uint32_t vendor_id_{0x1002};
    const std::string ep_version_{"0.1.0"};

    ComPtr<ID3D12Device> d3d12_device;
    ComPtr<IDMLDevice> m_dmlDevice;
    ComPtr<PluginDmlExecutionContext> m_context;
    //std::shared_ptr<OrtAllocator> m_cpuInputAllocator;
    std::unordered_map<int, std::vector<std::unique_ptr<DmlReusedCommandListState>>> m_capturedGraphs;
    std::shared_ptr<onnxruntime::KernelRegistry> m_kernelRegistry;
    std::shared_ptr<OrtKernelRegistry> m_ortKernelRegistry;
    std::shared_ptr<const Windows::AI::MachineLearning::Adapter::InternalRegistrationInfoMap> m_internalRegInfoMap;

    KernelCreateFuncState m_kernelCreateFuncStateTemplate;
    std::vector<std::unique_ptr<KernelCreateFuncState>> m_kernelCreateFuncStates;
    std::vector<UniqueOrtKernelDef> m_ortKernelDefs;

    bool m_native16BitShaderOpsSupported = false;
    bool m_isMcdmDevice = false;
    bool m_areCustomHeapsSupported = false;
    bool m_areMetacommandsEnabled = true;
    bool m_graphCaptured = false;
    bool m_graphCaptureEnabled = false;
    bool m_isGetCapabilityCompleted = false;
    bool m_closed = false;
    bool m_sessionInitialized = false;
    bool m_cpuSyncSpinningEnabled = false;
    bool m_memoryArenaDisabled = false;

    int m_currentGraphAnnotationId = 0;
    std::unordered_set<int> m_graphCapturingDone;

    std::shared_ptr<Dml::PluginDmlExecutionProviderImpl> m_executionProvider;
    std::unique_ptr<DMLDataTransfer> m_dataTransfer;
    static constexpr std::chrono::milliseconds m_batchFlushInterval = std::chrono::milliseconds(10);

    int64_t kSmallInitializerThreshold = 100;
};

}  // namespace Dml
