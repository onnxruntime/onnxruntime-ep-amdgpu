// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <filesystem>

#include "DmlExecutionProvider/inc/IWinmlExecutionProvider.h"
#include "OperatorAuthorHelper/MLOperatorAuthorHelper.h"
#include "DmlExecutionProvider/DmlEdgeShapes.h"
#include "core/framework/op_kernel.h"
#include "core/framework/customregistry.h"
#include "core/framework/tensorprotoutils.h"
#include <wrl/client.h>
#include <wrl/implements.h>
#include "DmlExecutionProvider/MLOperatorAuthorImpl.h"
#include "dml_execution_provider.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/framework/onnxruntime_sequence_type_info.h"

namespace dml_ep {
class PluginDmlExecutionProviderImpl;
}

namespace Windows::AI::MachineLearning::Adapter
{


struct LazyPass
{
    bool requiredConstantCpuInputsAvailable;
    bool isInputTensorShapesDefined;
    bool isRequiresInputShapesAtCreation;

    LazyPass(bool isInputTensorShapesDefined, bool requiredConstantCpuInputsAvailable, bool isRequiresInputShapesAtCreation)
        : isInputTensorShapesDefined(isInputTensorShapesDefined),
          requiredConstantCpuInputsAvailable(requiredConstantCpuInputsAvailable),
          isRequiresInputShapesAtCreation(isRequiresInputShapesAtCreation)
    {}
};

// ABI-safe version for plugin that uses AbiSafeProtoHelperNodeContext (NO UNSAFE CASTS)
class PluginAbiSafeMLSupportQueryContext final : public OpNodeInfoWrapper<
          onnxruntime::AbiSafeProtoHelperNodeContext,
          WRL::Base<Microsoft::WRL::ChainInterfaces<IMLOperatorSupportQueryContextPrivate,
                                                    IMLOperatorAttributes,
                                                    IMLOperatorAttributes1>>,
          onnxruntime::null_type> 
{
public:
    PluginAbiSafeMLSupportQueryContext() = delete;

    PluginAbiSafeMLSupportQueryContext(onnxruntime::OpNodeProtoHelper<onnxruntime::AbiSafeProtoHelperNodeContext>* info,
                                       const AttributeMap* defaultAttributes,
                                       MLOperatorTensorGetter& mLOperatorTensorGetter);

    static Microsoft::WRL::ComPtr<PluginAbiSafeMLSupportQueryContext>
    Create(onnxruntime::OpNodeProtoHelper<onnxruntime::AbiSafeProtoHelperNodeContext>* info,
           const AttributeMap* defaultAttributes);
};

// LEGACY: This class has protobuf ABI violations (TypeProto*, AttributeProto*, ValueInfoProto via
// NodeArg) through ProtoHelperNodeContext. It is no longer used by PluginDmlAbiOpKernel, which now
// uses AbiSafeKernelCreationContext instead. This class is retained only because the legacy graph
// fusion code (GraphPartitioner, GraphDescBuilder, etc.) still references the type, though that
// code is never called from plugin EP entry points. Scheduled for removal.
class PluginOpKernelInfoWrapper
    : public OpNodeInfoWrapper<onnxruntime::ProtoHelperNodeContext,
                               WRL::Base<Microsoft::WRL::ChainInterfaces<
                                             IMLOperatorKernelCreationContextNodeWrapperPrivate,
                                             IMLOperatorKernelCreationContextPrivate, IMLOperatorKernelCreationContext>,
                                         IMLOperatorTensorShapeDescription, IMLOperatorTensorShapeDescriptionPrivate,
                                         IMLOperatorAttributes1>,
                               onnxruntime::null_type> {
public:
    PluginOpKernelInfoWrapper(const onnxruntime::OpKernelInfo* kerneInfo, IUnknown* abiExecutionObject,
                        const EdgeShapes* inputShapeOverrides, const EdgeShapes* inferredOutputShapes,
                        bool allowInputShapeQuery, bool allowOutputShapeQuery, bool isInternalOperator,
                        const AttributeMap* defaultAttributes, gsl::span<const uint32_t> requiredConstantCpuInputs,
                        MLOperatorTensorGetter& constantInputGetter,
                        const dml_ep::PluginDmlExecutionProviderImpl* pluginDmlEp,
                        const onnxruntime::OpKernelContext* kernelContext = nullptr);

    // HasTensorShapeDescription returns false if and only if the kernel is registered using
    // MLOperatorKernelOptions::AllowDynamicInputTensorSizes.    If this flag is specified and upstream
    // shapes are known when the kernel is created, HasTensorShapeDescription still returns false.
    bool STDMETHODCALLTYPE HasTensorShapeDescription() const noexcept override;
    HRESULT STDMETHODCALLTYPE
    GetTensorShapeDescription(IMLOperatorTensorShapeDescription** shapeInfo) const noexcept override;

    void STDMETHODCALLTYPE GetExecutionInterface(IUnknown** executionInterface) const noexcept override;

    // IMLOperatorTensorShapeDescription methods.
    HRESULT STDMETHODCALLTYPE GetOutputTensorDimensionCount(uint32_t inputIndex,
                                                            uint32_t* dimensionCount) const noexcept override;
    bool STDMETHODCALLTYPE HasOutputShapeDescription() const noexcept override;
    HRESULT STDMETHODCALLTYPE GetOutputTensorShape(uint32_t inputIndex, uint32_t dimensionCount,
                                                   uint32_t* dimensions) const noexcept override;

    bool STDMETHODCALLTYPE IsDmlGraphNode() const noexcept override { return false; }

    HRESULT STDMETHODCALLTYPE
    SetDmlOperator(_In_ const MLOperatorGraphDesc* operatorGraphDesc) const noexcept override {
        return E_NOTIMPL;
    }

    // IMLOperatorKernelCreationContextNodeWrapperPrivate methods.

    uint32_t STDMETHODCALLTYPE GetUtf8NameBufferSizeInBytes() const noexcept override;
    HRESULT STDMETHODCALLTYPE GetUtf8Name(uint32_t bufferSizeInBytes, char* name) const noexcept override;

    uint32_t STDMETHODCALLTYPE GetWideNameBufferSizeInBytes() const noexcept override;
    HRESULT STDMETHODCALLTYPE GetWideName(uint32_t bufferSizeInBytes, wchar_t* name) const noexcept override;

    HRESULT STDMETHODCALLTYPE
    GetExecutionProvider(_Outptr_result_maybenull_ IUnknown** executionProvider) const noexcept override {
        return m_winmlProvider.CopyTo(executionProvider);
    }

private:
    // For shape info, in addition to the info
    const EdgeShapes* m_inferredOutputShapes = nullptr;
    bool m_allowInputShapeQuery = false;
    bool m_allowOutputShapeQuery = false;

    bool m_internalOperator = false;
    Microsoft::WRL::ComPtr<IWinmlExecutionProvider> m_winmlProvider;

    const onnxruntime::OpKernelInfo* m_impl = nullptr;

    // The execution object returned through the ABI, which may vary according to kernel
    // registration options.
    Microsoft::WRL::ComPtr<IUnknown> m_abiExecutionObject;
};

class PluginOpKernelContextWrapper : public WRL::Base<IMLOperatorKernelContext, IMLOperatorKernelContextPrivate>,
                               public Closable {
public:
    ~PluginOpKernelContextWrapper();

    PluginOpKernelContextWrapper(onnxruntime::OpKernelContext* context,
                                 const dml_ep::PluginDmlExecutionProviderImpl* pluginDmlEp,
                           bool isInternalOperator, const EdgeShapes* outputShapes);

    bool STDMETHODCALLTYPE IsSequenceInputTensor(uint32_t inputIndex) const noexcept override;
    HRESULT STDMETHODCALLTYPE GetSequenceInputInfo(uint32_t inputIndex, uint32_t* inputCount,
                                                   MLOperatorTensorDataType* dataType) const noexcept override;
    HRESULT STDMETHODCALLTYPE GetSequenceInputTensor(uint32_t inputIndex, uint32_t sequenceIndex,
                                                     IMLOperatorTensor** tensor) const noexcept override;

    HRESULT STDMETHODCALLTYPE PrepareSequenceOutput(uint32_t outputIndex,
                                                    MLOperatorTensorDataType dataType) const noexcept override;

    HRESULT STDMETHODCALLTYPE GetSequenceOutputTensor(uint32_t outputIndex, uint32_t sequenceIndex,
                                                      MLOperatorTensorDataType dataType, uint32_t dimensions,
                                                      const uint32_t* dimensionSizes, bool gpuOutput,
                                                      IMLOperatorTensor** tensor) const noexcept override;

    HRESULT STDMETHODCALLTYPE GetInputTensor(uint32_t inputIndex, IMLOperatorTensor** tensor) const noexcept override;

    HRESULT STDMETHODCALLTYPE GetOutputTensor(uint32_t outputIndex, IMLOperatorTensor** tensor) noexcept override;
    HRESULT STDMETHODCALLTYPE GetOutputTensor(uint32_t outputIndex, uint32_t dimensions, const uint32_t* dimensionSizes,
                                              IMLOperatorTensor** tensor) noexcept override;

    HRESULT STDMETHODCALLTYPE AllocateTemporaryData(size_t size, IUnknown** data) const noexcept override;
    HRESULT STDMETHODCALLTYPE AllocateTemporaryData(size_t size, IUnknown** data, uint64_t* allocId) const;

    void STDMETHODCALLTYPE GetExecutionInterface(IUnknown** executionInterface) const noexcept override;

    void Close() override;

    std::vector<IMLOperatorTensor*> GetInputTensors();
    std::vector<IMLOperatorTensor*> GetOutputTensors(const EdgeShapes& outputShapes);

    onnxruntime::OpKernelContext* GetOpKernelContext() { return m_impl; }

    static bool IsAllocationInterface(const ::OrtMemoryInfo& info);
    static void TranslateAllocationDataToAbi(IWinmlExecutionProvider* winmlProvider,
                                             bool isInternalOperator,
                                             const ::OrtMemoryInfo& allocInfo,
                                             IUnknown* allocation,
                                             IUnknown** abiAllocation);

protected:
    void ClearTempAllocations();
    void TransitionResourcesForOperatorIfRequired(bool isBeforeOp);

    // Lifetime is managed by the caller and guaranteed to outlive this class
    onnxruntime::OpKernelContext* m_impl = nullptr;
    const EdgeShapes* m_outputShapes = nullptr;

    std::vector<std::vector<Microsoft::WRL::ComPtr<TensorWrapper>>> m_inputTensors;
    std::vector<std::vector<Microsoft::WRL::ComPtr<TensorWrapper>>> m_outputTensors;

    const dml_ep::PluginDmlExecutionProviderImpl* m_provider = nullptr;
    Microsoft::WRL::ComPtr<IWinmlExecutionProvider> m_winmlProvider;
    bool m_internalOperator = false;

    // The execution object returned to the kernel may vary according to kernel execution options
    Microsoft::WRL::ComPtr<IUnknown> m_providerExecutionObject;
    Microsoft::WRL::ComPtr<IUnknown> m_abiExecutionObject;

    // Temporary allocations created by the kernel.  These will be freed to the allocator following
    // Compute being called on the kernel.  This list is used to maintain their lifetime.
    mutable std::vector<Microsoft::WRL::ComPtr<IUnknown>> m_temporaryAllocations;
    mutable std::vector<Microsoft::WRL::ComPtr<IUnknown>> m_temporaryAbiAllocations;
};

class PluginDmlAbiOpKernel : public onnxruntime::OpKernel
{
 public:
    PluginDmlAbiOpKernel(
        IMLOperatorKernelFactory* operatorFactory,
        const onnxruntime::OpKernelInfo& kerneInfo,
        bool requiresInputShapesAtCreation,
        bool requiresOutputShapesAtCreation,
        bool isInternalOperator,
        gsl::span<const uint32_t> requiredConstantCpuInputs,
        IMLOperatorShapeInferrer* shapeInferrer,
        const AttributeMap* defaultAttributes,
        const dml_ep::PluginDmlExecutionProviderImpl* pluginDmlEp);

    Ort::Status Compute(onnxruntime::OpKernelContext* context) const override;
 protected:
    static bool IsAllocationInterface(const ::OrtMemoryInfo& info) {
        return strcmp(info.name.c_str(), onnxruntime::CPU) &&
            !(info.mem_type == ::OrtMemType::OrtMemTypeCPUOutput || info.mem_type == ::OrtMemType::OrtMemTypeCPUInput);
    }
    bool RequiresLazyInitialization() const { return (m_operatorFactory != nullptr) && !m_lazyInitialized; };
    void SetLazyInitialized() const { m_lazyInitialized = true; };

    EdgeShapes GetInputShapes(onnxruntime::OpKernelContext* context) const;

    bool InputTensorShapesDefined() const;
    bool InputSizesInferencedFromSchema() const;
    void InferAndVerifyOutputSizes(gsl::span<const uint32_t> requiredConstantCpuInputs, MLOperatorTensorGetter& constantInputGetter, const EdgeShapes* inputShapes, EdgeShapes& outputShapes, OrtKernelContext* ortContext = nullptr) const;
    bool m_requiresInputShapesAtCreation = false;
    bool m_requiresOutputShapesAtCreation = false;

    mutable Microsoft::WRL::ComPtr<IMLOperatorKernel> m_kernel;

    // This is null unless the kernel requires lazy initialization
    Microsoft::WRL::ComPtr<IMLOperatorKernelFactory> m_operatorFactory;
    mutable volatile bool m_lazyInitialized = false;

    Microsoft::WRL::ComPtr<IMLOperatorShapeInferrer> m_shapeInferrer;

    // Used to determine whether anything has changed since creation when shapes or
    // inputs treated as constant by the operator are not inferred / constant.
    mutable EdgeShapes m_inputShapesOfKernelInference;

    struct TensorContent
    {
        bool isValid;
        std::vector<uint32_t> shape;
        MLOperatorTensorDataType type;
        std::vector<std::byte> data;
    };

    mutable std::vector<std::variant<TensorContent, std::vector<TensorContent>>> m_constantInputTensorContentsOfKernel;

    mutable std::mutex m_mutex;
    mutable EdgeShapes m_inferredOutputShapes;

    Microsoft::WRL::ComPtr<IWinmlExecutionProvider> m_winmlProvider;
    bool m_internalOperator = false;
    std::vector<uint32_t> m_requiredConstantCpuInputs;

    // The execution object returned through the ABI may vary according to kernel
    // registration options.
    Microsoft::WRL::ComPtr<IUnknown> m_providerExecutionObject;
    Microsoft::WRL::ComPtr<IUnknown> m_abiExecutionObject;

    const AttributeMap* m_defaultAttributes = nullptr;
    const dml_ep::PluginDmlExecutionProviderImpl* m_dmlPluginExecutionProvider = nullptr;

    const OrtKernelInfo* m_ortKernelInfo = nullptr;
    const OrtApi* m_ortApi = nullptr;
    std::vector<std::string> m_tensorAttributeNames;  // Tensor-typed ONNX attribute names for FetchAllTensorAttributes

private:
    bool RequiredCpuInputChanged(const Microsoft::WRL::ComPtr<IMLOperatorTensor>& constantTensor, uint32_t index) const;
    bool RequiredCpuInputChanged(const std::vector<ComPtr<IMLOperatorTensor>>& constantTensorSequence, uint32_t index) const;
    void FillConstantInputs(const Microsoft::WRL::ComPtr<IMLOperatorTensor>& constantTensor, onnxruntime::OpKernelContext* context, uint32_t index) const;
    void FillConstantInputs(const std::vector<Microsoft::WRL::ComPtr<IMLOperatorTensor>>& constantTensor, onnxruntime::OpKernelContext* context, uint32_t index) const;
};

}    // namespace Windows::AI::MachineLearning::Adapter
