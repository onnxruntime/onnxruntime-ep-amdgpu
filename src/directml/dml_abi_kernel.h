// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License

#pragma once

// Only use public C API headers for ABI safety
#include <onnxruntime_c_api.h>

#include "OperatorAuthorHelper/MLOperatorAuthorHelper.h"
#include <wrl/client.h>
#include <wrl/implements.h>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstddef>
#include <cstring>

#include "DmlExecutionProvider/DmlEdgeShapes.h"

// Forward declarations
namespace Windows::AI::MachineLearning::Adapter {
    class AttributeValue;
    using AttributeMap = std::map<std::string, AttributeValue>;
}

// IWinmlExecutionProvider must be fully defined before this header is parsed
// because AbiSafeKernelContext holds ComPtr<IWinmlExecutionProvider>.
// Include dml_execution_provider.h or IWinmlExecutionProvider.h before this header.
// (dml_abi_kernel.cpp does this explicitly; other TUs reach it via precomp.h or dml_ep.h.)

namespace Dml {

// Forward declarations
class PluginDmlExecutionProviderImpl;

// ============================================================================
// ABI-Safe Tensor Wrapper - implements IMLOperatorTensor using C API
// ============================================================================

class AbiSafeTensor : public Microsoft::WRL::RuntimeClass<
    Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
    IMLOperatorTensor>
{
public:
    AbiSafeTensor(
        const OrtValue* ort_value,
        const OrtApi* ort_api,
        const PluginDmlExecutionProviderImpl* execution_provider,
        bool is_internal_operator = false);

    // IMLOperatorTensor methods - implemented using only C API
    STDMETHOD_(uint32_t, GetDimensionCount)() const noexcept override;
    STDMETHOD(GetShape)(uint32_t dimensionCount, uint32_t* dimensions) const noexcept override;
    STDMETHOD_(MLOperatorTensorDataType, GetTensorDataType)() const noexcept override;
    STDMETHOD_(bool, IsCpuData)() const noexcept override;
    STDMETHOD_(bool, IsDataInterface)() const noexcept override;
    STDMETHOD_(void*, GetData)() noexcept override;
    STDMETHOD_(void, GetDataInterface)(IUnknown** dataInterface) noexcept override;

private:
    const OrtValue* ort_value_;
    const OrtApi* ort_api_;
    const PluginDmlExecutionProviderImpl* execution_provider_;
    bool is_internal_operator_;
    mutable std::vector<int64_t> shape_cache_;  // Cache for shape
};

// ============================================================================
// ABI-Safe Kernel Context - implements IMLOperatorKernelContext using C API
// ============================================================================

class AbiSafeKernelContext : public Microsoft::WRL::RuntimeClass<
    Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
    IMLOperatorKernelContext,
    IMLOperatorKernelContextPrivate>
{
public:
    AbiSafeKernelContext(
        OrtKernelContext* kernel_context,
        const OrtApi* ort_api,
        const PluginDmlExecutionProviderImpl* execution_provider,
        bool is_internal_operator,
        const std::vector<std::vector<uint32_t>>* inferred_output_shapes = nullptr,
        IMLOperatorShapeInferrer* shape_inferrer = nullptr,
        const std::vector<uint32_t>* required_constant_cpu_inputs = nullptr,
        const Windows::AI::MachineLearning::Adapter::AttributeMap* default_attributes = nullptr,
        const OrtKernelInfo* kernel_info = nullptr);

    // IMLOperatorKernelContext methods - implemented using only C API
    STDMETHOD(GetInputTensor)(uint32_t inputIndex, IMLOperatorTensor** tensor) const noexcept override;
    STDMETHOD(GetOutputTensor)(uint32_t outputIndex, IMLOperatorTensor** tensor) noexcept override;
    STDMETHOD(GetOutputTensor)(uint32_t outputIndex, uint32_t dimensionCount, const uint32_t* dimensionSizes,
        IMLOperatorTensor** tensor) noexcept override;
    STDMETHOD(AllocateTemporaryData)(size_t size, IUnknown** data) const noexcept override;
    STDMETHOD_(void, GetExecutionInterface)(IUnknown** executionInterface) const noexcept override;

    // Resource state transitions — mirrors PluginOpKernelContextWrapper::TransitionResourcesForOperatorIfRequired.
    // Must be called before (isBeforeOp=true) and after (isBeforeOp=false) ml_operator_kernel->Compute().
    void TransitionResourcesForOperatorIfRequired(bool isBeforeOp);

    // IMLOperatorKernelContextPrivate methods
    STDMETHOD(GetSequenceInputTensor)(uint32_t inputIndex, uint32_t sequenceIndex, IMLOperatorTensor** tensor) const noexcept override;
    STDMETHOD(PrepareSequenceOutput)(uint32_t outputIndex, MLOperatorTensorDataType dataType) const noexcept override;
    STDMETHOD(GetSequenceOutputTensor)(uint32_t outputIndex, uint32_t sequenceIndex, MLOperatorTensorDataType dataType,
        uint32_t dimensions, const uint32_t* dimensionSizes, bool gpuOutput, IMLOperatorTensor** tensor) const noexcept override;
    STDMETHOD(GetSequenceInputInfo)(uint32_t inputIndex, uint32_t* inputCount, MLOperatorTensorDataType* dataType) const noexcept override;
    STDMETHOD_(bool, IsSequenceInputTensor)(uint32_t inputIndex) const noexcept override;

private:
    OrtKernelContext* kernel_context_;
    const OrtApi* ort_api_;
    const PluginDmlExecutionProviderImpl* execution_provider_;
    bool is_internal_operator_;  // For resource state transitions (MemcpyToHost/FromHost)
    const std::vector<std::vector<uint32_t>>* inferred_output_shapes_;
    mutable std::vector<Microsoft::WRL::ComPtr<AbiSafeTensor>> tensor_cache_;         // input tensors
    mutable std::vector<Microsoft::WRL::ComPtr<AbiSafeTensor>> output_tensor_cache_;  // output tensors (for post-op transitions)
    mutable Microsoft::WRL::ComPtr<IUnknown> abi_execution_object_;
    Microsoft::WRL::ComPtr<Windows::AI::MachineLearning::Adapter::IWinmlExecutionProvider> winml_provider_;

    // For runtime shape inference (e.g., Reshape with constant shape input)
    IMLOperatorShapeInferrer* shape_inferrer_;
    const std::vector<uint32_t>* required_constant_cpu_inputs_;
    const Windows::AI::MachineLearning::Adapter::AttributeMap* default_attributes_;
    const OrtKernelInfo* kernel_info_ = nullptr;  // For AbiSafeShapeInferenceContext attribute reads
};

// ============================================================================
// ABI-Safe Shape Inference Context - for runtime shape inference
// ============================================================================

class AbiSafeShapeInferenceContext : public Microsoft::WRL::RuntimeClass<
    Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
    Microsoft::WRL::ChainInterfaces<IMLOperatorShapeInferenceContextPrivate, IMLOperatorShapeInferenceContext>>
{
public:
    AbiSafeShapeInferenceContext(
        OrtKernelContext* kernel_context,
        const OrtApi* ort_api,
        const Windows::AI::MachineLearning::Adapter::AttributeMap* default_attributes,
        const PluginDmlExecutionProviderImpl* execution_provider,
        const OrtKernelInfo* kernel_info = nullptr,
        const std::vector<uint32_t>* required_constant_cpu_inputs = nullptr,
        const std::unordered_map<uint32_t, Microsoft::WRL::ComPtr<IMLOperatorTensor>>* prefetched_constant_tensors = nullptr);

    // IMLOperatorAttributes methods
    STDMETHOD(GetAttributeElementCount)(
        _In_z_ const char* name,
        MLOperatorAttributeType type,
        _Out_ uint32_t* elementCount) const noexcept override;

    STDMETHOD(GetAttribute)(
        _In_z_ const char* name,
        MLOperatorAttributeType type,
        uint32_t elementCount,
        size_t elementByteSize,
        _Out_writes_bytes_(elementCount * elementByteSize) void* value) const noexcept override;

    STDMETHOD(GetStringAttributeElementLength)(
        _In_z_ const char* name,
        uint32_t elementIndex,
        _Out_ uint32_t* attributeElementByteLength) const noexcept override;

    STDMETHOD(GetStringAttributeElement)(
        _In_z_ const char* name,
        uint32_t elementIndex,
        uint32_t attributeElementByteLength,
        _Out_writes_(attributeElementByteLength) char* attributeElement) const noexcept override;

    // IMLOperatorShapeInferenceContext methods
    STDMETHOD_(uint32_t, GetInputCount)() const noexcept override;
    STDMETHOD_(uint32_t, GetOutputCount)() const noexcept override;
    STDMETHOD_(bool, IsInputValid)(uint32_t inputIndex) const noexcept override;
    STDMETHOD_(bool, IsOutputValid)(uint32_t outputIndex) const noexcept override;

    STDMETHOD(GetInputEdgeDescription)(
        uint32_t inputIndex,
        _Out_ MLOperatorEdgeDescription* edgeDescription) const noexcept override;

    STDMETHOD(GetInputTensorDimensionCount)(
        uint32_t inputIndex,
        _Out_ uint32_t* dimensionCount) const noexcept override;

    STDMETHOD(GetInputTensorShape)(
        uint32_t inputIndex,
        uint32_t dimensionCount,
        _Out_writes_(dimensionCount) uint32_t* dimensions) const noexcept override;

    STDMETHOD(SetOutputTensorShape)(
        uint32_t outputIndex,
        uint32_t dimensionCount,
        const uint32_t* dimensions) noexcept override;

    // IMLOperatorShapeInferenceContextPrivate methods
    STDMETHOD(GetConstantInputTensor)(
        uint32_t inputIndex,
        _Outptr_ IMLOperatorTensor** tensor) const noexcept override;

    STDMETHOD(TryGetConstantInputTensor)(
        uint32_t inputIndex,
        _Outptr_ IMLOperatorTensor** tensor) const noexcept override;

    STDMETHOD(GetSequenceInputInfo)(
        uint32_t inputIndex,
        _Out_ uint32_t* inputCount,
        MLOperatorTensorDataType* dataType) const noexcept override;

    STDMETHOD(GetSequenceInputTensorDimensionCount)(
        uint32_t inputIndex,
        uint32_t sequenceIndex,
        _Out_ uint32_t* dimensionCount) const noexcept override;

    STDMETHOD(GetSequenceInputTensorShape)(
        uint32_t inputIndex,
        uint32_t sequenceIndex,
        uint32_t dimensionCount,
        _Out_writes_(dimensionCount) uint32_t* dimensions) const noexcept override;

    // Retrieve inferred output shapes after calling shape inferrer
    const std::vector<std::vector<uint32_t>>& GetInferredOutputShapes() const { return inferred_output_shapes_; }

private:
    OrtKernelContext* kernel_context_;
    const OrtApi* ort_api_;
    const Windows::AI::MachineLearning::Adapter::AttributeMap* default_attributes_;
    const PluginDmlExecutionProviderImpl* execution_provider_;
    const OrtKernelInfo* kernel_info_;  // For accessing actual node attributes
    const std::vector<uint32_t>* required_constant_cpu_inputs_;  // non-owning; mirrors unsafe path required list
    // Pre-fetched constant tensors from lazy-init — same tensors the unsafe path's constantInputGetter provides.
    // When set, GetConstantInputTensor serves from this map rather than re-fetching via the C API.
    const std::unordered_map<uint32_t, Microsoft::WRL::ComPtr<IMLOperatorTensor>>* prefetched_constant_tensors_;

    // Stores output shapes set by the shape inferrer
    std::vector<std::vector<uint32_t>> inferred_output_shapes_;

    // Cache for constant input tensors
    mutable std::unordered_map<uint32_t, Microsoft::WRL::ComPtr<AbiSafeTensor>> constant_tensor_cache_;
};

// ============================================================================
// ABI-Safe Kernel Creation Context - implements IMLOperatorKernelCreationContext using C API
// ============================================================================

// Pre-fetched tensor attribute data — plain bytes, no protobuf types.
// Populated before AbiSafeKernelCreationContext construction by TryFetchTensorAttribute().
struct PreFetchedTensorAttr {
    MLOperatorTensorDataType data_type = MLOperatorTensorDataType::Undefined;
    std::vector<uint32_t> shape;
    std::vector<std::byte> raw_bytes;
};

// ABI-safe IMLOperatorTensor wrapping pre-fetched tensor attribute bytes.
// Owns its data; no protobuf pointers. Used as the return value of GetTensorAttribute().
// Implementation is in dml_abi_kernel.cpp to avoid inline WRL template instantiation in the header,
// which would require IWinmlExecutionProvider (a heavy ORT-internal type) to be fully defined.
class PreFetchedTensorAttrWrapper : public Microsoft::WRL::RuntimeClass<
    Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
    IMLOperatorTensor>
{
public:
    PreFetchedTensorAttrWrapper(
        MLOperatorTensorDataType data_type,
        std::vector<uint32_t> shape,
        std::vector<std::byte> raw_bytes);

    STDMETHOD_(uint32_t, GetDimensionCount)() const noexcept override;
    STDMETHOD(GetShape)(uint32_t dim_count, uint32_t* dims) const noexcept override;
    STDMETHOD_(MLOperatorTensorDataType, GetTensorDataType)() const noexcept override;
    STDMETHOD_(bool, IsCpuData)() const noexcept override;
    STDMETHOD_(bool, IsDataInterface)() const noexcept override;
    STDMETHOD_(void*, GetData)() noexcept override;
    STDMETHOD_(void, GetDataInterface)(IUnknown** dataInterface) noexcept override;

private:
    MLOperatorTensorDataType data_type_;
    std::vector<uint32_t> shape_;
    std::vector<std::byte> raw_bytes_;
};

class AbiSafeKernelCreationContext : public Microsoft::WRL::RuntimeClass<
    Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
    Microsoft::WRL::ChainInterfaces<IMLOperatorKernelCreationContextNodeWrapperPrivate, IMLOperatorKernelCreationContextPrivate, IMLOperatorKernelCreationContext>,
    IMLOperatorTensorShapeDescription,
    IMLOperatorAttributes1>
{
public:
    AbiSafeKernelCreationContext(
        const OrtKernelInfo* kernel_info,
        const OrtApi* ort_api,
        const Windows::AI::MachineLearning::Adapter::AttributeMap* default_attributes,
        const std::vector<uint32_t>* required_constant_cpu_inputs,
        const PluginDmlExecutionProviderImpl* execution_provider,
        std::unordered_map<uint32_t, Microsoft::WRL::ComPtr<IMLOperatorTensor>>&& constant_tensors = {},
        OrtKernelContext* runtime_context = nullptr,  // For lazy init with runtime shapes
        const char* operator_name = nullptr,  // For operator-specific logic
        bool is_internal_operator = false,  // Controls GetExecutionInterface return value
        bool requires_input_shapes_at_creation = true,  // Mirrors old m_allowInputShapeQuery
        std::unordered_map<std::string, PreFetchedTensorAttr> tensor_attribute_cache = {},  // Pre-fetched tensor attrs
        const Windows::AI::MachineLearning::Adapter::EdgeShapes* input_shapes_override = nullptr);  // Runtime shapes (lazy-init)

    // IMLOperatorKernelCreationContext methods
    STDMETHOD_(uint32_t, GetInputCount)() const noexcept override;
    STDMETHOD_(uint32_t, GetOutputCount)() const noexcept override;
    STDMETHOD_(bool, IsInputValid)(uint32_t inputIndex) const noexcept override;
    STDMETHOD_(bool, IsOutputValid)(uint32_t outputIndex) const noexcept override;
    STDMETHOD(GetInputEdgeDescription)(uint32_t inputIndex, MLOperatorEdgeDescription* edgeDesc) const noexcept override;
    STDMETHOD(GetOutputEdgeDescription)(uint32_t outputIndex, MLOperatorEdgeDescription* edgeDesc) const noexcept override;
    STDMETHOD_(bool, HasTensorShapeDescription)() const noexcept override;
    STDMETHOD(GetTensorShapeDescription)(IMLOperatorTensorShapeDescription** shapeInfo) const noexcept override;
    STDMETHOD_(void, GetExecutionInterface)(IUnknown** executionInterface) const noexcept override;

    // IMLOperatorAttributes methods
    STDMETHOD(GetAttribute)(_In_z_ const char* name, MLOperatorAttributeType type,
        uint32_t elementCount, size_t elementByteSize, void* value) const noexcept override;
    STDMETHOD(GetAttributeElementCount)(_In_z_ const char* name, MLOperatorAttributeType type, uint32_t* elementCount) const noexcept override;
    STDMETHOD(GetStringAttributeElementLength)(_In_z_ const char* name, uint32_t elementIndex, uint32_t* attributeElementByteSize) const noexcept override;
    STDMETHOD(GetStringAttributeElement)(_In_z_ const char* name, uint32_t elementIndex, uint32_t attributeElementByteSize,
        char* attributeElement) const noexcept override;

    // IMLOperatorAttributes1 methods
    STDMETHOD(GetTensorAttribute)(_In_z_ const char* name, _COM_Outptr_ IMLOperatorTensor** tensor) const noexcept override;

    // IMLOperatorKernelCreationContextPrivate methods
    STDMETHOD(GetConstantInputTensor)(uint32_t inputIndex, IMLOperatorTensor** tensor) const noexcept override;
    STDMETHOD(TryGetConstantInputTensor)(uint32_t inputIndex, IMLOperatorTensor** tensor) const noexcept override;
    STDMETHOD_(bool, IsDmlGraphNode)() const noexcept override { return false; }
    STDMETHOD(SetDmlOperator)(_In_ const MLOperatorGraphDesc* operatorGraphDesc) const noexcept override { return E_NOTIMPL; }

    // IMLOperatorKernelCreationContextNodeWrapperPrivate methods - for node name access
    STDMETHOD_(uint32_t, GetUtf8NameBufferSizeInBytes)() const noexcept override;
    STDMETHOD(GetUtf8Name)(uint32_t bufferSizeInBytes, _Out_writes_bytes_(bufferSizeInBytes) char* name) const noexcept override;
    STDMETHOD_(uint32_t, GetWideNameBufferSizeInBytes)() const noexcept override;
    STDMETHOD(GetWideName)(uint32_t bufferSizeInBytes, _Out_writes_bytes_(bufferSizeInBytes) wchar_t* name) const noexcept override;
    STDMETHOD(GetExecutionProvider)(_Outptr_result_maybenull_ IUnknown** executionProvider) const noexcept override;

    // IMLOperatorTensorShapeDescription methods - implemented using C API
    STDMETHOD(GetInputTensorDimensionCount)(uint32_t inputIndex, _Out_ uint32_t* dimensionCount) const noexcept override;
    STDMETHOD(GetInputTensorShape)(uint32_t inputIndex, uint32_t dimensionCount, _Out_writes_(dimensionCount) uint32_t* dimensions) const noexcept override;
    STDMETHOD_(bool, HasOutputShapeDescription)() const noexcept override;
    STDMETHOD(GetOutputTensorDimensionCount)(uint32_t outputIndex, _Out_ uint32_t* dimensionCount) const noexcept override;
    STDMETHOD(GetOutputTensorShape)(uint32_t outputIndex, uint32_t dimensionCount, _Out_writes_(dimensionCount) uint32_t* dimensions) const noexcept override;

    // Set pre-computed output shapes (e.g., from shape inferrer)
    void SetPrecomputedOutputShapes(const std::vector<std::vector<uint32_t>>& shapes) {
        precomputed_output_shapes_ = shapes;
    }

private:
    const OrtKernelInfo* kernel_info_;
    const OrtApi* ort_api_;
    const Windows::AI::MachineLearning::Adapter::AttributeMap* default_attributes_;
    const std::vector<uint32_t>* required_constant_cpu_inputs_;
    const PluginDmlExecutionProviderImpl* execution_provider_;

    // Pre-fetched constant tensors (ABI-safe, no unsafe casts)
    std::unordered_map<uint32_t, Microsoft::WRL::ComPtr<IMLOperatorTensor>> constant_tensors_;

    // For lazy initialization - get shapes from actual runtime tensors
    OrtKernelContext* runtime_context_ = nullptr;
    const char* operator_name_ = nullptr;
    bool is_internal_operator_ = false;  // Controls GetExecutionInterface (ID3D12GraphicsCommandList* vs provider)
    // Captured runtime input shapes (mirrors old plugin's m_inputShapesOverride passed to PluginOpKernelInfoWrapper)
    const Windows::AI::MachineLearning::Adapter::EdgeShapes* input_shapes_override_ = nullptr;
    bool requires_input_shapes_at_creation_ = true;  // Mirrors old m_allowInputShapeQuery static flag

    // Pre-computed output shapes from shape inferrer (takes precedence over fallback logic)
    std::vector<std::vector<uint32_t>> precomputed_output_shapes_;

    // Pre-fetched tensor attribute cache (ABI-safe, no protobuf). Populated at construction
    // by TryFetchTensorAttribute() called at each AbiSafeKernelCreationContext creation site.
    std::unordered_map<std::string, PreFetchedTensorAttr> tensor_attribute_cache_;

    // Cached ABI execution object. GetABIExecutionInterfaceAndInvalidateState is called exactly
    // once in the constructor (matching PluginOpKernelInfoWrapper behavior). GetExecutionInterface
    // returns this cached value rather than calling the invalidating function on every access.
    Microsoft::WRL::ComPtr<IUnknown> abi_execution_object_;
};

// ============================================================================
// Kernel Creation and Storage
// ============================================================================

// State for ABI-safe kernel creation
struct DmlKernelCreationState {
    IMLOperatorKernelFactory* kernel_factory = nullptr;
    IMLOperatorShapeInferrer* shape_inferrer = nullptr;
    const Windows::AI::MachineLearning::Adapter::AttributeMap* default_attributes = nullptr;

    std::vector<uint32_t> required_constant_cpu_inputs;
    bool requires_input_shapes_at_creation = false;
    bool requires_output_shapes_at_creation = false;
    bool is_internal_operator = false;  // For resource state transitions (MemcpyToHost/FromHost)
    std::vector<std::string> tensor_attribute_names;  // Tensor-typed ONNX attribute names (e.g., ConstantOfShape's "value")

    const PluginDmlExecutionProviderImpl* dml_execution_provider = nullptr;
    const OrtApi* ort_api = nullptr;
    const char* operator_name = nullptr;  // For debugging
};

// Snapshot of a constant CPU input tensor's content, used to detect value changes between Compute
// calls. Mirrors the TensorContent struct in PluginDmlAbiOpKernel (MLOperatorAuthorImpl.h:598).
struct TensorContent {
    bool isValid = false;
    std::vector<uint32_t> shape;
    MLOperatorTensorDataType type = MLOperatorTensorDataType::Undefined;
    std::vector<std::byte> data;
};

// ABI-safe kernel - stores DML operator directly
struct DmlAbiKernel {
    Microsoft::WRL::ComPtr<IMLOperatorKernel> ml_operator_kernel;
    const OrtApi* ort_api = nullptr;
    const PluginDmlExecutionProviderImpl* dml_execution_provider = nullptr;
    bool is_internal_operator = false;  // For resource state transitions (MemcpyToHost/FromHost)
    std::vector<std::vector<uint32_t>> inferred_output_shapes;  // Shapes from graph inference
    std::string operator_name;  // For debugging

    // For runtime shape inference (e.g., Reshape with constant shape input)
    Microsoft::WRL::ComPtr<IMLOperatorShapeInferrer> shape_inferrer;
    std::vector<uint32_t> required_constant_cpu_inputs;
    const Windows::AI::MachineLearning::Adapter::AttributeMap* default_attributes = nullptr;

    // For lazy kernel creation when shapes are dynamic
    bool needs_lazy_init = false;  // True if kernel wasn't created yet due to dynamic shapes
    bool requires_input_shapes_at_creation = false;  // Static flag for HasTensorShapeDescription
    Microsoft::WRL::ComPtr<IMLOperatorKernelFactory> kernel_factory;  // Factory to create kernel lazily
    const OrtKernelInfo* kernel_info = nullptr;  // Stored for lazy initialization (lifetime managed by ORT)

    // For resource state transitions and QueueReference after Compute
    Microsoft::WRL::ComPtr<Windows::AI::MachineLearning::Adapter::IWinmlExecutionProvider> winml_provider;

    // Names of tensor-typed ONNX attributes (e.g., ConstantOfShape's "value") — for ABI-safe attribute fetch
    std::vector<std::string> tensor_attribute_names;

    // For shape-change detection between Compute calls (mirrors m_inputShapesOfKernelInference in unsafe path)
    Windows::AI::MachineLearning::Adapter::EdgeShapes input_shapes_of_kernel_inference;

    // Snapshots of required constant CPU input values from the last kernel creation. Indexed by input
    // index (sparse — only indices in required_constant_cpu_inputs are populated). Used to detect value
    // changes between calls when shapes are unchanged. Mirrors m_constantInputTensorContentsOfKernel
    // in PluginDmlAbiOpKernel. Only the tensor (non-sequence) case is needed here since sequence-typed
    // constant CPU inputs do not occur for DML operators.
    std::vector<TensorContent> constant_input_tensor_contents;
};

// Pre-fetches tensor-typed attributes from kernel_info into an ABI-safe cache map.
// Enumerates only the names listed in tensor_attribute_names — fully ABI-safe (ORT C API only).
std::unordered_map<std::string, PreFetchedTensorAttr>
FetchAllTensorAttributes(
    const OrtKernelInfo* kernel_info,
    const OrtApi* ort_api,
    const std::vector<std::string>& tensor_attribute_names);

// C API kernel creation function
OrtStatus* ORT_API_CALL DmlAbiKernel_Create(
    void* kernel_create_state,
    const OrtKernelInfo* kernel_info,
    OrtKernelImpl** kernel_out,
    std::unordered_map<uint32_t, Microsoft::WRL::ComPtr<IMLOperatorTensor>>&& constant_tensors = {}
) noexcept;

// C API kernel compute function
OrtStatus* ORT_API_CALL DmlAbiKernel_Compute(
    OrtKernelImpl* this_ptr,
    OrtKernelContext* context
) noexcept;

// C API kernel release function
void ORT_API_CALL DmlAbiKernel_Release(OrtKernelImpl* this_ptr) noexcept;

// Helper to get DmlAbiKernel from OrtKernelImpl
inline DmlAbiKernel* GetDmlAbiKernelFromImpl(OrtKernelImpl* impl) {
    return reinterpret_cast<DmlAbiKernel*>(
        reinterpret_cast<char*>(impl) + sizeof(OrtKernelImpl)
    );
}

} // namespace Dml
