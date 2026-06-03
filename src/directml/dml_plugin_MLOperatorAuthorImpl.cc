// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "DmlExecutionProvider/precomp.h"

#include "core/framework/execution_frame.h"
#include "core/framework/TensorSeq.h"

#include <onnxruntime_c_api.h>
#include "DmlExecutionProvider/inc/MLOperatorAuthor.h"

#include "dml_plugin_MLOperatorAuthorImpl.h"
#include "dml_abi_kernel.h"
#include "dml_ep.h"

namespace dml_ep {

    #define ML_TENSOR_TYPE_CASE(x)                                                                                         \
    if (onnxruntime::utils::IsPrimitiveDataType<x>(type)) {                                                            \
        return MLTypeTraits<x>::TensorType;                                                                            \
    }

#pragma warning(push)
#pragma warning(disable : 4702)
::MLOperatorTensorDataType ToMLTensorDataTypePlugin(onnxruntime::MLDataType type) {
    if (onnxruntime::utils::IsDataTypeString(type)) {
        return MLOperatorTensorDataType::String;
    }

    ML_TENSOR_TYPE_CASE(float);
    ML_TENSOR_TYPE_CASE(onnxruntime::Int4x2Base<false>);
    ML_TENSOR_TYPE_CASE(onnxruntime::Int4x2Base<true>);
    ML_TENSOR_TYPE_CASE(uint8_t);
    ML_TENSOR_TYPE_CASE(int8_t);
    ML_TENSOR_TYPE_CASE(uint16_t);
    ML_TENSOR_TYPE_CASE(int16_t);
    ML_TENSOR_TYPE_CASE(int32_t);
    ML_TENSOR_TYPE_CASE(int64_t);
    ML_TENSOR_TYPE_CASE(bool);
    ML_TENSOR_TYPE_CASE(double);
    ML_TENSOR_TYPE_CASE(uint32_t);
    ML_TENSOR_TYPE_CASE(uint64_t);
    ML_TENSOR_TYPE_CASE(onnxruntime::MLFloat16);

    ORT_THROW_HR(E_NOTIMPL);
    return MLOperatorTensorDataType::Undefined;
#pragma warning(pop)
}

#undef ML_TENSOR_TYPE_CASE
#define ML_TENSOR_TYPE_CASE(x)                                                                                         \
    if (type == MLTypeTraits<x>::TensorType) {                                                                         \
        return onnxruntime::DataTypeImpl::GetTensorType<x>();                                                          \
    }

#define ML_SEQUENCE_TENSOR_TYPE_CASE(x)                                                                                \
    if (type == MLTypeTraits<x>::TensorType) {                                                                         \
        return onnxruntime::DataTypeImpl::GetSequenceTensorType<x>();                                                  \
    }

#define ML_PRIMITIVE_TYPE_CASE(x)                                                                                      \
    if (type == MLTypeTraits<x>::TensorType) {                                                                         \
        return onnxruntime::DataTypeImpl::GetType<x>();                                                                \
    }

#pragma warning(push)
#pragma warning(disable : 4702)
onnxruntime::MLDataType ToMLDataTypePlugin(::MLOperatorEdgeType edgeType, ::MLOperatorTensorDataType type) {
    if (edgeType == ::MLOperatorEdgeType::Tensor) {
        if (type == MLOperatorTensorDataType::String)
            return onnxruntime::DataTypeImpl::GetTensorType<std::string>();

        ML_TENSOR_TYPE_CASE(float);
        ML_TENSOR_TYPE_CASE(onnxruntime::Int4x2Base<false>);
        ML_TENSOR_TYPE_CASE(onnxruntime::Int4x2Base<true>);
        ML_TENSOR_TYPE_CASE(uint8_t);
        ML_TENSOR_TYPE_CASE(int8_t);
        ML_TENSOR_TYPE_CASE(uint16_t);
        ML_TENSOR_TYPE_CASE(int16_t);
        ML_TENSOR_TYPE_CASE(int32_t);
        ML_TENSOR_TYPE_CASE(int64_t);
        ML_TENSOR_TYPE_CASE(bool);
        ML_TENSOR_TYPE_CASE(double);
        ML_TENSOR_TYPE_CASE(uint32_t);
        ML_TENSOR_TYPE_CASE(uint64_t);
        ML_TENSOR_TYPE_CASE(onnxruntime::MLFloat16);

        ORT_THROW_HR(E_NOTIMPL);
        return onnxruntime::DataTypeImpl::GetTensorType<float>();
    } else if (edgeType == ::MLOperatorEdgeType::SequenceTensor) {
        if (type == MLOperatorTensorDataType::String)
            return onnxruntime::DataTypeImpl::GetSequenceTensorType<std::string>();

        ML_SEQUENCE_TENSOR_TYPE_CASE(float);
        ML_SEQUENCE_TENSOR_TYPE_CASE(onnxruntime::Int4x2Base<false>);
        ML_SEQUENCE_TENSOR_TYPE_CASE(onnxruntime::Int4x2Base<true>);
        ML_SEQUENCE_TENSOR_TYPE_CASE(uint8_t);
        ML_SEQUENCE_TENSOR_TYPE_CASE(int8_t);
        ML_SEQUENCE_TENSOR_TYPE_CASE(uint16_t);
        ML_SEQUENCE_TENSOR_TYPE_CASE(int16_t);
        ML_SEQUENCE_TENSOR_TYPE_CASE(int32_t);
        ML_SEQUENCE_TENSOR_TYPE_CASE(int64_t);
        ML_SEQUENCE_TENSOR_TYPE_CASE(bool);
        ML_SEQUENCE_TENSOR_TYPE_CASE(double);
        ML_SEQUENCE_TENSOR_TYPE_CASE(uint32_t);
        ML_SEQUENCE_TENSOR_TYPE_CASE(uint64_t);
        ML_SEQUENCE_TENSOR_TYPE_CASE(onnxruntime::MLFloat16);

        ORT_THROW_HR(E_NOTIMPL);
        return onnxruntime::DataTypeImpl::GetSequenceTensorType<float>();
    } else if (edgeType == ::MLOperatorEdgeType::Primitive) {
        if (type == MLOperatorTensorDataType::String)
            return onnxruntime::DataTypeImpl::GetType<std::string>();

        ML_PRIMITIVE_TYPE_CASE(float);
        ML_PRIMITIVE_TYPE_CASE(onnxruntime::Int4x2Base<false>);
        ML_PRIMITIVE_TYPE_CASE(onnxruntime::Int4x2Base<true>);
        ML_PRIMITIVE_TYPE_CASE(uint8_t);
        ML_PRIMITIVE_TYPE_CASE(int8_t);
        ML_PRIMITIVE_TYPE_CASE(uint16_t);
        ML_PRIMITIVE_TYPE_CASE(int16_t);
        ML_PRIMITIVE_TYPE_CASE(int32_t);
        ML_PRIMITIVE_TYPE_CASE(int64_t);
        ML_PRIMITIVE_TYPE_CASE(bool);
        ML_PRIMITIVE_TYPE_CASE(double);
        ML_PRIMITIVE_TYPE_CASE(uint32_t);
        ML_PRIMITIVE_TYPE_CASE(uint64_t);
        ML_PRIMITIVE_TYPE_CASE(onnxruntime::MLFloat16);

        ORT_THROW_HR(E_NOTIMPL);
        return onnxruntime::DataTypeImpl::GetType<float>();
    }
#pragma warning(pop)
    ORT_THROW_HR(E_NOTIMPL);
}

PluginOpKernelInfoWrapper::PluginOpKernelInfoWrapper(
    const onnxruntime::OpKernelInfo* kerneInfo,
    IUnknown* abiExecutionObject,
    const EdgeShapes* inputShapeOverrides,
    const EdgeShapes* inferredOutputShapes,
    bool allowInputShapeQuery,
    bool allowOutputShapeQuery,
    bool isInternalOperator,
    const AttributeMap* defaultAttributes,
    gsl::span<const uint32_t> requiredConstantCpuInputs,
    MLOperatorTensorGetter& constantInputGetter,
    const dml_ep::PluginDmlExecutionProviderImpl* pluginDmlEp,
    const onnxruntime::OpKernelContext* kernelContext) 
    : OpNodeInfoWrapper(kerneInfo, inputShapeOverrides, defaultAttributes, requiredConstantCpuInputs, constantInputGetter,
                      kernelContext)
    , m_inferredOutputShapes(inferredOutputShapes)
    , m_allowInputShapeQuery(allowInputShapeQuery)
    , m_allowOutputShapeQuery(allowOutputShapeQuery)
    , m_internalOperator(isInternalOperator)
    , m_impl(kerneInfo)
    , m_abiExecutionObject(abiExecutionObject)
{
    const void* executionHandle = pluginDmlEp; // kerneInfo->GetExecutionProvider()->GetExecutionHandle();
    if (executionHandle) {
        // We assume the execution object inherits IUnknown as its first base
        Microsoft::WRL::ComPtr<IUnknown> providerExecutionObject = const_cast<IUnknown*>(static_cast<const IUnknown*>(executionHandle));
        providerExecutionObject.As(&m_winmlProvider);
    }

    assert(allowInputShapeQuery || !allowOutputShapeQuery);

    // The input may be exposed using non-overridden sizes.    Exposing output shapes requires
    // those shapes be provided here.
    assert(!allowOutputShapeQuery || (inferredOutputShapes != nullptr));
}

HRESULT STDMETHODCALLTYPE PluginOpKernelInfoWrapper::GetOutputTensorShape(uint32_t outputIndex, uint32_t dimensionCount,
                                                                    uint32_t* dimensions) const noexcept {
    ORT_TRY{VerifyNotClosed();

memset(dimensions, 0, dimensionCount * sizeof(dimensions[0]));

if (!HasOutputShapeDescription()) {
    return E_FAIL;
}

if (outputIndex >= GetOutputCount()) {
    return E_INVALIDARG;
}

if (m_inferredOutputShapes->GetShape(outputIndex).size() != dimensionCount) {
    return E_INVALIDARG;
}

for (uint32_t i = 0; i < dimensionCount; ++i) {
    dimensions[i] = m_inferredOutputShapes->GetShape(outputIndex)[i];
}

return S_OK;
}  // namespace dml_ep
ORT_CATCH_RETURN
}

HRESULT STDMETHODCALLTYPE PluginOpKernelInfoWrapper::GetOutputTensorDimensionCount(uint32_t outputIndex,
                                                                             uint32_t* dimensionCount) const noexcept {
    ORT_TRY {
        VerifyNotClosed();

        *dimensionCount = 0;

        if (!HasOutputShapeDescription()) {
            return E_FAIL;
        }

        if (outputIndex >= GetOutputCount()) {
            return E_INVALIDARG;
        }

        *dimensionCount = gsl::narrow_cast<uint32_t>(m_inferredOutputShapes->GetShape(outputIndex).size());

        return S_OK;
    }
    ORT_CATCH_RETURN
}

bool STDMETHODCALLTYPE PluginOpKernelInfoWrapper::HasTensorShapeDescription() const noexcept {
    return m_allowInputShapeQuery;
}

HRESULT STDMETHODCALLTYPE
PluginOpKernelInfoWrapper::GetTensorShapeDescription(IMLOperatorTensorShapeDescription** shapeInfo) const noexcept {
    ORT_TRY {
        VerifyNotClosed();

        *shapeInfo = nullptr;

        if (!HasTensorShapeDescription()) {
            *shapeInfo = nullptr;
            return E_FAIL;
            // return MLStatus::REQUIREMENT_NOT_REGISTERED;
        }

        Microsoft::WRL::ComPtr<IMLOperatorTensorShapeDescription> ret = const_cast<PluginOpKernelInfoWrapper*>(this);
        *shapeInfo = ret.Detach();
        return S_OK;
    }
    ORT_CATCH_RETURN
}

void STDMETHODCALLTYPE PluginOpKernelInfoWrapper::GetExecutionInterface(IUnknown** executionInterface) const noexcept {
    m_abiExecutionObject.CopyTo(executionInterface);
}

uint32_t STDMETHODCALLTYPE PluginOpKernelInfoWrapper::GetUtf8NameBufferSizeInBytes() const noexcept {
    // Include null terminator.
    return static_cast<uint32_t>(m_impl->node().Name().size() + 1);
}

HRESULT STDMETHODCALLTYPE PluginOpKernelInfoWrapper::GetUtf8Name(uint32_t bufferSizeInBytes,
                                                           char* outputName) const noexcept {
    if (bufferSizeInBytes == 0) {
        return E_INVALIDARG;
    }

    // Copy as many characters as possible, leaving room for the null terminator.
    const auto& nodeName = m_impl->node().Name();
    size_t charsCopied = nodeName.copy(outputName, bufferSizeInBytes - 1);

    // Write the null terminator.
    assert(charsCopied >= 0 && charsCopied < bufferSizeInBytes);
    outputName[charsCopied] = '\0';

    return S_OK;
}

uint32_t STDMETHODCALLTYPE PluginOpKernelInfoWrapper::GetWideNameBufferSizeInBytes() const noexcept {
    const auto& name = m_impl->node().Name();
    if (name.empty()) {
        // Include null terminator.
        return sizeof(wchar_t);
    }

    int requiredSizeInChars = MultiByteToWideChar(CP_UTF8, 0, name.data(), static_cast<int>(name.size()), nullptr, 0);
    assert(requiredSizeInChars > 0);

    // Include null terminator.
    return static_cast<uint32_t>((requiredSizeInChars + 1) * sizeof(wchar_t));
}

HRESULT STDMETHODCALLTYPE PluginOpKernelInfoWrapper::GetWideName(uint32_t bufferSizeInBytes,
                                                           wchar_t* outputName) const noexcept {
    // Buffer needs to be large enough to at least hold a null terminator.
    if (bufferSizeInBytes < sizeof(wchar_t)) {
        return E_INVALIDARG;
    }

    const auto& nodeName = m_impl->node().Name();
    if (nodeName.empty()) {
        outputName[0] = L'\0';
        return S_OK;
    }

    uint32_t bufferSizeInChars = bufferSizeInBytes / sizeof(wchar_t);
    int charsCopiedIfSucceeded = MultiByteToWideChar(CP_UTF8, 0, nodeName.data(), static_cast<int>(nodeName.size()),
                                                     outputName, bufferSizeInChars);

    if (charsCopiedIfSucceeded > 0) {
        // The return value is only > 0 if ALL characters copied successfully.
        // Write null terminator at the end of copied chars, which may not be at the end of the buffer.
        outputName[charsCopiedIfSucceeded] = L'\0';
        return S_OK;
    }

    // An error must have occurred in MultiByteToWideChar.
    assert(charsCopiedIfSucceeded <= 0);
    auto lastError = GetLastError();

    if (lastError == ERROR_INSUFFICIENT_BUFFER) {
        // The buffer was too small, but MultiByteToWideChar will have copied as many chars as possible.
        // Truncate and overwrite last char with null terminator. Don't treat this as an error.
        outputName[bufferSizeInChars - 1] = L'\0';
        return S_OK;
    }

    assert(lastError == ERROR_INVALID_PARAMETER || lastError == ERROR_NO_UNICODE_TRANSLATION);
    return E_INVALIDARG;
}

bool STDMETHODCALLTYPE PluginOpKernelInfoWrapper::HasOutputShapeDescription() const noexcept
{
    return m_allowOutputShapeQuery;
}



void PluginOpKernelContextWrapper::TransitionResourcesForOperatorIfRequired(bool isBeforeOp) {
    if (m_winmlProvider->TransitionsRequiredForOperator(m_internalOperator)) {
        uint32_t totalInputTensorCount = 0;
        for (auto inputTensor : m_inputTensors) {
            totalInputTensorCount += static_cast<uint32_t>(inputTensor.size());
        }
        std::vector<IUnknown*> resourcesToTransition;
        resourcesToTransition.reserve(totalInputTensorCount + m_outputTensors.size() + m_temporaryAllocations.size());

        for (uint32_t i = 0; i < m_inputTensors.size(); ++i) {
            for (uint32_t j = 0; j < m_inputTensors[i].size(); ++j) {
                Microsoft::WRL::ComPtr<IMLOperatorTensor> tensor;
                if (m_inputTensors[i].size() == 1) {
                    ORT_THROW_IF_FAILED(GetInputTensor(i, tensor.GetAddressOf()));
                } else {
                    ORT_THROW_IF_FAILED(GetSequenceInputTensor(i, j, tensor.GetAddressOf()));
                }

                if (tensor) {
                    Microsoft::WRL::ComPtr<IUnknown> resource;
                    tensor->GetDataInterface(resource.GetAddressOf());
                    if (resource) {
                        resourcesToTransition.push_back(resource.Get());
                    }
                }
            }
        }

        for (uint32_t i = 0; i < m_outputTensors.size(); ++i) {
            Microsoft::WRL::ComPtr<IMLOperatorTensor> tensor;
            ORT_THROW_IF_FAILED(GetOutputTensor(i, tensor.GetAddressOf()));

            Microsoft::WRL::ComPtr<IUnknown> resource;
            tensor->GetDataInterface(resource.GetAddressOf());
            if (resource) {
                resourcesToTransition.push_back(resource.Get());
            }
        }

        for (auto& tempAlloc : m_temporaryAbiAllocations) {
            resourcesToTransition.push_back(tempAlloc.Get());
        }

        m_winmlProvider->TransitionResourcesForOperator(
            isBeforeOp, gsl::narrow_cast<uint32_t>(resourcesToTransition.size()), resourcesToTransition.data());
    }
}

PluginOpKernelContextWrapper::PluginOpKernelContextWrapper(
    onnxruntime::OpKernelContext* context,
    const dml_ep::PluginDmlExecutionProviderImpl* provider,
    bool isInternalOperator,
    const EdgeShapes* outputShapes) :
    m_impl(context), m_outputShapes(outputShapes), m_provider(provider), m_internalOperator(isInternalOperator) {
    // Pre-size tensor arrays.    Member methods return pointers to these which
    // are stored in these arrays, which would become stale if the vectors reallocate
    // their internal storage.
    m_inputTensors.resize(context->InputCount(), std::vector<Microsoft::WRL::ComPtr<TensorWrapper>>(1));
    m_outputTensors.resize(context->OutputCount(), std::vector<Microsoft::WRL::ComPtr<TensorWrapper>>(1));

    const void* executionHandle = m_provider;
    if (executionHandle) {
        // We assume the execution object inherits IUnknown as its first base
        m_providerExecutionObject = const_cast<IUnknown*>(static_cast<const IUnknown*>(executionHandle));
        m_providerExecutionObject.As(&m_winmlProvider);

        // Query the actual object to return through the ABI, based on options registered
        // with the kernel
        m_abiExecutionObject = m_providerExecutionObject;
        if (m_winmlProvider) {
            m_winmlProvider->GetABIExecutionInterfaceAndInvalidateState(isInternalOperator,
                                                                        m_abiExecutionObject.ReleaseAndGetAddressOf());
        }

        TransitionResourcesForOperatorIfRequired(true);
    }
}

PluginOpKernelContextWrapper::~PluginOpKernelContextWrapper() { ClearTempAllocations(); }

void PluginOpKernelContextWrapper::ClearTempAllocations() {
    if (m_winmlProvider) {
        m_temporaryAllocations.clear();
        m_temporaryAbiAllocations.clear();
    }
}

void PluginOpKernelContextWrapper::Close() {
    if (m_winmlProvider && m_winmlProvider->TransitionsRequiredForOperator(m_internalOperator)) {
        TransitionResourcesForOperatorIfRequired(false);
    }

    for (auto& tensors : m_inputTensors) {
        for (auto& tensor : tensors) {
            if (tensor) {
                tensor->Close();
            }
        }
    }

    for (auto& tensors : m_outputTensors) {
        for (auto& tensor : tensors) {
            if (tensor) {
                tensor->Close();
            }
        }
    }

    ClearTempAllocations();

    Closable::Close();
}

bool STDMETHODCALLTYPE PluginOpKernelContextWrapper::IsSequenceInputTensor(uint32_t inputIndex) const noexcept {
    assert(inputIndex < gsl::narrow_cast<uint32_t>(m_impl->InputCount()));
    return m_impl->InputType(inputIndex)->IsTensorSequenceType();
}

HRESULT STDMETHODCALLTYPE PluginOpKernelContextWrapper::GetInputTensor(uint32_t inputIndex,
                                                                 IMLOperatorTensor** tensor) const noexcept {
    ORT_TRY{VerifyNotClosed();
*tensor = nullptr;

ML_CHECK_BOOL(inputIndex < m_inputTensors.size());

auto opKernelContextWrapper = const_cast<PluginOpKernelContextWrapper*>(this);
if (m_inputTensors[inputIndex][0] == nullptr) {
    auto inputTensor = m_impl->Input<onnxruntime::Tensor>(gsl::narrow_cast<int>(inputIndex));
    if (inputTensor != nullptr) {
        Microsoft::WRL::ComPtr<TensorWrapper> tensorWrapper = wil::MakeOrThrow<TensorWrapper>(
            const_cast<onnxruntime::Tensor*>(inputTensor), IsAllocationInterface(inputTensor->Location()),
            m_winmlProvider.Get(), m_internalOperator);

        opKernelContextWrapper->m_inputTensors[inputIndex][0] = tensorWrapper;
    }
}

if (opKernelContextWrapper->m_inputTensors[inputIndex][0] != nullptr) {
    opKernelContextWrapper->m_inputTensors[inputIndex][0].CopyTo(tensor);
}
return S_OK;
}  // namespace dml_ep
ORT_CATCH_RETURN
}

HRESULT STDMETHODCALLTYPE PluginOpKernelContextWrapper::GetSequenceInputTensor(uint32_t inputIndex, uint32_t sequenceIndex,
                                                                         IMLOperatorTensor** tensor) const noexcept {
    ORT_TRY{VerifyNotClosed();
*tensor = nullptr;

auto opKernelContextWrapper = const_cast<PluginOpKernelContextWrapper*>(this);

ML_CHECK_BOOL(inputIndex < m_inputTensors.size());
if (sequenceIndex >= m_inputTensors[inputIndex].size()) {
    opKernelContextWrapper->m_inputTensors[inputIndex].resize(static_cast<size_t>(sequenceIndex) + 1);
}

if (m_inputTensors[inputIndex][sequenceIndex] == nullptr) {
    auto inputTensorSeq = m_impl->Input<onnxruntime::TensorSeq>(gsl::narrow_cast<int>(inputIndex));
    ML_CHECK_BOOL(inputTensorSeq != nullptr);

    auto elemTensor = const_cast<onnxruntime::Tensor*>(&inputTensorSeq->Get(sequenceIndex));
    if (elemTensor != nullptr) {
        Microsoft::WRL::ComPtr<TensorWrapper> tensorWrapper = wil::MakeOrThrow<TensorWrapper>(
            elemTensor, IsAllocationInterface(elemTensor->Location()), m_winmlProvider.Get(), m_internalOperator);

        opKernelContextWrapper->m_inputTensors[inputIndex][sequenceIndex] = tensorWrapper;
    }
}

if (opKernelContextWrapper->m_inputTensors[inputIndex][sequenceIndex] != nullptr) {
    opKernelContextWrapper->m_inputTensors[inputIndex][sequenceIndex].CopyTo(tensor);
}
return S_OK;
}
ORT_CATCH_RETURN
}

HRESULT STDMETHODCALLTYPE PluginOpKernelContextWrapper::PrepareSequenceOutput(
    uint32_t outputIndex, MLOperatorTensorDataType dataType) const noexcept {ORT_TRY{VerifyNotClosed();

auto opKernelContextWrapper = const_cast<PluginOpKernelContextWrapper*>(this);

ML_CHECK_BOOL(outputIndex < m_outputTensors.size());
auto outputTensorSeq = m_impl->Output<onnxruntime::TensorSeq>(gsl::narrow_cast<int>(outputIndex));
ML_CHECK_BOOL(outputTensorSeq != nullptr);

auto mlDataType = ToMLDataType(MLOperatorEdgeType::Primitive, dataType);
outputTensorSeq->SetType(mlDataType);

return S_OK;
}
ORT_CATCH_RETURN
}

HRESULT STDMETHODCALLTYPE PluginOpKernelContextWrapper::GetSequenceOutputTensor(
    uint32_t outputIndex, uint32_t sequenceIndex, MLOperatorTensorDataType dataType, uint32_t dimensions,
    const uint32_t* dimensionSizes, bool gpuOutput,
    IMLOperatorTensor** tensor) const noexcept
{
    ORT_TRY
    {
        VerifyNotClosed();
        *tensor = nullptr;

        auto opKernelContextWrapper = const_cast<PluginOpKernelContextWrapper*>(this);

        ML_CHECK_BOOL(outputIndex < m_outputTensors.size());
        if (sequenceIndex >= m_outputTensors[outputIndex].size()) {
            opKernelContextWrapper->m_outputTensors[outputIndex].resize(sequenceIndex + 1);
        }

        // Verify that the provided shape matches the shape determined using the kernel's shape inference function.
        if (m_outputTensors[outputIndex][sequenceIndex] == nullptr) {
            auto outputTensorSeq = m_impl->Output<onnxruntime::TensorSeq>(gsl::narrow_cast<int>(outputIndex));
            ML_CHECK_BOOL(outputTensorSeq != nullptr);

            auto mlDataType = ToMLDataType(MLOperatorEdgeType::Primitive, dataType);

            if (outputTensorSeq->Size() == 0) {
                outputTensorSeq->SetType(mlDataType);
            }

            onnxruntime::AllocatorPtr alloc;
            if (gpuOutput) {
                auto status = m_impl->GetTempSpaceAllocator(&alloc);
                ORT_THROW_HR_IF(E_INVALIDARG, !status.IsOK());
            } else {
                auto status = m_impl->GetTempSpaceCPUAllocator(&alloc);
                ORT_THROW_HR_IF(E_INVALIDARG, !status.IsOK());
            }

            std::vector<int64_t> shapeDims(dimensions);
            for (uint32_t i = 0; i < dimensions; ++i) {
                shapeDims[i] = dimensionSizes[i];
            }

            auto target_tensor = onnxruntime::Tensor(mlDataType, onnxruntime::TensorShape(shapeDims), alloc);
            outputTensorSeq->Add(std::move(target_tensor));

            auto elemTensor = const_cast<onnxruntime::Tensor*>(&outputTensorSeq->Get(sequenceIndex));
            if (elemTensor != nullptr) {
                Microsoft::WRL::ComPtr<TensorWrapper> tensorWrapper = wil::MakeOrThrow<TensorWrapper>(
                    elemTensor, IsAllocationInterface(elemTensor->Location()), m_winmlProvider.Get(), m_internalOperator);

                opKernelContextWrapper->m_outputTensors[outputIndex][sequenceIndex] = tensorWrapper;
            }
        }

        if (opKernelContextWrapper->m_outputTensors[outputIndex][sequenceIndex] != nullptr) {
            opKernelContextWrapper->m_outputTensors[outputIndex][sequenceIndex].CopyTo(tensor);
        }

        return S_OK;
    }
    ORT_CATCH_RETURN
}

    HRESULT STDMETHODCALLTYPE PluginOpKernelContextWrapper::GetSequenceInputInfo(
        uint32_t inputIndex, uint32_t* inputCount,
        MLOperatorTensorDataType* dataType) const noexcept {ORT_TRY{VerifyNotClosed();

    ML_CHECK_BOOL(inputIndex < m_inputTensors.size());

    assert(m_impl->InputType(gsl::narrow_cast<int>(inputIndex))->IsTensorSequenceType());
    ML_CHECK_BOOL(m_impl->InputType(gsl::narrow_cast<int>(inputIndex))->IsTensorSequenceType());
    auto inputTensorSeq = m_impl->Input<onnxruntime::TensorSeq>(gsl::narrow_cast<int>(inputIndex));
    ML_CHECK_BOOL(inputTensorSeq != nullptr);
    *inputCount = static_cast<uint32_t>(inputTensorSeq->Size());
    *dataType = ToMLTensorDataTypePlugin(inputTensorSeq->DataType());
    return S_OK;
    }
    ORT_CATCH_RETURN
}

HRESULT STDMETHODCALLTYPE PluginOpKernelContextWrapper::GetOutputTensor(uint32_t outputIndex,
                                                                  IMLOperatorTensor** tensor) noexcept {
    ORT_TRY{VerifyNotClosed();

*tensor = nullptr;

ML_CHECK_BOOL(outputIndex < m_outputTensors.size());

// GetOutputTensor must be called unless a kernel provides shape inferencing,
// in which case m_outputShapes will be valid here.
if (!m_outputShapes) {
    return E_FAIL;
    // return MLStatus::SHAPE_INFERENCE_NOT_REGISTERED;
}

uint32_t dimensionCount = gsl::narrow_cast<uint32_t>(m_outputShapes->GetShape(outputIndex).size());
return GetOutputTensor(outputIndex, dimensionCount, m_outputShapes->GetShape(outputIndex).data(), tensor);
}
ORT_CATCH_RETURN
}

HRESULT STDMETHODCALLTYPE PluginOpKernelContextWrapper::GetOutputTensor(uint32_t outputIndex, uint32_t dimensions,
                                                                  const uint32_t* dimensionSizes,
                                                                  IMLOperatorTensor** tensor) noexcept {
    ORT_TRY{VerifyNotClosed();
*tensor = nullptr;

ML_CHECK_BOOL(outputIndex < m_outputTensors.size());

// Verify that the provided shape matches the shape determined using the kernel's shape inference function.
if (m_outputTensors[outputIndex][0] == nullptr) {
    if (m_outputShapes) {
        if ((m_outputShapes->GetShape(outputIndex).size() != dimensions ||
             memcmp(dimensionSizes, m_outputShapes->GetShape(outputIndex).data(),
                    dimensions * sizeof(*dimensionSizes)))) {
            return E_INVALIDARG;
        }
    }
    std::vector<int64_t> convertedSizes(dimensions);
    for (size_t i = 0; i < dimensions; ++i) {
        convertedSizes[i] = dimensionSizes[i];
    }

    onnxruntime::TensorShape shape(convertedSizes.data(), dimensions);
    auto outputTensor = m_impl->Output(outputIndex, shape);
    if (outputTensor) {
        Microsoft::WRL::ComPtr<TensorWrapper> tensorWrapper = wil::MakeOrThrow<TensorWrapper>(
            const_cast<onnxruntime::Tensor*>(outputTensor), IsAllocationInterface(outputTensor->Location()),
            m_winmlProvider.Get(), m_internalOperator);

        const_cast<PluginOpKernelContextWrapper*>(this)->m_outputTensors[outputIndex][0] = tensorWrapper;
    }
}

m_outputTensors[outputIndex][0].CopyTo(tensor);

return S_OK;
}
ORT_CATCH_RETURN
}

HRESULT STDMETHODCALLTYPE PluginOpKernelContextWrapper::AllocateTemporaryData(size_t size,
                                                                        IUnknown** abiAllocation) const noexcept {
    ORT_TRY{uint64_t allocId;
return AllocateTemporaryData(size, abiAllocation, &allocId);
}
ORT_CATCH_RETURN
}

HRESULT STDMETHODCALLTYPE PluginOpKernelContextWrapper::AllocateTemporaryData(size_t size, IUnknown** abiAllocation,
                                                                        uint64_t* allocId) const {
    ORT_TRY {
        VerifyNotClosed();

        *abiAllocation = nullptr;
        onnxruntime::AllocatorPtr alloc;
        THROW_IF_NOT_OK(m_impl->GetTempSpaceAllocator(&alloc));

        if (!IsAllocationInterface(alloc->Info())) {
            return E_FAIL;
        }

        Microsoft::WRL::ComPtr<IUnknown> allocation;
        allocation.Attach(static_cast<IUnknown*>(alloc->Alloc(size)));

        *allocId = m_winmlProvider->TryGetPooledAllocationId(allocation.Get(), 0);

        TranslateAllocationDataToAbi(m_winmlProvider.Get(), m_internalOperator, alloc->Info(), allocation.Get(),
                                     abiAllocation);

        if (m_winmlProvider->TransitionsRequiredForOperator(m_internalOperator)) {
            m_winmlProvider->TransitionResourcesForOperator(true, 1, abiAllocation);
        }

        // Ensure the allocation is freed and transitioned when the context destructs
        m_temporaryAllocations.push_back(allocation);
        m_temporaryAbiAllocations.push_back(*abiAllocation);

        return S_OK;
    }
    ORT_CATCH_RETURN
}

void STDMETHODCALLTYPE
PluginOpKernelContextWrapper::GetExecutionInterface(IUnknown** executionInterface) const noexcept {
    m_abiExecutionObject.CopyTo(executionInterface);
}

std::vector<IMLOperatorTensor*> PluginOpKernelContextWrapper::GetInputTensors() {
    std::vector<IMLOperatorTensor*> ret;
    ret.reserve(m_inputTensors.size());

    for (int i = 0; i < m_impl->InputCount(); ++i) {
        Microsoft::WRL::ComPtr<IMLOperatorTensor> tensor;
        ORT_THROW_IF_FAILED(GetInputTensor(i, tensor.GetAddressOf()));
        ret.push_back(m_inputTensors[i][0].Get());
    }

    return ret;
}

std::vector<IMLOperatorTensor*> PluginOpKernelContextWrapper::GetOutputTensors(const EdgeShapes& outputShapes) {
    std::vector<IMLOperatorTensor*> ret;
    ret.reserve(m_outputTensors.size());

    ORT_THROW_HR_IF(E_INVALIDARG, static_cast<size_t>(m_impl->OutputCount()) != outputShapes.EdgeCount());

    for (int i = 0; i < m_impl->OutputCount(); ++i) {
        Microsoft::WRL::ComPtr<IMLOperatorTensor> tensor;
        ORT_THROW_IF_FAILED(GetOutputTensor(i, static_cast<uint32_t>(outputShapes.GetShape(i).size()),
                                            outputShapes.GetShape(i).data(), tensor.GetAddressOf()));

        ret.push_back(m_outputTensors[i][0].Get());
    }

    return ret;
}

    bool PluginOpKernelContextWrapper::IsAllocationInterface(const ::OrtMemoryInfo& info) 
    {
        return strcmp(info.name.c_str(), onnxruntime::CPU) && !(info.mem_type == ::OrtMemType::OrtMemTypeCPUOutput || info.mem_type == ::OrtMemType::OrtMemTypeCPUInput);
    }

    // Translate the data object stored in a tensor to the type which will be returned through
    // the ABI. The translation is determined by the provider and based on options with which the
    // kernels are registered.
    void PluginOpKernelContextWrapper::TranslateAllocationDataToAbi(
        IWinmlExecutionProvider* winmlProvider,
        bool isInternalOperator,
        const ::OrtMemoryInfo& allocInfo,
        IUnknown* allocation,
        IUnknown** abiAllocation) 
    {
        if (winmlProvider) {
            winmlProvider->GetABIDataInterface(isInternalOperator, allocation, abiAllocation);
        } else {
            Microsoft::WRL::ComPtr<IUnknown> tmp = allocation;
            *abiAllocation = tmp.Detach();
        }
    }

    PluginDmlAbiOpKernel::PluginDmlAbiOpKernel(
        IMLOperatorKernelFactory* operatorFactory,
        const onnxruntime::OpKernelInfo& kerneInfo,
        bool requiresInputShapesAtCreation,
        bool requiresOutputShapesAtCreation,
        bool isInternalOperator,
        gsl::span<const uint32_t> requiredConstantCpuInputs,
        IMLOperatorShapeInferrer* shapeInferrer,
        const AttributeMap* defaultAttributes,
        const dml_ep::PluginDmlExecutionProviderImpl* pluginDmlEp)
        : OpKernel(kerneInfo),
        m_requiresInputShapesAtCreation(requiresInputShapesAtCreation),
        m_requiresOutputShapesAtCreation(requiresOutputShapesAtCreation),
        m_shapeInferrer(shapeInferrer),
        m_internalOperator(isInternalOperator),
        m_defaultAttributes(defaultAttributes),
        m_dmlPluginExecutionProvider(pluginDmlEp),
        m_ortKernelInfo(reinterpret_cast<const OrtKernelInfo*>(&kerneInfo)),
        m_ortApi(OrtGetApiBase()->GetApi(ORT_API_VERSION))
    {
        assert(requiresInputShapesAtCreation || !requiresOutputShapesAtCreation);

        m_requiredConstantCpuInputs.assign(requiredConstantCpuInputs.begin(), requiredConstantCpuInputs.end());

        // Populate tensor attribute names by operator type — matches the same lookup in dml_ep.cc.
        // Only ConstantOfShape has a tensor-typed ONNX attribute ("value") in this codebase.
        static const std::unordered_map<std::string, std::vector<std::string>> kTensorAttrNames = {
            {"ConstantOfShape", {"value"}},
        };
        auto tensor_attr_it = kTensorAttrNames.find(kerneInfo.node().OpType());
        if (tensor_attr_it != kTensorAttrNames.end()) {
            m_tensorAttributeNames = tensor_attr_it->second;
        }

        const void* executionHandle = m_dmlPluginExecutionProvider;//kerneInfo.GetExecutionProvider()->GetExecutionHandle();

        if (executionHandle)
        {
            // We assume the execution object inherits IUnknown as its first base
            Microsoft::WRL::ComPtr<IUnknown> providerExecutionObject = const_cast<IUnknown*>(static_cast<const IUnknown*>(executionHandle));
            m_abiExecutionObject = providerExecutionObject;

            // Get the WinML-specific execution provider interface from the execution object.
            providerExecutionObject.As(&m_winmlProvider);

            if (m_winmlProvider)
            {
                // Get the particular object to return to a isInternalOperator based on the registration of that kernel.
                m_winmlProvider->GetABIExecutionInterfaceAndInvalidateState(isInternalOperator, m_abiExecutionObject.ReleaseAndGetAddressOf());
            }
        }

        bool requiredConstantCpuInputsAvailable = true;
        DML_PERF_LOG("[ABI_UNSAFE] ctor: op=", kerneInfo.node().OpType(),
            "  since_ver=", kerneInfo.node().SinceVersion(),
            "  required_const_count=", requiredConstantCpuInputs.size(), "\n");
        for (uint32_t index : requiredConstantCpuInputs)
        {
            int isConstant = 0;
            const OrtValue* constantValue = nullptr;
            OrtStatus* checkStatus = m_ortApi->KernelInfoGetConstantInput_tensor(
                m_ortKernelInfo, index, &isConstant, &constantValue);
            if (checkStatus) m_ortApi->ReleaseStatus(checkStatus);
            bool available = (isConstant && constantValue != nullptr);
            DML_PERF_LOG("[ABI_UNSAFE] ctor const[", index, "]: op=", kerneInfo.node().OpType(),
                "  ki_isconst=", isConstant, "  ki_val=", (void*)constantValue, "  available=", available, "\n");
            if (!available)
            {
                requiredConstantCpuInputsAvailable = false;
                break;
            }
        }

        // If input sizes are either available or not required at creation, no need to delay kernel creation.
        bool immediateCreate = requiredConstantCpuInputsAvailable && (!m_requiresInputShapesAtCreation || InputTensorShapesDefined());
        if (immediateCreate)
        {
            auto winmlProviderCapture = m_winmlProvider;
            auto internalOpCapture = m_internalOperator;

            MLOperatorTensorGetter constantInputGetter = [kerneInfo, winmlProviderCapture, internalOpCapture](uint32_t index)
            {
                Microsoft::WRL::ComPtr<IMLOperatorTensor> tensorWrapper = nullptr;
                const onnxruntime::Tensor* tensor = nullptr;
                if (kerneInfo.TryGetConstantInput(index, &tensor))
                {
                    tensorWrapper = wil::MakeOrThrow<TensorWrapper>(
                        const_cast<onnxruntime::Tensor*>(tensor),
                        IsAllocationInterface(tensor->Location()),
                        winmlProviderCapture.Get(),
                        internalOpCapture);
                }

                return tensorWrapper;
            };

            // If the output size is not dynamic, infer it using the kernel.  Then if the output size was predicted
            // by schema, verify consistency.  The result of inference is stored in m_inferredOutputShapes.
            if (m_requiresOutputShapesAtCreation)
            {
                // Use the same list of required inputs for the shape inferrer and the kernel.
                InferAndVerifyOutputSizes(m_requiredConstantCpuInputs, constantInputGetter, nullptr, m_inferredOutputShapes);
            }

            std::unordered_map<uint32_t, Microsoft::WRL::ComPtr<IMLOperatorTensor>> constantTensors;
            for (uint32_t inputIndex : requiredConstantCpuInputs)
            {
                int isConstant = 0;
                const OrtValue* constantValue = nullptr;
                OrtStatus* getStatus = m_ortApi->KernelInfoGetConstantInput_tensor(
                    m_ortKernelInfo, inputIndex, &isConstant, &constantValue);
                if (getStatus == nullptr && isConstant && constantValue != nullptr)
                {
                    constantTensors[inputIndex] = Microsoft::WRL::Make<dml_ep::AbiSafeTensor>(
                        constantValue, m_ortApi, m_dmlPluginExecutionProvider);
                }
                else if (getStatus)
                {
                    m_ortApi->ReleaseStatus(getStatus);
                }
            }

            auto tensorAttrCache = dml_ep::FetchAllTensorAttributes(m_ortKernelInfo, m_ortApi, m_tensorAttributeNames);

            auto kernelCreationContext = Microsoft::WRL::Make<dml_ep::AbiSafeKernelCreationContext>(
                m_ortKernelInfo,
                m_ortApi,
                m_defaultAttributes,
                &m_requiredConstantCpuInputs,
                m_dmlPluginExecutionProvider,
                std::move(constantTensors),
                nullptr,  // runtime_context
                nullptr,  // operator_name
                m_internalOperator,
                m_requiresInputShapesAtCreation,
                std::move(tensorAttrCache));

            if (m_requiresOutputShapesAtCreation)
            {
                std::vector<std::vector<uint32_t>> outputShapeVecs(m_inferredOutputShapes.EdgeCount());
                for (size_t i = 0; i < m_inferredOutputShapes.EdgeCount(); ++i)
                {
                    outputShapeVecs[i] = m_inferredOutputShapes.GetShape(i);
                }
                kernelCreationContext->SetPrecomputedOutputShapes(outputShapeVecs);
            }

            ORT_THROW_IF_FAILED(operatorFactory->CreateKernel(kernelCreationContext.Get(), m_kernel.GetAddressOf()));

            // Ensure that scheduled work, if any, is completed before freeing the kernel if the execution
            // provider requires this.
            if (m_winmlProvider)
            {
                m_winmlProvider->QueueReference(m_kernel.Get());
            }
        }
        else
        {
            m_operatorFactory = operatorFactory;
        }
    }

    Ort::Status PluginDmlAbiOpKernel::Compute(onnxruntime::OpKernelContext* context) const
    {
        auto winmlProviderCapture = m_winmlProvider;
        auto internalOpCapture = m_internalOperator;

        MLOperatorTensorGetter constantInputGetter = [context, winmlProviderCapture, internalOpCapture](uint32_t index)
        {
            auto inputType = context->InputType(gsl::narrow_cast<int>(index));

            if (inputType != nullptr)
            {
                if (inputType->IsTensorType())
                {
                    Microsoft::WRL::ComPtr<IMLOperatorTensor> tensorWrapper = nullptr;

                    const auto* tensor = context->Input<onnxruntime::Tensor>(gsl::narrow_cast<int>(index));
                    if (tensor != nullptr)
                    {
                        tensorWrapper = wil::MakeOrThrow<TensorWrapper>(
                            const_cast<onnxruntime::Tensor*>(tensor),
                            IsAllocationInterface(tensor->Location()),
                            winmlProviderCapture.Get(),
                            internalOpCapture);
                    }

                    return tensorWrapper;
                }
                else if (inputType->IsTensorSequenceType())
                {
                    std::vector<Microsoft::WRL::ComPtr<IMLOperatorTensor>> tensorWrappers;

                    const auto* tensorSequence = context->Input<onnxruntime::TensorSeq>(gsl::narrow_cast<int>(index));
                    if (tensorSequence != nullptr)
                    {
                        tensorWrappers.reserve(tensorSequence->Size());

                        for (uint32_t sequenceIndex = 0; sequenceIndex < tensorSequence->Size(); ++sequenceIndex)
                        {
                            auto& tensor = tensorSequence->Get(sequenceIndex);
                            auto tensorWrapper = wil::MakeOrThrow<TensorWrapper>(
                                const_cast<onnxruntime::Tensor*>(&tensor),
                                IsAllocationInterface(tensor.Location()),
                                winmlProviderCapture.Get(),
                                internalOpCapture);
                        }
                    }
                }
                else
                {
                    assert(false);
                    ORT_THROW_HR(E_INVALIDARG);
                }
            }

            return Microsoft::WRL::ComPtr<IMLOperatorTensor>();
        };

        auto inferShapesAndCreateKernel = [&, context](const EdgeShapes& inputShapes, EdgeShapes& outputShapes) -> Microsoft::WRL::ComPtr<IMLOperatorKernel> {
            // If the output size is not dynamic, infer it using the kernel. The result of inference is stored in m_inferredOutputShapes.
            if (m_requiresOutputShapesAtCreation)
            {
                // Use the same list of required inputs for the shape inferrer and the kernel.
                OrtKernelContext* ortKernelCtx = reinterpret_cast<OrtKernelContext*>(context);
                InferAndVerifyOutputSizes(m_requiredConstantCpuInputs, constantInputGetter, &inputShapes, outputShapes, ortKernelCtx);
            }

            std::unordered_map<uint32_t, Microsoft::WRL::ComPtr<IMLOperatorTensor>> constantTensorsForKernel;
            for (uint32_t inputIndex : m_requiredConstantCpuInputs)
            {
                int isConstant = 0;
                const OrtValue* constantValue = nullptr;
                OrtStatus* getStatus = m_ortApi->KernelInfoGetConstantInput_tensor(
                    m_ortKernelInfo, inputIndex, &isConstant, &constantValue);
                DML_PERF_LOG("[ABI_UNSAFE] lazy const[", inputIndex, "]: op=", Node().OpType(),
                    "  ki_isconst=", isConstant, "  ki_val=", (void*)constantValue, "  ki_status=", (void*)getStatus, "\n");
                if (getStatus == nullptr && isConstant && constantValue != nullptr)
                {
                    auto wrappedTensor = Microsoft::WRL::Make<dml_ep::AbiSafeTensor>(
                        constantValue, m_ortApi, m_dmlPluginExecutionProvider);
                    constantTensorsForKernel[inputIndex] = wrappedTensor;
                }
                else
                {
                    bool hadError = (getStatus != nullptr);
                    if (getStatus) m_ortApi->ReleaseStatus(getStatus);
                    // Fall back to runtime input tensor (mirrors old plugin's constantInputGetter using context->Input<onnxruntime::Tensor>())
                    const onnxruntime::Tensor* tensor = context->Input<onnxruntime::Tensor>(inputIndex);
                    DML_PERF_LOG("[ABI_UNSAFE] lazy const[", inputIndex, "] ki-miss -> ctx fallback: op=",
                        Node().OpType(), "  ctx_tensor=", (void*)tensor, "\n");
                    if (tensor)
                    {
                        auto internalOpCapture = m_internalOperator;
                        auto winmlProviderCapture = m_winmlProvider;
                        Microsoft::WRL::ComPtr<TensorWrapper> tensorWrapper = wil::MakeOrThrow<TensorWrapper>(
                            const_cast<onnxruntime::Tensor*>(tensor),
                            IsAllocationInterface(tensor->Location()),
                            winmlProviderCapture.Get(),
                            internalOpCapture);
                        constantTensorsForKernel[inputIndex] = tensorWrapper;
                    }
                    else
                    {
                    }
                }
            }

            OrtKernelContext* ortKernelContext = reinterpret_cast<OrtKernelContext*>(context);
            auto tensorAttrCacheLazy = dml_ep::FetchAllTensorAttributes(m_ortKernelInfo, m_ortApi, m_tensorAttributeNames);

            auto kernelCreationContext = Microsoft::WRL::Make<dml_ep::AbiSafeKernelCreationContext>(
                m_ortKernelInfo,
                m_ortApi,
                m_defaultAttributes,
                &m_requiredConstantCpuInputs,
                m_dmlPluginExecutionProvider,
                std::move(constantTensorsForKernel),
                ortKernelContext,
                nullptr,  // operator_name
                m_internalOperator,
                m_requiresInputShapesAtCreation,
                std::move(tensorAttrCacheLazy),
                &inputShapes);  // mirrors old plugin's m_inputShapesOverride passed to PluginOpKernelInfoWrapper

            if (m_requiresOutputShapesAtCreation)
            {
                std::vector<std::vector<uint32_t>> outputShapeVecs(outputShapes.EdgeCount());
                for (size_t i = 0; i < outputShapes.EdgeCount(); ++i)
                {
                    outputShapeVecs[i] = outputShapes.GetShape(i);
                }
                kernelCreationContext->SetPrecomputedOutputShapes(outputShapeVecs);
            }

            Microsoft::WRL::ComPtr<IMLOperatorKernel> ret;
            ORT_THROW_IF_FAILED(m_operatorFactory->CreateKernel(kernelCreationContext.Get(), ret.GetAddressOf()));

            return ret;
        };

        // The kernel creation may have been delayed because input shapes were required but not inferred by schema.
        if (RequiresLazyInitialization())
        {
            std::lock_guard<std::mutex> lock(m_mutex);

            if (RequiresLazyInitialization())
            {
                DML_PERF_LOG("[ABI_UNSAFE] lazy-init triggered: op=", Node().OpType(),
                    "  ctx_input_count=", context->InputCount(), "\n");
                m_inputShapesOfKernelInference = GetInputShapes(context);

                m_constantInputTensorContentsOfKernel.resize(context->InputCount());
                for (uint32_t index : m_requiredConstantCpuInputs)
                {
                    if (index >= m_constantInputTensorContentsOfKernel.size())
                    {
                        continue;
                    }

                    auto constantInput = constantInputGetter(index);

                    std::visit([this, context, index](auto&& arg) {
                        FillConstantInputs(arg, context, index);
                    }, constantInput);
                }

                m_kernel = inferShapesAndCreateKernel(m_inputShapesOfKernelInference, m_inferredOutputShapes);
                SetLazyInitialized();
            }
        }
        else if (m_inputShapesOfKernelInference.EdgeCount() > 0)
        {
            EdgeShapes local_input_shapes = GetInputShapes(context);

            bool requiredCpuInputsChanged = false;
            for (uint32_t index : m_requiredConstantCpuInputs)
            {
                if (index >= m_constantInputTensorContentsOfKernel.size())
                {
                    continue;
                }

                auto constantInput = constantInputGetter(index);
                requiredCpuInputsChanged = std::visit([this, index](auto&& arg){
                    return RequiredCpuInputChanged(arg, index);
                }, constantInput);

                if (requiredCpuInputsChanged)
                {
                    break;
                }
            }

            // In the edge case that the input size is changing across invocations and the kernel requires
            // its input size at construction, use a local instance of the kernel.
            if (local_input_shapes != m_inputShapesOfKernelInference || requiredCpuInputsChanged)
            {
                EdgeShapes localInferredOutputShapes;
                Microsoft::WRL::ComPtr<IMLOperatorKernel> localKernel = inferShapesAndCreateKernel(local_input_shapes, localInferredOutputShapes);

                Microsoft::WRL::ComPtr<PluginOpKernelContextWrapper> kernelContextWrapper = wil::MakeOrThrow<PluginOpKernelContextWrapper>(
                    context,
                    m_dmlPluginExecutionProvider,
                    m_internalOperator,
                    m_requiresOutputShapesAtCreation ? &localInferredOutputShapes : nullptr);

                ORT_THROW_IF_FAILED(localKernel->Compute(kernelContextWrapper.Get()));
                kernelContextWrapper->Close();

                // Ensure that scheduled work, if any, is completed before freeing the kernel if the execution
                // provider requires this.
                if (m_winmlProvider)
                {
                    m_winmlProvider->QueueReference(localKernel.Get());
                }
                return STATUS_OK;
            }
        }

        Microsoft::WRL::ComPtr<PluginOpKernelContextWrapper> kernelContextWrapper = wil::MakeOrThrow<PluginOpKernelContextWrapper>(
            context,
            m_dmlPluginExecutionProvider,
            m_internalOperator,
            m_requiresOutputShapesAtCreation ? &m_inferredOutputShapes : nullptr);

        ORT_THROW_IF_FAILED(m_kernel->Compute(kernelContextWrapper.Get()));
        kernelContextWrapper->Close();

        // Ensure that scheduled work, if any, is completed before freeing the kernel if the execution
        // provider requires this.
        if (m_winmlProvider)
        {
            m_winmlProvider->QueueReference(m_kernel.Get());
        }

        return STATUS_OK;
    }

    bool PluginDmlAbiOpKernel::RequiredCpuInputChanged(const Microsoft::WRL::ComPtr<IMLOperatorTensor>& constantTensor, uint32_t index) const
    {
        assert(std::holds_alternative<TensorContent>(m_constantInputTensorContentsOfKernel[index]));

        auto lastValue = std::get<TensorContent>(m_constantInputTensorContentsOfKernel[index]);
        MLOperatorTensor currentValue(constantTensor.Get());

        if (lastValue.isValid != (currentValue.GetInterface() != nullptr))
        {
            return false;
        }

        if (lastValue.isValid)
        {
            if (lastValue.shape != currentValue.GetShape() ||
                lastValue.type != currentValue.GetTensorDataType() ||
                currentValue.GetUnalignedTensorByteSize() != lastValue.data.size() ||
                (memcmp(lastValue.data.data(), currentValue.GetByteData(), lastValue.data.size()) != 0))
            {
                return true;
            }
        }

        return false;
    }

    bool PluginDmlAbiOpKernel::RequiredCpuInputChanged(const std::vector<Microsoft::WRL::ComPtr<IMLOperatorTensor>>& constantTensorSequence, uint32_t index) const
    {
        assert(std::holds_alternative<std::vector<TensorContent>>(m_constantInputTensorContentsOfKernel[index]));
        auto lastValues = std::get<std::vector<TensorContent>>(m_constantInputTensorContentsOfKernel[index]);

        for (uint32_t sequenceIndex = 0; sequenceIndex < constantTensorSequence.size(); ++sequenceIndex)
        {
            const auto& lastValue = lastValues[sequenceIndex];
            MLOperatorTensor currentValue(constantTensorSequence[sequenceIndex].Get());

            if (lastValue.isValid != (currentValue.GetInterface() != nullptr))
            {
                return false;
            }

            if (lastValue.isValid)
            {
                if (lastValue.shape != currentValue.GetShape() ||
                    lastValue.type != currentValue.GetTensorDataType() ||
                    currentValue.GetUnalignedTensorByteSize() != lastValue.data.size() ||
                    (memcmp(lastValue.data.data(), currentValue.GetByteData(), lastValue.data.size()) != 0))
                {
                    return true;
                }
            }
        }

        return false;
    }

    void PluginDmlAbiOpKernel::FillConstantInputs(const Microsoft::WRL::ComPtr<IMLOperatorTensor>& constantTensor, onnxruntime::OpKernelContext* context, uint32_t index) const
    {
        // Skip optional constant tensors.
        if (constantTensor != nullptr)
        {
            MLOperatorTensor tensor = MLOperatorTensor(constantTensor.Get());

            if (index >= static_cast<uint32_t>(context->InputCount()))
            {
                return;
            }

            TensorContent tensorContent{};
            tensorContent.isValid = (tensor.GetInterface() != nullptr);

            if (tensor.GetInterface() != nullptr)
            {
                tensorContent.shape = tensor.GetShape();
                tensorContent.type = tensor.GetTensorDataType();
                tensorContent.data.resize(tensor.GetUnalignedTensorByteSize());
            }

            tensorContent.data.assign(
                reinterpret_cast<const std::byte*>(tensor.GetByteData()),
                reinterpret_cast<const std::byte*>(tensor.GetByteData()) + tensor.GetUnalignedTensorByteSize());

            m_constantInputTensorContentsOfKernel[index] = std::move(tensorContent);
        }
    }

    void PluginDmlAbiOpKernel::FillConstantInputs(const std::vector<Microsoft::WRL::ComPtr<IMLOperatorTensor>>& constantTensorSequence, onnxruntime::OpKernelContext* context, uint32_t index) const
    {
        std::vector<TensorContent> tensorContent(constantTensorSequence.size());

        for (uint32_t i = 0; i < constantTensorSequence.size(); ++i)
        {
            const Microsoft::WRL::ComPtr<IMLOperatorTensor>& constantTensor = constantTensorSequence[i];

            // Skip optional constant tensors.
            if (constantTensor == nullptr)
            {
                continue;
            }

            MLOperatorTensor tensor = MLOperatorTensor(constantTensor.Get());

            if (index >= static_cast<uint32_t>(context->InputCount()))
            {
                continue;
            }
            tensorContent[i].isValid = (tensor.GetInterface() != nullptr);

            if (tensor.GetInterface() != nullptr)
            {
                tensorContent[i].shape = tensor.GetShape();
                tensorContent[i].type = tensor.GetTensorDataType();
                tensorContent[i].data.resize(tensor.GetUnalignedTensorByteSize());
            }
            tensorContent[i].data.assign(
                reinterpret_cast<const std::byte*>(tensor.GetByteData()),
                reinterpret_cast<const std::byte*>(tensor.GetByteData()) + tensor.GetUnalignedTensorByteSize());
        }

        m_constantInputTensorContentsOfKernel[index] = std::move(tensorContent);
    }

    bool PluginDmlAbiOpKernel::InputTensorShapesDefined() const
    {
        // Mirrors the old version's InputTensorShapesDefinedOnNode logic, using the ORT C API
        // instead of TypeProto to avoid protobuf ABI issues.
        // Returns true only if every tensor input has a fully static shape (all dims concrete).
        size_t inputCount = 0;
        OrtStatus* status = m_ortApi->KernelInfo_GetInputCount(m_ortKernelInfo, &inputCount);
        if (status) { m_ortApi->ReleaseStatus(status); return false; }

        for (size_t i = 0; i < inputCount; ++i)
        {
            OrtTypeInfo* typeInfo = nullptr;
            status = m_ortApi->KernelInfo_GetInputTypeInfo(m_ortKernelInfo, i, &typeInfo);
            if (status) { m_ortApi->ReleaseStatus(status); return false; }
            if (!typeInfo) continue;

            ONNXType onnxType = ONNX_TYPE_UNKNOWN;
            status = m_ortApi->GetOnnxTypeFromTypeInfo(typeInfo, &onnxType);
            if (status) { m_ortApi->ReleaseStatus(status); m_ortApi->ReleaseTypeInfo(typeInfo); return false; }

            if (onnxType == ONNX_TYPE_SEQUENCE)
            {
                m_ortApi->ReleaseTypeInfo(typeInfo);
                return false;
            }

            if (onnxType == ONNX_TYPE_TENSOR)
            {
                const OrtTensorTypeAndShapeInfo* shapeInfo = nullptr;
                status = m_ortApi->CastTypeInfoToTensorInfo(typeInfo, &shapeInfo);
                if (status) { m_ortApi->ReleaseStatus(status); m_ortApi->ReleaseTypeInfo(typeInfo); return false; }

                if (!shapeInfo || !m_ortApi->TensorTypeAndShape_HasShape(shapeInfo))
                {
                    m_ortApi->ReleaseTypeInfo(typeInfo);
                    return false;
                }

                size_t dimCount = 0;
                status = m_ortApi->GetDimensionsCount(shapeInfo, &dimCount);
                if (status) { m_ortApi->ReleaseStatus(status); m_ortApi->ReleaseTypeInfo(typeInfo); return false; }

                if (dimCount > 0)
                {
                    std::vector<int64_t> dims(dimCount);
                    status = m_ortApi->GetDimensions(shapeInfo, dims.data(), dimCount);
                    if (status) { m_ortApi->ReleaseStatus(status); m_ortApi->ReleaseTypeInfo(typeInfo); return false; }

                    for (int64_t d : dims)
                    {
                        if (d < 0) // symbolic / dynamic dimension
                        {
                            m_ortApi->ReleaseTypeInfo(typeInfo);
                            return false;
                        }
                    }
                }
            }

            m_ortApi->ReleaseTypeInfo(typeInfo);
        }

        return true;
    }

    EdgeShapes PluginDmlAbiOpKernel::GetInputShapes(onnxruntime::OpKernelContext* context) const
    {
        EdgeShapes ret(context->InputCount());

        for (size_t i = 0; i < ret.EdgeCount(); ++i)
        {
            // The input type is null if unused
            auto inputType = context->InputType(static_cast<int>(i));
            if (inputType != nullptr && inputType->IsTensorType())
            {
                if (context->InputType(gsl::narrow_cast<int>(i))->IsTensorSequenceType())
                {
                    auto inputTensorSeq = context->Input<onnxruntime::TensorSeq>(gsl::narrow_cast<int>(i));
                    for (uint32_t sequenceIndex = 0; sequenceIndex < inputTensorSeq->Size(); ++sequenceIndex)
                    {
                        const auto& tensor = inputTensorSeq->Get(sequenceIndex);
                        ret.GetMutableShape(i).resize(tensor.Shape().GetDims().size());
                        for (size_t j = 0; j < ret.GetMutableShape(i).size(); ++j)
                        {
                            ret.GetMutableShape(i)[j] = gsl::narrow_cast<uint32_t>(tensor.Shape().GetDims()[j]);
                        }
                    }
                }
                else if (context->InputType(gsl::narrow_cast<int>(i))->IsTensorType())
                {
                    const onnxruntime::Tensor* tensor = context->Input<onnxruntime::Tensor>(gsl::narrow_cast<int>(i));
                    if (tensor)
                    {
                        ret.GetMutableShape(i).resize(tensor->Shape().GetDims().size());
                        for (size_t j = 0; j < ret.GetMutableShape(i).size(); ++j)
                        {
                            ret.GetMutableShape(i)[j] = gsl::narrow_cast<uint32_t>(tensor->Shape().GetDims()[j]);
                        }
                    }
                }
                else
                {
                    ORT_THROW_HR(E_INVALIDARG);
                }
            }
        }

        return ret;
    }

    void PluginDmlAbiOpKernel::InferAndVerifyOutputSizes(
        gsl::span<const uint32_t> requiredConstantCpuInputs,
        MLOperatorTensorGetter& constantInputGetter,
        const EdgeShapes* inputShapes,
        EdgeShapes& outputShapes,
        OrtKernelContext* ortContext) const
    {
        auto inferenceContext = Microsoft::WRL::Make<dml_ep::AbiSafeShapeInferenceContext>(
            ortContext,
            m_ortApi,
            m_defaultAttributes,
            m_dmlPluginExecutionProvider,
            m_ortKernelInfo);

        size_t outputCount = 0;
        m_ortApi->KernelInfo_GetOutputCount(m_ortKernelInfo, &outputCount);
        outputShapes.Reset(static_cast<uint32_t>(outputCount));

        ORT_THROW_IF_FAILED(m_shapeInferrer->InferOutputShapes(inferenceContext.Get()));

        const auto& inferredShapes = inferenceContext->GetInferredOutputShapes();
        for (size_t outputIndex = 0; outputIndex < outputShapes.EdgeCount(); ++outputIndex)
        {
            if (outputIndex < inferredShapes.size() && !inferredShapes[outputIndex].empty())
            {
                outputShapes.GetMutableShape(outputIndex) = inferredShapes[outputIndex];
            }
        }

        for (size_t outputIndex = 0; outputIndex < outputShapes.EdgeCount(); ++outputIndex)
        {
            OrtTypeInfo* typeInfo = nullptr;
            OrtStatus* status = m_ortApi->KernelInfo_GetOutputTypeInfo(m_ortKernelInfo, outputIndex, &typeInfo);
            if (status)
            {
                m_ortApi->ReleaseStatus(status);
                continue;
            }
            if (!typeInfo) continue;

            ONNXType onnxType;
            status = m_ortApi->GetOnnxTypeFromTypeInfo(typeInfo, &onnxType);
            if (status || onnxType != ONNX_TYPE_TENSOR)
            {
                if (status) m_ortApi->ReleaseStatus(status);
                assert(outputShapes.GetShape(outputIndex).empty());
                ML_CHECK_BOOL(outputShapes.GetShape(outputIndex).empty());
                m_ortApi->ReleaseTypeInfo(typeInfo);
                continue;
            }

            const OrtTensorTypeAndShapeInfo* tensorInfo = nullptr;
            status = m_ortApi->CastTypeInfoToTensorInfo(typeInfo, &tensorInfo);
            if (status || !tensorInfo)
            {
                if (status) m_ortApi->ReleaseStatus(status);
                m_ortApi->ReleaseTypeInfo(typeInfo);
                continue;
            }

            if (m_ortApi->TensorTypeAndShape_HasShape(tensorInfo))
            {
                size_t dimCount = 0;
                m_ortApi->GetDimensionsCount(tensorInfo, &dimCount);
                assert(dimCount == outputShapes.GetShape(outputIndex).size());
                ML_CHECK_BOOL(dimCount == outputShapes.GetShape(outputIndex).size());

                std::vector<int64_t> dims(dimCount);
                m_ortApi->GetDimensions(tensorInfo, dims.data(), dimCount);

                for (uint32_t dim = 0; dim < outputShapes.GetShape(outputIndex).size(); ++dim)
                {
                    if (dims[dim] > 0)
                    {
                        int64_t expected_size = dims[dim];
                        int64_t actual_size = outputShapes.GetShape(outputIndex)[dim];
                        assert(expected_size == actual_size);
                        ML_CHECK_BOOL(expected_size == actual_size);
                    }
                }
            }

            m_ortApi->ReleaseTypeInfo(typeInfo);
        }
    }

    //==============================================================================
    // ABI-Safe ToMLEdgeDesc Overload for OrtTypeInfo (NO UNSAFE CASTS)
    // Static to avoid conflict with MLOperatorAuthorImpl.cpp version
    //==============================================================================

    static ::MLOperatorEdgeDescription ToMLEdgeDesc(const OrtTypeInfo* type_info) {
        // Initialized to undefined class and data type
        MLOperatorEdgeDescription ret = {};

        if (!type_info) {
            return ret;
        }

        if (type_info->type == ONNX_TYPE_TENSOR) {
            ret.edgeType = MLOperatorEdgeType::Tensor;
            if (type_info->tensor_type_info) {
                auto elem_type = type_info->tensor_type_info.get()->GetElementType();
                ret.tensorDataType = ToMLTensorDataTypePlugin(
                    onnxruntime::DataTypeImpl::TensorTypeFromONNXEnum(static_cast<int>(elem_type))
                        ->AsPrimitiveDataType());
            }
        } else if (type_info->type == ONNX_TYPE_SEQUENCE) {
            ret.edgeType = MLOperatorEdgeType::SequenceTensor;
            if (type_info->sequence_type_info && type_info->sequence_type_info->sequence_key_type_) {
                auto* elem_type_info = type_info->sequence_type_info->sequence_key_type_.get();
                if (elem_type_info && elem_type_info->tensor_type_info) {
                    auto elem_type = elem_type_info->tensor_type_info->GetElementType();
                    ret.tensorDataType = ToMLTensorDataTypePlugin(
                        onnxruntime::DataTypeImpl::TensorTypeFromONNXEnum(static_cast<int>(elem_type))
                            ->AsPrimitiveDataType());
                }
            }
        }

        return ret;
    }

    //==============================================================================
    // PluginAbiSafeMLSupportQueryContext Implementation (NO UNSAFE CASTS)
    //==============================================================================

    Microsoft::WRL::ComPtr<PluginAbiSafeMLSupportQueryContext> PluginAbiSafeMLSupportQueryContext::Create(
        onnxruntime::OpNodeProtoHelper<onnxruntime::AbiSafeProtoHelperNodeContext>* info,
        const AttributeMap* defaultAttributes)
    {
        MLOperatorTensorGetter mLOperatorTensorGetter = MLOperatorTensorGetter();
        return wil::MakeOrThrow<PluginAbiSafeMLSupportQueryContext>(info, defaultAttributes, mLOperatorTensorGetter);
    }

    PluginAbiSafeMLSupportQueryContext::PluginAbiSafeMLSupportQueryContext(
        onnxruntime::OpNodeProtoHelper<onnxruntime::AbiSafeProtoHelperNodeContext>* info,
        const AttributeMap* defaultAttributes,
        MLOperatorTensorGetter& mLOperatorTensorGetter) :
        OpNodeInfoWrapper(info, nullptr, defaultAttributes, gsl::span<const uint32_t>(), mLOperatorTensorGetter)
    {}


}  // namespace winrt::Windows::AI::MachineLearning::implementation
