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
