// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace dml_ep {

// Copies first input and ignores others.  Used for operators which perform reshaping.
class DmlOperatorCopy : public DmlOperator
{
public:
    using Self = DmlOperatorCopy;

    DmlOperatorCopy(const MLOperatorKernelCreationContext& kernelInfo) : DmlOperator(kernelInfo)
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() >= 1);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);

        std::vector<std::optional<uint32_t>> kernelInputOutputIndices  = {0};

        Initialize(kernelInfo, kernelInputOutputIndices);

        // DirectML requires the input & output dimensions to be identical, even if the
        // element counts are the same. All this operator does is copy the resource and
        // rearrange the dimensions, so we tell DML that the output dimensions are the
        // same as the input dimensions.
        m_outputTensorDescs.front() = m_inputTensorDescs.front();

        Microsoft::WRL::ComPtr<IMLOperatorKernelCreationContextPrivate> contextPrivate;
        ORT_THROW_IF_FAILED(kernelInfo.GetInterface()->QueryInterface(contextPrivate.GetAddressOf()));

        if (contextPrivate->IsDmlGraphNode())
        {
            std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
            std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

            DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC opDesc = {};
            opDesc.InputTensor = inputDescs.data();
            opDesc.OutputTensor = outputDescs.data();

            SetDmlOperatorDesc({ DML_OPERATOR_ELEMENT_WISE_IDENTITY, &opDesc }, kernelInfo);
        }
    }

    void Compute(const MLOperatorKernelContext& kernelContext)
    {
        MLOperatorTensor inputTensor = kernelContext.GetInputTensor(0);

        // Reshape the output tensor.
        MLOperatorTensor outputTensor = kernelContext.GetOutputTensor(0);

        // Avoid self copying. GetDataInterface() asserts the tensor is a GPU data interface
        // (IsDataInterface()); calling it on a CPU-classified tensor throws E_INVALIDARG. With
        // host-accessible decode inputs the aliased output can be CPU-classified -> the original
        // guard threw -> unhandled -> 0xC0000409 fastfail. A self-copy is only possible when BOTH
        // sides are GPU data interfaces backed by the same resource; if either side is CPU they
        // cannot alias the same D3D12 resource, so treat them as different and perform the copy.
        const bool bothDeviceInterfaces = inputTensor.IsDataInterface() && outputTensor.IsDataInterface();
        const bool isSelfCopy = bothDeviceInterfaces &&
            (inputTensor.GetDataInterface().Get() == outputTensor.GetDataInterface().Get());

        if (!isSelfCopy)
        {
            // Copy elements from input tensor to output tensor.
            ORT_THROW_IF_FAILED(m_executionProvider->CopyTensor(
                outputTensor.GetInterface().Get(),
                inputTensor.GetInterface().Get()));
        }
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(Copy, DmlOperatorCopy);

}  // namespace dml_ep
