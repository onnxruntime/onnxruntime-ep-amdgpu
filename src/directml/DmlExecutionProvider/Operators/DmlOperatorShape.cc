// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace dml_ep {


class DmlOperatorShape : public DmlOperator, OperatorHelper::ShapeHelper
{
public:
    explicit DmlOperatorShape(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext),
        ShapeHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription())
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 1, "Shape expects 1 input tensor.");
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1, "Shape expects 1 output tensor.");
        DmlOperator::Initialize(kernelCreationContext);
    }

    // Takes a tensor as input and outputs a 1D int64 tensor containing the shape of the input tensor.
    void Compute(const MLOperatorKernelContext& kernelContext) override {
        const auto inputs{GetInputTensors(kernelContext)};
        const auto outputs{GetOutputTensors(kernelContext)};

        const auto input{inputs[0]};
        const auto output{outputs[0]};

        const uint32_t inputDimCount = input->GetDimensionCount();
        std::vector<uint32_t> inputShape(inputDimCount);
        THROW_IF_FAILED(input->GetShape(inputDimCount, inputShape.data()));

        const std::vector<uint32_t> outputShape(inputShape.begin() + m_sliceStart, inputShape.begin() + m_sliceEnd);

        ML_CHECK_VALID_ARGUMENT(output->IsCpuData(), "Output must be a CPU tensor.");
        const auto outputData{output->GetData<int64_t>()};

        // Write input shape to output data
        for (uint32_t i = 0U; i < outputShape.size(); ++i) {
            outputData[i] = static_cast<int64_t>(outputShape[i]);
        }
    }
};

// Shape is a special case which is hardcoded in AbiCustomRegistry.cpp. If name changes this must be updated.
// Special case makes sure that the input/output resource is created using the CPU allocator.
DML_OP_DEFINE_CREATION_FUNCTION(Shape, DmlOperatorShape);

}  // namespace dml_ep
