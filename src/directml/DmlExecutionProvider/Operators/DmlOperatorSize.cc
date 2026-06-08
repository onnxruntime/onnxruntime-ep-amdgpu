// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace dml_ep {

class DmlOperatorSize : public DmlOperator, OperatorHelper::SizeHelper
{
public:
    DmlOperatorSize(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext),
        SizeHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription())
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 1, "Size expects 1 input tensor.");
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1, "Size expects 1 output tensor.");
        DmlOperator::Initialize(kernelCreationContext);
    }

    // Takes a tensor as input and outputs a scalar int64 tensor containing the size of the input tensor.
    void Compute(const MLOperatorKernelContext& kernelContext) override {
        const auto inputs{GetInputTensors(kernelContext)};
        const auto& input{inputs[0]};
        const auto inputDimCount{input->GetDimensionCount()};
        std::vector<uint32_t> inputShape(inputDimCount);
        THROW_IF_FAILED(input->GetShape(inputDimCount, inputShape.data()));
        const auto outputs{GetOutputTensors(kernelContext)};
        const auto& output{outputs[0]};
        ML_CHECK_VALID_ARGUMENT(output->IsCpuData(), "Output must be a CPU tensor.");
        const auto outputData{output->GetData<int64_t>()};
        *outputData = static_cast<uint64_t>(ComputeElementCountFromDimensions(inputShape));
    }
};

// Size is a special case which is hardcoded in AbiCustomRegistry.cpp. If name changes this must be updated.
// Special case makes sure that the input/output resource is created using the CPU allocator.
DML_OP_DEFINE_CREATION_FUNCTION(Size, DmlOperatorSize);

}  // namespace dml_ep
