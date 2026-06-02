// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace dml_ep {


class DmlOperatorCrop : public DmlOperator, public OperatorHelper::CropHelper
{
public:
    DmlOperatorCrop(const MLOperatorKernelCreationContext& kernelInfo)
    :   DmlOperator(kernelInfo),
        CropHelper(kernelInfo, kernelInfo.GetTensorShapeDescription())
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() == 1);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);

        DmlOperator::Initialize(kernelInfo);

        // CropHelper coerces the input into 4D by this point.
        auto outputShape = kernelInfo.GetTensorShapeDescription().GetOutputTensorShape(0);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_SLICE_OPERATOR_DESC opDesc = {};
        opDesc.InputTensor = inputDescs.data();
        opDesc.OutputTensor = outputDescs.data();
        opDesc.DimensionCount = OperatorHelper::NchwDimensionCount;
        opDesc.Offsets = m_offsets;
        opDesc.Sizes = outputShape.data();
        opDesc.Strides = c_strides;

        SetDmlOperatorDesc({ DML_OPERATOR_SLICE, &opDesc}, kernelInfo);
    }

    static const uint32_t c_strides[OperatorHelper::NchwDimensionCount];
};

const uint32_t DmlOperatorCrop::c_strides[OperatorHelper::NchwDimensionCount] = {1, 1, 1, 1};

DML_OP_DEFINE_CREATION_FUNCTION(Crop, DmlOperatorCrop);

}  // namespace dml_ep
