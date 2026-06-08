// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace dml_ep {


class DmlOperatorConcat : public DmlOperator, public OperatorHelper::ConcatHelper
{
public:
    using Self = DmlOperatorConcat;

    DmlOperatorConcat(const MLOperatorKernelCreationContext& kernelInfo)
    :   DmlOperator(kernelInfo),
        ConcatHelper(kernelInfo, kernelInfo.GetTensorShapeDescription())
    {
        auto tensorShapeDescription = kernelInfo.GetTensorShapeDescription();
        std::vector<std::optional<uint32_t>> kernelInputIndices;

        std::vector<OperatorHelper::DimensionType> tensorShape;

        for (uint32_t i = 0; i < kernelInfo.GetInputCount(); i++) {
            // Only keep the non-empty tensors
            if (!OperatorHelper::ContainsEmptyDimensions(tensorShapeDescription.GetInputTensorShape(i))) {
                kernelInputIndices.emplace_back(i);
            }
        }

        DmlOperator::Initialize(kernelInfo, kernelInputIndices);

        // Only execute Concat if it has at least one non-empty input
        if (!m_inputTensorDescs.empty())
        {
            uint32_t dmlAxis = GetDmlAdjustedAxis(m_axis, kernelInfo, m_inputTensorDescs.front().GetDimensionCount());

            std::vector<DML_TENSOR_DESC> inputDescs(m_inputTensorDescs.size());
            for (auto & m_inputTensorDesc : m_inputTensorDescs) {
                if (!OperatorHelper::ContainsEmptyDimensions(m_inputTensorDesc.GetSizes())) {
                    inputDescs.push_back(m_inputTensorDesc.GetDmlDesc());
                }
            }

            const auto outputDescs{GetDmlOutputDescs()};
            DML_JOIN_OPERATOR_DESC joinDesc = {};
            joinDesc.InputCount = gsl::narrow_cast<uint32_t>(inputDescs.size());
            joinDesc.InputTensors = inputDescs.data();
            joinDesc.OutputTensor = outputDescs.data();
            joinDesc.Axis = dmlAxis;

            DML_OPERATOR_DESC opDesc = { DML_OPERATOR_JOIN, &joinDesc };

            SetDmlOperatorDesc(opDesc, kernelInfo);
        }
    }

    void Compute(const MLOperatorKernelContext& kernelContext) override {
        if (const auto inputTensors{GetInputTensorsForExecute(kernelContext)};
            !inputTensors.empty()) {
            THROW_IF_FAILED(m_executionProvider->ExecuteOperator(
                m_compiledOperator, m_persistentResourceBinding,
                inputTensors, GetOutputTensorsForExecute(kernelContext)));
        }
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(Concat, DmlOperatorConcat);

}  // namespace dml_ep
