// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "GraphDescBuilder.h"
#include "DmlGraphFusionTransformer.h"

namespace dml_ep {

    onnxruntime::OpKernel* CreateFusedGraphKernel(
        const onnxruntime::OpKernelInfo& info,
        Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiledExecutionPlanOperator,
        EdgeShapes& outputShapes,
        bool reuseCommandList,
        std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>>& nonOwnedGraphInputsFromInitializers,
        std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>>& initializeResourceRefs,
        std::vector<DML_BUFFER_BINDING> initInputBindings,
        std::vector<uint8_t>&& isInputsUploadedByDmlEP,
        std::vector<bool>&& inputsUsed
    );

}  // namespace dml_ep
