// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include "plugin_ep_utils.h"
#include "DmlExecutionProvider/AllocationInfo.h"
#include "OperatorAuthorHelper/MLOperatorAuthorHelper.h"
#include "dml_ep.h"
//#include "DmlExecutionProvider/DmlCommon.h"
#include "dml_common.h"
#include "dml_bucketized_buffer_allocator.h"
#include "dml_execution_context.h"
#include "dml_execution_provider.h"
#include "dml_pooled_upload_heap.h"
#include "dml_readback_heap.h"
#include "DmlExecutionProvider/DmlCommittedResourceAllocator.h"
#include <memory>

namespace dml_ep {

class ExecutionProviderPlugin;

class DMLDataTransfer : public OrtDataTransferImpl, public ApiPtrs
{
public:
    DMLDataTransfer(ApiPtrs api_ptrs);

    static bool ORT_API_CALL CanCopyImpl(const OrtDataTransferImpl* this_ptr, const OrtMemoryDevice* src_memory_device,
                                         const OrtMemoryDevice* dst_memory_device) noexcept;

    // function to copy one or more tensors.
    static OrtStatus* ORT_API_CALL CopyTensorsImpl(OrtDataTransferImpl* this_ptr, const OrtValue** src_tensors_ptr,
                                                   OrtValue** dst_tensors_ptr, OrtSyncStream** streams_ptr,
                                                   size_t num_tensors) noexcept;

    static void ORT_API_CALL ReleaseImpl(OrtDataTransferImpl* this_ptr) noexcept;

    void AttachExecutionProvider(std::shared_ptr<PluginDmlExecutionProviderImpl> ep);
    // Stores a pointer to the factory's m_ep_raw so that CopyTensorsImpl can lazily resolve
    // the EP the first time a copy is requested (factory CreateDataTransfer is called before
    // the EP instance exists).
    void AttachFactoryEpRef(ExecutionProviderPlugin** ep_raw_ref);

private:
    static bool IsGpuTensor(const onnxruntime::Tensor& tensor);

    std::shared_ptr<PluginDmlExecutionProviderImpl> m_executionProvider;
    ExecutionProviderPlugin** m_ep_raw_ref = nullptr; // non-owning ptr to factory's m_ep_raw
};

}  // namespace dml_ep