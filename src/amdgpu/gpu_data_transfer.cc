// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gpu_factory.h"
#include "gpu_data_transfer.h"

namespace gpu_ep {

DataTransfer::DataTransfer(const ProviderFactory& factory)
    : OrtDataTransferImpl{ORT_API_VERSION}, factory_{factory}
{
    OrtDataTransferImpl::Release = [](OrtDataTransferImpl* this_) noexcept {
        // Factories should own and manage the DataTransfer object - do not delete it here!
    };
    OrtDataTransferImpl::CanCopy = [](const OrtDataTransferImpl* this_,
            const OrtMemoryDevice* src_memory_device,
            const OrtMemoryDevice* dst_memory_device) noexcept
    {
        API_CALL_T(const DataTransfer, this_, CanCopy, false, src_memory_device, dst_memory_device);
    };
    OrtDataTransferImpl::CopyTensors = [](OrtDataTransferImpl* this_,
        const OrtValue** src_tensors, OrtValue** dst_tensors, OrtSyncStream** streams,
        size_t num_tensors) noexcept
    {
        API_CALL_S(DataTransfer, this_, CopyTensors, src_tensors, dst_tensors, streams, num_tensors);
    };
}

bool DataTransfer::CanCopy(const OrtMemoryDevice* src_memory_device,
    const OrtMemoryDevice* dst_memory_device) const noexcept
{
    if (backend_data_transfer_ == nullptr) {
        const auto backend_factory{factory_.GetBackendFactory()};
        if (backend_factory == nullptr) {
            return false;
        }
        if (backend_factory->CreateDataTransfer(backend_factory, &backend_data_transfer_) != nullptr) {
            return false;
        }
    }
    return backend_data_transfer_->CanCopy(backend_data_transfer_,
        src_memory_device, dst_memory_device);
}

Ort::Status DataTransfer::CopyTensors(const OrtValue** src_tensors,
    OrtValue** dst_tensors, OrtSyncStream** streams, size_t num_tensors) const noexcept
{
    if (backend_data_transfer_ == nullptr) {
        const auto backend_factory{factory_.GetBackendFactory()};
        if (backend_factory == nullptr) {
            return MAKE_STATUS(ORT_EP_FAIL, "invalid backend factory");
        }
        RETURN_IF_ERROR(backend_factory->CreateDataTransfer(backend_factory, &backend_data_transfer_));
    }
    RETURN_IF_ERROR(backend_data_transfer_->CopyTensors(backend_data_transfer_,
        src_tensors, dst_tensors, streams, num_tensors));
    return STATUS_OK;
}

}  // namespace gpu_ep