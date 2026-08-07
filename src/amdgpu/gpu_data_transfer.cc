// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gpu_factory.h"
#include "gpu_data_transfer.h"

namespace gpu_ep {

DataTransfer::DataTransfer(const ProviderFactory& factory, OrtEpFactory* backend_factory)
    : OrtDataTransferImpl{ORT_API_VERSION}, factory_{factory},
      follows_selected_backend_{backend_factory == nullptr}, backend_factory_{backend_factory}
{
    OrtDataTransferImpl::Release = [](OrtDataTransferImpl* this_) noexcept {
        // ORT creates one DataTransfer per session and owns it, delete on Release.
        // The backend's data transfer is a factory-owned singleton (no-op Release),
        // so there is nothing to release for backend_data_transfer_ here.
        delete static_cast<DataTransfer*>(this_);
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

    // Snapshot the backend transfer once from the backend this session selected; the
    // registration-time instance has none yet and resolves in GetBackendDataTransfer().
    if (backend_factory_ != nullptr) {
        if (backend_factory_->CreateDataTransfer(backend_factory_, &backend_data_transfer_) != nullptr) {
            backend_data_transfer_ = nullptr;
            backend_factory_ = nullptr;
        }
    }
}

// The instance ORT creates at library-registration time backs OrtApi::CopyTensors outside any
// session (OGA zeroes its KV cache through it). It has no backend yet, and by the time it is used
// two sessions may have selected different backends, so it follows the currently selected one
// rather than snapshotting — the same lazy resolution the Allocator wrapper does.
OrtDataTransferImpl* DataTransfer::GetBackendDataTransfer() const noexcept {
    if (!follows_selected_backend_) {
        return backend_data_transfer_;
    }
    const auto backend_factory{factory_.GetBackendFactory()};
    if (backend_factory == nullptr) {
        return nullptr;
    }
    if (backend_factory != backend_factory_) {
        backend_factory_ = nullptr;
        backend_data_transfer_ = nullptr;
        if (backend_factory->CreateDataTransfer(backend_factory, &backend_data_transfer_) != nullptr) {
            backend_data_transfer_ = nullptr;
            return nullptr;
        }
        backend_factory_ = backend_factory;
    }
    return backend_data_transfer_;
}

bool DataTransfer::CanCopy(const OrtMemoryDevice* src_memory_device,
    const OrtMemoryDevice* dst_memory_device) const noexcept
{
    const auto backend_data_transfer{GetBackendDataTransfer()};
    if (backend_data_transfer == nullptr) {
        return false;
    }
    return backend_data_transfer->CanCopy(backend_data_transfer,
        src_memory_device, dst_memory_device);
}

Ort::Status DataTransfer::CopyTensors(const OrtValue** src_tensors,
    OrtValue** dst_tensors, OrtSyncStream** streams, size_t num_tensors) const noexcept
{
    const auto backend_data_transfer{GetBackendDataTransfer()};
    if (backend_data_transfer == nullptr) {
        return MAKE_STATUS(ORT_EP_FAIL, "invalid backend factory");
    }
    RETURN_IF_ERROR(backend_data_transfer->CopyTensors(backend_data_transfer,
        src_tensors, dst_tensors, streams, num_tensors));
    return STATUS_OK;
}

}  // namespace gpu_ep