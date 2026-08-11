// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gpu_factory.h"
#include "gpu_data_transfer.h"

namespace gpu_ep {

DataTransfer::DataTransfer(const ProviderFactory& factory)
    : OrtDataTransferImpl{ORT_API_VERSION}, factory_{factory}
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

    // The backend is resolved on first use, not here: this constructor also runs at
    // library-registration time, before any backend has been selected.
}

OrtDataTransferImpl* DataTransfer::GetBackendDataTransfer() const noexcept {
    const auto backend_factory{factory_.GetBackendFactory()};
    if (backend_factory == nullptr) {
        // No backend selected yet (e.g. called before any CreateEp). Not an error —
        // this instance resolves once a backend exists.
        return nullptr;
    }
    if (backend_factory != backend_factory_) {
        // First use, or the backend changed (e.g. profile switch) — (re-)query the
        // transfer from the currently selected backend factory. The backend owns it
        // (Release is a no-op there), so there is nothing to release for the old one.
        backend_data_transfer_ = nullptr;
        backend_factory_ = nullptr;
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
