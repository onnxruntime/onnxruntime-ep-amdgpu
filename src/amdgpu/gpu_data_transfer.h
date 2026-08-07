// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/plugin_ep_utils.h"

namespace gpu_ep {

struct ProviderFactory;

struct DataTransfer : OrtDataTransferImpl {
    DataTransfer() = delete;
    // Per-session: the backend is snapshotted at construction from the factory that
    // the session's CreateEp just selected, and never re-queried. ORT creates one
    // DataTransfer per session (CreateDataTransfer -> GetDataTransfer) and owns it,
    // Releasing (deleting) it at session teardown. backend_factory is null when ORT
    // creates a transfer at library-registration time (before any backend is
    // selected); that instance follows the selected backend instead.
    DataTransfer(const ProviderFactory& factory, OrtEpFactory* backend_factory);

private:
    bool CanCopy(const OrtMemoryDevice* src_memory_device,
        const OrtMemoryDevice* dst_memory_device) const noexcept;

    [[nodiscard]] Ort::Status CopyTensors(const OrtValue** src_tensors,
        OrtValue** dst_tensors, OrtSyncStream** streams, size_t num_tensors) const noexcept;

    OrtDataTransferImpl* GetBackendDataTransfer() const noexcept;

    const ProviderFactory& factory_;
    const bool follows_selected_backend_;
    mutable OrtEpFactory* backend_factory_{};
    mutable OrtDataTransferImpl* backend_data_transfer_{};
};

}  // namespace gpu_ep
