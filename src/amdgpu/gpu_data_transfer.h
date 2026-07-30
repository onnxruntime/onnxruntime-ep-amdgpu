// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/plugin_ep_utils.h"

namespace gpu_ep {

struct ProviderFactory;

struct DataTransfer : OrtDataTransferImpl {
    DataTransfer() = delete;
    explicit DataTransfer(const ProviderFactory& factory);

private:
    bool CanCopy(const OrtMemoryDevice* src_memory_device,
        const OrtMemoryDevice* dst_memory_device) const noexcept;

    [[nodiscard]] Ort::Status CopyTensors(const OrtValue** src_tensors,
        OrtValue** dst_tensors, OrtSyncStream** streams, size_t num_tensors) const noexcept;

    // Returns the backend's data transfer, re-creating it if the selected backend
    // changed (e.g. profile switch). Returns nullptr if no backend is available.
    OrtDataTransferImpl* GetBackendDataTransfer() const noexcept;

    const ProviderFactory& factory_;
    mutable OrtEpFactory* backend_factory_{};
    mutable OrtDataTransferImpl* backend_data_transfer_{};
};

}  // namespace gpu_ep
