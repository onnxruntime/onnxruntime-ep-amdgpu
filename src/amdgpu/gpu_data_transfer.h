// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/plugin_ep_utils.h"

namespace gpu_ep {

struct ProviderFactory;

struct DataTransfer : OrtDataTransferImpl {
    DataTransfer() = delete;
    // The backend is resolved lazily, on first use, and re-resolved if the selected
    // backend changes (e.g. profile switch) — see GetBackendDataTransfer.
    //
    // Lazy is required, not merely convenient: ORT creates a DataTransfer per session
    // (CreateDataTransfer -> GetDataTransfer) *and* one per factory at library
    // registration time, before any CreateEp has run. That registration-time instance
    // is the only one reachable from the env-level OrtApi::CopyTensors, so resolving
    // the backend in the constructor leaves that path permanently broken.
    // ORT owns each instance and Releases (deletes) it at teardown.
    explicit DataTransfer(const ProviderFactory& factory);

private:
    bool CanCopy(const OrtMemoryDevice* src_memory_device,
        const OrtMemoryDevice* dst_memory_device) const noexcept;

    [[nodiscard]] Ort::Status CopyTensors(const OrtValue** src_tensors,
        OrtValue** dst_tensors, OrtSyncStream** streams, size_t num_tensors) const noexcept;

    // Returns the backend's data transfer, re-querying it if the selected backend
    // changed (e.g. profile switch). Null until a backend has been selected.
    OrtDataTransferImpl* GetBackendDataTransfer() const noexcept;

    mutable OrtEpFactory* backend_factory_{};
    mutable OrtDataTransferImpl* backend_data_transfer_{};

    const ProviderFactory& factory_;
};

}  // namespace gpu_ep
