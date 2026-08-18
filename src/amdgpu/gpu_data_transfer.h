// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/plugin_ep_utils.h"

namespace gpu_ep {

struct ProviderFactory;

struct DataTransfer : OrtDataTransferImpl {
    DataTransfer() = delete;
    // ORT creates one instance per session (after that session's CreateEp) and one per
    // factory at library registration (before any CreateEp). These need opposite
    // behaviour, so the constructor branches on whether a backend exists yet:
    //
    //  - backend selected -> per-session: snapshot and freeze. Re-resolving would follow
    //    ProviderFactory's process-global slot into a *later* session's backend.
    //  - no backend -> registration-time instance, the only one the env-level
    //    OrtApi::CopyTensors can reach. Stay lazy; resolving here breaks it permanently.
    //
    // ORT owns each instance and Releases (deletes) it at teardown.
    explicit DataTransfer(const ProviderFactory& factory);

private:
    bool CanCopy(const OrtMemoryDevice* src_memory_device,
        const OrtMemoryDevice* dst_memory_device) const noexcept;

    [[nodiscard]] Ort::Status CopyTensors(const OrtValue** src_tensors,
        OrtValue** dst_tensors, OrtSyncStream** streams, size_t num_tensors) const noexcept;

    // Returns the backend's data transfer. When frozen_, the snapshot taken in the
    // constructor. Otherwise re-queried if the selected backend changed (e.g. profile
    // switch), and null until a backend has been selected.
    OrtDataTransferImpl* GetBackendDataTransfer() const noexcept;

    // True when a backend already existed at construction — i.e. the per-session
    // instance. Not mutable: written only in the constructor, by design. The decision
    // belongs there, not in the const GetBackendDataTransfer().
    bool frozen_{};

    mutable OrtEpFactory* backend_factory_{};
    mutable OrtDataTransferImpl* backend_data_transfer_{};

    const ProviderFactory& factory_;
};

}  // namespace gpu_ep
