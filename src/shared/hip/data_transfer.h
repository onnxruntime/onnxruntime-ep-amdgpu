// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "plugin_ep_utils.h"

namespace hip {

struct DataTransfer final : OrtDataTransferImpl, ApiPtrs {
    DataTransfer() = delete;
    DataTransfer(const ApiPtrs& api_ptrs) : OrtDataTransferImpl{ORT_API_VERSION}, ApiPtrs{api_ptrs}
    {
        OrtDataTransferImpl::Release = [](OrtDataTransferImpl* this_) noexcept {
            // Factories should own and manage the DataTransfer object - do not delete it here!
        };
        OrtDataTransferImpl::CanCopy = [](const OrtDataTransferImpl* this_,
                const OrtMemoryDevice* src_memory_device,
                const OrtMemoryDevice* dst_memory_device) noexcept
        {
            return reinterpret_cast<const DataTransfer*>(this_)->CanCopy(src_memory_device, dst_memory_device);
        };
        OrtDataTransferImpl::CopyTensors = [](OrtDataTransferImpl* this_,
            const OrtValue** src_tensors, OrtValue** dst_tensors, OrtSyncStream** streams,
            size_t num_tensors) noexcept
        {
            API_CALL_S(DataTransfer, this_, CopyTensors, {src_tensors, src_tensors + num_tensors},
                {dst_tensors, dst_tensors + num_tensors}, {streams, streams + num_tensors});
        };
    }

private:
    bool CanCopy(const OrtMemoryDevice* src_memory_device,
        const OrtMemoryDevice* dst_memory_device) const noexcept;

    [[nodiscard]] Ort::Status CopyTensors(const std::vector<Ort::ConstValue>& src_tensors,
        std::vector<Ort::UnownedValue> dst_tensors, std::vector<Ort::UnownedSyncStream> streams) const noexcept;
};

}  // namespace hip