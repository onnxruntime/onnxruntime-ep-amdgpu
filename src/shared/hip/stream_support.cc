// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "hip/stream_support.h"

namespace hip {

void* SyncStream::GetHandle() const noexcept {
    return stream_;
}

OrtStatus* SyncStream::CreateNotification(OrtSyncNotificationImpl** sync_notification) const noexcept {
    try {
        auto n{std::make_unique<SyncNotification>(*this, stream_)};
        *sync_notification = n.release();
    } catch (const std::exception& e) {
        RETURN_STATUS(ORT_RUNTIME_EXCEPTION, e.what());
    }
    return nullptr;
}

OrtStatus* SyncStream::Flush() const noexcept {
    return HIP_CALL(hipStreamSynchronize(stream_));
}

OrtStatus* SyncStream::OnSessionRunEnd() noexcept {
    return nullptr;
}

OrtStatus* SyncNotification::Activate() const noexcept {
    return HIP_CALL(hipEventRecord(event_, stream_));
}

OrtStatus* SyncNotification::WaitOnDevice(OrtSyncStream* consumer_stream) const noexcept {
    const auto handle{static_cast<hipStream_t>(ort_api.SyncStream_GetHandle(consumer_stream))};
    return HIP_CALL(hipStreamWaitEvent(handle, event_));
}

OrtStatus* SyncNotification::WaitOnHost() const noexcept {
    return HIP_CALL(hipEventSynchronize(event_));
}

}  // namespace hip
