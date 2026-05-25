// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "hip/utils.h"

namespace hip {

struct SyncStream : OrtSyncStreamImpl, ApiPtrs {
    SyncStream(const ApiPtrs& api_ptrs, int device_id, const OrtKeyValuePairs* /* stream_options */)
        : OrtSyncStreamImpl{ORT_API_VERSION}, ApiPtrs{api_ptrs}
    {
        HIP_CALL_THROW(hipSetDevice(device_id));
        HIP_CALL_THROW(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));

        OrtSyncStreamImpl::Release = [](OrtSyncStreamImpl* this_) noexcept {
            delete reinterpret_cast<SyncStream*>(this_);
        };
        OrtSyncStreamImpl::GetHandle = [](OrtSyncStreamImpl* this_) noexcept {
            return reinterpret_cast<SyncStream*>(this_)->GetHandle();
        };
        OrtSyncStreamImpl::CreateNotification = [](OrtSyncStreamImpl* this_,
                OrtSyncNotificationImpl** notification) noexcept {
            return reinterpret_cast<SyncStream*>(this_)->CreateNotification(notification);
        };
        OrtSyncStreamImpl::Flush = [](OrtSyncStreamImpl* this_) noexcept {
            return reinterpret_cast<SyncStream*>(this_)->Flush();
        };
        OrtSyncStreamImpl::OnSessionRunEnd = [](OrtSyncStreamImpl* this_) noexcept {
            return reinterpret_cast<SyncStream*>(this_)->OnSessionRunEnd();
        };
    }

    ~SyncStream() noexcept {
        HIP_CALL_THROW(hipStreamDestroy(stream_));
    }

private:
    [[nodiscard]] void* GetHandle() const noexcept;
    OrtStatus* CreateNotification(OrtSyncNotificationImpl** sync_notification) const noexcept;
    [[nodiscard]] OrtStatus* Flush() const noexcept;
    OrtStatus* OnSessionRunEnd() noexcept;

    hipStream_t stream_{};
};

struct SyncNotification : OrtSyncNotificationImpl, ApiPtrs {
    SyncNotification(const ApiPtrs& apis, hipStream_t stream)
        : OrtSyncNotificationImpl{ORT_API_VERSION}, ApiPtrs{apis}, stream_{stream}
    {
        HIP_CALL_THROW(hipEventCreateWithFlags(&event_, hipEventDisableTiming));

        OrtSyncNotificationImpl::Release = [](OrtSyncNotificationImpl* this_) noexcept {
            delete reinterpret_cast<SyncNotification*>(this_);
        };
        OrtSyncNotificationImpl::Activate = [](OrtSyncNotificationImpl* this_) noexcept {
            return reinterpret_cast<SyncNotification*>(this_)->Activate();
        };
        OrtSyncNotificationImpl::WaitOnDevice = [](OrtSyncNotificationImpl* this_,
                OrtSyncStream* consumer_stream) noexcept {
            return reinterpret_cast<SyncNotification*>(this_)->WaitOnDevice(consumer_stream);
        };
        OrtSyncNotificationImpl::WaitOnHost = [](OrtSyncNotificationImpl* this_) noexcept {
            return reinterpret_cast<SyncNotification*>(this_)->WaitOnHost();
        };
    }

    ~SyncNotification() noexcept {
        HIP_CALL_THROW(hipEventDestroy(event_));
    }

private:
    [[nodiscard]] OrtStatus* Activate() const noexcept;
    OrtStatus* WaitOnDevice(OrtSyncStream* consumer_stream) const noexcept;
    [[nodiscard]] OrtStatus* WaitOnHost() const noexcept;

    hipEvent_t event_{};
    hipStream_t stream_{};
};

}  // namespace hip
