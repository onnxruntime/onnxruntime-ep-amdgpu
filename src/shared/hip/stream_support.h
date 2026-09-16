// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "hip/utils.h"

namespace hip {

struct SyncStream : OrtSyncStreamImpl, ApiPtrs {
    // When external_stream is non-null the stream is application-owned: we adopt it
    // (compute + ORT's copies share it, so there is no cross-stream race and no need
    // for a device-wide drain) and must NOT destroy it. This mirrors the classic
    // built-in MIGraphXExecutionProvider's user_compute_stream / external_stream_
    // path. When null we create and own a non-blocking stream as before.
    SyncStream(const ApiPtrs& api_ptrs, int device_id, hipStream_t external_stream = nullptr)
        : OrtSyncStreamImpl{NegotiatedOrtApiVersion()}, ApiPtrs{api_ptrs}
    {
        HIP_CALL_THROW(hipSetDevice(device_id));
        if (external_stream != nullptr) {
            stream_ = external_stream;
            owns_stream_ = false;
        } else {
            HIP_CALL_THROW(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
            owns_stream_ = true;
        }

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
        // Never destroy an adopted application-owned stream.
        if (owns_stream_) {
            HIP_CALL_THROW(hipStreamDestroy(stream_));
        }
    }

private:
    [[nodiscard]] void* GetHandle() const noexcept;
    OrtStatus* CreateNotification(OrtSyncNotificationImpl** sync_notification) const noexcept;
    [[nodiscard]] OrtStatus* Flush() const noexcept;
    OrtStatus* OnSessionRunEnd() noexcept;

    hipStream_t stream_{};
    bool owns_stream_{true};
};

struct SyncNotification : OrtSyncNotificationImpl, ApiPtrs {
    SyncNotification(const ApiPtrs& apis, hipStream_t stream)
        : OrtSyncNotificationImpl{NegotiatedOrtApiVersion()}, ApiPtrs{apis}, stream_{stream}
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
