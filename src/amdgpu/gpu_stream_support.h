// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/plugin_ep_utils.h"

namespace gpu_ep {

struct ProviderFactory;

struct SyncStream : OrtSyncStreamImpl {
    explicit SyncStream(const ProviderFactory& factory)
        : OrtSyncStreamImpl{ORT_API_VERSION}, factory_{factory}
    {
        OrtSyncStreamImpl::Release = [](OrtSyncStreamImpl* this_) noexcept {
            delete reinterpret_cast<SyncStream*>(this_);
        };
        OrtSyncStreamImpl::GetHandle = [](OrtSyncStreamImpl* this_) noexcept {
            API_CALL_T(SyncStream, this_, GetHandle, static_cast<void*>(nullptr));
        };
        OrtSyncStreamImpl::CreateNotification = [](OrtSyncStreamImpl* this_,
            OrtSyncNotificationImpl** notification) noexcept {
            API_CALL_S(SyncStream, this_, CreateNotification, notification);
        };
        OrtSyncStreamImpl::Flush = [](OrtSyncStreamImpl* this_) noexcept {
            API_CALL_S(SyncStream, this_, Flush);
        };
        OrtSyncStreamImpl::OnSessionRunEnd = [](OrtSyncStreamImpl* this_) noexcept {
            API_CALL_S(SyncStream, this_, OnSessionRunEnd);
        };
    }

private:
    [[nodiscard]] void* GetHandle() const noexcept;
    Ort::Status CreateNotification(OrtSyncNotificationImpl** sync_notification) const noexcept;
    Ort::Status Flush() const noexcept;
    Ort::Status OnSessionRunEnd() noexcept;

    const ProviderFactory& factory_;
};

struct SyncNotification : OrtSyncNotificationImpl {
    explicit SyncNotification(const ProviderFactory& factory)
        : OrtSyncNotificationImpl{ORT_API_VERSION}, factory_{factory}
    {
        OrtSyncNotificationImpl::Release = [](OrtSyncNotificationImpl* this_) noexcept {
            delete reinterpret_cast<SyncNotification*>(this_);
        };
        OrtSyncNotificationImpl::Activate = [](OrtSyncNotificationImpl* this_) noexcept {
            API_CALL_S(SyncNotification, this_, Activate);
        };
        OrtSyncNotificationImpl::WaitOnDevice = [](OrtSyncNotificationImpl* this_,
            OrtSyncStream* consumer_stream) noexcept {
            API_CALL_S(SyncNotification, this_, WaitOnDevice, consumer_stream);
        };
        OrtSyncNotificationImpl::WaitOnHost = [](OrtSyncNotificationImpl* this_) noexcept {
            API_CALL_S(SyncNotification, this_, WaitOnHost);
        };
    }

private:
    [[nodiscard]] Ort::Status Activate() const noexcept;
    Ort::Status WaitOnDevice(OrtSyncStream* consumer_stream) const noexcept;
    [[nodiscard]] Ort::Status WaitOnHost() const noexcept;

    const ProviderFactory& factory_;
};

}  // namespace gpu_ep
