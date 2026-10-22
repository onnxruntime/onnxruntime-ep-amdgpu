// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "dml_client.h"
#include "core/common/spin_pause.h"
#include "DmlExecutionProvider/ErrorHandling.h"

namespace dml_ep {

    // Represents a fence which will be signaled at some point (usually by the GPU).
    struct GpuEvent
    {
        uint64_t fenceValue;
        Microsoft::WRL::ComPtr<ID3D12Fence> fence;

        bool IsSignaled() const
        {
            return fence->GetCompletedValue() >= fenceValue;
        }

        // True once the fence's owning device has been removed (TDR / device-hung / reset).
        // After removal the fence will never reach fenceValue, so every wait must bail out.
        bool IsDeviceRemoved() const
        {
            Microsoft::WRL::ComPtr<ID3D12Device> device;
            if (FAILED(fence->GetDevice(IID_PPV_ARGS(device.GetAddressOf()))))
                return true; // device gone entirely -> treat as removed
            return FAILED(device->GetDeviceRemovedReason());
        }

        // Blocks until IsSignaled returns true, or the device is removed (TDR).
        // On device removal this returns instead of waiting forever; callers that need to
        // distinguish success from removal should re-check GetDeviceRemovedReason() afterwards.
        void WaitForSignal(bool cpuSyncSpinningEnabled) const
        {
            if (IsSignaled())
                return; // early-out

            if (cpuSyncSpinningEnabled)
            {
                while (!IsSignaled())
                {
                    if (IsDeviceRemoved())
                        return; // fence will never signal on a removed device
                    // We keep spinning until the fence gets signaled
                    onnxruntime::concurrency::SpinPause();
                }
            }
            else
            {
                wil::unique_handle h(CreateEvent(nullptr, TRUE, FALSE, nullptr));
                ORT_THROW_LAST_ERROR_IF(!h);
                ORT_THROW_IF_FAILED(fence->SetEventOnCompletion(fenceValue, h.get()));

                // Bounded waits with a device-removed re-check between them, so a TDR
                // can't wedge teardown/readback in an INFINITE wait (fence never fires).
                constexpr DWORD kWaitSliceMs = 100;
                for (;;)
                {
                    DWORD waitResult = WaitForSingleObject(h.get(), kWaitSliceMs);
                    if (waitResult == WAIT_OBJECT_0)
                        return; // signaled
                    if (waitResult != WAIT_TIMEOUT)
                        return; // WAIT_FAILED / WAIT_ABANDONED -> stop waiting
                    if (IsSignaled() || IsDeviceRemoved())
                        return;
                }
            }
        }
    };

}  // namespace dml_ep
