// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gsl/gsl>

#include "hip/data_transfer.h"
#include "hip/utils.h"

namespace hip {

bool DataTransfer::CanCopy(const OrtMemoryDevice* src_memory_device, const OrtMemoryDevice* dst_memory_device) const noexcept
{
    const auto src_type{ep_api.MemoryDevice_GetDeviceType(src_memory_device)};
    const auto src_vendor_id{ep_api.MemoryDevice_GetVendorId(src_memory_device)};
    const auto dst_type{ep_api.MemoryDevice_GetDeviceType(dst_memory_device)};
    const auto dst_vendor_id{ep_api.MemoryDevice_GetVendorId(dst_memory_device)};

    if ((src_type == OrtMemoryInfoDeviceType_GPU &&
         src_vendor_id != amd::VendorId) ||
        (dst_type == OrtMemoryInfoDeviceType_GPU &&
         dst_vendor_id != amd::VendorId)) {
        return false;
    }

    return (src_type == OrtMemoryInfoDeviceType_GPU &&
            dst_type == OrtMemoryInfoDeviceType_GPU) ||
           (src_type == OrtMemoryInfoDeviceType_GPU &&
            dst_type == OrtMemoryInfoDeviceType_CPU) ||
           (src_type == OrtMemoryInfoDeviceType_CPU &&
            dst_type == OrtMemoryInfoDeviceType_GPU);
}

Ort::Status DataTransfer::CopyTensors(const std::vector<Ort::ConstValue>& src_tensors,
    std::vector<Ort::UnownedValue> dst_tensors, std::vector<Ort::UnownedSyncStream> streams) const noexcept
{
    for (auto [src, dst, stream] : zip(src_tensors, dst_tensors, streams)) {
        const auto src_memory_info{src.GetTensorMemoryInfo()};
        const auto dst_memory_info{dst.GetTensorMemoryInfo()};

        // A null stream = synchronous copy request (e.g. OGA device-input staging). Fall
        // through with a null handle to the sync branches below rather than skipping; the
        // unbacked-buffer guard keeps the MIGraphX-initializer skip. See PR for details.
        const hipStream_t stream_handle{stream == nullptr
            ? static_cast<hipStream_t>(nullptr)
            : static_cast<hipStream_t>(stream.GetHandle())};

        const auto src_device_type{src_memory_info.GetDeviceType()};
        const auto dst_device_type{dst_memory_info.GetDeviceType()};

        void* dst_data{dst.GetTensorMutableData<void*>()};
        const void* src_data{src.GetTensorData<const void*>()};

        const size_t bytes{src.GetTensorSizeInBytes()};

        if (dst_data == nullptr || src_data == nullptr || bytes == 0) {
            // Nothing valid to copy (e.g. MIGraphX-owned initializer not backed here).
            continue;
        }

        if (dst_device_type == OrtMemoryInfoDeviceType_GPU) {
            if (src_device_type == OrtMemoryInfoDeviceType_GPU) {
                if (dst_data != src_data && bytes > 0) {
                    if (stream_handle != nullptr) {
                        HIP_RETURN_IF_ERROR(hipMemcpyAsync(dst_data, src_data, bytes,
                            hipMemcpyDeviceToDevice, stream_handle));
                        // TODO: incomplete, requires event registration for ack
                    } else {
                        HIP_RETURN_IF_ERROR(hipMemcpy(dst_data, src_data, bytes, hipMemcpyDeviceToDevice));
                        HIP_RETURN_IF_ERROR(hipStreamSynchronize(nullptr));
                    }
                }
            } else {
                if (stream_handle != nullptr) {
                    // ORT gave us the compute stream: enqueue the H2D on it and let ORT's
                    // run start/end stream synchronization order it against the compute that
                    // consumes it. Blocking here (or on the null stream) would serialize the
                    // hot path device-wide across all parallel instances for no benefit.
                    HIP_RETURN_IF_ERROR(hipMemcpyAsync(dst_data, src_data, bytes,
                        hipMemcpyHostToDevice, stream_handle));
                } else {
                    // No stream: a synchronous copy request, so the copy must be complete
                    // when we return. Pinned (host-accessible) source is already synchronous.
                    HIP_RETURN_IF_ERROR(hipMemcpy(dst_data, src_data, bytes, hipMemcpyHostToDevice));
                    if (src_memory_info.GetDeviceMemoryType() != OrtDeviceMemoryType_HOST_ACCESSIBLE) {
                        HIP_RETURN_IF_ERROR(hipStreamSynchronize(nullptr));
                    }
                }
            }
        } else if (src_device_type == OrtMemoryInfoDeviceType_GPU) {
            if (stream_handle != nullptr) {
                // Order the read against the producer stream ORT gave us. A plain hipMemcpy
                // runs on the null stream, which does not wait for a hipStreamNonBlocking
                // one (hip/stream_support.h) and so can read a buffer still being written.
                HIP_RETURN_IF_ERROR(hipMemcpyWithStream(dst_data, src_data, bytes,
                    hipMemcpyDeviceToHost, stream_handle));
            } else {
                HIP_RETURN_IF_ERROR(hipMemcpy(dst_data, src_data, bytes, hipMemcpyDeviceToHost));
            }
        } else {
            memcpy(dst_data, src_data, bytes);
        }
    }

    return STATUS_OK;
}

}  // namespace hip