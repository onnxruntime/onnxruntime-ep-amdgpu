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

        if (stream == nullptr) {
            // MIGraphX handles the allocation and copying of initializers to a GPU memory.
            continue;
        }

        const auto stream_handle{static_cast<hipStream_t>(stream.GetHandle())};

        const auto src_device_type{src_memory_info.GetDeviceType()};
        const auto dst_device_type{dst_memory_info.GetDeviceType()};

        void* dst_data{dst.GetTensorMutableData<void*>()};
        const void* src_data{src.GetTensorData<const void*>()};

        const size_t bytes{src.GetTensorSizeInBytes()};

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
                HIP_RETURN_IF_ERROR(hipMemcpy(dst_data, src_data, bytes, hipMemcpyHostToDevice));
                if (src_memory_info.GetDeviceMemoryType() != OrtDeviceMemoryType_HOST_ACCESSIBLE) {
                    HIP_RETURN_IF_ERROR(hipStreamSynchronize(nullptr));
                }
            }
        } else if (src_device_type == OrtMemoryInfoDeviceType_GPU) {
            HIP_RETURN_IF_ERROR(hipMemcpy(dst_data, src_data, bytes, hipMemcpyDeviceToHost));
        } else {
            memcpy(dst_data, src_data, bytes);
        }
    }

    return STATUS_OK;
}

}  // namespace hip