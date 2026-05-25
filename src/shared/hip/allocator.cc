// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <stdexcept>
#include "hip/allocator.h"

namespace hip {

namespace {
void SetDevice(const int device_id) {
    int current_device;
    HIP_CALL_THROW(hipGetDevice(&current_device));
    if (current_device != device_id) {
        HIP_CALL_THROW(hipSetDevice(device_id));
    }
#ifdef _DEBUG
    HIP_CALL_THROW(hipGetDevice(&current_device));
    ENFORCE(current_device == device_id, "HIP device mismatch: ",
        current_device, " != ", device_id);
#endif
}
}  // namespace

void* Allocator::Alloc(const size_t size) const {
    void* p{};
    if (size > 0) {
        SetDevice(device_id_);
        HIP_CALL_THROW(hipMalloc(&p, size));
    }
    return p;
}

void Allocator::Free(void* p) const {
    SetDevice(device_id_);
    HIP_CALL_THROW(hipFree(p));
}

const OrtMemoryInfo* Allocator::Info() const {
    return memory_info_;
}

void* PinnedAllocator::Alloc(const size_t size) const {
    void* p{};
    if (size > 0) {
        SetDevice(device_id_);
        HIP_CALL_THROW(hipHostMalloc(&p, size));
    }
    return p;
}

void PinnedAllocator::Free(void* p) const {
    SetDevice(device_id_);
    HIP_CALL_THROW(hipHostFree(p));
}

const OrtMemoryInfo* PinnedAllocator::Info() const {
    return memory_info_;
}

}  // namespace hip
