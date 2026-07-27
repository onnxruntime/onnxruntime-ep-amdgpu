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
    // Free is wired as OrtAllocator::Free — ORT's deleter, invoked from ~Tensor, which is implicitly
    // noexcept. A throw here (from SetDevice or hipFree) escapes the noexcept boundary -> std::terminate
    // -> __fastfail (0xC0000409). So the free path MUST NOT throw: best-effort, ignore HIP errors
    // (matches the (void)hipSetDevice/(void)hipFree cleanup pattern in mgx_ep.cc).
    (void)hipSetDevice(device_id_);
    (void)hipFree(p);
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
    // Must not throw — same reason as Allocator::Free above (invoked from ~Tensor, a noexcept deleter).
    (void)hipSetDevice(device_id_);
    (void)hipHostFree(p);
}

const OrtMemoryInfo* PinnedAllocator::Info() const {
    return memory_info_;
}

}  // namespace hip
