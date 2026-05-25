// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "hip/utils.h"

namespace hip {

struct Allocator : OrtAllocator {
    Allocator(const OrtMemoryInfo* memory_info, const int device_id)
        : OrtAllocator{ORT_API_VERSION}, device_id_{device_id}, memory_info_{memory_info}
    {
        OrtAllocator::Alloc = [](OrtAllocator* this_, size_t size) {
            return reinterpret_cast<Allocator*>(this_)->Alloc(size);
        };
        OrtAllocator::Free = [](OrtAllocator* this_, void* p) {
            return reinterpret_cast<Allocator*>(this_)->Free(p);
        };
        OrtAllocator::Info = [](const OrtAllocator* this_) {
            return reinterpret_cast<const Allocator*>(this_)->Info();
        };
    }

private:
    [[nodiscard]] void* Alloc(size_t size) const;
    void Free(void* p) const;
    [[nodiscard]] const OrtMemoryInfo* Info() const;

    int device_id_{};
    const OrtMemoryInfo* memory_info_{};
};

struct PinnedAllocator final : OrtAllocator {
    PinnedAllocator(const OrtMemoryInfo* memory_info, const int device_id)
        : OrtAllocator{ORT_API_VERSION}, device_id_{device_id}, memory_info_{memory_info}
    {
        OrtAllocator::Alloc = [](OrtAllocator* this_, const size_t size) {
            return reinterpret_cast<PinnedAllocator*>(this_)->Alloc(size);
        };
        OrtAllocator::Free = [](OrtAllocator* this_, void* p) {
            return reinterpret_cast<PinnedAllocator*>(this_)->Free(p);
        };
        OrtAllocator::Info = [](const OrtAllocator* this_) {
            return reinterpret_cast<const PinnedAllocator*>(this_)->Info();
        };
    }

private:
    [[nodiscard]] void* Alloc(size_t size) const;
    void Free(void* p) const;
    [[nodiscard]] const OrtMemoryInfo* Info() const;

    int device_id_{};
    const OrtMemoryInfo* memory_info_{};
};

}  // namespace hip