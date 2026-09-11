// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "hip/utils.h"

namespace hip {

struct Allocator : OrtAllocator {
    Allocator(const OrtMemoryInfo* memory_info, const int device_id)
        : OrtAllocator{NegotiatedOrtApiVersion()}, device_id_{device_id}, memory_info_{memory_info},
          pool_enabled_{PoolModeFromEnv()}
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

    ~Allocator();

private:
    [[nodiscard]] void* Alloc(size_t size) const;
    void Free(void* p) const;
    [[nodiscard]] const OrtMemoryInfo* Info() const;

    // Whether to recycle device allocations through a size-keyed free list instead
    // of calling hipMalloc/hipFree on every request.  hipMalloc/hipFree take a
    // device-wide runtime lock and implicitly synchronize, so with this allocator
    // shared process-wide across sessions they serialize the hot path across all
    // parallel instances.  Pooling keeps those calls off the steady-state path.
    // Mirrors MIGraphXAllocator::EnablePoolMode in the integrated ROCm EP.
    static bool PoolModeFromEnv();

    int device_id_{};
    const OrtMemoryInfo* memory_info_{};

    // ── Recycling pool (only touched when pool_enabled_) ──────────────────────
    bool pool_enabled_{};
    mutable std::mutex pool_mutex_{};
    mutable std::unordered_map<size_t, std::vector<void*>> free_list_{};  // idle blocks by size
    mutable std::unordered_map<void*, size_t> alloc_sizes_{};             // live block -> size
};

struct PinnedAllocator final : OrtAllocator {
    PinnedAllocator(const OrtMemoryInfo* memory_info, const int device_id)
        : OrtAllocator{NegotiatedOrtApiVersion()}, device_id_{device_id}, memory_info_{memory_info}
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

// Pin the next device Alloc() on this thread to a caller-owned buffer: the EP arms it
// before GetOutput so the ORT output tensor reuses a pointer-stable staging buffer
// (drift-free direct-bind, no copy-back).  Consumed once; oversize falls through to a
// normal allocation.
void ArmOutputAlloc(void* buffer, std::size_t capacity) noexcept;
void DisarmOutputAlloc() noexcept;

// Drop a borrowed buffer from the registry when the EP frees it, so a later reuse of the
// address is freed normally.
void ReleaseBorrowedOutput(void* buffer) noexcept;

}  // namespace hip