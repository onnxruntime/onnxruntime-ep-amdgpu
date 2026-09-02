// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

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

bool EnvTruthy(const char* name) {
    const char* v{std::getenv(name)};
    if (v == nullptr) {
        return false;
    }
    const std::string_view s{v};
    return s == "1" || s == "true" || s == "True" || s == "TRUE" || s == "on" || s == "ON";
}

// Output-buffer allocation hint state (see hip::ArmOutputAlloc).
struct OutputAllocHint {
    void* buffer{nullptr};
    std::size_t capacity{0};
    bool armed{false};
};
thread_local OutputAllocHint t_output_hint{};

// EP-owned buffers currently lent to ORT as output storage; process-wide, keyed by the
// (globally unique) device pointer so it is correct across allocator instances.
std::mutex& BorrowedOutputsMutex() {
    static std::mutex mutex;
    return mutex;
}
std::unordered_set<void*>& BorrowedOutputs() {
    static std::unordered_set<void*> set;
    return set;
}
}  // namespace

void ArmOutputAlloc(void* buffer, const std::size_t capacity) noexcept {
    t_output_hint = OutputAllocHint{buffer, capacity, buffer != nullptr};
}

void DisarmOutputAlloc() noexcept {
    t_output_hint.armed = false;
}

void ReleaseBorrowedOutput(void* buffer) noexcept {
    if (buffer == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock{BorrowedOutputsMutex()};
    BorrowedOutputs().erase(buffer);
}

bool Allocator::PoolModeFromEnv() {
    // Explicit override wins in either direction (ORT_MIGRAPHX_ALLOCATOR_POOL=0 forces off).
    // Otherwise default to on when hipGraph is enabled, matching the integrated ROCm EP,
    // which turns on its allocator pool under hipGraph.
    if (std::getenv("ORT_MIGRAPHX_ALLOCATOR_POOL") != nullptr) {
        return EnvTruthy("ORT_MIGRAPHX_ALLOCATOR_POOL");
    }
    return EnvTruthy("ORT_MIGRAPHX_HIP_GRAPH_ENABLE");
}

Allocator::~Allocator() {
    // Return idle pooled blocks to the driver. Best-effort: the HIP context may already
    // be torn down at process exit, and this runs on the noexcept deleter path.
    (void)hipSetDevice(device_id_);
    for (auto& [size, blocks] : free_list_) {
        for (void* p : blocks) {
            (void)hipFree(p);
        }
    }
}

void* Allocator::Alloc(const size_t size) const {
    if (size == 0) {
        return nullptr;
    }
    // Consume an armed output hint: hand back the EP-owned buffer (drift-free) instead of
    // allocating.  Consumed once; oversize falls through to a normal allocation.
    if (t_output_hint.armed) {
        void* const hinted{t_output_hint.buffer};
        const std::size_t capacity{t_output_hint.capacity};
        t_output_hint.armed = false;
        if (hinted != nullptr && size <= capacity) {
            std::lock_guard<std::mutex> lock{BorrowedOutputsMutex()};
            BorrowedOutputs().insert(hinted);
            return hinted;
        }
    }
    // Reuse an idle block of the exact size if one is available.
    if (pool_enabled_) {
        std::lock_guard<std::mutex> lock{pool_mutex_};
        if (const auto it{free_list_.find(size)}; it != free_list_.end() && !it->second.empty()) {
            void* p{it->second.back()};
            it->second.pop_back();
            return p;
        }
    }
    void* p{};
    SetDevice(device_id_);
    HIP_CALL_THROW(hipMalloc(&p, size));
    if (pool_enabled_) {
        std::lock_guard<std::mutex> lock{pool_mutex_};
        alloc_sizes_[p] = size;
    }
    return p;
}

void Allocator::Free(void* p) const {
    if (p == nullptr) {
        return;
    }
    // Borrowed output buffers are EP-owned and reused across calls; never free them here
    // (released via ReleaseBorrowedOutput in FreeStaging / ~ExecutionProvider).
    {
        std::lock_guard<std::mutex> lock{BorrowedOutputsMutex()};
        if (BorrowedOutputs().count(p) != 0) {
            return;
        }
    }
    // Recycle instead of releasing to the driver so the next same-size request skips
    // hipMalloc entirely (and the device-wide lock/sync it takes).
    if (pool_enabled_) {
        std::lock_guard<std::mutex> lock{pool_mutex_};
        if (const auto it{alloc_sizes_.find(p)}; it != alloc_sizes_.end()) {
            free_list_[it->second].push_back(p);
            return;
        }
    }
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
