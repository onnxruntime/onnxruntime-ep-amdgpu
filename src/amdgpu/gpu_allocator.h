// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/plugin_ep_utils.h"

namespace gpu_ep {

struct ProviderFactory;

struct Allocator : OrtAllocator {
    Allocator(const ProviderFactory& factory,
        const OrtMemoryInfo* memory_info,
        const OrtKeyValuePairs* allocator_options);

    ~Allocator();

private:
    [[nodiscard]] void* Alloc(size_t) const noexcept;
    void Free(void* p) const noexcept;
    [[nodiscard]] const OrtMemoryInfo* Info() const noexcept;

    // Returns the backend's allocator, re-creating it if the selected backend
    // changed (e.g. profile switch)
    OrtAllocator* GetBackendAllocator() const noexcept;

    mutable OrtEpFactory* backend_factory_{};
    mutable OrtAllocator* backend_allocator_{};

    const ProviderFactory& factory_;
    const OrtKeyValuePairs* allocator_options_{};
    const OrtMemoryInfo* memory_info_{};
};

}  // namespace gpu_ep
