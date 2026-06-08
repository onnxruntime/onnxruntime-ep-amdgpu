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

    mutable OrtAllocator* backend_allocator_{};

    const ProviderFactory& factory_;
    const OrtKeyValuePairs* allocator_options_{};
    const OrtMemoryInfo* memory_info_{};
};

}  // namespace gpu_ep
