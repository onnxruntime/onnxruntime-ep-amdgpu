// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gpu_factory.h"
#include "gpu_allocator.h"

namespace gpu_ep {

Allocator::Allocator(const ProviderFactory& factory,
        const OrtMemoryInfo* memory_info, const OrtKeyValuePairs* allocator_options)
    : OrtAllocator{ORT_API_VERSION}, factory_{factory},
      allocator_options_{allocator_options}, memory_info_{memory_info}
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

Allocator::~Allocator() {
    if (const auto f{factory_.GetBackendFactory()}; f != nullptr) {
        f->ReleaseAllocator(f, backend_allocator_);
    }
}

void* Allocator::Alloc(size_t size) const noexcept {
    if (backend_allocator_ == nullptr) {
        if (const auto f{factory_.GetBackendFactory()}; f != nullptr) {
            if (f->CreateAllocator(f, memory_info_, allocator_options_, &backend_allocator_) != nullptr) {
                return nullptr;
            }
        }
    }
    return backend_allocator_->Alloc(backend_allocator_, size);
}

void Allocator::Free(void* p) const noexcept {
    if (backend_allocator_ == nullptr) {
        if (const auto f{factory_.GetBackendFactory()}; f != nullptr) {
            if (f->CreateAllocator(f, memory_info_, allocator_options_, &backend_allocator_) != nullptr) {
                return;
            }
        }
    }
    backend_allocator_->Free(backend_allocator_, p);
}

const OrtMemoryInfo* Allocator::Info() const noexcept {
    return memory_info_;
}

}  // namespace gpu_ep
