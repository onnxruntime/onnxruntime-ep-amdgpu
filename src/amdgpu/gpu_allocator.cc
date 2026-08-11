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
    // Release through the factory the allocator was created from, not whatever
    // backend is current now — a profile switch may have moved GetBackendFactory().
    if (backend_allocator_ != nullptr && backend_factory_ != nullptr) {
        backend_factory_->ReleaseAllocator(backend_factory_, backend_allocator_);
    }
}

// Always lazy — deliberately unlike DataTransfer, which freezes its per-session instance.
// Allocators are factory-owned and shared across sessions, so there is no single session's
// backend to pin to; freezing would bind every later session to the first one selected.
OrtAllocator* Allocator::GetBackendAllocator() const noexcept {
    const auto backend_factory{factory_.GetBackendFactory()};
    if (backend_factory == nullptr) {
        return nullptr;
    }
    if (backend_factory != backend_factory_) {
        // Backend changed (e.g. profile switch) — drop the stale allocator and
        // re-create it from the newly selected backend factory.
        if (backend_allocator_ != nullptr && backend_factory_ != nullptr) {
            backend_factory_->ReleaseAllocator(backend_factory_, backend_allocator_);
        }
        backend_allocator_ = nullptr;
        backend_factory_ = nullptr;
        if (backend_factory->CreateAllocator(backend_factory, memory_info_,
                allocator_options_, &backend_allocator_) != nullptr) {
            backend_allocator_ = nullptr;
            return nullptr;
        }
        backend_factory_ = backend_factory;
    }
    return backend_allocator_;
}

void* Allocator::Alloc(size_t size) const noexcept {
    const auto backend_allocator{GetBackendAllocator()};
    if (backend_allocator == nullptr) {
        return nullptr;
    }
    return backend_allocator->Alloc(backend_allocator, size);
}

void Allocator::Free(void* p) const noexcept {
    const auto backend_allocator{GetBackendAllocator()};
    if (backend_allocator == nullptr) {
        return;
    }
    backend_allocator->Free(backend_allocator, p);
}

const OrtMemoryInfo* Allocator::Info() const noexcept {
    return memory_info_;
}

}  // namespace gpu_ep
