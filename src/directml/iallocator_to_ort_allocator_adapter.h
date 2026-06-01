// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "core/framework/allocator.h"     // onnxruntime::IAllocator, onnxruntime::AllocatorPtr
#include <onnxruntime_c_api.h>            // OrtAllocator, ORT_API_VERSION

namespace dml_ep {

class IAllocatorToOrtAllocatorAdapter final : public OrtAllocator {
public:
  explicit IAllocatorToOrtAllocatorAdapter(onnxruntime::AllocatorPtr allocator)
      : OrtAllocator{ORT_API_VERSION},
        allocator_{std::move(allocator)} {
    // OrtAllocator function pointers (C ABI) forwarding to IAllocator (C++).
    Alloc = [](OrtAllocator* this_, size_t size) -> void* {
      auto* self = static_cast<IAllocatorToOrtAllocatorAdapter*>(this_);
      return self->allocator_->Alloc(size);
    };

    Free = [](OrtAllocator* this_, void* p) {
      auto* self = static_cast<IAllocatorToOrtAllocatorAdapter*>(this_);
      self->allocator_->Free(p);
    };

    Info = [](const OrtAllocator* this_) -> const OrtMemoryInfo* {
      auto* self = static_cast<const IAllocatorToOrtAllocatorAdapter*>(this_);
      // Important: must return a pointer that stays valid for the lifetime of the OrtAllocator.
      // self->memory_info_storage_ is owned by this adapter object, so the address is stable.
      return &self->memory_info_storage_;
    };

    Reserve = [](OrtAllocator* this_, size_t size) -> void* {
      auto* self = static_cast<IAllocatorToOrtAllocatorAdapter*>(this_);
      return self->allocator_->Reserve(size);
    };

    // Optional stats/stream-aware entry points (leave unimplemented if not needed)
    GetStats = nullptr;
    AllocOnStream = nullptr;

    // Copy OrtMemoryInfo into stable storage used by Info().
    // (IAllocator::Info() returns a reference, but its address/lifetime isn't part of the C ABI contract.)
    memory_info_storage_ = allocator_->Info();
  }

  ~IAllocatorToOrtAllocatorAdapter() = default;

  IAllocatorToOrtAllocatorAdapter(const IAllocatorToOrtAllocatorAdapter&) = delete;
  IAllocatorToOrtAllocatorAdapter& operator=(const IAllocatorToOrtAllocatorAdapter&) = delete;

private:
  onnxruntime::AllocatorPtr allocator_;
  OrtMemoryInfo memory_info_storage_;
};

/// Adapts a std::shared_ptr<OrtAllocator> (C ABI) into an onnxruntime::IAllocator (C++).
/// This is the reverse of IAllocatorToOrtAllocatorAdapter.
class OrtAllocatorToIAllocatorAdapter final : public onnxruntime::IAllocator {
public:
    explicit OrtAllocatorToIAllocatorAdapter(std::shared_ptr<OrtAllocator> ort_allocator) :
        IAllocator(*ort_allocator->Info(ort_allocator.get())), ort_allocator_(std::move(ort_allocator)) {}

    void* Alloc(size_t size) override { return ort_allocator_->Alloc(ort_allocator_.get(), size); }

    void Free(void* p) override { ort_allocator_->Free(ort_allocator_.get(), p); }

private:
    std::shared_ptr<OrtAllocator> ort_allocator_;
};

/// Helper to wrap a std::shared_ptr<OrtAllocator> as a std::shared_ptr<IAllocator>.
inline onnxruntime::AllocatorPtr WrapOrtAllocatorAsIAllocator(std::shared_ptr<OrtAllocator> ort_allocator) {
    return std::make_shared<OrtAllocatorToIAllocatorAdapter>(std::move(ort_allocator));
}

}  // namespace dml_ep
