// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/plugin_ep_utils.h"

#include <hip/hip_runtime_api.h>

namespace mgx_ep {

struct ExternalMemoryHandle : OrtExternalMemoryHandle {
    ExternalMemoryHandle(const OrtExternalMemoryDescriptor& descriptor_in,
        hipExternalMemory_t hip_ext_memory, void* mapped_ptr) noexcept;

    [[nodiscard]] hipExternalMemory_t HipExtMemory() const noexcept { return hip_ext_memory_; }
    [[nodiscard]] void* MappedPtr() const noexcept { return mapped_ptr_; }

private:
    static void ORT_API_CALL ReleaseImpl(OrtExternalMemoryHandle* handle) noexcept;

    hipExternalMemory_t hip_ext_memory_{};
    void* mapped_ptr_{};
};

struct ExternalSemaphoreHandle : OrtExternalSemaphoreHandle {
    explicit ExternalSemaphoreHandle(hipExternalSemaphore_t hip_ext_semaphore) noexcept;

    [[nodiscard]] hipExternalSemaphore_t HipExtSemaphore() const noexcept { return hip_ext_semaphore_; }

private:
    static void ORT_API_CALL ReleaseImpl(OrtExternalSemaphoreHandle* handle) noexcept;

    hipExternalSemaphore_t hip_ext_semaphore_{};
};

struct ExternalResourceImporter : OrtExternalResourceImporterImpl, ApiPtrs {
    ExternalResourceImporter(const ApiPtrs& api_ptrs, int device_id);

private:
    static bool ORT_API_CALL CanImportMemoryImpl(const OrtExternalResourceImporterImpl* this_ptr,
        OrtExternalMemoryHandleType handle_type) noexcept;
    static OrtStatus* ORT_API_CALL ImportMemoryImpl(OrtExternalResourceImporterImpl* this_ptr,
        const OrtExternalMemoryDescriptor* desc, OrtExternalMemoryHandle** out_handle) noexcept;
    static void ORT_API_CALL ReleaseMemoryImpl(OrtExternalResourceImporterImpl* this_ptr,
        OrtExternalMemoryHandle* handle) noexcept;
    static OrtStatus* ORT_API_CALL CreateTensorFromMemoryImpl(OrtExternalResourceImporterImpl* this_ptr,
        const OrtExternalMemoryHandle* mem_handle, const OrtExternalTensorDescriptor* tensor_desc,
        OrtValue** out_tensor) noexcept;

    static bool ORT_API_CALL CanImportSemaphoreImpl(const OrtExternalResourceImporterImpl* this_ptr,
        OrtExternalSemaphoreType type) noexcept;
    static OrtStatus* ORT_API_CALL ImportSemaphoreImpl(OrtExternalResourceImporterImpl* this_ptr,
        const OrtExternalSemaphoreDescriptor* desc, OrtExternalSemaphoreHandle** out_handle) noexcept;
    static void ORT_API_CALL ReleaseSemaphoreImpl(OrtExternalResourceImporterImpl* this_ptr,
        OrtExternalSemaphoreHandle* handle) noexcept;
    static OrtStatus* ORT_API_CALL WaitSemaphoreImpl(OrtExternalResourceImporterImpl* this_ptr,
        OrtExternalSemaphoreHandle* handle, OrtSyncStream* stream, uint64_t value) noexcept;
    static OrtStatus* ORT_API_CALL SignalSemaphoreImpl(OrtExternalResourceImporterImpl* this_ptr,
        OrtExternalSemaphoreHandle* handle, OrtSyncStream* stream, uint64_t value) noexcept;

    static void ORT_API_CALL ReleaseSelfImpl(OrtExternalResourceImporterImpl* this_ptr) noexcept;

    int device_id_{};
};

}  // namespace mgx_ep
