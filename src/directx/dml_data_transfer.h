// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/plugin_ep_utils.h"

#include <memory>
#include <mutex>
#include <vector>

namespace dml_ep {

class PluginDmlExecutionProviderImpl;
class ExecutionProviderPlugin;

class DMLDataTransfer : public OrtDataTransferImpl, public ApiPtrs
{
public:
    DMLDataTransfer(ApiPtrs api_ptrs);

    static bool ORT_API_CALL CanCopyImpl(const OrtDataTransferImpl* this_ptr, const OrtMemoryDevice* src_memory_device,
                                         const OrtMemoryDevice* dst_memory_device) noexcept;

    // function to copy one or more tensors.
    static OrtStatus* ORT_API_CALL CopyTensorsImpl(OrtDataTransferImpl* this_ptr, const OrtValue** src_tensors_ptr,
                                                   OrtValue** dst_tensors_ptr, OrtSyncStream** streams_ptr,
                                                   size_t num_tensors) noexcept;

    static void ORT_API_CALL ReleaseImpl(OrtDataTransferImpl* this_ptr) noexcept;

    void AttachExecutionProvider(std::shared_ptr<PluginDmlExecutionProviderImpl> ep);
    // Stores a pointer to the factory's m_ep_raw so that CopyTensorsImpl can lazily resolve
    // the EP the first time a copy is requested (factory CreateDataTransfer is called before
    // the EP instance exists).
    void AttachFactoryEpRef(ExecutionProviderPlugin** ep_raw_ref);

    // Register/unregister a live EP so CopyTensorsImpl can route each copy to the EP that owns the
    // tensor's allocation (its own ExecutionContext/queue/fence), instead of a single cached
    // provider. A shared DMLDataTransfer services multiple sessions, each with its own context.
    void RegisterProvider(const std::shared_ptr<PluginDmlExecutionProviderImpl>& ep);
    void UnregisterProvider(const PluginDmlExecutionProviderImpl* ep);

private:
    // Resolve the EP that owns a GPU tensor (by its allocation's owning allocator). Falls back to
    // the attached provider when ownership can't be determined (e.g. CPU-only copy).
    std::shared_ptr<PluginDmlExecutionProviderImpl> ResolveOwningProvider(
        const OrtValue* const* src_tensors, OrtValue* const* dst_tensors, size_t num_tensors);

    std::shared_ptr<PluginDmlExecutionProviderImpl> m_executionProvider;
    ExecutionProviderPlugin** m_ep_raw_ref = nullptr; // non-owning ptr to factory's m_ep_raw

    std::mutex m_providersMutex;
    std::vector<std::shared_ptr<PluginDmlExecutionProviderImpl>> m_providers; // all live EPs
};

}  // namespace dml_ep