// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "DmlSubAllocator.h"

namespace dml_ep {

    struct DmlResourceWrapper;

    class DmlCommittedResourceAllocator : public DmlSubAllocator
    {
    public:
        DmlCommittedResourceAllocator(ID3D12Device* device) : m_device(device) {}
        Microsoft::WRL::ComPtr<DmlResourceWrapper> Alloc(size_t size) final;

    private:
        ID3D12Device* m_device = nullptr;
    };

}  // namespace dml_ep
