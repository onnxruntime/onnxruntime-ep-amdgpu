// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "DmlResourceWrapper.h"

namespace dml_ep {

    class DmlCommittedResourceWrapper : public Com<DmlResourceWrapper>
    {
    public:
        DmlCommittedResourceWrapper(Microsoft::WRL::ComPtr<ID3D12Resource>&& d3d12Resource) : m_d3d12Resource(std::move(d3d12Resource)) {}
        ID3D12Resource* GetD3D12Resource() const final { return m_d3d12Resource.Get(); }

    private:
        Microsoft::WRL::ComPtr<ID3D12Resource> m_d3d12Resource;
    };

}  // namespace dml_ep
