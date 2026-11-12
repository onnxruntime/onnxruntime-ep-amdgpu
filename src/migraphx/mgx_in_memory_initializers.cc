// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mgx_in_memory_initializers.h"

#include <algorithm>

namespace mgx_ep {

namespace {
// Only informational (e.g. in dumped subgraphs): MIGraphX resolves the data by tensor name.
constexpr const char* kInMemoryLocation{"*/_MIGRAPHX_EP_IN_MEMORY_/*"};
}  // namespace

HandleInitializerDataFunc InMemoryInitializers::Collector()
{
    return [this](const Ort::ConstValueInfo& value_info, gsl::span<const std::byte> data,
                  bool& is_external, std::string& location, int64_t& offset) -> Ort::Status {
        is_external = !data.empty();
        if (is_external) {
            entries_.push_back({value_info.GetName(), data});
            location = kInMemoryLocation;
            offset = 0;
        }
        return STATUS_OK;
    };
}

void InMemoryInitializers::Retain()
{
    if (!storage_.empty()) {
        return;
    }
    storage_.reserve(entries_.size());
    for (auto& entry : entries_) {
        // new[] without () leaves the buffer uninitialized; it is overwritten by the copy.
        auto& buffer{storage_.emplace_back(new std::byte[entry.data.size()])};
        std::copy(entry.data.begin(), entry.data.end(), buffer.get());
        entry.data = {buffer.get(), entry.data.size()};
    }
}

void InMemoryInitializers::Register(migraphx::onnx_options& onnx_options) const
{
    for (const auto& [name, data] : entries_) {
        onnx_options.set_external_initializer(name, data.data(), data.size());
    }
}

}  // namespace mgx_ep
