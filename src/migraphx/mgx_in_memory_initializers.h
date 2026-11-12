// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <gsl/span>
#include <migraphx/migraphx.hpp>

#include "common/ort_graph_to_proto.h"

namespace mgx_ep {

// Initializers whose data lives in memory (not in an external data file) are handed to
// MIGraphX by pointer instead of being serialized into the ONNX proto. The proto then only
// carries the graph structure, which keeps it below protobuf's 2 GB limit for large models.
class InMemoryInitializers {
public:
    // GraphToProto hook: marks each non-empty in-memory initializer as external and records
    // a view of its data. The views point into ORT-owned memory, valid only during Compile.
    HandleInitializerDataFunc Collector();

    // Copy the recorded data into EP-owned memory. Compute re-parses the proto whenever no
    // compiled program matches the input shapes, which happens after ORT's data may be gone.
    void Retain();

    // Register the data with MIGraphX; must precede every parse_onnx_buffer of the proto.
    void Register(migraphx::onnx_options& onnx_options) const;

private:
    struct Entry {
        std::string name;
        gsl::span<const std::byte> data;
    };
    std::vector<Entry> entries_;
    std::vector<std::unique_ptr<std::byte[]>> storage_;
};

}  // namespace mgx_ep
