// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// EPContext support for the DX (DirectML) plugin backend.
//
// Generate: wrap a serialized DML graph (SerializeDmlGraph flatbuffer blob) in an
//   "EPContext" OrtNode (op_type EPContext / com.microsoft), tagged with the umbrella
//   EP name so ORT routes a reloaded EPContext model back to this provider. Mirrors
//   src/migraphx/mgx_ep_ctx.{h,cc}.
// Consume: detect an EPContext node produced by this EP and extract its embedded blob
//   (for DeserializeDmlGraph on reload).

#pragma once

#include <filesystem>
#include <string>
#include <string_view>

#include "common/plugin_ep_utils.h"

namespace dml_ep {

namespace fs = std::filesystem;

// The DX backend is created under the AMDGPU umbrella identity (gpu_factory passes
// ep_name_ to dml_create_ep_factories), so EPContext nodes are tagged with the
// umbrella's registered EP name. On reload ORT routes the EPContext node back here.
constexpr std::string_view kEpContextSource = "AMDGPUExecutionProvider";

// Builds an "EPContext" OrtNode that embeds a serialized DML graph (embed_mode=1)
// or references a cache file path (embed_mode=0).
struct EpContextNodeHelper : ApiPtrs {
    EpContextNodeHelper(const ApiPtrs& api_ptrs, const Ort::ConstGraph& graph, const Ort::ConstNode& fused_node)
        : ApiPtrs{api_ptrs}, graph_{graph}, fused_node_{fused_node} {}

    // serialized_graph: the SerializeDmlGraph() flatbuffer blob (embed_mode=1),
    //                   or a cache-relative path string (embed_mode=0).
    Ort::Status CreateEpContextNode(std::string_view serialized_graph,
        int64_t embed_mode, std::string_view compute_capability, const fs::path& model_path,
        std::string_view node_name_prefix, OrtNode*& ep_context_node) const;

private:
    const Ort::ConstGraph& graph_;
    const Ort::ConstNode& fused_node_;
};

// Detects an EPContext node compiled by this EP and extracts the embedded blob.
struct EpContextNodeReader : ApiPtrs {
    EpContextNodeReader(const ApiPtrs& api_ptrs, const Ort::ConstGraph& graph,
        std::string_view current_hw_arch = {});

    // The embedded serialized DML graph blob (feeds DeserializeDmlGraph).
    const std::string& GetSerializedGraph() const { return serialized_graph_; }

    static bool GraphHasContextNode(const Ort::ConstGraph& graph);

private:
    std::string serialized_graph_;
};

}  // namespace dml_ep
