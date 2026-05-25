// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <gsl/gsl>

#include "common/plugin_ep_utils.h"
#include "common/path_string.h"

namespace gpu_ep {

struct EpContextNodeHelper : ApiPtrs {
    EpContextNodeHelper(const ApiPtrs& api_ptrs, const Ort::ConstGraph& graph, const Ort::ConstNode& fused_node)
        : ApiPtrs{api_ptrs}, graph_{graph}, fused_node_{fused_node} {}

    Ort::Status CreateEpContextNode(const PathString& engine_cache_path,
        gsl::span<char*> engine_data, int64_t embed_mode, std::string_view compute_capability,
        const PathString& onnx_model_path, OrtNode*& ep_context_node);

private:
    const Ort::ConstGraph& graph_;
    const Ort::ConstNode& fused_node_;
};

struct EpContextNodeReader : ApiPtrs {
    static bool GraphHasContextNode(const Ort::ConstGraph& graph) { return false; }
};

}  // namespace gpu_ep
