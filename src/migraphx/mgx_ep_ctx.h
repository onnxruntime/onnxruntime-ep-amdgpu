// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <migraphx/migraphx.hpp>
#include <gsl/gsl>

#include "mgx_utils.h"

#include "common/plugin_ep_utils.h"

namespace mgx_ep {

constexpr std::string_view kEpContextSource = "MIGraphXExecutionProvider";

struct EpContextNodeHelper : ApiPtrs {
    EpContextNodeHelper(const ApiPtrs& api_ptrs, const Ort::Graph& graph, const Ort::ConstNode& fused_node)
        : ApiPtrs{api_ptrs}, graph_{graph}, fused_node_{fused_node} {}

    Ort::Status CreateEpContextNode(const fs::path& cache_path, const fs::path& cache_dir,
        int64_t embed_mode, std::string_view compute_capability, const fs::path& model_path,
        std::string_view node_name_prefix, OrtNode*& ep_context_node) const;

private:
    const Ort::Graph& graph_;
    const Ort::ConstNode& fused_node_;
};

struct EpContextNodeReader : ApiPtrs {
    EpContextNodeReader(const ApiPtrs& api_ptrs, const Ort::ConstGraph& graph, const Ort::Logger& logger,
        const fs::path& cache_dir, std::string_view current_hw_arch = {},
        std::string_view current_sdk_version = {});

    migraphx::program GetProgram() const {
        return std::move(program_);
    }

    static bool GraphHasContextNode(const Ort::ConstGraph& graph);

private:
    migraphx::program program_;
};

}  // namespace mgx_ep
