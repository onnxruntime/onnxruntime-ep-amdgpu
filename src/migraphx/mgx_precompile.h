// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <tuple>
#include <vector>

#include <migraphx/migraphx.hpp>

#include "common/murmurhash3.h"
#include "common/plugin_ep_utils.h"
#include "mgx_dynamic_batch.h"

namespace fs = std::filesystem;

namespace mgx_ep {

// (eligible, bucketed, batch_sizes, shapes_by_name)
// Bucketed: shapes_by_name holds non-batch dims per input; static: full shapes.
using PrecompilePlan = std::tuple<bool, bool, std::vector<std::size_t>, Map<std::vector<std::int64_t>>>;

// Decide what MXR targets exist for this model at Compile() time.
PrecompilePlan BuildPrecompilePlan(const Ort::ConstGraph& graph, const Ort::ConstNode& fused_node,
    const Map<std::size_t>& input_name_indices, std::size_t max_dynamic_batch,
    std::string_view compile_batches);

// Build the same shape hash Compute() uses for a bucket batch size.
std::string ShapeHashHexForBucketBatch(const Map<std::size_t>& input_name_indices,
    const Map<std::vector<std::int64_t>>& base_shapes_by_name, std::size_t bucket_batch);

// Build the same shape hash Compute() uses for fixed static input shapes.
std::string ShapeHashHexForStaticShapes(const Map<std::size_t>& input_name_indices,
    const Map<std::vector<std::int64_t>>& shapes_by_name);

// True when any planned target hash is absent from cached_programs.
bool AnyPlannedTargetMissing(const PrecompilePlan& plan, const Map<std::size_t>& input_name_indices,
    const Map<migraphx::program>& cached_programs);

// Load every planned MXR from disk into cached_programs (no compile).
Ort::Status PreloadMxrPrograms(const PrecompilePlan& plan, const Map<std::size_t>& input_name_indices,
    Map<migraphx::program>& cached_programs, bool force_recompile, const fs::path& cache_dir,
    const std::string& mxr_prefix);

// Compile and save any planned targets still missing from cached_programs.
Ort::Status CompileMissingPrograms(const PrecompilePlan& plan, const Map<std::size_t>& input_name_indices,
    std::string_view onnx_string, Map<migraphx::program>& cached_programs, const migraphx::target& target,
    bool fp16_enable, bool bf16_enable, bool int8_enable, bool fp8_enable,
    bool int8_calibration_cache_available, const Map<float>& dynamic_ranges, bool exhaustive_tune,
    const std::string& mlss_use_specific_ops, const std::vector<std::string>& problem_cache_paths,
    bool disable_compiled_model_caching,
    const fs::path& model_path, const fs::path& external_data_dir, const fs::path& cache_dir,
    const std::string& mxr_prefix);

// Pick the default program to install on ComputeState after load/compile.
migraphx::program SelectDefaultProgram(const Map<migraphx::program>& cached_programs, bool bucketed,
    const std::vector<std::size_t>& batch_sizes, const Map<std::vector<std::int64_t>>& shapes_by_name,
    const Map<std::size_t>& input_name_indices);

}  // namespace mgx_ep
