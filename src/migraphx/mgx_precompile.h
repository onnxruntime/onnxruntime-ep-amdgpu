// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <migraphx/migraphx.hpp>

#include "common/murmurhash3.h"
#include "common/plugin_ep_utils.h"
#include "mgx_dynamic_batch.h"
#include "mgx_info.h"

namespace fs = std::filesystem;

namespace mgx_ep {

// (eligible, bucketed, batch_sizes, shapes_by_name)
// Bucketed: shapes_by_name holds non-batch dims per input; static: full shapes.
using PrecompilePlan = std::tuple<bool, bool, std::vector<std::size_t>, Map<std::vector<std::int64_t>>>;

// Decide what MXR targets exist for this model at Compile() time.
PrecompilePlan BuildPrecompilePlan(const Ort::ConstGraph& graph, const Ort::ConstNode& fused_node,
    const Map<std::size_t>& input_name_indices, std::size_t max_dynamic_batch,
    std::string_view compile_batches);

// Per-ORT-input axis-0 extent as the graph declares it, indexed by ORT input index, with
// -1 where that dim is symbolic (i.e. it carries the batch).  Returned only when every
// model input's shape is [axis0, <concrete dims>], which makes the whole input shape set
// a function of the batch alone; std::nullopt otherwise.  Lets Compute() derive every
// input's shape from one representative input instead of rescanning all of them.
std::optional<std::vector<std::int64_t>> BuildBatchShapeProfile(const Ort::ConstGraph& graph,
    const Ort::ConstNode& fused_node, const Map<std::size_t>& input_name_indices);

// Per-ORT-input flag, indexed by ORT input index: 1 where the graph declares axis 0 as a
// symbolic dim, i.e. that input's leading axis is the one dynamic batching varies.  Unlike
// BuildBatchShapeProfile this says nothing about the remaining dims, so it is available
// for models whose other axes are dynamic too.  Returns an empty vector when the graph
// does not declare a shape for every model input, or when no input has a symbolic axis 0
// (there would be nothing to read the batch from).
std::vector<char> BuildBatchAxisMask(const Ort::ConstGraph& graph,
    const Ort::ConstNode& fused_node, const Map<std::size_t>& input_name_indices);

// Nodes that combine data across axis 0.  Dynamic batching rounds a request up to a
// compiled bucket and slices the pad rows back off the outputs, which is sound only when
// rows are computed independently -- and the pad rows carry whatever the input arena held
// on an earlier call, not zeros.  An op that reduces, sorts, or contracts over axis 0
// therefore folds that stale data into the rows the caller does read, producing wrong but
// entirely plausible numbers.  Returns a human-readable entry per offending node
// ("OpType(name): reason"); empty means the model is safe to bucket.  Errs toward
// reporting where the cost is low -- an axes list it cannot constant-fold counts as
// possibly axis 0 -- but reads a negative axis on an unranked tensor as a tail axis, and
// does not descend into If/Loop/Scan subgraph bodies.
std::vector<std::string> FindCrossBatchNodes(const std::vector<Ort::ConstNode>& nodes);

// The shape hash Compute() uses for a bucket batch size / fixed static shapes.  The
// caller derives the integer key (hash::ShapeKeyOf) and the MXR filename (ToHex).
hash::Value ShapeHashForBucketBatch(const Map<std::size_t>& input_name_indices,
    const Map<std::vector<std::int64_t>>& base_shapes_by_name, std::size_t bucket_batch);

hash::Value ShapeHashForStaticShapes(const Map<std::size_t>& input_name_indices,
    const Map<std::vector<std::int64_t>>& shapes_by_name);

// cached_programs is keyed by ShapeKey (the low 64 bits of the shape hash).
using CachedPrograms = std::unordered_map<hash::ShapeKey, migraphx::program>;

// True when any planned target hash is absent from cached_programs.
bool AnyPlannedTargetMissing(const PrecompilePlan& plan, const Map<std::size_t>& input_name_indices,
    const CachedPrograms& cached_programs);

// Load every planned MXR from disk into cached_programs (no compile).
Ort::Status PreloadMxrPrograms(const PrecompilePlan& plan, const Map<std::size_t>& input_name_indices,
    CachedPrograms& cached_programs, bool force_recompile, const fs::path& cache_dir,
    const std::string& mxr_prefix);

// Compile and save any planned targets still missing from cached_programs.
Ort::Status CompileMissingPrograms(const PrecompilePlan& plan, const Map<std::size_t>& input_name_indices,
    std::string_view onnx_string, CachedPrograms& cached_programs, const migraphx::target& target,
    bool fp16_enable, bool bf16_enable, bool int8_enable, bool fp8_enable,
    bool int8_calibration_cache_available, const Map<float>& dynamic_ranges, bool exhaustive_tune,
    const std::string& mlss_use_specific_ops, ComputeMode compute_mode,
    const std::vector<std::string>& problem_cache_paths, bool disable_compiled_model_caching,
    const fs::path& model_path, const fs::path& external_data_dir, const fs::path& cache_dir,
    const std::string& mxr_prefix);

// Pick the default program to install on ComputeState after load/compile.
migraphx::program SelectDefaultProgram(const CachedPrograms& cached_programs, bool bucketed,
    const std::vector<std::size_t>& batch_sizes, const Map<std::vector<std::int64_t>>& shapes_by_name,
    const Map<std::size_t>& input_name_indices);

}  // namespace mgx_ep
