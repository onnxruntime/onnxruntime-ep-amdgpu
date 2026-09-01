// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mgx_precompile.h"

#include <utility>

#include "mgx_program_ops.h"
#include "mgx_utils.h"

namespace mgx_ep {

namespace {

Map<std::vector<std::int64_t>> collect_graph_input_shapes(const Ort::ConstGraph& graph,
    const Ort::ConstNode& fused_node)
{
    Map<std::vector<std::int64_t>> graph_shapes;
    const auto add_shape = [&](const Ort::ConstValueInfo& input) {
        if (input == nullptr || input.IsConstantInitializer()) {
            return;
        }
        graph_shapes[input.GetName()] = input.TypeInfo().GetTensorTypeAndShapeInfo().GetShape();
    };
    for (const auto& input : GetValueInfos(fused_node.GetInputs())) {
        add_shape(input);
    }
    for (const auto& input : GetValueInfos(graph.GetInputs())) {
        add_shape(input);
    }
    for (const auto& input : GetValueInfos(fused_node.GetImplicitInputs())) {
        add_shape(input);
    }
    return graph_shapes;
}

bool non_batch_dims_are_concrete(const std::vector<std::int64_t>& shape)
{
    if (shape.empty()) {
        return false;
    }
    for (std::size_t j{1}; j < shape.size(); ++j) {
        if (shape[j] == -1) {
            return false;
        }
    }
    return true;
}

bool all_dims_are_concrete(const std::vector<std::int64_t>& shape)
{
    if (shape.empty()) {
        return false;
    }
    for (const auto dim : shape) {
        if (dim == -1) {
            return false;
        }
    }
    return true;
}

std::vector<std::int64_t> base_shape_from_graph_shape(const std::vector<std::int64_t>& shape)
{
    if (shape.size() > 1) {
        return {shape.begin() + 1, shape.end()};
    }
    return {};
}

std::vector<std::int64_t> effective_shape_for_bucket_batch(const std::vector<std::int64_t>& base,
    std::size_t bucket_batch)
{
    std::vector<std::int64_t> shape;
    shape.push_back(static_cast<std::int64_t>(bucket_batch));
    shape.insert(shape.end(), base.begin(), base.end());
    return shape;
}

PrecompilePlan ineligible_plan()
{
    return {false, false, {}, {}};
}

std::vector<std::pair<std::string, std::size_t>> build_plan_targets(bool bucketed,
    const std::vector<std::size_t>& batch_sizes, const Map<std::vector<std::int64_t>>& shapes_by_name,
    const Map<std::size_t>& input_name_indices)
{
    if (bucketed) {
        std::vector<std::pair<std::string, std::size_t>> targets;
        targets.reserve(batch_sizes.size());
        for (const auto batch_size : batch_sizes) {
            targets.emplace_back(
                ShapeHashHexForBucketBatch(input_name_indices, shapes_by_name, batch_size),
                batch_size);
        }
        return targets;
    }
    return {{ShapeHashHexForStaticShapes(input_name_indices, shapes_by_name), 0}};
}

}  // namespace

PrecompilePlan BuildPrecompilePlan(const Ort::ConstGraph& graph, const Ort::ConstNode& fused_node,
    const Map<std::size_t>& input_name_indices, std::size_t max_dynamic_batch,
    std::string_view compile_batches)
{
    if (input_name_indices.empty()) {
        return ineligible_plan();
    }

    const auto graph_shapes{collect_graph_input_shapes(graph, fused_node)};
    auto batch_sizes{GenerateCompiledBatchSizes(max_dynamic_batch, compile_batches)};

    if (!batch_sizes.empty()) {
        Map<std::vector<std::int64_t>> shapes_by_name;
        for (const auto& [name, _] : input_name_indices) {
            const auto shape_it{graph_shapes.find(name)};
            if (shape_it == graph_shapes.end()) {
                return ineligible_plan();
            }
            const auto& shape{shape_it->second};
            if (!non_batch_dims_are_concrete(shape)) {
                return ineligible_plan();
            }
            shapes_by_name.emplace(name, base_shape_from_graph_shape(shape));
        }
        return {true, true, std::move(batch_sizes), std::move(shapes_by_name)};
    }

    Map<std::vector<std::int64_t>> shapes_by_name;
    for (const auto& [name, _] : input_name_indices) {
        const auto shape_it{graph_shapes.find(name)};
        if (shape_it == graph_shapes.end()) {
            return ineligible_plan();
        }
        const auto& shape{shape_it->second};
        if (!all_dims_are_concrete(shape)) {
            return ineligible_plan();
        }
        shapes_by_name.emplace(name, shape);
    }

    batch_sizes.push_back(1);
    return {true, false, std::move(batch_sizes), std::move(shapes_by_name)};
}

std::string ShapeHashHexForBucketBatch(const Map<std::size_t>& input_name_indices,
    const Map<std::vector<std::int64_t>>& base_shapes_by_name, std::size_t bucket_batch)
{
    hash::Value input_shapes_hash{};
    for (const auto& [name, _] : input_name_indices) {
        const auto shape{effective_shape_for_bucket_batch(base_shapes_by_name.at(name), bucket_batch)};
        hash::Hash(input_shapes_hash, shape);
    }
    return hash::ToHex(input_shapes_hash);
}

std::string ShapeHashHexForStaticShapes(const Map<std::size_t>& input_name_indices,
    const Map<std::vector<std::int64_t>>& shapes_by_name)
{
    hash::Value input_shapes_hash{};
    for (const auto& [name, _] : input_name_indices) {
        hash::Hash(input_shapes_hash, shapes_by_name.at(name));
    }
    return hash::ToHex(input_shapes_hash);
}

bool AnyPlannedTargetMissing(const PrecompilePlan& plan, const Map<std::size_t>& input_name_indices,
    const Map<migraphx::program>& cached_programs)
{
    const auto& [eligible, bucketed, batch_sizes, shapes_by_name]{plan};
    if (!eligible) {
        return true;
    }
    for (const auto& [hash, _] : build_plan_targets(bucketed, batch_sizes, shapes_by_name, input_name_indices)) {
        if (cached_programs.count(hash) == 0) {
            return true;
        }
    }
    return false;
}

Ort::Status PreloadMxrPrograms(const PrecompilePlan& plan, const Map<std::size_t>& input_name_indices,
    Map<migraphx::program>& cached_programs, bool force_recompile, const fs::path& cache_dir,
    const std::string& mxr_prefix)
{
    const auto& [eligible, bucketed, batch_sizes, shapes_by_name]{plan};
    if (!eligible || cache_dir.empty()) {
        return STATUS_OK;
    }

    for (const auto& [hash, _] : build_plan_targets(bucketed, batch_sizes, shapes_by_name, input_name_indices)) {
        if (cached_programs.count(hash) != 0) {
            continue;
        }
        if (force_recompile) {
            continue;
        }

        const fs::path mxr_path{cache_dir / (mxr_prefix + hash + ".mxr")};
        migraphx::program program;
        if (load_compiled_program(program, mxr_path)) {
            cached_programs.emplace(hash, std::move(program));
        }
    }

    return STATUS_OK;
}

Ort::Status CompileMissingPrograms(const PrecompilePlan& plan, const Map<std::size_t>& input_name_indices,
    std::string_view onnx_string, Map<migraphx::program>& cached_programs, const migraphx::target& target,
    bool fp16_enable, bool bf16_enable, bool int8_enable, bool fp8_enable,
    bool int8_calibration_cache_available, const Map<float>& dynamic_ranges, bool exhaustive_tune,
    const std::string& mlss_use_specific_ops, ComputeMode compute_mode, bool disable_compiled_model_caching,
    const fs::path& model_path, const fs::path& external_data_dir, const fs::path& cache_dir,
    const std::string& mxr_prefix)
{
    const auto& [eligible, bucketed, batch_sizes, shapes_by_name]{plan};
    if (!eligible) {
        return STATUS_OK;
    }

    const auto external_data_dir_resolved{external_data_dir.empty() ?
        model_path.parent_path() : external_data_dir};

    for (const auto& [hash, batch_size] : build_plan_targets(bucketed, batch_sizes, shapes_by_name, input_name_indices)) {
        if (cached_programs.count(hash) != 0) {
            continue;
        }

        fs::path mxr_path;
        if (!cache_dir.empty()) {
            mxr_path = cache_dir / (mxr_prefix + hash + ".mxr");
        }

        migraphx::onnx_options onnx_options;
        onnx_options.set_external_data_path(external_data_dir_resolved.string());
        for (const auto& [name, _] : input_name_indices) {
            const auto shape{bucketed ?
                effective_shape_for_bucket_batch(shapes_by_name.at(name), batch_size) :
                shapes_by_name.at(name)};
            onnx_options.set_input_parameter_shape(name, {shape.begin(), shape.end()});
        }
        auto program{migraphx::parse_onnx_buffer(std::string{onnx_string}, onnx_options)};
        migraphx::program_parameters params;
        calibrate_and_quantize(program, target, params, fp16_enable, bf16_enable, int8_enable, fp8_enable,
            int8_calibration_cache_available, dynamic_ranges);
        compile_program(program, target, exhaustive_tune, mlss_use_specific_ops, compute_mode);
        if (!disable_compiled_model_caching) {
            save_compiled_program(program, mxr_path);
        }
        cached_programs.emplace(hash, std::move(program));
    }

    return STATUS_OK;
}

migraphx::program SelectDefaultProgram(const Map<migraphx::program>& cached_programs, bool bucketed,
    const std::vector<std::size_t>& batch_sizes, const Map<std::vector<std::int64_t>>& shapes_by_name,
    const Map<std::size_t>& input_name_indices)
{
    if (cached_programs.empty()) {
        return {};
    }

    const std::string hash{bucketed ?
        ShapeHashHexForBucketBatch(input_name_indices, shapes_by_name, batch_sizes.back()) :
        ShapeHashHexForStaticShapes(input_name_indices, shapes_by_name)};
    if (const auto it{cached_programs.find(hash)}; it != cached_programs.end()) {
        return it->second;
    }

    return cached_programs.begin()->second;
}

}  // namespace mgx_ep
