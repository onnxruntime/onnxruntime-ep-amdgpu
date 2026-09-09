// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mgx_program_ops.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace mgx_ep {

bool load_compiled_program(migraphx::program& prog, const fs::path& path)
try {
    if (!path.empty() && exists(path)) {
        prog = migraphx::load(path.string().c_str());
        return true;
    }
    return false;
} catch (const std::exception&) {
    return false;
}

void save_compiled_program(const migraphx::program& prog, const fs::path& path) {
    if (!path.empty()) {
        migraphx::file_options options;
        options.set_file_format("msgpack");
        migraphx::save(prog, path.string().c_str(), options);
    }
}

void calibrate_and_quantize(const migraphx::program& prog, const migraphx::target& target,
    const migraphx::program_parameters& params,
    bool fp16_enable, bool bf16_enable, bool int8_enable, bool fp8_enable, bool int8_calibration_cache_available,
    const std::unordered_map<std::string, float>& dynamic_range_map)
{
    if ((int8_enable ^ fp8_enable) && int8_calibration_cache_available) {
        const auto param_shapes{prog.get_parameter_shapes()};
        for (const auto& [key, value] : dynamic_range_map) {
            const auto shape{migraphx::shape(migraphx_shape_float_type)};
            params.add(key.c_str(), migraphx::argument(shape, const_cast<float*>(&value)));
        }
        if (int8_enable) {
            migraphx::quantize_int8_options options;
            options.add_calibration_data(params);
            options.add_op_name("convolution");
            options.add_op_name("dot");
            migraphx::quantize_int8(prog, target, options);
        } else if (fp8_enable) {
            migraphx::quantize_fp8_options options;
            options.add_calibration_data(params);
            migraphx::quantize_fp8(prog, target, options);
        }
    }
    if (fp16_enable) {
        migraphx::quantize_fp16(prog);
    }
    if (bf16_enable) {
        migraphx::quantize_bf16(prog);
    }
}

// Render already-escaped cache paths as the {"read_only_problem_cache_files":[...]} object
// that migraphx's relaxed-JSON backend-option parser expects.
static std::string build_read_only_problem_cache_option(const std::vector<std::string>& problem_cache_paths) {
    std::string json{R"({"read_only_problem_cache_files":[)"};
    for (std::size_t i{0}; i < problem_cache_paths.size(); ++i) {
        if (i != 0) {
            json += ',';
        }
        json += '"' + problem_cache_paths[i] + '"';
    }
    json += "]}";
    return json;
}

void compile_program(const migraphx::program& prog, const migraphx::target& target, bool exhaustive_tune,
    const std::string& mlss_use_specific_ops, ComputeMode compute_mode,
    const std::vector<std::string>& problem_cache_paths) {
    migraphx::compile_options options;
    options.set_fast_math(false);

    // The EP calls this a compute mode; migraphx calls the same concept a compile
    // mode. ComputeMode's enumerator values are migraphx's compile_modes values
    // (see mgx_info.h), so this cast is a straight pass-through. migraphx snaps
    // an unknown value to the nearest mode rather than reporting an error, so
    // assert the correspondence at compile time instead.
    static_assert(static_cast<std::int8_t>(ComputeMode::Eager) == migraphx_compile_mode_eager);
    static_assert(static_cast<std::int8_t>(ComputeMode::Balanced) == migraphx_compile_mode_balanced);
    static_assert(static_cast<std::int8_t>(ComputeMode::Maximum) == migraphx_compile_mode_max);
    options.set_compile_mode(static_cast<std::int8_t>(compute_mode));

    // migraphx's own max mode sets the exhaustive-tune flag on the context
    // (target.cpp), but constructs compile_ops from the unmutated
    // options.exhaustive_tune, so on this build max would be close to a no-op.
    // Set the flag here so Maximum means something without patching migraphx.
    options.set_exhaustive_tune_flag(exhaustive_tune || compute_mode == ComputeMode::Maximum);

    // read_only_problem_cache_files are system-level/shipped caches migraphx must never write
    // back. Passed as a %s argument so a '%' in a path is not read as a format specifier.
    // migraphx builds with the problem-cache feature consume the key; older ones ignore it.
    if (!problem_cache_paths.empty()) {
        const auto json{build_read_only_problem_cache_option(problem_cache_paths)};
        options.set_advance_backend_options("%s", json.c_str());
    }
    if (!mlss_use_specific_ops.empty()) {
        // MIGraphX expects a list of op names; split the comma-separated value.
        std::vector<std::string> ops;
        std::string_view rest{mlss_use_specific_ops};
        while (!rest.empty()) {
            const auto pos{rest.find(',')};
            if (const auto op{rest.substr(0, pos)}; !op.empty()) {
                ops.emplace_back(op);
            }
            if (pos == std::string_view::npos) {
                break;
            }
            rest.remove_prefix(pos + 1);
        }
        options.set_advance_backend_option("mlss_use_specific_ops", ops);
        if (ranges::any_of(ops, [](const std::string& op) { return op == "conv"; })) {
            options.set_advance_backend_option("convolution_layout", "channels_first");
        }
    }
    prog.compile(target, options);
}

}  // namespace mgx_ep
