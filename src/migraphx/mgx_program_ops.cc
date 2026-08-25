// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mgx_program_ops.h"

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

void compile_program(const migraphx::program& prog, const migraphx::target& target, bool exhaustive_tune,
    const std::string& mlss_use_specific_ops) {
    migraphx::compile_options options;
    options.set_fast_math(false);
    options.set_exhaustive_tune_flag(exhaustive_tune);
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
