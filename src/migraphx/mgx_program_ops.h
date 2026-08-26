// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <unordered_map>

#include <migraphx/migraphx.hpp>

#include "common/plugin_ep_utils.h"
#include "mgx_info.h"
#include "mgx_utils.h"

namespace mgx_ep {

bool load_compiled_program(migraphx::program& prog, const fs::path& path);

void save_compiled_program(const migraphx::program& prog, const fs::path& path);

void calibrate_and_quantize(const migraphx::program& prog, const migraphx::target& target,
    const migraphx::program_parameters& params,
    bool fp16_enable, bool bf16_enable, bool int8_enable, bool fp8_enable, bool int8_calibration_cache_available,
    const std::unordered_map<std::string, float>& dynamic_range_map);

// compute_mode has no default argument on purpose: every call site must state
// which mode it compiles under.
void compile_program(const migraphx::program& prog, const migraphx::target& target, bool exhaustive_tune,
    const std::string& mlss_use_specific_ops, ComputeMode compute_mode);

}  // namespace mgx_ep
