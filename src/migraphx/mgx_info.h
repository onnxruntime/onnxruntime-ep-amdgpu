// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>

#include "common/plugin_ep_utils.h"
#include "mgx_utils.h"

namespace mgx_ep {

// The enumerator values are load-bearing: they are migraphx's compile_modes
// values, so the value can be passed straight to
// migraphx::compile_options::set_compile_mode() without translation (see the
// static_asserts in mgx_ep.cc's compile_program()). Never renumber these
// positionally; migraphx snaps an out-of-range value to the nearest mode
// instead of rejecting it, so a mismatch fails silently.
enum class ComputeMode : std::int8_t {
    Eager = 0,
    Balanced = 50,
    Maximum = 100
};

// Parses a compute-mode spelling, case-insensitive: eager|0, balanced|50,
// maximum|100. Returns nullopt on anything else. Shared by the provider-option
// parser and the ORT_MIGRAPHX_COMPUTE_MODE environment variable.
std::optional<ComputeMode> ParseComputeMode(std::string_view value);

struct ProviderInfo {
    int device_id{};
    bool enable_fp16{};
    bool enable_bf16{};
    bool enable_fp8{};
    bool enable_int8{};
    std::string int8_calibration_table_name{};
    bool int8_use_native_calibration_table{};
    bool exhaustive_tune{};
    bool dump_subgraphs{};
    std::string mlss_use_specific_ops{};
    std::string model_arch{};
    // Explicit: value-initialization selects Eager (enumerator value 0), which
    // would silently downgrade every session that does not set the option.
    ComputeMode compute_mode{ComputeMode::Balanced};
    fs::path cache_dir{};
    bool disable_caching{};
    bool force_recompile{};
    bool context_embed_mode{};
    bool context_enable{};
    fs::path external_initializers_file_name{};
    fs::path context_file_path{};
    std::string context_node_name_prefix{};
    bool hip_graph_enable{};
    std::size_t max_dynamic_batch{};
    std::string compile_batches{};
    bool precompile_at_load{};
    bool coalesce_io{};
    bool cpu_control_flow{};
    bool static_pad_seq{};
    std::size_t static_pad_seq_len{};
    std::string static_pad_inputs{};
    std::string static_pad_outputs{};

    ProviderInfo() = default;

    explicit ProviderInfo(const ProviderOptions& provider_options);
};

}  // namespace mgx_ep
