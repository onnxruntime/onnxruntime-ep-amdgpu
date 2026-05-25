// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/plugin_ep_utils.h"
#include "mgx_utils.h"

namespace mgx_ep {

enum class ComputeMode {
    Eager,
    Balanced,
    Maximum
};

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
    ComputeMode compute_mode{};
    fs::path cache_dir{};
    bool disable_caching{};
    bool force_recompile{};
    bool context_embed_mode{};
    bool context_enable{};
    fs::path external_initializers_file_name{};
    fs::path context_file_path{};
    std::string context_node_name_prefix{};

    ProviderInfo() = default;

    explicit ProviderInfo(const ProviderOptions& provider_options);
};

}  // namespace mgx_ep
