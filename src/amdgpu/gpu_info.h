// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include "common/plugin_ep_utils.h"

namespace fs = std::filesystem;

namespace gpu_ep {

enum class Profile {
    Auto,
    Eager,
    Optimized,
    MIGraphX,
    DirectX,
    Hip
};

struct ProviderInfo {
    Profile profile{Profile::Auto};
    std::optional<int> device_id{};
    std::optional<bool> disable_caching{};
    std::optional<bool> force_recompile{};
    std::optional<bool> exhaustive_tune{};
    std::optional<fs::path> cache_dir{};
    std::optional<std::string> mlss_use_specific_ops{};
    std::optional<bool> cpu_control_flow{};
    std::optional<std::string> model_arch{};
    std::optional<std::string> model_fw{};  // caller framework (e.g. "webnn"); "webnn" -> DirectML
    std::optional<std::string> hip_graph_enable{};
    // Static seq-padding: recognized here only so the umbrella can relay them to the
    // migraphx backend (which implements the pad/slice). Stored as strings verbatim.
    std::optional<std::string> static_pad_seq{};
    std::optional<std::string> static_pad_seq_len{};
    std::optional<std::string> static_pad_inputs{};
    std::optional<std::string> static_pad_outputs{};

    ProviderInfo() = default;

    explicit ProviderInfo(const ProviderOptions& provider_options);
};

}  // namespace gpu_ep
