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
    DirectML,
    Llm
};

struct ProviderInfo {
    Profile profile{Profile::Auto};
    std::optional<int> device_id{};
    std::optional<bool> disable_caching{};
    std::optional<bool> force_recompile{};
    std::optional<bool> exhaustive_tune{};
    std::optional<fs::path> cache_dir{};
    std::optional<std::string> mlss_use_specific_ops{};

    ProviderInfo() = default;

    explicit ProviderInfo(const ProviderOptions& provider_options);
};

}  // namespace gpu_ep
