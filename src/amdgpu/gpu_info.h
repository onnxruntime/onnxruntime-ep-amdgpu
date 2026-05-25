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
    Optimized
};

struct ProviderInfo {
    Profile profile{Profile::Auto};
    std::optional<int> device_id{};
    std::optional<bool> disable_caching{};
    std::optional<bool> force_recompile{};
    std::optional<fs::path> cache_dir{};

    ProviderInfo() = default;

    explicit ProviderInfo(const ProviderOptions& provider_options);
};

}  // namespace gpu_ep
