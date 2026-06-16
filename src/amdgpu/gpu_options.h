// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>
using namespace std::literals::string_view_literals;  // NOLINT(build/namespaces_literals)

namespace gpu_ep {

constexpr auto kDirectMLBackend = "directml";
constexpr auto kMIGraphXBackend = "migraphx";

namespace provider_option {
constexpr auto kDeviceId = "device_id"sv;
constexpr auto kDisableCaching = "disable_caching"sv;
constexpr auto kForceRecompile = "force_recompile"sv;
constexpr auto kProfile = "profile"sv;
constexpr auto kCacheDir = "cache_dir"sv;
constexpr auto kMlssUseSpecificOps = "mlss_use_specific_ops"sv;
}  // provider_option

}  // gpu_ep
