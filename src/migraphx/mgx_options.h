// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>
using namespace std::literals::string_view_literals;  // NOLINT(build/namespaces_literals)

namespace mgx_ep::provider_option {

constexpr auto kDeviceId = "device_id"sv;
constexpr auto kFp16Enable = "fp16_enable"sv;
constexpr auto kBf16Enable = "bf16_enable"sv;
constexpr auto kFp8Enable = "fp8_enable"sv;
constexpr auto kInt8Enable = "int8_enable"sv;
constexpr auto kInt8CalibTable = "int8_calibration_table_name"sv;
constexpr auto kInt8UseNativeCalibTable = "int8_use_native_calibration_table"sv;
constexpr auto kDisableCaching = "disable_caching"sv;
constexpr auto kForceRecompile = "force_recompile"sv;
constexpr auto kComputeMode = "compute_mode"sv;
constexpr auto kCacheDir = "cache_dir"sv;
constexpr auto kExhaustiveTune = "exhaustive_tune"sv;
constexpr auto kDumpSubgraphs = "dump_subgraphs"sv;
constexpr auto kMlssUseSpecificOps = "mlss_use_specific_ops"sv;

}  // namespace mgx_ep::provider_option
