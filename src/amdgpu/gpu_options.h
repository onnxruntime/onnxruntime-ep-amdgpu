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
constexpr auto kExhaustiveTune = "exhaustive_tune"sv;
constexpr auto kProfile = "profile"sv;
constexpr auto kCacheDir = "cache_dir"sv;
constexpr auto kMlssUseSpecificOps = "mlss_use_specific_ops"sv;
constexpr auto kModelArch = "model_arch"sv;
constexpr auto kModelFw = "model_fw"sv;  // caller framework hint, e.g. "webnn" (routes to DirectML)
// Static sequence-length padding (forwarded verbatim to the migraphx backend, which
// implements the pad/slice). The umbrella only needs to recognize + relay these.
constexpr auto kStaticPadSeq = "static_pad_seq"sv;
constexpr auto kStaticPadSeqLen = "static_pad_seq_len"sv;
constexpr auto kStaticPadInputs = "static_pad_inputs"sv;
constexpr auto kStaticPadOutputs = "static_pad_outputs"sv;
}  // provider_option

}  // gpu_ep
