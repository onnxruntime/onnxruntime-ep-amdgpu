// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>
#include <utility>
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
constexpr auto kHipGraphEnable = "hip_graph_enable"sv;
constexpr auto kMaxDynamicBatch = "max_dynamic_batch"sv;
constexpr auto kCompileBatches = "compile_batches"sv;
constexpr auto kPrecompileAtLoad = "precompile_at_load"sv;
constexpr auto kCoalesceIO = "coalesce_io"sv;
constexpr auto kMlssUseSpecificOps = "mlss_use_specific_ops"sv;
constexpr auto kCpuControlFlow = "cpu_control_flow"sv;
constexpr auto kModelArch = "model_arch"sv;

// External application-owned compute stream. Mirrors the classic built-in
// MIGraphXExecutionProvider: when set, the EP adopts this hipStream_t instead of
// creating its own, so the application's pipeline and the EP share one stream
// (no cross-stream race, no forced device-wide drain). kUserComputeStream carries
// the stream handle as a decimal pointer address; kHasUserComputeStream is the
// matching boolean flag. Providing a non-null kUserComputeStream implies the flag.
constexpr auto kUserComputeStream = "user_compute_stream"sv;
constexpr auto kHasUserComputeStream = "has_user_compute_stream"sv;

// Static sequence-length padding.  When enabled, the EP pads the token axis of the
// named inputs up to static_pad_seq_len (so a varying prefill length compiles the
// program only once) and slices the named outputs back down to the real length.
// Set by OGA for MIGraphX LLM sessions; mirrors the pad OGA used to do itself.
constexpr auto kStaticPadSeq = "static_pad_seq"sv;             // "1" = enable
constexpr auto kStaticPadSeqLen = "static_pad_seq_len"sv;      // target token-axis length
constexpr auto kStaticPadInputs = "static_pad_inputs"sv;       // "input_ids:1,position_ids:1"
constexpr auto kStaticPadOutputs = "static_pad_outputs"sv;     // "logits:1"

// Legacy aliases: the classic (built-in) MIGraphXExecutionProvider prefixed
// its option names with "migraphx_", e.g. "migraphx_fp16_enable" rather than
// this plugin EP's "fp16_enable" (the "ep.<name>." prefix already added by
// SessionOptionsAppendExecutionProvider_V2/add_provider_for_devices() makes
// repeating "migraphx_" in the key itself redundant). These aliases let
// scripts/tools written against the classic EP's option names keep working
// unmodified against this plugin EP. Only options with a direct plugin
// equivalent are aliased; options with no equivalent here (e.g.
// migraphx_mem_limit, migraphx_arena_extend_strategy, migraphx_external_*,
// migraphx_model_cache_dir) are intentionally not aliased.
constexpr std::pair<std::string_view, std::string_view> kLegacyOptionAliases[] = {
    {"migraphx_fp16_enable", kFp16Enable},
    {"migraphx_bf16_enable", kBf16Enable},
    {"migraphx_fp8_enable", kFp8Enable},
    {"migraphx_int8_enable", kInt8Enable},
    {"migraphx_int8_calibration_table_name", kInt8CalibTable},
    {"migraphx_int8_use_native_calibration_table", kInt8UseNativeCalibTable},
    {"migraphx_exhaustive_tune", kExhaustiveTune},
    {"migraphx_user_compute_stream", kUserComputeStream},
    {"migraphx_has_user_compute_stream", kHasUserComputeStream},
};

}  // namespace mgx_ep::provider_option
