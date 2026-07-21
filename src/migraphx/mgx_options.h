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
constexpr auto kHipGraphEnable = "hip_graph_enable"sv;
constexpr auto kMaxDynamicBatch = "max_dynamic_batch"sv;
constexpr auto kCompileBatches = "compile_batches"sv;
constexpr auto kCoalesceIO = "coalesce_io"sv;
constexpr auto kMlssUseSpecificOps = "mlss_use_specific_ops"sv;
constexpr auto kModelArch = "model_arch"sv;

// Static sequence-length padding.  When enabled, the EP pads the token axis of the
// named inputs up to static_pad_seq_len (so a varying prefill length compiles the
// program only once) and slices the named outputs back down to the real length.
// Set by OGA for MIGraphX LLM sessions; mirrors the pad OGA used to do itself.
constexpr auto kStaticPadSeq = "static_pad_seq"sv;             // "1" = enable
constexpr auto kStaticPadSeqLen = "static_pad_seq_len"sv;      // target token-axis length
constexpr auto kStaticPadInputs = "static_pad_inputs"sv;       // "input_ids:1,position_ids:1"
constexpr auto kStaticPadOutputs = "static_pad_outputs"sv;     // "logits:1"

}  // namespace mgx_ep::provider_option
