// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mgx_options.h"
#include "mgx_info.h"

#include <algorithm>
#include <cctype>

#include "onnxruntime_session_options_config_keys.h"

#include "common/parse_string.h"
#include "common/provider_options_utils.h"

#include "hip/utils.h"

namespace mgx_ep {

namespace {

// Rewrites any legacy "migraphx_*"-prefixed option names (see
// provider_option::kLegacyOptionAliases) to this plugin EP's canonical
// (unprefixed) names, so scripts/tools written against the classic built-in
// MIGraphXExecutionProvider keep working unmodified against this plugin EP.
// If both an alias and its canonical name are present, the canonical name's
// value wins.
ProviderOptions ApplyLegacyOptionAliases(const ProviderOptions& options) {
    ProviderOptions normalized = options;
    for (const auto& [alias, canonical_name] : provider_option::kLegacyOptionAliases) {
        const auto alias_it = normalized.find(std::string{alias});
        if (alias_it != normalized.end()) {
            normalized.emplace(std::string{canonical_name}, alias_it->second);
            normalized.erase(alias_it);
        }
    }
    return normalized;
}

}  // namespace

std::optional<ComputeMode> ParseComputeMode(const std::string_view value) {
    std::string lower{value};
    std::transform(lower.begin(), lower.end(), lower.begin(),
        [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower == "eager" || lower == "0") {
        return ComputeMode::Eager;
    }
    if (lower == "balanced" || lower == "50") {
        return ComputeMode::Balanced;
    }
    if (lower == "maximum" || lower == "100") {
        return ComputeMode::Maximum;
    }
    return std::nullopt;
}

ProviderInfo::ProviderInfo(const ProviderOptions& provider_options) {
    THROW_IF_ERROR(
        ProviderOptionsParser{}
            .AddValueParser(
                provider_option::kDeviceId,
                [this](const std::string_view value) -> Ort::Status {
                    RETURN_IF_ERROR(ParseStringWithClassicLocale(value, device_id));
                    int num_devices{};
                    RETURN_IF_ERROR(HIP_CALL(hipGetDeviceCount(&num_devices)));
                    RETURN_IF_NOT(0 <= device_id && device_id < num_devices,
                        "Invalid device ID: ", device_id,
                        ", must be between 0 (inclusive) and ", num_devices, " (exclusive).");
                    return STATUS_OK;
                })
            .AddValueParser(
                provider_option::kCacheDir,
                [this](const std::string_view value) -> Ort::Status {
                    cache_dir = value;
                    return STATUS_OK;
                })
            .AddValueParser(
                provider_option::kCompileBatches,
                [this](const std::string_view value) -> Ort::Status {
                    compile_batches = value;
                    return STATUS_OK;
                })
            .AddValueParser(
                provider_option::kStaticPadInputs,
                [this](const std::string_view value) -> Ort::Status {
                    static_pad_inputs = value;
                    return STATUS_OK;
                })
            .AddValueParser(
                provider_option::kStaticPadOutputs,
                [this](const std::string_view value) -> Ort::Status {
                    static_pad_outputs = value;
                    return STATUS_OK;
                })
            .AddValueParser(
                provider_option::kComputeMode,
                [this](const std::string_view value) -> Ort::Status {
                    // This is the only layer that can reject a bad value: migraphx
                    // clamps an out-of-range compile mode instead of failing.
                    const auto parsed{ParseComputeMode(value)};
                    if (!parsed.has_value()) {
                        return Ort::Status{("unknown compute mode: " +
                            std::string{value}).c_str(), ORT_FAIL};
                    }
                    compute_mode = *parsed;
                    return STATUS_OK;
                })
            .AddAssignmentToReference(kOrtSessionOptionEpContextFilePath, context_file_path)
            .AddAssignmentToReference(kOrtSessionOptionEpContextEmbedMode, context_embed_mode)
            .AddAssignmentToReference(kOrtSessionOptionEpContextEnable, context_enable)
            .AddAssignmentToReference(kOrtSessionOptionEpContextNodeNamePrefix, context_node_name_prefix)
            .AddAssignmentToReference(kOrtSessionOptionsEpContextModelExternalInitializersFileName, external_initializers_file_name)
            .AddAssignmentToReference(provider_option::kFp16Enable, enable_fp16)
            .AddAssignmentToReference(provider_option::kBf16Enable, enable_bf16)
            .AddAssignmentToReference(provider_option::kFp8Enable, enable_fp8)
            .AddAssignmentToReference(provider_option::kInt8Enable, enable_int8)
            .AddAssignmentToReference(provider_option::kInt8UseNativeCalibTable, int8_use_native_calibration_table)
            .AddAssignmentToReference(provider_option::kInt8CalibTable, int8_calibration_table_name)
            .AddAssignmentToReference(provider_option::kExhaustiveTune, exhaustive_tune)
            .AddAssignmentToReference(provider_option::kDisableCaching, disable_caching)
            .AddAssignmentToReference(provider_option::kDumpSubgraphs, dump_subgraphs)
            .AddAssignmentToReference(provider_option::kForceRecompile, force_recompile)
            .AddAssignmentToReference(provider_option::kHipGraphEnable, hip_graph_enable)
            .AddAssignmentToReference(provider_option::kMaxDynamicBatch, max_dynamic_batch)
            .AddAssignmentToReference(provider_option::kStaticPadSeq, static_pad_seq)
            .AddAssignmentToReference(provider_option::kStaticPadSeqLen, static_pad_seq_len)
            .AddAssignmentToReference(provider_option::kCoalesceIO, coalesce_io)
            .AddAssignmentToReference(provider_option::kMlssUseSpecificOps, mlss_use_specific_ops)
            .AddAssignmentToReference(provider_option::kCpuControlFlow, cpu_control_flow)
            .AddAssignmentToReference(provider_option::kModelArch, model_arch)
            .Parse(ApplyLegacyOptionAliases(provider_options)));
}

}  // namespace mgx_ep
