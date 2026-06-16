// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mgx_options.h"
#include "mgx_info.h"

#include "onnxruntime_session_options_config_keys.h"

#include "common/parse_string.h"
#include "common/provider_options_utils.h"

#include "hip/utils.h"

namespace mgx_ep {

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
                provider_option::kComputeMode,
                [this](const std::string_view value) -> Ort::Status {
                    std::string lower{value};
                    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                    if (lower == "eager" || value == "0") {
                        compute_mode = ComputeMode::Eager;
                    } else if (lower == "balanced" || value == "50") {
                        compute_mode = ComputeMode::Balanced;
                    } else if (lower == "maximum" || value == "100") {
                        compute_mode = ComputeMode::Maximum;
                    } else {
                        return Ort::Status{("unknown compute mode: " +
                            std::string{value}).c_str(), ORT_FAIL};
                    }
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
            .AddAssignmentToReference(provider_option::kCoalesceIO, coalesce_io)
            .Parse(provider_options));
}

}  // namespace mgx_ep
