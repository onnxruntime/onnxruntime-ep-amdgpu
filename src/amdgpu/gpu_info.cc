// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gpu_options.h"
#include "gpu_info.h"

#include "common/parse_string.h"
#include "common/provider_options_utils.h"

namespace gpu_ep {

ProviderInfo::ProviderInfo(const ProviderOptions& provider_options) {
    THROW_IF_ERROR(
        ProviderOptionsParser{}
            .AddValueParser(
                provider_option::kProfile,
                [this](const std::string_view value) -> Ort::Status {
                    std::string lower{value};
                    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                    if (lower == "auto" || value == "0" || value.empty()) {
                        profile = Profile::Auto;
                    } else if (lower == "eager" || value == "1") {
                        profile = Profile::Eager;
                    } else if (lower == "optimize" || value == "2") {
                        profile = Profile::Optimized;
                    } else {
                        return MAKE_STATUS(ORT_FAIL, "unknown profile: '", value, "'");
                    }
                    return STATUS_OK;
                })
            .AddAssignmentToReference(provider_option::kDeviceId, device_id)
            .AddAssignmentToReference(provider_option::kDisableCaching, disable_caching)
            .AddAssignmentToReference(provider_option::kCacheDir, cache_dir)
            .AddAssignmentToReference(provider_option::kForceRecompile, force_recompile)
            .Parse(provider_options));
}

}  // namespace mgx_ep
