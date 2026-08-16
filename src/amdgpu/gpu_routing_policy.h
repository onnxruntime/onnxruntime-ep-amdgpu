// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Backend-selection policy for the AMD GPU umbrella EP, Auto profile.
// This header holds the pure decision logic (normalization, hashing helpers, and select_backend()). It
// has no I/O and no ORT/HIP dependency beyond the Profile enum, so the policy can be read, extended, and
// unit-tested in isolation. The routing DATA (LLM family set + (arch, model_arch) override table) lives
// in gpu_routing_tables.h; the EP-side glue (ASIC query via HIP, provider options, trace) is in gpu_ep.cc.

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "gpu_info.h"            // Profile (used in select_backend's signature)
#include "gpu_routing_tables.h"  // fnv1a, kNoModelArch, kLlmModelArch, kArchModelBackend, arch_model_backend

namespace gpu_ep {

// Normalize a model_arch string (lowercase + trim) before hashing, so "Llama", " llama" all match.
// The routing tables in gpu_routing_tables.h hash the same normalized literals via fnv1a(), so
// runtime and compile-time hashing stay consistent by construction.
inline std::string normalize_model_arch(std::string_view value) {
    std::string s{value};
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const auto not_space = [](unsigned char c) { return std::isspace(c) == 0; };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

// Prefix test on a gcnArchName. Matches the rocMLIR/MIGraphX idiom (AmdArchDb.cpp, device_name.cpp),
// which classify archs by gfx-name prefix rather than a parsed integer. Prefix matching also handles
// gcnArchName feature suffixes natively (e.g. "gfx1201:xnack-" starts_with "gfx12").
constexpr bool starts_with(std::string_view s, std::string_view prefix) {
    return s.substr(0, prefix.size()) == prefix;
}

// fnv1a of the normalized model_arch, or kNoModelArch when the caller supplied none.
inline std::uint64_t model_arch_hash(const std::optional<std::string>& model_arch) {
    return model_arch.has_value() ? fnv1a(normalize_model_arch(model_arch.value())) : kNoModelArch;
}

// True if the caller framework hint (model_fw provider option) names WebNN. Case-insensitive.
inline bool is_webnn(const std::optional<std::string>& model_fw) {
    return model_fw.has_value() && normalize_model_arch(model_fw.value()) == "webnn";
}

// Pure backend-selection policy. No I/O — unit-testable. Takes the raw gcnArchName ("gfx1201",
// "gfx1100:xnack-"). Only the Auto profile is heuristic-driven; every other profile (explicit backend,
// or Optimized) is honored/dispatched as-is. In Auto mode, in priority order:
//   1. WebNN caller                   -> DirectML (browser/WebNN compatibility carve-out, all ASICs)
//   2. kArchModelBackend (arch prefix + model_arch) override, if any row matches
//   3. HIP-enabled gfx1151 + LLM      -> Hip
//   4. Medusa gfx117x                 -> DirectML
//   5. gfx11 and newer                -> MIGraphX
//   6. everything below gfx11 (gfx9/gfx10) -> DirectML
inline Profile select_backend(std::string_view gfx, std::uint64_t arch_model_hash, bool is_webnn,
                              Profile profile) {
    if (profile != Profile::Auto) {
        return profile;  // explicit profile (and Optimized) honored/dispatched as-is
    }
    // 1. WebNN compatibility carve-out: always DirectML, regardless of ASIC (browser path).
    if (is_webnn) return Profile::DirectX;
    // 2. Exact (arch prefix, model_arch) override.
    for (const auto& row : kArchModelBackend) {
        if (arch_model_hash != kNoModelArch && row.model_arch_hash == arch_model_hash &&
            starts_with(gfx, row.arch_prefix)) {
            return row.backend;
        }
    }
#ifdef USE_HIP
    // 3. Strix Halo LLMs use the HIP backend when it is included in this build.
    if (starts_with(gfx, "gfx1151") && arch_model_hash == fnv1a("llm")) return Profile::Hip;
#endif
    // 4-6. Prefix buckets (order matters: gfx117x before the general gfx11x).
    if (starts_with(gfx, "gfx117")) return Profile::DirectX;  // Medusa MDS1/MDS2 (temporary)
    if (starts_with(gfx, "gfx11")) return Profile::MIGraphX;   // RDNA3 / RDNA3.5
    if (starts_with(gfx, "gfx12")) return Profile::MIGraphX;   // RDNA4
    return Profile::DirectX;                                   // pre-gfx11 (gfx9/gfx10)
}

}  // namespace gpu_ep
