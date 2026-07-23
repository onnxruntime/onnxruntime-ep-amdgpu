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

#include "gpu_routing_tables.h"  // fnv1a, kNoModelArch, kLlmModelArch, kArchModelBackend, arch_model_backend

namespace gpu_ep {

// Normalize a model_arch string (lowercase + trim) before hashing, so "Llama", " llama" all match.
// The table constants above are the same fnv1a() applied to normalized literals, so runtime and
// compile-time hashing stay consistent by construction.
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
//   3. Medusa gfx117x                 -> DirectML
//   4. gfx11.5+ (gfx115x Strix)       -> MIGraphX
//   5. gfx11.x pre-11.5 (gfx110x/1103): LLM -> MIGraphX, else DirectML  (the LLM cutoff at gfx11.0)
//   6. gfx12x (RDNA4)                 -> MIGraphX
//   7. everything below gfx11 (gfx9/gfx10, incl. LLM) -> DirectML
inline Profile select_backend(std::string_view gfx, std::uint64_t arch_model_hash, bool is_webnn,
                              Profile profile) {
    if (profile != Profile::Auto) {
        return profile;  // explicit profile (and Optimized) honored/dispatched as-is
    }
    // 1. WebNN compatibility carve-out: always DirectML, regardless of ASIC (browser path).
    if (is_webnn) return Profile::DirectML;
    // 2. Exact (arch prefix, model_arch) override.
    for (const auto& row : kArchModelBackend) {
        if (arch_model_hash != kNoModelArch && row.model_arch_hash == arch_model_hash &&
            starts_with(gfx, row.arch_prefix)) {
            return row.backend;
        }
    }
    const bool is_llm = arch_model_hash != kNoModelArch &&
        std::find(kLlmModelArch.begin(), kLlmModelArch.end(), arch_model_hash) != kLlmModelArch.end();
    // 2-6. Prefix buckets (order matters: gfx117x and gfx115x before the general gfx11x).
    if (starts_with(gfx, "gfx117")) return Profile::DirectML;  // Medusa MDS1/MDS2 (temporary)
    // TODO(routing): Strix Halo LLM (gfx1150/1151) targets HipEP starting next release; it folds to
    // MIGraphX today because HipEP is not built. Split out an is_llm -> Hip branch once it ships.
    if (starts_with(gfx, "gfx115")) return Profile::MIGraphX;  // Strix (gfx11.5)
    if (starts_with(gfx, "gfx11"))                             // pre-gfx11.5 RDNA3 (gfx110x/1103)
        return is_llm ? Profile::MIGraphX : Profile::DirectML;
    if (starts_with(gfx, "gfx12")) return Profile::MIGraphX;   // RDNA4
    return Profile::DirectML;                                  // pre-gfx11 (gfx9/gfx10), incl. LLM
}

}  // namespace gpu_ep
