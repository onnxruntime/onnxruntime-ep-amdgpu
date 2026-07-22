// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Backend-selection policy for the AMD GPU umbrella EP, Auto profile.
// This header holds the routing DATA (arch cutoffs, LLM family set, the (arch, model_arch) override
// table) and the pure decision function select_backend(). It has no I/O and no ORT/HIP dependency
// beyond the Profile enum, so the policy can be read, extended, and unit-tested in isolation. The
// EP-side glue (querying the ASIC via HIP, reading provider options, logging) lives in gpu_ep.cc.
//
// To pin a specific GPU arch + model combination to a backend, add a row to kArchModelBackend.
// To treat a new model architecture as an LLM (shifts the MIGraphX cutoff down to gfx11.0), add its
// normalized name to kLlmModelArch.

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "gpu_info.h"  // Profile

namespace gpu_ep {

// FNV-1a (64-bit). Used to store model_arch names as hash constants so the plaintext names appear in
// neither the source nor the shipped binary. Defined first because the tables below hash at compile time.
constexpr std::uint64_t fnv1a(std::string_view s) {
    std::uint64_t h{0xcbf29ce484222325ULL};
    for (const char c : s) {
        h ^= static_cast<std::uint8_t>(c);
        h *= 0x100000001b3ULL;
    }
    return h;
}

// A hash of 0 means "no model_arch supplied" (see select_backend). Because model names are only
// stored/compared as hashes, a family that hashed to 0 would be silently unmatchable; the static_asserts
// below guard against it. (A distinct-name hash collision is negligible at 64-bit but would be hard to
// debug, since the plaintext names are not in the binary.)
constexpr std::uint64_t kNoModelArch = 0;

// ============================================================================================
//  ROUTING TABLES — the two things maintainers edit. Everything below this block is machinery.
// ============================================================================================

// (1) Known LLM model_arch families (hashed). Currently the Windows ML P0 LLM set that is tested and
//     supported. On gfx11.0+ these route to MIGraphX (below gfx11.5 they flip the default).
//     TO ADD AN LLM FAMILY: add one `fnv1a("normalized_name")` entry (and bump the array size).
constexpr std::array<std::uint64_t, 4> kLlmModelArch{{
    fnv1a("llama"), fnv1a("qwen2"), fnv1a("phi3"), fnv1a("mistral"),
}};

// (2) Per-(arch, model) backend override. Highest priority in Auto mode: a matching row wins over the
//     arch defaults. This is the hook for specific GPU + model combinations (e.g. a customer model that
//     must pin to a backend on a given ASIC). Empty today.
//     TO ADD AN OVERRIDE: add a row `{"gfxNNNN", fnv1a("normalized_name"), Profile::X}` (and bump size).
struct arch_model_backend {
    std::string_view arch_prefix;   // gfx-name prefix, e.g. "gfx1201"
    std::uint64_t model_arch_hash;  // fnv1a of normalized model_arch
    Profile backend;
};
constexpr std::array<arch_model_backend, 0> kArchModelBackend{};

// ============================================================================================
//  Machinery below — not normally edited.
// ============================================================================================

static_assert([] {
    for (const auto h : kLlmModelArch) if (h == kNoModelArch) return false;
    return true;
}(), "an LLM family hashed to the no-model sentinel (0); rename or change the sentinel");
static_assert([] {
    for (const auto& r : kArchModelBackend) if (r.model_arch_hash == kNoModelArch) return false;
    return true;
}(), "an override row uses the no-model sentinel (0); it could never match");

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

// Pure backend-selection policy. No I/O — unit-testable. Takes the raw gcnArchName ("gfx1201",
// "gfx1100:xnack-"). Only the Auto profile is heuristic-driven; every other profile (explicit backend,
// or Optimized) is honored/dispatched as-is. In Auto mode, in priority order:
//   1. kArchModelBackend (arch prefix + model_arch) override, if any row matches
//   2. Medusa gfx117x                 -> DirectML
//   3. gfx11.5+ (gfx115x Strix)       -> MIGraphX
//   4. gfx11.x pre-11.5 (gfx110x/1103): LLM -> MIGraphX, else DirectML  (the LLM cutoff at gfx11.0)
//   5. gfx12x (RDNA4)                 -> MIGraphX
//   6. everything below gfx11 (gfx9/gfx10, incl. LLM) -> DirectML
inline Profile select_backend(std::string_view gfx, std::uint64_t arch_model_hash, Profile profile) {
    if (profile != Profile::Auto) {
        return profile;  // explicit profile (and Optimized) honored/dispatched as-is
    }
    // 1. Highest priority: exact (arch prefix, model_arch) override.
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
    if (starts_with(gfx, "gfx115")) return Profile::MIGraphX;  // Strix (gfx11.5)
    if (starts_with(gfx, "gfx11"))                             // pre-gfx11.5 RDNA3 (gfx110x/1103)
        return is_llm ? Profile::MIGraphX : Profile::DirectML;
    if (starts_with(gfx, "gfx12")) return Profile::MIGraphX;   // RDNA4
    return Profile::DirectML;                                  // pre-gfx11 (gfx9/gfx10), incl. LLM
}

}  // namespace gpu_ep
