// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Routing DATA for the AMD GPU umbrella EP, Auto profile — the two tables maintainers edit:
//   kLlmModelArch     : model_arch families treated as LLMs (shifts the MIGraphX cutoff to gfx11.0)
//   kArchModelBackend : per-(gfx prefix, model_arch) backend overrides (grows large over time)
// Kept in its own header so the growable tables live in one small, self-contained file. The pure
// decision logic that consumes them is in gpu_routing_policy.h; the EP-side glue is in gpu_ep.cc.
//
// TO ADD AN LLM FAMILY:    add one `fnv1a("normalized_name")` entry to kLlmModelArch (and bump size).
// TO PIN AN (arch, model): add `{"gfxNNNN", fnv1a("normalized_name"), Profile::X}` to kArchModelBackend.

#include <array>
#include <cstdint>
#include <string_view>

#include "gpu_info.h"  // Profile

namespace gpu_ep {

// FNV-1a (64-bit). model_arch names are stored as hash constants so the plaintext names appear in
// neither the source nor the shipped binary. Defined here because the tables below hash at compile time.
constexpr std::uint64_t fnv1a(std::string_view s) {
    std::uint64_t h{0xcbf29ce484222325ULL};
    for (const char c : s) {
        h ^= static_cast<std::uint8_t>(c);
        h *= 0x100000001b3ULL;
    }
    return h;
}

// A hash of 0 means "no model_arch supplied" (see model_arch_hash / select_backend). Because model names
// are only stored/compared as hashes, a family that hashed to 0 would be silently unmatchable; the
// static_asserts below guard against it. (A distinct-name hash collision is negligible at 64-bit but
// would be hard to debug, since the plaintext names are not in the binary.)
constexpr std::uint64_t kNoModelArch = 0;

// ============================================================================================
//  ROUTING TABLES — the two things maintainers edit. Machinery (logic) lives in gpu_routing_policy.h.
// ============================================================================================

// (1) Known LLM model_arch families (hashed). Currently the Windows ML P0 LLM set that is tested and
//     supported. On gfx11.0+ these route to MIGraphX (below gfx11.5 they flip the default).
//     "llm" is a generic forward-compat marker: a caller (e.g. a newer OGA whose specific arch this
//     umbrella build doesn't yet recognize) can send model_arch="llm" to force the generic LLM route.
//     TO ADD AN LLM FAMILY: add one `fnv1a("normalized_name")` entry (and bump the array size).
constexpr std::array<std::uint64_t, 5> kLlmModelArch{{
    fnv1a("llama"), fnv1a("qwen2"), fnv1a("phi3"), fnv1a("mistral"),
    fnv1a("llm"),
}};

// (2) Per-(arch, model) backend override. Highest priority in Auto mode: a matching row wins over the
//     arch defaults. This is the hook for specific GPU + model combinations (e.g. a customer model that
//     must pin to a backend on a given ASIC). Empty today; expected to grow.
//     TO ADD AN OVERRIDE: add a row `{"gfxNNNN", fnv1a("normalized_name"), Profile::X}` (and bump size).
struct arch_model_backend {
    std::string_view arch_prefix;   // gfx-name prefix, e.g. "gfx1201"
    std::uint64_t model_arch_hash;  // fnv1a of normalized model_arch
    Profile backend;
};
constexpr std::array<arch_model_backend, 0> kArchModelBackend{};

static_assert([] {
    for (const auto h : kLlmModelArch) if (h == kNoModelArch) return false;
    return true;
}(), "an LLM family hashed to the no-model sentinel (0); rename or change the sentinel");
static_assert([] {
    for (const auto& r : kArchModelBackend) if (r.model_arch_hash == kNoModelArch) return false;
    return true;
}(), "an override row uses the no-model sentinel (0); it could never match");

}  // namespace gpu_ep
