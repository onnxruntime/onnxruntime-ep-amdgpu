// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <migraphx/migraphx.hpp>

#include "mgx_ep.h"

namespace mgx_ep {

// Parse the output index from a MIGraphX output parameter name ("#output_0").
// Returns -1 if the name is not an output parameter.
int ComputeOutputIndex(std::string_view name);

// ── Scratch buffer management ────────────────────────────────────────────────
// MIGraphX programs expose a "scratch" parameter.  If the EP doesn't bind it,
// MIGraphX uses its internal arena whose contents persist across runs and bleed
// into captured hipGraph kernels that read scratch before writing.  Owning the
// buffer lets us zero it to a deterministic baseline before capture/replay.

struct ScratchBindInfo {
    void* ptr{nullptr};
    migraphx::shape shape{};
};

// Ensure an EP-owned scratch buffer exists (and is large enough) for shape_hash.
// Freshly allocated buffers are zeroed; existing buffers are left as-is (callers
// zero them via ZeroScratchFor immediately before use).  Returns std::nullopt
// when the program has no "scratch" parameter.
std::optional<ScratchBindInfo> GetOrAllocScratch(ComputeState& cs,
    const migraphx::program_parameter_shapes& param_shapes,
    const std::string& shape_hash, hipStream_t stream);

// Zero an already-allocated scratch buffer (no-op if none exists for shape_hash).
void ZeroScratchFor(ComputeState& cs, const std::string& shape_hash, hipStream_t stream);

// ── Staging buffer substrate ─────────────────────────────────────────────────
// EP-owned device buffers that give pointer stability for hipGraph capture.
// Allocated once (plain hipMalloc) and reused for the compute state's lifetime
// to avoid the stream-ordered-pool growth that hipMallocAsync incurs.

// Allocate staging buffers (one per program input/output parameter) sized to the
// program's parameter shapes.  No-op if already allocated.
void AllocateStaging(ComputeState& cs,
    const migraphx::program_parameter_shapes& param_shapes, hipStream_t stream);

// Copy ORT input tensors into their staging buffers.
void CopyInputsToStaging(ComputeState& cs,
    const migraphx::program_parameter_shapes& param_shapes,
    const Ort::KernelContext& ctx, hipStream_t stream);

// Result of binding staging buffers as program parameters.
struct StagingBindResult {
    migraphx::program_parameters params{};
    std::vector<std::size_t> prog_output_indices{};   // ORT output index per bound output
    std::vector<std::string> bound_output_names{};    // staging key per bound output
};

// Bind staging input/output buffers and the EP-owned scratch buffer as program
// parameters for the given compiled shape.
StagingBindResult BindStagingParams(ComputeState& cs,
    const migraphx::program_parameter_shapes& param_shapes,
    const std::string& shape_hash, hipStream_t stream);

// Copy staging output buffers back into the ORT output tensors.
void CopyStagingOutputsToOrt(ComputeState& cs, const StagingBindResult& bind,
    const Ort::KernelContext& ctx, hipStream_t stream);

}  // namespace mgx_ep
