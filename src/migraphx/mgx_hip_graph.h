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

// Dynamic-batch bucketing context for a single Compute call.  When active, the
// program is compiled/replayed at target_batch (a compiled bucket >= the request)
// and inputs/outputs are padded up / sliced down on the batch axis (axis 0).
struct DynamicBatchContext {
    bool active{false};
    std::size_t requested_batch{0};
    std::size_t target_batch{0};
};

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

// Allocate staging buffers (one per program input/output parameter).  Batched
// buffers (batch on axis 0) are sized to max_dynamic_batch so a single set of
// buffers serves every compiled bucket; all others are sized exactly.  No-op if
// already allocated.
void AllocateStaging(ComputeState& cs,
    const migraphx::program_parameter_shapes& param_shapes, hipStream_t stream,
    const DynamicBatchContext& dyn);

// Copy ORT input tensors into their staging buffers, padding batched inputs up to
// the target bucket batch when dynamic batching is active.
void CopyInputsToStaging(ComputeState& cs,
    const migraphx::program_parameter_shapes& param_shapes,
    const Ort::KernelContext& ctx, hipStream_t stream,
    const DynamicBatchContext& dyn);

// Result of binding staging buffers as program parameters.
struct StagingBindResult {
    migraphx::program_parameters params{};
    std::vector<std::size_t> prog_output_indices{};       // ORT output index per bound output
    std::vector<std::string> bound_output_names{};        // staging key per bound output
    std::vector<migraphx::shape> bound_output_shapes{};   // current bucket shape per bound output
};

// Bind staging input/output buffers and the EP-owned scratch buffer as program
// parameters for the given compiled shape.
StagingBindResult BindStagingParams(ComputeState& cs,
    const migraphx::program_parameter_shapes& param_shapes,
    const std::string& shape_hash, hipStream_t stream);

// Copy staging output buffers back into the ORT output tensors, slicing batched
// outputs down to the requested batch when dynamic batching is active.
void CopyStagingOutputsToOrt(ComputeState& cs, const StagingBindResult& bind,
    const Ort::KernelContext& ctx, hipStream_t stream,
    const DynamicBatchContext& dyn);

// Free all staging buffers and reset the allocation flag (used when the program
// is recompiled for a new shape so buffers are re-sized on next use).
void FreeStaging(ComputeState& cs);

// ── hipGraph capture / replay ────────────────────────────────────────────────

// Destroy all captured graphs held by a compute state (used to invalidate the
// cache when the underlying program is recompiled).
void DestroyHipGraphs(ComputeState& cs);

// Dispatch a program run: replay a cached hipGraph for shape_hash, capture one
// on first use, or fall back to an eager run if capture fails.  `params` must
// already be bound to staging buffers/scratch and inputs already staged.
// Extra (non-pre-bound) outputs are materialized into ORT after the launch.
void RunProgramOrHipGraph(ComputeState& cs, hipStream_t stream,
    const Ort::KernelContext& ctx,
    migraphx::program& program,
    migraphx::program_parameters& params,
    const std::vector<std::size_t>& prog_output_indices,
    const std::string& shape_hash,
    const DynamicBatchContext& dyn);

}  // namespace mgx_ep
