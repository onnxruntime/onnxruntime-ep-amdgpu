// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mgx_hip_graph.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <numeric>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "hip/utils.h"
#include "mgx_dynamic_batch.h"
#include "common/env_var.h"

namespace mgx_ep {

namespace {

constexpr std::string_view kScratchParam{"scratch"};

// A/B / safety gate (cached once): force zeroing ALL captured graph outputs before
// every replay instead of only the read-modify-write ones DetectRmwOutputs finds.
bool ForceZeroAllOutputsEnabled() {
    static const bool enabled{
        ParseEnvironmentVariableWithDefault<bool>(env_var::kForceZeroAllGraphOutputs, false)};
    return enabled;
}

// Arena slot alignment: every coalesced input sub-view starts on this boundary so
// the device kernels that consume it stay aligned.
constexpr std::size_t kArenaAlign = 256;

// Product of a length vector (1 for an empty/scalar shape).
std::size_t ProductOf(const std::vector<std::size_t>& lengths) {
    return std::accumulate(lengths.begin(), lengths.end(), std::size_t{1},
        std::multiplies<std::size_t>{});
}

std::size_t AlignUp(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

// Capture warm-in tuning.  A small pre-capture eager phase finalizes MIGraphX's
// lazy allocations; the post-capture phase replays the instantiated graph to push
// state-dependent kernels (atomic reductions, fused-attention accumulators) close
// to steady state before the first real inference.
constexpr int kCaptureFinalizeIterations = 2;
constexpr int kPostCaptureWarmInBase = 10;

// One extra post-capture replay per doubling of the compiled batch size.
int PostCaptureWarminFor(std::size_t batch) {
    int extra{0};
    for (std::size_t b{batch}; b > 1; b >>= 1) {
        ++extra;
    }
    return kPostCaptureWarmInBase + extra;
}

// Best-effort read of the compiled batch size from the input parameter shapes,
// used only to tune warm-in iteration counts.
std::size_t InferCompiledBatchFromParams(const migraphx::program_parameter_shapes& param_shapes,
    const Map<std::size_t>& input_name_indices)
{
    std::size_t batch{0};
    for (const auto& name : param_shapes.names()) {
        if (input_name_indices.count(std::string{name}) == 0) {
            continue;
        }
        const auto lens{param_shapes[name].lengths()};
        if (!lens.empty()) {
            batch = std::max(batch, static_cast<std::size_t>(lens.front()));
        }
    }
    return batch;
}

// Copy extra (non-pre-bound) MIGraphX outputs into their ORT output tensors.
// Their device pointers are stable across replays because the graph replays the
// same kernels into the same MIGraphX-managed memory.
void MaterializeExtraOutputs(const Ort::KernelContext& ctx, hipStream_t stream,
    const std::vector<CapturedHipGraph::ExtraOutput>& extras, const DynamicBatchContext& dyn)
{
    // Report the cached (bind-time) shape directly in the common no-slice case; only a
    // batched extra output needs a shrunk shape, built on a per-thread reusable buffer so
    // the steady state copies/allocates nothing per extra output.
    thread_local std::vector<std::int64_t> sliced_shape;
    for (const auto& extra : extras) {
        const std::vector<std::int64_t>* report_shape{&extra.ort_shape};
        std::size_t bytes{extra.bytes};

        // Slice a batched extra output (captured at target_batch) down to request.
        if (dyn.active && dyn.target_batch > dyn.requested_batch &&
            !extra.ort_shape.empty() &&
            static_cast<std::size_t>(extra.ort_shape.front()) == dyn.target_batch)
        {
            const std::size_t row_bytes{bytes / dyn.target_batch};
            sliced_shape.assign(extra.ort_shape.begin(), extra.ort_shape.end());
            sliced_shape.front() = static_cast<std::int64_t>(dyn.requested_batch);
            bytes = row_bytes * dyn.requested_batch;
            report_shape = &sliced_shape;
        }

        auto output_tensor{ctx.GetOutput(extra.output_index, report_shape->data(), report_shape->size())};
        void* dst{output_tensor.GetTensorMutableRawData()};
        if (bytes > 0) {
            // Stream-ordered async D2D, matching CopyStagingOutputsToOrt on this same
            // graph-replay path (correctness comes from the per-Run stream drain).
            HIP_CALL_THROW(hipMemcpyAsync(dst, extra.gpu_data, bytes,
                hipMemcpyDeviceToDevice, stream));
        }
    }
}

// Destroy a captured graph's exec + graph handles and mark the slot uncaptured, so
// it can be re-captured (drift / stale entry) or freed.  Cached pointers and extra
// outputs are left untouched -- the next capture overwrites them.
void ResetCapturedGraph(CapturedHipGraph& entry) {
    if (entry.exec != nullptr) {
        (void)hipGraphExecDestroy(entry.exec);
        entry.exec = nullptr;
    }
    if (entry.graph != nullptr) {
        (void)hipGraphDestroy(entry.graph);
        entry.graph = nullptr;
    }
    entry.captured = false;
}

// Determine which of `outputs` actually need zeroing before each replay, i.e.
// whose post-run contents depend on their pre-run contents (read-modify-write, or
// partially/never written).  Runs the already-instantiated `exec` twice from two
// different output fill patterns (holding scratch + inputs constant) and compares
// a 128-bit digest of each output; an output whose digest is unchanged by the fill
// is fully overwritten and can be skipped.  Cold path (capture only): the per-
// replay memset fan-out this removes is what dominates batch-1 many-output models.
// On any error it conservatively returns all outputs (zero everything).
std::vector<std::pair<void*, std::size_t>> DetectRmwOutputs(ComputeState& cs,
    ShapeKey shape_key, hipGraphExec_t exec, hipStream_t stream,
    const std::vector<std::pair<void*, std::size_t>>& outputs)
{
    // Gate / degenerate cases: fall back to zeroing everything (original behavior).
    if (ForceZeroAllOutputsEnabled() || exec == nullptr || outputs.empty()) {
        return outputs;
    }
    try {
        std::size_t max_bytes{0};
        for (const auto& [ptr, bytes] : outputs) {
            max_bytes = std::max(max_bytes, bytes);
        }
        std::vector<unsigned char> host(max_bytes);

        // Fill every output with `pattern`, replay once, then digest each output.
        const auto run_and_digest{[&](unsigned char pattern, std::vector<hash::Value>& digests) {
            ZeroScratchFor(cs, shape_key, stream);
            for (const auto& [ptr, bytes] : outputs) {
                HIP_CALL_THROW(hipMemsetAsync(ptr, pattern, bytes, stream));
            }
            HIP_CALL_THROW(hipGraphLaunch(exec, stream));
            HIP_CALL_THROW(hipStreamSynchronize(stream));
            digests.resize(outputs.size());
            for (std::size_t i{0}; i < outputs.size(); ++i) {
                const auto [ptr, bytes] = outputs[i];
                HIP_CALL_THROW(hipMemcpy(host.data(), ptr, bytes, hipMemcpyDeviceToHost));
                hash::Value v{};
                hash::murmur3::x86_128(host.data(), bytes, 0u, v.data());
                digests[i] = v;
            }
        }};

        std::vector<hash::Value> digest_zero, digest_ones;
        run_and_digest(0x00, digest_zero);
        run_and_digest(0xFF, digest_ones);

        std::vector<std::pair<void*, std::size_t>> rmw;
        for (std::size_t i{0}; i < outputs.size(); ++i) {
            if (digest_zero[i] != digest_ones[i]) {
                rmw.push_back(outputs[i]);
            }
        }
        return rmw;
    } catch (...) {
        return outputs;
    }
}

// Item 7: fold the pre-replay zeroing (scratch + the detected RMW outputs) into the
// captured graph so steady-state replay is a single launch instead of launch + N
// memsets.  The staging outputs and scratch are pointer-stable, so baking their
// addresses into the graph is safe; a scratch realloc is caught by the drift check,
// which forces a recapture that re-folds.  Must run only AFTER DetectRmwOutputs (the
// RMW probe must see the bare program, not a pre-zeroed one).  On any failure the
// original exec + out-of-graph zeroing are kept, so this can never regress.
void FoldZeroingIntoCapturedGraph(ComputeState& cs, hipStream_t stream,
    migraphx::program& program, migraphx::program_parameters& params,
    ShapeKey shape_key, CapturedHipGraph& entry)
{
    ScratchBuffer* scratch{nullptr};
    if (const auto it{cs.scratch_bufs.find(shape_key)}; it != cs.scratch_bufs.end() &&
        it->second.data != nullptr && it->second.size_bytes > 0) {
        scratch = &it->second;
    }
    // Nothing to fold -> replay is already a single launch.
    if (scratch == nullptr && entry.captured_output_zeroes.empty()) {
        return;
    }

    hipGraph_t folded{nullptr};
    try {
        HIP_CALL_THROW(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
        if (scratch != nullptr) {
            HIP_CALL_THROW(hipMemsetAsync(scratch->data, 0, scratch->size_bytes, stream));
        }
        for (const auto& [ptr, bytes] : entry.captured_output_zeroes) {
            HIP_CALL_THROW(hipMemsetAsync(ptr, 0, bytes, stream));
        }
        {
            std::lock_guard<std::mutex> lock{cs.mutex};
            program.run_async(params, stream);
        }
        if (hipStreamEndCapture(stream, &folded) != hipSuccess || folded == nullptr) {
            if (folded != nullptr) {
                (void)hipGraphDestroy(folded);
            }
            return;  // keep original exec + out-of-graph zeroing
        }
    } catch (...) {
        // Never leave a capture open on the stream.
        hipGraph_t dummy{nullptr};
        (void)hipStreamEndCapture(stream, &dummy);
        if (dummy != nullptr) {
            (void)hipGraphDestroy(dummy);
        }
        if (folded != nullptr) {
            (void)hipGraphDestroy(folded);
        }
        return;
    }

    hipGraphExec_t folded_exec{nullptr};
    if (hipGraphInstantiate(&folded_exec, folded, nullptr, nullptr, 0) != hipSuccess ||
        folded_exec == nullptr) {
        (void)hipGraphDestroy(folded);
        return;
    }

    // Swap in the folded graph and retire the original.
    if (entry.exec != nullptr) {
        (void)hipGraphExecDestroy(entry.exec);
    }
    if (entry.graph != nullptr) {
        (void)hipGraphDestroy(entry.graph);
    }
    entry.graph = folded;
    entry.exec = folded_exec;
    entry.captured_output_zeroes.clear();  // now zeroed inside the graph
    entry.zeroing_in_graph = true;
}

// Zero a set of (device pointer, byte-count) buffers on `stream`.  All entries are
// assumed non-null with non-zero size (the capture callers filter them out).  Used
// to reset outputs to a known baseline around warmup so warmup-derived bytes are
// not left behind in read-modify-write outputs.
void ZeroOutputBufs(const std::vector<std::pair<void*, std::size_t>>& bufs,
    hipStream_t stream)
{
    for (const auto& [ptr, bytes] : bufs) {
        HIP_CALL_THROW(hipMemsetAsync(ptr, 0, bytes, stream));
    }
}

// Shared warmup + hipGraph capture used by both the staging and direct-bind paths.
// The caller supplies the run params, the output buffers to zero / RMW-probe, and
// (direct-bind only) the input/output device pointers to bake into the entry for
// drift checks.  On success the captured entry lives in cs.hip_graph_cache[shape_key];
// on failure the slot is cleared, `enable_flag` is set false, and false is returned
// so the caller falls back to eager execution.  `direct_bind` selects the few
// path-specific steps: the staging path folds the pre-replay zeroing into the graph
// (item 7) and warms in with a bare launch, while the direct path bakes the ORT
// pointers, skips the fold, and zeroes scratch + RMW outputs before each warm-in launch.
bool WarmupAndCaptureHipGraphCommon(ComputeState& cs, hipStream_t stream,
    migraphx::program& program, migraphx::program_parameters& params,
    ShapeKey shape_key, const std::vector<std::size_t>& prog_output_indices,
    const std::vector<std::pair<void*, std::size_t>>& output_bufs, bool direct_bind,
    const std::vector<void*>& captured_input_ptrs,
    const std::vector<void*>& captured_output_ptrs, bool& enable_flag)
{
    // Zero outputs + scratch so warmup-derived bytes are not baked into the capture.
    ZeroOutputBufs(output_bufs, stream);
    ZeroScratchFor(cs, shape_key, stream);

    // Pre-capture eager loop to finalize MIGraphX's lazy allocations.
    std::optional<migraphx::arguments> warmup_outputs;
    for (int i{0}; i < kCaptureFinalizeIterations; ++i) {
        std::lock_guard<std::mutex> lock{cs.mutex};
        warmup_outputs = program.run_async(params, stream);
    }
    HIP_CALL_THROW(hipStreamSynchronize(stream));

    // Re-zero right before capture so the captured sequence starts from the baseline
    // (the warmup loop just dirtied it).  The direct path re-zeros its ORT output
    // buffers too; the staging path re-zeros only scratch because its outputs are
    // handled by DetectRmwOutputs / FoldZeroingIntoCapturedGraph below.
    if (direct_bind) {
        ZeroOutputBufs(output_bufs, stream);
    }
    ZeroScratchFor(cs, shape_key, stream);
    HIP_CALL_THROW(hipStreamSynchronize(stream));

    const std::size_t compiled_batch{
        InferCompiledBatchFromParams(program.get_parameter_shapes(), cs.input_name_indices)};
    const int post_warmin{PostCaptureWarminFor(compiled_batch)};

    auto& entry{cs.hip_graph_cache[shape_key]};

    // Drop any stale capture for this hash (e.g. a prior staging<->direct entry for
    // the same shape) before overwriting so its graph/exec are not leaked.  A no-op
    // for a fresh slot -- the staging caller resets drifted entries before reaching here.
    if (entry.exec != nullptr) {
        (void)hipGraphExecDestroy(entry.exec);
        entry.exec = nullptr;
    }
    if (entry.graph != nullptr) {
        (void)hipGraphDestroy(entry.graph);
        entry.graph = nullptr;
    }

    try {
        // ThreadLocal capture mode so concurrent serving threads don't have their
        // unrelated stream work swept into this capture.
        HIP_CALL_THROW(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
        {
            std::lock_guard<std::mutex> lock{cs.mutex};
            program.run_async(params, stream);
        }
        const hipError_t err{hipStreamEndCapture(stream, &entry.graph)};
        if (err != hipSuccess || entry.graph == nullptr) {
            entry.graph = nullptr;
            entry.captured = false;
            enable_flag = false;
            return false;
        }

        HIP_CALL_THROW(hipGraphInstantiate(&entry.exec, entry.graph, nullptr, nullptr, 0));
        entry.captured = true;
        entry.direct_bind = direct_bind;
        entry.zeroing_in_graph = false;  // reset; set only if the fold below succeeds
        if (direct_bind) {
            // Baked-in device addresses, compared positionally on every replay.
            entry.captured_input_ptrs = captured_input_ptrs;
            entry.captured_output_ptrs = captured_output_ptrs;
        }

        // Record the scratch pointer baked into the graph for drift detection.
        if (const auto it{cs.scratch_bufs.find(shape_key)}; it != cs.scratch_bufs.end()) {
            entry.captured_scratch_ptr = it->second.data;
        } else {
            entry.captured_scratch_ptr = nullptr;
        }

        // Record only the outputs that actually need zeroing before each replay
        // (item #1): detect read-modify-write / partially-written outputs instead
        // of blindly zeroing all of them.  Conservatively zeroes all on failure.
        entry.captured_output_zeroes =
            DetectRmwOutputs(cs, shape_key, entry.exec, stream, output_bufs);

        // Item 7: fold the (scratch + RMW-output) zeroing into the graph so replay is a
        // single launch.  Must run after DetectRmwOutputs; keeps the original exec on
        // failure.  Applies to both paths: for direct-bind the folded memsets target the
        // same ORT-output/scratch addresses already baked into the program graph, so any
        // pointer drift is caught by CheckCapturedPtrsMatch and forces a recapture that
        // re-folds against the new pointers.
        FoldZeroingIntoCapturedGraph(cs, stream, program, params, shape_key, entry);

        // Post-capture warm-in replays to settle workspace before first real use.  The
        // direct path zeroes scratch + RMW outputs before each launch; the staging path
        // folded that into the graph (or accepts an unzeroed warm-in), so it just launches.
        for (int i{0}; i < post_warmin; ++i) {
            // A folded graph zeroes scratch + RMW outputs inside the launch; only an
            // unfolded direct-bind graph needs the out-of-band zeroing here.
            if (direct_bind && !entry.zeroing_in_graph) {
                ZeroScratchFor(cs, shape_key, stream);
                ZeroOutputBufs(entry.captured_output_zeroes, stream);
            }
            HIP_CALL_THROW(hipGraphLaunch(entry.exec, stream));
        }
        HIP_CALL_THROW(hipStreamSynchronize(stream));

        // Record extra (non-pre-bound) outputs returned by run_async.
        const std::unordered_set<std::size_t> pre_alloc{
            prog_output_indices.begin(), prog_output_indices.end()};
        entry.extra_outputs.clear();
        if (warmup_outputs) {
            const auto output_num{warmup_outputs->size()};
            for (std::size_t i{0}; i < output_num; ++i) {
                if (pre_alloc.count(i) > 0) {
                    continue;
                }
                auto gpu_res{(*warmup_outputs)[i]};
                const migraphx::shape res_shape{gpu_res.get_shape()};
                const auto res_lens{res_shape.lengths()};
                std::vector<std::int64_t> ort_shape{res_lens.begin(), res_lens.end()};
                entry.extra_outputs.push_back(
                    CapturedHipGraph::ExtraOutput{i, std::move(ort_shape),
                        gpu_res.data(), res_shape.bytes()});
            }
        }
        return true;
    } catch (...) {
        hipGraph_t dummy{nullptr};
        (void)hipStreamEndCapture(stream, &dummy);
        if (dummy != nullptr) {
            (void)hipGraphDestroy(dummy);
        }
        entry.graph = nullptr;
        entry.exec = nullptr;
        entry.captured = false;
        enable_flag = false;
        return false;
    }
}

// Warm up, then capture a hipGraph for the currently bound staging params.  Returns
// false (and disables hipGraph on the state) if capture fails so callers fall back
// to eager execution.  Thin wrapper over WarmupAndCaptureHipGraphCommon that gathers
// the staging output buffers to zero / RMW-probe.
bool WarmupAndCaptureHipGraph(ComputeState& cs, hipStream_t stream,
    migraphx::program& program, migraphx::program_parameters& params,
    const std::vector<std::size_t>& prog_output_indices, ShapeKey shape_key)
{
    std::vector<std::pair<void*, std::size_t>> output_bufs;
    output_bufs.reserve(cs.staging_outputs.size());
    for (auto& [name, buf] : cs.staging_outputs) {
        if (buf.data != nullptr && buf.size_bytes > 0) {
            output_bufs.emplace_back(buf.data, buf.size_bytes);
        }
    }
    return WarmupAndCaptureHipGraphCommon(cs, stream, program, params, shape_key,
        prog_output_indices, output_bufs, /*direct_bind=*/false,
        /*captured_input_ptrs=*/{}, /*captured_output_ptrs=*/{}, cs.hip_graph_enable);
}

// Replay a previously captured graph: zero scratch + RMW outputs, then launch.
// scratch_slot is the bind's cached scratch buffer (or nullptr); passing it avoids
// the per-replay scratch_bufs lookup ZeroScratchFor would otherwise do.
void ReplayHipGraph(hipStream_t stream, CapturedHipGraph& entry,
    ScratchBuffer* scratch_slot)
{
    // When the pre-replay zeroing was folded into the captured graph (item 7), the
    // single launch already zeroes scratch + RMW outputs, so skip the out-of-graph
    // memsets entirely.
    if (!entry.zeroing_in_graph) {
        if (scratch_slot != nullptr && scratch_slot->data != nullptr &&
            scratch_slot->size_bytes > 0) {
            HIP_CALL_THROW(hipMemsetAsync(scratch_slot->data, 0,
                scratch_slot->size_bytes, stream));
        }
        for (const auto& [ptr, bytes] : entry.captured_output_zeroes) {
            HIP_CALL_THROW(hipMemsetAsync(ptr, 0, bytes, stream));
        }
    }
    HIP_CALL_THROW(hipGraphLaunch(entry.exec, stream));
}

// True when every current ORT input/output pointer (and the EP-owned scratch)
// matches the address baked into the captured direct-bind graph.  Any mismatch
// means an ORT buffer moved (e.g. allocator recycling) and the graph -- which
// hard-codes device addresses -- must be re-captured.  Pointers are compared
// positionally (same DirectBindCache order used at capture), so this is a flat
// vector walk with no per-call map build or string hashing.  check_inputs is false when
// the inputs are pointer-stable (coalesced arena sub-views): they cannot drift, so only
// the outputs (+ scratch) are compared.
bool CheckCapturedPtrsMatch(const CapturedHipGraph& entry,
    const std::vector<void*>& input_ptrs, const std::vector<void*>& output_ptrs,
    void* scratch_ptr, bool check_inputs)
{
    if (entry.captured_output_ptrs.size() != output_ptrs.size()) {
        return false;
    }
    if (check_inputs) {
        if (entry.captured_input_ptrs.size() != input_ptrs.size()) {
            return false;
        }
        for (std::size_t i{0}; i < input_ptrs.size(); ++i) {
            if (entry.captured_input_ptrs[i] != input_ptrs[i]) {
                return false;
            }
        }
    }
    for (std::size_t i{0}; i < output_ptrs.size(); ++i) {
        if (entry.captured_output_ptrs[i] != output_ptrs[i]) {
            return false;
        }
    }
    return entry.captured_scratch_ptr == scratch_ptr;
}

// Direct-bind capture: `dbc.params` is already bound (by the caller) to the
// current ORT tensor pointers (plus EP-owned scratch), so there are no staging
// copies to make.  Records the current pointers (flat, in dbc order) so replay can
// detect drift.  Returns false (and disables the direct path) on capture failure so
// the caller falls back to eager execution.  Thin wrapper over
// WarmupAndCaptureHipGraphCommon that gathers the ORT-bound output buffers to zero /
// RMW-probe and caches the resulting entry on the dbc.
bool WarmupAndCaptureHipGraphDirect(ComputeState& cs, hipStream_t stream,
    migraphx::program& program, DirectBindCache& dbc, ShapeKey shape_key,
    bool& enable_flag)
{
    // ORT-bound output buffers (ptr,bytes) in dbc order, for zeroing + RMW probe.
    // Inputs are NOT zeroed -- they carry the caller's data.
    std::vector<std::pair<void*, std::size_t>> output_bufs;
    output_bufs.reserve(dbc.outputs.size());
    for (std::size_t i{0}; i < dbc.outputs.size(); ++i) {
        void* ptr{dbc.cur_output_ptrs[i]};
        const std::size_t bytes{dbc.outputs[i].mgx_shape.bytes()};
        if (ptr != nullptr && bytes > 0) {
            output_bufs.emplace_back(ptr, bytes);
        }
    }

    // Cache the entry on the dbc so the steady-state replay path (and
    // MaterializeExtraOutputs) reaches it directly, without re-searching
    // hip_graph_cache by key every call.
    dbc.graph = &cs.hip_graph_cache[shape_key];

    return WarmupAndCaptureHipGraphCommon(cs, stream, program, dbc.params, shape_key,
        dbc.prog_output_indices, output_bufs, /*direct_bind=*/true,
        dbc.cur_input_ptrs, dbc.cur_output_ptrs, enable_flag);
}

}  // namespace

int ComputeOutputIndex(std::string_view name) {
    constexpr std::string_view prefix{"#output_"};
    const auto pos{name.find(prefix)};
    if (pos == std::string_view::npos) {
        return -1;
    }
    auto digits{name.substr(pos + prefix.size())};
    // MIGraphX pads output index >= 10 as "#output_:00056" (colon + zero-pad); skip the ':'.
    if (!digits.empty() && digits.front() == ':') {
        digits.remove_prefix(1);
    }
    const auto* begin{digits.data()};
    const auto* end{digits.data() + digits.size()};
    const auto* last{begin};
    while (last != end && std::isdigit(static_cast<unsigned char>(*last)) != 0) {
        ++last;
    }
    if (begin == last) {
        return -1;
    }
    int value{};
    std::from_chars(begin, last, value);
    return value;
}

// Ensure `slot` holds a buffer of at least `shape` bytes and return its bind info.
// (Re)allocates when missing or grown; plain hipMalloc (not hipMallocAsync) keeps
// these out of the stream-ordered pool.  Fresh/grown buffers are zeroed; callers
// re-zero before use.
static ScratchBindInfo AllocScratchSlot(ScratchBuffer& slot,
    const migraphx::shape& scratch_shape, hipStream_t stream)
{
    const std::size_t needed_bytes{scratch_shape.bytes()};
    if (slot.data == nullptr || needed_bytes > slot.size_bytes) {
        if (slot.data != nullptr) {
            (void)hipFree(slot.data);
            slot.data = nullptr;
            slot.size_bytes = 0;
        }
        void* ptr{nullptr};
        HIP_CALL_THROW(hipMalloc(&ptr, needed_bytes));
        slot.data = ptr;
        slot.size_bytes = needed_bytes;
        slot.shape = scratch_shape;
        HIP_CALL_THROW(hipMemsetAsync(slot.data, 0, slot.size_bytes, stream));
    } else {
        slot.shape = scratch_shape;
    }
    return ScratchBindInfo{slot.data, slot.shape};
}

std::optional<ScratchBindInfo> GetOrAllocScratchCached(ComputeState& cs,
    DirectBindCache& dbc, ShapeKey shape_key, hipStream_t stream)
{
    if (!dbc.has_scratch) {
        return std::nullopt;
    }
    // Resolve the scratch slot once and cache it on the dbc: scratch_bufs entries
    // are never erased mid-session, so the pointer stays valid and later calls skip
    // the map lookup (and ZeroScratchFor uses the same slot -- see the replay path).
    if (dbc.scratch_slot == nullptr) {
        dbc.scratch_slot = &cs.scratch_bufs[shape_key];
    }
    return AllocScratchSlot(*dbc.scratch_slot, dbc.scratch_shape, stream);
}

std::optional<ScratchBindInfo> GetOrAllocScratch(ComputeState& cs,
    const migraphx::program_parameter_shapes& param_shapes,
    ShapeKey shape_key, hipStream_t stream)
{
    bool has_scratch{false};
    for (const auto& name : param_shapes.names()) {
        if (std::string_view{name} == kScratchParam) {
            has_scratch = true;
            break;
        }
    }
    if (!has_scratch) {
        return std::nullopt;
    }
    return AllocScratchSlot(cs.scratch_bufs[shape_key], param_shapes[kScratchParam.data()], stream);
}

void ZeroScratchFor(ComputeState& cs, ShapeKey shape_key, hipStream_t stream) {
    const auto it{cs.scratch_bufs.find(shape_key)};
    if (it == cs.scratch_bufs.end() || it->second.data == nullptr || it->second.size_bytes == 0) {
        return;
    }
    HIP_CALL_THROW(hipMemsetAsync(it->second.data, 0, it->second.size_bytes, stream));
}

void AllocateStaging(ComputeState& cs,
    const migraphx::program_parameter_shapes& param_shapes, hipStream_t stream,
    const DynamicBatchContext& dyn)
{
    if (cs.staging_allocated) {
        return;
    }

    // Batched buffers (batch on axis 0) are sized at max_dynamic_batch so the same
    // allocation serves every compiled bucket; smaller buckets bind a prefix.
    const auto buffer_bytes{[&](const migraphx::shape& shape) -> std::size_t {
        std::size_t bytes{shape.bytes()};
        if (dyn.active && dyn.target_batch > 0) {
            const auto lens{shape.lengths()};
            if (!lens.empty() && lens.front() == dyn.target_batch) {
                const std::size_t row_bytes{bytes / dyn.target_batch};
                const std::size_t max_batch{std::max(cs.max_dynamic_batch, dyn.target_batch)};
                bytes = row_bytes * max_batch;
            }
        }
        return bytes;
    }};

    const auto alloc_buffer{[&](const migraphx::shape& shape) -> StagingBuffer {
        const std::size_t bytes{buffer_bytes(shape)};
        void* ptr{nullptr};
        // Stream-ordered: the alloc rides the compute stream (no device-wide sync)
        // and is completed by the hipStreamSynchronize at the end of AllocateStaging.
        // Must be released with hipFreeAsync (see FreeStaging / ~ExecutionProvider).
        HIP_CALL_THROW(hipMallocAsync(&ptr, bytes, stream));
        HIP_CALL_THROW(hipMemsetAsync(ptr, 0, bytes, stream));
        return StagingBuffer{ptr, bytes, shape};
    }};

    // ── Coalesced inputs: one device arena + one pinned host staging buffer ───
    // Inputs become sub-views into the arena so bind/capture paths are unchanged,
    // and copy gathers them host-side then issues a single H2D for the whole arena.
    if (cs.coalesce_io) {
        std::size_t offset{0};
        for (const auto& name : param_shapes.names()) {
            const std::string param_name{name};
            if (std::string_view{name} == kScratchParam ||
                cs.input_name_indices.count(param_name) == 0) {
                continue;
            }
            const auto shape{param_shapes[name]};
            const std::size_t bytes{buffer_bytes(shape)};
            StagingBuffer buf{};
            buf.size_bytes = bytes;
            buf.shape = shape;
            buf.arena_offset = offset;
            buf.is_arena_view = true;
            cs.staging_inputs.emplace(param_name, buf);
            offset += AlignUp(bytes, kArenaAlign);
        }
        cs.in_arena_bytes = offset;

        if (cs.in_arena_bytes > 0) {
            // Stream-ordered arena alloc (freed via hipFreeAsync); completed by the
            // hipStreamSynchronize at the end of AllocateStaging before first use.
            HIP_CALL_THROW(hipMallocAsync(&cs.in_arena_dev, cs.in_arena_bytes, stream));
            HIP_CALL_THROW(hipMemsetAsync(cs.in_arena_dev, 0, cs.in_arena_bytes, stream));
            HIP_CALL_THROW(hipHostMalloc(&cs.in_staging_host, cs.in_arena_bytes, hipHostMallocDefault));
            std::memset(cs.in_staging_host, 0, cs.in_arena_bytes);
            for (auto& [param_name, buf] : cs.staging_inputs) {
                buf.data = static_cast<char*>(cs.in_arena_dev) + buf.arena_offset;
            }
        }
        cs.staging_inputs_coalesced = true;
    } else {
        for (const auto& name : param_shapes.names()) {
            const std::string param_name{name};
            if (std::string_view{name} == kScratchParam) {
                continue;  // scratch is owned separately
            }
            if (cs.input_name_indices.count(param_name) > 0) {
                cs.staging_inputs.emplace(param_name, alloc_buffer(param_shapes[name]));
            }
        }
    }

    // Outputs are always per-buffer: each maps to a distinct ORT destination, so
    // there is no copy-count win from coalescing them at the EP layer.
    for (const auto& name : param_shapes.names()) {
        const std::string param_name{name};
        if (std::string_view{name} == kScratchParam ||
            cs.input_name_indices.count(param_name) > 0) {
            continue;
        }
        if (ComputeOutputIndex(name) != -1) {
            cs.staging_outputs.emplace(param_name, alloc_buffer(param_shapes[name]));
        }
    }

    HIP_CALL_THROW(hipStreamSynchronize(stream));
    cs.staging_allocated = true;
}

// True when `ptr` is page-locked (pinned) host memory: pinned sources DMA straight into
// the device arena, whereas pageable memory must first be gathered into a pinned host
// buffer.  Unregistered/unknown pointers report false (and clear the sticky error).
static bool IsPinnedHostPtr(const void* ptr) {
    if (ptr == nullptr) {
        return false;
    }
    hipPointerAttribute_t attr{};
    if (hipPointerGetAttributes(&attr, ptr) != hipSuccess) {
        (void)hipGetLastError();
        return false;
    }
    return attr.type == hipMemoryTypeHost;
}

// Classify the coalesced inputs once: any device-resident input disqualifies the
// coalesced path; otherwise all-host, split by whether every source is pinned (direct
// DMA) or some are pageable (gather + single H2D).
static ComputeState::CoalesceResidency ProbeCoalesceResidency(
    const StagingBindResult& bind, const Ort::KernelContext& ctx) {
    bool all_pinned{true};
    for (const auto& ib : bind.input_copies) {
        const auto in{ctx.GetInput(ib.ort_index)};
        if (in.GetTensorMemoryInfo().GetDeviceType() != OrtMemoryInfoDeviceType_CPU) {
            return ComputeState::CoalesceResidency::kHasDevice;
        }
        if (all_pinned && !IsPinnedHostPtr(in.GetTensorRawData())) {
            all_pinned = false;
        }
    }
    return all_pinned ? ComputeState::CoalesceResidency::kAllHostPinned
                      : ComputeState::CoalesceResidency::kAllHostPageable;
}

// Copy every coalesced input into its arena sub-view in one pass.  Pinned sources DMA
// straight into the device arena; pageable sources are gathered into the pinned host
// buffer then flushed with one whole-arena H2D.  A batched input copies only its real
// requested_batch rows (the pad tail is left as-is -- pad output rows are sliced off
// downstream); others copy in full.  When refresh_ptrs is set the ORT data pointers are
// (re)read and recorded here so the shape scan and this gather share one traversal;
// otherwise the pointers already recorded by the scan are reused.  Requires residency to
// be a resolved all-host state.
static void CoalesceInputsCore(ComputeState& cs, const StagingBindResult& bind,
    const Ort::KernelContext& ctx, const DynamicBatchContext& dyn,
    hipStream_t stream, bool refresh_ptrs) {
    const bool pinned{cs.coalesce_residency == ComputeState::CoalesceResidency::kAllHostPinned};
    const bool batch_pad{dyn.active && dyn.target_batch > dyn.requested_batch};
    char* const host_base{static_cast<char*>(cs.in_staging_host)};
    char* const arena_base{static_cast<char*>(cs.in_arena_dev)};
    for (const auto& ib : bind.input_copies) {
        const void* src{nullptr};
        if (refresh_ptrs) {
            src = ctx.GetInput(ib.ort_index).GetTensorRawData();
            if (ib.ort_index < cs.cur_input_data.size()) {
                cs.cur_input_data[ib.ort_index] = src;
            }
        } else {
            src = ib.ort_index < cs.cur_input_data.size() ? cs.cur_input_data[ib.ort_index] : nullptr;
            if (src == nullptr) {
                src = ctx.GetInput(ib.ort_index).GetTensorRawData();
            }
        }
        const bool batched{batch_pad && !ib.prog_lens.empty() &&
            ib.prog_lens.front() == dyn.target_batch &&
            ib.ort_index < cs.cur_input_axis0.size() &&
            cs.cur_input_axis0[ib.ort_index] == static_cast<std::int64_t>(dyn.requested_batch)};
        std::size_t copy_bytes{batched ? ib.row_bytes * dyn.requested_batch : ib.prog_bytes};
        if (copy_bytes > ib.stage_capacity) {
            copy_bytes = ib.stage_capacity;
        }
        if (copy_bytes == 0) {
            continue;
        }
        if (pinned) {
            HIP_CALL_THROW(hipMemcpyAsync(arena_base + ib.arena_offset, src, copy_bytes,
                hipMemcpyHostToDevice, stream));
        } else {
            std::memcpy(host_base + ib.arena_offset, src, copy_bytes);
        }
    }
    if (!pinned) {
        HIP_CALL_THROW(hipMemcpyAsync(cs.in_arena_dev, cs.in_staging_host,
            cs.in_arena_bytes, hipMemcpyHostToDevice, stream));
    }
}

void CopyInputsToStaging(ComputeState& cs,
    const StagingBindResult& bind,
    const Ort::KernelContext& ctx, hipStream_t stream,
    const DynamicBatchContext& dyn,
    const StaticSeqContext& seq)
{
    // The fused shape-scan already coalesced this call's inputs in one pass.
    if (cs.inputs_coalesced_this_call) {
        return;
    }

    // Coalesced fast path: gather every host-resident input into the arena (pinned
    // sources DMA directly; pageable sources gather then one H2D), driven by the
    // precomputed bind.input_copies plan (no names/strings/map lookups).  A device
    // input or active seq padding (real tokens at a per-slice interior offset) falls
    // through to the per-input path below.
    const bool seq_no_pad{!seq.active || seq.target_len == seq.real_len};
    if (cs.staging_inputs_coalesced && cs.in_staging_host != nullptr && seq_no_pad) {
        if (cs.coalesce_residency == ComputeState::CoalesceResidency::kUnknown) {
            cs.coalesce_residency = ProbeCoalesceResidency(bind, ctx);
        }
        if (cs.coalesce_residency == ComputeState::CoalesceResidency::kAllHostPinned ||
            cs.coalesce_residency == ComputeState::CoalesceResidency::kAllHostPageable) {
            CoalesceInputsCore(cs, bind, ctx, dyn, stream, /*refresh_ptrs=*/false);
            return;
        }
    }

    // Per-input fallback (a device-resident input, or seq padding is active).  Only
    // the seq-padding branch needs the actual ORT shape (to place the real token
    // span); batch padding is driven entirely by the bind-time program lengths + the
    // resolved dynamic-batch context, so it needs no per-call GetShape (item 4).
    const bool need_actual_shape{seq.active && seq.input_axes != nullptr};
    for (const auto& ib : bind.input_copies) {
        const auto input_tensor{ctx.GetInput(ib.ort_index)};
        const void* src{input_tensor.GetTensorRawData()};
        const auto& prog_lens{ib.prog_lens};

        std::vector<std::int64_t> actual_shape;
        if (need_actual_shape) {
            actual_shape = input_tensor.GetTensorTypeAndShapeInfo().GetShape();
        }

        // A batched input arrives with requested_batch rows but the program (and
        // staging) expect target_batch rows.  Batched == program axis-0 was bucketed up
        // to target_batch AND the actual axis-0 (cached during the input scan, item 4 --
        // no GetShape here) equals requested_batch; the second half rejects a fixed dim
        // that merely coincides with target_batch.
        const bool batched{dyn.active && dyn.target_batch > dyn.requested_batch &&
            !prog_lens.empty() && prog_lens.front() == dyn.target_batch &&
            ib.ort_index < cs.cur_input_axis0.size() &&
            cs.cur_input_axis0[ib.ort_index] == static_cast<std::int64_t>(dyn.requested_batch)};

        // A named seq input arrives with real_len tokens but the program expects
        // target_len; copy the real tokens per slice and zero-fill the pad tail.
        int seq_axis{-1};
        if (seq.active && seq.input_axes != nullptr && !batched) {
            if (const auto it{seq.input_axes->find(ib.name)}; it != seq.input_axes->end()) {
                const int axis{it->second};
                if (axis >= 0 && static_cast<std::size_t>(axis) < prog_lens.size() &&
                    prog_lens[axis] == seq.target_len && axis < static_cast<int>(actual_shape.size()) &&
                    static_cast<std::size_t>(actual_shape[axis]) == seq.real_len) {
                    seq_axis = axis;
                }
            }
        }

        if (seq_axis >= 0) {
            std::size_t outer{1};
            for (int a{0}; a < seq_axis; ++a) outer *= prog_lens[a];
            std::size_t inner{1};
            for (std::size_t a{static_cast<std::size_t>(seq_axis) + 1}; a < prog_lens.size(); ++a)
                inner *= prog_lens[a];
            PadSeqTensor(src, ib.staging_data, outer, seq.real_len, seq.target_len,
                inner, ib.element_size, stream);
        } else {
            // Batched: copy the real requested_batch rows; unbatched: copy in full.  The
            // pad tail is left as-is (no per-call zeroing) -- pad output rows are sliced
            // off downstream.
            std::size_t copy_bytes{batched ? ib.row_bytes * dyn.requested_batch
                                           : ib.prog_bytes};
            if (copy_bytes > ib.stage_capacity) {
                copy_bytes = ib.stage_capacity;
            }
            if (copy_bytes > 0) {
                HIP_CALL_THROW(hipMemcpyAsync(ib.staging_data, src, copy_bytes,
                    hipMemcpyDefault, stream));
            }
        }
    }
}

bool TryFusedCoalesceGather(ComputeState& cs,
    const Ort::KernelContext& ctx, const DynamicBatchContext& dyn,
    ShapeKey shape_key, hipStream_t stream) {
    if (!cs.coalesce_io || !cs.staging_inputs_coalesced ||
        cs.in_staging_host == nullptr || cs.in_arena_dev == nullptr) {
        return false;
    }
    if (cs.coalesce_residency != ComputeState::CoalesceResidency::kAllHostPinned &&
        cs.coalesce_residency != ComputeState::CoalesceResidency::kAllHostPageable) {
        return false;
    }
    const auto it{cs.staging_bind_cache.find(shape_key)};
    if (it == cs.staging_bind_cache.end()) {
        return false;
    }
    CoalesceInputsCore(cs, it->second, ctx, dyn, stream, /*refresh_ptrs=*/true);
    cs.inputs_coalesced_this_call = true;
    return true;
}

StagingBindResult BindStagingParams(ComputeState& cs,
    const migraphx::program_parameter_shapes& param_shapes,
    ShapeKey shape_key, hipStream_t stream)
{
    StagingBindResult result;
    // Item 1: build the hybrid (direct-bound outputs) plan alongside the staging
    // bind.  Eligible unless a program parameter is neither an input, an output, nor
    // scratch (an unbound/literal param the hybrid cannot direct-bind).
    result.hybrid.eligible = true;
    // Hybrid inputs are pointer-stable staging/arena sub-views, so replay drift checks
    // only the ORT output pointers.
    result.hybrid.inputs_pointer_stable = true;
    for (const auto& name : param_shapes.names()) {
        const std::string param_name{name};
        if (const auto idx_it{cs.input_name_indices.find(param_name)};
            idx_it != cs.input_name_indices.end()) {
            const auto stage_it{cs.staging_inputs.find(param_name)};
            if (stage_it == cs.staging_inputs.end()) {
                result.hybrid.eligible = false;  // input with no staging buffer
                continue;
            }
            const auto in_shape{param_shapes[name]};
            result.params.add(name, migraphx::argument{in_shape, stage_it->second.data});
            // Flat copy-plan entry so the per-call input copy needs no name/string/map work.
            StagingInputBind ib;
            ib.ort_index = idx_it->second;
            ib.staging_data = stage_it->second.data;
            ib.arena_offset = stage_it->second.arena_offset;
            ib.stage_capacity = stage_it->second.size_bytes;
            ib.prog_bytes = in_shape.bytes();
            const auto in_lens{in_shape.lengths()};
            ib.prog_lens.assign(in_lens.begin(), in_lens.end());
            const std::size_t total_elems{ProductOf(ib.prog_lens)};
            ib.element_size = total_elems > 0 ? ib.prog_bytes / total_elems : 0;
            // Precompute bytes-per-row (axis 0) so the batch-pad copy computes the real
            // byte count as row_bytes * requested_batch with no per-call shape read.
            ib.row_bytes = (!ib.prog_lens.empty() && ib.prog_lens.front() > 0)
                ? ib.prog_bytes / ib.prog_lens.front() : ib.prog_bytes;
            ib.name = param_name;
            result.input_copies.push_back(std::move(ib));
            // Hybrid: bind this input to its (pointer-stable) arena sub-view.  The
            // ptr never drifts, so it is recorded once here and never re-gathered.
            result.hybrid.inputs.push_back(
                CachedDirectInput{param_name, idx_it->second, in_shape});
            result.hybrid.cur_input_ptrs.push_back(stage_it->second.data);
        } else if (std::string_view{name} == kScratchParam) {
            if (const auto scratch{GetOrAllocScratch(cs, param_shapes, shape_key, stream)}) {
                result.params.add(name, migraphx::argument{scratch->shape, scratch->ptr});
                // Cache the scratch slot so replay zeroing + the drift check skip the
                // per-call scratch_bufs lookup (entries are never erased mid-session).
                result.scratch_slot = &cs.scratch_bufs[shape_key];
                // Hybrid shares the same scratch (same shape_key / slot).
                result.hybrid.has_scratch = true;
                result.hybrid.scratch_shape = scratch->shape;
            }
        } else if (const auto oi{ComputeOutputIndex(name)}; oi != -1) {
            const auto stage_it{cs.staging_outputs.find(param_name)};
            if (stage_it == cs.staging_outputs.end()) {
                continue;
            }
            const auto out_shape{param_shapes[name]};
            result.params.add(name, migraphx::argument{out_shape, stage_it->second.data});
            result.prog_output_indices.push_back(static_cast<std::size_t>(oi));
            result.bound_output_names.push_back(param_name);
            result.bound_output_shapes.push_back(out_shape);
            // Cache the staging source pointer so the per-call output copy needs no
            // staging_outputs map lookup (mirrors the input_copies plan).
            result.bound_output_data.push_back(stage_it->second.data);
            // Precompute the ORT (int64) bucket shape so the per-call output copy
            // does not rebuild it every inference (it only slices it when padding).
            const auto out_lens{out_shape.lengths()};
            result.bound_output_ort_shapes.emplace_back(out_lens.begin(), out_lens.end());
            // Precompute bytes-per-row (axis 0) so the batch slice avoids a per-call
            // division when shrinking a padded output to the requested batch.
            result.bound_output_row_bytes.push_back(
                !out_lens.empty() && out_lens.front() > 0 ? out_shape.bytes() / out_lens.front()
                                                          : out_shape.bytes());
            // Precompute the total byte count so the per-call copy skips shape.bytes().
            result.bound_output_bytes.push_back(out_shape.bytes());
            // Hybrid: this output can be bound directly to its ORT tensor.  ort_shape
            // == program shape here (the hybrid runs only when no padding is needed).
            result.hybrid.outputs.push_back(CachedDirectOutput{
                param_name, static_cast<std::size_t>(oi), out_shape,
                std::vector<std::int64_t>{out_lens.begin(), out_lens.end()}});
            result.hybrid.prog_output_indices.push_back(static_cast<std::size_t>(oi));
        } else {
            result.hybrid.eligible = false;  // unbound/literal param -> no direct bind
        }
    }
    return result;
}

void CopyStagingOutputsToOrt(const StagingBindResult& bind,
    const Ort::KernelContext& ctx, hipStream_t stream,
    const DynamicBatchContext& dyn,
    const StaticSeqContext& seq)
{
    for (std::size_t i{}; i < bind.prog_output_indices.size() &&
        i < bind.bound_output_shapes.size() &&
        i < bind.bound_output_ort_shapes.size() &&
        i < bind.bound_output_bytes.size() &&
        i < bind.bound_output_data.size(); ++i)
    {
        const auto oi{bind.prog_output_indices[i]};
        // Staging source pointer resolved once at bind time (no map lookup here).
        void* const src{bind.bound_output_data[i]};
        // Bucket shapes were precomputed at bind time; reuse the cached ORT (int64)
        // shape so the common (no-slice) path allocates nothing per output.  Use the
        // current bucket's shape, not the (first-bucket) staging shape, so dynamic
        // batch buckets report the correct dimensions.
        const auto& out_shape{bind.bound_output_shapes[i]};
        const auto& cached_ort_shape{bind.bound_output_ort_shapes[i]};
        // Precomputed at bind (item 6); bounded by the loop like its sibling vectors.
        std::size_t bytes{bind.bound_output_bytes[i]};

        // Slice a batched output down to the requested batch.
        const bool batch_sliced{dyn.active && dyn.target_batch > dyn.requested_batch &&
            !cached_ort_shape.empty() &&
            static_cast<std::size_t>(cached_ort_shape.front()) == dyn.target_batch};

        // Slice a named seq output (e.g. logits) back to real_len on its token axis.
        // Excludes batch-slicing: element_size below uses the un-shrunk `bytes`.
        int seq_axis{-1};
        if (seq.active && seq.output_axes_by_index != nullptr && seq.target_len > seq.real_len &&
            !batch_sliced) {
            if (const auto it{seq.output_axes_by_index->find(oi)}; it != seq.output_axes_by_index->end()) {
                const int axis{it->second};
                if (axis >= 0 && static_cast<std::size_t>(axis) < cached_ort_shape.size() &&
                    static_cast<std::size_t>(cached_ort_shape[axis]) == seq.target_len) {
                    seq_axis = axis;
                }
            }
        }

        // Report the cached bucket shape as-is unless a slice is needed (batch or
        // seq), in which case work on a per-thread reusable buffer.  The batch-1 steady
        // state needs no slice and never touches it; the padded-prefill slice reuses the
        // buffer's capacity (item 5), so it allocates nothing after warmup.
        const std::vector<std::int64_t>* report_shape{&cached_ort_shape};
        thread_local std::vector<std::int64_t> sliced_shape;
        if (batch_sliced) {
            // row_bytes precomputed at bind (item 3): bytes / target_batch.
            const std::size_t row_bytes{i < bind.bound_output_row_bytes.size()
                ? bind.bound_output_row_bytes[i] : bytes / dyn.target_batch};
            sliced_shape.assign(cached_ort_shape.begin(), cached_ort_shape.end());
            sliced_shape.front() = static_cast<std::int64_t>(dyn.requested_batch);
            bytes = row_bytes * dyn.requested_batch;
            report_shape = &sliced_shape;
        } else if (seq_axis >= 0) {
            sliced_shape.assign(cached_ort_shape.begin(), cached_ort_shape.end());
            sliced_shape[seq_axis] = static_cast<std::int64_t>(seq.real_len);
            report_shape = &sliced_shape;
        }

        auto output_tensor{ctx.GetOutput(oi, report_shape->data(), report_shape->size())};
        void* dst{output_tensor.GetTensorMutableRawData()};

        if (seq_axis >= 0) {
            // Per outer slice, copy the first real_len rows (inner_count elems each).
            std::size_t outer{1};
            for (int a{0}; a < seq_axis; ++a) outer *= static_cast<std::size_t>(cached_ort_shape[a]);
            std::size_t inner{1};
            for (std::size_t a{static_cast<std::size_t>(seq_axis) + 1}; a < cached_ort_shape.size(); ++a)
                inner *= static_cast<std::size_t>(cached_ort_shape[a]);
            const auto lengths{out_shape.lengths()};
            const std::size_t total_elems{ProductOf(lengths)};
            const std::size_t element_size{total_elems > 0 ? bytes / total_elems : 0};
            const std::size_t src_slice_bytes{seq.target_len * inner * element_size};
            const std::size_t dst_slice_bytes{seq.real_len * inner * element_size};
            for (std::size_t o{0}; o < outer; ++o) {
                if (dst_slice_bytes > 0) {
                    HIP_CALL_THROW(hipMemcpyAsync(
                        static_cast<char*>(dst) + o * dst_slice_bytes,
                        static_cast<const char*>(src) + o * src_slice_bytes,
                        dst_slice_bytes, hipMemcpyDefault, stream));
                }
            }
        } else if (bytes > 0) {
            HIP_CALL_THROW(hipMemcpyAsync(dst, src, bytes, hipMemcpyDefault, stream));
        }
    }
}

void FreeStaging(ComputeState& cs, hipStream_t stream) {
    // Staging inputs/outputs and the coalesce arena are allocated with
    // hipMallocAsync (AllocateStaging), so they must be released with hipFreeAsync
    // on a valid stream.  The subsequent AllocateStaging on the same stream reuses
    // the freed pool memory in stream order.
    for (auto& [name, buf] : cs.staging_inputs) {
        // Arena sub-views are not independent allocations; the arena is freed below.
        if (buf.data != nullptr && !buf.is_arena_view) {
            (void)hipFreeAsync(buf.data, stream);
        }
        buf.data = nullptr;
    }
    if (cs.in_arena_dev != nullptr) {
        (void)hipFreeAsync(cs.in_arena_dev, stream);
        cs.in_arena_dev = nullptr;
    }
    if (cs.in_staging_host != nullptr) {
        (void)hipHostFree(cs.in_staging_host);
        cs.in_staging_host = nullptr;
    }
    cs.in_arena_bytes = 0;
    cs.staging_inputs_coalesced = false;
    // Re-verify coalesce eligibility against the next allocation's inputs.
    cs.coalesce_residency = ComputeState::CoalesceResidency::kUnknown;
    for (auto& [name, buf] : cs.staging_outputs) {
        if (buf.data != nullptr) {
            (void)hipFreeAsync(buf.data, stream);
            buf.data = nullptr;
        }
    }
    cs.staging_inputs.clear();
    cs.staging_outputs.clear();
    cs.staging_allocated = false;
    // Cached bindings reference the staging buffers just freed; drop them so they
    // are rebuilt against the next allocation.
    cs.staging_bind_cache.clear();
}

void DestroyHipGraphs(ComputeState& cs) {
    for (auto& [hash, entry] : cs.hip_graph_cache) {
        ResetCapturedGraph(entry);
    }
    cs.hip_graph_cache.clear();
}

// Copy every program output not pre-bound (its index absent from prog_output_indices)
// to the matching ORT output tensor, device-to-device.  Shared by the staging,
// direct, and no-graph eager paths.
void CopyUnboundOutputsToOrt(const Ort::KernelContext& ctx, hipStream_t stream,
    migraphx::arguments& outputs, const std::vector<std::size_t>& prog_output_indices)
{
    const std::unordered_set<std::size_t> pre_bound{
        prog_output_indices.begin(), prog_output_indices.end()};
    for (std::size_t i{0}; i < outputs.size(); ++i) {
        if (pre_bound.count(i) > 0) {
            continue;
        }
        const auto out{outputs[i]};
        const auto shape{out.get_shape()};
        const auto lens{shape.lengths()};
        std::vector<std::int64_t> ort_shape{lens.begin(), lens.end()};
        auto dst{ctx.GetOutput(i, ort_shape.data(), ort_shape.size())};
        if (shape.bytes() > 0) {
            HIP_CALL_THROW(hipMemcpyWithStream(dst.GetTensorMutableRawData(), out.data(),
                shape.bytes(), hipMemcpyDeviceToDevice, stream));
        }
    }
}

// Eager run + copy of the non-pre-bound ("extra") outputs, shared by the staging and
// direct run paths.  Without this the extra outputs (e.g. a KV cache) are lost, since
// only the hipGraph replay path's MaterializeExtraOutputs copies them.
static void RunProgramEagerCopyingExtras(ComputeState& cs, hipStream_t stream,
    const Ort::KernelContext& ctx, migraphx::program& program,
    migraphx::program_parameters& params,
    const std::vector<std::size_t>& prog_output_indices)
{
    std::optional<migraphx::arguments> outputs;
    {
        std::lock_guard<std::mutex> lock{cs.mutex};
        outputs = program.run_async(params, stream);
    }
    if (!outputs) {
        return;
    }
    CopyUnboundOutputsToOrt(ctx, stream, *outputs, prog_output_indices);
}

void RunProgramOrHipGraph(ComputeState& cs, hipStream_t stream,
    const Ort::KernelContext& ctx,
    migraphx::program& program,
    StagingBindResult& bind,
    ShapeKey shape_key,
    const DynamicBatchContext& dyn)
{
    // Eager run + copy non-pre-bound outputs (KV present) to ORT tensors; without this the KV is
    // lost (only the hipGraph path's MaterializeExtraOutputs copies them).
    const auto eager_run_with_extras{[&]() {
        RunProgramEagerCopyingExtras(cs, stream, ctx, program, bind.params, bind.prog_output_indices);
    }};

    if (!cs.hip_graph_enable) {
        eager_run_with_extras();
        return;
    }

    // Resolve this shape's captured-graph slot once and cache it on the bind, so the
    // steady-state replay does a single map lookup only on the first call and none
    // thereafter (mirrors DirectBindCache.graph).  The two former lookups (stale
    // direct-bind drop + capture-status check) are merged into this one resolve.
    if (bind.graph == nullptr) {
        if (const auto it{cs.hip_graph_cache.find(shape_key)}; it != cs.hip_graph_cache.end()) {
            bind.graph = &it->second;
        }
    }
    CapturedHipGraph* entry{bind.graph};

    // A direct-bind capture (against ORT pointers) cannot be replayed through the
    // staging path.  This happens when the direct path was disabled at runtime
    // (pointer drift) and the same shape_key falls back here; drop the stale
    // entry so a staging graph is captured fresh below.
    if (entry != nullptr && entry->direct_bind) {
        ResetCapturedGraph(*entry);
        entry->direct_bind = false;
    }

    if (entry != nullptr && entry->captured) {
        // Re-capture if the scratch buffer was reallocated since capture.  The slot
        // pointer was cached on the bind (scratch_bufs entries never move), so this
        // drift check needs no scratch_bufs lookup.
        void* const current_scratch{
            bind.scratch_slot != nullptr ? bind.scratch_slot->data : nullptr};
        if (entry->captured_scratch_ptr != current_scratch) {
            ResetCapturedGraph(*entry);
        } else {
            ReplayHipGraph(stream, *entry, bind.scratch_slot);
            MaterializeExtraOutputs(ctx, stream, entry->extra_outputs, dyn);
            return;
        }
    }

    if (!WarmupAndCaptureHipGraph(cs, stream, program, bind.params, bind.prog_output_indices, shape_key)) {
        eager_run_with_extras();
        return;
    }
    // Capture created (or reused) the entry node; (re)cache the pointer so the next
    // call replays without a lookup.
    bind.graph = &cs.hip_graph_cache[shape_key];
    MaterializeExtraOutputs(ctx, stream, bind.graph->extra_outputs, dyn);
}

void RunProgramOrHipGraphDirect(ComputeState& cs, hipStream_t stream,
    const Ort::KernelContext& ctx,
    migraphx::program& program,
    DirectBindCache& dbc,
    ShapeKey shape_key,
    const std::optional<ScratchBindInfo>& scratch,
    const DynamicBatchContext& dyn,
    bool& enable_flag,
    int& recapture_count)
{
    void* const current_scratch{scratch ? scratch->ptr : nullptr};

    // (Re)bind params to the current ORT pointers + scratch.  Only the capture and
    // eager paths use `params`; the steady-state replay below never touches it, so
    // this per-parameter add() work is skipped on the hot path.
    const auto rebind_params{[&]() {
        for (std::size_t i{0}; i < dbc.inputs.size(); ++i) {
            dbc.params.add(dbc.inputs[i].name.c_str(),
                migraphx::argument{dbc.inputs[i].mgx_shape, dbc.cur_input_ptrs[i]});
        }
        for (std::size_t i{0}; i < dbc.outputs.size(); ++i) {
            dbc.params.add(dbc.outputs[i].name.c_str(),
                migraphx::argument{dbc.outputs[i].mgx_shape, dbc.cur_output_ptrs[i]});
        }
        if (scratch) {
            dbc.params.add("scratch", migraphx::argument{scratch->shape, scratch->ptr});
        }
    }};

    // Eager fallback: bind, run against the ORT pointers, and copy any non-pre-bound
    // (extra) outputs to their ORT tensors.  Used when capture fails or the direct
    // path has been permanently disabled by pointer drift.
    const auto eager_run_with_extras{[&]() {
        rebind_params();
        RunProgramEagerCopyingExtras(cs, stream, ctx, program, dbc.params, dbc.prog_output_indices);
    }};

    // The captured graph for this shape is cached on the dbc (resolved at capture),
    // so the steady-state replay reaches it directly -- no hip_graph_cache lookup.
    if (CapturedHipGraph* entry{dbc.graph};
        entry != nullptr && entry->captured && entry->direct_bind)
    {
        if (!CheckCapturedPtrsMatch(*entry, dbc.cur_input_ptrs, dbc.cur_output_ptrs,
                current_scratch, /*check_inputs=*/!dbc.inputs_pointer_stable)) {
            // ORT recycled a buffer under us.  Re-capture, but if drift is
            // sustained give up on this path (permanent eager/staging fallback)
            // rather than re-capturing on every call.
            ++recapture_count;
            if (recapture_count > ComputeState::kMaxDirectRecaptures) {
                enable_flag = false;
                eager_run_with_extras();
                return;
            }
            ResetCapturedGraph(*entry);
        } else {
            // Pointers matched: reset the drift counter so only *consecutive*
            // mismatches can trip the permanent fallback above.  No param rebind and
            // no map lookups.  When the zeroing was folded into the captured graph
            // (item 7) the single launch already zeroes scratch + RMW outputs, so skip
            // the out-of-graph memsets; otherwise zero via the cached slot + the RMW
            // outputs, then replay.
            recapture_count = 0;
            if (!entry->zeroing_in_graph) {
                if (dbc.scratch_slot != nullptr && dbc.scratch_slot->data != nullptr &&
                    dbc.scratch_slot->size_bytes > 0) {
                    HIP_CALL_THROW(hipMemsetAsync(dbc.scratch_slot->data, 0,
                        dbc.scratch_slot->size_bytes, stream));
                }
                for (const auto& [ptr, bytes] : entry->captured_output_zeroes) {
                    HIP_CALL_THROW(hipMemsetAsync(ptr, 0, bytes, stream));
                }
            }
            HIP_CALL_THROW(hipGraphLaunch(entry->exec, stream));
            MaterializeExtraOutputs(ctx, stream, entry->extra_outputs, dyn);
            return;
        }
    }

    // Capture path (first use for this shape or a re-capture after drift): bind
    // params to the current pointers, then warm up + capture.  WarmupAndCaptureHipGraphDirect
    // sets dbc.graph to the freshly captured entry.
    rebind_params();
    if (!WarmupAndCaptureHipGraphDirect(cs, stream, program, dbc, shape_key, enable_flag)) {
        eager_run_with_extras();
        return;
    }
    MaterializeExtraOutputs(ctx, stream, dbc.graph->extra_outputs, dyn);
}

}  // namespace mgx_ep
