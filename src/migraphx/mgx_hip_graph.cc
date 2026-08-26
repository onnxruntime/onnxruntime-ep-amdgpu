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
    for (const auto& extra : extras) {
        auto ort_shape{extra.ort_shape};
        std::size_t bytes{extra.bytes};

        // Slice a batched extra output (captured at target_batch) down to request.
        if (dyn.active && dyn.target_batch > dyn.requested_batch &&
            !ort_shape.empty() &&
            static_cast<std::size_t>(ort_shape.front()) == dyn.target_batch)
        {
            const std::size_t row_bytes{bytes / dyn.target_batch};
            ort_shape.front() = static_cast<std::int64_t>(dyn.requested_batch);
            bytes = row_bytes * dyn.requested_batch;
        }

        auto output_tensor{ctx.GetOutput(extra.output_index, ort_shape.data(), ort_shape.size())};
        void* dst{output_tensor.GetTensorMutableRawData()};
        if (bytes > 0) {
            HIP_CALL_THROW(hipMemcpyWithStream(dst, extra.gpu_data, bytes,
                hipMemcpyDeviceToDevice, stream));
        }
    }
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

// Warm up, then capture a hipGraph for the currently bound params.  Returns false
// (and disables hipGraph on the state) if capture fails so callers fall back to
// eager execution.
bool WarmupAndCaptureHipGraph(ComputeState& cs, hipStream_t stream,
    migraphx::program& program, migraphx::program_parameters& params,
    const std::vector<std::size_t>& prog_output_indices, ShapeKey shape_key)
{
    // Zero staging outputs and scratch so warmup-derived bytes are not baked into
    // the capture.
    for (auto& [name, buf] : cs.staging_outputs) {
        if (buf.data != nullptr) {
            HIP_CALL_THROW(hipMemsetAsync(buf.data, 0, buf.size_bytes, stream));
        }
    }
    ZeroScratchFor(cs, shape_key, stream);

    // Pre-capture eager loop to finalize MIGraphX's lazy allocations.
    std::optional<migraphx::arguments> warmup_outputs;
    for (int i{0}; i < kCaptureFinalizeIterations; ++i) {
        std::lock_guard<std::mutex> lock{cs.mutex};
        warmup_outputs = program.run_async(params, stream);
    }
    HIP_CALL_THROW(hipStreamSynchronize(stream));

    const std::size_t compiled_batch{
        InferCompiledBatchFromParams(program.get_parameter_shapes(), cs.input_name_indices)};
    const int post_warmin{PostCaptureWarminFor(compiled_batch)};

    auto& entry{cs.hip_graph_cache[shape_key]};

    // Re-zero scratch right before capture so the captured kernel sequence is
    // anchored to a known baseline (the warmup loop just dirtied it).
    ZeroScratchFor(cs, shape_key, stream);
    HIP_CALL_THROW(hipStreamSynchronize(stream));

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
            cs.hip_graph_enable = false;
            return false;
        }

        HIP_CALL_THROW(hipGraphInstantiate(&entry.exec, entry.graph, nullptr, nullptr, 0));
        entry.captured = true;
        entry.direct_bind = false;  // staging capture: replayed via ReplayHipGraph
        entry.zeroing_in_graph = false;  // reset; set only if the fold below succeeds

        // Record the scratch pointer baked into the graph for drift detection.
        if (const auto it{cs.scratch_bufs.find(shape_key)}; it != cs.scratch_bufs.end()) {
            entry.captured_scratch_ptr = it->second.data;
        } else {
            entry.captured_scratch_ptr = nullptr;
        }

        // Record only the outputs that actually need zeroing before each replay
        // (item #1): detect read-modify-write / partially-written outputs instead
        // of blindly zeroing all of them.  Conservatively zeroes all on failure.
        std::vector<std::pair<void*, std::size_t>> output_bufs;
        output_bufs.reserve(cs.staging_outputs.size());
        for (auto& [name, buf] : cs.staging_outputs) {
            if (buf.data != nullptr && buf.size_bytes > 0) {
                output_bufs.emplace_back(buf.data, buf.size_bytes);
            }
        }
        entry.captured_output_zeroes =
            DetectRmwOutputs(cs, shape_key, entry.exec, stream, output_bufs);

        // Item 7: fold the (scratch + RMW-output) zeroing into the graph so replay is
        // a single launch.  Must run after DetectRmwOutputs; keeps the original exec
        // on failure.  The warm-in below then settles whichever exec ends up active.
        FoldZeroingIntoCapturedGraph(cs, stream, program, params, shape_key, entry);

        // Post-capture warm-in replays to settle workspace before first real use.
        for (int i{0}; i < post_warmin; ++i) {
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
        cs.hip_graph_enable = false;
        return false;
    }
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
// vector walk with no per-call map build or string hashing.
bool CheckCapturedPtrsMatch(const CapturedHipGraph& entry,
    const std::vector<void*>& input_ptrs, const std::vector<void*>& output_ptrs, void* scratch_ptr)
{
    if (entry.captured_input_ptrs.size() != input_ptrs.size() ||
        entry.captured_output_ptrs.size() != output_ptrs.size()) {
        return false;
    }
    for (std::size_t i{0}; i < input_ptrs.size(); ++i) {
        if (entry.captured_input_ptrs[i] != input_ptrs[i]) {
            return false;
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
// copies to make.  Warm up to finalize MIGraphX's lazy allocations, then capture
// the graph against those pointers.  Records the current pointers (flat, in
// dbc order) so replay can detect drift.  Returns false (and disables the direct
// path) on capture failure so the caller falls back to eager execution.
bool WarmupAndCaptureHipGraphDirect(ComputeState& cs, hipStream_t stream,
    migraphx::program& program, DirectBindCache& dbc, ShapeKey shape_key,
    bool& enable_flag)
{
    auto& params{dbc.params};

    // ORT-bound output buffers (ptr,bytes) in dbc order, for zeroing + RMW probe.
    std::vector<std::pair<void*, std::size_t>> output_bufs;
    output_bufs.reserve(dbc.outputs.size());
    for (std::size_t i{0}; i < dbc.outputs.size(); ++i) {
        void* ptr{dbc.cur_output_ptrs[i]};
        const std::size_t bytes{dbc.outputs[i].mgx_shape.bytes()};
        if (ptr != nullptr && bytes > 0) {
            output_bufs.emplace_back(ptr, bytes);
        }
    }

    // Zero ORT-bound outputs (NOT inputs -- those carry the caller's data) and
    // scratch so read-modify-write kernels are anchored to a known baseline.
    const auto zero_outputs{[&]() {
        for (const auto& [ptr, bytes] : output_bufs) {
            HIP_CALL_THROW(hipMemsetAsync(ptr, 0, bytes, stream));
        }
    }};
    zero_outputs();
    ZeroScratchFor(cs, shape_key, stream);

    // Pre-capture eager loop to finalize MIGraphX's lazy allocations.
    std::optional<migraphx::arguments> warmup_outputs;
    for (int i{0}; i < kCaptureFinalizeIterations; ++i) {
        std::lock_guard<std::mutex> lock{cs.mutex};
        warmup_outputs = program.run_async(params, stream);
    }
    HIP_CALL_THROW(hipStreamSynchronize(stream));

    // The warmup runs wrote real values into the outputs/scratch; re-zero right
    // before capture so the captured sequence starts from the baseline.
    zero_outputs();
    ZeroScratchFor(cs, shape_key, stream);
    HIP_CALL_THROW(hipStreamSynchronize(stream));

    const std::size_t compiled_batch{
        InferCompiledBatchFromParams(program.get_parameter_shapes(), cs.input_name_indices)};
    const int post_warmin{PostCaptureWarminFor(compiled_batch)};

    auto& entry{cs.hip_graph_cache[shape_key]};
    // Cache the entry on the dbc so the steady-state replay path (and MaterializeExtraOutputs)
    // reaches it directly, without re-searching hip_graph_cache by key every call.
    dbc.graph = &entry;

    // Drop any stale capture for this hash (e.g. a prior staging-path entry for
    // the same shape) before overwriting so its graph/exec are not leaked.
    if (entry.exec != nullptr) {
        (void)hipGraphExecDestroy(entry.exec);
        entry.exec = nullptr;
    }
    if (entry.graph != nullptr) {
        (void)hipGraphDestroy(entry.graph);
        entry.graph = nullptr;
    }

    try {
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
        entry.direct_bind = true;
        entry.captured_input_ptrs = dbc.cur_input_ptrs;
        entry.captured_output_ptrs = dbc.cur_output_ptrs;

        if (const auto it{cs.scratch_bufs.find(shape_key)}; it != cs.scratch_bufs.end()) {
            entry.captured_scratch_ptr = it->second.data;
        } else {
            entry.captured_scratch_ptr = nullptr;
        }

        // Item #1: only zero the outputs that actually depend on their prior
        // contents each replay (detected by replaying twice from two fill
        // patterns), instead of blindly zeroing every output.
        entry.captured_output_zeroes =
            DetectRmwOutputs(cs, shape_key, entry.exec, stream, output_bufs);

        // Post-capture warm-in replays to settle workspace before first real use.
        for (int i{0}; i < post_warmin; ++i) {
            ZeroScratchFor(cs, shape_key, stream);
            for (const auto& [ptr, bytes] : entry.captured_output_zeroes) {
                HIP_CALL_THROW(hipMemsetAsync(ptr, 0, bytes, stream));
            }
            HIP_CALL_THROW(hipGraphLaunch(entry.exec, stream));
        }
        HIP_CALL_THROW(hipStreamSynchronize(stream));

        // Record extra (non-pre-bound) outputs returned by run_async.
        const std::unordered_set<std::size_t> pre_alloc{
            dbc.prog_output_indices.begin(), dbc.prog_output_indices.end()};
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

void CopyInputsToStaging(ComputeState& cs,
    const StagingBindResult& bind,
    const Ort::KernelContext& ctx, hipStream_t stream,
    const DynamicBatchContext& dyn,
    const StaticSeqContext& seq)
{
    // ── Coalesced fast path ───────────────────────────────────────────────────
    // When the input arena is active, there is no padding, and every input is
    // host-resident, gather all inputs into the pinned staging buffer and issue a
    // single H2D for the whole arena -- collapsing the per-input launch overhead
    // that dominates batch-1 many-input models.  Any other case (padding or a
    // device-resident input) falls through to the per-input loop below, which is
    // still correct because each staging buffer's data points into the arena.
    // Both batch padding AND seq padding disqualify the fast path: the memcpy below
    // copies only the real bytes and would skip the pad tail-zero / prefix stride.
    // Driven entirely by the precomputed bind.input_copies plan: no parameter
    // names, std::strings, or map lookups on this hot path.
    const bool batch_no_pad{!dyn.active || dyn.target_batch == dyn.requested_batch};
    const bool seq_no_pad{!seq.active || seq.target_len == seq.real_len};
    const bool no_padding{batch_no_pad && seq_no_pad};
    if (cs.staging_inputs_coalesced && cs.in_staging_host != nullptr && no_padding) {
        // Coalesce eligibility (all inputs host-resident) is determined once and
        // cached on the ComputeState: the caller binds each input to the same memory
        // type on every call, so rescanning N inputs with GetTensorMemoryInfo per
        // inference is wasted work.  FreeStaging resets it on any structural change.
        if (cs.coalesce_residency == ComputeState::CoalesceResidency::kUnknown) {
            auto residency{ComputeState::CoalesceResidency::kAllHost};
            for (const auto& ib : bind.input_copies) {
                const auto mem{ctx.GetInput(ib.ort_index).GetTensorMemoryInfo()};
                if (mem.GetDeviceType() != OrtMemoryInfoDeviceType_CPU) {
                    residency = ComputeState::CoalesceResidency::kHasDevice;
                    break;
                }
            }
            cs.coalesce_residency = residency;
        }

        if (cs.coalesce_residency == ComputeState::CoalesceResidency::kAllHost) {
            char* host_base{static_cast<char*>(cs.in_staging_host)};
            for (const auto& ib : bind.input_copies) {
                // Item 6: reuse the raw pointer captured during the per-call input
                // shape scan; fall back to GetInput only if it was not recorded.
                const void* src{ib.ort_index < cs.cur_input_data.size()
                    ? cs.cur_input_data[ib.ort_index] : nullptr};
                if (src == nullptr) {
                    src = ctx.GetInput(ib.ort_index).GetTensorRawData();
                }
                std::size_t copy_bytes{ib.prog_bytes};
                if (copy_bytes > ib.stage_capacity) {
                    copy_bytes = ib.stage_capacity;
                }
                if (copy_bytes > 0) {
                    std::memcpy(host_base + ib.arena_offset, src, copy_bytes);
                }
            }
            // One transfer for every input.  Copying the whole arena (including the
            // aligned gaps) keeps it a single contiguous DMA; the program only reads
            // the bound rows of each sub-view.
            HIP_CALL_THROW(hipMemcpyAsync(cs.in_arena_dev, cs.in_staging_host,
                cs.in_arena_bytes, hipMemcpyHostToDevice, stream));
            return;
        }
    }

    // Per-input fallback (a device-resident input, or padding is active).  Only the
    // padding branches need the actual ORT shape, so fetch it lazily.
    const bool maybe_pad{(dyn.active && dyn.target_batch > dyn.requested_batch) ||
                         (seq.active && seq.input_axes != nullptr)};
    for (const auto& ib : bind.input_copies) {
        const auto input_tensor{ctx.GetInput(ib.ort_index)};
        const void* src{input_tensor.GetTensorRawData()};
        const auto& prog_lens{ib.prog_lens};

        std::vector<std::int64_t> actual_shape;
        if (maybe_pad) {
            actual_shape = input_tensor.GetTensorTypeAndShapeInfo().GetShape();
        }

        // A batched input arrives with requested_batch rows but the program (and
        // staging) expect target_batch rows; replicate the last row to pad.
        const bool batched{dyn.active && dyn.target_batch > dyn.requested_batch &&
            !prog_lens.empty() && prog_lens.front() == dyn.target_batch &&
            !actual_shape.empty() &&
            static_cast<std::size_t>(actual_shape.front()) == dyn.requested_batch};

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
        } else if (batched) {
            const std::size_t elements_per_row{ProductOf(prog_lens) / dyn.target_batch};
            PadInputTensor(src, ib.staging_data, dyn.requested_batch, dyn.target_batch,
                ib.element_size, elements_per_row, stream);
        } else {
            std::size_t bytes{ib.prog_bytes};
            if (bytes > ib.stage_capacity) {
                bytes = ib.stage_capacity;
            }
            if (bytes > 0) {
                HIP_CALL_THROW(hipMemcpyAsync(ib.staging_data, src, bytes, hipMemcpyDefault, stream));
            }
        }
    }
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
        std::size_t bytes{out_shape.bytes()};

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
        // seq), in which case work on a local copy.  The batch-1 steady state needs
        // neither, so no per-output allocation happens there.
        const std::vector<std::int64_t>* report_shape{&cached_ort_shape};
        std::vector<std::int64_t> sliced_shape;
        if (batch_sliced) {
            const std::size_t row_bytes{bytes / dyn.target_batch};
            sliced_shape = cached_ort_shape;
            sliced_shape.front() = static_cast<std::int64_t>(dyn.requested_batch);
            bytes = row_bytes * dyn.requested_batch;
            report_shape = &sliced_shape;
        } else if (seq_axis >= 0) {
            sliced_shape = cached_ort_shape;
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
    cs.hip_graph_cache.clear();
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
        std::optional<migraphx::arguments> outputs;
        {
            std::lock_guard<std::mutex> lock{cs.mutex};
            outputs = program.run_async(bind.params, stream);
        }
        if (!outputs) {
            return;
        }
        const std::unordered_set<std::size_t> pre_bound{
            bind.prog_output_indices.begin(), bind.prog_output_indices.end()};
        for (std::size_t i{0}; i < outputs->size(); ++i) {
            if (pre_bound.count(i) > 0) {
                continue;
            }
            const auto out{(*outputs)[i]};
            const auto shape{out.get_shape()};
            const auto lens{shape.lengths()};
            std::vector<std::int64_t> ort_shape{lens.begin(), lens.end()};
            auto dst{ctx.GetOutput(i, ort_shape.data(), ort_shape.size())};
            if (shape.bytes() > 0) {
                HIP_CALL_THROW(hipMemcpyWithStream(dst.GetTensorMutableRawData(), out.data(),
                    shape.bytes(), hipMemcpyDeviceToDevice, stream));
            }
        }
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
        if (entry->exec != nullptr) {
            (void)hipGraphExecDestroy(entry->exec);
            entry->exec = nullptr;
        }
        if (entry->graph != nullptr) {
            (void)hipGraphDestroy(entry->graph);
            entry->graph = nullptr;
        }
        entry->captured = false;
        entry->direct_bind = false;
    }

    if (entry != nullptr && entry->captured) {
        // Re-capture if the scratch buffer was reallocated since capture.  The slot
        // pointer was cached on the bind (scratch_bufs entries never move), so this
        // drift check needs no scratch_bufs lookup.
        void* const current_scratch{
            bind.scratch_slot != nullptr ? bind.scratch_slot->data : nullptr};
        if (entry->captured_scratch_ptr != current_scratch) {
            if (entry->exec != nullptr) {
                (void)hipGraphExecDestroy(entry->exec);
                entry->exec = nullptr;
            }
            if (entry->graph != nullptr) {
                (void)hipGraphDestroy(entry->graph);
                entry->graph = nullptr;
            }
            entry->captured = false;
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
        std::optional<migraphx::arguments> outputs;
        {
            std::lock_guard<std::mutex> lock{cs.mutex};
            outputs = program.run_async(dbc.params, stream);
        }
        if (!outputs) {
            return;
        }
        const std::unordered_set<std::size_t> pre_bound{
            dbc.prog_output_indices.begin(), dbc.prog_output_indices.end()};
        for (std::size_t i{0}; i < outputs->size(); ++i) {
            if (pre_bound.count(i) > 0) {
                continue;
            }
            const auto out{(*outputs)[i]};
            const auto shape{out.get_shape()};
            const auto lens{shape.lengths()};
            std::vector<std::int64_t> ort_shape{lens.begin(), lens.end()};
            auto dst{ctx.GetOutput(i, ort_shape.data(), ort_shape.size())};
            if (shape.bytes() > 0) {
                HIP_CALL_THROW(hipMemcpyWithStream(dst.GetTensorMutableRawData(), out.data(),
                    shape.bytes(), hipMemcpyDeviceToDevice, stream));
            }
        }
    }};

    // The captured graph for this shape is cached on the dbc (resolved at capture),
    // so the steady-state replay reaches it directly -- no hip_graph_cache lookup.
    if (CapturedHipGraph* entry{dbc.graph};
        entry != nullptr && entry->captured && entry->direct_bind)
    {
        if (!CheckCapturedPtrsMatch(*entry, dbc.cur_input_ptrs, dbc.cur_output_ptrs,
                current_scratch)) {
            // ORT recycled a buffer under us.  Re-capture, but if drift is
            // sustained give up on this path (permanent eager/staging fallback)
            // rather than re-capturing on every call.
            ++recapture_count;
            if (recapture_count > ComputeState::kMaxDirectRecaptures) {
                enable_flag = false;
                eager_run_with_extras();
                return;
            }
            if (entry->exec != nullptr) {
                (void)hipGraphExecDestroy(entry->exec);
                entry->exec = nullptr;
            }
            if (entry->graph != nullptr) {
                (void)hipGraphDestroy(entry->graph);
                entry->graph = nullptr;
            }
            entry->captured = false;
        } else {
            // Pointers matched: reset the drift counter so only *consecutive*
            // mismatches can trip the permanent fallback above.  No param rebind and
            // no map lookups -- zero scratch via the cached slot + the RMW outputs,
            // then replay.
            recapture_count = 0;
            if (dbc.scratch_slot != nullptr && dbc.scratch_slot->data != nullptr &&
                dbc.scratch_slot->size_bytes > 0) {
                HIP_CALL_THROW(hipMemsetAsync(dbc.scratch_slot->data, 0,
                    dbc.scratch_slot->size_bytes, stream));
            }
            for (const auto& [ptr, bytes] : entry->captured_output_zeroes) {
                HIP_CALL_THROW(hipMemsetAsync(ptr, 0, bytes, stream));
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
