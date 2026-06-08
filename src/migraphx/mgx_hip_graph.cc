// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mgx_hip_graph.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <mutex>
#include <optional>
#include <string_view>
#include <unordered_set>

#include "hip/utils.h"

namespace mgx_ep {

namespace {

constexpr std::string_view kScratchParam{"scratch"};

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
    const std::vector<CapturedHipGraph::ExtraOutput>& extras)
{
    for (const auto& extra : extras) {
        auto ort_shape{extra.ort_shape};
        auto output_tensor{ctx.GetOutput(extra.output_index, ort_shape.data(), ort_shape.size())};
        void* dst{output_tensor.GetTensorMutableRawData()};
        if (extra.bytes > 0) {
            HIP_CALL_THROW(hipMemcpyWithStream(dst, extra.gpu_data, extra.bytes,
                hipMemcpyDeviceToDevice, stream));
        }
    }
}

// Warm up, then capture a hipGraph for the currently bound params.  Returns false
// (and disables hipGraph on the state) if capture fails so callers fall back to
// eager execution.
bool WarmupAndCaptureHipGraph(ComputeState& cs, hipStream_t stream,
    migraphx::program& program, migraphx::program_parameters& params,
    const std::vector<std::size_t>& prog_output_indices, const std::string& shape_hash)
{
    // Zero staging outputs and scratch so warmup-derived bytes are not baked into
    // the capture.
    for (auto& [name, buf] : cs.staging_outputs) {
        if (buf.data != nullptr) {
            HIP_CALL_THROW(hipMemsetAsync(buf.data, 0, buf.size_bytes, stream));
        }
    }
    ZeroScratchFor(cs, shape_hash, stream);

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

    auto& entry{cs.hip_graph_cache[shape_hash]};

    // Re-zero scratch right before capture so the captured kernel sequence is
    // anchored to a known baseline (the warmup loop just dirtied it).
    ZeroScratchFor(cs, shape_hash, stream);
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

        // Record the scratch pointer baked into the graph for drift detection.
        if (const auto it{cs.scratch_bufs.find(shape_hash)}; it != cs.scratch_bufs.end()) {
            entry.captured_scratch_ptr = it->second.data;
        } else {
            entry.captured_scratch_ptr = nullptr;
        }

        // Record the output buffers to zero before every replay (kernels may
        // read-modify-write their output).
        entry.captured_output_zeroes.clear();
        entry.captured_output_zeroes.reserve(cs.staging_outputs.size());
        for (auto& [name, buf] : cs.staging_outputs) {
            if (buf.data != nullptr) {
                entry.captured_output_zeroes.emplace_back(buf.data, buf.size_bytes);
            }
        }

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
void ReplayHipGraph(ComputeState& cs, hipStream_t stream,
    CapturedHipGraph& entry, const std::string& shape_hash)
{
    ZeroScratchFor(cs, shape_hash, stream);
    for (const auto& [ptr, bytes] : entry.captured_output_zeroes) {
        HIP_CALL_THROW(hipMemsetAsync(ptr, 0, bytes, stream));
    }
    HIP_CALL_THROW(hipGraphLaunch(entry.exec, stream));
}

}  // namespace

int ComputeOutputIndex(std::string_view name) {
    constexpr std::string_view prefix{"#output_"};
    const auto pos{name.find(prefix)};
    if (pos == std::string_view::npos) {
        return -1;
    }
    const auto digits{name.substr(pos + prefix.size())};
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

std::optional<ScratchBindInfo> GetOrAllocScratch(ComputeState& cs,
    const migraphx::program_parameter_shapes& param_shapes,
    const std::string& shape_hash, hipStream_t stream)
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

    const auto scratch_shape{param_shapes["scratch"]};
    const std::size_t needed_bytes{scratch_shape.bytes()};

    auto& slot{cs.scratch_bufs[shape_hash]};

    // (Re)allocate when missing or the required size has grown.  Plain hipMalloc
    // (not hipMallocAsync) keeps these allocations out of the stream-ordered pool.
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
        // Zero on fresh allocation only; callers re-zero via ZeroScratchFor.
        HIP_CALL_THROW(hipMemsetAsync(slot.data, 0, slot.size_bytes, stream));
    } else {
        slot.shape = scratch_shape;
    }

    return ScratchBindInfo{slot.data, slot.shape};
}

void ZeroScratchFor(ComputeState& cs, const std::string& shape_hash, hipStream_t stream) {
    const auto it{cs.scratch_bufs.find(shape_hash)};
    if (it == cs.scratch_bufs.end() || it->second.data == nullptr || it->second.size_bytes == 0) {
        return;
    }
    HIP_CALL_THROW(hipMemsetAsync(it->second.data, 0, it->second.size_bytes, stream));
}

void AllocateStaging(ComputeState& cs,
    const migraphx::program_parameter_shapes& param_shapes, hipStream_t stream)
{
    if (cs.staging_allocated) {
        return;
    }

    const auto alloc_buffer{[&stream](const migraphx::shape& shape) -> StagingBuffer {
        const std::size_t bytes{shape.bytes()};
        void* ptr{nullptr};
        HIP_CALL_THROW(hipMalloc(&ptr, bytes));
        HIP_CALL_THROW(hipMemsetAsync(ptr, 0, bytes, stream));
        return StagingBuffer{ptr, bytes, shape};
    }};

    for (const auto& name : param_shapes.names()) {
        const std::string param_name{name};
        if (std::string_view{name} == kScratchParam) {
            continue;  // scratch is owned separately
        }
        if (cs.input_name_indices.count(param_name) > 0) {
            cs.staging_inputs.emplace(param_name, alloc_buffer(param_shapes[name]));
        } else if (ComputeOutputIndex(name) != -1) {
            cs.staging_outputs.emplace(param_name, alloc_buffer(param_shapes[name]));
        }
    }

    HIP_CALL_THROW(hipStreamSynchronize(stream));
    cs.staging_allocated = true;
}

void CopyInputsToStaging(ComputeState& cs,
    const migraphx::program_parameter_shapes& param_shapes,
    const Ort::KernelContext& ctx, hipStream_t stream)
{
    for (const auto& name : param_shapes.names()) {
        const std::string param_name{name};
        const auto idx_it{cs.input_name_indices.find(param_name)};
        if (idx_it == cs.input_name_indices.end()) {
            continue;
        }
        const auto stage_it{cs.staging_inputs.find(param_name)};
        if (stage_it == cs.staging_inputs.end()) {
            continue;
        }
        const auto& stage{stage_it->second};
        const auto input_tensor{ctx.GetInput(idx_it->second)};
        const void* src{input_tensor.GetTensorRawData()};
        std::size_t bytes{param_shapes[name].bytes()};
        if (bytes > stage.size_bytes) {
            bytes = stage.size_bytes;
        }
        if (bytes > 0) {
            HIP_CALL_THROW(hipMemcpyAsync(stage.data, src, bytes, hipMemcpyDefault, stream));
        }
    }
}

StagingBindResult BindStagingParams(ComputeState& cs,
    const migraphx::program_parameter_shapes& param_shapes,
    const std::string& shape_hash, hipStream_t stream)
{
    StagingBindResult result;
    for (const auto& name : param_shapes.names()) {
        const std::string param_name{name};
        if (cs.input_name_indices.count(param_name) > 0) {
            const auto stage_it{cs.staging_inputs.find(param_name)};
            if (stage_it == cs.staging_inputs.end()) {
                continue;
            }
            result.params.add(name, migraphx::argument{param_shapes[name], stage_it->second.data});
        } else if (std::string_view{name} == kScratchParam) {
            if (const auto scratch{GetOrAllocScratch(cs, param_shapes, shape_hash, stream)}) {
                result.params.add(name, migraphx::argument{scratch->shape, scratch->ptr});
            }
        } else if (const auto oi{ComputeOutputIndex(name)}; oi != -1) {
            const auto stage_it{cs.staging_outputs.find(param_name)};
            if (stage_it == cs.staging_outputs.end()) {
                continue;
            }
            result.params.add(name, migraphx::argument{param_shapes[name], stage_it->second.data});
            result.prog_output_indices.push_back(static_cast<std::size_t>(oi));
            result.bound_output_names.push_back(param_name);
        }
    }
    return result;
}

void CopyStagingOutputsToOrt(ComputeState& cs, const StagingBindResult& bind,
    const Ort::KernelContext& ctx, hipStream_t stream)
{
    for (std::size_t i{}; i < bind.prog_output_indices.size() && i < bind.bound_output_names.size(); ++i) {
        const auto oi{bind.prog_output_indices[i]};
        const auto stage_it{cs.staging_outputs.find(bind.bound_output_names[i])};
        if (stage_it == cs.staging_outputs.end()) {
            continue;
        }
        const auto& stage{stage_it->second};
        const auto lengths{stage.shape.lengths()};
        std::vector<std::int64_t> ort_shape{lengths.begin(), lengths.end()};
        auto output_tensor{ctx.GetOutput(oi, ort_shape.data(), ort_shape.size())};
        void* dst{output_tensor.GetTensorMutableRawData()};
        if (stage.size_bytes > 0) {
            HIP_CALL_THROW(hipMemcpyAsync(dst, stage.data, stage.size_bytes, hipMemcpyDefault, stream));
        }
    }
}

void FreeStaging(ComputeState& cs) {
    for (auto& [name, buf] : cs.staging_inputs) {
        if (buf.data != nullptr) {
            (void)hipFree(buf.data);
            buf.data = nullptr;
        }
    }
    for (auto& [name, buf] : cs.staging_outputs) {
        if (buf.data != nullptr) {
            (void)hipFree(buf.data);
            buf.data = nullptr;
        }
    }
    cs.staging_inputs.clear();
    cs.staging_outputs.clear();
    cs.staging_allocated = false;
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
    migraphx::program_parameters& params,
    const std::vector<std::size_t>& prog_output_indices,
    const std::string& shape_hash)
{
    const auto eager_run{[&]() {
        std::lock_guard<std::mutex> lock{cs.mutex};
        program.run_async(params, stream);
    }};

    if (!cs.hip_graph_enable) {
        eager_run();
        return;
    }

    if (const auto it{cs.hip_graph_cache.find(shape_hash)};
        it != cs.hip_graph_cache.end() && it->second.captured)
    {
        // Re-capture if the scratch buffer was reallocated since capture.
        const auto scratch_it{cs.scratch_bufs.find(shape_hash)};
        void* current_scratch{scratch_it != cs.scratch_bufs.end() ? scratch_it->second.data : nullptr};
        if (it->second.captured_scratch_ptr != current_scratch) {
            if (it->second.exec != nullptr) {
                (void)hipGraphExecDestroy(it->second.exec);
                it->second.exec = nullptr;
            }
            if (it->second.graph != nullptr) {
                (void)hipGraphDestroy(it->second.graph);
                it->second.graph = nullptr;
            }
            it->second.captured = false;
        } else {
            ReplayHipGraph(cs, stream, it->second, shape_hash);
            MaterializeExtraOutputs(ctx, stream, it->second.extra_outputs);
            return;
        }
    }

    if (!WarmupAndCaptureHipGraph(cs, stream, program, params, prog_output_indices, shape_hash)) {
        eager_run();
        return;
    }
    MaterializeExtraOutputs(ctx, stream, cs.hip_graph_cache.at(shape_hash).extra_outputs);
}

}  // namespace mgx_ep
