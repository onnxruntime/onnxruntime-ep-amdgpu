// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mgx_hip_graph.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <string_view>

#include "hip/utils.h"

namespace mgx_ep {

namespace {

constexpr std::string_view kScratchParam{"scratch"};

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

}  // namespace mgx_ep
