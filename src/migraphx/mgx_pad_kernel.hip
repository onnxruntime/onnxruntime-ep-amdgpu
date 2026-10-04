// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mgx_pad_kernel.h"

#include <hip/hip_runtime.h>

namespace mgx_ep {

namespace {

constexpr unsigned kPadBlockSize{256};
// Blocks per input. Rows are small (a model row is typically hundreds of bytes to a
// few KB), so a handful of blocks saturates one row's copy; blocks that find no work
// retire immediately, which is cheaper than sizing the grid per descriptor.
constexpr unsigned kPadBlocksPerInput{8};

// blockIdx.y selects the input, blockIdx.x + threadIdx.x stride over its row.  The
// source row is read once per pad row and stays in cache, and the writes are
// contiguous, so this is bandwidth-bound on a trivial amount of data.
__global__ void PadReplicateRowsKernel(const PadRowDesc* __restrict__ descs)
{
    const PadRowDesc d{descs[blockIdx.y]};
    if (d.dst == nullptr || d.src == nullptr || d.row_bytes == 0 || d.real_rows == 0 ||
        d.real_rows > d.target_rows) {
        return;
    }

    const std::size_t tid{static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x};
    const std::size_t stride{static_cast<std::size_t>(gridDim.x) * blockDim.x};
    // Uniform for this block (one descriptor per blockIdx.y).  Arena slots are
    // 256B-aligned and each packed source is 16B-aligned; only the row length and
    // the two bases need checking.
    const bool vec{(d.row_bytes & 15u) == 0u &&
        (reinterpret_cast<uintptr_t>(d.dst) & 15u) == 0u &&
        (reinterpret_cast<uintptr_t>(d.src) & 15u) == 0u};

    const auto copy_row{[&](char* dst_row, const char* src_row) {
        if (vec) {
            const auto* __restrict__ src_vec{reinterpret_cast<const uint4*>(src_row)};
            auto* __restrict__ dst_vec{reinterpret_cast<uint4*>(dst_row)};
            const std::size_t vec_per_row{d.row_bytes >> 4};
            for (std::size_t v{tid}; v < vec_per_row; v += stride) {
                dst_vec[v] = src_vec[v];
            }
        } else {
            for (std::size_t b{tid}; b < d.row_bytes; b += stride) {
                dst_row[b] = src_row[b];
            }
        }
    }};

    for (std::uint32_t r{0}; r < d.real_rows; ++r) {
        copy_row(d.dst + static_cast<std::size_t>(r) * d.row_bytes,
            d.src + static_cast<std::size_t>(r) * d.row_bytes);
    }
    const char* const last{d.src + static_cast<std::size_t>(d.real_rows - 1) * d.row_bytes};
    for (std::uint32_t r{d.real_rows}; r < d.target_rows; ++r) {
        copy_row(d.dst + static_cast<std::size_t>(r) * d.row_bytes, last);
    }
}

}  // namespace

hipError_t LaunchPadReplicateRows(const PadRowDesc* descs, unsigned desc_count,
    hipStream_t stream)
{
    if (descs == nullptr || desc_count == 0) {
        return hipSuccess;
    }
    // gridDim.y caps at 65535; an input count anywhere near that is not a shape this
    // EP can bind, but refuse rather than silently truncate the fill.
    if (desc_count > 65535u) {
        return hipErrorInvalidConfiguration;
    }
    PadReplicateRowsKernel<<<dim3{kPadBlocksPerInput, desc_count, 1},
        dim3{kPadBlockSize, 1, 1}, 0, stream>>>(descs);
    return hipGetLastError();
}

}  // namespace mgx_ep
