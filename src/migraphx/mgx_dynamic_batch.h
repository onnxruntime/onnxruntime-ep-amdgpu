// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime_api.h>

namespace mgx_ep {

// Parse a comma-separated "name:axis" spec (e.g. "input_ids:1,position_ids:1") into
// a map of parameter name -> axis.  A token with no ":axis" suffix defaults to axis
// 1 (the token axis).  Empty / unparseable tokens are skipped.
std::unordered_map<std::string, int> ParseNameAxisSpec(std::string_view spec);

// Parse a comma-separated batch-size specification (e.g. "1,4,16,64") into a
// sorted, de-duplicated list of batch sizes.  Zero-valued and unparseable
// tokens are skipped.  Returns an empty vector when nothing valid is found.
std::vector<std::size_t> ParseCompileBatches(std::string_view spec);

// Generate power-of-two batch sizes from 1 up to the nearest power of two that
// is >= max_batch_size.  E.g. max_batch_size=100 -> {1, 2, 4, 8, 16, 32, 64, 128}.
// Returns an empty vector when max_batch_size is 0.
std::vector<std::size_t> GeneratePowerOfTwoBatchSizes(std::size_t max_batch_size);

// Two-tier batch-size generation:
//   1. If compile_batches_spec is non-empty and parses, use those explicit sizes.
//   2. Otherwise generate power-of-two sizes up to max_batch_size.
std::vector<std::size_t> GenerateCompiledBatchSizes(std::size_t max_batch_size,
    std::string_view compile_batches_spec);

// Find the smallest compiled batch size >= requested_batch.  The input vector
// must be sorted ascending.  Returns 0 when no suitable size exists.
std::size_t FindNearestCompiledBatchSize(std::size_t requested_batch,
    const std::vector<std::size_t>& compiled_batch_sizes);

// Pad a tensor along an inner "sequence" axis: for each of outer_count slices, copy
// real_len elements and zero-fill the remaining (target_len - real_len) elements of
// that slice.  Layout is row-major with the seq axis at `axis`: outer_count =
// product of dims before `axis`, inner_count = product of dims after `axis` (the
// contiguous block per seq position).  dst must be sized for target_len.  All ops
// are on the supplied stream (D2D copy + memset of the pad tail).
void PadSeqTensor(const void* src_data, void* dst_data,
    std::size_t outer_count, std::size_t real_len, std::size_t target_len,
    std::size_t inner_count, std::size_t element_size_bytes,
    hipStream_t stream);

}  // namespace mgx_ep
