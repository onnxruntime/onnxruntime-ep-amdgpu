// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include <hip/hip_runtime_api.h>

namespace mgx_ep {

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

// Pad a device input tensor up to a larger batch size by copying the original
// rows and replicating the last row to fill the padding.  The replication uses
// exponential doubling so the number of async copies is O(log N).  All copies
// are device-to-device on the supplied stream.
void PadInputTensor(const void* src_data, void* dst_data,
    std::size_t original_batch, std::size_t padded_batch,
    std::size_t element_size_bytes, std::size_t elements_per_batch,
    hipStream_t stream);

}  // namespace mgx_ep
