// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mgx_dynamic_batch.h"

#include <algorithm>
#include <charconv>
#include <sstream>

#include "hip/utils.h"

namespace mgx_ep {

std::vector<std::size_t> ParseCompileBatches(std::string_view spec) {
    std::vector<std::size_t> batch_sizes;
    if (spec.empty()) {
        return batch_sizes;
    }

    std::istringstream iss{std::string{spec}};
    std::string token;
    while (std::getline(iss, token, ',')) {
        if (token.empty()) {
            continue;
        }
        std::size_t value{};
        const auto* begin{token.data()};
        const auto* end{token.data() + token.size()};
        const auto [ptr, ec]{std::from_chars(begin, end, value)};
        if (ec != std::errc{} || ptr != end || value == 0) {
            // Skip zero-valued or unparseable entries.
            continue;
        }
        batch_sizes.push_back(value);
    }

    std::sort(batch_sizes.begin(), batch_sizes.end());
    batch_sizes.erase(std::unique(batch_sizes.begin(), batch_sizes.end()), batch_sizes.end());
    return batch_sizes;
}

std::vector<std::size_t> GeneratePowerOfTwoBatchSizes(std::size_t max_batch_size) {
    std::vector<std::size_t> batch_sizes;
    if (max_batch_size == 0) {
        return batch_sizes;
    }

    std::size_t target{1};
    while (target < max_batch_size) {
        target *= 2;
    }
    for (std::size_t bs{1}; bs <= target; bs *= 2) {
        batch_sizes.push_back(bs);
    }
    return batch_sizes;
}

std::vector<std::size_t> GenerateCompiledBatchSizes(std::size_t max_batch_size,
    std::string_view compile_batches_spec)
{
    if (!compile_batches_spec.empty()) {
        if (auto batch_sizes{ParseCompileBatches(compile_batches_spec)}; !batch_sizes.empty()) {
            return batch_sizes;
        }
    }
    return GeneratePowerOfTwoBatchSizes(max_batch_size);
}

std::size_t FindNearestCompiledBatchSize(std::size_t requested_batch,
    const std::vector<std::size_t>& compiled_batch_sizes)
{
    for (const auto bs : compiled_batch_sizes) {
        if (bs >= requested_batch) {
            return bs;
        }
    }
    return 0;
}

void PadInputTensor(const void* src_data, void* dst_data,
    std::size_t original_batch, std::size_t padded_batch,
    std::size_t element_size_bytes, std::size_t elements_per_batch,
    hipStream_t stream)
{
    const std::size_t bytes_per_batch{element_size_bytes * elements_per_batch};

    // Copy the original rows.
    HIP_CALL_THROW(hipMemcpyAsync(dst_data, src_data, original_batch * bytes_per_batch,
        hipMemcpyDeviceToDevice, stream));

    // Replicate the last row to fill the padding using exponential doubling so
    // the number of copies is O(log N) rather than O(N).
    if (original_batch > 0 && padded_batch > original_batch) {
        const char* last_batch{static_cast<const char*>(src_data) +
            (original_batch - 1) * bytes_per_batch};
        char* pad_start{static_cast<char*>(dst_data) + original_batch * bytes_per_batch};
        const std::size_t slots_to_fill{padded_batch - original_batch};

        HIP_CALL_THROW(hipMemcpyAsync(pad_start, last_batch, bytes_per_batch,
            hipMemcpyDeviceToDevice, stream));
        std::size_t filled{1};
        while (filled < slots_to_fill) {
            const std::size_t chunk{std::min(filled, slots_to_fill - filled)};
            HIP_CALL_THROW(hipMemcpyAsync(pad_start + filled * bytes_per_batch, pad_start,
                chunk * bytes_per_batch, hipMemcpyDeviceToDevice, stream));
            filled += chunk;
        }
    }
}

}  // namespace mgx_ep
