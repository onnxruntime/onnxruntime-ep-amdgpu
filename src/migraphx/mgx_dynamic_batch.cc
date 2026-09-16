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

std::unordered_map<std::string, int> ParseNameAxisSpec(std::string_view spec) {
    std::unordered_map<std::string, int> result;
    if (spec.empty()) {
        return result;
    }

    std::istringstream iss{std::string{spec}};
    std::string token;
    while (std::getline(iss, token, ',')) {
        if (token.empty()) {
            continue;
        }
        // Trim surrounding whitespace so "a:1, b: 1" parses cleanly instead of silently
        // dropping entries on a mismatched name (" b") or unparseable axis (" 1").
        const auto trim{[](std::string s) {
            const auto b{s.find_first_not_of(" \t\r\n")};
            if (b == std::string::npos) return std::string{};
            return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
        }};
        const auto colon{token.find(':')};
        std::string name{trim(colon == std::string::npos ? token : token.substr(0, colon))};
        if (name.empty()) {
            continue;
        }
        int axis{1};  // default token axis
        if (colon != std::string::npos) {
            const std::string axis_str{trim(token.substr(colon + 1))};
            std::size_t parsed{};
            const auto* begin{axis_str.data()};
            const auto* end{axis_str.data() + axis_str.size()};
            if (const auto [ptr, ec]{std::from_chars(begin, end, parsed)};
                ec != std::errc{} || ptr != end) {
                continue;  // unparseable axis -> skip this entry
            }
            axis = static_cast<int>(parsed);
        }
        result.emplace(std::move(name), axis);
    }
    return result;
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

void PadSeqTensor(const void* src_data, void* dst_data,
    std::size_t outer_count, std::size_t real_len, std::size_t target_len,
    std::size_t inner_count, std::size_t element_size_bytes,
    hipStream_t stream)
{
    if (target_len < real_len) {
        return;  // caller guarantees target_len >= real_len; nothing sensible to do otherwise
    }
    const std::size_t elem{element_size_bytes};
    const std::size_t src_slice_bytes{real_len * inner_count * elem};
    const std::size_t dst_slice_bytes{target_len * inner_count * elem};
    const std::size_t real_bytes{real_len * inner_count * elem};
    const std::size_t pad_bytes{(target_len - real_len) * inner_count * elem};

    const char* src{static_cast<const char*>(src_data)};
    char* dst{static_cast<char*>(dst_data)};

    // Fast path: a single outer slice (batch=1) is one contiguous copy + one memset.
    // The general path handles outer_count > 1 with per-slice copy/zero so that the
    // real tokens of each slice land at the start of that slice's padded region.
    for (std::size_t o{0}; o < outer_count; ++o) {
        if (real_bytes > 0) {
            HIP_CALL_THROW(hipMemcpyAsync(dst + o * dst_slice_bytes, src + o * src_slice_bytes,
                real_bytes, hipMemcpyDefault, stream));
        }
        if (pad_bytes > 0) {
            HIP_CALL_THROW(hipMemsetAsync(dst + o * dst_slice_bytes + real_bytes, 0,
                pad_bytes, stream));
        }
    }
}

}  // namespace mgx_ep
