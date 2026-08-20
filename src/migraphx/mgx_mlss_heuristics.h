// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string_view>

#include "onnx/onnx_pb.h"

namespace mgx_ep {

struct MlssGraphFeatures {
    std::uint64_t convolution_count{};
    std::uint64_t convolution_weight_elements{};
    std::uint64_t convolution_weight_elements_max{};
    std::uint64_t input_channels_sum{};
    std::uint64_t output_channels_sum{};
    std::uint64_t channels_max{};
    std::uint64_t kernel_area_sum{};
    std::uint64_t one_by_one_count{};
    std::uint64_t three_by_three_count{};
    std::uint64_t strided_count{};
    std::uint64_t dilated_count{};
    std::uint64_t grouped_count{};
    std::uint64_t depthwise_count{};
    std::uint64_t fp16_count{};
    std::uint64_t fp32_count{};
    std::uint64_t input_elements_max{};
    std::uint64_t input_spatial_max{};
    std::uint64_t input_channels_max{};
};

MlssGraphFeatures AnalyzeMlssGraph(const ONNX_NAMESPACE::ModelProto& model);

constexpr bool IsMlssArchPrefix(std::string_view value, std::string_view prefix) {
    return value.substr(0, prefix.size()) == prefix;
}

// The gfx115x thresholds were selected from grouped validation of STXH model
// measurements. Only leaves containing no measured >=5% regressions are
// enabled; unknown graph shapes remain on the MIGraphX default path.
constexpr bool ShouldForceMlssConv(std::string_view gfx, const MlssGraphFeatures& features) {
    if (features.convolution_count == 0) {
        return false;
    }
    // Preserve the existing gfx1200/gfx1201 policy.
    if (IsMlssArchPrefix(gfx, "gfx1200") || IsMlssArchPrefix(gfx, "gfx1201")) {
        return true;
    }
    if (!IsMlssArchPrefix(gfx, "gfx1150") && !IsMlssArchPrefix(gfx, "gfx1151")) {
        return false;
    }

    const bool fp32_low_pointwise =
        features.fp32_count == features.convolution_count &&
        features.input_elements_max > 2 &&
        features.one_by_one_count * 60 <= features.convolution_count * 19;
    return fp32_low_pointwise;
}

namespace detail {

constexpr MlssGraphFeatures TestFeatures(std::uint64_t convolutions,
                                         std::uint64_t one_by_one,
                                         std::uint64_t fp16,
                                         std::uint64_t fp32,
                                         std::uint64_t input_channels,
                                         std::uint64_t three_by_three = 0) {
    MlssGraphFeatures features{};
    features.convolution_count = convolutions;
    features.one_by_one_count = one_by_one;
    features.fp16_count = fp16;
    features.fp32_count = fp32;
    features.input_channels_max = input_channels;
    features.input_elements_max = 1024;
    features.three_by_three_count = three_by_three;
    return features;
}

static_assert(ShouldForceMlssConv("gfx1151", TestFeatures(4, 1, 0, 4, 3)));
static_assert(!ShouldForceMlssConv("gfx1151", TestFeatures(3, 1, 0, 3, 3)));
static_assert(!ShouldForceMlssConv("gfx1150", TestFeatures(2, 0, 2, 0, 949)));
static_assert(!ShouldForceMlssConv("gfx1150", TestFeatures(2, 0, 2, 0, 936)));
static_assert(!ShouldForceMlssConv("gfx1151", TestFeatures(10, 4, 0, 10, 3, 6)));
static_assert(!ShouldForceMlssConv("gfx1151", TestFeatures(10, 4, 0, 10, 3, 5)));
static_assert(!ShouldForceMlssConv("gfx1151", TestFeatures(10, 0, 5, 5, 1024)));
static_assert([] {
    auto features = TestFeatures(4, 1, 0, 4, 3);
    features.input_elements_max = 0;
    return !ShouldForceMlssConv("gfx1151", features);
}());
static_assert(ShouldForceMlssConv("gfx1201", TestFeatures(1, 0, 0, 0, 0)));
static_assert(!ShouldForceMlssConv("gfx1100", TestFeatures(10, 0, 0, 0, 0)));

}  // namespace detail
}  // namespace mgx_ep
