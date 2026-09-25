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
    std::uint64_t node_count{};
    std::uint64_t input_elements_max{};
    std::uint64_t input_spatial_max{};
    std::uint64_t input_channels_max{};
};

MlssGraphFeatures AnalyzeMlssGraph(const ONNX_NAMESPACE::ModelProto& model);

inline constexpr bool kEnableGfx115xMlssHeuristics = false;

constexpr bool IsMlssArchPrefix(std::string_view value, std::string_view prefix) {
    return value.substr(0, prefix.size()) == prefix;
}

// gfx1200/gfx1201: always on. gfx1150: graph heuristics. All other gfx
// (gfx1100, gfx1151, ...) stay off until a policy is added.
constexpr bool ShouldForceMlssConv(std::string_view gfx, const MlssGraphFeatures& features) {
    if (IsMlssArchPrefix(gfx, "gfx1200") || IsMlssArchPrefix(gfx, "gfx1201")) {
        return true;
    }
    if (!IsMlssArchPrefix(gfx, "gfx1150") || features.convolution_count == 0) {
        return false;
    }
    if (features.input_elements_max <= 32) {
        return false;
    }
    const auto grouped_nondw{
        features.grouped_count > features.depthwise_count
            ? features.grouped_count - features.depthwise_count
            : std::uint64_t{0}};
    if (features.fp32_count == features.convolution_count) {
        if (features.convolution_count < 2) {
            return false;
        }
        if (grouped_nondw * 20 > features.convolution_count * 3) {
            return false;
        }
        return features.one_by_one_count * 5 <= features.convolution_count * 4;
    }
    if (features.fp16_count != features.convolution_count ||
        features.node_count == 0 ||
        grouped_nondw != 0 ||
        features.depthwise_count != 0) {
        return false;
    }
    if (features.convolution_count >= 50 &&
        features.one_by_one_count * 2 <= features.convolution_count &&
        features.strided_count * 50 <= features.convolution_count &&
        features.convolution_count * 25 >= features.node_count * 7) {
        return true;
    }
    if (features.one_by_one_count != 0 || features.strided_count != 0) {
        return false;
    }
    if (features.convolution_count >= 30 &&
        features.convolution_count * 25 <= features.node_count * 2) {
        return true;
    }
    return features.convolution_count >= 20 &&
           features.convolution_count * 20 >= features.node_count * 3 &&
           features.convolution_count * 4 <= features.node_count;
}

// Applies the rollout gate to the tested policy
constexpr bool ShouldAutoForceMlssConv(std::string_view gfx, const MlssGraphFeatures& features) {
    if ((IsMlssArchPrefix(gfx, "gfx1150") || IsMlssArchPrefix(gfx, "gfx1151")) &&
        !kEnableGfx115xMlssHeuristics) {
        return false;
    }
    return ShouldForceMlssConv(gfx, features);
}

namespace detail {

constexpr MlssGraphFeatures TestFeatures(std::uint64_t convolutions,
                                         std::uint64_t one_by_one,
                                         std::uint64_t fp16,
                                         std::uint64_t fp32,
                                         std::uint64_t grouped = 0,
                                         std::uint64_t nodes = 0,
                                         std::uint64_t strided = 0) {
    MlssGraphFeatures features{};
    features.convolution_count = convolutions;
    features.one_by_one_count = one_by_one;
    features.fp16_count = fp16;
    features.fp32_count = fp32;
    features.grouped_count = grouped;
    features.node_count = nodes;
    features.strided_count = strided;
    features.input_elements_max = 1024;
    return features;
}

static_assert(ShouldForceMlssConv("gfx1150", TestFeatures(2, 0, 0, 2)));
static_assert(ShouldForceMlssConv("gfx1150", TestFeatures(10, 4, 0, 10)));
static_assert(ShouldForceMlssConv("gfx1150", TestFeatures(10, 8, 0, 10)));
static_assert(ShouldForceMlssConv("gfx1150", TestFeatures(57, 37, 0, 57)));
static_assert(ShouldForceMlssConv("gfx1150", TestFeatures(30, 9, 0, 30, 4)));
static_assert(!ShouldForceMlssConv("gfx1150", TestFeatures(1, 0, 0, 1)));
static_assert(!ShouldForceMlssConv("gfx1150", TestFeatures(2, 0, 2, 0)));
static_assert(!ShouldForceMlssConv("gfx1150", TestFeatures(10, 9, 0, 10)));
static_assert(!ShouldForceMlssConv("gfx1150", TestFeatures(5, 0, 0, 5, 3)));
static_assert(ShouldForceMlssConv("gfx1150", TestFeatures(299, 144, 299, 0, 0, 923, 1)));
static_assert(!ShouldForceMlssConv("gfx1150", TestFeatures(263, 108, 263, 0, 0, 1154, 1)));
static_assert(ShouldForceMlssConv("gfx1150", TestFeatures(37, 0, 37, 0, 0, 698, 0)));
static_assert(!ShouldForceMlssConv("gfx1150", TestFeatures(37, 0, 37, 0, 0, 250, 0)));
static_assert(ShouldForceMlssConv("gfx1150", TestFeatures(26, 0, 26, 0, 0, 117, 0)));
static_assert(!ShouldForceMlssConv("gfx1150", TestFeatures(33, 0, 33, 0, 0, 105, 0)));
static_assert(!ShouldForceMlssConv("gfx1151", TestFeatures(10, 0, 0, 10)));
static_assert(ShouldForceMlssConv("gfx1200", TestFeatures(0, 0, 0, 0)));
static_assert(ShouldForceMlssConv("gfx1201", TestFeatures(1, 0, 0, 0)));
static_assert(!ShouldForceMlssConv("gfx1100", TestFeatures(10, 0, 0, 0)));
static_assert([] {
    auto features = TestFeatures(4, 1, 0, 4);
    features.input_elements_max = 3;
    return !ShouldForceMlssConv("gfx1150", features);
}());

}  // namespace detail
}  // namespace mgx_ep
