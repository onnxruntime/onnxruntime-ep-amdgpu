// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mgx_mlss_heuristics.h"

#include <algorithm>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace mgx_ep {
namespace {

std::uint64_t SaturatingMultiply(std::uint64_t left, std::uint64_t right) {
    if (left == 0 || right == 0) {
        return 0;
    }
    if (left > std::numeric_limits<std::uint64_t>::max() / right) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return left * right;
}

template <typename Range>
std::uint64_t PositiveProduct(const Range& values, std::size_t begin = 0) {
    if (begin >= static_cast<std::size_t>(values.size())) {
        return 0;
    }
    std::uint64_t result{1};
    for (std::size_t index = begin; index < static_cast<std::size_t>(values.size()); ++index) {
        if (values[index] <= 0) {
            return 0;
        }
        result = SaturatingMultiply(result,
            static_cast<std::uint64_t>(values[index]));
    }
    return result;
}

const ONNX_NAMESPACE::AttributeProto* FindAttribute(
    const ONNX_NAMESPACE::NodeProto& node, std::string_view name) {
    for (const auto& attribute : node.attribute()) {
        if (attribute.name() == name) {
            return &attribute;
        }
    }
    return nullptr;
}

std::uint64_t AttributeProduct(
    const ONNX_NAMESPACE::NodeProto& node, std::string_view name) {
    const auto* attribute{FindAttribute(node, name)};
    if (attribute == nullptr || attribute->ints().empty()) {
        return 1;
    }
    return PositiveProduct(attribute->ints());
}

}  // namespace

MlssGraphFeatures AnalyzeMlssGraph(const ONNX_NAMESPACE::ModelProto& model) {
    MlssGraphFeatures features{};
    const auto& graph{model.graph()};
    std::unordered_map<std::string, const ONNX_NAMESPACE::TensorProto*> tensors;
    tensors.reserve(static_cast<std::size_t>(graph.initializer_size() + graph.node_size()));
    for (const auto& initializer : graph.initializer()) {
        tensors.emplace(initializer.name(), &initializer);
    }
    for (const auto& node : graph.node()) {
        if (node.op_type() != "Constant" || node.output().empty()) {
            continue;
        }
        if (const auto* value{FindAttribute(node, "value")}; value != nullptr && value->has_t()) {
            tensors.emplace(node.output(0), &value->t());
        }
    }

    for (const auto& input : graph.input()) {
        if (!input.type().has_tensor_type() || !input.type().tensor_type().has_shape()) {
            continue;
        }
        const auto& dimensions{input.type().tensor_type().shape().dim()};
        std::vector<std::int64_t> shape;
        shape.reserve(static_cast<std::size_t>(dimensions.size()));
        for (const auto& dimension : dimensions) {
            shape.push_back(dimension.has_dim_value() ? dimension.dim_value() : 0);
        }
        features.input_elements_max =
            std::max(features.input_elements_max, PositiveProduct(shape));
        if (shape.size() >= 2 && shape[1] > 0) {
            features.input_channels_max =
                std::max(features.input_channels_max, static_cast<std::uint64_t>(shape[1]));
        }
        if (shape.size() >= 4) {
            features.input_spatial_max =
                std::max(features.input_spatial_max, PositiveProduct(shape, 2));
        }
    }

    for (const auto& node : graph.node()) {
        if (node.op_type() != "Conv" || node.input_size() < 2) {
            continue;
        }
        const auto tensor{tensors.find(node.input(1))};
        if (tensor == tensors.end() || tensor->second->dims_size() < 3) {
            continue;
        }
        const auto& weight{*tensor->second};
        const std::uint64_t group = [&] {
            const auto* attribute{FindAttribute(node, "group")};
            return attribute != nullptr && attribute->i() > 0
                ? static_cast<std::uint64_t>(attribute->i())
                : std::uint64_t{1};
        }();
        const auto weight_elements{PositiveProduct(weight.dims())};
        const auto kernel_area{PositiveProduct(weight.dims(), 2)};
        const auto output_channels{
            static_cast<std::uint64_t>(std::max<std::int64_t>(weight.dims(0), 0))};
        const auto input_channels_per_group{
            static_cast<std::uint64_t>(std::max<std::int64_t>(weight.dims(1), 0))};
        const auto input_channels{SaturatingMultiply(input_channels_per_group, group)};

        ++features.convolution_count;
        features.convolution_weight_elements += weight_elements;
        features.convolution_weight_elements_max =
            std::max(features.convolution_weight_elements_max, weight_elements);
        features.input_channels_sum += input_channels;
        features.output_channels_sum += output_channels;
        features.channels_max =
            std::max(features.channels_max, std::max(input_channels, output_channels));
        features.kernel_area_sum += kernel_area;
        features.one_by_one_count += kernel_area == 1;
        features.three_by_three_count += kernel_area == 9;
        features.strided_count += AttributeProduct(node, "strides") > 1;
        features.dilated_count += AttributeProduct(node, "dilations") > 1;
        features.grouped_count += group > 1;
        features.depthwise_count += group > 1 && group == input_channels;
        features.fp16_count += weight.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
        features.fp32_count += weight.data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
    }
    return features;
}

}  // namespace mgx_ep
