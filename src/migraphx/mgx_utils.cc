// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <charconv>

#include "common/murmurhash3.h"

#include "mgx_utils.h"
#include "ort_trt_int8_cal_table.fbs.h"

namespace mgx_ep {

namespace {

bool ContainsByName(const std::vector<Ort::ConstValueInfo>& value_infos, std::string_view name) {
    const auto filtered_value_infos{GetValueInfos(value_infos)};
    return ranges::find_if(value_infos, [&name](const auto& value_info) {
        return value_info.GetName() == name;
    }) != value_infos.end();
}

bool IsGraphInput(const Ort::ConstGraph& graph, const std::string_view name) {
    return ContainsByName(graph.GetInputs(), name);
}

bool IsGraphInitializer(const Ort::ConstGraph& graph, const std::string_view name) {
    return ContainsByName(graph.GetInitializers(), name);
}

bool CanEvalShapeGeneral(const Ort::ConstGraph& graph, const Ort::ConstNode& node) {
    if (node.GetOperatorType() != "Shape") {
        for (const auto& value_info : GetValueInfos(node.GetInputs())) {
            const auto name{value_info.GetName()};
            if (IsGraphInitializer(graph, name)) {
                continue;
            }
            if (IsGraphInput(graph, name)) {
                return false;
            }
            const auto& [input_node, _]{value_info.GetProducerNode()};
            if (input_node == nullptr) {
                return false;
            }
            if (input_node.GetOperatorType() == "Shape") {
                continue;
            }
            if (CanEvalShapeGeneral(graph, input_node)) {
                continue;
            }
            return false;
        }
    }
    return true;
}
}  // namespace

bool CanEvalNodeArgument(const Ort::ConstGraph& graph, const Ort::ConstNode& node, const std::vector<size_t>& indices ) {
    const auto value_infos{GetValueInfos(node.GetInputs())};
    return ranges::all_of(indices,
        [&](const auto& index) {
            const auto value_info{value_infos[index]};
            const auto name{value_info.GetName()};
            if (IsGraphInitializer(graph, name)) {
                return true;
            }
            if (IsGraphInput(graph, name)) {
                return false;
            }
            const auto& [input_node, _]{value_info.GetProducerNode()};
            if (input_node == nullptr) {
                return false;
            }
            return CanEvalShapeGeneral(graph, input_node);
        });
}

float ConvertSinglePrecisionIEEE754ToFloat(unsigned long value) {
    const auto s{static_cast<int>(value >> 31) & 1};
    const auto e{static_cast<int>((value & 0x7f800000) >> 23) - 127};
    int p{-1};
    double m{};
    for (int i{}; i < 23; ++i) {
        m += (value >> (23 - i - 1) & 1) * pow(2.0, p--);
    }
    return static_cast<float>((s ? -1 : 1) * pow(2.0, e) * (m + 1.0));
}

bool ReadDynamicRange(const fs::path& path, const bool is_native_calibration_table, Map<float>& dynamic_ranges)
try {
    const auto length{fs::file_size(path)};
    std::ifstream ifs{path, std::ios::binary | std::ios::in};
    if (!ifs) {
        return false;
    }
    const auto data{std::make_unique<char[]>(length)};
    ifs.read(data.get(), static_cast<std::streamsize>(length));
    ifs.close();
    if (is_native_calibration_table) {
        // read MIGraphX generated calibration table
        std::istringstream ss{data.get()};
        std::string version;
        std::getline(ss, version, ':');
        auto pos{version.find("MGX-")};
        if (pos == std::string::npos) {
            // check for the native TensorRT generated calibration table
            pos = version.find("TRT-");
        }
        if (pos != std::string::npos) {
            std::string line;
            while (std::getline(ss, line)) {
                std::istringstream ls{line};
                std::string token;
                std::getline(ls, token, ':');
                std::string name{token};
                std::getline(ls, token, ':');
                unsigned long scale_int{std::strtoul(token.c_str(), nullptr, 16)};
                float scale_float{ConvertSinglePrecisionIEEE754ToFloat(scale_int)};
                dynamic_ranges[name] = scale_float * 127.0f;
            }
        } else {
            throw std::runtime_error{path.string() + ": not a MIGraphX-generated calibration table"};
        }
    } else {
        // read ONNXRuntime generated calibration table
        const auto flat_table{flatbuffers::GetRoot<CalTableFlatBuffers::TrtTable>(data.get())};
        const auto flat_dict{flat_table->dict()};
        for (size_t i = 0; i < flat_dict->size(); ++i) {
            const auto index{static_cast<flatbuffers::uoffset_t>(i)};
            const auto& elem{flat_dict->Get(index)};
            dynamic_ranges[elem->key()->str()] = std::stof(elem->value()->str());
        }
    }
    return true;
} catch (const std::exception&) {
    return false;
}

std::string GenerateGraphId(const Ort::ConstGraph& graph)
{
    hash::Value value = {0, 0, 0, 0};

    if (fs::path path{graph.GetModelPath()}; path.has_filename()) {
        const auto model_name{path.filename().string()};
        auto repeated_model_name{model_name};
        for (auto i{model_name.length()}; i > 0 && i < 500; i += model_name.length()) {
            repeated_model_name += model_name;
        }
        hash::Hash(value, repeated_model_name);
    }
    const auto inputs{GetValueInfos(graph.GetInputs())};
    for (const auto& node : inputs) {
        hash::Hash(value, node.GetName());
    }
    hash::Hash(value, inputs.size());

    for (const auto& node : GetKahnsVariantTopologicalSortedNodes(graph)) {
        auto node_outputs{GetValueInfos(node.GetOutputs())};
        hash::Hash(value, node_outputs.size());
        for (const auto& output : node_outputs) {
            hash::Hash(value, output.GetName());
        }
        auto node_inputs{GetValueInfos(node.GetInputs())};
        hash::Hash(value, node_inputs.size());
        for (const auto& input : node_inputs) {
            hash::Hash(value, input.GetName());
            hash::Hash(value, input.TypeInfo().GetTensorTypeAndShapeInfo().GetShape());
        }
    }

#if defined(_WIN32)
    hash::Hash(value, "WINDOWS");
#elif defined(__linux__)
    hash::Hash(value, "LINUX");
#endif

    return hash::ToHex(value);
}

}  // namespace mgx_ep
