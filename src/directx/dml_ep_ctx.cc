// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "dml_ep_ctx.h"

#include <array>
#include <stdexcept>
#include <vector>

namespace dml_ep {

Ort::Status EpContextNodeHelper::CreateEpContextNode(std::string_view serialized_graph,
    int64_t embed_mode, std::string_view compute_capability, const fs::path& model_path,
    std::string_view node_name_prefix, OrtNode*& ep_context_node) const
{
    std::vector<std::string> output_name_strings;
    for (const auto& info : GetValueInfos(fused_node_.GetOutputs())) {
        output_name_strings.emplace_back(info.GetName());
    }
    std::vector<const char*> output_names;
    output_names.reserve(output_name_strings.size());
    for (const auto& name : output_name_strings) output_names.emplace_back(name.c_str());

    std::vector<std::string> input_name_strings;
    for (const auto& input : GetValueInfos(fused_node_.GetInputs())) {
        if (!input.IsConstantInitializer()) input_name_strings.emplace_back(input.GetName());
    }
    std::vector<const char*> input_names;
    input_names.reserve(input_name_strings.size());
    for (const auto& name : input_name_strings) input_names.emplace_back(name.c_str());

    constexpr size_t kNumAttributes = 8;
    std::array<OrtOpAttr*, kNumAttributes> attributes{};
    DeferRelease<OrtOpAttr> defer_release_attr{attributes.data(), attributes.size(), ort_api.ReleaseOpAttr};

    RETURN_IF_ERROR(ort_api.CreateOpAttr("embed_mode", &embed_mode,
        sizeof(int64_t), ORT_OP_ATTR_INT, &attributes[0]));

    // embed_mode==1: the blob is the serialized DML graph. embed_mode==0: a cache path.
    const std::string ep_cache_context_data{serialized_graph};
    RETURN_IF_ERROR(ort_api.CreateOpAttr("ep_cache_context", ep_cache_context_data.data(),
        ep_cache_context_data.length(), ORT_OP_ATTR_STRING, &attributes[1]));

    RETURN_IF_ERROR(ort_api.CreateOpAttr("hardware_architecture", compute_capability.data(),
        compute_capability.length(), ORT_OP_ATTR_STRING, &attributes[2]));

    const auto onnx_model_filename{model_path.filename().string()};
    RETURN_IF_ERROR(ort_api.CreateOpAttr("onnx_model_filename", onnx_model_filename.data(),
        onnx_model_filename.length(), ORT_OP_ATTR_STRING, &attributes[3]));

    RETURN_IF_ERROR(ort_api.CreateOpAttr("source", kEpContextSource.data(),
        kEpContextSource.length(), ORT_OP_ATTR_STRING, &attributes[4]));

    const auto fused_node_name{fused_node_.GetName()};
    const auto partition_name{node_name_prefix.empty() ?
        fused_node_name : std::string{node_name_prefix} + "_" + fused_node_name};
    RETURN_IF_ERROR(ort_api.CreateOpAttr("partition_name", partition_name.data(),
        partition_name.length(), ORT_OP_ATTR_STRING, &attributes[5]));

    constexpr std::string_view kDmlEpSdkVersion = "1";
    RETURN_IF_ERROR(ort_api.CreateOpAttr("ep_sdk_version", kDmlEpSdkVersion.data(),
        kDmlEpSdkVersion.length(), ORT_OP_ATTR_STRING, &attributes[6]));

    const int64_t main_context{1};
    RETURN_IF_ERROR(ort_api.CreateOpAttr("main_context", &main_context,
        sizeof(int64_t), ORT_OP_ATTR_INT, &attributes[7]));

    const auto node_name{node_name_prefix.empty() ?
        fused_node_name : std::string{node_name_prefix} + "_" + fused_node_name};

    RETURN_IF_ERROR(model_editor_api.CreateNode("EPContext", "com.microsoft",
        node_name.c_str(), input_names.data(), input_names.size(), output_names.data(),
        output_names.size(), attributes.data(), attributes.size(), &ep_context_node));

    return STATUS_OK;
}

bool EpContextNodeReader::GraphHasContextNode(const Ort::ConstGraph& graph) {
    for (const auto& node : graph.GetNodes()) {
        if (node.GetOperatorType() == "EPContext") return true;
    }
    return false;
}

EpContextNodeReader::EpContextNodeReader(const ApiPtrs& api_ptrs, const Ort::ConstGraph& graph,
    std::string_view current_hw_arch)
    : ApiPtrs{api_ptrs}
{
    const auto nodes{graph.GetNodes()};
    if (nodes.empty()) throw std::runtime_error("The graph is empty");

    const Ort::ConstNode* ep_context_node{nullptr};
    for (const auto& node : nodes) {
        if (node.GetOperatorType() != "EPContext") continue;
        Ort::ConstOpAttr source_attr;
        if (node.GetAttributeByName("source", source_attr).IsOK() && source_attr) {
            std::string source;
            if (source_attr.GetValue(source).IsOK() && source == kEpContextSource) {
                ep_context_node = &node;
                break;
            }
        }
        if (ep_context_node == nullptr) ep_context_node = &node;
    }
    if (ep_context_node == nullptr) throw std::runtime_error("no EPContext node for AMDGPUExecutionProvider");

    const auto& node{*ep_context_node};

    Ort::ConstOpAttr hw_attr;
    if (node.GetAttributeByName("hardware_architecture", hw_attr).IsOK() && hw_attr
        && hw_attr.GetType() == ORT_OP_ATTR_STRING && !current_hw_arch.empty()) {
        std::string cached_hw_arch;
        if (hw_attr.GetValue(cached_hw_arch).IsOK() && !cached_hw_arch.empty()
            && cached_hw_arch != std::string{current_hw_arch}) {
            throw std::runtime_error("EP context binary was compiled for GPU architecture '" +
                cached_hw_arch + "' but current GPU is '" + std::string{current_hw_arch} +
                "'. Please recompile the model for the current hardware.");
        }
    }

    Ort::ConstOpAttr attr;
    if (!node.GetAttributeByName("ep_cache_context", attr).IsOK() || !attr)
        throw std::runtime_error("EPContext node missing `ep_cache_context`");
    if (attr.GetType() != ORT_OP_ATTR_STRING)
        throw std::runtime_error("`ep_cache_context` attribute must be a string type");
    if (!attr.GetValue(serialized_graph_).IsOK() || serialized_graph_.empty())
        throw std::runtime_error("`ep_cache_context` is empty");
}

}  // namespace dml_ep
