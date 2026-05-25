// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <fstream>

#include <migraphx/version.h>

#include "mgx_ep_ctx.h"

namespace mgx_ep {

namespace {

std::string GetMIGraphXSdkVersion() {
    return std::to_string(MIGRAPHX_VERSION_MAJOR) + "." +
           std::to_string(MIGRAPHX_VERSION_MINOR) + "." +
           std::to_string(MIGRAPHX_VERSION_PATCH);
}

// Read entire file into a string buffer (binary mode)
bool ReadFileToString(const fs::path& path, std::string& contents)
try {
    const auto size{fs::file_size(path)};
    if (std::ifstream ifs{path, std::ios::binary}; ifs.is_open()) {
        contents.resize(size);
        ifs.read(contents.data(), size);
        return ifs.good();
    }
    return false;
} catch (const std::exception& e) {
    return false;
}

}  // namespace

Ort::Status EpContextNodeHelper::CreateEpContextNode(const fs::path& cache_path, const fs::path& cache_dir,
    int64_t embed_mode, std::string_view compute_capability, const fs::path& model_path,
    std::string_view node_name_prefix, OrtNode*& ep_context_node) const
{
    std::vector<std::string> output_name_strings;
    for (const auto& info : GetValueInfos(fused_node_.GetOutputs())) {
        output_name_strings.emplace_back(info.GetName());
    }
    std::vector<const char*> output_names;
    output_names.reserve(output_name_strings.size());
    for (const auto& name : output_name_strings) {
        output_names.emplace_back(name.c_str());
    }

    std::vector<std::string> input_name_strings;
    for (const auto& input : GetValueInfos(fused_node_.GetInputs())) {
        if (!input.IsConstantInitializer()) {
            input_name_strings.emplace_back(input.GetName());
        }
    }
    std::vector<const char*> input_names;
    input_names.reserve(input_name_strings.size());
    for (const auto& name : input_name_strings) {
        input_names.emplace_back(name.c_str());
    }

    constexpr size_t kNumAttributes = 8;
    std::array<OrtOpAttr*, kNumAttributes> attributes{};
    DeferRelease<OrtOpAttr> defer_release_attr{attributes.data(), attributes.size(), ort_api.ReleaseOpAttr};

    RETURN_IF_ERROR(ort_api.CreateOpAttr("embed_mode", &embed_mode,
        sizeof(int64_t), ORT_OP_ATTR_INT, &attributes[0]));

    std::string ep_cache_context_data;
    if (embed_mode == 1) {
        const auto full_mxr_path = cache_dir.empty() ? cache_path : cache_dir / cache_path;
        RETURN_IF(!ReadFileToString(full_mxr_path, ep_cache_context_data),
            "failed to read MXR file for embedding: ", full_mxr_path);
    } else {
        ep_cache_context_data = cache_path.string();
    }
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

    const auto sdk_version{GetMIGraphXSdkVersion()};
    RETURN_IF_ERROR(ort_api.CreateOpAttr("ep_sdk_version", sdk_version.data(),
        sdk_version.length(), ORT_OP_ATTR_STRING, &attributes[6]));

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
    return ranges::any_of(graph.GetNodes(),
        [](const auto& node) {
            return node.GetOperatorType() == "EPContext";
        });
}

EpContextNodeReader::EpContextNodeReader(const ApiPtrs& api_ptrs, const Ort::ConstGraph& graph, const Ort::Logger& logger,
    const fs::path& cache_dir, std::string_view current_hw_arch, std::string_view current_sdk_version)
    : ApiPtrs{api_ptrs}
{
    const auto nodes{graph.GetNodes()};
    THROW_IF(nodes.empty(), "The graph is empty");

    const Ort::ConstNode* ep_context_node{nullptr};
    for (const auto& node : nodes) {
        if (node.GetOperatorType() != "EPContext") {
            continue;
        }
        Ort::ConstOpAttr source_attr;
        Ort::Status source_status{node.GetAttributeByName("source", source_attr)};
        if (source_status.IsOK() && source_attr) {
            std::string source;
            if (source_attr.GetValue(source).IsOK()) {
                if (source == kEpContextSource) {
                    ep_context_node = &node;
                    break;
                }
                continue;
            }
        }
        if (ep_context_node == nullptr) {
            ep_context_node = &node;
        }
    }
    THROW_IF(ep_context_node == nullptr, "no EPContext node found for MIGraphXExecutionProvider");

    const auto& node{*ep_context_node};

    Ort::ConstOpAttr attr;
    THROW_IF_ERROR(node.GetAttributeByName("embed_mode", attr));
    THROW_IF(attr.GetType() != ORT_OP_ATTR_INT, "`embed_mode` attribute must be an integer type");

    int64_t embed_mode;
    THROW_IF_ERROR(attr.GetValue(embed_mode));

    Ort::ConstOpAttr hw_attr;
    if (node.GetAttributeByName("hardware_architecture", hw_attr).IsOK() && hw_attr
        && hw_attr.GetType() == ORT_OP_ATTR_STRING && !current_hw_arch.empty()) {
        std::string cached_hw_arch;
        if (hw_attr.GetValue(cached_hw_arch).IsOK() && !cached_hw_arch.empty()
            && cached_hw_arch != current_hw_arch) {
            THROW("EP context binary was compiled for GPU architecture '" + cached_hw_arch +
                  "' but current GPU is '" + std::string{current_hw_arch} +
                  "'. Please recompile the model for the current hardware.");
        }
    }

    Ort::ConstOpAttr sdk_attr;
    if (node.GetAttributeByName("ep_sdk_version", sdk_attr).IsOK() && sdk_attr
        && sdk_attr.GetType() == ORT_OP_ATTR_STRING && !current_sdk_version.empty()) {
        std::string cached_sdk_version;
        if (sdk_attr.GetValue(cached_sdk_version).IsOK() && !cached_sdk_version.empty()) {
            auto get_major = [](const std::string& ver) {
                auto dot = ver.find('.');
                return dot != std::string::npos ? ver.substr(0, dot) : ver;
            };
            if (get_major(cached_sdk_version) != get_major(std::string{current_sdk_version})) {
                THROW("EP context binary was compiled with MIGraphX SDK " + cached_sdk_version +
                      " but current SDK is " + std::string{current_sdk_version} +
                      ". Major version mismatch - please recompile the model.");
            }
        }
    }

    THROW_IF_ERROR(node.GetAttributeByName("ep_cache_context", attr));
    THROW_IF(attr.GetType() != ORT_OP_ATTR_STRING, "`ep_cache_context` attribute must be a string type");

    std::string ep_cache_context;
    THROW_IF_ERROR(attr.GetValue(ep_cache_context));

    if (embed_mode == 1) {
        THROW_IF(ep_cache_context.empty(), "ep_cache_context is empty for embed_mode=1");

        auto temp_path = fs::temp_directory_path() / "migraphx_ep_context_temp.mxr";
        try {
            {
                std::ofstream ofs(temp_path, std::ios::binary | std::ios::trunc);
                THROW_IF(!ofs, "failed to create temporary MXR file: " + temp_path.string());
                ofs.write(ep_cache_context.data(), static_cast<std::streamsize>(ep_cache_context.size()));
            }
            program_ = migraphx::load(temp_path.string().c_str());
            fs::remove(temp_path);
        } catch (const std::exception& e) {
            fs::remove(temp_path);  // cleanup on failure
            THROW("loading embedded MXR binary failed: " + std::string{e.what()});
        }
    } else {
        fs::path mxr_path{ep_cache_context};

        THROW_IF(mxr_path.is_absolute(), "for security purpose, the `ep_cache_context` must contain a relative "
                                         "path to MXR file: " + ep_cache_context);

        for (const auto& section : mxr_path.lexically_normal()) {
            THROW_IF(section == "..", "for security purpose, the file path of the `ep_cache_context` attribute "
                                      "must not include '..' - it is not allowed to point outside the directory.");
        }

        if (!cache_dir.empty()) {
            mxr_path = cache_dir / mxr_path;
        }

        THROW_IF(!exists(mxr_path), "the MXR cache file does not exist: " + mxr_path.string());
        try {
            program_ = migraphx::load(mxr_path.string().c_str());
        } catch (const std::exception&) {
            THROW("loading MXR file failed: " + mxr_path.string());
        }
    }
}

}  // namespace mgx_ep
