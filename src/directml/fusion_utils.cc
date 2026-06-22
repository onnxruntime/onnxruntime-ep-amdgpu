// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "fusion_utils.h"

#include <cstdint>
#include <cstring>

namespace dml_ep {
namespace fusion_utils {

std::string GetValueInfoName(const OrtApi& api, const OrtValueInfo* vi) {
    if (!vi) return {};
    const char* name = nullptr;
    OrtStatus* st = api.GetValueInfoName(vi, &name);
    if (st) { api.ReleaseStatus(st); return {}; }
    return name ? std::string(name) : std::string{};
}

std::string GetNodeOpType(const OrtApi& api, const OrtNode* node) {
    const char* op = nullptr;
    OrtStatus* st = api.Node_GetOperatorType(node, &op);
    if (st) { api.ReleaseStatus(st); return {}; }
    return op ? std::string(op) : std::string{};
}

std::string GetNodeDomain(const OrtApi& api, const OrtNode* node) {
    const char* domain = nullptr;
    OrtStatus* st = api.Node_GetDomain(node, &domain);
    if (st) { api.ReleaseStatus(st); return {}; }
    return domain ? std::string(domain) : std::string{};
}

size_t GetNodeId(const OrtApi& api, const OrtNode* node) {
    size_t id = 0;
    OrtStatus* st = api.Node_GetId(node, &id);
    if (st) { api.ReleaseStatus(st); return SIZE_MAX; }
    return id;
}

std::vector<std::string> GetNodeInputNames(const OrtApi& api, const OrtNode* node) {
    size_t count = 0;
    OrtStatus* st = api.Node_GetNumInputs(node, &count);
    if (st) { api.ReleaseStatus(st); return {}; }
    if (count == 0) return {};

    std::vector<const OrtValueInfo*> inputs(count, nullptr);
    st = api.Node_GetInputs(node, inputs.data(), count);
    if (st) { api.ReleaseStatus(st); return {}; }

    std::vector<std::string> names;
    names.reserve(count);
    for (const OrtValueInfo* vi : inputs) {
        names.push_back(GetValueInfoName(api, vi));
    }
    return names;
}

std::vector<std::string> GetNodeOutputNames(const OrtApi& api, const OrtNode* node) {
    size_t count = 0;
    OrtStatus* st = api.Node_GetNumOutputs(node, &count);
    if (st) { api.ReleaseStatus(st); return {}; }
    if (count == 0) return {};

    std::vector<const OrtValueInfo*> outputs(count, nullptr);
    st = api.Node_GetOutputs(node, outputs.data(), count);
    if (st) { api.ReleaseStatus(st); return {}; }

    std::vector<std::string> names;
    names.reserve(count);
    for (const OrtValueInfo* vi : outputs) {
        names.push_back(GetValueInfoName(api, vi));
    }
    return names;
}

bool TryReadScalarFloat(const OrtApi& api, const OrtValue* val, float& out_value) {
    if (!val) return false;

    OrtTensorTypeAndShapeInfo* info = nullptr;
    OrtStatus* st = api.GetTensorTypeAndShape(const_cast<OrtValue*>(val), &info);
    if (st) { api.ReleaseStatus(st); return false; }

    size_t elem_count = 0;
    st = api.GetTensorShapeElementCount(info, &elem_count);
    if (st) { api.ReleaseStatus(st); api.ReleaseTensorTypeAndShapeInfo(info); return false; }

    ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    st = api.GetTensorElementType(info, &elem_type);
    api.ReleaseTensorTypeAndShapeInfo(info);
    if (st) { api.ReleaseStatus(st); return false; }

    if (elem_count != 1) return false;

    void* data = nullptr;
    st = api.GetTensorMutableData(const_cast<OrtValue*>(val), &data);
    if (st) { api.ReleaseStatus(st); return false; }
    if (!data) return false;

    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        out_value = *static_cast<const float*>(data);
        return true;
    }
    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
        out_value = static_cast<float>(*static_cast<const double*>(data));
        return true;
    }
    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        // Standard half-precision to float conversion via bit manipulation.
        uint16_t bits = *static_cast<const uint16_t*>(data);
        uint32_t sign     = static_cast<uint32_t>(bits & 0x8000u) << 16;
        uint32_t exponent = static_cast<uint32_t>(bits & 0x7C00u);
        uint32_t mantissa = static_cast<uint32_t>(bits & 0x03FFu);
        uint32_t result_bits;
        if (exponent == 0x7C00u) {
            result_bits = sign | 0x7F800000u | (mantissa << 13);
        } else if (exponent == 0) {
            uint32_t m = mantissa;
            uint32_t e = 0;
            while (m && !(m & 0x0400u)) { m <<= 1; ++e; }
            result_bits = sign | ((127u - 14u - e + 1u) << 23) | ((m & 0x03FFu) << 13);
        } else {
            result_bits = sign | (((exponent >> 10) + (127u - 15u)) << 23) | (mantissa << 13);
        }
        float f;
        std::memcpy(&f, &result_bits, sizeof(f));
        out_value = f;
        return true;
    }
    return false;
}

bool GraphConnectivity::HasSingleConsumer(const std::string& value_name) const {
    auto it = consumer_map.find(value_name);
    return (it != consumer_map.end()) && (it->second.size() == 1);
}

GraphConnectivity BuildGraphConnectivity(const OrtApi& api, const OrtGraph* graph) {
    GraphConnectivity gc;

    size_t num_nodes = 0;
    {
        OrtStatus* st = api.Graph_GetNumNodes(graph, &num_nodes);
        if (st) { api.ReleaseStatus(st); return gc; }
    }
    if (num_nodes == 0) return gc;

    std::vector<const OrtNode*> topo_nodes(num_nodes, nullptr);
    {
        OrtStatus* st = api.Graph_GetNodes(graph, topo_nodes.data(), num_nodes);
        if (st) { api.ReleaseStatus(st); return gc; }
    }

    gc.node_infos.reserve(num_nodes);
    for (const OrtNode* node : topo_nodes) {
        if (!node) continue;

        NodeInfo info;
        info.node         = node;
        info.op_type      = GetNodeOpType(api, node);
        info.domain       = GetNodeDomain(api, node);
        info.input_names  = GetNodeInputNames(api, node);
        info.output_names = GetNodeOutputNames(api, node);

        size_t idx     = gc.node_infos.size();
        size_t node_id = GetNodeId(api, node);
        gc.id_to_index[node_id] = idx;

        for (const std::string& out : info.output_names) {
            if (!out.empty()) {
                gc.producer_map[out] = idx;
            }
        }
        for (const std::string& in : info.input_names) {
            if (!in.empty()) {
                gc.consumer_map[in].push_back(idx);
            }
        }

        gc.node_infos.push_back(std::move(info));
    }

    return gc;
}

}  // namespace fusion_utils
}  // namespace dml_ep
