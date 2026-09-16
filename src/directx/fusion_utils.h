// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <onnxruntime_c_api.h>

namespace dml_ep {
namespace fusion_utils {

// ---------------------------------------------------------------------------
// Node property accessors — thin wrappers around OrtApi C API calls.
// ---------------------------------------------------------------------------

std::string GetValueInfoName(const OrtApi& api, const OrtValueInfo* vi);
std::string GetNodeOpType(const OrtApi& api, const OrtNode* node);
std::string GetNodeDomain(const OrtApi& api, const OrtNode* node);
size_t      GetNodeId(const OrtApi& api, const OrtNode* node);

std::vector<std::string> GetNodeInputNames(const OrtApi& api, const OrtNode* node);
std::vector<std::string> GetNodeOutputNames(const OrtApi& api, const OrtNode* node);

// ---------------------------------------------------------------------------
// Empty-tensor detection
// ---------------------------------------------------------------------------

// Returns true if the node has an empty (extent-0) edge that DML cannot
// represent inside a compiled graph: any output with a zero-sized dim, or any
// input with a zero-sized dim that is NOT a constant initializer. A constant
// initializer input is exempt — it is consumed at compile time and never
// becomes a runtime DML edge (e.g. Resize's empty roi input).
//
// Used both to split Tier-0 partitions at an empty-tensor boundary (group
// formation) and to reject a partition outright (ValidateTier0). Reads only
// static ORT shape metadata; a dim ORT reports as dynamic (< 0) is not empty.
bool NodeHasEmptyEdge(
    const OrtApi& api,
    const OrtNode* node,
    const std::unordered_map<std::string, const OrtValue*>& initializers);

// ---------------------------------------------------------------------------
// Initializer helpers
// ---------------------------------------------------------------------------

// Reads a scalar float from an OrtValue initializer (supports float32, float64, float16).
// Returns false if the value is not a scalar or not a floating-point type.
bool TryReadScalarFloat(const OrtApi& api, const OrtValue* val, float& out_value);

// ---------------------------------------------------------------------------
// Graph connectivity maps built from a topological node list.
// ---------------------------------------------------------------------------

struct NodeInfo {
    const OrtNode* node = nullptr;
    std::string op_type;
    std::string domain;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};

struct GraphConnectivity {
    std::vector<NodeInfo> node_infos;
    // value_name -> index into node_infos (the node that produces this value)
    std::unordered_map<std::string, size_t> producer_map;
    // value_name -> indices into node_infos (nodes that consume this value)
    std::unordered_map<std::string, std::vector<size_t>> consumer_map;
    // node id -> index into node_infos
    std::unordered_map<size_t, size_t> id_to_index;
    // value names that are graph outputs — these count as external consumers
    // even though no node in the graph consumes them directly.
    std::unordered_set<std::string> graph_output_values;

    // Returns true if the named value has exactly one consumer in the graph.
    bool HasSingleConsumer(const std::string& value_name) const;
};

// Builds a GraphConnectivity from all nodes returned by Graph_GetNodes.
// Returns an empty GraphConnectivity on error.
GraphConnectivity BuildGraphConnectivity(const OrtApi& api, const OrtGraph* graph);

}  // namespace fusion_utils
}  // namespace dml_ep
