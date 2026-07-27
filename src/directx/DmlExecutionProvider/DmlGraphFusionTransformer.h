// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <unordered_map>
#include "core/optimizer/graph_transformer.h"

namespace dml_ep {

class PluginDmlExecutionProviderImpl;

class DmlGraphFusionTransformer : public onnxruntime::GraphTransformer
{
public:
    DmlGraphFusionTransformer(
        const std::string& name,
        const PluginDmlExecutionProviderImpl* provider,
        const bool graphSerializationEnabled
    );

public:
    static inline const char* const DML_GRAPH_FUSION_NODE_NAME_PREFIX = "DmlFusedNode_";
    static inline const char* const DML_GRAPH_FUSION_NODE_DOMAIN = "DmlFusedNodeDomain";

private:
    Ort::Status ApplyImpl(onnxruntime::Graph& graph,
                                            bool& modified,
                                            int graph_level,
                                            const Ort::Logger& logger) const final;

    Ort::Status ApplyImplHelper(
        onnxruntime::Graph& graph,
        bool& modified,
        int graph_level,
        const Ort::Logger& logger,
        const std::unordered_map<std::string, const onnxruntime::NodeArg*>& implicitInputDefs) const;

private:
    const PluginDmlExecutionProviderImpl* m_providerImpl = nullptr;
    const bool graphSerializationEnabled = false;
};

}  // namespace dml_ep
