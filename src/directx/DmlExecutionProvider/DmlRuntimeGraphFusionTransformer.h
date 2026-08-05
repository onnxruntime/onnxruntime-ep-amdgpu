// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>

#include "core/framework/execution_providers.h"
#include "core/optimizer/graph_transformer.h"

namespace dml_ep {

class PluginDmlExecutionProviderImpl;

class DmlRuntimeGraphFusionTransformer : public onnxruntime::GraphTransformer
{
public:
    DmlRuntimeGraphFusionTransformer(
        const std::string& name,
        const PluginDmlExecutionProviderImpl* provider
    );

public:
    static inline const char* const DML_GRAPH_FUSION_NODE_NAME_PREFIX = "DmlRuntimeFusedNode_";
    static inline const char* const DML_GRAPH_FUSION_NODE_DOMAIN = "DmlRuntimeFusedNodeDomain";

private:
    Ort::Status ApplyImpl(onnxruntime::Graph& graph,
                                          bool& modified,
                                          int graphLevel,
                                          const Ort::Logger& logger) const final;

    Ort::Status ApplyImplHelper(
        onnxruntime::Graph& graph,
        bool& modified,
        int graphLevel,
        const Ort::Logger& logger,
        const std::unordered_map<std::string, const onnxruntime::NodeArg*>& implicitInputDefs) const;

private:
    const PluginDmlExecutionProviderImpl* m_providerImpl = nullptr;
};

}  // namespace dml_ep
