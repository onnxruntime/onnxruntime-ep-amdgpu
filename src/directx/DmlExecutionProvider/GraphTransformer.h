// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/optimizer/graph_transformer.h"

namespace dml_ep {

    class PluginDmlExecutionProviderImpl;

    // Applies transforms to a Lotus graph. The graph transformer is responsible for setting the execution provider
    // on the graph nodes which DML supports.
    class GraphTransformer : public onnxruntime::GraphTransformer
    {
    public:
        GraphTransformer(
            const std::string& name,
            const PluginDmlExecutionProviderImpl* provider
        );

    private:
     Ort::Status ApplyImpl(onnxruntime::Graph& graph, bool& modified, int graph_level, const Ort::Logger& logger) const final;

    private:
        void PerformOperatorFusion(onnxruntime::Graph* graph, bool isMcdmDevice, bool* modified) const;

        const PluginDmlExecutionProviderImpl* m_providerImpl = nullptr;
    };

}  // namespace dml_ep
