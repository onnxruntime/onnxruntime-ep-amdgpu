// Copyright (c) Microsoft Corporation.
// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Self-contained extraction of DmlGraphFusionHelper::ConvertGraphDesc: convert a serialized
// DmlSerializedGraphDesc back into a DML_GRAPH_DESC (operators + nodes + edges) so a compiled
// DML graph can be rebuilt from a blob without ONNX re-translation.
//
// The original DmlExecutionProvider/{GraphDescBuilder,DmlGraphFusionHelper}.h were removed as
// dead code by onnxruntime/onnxruntime-ep-amdgpu PR #144 ("trim ~600 unused files"): nothing on
// mainline called ConvertGraphDesc, because mainline has no EPContext reload (consume) path. The
// WebNN AOT consume path (FullGraphFusion::CompileEpContext) is the sole caller, so we lift just
// that one function + its small helper + the minimal GraphDesc struct here, depending only on the
// external/DirectMLHelpers headers that upstream keeps. Verbatim logic from the removed
// DmlGraphFusionHelper.cc, with ORT_THROW_IF_FAILED replaced by a std::runtime_error (the caller
// runs it inside a try/catch that returns nullptr on failure).
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

#include <wrl/client.h>

#include "DmlSerializedGraphDesc.h"
#include "SchemaHelpers.h"  // SchemaHelpers::ConvertOperatorDesc<N>; StackAllocator (via ApiHelpers.h)

namespace dml_ep {

namespace GraphDescBuilder {
    // Minimal form of the original struct: ConvertGraphDesc and CompileEpContext use only the
    // DmlSerializedGraphDesc base plus reuseCommandList.
    struct GraphDesc : DmlSerializedGraphDesc {
        bool reuseCommandList = false;
    };
}  // namespace GraphDescBuilder

namespace DmlGraphFusionHelper {

    inline uint32_t GetConstantNodeGraphInputIndex(
        const std::string& constantName,
        const std::unordered_map<std::string_view, uint32_t>* serializedGraphConstantNameToMainGraphInputIndex,
        uint32_t& graphMaxInputIndex,
        std::unordered_map<std::string_view, uint32_t>& localConstantNameToIndexMap)
    {
        if (serializedGraphConstantNameToMainGraphInputIndex == nullptr)
        {
            if (localConstantNameToIndexMap.find(constantName) == localConstantNameToIndexMap.end())
            {
                localConstantNameToIndexMap[constantName] = ++graphMaxInputIndex;
            }
            return localConstantNameToIndexMap[constantName];
        }
        else
        {
            graphMaxInputIndex = std::max(graphMaxInputIndex, serializedGraphConstantNameToMainGraphInputIndex->at(constantName));
            return serializedGraphConstantNameToMainGraphInputIndex->at(constantName);
        }
    }

    template <size_t AllocatorSize>
    void ConvertGraphDesc(
        const GraphDescBuilder::GraphDesc& graphDesc,
        const uint32_t inputCount,
        const uint32_t outputCount,
        IDMLDevice* device,
        StackAllocator<AllocatorSize>& allocator,
        const std::unordered_map<uint32_t, uint32_t>* serializedGraphInputIndexToSubgraphInputIndex,
        const std::unordered_map<std::string_view, uint32_t>* serializedGraphLargeConstantNameToSubgraphInputIndex,
        DML_GRAPH_DESC& dmlGraphDesc,
        std::vector<Microsoft::WRL::ComPtr<IDMLOperator>>& dmlOperators,
        std::vector<DML_GRAPH_NODE_DESC>& dmlGraphNodes,
        std::vector<DML_GRAPH_EDGE_DESC>& dmlInputEdges,
        std::vector<DML_GRAPH_EDGE_DESC>& dmlOutputEdges,
        std::vector<DML_GRAPH_EDGE_DESC>& dmlIntermediateEdges)
    {
        std::unordered_map<uint32_t, uint32_t> oldNodeIndexToNewNodeIndexMap;
        for (uint32_t index = 0; index < static_cast<uint32_t>(graphDesc.Nodes.size()); index++)
        {
            const DmlSerializedGraphNode& node = graphDesc.Nodes[index];
            if (std::holds_alternative<AbstractOperatorDesc>(node.Desc))
            {
                oldNodeIndexToNewNodeIndexMap[index] = static_cast<uint32_t>(dmlGraphNodes.size());
                DML_OPERATOR_DESC dmlDesc = SchemaHelpers::ConvertOperatorDesc<AllocatorSize>(std::get<AbstractOperatorDesc>(node.Desc), &allocator);
                Microsoft::WRL::ComPtr<IDMLOperator> op;
                if (FAILED(device->CreateOperator(&dmlDesc, IID_PPV_ARGS(&op))))
                    throw std::runtime_error("ConvertGraphDesc: IDMLDevice::CreateOperator failed");
                dmlOperators.push_back(op);
                DML_OPERATOR_GRAPH_NODE_DESC* dmlOperatorGraphNode = allocator.template Allocate<DML_OPERATOR_GRAPH_NODE_DESC>();
                dmlOperatorGraphNode->Name = node.Name.data();
                dmlOperatorGraphNode->Operator = op.Get();
                dmlGraphNodes.push_back(DML_GRAPH_NODE_DESC{DML_GRAPH_NODE_TYPE_OPERATOR, dmlOperatorGraphNode});
            }
            else
            {
                auto& constantNodeVariant = std::get<DmlSerializedGraphNodeConstantVariant>(node.Desc);
                if (std::holds_alternative<ConstantData>(constantNodeVariant))
                {
                    oldNodeIndexToNewNodeIndexMap[index] = static_cast<uint32_t>(dmlGraphNodes.size());

                    auto& constantData = std::get<ConstantData>(constantNodeVariant);

                    DML_CONSTANT_DATA_GRAPH_NODE_DESC* constantNode = allocator.template Allocate<DML_CONSTANT_DATA_GRAPH_NODE_DESC>();
                    constantNode->Name = node.Name.data();
                    constantNode->DataSize = gsl::narrow_cast<size_t>(constantData.dataSize);
                    constantNode->Data = constantData.data;
                    dmlGraphNodes.push_back(DML_GRAPH_NODE_DESC{DML_GRAPH_NODE_TYPE_CONSTANT, constantNode});
                }
            }
        }

        uint32_t graphMaxInputIndex = 0;

        for (size_t i = 0; i < graphDesc.InputEdges.size(); ++i)
        {
            DML_INPUT_GRAPH_EDGE_DESC* edge = allocator.template Allocate<DML_INPUT_GRAPH_EDGE_DESC>();
            edge->GraphInputIndex = serializedGraphInputIndexToSubgraphInputIndex == nullptr ?
                graphDesc.InputEdges[i].GraphInputIndex :
                serializedGraphInputIndexToSubgraphInputIndex->at(graphDesc.InputEdges[i].GraphInputIndex);
            edge->ToNodeIndex = oldNodeIndexToNewNodeIndexMap[graphDesc.InputEdges[i].ToNodeIndex];
            edge->ToNodeInputIndex = graphDesc.InputEdges[i].ToNodeInputIndex;
            edge->Name = graphDesc.InputEdges[i].Name.data();

            graphMaxInputIndex = std::max(graphMaxInputIndex, edge->GraphInputIndex);
            dmlInputEdges.push_back(DML_GRAPH_EDGE_DESC{DML_GRAPH_EDGE_TYPE_INPUT, edge});
        }

        for (size_t i = 0; i < graphDesc.OutputEdges.size(); ++i)
        {
            DML_OUTPUT_GRAPH_EDGE_DESC* edge = allocator.template Allocate<DML_OUTPUT_GRAPH_EDGE_DESC>();
            edge->GraphOutputIndex = graphDesc.OutputEdges[i].GraphOutputIndex;
            edge->FromNodeIndex = oldNodeIndexToNewNodeIndexMap[graphDesc.OutputEdges[i].FromNodeIndex];
            edge->FromNodeOutputIndex = graphDesc.OutputEdges[i].FromNodeOutputIndex;
            edge->Name = graphDesc.OutputEdges[i].Name.data();

            dmlOutputEdges.push_back(DML_GRAPH_EDGE_DESC{DML_GRAPH_EDGE_TYPE_OUTPUT, edge});
        }

        std::unordered_map<std::string_view, uint32_t> localConstantNameToIndexMap;
        for (uint32_t i = 0; i < static_cast<uint32_t>(graphDesc.IntermediateEdges.size()); ++i)
        {
            DmlSerializedGraphNodeDescVariant descVariant = graphDesc.Nodes[graphDesc.IntermediateEdges[i].FromNodeIndex].Desc;
            bool isConstantEdge = std::holds_alternative<DmlSerializedGraphNodeConstantVariant>(descVariant);
            if (isConstantEdge)
            {
                auto& constantNodeVariant = std::get<DmlSerializedGraphNodeConstantVariant>(descVariant);
                if (std::holds_alternative<ConstantData>(constantNodeVariant))
                {
                    DML_INTERMEDIATE_GRAPH_EDGE_DESC* edge = allocator.template Allocate<DML_INTERMEDIATE_GRAPH_EDGE_DESC>();
                    edge->FromNodeIndex = oldNodeIndexToNewNodeIndexMap[graphDesc.IntermediateEdges[i].FromNodeIndex];
                    edge->FromNodeOutputIndex = graphDesc.IntermediateEdges[i].FromNodeOutputIndex;
                    edge->ToNodeIndex = oldNodeIndexToNewNodeIndexMap[graphDesc.IntermediateEdges[i].ToNodeIndex];
                    edge->ToNodeInputIndex = graphDesc.IntermediateEdges[i].ToNodeInputIndex;
                    edge->Name = graphDesc.IntermediateEdges[i].Name.data();
                    dmlIntermediateEdges.push_back(DML_GRAPH_EDGE_DESC{DML_GRAPH_EDGE_TYPE_INTERMEDIATE, edge});
                }
                else
                {
                    const std::string& constantName = graphDesc.Nodes[graphDesc.IntermediateEdges[i].FromNodeIndex].Name;

                    DML_INPUT_GRAPH_EDGE_DESC* edge = allocator.template Allocate<DML_INPUT_GRAPH_EDGE_DESC>();
                    edge->GraphInputIndex = GetConstantNodeGraphInputIndex(
                        constantName,
                        serializedGraphLargeConstantNameToSubgraphInputIndex,
                        graphMaxInputIndex,
                        localConstantNameToIndexMap);
                    edge->ToNodeIndex = oldNodeIndexToNewNodeIndexMap[graphDesc.IntermediateEdges[i].ToNodeIndex];
                    edge->ToNodeInputIndex = graphDesc.IntermediateEdges[i].ToNodeInputIndex;
                    edge->Name = graphDesc.IntermediateEdges[i].Name.data();

                    dmlInputEdges.push_back({DML_GRAPH_EDGE_TYPE_INPUT, edge});
                }
            }
            else
            {
                DML_INTERMEDIATE_GRAPH_EDGE_DESC* edge = allocator.template Allocate<DML_INTERMEDIATE_GRAPH_EDGE_DESC>();
                edge->FromNodeIndex = oldNodeIndexToNewNodeIndexMap[graphDesc.IntermediateEdges[i].FromNodeIndex];
                edge->FromNodeOutputIndex = graphDesc.IntermediateEdges[i].FromNodeOutputIndex;
                edge->ToNodeIndex = oldNodeIndexToNewNodeIndexMap[graphDesc.IntermediateEdges[i].ToNodeIndex];
                edge->ToNodeInputIndex = graphDesc.IntermediateEdges[i].ToNodeInputIndex;
                edge->Name = graphDesc.IntermediateEdges[i].Name.data();
                dmlIntermediateEdges.push_back(DML_GRAPH_EDGE_DESC{DML_GRAPH_EDGE_TYPE_INTERMEDIATE, edge});
            }
        }

        dmlGraphDesc.InputCount = inputCount;
        dmlGraphDesc.OutputCount = outputCount;
        dmlGraphDesc.NodeCount = gsl::narrow_cast<uint32_t>(dmlGraphNodes.size());
        dmlGraphDesc.Nodes = dmlGraphNodes.data();
        dmlGraphDesc.InputEdgeCount = gsl::narrow_cast<uint32_t>(dmlInputEdges.size());
        dmlGraphDesc.InputEdges = dmlInputEdges.data();
        dmlGraphDesc.OutputEdgeCount = gsl::narrow_cast<uint32_t>(dmlOutputEdges.size());
        dmlGraphDesc.OutputEdges = dmlOutputEdges.data();
        dmlGraphDesc.IntermediateEdgeCount = gsl::narrow_cast<uint32_t>(dmlIntermediateEdges.size());
        dmlGraphDesc.IntermediateEdges = dmlIntermediateEdges.data();
    }

}  // namespace DmlGraphFusionHelper
}  // namespace dml_ep
