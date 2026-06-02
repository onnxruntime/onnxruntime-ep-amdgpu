// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

#pragma once

#include "precomp.h"
#include "GraphDescBuilder.h"
#include "DmlRuntimeGraphFusionTransformer.h"
#include "GraphPartitioner.h"
#include "core/framework/kernel_type_str_resolver.h"
#include "core/framework/kernel_lookup.h"
#include "core/optimizer/constant_sharing.h"
#include "DmlGraphFusionHelper.h"

namespace dml_ep {

    namespace
    {
        struct CompiledPartitionInfo
        {
            std::shared_ptr<onnxruntime::IndexedSubGraph> indexedSubGraph;
            std::unordered_map<std::string, std::pair<const ONNX_NAMESPACE::TensorProto*, bool>> isInitializerTransferable;
        };
    }

    DmlRuntimeGraphFusionTransformer::DmlRuntimeGraphFusionTransformer(
        const std::string& name,
        const PluginDmlExecutionProviderImpl* provider
    )
        : GraphTransformer(name),
          m_providerImpl(provider)
    {
    }

    Ort::Status DmlRuntimeGraphFusionTransformer::ApplyImpl(
        onnxruntime::Graph& graph,
        bool& modified,
        int graphLevel,
        const Ort::Logger& logger) const
    {
        return ApplyImplHelper(graph, modified, graphLevel, logger, {});
    }

    Ort::Status DmlRuntimeGraphFusionTransformer::ApplyImplHelper(
        onnxruntime::Graph& graph,
        bool& modified,
        int graphLevel,
        const Ort::Logger& logger,
        const std::unordered_map<std::string, const onnxruntime::NodeArg*>& implicitInputDefs) const
    {
        onnxruntime::ProviderType providerType = onnxruntime::kDmlExecutionProvider;
        const gsl::not_null<const onnxruntime::KernelRegistry*> registry = m_providerImpl->GetKernelRegistry().get();
        const auto kernelTypeStrResolver = onnxruntime::OpSchemaKernelTypeStrResolver{};
        const auto kernelLookup = onnxruntime::KernelLookup(
            providerType,
            gsl::make_span(&registry, 1),
            kernelTypeStrResolver,
            logger);

        onnxruntime::GraphViewer graphViewer(graph);
        const auto& nodeTopologyList = graphViewer.GetNodesInTopologicalOrder();

        for (auto nodeIndex : nodeTopologyList)
        {
            auto* node = graph.GetNode(nodeIndex);
            if (!node)
            {
                continue;  // node was removed
            }

            std::unordered_map<std::string, const onnxruntime::NodeArg*> subgraphImplicitInputDefs;
            for (const onnxruntime::NodeArg* inputDef : node->ImplicitInputDefs())
            {
                subgraphImplicitInputDefs[inputDef->Name()] = inputDef;
            }

            for (auto& entry : node->GetAttributeNameToMutableSubgraphMap())
            {
                auto& subgraph = *entry.second;
                ORT_RETURN_IF_ERROR(ApplyImplHelper(subgraph, modified, graphLevel + 1, logger, subgraphImplicitInputDefs));
            }
        }

        // Initializers needed by any graph partition
        std::vector<onnxruntime::NodeIndex> additionalSplittingNodes;
        std::unordered_map<const onnxruntime::Node*, GraphNodeProperties> graphNodePropertyMap;
        std::unordered_set<std::string> requiredInitializerMap;
        std::unordered_set<std::string> dynamicCpuInputMap;
        std::vector<std::unique_ptr<GraphPartition>> partitions = BuildPartitions(
            graphViewer,
            *m_providerImpl->GetInternalRegistrationInfoMap(),
            kernelLookup,
            m_providerImpl->GetSupportedDeviceDataTypeMask(),
            graphNodePropertyMap,
            requiredInitializerMap,
            dynamicCpuInputMap,
            additionalSplittingNodes,
            implicitInputDefs,
            true);

        // Reset the splitting nodes for the current iteration
        additionalSplittingNodes.clear();

        // Reset the compiled operators for the current iteration
        std::vector<std::shared_ptr<CompiledPartitionInfo>> compiledPartitionInfos(partitions.size());

        // Create a map between each initialized tensor and the partition(s) it is part of.
        auto initializerPartitionMap = DmlGraphFusionHelper::GetInitializerToPartitionMap(graphViewer, partitions);

        for (uint32_t partitionIndex = 0; partitionIndex < partitions.size(); ++partitionIndex)
        {
            auto& partition = partitions[partitionIndex];

            if (partition->GetRootMergedPartition() != partition.get() ||
                !partition->IsDmlPartition())
            {
                continue;
            }

            if (partition->IsDmlGraphPartition())
            {
                std::unordered_map<std::string, std::pair<const ONNX_NAMESPACE::TensorProto*, bool>> isInitializerTransferable;

                std::string partitionKernelPrefix = std::to_string(m_providerImpl->GetPartitionKernelPrefixVal()) + "_";
                m_providerImpl->IncreasePartitionKernelPrefixVal();

                // populate isInitializerTransferable
                for (const auto& input : partition->GetInputs())
                {
                    const onnx::TensorProto* tensor = nullptr;
                    if (graph.GetInitializedTensor(input, tensor) && requiredInitializerMap.find(input) != requiredInitializerMap.end())
                    {
                        isInitializerTransferable[input] = {tensor, false};
                    }
                }

                compiledPartitionInfos[partitionIndex] = std::make_shared<CompiledPartitionInfo>();
                compiledPartitionInfos[partitionIndex]->indexedSubGraph = std::make_shared<onnxruntime::IndexedSubGraph>(
                    DmlGraphFusionHelper::CreateIndexedSubGraph(partition.get(), partitionIndex, partitionKernelPrefix));
                compiledPartitionInfos[partitionIndex]->isInitializerTransferable = std::move(isInitializerTransferable);
            }
        }

        for (auto&& compiledPartitionInfo : compiledPartitionInfos)
        {
            // Null compiled operators were not DML partitions
            if (compiledPartitionInfo)
            {
                //DmlGraphFusionHelper::RegisterDynamicKernel(
                //    graph,
                //    m_providerImpl->GetKernelRegistry().get(),
                //    m_providerImpl,
                //    graphNodePropertyMap,
                //    dynamicCpuInputMap,
                //    std::move(compiledPartitionInfo->indexedSubGraph),
                //    std::move(compiledPartitionInfo->isInitializerTransferable));
            }
        }

        return STATUS_OK;
    }

}  // namespace dml_ep
