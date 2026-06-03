// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/plugin_ep_utils.h"

template <typename T>
struct VisitorPriorityQueue {
    using ComparatorType = std::function<bool(const T&, const T&)>;
    std::list<T> list_{};
    const ComparatorType comparator_{};

    explicit VisitorPriorityQueue(ComparatorType comparator)
        : comparator_{std::move(comparator)} {
    }
    void push(T node) {
        list_.insert(std::upper_bound(list_.begin(), list_.end(), node, comparator_), node);
    }
    bool empty() { return list_.empty(); }
    T top() { return list_.back(); }
    void pop() { list_.pop_back(); }
};

size_t GetNodeInputEdgeCount(const Ort::ConstNode& node) {
    size_t num_input_edges{};
    for (const auto& input : node.GetInputs()) {
        if (input == nullptr) {
            continue;
        }
        if (input.GetProducerNode().node != nullptr) {
            ++num_input_edges;
        }
    }
    return num_input_edges;
}

std::vector<Ort::ConstNode> GetOutputNodes(const Ort::ConstNode& node) {
    const auto outputs{node.GetOutputs()};
    std::vector<Ort::ConstNode> output_nodes;
    output_nodes.reserve(outputs.size());
    for (const auto& output : outputs) {
        if (output == nullptr) {
            continue;
        }
        const auto consumers{output.GetConsumers()};
        for (const auto& consumer : consumers) {
            output_nodes.emplace_back(consumer.node);
        }
    }
    return output_nodes;
}

// A sort that is a variant of Kahn's topological sort based on the implementation in CreateSupportedPartitionNodeGroups() in
// partitioning_utils.cc. The main difference is that this variant does not take
// into account supported/unsupported nodes and simply returns the topologically sorted nodes in a single group.
// This sort is used by the plugin EP to get the nodes in a topologically sorted order for compilation that matches the order
// created by onnxruntime in CreateSupportedPartitionNodeGroups() for input/output mapping.
// This function currently assumes all nodes in the graph are supported and is only used by migraphx.
std::vector<Ort::ConstNode> GetKahnsVariantTopologicalSortedNodes(const Ort::ConstGraph& graph)
{
    std::vector<std::vector<Ort::ConstNode>> supported_groups{};

    // number of inputs from unprocessed nodes (in-degree) per node
    std::unordered_map<size_t, size_t> in_degree{};
    // nodes that are ready to process
    std::deque<Ort::ConstNode> nodes_to_process{};

    const auto nodes{graph.GetNodes()};
    for (const auto node : nodes) {
        const auto node_input_edge_count{GetNodeInputEdgeCount(node)};
        in_degree.insert({node.GetId(), node_input_edge_count});

        if (node_input_edge_count == 0) {
            nodes_to_process.push_back(node);
        }
    }

    std::vector<Ort::ConstNode> supported_group{};

    size_t num_nodes_processed = 0;
    std::deque<size_t> node_id;
    while (!nodes_to_process.empty()) {

        const Ort::ConstNode node = nodes_to_process.front();
        std::string current_node_name = node.GetName();
        nodes_to_process.pop_front();

        const bool is_node_supported = true;

        if (is_node_supported) {
            supported_group.push_back(node);
        }

        for (const auto& output : node.GetOutputs()) {

            if (output == nullptr)
                continue;

            for (auto values : output.GetConsumers()) {

                auto& downstream_node_in_degree = in_degree[values.node.GetId()];
                --downstream_node_in_degree;

                if (downstream_node_in_degree == 0) {
                    nodes_to_process.emplace_back(values.node);
                }
            }
        }

        ++num_nodes_processed;
    }

    ENFORCE(num_nodes_processed == in_degree.size(), "Processed ", num_nodes_processed,
                " nodes. Expected to process ", in_degree.size());

    return supported_group;
}
