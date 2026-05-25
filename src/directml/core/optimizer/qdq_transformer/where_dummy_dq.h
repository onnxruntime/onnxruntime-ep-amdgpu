// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
    @Class WhereDummyDq

    Graph transformer that inserts a dummy DQ on Where node's initializer input
    to form Node Unit when Where node has one DQ and one scalar initializer input
*/
class WhereDummyDq : public GraphTransformer {
 public:
  WhereDummyDq() noexcept : GraphTransformer("WhereDummyDq") {}

 private:
  Ort::Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const Ort::Logger& logger) const override;

  bool SatisfyCondition(const Graph& graph, const Node& node) const;
  Ort::Status InsertDummyDQ(Node& node, Graph& graph, bool& modified, const Ort::Logger& logger) const;
};
}  // namespace onnxruntime