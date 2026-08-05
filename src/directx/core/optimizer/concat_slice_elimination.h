// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class ConcatSliceElimination
*/
class ConcatSliceElimination : public GraphTransformer {
 public:
  ConcatSliceElimination(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("ConcatSliceElimination", compatible_execution_providers) {}

  Ort::Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const Ort::Logger& logger) const override;

 private:
  static bool FuseConcatSliceSubgraph(Node& concat, Graph& graph, const Ort::Logger& logger);
};

}  // namespace onnxruntime
