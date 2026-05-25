// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class SkipLayerNormFusion

Rewrite graph fusing Add + Layer Normalization subgraph to a single SkipLayerNormalization node.

*/
class SkipLayerNormFusion : public GraphTransformer {
 public:
  explicit SkipLayerNormFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("SkipLayerNormFusion", compatible_execution_providers) {}

  Ort::Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const Ort::Logger& logger) const override;
};

}  // namespace onnxruntime
