// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class GroupQueryAttention
*/
class GroupQueryAttentionFusion : public GraphTransformer {
 public:
  explicit GroupQueryAttentionFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("GroupQueryAttentionFusion", compatible_execution_providers) {
  }

  Ort::Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const Ort::Logger& logger) const override;
};

}  // namespace onnxruntime
