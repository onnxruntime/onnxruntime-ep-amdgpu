// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {

class MatmulTransposeFusion : public GraphTransformer {
 public:
  MatmulTransposeFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("MatmulTransposeFusion", compatible_execution_providers) {}

  Ort::Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const Ort::Logger& logger) const override;
};

}  // namespace onnxruntime
