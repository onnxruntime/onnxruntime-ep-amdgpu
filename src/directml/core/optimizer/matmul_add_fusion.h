// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

class MatMulAddFusion : public GraphTransformer {
 public:
  MatMulAddFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {},
                  const bool preserve_attention_pattern = true) noexcept
      : GraphTransformer("MatMulAddFusion", compatible_execution_providers),
        preserve_attention_pattern_(preserve_attention_pattern) {}

  Ort::Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const Ort::Logger& logger) const override;

 private:
  bool preserve_attention_pattern_;
};

}  // namespace onnxruntime
