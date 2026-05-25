// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/framework/ort_value.h"
#include <memory>
#include "core/framework/execution_provider.h"

namespace onnxruntime {

/**
@class STFTDecomposition

Transformer that traverses the graph top-down and decomposes
STFT into convolution.
*/
class STFTDecomposition : public GraphTransformer {
 public:
  /*! STFT decomposition .
      \param execution_provider Execution provider instance to execute constant folding.
  */
  STFTDecomposition(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept;

 private:
  Ort::Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const Ort::Logger& logger) const override;
};

}  // namespace onnxruntime
