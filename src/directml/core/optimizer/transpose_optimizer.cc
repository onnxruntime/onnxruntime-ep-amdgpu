// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "transpose_optimizer.h"
#include <deque>
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/providers/cpu/tensor/transpose.h"
#include "core/optimizer/transpose_optimization/ort_optimizer_utils.h"
#include "core/optimizer/transpose_optimization/ort_transpose_optimization.h"

using namespace ONNX_NAMESPACE;
using namespace onnx_transpose_optimization;

namespace onnxruntime {

Ort::Status TransposeOptimizer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                     const Ort::Logger& logger) const {
  OptimizeResult result;

  if (ep_.empty()) {
    // basic usage - no EP specific optimizations
    auto api_graph = MakeApiGraph(graph, cpu_allocator_, /*new_node_ep*/ nullptr);
    result = onnx_transpose_optimization::Optimize(*api_graph, "", /* default cost check*/ nullptr,
                                                   OrtExtendedHandlers());
  } else {
    // EP specific optimizations enabled. Currently only used for CPU EP.
    auto api_graph = MakeApiGraph(graph, cpu_allocator_, /*new_node_ep*/ ep_.c_str());
    result = onnx_transpose_optimization::Optimize(*api_graph, ep_, OrtEPCostCheck, OrtExtendedHandlers());
  }

  if (result.graph_modified) {
    modified = true;
  }

  GraphViewer graph_viewer(graph);
  auto nodes = std::vector<std::unique_ptr<api::NodeRef>>();
  for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    ORT_RETURN_IF_ERROR(Recurse(*graph.GetNode(index), modified, graph_level, logger));
  }

  return STATUS_OK;
}

}  // namespace onnxruntime
