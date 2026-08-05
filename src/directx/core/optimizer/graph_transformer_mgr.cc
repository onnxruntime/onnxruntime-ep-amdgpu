// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/graph_transformer_mgr.h"
#include "core/optimizer/rule_based_graph_transformer.h"

#include <memory>
#include <utility>

using namespace onnxruntime;

namespace onnxruntime {

Ort::Status GraphTransformerManager::SetSteps(unsigned steps) {
  steps_ = steps;
  return STATUS_OK;
}

Ort::Status GraphTransformerManager::GetSteps(unsigned& steps) const {
  steps = steps_;
  return STATUS_OK;
}

Ort::Status GraphTransformerManager::ApplyTransformers(Graph& graph, TransformerLevel level,
                                                          const Ort::Logger& logger) const {
  _is_graph_modified = false;
  const auto& transformers = level_to_transformer_map_.find(level);
  if (transformers == level_to_transformer_map_.end()) {
    return STATUS_OK;
  }

  for (unsigned step = 0; step < steps_; ++step) {
    if (IsLoadCancellationFlagSet()) {
      return ORT_MAKE_STATUS(ORT_MODEL_LOAD_CANCELED, "Graph transformation canceled due to user request.");
    }
    bool graph_changed = false;
    for (const auto& transformer : transformers->second) {
      if (step > 0 && transformer->ShouldOnlyApplyOnce())
        continue;

      bool modified = false;
      ORT_RETURN_IF_ERROR(transformer->Apply(graph, modified, logger));
      graph_changed = graph_changed || modified;
      _is_graph_modified = _is_graph_modified || modified;
    }
    if (!graph_changed) {
      break;
    }
  }

  return STATUS_OK;
}

const bool& GraphTransformerManager::IsGraphModified(void) const {
  return _is_graph_modified;
}

void GraphTransformerManager::ClearGraphModified(void) {
  _is_graph_modified = false;
}

Ort::Status GraphTransformerManager::Register(std::unique_ptr<GraphTransformer> transformer,
                                                 TransformerLevel level) {
  const auto& name = transformer->Name();
  const auto& registered = level_to_transformer_map_[level];
  if (std::find(registered.begin(), registered.end(), transformer) != registered.end()) {
    return ORT_MAKE_STATUS(ORT_FAIL, "This transformer is already registered " + name);
  }

  transformers_info_[name] = transformer.get();
  level_to_transformer_map_[level].push_back(std::move(transformer));
  return STATUS_OK;
}
}  // namespace onnxruntime
