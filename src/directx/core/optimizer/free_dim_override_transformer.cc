// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/session_options.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/free_dim_override_transformer.h"

using namespace ONNX_NAMESPACE;
namespace onnxruntime {

static std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](char c) {
    return static_cast<char>(::tolower(c));
  });

  return s;
}

FreeDimensionOverrideTransformer::FreeDimensionOverrideTransformer(gsl::span<const FreeDimensionOverride> overrides_to_apply)
    : GraphTransformer("FreeDimensionOverrideTransformer") {
  for (const auto& o : overrides_to_apply) {
    // Convert to lowercase to perform case-insensitive comparisons later
    if (o.dim_identifier_type == FreeDimensionOverrideType::Denotation) {
      dimension_override_by_denotation_.emplace(ToLower(o.dim_identifier), o.dim_value);
    } else if (o.dim_identifier_type == FreeDimensionOverrideType::Name) {
      dimension_override_by_name_.emplace(o.dim_identifier, o.dim_value);
    } else {
      ORT_THROW("Invalid free dimension override.");
    }
  }
}

Ort::Status FreeDimensionOverrideTransformer::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/, const Ort::Logger& logger) const {
  for (const onnxruntime::NodeArg* graph_input : graph.GetInputs()) {
    // Get the current input's type and shape
    const auto* input_type = graph_input->TypeAsProto();
    const auto* input_shape = graph_input->Shape();

    if (!input_type || !input_shape || !input_type->has_tensor_type()) {
      continue;
    }

    // Construct a new shape for this input, replacing free dimensions with their overrides
    onnx::TensorShapeProto new_shape;
    bool shape_modified = false;
    for (int32_t dim_index = 0; dim_index < input_shape->dim_size(); ++dim_index) {
      const auto& dimension = input_shape->dim(dim_index);

      // By default just make a copy of the dimension
      auto* new_dimension = new_shape.add_dim();
      *new_dimension = dimension;

      bool overridden = false;
      int64_t dimension_override = 0;

      if (dimension.has_denotation()) {
        // Convert to lowercase to perform case-insensitive comparison
        auto it = dimension_override_by_denotation_.find(ToLower(dimension.denotation()));
        if (it != dimension_override_by_denotation_.end()) {
          overridden = true;
          dimension_override = it->second;
        }
      }

      if (dimension.has_dim_param()) {
        auto it = dimension_override_by_name_.find(dimension.dim_param());
        if (it != dimension_override_by_name_.end()) {
          if (overridden && dimension_override != it->second) {
            return ORT_MAKE_STATUS(ORT_INVALID_ARGUMENT, "Conflicting free dimension overrides.");
          }

          overridden = true;
          dimension_override = it->second;
        }
      }

      if (overridden) {
        if (dimension.has_dim_value()) {
          // If this dimension actually has a value but it doesn't match the override value, return an
          // error.
          if (dimension.dim_value() != dimension_override) {
            return ORT_MAKE_STATUS(ORT_INVALID_ARGUMENT, "Invalid free dimension override.");
          }
        } else {
          // Set the dimension override
          new_dimension->set_dim_value(dimension_override);
          shape_modified = true;
        }
      }
    }

    if (shape_modified) {
      // Set the new shape
      auto* mutable_graph_input = graph.GetNodeArg(graph_input->Name());
      assert(mutable_graph_input != nullptr);
      mutable_graph_input->SetShape(new_shape);

      graph.SetGraphResolveNeeded();
      modified = true;
    }
  }

  return STATUS_OK;
}

}  // namespace onnxruntime
