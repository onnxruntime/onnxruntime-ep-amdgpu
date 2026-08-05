// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

/**
    @Class ReluQuantFusion

    Rewrite rule that fuses Relu into followed QuantizeLinear
 */
class ReluQuantFusion : public RewriteRule {
 public:
  ReluQuantFusion() noexcept : RewriteRule("ReluQuantRewrite") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"Relu"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const Ort::Logger& logger) const override;

  Ort::Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const Ort::Logger& logger) const override;
};

}  // namespace onnxruntime
