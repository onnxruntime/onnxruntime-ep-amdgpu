// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {
/*
 *   This fusion submerges a BatchNormalization operator to it's super
 *   preceding MatMul operator, if and only if MatmulBNFusion::SatisfyCondition()
 *   is true.
 */
class MatmulBNFusion : public RewriteRule {
 public:
  MatmulBNFusion() : RewriteRule("MatMul_BatchNormalization_Fusion") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"MatMul"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const Ort::Logger& logger) const override;

  Ort::Status Apply(Graph& graph, Node& matmul_node, RewriteRuleEffect& rule_effect, const Ort::Logger& logger) const override;
};
}  // namespace onnxruntime
