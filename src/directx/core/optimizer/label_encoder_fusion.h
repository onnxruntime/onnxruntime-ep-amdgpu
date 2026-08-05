// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {
/**
@Class LabelEncoderFusion

Rewrite rule that fuses two LabelEncoder -> LabelEncoder nodes to a single
LabelEncoder node.

*/
class LabelEncoderFusion : public RewriteRule {
 public:
  LabelEncoderFusion() noexcept : RewriteRule("LabelEncoderFusion") {}

  std::vector<std::string> TargetOpTypes() const noexcept override {
    return {"LabelEncoder"};
  }

 private:
  bool SatisfyCondition(const Graph& graph, const Node& node, const Ort::Logger& logger) const override;
  Ort::Status Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const Ort::Logger& logger) const override;

  template <typename T1, typename T2, typename T3>
  Ort::Status ApplyHelper(Graph& graph, Node& node, Node& next_node, RewriteRuleEffect& rule_effect) const;

  template <typename T1, typename T2, typename T3>
  bool IsValidForFusion(const Node& node, const Node& next) const;
};

}  // namespace onnxruntime
