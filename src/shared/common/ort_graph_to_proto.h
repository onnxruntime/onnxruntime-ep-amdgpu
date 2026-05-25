// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <gsl/gsl>

#include "plugin_ep_utils.h"
#include "onnx/onnx_pb.h"

using HandleInitializerDataFunc = std::function<Ort::Status(const Ort::ConstValueInfo& value_info,
                                                            gsl::span<const std::byte> data,
                                                            bool& is_external, std::string& location,
                                                            int64_t& offset)>;

Ort::Status GraphToProto(const OrtGraph* ort_graph,
                         ONNX_NAMESPACE::GraphProto& graph_proto,
                         const HandleInitializerDataFunc& handle_initializer_data_func = {});

Ort::Status GraphToProto(const OrtGraph* ort_graph,
                         ONNX_NAMESPACE::ModelProto& model_proto,
                         const HandleInitializerDataFunc& handle_initializer_data_func = {});
