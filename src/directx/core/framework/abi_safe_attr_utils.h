// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

#pragma once

#include <onnxruntime_c_api.h>
#include <onnx/onnx_pb.h>
#include <memory>
#include <string>

namespace dml_ep {

// Builds a plugin-side AttributeProto from an ORT-side OrtOpAttr* using only
// the ORT C API (ReadOpAttr, OpAttr_GetName, OpAttr_GetType). This avoids
// accessing OrtOpAttr::attr_proto directly, which is unsafe when the plugin
// and ORT host use different protobuf versions.
std::unique_ptr<ONNX_NAMESPACE::AttributeProto> BuildPluginAttributeProto(
    const OrtOpAttr* ort_attr, const OrtApi& api);

// Gets the attribute name via the C API (OpAttr_GetName).
// Returns empty string on failure.
std::string GetOpAttrName(const OrtOpAttr* ort_attr, const OrtApi& api);

}  // namespace dml_ep
