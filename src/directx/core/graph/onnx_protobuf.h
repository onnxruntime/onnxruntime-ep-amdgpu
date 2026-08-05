// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4244)
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic push

#ifdef HAS_SHORTEN_64_TO_32
#pragma GCC diagnostic ignored "-Wshorten-64-to-32"
#endif

#endif

#if !defined(ORT_MINIMAL_BUILD)
#include "onnx/defs/schema.h"
#else
#include "onnx/defs/data_type_utils.h"
#endif

#include "onnx/onnx_pb.h"
#include "onnx/onnx-operators_pb.h"

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#ifdef _WIN32
#pragma warning(pop)
#endif

#include "common/plugin_ep_utils.h"

template <> struct fmt::formatter<ONNX_NAMESPACE::TypeProto::ValueCase> : formatter<std::string_view> {
    format_context::iterator format(ONNX_NAMESPACE::TypeProto::ValueCase value_case, format_context& ctx) const {
        std::string_view name{"unknown"};
        switch (value_case) {
        case ONNX_NAMESPACE::TypeProto::ValueCase::kTensorType:
            name = "Tensor";
            break;
        case ONNX_NAMESPACE::TypeProto::ValueCase::kSequenceType:
            name = "Sequence";
            break;
        case ONNX_NAMESPACE::TypeProto::ValueCase::kMapType:
            name = "Map";
            break;
        case ONNX_NAMESPACE::TypeProto::ValueCase::kOptionalType:
            name = "Optional";
            break;
        case ONNX_NAMESPACE::TypeProto::ValueCase::kSparseTensorType:
            name = "SparseTensor";
            break;
        case ONNX_NAMESPACE::TypeProto::ValueCase::kOpaqueType:
            name = "Opaque";
            break;
        case ONNX_NAMESPACE::TypeProto::VALUE_NOT_SET:
            name = "value not set";
            break;
        }
        return formatter<std::string_view>::format(name, ctx);
    }
};
