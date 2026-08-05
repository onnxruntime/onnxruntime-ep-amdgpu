// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

#include "abi_safe_attr_utils.h"
#include <vector>
#include <stdexcept>
#include <cstring>
namespace {

void ThrowOnError(const OrtApi& api, OrtStatus* status) {
    if (status != nullptr) {
        std::string message = api.GetErrorMessage(status);
        api.ReleaseStatus(status);
        throw std::runtime_error(message);
    }
}

size_t ElementSizeForTensorDataType(int32_t data_type) {
    switch (data_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
        return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return 8;
    default:
        throw std::runtime_error("Unsupported tensor element data type: " + std::to_string(data_type));
    }
}

int32_t OnnxTensorElementToProtoDataType(ONNXTensorElementDataType onnx_type) {
    switch (onnx_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4:   return ONNX_NAMESPACE::TensorProto_DataType_UINT4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:    return ONNX_NAMESPACE::TensorProto_DataType_INT4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return ONNX_NAMESPACE::TensorProto_DataType_UINT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return ONNX_NAMESPACE::TensorProto_DataType_INT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  return ONNX_NAMESPACE::TensorProto_DataType_UINT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   return ONNX_NAMESPACE::TensorProto_DataType_INT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  return ONNX_NAMESPACE::TensorProto_DataType_UINT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return ONNX_NAMESPACE::TensorProto_DataType_INT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:  return ONNX_NAMESPACE::TensorProto_DataType_UINT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return ONNX_NAMESPACE::TensorProto_DataType_INT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return ONNX_NAMESPACE::TensorProto_DataType_BOOL;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return ONNX_NAMESPACE::TensorProto_DataType_DOUBLE;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:   return ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ: return ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FNUZ;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:     return ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ: return ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2FNUZ;
    default:
        throw std::runtime_error("Unsupported ONNXTensorElementDataType: " + std::to_string(onnx_type));
    }
}

}  // namespace

namespace dml_ep {

std::string GetOpAttrName(const OrtOpAttr* ort_attr, const OrtApi& api) {
    if (!ort_attr) return {};
    const char* name = nullptr;
    ThrowOnError(api, api.OpAttr_GetName(ort_attr, &name));
    return name ? std::string(name) : std::string();
}

std::unique_ptr<ONNX_NAMESPACE::AttributeProto> BuildPluginAttributeProto(
    const OrtOpAttr* ort_attr, const OrtApi& api)
{
    if (!ort_attr) return nullptr;

    auto attr_proto = std::make_unique<ONNX_NAMESPACE::AttributeProto>();

    // Get name via C API
    std::string attr_name = GetOpAttrName(ort_attr, api);
    attr_proto->set_name(attr_name);

    // Get type via C API
    OrtOpAttrType attr_type{};
    ThrowOnError(api, api.OpAttr_GetType(ort_attr, &attr_type));

    switch (attr_type) {
    case ORT_OP_ATTR_INT: {
        attr_proto->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
        int64_t value{};
        size_t out_size{};
        ThrowOnError(api, api.ReadOpAttr(ort_attr, ORT_OP_ATTR_INT, &value, sizeof(value), &out_size));
        attr_proto->set_i(value);
        break;
    }
    case ORT_OP_ATTR_INTS: {
        attr_proto->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
        // Query size first
        size_t total_size{};
        api.ReadOpAttr(ort_attr, ORT_OP_ATTR_INTS, nullptr, 0, &total_size);
        if (total_size > 0) {
            std::vector<int64_t> values(total_size / sizeof(int64_t));
            ThrowOnError(api, api.ReadOpAttr(ort_attr, ORT_OP_ATTR_INTS, values.data(), total_size, &total_size));
            attr_proto->mutable_ints()->Assign(values.begin(), values.end());
        }
        break;
    }
    case ORT_OP_ATTR_FLOAT: {
        attr_proto->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
        float value{};
        size_t out_size{};
        ThrowOnError(api, api.ReadOpAttr(ort_attr, ORT_OP_ATTR_FLOAT, &value, sizeof(value), &out_size));
        attr_proto->set_f(value);
        break;
    }
    case ORT_OP_ATTR_FLOATS: {
        attr_proto->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS);
        size_t total_size{};
        api.ReadOpAttr(ort_attr, ORT_OP_ATTR_FLOATS, nullptr, 0, &total_size);
        if (total_size > 0) {
            std::vector<float> values(total_size / sizeof(float));
            ThrowOnError(api, api.ReadOpAttr(ort_attr, ORT_OP_ATTR_FLOATS, values.data(), total_size, &total_size));
            attr_proto->mutable_floats()->Assign(values.begin(), values.end());
        }
        break;
    }
    case ORT_OP_ATTR_STRING: {
        attr_proto->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
        size_t total_size{};
        api.ReadOpAttr(ort_attr, ORT_OP_ATTR_STRING, nullptr, 0, &total_size);
        if (total_size > 0) {
            std::string value(total_size, '\0');
            ThrowOnError(api, api.ReadOpAttr(ort_attr, ORT_OP_ATTR_STRING, &value[0], total_size, &total_size));
            // Remove trailing null if present
            if (!value.empty() && value.back() == '\0') {
                value.pop_back();
            }
            attr_proto->set_s(value);
        }
        break;
    }
    case ORT_OP_ATTR_STRINGS: {
        attr_proto->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRINGS);
        size_t total_size{};
        api.ReadOpAttr(ort_attr, ORT_OP_ATTR_STRINGS, nullptr, 0, &total_size);
        if (total_size > 0) {
            std::vector<char> buffer(total_size);
            ThrowOnError(api, api.ReadOpAttr(ort_attr, ORT_OP_ATTR_STRINGS, buffer.data(), total_size, &total_size));
            // Strings are packed as null-terminated sequences
            const char* ptr = buffer.data();
            const char* end = ptr + total_size;
            while (ptr < end) {
                attr_proto->add_strings(ptr);
                ptr += std::strlen(ptr) + 1;
            }
        }
        break;
    }
    case ORT_OP_ATTR_TENSOR: {
        attr_proto->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
        OrtValue* tensor_value = nullptr;
        ThrowOnError(api, api.OpAttr_GetTensorAttributeAsOrtValue(ort_attr, &tensor_value));
        if (tensor_value) {
            OrtTensorTypeAndShapeInfo* type_shape_info = nullptr;
            ThrowOnError(api, api.GetTensorTypeAndShape(tensor_value, &type_shape_info));

            ONNXTensorElementDataType elem_type{};
            ThrowOnError(api, api.GetTensorElementType(type_shape_info, &elem_type));

            size_t dim_count{};
            ThrowOnError(api, api.GetDimensionsCount(type_shape_info, &dim_count));
            std::vector<int64_t> dims(dim_count);
            ThrowOnError(api, api.GetDimensions(type_shape_info, dims.data(), dim_count));

            size_t elem_count{};
            ThrowOnError(api, api.GetTensorShapeElementCount(type_shape_info, &elem_count));
            api.ReleaseTensorTypeAndShapeInfo(type_shape_info);

            ONNX_NAMESPACE::TensorProto& tensor_proto = *attr_proto->mutable_t();
            tensor_proto.set_data_type(OnnxTensorElementToProtoDataType(elem_type));
            for (int64_t d : dims) {
                tensor_proto.add_dims(d);
            }

            const void* raw_data = nullptr;
            ThrowOnError(api, api.GetTensorData(tensor_value, &raw_data));
            size_t element_size = ElementSizeForTensorDataType(elem_type);
            size_t total_bytes = elem_count * element_size;
            tensor_proto.set_raw_data(raw_data, total_bytes);

            api.ReleaseValue(tensor_value);
        }
        break;
    }
    default:
        throw std::runtime_error("Unsupported OrtOpAttrType: " + std::to_string(attr_type));
    }

    return attr_proto;
}

}  // namespace dml_ep
