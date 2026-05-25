// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <string>
#include <string_view>
#include <map>
#include <vector>

#include <onnx/onnx_pb.h>
#include "ort_graph_to_proto.h"

#include <onnx/common/ir.h>

namespace {

Ort::ConstTensorTypeAndShapeInfo GetTensorTypeAndShapeInfo(const Ort::ConstValueInfo& value_info) {
    const auto type_info{value_info.TypeInfo()};
    THROW_IF(type_info.GetONNXType() != ONNX_TYPE_TENSOR, "Expected ValueInfo to represent a Tensor");
    return type_info.GetTensorTypeAndShapeInfo();
}

std::vector<std::string> GetSymbolicDimensions(const Ort::ConstTensorTypeAndShapeInfo& type_and_shape_info) {
    const auto t{type_and_shape_info.GetSymbolicDimensions()};
    return {t.begin(), t.end()};
}

void ValueInfoToProto(const Ort::ConstValueInfo& value_info, ONNX_NAMESPACE::ValueInfoProto& value_info_proto) {
    const auto type_and_shape_info{GetTensorTypeAndShapeInfo(value_info)};

    const auto dims{type_and_shape_info.GetShape()};
    const auto symbolic_dims{GetSymbolicDimensions(type_and_shape_info)};

    value_info_proto.set_name(value_info.GetName());

    const auto type_proto_tensor{value_info_proto.mutable_type()->mutable_tensor_type()};
    type_proto_tensor->set_elem_type(type_and_shape_info.GetElementType());

    if (!dims.empty()) {
        const auto shape_proto{type_proto_tensor->mutable_shape()};
        for (const auto& [dim, symbolic] : zip(dims, symbolic_dims)) {
            const auto dim_proto{shape_proto->add_dim()};
            if (dim >= 0) {
                dim_proto->set_dim_value(dim);
            } else {
                if (!symbolic.empty()) {
                    dim_proto->set_dim_param(symbolic);
                }
            }
        }
    }
}

Ort::Status OpAttrToProto(const Ort::ConstOpAttr& attr, ONNX_NAMESPACE::AttributeProto& attr_proto)
try {
    attr_proto.set_name(attr.GetName());
    switch (const auto attr_type{attr.GetType()}) {
    case ORT_OP_ATTR_INT: {
        attr_proto.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
        int64_t i_val{};
        RETURN_IF_ERROR(attr.GetValue(i_val));
        attr_proto.set_i(i_val);
        break;
    }
    case ORT_OP_ATTR_INTS: {
        attr_proto.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
        std::vector<int64_t> i_vals{};
        RETURN_IF_ERROR(attr.GetValueArray(i_vals));
        attr_proto.mutable_ints()->Assign(i_vals.begin(), i_vals.end());
        break;
    }
    case ORT_OP_ATTR_FLOAT: {
        attr_proto.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
        float f_val{};
        RETURN_IF_ERROR(attr.GetValue(f_val));
        attr_proto.set_f(f_val);
        break;
    }
    case ORT_OP_ATTR_FLOATS: {
        attr_proto.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS);
        std::vector<float> f_vals{};
        RETURN_IF_ERROR(attr.GetValueArray(f_vals));
        attr_proto.mutable_floats()->Assign(f_vals.begin(), f_vals.end());
        break;
    }
    case ORT_OP_ATTR_STRING: {
        attr_proto.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
        std::string str_val{};
        RETURN_IF_ERROR(attr.GetValue(str_val));
        attr_proto.set_s(str_val.c_str());
        break;
    }
    case ORT_OP_ATTR_STRINGS: {
        attr_proto.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRINGS);
        std::vector<std::string> str_vals{};
        RETURN_IF_ERROR(attr.GetValueArray(str_vals));
        attr_proto.mutable_strings()->Assign(str_vals.begin(), str_vals.end());
        break;
    }
    case ORT_OP_ATTR_TENSOR: {
        attr_proto.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
        Ort::Value value;
        RETURN_IF_ERROR(attr.GetTensorAttributeAsOrtValue(value));
        const auto info{value.GetTensorTypeAndShapeInfo()};

        ONNX_NAMESPACE::TensorProto& tensor_proto{*attr_proto.mutable_t()};
        size_t element_size{};
        switch (const auto element_type{info.GetElementType()}) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4:
            tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT4);
            element_size = sizeof(uint8_t);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:
            tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT4);
            element_size = sizeof(uint8_t);;
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
            element_size = sizeof(uint8_t);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT8);
            element_size = sizeof(int8_t);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT16);
            element_size = sizeof(uint16_t);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT16);
            element_size = sizeof(int16_t);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT32);
            element_size = sizeof(uint32_t);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
            element_size = sizeof(int32_t);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT64);
            element_size = sizeof(uint64_t);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
            element_size = sizeof(int64_t);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
            element_size = sizeof(bool);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_DOUBLE);
            element_size = sizeof(double);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
            element_size = sizeof(float);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
            element_size = sizeof(uint16_t);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16);
            element_size = sizeof(uint16_t);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
            tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN);
            element_size = sizeof(uint8_t);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
            tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FNUZ);
            element_size = sizeof(uint8_t);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
            tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2);
            element_size = sizeof(uint8_t);
            break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
            tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2FNUZ);
            element_size = sizeof(uint8_t);
            break;
        default:
            return {("Unexpected ONNXTensorElementDataType: " + std::to_string(element_type)).c_str(), ORT_FAIL};
        }
        for (const auto& dim : info.GetShape()) {
            tensor_proto.add_dims(dim);
        }
        tensor_proto.set_raw_data(value.GetTensorRawData(), info.GetElementCount() * element_size);
        break;
    }
    default:
        return {("Unexpected OrtOpAttrType with value " + std::to_string(attr_type)).c_str(), ORT_FAIL};
    }
    return STATUS_OK;
} catch (const Ort::Exception& e) {
    return Ort::Status{e};
} catch (const std::exception& e) {
    return Ort::Status{e.what(), ORT_EP_FAIL};
}

}  // namespace

Ort::Status GraphToProto(const OrtGraph* ort_graph, ONNX_NAMESPACE::GraphProto& graph_proto,
    const HandleInitializerDataFunc& handle_initializer_data_func)
try {
    Ort::ConstGraph graph{ort_graph};

    graph_proto.set_name(graph.GetName());
    graph_proto.set_doc_string("Serialized from OrtGraph");

    const auto graph_inputs{graph.GetInputs()};
    for (const auto& value_info : GetValueInfos(graph_inputs)) {
        auto value_info_proto{graph_proto.mutable_input()->Add()};
        ValueInfoToProto(value_info, *value_info_proto);
    }

    const auto graph_outputs{graph.GetOutputs()};
    for (const auto& value_info : GetValueInfos(graph_outputs)) {
        auto value_info_proto{graph_proto.mutable_output()->Add()};
        ValueInfoToProto(value_info, *value_info_proto);
    }
    std::map<std::string, Ort::ConstValueInfo> value_infos;
    std::map<std::string, Ort::ConstValueInfo> initializer_value_infos;

    auto collect_value_info = [&value_infos, &initializer_value_infos](
        const Ort::ConstValueInfo& value_info, std::string& value_name)
    {
        value_name = value_info.GetName();
        const auto is_required_graph_input{value_info.IsRequiredGraphInput()};
        const auto is_optional_graph_input{value_info.IsOptionalGraphInput()};
        const auto is_graph_output{value_info.IsGraphOutput()};
        const auto is_constant_initializer{value_info.IsConstantInitializer()};
        const auto is_from_outer_scope{value_info.IsFromOuterScope()};

        if (value_infos.count(value_name) == 0 && initializer_value_infos.count(value_name) == 0) {
            if (is_from_outer_scope) {
                value_infos.emplace(value_name, value_info);
                if (is_constant_initializer) {
                    initializer_value_infos.emplace(value_name, value_info);
                }
            } else if (is_optional_graph_input) {
                initializer_value_infos.emplace(value_name, value_info);
            } else if (is_constant_initializer) {
                value_infos.emplace(value_name, value_info);
                initializer_value_infos.emplace(value_name, value_info);
            } else if (!is_required_graph_input && !is_graph_output) {
                value_infos.emplace(value_name, value_info);
            }
        }
    };

    for (const auto& node : graph.GetNodes()) {
        const auto node_proto{graph_proto.add_node()};
        node_proto->set_name(node.GetName());
        node_proto->set_domain(node.GetDomain());
        node_proto->set_op_type(node.GetOperatorType());

        // Handle node attributes
        for (const auto& attr : node.GetAttributes()) {
            if (attr.GetType() != ORT_OP_ATTR_GRAPH) {
                RETURN_IF_ERROR(OpAttrToProto(attr, *node_proto->add_attribute()));
            }
        }

        // Handle node subgraphs
        for (const auto& [attr_name, subgraph] : node.GetSubgraphs()) {
            const auto attr_proto{node_proto->add_attribute()};
            attr_proto->set_name(attr_name);
            attr_proto->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
            RETURN_IF_ERROR(GraphToProto(subgraph, *attr_proto->mutable_g()));
        }

        // Handle node inputs
        for (const auto& value_info : node.GetInputs()) {
            if (value_info) {
                std::string value_name;
                collect_value_info(value_info, value_name);
                node_proto->add_input(value_name);
            } else {
                node_proto->add_input("");
            }
        }

        // Handle implicit inputs to this node.
        for (const auto& value_info : node.GetImplicitInputs()) {
            std::string value_name;
            collect_value_info(value_info, value_name);
        }

        // Handle node outputs
        for (const auto& value_info : node.GetOutputs()) {
            if (value_info) {
                std::string value_name;
                collect_value_info(value_info, value_name);
                node_proto->add_output(value_name);
            } else {
                node_proto->add_output("");
            }
        }
    }

    // Add value_infos to GraphProto as ValueInfoProto objects.
    for (const auto& [_, value_info] : value_infos) {
        auto value_info_proto{graph_proto.mutable_value_info()->Add()};
        ValueInfoToProto(value_info, *value_info_proto);
    }

    // Add missing initializers and skip the ones that already on the list
    for (const auto& initializer : GetValueInfos(graph.GetInitializers())) {
        initializer_value_infos.emplace(initializer.GetName(), initializer);
    }

    // Add initializers to GraphProto as TensorProto objects.
    for (const auto& [name, value_info] : initializer_value_infos) {
        const auto info{GetTensorTypeAndShapeInfo(value_info)};
        const auto tensor_proto{graph_proto.add_initializer()};
        tensor_proto->set_name(name.c_str());
        tensor_proto->set_data_type(info.GetElementType());
        for (const auto dim : info.GetShape()) {
            tensor_proto->mutable_dims()->Add(dim);
        }
        
        // Check if the initializer has external data info
        // If the model has external data files (e.g., model.onnx.data), the serialized proto
        // will reference those files directly without loading the data into memory,
        // avoiding the 2GB protobuf serialization limit
        Ort::ExternalInitializerInfo ext_init_info{};
        RETURN_IF_ERROR(value_info.GetExternalInitializerInfo(ext_init_info));
        if (ext_init_info != nullptr) {
            tensor_proto->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
            const auto data_proto{tensor_proto->mutable_external_data()};
            auto location_proto{data_proto->Add()};
            location_proto->set_key("location");
#ifdef _WIN32
            const auto filepath{ext_init_info.GetFilePath()};
            location_proto->set_value({filepath.begin(), filepath.end()});
#else
            location_proto->set_value(ext_init_info.GetFilePath());
#endif
            auto offset_proto = data_proto->Add();
            offset_proto->set_key("offset");
            offset_proto->set_value(std::to_string(ext_init_info.GetFileOffset()));
            auto length_proto = data_proto->Add();
            length_proto->set_key("length");
            length_proto->set_value(std::to_string(ext_init_info.GetByteSize()));
        } else {
            // Initializer data is inline 
            Ort::ConstValue initializer;
            RETURN_IF_ERROR(value_info.GetInitializer(initializer));
            const auto data{initializer.GetTensorRawData()};
            const auto data_bytes{initializer.GetTensorSizeInBytes()};

            bool is_external{};
            std::string ext_location;
            int64_t ext_offset{};

            if (handle_initializer_data_func) {
                RETURN_IF_ERROR(handle_initializer_data_func(value_info,
                    {static_cast<const std::byte*>(data), data_bytes},
                    is_external, ext_location, ext_offset));
            }
            if (is_external) {
                tensor_proto->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
                const auto data_proto{tensor_proto->mutable_external_data()};
                auto location_proto{data_proto->Add()};
                location_proto->set_key("location");
                location_proto->set_value(ext_location);
                auto offset_proto = data_proto->Add();
                offset_proto->set_key("offset");
                offset_proto->set_value(std::to_string(ext_offset));
                auto length_proto = data_proto->Add();
                length_proto->set_key("length");
                length_proto->set_value(std::to_string(data_bytes));
            } else {
                tensor_proto->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_DEFAULT);
                tensor_proto->set_raw_data(data, data_bytes);
            }
        }
    }
    return STATUS_OK;
} catch (const Ort::Exception& e) {
    return Ort::Status{e};
} catch (const std::exception& e) {
    return Ort::Status{e.what(), ORT_EP_FAIL};
}

Ort::Status GraphToProto(const OrtGraph* ort_graph, ONNX_NAMESPACE::ModelProto& model_proto,
    const HandleInitializerDataFunc& handle_initializer_data_func)
try {
    Ort::ConstGraph graph{ort_graph};
    model_proto.set_doc_string("Serialized from OrtGraph");
    model_proto.set_producer_name("GraphToProto");
    model_proto.set_ir_version(graph.GetOnnxIRVersion());
    const auto op_sets{graph.GetOperatorSets()};
    RETURN_IF(op_sets.empty(), "OrtGraph must have at least one operator set.");
    const auto op_sets_proto{model_proto.mutable_opset_import()};
    for (const auto& [domain, version] : graph.GetOperatorSets()) {
        const auto op_set{op_sets_proto->Add()};
        op_set->set_domain(domain);
        op_set->set_version(version);
    }
    model_proto.clear_graph();
    return GraphToProto(graph, *model_proto.mutable_graph(), handle_initializer_data_func);
} catch (Ort::Exception& e) {
    return Ort::Status{e};
} catch (std::exception& e) {
    return Ort::Status{e.what(), ORT_EP_FAIL};
}
