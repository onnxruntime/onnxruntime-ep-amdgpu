// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

#include <gsl/gsl>
#include <stdexcept>
#include <sstream>

#include "ort_node_adapter.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/framework/abi_safe_attr_utils.h"

namespace dml_ep {

//==============================================================================
// OrtValueInfoAdapter Implementation
//==============================================================================

OrtValueInfoAdapter::OrtValueInfoAdapter(const OrtValueInfo* value_info)
    : value_info_(value_info)
    , type_info_(nullptr)
    , type_info_cached_(false)
    , onnx_type_(ONNX_TYPE_UNKNOWN)
    , tensor_elem_type_(ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED)
{
    if (!value_info_) {
        return;  // Allow nullptr for optional inputs
    }

    // Get name using virtual method
    name_ = value_info_->GetName();

    // Get type info
    type_info_ = value_info_->GetTypeInfo();
}

bool OrtValueInfoAdapter::IsTensor() const {
    if (!type_info_cached_) {
        CacheTypeInfo();
    }
    return onnx_type_ == ONNX_TYPE_TENSOR;
}

bool OrtValueInfoAdapter::IsSequence() const {
    if (!type_info_cached_) {
        CacheTypeInfo();
    }
    return onnx_type_ == ONNX_TYPE_SEQUENCE;
}

bool OrtValueInfoAdapter::IsMap() const {
    if (!type_info_cached_) {
        CacheTypeInfo();
    }
    return onnx_type_ == ONNX_TYPE_MAP;
}

bool OrtValueInfoAdapter::IsOptional() const {
    if (!type_info_cached_) {
        CacheTypeInfo();
    }
    return onnx_type_ == ONNX_TYPE_OPTIONAL;
}

ONNXTensorElementDataType OrtValueInfoAdapter::GetTensorElementType() const {
    if (!type_info_cached_) {
        CacheTypeInfo();
    }
    return tensor_elem_type_;
}

void OrtValueInfoAdapter::CacheTypeInfo() const {
    type_info_cached_ = true;

    if (!type_info_) {
        return;
    }

    // Get ONNX type
    onnx_type_ = type_info_->type;

    // If tensor, get element type from tensor_type_info
    if (onnx_type_ == ONNX_TYPE_TENSOR && type_info_->tensor_type_info) {
    tensor_elem_type_ = type_info_->tensor_type_info->GetElementType();
    }
}

//==============================================================================
// OrtOpAttrAdapter Implementation
//==============================================================================

OrtOpAttrAdapter::OrtOpAttrAdapter(const char* name, const OrtOpAttr* attr, const OrtApi& api)
    : name_(name ? name : ""), attr_(attr), api_(&api) {}

void OrtOpAttrAdapter::EnsureCachedProto() const {
    if (!cached_proto_) {
        cached_proto_ = BuildPluginAttributeProto(attr_, *api_);
    }
}

const ONNX_NAMESPACE::AttributeProto& OrtOpAttrAdapter::GetAttrProto() const {
    EnsureCachedProto();
    return *cached_proto_;
}

float OrtOpAttrAdapter::GetFloat() const {
    EnsureCachedProto();
    if (!cached_proto_ || cached_proto_->type() != ONNX_NAMESPACE::AttributeProto::FLOAT) {
        throw std::runtime_error("Attribute is not a float");
    }
    return cached_proto_->f();
}

int64_t OrtOpAttrAdapter::GetInt() const {
    EnsureCachedProto();
    if (!cached_proto_ || cached_proto_->type() != ONNX_NAMESPACE::AttributeProto::INT) {
        throw std::runtime_error("Attribute is not an int");
    }
    return cached_proto_->i();
}

const char* OrtOpAttrAdapter::GetString() const {
    EnsureCachedProto();
    if (!cached_proto_ || cached_proto_->type() != ONNX_NAMESPACE::AttributeProto::STRING) {
        throw std::runtime_error("Attribute is not a string");
    }
    return cached_proto_->s().c_str();
}

std::vector<float> OrtOpAttrAdapter::GetFloats() const {
    EnsureCachedProto();
    if (!cached_proto_ || cached_proto_->type() != ONNX_NAMESPACE::AttributeProto::FLOATS) {
        throw std::runtime_error("Attribute is not a floats array");
    }
    std::vector<float> result;
    result.reserve(cached_proto_->floats_size());
    for (int i = 0; i < cached_proto_->floats_size(); ++i) {
        result.push_back(cached_proto_->floats(i));
    }
    return result;
}

std::vector<int64_t> OrtOpAttrAdapter::GetInts() const {
    EnsureCachedProto();
    if (!cached_proto_ || cached_proto_->type() != ONNX_NAMESPACE::AttributeProto::INTS) {
        throw std::runtime_error("Attribute is not an ints array");
    }
    std::vector<int64_t> result;
    result.reserve(cached_proto_->ints_size());
    for (int i = 0; i < cached_proto_->ints_size(); ++i) {
        result.push_back(cached_proto_->ints(i));
    }
    return result;
}

std::vector<std::string> OrtOpAttrAdapter::GetStrings() const {
    EnsureCachedProto();
    if (!cached_proto_ || cached_proto_->type() != ONNX_NAMESPACE::AttributeProto::STRINGS) {
        throw std::runtime_error("Attribute is not a strings array");
    }
    std::vector<std::string> result;
    result.reserve(cached_proto_->strings_size());
    for (int i = 0; i < cached_proto_->strings_size(); ++i) {
        result.push_back(cached_proto_->strings(i));
    }
    return result;
}

//==============================================================================
// OrtGraphAdapter Implementation
//==============================================================================

OrtGraphAdapter::OrtGraphAdapter(const OrtGraph* graph) : graph_(graph) {
    if (graph_) {
        name_ = graph_->GetName();
    }
}

size_t OrtGraphAdapter::GetNumNodes() const {
    if (!graph_) {
        return 0;
}
    return graph_->GetNumNodes();
}

//==============================================================================
// OrtNodeAdapter Implementation
//==============================================================================

OrtNodeAdapter::OrtNodeAdapter(const OrtNodePlugin* node)
: id_(0), since_version_(0), parent_graph_(nullptr),
  api_(OrtGetApiBase()->GetApi(ORT_API_VERSION))
{
    if (!node) {
        throw std::invalid_argument("OrtNodePlugin cannot be nullptr");
    }
    InitializeFromPlugin(node);
}

OrtNodeAdapter::OrtNodeAdapter(const OrtNode* node, const OrtApi& ort_api)
    : id_(0), since_version_(0), parent_graph_(nullptr),
      api_(&ort_api)
{
    if (!node) {
        throw std::invalid_argument("OrtNode cannot be nullptr");
    }
    InitializeFromCApi(node, ort_api);
}

void OrtNodeAdapter::InitializeFromPlugin(const OrtNodePlugin* node) {
    id_ = node->GetId();
    name_ = node->GetName();
    op_type_ = node->GetOpType();
    domain_ = node->GetDomain();

    int version = 0;
    auto status = node->GetSinceVersion(version);
    if (status.IsOK()) {
   since_version_ = version;
    }

    status = node->GetGraph(parent_graph_);
    if (!status.IsOK()) {
        parent_graph_ = nullptr;
    }

    // Inputs
    size_t num_inputs = node->GetNumInputs();
    if (num_inputs > 0) {
   std::vector<const OrtValueInfo*> input_ptrs(num_inputs);
        status = node->GetInputs(gsl::make_span(input_ptrs));
    if (status.IsOK()) {
            inputs_.reserve(num_inputs);
     for (const OrtValueInfo* input : input_ptrs) {
     inputs_.push_back(std::make_unique<OrtValueInfoAdapter>(input));
    }
   }
    }

    // Outputs
    size_t num_outputs = node->GetNumOutputs();
    if (num_outputs > 0) {
        std::vector<const OrtValueInfo*> output_ptrs(num_outputs);
        status = node->GetOutputs(gsl::make_span(output_ptrs));
        if (status.IsOK()) {
     outputs_.reserve(num_outputs);
       for (const OrtValueInfo* output : output_ptrs) {
      outputs_.push_back(std::make_unique<OrtValueInfoAdapter>(output));
       }
     }
    }

    // Implicit inputs
    size_t num_implicit_inputs = 0;
    status = node->GetNumImplicitInputs(num_implicit_inputs);
    if (status.IsOK() && num_implicit_inputs > 0) {
        std::vector<const OrtValueInfo*> implicit_input_ptrs(num_implicit_inputs);
        status = node->GetImplicitInputs(gsl::make_span(implicit_input_ptrs));
        if (status.IsOK()) {
   implicit_inputs_.reserve(num_implicit_inputs);
    for (const OrtValueInfo* input : implicit_input_ptrs) {
      implicit_inputs_.push_back(std::make_unique<OrtValueInfoAdapter>(input));
      }
        }
    }

// Attributes
    size_t num_attrs = node->GetNumAttributes();
 if (num_attrs > 0) {
  std::vector<const OrtOpAttr*> attr_ptrs(num_attrs);
        status = node->GetAttributes(gsl::make_span(attr_ptrs));
        if (status.IsOK()) {
            attributes_.reserve(num_attrs);
            for (size_t i = 0; i < num_attrs; ++i) {
                const OrtOpAttr* attr = attr_ptrs[i];
                if (!attr) continue;
                std::string attr_name = GetOpAttrName(attr, *api_);
                attributes_.push_back(std::make_unique<OrtOpAttrAdapter>(attr_name.c_str(), attr, *api_));
                attribute_map_[attr_name] = attributes_.size() - 1;
            }
        }
    }

    // Subgraphs
    size_t num_subgraphs = 0;
    status = node->GetNumSubgraphs(num_subgraphs);
    if (status.IsOK() && num_subgraphs > 0) {
    std::vector<const OrtGraph*> subgraph_ptrs(num_subgraphs);
        std::vector<const char*> attr_name_ptrs(num_subgraphs, nullptr);
status = node->GetSubgraphs(gsl::make_span(subgraph_ptrs), attr_name_ptrs.data());
if (status.IsOK()) {
            subgraphs_.reserve(num_subgraphs);
      for (size_t i = 0; i < num_subgraphs; ++i) {
                if (subgraph_ptrs[i]) {
        subgraphs_.push_back(std::make_unique<OrtGraphAdapter>(subgraph_ptrs[i]));
  }
            }
        }
  }
}

void OrtNodeAdapter::InitializeFromCApi(const OrtNode* node, const OrtApi& ort_api) {
    ort_api.Node_GetId(node, &id_);

    const char* name_ptr = nullptr;
    ort_api.Node_GetName(node, &name_ptr);
    if (name_ptr) name_ = name_ptr;

    const char* op_type_ptr = nullptr;
    ort_api.Node_GetOperatorType(node, &op_type_ptr);
    if (op_type_ptr) op_type_ = op_type_ptr;

    const char* domain_ptr = nullptr;
    ort_api.Node_GetDomain(node, &domain_ptr);
    if (domain_ptr) domain_ = domain_ptr;

    ort_api.Node_GetSinceVersion(node, &since_version_);

    // Inputs
    size_t num_inputs = 0;
    ort_api.Node_GetNumInputs(node, &num_inputs);
    if (num_inputs > 0) {
        std::vector<const OrtValueInfo*> input_ptrs(num_inputs);
     ort_api.Node_GetInputs(node, input_ptrs.data(), num_inputs);
        inputs_.reserve(num_inputs);
      for (const OrtValueInfo* input : input_ptrs) {
            inputs_.push_back(std::make_unique<OrtValueInfoAdapter>(input));
        }
    }

    // Outputs
    size_t num_outputs = 0;
    ort_api.Node_GetNumOutputs(node, &num_outputs);
    if (num_outputs > 0)
    {
        std::vector<const OrtValueInfo*> output_ptrs(num_outputs);
        ort_api.Node_GetOutputs(node, output_ptrs.data(), num_outputs);
        outputs_.reserve(num_outputs);
        for (const OrtValueInfo* output : output_ptrs)
        {
            outputs_.push_back(std::make_unique<OrtValueInfoAdapter>(output));
        }
    }

    // Attributes
    size_t num_attrs = 0;
    ort_api.Node_GetNumAttributes(node, &num_attrs);
    if (num_attrs > 0)
    {
        std::vector<const OrtOpAttr*> attr_ptrs(num_attrs);
      ort_api.Node_GetAttributes(node, attr_ptrs.data(), num_attrs);
        attributes_.reserve(num_attrs);
        for (size_t i = 0; i < num_attrs; ++i)
        {
            const OrtOpAttr* attr = attr_ptrs[i];
            if (!attr)
                continue;
            std::string attr_name = GetOpAttrName(attr, ort_api);
            attributes_.push_back(std::make_unique<OrtOpAttrAdapter>(attr_name.c_str(), attr, ort_api));
            attribute_map_[attr_name] = attributes_.size() - 1;
        }
    }
}

const OrtValueInfoAdapter* OrtNodeAdapter::GetInput(size_t index) const {
    return index < inputs_.size() ? inputs_[index].get() : nullptr;
}

const OrtValueInfoAdapter* OrtNodeAdapter::GetOutput(size_t index) const {
  return index < outputs_.size() ? outputs_[index].get() : nullptr;
}

const OrtOpAttrAdapter* OrtNodeAdapter::GetAttribute(const char* name) const {
    auto it = attribute_map_.find(name);
    return it != attribute_map_.end() ? attributes_[it->second].get() : nullptr;
}

float OrtNodeAdapter::GetAttributeFloat(const char* name, float default_value) const {
    const OrtOpAttrAdapter* attr = GetAttribute(name);
  if (!attr) return default_value;
    try { return attr->GetFloat(); } catch (...) { return default_value; }
}

int64_t OrtNodeAdapter::GetAttributeInt(const char* name, int64_t default_value) const {
    const OrtOpAttrAdapter* attr = GetAttribute(name);
    if (!attr) return default_value;
    try { return attr->GetInt(); } catch (...) { return default_value; }
}

const char* OrtNodeAdapter::GetAttributeString(const char* name, const char* default_value) const {
    const OrtOpAttrAdapter* attr = GetAttribute(name);
    if (!attr) return default_value;
    try { return attr->GetString(); } catch (...) { return default_value; }
}

std::vector<float> OrtNodeAdapter::GetAttributeFloats(const char* name) const {
    const OrtOpAttrAdapter* attr = GetAttribute(name);
    if (!attr) return {};
    try { return attr->GetFloats(); } catch (...) { return {}; }
}

std::vector<int64_t> OrtNodeAdapter::GetAttributeInts(const char* name) const {
    const OrtOpAttrAdapter* attr = GetAttribute(name);
    if (!attr) return {};
    try { return attr->GetInts(); } catch (...) { return {}; }
}

const OrtGraphAdapter* OrtNodeAdapter::GetSubgraph(size_t index) const {
    return index < subgraphs_.size() ? subgraphs_[index].get() : nullptr;
}

std::string OrtNodeAdapter::ToString() const {
    std::ostringstream oss;
 oss << "Node{id=" << id_
        << ", name=\"" << name_ << "\""
        << ", op=\"" << op_type_ << "\""
    << ", domain=\"" << domain_ << "\""
   << ", version=" << since_version_
        << ", inputs=" << inputs_.size()
  << ", outputs=" << outputs_.size()
        << ", attrs=" << attributes_.size()
        << ", subgraphs=" << subgraphs_.size()
  << "}";
    return oss.str();
}

}  // namespace dml_ep
