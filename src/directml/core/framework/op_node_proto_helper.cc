// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/onnx_protobuf.h"
#include "core/graph/op.h"
#include <gsl/gsl>
#include "core/graph/abi_graph_types.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/framework/abi_safe_attr_utils.h"
using namespace ONNX_NAMESPACE;

namespace onnxruntime {

template <class T>
bool HasTyped(const AttributeProto*);

template <>
inline bool HasTyped<float>(const AttributeProto* attr) {
  return utils::HasFloat(*attr);
}
template <>
inline bool HasTyped<int64_t>(const AttributeProto* attr) {
  return utils::HasInt(*attr);
}
template <>
inline bool HasTyped<std::string>(const AttributeProto* attr) {
  return utils::HasString(*attr);
}

template <>
inline bool HasTyped<TensorProto>(const AttributeProto* attr) {
  return utils::HasTensor(*attr);
}
template <>
inline bool HasTyped<GraphProto>(const AttributeProto* attr) {
  return utils::HasGraph(*attr);
}

template <typename T>
inline bool HasTypedList(const AttributeProto* attr);

template <>
inline bool HasTypedList<float>(const AttributeProto* attr) {
  return utils::HasFloats(*attr);
}

template <>
inline bool HasTypedList<int64_t>(const AttributeProto* attr) {
  return utils::HasInts(*attr);
}

template <>
inline bool HasTypedList<std::string>(const AttributeProto* attr) {
  return utils::HasStrings(*attr);
}

template <typename T>
inline constexpr int ArrayTypeToAttributeType();

template <>
inline constexpr int ArrayTypeToAttributeType<float>() {
  return AttributeProto_AttributeType_FLOATS;
}

template <>
inline constexpr int ArrayTypeToAttributeType<int64_t>() {
  return AttributeProto_AttributeType_INTS;
}

template <>
inline constexpr int ArrayTypeToAttributeType<std::string>() {
  return AttributeProto_AttributeType_STRINGS;
}

#define ORT_DEFINE_GET_ATTR(IMPL_T, T, type)                                                       \
  template <>                                                                                      \
  template <>                                                                                      \
  Ort::Status OpNodeProtoHelper<IMPL_T>::GetAttr<T>(                                                    \
      const std::string& name, T* value) const {                                                   \
    const AttributeProto* attr = TryGetAttribute(name);                                            \
    if (!attr) {                                                                                   \
      return ORT_MAKE_STATUS(ORT_FAIL, "No attribute with name:'", name, "'is defined."); \
    }                                                                                              \
    if (!HasTyped<T>(attr)) {                                                                      \
      return ORT_MAKE_STATUS(ORT_FAIL, "Attribute name and type don't match");                     \
    } else {                                                                                       \
      *value = static_cast<T>(attr->type());                                                       \
      return STATUS_OK;                                                                         \
    }                                                                                              \
  }

#define ORT_DEFINE_GET_ATTRS(IMPL_T, T, list)                                                                \
  template <>                                                                                                \
  template <>                                                                                                \
  Ort::Status OpNodeProtoHelper<IMPL_T>::GetAttrs<T>(                                                             \
      const std::string& name, std::vector<T>& values) const {                                               \
    const AttributeProto* attr = TryGetAttribute(name);                                                      \
    if (!attr) {                                                                                             \
      return ORT_MAKE_STATUS(ORT_FAIL, "No attribute with this name is defined.");                           \
    }                                                                                                        \
    values.reserve(attr->list##_size());                                                                     \
    for (int i = 0; i < attr->list##_size(); ++i) {                                                          \
      values.push_back(static_cast<T>(attr->list(i)));                                                       \
    }                                                                                                        \
    return STATUS_OK;                                                                                     \
  }                                                                                                          \
  template <>                                                                                                \
  template <>                                                                                                \
  Ort::Status OpNodeProtoHelper<IMPL_T>::GetAttrs<T>(                                                             \
      const std::string& name, gsl::span<T> values) const {                                                  \
    const AttributeProto* attr = TryGetAttribute(name);                                                      \
    if (!attr) {                                                                                             \
      return ORT_MAKE_STATUS(ORT_FAIL, "No attribute with this name is defined.");                           \
    }                                                                                                        \
    ORT_RETURN_IF(values.size() != static_cast<size_t>(attr->list##_size()),                                 \
                  "GetAttrs failed. Expect values.size()=", (attr->list##_size()), ", got ", values.size()); \
    for (int i = 0; i < attr->list##_size(); ++i) {                                                          \
      values[i] = static_cast<T>(attr->list(i));                                                             \
    }                                                                                                        \
    return STATUS_OK;                                                                                     \
  }

// Will not work for std::strings
#define ORT_DEFINE_GET_ATTRS_AS_SPAN(IMPL_T, T, list)                                              \
  template <>                                                                                      \
  template <>                                                                                      \
  Ort::Status OpNodeProtoHelper<IMPL_T>::GetAttrsAsSpan<T>(                                             \
      const std::string& name, gsl::span<const T>& values) const {                                 \
    const AttributeProto* attr = TryGetAttribute(name);                                            \
    if (!attr) {                                                                                   \
      return ORT_MAKE_STATUS(ORT_FAIL, "No attribute with name: ", name, " is defined."); \
    }                                                                                              \
    if (!HasTypedList<T>(attr)) {                                                                  \
      return ORT_MAKE_STATUS(ORT_FAIL, "Attribute: ", name,                               \
                             " expected to be of type: ",                                          \
                             AttributeProto::AttributeType_Name(ArrayTypeToAttributeType<T>()),    \
                             " but is of type: ",                                                  \
                             AttributeProto::AttributeType_Name(attr->type()));                    \
    }                                                                                              \
    values = gsl::make_span<const T>(reinterpret_cast<const T*>(attr->list().data()),              \
                                     attr->list##_size());                                         \
    return STATUS_OK;                                                                           \
  }

#if !defined(ORT_MINIMAL_BUILD)
#define ORT_DEFINE_GET_ATTR_SPECIALIZATIONS(type, list)                                                                \
    ORT_DEFINE_GET_ATTR(ProtoHelperNodeContext, type, list)                                                            \
    ORT_DEFINE_GET_ATTR(AbiSafeProtoHelperNodeContext, type, list)                                                     \
    ORT_DEFINE_GET_ATTR(InferenceContext, type, list)

#define ORT_DEFINE_GET_ATTRS_SPECIALIZATIONS(type, list)                                                               \
    ORT_DEFINE_GET_ATTRS(ProtoHelperNodeContext, type, list)                                                           \
    ORT_DEFINE_GET_ATTRS(AbiSafeProtoHelperNodeContext, type, list)                                                    \
    ORT_DEFINE_GET_ATTRS(InferenceContext, type, list)

#define ORT_DEFINE_GET_ATTRS_SPAN_SPECIALIZATION(type, list)                                                           \
    ORT_DEFINE_GET_ATTRS_AS_SPAN(ProtoHelperNodeContext, type, list)                                                   \
    ORT_DEFINE_GET_ATTRS_AS_SPAN(AbiSafeProtoHelperNodeContext, type, list)                                            \
    ORT_DEFINE_GET_ATTRS_AS_SPAN(InferenceContext, type, list)

#else
#define ORT_DEFINE_GET_ATTR_SPECIALIZATIONS(type, list)                                                                \
    ORT_DEFINE_GET_ATTR(ProtoHelperNodeContext, type, list)                                                            \
    ORT_DEFINE_GET_ATTR(AbiSafeProtoHelperNodeContext, type, list)

#define ORT_DEFINE_GET_ATTRS_SPECIALIZATIONS(type, list)                                                               \
    ORT_DEFINE_GET_ATTRS(ProtoHelperNodeContext, type, list)                                                           \
    ORT_DEFINE_GET_ATTRS(AbiSafeProtoHelperNodeContext, type, list)

#define ORT_DEFINE_GET_ATTRS_SPAN_SPECIALIZATION(type, list)                                                           \
    ORT_DEFINE_GET_ATTRS_AS_SPAN(ProtoHelperNodeContext, type, list)                                                   \
    ORT_DEFINE_GET_ATTRS_AS_SPAN(AbiSafeProtoHelperNodeContext, type, list)
#endif

ORT_DEFINE_GET_ATTR_SPECIALIZATIONS(float, f)
ORT_DEFINE_GET_ATTR_SPECIALIZATIONS(int64_t, i)
ORT_DEFINE_GET_ATTR_SPECIALIZATIONS(std::string, s)
ORT_DEFINE_GET_ATTR_SPECIALIZATIONS(TensorProto, t)
ORT_DEFINE_GET_ATTR_SPECIALIZATIONS(GraphProto, g)
ORT_DEFINE_GET_ATTRS_SPECIALIZATIONS(float, floats)
ORT_DEFINE_GET_ATTRS_SPECIALIZATIONS(int64_t, ints)
ORT_DEFINE_GET_ATTRS_SPECIALIZATIONS(std::string, strings)
ORT_DEFINE_GET_ATTRS_SPECIALIZATIONS(TensorProto, tensors)
ORT_DEFINE_GET_ATTRS_SPECIALIZATIONS(GraphProto, graphs)

ORT_DEFINE_GET_ATTRS_SPAN_SPECIALIZATION(float, floats)
ORT_DEFINE_GET_ATTRS_SPAN_SPECIALIZATION(int64_t, ints)

template <typename Impl_t>
Ort::Status OpNodeProtoHelper<Impl_t>::GetAttrs(const std::string& name, TensorShapeVector& out) const {
  gsl::span<const int64_t> span;
  Ort::Status status = this->GetAttrsAsSpan<int64_t>(name, span);
  if (status.IsOK()) {
    out.reserve(span.size());
    out.assign(span.begin(), span.end());
  }
  return status;
}

template <typename Impl_t>
Ort::Status OpNodeProtoHelper<Impl_t>::GetAttrsStringRefs(
    const std::string& name,
    std::vector<std::reference_wrapper<const std::string>>& refs) const {
  const AttributeProto* attr = TryGetAttribute(name);
  if (!attr) {
    return ORT_MAKE_STATUS(ORT_FAIL, "No attribute with name: ", name, " is defined.");
  }
  if (!HasTypedList<std::string>(attr)) {
    return ORT_MAKE_STATUS(ORT_FAIL, "Requested attribute: ",
                           name, " is expected to have type: ",
                           AttributeProto::AttributeType_Name(AttributeProto_AttributeType_STRINGS),
                           " but is of type: ",
                           AttributeProto::AttributeType_Name(attr->type()));
  }
  std::vector<std::reference_wrapper<const std::string>> result;
  if (attr->strings_size() > 0) {
    result.reserve(attr->strings_size());
    std::copy(attr->strings().cbegin(), attr->strings().cend(), std::back_inserter(result));
  }
  refs.swap(result);
  return STATUS_OK;
}

size_t ProtoHelperNodeContext::getNumInputs() const {
  return node_.InputDefs().size();
}

size_t ProtoHelperNodeContext::getNumOutputs() const {
  return node_.OutputDefs().size();
}

const AttributeProto* ProtoHelperNodeContext::getAttribute(const std::string& name) const {
  const onnxruntime::NodeAttributes& attributes = node_.GetAttributes();
  auto it = attributes.find(name);
  if (it != attributes.end()) {
    return &it->second;
  }
  return nullptr;
}

const TypeProto* ProtoHelperNodeContext::getInputType(size_t index) const {
  return node_.InputDefs()[index]->TypeAsProto();
}

const TypeProto* ProtoHelperNodeContext::getOutputType(size_t index) const {
  return node_.OutputDefs()[index]->TypeAsProto();
}

AbiSafeProtoHelperNodeContext::AbiSafeProtoHelperNodeContext(const OrtNodePlugin* node)
    : plugin_node_(node), api_(OrtGetApiBase()->GetApi(ORT_API_VERSION))
{
    InitializeFromPlugin(node);
}

AbiSafeProtoHelperNodeContext::AbiSafeProtoHelperNodeContext(const OrtNode* node, const OrtApi& ort_api)
    : plugin_node_(nullptr), api_(&ort_api)
{
    InitializeFromCApi(node, ort_api);
}

void AbiSafeProtoHelperNodeContext::InitializeFromPlugin(const OrtNodePlugin* node)
{

    if (!node)
    {
      return;
    }

    // Get inputs using virtual method API
    size_t num_inputs = node->GetNumInputs();
    if (num_inputs > 0)
    {
        inputs_.resize(num_inputs);
        auto status = node->GetInputs(gsl::make_span(inputs_));
    }

    // Get outputs using virtual method API
    size_t num_outputs = node->GetNumOutputs();
    if (num_outputs > 0)
    {
        outputs_.resize(num_outputs);
        auto status = node->GetOutputs(gsl::make_span(outputs_));
    }

    // Get attributes using virtual method API, then build plugin-side AttributeProto copies
    size_t num_attrs = node->GetNumAttributes();
    if (num_attrs > 0)
    {
        std::vector<const OrtOpAttr*> attr_values(num_attrs);
        auto status = node->GetAttributes(gsl::make_span(attr_values));

        for (size_t i = 0; i < num_attrs; ++i)
        {
            if (attr_values[i])
            {
                std::string name = dml_ep::GetOpAttrName(attr_values[i], *api_);
                auto built = dml_ep::BuildPluginAttributeProto(attr_values[i], *api_);
                attributes_[name] = std::move(built);
            }
        }
    }
}

void AbiSafeProtoHelperNodeContext::InitializeFromCApi(const OrtNode* node, const OrtApi& ort_api) {
    if (!node) {
        return;
    }

    // Get inputs via C API
    size_t num_inputs = 0;
    ort_api.Node_GetNumInputs(node, &num_inputs);
    if (num_inputs > 0)
    {
        inputs_.resize(num_inputs);
        ort_api.Node_GetInputs(node, inputs_.data(), num_inputs);
    }

    // Get outputs via C API
    size_t num_outputs = 0;
    ort_api.Node_GetNumOutputs(node, &num_outputs);
    if (num_outputs > 0)
    {
        outputs_.resize(num_outputs);
        ort_api.Node_GetOutputs(node, outputs_.data(), num_outputs);
    }

    // Get attributes via C API, then build plugin-side AttributeProto copies
    size_t num_attrs = 0;
    ort_api.Node_GetNumAttributes(node, &num_attrs);
    if (num_attrs > 0)
    {
        std::vector<const OrtOpAttr*> attr_values(num_attrs);
        ort_api.Node_GetAttributes(node, attr_values.data(), num_attrs);

        for (size_t i = 0; i < num_attrs; ++i)
        {
            if (attr_values[i])
            {
                std::string name = dml_ep::GetOpAttrName(attr_values[i], ort_api);
                auto built = dml_ep::BuildPluginAttributeProto(attr_values[i], ort_api);
                attributes_[name] = std::move(built);
            }
        }
    }
}

const ONNX_NAMESPACE::AttributeProto* AbiSafeProtoHelperNodeContext::getAttribute(const std::string& name) const {
    auto it = attributes_.find(name);
    if (it != attributes_.end() && it->second) {
        return it->second.get();
    }
    return nullptr;
}

size_t AbiSafeProtoHelperNodeContext::getNumInputs() const { return inputs_.size(); }

const OrtTypeInfo* AbiSafeProtoHelperNodeContext::getInputType(size_t index) const {
    if (index >= inputs_.size() || !inputs_[index]) {
        return nullptr;
    }
    return inputs_[index]->GetTypeInfo();
}

size_t AbiSafeProtoHelperNodeContext::getNumOutputs() const { return outputs_.size(); }

const OrtTypeInfo* AbiSafeProtoHelperNodeContext::getOutputType(size_t index) const {
    if (index >= outputs_.size() || !outputs_[index]) {
        return nullptr;
    }
    return outputs_[index]->GetTypeInfo();
}

template <class Impl_t>
uint32_t OpNodeProtoHelper<Impl_t>::GetPrimitiveAttrElementCount(AttributeProto_AttributeType type, const std::string& name) const noexcept {
    const AttributeProto* attr = impl_->getAttribute(name);
    if (attr)
    {
        switch (type) {
            case AttributeProto_AttributeType_FLOAT:
            case AttributeProto_AttributeType_INT:
            case AttributeProto_AttributeType_STRING:
                return 1;

            case AttributeProto_AttributeType_FLOATS:
                return attr->floats_size();
            case AttributeProto_AttributeType_INTS:
                return attr->ints_size();
            case AttributeProto_AttributeType_STRINGS:
                return attr->strings_size();

            case AttributeProto_AttributeType_UNDEFINED:
            case AttributeProto_AttributeType_TENSOR:
            case AttributeProto_AttributeType_GRAPH:
            case AttributeProto_AttributeType_SPARSE_TENSOR:
            case AttributeProto_AttributeType_TENSORS:
            case AttributeProto_AttributeType_GRAPHS:
            case AttributeProto_AttributeType_SPARSE_TENSORS:
            default:
            return 0;
        }
    }

    return 0;
}

template <class Impl_t>
bool OpNodeProtoHelper<Impl_t>::HasPrimitiveAttribute(AttributeProto_AttributeType type, const std::string& name) const noexcept {
  return GetPrimitiveAttrElementCount(type, name) > 0;
}

template class OpNodeProtoHelper<ProtoHelperNodeContext>;
template class OpNodeProtoHelper<AbiSafeProtoHelperNodeContext>;
#if !defined(ORT_MINIMAL_BUILD)
template class OpNodeProtoHelper<InferenceContext>;
#endif

}  // namespace onnxruntime
