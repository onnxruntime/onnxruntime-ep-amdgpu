// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/graph/abi_graph_types.h"
#include <onnxruntime_c_api.h>
#include <onnx/onnx_pb.h>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace Dml {

// Forward declarations
class OrtNodeAdapter;
class OrtValueInfoAdapter;
class OrtOpAttrAdapter;
class OrtGraphAdapter;

//==============================================================================
// OrtValueInfoAdapter - Wraps const OrtValueInfo* with helper methods
//==============================================================================
class OrtValueInfoAdapter {
public:
    explicit OrtValueInfoAdapter(const OrtValueInfo* value_info);
    ~OrtValueInfoAdapter() = default;

    // Prevent copying, allow moving
    OrtValueInfoAdapter(const OrtValueInfoAdapter&) = delete;
    OrtValueInfoAdapter& operator=(const OrtValueInfoAdapter&) = delete;
    OrtValueInfoAdapter(OrtValueInfoAdapter&&) = default;
    OrtValueInfoAdapter& operator=(OrtValueInfoAdapter&&) = default;

    const char* GetName() const { return name_.c_str(); }
    const OrtValueInfo* GetOrtValueInfo() const { return value_info_; }
    const OrtTypeInfo* GetTypeInfo() const { return type_info_; }

    // Type checking
    bool IsTensor() const;
    bool IsSequence() const;
    bool IsMap() const;
    bool IsOptional() const;

    // Tensor-specific (returns ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED if not a tensor)
    ONNXTensorElementDataType GetTensorElementType() const;

private:
    const OrtValueInfo* value_info_;
    std::string name_;
    const OrtTypeInfo* type_info_;

    // Cached type info
    mutable bool type_info_cached_;
    mutable ONNXType onnx_type_;
    mutable ONNXTensorElementDataType tensor_elem_type_;

    void CacheTypeInfo() const;
};

//==============================================================================
// OrtOpAttrAdapter - Wraps const OrtOpAttr* with type-safe attribute access
//==============================================================================
class OrtOpAttrAdapter {
public:
    OrtOpAttrAdapter(const char* name, const OrtOpAttr* attr, const OrtApi& api);
    ~OrtOpAttrAdapter() = default;

    const char* GetName() const { return name_.c_str(); }
    const OrtOpAttr* GetOrtOpAttr() const { return attr_; }

    // Returns a plugin-side AttributeProto built via the C API (ABI-safe).
    const ONNX_NAMESPACE::AttributeProto& GetAttrProto() const;

    // Type-safe getters (use cached plugin-side AttributeProto)
    float GetFloat() const;
    int64_t GetInt() const;
    const char* GetString() const;
    std::vector<float> GetFloats() const;
    std::vector<int64_t> GetInts() const;
    std::vector<std::string> GetStrings() const;

private:
    std::string name_;
    const OrtOpAttr* attr_;
    const OrtApi* api_;
    mutable std::unique_ptr<ONNX_NAMESPACE::AttributeProto> cached_proto_;

    void EnsureCachedProto() const;
};

//==============================================================================
// OrtGraphAdapter - Wraps const OrtGraph* for subgraph access
//==============================================================================
class OrtGraphAdapter {
public:
    explicit OrtGraphAdapter(const OrtGraph* graph);
    ~OrtGraphAdapter() = default;

    const OrtGraph* GetOrtGraph() const { return graph_; }
    const char* GetName() const { return name_.c_str(); }
    size_t GetNumNodes() const;

private:
    const OrtGraph* graph_;
    std::string name_;
};

//==============================================================================
// OrtNodeAdapter - Full replacement for EpNode using ORT C API
// Uses ort_api function pointers to access node properties (ABI-safe).
//==============================================================================
class OrtNodeAdapter {
public:
    // Construction from opaque ORT C API node + API function table
    explicit OrtNodeAdapter(const OrtNode* node, const OrtApi& ort_api);

    // Construction from local plugin node type (uses virtual methods directly)
    explicit OrtNodeAdapter(const OrtNodePlugin* node);

    ~OrtNodeAdapter() = default;

    // Prevent copying, allow moving
    OrtNodeAdapter(const OrtNodeAdapter&) = delete;
    OrtNodeAdapter& operator=(const OrtNodeAdapter&) = delete;
    OrtNodeAdapter(OrtNodeAdapter&&) = default;
    OrtNodeAdapter& operator=(OrtNodeAdapter&&) = default;

    //==========================================================================
    // Core Node Information
    //==========================================================================

    size_t GetId() const { return id_; }
    const std::string& GetName() const { return name_; }
    const std::string& GetOpType() const { return op_type_; }
    const std::string& GetDomain() const { return domain_; }
    int GetSinceVersion() const { return since_version_; }

    //==========================================================================
    // Inputs/Outputs
    //==========================================================================

    size_t GetNumInputs() const { return inputs_.size(); }
    const std::vector<std::unique_ptr<OrtValueInfoAdapter>>& GetInputs() const { return inputs_; }
    const OrtValueInfoAdapter* GetInput(size_t index) const;

    size_t GetNumOutputs() const { return outputs_.size(); }
    const std::vector<std::unique_ptr<OrtValueInfoAdapter>>& GetOutputs() const { return outputs_; }
    const OrtValueInfoAdapter* GetOutput(size_t index) const;

    size_t GetNumImplicitInputs() const { return implicit_inputs_.size(); }
    const std::vector<std::unique_ptr<OrtValueInfoAdapter>>& GetImplicitInputs() const { return implicit_inputs_; }

    //==========================================================================
    // Attributes
    //==========================================================================

    size_t GetNumAttributes() const { return attributes_.size(); }
    const std::vector<std::unique_ptr<OrtOpAttrAdapter>>& GetAttributes() const { return attributes_; }

    // Get attribute by name (returns nullptr if not found)
    const OrtOpAttrAdapter* GetAttribute(const char* name) const;

    // Type-safe attribute getters with default values
    float GetAttributeFloat(const char* name, float default_value = 0.0f) const;
    int64_t GetAttributeInt(const char* name, int64_t default_value = 0) const;
    const char* GetAttributeString(const char* name, const char* default_value = "") const;
    std::vector<float> GetAttributeFloats(const char* name) const;
    std::vector<int64_t> GetAttributeInts(const char* name) const;

    //==========================================================================
    // Subgraphs
    //==========================================================================

    size_t GetNumSubgraphs() const { return subgraphs_.size(); }
    const std::vector<std::unique_ptr<OrtGraphAdapter>>& GetSubgraphs() const { return subgraphs_; }
    const OrtGraphAdapter* GetSubgraph(size_t index) const;

    //==========================================================================
    // Parent Graph
    //==========================================================================

    const OrtGraph* GetParentGraph() const { return parent_graph_; }

    //==========================================================================
    // Convenience Methods
    //==========================================================================

    // Iterator-style traversal (replacement for Node::ForEachDef)
    template<typename Func>
    void ForEachInput(Func callback) const {
        for (const auto& input : inputs_) {
            if (input) {
                callback(*input);
            }
        }
    }

    template<typename Func>
    void ForEachOutput(Func callback) const {
        for (const auto& output : outputs_) {
            if (output) {
                callback(*output);
            }
        }
    }

    template<typename Func>
    void ForEachDef(Func callback) const {
        for (const auto& input : inputs_) {
            if (input) {
                callback(*input, true);  // true = isInput
            }
        }
        for (const auto& output : outputs_) {
            if (output) {
                callback(*output, false);  // false = isOutput
            }
        }
    }

    // Attribute existence check
    bool HasAttribute(const char* name) const {
        return attribute_map_.find(name) != attribute_map_.end();
    }

    // String representation for debugging
    std::string ToString() const;

private:
    // Cached node properties
    size_t id_;
    std::string name_;
    std::string op_type_;
    std::string domain_;
    int since_version_;

    // Inputs/Outputs
    std::vector<std::unique_ptr<OrtValueInfoAdapter>> inputs_;
    std::vector<std::unique_ptr<OrtValueInfoAdapter>> outputs_;
    std::vector<std::unique_ptr<OrtValueInfoAdapter>> implicit_inputs_;

    // Attributes
    std::vector<std::unique_ptr<OrtOpAttrAdapter>> attributes_;
    std::unordered_map<std::string, size_t> attribute_map_;  // name -> index

    // Subgraphs
    std::vector<std::unique_ptr<OrtGraphAdapter>> subgraphs_;

    // Parent graph
    const OrtGraph* parent_graph_;

    // ORT API for ABI-safe attribute access
    const OrtApi* api_;

    // Initialization from OrtNodePlugin* (virtual method calls - used within plugin)
    void InitializeFromPlugin(const OrtNodePlugin* node);

    // Initialization from OrtNode* + OrtApi (C API calls - ABI-safe)
    void InitializeFromCApi(const OrtNode* node, const OrtApi& ort_api);
};

} // namespace Dml
