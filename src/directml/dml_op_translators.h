// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <DirectML.h>
#include <onnxruntime_c_api.h>
#include <wrl/client.h>

namespace dml_ep {

struct DmlTensorInfo {
    DML_TENSOR_DATA_TYPE  data_type = DML_TENSOR_DATA_TYPE_UNKNOWN;
    std::vector<uint32_t> sizes;
    std::vector<uint32_t> strides;
    uint64_t              total_bytes = 0;
    uint32_t              original_rank = 0;  // rank before 4D padding (0 = unknown)

    DML_BUFFER_TENSOR_DESC ToBufferDesc() const;
};

// A sub-node within a multi-operator expansion.  When a single ONNX node
// translates to multiple DML operators (e.g. GroupNorm + SiLU), extra ops
// beyond the primary are stored here.
struct SubNode {
    std::vector<DmlTensorInfo>                 input_tensors;
    std::vector<DmlTensorInfo>                 output_tensors;
    std::vector<DML_BUFFER_TENSOR_DESC>        input_buffer_descs;
    std::vector<DML_TENSOR_DESC>               input_tensor_descs;
    std::vector<DML_BUFFER_TENSOR_DESC>        output_buffer_descs;
    std::vector<DML_TENSOR_DESC>               output_tensor_descs;
    std::shared_ptr<void>                      desc_storage;
    DML_OPERATOR_DESC                          op_desc{};
    Microsoft::WRL::ComPtr<IDMLOperator>       dml_operator;

    // Each input comes from {sub_node_index, output_slot}.
    // -1 means the primary node's output.
    std::vector<std::pair<int, int>>           input_from;

    using FixupFn = std::function<void(SubNode& self)>;
    FixupFn fixup;
    void FixupPointers() { if (fixup) fixup(*this); }
};

struct TranslatedOp {
    std::vector<DML_BUFFER_TENSOR_DESC>        input_buffer_descs;
    std::vector<DML_TENSOR_DESC>               input_tensor_descs;
    std::vector<DML_BUFFER_TENSOR_DESC>        output_buffer_descs;
    std::vector<DML_TENSOR_DESC>               output_tensor_descs;

    std::vector<DmlTensorInfo>                 input_tensors;
    std::vector<DmlTensorInfo>                 output_tensors;

    std::shared_ptr<void>                      desc_storage;
    DML_OPERATOR_DESC                          op_desc{};
    Microsoft::WRL::ComPtr<IDMLOperator>       dml_operator;

    // When true, this node is a no-op (e.g. Reshape) that should be elided
    // from the DML graph.  Edges pass through: output[0] aliases input[0].
    bool passthrough = false;

    // Additional DML operators for multi-op expansion (e.g. GroupNorm+SiLU).
    // When non-empty, the graph output comes from the last sub_node.
    std::vector<SubNode> sub_nodes;

    // Re-links internal pointers (DML_TENSOR_DESC → DML_BUFFER_TENSOR_DESC,
    // operator desc fields → tensor descs) after the struct has been moved
    // to its final memory location.  Must be called before CreateOperator.
    using FixupFn = std::function<void(TranslatedOp& self)>;
    FixupFn fixup;

    void FixupPointers() {
        if (fixup) fixup(*this);
        for (auto& sn : sub_nodes) sn.FixupPointers();
    }
};

using OpTranslatorFn = std::function<std::optional<TranslatedOp>(
    const OrtApi&                                            ort_api,
    const OrtNode*                                           node,
    const std::unordered_map<std::string, DmlTensorInfo>&    value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&  initializers)>;

using OpTranslatorRegistry = std::unordered_map<std::string, OpTranslatorFn>;

OpTranslatorRegistry BuildOpTranslatorRegistry();

// Pad shape with leading 1s to reach at least min_dims dimensions.
std::vector<uint32_t> PadToMinDims(const std::vector<uint32_t>& sizes, size_t min_dims = 4);

// Compute packed strides for a shape (descending / row-major).
std::vector<uint32_t> ComputePackedStrides(const std::vector<uint32_t>& sizes);

// Compute DWORD-aligned total tensor size in bytes (minimum 4).
uint64_t ComputeAlignedTotalBytes(const std::vector<uint32_t>& sizes, DML_TENSOR_DATA_TYPE dtype);

// Compute element size in bytes for a DML data type.
size_t DmlDataTypeSize(DML_TENSOR_DATA_TYPE dtype);

// Build a DmlTensorInfo from sizes and data type, with 4D padding and alignment.
DmlTensorInfo MakeTensorInfo(const std::vector<uint32_t>& sizes, DML_TENSOR_DATA_TYPE dtype);

// Build a DmlTensorInfo placing source dims at a given axis within a target
// dimension count.  Mirrors ORT's TensorDesc(placement=C, LeftAligned) pattern.
// e.g. sizes=[384], placement=1, target_dim_count=4 → [1,384,1,1]
DmlTensorInfo MakeTensorInfoAtAxis(
    const std::vector<uint32_t>& sizes,
    DML_TENSOR_DATA_TYPE dtype,
    uint32_t placement,
    uint32_t target_dim_count);

// Build a DmlTensorInfo with explicit strides (padded to 4D).
DmlTensorInfo MakeTensorInfoWithStrides(
    const std::vector<uint32_t>& sizes,
    const std::vector<uint32_t>& strides,
    DML_TENSOR_DATA_TYPE dtype,
    uint64_t total_bytes);

// Broadcast two shapes following ONNX multidirectional broadcast rules.
// Returns (broadcast_a, broadcast_b, output_sizes) or nullopt on incompatible shapes.
struct BroadcastResult {
    DmlTensorInfo a;
    DmlTensorInfo b;
    std::vector<uint32_t> output_sizes;
};
std::optional<BroadcastResult> BroadcastShapes(
    const DmlTensorInfo& a,
    const DmlTensorInfo& b);

// Map ONNX element data type to DML tensor data type.
DML_TENSOR_DATA_TYPE OnnxDtypeToDml(ONNXTensorElementDataType onnx_dtype);

}  // namespace dml_ep
