// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "dml_op_translators.h"
#include "fusion_utils.h"
#include "ort_node_adapter.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <numeric>

namespace dml_ep {

// ---------------------------------------------------------------------------
// Utility implementations
// ---------------------------------------------------------------------------

size_t DmlDataTypeSize(DML_TENSOR_DATA_TYPE dtype) {
    switch (dtype) {
    case DML_TENSOR_DATA_TYPE_FLOAT32: return 4;
    case DML_TENSOR_DATA_TYPE_FLOAT16: return 2;
    case DML_TENSOR_DATA_TYPE_UINT32:  return 4;
    case DML_TENSOR_DATA_TYPE_UINT16:  return 2;
    case DML_TENSOR_DATA_TYPE_UINT8:   return 1;
    case DML_TENSOR_DATA_TYPE_INT32:   return 4;
    case DML_TENSOR_DATA_TYPE_INT16:   return 2;
    case DML_TENSOR_DATA_TYPE_INT8:    return 1;
    case DML_TENSOR_DATA_TYPE_FLOAT64: return 8;
    case DML_TENSOR_DATA_TYPE_UINT64:  return 8;
    case DML_TENSOR_DATA_TYPE_INT64:   return 8;
    case DML_TENSOR_DATA_TYPE_UINT4:   return 0;  // sub-byte; callers use special handling
    default: return 0;
    }
}

DML_TENSOR_DATA_TYPE OnnxDtypeToDml(ONNXTensorElementDataType onnx_dtype) {
    switch (onnx_dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return DML_TENSOR_DATA_TYPE_FLOAT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return DML_TENSOR_DATA_TYPE_FLOAT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return DML_TENSOR_DATA_TYPE_FLOAT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return DML_TENSOR_DATA_TYPE_INT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   return DML_TENSOR_DATA_TYPE_INT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return DML_TENSOR_DATA_TYPE_INT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return DML_TENSOR_DATA_TYPE_INT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return DML_TENSOR_DATA_TYPE_UINT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  return DML_TENSOR_DATA_TYPE_UINT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  return DML_TENSOR_DATA_TYPE_UINT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:  return DML_TENSOR_DATA_TYPE_UINT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:   return DML_TENSOR_DATA_TYPE_UINT32;
    default: return DML_TENSOR_DATA_TYPE_UNKNOWN;
    }
}

std::vector<uint32_t> PadToMinDims(const std::vector<uint32_t>& sizes, size_t min_dims) {
    if (sizes.size() >= min_dims) return sizes;
    std::vector<uint32_t> padded(min_dims - sizes.size(), 1u);
    padded.insert(padded.end(), sizes.begin(), sizes.end());
    return padded;
}

std::vector<uint32_t> ComputePackedStrides(const std::vector<uint32_t>& sizes) {
    std::vector<uint32_t> strides(sizes.size());
    uint32_t stride = 1;
    for (int i = static_cast<int>(sizes.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= sizes[i];
    }
    return strides;
}

uint64_t ComputeAlignedTotalBytes(const std::vector<uint32_t>& sizes, DML_TENSOR_DATA_TYPE dtype) {
    size_t elem_size = DmlDataTypeSize(dtype);
    if (elem_size == 0) return 0;
    uint64_t elem_count = 1;
    for (uint32_t s : sizes) elem_count *= s;
    uint64_t bytes = elem_count * elem_size;
    bytes = std::max<uint64_t>(bytes, 4);
    bytes = (bytes + 3) & ~3ull;
    return bytes;
}

DmlTensorInfo MakeTensorInfo(const std::vector<uint32_t>& sizes, DML_TENSOR_DATA_TYPE dtype) {
    DmlTensorInfo info;
    info.data_type = dtype;
    info.sizes = PadToMinDims(sizes);
    info.total_bytes = ComputeAlignedTotalBytes(info.sizes, dtype);
    return info;
}

DmlTensorInfo MakeTensorInfoAtAxis(
    const std::vector<uint32_t>& sizes,
    DML_TENSOR_DATA_TYPE dtype,
    uint32_t placement,
    uint32_t target_dim_count) {
    std::vector<uint32_t> placed(target_dim_count, 1u);
    for (size_t i = 0; i < sizes.size() && placement + i < target_dim_count; ++i)
        placed[placement + i] = sizes[i];
    return MakeTensorInfo(placed, dtype);
}

DmlTensorInfo MakeTensorInfoWithStrides(
    const std::vector<uint32_t>& sizes,
    const std::vector<uint32_t>& strides,
    DML_TENSOR_DATA_TYPE dtype,
    uint64_t total_bytes) {
    DmlTensorInfo info;
    info.data_type = dtype;
    info.sizes = PadToMinDims(sizes);
    // Pad strides with leading 0s to match padded sizes.
    if (strides.size() < info.sizes.size()) {
        std::vector<uint32_t> padded_strides(info.sizes.size() - strides.size(), 0u);
        padded_strides.insert(padded_strides.end(), strides.begin(), strides.end());
        info.strides = std::move(padded_strides);
    } else {
        info.strides = strides;
    }
    info.total_bytes = std::max<uint64_t>(total_bytes, 4);
    info.total_bytes = (info.total_bytes + 3) & ~3ull;
    return info;
}

DML_BUFFER_TENSOR_DESC DmlTensorInfo::ToBufferDesc() const {
    DML_BUFFER_TENSOR_DESC desc{};
    desc.DataType = data_type;
    desc.Flags = DML_TENSOR_FLAG_NONE;
    desc.DimensionCount = static_cast<UINT>(sizes.size());
    desc.Sizes = sizes.data();
    desc.Strides = strides.empty() ? nullptr : strides.data();
    desc.TotalTensorSizeInBytes = total_bytes;
    desc.GuaranteedBaseOffsetAlignment = 0;
    return desc;
}

std::optional<BroadcastResult> BroadcastShapes(
    const DmlTensorInfo& a_in,
    const DmlTensorInfo& b_in) {
    // Use the un-padded sizes for broadcast logic, then pad result to 4D.
    auto a_sizes = a_in.sizes;
    auto b_sizes = b_in.sizes;
    DML_TENSOR_DATA_TYPE dtype = a_in.data_type;

    // Equalize ranks by prepending 1s.
    size_t max_rank = std::max(a_sizes.size(), b_sizes.size());
    while (a_sizes.size() < max_rank) a_sizes.insert(a_sizes.begin(), 1u);
    while (b_sizes.size() < max_rank) b_sizes.insert(b_sizes.begin(), 1u);

    std::vector<uint32_t> out_sizes(max_rank);
    for (size_t d = 0; d < max_rank; ++d) {
        if (a_sizes[d] == b_sizes[d]) {
            out_sizes[d] = a_sizes[d];
        } else if (a_sizes[d] == 1) {
            out_sizes[d] = b_sizes[d];
        } else if (b_sizes[d] == 1) {
            out_sizes[d] = a_sizes[d];
        } else {
            return std::nullopt;  // incompatible
        }
    }

    // DML requires stride=0 for broadcast dimensions (size 1 expanding to size N).
    auto a_strides = ComputePackedStrides(a_sizes);
    auto b_strides = ComputePackedStrides(b_sizes);

    for (size_t d = 0; d < max_rank; ++d) {
        // Zero stride if expanding broadcast (size 1 → larger output).
        if (a_sizes[d] == 1 && out_sizes[d] > 1) {
            a_sizes[d] = out_sizes[d];
            a_strides[d] = 0;
        }
        if (b_sizes[d] == 1 && out_sizes[d] > 1) {
            b_sizes[d] = out_sizes[d];
            b_strides[d] = 0;
        }
        // Zero strides for any remaining size-1 dimensions in b — DML requires
        // stride=0 for size-1 dims that carry no data variation (e.g. a [1,1,1,3]
        // broadcast tensor's leading dims). Only applied to b to avoid incorrectly
        // zeroing runtime tensors like lr_curr [1,H,W,C] where N=1 is a real batch.
        if (b_sizes[d] == 1) b_strides[d] = 0;
    }

    // Pad all to 4D.
    auto pad = [](std::vector<uint32_t>& sizes, std::vector<uint32_t>& strides) {
        while (sizes.size() < 4) {
            sizes.insert(sizes.begin(), 1u);
            strides.insert(strides.begin(), 0u);
        }
    };
    pad(a_sizes, a_strides);
    pad(b_sizes, b_strides);
    out_sizes = PadToMinDims(out_sizes);

    BroadcastResult result;
    result.a = MakeTensorInfoWithStrides(a_sizes, a_strides, dtype,
        ComputeAlignedTotalBytes(a_in.sizes, dtype));
    result.a.sizes = a_sizes;
    result.a.strides = a_strides;

    result.b = MakeTensorInfoWithStrides(b_sizes, b_strides, dtype,
        ComputeAlignedTotalBytes(b_in.sizes, dtype));
    result.b.sizes = b_sizes;
    result.b.strides = b_strides;

    result.output_sizes = out_sizes;
    return result;
}

// ---------------------------------------------------------------------------
// Translator helpers
// ---------------------------------------------------------------------------

static std::vector<std::string> GetInputNames(const OrtApi& ort_api, const OrtNode* node) {
    return fusion_utils::GetNodeInputNames(ort_api, node);
}

static std::vector<std::string> GetOutputNames(const OrtApi& ort_api, const OrtNode* node) {
    return fusion_utils::GetNodeOutputNames(ort_api, node);
}

static const DmlTensorInfo* LookupShape(
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::string& name) {
    auto it = value_shapes.find(name);
    return (it != value_shapes.end()) ? &it->second : nullptr;
}

// ---------------------------------------------------------------------------
// Rebuild DML_TENSOR_DESC → DML_BUFFER_TENSOR_DESC pointers and re-link the
// operator desc's tensor pointer fields.  Called after the TranslatedOp is in
// its final memory location (no further moves).
// ---------------------------------------------------------------------------

static void RebuildSubNodePointers(SubNode& sn) {
    for (size_t i = 0; i < sn.input_buffer_descs.size() && i < sn.input_tensors.size(); ++i) {
        sn.input_buffer_descs[i].Sizes = sn.input_tensors[i].sizes.data();
        sn.input_buffer_descs[i].Strides = sn.input_tensors[i].strides.empty()
            ? nullptr : sn.input_tensors[i].strides.data();
    }
    for (size_t i = 0; i < sn.output_buffer_descs.size() && i < sn.output_tensors.size(); ++i) {
        sn.output_buffer_descs[i].Sizes = sn.output_tensors[i].sizes.data();
        sn.output_buffer_descs[i].Strides = sn.output_tensors[i].strides.empty()
            ? nullptr : sn.output_tensors[i].strides.data();
    }
    for (size_t i = 0; i < sn.input_tensor_descs.size(); ++i)
        sn.input_tensor_descs[i] = { DML_TENSOR_TYPE_BUFFER, &sn.input_buffer_descs[i] };
    for (size_t i = 0; i < sn.output_tensor_descs.size(); ++i)
        sn.output_tensor_descs[i] = { DML_TENSOR_TYPE_BUFFER, &sn.output_buffer_descs[i] };
    sn.op_desc.Desc = sn.desc_storage.get();
}

static void RebuildTensorDescPointers(TranslatedOp& op) {
    // Rebuild buffer descs from the owned DmlTensorInfo data, ensuring
    // Sizes/Strides point into the stable input_tensors/output_tensors vectors.
    for (size_t i = 0; i < op.input_buffer_descs.size() && i < op.input_tensors.size(); ++i) {
        op.input_buffer_descs[i].Sizes = op.input_tensors[i].sizes.data();
        op.input_buffer_descs[i].Strides = op.input_tensors[i].strides.empty()
            ? nullptr : op.input_tensors[i].strides.data();
    }
    for (size_t i = 0; i < op.output_buffer_descs.size() && i < op.output_tensors.size(); ++i) {
        op.output_buffer_descs[i].Sizes = op.output_tensors[i].sizes.data();
        op.output_buffer_descs[i].Strides = op.output_tensors[i].strides.empty()
            ? nullptr : op.output_tensors[i].strides.data();
    }
    for (size_t i = 0; i < op.input_tensor_descs.size(); ++i)
        op.input_tensor_descs[i] = { DML_TENSOR_TYPE_BUFFER, &op.input_buffer_descs[i] };
    for (size_t i = 0; i < op.output_tensor_descs.size(); ++i)
        op.output_tensor_descs[i] = { DML_TENSOR_TYPE_BUFFER, &op.output_buffer_descs[i] };
    op.op_desc.Desc = op.desc_storage.get();
}

// ---------------------------------------------------------------------------
// Binary elementwise translator template (Add, Mul, Sub, Div)
// ---------------------------------------------------------------------------

template <typename DescType, DML_OPERATOR_TYPE OpType>
static std::optional<TranslatedOp> TranslateBinaryElementwise(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 2 || outputs.empty()) return std::nullopt;

    auto* a_info = LookupShape(value_shapes, inputs[0]);
    auto* b_info = LookupShape(value_shapes, inputs[1]);
    if (!a_info || !b_info) return std::nullopt;

    auto bc = BroadcastShapes(*a_info, *b_info);
    if (!bc) return std::nullopt;

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    auto out_info = out_edge ? MakeTensorInfo(out_edge->sizes, out_edge->data_type)
                             : MakeTensorInfo(bc->output_sizes, a_info->data_type);

    auto storage = std::make_shared<DescType>();

    TranslatedOp result;
    result.input_tensors = { bc->a, bc->b };
    result.output_tensors = { out_info };

    result.input_buffer_descs = { bc->a.ToBufferDesc(), bc->b.ToBufferDesc() };
    result.input_tensor_descs.resize(2);
    result.output_buffer_descs = { out_info.ToBufferDesc() };
    result.output_tensor_descs.resize(1);

    result.desc_storage = storage;
    result.op_desc = { OpType, storage.get() };

    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->ATensor = &self.input_tensor_descs[0];
        storage->BTensor = &self.input_tensor_descs[1];
        storage->OutputTensor = &self.output_tensor_descs[0];
    };
    result.FixupPointers();

    return result;
}

// ---------------------------------------------------------------------------
// Unary activation translator template (Relu, Sigmoid, Tanh)
// ---------------------------------------------------------------------------

template <typename DescType, DML_OPERATOR_TYPE OpType>
static std::optional<TranslatedOp> TranslateUnaryActivation(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    auto t_info = MakeTensorInfo(in_info->sizes, in_info->data_type);

    auto storage = std::make_shared<DescType>();

    TranslatedOp result;
    result.input_tensors = { t_info };
    result.output_tensors = { t_info };

    result.input_buffer_descs = { t_info.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { t_info.ToBufferDesc() };
    result.output_tensor_descs.resize(1);

    result.desc_storage = storage;
    result.op_desc = { OpType, storage.get() };

    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->InputTensor = &self.input_tensor_descs[0];
        storage->OutputTensor = &self.output_tensor_descs[0];
    };
    result.FixupPointers();

    return result;
}

// ---------------------------------------------------------------------------
// MatMul → DML_OPERATOR_GEMM
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateMatMul(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 2 || outputs.empty()) return std::nullopt;

    auto* a_info = LookupShape(value_shapes, inputs[0]);
    auto* b_info = LookupShape(value_shapes, inputs[1]);
    if (!a_info || !b_info) return std::nullopt;

    auto a_sizes = a_info->sizes;
    auto b_sizes = b_info->sizes;
    DML_TENSOR_DATA_TYPE dtype = a_info->data_type;

    // Compute packed strides before any modifications.
    auto a_strides = ComputePackedStrides(a_sizes);
    auto b_strides = ComputePackedStrides(b_sizes);

    // Handle rank-1: A[N] → [1,N], B[N] → [N,1].
    bool a_was_1d = (a_sizes.size() == 1);
    bool b_was_1d = (b_sizes.size() == 1);
    if (a_was_1d) {
        a_sizes.insert(a_sizes.begin(), 1u);
        a_strides.insert(a_strides.begin(), 0u);
    }
    if (b_was_1d) {
        b_sizes.push_back(1u);
        b_strides.push_back(0u);
    }
    if (a_sizes.size() < 2 || b_sizes.size() < 2) return std::nullopt;

    // Equalize ranks.
    size_t max_rank = std::max(a_sizes.size(), b_sizes.size());
    while (a_sizes.size() < max_rank) { a_sizes.insert(a_sizes.begin(), 1u); a_strides.insert(a_strides.begin(), 0u); }
    while (b_sizes.size() < max_rank) { b_sizes.insert(b_sizes.begin(), 1u); b_strides.insert(b_strides.begin(), 0u); }

    // Broadcast batch dimensions (all except last 2).
    for (size_t d = 0; d + 2 < max_rank; ++d) {
        if (a_sizes[d] == b_sizes[d]) continue;
        uint32_t bd = std::max(a_sizes[d], b_sizes[d]);
        if (a_sizes[d] == 1) { a_sizes[d] = bd; a_strides[d] = 0; }
        if (b_sizes[d] == 1) { b_sizes[d] = bd; b_strides[d] = 0; }
    }

    // Pad to 4D.
    while (a_sizes.size() < 4) { a_sizes.insert(a_sizes.begin(), 1u); a_strides.insert(a_strides.begin(), 0u); }
    while (b_sizes.size() < 4) { b_sizes.insert(b_sizes.begin(), 1u); b_strides.insert(b_strides.begin(), 0u); }

    uint32_t M = a_sizes[2];
    uint32_t K_a = a_sizes[3];
    uint32_t K_b = b_sizes[2];
    uint32_t N = b_sizes[3];
    if (K_a != K_b) return std::nullopt;

    std::vector<uint32_t> c_sizes = { a_sizes[0], a_sizes[1], M, N };

    uint64_t a_bytes = ComputeAlignedTotalBytes(a_info->sizes, dtype);
    uint64_t b_bytes = ComputeAlignedTotalBytes(b_info->sizes, dtype);
    uint64_t c_bytes = ComputeAlignedTotalBytes(c_sizes, dtype);

    auto a_tensor = MakeTensorInfoWithStrides(a_sizes, a_strides, dtype, a_bytes);
    a_tensor.sizes = a_sizes;
    a_tensor.strides = a_strides;
    auto b_tensor = MakeTensorInfoWithStrides(b_sizes, b_strides, dtype, b_bytes);
    b_tensor.sizes = b_sizes;
    b_tensor.strides = b_strides;

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    auto c_tensor = out_edge ? MakeTensorInfo(out_edge->sizes, dtype)
                             : MakeTensorInfo(c_sizes, dtype);
    if (!out_edge) c_tensor.total_bytes = c_bytes;

    auto storage = std::make_shared<DML_GEMM_OPERATOR_DESC>();
    storage->CTensor = nullptr;
    storage->TransA = DML_MATRIX_TRANSFORM_NONE;
    storage->TransB = DML_MATRIX_TRANSFORM_NONE;
    storage->Alpha = 1.0f;
    storage->Beta = 0.0f;
    storage->FusedActivation = nullptr;

    TranslatedOp result;
    result.input_tensors = { a_tensor, b_tensor };
    result.output_tensors = { c_tensor };

    result.input_buffer_descs = { a_tensor.ToBufferDesc(), b_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(2);
    result.output_buffer_descs = { c_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);

    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_GEMM, storage.get() };

    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->ATensor = &self.input_tensor_descs[0];
        storage->BTensor = &self.input_tensor_descs[1];
        storage->OutputTensor = &self.output_tensor_descs[0];
    };
    result.FixupPointers();

    return result;
}

// ---------------------------------------------------------------------------
// Softmax → DML_OPERATOR_ACTIVATION_SOFTMAX1
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateSoftmax(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    int64_t axis = adapter.GetAttributeInt("axis", -1);
    size_t padded_rank = in_info->sizes.size();
    size_t orig_rank = in_info->original_rank ? in_info->original_rank : padded_rank;
    size_t pad_offset = padded_rank - orig_rank;
    if (axis < 0) axis += static_cast<int64_t>(orig_rank);
    if (axis < 0 || axis >= static_cast<int64_t>(orig_rank)) return std::nullopt;
    axis += static_cast<int64_t>(pad_offset);
    UINT dml_axis = static_cast<UINT>(axis);

    auto t_info = MakeTensorInfo(in_info->sizes, in_info->data_type);

    struct SoftmaxStorage {
        DML_ACTIVATION_SOFTMAX1_OPERATOR_DESC desc{};
        UINT axis_value;
    };
    auto storage = std::make_shared<SoftmaxStorage>();
    storage->axis_value = dml_axis;
    storage->desc.AxisCount = 1;
    storage->desc.Axes = &storage->axis_value;

    TranslatedOp result;
    result.input_tensors = { t_info };
    result.output_tensors = { t_info };

    result.input_buffer_descs = { t_info.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { t_info.ToBufferDesc() };
    result.output_tensor_descs.resize(1);

    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_ACTIVATION_SOFTMAX1, &storage->desc };

    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor = &self.input_tensor_descs[0];
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
    };
    result.FixupPointers();

    return result;
}

// ---------------------------------------------------------------------------
// Reshape → DML_OPERATOR_ELEMENT_WISE_IDENTITY
//
// Reshape only reinterprets dimensions without moving data. DML identity
// requires input and output to have identical Sizes. We express both
// using the OUTPUT shape — the input is contiguous so its packed strides
// under the output shape are the same. This makes the identity a true
// no-op copy while matching the output shape that downstream consumers
// expect on the intermediate edge.
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateReshape(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>& initializers) {
    auto inputs = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 2 || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    auto* out_info = LookupShape(value_shapes, outputs[0]);
    if (!in_info || !out_info) return std::nullopt;

    auto out_tensor = MakeTensorInfo(out_info->sizes, in_info->data_type);

    TranslatedOp result;
    result.input_tensors  = { out_tensor };
    result.output_tensors = { out_tensor };
    result.passthrough = true;
    return result;
}

// ---------------------------------------------------------------------------
// Shape-only ops (Flatten, Squeeze, Unsqueeze) — same identity pattern as
// Reshape.  Only the data tensor (inputs[0]) becomes a DML input; any ONNX
// shape/axes inputs are consumed at shape-inference time and omitted.
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateShapeOnly(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    auto* out_info = LookupShape(value_shapes, outputs[0]);
    if (!in_info || !out_info) return std::nullopt;

    auto out_tensor = MakeTensorInfo(out_info->sizes, in_info->data_type);

    TranslatedOp result;
    result.input_tensors = { out_tensor };
    result.output_tensors = { out_tensor };
    result.passthrough = true;
    return result;
}

// ---------------------------------------------------------------------------
// Unary activation with a single float attribute (alpha)
// ---------------------------------------------------------------------------

template <typename DescType, DML_OPERATOR_TYPE OpType, typename SetAttrsFn>
static std::optional<TranslatedOp> TranslateUnaryWithAttrs(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&,
    SetAttrsFn set_attrs) {
    auto inputs = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    auto t_info = MakeTensorInfo(in_info->sizes, in_info->data_type);
    auto storage = std::make_shared<DescType>();
    set_attrs(storage.get(), node, ort_api);

    TranslatedOp result;
    result.input_tensors = { t_info };
    result.output_tensors = { t_info };
    result.input_buffer_descs = { t_info.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { t_info.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { OpType, storage.get() };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->InputTensor = &self.input_tensor_descs[0];
        storage->OutputTensor = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// Cast → DML_OPERATOR_CAST
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateCast(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    int64_t to_type = adapter.GetAttributeInt("to", 0);
    auto out_dtype = OnnxDtypeToDml(static_cast<ONNXTensorElementDataType>(to_type));
    if (out_dtype == DML_TENSOR_DATA_TYPE_UNKNOWN) return std::nullopt;

    auto in_tensor = MakeTensorInfo(in_info->sizes, in_info->data_type);
    auto out_tensor = MakeTensorInfo(in_info->sizes, out_dtype);

    struct CastStorage {
        DML_CAST_OPERATOR_DESC desc{};
    };
    auto storage = std::make_shared<CastStorage>();

    TranslatedOp result;
    result.input_tensors = { in_tensor };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs = { in_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_CAST, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor = &self.input_tensor_descs[0];
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// IsInf → DML_OPERATOR_ELEMENT_WISE_IS_INFINITY (detect_positive/negative)
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateIsInf(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    int64_t detect_pos = adapter.GetAttributeInt("detect_positive", 1);
    int64_t detect_neg = adapter.GetAttributeInt("detect_negative", 1);

    auto in_tensor  = MakeTensorInfo(in_info->sizes, in_info->data_type);
    auto out_tensor = MakeTensorInfo(in_info->sizes, DML_TENSOR_DATA_TYPE_UINT8);

    struct IsInfStorage {
        DML_ELEMENT_WISE_IS_INFINITY_OPERATOR_DESC desc{};
    };
    auto storage = std::make_shared<IsInfStorage>();
    if (detect_pos && !detect_neg)
        storage->desc.InfinityMode = DML_IS_INFINITY_MODE_POSITIVE;
    else if (!detect_pos && detect_neg)
        storage->desc.InfinityMode = DML_IS_INFINITY_MODE_NEGATIVE;
    else
        storage->desc.InfinityMode = DML_IS_INFINITY_MODE_EITHER;

    TranslatedOp result;
    result.input_tensors = { in_tensor };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs = { in_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_ELEMENT_WISE_IS_INFINITY, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor  = &self.input_tensor_descs[0];
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// Affine / ImageScaler → DML_OPERATOR_ELEMENT_WISE_IDENTITY with ScaleBias
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateAffine(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs = GetInputNames(ort_api, node);
    if (inputs.empty()) return std::nullopt;
    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    float scale = adapter.GetAttributeFloat("scale", 1.0f);
    float bias  = adapter.GetAttributeFloat("bias",  0.0f);

    auto t_info = MakeTensorInfo(in_info->sizes, in_info->data_type);

    struct AffineStorage {
        DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC desc{};
        DML_SCALE_BIAS sb{};
    };
    auto storage = std::make_shared<AffineStorage>();
    storage->sb.Scale = scale;
    storage->sb.Bias  = bias;
    storage->desc.ScaleBias = &storage->sb;

    TranslatedOp result;
    result.input_tensors = { t_info };
    result.output_tensors = { t_info };
    result.input_buffer_descs = { t_info.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { t_info.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_ELEMENT_WISE_IDENTITY, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor  = &self.input_tensor_descs[0];
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
        storage->desc.ScaleBias    = &storage->sb;
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// Bitwise binary ops — same pattern as binary elementwise but output dtype
// may be bool for logical ops; use input dtype for bitwise ops.
// ---------------------------------------------------------------------------

// Where (condition, X, Y) → DML_OPERATOR_ELEMENT_WISE_IF
static std::optional<TranslatedOp> TranslateWhere(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 3 || outputs.empty()) return std::nullopt;

    auto* cond_info = LookupShape(value_shapes, inputs[0]);
    auto* x_info    = LookupShape(value_shapes, inputs[1]);
    auto* y_info    = LookupShape(value_shapes, inputs[2]);
    if (!cond_info || !x_info || !y_info) return std::nullopt;

    // Broadcast X and Y together first to get output shape.
    auto bc = BroadcastShapes(*x_info, *y_info);
    if (!bc) return std::nullopt;

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    const auto& out_sizes = out_edge ? out_edge->sizes : bc->output_sizes;
    auto cond_tensor = MakeTensorInfo(out_sizes, cond_info->data_type);
    auto out_tensor  = MakeTensorInfo(out_sizes, x_info->data_type);

    struct WhereStorage {
        DML_ELEMENT_WISE_IF_OPERATOR_DESC desc{};
    };
    auto storage = std::make_shared<WhereStorage>();

    TranslatedOp result;
    result.input_tensors = { cond_tensor, bc->a, bc->b };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs = { cond_tensor.ToBufferDesc(), bc->a.ToBufferDesc(), bc->b.ToBufferDesc() };
    result.input_tensor_descs.resize(3);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_ELEMENT_WISE_IF, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.ConditionTensor = &self.input_tensor_descs[0];
        storage->desc.ATensor         = &self.input_tensor_descs[1];
        storage->desc.BTensor         = &self.input_tensor_descs[2];
        storage->desc.OutputTensor    = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

// Sum (N-ary) — chain pairwise adds.  For the graph fusion context we only
// support 2-input Sum here; the graph partitioner already limits to that.
static std::optional<TranslatedOp> TranslateSum(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>& init) {
    return TranslateBinaryElementwise<DML_ELEMENT_WISE_ADD_OPERATOR_DESC, DML_OPERATOR_ELEMENT_WISE_ADD>(
        ort_api, node, value_shapes, init);
}

// Mod — fmod attr selects TRUNCATE vs FLOOR
static std::optional<TranslatedOp> TranslateMod(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 2 || outputs.empty()) return std::nullopt;

    auto* a_info = LookupShape(value_shapes, inputs[0]);
    auto* b_info = LookupShape(value_shapes, inputs[1]);
    if (!a_info || !b_info) return std::nullopt;

    auto bc = BroadcastShapes(*a_info, *b_info);
    if (!bc) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    int64_t fmod = adapter.GetAttributeInt("fmod", 0);

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    auto out_tensor = out_edge ? MakeTensorInfo(out_edge->sizes, a_info->data_type)
                               : MakeTensorInfo(bc->output_sizes, a_info->data_type);

    struct ModStorage {
        DML_ELEMENT_WISE_MODULUS_FLOOR_OPERATOR_DESC  floor_desc{};
        DML_ELEMENT_WISE_MODULUS_TRUNCATE_OPERATOR_DESC trunc_desc{};
        bool use_trunc;
    };
    auto storage = std::make_shared<ModStorage>();
    storage->use_trunc = (fmod != 0);

    TranslatedOp result;
    result.input_tensors = { bc->a, bc->b };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs = { bc->a.ToBufferDesc(), bc->b.ToBufferDesc() };
    result.input_tensor_descs.resize(2);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;

    if (fmod) {
        result.op_desc = { DML_OPERATOR_ELEMENT_WISE_MODULUS_TRUNCATE, &storage->trunc_desc };
    } else {
        result.op_desc = { DML_OPERATOR_ELEMENT_WISE_MODULUS_FLOOR, &storage->floor_desc };
    }

    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        if (storage->use_trunc) {
            storage->trunc_desc.ATensor     = &self.input_tensor_descs[0];
            storage->trunc_desc.BTensor     = &self.input_tensor_descs[1];
            storage->trunc_desc.OutputTensor= &self.output_tensor_descs[0];
            self.op_desc.Desc = &storage->trunc_desc;
        } else {
            storage->floor_desc.ATensor     = &self.input_tensor_descs[0];
            storage->floor_desc.BTensor     = &self.input_tensor_descs[1];
            storage->floor_desc.OutputTensor= &self.output_tensor_descs[0];
            self.op_desc.Desc = &storage->floor_desc;
        }
    };
    result.FixupPointers();
    return result;
}

// BitShift — direction attr: "LEFT" or "RIGHT"
static std::optional<TranslatedOp> TranslateBitShift(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 2 || outputs.empty()) return std::nullopt;

    auto* a_info = LookupShape(value_shapes, inputs[0]);
    auto* b_info = LookupShape(value_shapes, inputs[1]);
    if (!a_info || !b_info) return std::nullopt;

    auto bc = BroadcastShapes(*a_info, *b_info);
    if (!bc) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    std::string dir = adapter.GetAttributeString("direction", "LEFT");
    bool left_shift = (dir == "LEFT");

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    auto out_tensor = out_edge ? MakeTensorInfo(out_edge->sizes, a_info->data_type)
                               : MakeTensorInfo(bc->output_sizes, a_info->data_type);

    struct ShiftStorage {
        DML_ELEMENT_WISE_BIT_SHIFT_LEFT_OPERATOR_DESC  left_desc{};
        DML_ELEMENT_WISE_BIT_SHIFT_RIGHT_OPERATOR_DESC right_desc{};
        bool left;
    };
    auto storage = std::make_shared<ShiftStorage>();
    storage->left = left_shift;

    TranslatedOp result;
    result.input_tensors = { bc->a, bc->b };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs = { bc->a.ToBufferDesc(), bc->b.ToBufferDesc() };
    result.input_tensor_descs.resize(2);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = left_shift
        ? DML_OPERATOR_DESC{ DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_LEFT,  &storage->left_desc }
        : DML_OPERATOR_DESC{ DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_RIGHT, &storage->right_desc };

    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        if (storage->left) {
            storage->left_desc.ATensor      = &self.input_tensor_descs[0];
            storage->left_desc.BTensor      = &self.input_tensor_descs[1];
            storage->left_desc.OutputTensor = &self.output_tensor_descs[0];
            self.op_desc = { DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_LEFT, &storage->left_desc };
        } else {
            storage->right_desc.ATensor      = &self.input_tensor_descs[0];
            storage->right_desc.BTensor      = &self.input_tensor_descs[1];
            storage->right_desc.OutputTensor = &self.output_tensor_descs[0];
            self.op_desc = { DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_RIGHT, &storage->right_desc };
        }
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// Transpose → DML_OPERATOR_ELEMENT_WISE_IDENTITY with permuted strides
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateTranspose(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    std::vector<int64_t> perm = adapter.GetAttributeInts("perm");
    size_t rank = in_info->sizes.size();
    if (perm.empty()) {
        // Default perm: reverse all dims.
        perm.resize(rank);
        std::iota(perm.rbegin(), perm.rend(), 0);
    }
    // Adjust perm for 4D padding: perm refers to original rank, but sizes are
    // padded with leading 1s. Prepend identity dims to perm.
    size_t orig_rank = perm.size();
    if (orig_rank < rank) {
        size_t pad = rank - orig_rank;
        std::vector<int64_t> padded_perm(rank);
        for (size_t i = 0; i < pad; ++i)
            padded_perm[i] = static_cast<int64_t>(i);
        for (size_t i = 0; i < orig_rank; ++i)
            padded_perm[pad + i] = perm[i] + static_cast<int64_t>(pad);
        perm = std::move(padded_perm);
    }
    if (perm.size() != rank) return std::nullopt;

    // Compute output sizes and input strides for the transpose.
    auto in_strides = ComputePackedStrides(in_info->sizes);
    std::vector<uint32_t> out_sizes(rank);
    std::vector<uint32_t> transposed_strides(rank);
    for (size_t i = 0; i < rank; ++i) {
        out_sizes[i] = in_info->sizes[perm[i]];
        transposed_strides[i] = in_strides[perm[i]];
    }

    // The input tensor uses permuted strides so that reading in the new
    // dimension order traverses the original data correctly.
    auto in_tensor = MakeTensorInfoWithStrides(
        out_sizes, transposed_strides, in_info->data_type, in_info->total_bytes);
    in_tensor.sizes = PadToMinDims(out_sizes);
    // Strides must match padded sizes length.
    if (transposed_strides.size() < in_tensor.sizes.size()) {
        std::vector<uint32_t> padded(in_tensor.sizes.size() - transposed_strides.size(), 0u);
        padded.insert(padded.end(), transposed_strides.begin(), transposed_strides.end());
        in_tensor.strides = std::move(padded);
    } else {
        in_tensor.strides = transposed_strides;
    }

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    auto out_tensor = out_edge ? MakeTensorInfo(out_edge->sizes, in_info->data_type)
                               : MakeTensorInfo(out_sizes, in_info->data_type);

    auto storage = std::make_shared<DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC>();
    storage->ScaleBias = nullptr;

    TranslatedOp result;
    result.input_tensors = { in_tensor };
    result.output_tensors = { out_tensor };

    result.input_buffer_descs = { in_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);

    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_ELEMENT_WISE_IDENTITY, storage.get() };

    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->InputTensor = &self.input_tensor_descs[0];
        storage->OutputTensor = &self.output_tensor_descs[0];
    };
    result.FixupPointers();

    return result;
}

// ---------------------------------------------------------------------------
// P2 — Reduce ops
// ---------------------------------------------------------------------------

template <DML_REDUCE_FUNCTION ReduceFunc>
static std::optional<TranslatedOp> TranslateReduce(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>& initializers) {
    auto inputs = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    int64_t keepdims = adapter.GetAttributeInt("keepdims", 1);
    int64_t noop_with_empty_axes = adapter.GetAttributeInt("noop_with_empty_axes", 0);

    std::vector<int64_t> axes_i64;
    if (inputs.size() > 1 && !inputs[1].empty()) {
        // Opset 13 (ReduceSum) / Opset 18 (other Reduce ops): axes from input[1]
        auto it = initializers.find(inputs[1]);
        if (it != initializers.end() && it->second) {
            OrtTensorTypeAndShapeInfo* tsi = nullptr;
            ort_api.GetTensorTypeAndShape(const_cast<OrtValue*>(it->second), &tsi);
            if (tsi) {
                size_t count = 0;
                ort_api.GetTensorShapeElementCount(tsi, &count);
                ort_api.ReleaseTensorTypeAndShapeInfo(tsi);
                void* data = nullptr;
                OrtStatus* st = ort_api.GetTensorMutableData(const_cast<OrtValue*>(it->second), &data);
                if (st) { ort_api.ReleaseStatus(st); }
                else if (data && count > 0) {
                    auto* vals = static_cast<const int64_t*>(data);
                    axes_i64.assign(vals, vals + count);
                }
            }
        }
    } else {
        axes_i64 = adapter.GetAttributeInts("axes");
    }

    size_t padded_rank = in_info->sizes.size();
    size_t orig_rank = in_info->original_rank ? in_info->original_rank : padded_rank;
    size_t pad_offset = padded_rank - orig_rank;
    std::vector<uint32_t> axes;
    if (axes_i64.empty()) {
        if (noop_with_empty_axes) {
            return std::nullopt;
        }
        axes.resize(padded_rank);
        std::iota(axes.begin(), axes.end(), 0u);
    } else {
        for (int64_t a : axes_i64) {
            if (a < 0) a += static_cast<int64_t>(orig_rank);
            a += static_cast<int64_t>(pad_offset);
            axes.push_back(static_cast<uint32_t>(a));
        }
    }

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    std::vector<uint32_t> out_sizes;
    if (out_edge) {
        out_sizes = out_edge->sizes;
    } else {
        out_sizes = in_info->sizes;
        for (uint32_t ax : axes) {
            if (ax < out_sizes.size()) out_sizes[ax] = (keepdims ? 1u : 0u);
        }
        if (!keepdims) {
            out_sizes.erase(std::remove(out_sizes.begin(), out_sizes.end(), 0u), out_sizes.end());
            if (out_sizes.empty()) out_sizes.push_back(1u);
        }
    }

    auto in_tensor  = MakeTensorInfo(in_info->sizes, in_info->data_type);
    auto out_tensor = MakeTensorInfo(out_sizes, in_info->data_type);

    struct ReduceStorage {
        DML_REDUCE_OPERATOR_DESC desc{};
        std::vector<uint32_t> axes_vec;
    };
    auto storage = std::make_shared<ReduceStorage>();
    storage->desc.Function = ReduceFunc;
    storage->axes_vec = axes;
    storage->desc.AxisCount = static_cast<UINT>(storage->axes_vec.size());
    storage->desc.Axes = storage->axes_vec.data();

    TranslatedOp result;
    result.input_tensors  = { in_tensor };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs  = { in_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_REDUCE, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor  = &self.input_tensor_descs[0];
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
        storage->desc.Axes = storage->axes_vec.data();
    };
    result.FixupPointers();
    return result;
}

// ArgMax / ArgMin share same descriptor layout: AxisCount/Axes array, not a single Axis.
template <DML_OPERATOR_TYPE OpType, typename DescType>
static std::optional<TranslatedOp> TranslateArgMaxMin(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    int64_t axis      = adapter.GetAttributeInt("axis", 0);
    int64_t keepdims  = adapter.GetAttributeInt("keepdims", 1);
    int64_t select_last = adapter.GetAttributeInt("select_last_index", 0);

    size_t padded_rank = in_info->sizes.size();
    size_t orig_rank = in_info->original_rank ? in_info->original_rank : padded_rank;
    size_t pad_offset = padded_rank - orig_rank;
    if (axis < 0) axis += static_cast<int64_t>(orig_rank);
    axis += static_cast<int64_t>(pad_offset);
    UINT dml_axis = static_cast<UINT>(axis);

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    std::vector<uint32_t> out_sizes;
    if (out_edge) {
        out_sizes = out_edge->sizes;
    } else {
        out_sizes = in_info->sizes;
        if (keepdims) out_sizes[axis] = 1u;
        else {
            out_sizes.erase(out_sizes.begin() + axis);
            if (out_sizes.empty()) out_sizes.push_back(1u);
        }
    }

    auto in_tensor  = MakeTensorInfo(in_info->sizes, in_info->data_type);
    auto out_tensor = MakeTensorInfo(out_sizes, DML_TENSOR_DATA_TYPE_INT64);

    struct ArgStorage {
        DescType desc{};
        UINT axis_val;
    };
    auto storage = std::make_shared<ArgStorage>();
    storage->axis_val = dml_axis;
    storage->desc.AxisCount = 1;
    storage->desc.Axes = &storage->axis_val;
    storage->desc.AxisDirection = (select_last ? DML_AXIS_DIRECTION_DECREASING : DML_AXIS_DIRECTION_INCREASING);

    TranslatedOp result;
    result.input_tensors  = { in_tensor };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs  = { in_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { OpType, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor  = &self.input_tensor_descs[0];
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
        storage->desc.Axes = &storage->axis_val;
    };
    result.FixupPointers();
    return result;
}

// Hardmax → DML_OPERATOR_ACTIVATION_HARDMAX (axis attr)
static std::optional<TranslatedOp> TranslateHardmax(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    int64_t axis = adapter.GetAttributeInt("axis", -1);
    size_t padded_rank = in_info->sizes.size();
    size_t orig_rank = in_info->original_rank ? in_info->original_rank : padded_rank;
    size_t pad_offset = padded_rank - orig_rank;
    if (axis < 0) axis += static_cast<int64_t>(orig_rank);
    axis += static_cast<int64_t>(pad_offset);
    UINT dml_axis = static_cast<UINT>(axis);

    auto t_info = MakeTensorInfo(in_info->sizes, in_info->data_type);

    struct HardmaxStorage {
        DML_ACTIVATION_HARDMAX1_OPERATOR_DESC desc{};
        UINT axis_val;
    };
    auto storage = std::make_shared<HardmaxStorage>();
    storage->axis_val = dml_axis;
    storage->desc.AxisCount = 1;
    storage->desc.Axes = &storage->axis_val;

    TranslatedOp result;
    result.input_tensors  = { t_info };
    result.output_tensors = { t_info };
    result.input_buffer_descs  = { t_info.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { t_info.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_ACTIVATION_HARDMAX1, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor  = &self.input_tensor_descs[0];
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
        storage->desc.Axes = &storage->axis_val;
    };
    result.FixupPointers();
    return result;
}

// LogSoftmax → DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1 (same axis pattern as Softmax)
static std::optional<TranslatedOp> TranslateLogSoftmax(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    int64_t axis = adapter.GetAttributeInt("axis", -1);
    size_t padded_rank = in_info->sizes.size();
    size_t orig_rank = in_info->original_rank ? in_info->original_rank : padded_rank;
    size_t pad_offset = padded_rank - orig_rank;
    if (axis < 0) axis += static_cast<int64_t>(orig_rank);
    axis += static_cast<int64_t>(pad_offset);
    UINT dml_axis = static_cast<UINT>(axis);

    auto t_info = MakeTensorInfo(in_info->sizes, in_info->data_type);

    struct LogSoftmaxStorage {
        DML_ACTIVATION_LOG_SOFTMAX1_OPERATOR_DESC desc{};
        UINT axis_val;
    };
    auto storage = std::make_shared<LogSoftmaxStorage>();
    storage->axis_val = dml_axis;
    storage->desc.AxisCount = 1;
    storage->desc.Axes = &storage->axis_val;

    TranslatedOp result;
    result.input_tensors  = { t_info };
    result.output_tensors = { t_info };
    result.input_buffer_descs  = { t_info.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { t_info.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_ACTIVATION_LOG_SOFTMAX1, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor  = &self.input_tensor_descs[0];
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
        storage->desc.Axes = &storage->axis_val;
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// P3 — Conv helpers
// ---------------------------------------------------------------------------

// Read kernel_shape, strides, pads, dilations from node attributes.
// Returns spatial dimension count (1 or 2 for DML; we force to 2).
struct ConvKernelArgs {
    uint32_t spatial_dim_count;
    std::vector<uint32_t> kernel_shape;
    std::vector<uint32_t> strides;
    std::vector<uint32_t> dilations;
    std::vector<uint32_t> start_padding;
    std::vector<uint32_t> end_padding;
    std::vector<uint32_t> output_padding;
    bool auto_pad = false;
    bool auto_pad_same_upper = false;
};

static ConvKernelArgs ReadConvKernelArgs(
    const OrtNodeAdapter& adapter,
    const std::vector<uint32_t>& input_sizes) {  // NCHW padded to 4D
    ConvKernelArgs k;
    // spatial dims = rank - 2 (N,C + spatial)
    uint32_t spatial = static_cast<uint32_t>(input_sizes.size()) - 2;
    if (spatial < 1) spatial = 1;
    // DML requires at least 2 spatial dims
    if (spatial < 2) spatial = 2;
    k.spatial_dim_count = spatial;

    auto ks = adapter.GetAttributeInts("kernel_shape");
    for (int64_t v : ks) k.kernel_shape.push_back(static_cast<uint32_t>(v));
    while (k.kernel_shape.size() < spatial) k.kernel_shape.insert(k.kernel_shape.begin(), 1u);

    auto st = adapter.GetAttributeInts("strides");
    for (int64_t v : st) k.strides.push_back(static_cast<uint32_t>(v));
    while (k.strides.size() < spatial) k.strides.insert(k.strides.begin(), 1u);

    auto di = adapter.GetAttributeInts("dilations");
    for (int64_t v : di) k.dilations.push_back(static_cast<uint32_t>(v));
    while (k.dilations.size() < spatial) k.dilations.insert(k.dilations.begin(), 1u);

    std::string auto_pad_str = adapter.GetAttributeString("auto_pad", "NOTSET");
    if (auto_pad_str == "SAME_UPPER" || auto_pad_str == "SAME_LOWER") {
        k.auto_pad = true;
        k.auto_pad_same_upper = (auto_pad_str == "SAME_UPPER");
        k.start_padding.assign(spatial, 0u);
        k.end_padding.assign(spatial, 0u);
    } else if (auto_pad_str == "VALID") {
        k.start_padding.assign(spatial, 0u);
        k.end_padding.assign(spatial, 0u);
    } else {
        auto pads = adapter.GetAttributeInts("pads");
        if (pads.size() >= 2 * spatial) {
            for (size_t i = 0; i < spatial; ++i) k.start_padding.push_back(static_cast<uint32_t>(pads[i]));
            for (size_t i = 0; i < spatial; ++i) k.end_padding.push_back(static_cast<uint32_t>(pads[spatial + i]));
        } else {
            k.start_padding.assign(spatial, 0u);
            k.end_padding.assign(spatial, 0u);
        }
    }

    auto op = adapter.GetAttributeInts("output_padding");
    for (int64_t v : op) k.output_padding.push_back(static_cast<uint32_t>(v));
    while (k.output_padding.size() < spatial) k.output_padding.insert(k.output_padding.begin(), 0u);

    // Note: for transposed convolution, DML padding is computed from the
    // known output shape in TranslateConvImpl, not from ONNX pads here.

    return k;
}

static std::optional<TranslatedOp> TranslateConvImpl(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&,
    DML_CONVOLUTION_DIRECTION direction,
    bool nhwc = false) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 2 || outputs.empty()) return std::nullopt;

    auto* x_info = LookupShape(value_shapes, inputs[0]);
    auto* w_info = LookupShape(value_shapes, inputs[1]);
    if (!x_info || !w_info) return std::nullopt;

    bool has_bias = (inputs.size() > 2 && !inputs[2].empty() &&
                     value_shapes.count(inputs[2]));

    OrtNodeAdapter adapter(node, ort_api);
    int64_t group = adapter.GetAttributeInt("group", 1);

    auto x_sizes = x_info->sizes;  // already 4D from MakeTensorInfo
    auto w_sizes = w_info->sizes;

    // NHWC: permute to NCHW for all internal computation. Stride reinterpretation
    // happens later when building DML tensor descs.
    if (nhwc && x_sizes.size() == 4) {
        // [N,H,W,C] → [N,C,H,W]
        x_sizes = { x_sizes[0], x_sizes[3], x_sizes[1], x_sizes[2] };
        // Weights [K,kH,kW,CpG] → [K,CpG,kH,kW]
        w_sizes = { w_sizes[0], w_sizes[3], w_sizes[1], w_sizes[2] };
    }

    // 1D convolution: PadToMinDims produces [1,N,C,W] but DML needs [N,C,1,W].
    // Insert spatial dim between C and W to match ORT's NonspatialDimensionCount handling.
    uint32_t x_orig_rank = x_info->original_rank ? x_info->original_rank : static_cast<uint32_t>(x_sizes.size());
    bool is_1d = (!nhwc && x_orig_rank == 3 && x_sizes.size() == 4);
    if (is_1d) {
        // x_sizes is [1,N,C,W] → [N,C,1,W]
        x_sizes = { x_sizes[1], x_sizes[2], 1u, x_sizes[3] };
        // w_sizes is [1,K,C,W] → [K,C,1,W]
        w_sizes = { w_sizes[1], w_sizes[2], 1u, w_sizes[3] };
    }

    auto k = ReadConvKernelArgs(adapter, x_sizes);
    DML_TENSOR_DATA_TYPE dtype = x_info->data_type;

    // Resolve auto_pad for forward convolution (computes padding from input dims).
    // For backward conv, padding is derived from the known output shape below.
    if (k.auto_pad && direction == DML_CONVOLUTION_DIRECTION_FORWARD) {
        for (uint32_t d = 0; d < k.spatial_dim_count; ++d) {
            uint32_t input_length = x_sizes[2 + d];
            uint32_t stride = k.strides[d];
            uint32_t strided_output_length = (input_length + stride - 1) / stride;
            uint32_t kernel_length = 1 + (k.kernel_shape[d] - 1) * k.dilations[d];
            uint32_t length_needed = stride * (strided_output_length - 1) + kernel_length;
            uint32_t padding = (length_needed <= input_length) ? 0 : (length_needed - input_length);
            if (k.auto_pad_same_upper) {
                k.start_padding[d] = padding / 2;
            } else {
                k.start_padding[d] = (padding + 1) / 2;
            }
            k.end_padding[d] = padding - k.start_padding[d];
        }
    }

    // Compute output shape from convolution formula.
    // First try value_shapes (if graph already has the shape), else compute.
    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    std::vector<uint32_t> out_sizes;
    if (out_edge) {
        out_sizes = out_edge->sizes;
        if (nhwc && out_sizes.size() == 4) {
            out_sizes = { out_sizes[0], out_sizes[3], out_sizes[1], out_sizes[2] };
        }
        if (is_1d && out_sizes.size() == 4) {
            out_sizes = { out_sizes[1], out_sizes[2], 1u, out_sizes[3] };
        }
    } else {
        out_sizes = x_sizes;
        out_sizes[1] = (direction == DML_CONVOLUTION_DIRECTION_FORWARD)
            ? w_sizes[0]
            : w_sizes[1] * static_cast<uint32_t>(group);

        // ConvTranspose: use output_shape attribute when available (matches ORT's
        // InitializeKernelAndShapesTransposed which reads AttrName::OutputShape).
        auto output_shape_attr = adapter.GetAttributeInts("output_shape");
        if (!output_shape_attr.empty() && direction == DML_CONVOLUTION_DIRECTION_BACKWARD) {
            size_t attr_spatial = output_shape_attr.size();
            for (uint32_t d = 0; d < k.spatial_dim_count && d < attr_spatial; ++d) {
                size_t attr_idx = attr_spatial - k.spatial_dim_count + d;
                out_sizes[2 + d] = static_cast<uint32_t>(output_shape_attr[attr_idx]);
            }
        } else {
            for (uint32_t d = 0; d < k.spatial_dim_count; ++d) {
                uint32_t in_d  = x_sizes[2 + d];
                uint32_t ks    = k.kernel_shape[d];
                uint32_t st    = k.strides[d];
                uint32_t dil   = k.dilations[d];
                uint32_t ps    = k.start_padding[d];
                uint32_t pe    = k.end_padding[d];
                uint32_t op    = k.output_padding[d];
                uint32_t dk    = (ks - 1) * dil + 1;
                if (direction == DML_CONVOLUTION_DIRECTION_FORWARD) {
                    out_sizes[2 + d] = (in_d + ps + pe - dk) / st + 1;
                } else {
                    out_sizes[2 + d] = (in_d - 1) * st - ps - pe + dk + op;
                }
            }
        }
    }

    // For backward conv, recompute DML padding from the known output shape.
    // Matches ORT's InitializeKernelAndShapesTransposed logic: uses raw windowSize,
    // not dilated kernel size, for the padding derivation.
    if (direction == DML_CONVOLUTION_DIRECTION_BACKWARD) {
        for (uint32_t d = 0; d < k.spatial_dim_count; ++d) {
            uint32_t in_d = x_sizes[2 + d];
            uint32_t out_d = out_sizes[2 + d];
            uint32_t window_size = k.kernel_shape[d];
            uint32_t st = k.strides[d];
            int32_t paddings = static_cast<int32_t>((in_d - 1) * st + window_size) - static_cast<int32_t>(out_d);
            if (paddings < 0) paddings = 0;
            if (k.auto_pad_same_upper || !k.auto_pad) {
                k.start_padding[d] = static_cast<uint32_t>(paddings) / 2;
            } else {
                k.start_padding[d] = static_cast<uint32_t>(paddings + 1) / 2;
            }
            k.end_padding[d] = static_cast<uint32_t>(paddings) - k.start_padding[d];
        }
    }

    DmlTensorInfo x_tensor, w_tensor, out_tensor;
    if (nhwc && x_sizes.size() == 4) {
        // NCHW sizes with NHWC strides: data in memory is [N,H,W,C].
        auto nhwc_input_strides = [](const std::vector<uint32_t>& nchw) -> std::vector<uint32_t> {
            uint32_t C = nchw[1], H = nchw[2], W = nchw[3];
            return { H * W * C, 1, W * C, C };
        };
        size_t elem_sz = DmlDataTypeSize(dtype);
        uint64_t x_bytes = static_cast<uint64_t>(x_sizes[0]) * x_sizes[1] * x_sizes[2] * x_sizes[3] * elem_sz;
        uint64_t w_bytes = static_cast<uint64_t>(w_sizes[0]) * w_sizes[1] * w_sizes[2] * w_sizes[3] * elem_sz;
        uint64_t o_bytes = static_cast<uint64_t>(out_sizes[0]) * out_sizes[1] * out_sizes[2] * out_sizes[3] * elem_sz;
        x_tensor   = MakeTensorInfoWithStrides(x_sizes, nhwc_input_strides(x_sizes), dtype, x_bytes);
        w_tensor   = MakeTensorInfoWithStrides(w_sizes, nhwc_input_strides(w_sizes), dtype, w_bytes);
        out_tensor = MakeTensorInfoWithStrides(out_sizes, nhwc_input_strides(out_sizes), dtype, o_bytes);
    } else {
        x_tensor   = MakeTensorInfo(x_sizes, dtype);
        w_tensor   = MakeTensorInfo(w_sizes, dtype);
        out_tensor = MakeTensorInfo(out_sizes, dtype);
    }

    struct ConvStorage {
        DML_CONVOLUTION_OPERATOR_DESC desc{};
        std::vector<uint32_t> kernel_shape, strides, dilations, start_pad, end_pad, out_pad;
        // Fused activation support
        bool has_fused_activation = false;
        DML_OPERATOR_DESC fused_activation_op_desc{};
        union {
            DML_ACTIVATION_RELU_OPERATOR_DESC relu;
            DML_ACTIVATION_LEAKY_RELU_OPERATOR_DESC leaky_relu;
            DML_ACTIVATION_SIGMOID_OPERATOR_DESC sigmoid;
            DML_ACTIVATION_TANH_OPERATOR_DESC tanh_desc;
            DML_ACTIVATION_ELU_OPERATOR_DESC elu;
            DML_ACTIVATION_HARD_SIGMOID_OPERATOR_DESC hard_sigmoid;
            DML_ACTIVATION_LINEAR_OPERATOR_DESC linear;
        } activation_desc{};
    };
    auto storage = std::make_shared<ConvStorage>();
    storage->kernel_shape = k.kernel_shape;
    storage->strides      = k.strides;
    storage->dilations    = k.dilations;
    storage->start_pad    = k.start_padding;
    storage->end_pad      = k.end_padding;
    // ORT zeros output_padding before sending to DML — the ONNX output_padding
    // is only used to compute the output shape, not passed to the DML operator.
    storage->out_pad.assign(k.spatial_dim_count, 0u);

    // Parse fused activation (DmlFusedConv/DmlFusedConvTranspose set these attributes).
    std::string fused_act = adapter.GetAttributeString("activation", "");
    if (!fused_act.empty()) {
        storage->has_fused_activation = true;
        if (fused_act == "Relu") {
            storage->activation_desc.relu = {};
            storage->fused_activation_op_desc = { DML_OPERATOR_ACTIVATION_RELU, &storage->activation_desc.relu };
        } else if (fused_act == "LeakyRelu") {
            storage->activation_desc.leaky_relu = {};
            storage->activation_desc.leaky_relu.Alpha = adapter.GetAttributeFloat("activation_alpha", 0.01f);
            storage->fused_activation_op_desc = { DML_OPERATOR_ACTIVATION_LEAKY_RELU, &storage->activation_desc.leaky_relu };
        } else if (fused_act == "Sigmoid") {
            storage->activation_desc.sigmoid = {};
            storage->fused_activation_op_desc = { DML_OPERATOR_ACTIVATION_SIGMOID, &storage->activation_desc.sigmoid };
        } else if (fused_act == "Tanh") {
            storage->activation_desc.tanh_desc = {};
            storage->fused_activation_op_desc = { DML_OPERATOR_ACTIVATION_TANH, &storage->activation_desc.tanh_desc };
        } else if (fused_act == "Elu") {
            storage->activation_desc.elu = {};
            storage->activation_desc.elu.Alpha = adapter.GetAttributeFloat("activation_alpha", 1.0f);
            storage->fused_activation_op_desc = { DML_OPERATOR_ACTIVATION_ELU, &storage->activation_desc.elu };
        } else if (fused_act == "HardSigmoid") {
            storage->activation_desc.hard_sigmoid = {};
            storage->activation_desc.hard_sigmoid.Alpha = adapter.GetAttributeFloat("activation_alpha", 0.2f);
            storage->activation_desc.hard_sigmoid.Beta = adapter.GetAttributeFloat("activation_beta", 0.5f);
            storage->fused_activation_op_desc = { DML_OPERATOR_ACTIVATION_HARD_SIGMOID, &storage->activation_desc.hard_sigmoid };
        } else if (fused_act == "Linear") {
            storage->activation_desc.linear = {};
            storage->activation_desc.linear.Alpha = adapter.GetAttributeFloat("activation_alpha", 1.0f);
            storage->activation_desc.linear.Beta = adapter.GetAttributeFloat("activation_beta", 0.0f);
            storage->fused_activation_op_desc = { DML_OPERATOR_ACTIVATION_LINEAR, &storage->activation_desc.linear };
        } else {
            storage->has_fused_activation = false;
        }
    }

    storage->desc.Mode           = DML_CONVOLUTION_MODE_CROSS_CORRELATION;
    storage->desc.Direction      = direction;
    storage->desc.DimensionCount = k.spatial_dim_count;
    storage->desc.Strides        = storage->strides.data();
    storage->desc.Dilations      = storage->dilations.data();
    storage->desc.StartPadding   = storage->start_pad.data();
    storage->desc.EndPadding     = storage->end_pad.data();
    storage->desc.OutputPadding  = storage->out_pad.data();
    storage->desc.GroupCount     = static_cast<UINT>(group);
    storage->desc.FusedActivation= storage->has_fused_activation ? &storage->fused_activation_op_desc : nullptr;

    TranslatedOp result;
    if (has_bias) {
        auto* b_info = LookupShape(value_shapes, inputs[2]);
        // Bias matches output channels: w_sizes[0] for forward, w_sizes[1]*group for backward.
        uint32_t bias_channels = (direction == DML_CONVOLUTION_DIRECTION_FORWARD)
            ? w_sizes[0]
            : w_sizes[1] * static_cast<uint32_t>(group);
        auto b_tensor = MakeTensorInfoAtAxis(
            {bias_channels}, dtype, 1 /*C-axis*/,
            static_cast<uint32_t>(x_tensor.sizes.size()));
        result.input_tensors = { x_tensor, w_tensor, b_tensor };
        result.input_buffer_descs = { x_tensor.ToBufferDesc(), w_tensor.ToBufferDesc(), b_tensor.ToBufferDesc() };
        result.input_tensor_descs.resize(3);
    } else {
        result.input_tensors = { x_tensor, w_tensor };
        result.input_buffer_descs = { x_tensor.ToBufferDesc(), w_tensor.ToBufferDesc() };
        result.input_tensor_descs.resize(2);
    }
    result.output_tensors = { out_tensor };
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_CONVOLUTION, &storage->desc };

    bool local_has_bias = has_bias;
    result.fixup = [storage, local_has_bias](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor   = &self.input_tensor_descs[0];
        storage->desc.FilterTensor  = &self.input_tensor_descs[1];
        storage->desc.BiasTensor    = local_has_bias ? &self.input_tensor_descs[2] : nullptr;
        storage->desc.OutputTensor  = &self.output_tensor_descs[0];
        storage->desc.Strides       = storage->strides.data();
        storage->desc.Dilations     = storage->dilations.data();
        storage->desc.StartPadding  = storage->start_pad.data();
        storage->desc.EndPadding    = storage->end_pad.data();
        storage->desc.OutputPadding = storage->out_pad.data();
        storage->desc.FusedActivation = storage->has_fused_activation ? &storage->fused_activation_op_desc : nullptr;
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// P3 — Pooling helpers
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateAveragePool(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&,
    bool global = false) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    int64_t count_include_pad = adapter.GetAttributeInt("count_include_pad", 0);

    auto in_sizes = in_info->sizes;  // 4D NCHW
    std::vector<uint32_t> out_sizes;
    uint32_t spatial = static_cast<uint32_t>(in_sizes.size()) - 2;
    if (spatial < 1) spatial = 1;

    std::vector<uint32_t> window_size, strides, start_pad, end_pad;
    if (global) {
        // Pool over entire spatial extent.
        for (uint32_t i = 0; i < spatial; ++i) window_size.push_back(in_sizes[2 + i]);
        strides.assign(spatial, 1u);
        start_pad.assign(spatial, 0u);
        end_pad.assign(spatial, 0u);
    } else {
        auto ks = adapter.GetAttributeInts("kernel_shape");
        for (int64_t v : ks) window_size.push_back(static_cast<uint32_t>(v));
        while (window_size.size() < spatial) window_size.insert(window_size.begin(), 1u);
        auto st = adapter.GetAttributeInts("strides");
        for (int64_t v : st) strides.push_back(static_cast<uint32_t>(v));
        while (strides.size() < spatial) strides.push_back(1u);
        auto pads = adapter.GetAttributeInts("pads");
        if (pads.size() >= 2 * spatial) {
            for (size_t i = 0; i < spatial; ++i) start_pad.push_back(static_cast<uint32_t>(pads[i]));
            for (size_t i = 0; i < spatial; ++i) end_pad.push_back(static_cast<uint32_t>(pads[spatial + i]));
        } else {
            start_pad.assign(spatial, 0u);
            end_pad.assign(spatial, 0u);
        }
    }

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    if (out_edge) {
        out_sizes = out_edge->sizes;
    } else {
        out_sizes = in_sizes;
        for (uint32_t d = 0; d < spatial; ++d)
            out_sizes[2 + d] = (in_sizes[2 + d] + start_pad[d] + end_pad[d] - window_size[d]) / strides[d] + 1;
    }

    auto in_tensor  = MakeTensorInfo(in_sizes,  in_info->data_type);
    auto out_tensor = MakeTensorInfo(out_sizes, in_info->data_type);

    struct AvgPoolStorage {
        DML_AVERAGE_POOLING1_OPERATOR_DESC desc{};
        std::vector<uint32_t> window_size, strides, start_pad, end_pad, dilations;
    };
    auto storage = std::make_shared<AvgPoolStorage>();
    storage->window_size = window_size;
    storage->strides     = strides;
    storage->start_pad   = start_pad;
    storage->end_pad     = end_pad;
    storage->dilations.assign(spatial, 1u);
    storage->desc.DimensionCount = spatial;
    storage->desc.WindowSize     = storage->window_size.data();
    storage->desc.Strides        = storage->strides.data();
    storage->desc.StartPadding   = storage->start_pad.data();
    storage->desc.EndPadding     = storage->end_pad.data();
    storage->desc.Dilations      = storage->dilations.data();
    storage->desc.IncludePadding = (count_include_pad != 0);

    TranslatedOp result;
    result.input_tensors = { in_tensor };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs = { in_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_AVERAGE_POOLING1, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor  = &self.input_tensor_descs[0];
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
        storage->desc.WindowSize   = storage->window_size.data();
        storage->desc.Strides      = storage->strides.data();
        storage->desc.StartPadding = storage->start_pad.data();
        storage->desc.EndPadding   = storage->end_pad.data();
        storage->desc.Dilations    = storage->dilations.data();
    };
    result.FixupPointers();
    return result;
}

static std::optional<TranslatedOp> TranslateMaxPool(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&,
    bool global = false) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    auto in_sizes = in_info->sizes;
    std::vector<uint32_t> out_sizes;
    uint32_t spatial = static_cast<uint32_t>(in_sizes.size()) - 2;
    if (spatial < 1) spatial = 1;

    std::vector<uint32_t> window_size, strides, dilations, start_pad, end_pad;
    if (global) {
        for (uint32_t i = 0; i < spatial; ++i) window_size.push_back(in_sizes[2 + i]);
        strides.assign(spatial, 1u);
        dilations.assign(spatial, 1u);
        start_pad.assign(spatial, 0u);
        end_pad.assign(spatial, 0u);
    } else {
        auto ks = adapter.GetAttributeInts("kernel_shape");
        for (int64_t v : ks) window_size.push_back(static_cast<uint32_t>(v));
        while (window_size.size() < spatial) window_size.insert(window_size.begin(), 1u);
        auto st = adapter.GetAttributeInts("strides");
        for (int64_t v : st) strides.push_back(static_cast<uint32_t>(v));
        while (strides.size() < spatial) strides.push_back(1u);
        auto di = adapter.GetAttributeInts("dilations");
        for (int64_t v : di) dilations.push_back(static_cast<uint32_t>(v));
        while (dilations.size() < spatial) dilations.push_back(1u);
        auto pads = adapter.GetAttributeInts("pads");
        if (pads.size() >= 2 * spatial) {
            for (size_t i = 0; i < spatial; ++i) start_pad.push_back(static_cast<uint32_t>(pads[i]));
            for (size_t i = 0; i < spatial; ++i) end_pad.push_back(static_cast<uint32_t>(pads[spatial + i]));
        } else {
            start_pad.assign(spatial, 0u);
            end_pad.assign(spatial, 0u);
        }
    }

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    if (out_edge) {
        out_sizes = out_edge->sizes;
    } else {
        out_sizes = in_sizes;
        for (uint32_t d = 0; d < spatial; ++d) {
            uint32_t dk = (window_size[d] - 1) * dilations[d] + 1;
            out_sizes[2 + d] = (in_sizes[2 + d] + start_pad[d] + end_pad[d] - dk) / strides[d] + 1;
        }
    }

    auto in_tensor  = MakeTensorInfo(in_sizes,  in_info->data_type);
    auto out_tensor = MakeTensorInfo(out_sizes, in_info->data_type);

    struct MaxPoolStorage {
        DML_MAX_POOLING2_OPERATOR_DESC desc{};
        std::vector<uint32_t> window_size, strides, dilations, start_pad, end_pad;
    };
    auto storage = std::make_shared<MaxPoolStorage>();
    storage->window_size = window_size;
    storage->strides     = strides;
    storage->dilations   = dilations;
    storage->start_pad   = start_pad;
    storage->end_pad     = end_pad;
    storage->desc.DimensionCount = spatial;
    storage->desc.WindowSize     = storage->window_size.data();
    storage->desc.Strides        = storage->strides.data();
    storage->desc.Dilations      = storage->dilations.data();
    storage->desc.StartPadding   = storage->start_pad.data();
    storage->desc.EndPadding     = storage->end_pad.data();
    storage->desc.OutputIndicesTensor = nullptr;

    TranslatedOp result;
    result.input_tensors = { in_tensor };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs = { in_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_MAX_POOLING2, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor  = &self.input_tensor_descs[0];
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
        storage->desc.WindowSize   = storage->window_size.data();
        storage->desc.Strides      = storage->strides.data();
        storage->desc.Dilations    = storage->dilations.data();
        storage->desc.StartPadding = storage->start_pad.data();
        storage->desc.EndPadding   = storage->end_pad.data();
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// P4 — Data movement: Concat, Split, Pad, Gather, GatherElements, GatherND
//      Tile, Slice, Expand, DepthToSpace, SpaceToDepth, OneHot,
//      ScatterElements, ScatterND, ConstantOfShape
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateConcat(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    int64_t axis = adapter.GetAttributeInt("axis", 0);

    auto* first_info = LookupShape(value_shapes, inputs[0]);
    if (!first_info) return std::nullopt;

    std::vector<DmlTensorInfo> in_tensors;
    for (auto& name : inputs) {
        auto* info = LookupShape(value_shapes, name);
        if (!info) return std::nullopt;
        in_tensors.push_back(MakeTensorInfo(info->sizes, info->data_type));
    }

    size_t padded_rank = in_tensors[0].sizes.size();
    size_t orig_rank = first_info->original_rank ? first_info->original_rank : padded_rank;
    size_t pad_offset = padded_rank - orig_rank;
    if (axis < 0) axis += static_cast<int64_t>(orig_rank);
    axis += static_cast<int64_t>(pad_offset);
    UINT dml_axis = static_cast<UINT>(axis);

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    std::vector<uint32_t> out_sizes;
    if (out_edge) {
        out_sizes = PadToMinDims(out_edge->sizes);
    } else {
        out_sizes = in_tensors[0].sizes;
        uint32_t concat_dim = 0;
        for (auto& t : in_tensors) concat_dim += t.sizes[dml_axis];
        out_sizes[dml_axis] = concat_dim;
    }

    auto out_tensor = MakeTensorInfo(out_sizes, in_tensors[0].data_type);

    struct ConcatStorage {
        DML_JOIN_OPERATOR_DESC desc{};
        UINT axis_val;
    };
    auto storage = std::make_shared<ConcatStorage>();
    storage->axis_val = dml_axis;
    storage->desc.Axis       = dml_axis;
    storage->desc.InputCount = static_cast<UINT>(inputs.size());

    TranslatedOp result;
    result.input_tensors = std::move(in_tensors);
    result.output_tensors = { out_tensor };
    for (auto& t : result.input_tensors) result.input_buffer_descs.push_back(t.ToBufferDesc());
    result.input_tensor_descs.resize(result.input_tensors.size());
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_JOIN, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensors  = self.input_tensor_descs.data();
        storage->desc.OutputTensor  = &self.output_tensor_descs[0];
        storage->desc.InputCount    = static_cast<UINT>(self.input_tensor_descs.size());
    };
    result.FixupPointers();
    return result;
}

static std::optional<TranslatedOp> TranslateSplit(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    int64_t axis = adapter.GetAttributeInt("axis", 0);
    size_t padded_rank = in_info->sizes.size();
    size_t orig_rank = in_info->original_rank ? in_info->original_rank : padded_rank;
    size_t pad_offset = padded_rank - orig_rank;
    if (axis < 0) axis += static_cast<int64_t>(orig_rank);
    axis += static_cast<int64_t>(pad_offset);
    UINT dml_axis = static_cast<UINT>(axis);

    // Compute output shapes: split along axis, either from 'split' attribute or equal partitions.
    auto split_attr = adapter.GetAttributeInts("split");
    std::vector<DmlTensorInfo> out_tensors;
    std::vector<uint32_t> split_sizes_vec;
    if (!split_attr.empty()) {
        for (int64_t s : split_attr) split_sizes_vec.push_back(static_cast<uint32_t>(s));
    } else {
        uint32_t total = in_info->sizes[axis];
        uint32_t per_split = total / static_cast<uint32_t>(outputs.size());
        split_sizes_vec.assign(outputs.size(), per_split);
    }
    for (size_t o = 0; o < outputs.size(); ++o) {
        auto* out_edge = LookupShape(value_shapes, outputs[o]);
        if (out_edge) {
            out_tensors.push_back(MakeTensorInfo(out_edge->sizes, in_info->data_type));
        } else {
            auto osizes = in_info->sizes;
            osizes[axis] = split_sizes_vec[o];
            out_tensors.push_back(MakeTensorInfo(osizes, in_info->data_type));
        }
    }

    auto in_tensor = MakeTensorInfo(in_info->sizes, in_info->data_type);

    struct SplitStorage {
        DML_SPLIT_OPERATOR_DESC desc{};
        std::vector<uint32_t> split_sizes;
        UINT axis_val;
    };
    auto storage = std::make_shared<SplitStorage>();
    storage->axis_val    = dml_axis;
    storage->split_sizes = split_sizes_vec;
    storage->desc.Axis        = dml_axis;
    storage->desc.OutputCount = static_cast<UINT>(outputs.size());

    TranslatedOp result;
    result.input_tensors  = { in_tensor };
    result.output_tensors = std::move(out_tensors);
    result.input_buffer_descs = { in_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    for (auto& t : result.output_tensors) result.output_buffer_descs.push_back(t.ToBufferDesc());
    result.output_tensor_descs.resize(result.output_tensors.size());
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_SPLIT, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor   = &self.input_tensor_descs[0];
        storage->desc.OutputTensors = self.output_tensor_descs.data();
        storage->desc.OutputCount   = static_cast<UINT>(self.output_tensor_descs.size());
    };
    result.FixupPointers();
    return result;
}

static std::optional<TranslatedOp> TranslateGather(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 2 || outputs.empty()) return std::nullopt;

    auto* data_info    = LookupShape(value_shapes, inputs[0]);
    auto* indices_info = LookupShape(value_shapes, inputs[1]);
    auto* out_edge     = LookupShape(value_shapes, outputs[0]);
    if (!data_info || !indices_info || !out_edge) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);

    // Use original_rank from pre-seeded DmlTensorInfo (set during shape propagation).
    // This is the ONNX rank before PadToMinDims — critical for axis adjustment and
    // IndexDimensions (scalar indices have original_rank=0).
    size_t ort_data_rank    = data_info->original_rank ? data_info->original_rank : data_info->sizes.size();
    size_t ort_indices_rank = indices_info->original_rank;  // 0 for scalar indices
    size_t ort_out_rank     = out_edge->original_rank ? out_edge->original_rank : out_edge->sizes.size();

    int64_t axis = adapter.GetAttributeInt("axis", 0);
    if (axis < 0) axis += static_cast<int64_t>(ort_data_rank);

    // DML requires all tensors to have the same DimensionCount (min 4).
    size_t dim_count = std::max({ort_data_rank, ort_indices_rank, ort_out_rank, size_t(4)});
    UINT dml_axis = static_cast<UINT>((dim_count - ort_data_rank) + axis);

    auto pad_to = [](const std::vector<uint32_t>& sizes, size_t target) {
        if (sizes.size() >= target) return sizes;
        std::vector<uint32_t> padded(target - sizes.size(), 1u);
        padded.insert(padded.end(), sizes.begin(), sizes.end());
        return padded;
    };

    auto data_tensor    = MakeTensorInfo(pad_to(data_info->sizes, dim_count), data_info->data_type);
    auto indices_tensor = MakeTensorInfo(pad_to(indices_info->sizes, dim_count), indices_info->data_type);
    auto out_tensor     = MakeTensorInfo(pad_to(out_edge->sizes, dim_count), data_info->data_type);

    struct GatherStorage { DML_GATHER_OPERATOR_DESC desc{}; };
    auto storage = std::make_shared<GatherStorage>();
    storage->desc.Axis            = dml_axis;
    storage->desc.IndexDimensions = static_cast<UINT>(ort_indices_rank);

    TranslatedOp result;
    result.input_tensors  = { data_tensor, indices_tensor };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs  = { data_tensor.ToBufferDesc(), indices_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(2);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_GATHER, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor   = &self.input_tensor_descs[0];
        storage->desc.IndicesTensor = &self.input_tensor_descs[1];
        storage->desc.OutputTensor  = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

static std::optional<TranslatedOp> TranslateGatherElements(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 2 || outputs.empty()) return std::nullopt;

    auto* data_info    = LookupShape(value_shapes, inputs[0]);
    auto* indices_info = LookupShape(value_shapes, inputs[1]);
    if (!data_info || !indices_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    int64_t axis = adapter.GetAttributeInt("axis", 0);
    size_t padded_rank = data_info->sizes.size();
    size_t orig_rank = data_info->original_rank ? data_info->original_rank : padded_rank;
    size_t pad_offset = padded_rank - orig_rank;
    if (axis < 0) axis += static_cast<int64_t>(orig_rank);
    axis += static_cast<int64_t>(pad_offset);
    UINT dml_axis = static_cast<UINT>(axis);

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    auto data_tensor    = MakeTensorInfo(data_info->sizes, data_info->data_type);
    auto indices_tensor = MakeTensorInfo(indices_info->sizes, indices_info->data_type);
    auto out_tensor     = out_edge
        ? MakeTensorInfo(out_edge->sizes, data_info->data_type)
        : MakeTensorInfo(indices_info->sizes, data_info->data_type);

    struct GEStorage { DML_GATHER_ELEMENTS_OPERATOR_DESC desc{}; };
    auto storage = std::make_shared<GEStorage>();
    storage->desc.Axis = dml_axis;

    TranslatedOp result;
    result.input_tensors  = { data_tensor, indices_tensor };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs  = { data_tensor.ToBufferDesc(), indices_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(2);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_GATHER_ELEMENTS, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor   = &self.input_tensor_descs[0];
        storage->desc.IndicesTensor = &self.input_tensor_descs[1];
        storage->desc.OutputTensor  = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

static std::optional<TranslatedOp> TranslateGatherND(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 2 || outputs.empty()) return std::nullopt;

    auto* data_info    = LookupShape(value_shapes, inputs[0]);
    auto* indices_info = LookupShape(value_shapes, inputs[1]);
    auto* out_edge     = LookupShape(value_shapes, outputs[0]);
    if (!data_info || !indices_info || !out_edge) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    int64_t batch_dims = adapter.GetAttributeInt("batch_dims", 0);

    size_t ort_data_rank    = data_info->original_rank ? data_info->original_rank : data_info->sizes.size();
    size_t ort_indices_rank = indices_info->original_rank ? indices_info->original_rank : indices_info->sizes.size();
    size_t ort_out_rank     = out_edge->original_rank ? out_edge->original_rank : out_edge->sizes.size();

    size_t dim_count = std::max({ort_data_rank, ort_indices_rank, ort_out_rank});

    auto pad_to = [](const std::vector<uint32_t>& sizes, size_t target) {
        if (sizes.size() >= target) return sizes;
        std::vector<uint32_t> padded(target - sizes.size(), 1u);
        padded.insert(padded.end(), sizes.begin(), sizes.end());
        return padded;
    };

    auto data_tensor    = MakeTensorInfo(pad_to(data_info->sizes, dim_count), data_info->data_type);
    auto indices_tensor = MakeTensorInfo(pad_to(indices_info->sizes, dim_count), indices_info->data_type);
    auto out_tensor     = MakeTensorInfo(pad_to(out_edge->sizes, dim_count), data_info->data_type);

    struct GNDStorage { DML_GATHER_ND1_OPERATOR_DESC desc{}; };
    auto storage = std::make_shared<GNDStorage>();
    storage->desc.InputDimensionCount   = static_cast<UINT>(ort_data_rank);
    storage->desc.IndicesDimensionCount = static_cast<UINT>(ort_indices_rank);
    storage->desc.BatchDimensionCount   = static_cast<UINT>(batch_dims);

    TranslatedOp result;
    result.input_tensors  = { data_tensor, indices_tensor };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs  = { data_tensor.ToBufferDesc(), indices_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(2);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_GATHER_ND1, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor   = &self.input_tensor_descs[0];
        storage->desc.IndicesTensor = &self.input_tensor_descs[1];
        storage->desc.OutputTensor  = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

static std::optional<TranslatedOp> TranslatePad(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>& initializers) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    std::string mode_str = adapter.GetAttributeString("mode", "constant");
    DML_PADDING_MODE mode = DML_PADDING_MODE_CONSTANT;
    if (mode_str == "reflect")  mode = DML_PADDING_MODE_REFLECTION;
    if (mode_str == "edge")     mode = DML_PADDING_MODE_EDGE;
    if (mode_str == "wrap")     mode = DML_PADDING_MODE_WRAP;

    size_t rank = in_info->sizes.size();
    // pads input: [begin_0, begin_1, ..., end_0, end_1, ...]
    std::vector<uint32_t> start_pad(rank, 0u), end_pad(rank, 0u);
    if (inputs.size() > 1 && !inputs[1].empty()) {
        // Opset 11+: pads from input[1] tensor
        auto it = initializers.find(inputs[1]);
        if (it != initializers.end() && it->second) {
            void* data = nullptr;
            ort_api.GetTensorMutableData(const_cast<OrtValue*>(it->second), &data);
            if (data) {
                auto* pads = static_cast<const int64_t*>(data);
                for (size_t i = 0; i < rank; ++i) start_pad[i] = static_cast<uint32_t>(pads[i]);
                for (size_t i = 0; i < rank; ++i) end_pad[i]   = static_cast<uint32_t>(pads[rank + i]);
            }
        }
    } else {
        // Opset 7: pads from attribute
        auto pads_attr = adapter.GetAttributeInts("pads");
        if (pads_attr.size() >= 2 * rank) {
            for (size_t i = 0; i < rank; ++i) start_pad[i] = static_cast<uint32_t>(pads_attr[i]);
            for (size_t i = 0; i < rank; ++i) end_pad[i]   = static_cast<uint32_t>(pads_attr[rank + i]);
        }
    }

    // Pad to 4D
    size_t padded_rank = std::max<size_t>(rank, 4);
    size_t pad_count   = padded_rank - rank;
    std::vector<uint32_t> dml_start(pad_count, 0u), dml_end(pad_count, 0u);
    dml_start.insert(dml_start.end(), start_pad.begin(), start_pad.end());
    dml_end.insert(dml_end.end(), end_pad.begin(), end_pad.end());

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    std::vector<uint32_t> out_sizes;
    if (out_edge) {
        out_sizes = out_edge->sizes;
    } else {
        out_sizes.resize(rank);
        for (size_t d = 0; d < rank; ++d) out_sizes[d] = in_info->sizes[d] + start_pad[d] + end_pad[d];
    }

    auto in_tensor  = MakeTensorInfo(in_info->sizes, in_info->data_type);
    auto out_tensor = MakeTensorInfo(out_sizes, in_info->data_type);

    struct PadStorage {
        DML_PADDING1_OPERATOR_DESC desc{};
        std::vector<uint32_t> start_pad, end_pad;
        float constant_value = 0.0f;
    };
    auto storage = std::make_shared<PadStorage>();
    storage->start_pad = dml_start;
    storage->end_pad   = dml_end;
    storage->desc.PaddingMode      = mode;
    storage->desc.PaddingValueDataType = in_info->data_type;
    storage->desc.PaddingValue     = {};  // zero-initialized DML_SCALAR_UNION
    storage->desc.DimensionCount   = static_cast<UINT>(padded_rank);
    storage->desc.StartPadding     = storage->start_pad.data();
    storage->desc.EndPadding       = storage->end_pad.data();

    TranslatedOp result;
    result.input_tensors  = { in_tensor };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs  = { in_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_PADDING1, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor  = &self.input_tensor_descs[0];
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
        storage->desc.StartPadding = storage->start_pad.data();
        storage->desc.EndPadding   = storage->end_pad.data();
    };
    result.FixupPointers();
    return result;
}

static std::optional<TranslatedOp> TranslateDepthToSpace(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    int64_t blocksize = adapter.GetAttributeInt("blocksize", 2);
    uint32_t bs = static_cast<uint32_t>(blocksize);
    std::string mode_str = adapter.GetAttributeString("mode", "DCR");
    DML_DEPTH_SPACE_ORDER order = (mode_str == "CRD")
        ? DML_DEPTH_SPACE_ORDER_COLUMN_ROW_DEPTH
        : DML_DEPTH_SPACE_ORDER_DEPTH_COLUMN_ROW;

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    std::vector<uint32_t> out_sizes;
    if (out_edge) {
        out_sizes = out_edge->sizes;
    } else {
        out_sizes = in_info->sizes;
        out_sizes[1] /= (bs * bs);
        if (out_sizes.size() > 2) out_sizes[2] *= bs;
        if (out_sizes.size() > 3) out_sizes[3] *= bs;
    }

    auto in_tensor  = MakeTensorInfo(in_info->sizes, in_info->data_type);
    auto out_tensor = MakeTensorInfo(out_sizes, in_info->data_type);

    struct D2SStorage { DML_DEPTH_TO_SPACE1_OPERATOR_DESC desc{}; };
    auto storage = std::make_shared<D2SStorage>();
    storage->desc.BlockSize  = static_cast<UINT>(blocksize);
    storage->desc.Order      = order;

    TranslatedOp result;
    result.input_tensors  = { in_tensor };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs  = { in_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_DEPTH_TO_SPACE1, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor  = &self.input_tensor_descs[0];
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

static std::optional<TranslatedOp> TranslateSpaceToDepth(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    int64_t blocksize = adapter.GetAttributeInt("blocksize", 2);
    uint32_t bs = static_cast<uint32_t>(blocksize);

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    std::vector<uint32_t> out_sizes;
    if (out_edge) {
        out_sizes = out_edge->sizes;
    } else {
        out_sizes = in_info->sizes;
        out_sizes[1] *= (bs * bs);
        if (out_sizes.size() > 2) out_sizes[2] /= bs;
        if (out_sizes.size() > 3) out_sizes[3] /= bs;
    }

    auto in_tensor  = MakeTensorInfo(in_info->sizes, in_info->data_type);
    auto out_tensor = MakeTensorInfo(out_sizes, in_info->data_type);

    struct S2DStorage { DML_SPACE_TO_DEPTH1_OPERATOR_DESC desc{}; };
    auto storage = std::make_shared<S2DStorage>();
    storage->desc.BlockSize = static_cast<UINT>(blocksize);
    storage->desc.Order     = DML_DEPTH_SPACE_ORDER_DEPTH_COLUMN_ROW;

    TranslatedOp result;
    result.input_tensors  = { in_tensor };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs  = { in_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_SPACE_TO_DEPTH1, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor  = &self.input_tensor_descs[0];
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

static std::optional<TranslatedOp> TranslateScatterElements(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 3 || outputs.empty()) return std::nullopt;

    auto* data_info    = LookupShape(value_shapes, inputs[0]);
    auto* indices_info = LookupShape(value_shapes, inputs[1]);
    auto* updates_info = LookupShape(value_shapes, inputs[2]);
    if (!data_info || !indices_info || !updates_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    int64_t axis = adapter.GetAttributeInt("axis", 0);
    size_t padded_rank = data_info->sizes.size();
    size_t orig_rank = data_info->original_rank ? data_info->original_rank : padded_rank;
    size_t pad_offset = padded_rank - orig_rank;
    if (axis < 0) axis += static_cast<int64_t>(orig_rank);
    axis += static_cast<int64_t>(pad_offset);
    UINT dml_axis = static_cast<UINT>(axis);

    auto data_tensor    = MakeTensorInfo(data_info->sizes, data_info->data_type);
    auto indices_tensor = MakeTensorInfo(indices_info->sizes, indices_info->data_type);
    auto updates_tensor = MakeTensorInfo(updates_info->sizes, data_info->data_type);
    auto out_tensor     = MakeTensorInfo(data_info->sizes, data_info->data_type);

    struct ScatterStorage { DML_SCATTER_OPERATOR_DESC desc{}; };
    auto storage = std::make_shared<ScatterStorage>();
    storage->desc.Axis = dml_axis;

    TranslatedOp result;
    result.input_tensors  = { data_tensor, indices_tensor, updates_tensor };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs  = { data_tensor.ToBufferDesc(), indices_tensor.ToBufferDesc(), updates_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(3);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_SCATTER, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor   = &self.input_tensor_descs[0];
        storage->desc.IndicesTensor = &self.input_tensor_descs[1];
        storage->desc.UpdatesTensor = &self.input_tensor_descs[2];
        storage->desc.OutputTensor  = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

static std::optional<TranslatedOp> TranslateScatterND(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 3 || outputs.empty()) return std::nullopt;

    auto* data_info    = LookupShape(value_shapes, inputs[0]);
    auto* indices_info = LookupShape(value_shapes, inputs[1]);
    auto* updates_info = LookupShape(value_shapes, inputs[2]);
    if (!data_info || !indices_info || !updates_info) return std::nullopt;

    // Read original ranks from ORT node type info (value_shapes has padded ranks).
    auto read_ort_rank = [&](bool is_input, size_t idx) -> size_t {
        size_t n = 0;
        if (is_input) ort_api.Node_GetNumInputs(node, &n); else ort_api.Node_GetNumOutputs(node, &n);
        if (idx >= n) return 0;
        std::vector<const OrtValueInfo*> vis(n, nullptr);
        if (is_input) ort_api.Node_GetInputs(node, vis.data(), n); else ort_api.Node_GetOutputs(node, vis.data(), n);
        if (!vis[idx] || !vis[idx]->GetTypeInfo() || !vis[idx]->GetTypeInfo()->tensor_type_info) return 0;
        size_t r = 0;
        ort_api.GetDimensionsCount(vis[idx]->GetTypeInfo()->tensor_type_info.get(), &r);
        return r;
    };
    size_t ort_data_rank    = read_ort_rank(true, 0);
    size_t ort_indices_rank = read_ort_rank(true, 1);
    size_t ort_updates_rank = read_ort_rank(true, 2);
    if (ort_data_rank == 0)    ort_data_rank    = data_info->sizes.size();
    if (ort_indices_rank == 0) ort_indices_rank = indices_info->sizes.size();
    if (ort_updates_rank == 0) ort_updates_rank = updates_info->sizes.size();

    size_t dim_count = std::max({ort_data_rank, ort_indices_rank, ort_updates_rank});

    auto pad_to = [](const std::vector<uint32_t>& sizes, size_t target) {
        if (sizes.size() >= target) return sizes;
        std::vector<uint32_t> padded(target - sizes.size(), 1u);
        padded.insert(padded.end(), sizes.begin(), sizes.end());
        return padded;
    };

    auto data_tensor    = MakeTensorInfo(pad_to(data_info->sizes, dim_count), data_info->data_type);
    auto indices_tensor = MakeTensorInfo(pad_to(indices_info->sizes, dim_count), indices_info->data_type);
    auto updates_tensor = MakeTensorInfo(pad_to(updates_info->sizes, dim_count), data_info->data_type);
    auto out_tensor     = MakeTensorInfo(pad_to(data_info->sizes, dim_count), data_info->data_type);

    struct ScatterNDStorage { DML_SCATTER_ND_OPERATOR_DESC desc{}; };
    auto storage = std::make_shared<ScatterNDStorage>();
    storage->desc.InputDimensionCount   = static_cast<UINT>(ort_data_rank);
    storage->desc.IndicesDimensionCount = static_cast<UINT>(ort_indices_rank);

    TranslatedOp result;
    result.input_tensors  = { data_tensor, indices_tensor, updates_tensor };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs  = { data_tensor.ToBufferDesc(), indices_tensor.ToBufferDesc(), updates_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(3);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_SCATTER_ND, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor   = &self.input_tensor_descs[0];
        storage->desc.IndicesTensor = &self.input_tensor_descs[1];
        storage->desc.UpdatesTensor = &self.input_tensor_descs[2];
        storage->desc.OutputTensor  = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// Slice → DML_OPERATOR_SLICE1
// Opset 10+: starts/ends/axes/steps come from inputs (constant initializers).
// Opset <10: they come from attributes.
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateSlice(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>& initializers) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    size_t padded_rank = in_info->sizes.size();
    size_t orig_rank = in_info->original_rank ? in_info->original_rank : padded_rank;
    size_t pad_offset = padded_rank - orig_rank;
    size_t rank = padded_rank;

    // Helper to read an int64 initializer tensor into a vector.
    auto read_int64_init = [&](size_t idx) -> std::vector<int64_t> {
        std::vector<int64_t> result;
        if (idx >= inputs.size() || inputs[idx].empty()) return result;
        auto it = initializers.find(inputs[idx]);
        if (it == initializers.end() || !it->second) return result;
        OrtTensorTypeAndShapeInfo* tsi = nullptr;
        ort_api.GetTensorTypeAndShape(const_cast<OrtValue*>(it->second), &tsi);
        if (!tsi) return result;
        size_t count = 0;
        ort_api.GetTensorShapeElementCount(tsi, &count);
        ONNXTensorElementDataType dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        ort_api.GetTensorElementType(tsi, &dt);
        ort_api.ReleaseTensorTypeAndShapeInfo(tsi);
        if (dt != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 || count == 0) return result;
        void* data = nullptr;
        OrtStatus* st = ort_api.GetTensorMutableData(const_cast<OrtValue*>(it->second), &data);
        if (st || !data) { if (st) ort_api.ReleaseStatus(st); return result; }
        auto* vals = static_cast<const int64_t*>(data);
        result.assign(vals, vals + count);
        return result;
    };

    std::vector<int64_t> starts_raw, ends_raw, axes_raw, steps_raw;

    OrtNodeAdapter adapter(node, ort_api);
    if (inputs.size() >= 3) {
        // Opset 10+: inputs[1]=starts, inputs[2]=ends, optional inputs[3]=axes, inputs[4]=steps
        starts_raw = read_int64_init(1);
        ends_raw   = read_int64_init(2);
        axes_raw   = read_int64_init(3);
        steps_raw  = read_int64_init(4);
    } else {
        // Opset <10: attributes
        starts_raw = adapter.GetAttributeInts("starts");
        ends_raw   = adapter.GetAttributeInts("ends");
        axes_raw   = adapter.GetAttributeInts("axes");
    }

    if (starts_raw.empty() || ends_raw.empty()) return std::nullopt;
    size_t slice_count = starts_raw.size();

    // Default axes = [0, 1, ..., N-1], default steps = all 1s.
    if (axes_raw.empty()) {
        axes_raw.resize(slice_count);
        std::iota(axes_raw.begin(), axes_raw.end(), 0);
    }
    if (steps_raw.empty()) {
        steps_raw.assign(slice_count, 1);
    }

    // Normalize axes and clamp starts/ends to valid ranges.
    // Build per-dimension offsets, sizes, strides (full rank).
    std::vector<uint32_t> offsets(rank, 0u);
    std::vector<uint32_t> sizes = in_info->sizes;
    std::vector<int32_t>  strides(rank, 1);

    for (size_t i = 0; i < slice_count; ++i) {
        int64_t axis = axes_raw[i];
        // ONNX axes are relative to the original (unpadded) rank.
        if (axis < 0) axis += static_cast<int64_t>(orig_rank);
        axis += static_cast<int64_t>(pad_offset);
        if (axis < 0 || axis >= static_cast<int64_t>(rank)) return std::nullopt;

        int64_t dim_size = static_cast<int64_t>(in_info->sizes[axis]);
        int64_t step = steps_raw[i];
        if (step == 0) return std::nullopt;

        int64_t start = starts_raw[i];
        int64_t end   = ends_raw[i];

        // Clamp start/end following ONNX Slice semantics.
        if (start < 0) start += dim_size;
        if (end < 0)   end   += dim_size;
        start = std::clamp<int64_t>(start, 0, dim_size);
        end   = std::clamp<int64_t>(end,   0, dim_size);

        if (step > 0) {
            if (start >= end) {
                offsets[axis] = static_cast<uint32_t>(start);
                sizes[axis] = 0;
            } else {
                offsets[axis] = static_cast<uint32_t>(start);
                sizes[axis] = static_cast<uint32_t>((end - start + step - 1) / step);
            }
        } else {
            // Negative step: DML SLICE1 supports negative InputWindowStrides.
            if (end >= start) {
                offsets[axis] = static_cast<uint32_t>(start);
                sizes[axis] = 0;
            } else {
                offsets[axis] = static_cast<uint32_t>(start);
                sizes[axis] = static_cast<uint32_t>((start - end + (-step) - 1) / (-step));
            }
        }
        strides[axis] = static_cast<int32_t>(step);
    }

    // Pad to 4D (rank is already >= 4 from value_shapes padding).
    size_t pad_count = 0;
    std::vector<uint32_t> dml_offsets(pad_count, 0u);
    dml_offsets.insert(dml_offsets.end(), offsets.begin(), offsets.end());
    std::vector<uint32_t> dml_sizes(pad_count, 1u);
    dml_sizes.insert(dml_sizes.end(), sizes.begin(), sizes.end());
    std::vector<int32_t> dml_strides(pad_count, 1);
    dml_strides.insert(dml_strides.end(), strides.begin(), strides.end());

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    auto in_tensor  = MakeTensorInfo(in_info->sizes, in_info->data_type);
    auto out_tensor = out_edge ? MakeTensorInfo(out_edge->sizes, in_info->data_type)
                               : MakeTensorInfo(sizes, in_info->data_type);

    struct SliceStorage {
        DML_SLICE1_OPERATOR_DESC desc{};
        std::vector<uint32_t> offsets;
        std::vector<uint32_t> sizes;
        std::vector<int32_t>  strides;
    };
    auto storage = std::make_shared<SliceStorage>();
    storage->offsets = dml_offsets;
    storage->sizes   = dml_sizes;
    storage->strides = dml_strides;
    storage->desc.DimensionCount     = static_cast<UINT>(rank);
    storage->desc.InputWindowOffsets = storage->offsets.data();
    storage->desc.InputWindowSizes   = storage->sizes.data();
    storage->desc.InputWindowStrides = storage->strides.data();

    TranslatedOp result;
    result.input_tensors  = { in_tensor };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs  = { in_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_SLICE1, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor        = &self.input_tensor_descs[0];
        storage->desc.OutputTensor       = &self.output_tensor_descs[0];
        storage->desc.InputWindowOffsets = storage->offsets.data();
        storage->desc.InputWindowSizes   = storage->sizes.data();
        storage->desc.InputWindowStrides = storage->strides.data();
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// Resize → DML_OPERATOR_RESAMPLE2
// Computes scales from input/output shapes and applies coordinate transform
// pixel offsets following the ONNX spec.
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateResize(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>& initializers) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);

    const char* node_op = nullptr;
    ort_api.Node_GetOperatorType(node, &node_op);
    bool is_upsample = node_op && std::strcmp(node_op, "Upsample") == 0;

    std::string mode_str = adapter.GetAttributeString("mode", "nearest");
    std::string coord_transform = adapter.GetAttributeString(
        "coordinate_transformation_mode", is_upsample ? "asymmetric" : "half_pixel");
    std::string nearest_mode = adapter.GetAttributeString(
        "nearest_mode", is_upsample ? "floor" : "round_prefer_floor");

    DML_INTERPOLATION_MODE interp_mode = DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR;
    if (mode_str == "linear" || mode_str == "bilinear")
        interp_mode = DML_INTERPOLATION_MODE_LINEAR;

    // Determine output shape and scales.
    // Opset 11+: inputs are [X, roi, scales, sizes]. Sizes takes priority over scales.
    // We try output edge shape first (if available from value_shapes), else derive from
    // scales/sizes initializers, else fail.
    size_t rank = in_info->sizes.size();
    std::vector<float> scales(rank, 1.0f);
    std::vector<uint32_t> out_sizes(rank);

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    if (out_edge) {
        out_sizes = out_edge->sizes;
        for (size_t i = 0; i < rank; ++i) {
            scales[i] = (in_info->sizes[i] > 0)
                ? static_cast<float>(out_sizes[i]) / static_cast<float>(in_info->sizes[i])
                : 1.0f;
        }
    } else {
        // Try sizes input (index 3), then scales input (index 2).
        bool have_output = false;
        // Sizes input (opset 11+, index 3)
        if (inputs.size() > 3 && !inputs[3].empty()) {
            auto* sizes_info = LookupShape(value_shapes, inputs[3]);
            if (sizes_info && !sizes_info->sizes.empty()) {
                out_sizes = sizes_info->sizes;
                for (size_t i = 0; i < rank; ++i)
                    scales[i] = static_cast<float>(out_sizes[i]) / static_cast<float>(in_info->sizes[i]);
                have_output = true;
            }
        }
        // Scales input — Upsample: index 1; Resize opset 11+: index 2.
        size_t scales_idx = is_upsample ? 1 : 2;
        if (!have_output && inputs.size() > scales_idx && !inputs[scales_idx].empty()) {
            auto it = initializers.find(inputs[scales_idx]);
            if (it != initializers.end() && it->second) {
                OrtTensorTypeAndShapeInfo* tsi = nullptr;
                ort_api.GetTensorTypeAndShape(const_cast<OrtValue*>(it->second), &tsi);
                if (tsi) {
                    size_t count = 0;
                    ort_api.GetTensorShapeElementCount(tsi, &count);
                    ort_api.ReleaseTensorTypeAndShapeInfo(tsi);
                    void* data = nullptr;
                    OrtStatus* st = ort_api.GetTensorMutableData(const_cast<OrtValue*>(it->second), &data);
                    if (st) { ort_api.ReleaseStatus(st); }
                    else if (data && count > 0) {
                        auto* vals = static_cast<const float*>(data);
                        size_t n = std::min(count, rank);
                        for (size_t i = 0; i < n; ++i)
                            scales[i] = vals[i];
                    }
                }
                for (size_t i = 0; i < rank; ++i)
                    out_sizes[i] = static_cast<uint32_t>(std::round(in_info->sizes[i] * scales[i]));
                have_output = true;
            }
        }
        if (!have_output)
            return std::nullopt;
    }

    // Compute pixel offsets based on coordinate_transformation_mode.
    std::vector<float> input_pixel_offsets(rank, 0.0f);
    std::vector<float> output_pixel_offsets(rank, 0.0f);

    for (size_t i = 0; i < rank; ++i) {
        if (coord_transform == "half_pixel") {
            input_pixel_offsets[i]  =  0.5f;
            output_pixel_offsets[i] = -0.5f;
        } else if (coord_transform == "pytorch_half_pixel") {
            if (in_info->sizes[i] <= 1) {
                input_pixel_offsets[i]  = 0.0f;
                output_pixel_offsets[i] = 0.0f;
                scales[i] = FLT_MAX;
            } else {
                input_pixel_offsets[i]  =  0.5f;
                output_pixel_offsets[i] = -0.5f;
            }
        } else if (coord_transform == "align_corners") {
            input_pixel_offsets[i]  = 0.0f;
            output_pixel_offsets[i] = 0.0f;
            if (out_edge->sizes[i] > 1 && in_info->sizes[i] > 1) {
                scales[i] = static_cast<float>(out_edge->sizes[i] - 1) /
                            static_cast<float>(in_info->sizes[i] - 1);
            } else {
                scales[i] = FLT_MAX;
            }
        } else if (coord_transform == "asymmetric") {
            input_pixel_offsets[i]  = 0.0f;
            output_pixel_offsets[i] = 0.0f;
        } else if (coord_transform == "tf_half_pixel_for_nn") {
            input_pixel_offsets[i]  = 0.0f;
            output_pixel_offsets[i] = -0.5f;
        } else if (coord_transform == "half_pixel_symmetric") {
            input_pixel_offsets[i] = 0.5f - (static_cast<float>(in_info->sizes[i]) / 2.0f) *
                (1.0f - static_cast<float>(out_edge->sizes[i]) / (scales[i] * in_info->sizes[i]));
            output_pixel_offsets[i] = -0.5f;
        } else {
            // Default to asymmetric
            input_pixel_offsets[i]  = 0.0f;
            output_pixel_offsets[i] = 0.0f;
        }
    }

    // Apply nearest-neighbor rounding offset adjustments.
    DML_AXIS_DIRECTION rounding_dir = DML_AXIS_DIRECTION_DECREASING;
    if (interp_mode == DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR) {
        float offset_adj = 0.5f;
        if (nearest_mode == "round_prefer_floor") {
            rounding_dir = DML_AXIS_DIRECTION_INCREASING;
            offset_adj = 0.5f;
        } else if (nearest_mode == "round_prefer_ceil") {
            rounding_dir = DML_AXIS_DIRECTION_DECREASING;
            offset_adj = -0.5f;
        } else if (nearest_mode == "floor") {
            rounding_dir = DML_AXIS_DIRECTION_DECREASING;
            offset_adj = 0.0f;
        } else if (nearest_mode == "ceil") {
            rounding_dir = DML_AXIS_DIRECTION_INCREASING;
            offset_adj = 0.0f;
        }
        if (offset_adj != 0.0f) {
            for (auto& off : input_pixel_offsets) off += offset_adj;
        }
    }

    // Pad to 4D.
    size_t padded_rank = std::max<size_t>(rank, 4);
    size_t pad_count   = padded_rank - rank;
    std::vector<float> padded_scales(pad_count, 1.0f);
    padded_scales.insert(padded_scales.end(), scales.begin(), scales.end());
    std::vector<float> padded_in_offsets(pad_count, 0.5f);
    padded_in_offsets.insert(padded_in_offsets.end(), input_pixel_offsets.begin(), input_pixel_offsets.end());
    std::vector<float> padded_out_offsets(pad_count, -0.5f);
    padded_out_offsets.insert(padded_out_offsets.end(), output_pixel_offsets.begin(), output_pixel_offsets.end());

    auto in_tensor  = MakeTensorInfo(in_info->sizes, in_info->data_type);
    auto out_tensor = MakeTensorInfo(out_sizes, in_info->data_type);

    struct ResizeStorage {
        DML_RESAMPLE2_OPERATOR_DESC desc{};
        std::vector<float> scales;
        std::vector<float> input_offsets;
        std::vector<float> output_offsets;
    };
    auto storage = std::make_shared<ResizeStorage>();
    storage->scales         = padded_scales;
    storage->input_offsets  = padded_in_offsets;
    storage->output_offsets = padded_out_offsets;
    storage->desc.InterpolationMode = interp_mode;
    storage->desc.RoundingDirection = rounding_dir;
    storage->desc.DimensionCount    = static_cast<UINT>(padded_rank);
    storage->desc.Scales            = storage->scales.data();
    storage->desc.InputPixelOffsets  = storage->input_offsets.data();
    storage->desc.OutputPixelOffsets = storage->output_offsets.data();

    TranslatedOp result;
    result.input_tensors  = { in_tensor };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs  = { in_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_RESAMPLE2, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor        = &self.input_tensor_descs[0];
        storage->desc.OutputTensor       = &self.output_tensor_descs[0];
        storage->desc.Scales             = storage->scales.data();
        storage->desc.InputPixelOffsets   = storage->input_offsets.data();
        storage->desc.OutputPixelOffsets  = storage->output_offsets.data();
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// P5 — Normalization
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateLayerNorm(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&,
    bool simplified = false,
    int64_t default_axis = -1) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 2 || outputs.empty()) return std::nullopt;

    auto* in_info    = LookupShape(value_shapes, inputs[0]);
    auto* scale_info = LookupShape(value_shapes, inputs[1]);
    if (!in_info || !scale_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    float epsilon = adapter.GetAttributeFloat("epsilon", 1e-5f);
    int64_t axis  = adapter.GetAttributeInt("axis", default_axis);
    size_t padded_rank = in_info->sizes.size();
    size_t orig_rank = in_info->original_rank ? in_info->original_rank : padded_rank;
    size_t pad_offset = padded_rank - orig_rank;
    if (axis < 0) axis += static_cast<int64_t>(orig_rank);
    axis += static_cast<int64_t>(pad_offset);

    // Build axes array: from axis to padded_rank-1.
    std::vector<uint32_t> axes_vec;
    for (size_t i = static_cast<size_t>(axis); i < padded_rank; ++i)
        axes_vec.push_back(static_cast<uint32_t>(i));

    bool has_bias = (!simplified && inputs.size() > 2 && !inputs[2].empty() &&
                     value_shapes.count(inputs[2]));

    // ORT requires all MVN1 tensors to have the same DimensionCount.
    size_t dim_count = std::max<size_t>(in_info->sizes.size(), 4);
    auto in_tensor    = MakeTensorInfo(PadToMinDims(in_info->sizes, dim_count), in_info->data_type);
    auto out_tensor   = MakeTensorInfo(PadToMinDims(in_info->sizes, dim_count), in_info->data_type);

    // For InstanceNorm (default_axis==2), scale/bias are per-channel and must
    // be placed at the C axis (axis 1 in original rank). PadToMinDims would
    // right-align them to the wrong position. Use MakeTensorInfoAtAxis instead.
    DmlTensorInfo scale_tensor;
    bool is_instance_norm = (default_axis == 2);
    if (is_instance_norm) {
        uint32_t num_channels = scale_info->sizes.back();
        scale_tensor = MakeTensorInfoAtAxis(
            {num_channels}, scale_info->data_type,
            static_cast<uint32_t>(pad_offset + 1),
            static_cast<uint32_t>(dim_count));
    } else {
        scale_tensor = MakeTensorInfo(PadToMinDims(scale_info->sizes, dim_count), scale_info->data_type);
    }

    struct LNStorage {
        DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC desc{};
        std::vector<uint32_t> axes;
    };
    auto storage = std::make_shared<LNStorage>();
    storage->axes = axes_vec;
    storage->desc.Epsilon        = epsilon;
    storage->desc.NormalizeVariance = TRUE;
    storage->desc.AxisCount      = static_cast<UINT>(axes_vec.size());
    storage->desc.Axes           = storage->axes.data();
    storage->desc.FusedActivation= nullptr;

    TranslatedOp result;
    if (has_bias) {
        auto* bias_info = LookupShape(value_shapes, inputs[2]);
        DmlTensorInfo bias_tensor;
        if (is_instance_norm) {
            uint32_t num_channels = bias_info->sizes.back();
            bias_tensor = MakeTensorInfoAtAxis(
                {num_channels}, bias_info->data_type,
                static_cast<uint32_t>(pad_offset + 1),
                static_cast<uint32_t>(dim_count));
        } else {
            bias_tensor = MakeTensorInfo(PadToMinDims(bias_info->sizes, dim_count), bias_info->data_type);
        }
        result.input_tensors = { in_tensor, scale_tensor, bias_tensor };
        result.input_buffer_descs = { in_tensor.ToBufferDesc(), scale_tensor.ToBufferDesc(), bias_tensor.ToBufferDesc() };
        result.input_tensor_descs.resize(3);
    } else {
        result.input_tensors = { in_tensor, scale_tensor };
        result.input_buffer_descs = { in_tensor.ToBufferDesc(), scale_tensor.ToBufferDesc() };
        result.input_tensor_descs.resize(2);
    }
    result.output_tensors = { out_tensor };
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1, &storage->desc };

    bool local_has_bias = has_bias;
    result.fixup = [storage, local_has_bias](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor  = &self.input_tensor_descs[0];
        storage->desc.ScaleTensor  = &self.input_tensor_descs[1];
        storage->desc.BiasTensor   = local_has_bias ? &self.input_tensor_descs[2] : nullptr;
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
        storage->desc.Axes         = storage->axes.data();
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// GroupNorm — reshape to [B, G, C/G, H*W], MVN with CrossChannel=false, optional SiLU.
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateGroupNorm(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 3 || outputs.empty()) return std::nullopt;

    auto* in_info    = LookupShape(value_shapes, inputs[0]);
    auto* gamma_info = LookupShape(value_shapes, inputs[1]);
    auto* beta_info  = LookupShape(value_shapes, inputs[2]);
    auto* out_info   = LookupShape(value_shapes, outputs[0]);
    if (!in_info || !gamma_info || !beta_info || !out_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    float epsilon      = adapter.GetAttributeFloat("epsilon", 1e-5f);
    int64_t groups     = adapter.GetAttributeInt("groups", 32);
    int64_t activation = adapter.GetAttributeInt("activation", 0);
    int64_t channels_last = adapter.GetAttributeInt("channels_last", 1);

    auto& shape = in_info->sizes;  // pre-padded in value_shapes
    if (shape.size() < 4) return std::nullopt;

    // Determine layout — unpad to find actual dims.
    // value_shapes right-pads to 4D, so for 4D NCHW input the shape is already correct.
    uint32_t B, C, H, W;
    if (channels_last) {
        B = shape[0]; H = shape[1]; W = shape[2]; C = shape[3];
    } else {
        B = shape[0]; C = shape[1]; H = shape[2]; W = shape[3];
    }

    uint32_t G   = static_cast<uint32_t>(groups);
    uint32_t CpG = C / G;
    uint32_t S   = H * W;

    // MVN1 internal shape: [B, G, CpG, S]
    std::vector<uint32_t> mvn_sizes = { B, G, CpG, S };
    auto dtype = in_info->data_type;
    size_t elem_size = DmlDataTypeSize(dtype);

    DmlTensorInfo mvn_input;
    if (channels_last) {
        // NHWC → [B, G, CpG, S] with strides
        std::vector<uint32_t> strides = { S * C, CpG, 1, C };
        uint64_t total = static_cast<uint64_t>(B) * H * W * C * elem_size;
        mvn_input = MakeTensorInfoWithStrides(mvn_sizes, strides, dtype, total);
        mvn_input.sizes = mvn_sizes;  // override PadToMinDims since already 4D
        mvn_input.strides = strides;
    } else {
        // NCHW — flat reinterpret, no strides needed
        mvn_input = MakeTensorInfo(mvn_sizes, dtype);
    }

    // Gamma/beta: sizes match input ([B,G,CpG,S]), strides {0,CpG,1,0} for broadcast.
    // DML MVN requires Scale/Bias sizes == Input sizes.
    std::vector<uint32_t> gb_strides = { 0, CpG, 1, 0 };
    auto gamma_tensor = MakeTensorInfoWithStrides(mvn_sizes, gb_strides, gamma_info->data_type, gamma_info->total_bytes);
    gamma_tensor.sizes = mvn_sizes;
    gamma_tensor.strides = gb_strides;
    auto beta_tensor = MakeTensorInfoWithStrides(mvn_sizes, gb_strides, beta_info->data_type, beta_info->total_bytes);
    beta_tensor.sizes = mvn_sizes;
    beta_tensor.strides = gb_strides;

    auto mvn_output = MakeTensorInfo(mvn_sizes, dtype);

    struct GNStorage {
        DML_MEAN_VARIANCE_NORMALIZATION_OPERATOR_DESC desc{};
        // MVN output uses internal [B,G,CpG,S] shape; keep a separate desc
        // so RebuildTensorDescPointers (which writes the ONNX output shape
        // into output_buffer_descs[0]) doesn't clobber the MVN desc.
        DmlTensorInfo mvn_out_info;
        DML_BUFFER_TENSOR_DESC mvn_out_buf{};
        DML_TENSOR_DESC mvn_out_td{};
    };
    auto storage = std::make_shared<GNStorage>();
    storage->desc.Epsilon            = epsilon;
    storage->desc.NormalizeVariance  = TRUE;
    storage->desc.CrossChannel       = FALSE;
    storage->mvn_out_info            = mvn_output;

    TranslatedOp result;
    result.input_tensors  = { mvn_input, gamma_tensor, beta_tensor };
    result.output_tensors = { MakeTensorInfo(out_info->sizes, dtype) };

    result.input_buffer_descs  = { mvn_input.ToBufferDesc(), gamma_tensor.ToBufferDesc(), beta_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(3);
    result.output_buffer_descs = { mvn_output.ToBufferDesc() };
    result.output_tensor_descs.resize(1);

    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION, &storage->desc };

    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor  = &self.input_tensor_descs[0];
        storage->desc.ScaleTensor  = &self.input_tensor_descs[1];
        storage->desc.BiasTensor   = &self.input_tensor_descs[2];
        // Use the internal [B,G,CpG,S] shape for MVN output, not the ONNX shape.
        storage->mvn_out_buf = storage->mvn_out_info.ToBufferDesc();
        storage->mvn_out_buf.Sizes = storage->mvn_out_info.sizes.data();
        storage->mvn_out_buf.Strides = storage->mvn_out_info.strides.empty()
            ? nullptr : storage->mvn_out_info.strides.data();
        storage->mvn_out_td = { DML_TENSOR_TYPE_BUFFER, &storage->mvn_out_buf };
        storage->desc.OutputTensor = &storage->mvn_out_td;
    };
    result.FixupPointers();

    if (activation == 1) {
        // SiLU: x * sigmoid(x) — two sub_nodes after MVN1.
        // SubNode 0: Sigmoid(mvn_output)
        auto sig_storage = std::make_shared<DML_ACTIVATION_SIGMOID_OPERATOR_DESC>();

        SubNode sigmoid_node;
        sigmoid_node.input_tensors  = { mvn_output };
        sigmoid_node.output_tensors = { mvn_output };
        sigmoid_node.input_buffer_descs  = { mvn_output.ToBufferDesc() };
        sigmoid_node.input_tensor_descs.resize(1);
        sigmoid_node.output_buffer_descs = { mvn_output.ToBufferDesc() };
        sigmoid_node.output_tensor_descs.resize(1);
        sigmoid_node.desc_storage = sig_storage;
        sigmoid_node.op_desc = { DML_OPERATOR_ACTIVATION_SIGMOID, sig_storage.get() };
        sigmoid_node.input_from = { {-1, 0} };
        sigmoid_node.fixup = [sig_storage](SubNode& self) {
            RebuildSubNodePointers(self);
            sig_storage->InputTensor  = &self.input_tensor_descs[0];
            sig_storage->OutputTensor = &self.output_tensor_descs[0];
        };
        sigmoid_node.FixupPointers();

        // SubNode 1: Multiply(mvn_output, sigmoid_output)
        auto mul_storage = std::make_shared<DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC>();

        SubNode mul_node;
        mul_node.input_tensors  = { mvn_output, mvn_output };
        mul_node.output_tensors = { mvn_output };
        mul_node.input_buffer_descs  = { mvn_output.ToBufferDesc(), mvn_output.ToBufferDesc() };
        mul_node.input_tensor_descs.resize(2);
        mul_node.output_buffer_descs = { mvn_output.ToBufferDesc() };
        mul_node.output_tensor_descs.resize(1);
        mul_node.desc_storage = mul_storage;
        mul_node.op_desc = { DML_OPERATOR_ELEMENT_WISE_MULTIPLY, mul_storage.get() };
        mul_node.input_from = { {-1, 0}, {0, 0} };  // A=MVN1 output, B=Sigmoid output
        mul_node.fixup = [mul_storage](SubNode& self) {
            RebuildSubNodePointers(self);
            mul_storage->ATensor     = &self.input_tensor_descs[0];
            mul_storage->BTensor     = &self.input_tensor_descs[1];
            mul_storage->OutputTensor = &self.output_tensor_descs[0];
        };
        mul_node.FixupPointers();

        result.sub_nodes = { std::move(sigmoid_node), std::move(mul_node) };
    }

    return result;
}

static std::optional<TranslatedOp> TranslateBatchNorm(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    // inputs: X, scale, B, input_mean, input_var
    if (inputs.size() < 5 || outputs.empty()) return std::nullopt;

    auto* x_info     = LookupShape(value_shapes, inputs[0]);
    auto* scale_info = LookupShape(value_shapes, inputs[1]);
    auto* b_info     = LookupShape(value_shapes, inputs[2]);
    auto* mean_info  = LookupShape(value_shapes, inputs[3]);
    auto* var_info   = LookupShape(value_shapes, inputs[4]);
    if (!x_info || !scale_info || !b_info || !mean_info || !var_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    float epsilon = adapter.GetAttributeFloat("epsilon", 1e-5f);

    auto x_tensor   = MakeTensorInfo(x_info->sizes, x_info->data_type);
    auto out_tensor = MakeTensorInfo(x_info->sizes, x_info->data_type);

    uint32_t dim_count = static_cast<uint32_t>(x_tensor.sizes.size());
    std::vector<uint32_t> c_shape = {x_info->sizes[1]};
    auto scale_tensor = MakeTensorInfoAtAxis(c_shape, scale_info->data_type, 1, dim_count);
    auto b_tensor     = MakeTensorInfoAtAxis(c_shape, b_info->data_type,     1, dim_count);
    auto mean_tensor  = MakeTensorInfoAtAxis(c_shape, mean_info->data_type,  1, dim_count);
    auto var_tensor   = MakeTensorInfoAtAxis(c_shape, var_info->data_type,   1, dim_count);

    struct BNStorage { DML_BATCH_NORMALIZATION_OPERATOR_DESC desc{}; };
    auto storage = std::make_shared<BNStorage>();
    storage->desc.Spatial        = TRUE;
    storage->desc.Epsilon        = epsilon;
    storage->desc.FusedActivation= nullptr;

    TranslatedOp result;
    result.input_tensors  = { x_tensor, mean_tensor, var_tensor, scale_tensor, b_tensor };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs = {
        x_tensor.ToBufferDesc(), mean_tensor.ToBufferDesc(), var_tensor.ToBufferDesc(),
        scale_tensor.ToBufferDesc(), b_tensor.ToBufferDesc()
    };
    result.input_tensor_descs.resize(5);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_BATCH_NORMALIZATION, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor  = &self.input_tensor_descs[0];
        storage->desc.MeanTensor   = &self.input_tensor_descs[1];
        storage->desc.VarianceTensor= &self.input_tensor_descs[2];
        storage->desc.ScaleTensor  = &self.input_tensor_descs[3];
        storage->desc.BiasTensor   = &self.input_tensor_descs[4];
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

static std::optional<TranslatedOp> TranslateLRN(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    int64_t size  = adapter.GetAttributeInt("size", 1);
    float alpha   = adapter.GetAttributeFloat("alpha", 0.0001f);
    float beta    = adapter.GetAttributeFloat("beta", 0.75f);
    float bias    = adapter.GetAttributeFloat("bias", 1.0f);

    auto in_tensor  = MakeTensorInfo(in_info->sizes, in_info->data_type);
    auto out_tensor = MakeTensorInfo(in_info->sizes, in_info->data_type);

    struct LRNStorage { DML_LOCAL_RESPONSE_NORMALIZATION_OPERATOR_DESC desc{}; };
    auto storage = std::make_shared<LRNStorage>();
    storage->desc.CrossChannel = TRUE;
    storage->desc.LocalSize    = static_cast<UINT>(size);
    storage->desc.Alpha        = alpha;
    storage->desc.Beta         = beta;
    storage->desc.Bias         = bias;

    TranslatedOp result;
    result.input_tensors  = { in_tensor };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs  = { in_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_LOCAL_RESPONSE_NORMALIZATION, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor  = &self.input_tensor_descs[0];
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// P6 — Gemm, Clip, QuantizeLinear, DequantizeLinear
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateGemm(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 2 || outputs.empty()) return std::nullopt;

    auto* a_info = LookupShape(value_shapes, inputs[0]);
    auto* b_info = LookupShape(value_shapes, inputs[1]);
    if (!a_info || !b_info) return std::nullopt;

    bool has_c = (inputs.size() > 2 && !inputs[2].empty() && value_shapes.count(inputs[2]));

    OrtNodeAdapter adapter(node, ort_api);
    float alpha   = adapter.GetAttributeFloat("alpha", 1.0f);
    float beta    = adapter.GetAttributeFloat("beta", 1.0f);
    int64_t transA = adapter.GetAttributeInt("transA", 0);
    int64_t transB = adapter.GetAttributeInt("transB", 0);

    auto a_sizes = a_info->sizes;
    auto b_sizes = b_info->sizes;

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    std::vector<uint32_t> out_sizes;
    if (out_edge) {
        out_sizes = out_edge->sizes;
    } else {
        size_t a_rank = a_sizes.size();
        size_t b_rank = b_sizes.size();
        uint32_t M = transA ? a_sizes[a_rank - 1] : a_sizes[a_rank - 2];
        uint32_t N = transB ? b_sizes[b_rank - 2] : b_sizes[b_rank - 1];
        out_sizes = { M, N };
    }

    auto a_tensor   = MakeTensorInfo(a_info->sizes, a_info->data_type);
    auto b_tensor   = MakeTensorInfo(b_info->sizes, b_info->data_type);
    auto out_tensor = MakeTensorInfo(out_sizes, a_info->data_type);

    struct GemmStorage { DML_GEMM_OPERATOR_DESC desc{}; };
    auto storage = std::make_shared<GemmStorage>();
    storage->desc.TransA          = transA ? DML_MATRIX_TRANSFORM_TRANSPOSE : DML_MATRIX_TRANSFORM_NONE;
    storage->desc.TransB          = transB ? DML_MATRIX_TRANSFORM_TRANSPOSE : DML_MATRIX_TRANSFORM_NONE;
    storage->desc.Alpha           = alpha;
    storage->desc.Beta            = beta;
    storage->desc.FusedActivation = nullptr;

    TranslatedOp result;
    if (has_c) {
        auto* c_info = LookupShape(value_shapes, inputs[2]);
        // Broadcast C to output shape using strides (matching ORT reference).
        auto c_sizes = out_sizes;
        std::vector<uint32_t> c_strides(c_sizes.size(), 0u);
        // Compute packed strides for the original C shape, then set stride=0
        // for dimensions that need broadcasting (size 1 in C, >1 in output).
        auto orig_c = PadToMinDims(c_info->sizes, c_sizes.size());
        auto packed = ComputePackedStrides(orig_c);
        for (size_t d = 0; d < c_sizes.size(); ++d)
            c_strides[d] = (orig_c[d] == 1 && c_sizes[d] > 1) ? 0 : packed[d];
        auto c_tensor = MakeTensorInfoWithStrides(c_sizes, c_strides, a_info->data_type, c_info->total_bytes);
        c_tensor.sizes = c_sizes;
        c_tensor.strides = c_strides;
        result.input_tensors  = { a_tensor, b_tensor, c_tensor };
        result.input_buffer_descs = { a_tensor.ToBufferDesc(), b_tensor.ToBufferDesc(), c_tensor.ToBufferDesc() };
        result.input_tensor_descs.resize(3);
    } else {
        result.input_tensors  = { a_tensor, b_tensor };
        result.input_buffer_descs = { a_tensor.ToBufferDesc(), b_tensor.ToBufferDesc() };
        result.input_tensor_descs.resize(2);
    }
    result.output_tensors = { out_tensor };
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_GEMM, &storage->desc };

    bool local_has_c = has_c;
    result.fixup = [storage, local_has_c](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.ATensor     = &self.input_tensor_descs[0];
        storage->desc.BTensor     = &self.input_tensor_descs[1];
        storage->desc.CTensor     = local_has_c ? &self.input_tensor_descs[2] : nullptr;
        storage->desc.OutputTensor= &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// Tile — DML_OPERATOR_TILE
// ONNX input[0] = data, input[1] = repeats (1D int64 shape tensor, NOT wired to DML).
// Only input[0] is a DML input. Repeats are read from the initializer at translation time.
// ---------------------------------------------------------------------------
static std::optional<TranslatedOp> TranslateTile(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>& initializers) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 2 || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    // Read repeats from initializer (input[1] is a CPU shape tensor).
    std::vector<uint32_t> repeats;
    {
        auto it = initializers.find(inputs[1]);
        if (it == initializers.end() || !it->second) return std::nullopt;
        OrtTensorTypeAndShapeInfo* tsi = nullptr;
        ort_api.GetTensorTypeAndShape(const_cast<OrtValue*>(it->second), &tsi);
        if (!tsi) return std::nullopt;
        size_t count = 0;
        ort_api.GetTensorShapeElementCount(tsi, &count);
        ort_api.ReleaseTensorTypeAndShapeInfo(tsi);
        void* data = nullptr;
        OrtStatus* st = ort_api.GetTensorMutableData(const_cast<OrtValue*>(it->second), &data);
        if (st || !data) { if (st) ort_api.ReleaseStatus(st); return std::nullopt; }
        auto* vals = static_cast<const int64_t*>(data);
        for (size_t i = 0; i < count; ++i)
            repeats.push_back(static_cast<uint32_t>(std::max<int64_t>(vals[i], 1)));
    }

    // Compute output shape: input_size[i] * repeats[i].
    // Use pre-seeded output shape if available.
    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    std::vector<uint32_t> out_sizes;
    if (out_edge) {
        out_sizes = out_edge->sizes;
    } else {
        // Repeats correspond to original rank; pad input sizes match padded rank.
        // Pad repeats to match padded sizes.
        auto padded_in = in_info->sizes;
        std::vector<uint32_t> padded_repeats;
        if (repeats.size() < padded_in.size()) {
            padded_repeats.assign(padded_in.size() - repeats.size(), 1u);
            padded_repeats.insert(padded_repeats.end(), repeats.begin(), repeats.end());
        } else {
            padded_repeats = repeats;
        }
        out_sizes.resize(padded_in.size());
        for (size_t i = 0; i < padded_in.size(); ++i)
            out_sizes[i] = padded_in[i] * padded_repeats[i];
    }

    auto in_tensor  = MakeTensorInfo(in_info->sizes, in_info->data_type);
    auto out_tensor = MakeTensorInfo(out_sizes, in_info->data_type);

    // Pad repeats to match DML tensor DimensionCount (min 4D).
    size_t dml_dim_count = in_tensor.sizes.size();
    std::vector<uint32_t> dml_repeats;
    if (repeats.size() < dml_dim_count) {
        dml_repeats.assign(dml_dim_count - repeats.size(), 1u);
        dml_repeats.insert(dml_repeats.end(), repeats.begin(), repeats.end());
    } else {
        dml_repeats = repeats;
    }

    struct TileStorage {
        DML_TILE_OPERATOR_DESC desc{};
        std::vector<uint32_t> repeats;
    };
    auto storage = std::make_shared<TileStorage>();
    storage->repeats = std::move(dml_repeats);
    storage->desc.RepeatsCount = static_cast<UINT>(storage->repeats.size());
    storage->desc.Repeats = storage->repeats.data();

    TranslatedOp result;
    result.input_tensors  = { in_tensor };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs  = { in_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_TILE, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor  = &self.input_tensor_descs[0];
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
        storage->desc.RepeatsCount = static_cast<UINT>(storage->repeats.size());
        storage->desc.Repeats      = storage->repeats.data();
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// ConstantOfShape — DML_OPERATOR_FILL_VALUE_CONSTANT
// ONNX input[0] = shape tensor (CPU, not wired to DML). No DML inputs.
// Output filled with scalar from "value" tensor attribute (default 0).
// ---------------------------------------------------------------------------
static std::optional<TranslatedOp> TranslateConstantOfShape(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto outputs = GetOutputNames(ort_api, node);
    if (outputs.empty()) return std::nullopt;

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    if (!out_edge) return std::nullopt;

    auto out_tensor = MakeTensorInfo(out_edge->sizes, out_edge->data_type);

    DML_SCALAR_UNION fill_value{};
    DML_TENSOR_DATA_TYPE value_dtype = out_edge->data_type;

    // Read the "value" tensor attribute (single-element tensor).
    OrtNodeAdapter adapter(node, ort_api);
    if (adapter.HasAttribute("value")) {
        auto* attr = adapter.GetAttribute("value");
        if (attr) {
            try {
                auto& proto = attr->GetAttrProto();
                if (proto.type() == ONNX_NAMESPACE::AttributeProto::TENSOR && proto.has_t()) {
                    auto& tp = proto.t();
                    const auto& raw = tp.raw_data();
                    size_t copy_size = std::min(raw.size(), sizeof(fill_value.Bytes));
                    if (copy_size > 0) {
                        std::memcpy(fill_value.Bytes, raw.data(), copy_size);
                    }
                }
            } catch (...) {
                // If tensor attribute parsing fails, use default 0.
            }
        }
    }

    struct FillConstStorage {
        DML_FILL_VALUE_CONSTANT_OPERATOR_DESC desc{};
    };
    auto storage = std::make_shared<FillConstStorage>();
    storage->desc.ValueDataType = value_dtype;
    storage->desc.Value = fill_value;

    TranslatedOp result;
    result.input_tensors  = {};  // No DML inputs
    result.output_tensors = { out_tensor };
    result.input_buffer_descs  = {};
    result.input_tensor_descs  = {};
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_FILL_VALUE_CONSTANT, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// Range — DML_OPERATOR_FILL_VALUE_SEQUENCE
// ONNX inputs: [0]=start, [1]=limit, [2]=delta — all scalar CPU tensors.
// No DML inputs. All values read from initializers at translation time.
// Output is 1D: ceil((limit - start) / delta) elements.
// ---------------------------------------------------------------------------
static std::optional<TranslatedOp> TranslateRange(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>& initializers) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 3 || outputs.empty()) return std::nullopt;

    // Read scalar initializer values.
    auto read_scalar_bytes = [&](size_t idx, void* out, size_t max_bytes,
                                  ONNXTensorElementDataType& out_dtype) -> bool {
        if (idx >= inputs.size() || inputs[idx].empty()) return false;
        auto it = initializers.find(inputs[idx]);
        if (it == initializers.end() || !it->second) return false;
        OrtTensorTypeAndShapeInfo* tsi = nullptr;
        ort_api.GetTensorTypeAndShape(const_cast<OrtValue*>(it->second), &tsi);
        if (!tsi) return false;
        size_t count = 0;
        ort_api.GetTensorShapeElementCount(tsi, &count);
        ONNXTensorElementDataType dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        ort_api.GetTensorElementType(tsi, &dt);
        ort_api.ReleaseTensorTypeAndShapeInfo(tsi);
        if (count != 1) return false;
        void* data = nullptr;
        OrtStatus* st = ort_api.GetTensorMutableData(const_cast<OrtValue*>(it->second), &data);
        if (st || !data) { if (st) ort_api.ReleaseStatus(st); return false; }
        out_dtype = dt;
        size_t elem_size = DmlDataTypeSize(OnnxDtypeToDml(dt));
        if (elem_size == 0) return false;
        std::memcpy(out, data, std::min(elem_size, max_bytes));
        return true;
    };

    DML_SCALAR_UNION start_val{}, limit_val{}, delta_val{};
    ONNXTensorElementDataType start_dt{}, limit_dt{}, delta_dt{};
    if (!read_scalar_bytes(0, &start_val, sizeof(start_val), start_dt)) return std::nullopt;
    if (!read_scalar_bytes(1, &limit_val, sizeof(limit_val), limit_dt)) return std::nullopt;
    if (!read_scalar_bytes(2, &delta_val, sizeof(delta_val), delta_dt)) return std::nullopt;

    DML_TENSOR_DATA_TYPE dml_dtype = OnnxDtypeToDml(start_dt);
    if (dml_dtype == DML_TENSOR_DATA_TYPE_UNKNOWN) return std::nullopt;

    // Compute output element count: max(ceil((limit - start) / delta), 0).
    uint32_t output_count = 0;
    bool is_float = (start_dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                     start_dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE ||
                     start_dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

    if (is_float) {
        double s, l, d;
        if (start_dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            s = static_cast<double>(start_val.Float32);
            l = static_cast<double>(limit_val.Float32);
            d = static_cast<double>(delta_val.Float32);
        } else if (start_dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
            s = start_val.Float64;
            l = limit_val.Float64;
            d = delta_val.Float64;
        } else {
            // float16 — use pre-seeded output shape instead
            s = 0; l = 0; d = 1;
        }
        output_count = static_cast<uint32_t>(std::max(std::ceil((l - s) / d), 0.0));
    } else {
        int64_t s, l, d;
        if (start_dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            s = static_cast<int64_t>(start_val.Int32);
            l = static_cast<int64_t>(limit_val.Int32);
            d = static_cast<int64_t>(delta_val.Int32);
        } else if (start_dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            s = start_val.Int64;
            l = limit_val.Int64;
            d = delta_val.Int64;
        } else if (start_dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16) {
            s = static_cast<int64_t>(start_val.Int16);
            l = static_cast<int64_t>(limit_val.Int16);
            d = static_cast<int64_t>(delta_val.Int16);
        } else {
            return std::nullopt;
        }
        if (d == 0) return std::nullopt;
        int64_t range = l - s;
        output_count = static_cast<uint32_t>(std::max((range / d) + (range % d != 0 ? 1 : 0), int64_t(0)));
    }

    // Prefer pre-seeded output shape.
    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    std::vector<uint32_t> out_sizes;
    if (out_edge) {
        out_sizes = out_edge->sizes;
    } else {
        if (output_count == 0) return std::nullopt;
        out_sizes = { output_count };
    }

    auto out_tensor = MakeTensorInfo(out_sizes, dml_dtype);

    struct RangeStorage {
        DML_FILL_VALUE_SEQUENCE_OPERATOR_DESC desc{};
    };
    auto storage = std::make_shared<RangeStorage>();
    storage->desc.ValueDataType = dml_dtype;
    std::memcpy(&storage->desc.ValueStart, &start_val, sizeof(start_val));
    std::memcpy(&storage->desc.ValueDelta, &delta_val, sizeof(delta_val));

    TranslatedOp result;
    result.input_tensors  = {};  // No DML inputs
    result.output_tensors = { out_tensor };
    result.input_buffer_descs  = {};
    result.input_tensor_descs  = {};
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_FILL_VALUE_SEQUENCE, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

static std::optional<TranslatedOp> TranslateClip(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>& initializers) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    DML_TENSOR_DATA_TYPE dml_dtype = in_info->data_type;

    // Initialize min/max to type-appropriate extremes.
    // Following ORT: float types use -/+DBL_MAX clamped, int types use -/+INT64_MAX.
    DML_SCALAR_UNION min_val{}, max_val{};
    switch (dml_dtype) {
    case DML_TENSOR_DATA_TYPE_FLOAT16:
        min_val.UInt16 = 0xFBFF;  // -65504 (fp16 min finite)
        max_val.UInt16 = 0x7BFF;  //  65504 (fp16 max finite)
        break;
    case DML_TENSOR_DATA_TYPE_FLOAT64:
        min_val.Float64 = -std::numeric_limits<double>::max();
        max_val.Float64 =  std::numeric_limits<double>::max();
        break;
    case DML_TENSOR_DATA_TYPE_INT8:
        min_val.Int8 = INT8_MIN;  max_val.Int8 = INT8_MAX; break;
    case DML_TENSOR_DATA_TYPE_UINT8:
        min_val.UInt8 = 0;        max_val.UInt8 = UINT8_MAX; break;
    case DML_TENSOR_DATA_TYPE_INT16:
        min_val.Int16 = INT16_MIN; max_val.Int16 = INT16_MAX; break;
    case DML_TENSOR_DATA_TYPE_UINT16:
        min_val.UInt16 = 0;        max_val.UInt16 = UINT16_MAX; break;
    case DML_TENSOR_DATA_TYPE_INT32:
        min_val.Int32 = INT32_MIN; max_val.Int32 = INT32_MAX; break;
    case DML_TENSOR_DATA_TYPE_UINT32:
        min_val.UInt32 = 0;        max_val.UInt32 = UINT32_MAX; break;
    case DML_TENSOR_DATA_TYPE_INT64:
        min_val.Int64 = INT64_MIN; max_val.Int64 = INT64_MAX; break;
    case DML_TENSOR_DATA_TYPE_UINT64:
        min_val.UInt64 = 0;        max_val.UInt64 = UINT64_MAX; break;
    default: // FLOAT32
        min_val.Float32 = -std::numeric_limits<float>::max();
        max_val.Float32 =  std::numeric_limits<float>::max();
        break;
    }

    // Opset <11: min/max as float attributes (always float32).
    OrtNodeAdapter adapter(node, ort_api);
    if (adapter.HasAttribute("min")) {
        float v = adapter.GetAttributeFloat("min", 0.0f);
        if (dml_dtype == DML_TENSOR_DATA_TYPE_FLOAT32) min_val.Float32 = v;
        else if (dml_dtype == DML_TENSOR_DATA_TYPE_FLOAT64) min_val.Float64 = static_cast<double>(v);
    }
    if (adapter.HasAttribute("max")) {
        float v = adapter.GetAttributeFloat("max", 0.0f);
        if (dml_dtype == DML_TENSOR_DATA_TYPE_FLOAT32) max_val.Float32 = v;
        else if (dml_dtype == DML_TENSOR_DATA_TYPE_FLOAT64) max_val.Float64 = static_cast<double>(v);
    }

    // Opset 11+: inputs[1]=min, inputs[2]=max as scalar initializer tensors.
    // Read raw bytes matching the tensor's data type (works for all types).
    auto read_scalar_raw = [&](size_t idx, DML_SCALAR_UNION& out) {
        if (inputs.size() <= idx || inputs[idx].empty()) return;
        auto it = initializers.find(inputs[idx]);
        if (it == initializers.end() || !it->second) return;
        void* data = nullptr;
        ort_api.GetTensorMutableData(const_cast<OrtValue*>(it->second), &data);
        if (!data) return;
        size_t elem_size = DmlDataTypeSize(dml_dtype);
        if (elem_size > 0 && elem_size <= sizeof(out.Bytes))
            std::memcpy(out.Bytes, data, elem_size);
    };
    read_scalar_raw(1, min_val);
    read_scalar_raw(2, max_val);

    auto in_tensor  = MakeTensorInfo(in_info->sizes, dml_dtype);
    auto out_tensor = MakeTensorInfo(in_info->sizes, dml_dtype);

    struct ClipStorage { DML_ELEMENT_WISE_CLIP1_OPERATOR_DESC desc{}; };
    auto storage = std::make_shared<ClipStorage>();
    storage->desc.ScaleBias    = nullptr;
    storage->desc.MinMaxDataType = dml_dtype;
    storage->desc.Min          = min_val;
    storage->desc.Max          = max_val;

    TranslatedOp result;
    result.input_tensors  = { in_tensor };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs  = { in_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_ELEMENT_WISE_CLIP1, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor  = &self.input_tensor_descs[0];
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

static std::optional<TranslatedOp> TranslateQuantizeLinear(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 2 || outputs.empty()) return std::nullopt;

    auto* x_info     = LookupShape(value_shapes, inputs[0]);
    auto* scale_info = LookupShape(value_shapes, inputs[1]);
    if (!x_info || !scale_info) return std::nullopt;

    bool has_zp = (inputs.size() > 2 && !inputs[2].empty() && value_shapes.count(inputs[2]));

    // Output shape = input shape. Output dtype = zero_point dtype or UINT8.
    DML_TENSOR_DATA_TYPE out_dtype = DML_TENSOR_DATA_TYPE_UINT8;
    if (has_zp) {
        auto* zp_dtype_info = LookupShape(value_shapes, inputs[2]);
        if (zp_dtype_info) out_dtype = zp_dtype_info->data_type;
    }

    auto x_tensor     = MakeTensorInfo(x_info->sizes, x_info->data_type);
    auto scale_tensor = MakeTensorInfo(scale_info->sizes, scale_info->data_type);
    auto out_tensor   = MakeTensorInfo(x_info->sizes, out_dtype);

    // DML_QUANTIZE_OPERATOR_DESC uses a QuantizationTensors array: [scale] or [scale, zero_point]
    struct QLStorage {
        DML_QUANTIZE_OPERATOR_DESC desc{};
        std::vector<DML_TENSOR_DESC> quant_tensor_descs;
    };
    auto storage = std::make_shared<QLStorage>();
    storage->desc.QuantizationType = has_zp
        ? DML_QUANTIZATION_TYPE_SCALE_ZERO_POINT
        : DML_QUANTIZATION_TYPE_SCALE;

    TranslatedOp result;
    if (has_zp) {
        auto* zp_info = LookupShape(value_shapes, inputs[2]);
        auto zp_tensor = MakeTensorInfo(zp_info->sizes, zp_info->data_type);
        result.input_tensors  = { x_tensor, scale_tensor, zp_tensor };
        result.input_buffer_descs = { x_tensor.ToBufferDesc(), scale_tensor.ToBufferDesc(), zp_tensor.ToBufferDesc() };
        result.input_tensor_descs.resize(3);
    } else {
        result.input_tensors  = { x_tensor, scale_tensor };
        result.input_buffer_descs = { x_tensor.ToBufferDesc(), scale_tensor.ToBufferDesc() };
        result.input_tensor_descs.resize(2);
    }
    result.output_tensors = { out_tensor };
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_QUANTIZE, &storage->desc };

    bool local_has_zp = has_zp;
    result.fixup = [storage, local_has_zp](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        // Rebuild QuantizationTensors array from input slots 1 (scale) and optionally 2 (zp).
        storage->quant_tensor_descs.clear();
        storage->quant_tensor_descs.push_back(self.input_tensor_descs[1]);
        if (local_has_zp) storage->quant_tensor_descs.push_back(self.input_tensor_descs[2]);
        storage->desc.InputTensor           = &self.input_tensor_descs[0];
        storage->desc.QuantizationTensorCount = static_cast<UINT>(storage->quant_tensor_descs.size());
        storage->desc.QuantizationTensors   = storage->quant_tensor_descs.data();
        storage->desc.OutputTensor          = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

static std::optional<TranslatedOp> TranslateDequantizeLinear(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 2 || outputs.empty()) return std::nullopt;

    auto* x_info     = LookupShape(value_shapes, inputs[0]);
    auto* scale_info = LookupShape(value_shapes, inputs[1]);
    if (!x_info || !scale_info) return std::nullopt;

    bool has_zp = (inputs.size() > 2 && !inputs[2].empty() && value_shapes.count(inputs[2]));

    // Output shape = input shape. Output dtype = scale dtype (float32 or float16).
    auto x_tensor     = MakeTensorInfo(x_info->sizes, x_info->data_type);
    auto scale_tensor = MakeTensorInfo(scale_info->sizes, scale_info->data_type);
    auto out_tensor   = MakeTensorInfo(x_info->sizes, scale_info->data_type);

    struct DQLStorage {
        DML_DEQUANTIZE_OPERATOR_DESC desc{};
        std::vector<DML_TENSOR_DESC> quant_tensor_descs;
    };
    auto storage = std::make_shared<DQLStorage>();
    storage->desc.QuantizationType = has_zp
        ? DML_QUANTIZATION_TYPE_SCALE_ZERO_POINT
        : DML_QUANTIZATION_TYPE_SCALE;

    TranslatedOp result;
    if (has_zp) {
        auto* zp_info = LookupShape(value_shapes, inputs[2]);
        auto zp_tensor = MakeTensorInfo(zp_info->sizes, zp_info->data_type);
        result.input_tensors  = { x_tensor, scale_tensor, zp_tensor };
        result.input_buffer_descs = { x_tensor.ToBufferDesc(), scale_tensor.ToBufferDesc(), zp_tensor.ToBufferDesc() };
        result.input_tensor_descs.resize(3);
    } else {
        result.input_tensors  = { x_tensor, scale_tensor };
        result.input_buffer_descs = { x_tensor.ToBufferDesc(), scale_tensor.ToBufferDesc() };
        result.input_tensor_descs.resize(2);
    }
    result.output_tensors = { out_tensor };
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_DEQUANTIZE, &storage->desc };

    bool local_has_zp = has_zp;
    result.fixup = [storage, local_has_zp](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->quant_tensor_descs.clear();
        storage->quant_tensor_descs.push_back(self.input_tensor_descs[1]);
        if (local_has_zp) storage->quant_tensor_descs.push_back(self.input_tensor_descs[2]);
        storage->desc.InputTensor             = &self.input_tensor_descs[0];
        storage->desc.QuantizationTensorCount = static_cast<UINT>(storage->quant_tensor_descs.size());
        storage->desc.QuantizationTensors     = storage->quant_tensor_descs.data();
        storage->desc.OutputTensor            = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

// Gelu → DML_OPERATOR_ACTIVATION_GELU
static std::optional<TranslatedOp> TranslateGelu(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;
    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;
    auto t_info = MakeTensorInfo(in_info->sizes, in_info->data_type);
    auto storage = std::make_shared<DML_ACTIVATION_GELU_OPERATOR_DESC>();
    TranslatedOp result;
    result.input_tensors = { t_info };
    result.output_tensors = { t_info };
    result.input_buffer_descs  = { t_info.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { t_info.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_ACTIVATION_GELU, storage.get() };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->InputTensor  = &self.input_tensor_descs[0];
        storage->OutputTensor = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// QuickGelu → x * sigmoid(alpha * x), decomposed via SubNode graph.
// ORT ref: DmlOperatorQuickGelu.cpp
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateQuickGelu(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    float alpha = adapter.GetAttributeFloat("alpha", 1.0f);

    auto t_info = MakeTensorInfo(in_info->sizes, in_info->data_type);

    // Primary: Identity (x → x) — passes input through so sub_nodes can
    // reference the original x via {-1, 0}.
    auto id_storage = std::make_shared<DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC>();

    TranslatedOp result;
    result.input_tensors  = { t_info };
    result.output_tensors = { t_info };
    result.input_buffer_descs  = { t_info.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { t_info.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = id_storage;
    result.op_desc = { DML_OPERATOR_ELEMENT_WISE_IDENTITY, id_storage.get() };
    result.fixup = [id_storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        id_storage->InputTensor  = &self.input_tensor_descs[0];
        id_storage->OutputTensor = &self.output_tensor_descs[0];
    };
    result.FixupPointers();

    if (alpha != 1.0f) {
        // Sub 0: Identity + ScaleBias{alpha, 0} → alpha*x
        struct AlphaStorage {
            DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC desc{};
            DML_SCALE_BIAS scale_bias;
        };
        auto alpha_store = std::make_shared<AlphaStorage>();
        alpha_store->scale_bias = { alpha, 0.0f };

        SubNode alpha_node;
        alpha_node.input_tensors  = { t_info };
        alpha_node.output_tensors = { t_info };
        alpha_node.input_buffer_descs  = { t_info.ToBufferDesc() };
        alpha_node.input_tensor_descs.resize(1);
        alpha_node.output_buffer_descs = { t_info.ToBufferDesc() };
        alpha_node.output_tensor_descs.resize(1);
        alpha_node.desc_storage = alpha_store;
        alpha_node.op_desc = { DML_OPERATOR_ELEMENT_WISE_IDENTITY, &alpha_store->desc };
        alpha_node.input_from = { {-1, 0} };
        alpha_node.fixup = [alpha_store](SubNode& self) {
            RebuildSubNodePointers(self);
            alpha_store->desc.InputTensor  = &self.input_tensor_descs[0];
            alpha_store->desc.OutputTensor = &self.output_tensor_descs[0];
            alpha_store->desc.ScaleBias    = &alpha_store->scale_bias;
        };
        alpha_node.FixupPointers();

        // Sub 1: Sigmoid(alpha*x)
        auto sig_store = std::make_shared<DML_ACTIVATION_SIGMOID_OPERATOR_DESC>();
        SubNode sig_node;
        sig_node.input_tensors  = { t_info };
        sig_node.output_tensors = { t_info };
        sig_node.input_buffer_descs  = { t_info.ToBufferDesc() };
        sig_node.input_tensor_descs.resize(1);
        sig_node.output_buffer_descs = { t_info.ToBufferDesc() };
        sig_node.output_tensor_descs.resize(1);
        sig_node.desc_storage = sig_store;
        sig_node.op_desc = { DML_OPERATOR_ACTIVATION_SIGMOID, sig_store.get() };
        sig_node.input_from = { {0, 0} };
        sig_node.fixup = [sig_store](SubNode& self) {
            RebuildSubNodePointers(self);
            sig_store->InputTensor  = &self.input_tensor_descs[0];
            sig_store->OutputTensor = &self.output_tensor_descs[0];
        };
        sig_node.FixupPointers();

        // Sub 2: Multiply(x, sigmoid(alpha*x))
        auto mul_store = std::make_shared<DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC>();
        SubNode mul_node;
        mul_node.input_tensors  = { t_info, t_info };
        mul_node.output_tensors = { t_info };
        mul_node.input_buffer_descs  = { t_info.ToBufferDesc(), t_info.ToBufferDesc() };
        mul_node.input_tensor_descs.resize(2);
        mul_node.output_buffer_descs = { t_info.ToBufferDesc() };
        mul_node.output_tensor_descs.resize(1);
        mul_node.desc_storage = mul_store;
        mul_node.op_desc = { DML_OPERATOR_ELEMENT_WISE_MULTIPLY, mul_store.get() };
        mul_node.input_from = { {-1, 0}, {1, 0} };
        mul_node.fixup = [mul_store](SubNode& self) {
            RebuildSubNodePointers(self);
            mul_store->ATensor     = &self.input_tensor_descs[0];
            mul_store->BTensor     = &self.input_tensor_descs[1];
            mul_store->OutputTensor = &self.output_tensor_descs[0];
        };
        mul_node.FixupPointers();

        result.sub_nodes = { std::move(alpha_node), std::move(sig_node), std::move(mul_node) };
    } else {
        // Sub 0: Sigmoid(x)
        auto sig_store = std::make_shared<DML_ACTIVATION_SIGMOID_OPERATOR_DESC>();
        SubNode sig_node;
        sig_node.input_tensors  = { t_info };
        sig_node.output_tensors = { t_info };
        sig_node.input_buffer_descs  = { t_info.ToBufferDesc() };
        sig_node.input_tensor_descs.resize(1);
        sig_node.output_buffer_descs = { t_info.ToBufferDesc() };
        sig_node.output_tensor_descs.resize(1);
        sig_node.desc_storage = sig_store;
        sig_node.op_desc = { DML_OPERATOR_ACTIVATION_SIGMOID, sig_store.get() };
        sig_node.input_from = { {-1, 0} };
        sig_node.fixup = [sig_store](SubNode& self) {
            RebuildSubNodePointers(self);
            sig_store->InputTensor  = &self.input_tensor_descs[0];
            sig_store->OutputTensor = &self.output_tensor_descs[0];
        };
        sig_node.FixupPointers();

        // Sub 1: Multiply(x, sigmoid(x))
        auto mul_store = std::make_shared<DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC>();
        SubNode mul_node;
        mul_node.input_tensors  = { t_info, t_info };
        mul_node.output_tensors = { t_info };
        mul_node.input_buffer_descs  = { t_info.ToBufferDesc(), t_info.ToBufferDesc() };
        mul_node.input_tensor_descs.resize(2);
        mul_node.output_buffer_descs = { t_info.ToBufferDesc() };
        mul_node.output_tensor_descs.resize(1);
        mul_node.desc_storage = mul_store;
        mul_node.op_desc = { DML_OPERATOR_ELEMENT_WISE_MULTIPLY, mul_store.get() };
        mul_node.input_from = { {-1, 0}, {0, 0} };
        mul_node.fixup = [mul_store](SubNode& self) {
            RebuildSubNodePointers(self);
            mul_store->ATensor     = &self.input_tensor_descs[0];
            mul_store->BTensor     = &self.input_tensor_descs[1];
            mul_store->OutputTensor = &self.output_tensor_descs[0];
        };
        mul_node.FixupPointers();

        result.sub_nodes = { std::move(sig_node), std::move(mul_node) };
    }

    return result;
}

// ---------------------------------------------------------------------------
// BiasSplitGelu → Add(input,bias) → Split → Gelu(half) → Multiply
// ORT ref: DmlOperatorBiasSplitGelu.cpp
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateBiasSplitGelu(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 2 || outputs.empty()) return std::nullopt;

    auto* x_info    = LookupShape(value_shapes, inputs[0]);
    auto* bias_info = LookupShape(value_shapes, inputs[1]);
    if (!x_info || !bias_info) return std::nullopt;

    auto bc = BroadcastShapes(*x_info, *bias_info);
    if (!bc) return std::nullopt;

    auto add_out = MakeTensorInfo(bc->output_sizes, x_info->data_type);

    // Output is half the last dim.
    auto half_sizes = add_out.sizes;
    UINT split_axis = static_cast<UINT>(half_sizes.size() - 1);
    if (half_sizes[split_axis] % 2 != 0) return std::nullopt;
    half_sizes[split_axis] /= 2;

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    auto half_info = out_edge ? MakeTensorInfo(out_edge->sizes, x_info->data_type)
                              : MakeTensorInfo(half_sizes, x_info->data_type);

    // Primary: Add(x, bias)
    auto add_storage = std::make_shared<DML_ELEMENT_WISE_ADD_OPERATOR_DESC>();

    TranslatedOp result;
    result.input_tensors  = { bc->a, bc->b };
    result.output_tensors = { half_info };
    result.input_buffer_descs  = { bc->a.ToBufferDesc(), bc->b.ToBufferDesc() };
    result.input_tensor_descs.resize(2);
    result.output_buffer_descs = { add_out.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = add_storage;
    result.op_desc = { DML_OPERATOR_ELEMENT_WISE_ADD, add_storage.get() };
    result.fixup = [add_storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        add_storage->ATensor     = &self.input_tensor_descs[0];
        add_storage->BTensor     = &self.input_tensor_descs[1];
        add_storage->OutputTensor = &self.output_tensor_descs[0];
    };
    result.FixupPointers();

    // Sub 0: Split (2 outputs on last axis)
    struct SplitStorage {
        DML_SPLIT_OPERATOR_DESC desc{};
    };
    auto split_store = std::make_shared<SplitStorage>();
    split_store->desc.Axis = split_axis;
    split_store->desc.OutputCount = 2;

    SubNode split_node;
    split_node.input_tensors  = { add_out };
    split_node.output_tensors = { half_info, half_info };
    split_node.input_buffer_descs  = { add_out.ToBufferDesc() };
    split_node.input_tensor_descs.resize(1);
    split_node.output_buffer_descs = { half_info.ToBufferDesc(), half_info.ToBufferDesc() };
    split_node.output_tensor_descs.resize(2);
    split_node.desc_storage = split_store;
    split_node.op_desc = { DML_OPERATOR_SPLIT, &split_store->desc };
    split_node.input_from = { {-1, 0} };
    split_node.fixup = [split_store](SubNode& self) {
        RebuildSubNodePointers(self);
        split_store->desc.InputTensor   = &self.input_tensor_descs[0];
        split_store->desc.OutputTensors = self.output_tensor_descs.data();
        split_store->desc.OutputCount   = static_cast<UINT>(self.output_tensor_descs.size());
    };
    split_node.FixupPointers();

    // Sub 1: Gelu (second split half)
    auto gelu_store = std::make_shared<DML_ACTIVATION_GELU_OPERATOR_DESC>();
    SubNode gelu_node;
    gelu_node.input_tensors  = { half_info };
    gelu_node.output_tensors = { half_info };
    gelu_node.input_buffer_descs  = { half_info.ToBufferDesc() };
    gelu_node.input_tensor_descs.resize(1);
    gelu_node.output_buffer_descs = { half_info.ToBufferDesc() };
    gelu_node.output_tensor_descs.resize(1);
    gelu_node.desc_storage = gelu_store;
    gelu_node.op_desc = { DML_OPERATOR_ACTIVATION_GELU, gelu_store.get() };
    gelu_node.input_from = { {0, 1} };
    gelu_node.fixup = [gelu_store](SubNode& self) {
        RebuildSubNodePointers(self);
        gelu_store->InputTensor  = &self.input_tensor_descs[0];
        gelu_store->OutputTensor = &self.output_tensor_descs[0];
    };
    gelu_node.FixupPointers();

    // Sub 2: Multiply(first_half, gelu'd_second_half)
    auto mul_store = std::make_shared<DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC>();
    SubNode mul_node;
    mul_node.input_tensors  = { half_info, half_info };
    mul_node.output_tensors = { half_info };
    mul_node.input_buffer_descs  = { half_info.ToBufferDesc(), half_info.ToBufferDesc() };
    mul_node.input_tensor_descs.resize(2);
    mul_node.output_buffer_descs = { half_info.ToBufferDesc() };
    mul_node.output_tensor_descs.resize(1);
    mul_node.desc_storage = mul_store;
    mul_node.op_desc = { DML_OPERATOR_ELEMENT_WISE_MULTIPLY, mul_store.get() };
    mul_node.input_from = { {0, 0}, {1, 0} };
    mul_node.fixup = [mul_store](SubNode& self) {
        RebuildSubNodePointers(self);
        mul_store->ATensor     = &self.input_tensor_descs[0];
        mul_store->BTensor     = &self.input_tensor_descs[1];
        mul_store->OutputTensor = &self.output_tensor_descs[0];
    };
    mul_node.FixupPointers();

    result.sub_nodes = { std::move(split_node), std::move(gelu_node), std::move(mul_node) };
    return result;
}

// ---------------------------------------------------------------------------
// SkipLayerNormalization → Add(input,skip) + optional Add(bias) + MVN2
// ORT ref: DmlOperatorSkipLayerNormalization.cpp
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateSkipLayerNorm(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&,
    bool simplified) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    // inputs: input(0), skip(1), gamma(2), beta(3?), bias(4?)
    if (inputs.size() < 3 || outputs.empty()) return std::nullopt;

    auto* in_info    = LookupShape(value_shapes, inputs[0]);
    auto* skip_info  = LookupShape(value_shapes, inputs[1]);
    auto* gamma_info = LookupShape(value_shapes, inputs[2]);
    if (!in_info || !skip_info || !gamma_info) return std::nullopt;

    bool has_beta = inputs.size() > 3 && !inputs[3].empty() && value_shapes.count(inputs[3]);
    bool has_bias = inputs.size() > 4 && !inputs[4].empty() && value_shapes.count(inputs[4]);

    OrtNodeAdapter adapter(node, ort_api);
    float epsilon = adapter.GetAttributeFloat("epsilon", 1e-5f);

    // Reshape to ORT layout: [batchSize, sequenceLength, hiddenSize, 1].
    // value_shapes pre-pads to 4D with leading 1s, but ORT places hidden at axis 2.
    auto orig_sizes = in_info->sizes;
    size_t orig_rank = in_info->original_rank ? in_info->original_rank : orig_sizes.size();
    uint32_t hidden_size = orig_sizes.back();

    // Compute B, S from original ONNX rank.
    uint32_t batch_size, seq_len;
    if (orig_rank == 3) {
        // Padded shape: [1, B, S, H] — extract from original_rank-relative positions.
        size_t pad_off = orig_sizes.size() - orig_rank;
        batch_size = orig_sizes[pad_off];
        seq_len    = orig_sizes[pad_off + 1];
    } else {
        // 2D: [B*S, H] or [B, H]
        size_t pad_off = orig_sizes.size() - orig_rank;
        batch_size = orig_sizes[pad_off];
        seq_len    = 1;
    }

    // Build 4D tensors in ORT layout [B, S, H, 1].
    std::vector<uint32_t> tensor_shape = { batch_size, seq_len, hidden_size, 1 };
    std::vector<uint32_t> vector_shape = { 1, 1, hidden_size, 1 };

    auto in_tensor   = MakeTensorInfo(tensor_shape, in_info->data_type);
    auto skip_tensor = MakeTensorInfo(tensor_shape, skip_info->data_type);

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    auto out_tensor = out_edge ? MakeTensorInfo(out_edge->sizes, in_info->data_type)
                               : MakeTensorInfo(tensor_shape, in_info->data_type);

    // gamma/beta/bias: [hidden] → [1, 1, hidden, 1] with stride-based broadcast.
    std::vector<uint32_t> gamma_sizes = { hidden_size };
    auto gamma_tensor = MakeTensorInfoAtAxis(gamma_sizes, gamma_info->data_type, 2, 4);

    // Build input_tensors: [input, skip, gamma, beta?, bias?]
    TranslatedOp result;
    result.input_tensors = { in_tensor, skip_tensor, gamma_tensor };
    result.input_buffer_descs = { in_tensor.ToBufferDesc(), skip_tensor.ToBufferDesc(), gamma_tensor.ToBufferDesc() };

    DmlTensorInfo beta_tensor{};
    if (has_beta) {
        auto* beta_info = LookupShape(value_shapes, inputs[3]);
        if (!beta_info) return std::nullopt;
        beta_tensor = MakeTensorInfoAtAxis(gamma_sizes, beta_info->data_type, 2, 4);
        result.input_tensors.push_back(beta_tensor);
        result.input_buffer_descs.push_back(beta_tensor.ToBufferDesc());
    }

    DmlTensorInfo bias_tensor{};
    if (has_bias) {
        auto* bias_info_ptr = LookupShape(value_shapes, inputs[4]);
        if (!bias_info_ptr) return std::nullopt;
        // Broadcast bias [hidden] → [B, S, H, 1] with strides {0, 0, 1, 0} matching ORT.
        std::vector<uint32_t> bias_strides = { 0, 0, 1, 0 };
        uint64_t bias_bytes = ComputeAlignedTotalBytes(gamma_sizes, bias_info_ptr->data_type);
        bias_tensor = MakeTensorInfoWithStrides(tensor_shape, bias_strides,
            bias_info_ptr->data_type, bias_bytes);
        result.input_tensors.push_back(bias_tensor);
        result.input_buffer_descs.push_back(bias_tensor.ToBufferDesc());
    }

    result.input_tensor_descs.resize(result.input_tensors.size());
    result.output_tensors = { out_tensor };
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.primary_input_count = 2;

    // Primary: Add(input, skip)
    auto add_storage = std::make_shared<DML_ELEMENT_WISE_ADD_OPERATOR_DESC>();
    result.desc_storage = add_storage;
    result.op_desc = { DML_OPERATOR_ELEMENT_WISE_ADD, add_storage.get() };
    result.fixup = [add_storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        add_storage->ATensor     = &self.input_tensor_descs[0];
        add_storage->BTensor     = &self.input_tensor_descs[1];
        add_storage->OutputTensor = &self.output_tensor_descs[0];
    };
    result.FixupPointers();

    // Build sub_nodes depending on bias presence.
    auto add_result = MakeTensorInfo(tensor_shape, in_info->data_type);

    if (has_bias) {
        // Sub 0: Add(input+skip, bias)
        auto bias_add_store = std::make_shared<DML_ELEMENT_WISE_ADD_OPERATOR_DESC>();
        SubNode bias_add_node;
        bias_add_node.input_tensors  = { add_result, bias_tensor };
        bias_add_node.output_tensors = { add_result };
        bias_add_node.input_buffer_descs  = { add_result.ToBufferDesc(), bias_tensor.ToBufferDesc() };
        bias_add_node.input_tensor_descs.resize(2);
        bias_add_node.output_buffer_descs = { add_result.ToBufferDesc() };
        bias_add_node.output_tensor_descs.resize(1);
        bias_add_node.desc_storage = bias_add_store;
        bias_add_node.op_desc = { DML_OPERATOR_ELEMENT_WISE_ADD, bias_add_store.get() };
        bias_add_node.input_from = { {-1, 0} };
        bias_add_node.graph_inputs = { {4, 1} };
        bias_add_node.fixup = [bias_add_store](SubNode& self) {
            RebuildSubNodePointers(self);
            bias_add_store->ATensor     = &self.input_tensor_descs[0];
            bias_add_store->BTensor     = &self.input_tensor_descs[1];
            bias_add_store->OutputTensor = &self.output_tensor_descs[0];
        };
        bias_add_node.FixupPointers();
        result.sub_nodes.push_back(std::move(bias_add_node));
    }

    // MVN2 sub_node
    struct MVN2Storage {
        DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_DESC desc{};
        std::vector<uint32_t> axes;
    };
    auto mvn_store = std::make_shared<MVN2Storage>();
    // Normalize over axes {2, 3} matching ORT's [B, S, H, 1] layout.
    mvn_store->axes = { 2, 3 };
    mvn_store->desc.AxisCount = static_cast<UINT>(mvn_store->axes.size());
    mvn_store->desc.Axes = mvn_store->axes.data();
    mvn_store->desc.UseMean = simplified ? FALSE : TRUE;
    mvn_store->desc.UseVariance = TRUE;
    mvn_store->desc.Epsilon = epsilon;
    mvn_store->desc.FusedActivation = nullptr;

    size_t mvn_input_count = 2 + (has_beta ? 1 : 0);
    SubNode mvn_node;
    mvn_node.input_tensors  = { add_result, gamma_tensor };
    mvn_node.input_buffer_descs  = { add_result.ToBufferDesc(), gamma_tensor.ToBufferDesc() };
    if (has_beta) {
        mvn_node.input_tensors.push_back(beta_tensor);
        mvn_node.input_buffer_descs.push_back(beta_tensor.ToBufferDesc());
    }
    mvn_node.input_tensor_descs.resize(mvn_input_count);
    auto mvn_out = MakeTensorInfo(tensor_shape, in_info->data_type);
    mvn_node.output_tensors = { mvn_out };
    mvn_node.output_buffer_descs = { mvn_out.ToBufferDesc() };
    mvn_node.output_tensor_descs.resize(1);
    mvn_node.desc_storage = mvn_store;
    mvn_node.op_desc = { DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION2, &mvn_store->desc };
    // Input 0: from bias_add sub_node or primary
    mvn_node.input_from = { {has_bias ? 0 : -1, 0} };
    // gamma (input 1) and optionally beta (input 2) from ONNX graph inputs
    mvn_node.graph_inputs = { {2, 1} };
    if (has_beta) mvn_node.graph_inputs.push_back({3, 2});

    bool local_has_beta = has_beta;
    mvn_node.fixup = [mvn_store, local_has_beta](SubNode& self) {
        RebuildSubNodePointers(self);
        mvn_store->desc.InputTensor  = &self.input_tensor_descs[0];
        mvn_store->desc.ScaleTensor  = &self.input_tensor_descs[1];
        mvn_store->desc.BiasTensor   = local_has_beta ? &self.input_tensor_descs[2] : nullptr;
        mvn_store->desc.OutputTensor = &self.output_tensor_descs[0];
    };
    mvn_node.FixupPointers();
    result.sub_nodes.push_back(std::move(mvn_node));

    return result;
}

// ---------------------------------------------------------------------------
// DynamicQuantizeLinear → DML_OPERATOR_DYNAMIC_QUANTIZE_LINEAR
// ORT ref: DmlOperatorDynamicQuantizeLinear.cpp
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateDynamicQuantizeLinear(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.size() < 3) return std::nullopt;

    auto* x_info = LookupShape(value_shapes, inputs[0]);
    if (!x_info) return std::nullopt;

    auto x_tensor     = MakeTensorInfo(x_info->sizes, x_info->data_type);
    auto y_tensor     = MakeTensorInfo(x_info->sizes, DML_TENSOR_DATA_TYPE_UINT8);
    auto scale_tensor = MakeTensorInfo({1, 1, 1, 1}, DML_TENSOR_DATA_TYPE_FLOAT32);
    auto zp_tensor    = MakeTensorInfo({1, 1, 1, 1}, DML_TENSOR_DATA_TYPE_UINT8);

    auto storage = std::make_shared<DML_DYNAMIC_QUANTIZE_LINEAR_OPERATOR_DESC>();

    TranslatedOp result;
    result.input_tensors  = { x_tensor };
    result.output_tensors = { y_tensor, scale_tensor, zp_tensor };
    result.input_buffer_descs  = { x_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { y_tensor.ToBufferDesc(), scale_tensor.ToBufferDesc(), zp_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(3);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_DYNAMIC_QUANTIZE_LINEAR, storage.get() };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->InputTensor           = &self.input_tensor_descs[0];
        storage->OutputTensor          = &self.output_tensor_descs[0];
        storage->OutputScaleTensor     = &self.output_tensor_descs[1];
        storage->OutputZeroPointTensor = &self.output_tensor_descs[2];
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// QLinearMatMul → DML_OPERATOR_QUANTIZED_LINEAR_MATRIX_MULTIPLY
// ORT ref: DmlOperatorQLinearMatMul.cpp
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateQLinearMatMul(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    // A(0), A_scale(1), A_zp(2), B(3), B_scale(4), B_zp(5), Y_scale(6), Y_zp(7)
    if (inputs.size() < 8 || outputs.empty()) return std::nullopt;

    auto* a_info       = LookupShape(value_shapes, inputs[0]);
    auto* a_scale_info = LookupShape(value_shapes, inputs[1]);
    auto* b_info       = LookupShape(value_shapes, inputs[3]);
    auto* b_scale_info = LookupShape(value_shapes, inputs[4]);
    auto* y_scale_info = LookupShape(value_shapes, inputs[6]);
    if (!a_info || !a_scale_info || !b_info || !b_scale_info || !y_scale_info)
        return std::nullopt;

    bool has_a_zp = !inputs[2].empty() && value_shapes.count(inputs[2]);
    bool has_b_zp = !inputs[5].empty() && value_shapes.count(inputs[5]);
    bool has_y_zp = !inputs[7].empty() && value_shapes.count(inputs[7]);

    // MatMul shape handling for A and B.
    auto a_sizes = a_info->sizes;
    auto b_sizes = b_info->sizes;
    auto a_strides = ComputePackedStrides(a_sizes);
    auto b_strides = ComputePackedStrides(b_sizes);

    bool a_was_1d = (a_sizes.size() == 1);
    bool b_was_1d = (b_sizes.size() == 1);
    if (a_was_1d) { a_sizes.insert(a_sizes.begin(), 1u); a_strides.insert(a_strides.begin(), 0u); }
    if (b_was_1d) { b_sizes.push_back(1u); b_strides.push_back(0u); }
    if (a_sizes.size() < 2 || b_sizes.size() < 2) return std::nullopt;

    size_t max_rank = std::max(a_sizes.size(), b_sizes.size());
    while (a_sizes.size() < max_rank) { a_sizes.insert(a_sizes.begin(), 1u); a_strides.insert(a_strides.begin(), 0u); }
    while (b_sizes.size() < max_rank) { b_sizes.insert(b_sizes.begin(), 1u); b_strides.insert(b_strides.begin(), 0u); }

    for (size_t d = 0; d + 2 < max_rank; ++d) {
        uint32_t bd = std::max(a_sizes[d], b_sizes[d]);
        if (a_sizes[d] == 1) { a_sizes[d] = bd; a_strides[d] = 0; }
        if (b_sizes[d] == 1) { b_sizes[d] = bd; b_strides[d] = 0; }
    }

    while (a_sizes.size() < 4) { a_sizes.insert(a_sizes.begin(), 1u); a_strides.insert(a_strides.begin(), 0u); }
    while (b_sizes.size() < 4) { b_sizes.insert(b_sizes.begin(), 1u); b_strides.insert(b_strides.begin(), 0u); }

    uint32_t M = a_sizes[2], N = b_sizes[3];
    std::vector<uint32_t> y_sizes = { a_sizes[0], a_sizes[1], M, N };

    uint64_t a_bytes = ComputeAlignedTotalBytes(a_info->sizes, a_info->data_type);
    uint64_t b_bytes = ComputeAlignedTotalBytes(b_info->sizes, b_info->data_type);

    auto a_tensor = MakeTensorInfoWithStrides(a_sizes, a_strides, a_info->data_type, a_bytes);
    auto b_tensor = MakeTensorInfoWithStrides(b_sizes, b_strides, b_info->data_type, b_bytes);

    auto a_scale_tensor = MakeTensorInfo(a_scale_info->sizes, a_scale_info->data_type);
    auto b_scale_tensor = MakeTensorInfo(b_scale_info->sizes, b_scale_info->data_type);
    auto y_scale_tensor = MakeTensorInfo(y_scale_info->sizes, y_scale_info->data_type);

    DML_TENSOR_DATA_TYPE y_dtype = has_y_zp
        ? LookupShape(value_shapes, inputs[7])->data_type
        : DML_TENSOR_DATA_TYPE_UINT8;
    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    auto y_tensor = out_edge ? MakeTensorInfo(out_edge->sizes, y_dtype)
                             : MakeTensorInfo(y_sizes, y_dtype);

    struct QLMMStorage {
        DML_QUANTIZED_LINEAR_MATRIX_MULTIPLY_OPERATOR_DESC desc{};
    };
    auto storage = std::make_shared<QLMMStorage>();

    TranslatedOp result;
    // Build input_tensors in DML schema order, skipping absent optional inputs.
    // input_name_reorder maps each DML slot to the ONNX input index.
    auto push_input = [&](const DmlTensorInfo& t, size_t onnx_idx) {
        result.input_tensors.push_back(t);
        result.input_buffer_descs.push_back(t.ToBufferDesc());
        result.input_name_reorder.push_back(onnx_idx);
    };

    push_input(a_tensor, 0);        // A
    push_input(a_scale_tensor, 1);  // A_scale
    DmlTensorInfo a_zp_tensor{};
    if (has_a_zp) {
        a_zp_tensor = MakeTensorInfo(LookupShape(value_shapes, inputs[2])->sizes,
                                     LookupShape(value_shapes, inputs[2])->data_type);
        push_input(a_zp_tensor, 2); // A_zp
    }
    push_input(b_tensor, 3);        // B
    push_input(b_scale_tensor, 4);  // B_scale
    DmlTensorInfo b_zp_tensor{};
    if (has_b_zp) {
        b_zp_tensor = MakeTensorInfo(LookupShape(value_shapes, inputs[5])->sizes,
                                     LookupShape(value_shapes, inputs[5])->data_type);
        push_input(b_zp_tensor, 5); // B_zp
    }
    push_input(y_scale_tensor, 6);  // Y_scale
    DmlTensorInfo y_zp_tensor{};
    if (has_y_zp) {
        y_zp_tensor = MakeTensorInfo(LookupShape(value_shapes, inputs[7])->sizes,
                                     LookupShape(value_shapes, inputs[7])->data_type);
        push_input(y_zp_tensor, 7); // Y_zp
    }

    result.input_tensor_descs.resize(result.input_tensors.size());
    result.output_tensors = { y_tensor };
    result.output_buffer_descs = { y_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_QUANTIZED_LINEAR_MATRIX_MULTIPLY, &storage->desc };

    bool la_zp = has_a_zp, lb_zp = has_b_zp, ly_zp = has_y_zp;
    result.fixup = [storage, la_zp, lb_zp, ly_zp](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        size_t idx = 0;
        storage->desc.ATensor      = &self.input_tensor_descs[idx++];
        storage->desc.AScaleTensor = &self.input_tensor_descs[idx++];
        storage->desc.AZeroPointTensor = la_zp ? &self.input_tensor_descs[idx++] : nullptr;
        storage->desc.BTensor      = &self.input_tensor_descs[idx++];
        storage->desc.BScaleTensor = &self.input_tensor_descs[idx++];
        storage->desc.BZeroPointTensor = lb_zp ? &self.input_tensor_descs[idx++] : nullptr;
        storage->desc.OutputScaleTensor = &self.input_tensor_descs[idx++];
        storage->desc.OutputZeroPointTensor = ly_zp ? &self.input_tensor_descs[idx++] : nullptr;
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// MatMulInteger → DML_OPERATOR_MATRIX_MULTIPLY_INTEGER
// ORT ref: DmlOperatorMatMulInteger.cpp
// DML schema order: [ATensor, AZeroPointTensor, BTensor, BZeroPointTensor]
// ONNX input order: [A, B, a_zero_point, b_zero_point]
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateMatMulInteger(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 2 || outputs.empty()) return std::nullopt;

    auto* a_info = LookupShape(value_shapes, inputs[0]);
    auto* b_info = LookupShape(value_shapes, inputs[1]);
    if (!a_info || !b_info) return std::nullopt;

    bool has_a_zp = inputs.size() > 2 && !inputs[2].empty() && value_shapes.count(inputs[2]);
    bool has_b_zp = inputs.size() > 3 && !inputs[3].empty() && value_shapes.count(inputs[3]);

    // MatMul shape handling.
    auto a_sizes = a_info->sizes;
    auto b_sizes = b_info->sizes;
    auto a_strides = ComputePackedStrides(a_sizes);
    auto b_strides = ComputePackedStrides(b_sizes);

    bool a_was_1d = (a_sizes.size() == 1);
    bool b_was_1d = (b_sizes.size() == 1);
    if (a_was_1d) { a_sizes.insert(a_sizes.begin(), 1u); a_strides.insert(a_strides.begin(), 0u); }
    if (b_was_1d) { b_sizes.push_back(1u); b_strides.push_back(0u); }
    if (a_sizes.size() < 2 || b_sizes.size() < 2) return std::nullopt;

    size_t max_rank = std::max(a_sizes.size(), b_sizes.size());
    while (a_sizes.size() < max_rank) { a_sizes.insert(a_sizes.begin(), 1u); a_strides.insert(a_strides.begin(), 0u); }
    while (b_sizes.size() < max_rank) { b_sizes.insert(b_sizes.begin(), 1u); b_strides.insert(b_strides.begin(), 0u); }

    for (size_t d = 0; d + 2 < max_rank; ++d) {
        uint32_t bd = std::max(a_sizes[d], b_sizes[d]);
        if (a_sizes[d] == 1) { a_sizes[d] = bd; a_strides[d] = 0; }
        if (b_sizes[d] == 1) { b_sizes[d] = bd; b_strides[d] = 0; }
    }

    while (a_sizes.size() < 4) { a_sizes.insert(a_sizes.begin(), 1u); a_strides.insert(a_strides.begin(), 0u); }
    while (b_sizes.size() < 4) { b_sizes.insert(b_sizes.begin(), 1u); b_strides.insert(b_strides.begin(), 0u); }

    uint32_t M = a_sizes[2], N = b_sizes[3];
    std::vector<uint32_t> y_sizes = { a_sizes[0], a_sizes[1], M, N };

    uint64_t a_bytes = ComputeAlignedTotalBytes(a_info->sizes, a_info->data_type);
    uint64_t b_bytes = ComputeAlignedTotalBytes(b_info->sizes, b_info->data_type);

    auto a_tensor = MakeTensorInfoWithStrides(a_sizes, a_strides, a_info->data_type, a_bytes);
    auto b_tensor = MakeTensorInfoWithStrides(b_sizes, b_strides, b_info->data_type, b_bytes);

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    auto y_tensor = out_edge ? MakeTensorInfo(out_edge->sizes, DML_TENSOR_DATA_TYPE_INT32)
                             : MakeTensorInfo(y_sizes, DML_TENSOR_DATA_TYPE_INT32);

    // Build input_tensors in DML schema order: [A, AZP, B, BZP].
    // input_name_reorder maps these slots to ONNX input indices.
    struct MMIStorage {
        DML_MATRIX_MULTIPLY_INTEGER_OPERATOR_DESC desc{};
    };
    auto storage = std::make_shared<MMIStorage>();

    TranslatedOp result;
    result.input_tensors = { a_tensor };
    result.input_buffer_descs = { a_tensor.ToBufferDesc() };
    result.input_name_reorder = { 0 };  // DML slot 0 → ONNX input 0 (A)

    DmlTensorInfo a_zp_tensor{}, b_zp_tensor{};
    if (has_a_zp) {
        a_zp_tensor = MakeTensorInfo(LookupShape(value_shapes, inputs[2])->sizes,
                                     LookupShape(value_shapes, inputs[2])->data_type);
        result.input_tensors.push_back(a_zp_tensor);
        result.input_buffer_descs.push_back(a_zp_tensor.ToBufferDesc());
        result.input_name_reorder.push_back(2);  // DML slot 1 → ONNX input 2 (a_zp)
    }

    result.input_tensors.push_back(b_tensor);
    result.input_buffer_descs.push_back(b_tensor.ToBufferDesc());
    result.input_name_reorder.push_back(1);  // DML slot 2 (or 1) → ONNX input 1 (B)

    if (has_b_zp) {
        b_zp_tensor = MakeTensorInfo(LookupShape(value_shapes, inputs[3])->sizes,
                                     LookupShape(value_shapes, inputs[3])->data_type);
        result.input_tensors.push_back(b_zp_tensor);
        result.input_buffer_descs.push_back(b_zp_tensor.ToBufferDesc());
        result.input_name_reorder.push_back(3);  // DML slot 3 → ONNX input 3 (b_zp)
    }

    result.input_tensor_descs.resize(result.input_tensors.size());
    result.output_tensors = { y_tensor };
    result.output_buffer_descs = { y_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_MATRIX_MULTIPLY_INTEGER, &storage->desc };

    bool la_zp = has_a_zp, lb_zp = has_b_zp;
    result.fixup = [storage, la_zp, lb_zp](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        size_t idx = 0;
        storage->desc.ATensor = &self.input_tensor_descs[idx++];
        storage->desc.AZeroPointTensor = la_zp ? &self.input_tensor_descs[idx++] : nullptr;
        storage->desc.BTensor = &self.input_tensor_descs[idx++];
        storage->desc.BZeroPointTensor = lb_zp ? &self.input_tensor_descs[idx++] : nullptr;
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// MatMulIntegerToFloat → DML_OPERATOR_MATRIX_MULTIPLY_INTEGER_TO_FLOAT
// ORT ref: DmlOperatorMatMulIntegerToFloat.cpp
// ONNX inputs: A(0), B(1), A_scale(2), B_scale(3), A_zp(4?), B_zp(5?), Bias(6?)
// DML schema:  A, A_scale, A_zp, B, B_scale, B_zp, Bias
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateMatMulIntegerToFloat(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 4 || outputs.empty()) return std::nullopt;

    auto* a_info = LookupShape(value_shapes, inputs[0]);
    auto* b_info = LookupShape(value_shapes, inputs[1]);
    auto* a_scale_info = LookupShape(value_shapes, inputs[2]);
    auto* b_scale_info = LookupShape(value_shapes, inputs[3]);
    if (!a_info || !b_info || !a_scale_info || !b_scale_info) return std::nullopt;

    bool has_a_zp = inputs.size() > 4 && !inputs[4].empty() && value_shapes.count(inputs[4]);
    bool has_b_zp = inputs.size() > 5 && !inputs[5].empty() && value_shapes.count(inputs[5]);
    bool has_bias = inputs.size() > 6 && !inputs[6].empty() && value_shapes.count(inputs[6]);

    auto a_sizes = a_info->sizes;
    auto b_sizes = b_info->sizes;
    auto a_strides = ComputePackedStrides(a_sizes);
    auto b_strides = ComputePackedStrides(b_sizes);

    if (a_sizes.size() == 1) { a_sizes.insert(a_sizes.begin(), 1u); a_strides.insert(a_strides.begin(), 0u); }
    if (b_sizes.size() == 1) { b_sizes.push_back(1u); b_strides.push_back(0u); }
    if (a_sizes.size() < 2 || b_sizes.size() < 2) return std::nullopt;

    size_t max_rank = std::max(a_sizes.size(), b_sizes.size());
    while (a_sizes.size() < max_rank) { a_sizes.insert(a_sizes.begin(), 1u); a_strides.insert(a_strides.begin(), 0u); }
    while (b_sizes.size() < max_rank) { b_sizes.insert(b_sizes.begin(), 1u); b_strides.insert(b_strides.begin(), 0u); }
    for (size_t d = 0; d + 2 < max_rank; ++d) {
        uint32_t bd = std::max(a_sizes[d], b_sizes[d]);
        if (a_sizes[d] == 1) { a_sizes[d] = bd; a_strides[d] = 0; }
        if (b_sizes[d] == 1) { b_sizes[d] = bd; b_strides[d] = 0; }
    }
    while (a_sizes.size() < 4) { a_sizes.insert(a_sizes.begin(), 1u); a_strides.insert(a_strides.begin(), 0u); }
    while (b_sizes.size() < 4) { b_sizes.insert(b_sizes.begin(), 1u); b_strides.insert(b_strides.begin(), 0u); }

    uint32_t M = a_sizes[2], N = b_sizes[3];
    std::vector<uint32_t> y_sizes = { a_sizes[0], a_sizes[1], M, N };

    auto a_tensor = MakeTensorInfoWithStrides(a_sizes, a_strides, a_info->data_type,
        ComputeAlignedTotalBytes(a_info->sizes, a_info->data_type));
    auto b_tensor = MakeTensorInfoWithStrides(b_sizes, b_strides, b_info->data_type,
        ComputeAlignedTotalBytes(b_info->sizes, b_info->data_type));

    auto a_scale_tensor = MakeTensorInfo(a_scale_info->sizes, a_scale_info->data_type);
    auto b_scale_tensor = MakeTensorInfo(b_scale_info->sizes, b_scale_info->data_type);

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    auto y_tensor = out_edge ? MakeTensorInfo(out_edge->sizes, DML_TENSOR_DATA_TYPE_FLOAT32)
                             : MakeTensorInfo(y_sizes, DML_TENSOR_DATA_TYPE_FLOAT32);

    struct Storage { DML_MATRIX_MULTIPLY_INTEGER_TO_FLOAT_OPERATOR_DESC desc{}; };
    auto storage = std::make_shared<Storage>();

    TranslatedOp result;
    auto push = [&](const DmlTensorInfo& t, size_t onnx_idx) {
        result.input_tensors.push_back(t);
        result.input_buffer_descs.push_back(t.ToBufferDesc());
        result.input_name_reorder.push_back(onnx_idx);
    };

    push(a_tensor, 0);
    push(a_scale_tensor, 2);
    DmlTensorInfo a_zp_t{};
    if (has_a_zp) { a_zp_t = MakeTensorInfo(LookupShape(value_shapes, inputs[4])->sizes, LookupShape(value_shapes, inputs[4])->data_type); push(a_zp_t, 4); }
    push(b_tensor, 1);
    push(b_scale_tensor, 3);
    DmlTensorInfo b_zp_t{};
    if (has_b_zp) { b_zp_t = MakeTensorInfo(LookupShape(value_shapes, inputs[5])->sizes, LookupShape(value_shapes, inputs[5])->data_type); push(b_zp_t, 5); }
    DmlTensorInfo bias_t{};
    if (has_bias) { bias_t = MakeTensorInfo(LookupShape(value_shapes, inputs[6])->sizes, LookupShape(value_shapes, inputs[6])->data_type); push(bias_t, 6); }

    result.input_tensor_descs.resize(result.input_tensors.size());
    result.output_tensors = { y_tensor };
    result.output_buffer_descs = { y_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_MATRIX_MULTIPLY_INTEGER_TO_FLOAT, &storage->desc };

    bool la = has_a_zp, lb = has_b_zp, lbias = has_bias;
    result.fixup = [storage, la, lb, lbias](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        size_t i = 0;
        storage->desc.ATensor      = &self.input_tensor_descs[i++];
        storage->desc.AScaleTensor = &self.input_tensor_descs[i++];
        storage->desc.AZeroPointTensor = la ? &self.input_tensor_descs[i++] : nullptr;
        storage->desc.BTensor      = &self.input_tensor_descs[i++];
        storage->desc.BScaleTensor = &self.input_tensor_descs[i++];
        storage->desc.BZeroPointTensor = lb ? &self.input_tensor_descs[i++] : nullptr;
        storage->desc.BiasTensor   = lbias ? &self.input_tensor_descs[i++] : nullptr;
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// OneHot → DML_OPERATOR_ONE_HOT
// ORT ref: DmlOperatorOneHot.cpp
// ONNX inputs: indices(0), depth(1, scalar initializer), values(2, [off,on])
// DML inputs: indices, values (depth consumed at translation time)
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateOneHot(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>& initializers) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 3 || outputs.empty()) return std::nullopt;

    auto* idx_info = LookupShape(value_shapes, inputs[0]);
    auto* val_info = LookupShape(value_shapes, inputs[2]);
    auto* out_info = LookupShape(value_shapes, outputs[0]);
    if (!idx_info || !val_info || !out_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    int64_t axis = adapter.GetAttributeInt("axis", -1);

    size_t idx_orig_rank = idx_info->original_rank ? idx_info->original_rank : idx_info->sizes.size();
    if (axis < 0) axis += static_cast<int64_t>(idx_orig_rank) + 1;

    auto idx_tensor = MakeTensorInfo(idx_info->sizes, idx_info->data_type);
    auto val_tensor = MakeTensorInfo(val_info->sizes, val_info->data_type);
    auto out_tensor = MakeTensorInfo(out_info->sizes, out_info->data_type);

    size_t out_padded_rank = out_info->sizes.size();
    size_t out_orig_rank = out_info->original_rank ? out_info->original_rank : out_padded_rank;
    size_t pad_offset = out_padded_rank - out_orig_rank;
    UINT dml_axis = static_cast<UINT>(axis + pad_offset);

    struct Storage { DML_ONE_HOT_OPERATOR_DESC desc{}; UINT axis_val; };
    auto storage = std::make_shared<Storage>();
    storage->axis_val = dml_axis;
    storage->desc.Axis = dml_axis;

    TranslatedOp result;
    result.input_tensors  = { idx_tensor, val_tensor };
    result.input_buffer_descs = { idx_tensor.ToBufferDesc(), val_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(2);
    result.input_name_reorder = { 0, 2 };
    result.output_tensors = { out_tensor };
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_ONE_HOT, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.IndicesTensor = &self.input_tensor_descs[0];
        storage->desc.ValuesTensor  = &self.input_tensor_descs[1];
        storage->desc.OutputTensor  = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// Trilu → DML_OPERATOR_DIAGONAL_MATRIX1
// ORT ref: DmlOperatorTrilu.cpp
// ONNX inputs: input(0), k(1, optional scalar initializer)
// DML inputs: input only (k consumed at translation time)
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateTrilu(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>& initializers) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    int64_t upper = adapter.GetAttributeInt("upper", 1);

    int32_t k = 0;
    if (inputs.size() > 1 && !inputs[1].empty()) {
        auto it = initializers.find(inputs[1]);
        if (it != initializers.end() && it->second) {
            void* data = nullptr;
            OrtStatus* st = ort_api.GetTensorMutableData(const_cast<OrtValue*>(it->second), &data);
            if (!st && data) k = static_cast<int32_t>(*static_cast<const int64_t*>(data));
            if (st) ort_api.ReleaseStatus(st);
        }
    }

    auto t_info = MakeTensorInfo(in_info->sizes, in_info->data_type);

    struct Storage {
        DML_DIAGONAL_MATRIX1_OPERATOR_DESC desc{};
        DML_SCALAR_UNION value{};
    };
    auto storage = std::make_shared<Storage>();
    storage->desc.DiagonalFillBegin = upper ? INT32_MIN : k + 1;
    storage->desc.DiagonalFillEnd   = upper ? k : INT32_MAX;
    storage->desc.ValueDataType = in_info->data_type;
    storage->value = {};  // zero-fill

    TranslatedOp result;
    result.input_tensors  = { t_info };
    result.output_tensors = { t_info };
    result.input_buffer_descs  = { t_info.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { t_info.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_DIAGONAL_MATRIX1, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor  = &self.input_tensor_descs[0];
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
        storage->desc.Value = storage->value;
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// QLinearAdd → DML_OPERATOR_ELEMENT_WISE_QUANTIZED_LINEAR_ADD
// ORT ref: DmlOperatorQLinearAdd.cpp
// ONNX inputs (8): A(0), A_scale(1), A_zp(2), B(3), B_scale(4), B_zp(5), C_scale(6), C_zp(7)
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateQLinearAdd(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 8 || outputs.empty()) return std::nullopt;

    auto* a_info = LookupShape(value_shapes, inputs[0]);
    auto* b_info = LookupShape(value_shapes, inputs[3]);
    if (!a_info || !b_info) return std::nullopt;

    auto bc = BroadcastShapes(*a_info, *b_info);
    if (!bc) return std::nullopt;

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    DML_TENSOR_DATA_TYPE out_dtype = a_info->data_type;
    auto out_tensor = out_edge ? MakeTensorInfo(out_edge->sizes, out_dtype)
                               : MakeTensorInfo(bc->output_sizes, out_dtype);

    struct Storage { DML_ELEMENT_WISE_QUANTIZED_LINEAR_ADD_OPERATOR_DESC desc{}; };
    auto storage = std::make_shared<Storage>();

    TranslatedOp result;
    auto push = [&](size_t idx) {
        auto* info = LookupShape(value_shapes, inputs[idx]);
        if (!info) return;
        auto t = MakeTensorInfo(info->sizes, info->data_type);
        result.input_tensors.push_back(t);
        result.input_buffer_descs.push_back(t.ToBufferDesc());
    };
    for (size_t i = 0; i < 8; ++i) push(i);
    if (result.input_tensors.size() < 8) return std::nullopt;

    result.input_tensors[0] = bc->a;
    result.input_buffer_descs[0] = bc->a.ToBufferDesc();
    result.input_tensors[3] = bc->b;
    result.input_buffer_descs[3] = bc->b.ToBufferDesc();

    result.input_tensor_descs.resize(8);
    result.output_tensors = { out_tensor };
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_ELEMENT_WISE_QUANTIZED_LINEAR_ADD, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.ATensor = &self.input_tensor_descs[0];
        storage->desc.AScaleTensor = &self.input_tensor_descs[1];
        storage->desc.AZeroPointTensor = &self.input_tensor_descs[2];
        storage->desc.BTensor = &self.input_tensor_descs[3];
        storage->desc.BScaleTensor = &self.input_tensor_descs[4];
        storage->desc.BZeroPointTensor = &self.input_tensor_descs[5];
        storage->desc.OutputScaleTensor = &self.input_tensor_descs[6];
        storage->desc.OutputZeroPointTensor = &self.input_tensor_descs[7];
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// LpNormalization → DML_OPERATOR_LP_NORMALIZATION
// ORT ref: DmlOperatorLpNormalization.cpp
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateLpNormalization(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    int64_t axis = adapter.GetAttributeInt("axis", -1);
    int64_t p    = adapter.GetAttributeInt("p", 2);

    size_t padded_rank = in_info->sizes.size();
    size_t orig_rank = in_info->original_rank ? in_info->original_rank : padded_rank;
    size_t pad_offset = padded_rank - orig_rank;
    if (axis < 0) axis += static_cast<int64_t>(orig_rank);
    axis += static_cast<int64_t>(pad_offset);

    auto t_info = MakeTensorInfo(in_info->sizes, in_info->data_type);

    struct Storage { DML_LP_NORMALIZATION_OPERATOR_DESC desc{}; };
    auto storage = std::make_shared<Storage>();
    storage->desc.Axis = static_cast<UINT>(axis);
    storage->desc.Epsilon = 1e-5f;
    storage->desc.P = static_cast<UINT>(p);

    TranslatedOp result;
    result.input_tensors  = { t_info };
    result.output_tensors = { t_info };
    result.input_buffer_descs  = { t_info.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { t_info.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_LP_NORMALIZATION, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor  = &self.input_tensor_descs[0];
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// EyeLike → DML_OPERATOR_DIAGONAL_MATRIX (output-only, no DML inputs)
// ORT ref: DmlOperatorEyeLike.cpp
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateEyeLike(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info  = LookupShape(value_shapes, inputs[0]);
    auto* out_info = LookupShape(value_shapes, outputs[0]);
    if (!in_info || !out_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    int64_t k = adapter.GetAttributeInt("k", 0);

    auto out_tensor = MakeTensorInfo(out_info->sizes, out_info->data_type);

    struct Storage { DML_DIAGONAL_MATRIX_OPERATOR_DESC desc{}; };
    auto storage = std::make_shared<Storage>();
    storage->desc.Offset = static_cast<INT>(k);
    storage->desc.Value  = 1.0f;

    TranslatedOp result;
    result.output_tensors = { out_tensor };
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_DIAGONAL_MATRIX, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// ReverseSequence → DML_OPERATOR_REVERSE_SUBSEQUENCES
// ORT ref: DmlOperatorReverseSequence.cpp
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateReverseSequence(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 2 || outputs.empty()) return std::nullopt;

    auto* in_info  = LookupShape(value_shapes, inputs[0]);
    auto* seq_info = LookupShape(value_shapes, inputs[1]);
    if (!in_info || !seq_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    int64_t time_axis  = adapter.GetAttributeInt("time_axis", 0);
    int64_t batch_axis = adapter.GetAttributeInt("batch_axis", 1);

    size_t padded_rank = in_info->sizes.size();
    size_t orig_rank = in_info->original_rank ? in_info->original_rank : padded_rank;
    size_t pad_offset = padded_rank - orig_rank;
    if (time_axis < 0) time_axis += static_cast<int64_t>(orig_rank);
    time_axis += static_cast<int64_t>(pad_offset);

    auto in_tensor  = MakeTensorInfo(in_info->sizes, in_info->data_type);
    auto seq_tensor = MakeTensorInfo(seq_info->sizes, seq_info->data_type);

    struct Storage { DML_REVERSE_SUBSEQUENCES_OPERATOR_DESC desc{}; };
    auto storage = std::make_shared<Storage>();
    storage->desc.Axis = static_cast<UINT>(time_axis);

    TranslatedOp result;
    result.input_tensors  = { in_tensor, seq_tensor };
    result.output_tensors = { in_tensor };
    result.input_buffer_descs  = { in_tensor.ToBufferDesc(), seq_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(2);
    result.output_buffer_descs = { in_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_REVERSE_SUBSEQUENCES, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor           = &self.input_tensor_descs[0];
        storage->desc.SequenceLengthsTensor = &self.input_tensor_descs[1];
        storage->desc.OutputTensor          = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// Crop → DML_OPERATOR_SLICE (legacy, deprecated)
// ORT ref: DmlOperatorCrop.cpp — uses border attr for offsets
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateCrop(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info  = LookupShape(value_shapes, inputs[0]);
    auto* out_info = LookupShape(value_shapes, outputs[0]);
    if (!in_info || !out_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    auto border = adapter.GetAttributeInts("border");
    if (border.size() < 4) return std::nullopt;

    auto in_tensor  = MakeTensorInfo(in_info->sizes, in_info->data_type);
    auto out_tensor = MakeTensorInfo(out_info->sizes, out_info->data_type);

    struct Storage {
        DML_SLICE_OPERATOR_DESC desc{};
        std::vector<uint32_t> offsets;
        std::vector<uint32_t> sizes;
        std::vector<uint32_t> strides;
    };
    auto storage = std::make_shared<Storage>();
    storage->offsets = { 0, 0, static_cast<uint32_t>(border[0]), static_cast<uint32_t>(border[1]) };
    storage->sizes   = out_info->sizes;
    storage->strides = { 1, 1, 1, 1 };
    storage->desc.DimensionCount = 4;

    TranslatedOp result;
    result.input_tensors  = { in_tensor };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs  = { in_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_SLICE, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor  = &self.input_tensor_descs[0];
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
        storage->desc.Offsets = storage->offsets.data();
        storage->desc.Sizes   = storage->sizes.data();
        storage->desc.Strides = storage->strides.data();
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// MaxRoiPool → DML_OPERATOR_ROI_POOLING
// ORT ref: DmlOperatorRoiPooling.cpp
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateMaxRoiPool(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 2 || outputs.empty()) return std::nullopt;

    auto* x_info   = LookupShape(value_shapes, inputs[0]);
    auto* roi_info = LookupShape(value_shapes, inputs[1]);
    auto* out_info = LookupShape(value_shapes, outputs[0]);
    if (!x_info || !roi_info || !out_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    float spatial_scale = adapter.GetAttributeFloat("spatial_scale", 1.0f);
    auto pooled_shape = adapter.GetAttributeInts("pooled_shape");
    if (pooled_shape.size() < 2) return std::nullopt;

    auto x_tensor   = MakeTensorInfo(x_info->sizes, x_info->data_type);
    auto roi_tensor = MakeTensorInfo(roi_info->sizes, roi_info->data_type);
    auto out_tensor = MakeTensorInfo(out_info->sizes, out_info->data_type);

    struct Storage {
        DML_ROI_POOLING_OPERATOR_DESC desc{};
        DML_SIZE_2D pooled_size;
    };
    auto storage = std::make_shared<Storage>();
    storage->pooled_size = { static_cast<UINT>(pooled_shape[0]), static_cast<UINT>(pooled_shape[1]) };
    storage->desc.SpatialScale = spatial_scale;
    storage->desc.PooledSize = storage->pooled_size;

    TranslatedOp result;
    result.input_tensors  = { x_tensor, roi_tensor };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs  = { x_tensor.ToBufferDesc(), roi_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(2);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_ROI_POOLING, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor  = &self.input_tensor_descs[0];
        storage->desc.ROITensor    = &self.input_tensor_descs[1];
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// MaxUnpool → DML_OPERATOR_MAX_UNPOOLING
// ORT ref: DmlOperatorMaxUnpool.cpp
// ONNX inputs: X(0), I(1), output_shape(2, optional — consumed at translation time)
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateMaxUnpool(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 2 || outputs.empty()) return std::nullopt;

    auto* x_info = LookupShape(value_shapes, inputs[0]);
    auto* i_info = LookupShape(value_shapes, inputs[1]);
    auto* out_info = LookupShape(value_shapes, outputs[0]);
    if (!x_info || !i_info || !out_info) return std::nullopt;

    auto x_tensor = MakeTensorInfo(x_info->sizes, x_info->data_type);
    auto i_tensor = MakeTensorInfo(i_info->sizes, i_info->data_type);
    auto out_tensor = MakeTensorInfo(out_info->sizes, out_info->data_type);

    struct Storage { DML_MAX_UNPOOLING_OPERATOR_DESC desc{}; };
    auto storage = std::make_shared<Storage>();

    TranslatedOp result;
    result.input_tensors  = { x_tensor, i_tensor };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs  = { x_tensor.ToBufferDesc(), i_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(2);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_MAX_UNPOOLING, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor   = &self.input_tensor_descs[0];
        storage->desc.IndicesTensor = &self.input_tensor_descs[1];
        storage->desc.OutputTensor  = &self.output_tensor_descs[0];
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// MatMulNBits → Dequantize + Gemm SubNode graph
// ORT ref: DmlOperatorMatMulNBits.cpp
// ONNX inputs: A(0), B(1), scales(2), zero_point(3?), g_idx(4?), bias(5?)
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateMatMulNBits(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 3 || outputs.empty()) return std::nullopt;

    // Reject unsupported optional inputs (g_idx and bias).
    bool has_g_idx = inputs.size() > 4 && !inputs[4].empty() && value_shapes.count(inputs[4]);
    bool has_bias  = inputs.size() > 5 && !inputs[5].empty() && value_shapes.count(inputs[5]);
    if (has_g_idx || has_bias) return std::nullopt;

    auto* a_info = LookupShape(value_shapes, inputs[0]);
    auto* b_info = LookupShape(value_shapes, inputs[1]);
    auto* scale_info = LookupShape(value_shapes, inputs[2]);
    if (!a_info || !b_info || !scale_info) return std::nullopt;

    bool has_zp = inputs.size() > 3 && !inputs[3].empty() && value_shapes.count(inputs[3]);

    OrtNodeAdapter adapter(node, ort_api);
    uint32_t bRowCount = static_cast<uint32_t>(adapter.GetAttributeInt("N", 0));
    uint32_t bColCount = static_cast<uint32_t>(adapter.GetAttributeInt("K", 0));
    int64_t bits = adapter.GetAttributeInt("bits", 0);
    int64_t block_size = adapter.GetAttributeInt("block_size", 0);

    if (bits != 4 && bits != 8) return std::nullopt;
    if (block_size <= 0 || (bColCount % block_size) != 0) return std::nullopt;

    DML_TENSOR_DATA_TYPE quantized_dtype = (bits == 4)
        ? DML_TENSOR_DATA_TYPE_UINT4 : DML_TENSOR_DATA_TYPE_UINT8;

    auto a_sizes = a_info->sizes;
    if (a_sizes.size() < 2) return std::nullopt;

    // B: real shape from N/K attrs. Batch dims are 1, broadcast to match A.
    std::vector<uint32_t> b_broadcasted(a_sizes.size(), 1u);
    b_broadcasted[b_broadcasted.size() - 2] = bRowCount;
    b_broadcasted[b_broadcasted.size() - 1] = bColCount;

    std::vector<uint32_t> b_shape(a_sizes.size(), 1u);
    b_shape[b_shape.size() - 2] = bRowCount;
    b_shape[b_shape.size() - 1] = bColCount;

    // Broadcast strides: stride=0 for batch dims that are 1.
    auto b_strides = ComputePackedStrides(b_shape);
    for (size_t d = 0; d + 2 < b_shape.size(); ++d) {
        if (b_shape[d] == 1 && a_sizes[d] > 1) {
            b_broadcasted[d] = a_sizes[d];
            b_strides[d] = 0;
        }
    }

    // B total_bytes from the original ONNX uint8-packed tensor.
    uint64_t b_total_bytes = b_info->total_bytes;
    if (b_total_bytes == 0)
        b_total_bytes = ComputeAlignedTotalBytes(b_info->sizes, b_info->data_type);

    auto b_tensor = MakeTensorInfoWithStrides(b_broadcasted, b_strides, quantized_dtype, b_total_bytes);

    // Scale: scaleElementCount / N columns per row.
    uint64_t scale_elem_count = 1;
    for (auto s : scale_info->sizes) scale_elem_count *= s;
    uint32_t scale_last_dim = static_cast<uint32_t>(scale_elem_count / bRowCount);

    std::vector<uint32_t> scale_shape(a_sizes.size(), 1u);
    scale_shape[scale_shape.size() - 2] = bRowCount;
    scale_shape[scale_shape.size() - 1] = scale_last_dim;

    std::vector<uint32_t> scale_broadcasted(b_broadcasted);
    scale_broadcasted.back() = scale_last_dim;

    auto scale_strides = ComputePackedStrides(scale_shape);
    for (size_t d = 0; d + 2 < scale_shape.size(); ++d) {
        if (scale_shape[d] == 1 && a_sizes[d] > 1) {
            scale_strides[d] = 0;
        }
    }

    uint64_t scale_bytes = ComputeAlignedTotalBytes(scale_info->sizes, scale_info->data_type);
    auto scale_tensor = MakeTensorInfoWithStrides(scale_broadcasted, scale_strides,
        a_info->data_type, scale_bytes);

    // A tensor.
    auto a_tensor = MakeTensorInfo(a_sizes, a_info->data_type);

    // Dequantized B intermediate.
    auto deq_out_tensor = MakeTensorInfo(b_broadcasted, a_info->data_type);

    // Output.
    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    std::vector<uint32_t> out_sizes;
    if (out_edge) {
        out_sizes = out_edge->sizes;
    } else {
        out_sizes = a_sizes;
        out_sizes.back() = bRowCount;
    }
    auto out_tensor = MakeTensorInfo(out_sizes, a_info->data_type);

    // Storage for both operator descs.
    struct MatMulNBitsStorage {
        DML_DEQUANTIZE_OPERATOR_DESC deq_desc{};
        std::vector<DML_TENSOR_DESC> quant_tensor_descs;
        DmlTensorInfo deq_out_info;
        DML_BUFFER_TENSOR_DESC deq_out_buf{};
        DML_TENSOR_DESC deq_out_td{};
        DML_GEMM_OPERATOR_DESC gemm_desc{};
    };
    auto storage = std::make_shared<MatMulNBitsStorage>();
    storage->deq_out_info = deq_out_tensor;
    storage->deq_out_buf = deq_out_tensor.ToBufferDesc();
    storage->deq_out_td = { DML_TENSOR_TYPE_BUFFER, &storage->deq_out_buf };

    // Build TranslatedOp: input_tensors in DML Dequantize order + A appended.
    TranslatedOp result;
    result.input_tensors = { b_tensor, scale_tensor };
    result.input_buffer_descs = { b_tensor.ToBufferDesc(), scale_tensor.ToBufferDesc() };
    result.input_name_reorder = { 1, 2 };  // DML slot 0→ONNX 1 (B), slot 1→ONNX 2 (scale)

    DmlTensorInfo zp_tensor{};
    if (has_zp) {
        auto* zp_info = LookupShape(value_shapes, inputs[3]);
        if (!zp_info) return std::nullopt;

        auto zp_strides = ComputePackedStrides(scale_shape);
        for (size_t d = 0; d + 2 < scale_shape.size(); ++d) {
            if (scale_shape[d] == 1 && a_sizes[d] > 1)
                zp_strides[d] = 0;
        }

        // Odd-dimension stride fix for UINT4 alignment.
        if (bits == 4 && scale_last_dim % 2 != 0) {
            zp_strides[zp_strides.size() - 2] = scale_last_dim + 1;
        }

        uint64_t zp_bytes = zp_info->total_bytes;
        if (zp_bytes == 0)
            zp_bytes = ComputeAlignedTotalBytes(zp_info->sizes, zp_info->data_type);

        zp_tensor = MakeTensorInfoWithStrides(scale_broadcasted, zp_strides, quantized_dtype, zp_bytes);
        result.input_tensors.push_back(zp_tensor);
        result.input_buffer_descs.push_back(zp_tensor.ToBufferDesc());
        result.input_name_reorder.push_back(3);  // DML slot 2→ONNX 3 (zp)
    }

    // A comes after the Dequantize inputs (used only by Gemm sub_node via graph_inputs).
    result.input_tensors.push_back(a_tensor);
    result.input_buffer_descs.push_back(a_tensor.ToBufferDesc());
    result.input_name_reorder.push_back(0);  // last slot→ONNX 0 (A)

    size_t deq_input_count = has_zp ? 3 : 2;
    result.primary_input_count = deq_input_count;

    result.input_tensor_descs.resize(result.input_tensors.size());
    result.output_tensors = { out_tensor };
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);

    // Primary: DML_DEQUANTIZE_OPERATOR_DESC
    storage->deq_desc.QuantizationType = has_zp
        ? DML_QUANTIZATION_TYPE_SCALE_ZERO_POINT
        : DML_QUANTIZATION_TYPE_SCALE;
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_DEQUANTIZE, &storage->deq_desc };

    bool local_has_zp = has_zp;
    result.fixup = [storage, local_has_zp](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        // Rebuild dequantized output tensor pointers.
        storage->deq_out_buf.Sizes = storage->deq_out_info.sizes.data();
        storage->deq_out_buf.Strides = storage->deq_out_info.strides.empty()
            ? nullptr : storage->deq_out_info.strides.data();
        storage->deq_out_td = { DML_TENSOR_TYPE_BUFFER, &storage->deq_out_buf };
        // Build quantization tensor desc array.
        storage->quant_tensor_descs.clear();
        storage->quant_tensor_descs.push_back(self.input_tensor_descs[1]);  // scale
        if (local_has_zp)
            storage->quant_tensor_descs.push_back(self.input_tensor_descs[2]);  // zp
        storage->deq_desc.InputTensor = &self.input_tensor_descs[0];  // B
        storage->deq_desc.QuantizationTensorCount = static_cast<UINT>(storage->quant_tensor_descs.size());
        storage->deq_desc.QuantizationTensors = storage->quant_tensor_descs.data();
        storage->deq_desc.OutputTensor = &storage->deq_out_td;
    };
    result.FixupPointers();

    // Sub_node 0: Gemm(A, dequantized_B).
    SubNode gemm_node;
    gemm_node.input_tensors = { a_tensor, deq_out_tensor };
    gemm_node.output_tensors = { out_tensor };
    gemm_node.input_buffer_descs = { a_tensor.ToBufferDesc(), deq_out_tensor.ToBufferDesc() };
    gemm_node.input_tensor_descs.resize(2);
    gemm_node.output_buffer_descs = { out_tensor.ToBufferDesc() };
    gemm_node.output_tensor_descs.resize(1);
    gemm_node.desc_storage = storage;
    gemm_node.op_desc = { DML_OPERATOR_GEMM, &storage->gemm_desc };
    // Slot 0 (ATensor): from graph_inputs (ONNX input A).
    // Slot 1 (BTensor): from primary output 0 (dequantized_B).
    gemm_node.input_from = { {-2, 0}, {-1, 0} };  // -2 = skip (slot 0 wired by graph_inputs)
    gemm_node.graph_inputs = { {0, 0} };  // ONNX input 0 (A) → Gemm slot 0

    gemm_node.fixup = [storage](SubNode& self) {
        RebuildSubNodePointers(self);
        storage->gemm_desc.ATensor = &self.input_tensor_descs[0];
        storage->gemm_desc.BTensor = &self.input_tensor_descs[1];
        storage->gemm_desc.CTensor = nullptr;
        storage->gemm_desc.OutputTensor = &self.output_tensor_descs[0];
        storage->gemm_desc.TransA = DML_MATRIX_TRANSFORM_NONE;
        storage->gemm_desc.TransB = DML_MATRIX_TRANSFORM_TRANSPOSE;
        storage->gemm_desc.Alpha = 1.0f;
        storage->gemm_desc.Beta = 0.0f;
        storage->gemm_desc.FusedActivation = nullptr;
    };
    gemm_node.FixupPointers();
    result.sub_nodes.push_back(std::move(gemm_node));

    return result;
}

// ---------------------------------------------------------------------------
// Attention → Gemm + Identity(transpose) + MHA SubNode graph
// ORT ref: DmlOperatorAttention.cpp
// Decomposition: Gemm(input × weights + bias) → transpose QKV → MHA
// ORT restricts: no past, no unidirectional, no do_rotary, no past_present_share_buffer.
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateAttention(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 2 || outputs.empty()) return std::nullopt;

    enum OnnxInput : size_t { inputIdx=0, weightsIdx=1, biasIdx=2,
                              maskIdx=3, pastIdx=4, attBiasIdx=5, pastSeqLenIdx=6 };

    auto valid = [&](size_t idx) {
        return idx < inputs.size() && !inputs[idx].empty() && value_shapes.count(inputs[idx]);
    };

    // ORT QueryAttention restrictions.
    if (valid(pastIdx)) return std::nullopt;
    if (valid(pastSeqLenIdx)) return std::nullopt;
    if (outputs.size() > 1 && !outputs[1].empty() && value_shapes.count(outputs[1]))
        return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    if (adapter.GetAttributeInt("unidirectional", 0) != 0) return std::nullopt;
    if (adapter.GetAttributeInt("do_rotary", 0) != 0) return std::nullopt;
    if (adapter.GetAttributeInt("past_present_share_buffer", 0) != 0) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[inputIdx]);
    auto* w_info  = LookupShape(value_shapes, inputs[weightsIdx]);
    if (!in_info || !w_info) return std::nullopt;

    bool hasBias = valid(biasIdx);
    bool hasMask = valid(maskIdx);
    bool hasAttBias = valid(attBiasIdx);

    uint32_t numHeads = static_cast<uint32_t>(adapter.GetAttributeInt("num_heads", 0));
    if (numHeads == 0) return std::nullopt;

    // Extract dimensions.  input: [B, S, inputHidden], weights: [inputHidden, 3*H].
    auto in_sizes = in_info->sizes;
    auto w_sizes  = w_info->sizes;
    uint32_t in_rank = in_info->original_rank ? in_info->original_rank
                     : static_cast<uint32_t>(in_sizes.size());
    size_t in_off = in_sizes.size() >= in_rank ? in_sizes.size() - in_rank : 0;
    uint32_t batchSize = in_sizes[in_off];
    uint32_t sequenceLength = in_sizes[in_off + 1];
    uint32_t inputHidden = in_sizes[in_off + 2];

    // qkv_hidden_sizes attribute.
    auto qkvHiddenSizes = adapter.GetAttributeInts("qkv_hidden_sizes");
    uint32_t hiddenSize, vHiddenSize;
    if (!qkvHiddenSizes.empty() && qkvHiddenSizes.size() == 3) {
        hiddenSize  = static_cast<uint32_t>(qkvHiddenSizes[0]);
        vHiddenSize = static_cast<uint32_t>(qkvHiddenSizes[2]);
    } else {
        uint32_t w_rank = w_info->original_rank ? w_info->original_rank
                        : static_cast<uint32_t>(w_sizes.size());
        size_t w_off = w_sizes.size() >= w_rank ? w_sizes.size() - w_rank : 0;
        uint32_t totalHidden = w_sizes[w_off + 1];
        hiddenSize  = totalHidden / 3;
        vHiddenSize = hiddenSize;
    }
    uint32_t headSize  = hiddenSize / numHeads;
    uint32_t vHeadSize = vHiddenSize / numHeads;
    uint32_t totalQkvHidden = hiddenSize + hiddenSize + vHiddenSize;
    bool hasSlicedValue = (hiddenSize != vHiddenSize);

    float scale = adapter.GetAttributeFloat("scale", 0.0f);
    if (scale == 0.0f) scale = 1.0f / std::sqrt(static_cast<float>(headSize));
    float maskFilterValue = adapter.GetAttributeFloat("mask_filter_value", -10000.0f);

    // Data types.
    DML_TENSOR_DATA_TYPE dataType = in_info->data_type;

    // --- Mask detection (reuse TranslateMultiHeadAttention pattern) ---
    DML_MULTIHEAD_ATTENTION_MASK_TYPE maskType = DML_MULTIHEAD_ATTENTION_MASK_TYPE_NONE;
    DmlTensorInfo mask_tensor{};
    bool hasMaxSequenceMask = false;
    if (hasMask) {
        auto* mask_info = LookupShape(value_shapes, inputs[maskIdx]);
        if (!mask_info) return std::nullopt;
        uint32_t mask_rank = mask_info->original_rank ? mask_info->original_rank
                           : static_cast<uint32_t>(mask_info->sizes.size());
        if (mask_rank == 1) {
            uint32_t mask_size = mask_info->sizes.back();
            uint32_t batchGroupCount = mask_size / batchSize;
            if (batchGroupCount == 1) {
                maskType = DML_MULTIHEAD_ATTENTION_MASK_TYPE_KEY_SEQUENCE_LENGTH;
                mask_tensor = MakeTensorInfo({1, batchSize}, mask_info->data_type);
            } else {
                maskType = DML_MULTIHEAD_ATTENTION_MASK_TYPE_KEY_SEQUENCE_END_START;
                mask_tensor = MakeTensorInfo({batchGroupCount, batchSize}, mask_info->data_type);
            }
        } else {
            maskType = DML_MULTIHEAD_ATTENTION_MASK_TYPE_BOOLEAN;
            size_t m_off = mask_info->sizes.size() >= mask_rank
                         ? mask_info->sizes.size() - mask_rank : 0;
            std::vector<uint32_t> actual_shape(4, 1u), desired_shape(4, 1u);
            if (mask_rank == 2) {
                actual_shape  = {batchSize, 1, 1, sequenceLength};
                desired_shape = {batchSize, numHeads, sequenceLength, sequenceLength};
            } else if (mask_rank == 3) {
                actual_shape  = {batchSize, 1, sequenceLength, sequenceLength};
                desired_shape = {batchSize, numHeads, sequenceLength, sequenceLength};
            } else {
                uint32_t maskSeqLen = mask_info->sizes[m_off + 2];
                if (maskSeqLen != sequenceLength) {
                    hasMaxSequenceMask = true;
                    actual_shape  = {batchSize, numHeads, maskSeqLen, maskSeqLen};
                    desired_shape = actual_shape;
                } else {
                    actual_shape  = {batchSize, numHeads, sequenceLength, sequenceLength};
                    desired_shape = actual_shape;
                }
            }
            auto strides = ComputePackedStrides(actual_shape);
            for (size_t d = 0; d < 4; ++d) {
                if (actual_shape[d] == 1 && desired_shape[d] > 1)
                    strides[d] = 0;
            }
            uint64_t mask_bytes = mask_info->total_bytes > 0
                ? mask_info->total_bytes
                : ComputeAlignedTotalBytes(actual_shape, mask_info->data_type);
            if (hasMaxSequenceMask) {
                mask_tensor = MakeTensorInfo(actual_shape, mask_info->data_type);
            } else {
                mask_tensor = MakeTensorInfoWithStrides(desired_shape, strides,
                    mask_info->data_type, mask_bytes);
            }
        }
    }

    // --- Storage struct ---
    struct AttentionStorage {
        DML_GEMM_OPERATOR_DESC gemm_desc{};
        DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identity_desc{};
        DML_MULTIHEAD_ATTENTION_OPERATOR_DESC mha_desc{};
        DML_SLICE1_OPERATOR_DESC qk_slice_desc{};
        DML_SLICE1_OPERATOR_DESC v_slice_desc{};
        DML_SLICE1_OPERATOR_DESC mask_slice_desc{};
        DmlTensorInfo gemm_out_info;
        DML_BUFFER_TENSOR_DESC gemm_out_buf{};
        DML_TENSOR_DESC gemm_out_td{};
        DmlTensorInfo transpose_in_info, transpose_out_info;
        DML_BUFFER_TENSOR_DESC transpose_in_buf{}, transpose_out_buf{};
        DML_TENSOR_DESC transpose_in_td{}, transpose_out_td{};
        DmlTensorInfo qk_sliced_info, v_sliced_info;
        DML_BUFFER_TENSOR_DESC qk_sliced_buf{}, v_sliced_buf{};
        DML_TENSOR_DESC qk_sliced_td{}, v_sliced_td{};
        DmlTensorInfo mask_sliced_info;
        DML_BUFFER_TENSOR_DESC mask_sliced_buf{};
        DML_TENSOR_DESC mask_sliced_td{};
        std::vector<uint32_t> qk_offsets, qk_sizes, v_offsets, v_sizes;
        std::vector<int32_t> qk_strides_i, v_strides_i;
        std::vector<uint32_t> mask_offsets, mask_sizes;
        std::vector<int32_t> mask_strides_i;
    };
    auto storage = std::make_shared<AttentionStorage>();

    // GEMM output: [B, S, totalQkvHidden]
    std::vector<uint32_t> gemm_out_shape = {batchSize, sequenceLength, totalQkvHidden};
    storage->gemm_out_info = MakeTensorInfo(gemm_out_shape, dataType);
    storage->gemm_out_buf = storage->gemm_out_info.ToBufferDesc();
    storage->gemm_out_td  = { DML_TENSOR_TYPE_BUFFER, &storage->gemm_out_buf };

    // Weight tensor: [inputHidden, totalQkvHidden] → broadcast [B, inputHidden, totalQkvHidden]
    std::vector<uint32_t> w_actual = {1, inputHidden, totalQkvHidden};
    std::vector<uint32_t> w_desired = {batchSize, inputHidden, totalQkvHidden};
    auto w_strides = ComputePackedStrides(w_actual);
    w_strides[0] = 0;
    uint64_t w_bytes = ComputeAlignedTotalBytes(w_actual, w_info->data_type);
    auto weight_tensor = MakeTensorInfoWithStrides(w_desired, w_strides, w_info->data_type, w_bytes);

    // Bias tensor: [totalQkvHidden] → broadcast [B, S, totalQkvHidden]
    DmlTensorInfo bias_tensor{};
    if (hasBias) {
        auto* bias_info = LookupShape(value_shapes, inputs[biasIdx]);
        if (!bias_info) return std::nullopt;
        std::vector<uint32_t> bias_desired = {batchSize, sequenceLength, totalQkvHidden};
        std::vector<uint32_t> bias_strides = {0, 0, 1};
        uint64_t bias_bytes = ComputeAlignedTotalBytes({totalQkvHidden}, bias_info->data_type);
        bias_tensor = MakeTensorInfoWithStrides(bias_desired, bias_strides,
            bias_info->data_type, bias_bytes);
    }

    // Input tensor.
    auto input_tensor = MakeTensorInfo(in_sizes, dataType);

    // Output tensor.
    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    auto out_tensor = out_edge
        ? MakeTensorInfo(out_edge->sizes, dataType)
        : MakeTensorInfo({batchSize, sequenceLength, vHiddenSize}, dataType);

    // --- Build TranslatedOp: Primary = GEMM ---
    TranslatedOp result;
    result.input_tensors  = { input_tensor, weight_tensor };
    result.input_buffer_descs = { input_tensor.ToBufferDesc(), weight_tensor.ToBufferDesc() };
    result.input_name_reorder = { inputIdx, weightsIdx };

    if (hasBias) {
        result.input_tensors.push_back(bias_tensor);
        result.input_buffer_descs.push_back(bias_tensor.ToBufferDesc());
        result.input_name_reorder.push_back(biasIdx);
    }

    size_t primary_count = result.input_tensors.size();

    // Append mask and attBias for graph_inputs wiring.
    size_t mask_onnx_pos = SIZE_MAX, attbias_onnx_pos = SIZE_MAX;
    if (hasMask) {
        mask_onnx_pos = result.input_tensors.size();
        result.input_tensors.push_back(mask_tensor);
        result.input_buffer_descs.push_back(mask_tensor.ToBufferDesc());
        result.input_name_reorder.push_back(maskIdx);
    }
    if (hasAttBias) {
        auto* ab_info = LookupShape(value_shapes, inputs[attBiasIdx]);
        if (!ab_info) return std::nullopt;
        auto abt = MakeTensorInfo(ab_info->sizes, ab_info->data_type);
        attbias_onnx_pos = result.input_tensors.size();
        result.input_tensors.push_back(abt);
        result.input_buffer_descs.push_back(abt.ToBufferDesc());
        result.input_name_reorder.push_back(attBiasIdx);
    }

    result.primary_input_count = primary_count;
    result.input_tensor_descs.resize(result.input_tensors.size());
    result.output_tensors = { out_tensor };
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);

    // Primary fixup: GEMM desc.
    bool lHasBias = hasBias;
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_GEMM, &storage->gemm_desc };
    result.fixup = [storage, lHasBias](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->gemm_out_buf.Sizes = storage->gemm_out_info.sizes.data();
        storage->gemm_out_buf.Strides = nullptr;
        storage->gemm_out_td = { DML_TENSOR_TYPE_BUFFER, &storage->gemm_out_buf };
        storage->gemm_desc.ATensor = &self.input_tensor_descs[0];
        storage->gemm_desc.BTensor = &self.input_tensor_descs[1];
        storage->gemm_desc.CTensor = lHasBias ? &self.input_tensor_descs[2] : nullptr;
        storage->gemm_desc.OutputTensor = &storage->gemm_out_td;
        storage->gemm_desc.TransA = DML_MATRIX_TRANSFORM_NONE;
        storage->gemm_desc.TransB = DML_MATRIX_TRANSFORM_NONE;
        storage->gemm_desc.Alpha = 1.0f;
        storage->gemm_desc.Beta = 1.0f;
        storage->gemm_desc.FusedActivation = nullptr;
    };
    result.FixupPointers();

    // --- Build sub_nodes ---
    int sub_idx = 0;

    if (hasSlicedValue) {
        // Sub_node 0: Slice QK [B, S, 2*hiddenSize]
        std::vector<uint32_t> qk_shape = {batchSize, sequenceLength, hiddenSize + hiddenSize};
        storage->qk_offsets = {0, 0, 0};
        storage->qk_sizes   = qk_shape;
        storage->qk_strides_i = {1, 1, 1};
        storage->qk_sliced_info = MakeTensorInfo(qk_shape, dataType);

        SubNode qk_slice;
        qk_slice.input_tensors  = { storage->gemm_out_info };
        qk_slice.output_tensors = { storage->qk_sliced_info };
        qk_slice.input_buffer_descs  = { storage->gemm_out_info.ToBufferDesc() };
        qk_slice.output_buffer_descs = { storage->qk_sliced_info.ToBufferDesc() };
        qk_slice.input_tensor_descs.resize(1);
        qk_slice.output_tensor_descs.resize(1);
        qk_slice.input_from = { {-1, 0} };
        qk_slice.desc_storage = storage;
        qk_slice.op_desc = { DML_OPERATOR_SLICE1, &storage->qk_slice_desc };
        qk_slice.fixup = [storage](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->qk_slice_desc.InputTensor = &self.input_tensor_descs[0];
            storage->qk_slice_desc.OutputTensor = &self.output_tensor_descs[0];
            storage->qk_slice_desc.DimensionCount = static_cast<UINT>(storage->qk_offsets.size());
            storage->qk_slice_desc.InputWindowOffsets = storage->qk_offsets.data();
            storage->qk_slice_desc.InputWindowSizes = storage->qk_sizes.data();
            storage->qk_slice_desc.InputWindowStrides = storage->qk_strides_i.data();
        };
        qk_slice.FixupPointers();
        result.sub_nodes.push_back(std::move(qk_slice));
        int qk_slice_idx = sub_idx++;

        // Sub_node 1: Slice V [B, S, vHiddenSize]
        std::vector<uint32_t> v_shape = {batchSize, sequenceLength, vHiddenSize};
        storage->v_offsets = {0, 0, 2 * hiddenSize};
        storage->v_sizes   = v_shape;
        storage->v_strides_i = {1, 1, 1};
        storage->v_sliced_info = MakeTensorInfo(v_shape, dataType);

        SubNode v_slice;
        v_slice.input_tensors  = { storage->gemm_out_info };
        v_slice.output_tensors = { storage->v_sliced_info };
        v_slice.input_buffer_descs  = { storage->gemm_out_info.ToBufferDesc() };
        v_slice.output_buffer_descs = { storage->v_sliced_info.ToBufferDesc() };
        v_slice.input_tensor_descs.resize(1);
        v_slice.output_tensor_descs.resize(1);
        v_slice.input_from = { {-1, 0} };
        v_slice.desc_storage = storage;
        v_slice.op_desc = { DML_OPERATOR_SLICE1, &storage->v_slice_desc };
        v_slice.fixup = [storage](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->v_slice_desc.InputTensor = &self.input_tensor_descs[0];
            storage->v_slice_desc.OutputTensor = &self.output_tensor_descs[0];
            storage->v_slice_desc.DimensionCount = static_cast<UINT>(storage->v_offsets.size());
            storage->v_slice_desc.InputWindowOffsets = storage->v_offsets.data();
            storage->v_slice_desc.InputWindowSizes = storage->v_sizes.data();
            storage->v_slice_desc.InputWindowStrides = storage->v_strides_i.data();
        };
        v_slice.FixupPointers();
        result.sub_nodes.push_back(std::move(v_slice));
        int v_slice_idx = sub_idx++;

        // Sub_node 2: Identity (transpose QK from [B,S,2,N,H] to [B,S,N,2,H])
        std::vector<uint32_t> qk_t_shape = {batchSize, sequenceLength, numHeads, 2, headSize};
        std::vector<uint32_t> qk_t_strides = {
            sequenceLength * numHeads * 2 * headSize,
            numHeads * 2 * headSize,
            headSize,
            numHeads * headSize,
            1
        };
        uint64_t qk_bytes = ComputeAlignedTotalBytes({batchSize, sequenceLength, hiddenSize + hiddenSize}, dataType);
        storage->transpose_in_info = MakeTensorInfoWithStrides(qk_t_shape, qk_t_strides, dataType, qk_bytes);
        storage->transpose_out_info = MakeTensorInfo(qk_t_shape, dataType);

        SubNode qk_transpose;
        qk_transpose.input_tensors  = { storage->transpose_in_info };
        qk_transpose.output_tensors = { storage->transpose_out_info };
        qk_transpose.input_buffer_descs  = { storage->transpose_in_info.ToBufferDesc() };
        qk_transpose.output_buffer_descs = { storage->transpose_out_info.ToBufferDesc() };
        qk_transpose.input_tensor_descs.resize(1);
        qk_transpose.output_tensor_descs.resize(1);
        qk_transpose.input_from = { {qk_slice_idx, 0} };
        qk_transpose.desc_storage = storage;
        qk_transpose.op_desc = { DML_OPERATOR_ELEMENT_WISE_IDENTITY, &storage->identity_desc };
        qk_transpose.fixup = [storage](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->identity_desc.InputTensor  = &self.input_tensor_descs[0];
            storage->identity_desc.OutputTensor = &self.output_tensor_descs[0];
            storage->identity_desc.ScaleBias    = nullptr;
        };
        qk_transpose.FixupPointers();
        result.sub_nodes.push_back(std::move(qk_transpose));
        int qk_transpose_idx = sub_idx++;

        // Optional: mask slice sub_node
        int mask_slice_idx = -1;
        if (hasMaxSequenceMask) {
            std::vector<uint32_t> mask_out_shape = {batchSize, numHeads, sequenceLength, sequenceLength};
            storage->mask_offsets = {0, 0, 0, 0};
            storage->mask_sizes   = mask_out_shape;
            storage->mask_strides_i = {1, 1, 1, 1};
            storage->mask_sliced_info = MakeTensorInfo(mask_out_shape, mask_tensor.data_type);

            SubNode mask_slice;
            mask_slice.input_tensors  = { mask_tensor };
            mask_slice.output_tensors = { storage->mask_sliced_info };
            mask_slice.input_buffer_descs  = { mask_tensor.ToBufferDesc() };
            mask_slice.output_buffer_descs = { storage->mask_sliced_info.ToBufferDesc() };
            mask_slice.input_tensor_descs.resize(1);
            mask_slice.output_tensor_descs.resize(1);
            mask_slice.input_from = { {-2, 0} };
            mask_slice.graph_inputs = { {maskIdx, 0} };
            mask_slice.desc_storage = storage;
            mask_slice.op_desc = { DML_OPERATOR_SLICE1, &storage->mask_slice_desc };
            mask_slice.fixup = [storage](SubNode& self) {
                RebuildSubNodePointers(self);
                storage->mask_slice_desc.InputTensor = &self.input_tensor_descs[0];
                storage->mask_slice_desc.OutputTensor = &self.output_tensor_descs[0];
                storage->mask_slice_desc.DimensionCount = static_cast<UINT>(storage->mask_offsets.size());
                storage->mask_slice_desc.InputWindowOffsets = storage->mask_offsets.data();
                storage->mask_slice_desc.InputWindowSizes = storage->mask_sizes.data();
                storage->mask_slice_desc.InputWindowStrides = storage->mask_strides_i.data();
            };
            mask_slice.FixupPointers();
            result.sub_nodes.push_back(std::move(mask_slice));
            mask_slice_idx = sub_idx++;
        }

        // Sub_node N: MHA (last sub_node — outputs route from here)
        DmlTensorInfo v_sliced_tensor = storage->v_sliced_info;
        DmlTensorInfo qk_transposed_tensor = storage->transpose_out_info;
        DmlTensorInfo mask_sliced_tensor = storage->mask_sliced_info;

        SubNode mha_node;
        mha_node.input_tensors.resize(11);
        mha_node.input_buffer_descs.resize(11);
        mha_node.input_tensor_descs.resize(11);
        mha_node.input_from.resize(11, {-2, 0});

        // Slot 2: ValueTensor from v_slice
        mha_node.input_tensors[2] = v_sliced_tensor;
        mha_node.input_buffer_descs[2] = v_sliced_tensor.ToBufferDesc();
        mha_node.input_from[2] = {v_slice_idx, 0};

        // Slot 3: StackedQueryKeyTensor from qk_transpose
        mha_node.input_tensors[3] = qk_transposed_tensor;
        mha_node.input_buffer_descs[3] = qk_transposed_tensor.ToBufferDesc();
        mha_node.input_from[3] = {qk_transpose_idx, 0};

        // Slot 7: MaskTensor
        if (hasMask) {
            if (hasMaxSequenceMask) {
                mha_node.input_tensors[7] = mask_sliced_tensor;
                mha_node.input_buffer_descs[7] = mask_sliced_tensor.ToBufferDesc();
                mha_node.input_from[7] = {mask_slice_idx, 0};
            } else {
                mha_node.input_tensors[7] = mask_tensor;
                mha_node.input_buffer_descs[7] = mask_tensor.ToBufferDesc();
                mha_node.graph_inputs.push_back({maskIdx, 7});
            }
        }

        // Slot 8: RelativePositionBiasTensor
        if (hasAttBias) {
            auto* ab_info = LookupShape(value_shapes, inputs[attBiasIdx]);
            auto abt = MakeTensorInfo(ab_info->sizes, ab_info->data_type);
            mha_node.input_tensors[8] = abt;
            mha_node.input_buffer_descs[8] = abt.ToBufferDesc();
            mha_node.graph_inputs.push_back({attBiasIdx, 8});
        }

        mha_node.output_tensors = { out_tensor };
        mha_node.output_buffer_descs = { out_tensor.ToBufferDesc() };
        mha_node.output_tensor_descs.resize(1);
        mha_node.desc_storage = storage;
        mha_node.op_desc = { DML_OPERATOR_MULTIHEAD_ATTENTION, &storage->mha_desc };

        float l_scale = scale, l_mask_filter = maskFilterValue;
        uint32_t l_numHeads = numHeads;
        DML_MULTIHEAD_ATTENTION_MASK_TYPE l_maskType = maskType;
        mha_node.fixup = [storage, l_scale, l_mask_filter, l_numHeads, l_maskType](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->mha_desc = {};
            storage->mha_desc.ValueTensor              = &self.input_tensor_descs[2];
            storage->mha_desc.StackedQueryKeyTensor    = &self.input_tensor_descs[3];
            storage->mha_desc.MaskTensor               = self.input_tensor_descs[7].Desc ? &self.input_tensor_descs[7] : nullptr;
            storage->mha_desc.RelativePositionBiasTensor = self.input_tensor_descs[8].Desc ? &self.input_tensor_descs[8] : nullptr;
            storage->mha_desc.OutputTensor             = &self.output_tensor_descs[0];
            storage->mha_desc.Scale = l_scale;
            storage->mha_desc.MaskFilterValue = l_mask_filter;
            storage->mha_desc.HeadCount = l_numHeads;
            storage->mha_desc.MaskType = l_maskType;
        };
        mha_node.FixupPointers();
        result.sub_nodes.push_back(std::move(mha_node));
    } else {
        // Common path: hiddenSize == vHiddenSize.
        // Sub_node 0: Identity (transpose QKV from [B,S,3,N,H] to [B,S,N,3,H])
        std::vector<uint32_t> qkv_shape = {batchSize, sequenceLength, numHeads, 3, headSize};
        std::vector<uint32_t> qkv_strides = {
            sequenceLength * numHeads * 3 * headSize,
            numHeads * 3 * headSize,
            headSize,
            numHeads * headSize,
            1
        };
        uint64_t qkv_bytes = ComputeAlignedTotalBytes(gemm_out_shape, dataType);
        storage->transpose_in_info  = MakeTensorInfoWithStrides(qkv_shape, qkv_strides, dataType, qkv_bytes);
        storage->transpose_out_info = MakeTensorInfo(qkv_shape, dataType);

        SubNode transpose_node;
        transpose_node.input_tensors  = { storage->transpose_in_info };
        transpose_node.output_tensors = { storage->transpose_out_info };
        transpose_node.input_buffer_descs  = { storage->transpose_in_info.ToBufferDesc() };
        transpose_node.output_buffer_descs = { storage->transpose_out_info.ToBufferDesc() };
        transpose_node.input_tensor_descs.resize(1);
        transpose_node.output_tensor_descs.resize(1);
        transpose_node.input_from = { {-1, 0} };
        transpose_node.desc_storage = storage;
        transpose_node.op_desc = { DML_OPERATOR_ELEMENT_WISE_IDENTITY, &storage->identity_desc };
        transpose_node.fixup = [storage](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->identity_desc.InputTensor  = &self.input_tensor_descs[0];
            storage->identity_desc.OutputTensor = &self.output_tensor_descs[0];
            storage->identity_desc.ScaleBias    = nullptr;
        };
        transpose_node.FixupPointers();
        result.sub_nodes.push_back(std::move(transpose_node));
        int transpose_idx = sub_idx++;

        // Optional: mask slice sub_node
        int mask_slice_idx = -1;
        if (hasMaxSequenceMask) {
            std::vector<uint32_t> mask_out_shape = {batchSize, numHeads, sequenceLength, sequenceLength};
            storage->mask_offsets = {0, 0, 0, 0};
            storage->mask_sizes   = mask_out_shape;
            storage->mask_strides_i = {1, 1, 1, 1};
            storage->mask_sliced_info = MakeTensorInfo(mask_out_shape, mask_tensor.data_type);

            SubNode mask_slice;
            mask_slice.input_tensors  = { mask_tensor };
            mask_slice.output_tensors = { storage->mask_sliced_info };
            mask_slice.input_buffer_descs  = { mask_tensor.ToBufferDesc() };
            mask_slice.output_buffer_descs = { storage->mask_sliced_info.ToBufferDesc() };
            mask_slice.input_tensor_descs.resize(1);
            mask_slice.output_tensor_descs.resize(1);
            mask_slice.input_from = { {-2, 0} };
            mask_slice.graph_inputs = { {maskIdx, 0} };
            mask_slice.desc_storage = storage;
            mask_slice.op_desc = { DML_OPERATOR_SLICE1, &storage->mask_slice_desc };
            mask_slice.fixup = [storage](SubNode& self) {
                RebuildSubNodePointers(self);
                storage->mask_slice_desc.InputTensor = &self.input_tensor_descs[0];
                storage->mask_slice_desc.OutputTensor = &self.output_tensor_descs[0];
                storage->mask_slice_desc.DimensionCount = static_cast<UINT>(storage->mask_offsets.size());
                storage->mask_slice_desc.InputWindowOffsets = storage->mask_offsets.data();
                storage->mask_slice_desc.InputWindowSizes = storage->mask_sizes.data();
                storage->mask_slice_desc.InputWindowStrides = storage->mask_strides_i.data();
            };
            mask_slice.FixupPointers();
            result.sub_nodes.push_back(std::move(mask_slice));
            mask_slice_idx = sub_idx++;
        }

        // Sub_node N: MHA (last sub_node)
        DmlTensorInfo qkv_transposed = storage->transpose_out_info;
        DmlTensorInfo mask_sliced_t = storage->mask_sliced_info;

        SubNode mha_node;
        mha_node.input_tensors.resize(11);
        mha_node.input_buffer_descs.resize(11);
        mha_node.input_tensor_descs.resize(11);
        mha_node.input_from.resize(11, {-2, 0});

        // Slot 5: StackedQueryKeyValueTensor from transpose
        mha_node.input_tensors[5] = qkv_transposed;
        mha_node.input_buffer_descs[5] = qkv_transposed.ToBufferDesc();
        mha_node.input_from[5] = {transpose_idx, 0};

        // Slot 7: MaskTensor
        if (hasMask) {
            if (hasMaxSequenceMask) {
                mha_node.input_tensors[7] = mask_sliced_t;
                mha_node.input_buffer_descs[7] = mask_sliced_t.ToBufferDesc();
                mha_node.input_from[7] = {mask_slice_idx, 0};
            } else {
                mha_node.input_tensors[7] = mask_tensor;
                mha_node.input_buffer_descs[7] = mask_tensor.ToBufferDesc();
                mha_node.graph_inputs.push_back({maskIdx, 7});
            }
        }

        // Slot 8: RelativePositionBiasTensor
        if (hasAttBias) {
            auto* ab_info = LookupShape(value_shapes, inputs[attBiasIdx]);
            auto abt = MakeTensorInfo(ab_info->sizes, ab_info->data_type);
            mha_node.input_tensors[8] = abt;
            mha_node.input_buffer_descs[8] = abt.ToBufferDesc();
            mha_node.graph_inputs.push_back({attBiasIdx, 8});
        }

        mha_node.output_tensors = { out_tensor };
        mha_node.output_buffer_descs = { out_tensor.ToBufferDesc() };
        mha_node.output_tensor_descs.resize(1);
        mha_node.desc_storage = storage;
        mha_node.op_desc = { DML_OPERATOR_MULTIHEAD_ATTENTION, &storage->mha_desc };

        float l_scale = scale, l_mask_filter = maskFilterValue;
        uint32_t l_numHeads = numHeads;
        DML_MULTIHEAD_ATTENTION_MASK_TYPE l_maskType = maskType;
        mha_node.fixup = [storage, l_scale, l_mask_filter, l_numHeads, l_maskType](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->mha_desc = {};
            storage->mha_desc.StackedQueryKeyValueTensor = &self.input_tensor_descs[5];
            storage->mha_desc.MaskTensor               = self.input_tensor_descs[7].Desc ? &self.input_tensor_descs[7] : nullptr;
            storage->mha_desc.RelativePositionBiasTensor = self.input_tensor_descs[8].Desc ? &self.input_tensor_descs[8] : nullptr;
            storage->mha_desc.OutputTensor             = &self.output_tensor_descs[0];
            storage->mha_desc.Scale = l_scale;
            storage->mha_desc.MaskFilterValue = l_mask_filter;
            storage->mha_desc.HeadCount = l_numHeads;
            storage->mha_desc.MaskType = l_maskType;
        };
        mha_node.FixupPointers();
        result.sub_nodes.push_back(std::move(mha_node));
    }

    return result;
}

// ---------------------------------------------------------------------------
// QAttention → MatMulIntegerToFloat + Identity(transpose) + MHA SubNode graph
// ORT ref: DmlOperatorQAttention.cpp
// Decomposition: MMITF(quant input × quant weights + bias) → transpose QKV → MHA
// Optional: past/present KV cache via Slice + Join, causal mask via DiagonalMatrix1.
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateQAttention(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 5 || outputs.empty()) return std::nullopt;

    enum OnnxInput : size_t { inputIdx=0, weightsIdx=1, biasIdx=2,
                              inputScaleIdx=3, weightScaleIdx=4, maskIdx=5,
                              inputZpIdx=6, weightZpIdx=7, pastIdx=8 };
    enum OnnxOutput : size_t { outputIdx=0, presentIdx=1 };

    auto valid = [&](size_t idx) {
        return idx < inputs.size() && !inputs[idx].empty() && value_shapes.count(inputs[idx]);
    };
    auto out_valid = [&](size_t idx) {
        return idx < outputs.size() && !outputs[idx].empty() && value_shapes.count(outputs[idx]);
    };

    OrtNodeAdapter adapter(node, ort_api);
    bool unidirectional = adapter.GetAttributeInt("unidirectional", 0) != 0;
    bool hasMask = valid(maskIdx);
    bool hasPast = valid(pastIdx);
    bool hasPresent = out_valid(presentIdx);

    // ORT QueryQAttention restrictions.
    if (unidirectional && hasMask) return std::nullopt;
    if (adapter.GetAttributeInt("do_rotary", 0) != 0) return std::nullopt;
    if (adapter.GetAttributeInt("past_present_share_buffer", 0) != 0) return std::nullopt;
    if (hasPast && !hasPresent) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[inputIdx]);
    auto* w_info  = LookupShape(value_shapes, inputs[weightsIdx]);
    if (!in_info || !w_info) return std::nullopt;

    bool hasBias = valid(biasIdx);
    bool hasInputZp = valid(inputZpIdx);
    bool hasWeightZp = valid(weightZpIdx);

    uint32_t numHeads = static_cast<uint32_t>(adapter.GetAttributeInt("num_heads", 0));
    if (numHeads == 0) return std::nullopt;

    auto in_sizes = in_info->sizes;
    auto w_sizes  = w_info->sizes;
    uint32_t in_rank = in_info->original_rank ? in_info->original_rank
                     : static_cast<uint32_t>(in_sizes.size());
    size_t in_off = in_sizes.size() >= in_rank ? in_sizes.size() - in_rank : 0;
    uint32_t batchSize = in_sizes[in_off];
    uint32_t sequenceLength = in_sizes[in_off + 1];
    uint32_t inputHidden = in_sizes[in_off + 2];

    uint32_t w_rank = w_info->original_rank ? w_info->original_rank
                    : static_cast<uint32_t>(w_sizes.size());
    size_t w_off = w_sizes.size() >= w_rank ? w_sizes.size() - w_rank : 0;
    uint32_t hiddenSize = w_sizes[w_off + 1] / 3;
    uint32_t headSize = hiddenSize / numHeads;

    uint32_t pastSequenceLength = 0;
    if (hasPast) {
        auto* past_info = LookupShape(value_shapes, inputs[pastIdx]);
        if (!past_info) return std::nullopt;
        uint32_t past_rank = past_info->original_rank ? past_info->original_rank
                           : static_cast<uint32_t>(past_info->sizes.size());
        size_t past_off = past_info->sizes.size() >= past_rank
                        ? past_info->sizes.size() - past_rank : 0;
        pastSequenceLength = past_info->sizes[past_off + 3];
    }
    uint32_t totalSequenceLength = pastSequenceLength + sequenceLength;

    float scale = adapter.GetAttributeFloat("scale", 0.0f);
    if (scale == 0.0f) scale = 1.0f / std::sqrt(static_cast<float>(headSize));
    float maskFilterValue = unidirectional
        ? std::numeric_limits<float>::lowest()
        : adapter.GetAttributeFloat("mask_filter_value", -10000.0f);

    // Output data type (float, matching ORT).
    DML_TENSOR_DATA_TYPE outDataType = DML_TENSOR_DATA_TYPE_FLOAT32;
    if (auto* out_info = LookupShape(value_shapes, outputs[outputIdx]))
        outDataType = out_info->data_type;

    // --- Mask detection ---
    DML_MULTIHEAD_ATTENTION_MASK_TYPE maskType = DML_MULTIHEAD_ATTENTION_MASK_TYPE_NONE;
    DmlTensorInfo mask_tensor{};
    bool hasMaxSequenceMask = false;
    if (hasMask) {
        auto* mask_info = LookupShape(value_shapes, inputs[maskIdx]);
        if (!mask_info) return std::nullopt;
        uint32_t mask_rank = mask_info->original_rank ? mask_info->original_rank
                           : static_cast<uint32_t>(mask_info->sizes.size());
        if (mask_rank == 1) {
            uint32_t mask_size = mask_info->sizes.back();
            uint32_t batchGroupCount = mask_size / batchSize;
            if (batchGroupCount == 1) {
                maskType = DML_MULTIHEAD_ATTENTION_MASK_TYPE_KEY_SEQUENCE_LENGTH;
                mask_tensor = MakeTensorInfo({1, batchSize}, mask_info->data_type);
            } else {
                maskType = DML_MULTIHEAD_ATTENTION_MASK_TYPE_KEY_SEQUENCE_END_START;
                mask_tensor = MakeTensorInfo({batchGroupCount, batchSize}, mask_info->data_type);
            }
        } else {
            maskType = DML_MULTIHEAD_ATTENTION_MASK_TYPE_BOOLEAN;
            size_t m_off = mask_info->sizes.size() >= mask_rank
                         ? mask_info->sizes.size() - mask_rank : 0;
            std::vector<uint32_t> actual_shape(4, 1u), desired_shape(4, 1u);
            if (mask_rank == 2) {
                actual_shape  = {batchSize, 1, 1, totalSequenceLength};
                desired_shape = {batchSize, numHeads, sequenceLength, totalSequenceLength};
            } else if (mask_rank == 3) {
                actual_shape  = {batchSize, 1, sequenceLength, totalSequenceLength};
                desired_shape = {batchSize, numHeads, sequenceLength, totalSequenceLength};
            } else {
                uint32_t maskSeqLen = mask_info->sizes[m_off + 2];
                if (maskSeqLen != sequenceLength) {
                    hasMaxSequenceMask = true;
                    actual_shape  = {batchSize, numHeads, maskSeqLen, maskSeqLen};
                    desired_shape = actual_shape;
                } else {
                    actual_shape  = {batchSize, numHeads, sequenceLength, totalSequenceLength};
                    desired_shape = actual_shape;
                }
            }
            auto strides = ComputePackedStrides(actual_shape);
            for (size_t d = 0; d < 4; ++d) {
                if (actual_shape[d] == 1 && desired_shape[d] > 1) strides[d] = 0;
            }
            uint64_t mask_bytes = mask_info->total_bytes > 0
                ? mask_info->total_bytes
                : ComputeAlignedTotalBytes(actual_shape, mask_info->data_type);
            if (hasMaxSequenceMask) {
                mask_tensor = MakeTensorInfo(actual_shape, mask_info->data_type);
            } else {
                mask_tensor = MakeTensorInfoWithStrides(desired_shape, strides,
                    mask_info->data_type, mask_bytes);
            }
        }
    } else if (unidirectional) {
        maskType = DML_MULTIHEAD_ATTENTION_MASK_TYPE_BOOLEAN;
    }

    // --- Storage struct ---
    struct QAttentionStorage {
        DML_MATRIX_MULTIPLY_INTEGER_TO_FLOAT_OPERATOR_DESC mmitf_desc{};
        DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identity_desc{};
        DML_MULTIHEAD_ATTENTION_OPERATOR_DESC mha_desc{};
        DML_DIAGONAL_MATRIX1_OPERATOR_DESC causal_desc{};
        DML_SLICE1_OPERATOR_DESC mask_slice_desc{};
        DML_SLICE1_OPERATOR_DESC past_key_slice_desc{};
        DML_SLICE1_OPERATOR_DESC past_value_slice_desc{};
        DML_JOIN_OPERATOR_DESC join_desc{};
        DmlTensorInfo mmitf_out_info;
        DML_BUFFER_TENSOR_DESC mmitf_out_buf{};
        DML_TENSOR_DESC mmitf_out_td{};
        DmlTensorInfo transpose_in_info, transpose_out_info;
        DML_BUFFER_TENSOR_DESC transpose_in_buf{}, transpose_out_buf{};
        DML_TENSOR_DESC transpose_in_td{}, transpose_out_td{};
        DmlTensorInfo causal_info, causal_broadcast_info;
        DML_BUFFER_TENSOR_DESC causal_buf{}, causal_broadcast_buf{};
        DML_TENSOR_DESC causal_td{}, causal_broadcast_td{};
        DmlTensorInfo mask_sliced_info;
        DML_BUFFER_TENSOR_DESC mask_sliced_buf{};
        DML_TENSOR_DESC mask_sliced_td{};
        DmlTensorInfo past_key_info, past_value_info;
        DmlTensorInfo present_key_info, present_value_info;
        DML_BUFFER_TENSOR_DESC present_key_buf{}, present_value_buf{};
        DML_TENSOR_DESC present_key_td{}, present_value_td{};
        std::vector<DML_BUFFER_TENSOR_DESC> join_in_bufs;
        std::vector<DML_TENSOR_DESC> join_in_tds;
        std::vector<uint32_t> mask_offsets, mask_sizes;
        std::vector<int32_t> mask_strides_i;
        std::vector<uint32_t> pk_offsets, pk_sizes, pv_offsets, pv_sizes;
        std::vector<int32_t> pk_strides_i, pv_strides_i;
    };
    auto storage = std::make_shared<QAttentionStorage>();

    // MMITF output: [B, S, 3*hiddenSize]
    std::vector<uint32_t> mmitf_out_shape = {batchSize, sequenceLength, 3 * hiddenSize};
    storage->mmitf_out_info = MakeTensorInfo(mmitf_out_shape, outDataType);

    // --- Build primary inputs (MMITF) ---
    // DML MMITF slots: A(0), AScale(1), AZP(2), B(3), BScale(4), BZP(5), Bias(6)
    // ONNX order:      input(0), weights(1), bias(2), in_scale(3), w_scale(4), mask(5), in_zp(6), w_zp(7)
    auto input_tensor = MakeTensorInfo(in_sizes, in_info->data_type);

    auto* is_info = LookupShape(value_shapes, inputs[inputScaleIdx]);
    auto* ws_info = LookupShape(value_shapes, inputs[weightScaleIdx]);
    if (!is_info || !ws_info) return std::nullopt;
    auto input_scale_tensor = MakeTensorInfo(is_info->sizes, is_info->data_type);
    auto weight_scale_tensor = MakeTensorInfo(ws_info->sizes, ws_info->data_type);

    DmlTensorInfo input_zp_tensor{};
    if (hasInputZp) {
        auto* izp_info = LookupShape(value_shapes, inputs[inputZpIdx]);
        if (!izp_info) return std::nullopt;
        input_zp_tensor = MakeTensorInfo(izp_info->sizes, izp_info->data_type);
    }
    DmlTensorInfo weight_zp_tensor{};
    if (hasWeightZp) {
        auto* wzp_info = LookupShape(value_shapes, inputs[weightZpIdx]);
        if (!wzp_info) return std::nullopt;
        weight_zp_tensor = MakeTensorInfo(wzp_info->sizes, wzp_info->data_type);
    }

    // Weights broadcast: [inputHidden, 3*hiddenSize] → [B, inputHidden, 3*hiddenSize]
    std::vector<uint32_t> w_actual = {1, inputHidden, 3 * hiddenSize};
    std::vector<uint32_t> w_desired = {batchSize, inputHidden, 3 * hiddenSize};
    auto w_strides = ComputePackedStrides(w_actual);
    w_strides[0] = 0;
    uint64_t w_bytes = ComputeAlignedTotalBytes(w_actual, w_info->data_type);
    auto weight_tensor = MakeTensorInfoWithStrides(w_desired, w_strides, w_info->data_type, w_bytes);

    DmlTensorInfo bias_tensor{};
    if (hasBias) {
        auto* bias_info = LookupShape(value_shapes, inputs[biasIdx]);
        if (!bias_info) return std::nullopt;
        std::vector<uint32_t> bias_desired = {batchSize, sequenceLength, 3 * hiddenSize};
        std::vector<uint32_t> bias_strides_v = {0, 0, 1};
        uint64_t bias_bytes = ComputeAlignedTotalBytes({3 * hiddenSize}, bias_info->data_type);
        bias_tensor = MakeTensorInfoWithStrides(bias_desired, bias_strides_v,
            bias_info->data_type, bias_bytes);
    }

    auto* out_edge = LookupShape(value_shapes, outputs[outputIdx]);
    auto out_tensor = out_edge
        ? MakeTensorInfo(out_edge->sizes, outDataType)
        : MakeTensorInfo({batchSize, sequenceLength, hiddenSize}, outDataType);

    TranslatedOp result;
    // Inputs in DML MMITF order: A, AScale, AZP, B, BScale, BZP, Bias
    result.input_tensors = { input_tensor, input_scale_tensor };
    result.input_buffer_descs = { input_tensor.ToBufferDesc(), input_scale_tensor.ToBufferDesc() };
    result.input_name_reorder = { inputIdx, inputScaleIdx };

    if (hasInputZp) {
        result.input_tensors.push_back(input_zp_tensor);
        result.input_buffer_descs.push_back(input_zp_tensor.ToBufferDesc());
        result.input_name_reorder.push_back(inputZpIdx);
    }

    result.input_tensors.push_back(weight_tensor);
    result.input_buffer_descs.push_back(weight_tensor.ToBufferDesc());
    result.input_name_reorder.push_back(weightsIdx);

    result.input_tensors.push_back(weight_scale_tensor);
    result.input_buffer_descs.push_back(weight_scale_tensor.ToBufferDesc());
    result.input_name_reorder.push_back(weightScaleIdx);

    if (hasWeightZp) {
        result.input_tensors.push_back(weight_zp_tensor);
        result.input_buffer_descs.push_back(weight_zp_tensor.ToBufferDesc());
        result.input_name_reorder.push_back(weightZpIdx);
    }

    if (hasBias) {
        result.input_tensors.push_back(bias_tensor);
        result.input_buffer_descs.push_back(bias_tensor.ToBufferDesc());
        result.input_name_reorder.push_back(biasIdx);
    }

    size_t primary_count = result.input_tensors.size();

    // Append non-primary inputs for graph_inputs wiring.
    if (hasMask) {
        result.input_tensors.push_back(mask_tensor);
        result.input_buffer_descs.push_back(mask_tensor.ToBufferDesc());
        result.input_name_reorder.push_back(maskIdx);
    }
    if (hasPast) {
        auto* past_info = LookupShape(value_shapes, inputs[pastIdx]);
        auto past_tensor = MakeTensorInfo(past_info->sizes, past_info->data_type);
        result.input_tensors.push_back(past_tensor);
        result.input_buffer_descs.push_back(past_tensor.ToBufferDesc());
        result.input_name_reorder.push_back(pastIdx);
    }

    result.primary_input_count = primary_count;
    result.input_tensor_descs.resize(result.input_tensors.size());

    // Outputs.
    result.output_tensors = { out_tensor };
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    if (hasPresent) {
        auto* pres_info = LookupShape(value_shapes, outputs[presentIdx]);
        auto present_tensor = pres_info
            ? MakeTensorInfo(pres_info->sizes, outDataType)
            : MakeTensorInfo({2, batchSize, numHeads, totalSequenceLength, headSize}, outDataType);
        result.output_tensors.push_back(present_tensor);
        result.output_buffer_descs.push_back(present_tensor.ToBufferDesc());
    }
    result.output_tensor_descs.resize(result.output_tensors.size());

    // Primary fixup: MMITF desc.
    bool lHasBias = hasBias, lHasIZP = hasInputZp, lHasWZP = hasWeightZp;
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_MATRIX_MULTIPLY_INTEGER_TO_FLOAT, &storage->mmitf_desc };
    result.fixup = [storage, lHasBias, lHasIZP, lHasWZP](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->mmitf_out_buf.Sizes = storage->mmitf_out_info.sizes.data();
        storage->mmitf_out_buf.Strides = nullptr;
        storage->mmitf_out_td = { DML_TENSOR_TYPE_BUFFER, &storage->mmitf_out_buf };
        int i = 0;
        storage->mmitf_desc.ATensor           = &self.input_tensor_descs[i++];
        storage->mmitf_desc.AScaleTensor      = &self.input_tensor_descs[i++];
        storage->mmitf_desc.AZeroPointTensor  = lHasIZP ? &self.input_tensor_descs[i++] : nullptr;
        storage->mmitf_desc.BTensor           = &self.input_tensor_descs[i++];
        storage->mmitf_desc.BScaleTensor      = &self.input_tensor_descs[i++];
        storage->mmitf_desc.BZeroPointTensor  = lHasWZP ? &self.input_tensor_descs[i++] : nullptr;
        storage->mmitf_desc.BiasTensor        = lHasBias ? &self.input_tensor_descs[i++] : nullptr;
        storage->mmitf_desc.OutputTensor       = &storage->mmitf_out_td;
    };
    result.FixupPointers();

    // --- Sub_nodes ---
    int sub_idx = 0;

    // Sub_node 0: Identity (transpose QKV [B,S,3,N,H] → [B,S,N,3,H])
    std::vector<uint32_t> qkv_shape = {batchSize, sequenceLength, numHeads, 3, headSize};
    std::vector<uint32_t> qkv_strides = {
        sequenceLength * numHeads * 3 * headSize,
        numHeads * 3 * headSize,
        headSize,
        numHeads * headSize,
        1
    };
    uint64_t qkv_bytes = ComputeAlignedTotalBytes(mmitf_out_shape, outDataType);
    storage->transpose_in_info  = MakeTensorInfoWithStrides(qkv_shape, qkv_strides, outDataType, qkv_bytes);
    storage->transpose_out_info = MakeTensorInfo(qkv_shape, outDataType);

    SubNode transpose_node;
    transpose_node.input_tensors  = { storage->transpose_in_info };
    transpose_node.output_tensors = { storage->transpose_out_info };
    transpose_node.input_buffer_descs  = { storage->transpose_in_info.ToBufferDesc() };
    transpose_node.output_buffer_descs = { storage->transpose_out_info.ToBufferDesc() };
    transpose_node.input_tensor_descs.resize(1);
    transpose_node.output_tensor_descs.resize(1);
    transpose_node.input_from = { {-1, 0} };
    transpose_node.desc_storage = storage;
    transpose_node.op_desc = { DML_OPERATOR_ELEMENT_WISE_IDENTITY, &storage->identity_desc };
    transpose_node.fixup = [storage](SubNode& self) {
        RebuildSubNodePointers(self);
        storage->identity_desc.InputTensor  = &self.input_tensor_descs[0];
        storage->identity_desc.OutputTensor = &self.output_tensor_descs[0];
        storage->identity_desc.ScaleBias    = nullptr;
    };
    transpose_node.FixupPointers();
    result.sub_nodes.push_back(std::move(transpose_node));
    int transpose_idx = sub_idx++;

    // Optional: past key/value slice sub_nodes
    int past_key_slice_idx = -1, past_value_slice_idx = -1;
    DML_TENSOR_DATA_TYPE pastDataType = outDataType;
    DML_TENSOR_DATA_TYPE presentDataType = outDataType;
    if (hasPast) {
        auto* past_info_dt = LookupShape(value_shapes, inputs[pastIdx]);
        if (past_info_dt) pastDataType = past_info_dt->data_type;
        auto* pres_info_dt = LookupShape(value_shapes, outputs[presentIdx]);
        if (pres_info_dt) presentDataType = pres_info_dt->data_type;

        // Past shape: [2, B, N, pastS, H]
        std::vector<uint32_t> pk_shape = {1, batchSize, numHeads, pastSequenceLength, headSize};
        storage->pk_offsets = {0, 0, 0, 0, 0};
        storage->pk_sizes = pk_shape;
        storage->pk_strides_i = {1, 1, 1, 1, 1};
        storage->past_key_info = MakeTensorInfo(pk_shape, pastDataType);

        SubNode pk_slice;
        auto* past_info = LookupShape(value_shapes, inputs[pastIdx]);
        auto past_tensor = MakeTensorInfo(past_info->sizes, past_info->data_type);
        pk_slice.input_tensors  = { past_tensor };
        pk_slice.output_tensors = { storage->past_key_info };
        pk_slice.input_buffer_descs  = { past_tensor.ToBufferDesc() };
        pk_slice.output_buffer_descs = { storage->past_key_info.ToBufferDesc() };
        pk_slice.input_tensor_descs.resize(1);
        pk_slice.output_tensor_descs.resize(1);
        pk_slice.input_from = { {-2, 0} };
        pk_slice.graph_inputs = { {pastIdx, 0} };
        pk_slice.desc_storage = storage;
        pk_slice.op_desc = { DML_OPERATOR_SLICE1, &storage->past_key_slice_desc };
        pk_slice.fixup = [storage](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->past_key_slice_desc.InputTensor = &self.input_tensor_descs[0];
            storage->past_key_slice_desc.OutputTensor = &self.output_tensor_descs[0];
            storage->past_key_slice_desc.DimensionCount = static_cast<UINT>(storage->pk_offsets.size());
            storage->past_key_slice_desc.InputWindowOffsets = storage->pk_offsets.data();
            storage->past_key_slice_desc.InputWindowSizes = storage->pk_sizes.data();
            storage->past_key_slice_desc.InputWindowStrides = storage->pk_strides_i.data();
        };
        pk_slice.FixupPointers();
        result.sub_nodes.push_back(std::move(pk_slice));
        past_key_slice_idx = sub_idx++;

        // Past value slice
        std::vector<uint32_t> pv_shape = pk_shape;
        storage->pv_offsets = {1, 0, 0, 0, 0};
        storage->pv_sizes = pv_shape;
        storage->pv_strides_i = {1, 1, 1, 1, 1};
        storage->past_value_info = MakeTensorInfo(pv_shape, pastDataType);

        SubNode pv_slice;
        pv_slice.input_tensors  = { past_tensor };
        pv_slice.output_tensors = { storage->past_value_info };
        pv_slice.input_buffer_descs  = { past_tensor.ToBufferDesc() };
        pv_slice.output_buffer_descs = { storage->past_value_info.ToBufferDesc() };
        pv_slice.input_tensor_descs.resize(1);
        pv_slice.output_tensor_descs.resize(1);
        pv_slice.input_from = { {-2, 0} };
        pv_slice.graph_inputs = { {pastIdx, 0} };
        pv_slice.desc_storage = storage;
        pv_slice.op_desc = { DML_OPERATOR_SLICE1, &storage->past_value_slice_desc };
        pv_slice.fixup = [storage](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->past_value_slice_desc.InputTensor = &self.input_tensor_descs[0];
            storage->past_value_slice_desc.OutputTensor = &self.output_tensor_descs[0];
            storage->past_value_slice_desc.DimensionCount = static_cast<UINT>(storage->pv_offsets.size());
            storage->past_value_slice_desc.InputWindowOffsets = storage->pv_offsets.data();
            storage->past_value_slice_desc.InputWindowSizes = storage->pv_sizes.data();
            storage->past_value_slice_desc.InputWindowStrides = storage->pv_strides_i.data();
        };
        pv_slice.FixupPointers();
        result.sub_nodes.push_back(std::move(pv_slice));
        past_value_slice_idx = sub_idx++;
    }

    // Optional: causal mask sub_node
    int causal_idx = -1;
    if (unidirectional && !hasMask) {
        std::vector<uint32_t> causal_shape = {1, 1, sequenceLength, totalSequenceLength};
        storage->causal_info = MakeTensorInfo(causal_shape, DML_TENSOR_DATA_TYPE_INT32);
        // Broadcast to [B, N, S, totalS]
        std::vector<uint32_t> causal_bcast_shape = {batchSize, numHeads, sequenceLength, totalSequenceLength};
        auto c_strides = ComputePackedStrides(causal_shape);
        c_strides[0] = 0; c_strides[1] = 0;
        uint64_t c_bytes = ComputeAlignedTotalBytes(causal_shape, DML_TENSOR_DATA_TYPE_INT32);
        storage->causal_broadcast_info = MakeTensorInfoWithStrides(causal_bcast_shape, c_strides,
            DML_TENSOR_DATA_TYPE_INT32, c_bytes);

        SubNode causal_node;
        causal_node.output_tensors = { storage->causal_info };
        causal_node.output_buffer_descs = { storage->causal_info.ToBufferDesc() };
        causal_node.output_tensor_descs.resize(1);
        causal_node.desc_storage = storage;
        causal_node.op_desc = { DML_OPERATOR_DIAGONAL_MATRIX1, &storage->causal_desc };
        uint32_t l_pastSeqLen = pastSequenceLength;
        causal_node.fixup = [storage, l_pastSeqLen](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->causal_desc.OutputTensor = &self.output_tensor_descs[0];
            storage->causal_desc.ValueDataType = DML_TENSOR_DATA_TYPE_INT32;
            storage->causal_desc.DiagonalFillBegin = INT32_MIN;
            storage->causal_desc.DiagonalFillEnd = static_cast<int32_t>(l_pastSeqLen + 1);
            storage->causal_desc.Value.Int32 = 1;
        };
        causal_node.FixupPointers();
        result.sub_nodes.push_back(std::move(causal_node));
        causal_idx = sub_idx++;
    }

    // Optional: mask slice sub_node
    int mask_slice_idx = -1;
    if (hasMaxSequenceMask) {
        std::vector<uint32_t> mask_out_shape = {batchSize, numHeads, sequenceLength, sequenceLength};
        storage->mask_offsets = {0, 0, 0, 0};
        storage->mask_sizes = mask_out_shape;
        storage->mask_strides_i = {1, 1, 1, 1};
        storage->mask_sliced_info = MakeTensorInfo(mask_out_shape, mask_tensor.data_type);

        SubNode mask_slice;
        mask_slice.input_tensors  = { mask_tensor };
        mask_slice.output_tensors = { storage->mask_sliced_info };
        mask_slice.input_buffer_descs  = { mask_tensor.ToBufferDesc() };
        mask_slice.output_buffer_descs = { storage->mask_sliced_info.ToBufferDesc() };
        mask_slice.input_tensor_descs.resize(1);
        mask_slice.output_tensor_descs.resize(1);
        mask_slice.input_from = { {-2, 0} };
        mask_slice.graph_inputs = { {maskIdx, 0} };
        mask_slice.desc_storage = storage;
        mask_slice.op_desc = { DML_OPERATOR_SLICE1, &storage->mask_slice_desc };
        mask_slice.fixup = [storage](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->mask_slice_desc.InputTensor = &self.input_tensor_descs[0];
            storage->mask_slice_desc.OutputTensor = &self.output_tensor_descs[0];
            storage->mask_slice_desc.DimensionCount = static_cast<UINT>(storage->mask_offsets.size());
            storage->mask_slice_desc.InputWindowOffsets = storage->mask_offsets.data();
            storage->mask_slice_desc.InputWindowSizes = storage->mask_sizes.data();
            storage->mask_slice_desc.InputWindowStrides = storage->mask_strides_i.data();
        };
        mask_slice.FixupPointers();
        result.sub_nodes.push_back(std::move(mask_slice));
        mask_slice_idx = sub_idx++;
    }

    // MHA sub_node.
    DmlTensorInfo qkv_transposed = storage->transpose_out_info;

    // Determine MHA mask tensor.
    DmlTensorInfo mha_mask_tensor{};
    int mha_mask_from_sub = -1;
    bool mha_mask_via_graph_input = false;
    if (unidirectional && !hasMask) {
        mha_mask_tensor = storage->causal_broadcast_info;
        mha_mask_from_sub = causal_idx;
    } else if (hasMaxSequenceMask) {
        mha_mask_tensor = storage->mask_sliced_info;
        mha_mask_from_sub = mask_slice_idx;
    } else if (hasMask) {
        mha_mask_tensor = mask_tensor;
        mha_mask_via_graph_input = true;
    }

    // MHA present key/value outputs (when hasPast).
    DmlTensorInfo present_key_tensor{}, present_value_tensor{};
    if (hasPast) {
        std::vector<uint32_t> pk_out_shape = {1, batchSize, numHeads, totalSequenceLength, headSize};
        storage->present_key_info = MakeTensorInfo(pk_out_shape, presentDataType);
        storage->present_value_info = MakeTensorInfo(pk_out_shape, presentDataType);
        present_key_tensor = storage->present_key_info;
        present_value_tensor = storage->present_value_info;
    }

    SubNode mha_node;
    mha_node.input_tensors.resize(11);
    mha_node.input_buffer_descs.resize(11);
    mha_node.input_tensor_descs.resize(11);
    mha_node.input_from.resize(11, {-2, 0});

    // Slot 5: StackedQueryKeyValueTensor from transpose
    mha_node.input_tensors[5] = qkv_transposed;
    mha_node.input_buffer_descs[5] = qkv_transposed.ToBufferDesc();
    mha_node.input_from[5] = {transpose_idx, 0};

    // Slot 7: MaskTensor
    if (mha_mask_tensor.data_type != DML_TENSOR_DATA_TYPE_UNKNOWN) {
        mha_node.input_tensors[7] = mha_mask_tensor;
        mha_node.input_buffer_descs[7] = mha_mask_tensor.ToBufferDesc();
        if (mha_mask_via_graph_input) {
            mha_node.graph_inputs.push_back({maskIdx, 7});
        } else {
            mha_node.input_from[7] = {mha_mask_from_sub, 0};
        }
    }

    // Slot 9/10: PastKey/PastValue from slice sub_nodes
    if (hasPast) {
        mha_node.input_tensors[9] = storage->past_key_info;
        mha_node.input_buffer_descs[9] = storage->past_key_info.ToBufferDesc();
        mha_node.input_from[9] = {past_key_slice_idx, 0};

        mha_node.input_tensors[10] = storage->past_value_info;
        mha_node.input_buffer_descs[10] = storage->past_value_info.ToBufferDesc();
        mha_node.input_from[10] = {past_value_slice_idx, 0};
    }

    // MHA outputs: output(0), and optionally presentKey(1), presentValue(2)
    mha_node.output_tensors = { out_tensor };
    mha_node.output_buffer_descs = { out_tensor.ToBufferDesc() };
    if (hasPast) {
        mha_node.output_tensors.push_back(present_key_tensor);
        mha_node.output_buffer_descs.push_back(present_key_tensor.ToBufferDesc());
        mha_node.output_tensors.push_back(present_value_tensor);
        mha_node.output_buffer_descs.push_back(present_value_tensor.ToBufferDesc());
    }
    mha_node.output_tensor_descs.resize(mha_node.output_tensors.size());
    mha_node.desc_storage = storage;
    mha_node.op_desc = { DML_OPERATOR_MULTIHEAD_ATTENTION, &storage->mha_desc };

    float l_scale = scale, l_mask_filter = maskFilterValue;
    uint32_t l_numHeads = numHeads;
    DML_MULTIHEAD_ATTENTION_MASK_TYPE l_maskType = maskType;
    bool l_hasPast = hasPast;
    mha_node.fixup = [storage, l_scale, l_mask_filter, l_numHeads, l_maskType, l_hasPast](SubNode& self) {
        RebuildSubNodePointers(self);
        storage->mha_desc = {};
        storage->mha_desc.StackedQueryKeyValueTensor = &self.input_tensor_descs[5];
        storage->mha_desc.MaskTensor = self.input_tensor_descs[7].Desc ? &self.input_tensor_descs[7] : nullptr;
        storage->mha_desc.OutputTensor = &self.output_tensor_descs[0];
        if (l_hasPast) {
            storage->mha_desc.PastKeyTensor   = &self.input_tensor_descs[9];
            storage->mha_desc.PastValueTensor = &self.input_tensor_descs[10];
            storage->mha_desc.OutputPresentKeyTensor   = &self.output_tensor_descs[1];
            storage->mha_desc.OutputPresentValueTensor = &self.output_tensor_descs[2];
        }
        storage->mha_desc.Scale = l_scale;
        storage->mha_desc.MaskFilterValue = l_mask_filter;
        storage->mha_desc.HeadCount = l_numHeads;
        storage->mha_desc.MaskType = l_maskType;
    };
    mha_node.FixupPointers();
    result.sub_nodes.push_back(std::move(mha_node));
    int mha_idx = sub_idx++;

    // Optional: Join sub_node for present output (concat presentKey + presentValue)
    int join_idx = -1;
    if (hasPast) {
        SubNode join_node;
        join_node.input_tensors = { present_key_tensor, present_value_tensor };
        join_node.input_buffer_descs = { present_key_tensor.ToBufferDesc(), present_value_tensor.ToBufferDesc() };
        join_node.input_tensor_descs.resize(2);
        join_node.input_from = { {mha_idx, 1}, {mha_idx, 2} };

        auto& pres_out = result.output_tensors[1];
        join_node.output_tensors = { pres_out };
        join_node.output_buffer_descs = { pres_out.ToBufferDesc() };
        join_node.output_tensor_descs.resize(1);
        join_node.desc_storage = storage;
        join_node.op_desc = { DML_OPERATOR_JOIN, &storage->join_desc };

        storage->join_in_bufs = { present_key_tensor.ToBufferDesc(), present_value_tensor.ToBufferDesc() };
        storage->join_in_tds.resize(2);

        join_node.fixup = [storage](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->join_in_bufs[0].Sizes = self.input_tensors[0].sizes.data();
            storage->join_in_bufs[0].Strides = nullptr;
            storage->join_in_bufs[1].Sizes = self.input_tensors[1].sizes.data();
            storage->join_in_bufs[1].Strides = nullptr;
            storage->join_in_tds[0] = { DML_TENSOR_TYPE_BUFFER, &storage->join_in_bufs[0] };
            storage->join_in_tds[1] = { DML_TENSOR_TYPE_BUFFER, &storage->join_in_bufs[1] };
            storage->join_desc.InputCount = 2;
            storage->join_desc.InputTensors = storage->join_in_tds.data();
            storage->join_desc.OutputTensor = &self.output_tensor_descs[0];
            storage->join_desc.Axis = 0;
        };
        join_node.FixupPointers();
        result.sub_nodes.push_back(std::move(join_node));
        join_idx = sub_idx++;

        // Per-output routing: output(0) from MHA, present(1) from Join
        result.output_source = { {mha_idx, 0}, {join_idx, 0} };
    }

    return result;
}

// ---------------------------------------------------------------------------
// EmbedLayerNormalization → Gather+Add+MVN SubNode graph
// ORT ref: DmlOperatorEmbedLayerNormalization.cpp
// Outputs from 3 different sub_nodes → uses output_source for routing.
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateEmbedLayerNormalization(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 7 || outputs.empty()) return std::nullopt;

    enum OnnxInput : size_t { inputIdsIdx=0, segmentIdsIdx=1, wordEmbIdx=2,
                              posEmbIdx=3, segEmbIdx=4, gammaIdx=5, betaIdx=6,
                              maskIdx=7, positionIdsIdx=8 };
    enum OnnxOutput : size_t { outputIdx=0, maskIndexIdx=1, embSumIdx=2 };

    auto valid = [&](size_t idx) {
        return idx < inputs.size() && !inputs[idx].empty() && value_shapes.count(inputs[idx]);
    };
    auto out_valid = [&](size_t idx) {
        return idx < outputs.size() && !outputs[idx].empty() && value_shapes.count(outputs[idx]);
    };

    auto* ids_info   = LookupShape(value_shapes, inputs[inputIdsIdx]);
    auto* wemb_info  = LookupShape(value_shapes, inputs[wordEmbIdx]);
    auto* pemb_info  = LookupShape(value_shapes, inputs[posEmbIdx]);
    auto* gamma_info = LookupShape(value_shapes, inputs[gammaIdx]);
    auto* beta_info  = LookupShape(value_shapes, inputs[betaIdx]);
    if (!ids_info || !wemb_info || !pemb_info || !gamma_info || !beta_info)
        return std::nullopt;

    bool hasSegment = valid(segEmbIdx) && valid(segmentIdsIdx);
    bool hasMask = valid(maskIdx);
    bool hasPositionIds = valid(positionIdsIdx);
    bool hasMaskIndex = out_valid(maskIndexIdx);
    bool hasEmbSum = out_valid(embSumIdx);

    OrtNodeAdapter adapter(node, ort_api);
    float epsilon = adapter.GetAttributeFloat("epsilon", 1e-5f);

    // Dimensions (4D padding matching ORT exactly).
    DML_TENSOR_DATA_TYPE indicesType = ids_info->data_type;
    DML_TENSOR_DATA_TYPE valuesType  = wemb_info->data_type;

    // Extract B, S from input_ids (originally 2D [B,S]).
    uint32_t ids_rank = ids_info->original_rank ? ids_info->original_rank
                      : static_cast<uint32_t>(ids_info->sizes.size());
    size_t ids_off = ids_info->sizes.size() >= ids_rank
                   ? ids_info->sizes.size() - ids_rank : 0;
    uint32_t batchSize, sequenceLength;
    if (ids_rank == 2) {
        batchSize = ids_info->sizes[ids_off];
        sequenceLength = ids_info->sizes[ids_off + 1];
    } else {
        batchSize = 1;
        sequenceLength = ids_info->sizes.back();
    }

    // Word embedding: [V, H]
    uint32_t wemb_rank = wemb_info->original_rank ? wemb_info->original_rank
                       : static_cast<uint32_t>(wemb_info->sizes.size());
    size_t wemb_off = wemb_info->sizes.size() >= wemb_rank
                    ? wemb_info->sizes.size() - wemb_rank : 0;
    uint32_t hiddenSize = wemb_info->sizes[wemb_off + wemb_rank - 1];

    // 4D shapes (matching ORT).
    std::vector<uint32_t> ids_4d     = {1, 1, batchSize, sequenceLength};
    std::vector<uint32_t> wemb_4d    = {1, 1, wemb_info->sizes[wemb_off], hiddenSize};
    std::vector<uint32_t> pemb_4d    = {1, 1, pemb_info->sizes[pemb_info->sizes.size() >= 2 ? pemb_info->sizes.size() - 2 : 0], hiddenSize};
    // ORT Gather output shape matches the ONNX output shape [B,S,H] padded to 4D.
    std::vector<uint32_t> gathered_4d = PadToMinDims({batchSize, sequenceLength, hiddenSize});

    // Storage struct.
    struct ELNStorage {
        DML_GATHER_OPERATOR_DESC word_gather_desc{};
        DML_GATHER_OPERATOR_DESC pos_gather_desc{};
        DML_GATHER_OPERATOR_DESC seg_gather_desc{};
        DML_FILL_VALUE_SEQUENCE_OPERATOR_DESC seq_desc{};
        DML_ELEMENT_WISE_ADD_OPERATOR_DESC add_desc{};
        DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC mvn_desc{};
        DML_FILL_VALUE_CONSTANT_OPERATOR_DESC ones_desc{};
        DML_FILL_VALUE_CONSTANT_OPERATOR_DESC zeros_desc{};
        DML_ELEMENT_WISE_LOGICAL_EQUALS_OPERATOR_DESC equals_desc{};
        DML_REDUCE_OPERATOR_DESC reduce_desc{};
        DmlTensorInfo seq_info;
        DML_BUFFER_TENSOR_DESC seq_buf{};
        DML_TENSOR_DESC seq_td{};
        DmlTensorInfo scalar_info;
        DML_BUFFER_TENSOR_DESC scalar_buf{};
        DML_TENSOR_DESC scalar_td{};
        DmlTensorInfo equals_out_info;
        DML_BUFFER_TENSOR_DESC equals_out_buf{};
        DML_TENSOR_DESC equals_out_td{};
        DmlTensorInfo sparse_mask_info;
        DML_BUFFER_TENSOR_DESC sparse_mask_buf{};
        DML_TENSOR_DESC sparse_mask_td{};
        DmlTensorInfo broadcast_ones_info;
        DML_BUFFER_TENSOR_DESC broadcast_ones_buf{};
        DML_TENSOR_DESC broadcast_ones_td{};
        std::vector<uint32_t> mvn_axes;
        std::vector<uint32_t> reduce_axes;
    };
    auto storage = std::make_shared<ELNStorage>();

    // Output tensor.
    auto* out_info = LookupShape(value_shapes, outputs[outputIdx]);
    auto output_tensor = out_info
        ? MakeTensorInfo(out_info->sizes, valuesType)
        : MakeTensorInfo(gathered_4d, valuesType);

    // mask_index output: [B] → [1, 1, B, 1] in 4D
    DmlTensorInfo mask_index_tensor{};
    if (hasMaskIndex) {
        auto* mi_info = LookupShape(value_shapes, outputs[maskIndexIdx]);
        mask_index_tensor = mi_info
            ? MakeTensorInfo(mi_info->sizes, indicesType)
            : MakeTensorInfo({1, 1, batchSize, 1}, indicesType);
    }

    // embedding_sum output: same shape as gathered.
    DmlTensorInfo emb_sum_tensor{};
    if (hasEmbSum) {
        auto* es_info = LookupShape(value_shapes, outputs[embSumIdx]);
        emb_sum_tensor = es_info
            ? MakeTensorInfo(es_info->sizes, valuesType)
            : MakeTensorInfo(gathered_4d, valuesType);
    }

    // Gathered tensor (intermediate): [1, 1, B*S, H]
    auto gathered_tensor = MakeTensorInfo(gathered_4d, valuesType);

    // Input_ids tensor (4D).
    auto ids_tensor = MakeTensorInfo(ids_4d, indicesType);

    // Word embedding tensor (4D).
    auto wemb_tensor = MakeTensorInfo(wemb_4d, valuesType);

    // --- Build TranslatedOp: Primary = Gather(word_emb, input_ids) ---
    TranslatedOp result;
    result.input_tensors = { wemb_tensor, ids_tensor };
    result.input_buffer_descs = { wemb_tensor.ToBufferDesc(), ids_tensor.ToBufferDesc() };
    result.input_name_reorder = { wordEmbIdx, inputIdsIdx };
    result.primary_input_count = 2;

    // Append remaining inputs for graph_inputs wiring.
    auto pemb_tensor = MakeTensorInfo(pemb_4d, valuesType);
    result.input_tensors.push_back(pemb_tensor);
    result.input_buffer_descs.push_back(pemb_tensor.ToBufferDesc());
    result.input_name_reorder.push_back(posEmbIdx);

    DmlTensorInfo seg_emb_tensor{}, seg_ids_tensor{};
    if (hasSegment) {
        auto* semb_info = LookupShape(value_shapes, inputs[segEmbIdx]);
        if (!semb_info) return std::nullopt;
        std::vector<uint32_t> semb_4d = {1, 1, semb_info->sizes[semb_info->sizes.size() >= 2 ? semb_info->sizes.size() - 2 : 0], hiddenSize};
        seg_emb_tensor = MakeTensorInfo(semb_4d, valuesType);
        result.input_tensors.push_back(seg_emb_tensor);
        result.input_buffer_descs.push_back(seg_emb_tensor.ToBufferDesc());
        result.input_name_reorder.push_back(segEmbIdx);

        auto* sids_info = LookupShape(value_shapes, inputs[segmentIdsIdx]);
        if (!sids_info) return std::nullopt;
        seg_ids_tensor = MakeTensorInfo(ids_4d, indicesType);
        result.input_tensors.push_back(seg_ids_tensor);
        result.input_buffer_descs.push_back(seg_ids_tensor.ToBufferDesc());
        result.input_name_reorder.push_back(segmentIdsIdx);
    }

    auto gamma_tensor = MakeTensorInfo(gamma_info->sizes, gamma_info->data_type);
    result.input_tensors.push_back(gamma_tensor);
    result.input_buffer_descs.push_back(gamma_tensor.ToBufferDesc());
    result.input_name_reorder.push_back(gammaIdx);

    auto beta_tensor = MakeTensorInfo(beta_info->sizes, beta_info->data_type);
    result.input_tensors.push_back(beta_tensor);
    result.input_buffer_descs.push_back(beta_tensor.ToBufferDesc());
    result.input_name_reorder.push_back(betaIdx);

    DmlTensorInfo mask_input_tensor{};
    if (hasMask) {
        auto* mask_info = LookupShape(value_shapes, inputs[maskIdx]);
        if (!mask_info) return std::nullopt;
        mask_input_tensor = MakeTensorInfo(ids_4d, mask_info->data_type);
        result.input_tensors.push_back(mask_input_tensor);
        result.input_buffer_descs.push_back(mask_input_tensor.ToBufferDesc());
        result.input_name_reorder.push_back(maskIdx);
    }

    DmlTensorInfo pos_ids_tensor{};
    if (hasPositionIds) {
        auto* pids_info = LookupShape(value_shapes, inputs[positionIdsIdx]);
        if (!pids_info) return std::nullopt;
        // Broadcast [1,S] to [B,S] if needed.
        uint32_t pids_rank = pids_info->original_rank ? pids_info->original_rank
                           : static_cast<uint32_t>(pids_info->sizes.size());
        size_t pids_off = pids_info->sizes.size() >= pids_rank
                        ? pids_info->sizes.size() - pids_rank : 0;
        uint32_t pids_batch = pids_rank >= 2 ? pids_info->sizes[pids_off] : 1;
        if (pids_batch == 1 && batchSize > 1) {
            std::vector<uint32_t> pids_strides = {0, 0, 0, 1};
            uint64_t pids_bytes = ComputeAlignedTotalBytes({1, 1, 1, sequenceLength}, indicesType);
            pos_ids_tensor = MakeTensorInfoWithStrides(ids_4d, pids_strides, indicesType, pids_bytes);
        } else {
            pos_ids_tensor = MakeTensorInfo(ids_4d, indicesType);
        }
        result.input_tensors.push_back(pos_ids_tensor);
        result.input_buffer_descs.push_back(pos_ids_tensor.ToBufferDesc());
        result.input_name_reorder.push_back(positionIdsIdx);
    }

    result.input_tensor_descs.resize(result.input_tensors.size());

    // Outputs.
    result.output_tensors = { output_tensor };
    result.output_buffer_descs = { output_tensor.ToBufferDesc() };
    if (hasMaskIndex) {
        result.output_tensors.push_back(mask_index_tensor);
        result.output_buffer_descs.push_back(mask_index_tensor.ToBufferDesc());
    }
    if (hasEmbSum) {
        result.output_tensors.push_back(emb_sum_tensor);
        result.output_buffer_descs.push_back(emb_sum_tensor.ToBufferDesc());
    }
    result.output_tensor_descs.resize(result.output_tensors.size());

    // Primary fixup: Gather(word_emb, input_ids)
    // The primary's output is the intermediate gathered word embeddings, not
    // the final ONNX output.  Store the intermediate tensor info in storage
    // so the framework creates the right intermediate buffer size.
    struct GatherOutHelper {
        DmlTensorInfo info;
        DML_BUFFER_TENSOR_DESC buf{};
        DML_TENSOR_DESC td{};
    };
    auto gather_out = std::make_shared<GatherOutHelper>();
    gather_out->info = gathered_tensor;
    gather_out->buf  = gathered_tensor.ToBufferDesc();
    gather_out->td   = { DML_TENSOR_TYPE_BUFFER, &gather_out->buf };

    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_GATHER, &storage->word_gather_desc };
    result.fixup = [storage, gather_out](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        gather_out->buf.Sizes = gather_out->info.sizes.data();
        gather_out->buf.Strides = gather_out->info.strides.empty() ? nullptr : gather_out->info.strides.data();
        gather_out->td = { DML_TENSOR_TYPE_BUFFER, &gather_out->buf };
        storage->word_gather_desc.InputTensor  = &self.input_tensor_descs[0];
        storage->word_gather_desc.IndicesTensor = &self.input_tensor_descs[1];
        storage->word_gather_desc.OutputTensor = &gather_out->td;
        storage->word_gather_desc.Axis = 2;
        storage->word_gather_desc.IndexDimensions = 2;
    };
    result.FixupPointers();

    // --- Sub_nodes ---
    int sub_idx = 0;

    // Optional: FillValueSequence for position indices when no position_ids.
    int seq_node_idx = -1;
    if (!hasPositionIds) {
        std::vector<uint32_t> seq_shape = {1, 1, 1, sequenceLength};
        storage->seq_info = MakeTensorInfo(seq_shape, indicesType);

        SubNode seq_node;
        seq_node.output_tensors = { storage->seq_info };
        seq_node.output_buffer_descs = { storage->seq_info.ToBufferDesc() };
        seq_node.output_tensor_descs.resize(1);
        seq_node.desc_storage = storage;
        seq_node.op_desc = { DML_OPERATOR_FILL_VALUE_SEQUENCE, &storage->seq_desc };
        seq_node.fixup = [storage](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->seq_desc.OutputTensor = &self.output_tensor_descs[0];
            storage->seq_desc.ValueDataType = storage->seq_info.data_type;
            storage->seq_desc.ValueStart.Int32 = 0;
            storage->seq_desc.ValueDelta.Int32 = 1;
        };
        seq_node.FixupPointers();
        result.sub_nodes.push_back(std::move(seq_node));
        seq_node_idx = sub_idx++;
    }

    // Sub_node: Gather(position_emb, position_ids or sequence)
    SubNode pos_gather;
    pos_gather.input_tensors  = { pemb_tensor, hasPositionIds ? pos_ids_tensor : storage->seq_info };
    pos_gather.output_tensors = { gathered_tensor };
    pos_gather.input_buffer_descs  = { pemb_tensor.ToBufferDesc(),
        (hasPositionIds ? pos_ids_tensor : storage->seq_info).ToBufferDesc() };
    pos_gather.output_buffer_descs = { gathered_tensor.ToBufferDesc() };
    pos_gather.input_tensor_descs.resize(2);
    pos_gather.output_tensor_descs.resize(1);
    pos_gather.input_from = { {-2, 0}, hasPositionIds ? std::pair<int,int>{-2, 0} : std::pair<int,int>{seq_node_idx, 0} };
    pos_gather.graph_inputs.push_back({posEmbIdx, 0});
    if (hasPositionIds)
        pos_gather.graph_inputs.push_back({positionIdsIdx, 1});
    pos_gather.desc_storage = storage;
    pos_gather.op_desc = { DML_OPERATOR_GATHER, &storage->pos_gather_desc };
    pos_gather.fixup = [storage](SubNode& self) {
        RebuildSubNodePointers(self);
        storage->pos_gather_desc.InputTensor = &self.input_tensor_descs[0];
        storage->pos_gather_desc.IndicesTensor = &self.input_tensor_descs[1];
        storage->pos_gather_desc.OutputTensor = &self.output_tensor_descs[0];
        storage->pos_gather_desc.Axis = 2;
        storage->pos_gather_desc.IndexDimensions = 2;
    };
    pos_gather.FixupPointers();
    result.sub_nodes.push_back(std::move(pos_gather));
    int pos_gather_idx = sub_idx++;

    // Optional: Gather(segment_emb, segment_ids)
    int seg_gather_idx = -1;
    if (hasSegment) {
        SubNode seg_gather;
        seg_gather.input_tensors  = { seg_emb_tensor, seg_ids_tensor };
        seg_gather.output_tensors = { gathered_tensor };
        seg_gather.input_buffer_descs  = { seg_emb_tensor.ToBufferDesc(), seg_ids_tensor.ToBufferDesc() };
        seg_gather.output_buffer_descs = { gathered_tensor.ToBufferDesc() };
        seg_gather.input_tensor_descs.resize(2);
        seg_gather.output_tensor_descs.resize(1);
        seg_gather.input_from = { {-2, 0}, {-2, 0} };
        seg_gather.graph_inputs = { {segEmbIdx, 0}, {segmentIdsIdx, 1} };
        seg_gather.desc_storage = storage;
        seg_gather.op_desc = { DML_OPERATOR_GATHER, &storage->seg_gather_desc };
        seg_gather.fixup = [storage](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->seg_gather_desc.InputTensor = &self.input_tensor_descs[0];
            storage->seg_gather_desc.IndicesTensor = &self.input_tensor_descs[1];
            storage->seg_gather_desc.OutputTensor = &self.output_tensor_descs[0];
            storage->seg_gather_desc.Axis = 2;
            storage->seg_gather_desc.IndexDimensions = 2;
        };
        seg_gather.FixupPointers();
        result.sub_nodes.push_back(std::move(seg_gather));
        seg_gather_idx = sub_idx++;
    }

    // Sub_node: Add(word_gathered, pos_gathered) → word+pos
    SubNode wp_add;
    wp_add.input_tensors  = { gathered_tensor, gathered_tensor };
    wp_add.output_tensors = { gathered_tensor };
    wp_add.input_buffer_descs  = { gathered_tensor.ToBufferDesc(), gathered_tensor.ToBufferDesc() };
    wp_add.output_buffer_descs = { gathered_tensor.ToBufferDesc() };
    wp_add.input_tensor_descs.resize(2);
    wp_add.output_tensor_descs.resize(1);
    wp_add.input_from = { {-1, 0}, {pos_gather_idx, 0} };
    wp_add.desc_storage = storage;
    wp_add.op_desc = { DML_OPERATOR_ELEMENT_WISE_ADD, &storage->add_desc };
    wp_add.fixup = [storage](SubNode& self) {
        RebuildSubNodePointers(self);
        storage->add_desc.ATensor = &self.input_tensor_descs[0];
        storage->add_desc.BTensor = &self.input_tensor_descs[1];
        storage->add_desc.OutputTensor = &self.output_tensor_descs[0];
    };
    wp_add.FixupPointers();
    result.sub_nodes.push_back(std::move(wp_add));
    int wp_add_idx = sub_idx++;

    // Optional: Add(word+pos, seg_gathered) → word+pos+seg
    int embedding_sum_idx = wp_add_idx;
    if (hasSegment) {
        auto wps_add_store = std::make_shared<DML_ELEMENT_WISE_ADD_OPERATOR_DESC>();
        SubNode wps_add;
        wps_add.input_tensors  = { gathered_tensor, gathered_tensor };
        wps_add.output_tensors = { gathered_tensor };
        wps_add.input_buffer_descs  = { gathered_tensor.ToBufferDesc(), gathered_tensor.ToBufferDesc() };
        wps_add.output_buffer_descs = { gathered_tensor.ToBufferDesc() };
        wps_add.input_tensor_descs.resize(2);
        wps_add.output_tensor_descs.resize(1);
        wps_add.input_from = { {wp_add_idx, 0}, {seg_gather_idx, 0} };
        wps_add.desc_storage = wps_add_store;
        wps_add.op_desc = { DML_OPERATOR_ELEMENT_WISE_ADD, wps_add_store.get() };
        wps_add.fixup = [wps_add_store](SubNode& self) {
            RebuildSubNodePointers(self);
            wps_add_store->ATensor = &self.input_tensor_descs[0];
            wps_add_store->BTensor = &self.input_tensor_descs[1];
            wps_add_store->OutputTensor = &self.output_tensor_descs[0];
        };
        wps_add.FixupPointers();
        result.sub_nodes.push_back(std::move(wps_add));
        embedding_sum_idx = sub_idx++;
    }

    // Sub_node: MVN1(embedding_sum, gamma, beta) → output(0)
    storage->mvn_axes = { static_cast<uint32_t>(gathered_4d.size() - 1) };
    SubNode mvn_node;
    mvn_node.input_tensors  = { gathered_tensor, gamma_tensor, beta_tensor };
    mvn_node.output_tensors = { output_tensor };
    mvn_node.input_buffer_descs  = { gathered_tensor.ToBufferDesc(), gamma_tensor.ToBufferDesc(), beta_tensor.ToBufferDesc() };
    mvn_node.output_buffer_descs = { output_tensor.ToBufferDesc() };
    mvn_node.input_tensor_descs.resize(3);
    mvn_node.output_tensor_descs.resize(1);
    mvn_node.input_from = { {embedding_sum_idx, 0}, {-2, 0}, {-2, 0} };
    mvn_node.graph_inputs = { {gammaIdx, 1}, {betaIdx, 2} };
    mvn_node.desc_storage = storage;
    mvn_node.op_desc = { DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1, &storage->mvn_desc };
    float l_epsilon = epsilon;
    mvn_node.fixup = [storage, l_epsilon](SubNode& self) {
        RebuildSubNodePointers(self);
        storage->mvn_desc.InputTensor = &self.input_tensor_descs[0];
        storage->mvn_desc.ScaleTensor = &self.input_tensor_descs[1];
        storage->mvn_desc.BiasTensor  = &self.input_tensor_descs[2];
        storage->mvn_desc.OutputTensor = &self.output_tensor_descs[0];
        storage->mvn_desc.Axes = storage->mvn_axes.data();
        storage->mvn_desc.AxisCount = static_cast<UINT>(storage->mvn_axes.size());
        storage->mvn_desc.NormalizeVariance = true;
        storage->mvn_desc.Epsilon = l_epsilon;
        storage->mvn_desc.FusedActivation = nullptr;
    };
    mvn_node.FixupPointers();
    result.sub_nodes.push_back(std::move(mvn_node));
    int mvn_idx = sub_idx++;

    // Mask index chain.
    int mask_index_sub_idx = -1;
    if (hasMask) {
        // FillValueConstant(ones)
        std::vector<uint32_t> scalar_4d(4, 1u);
        storage->scalar_info = MakeTensorInfo(scalar_4d, indicesType);

        SubNode ones_node;
        ones_node.output_tensors = { storage->scalar_info };
        ones_node.output_buffer_descs = { storage->scalar_info.ToBufferDesc() };
        ones_node.output_tensor_descs.resize(1);
        ones_node.desc_storage = storage;
        ones_node.op_desc = { DML_OPERATOR_FILL_VALUE_CONSTANT, &storage->ones_desc };
        ones_node.fixup = [storage](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->ones_desc.OutputTensor = &self.output_tensor_descs[0];
            storage->ones_desc.ValueDataType = storage->scalar_info.data_type;
            storage->ones_desc.Value.Int32 = 1;
        };
        ones_node.FixupPointers();
        result.sub_nodes.push_back(std::move(ones_node));
        int ones_idx = sub_idx++;

        // Broadcast ones via stride=0.
        std::vector<uint32_t> bcast_strides(4, 0u);
        uint64_t ones_bytes = ComputeAlignedTotalBytes(scalar_4d, indicesType);
        storage->broadcast_ones_info = MakeTensorInfoWithStrides(ids_4d, bcast_strides, indicesType, ones_bytes);

        // Equals(mask, broadcast(ones))
        storage->equals_out_info = MakeTensorInfo(ids_4d, DML_TENSOR_DATA_TYPE_UINT32);

        SubNode equals_node;
        equals_node.input_tensors  = { mask_input_tensor, storage->broadcast_ones_info };
        equals_node.output_tensors = { storage->equals_out_info };
        equals_node.input_buffer_descs  = { mask_input_tensor.ToBufferDesc(), storage->broadcast_ones_info.ToBufferDesc() };
        equals_node.output_buffer_descs = { storage->equals_out_info.ToBufferDesc() };
        equals_node.input_tensor_descs.resize(2);
        equals_node.output_tensor_descs.resize(1);
        equals_node.input_from = { {-2, 0}, {ones_idx, 0} };
        equals_node.graph_inputs = { {maskIdx, 0} };
        equals_node.desc_storage = storage;
        equals_node.op_desc = { DML_OPERATOR_ELEMENT_WISE_LOGICAL_EQUALS, &storage->equals_desc };
        equals_node.fixup = [storage](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->equals_desc.ATensor = &self.input_tensor_descs[0];
            storage->equals_desc.BTensor = &self.input_tensor_descs[1];
            storage->equals_desc.OutputTensor = &self.output_tensor_descs[0];
        };
        equals_node.FixupPointers();
        result.sub_nodes.push_back(std::move(equals_node));
        int equals_idx = sub_idx++;

        // Reduce(equals_output, SUM, axis=3) → mask_index
        // Reinterpret uint32 as int32 for reduce input.
        storage->sparse_mask_info = MakeTensorInfo(ids_4d, indicesType);
        std::vector<uint32_t> reduced_shape = {1, 1, batchSize, 1};
        storage->reduce_axes = { 3 };

        SubNode reduce_node;
        reduce_node.input_tensors  = { storage->sparse_mask_info };
        reduce_node.output_tensors = { mask_index_tensor };
        reduce_node.input_buffer_descs  = { storage->sparse_mask_info.ToBufferDesc() };
        reduce_node.output_buffer_descs = { mask_index_tensor.ToBufferDesc() };
        reduce_node.input_tensor_descs.resize(1);
        reduce_node.output_tensor_descs.resize(1);
        reduce_node.input_from = { {equals_idx, 0} };
        reduce_node.desc_storage = storage;
        reduce_node.op_desc = { DML_OPERATOR_REDUCE, &storage->reduce_desc };
        reduce_node.fixup = [storage](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->reduce_desc.InputTensor = &self.input_tensor_descs[0];
            storage->reduce_desc.OutputTensor = &self.output_tensor_descs[0];
            storage->reduce_desc.Function = DML_REDUCE_FUNCTION_SUM;
            storage->reduce_desc.Axes = storage->reduce_axes.data();
            storage->reduce_desc.AxisCount = static_cast<UINT>(storage->reduce_axes.size());
        };
        reduce_node.FixupPointers();
        result.sub_nodes.push_back(std::move(reduce_node));
        mask_index_sub_idx = sub_idx++;
    } else if (hasMaskIndex) {
        // No mask but mask_index output requested → fill with zeros.
        SubNode zeros_node;
        zeros_node.output_tensors = { mask_index_tensor };
        zeros_node.output_buffer_descs = { mask_index_tensor.ToBufferDesc() };
        zeros_node.output_tensor_descs.resize(1);
        zeros_node.desc_storage = storage;
        zeros_node.op_desc = { DML_OPERATOR_FILL_VALUE_CONSTANT, &storage->zeros_desc };
        zeros_node.fixup = [storage](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->zeros_desc.OutputTensor = &self.output_tensor_descs[0];
            storage->zeros_desc.ValueDataType = DML_TENSOR_DATA_TYPE_INT32;
            storage->zeros_desc.Value.Int32 = 0;
        };
        zeros_node.FixupPointers();
        result.sub_nodes.push_back(std::move(zeros_node));
        mask_index_sub_idx = sub_idx++;
    }

    // Build output_source routing.
    // output(0) → MVN
    // mask_index(1) → Reduce or Zeros (if present)
    // embedding_sum(2) → last Add (if present)
    result.output_source.push_back({mvn_idx, 0});
    if (hasMaskIndex)
        result.output_source.push_back({mask_index_sub_idx, 0});
    if (hasEmbSum) {
        // embedding_sum is at output index 1 or 2 depending on mask_index presence.
        result.output_source.push_back({embedding_sum_idx, 0});
    }

    return result;
}

// ---------------------------------------------------------------------------
// MultiHeadAttention → DML_OPERATOR_MULTIHEAD_ATTENTION
// ORT ref: DmlOperatorMultiHeadAttention.cpp
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateMultiHeadAttention(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    // Input indices.
    enum { queryIdx=0, keyIdx=1, valueIdx=2, biasIdx=3,
           maskIdx=4, attBiasIdx=5, pastKeyIdx=6, pastValueIdx=7 };

    auto valid = [&](size_t idx) {
        return idx < inputs.size() && !inputs[idx].empty() && value_shapes.count(inputs[idx]);
    };

    auto* q_info = LookupShape(value_shapes, inputs[queryIdx]);
    if (!q_info) return std::nullopt;

    uint32_t q_rank = q_info->original_rank ? q_info->original_rank
                    : static_cast<uint32_t>(q_info->sizes.size());

    bool stackedQkv = (q_rank == 5);
    bool stackedKv = valid(keyIdx) && [&]{
        auto* k = LookupShape(value_shapes, inputs[keyIdx]);
        uint32_t kr = k ? (k->original_rank ? k->original_rank : static_cast<uint32_t>(k->sizes.size())) : 0;
        return kr == 5;
    }();
    bool keyValueIsPast = !stackedKv && valid(keyIdx) && [&]{
        auto* k = LookupShape(value_shapes, inputs[keyIdx]);
        uint32_t kr = k ? (k->original_rank ? k->original_rank : static_cast<uint32_t>(k->sizes.size())) : 0;
        return kr == 4;
    }();
    bool hasKey = !stackedKv && !keyValueIsPast && valid(keyIdx);
    bool hasValue = valid(valueIdx) && !keyValueIsPast;
    bool hasBias = valid(biasIdx);
    bool hasMask = valid(maskIdx);
    bool hasAttBias = valid(attBiasIdx);
    bool hasPastKey = keyValueIsPast || (valid(pastKeyIdx) && [&]{
        auto* pk = LookupShape(value_shapes, inputs[pastKeyIdx]);
        if (!pk) return false;
        size_t pad = pk->sizes.size() >= 4 ? pk->sizes.size() - 4 : 0;
        return pk->sizes[pad + 2] != 0;
    }());
    bool hasPastValue = keyValueIsPast || (valid(pastValueIdx) && [&]{
        auto* pv = LookupShape(value_shapes, inputs[pastValueIdx]);
        if (!pv) return false;
        size_t pad = pv->sizes.size() >= 4 ? pv->sizes.size() - 4 : 0;
        return pv->sizes[pad + 2] != 0;
    }());

    // Output validity.
    bool hasPresentKey = outputs.size() > 1 && !outputs[1].empty() && value_shapes.count(outputs[1]);
    bool hasPresentValue = outputs.size() > 2 && !outputs[2].empty() && value_shapes.count(outputs[2]);

    // Extract dimensions from padded shapes.
    auto get_rank_offset = [](const DmlTensorInfo* info) -> size_t {
        uint32_t r = info->original_rank ? info->original_rank : static_cast<uint32_t>(info->sizes.size());
        return info->sizes.size() >= r ? info->sizes.size() - r : 0;
    };

    OrtNodeAdapter adapter(node, ort_api);
    uint32_t numHeads = static_cast<uint32_t>(adapter.GetAttributeInt("num_heads", 0));
    if (numHeads == 0) return std::nullopt;

    size_t q_off = get_rank_offset(q_info);
    uint32_t batchSize, sequenceLength, hiddenSize, headSize;
    if (stackedQkv) {
        // [B, S, numHeads, 3, headSize] — 5D
        batchSize = q_info->sizes[q_off];
        sequenceLength = q_info->sizes[q_off + 1];
        headSize = q_info->sizes[q_off + 4];
        hiddenSize = numHeads * headSize;
    } else {
        // [B, S, hiddenSize] — 3D
        batchSize = q_info->sizes[q_off];
        sequenceLength = q_info->sizes[q_off + 1];
        hiddenSize = q_info->sizes[q_off + 2];
        headSize = hiddenSize / numHeads;
    }

    uint32_t kvSequenceLength;
    if (hasKey) {
        auto* k_info = LookupShape(value_shapes, inputs[keyIdx]);
        size_t k_off = get_rank_offset(k_info);
        kvSequenceLength = k_info->sizes[k_off + 1];
    } else if (stackedKv) {
        auto* k_info = LookupShape(value_shapes, inputs[keyIdx]);
        size_t k_off = get_rank_offset(k_info);
        kvSequenceLength = k_info->sizes[k_off + 1];
    } else if (hasPastKey) {
        auto* pk = LookupShape(value_shapes, inputs[keyValueIsPast ? keyIdx : pastKeyIdx]);
        size_t pk_off = get_rank_offset(pk);
        kvSequenceLength = pk->sizes[pk_off + 2];
    } else {
        kvSequenceLength = sequenceLength;
    }

    uint32_t vHiddenSize = hiddenSize;
    if (hasValue) {
        auto* v_info = LookupShape(value_shapes, inputs[valueIdx]);
        size_t v_off = get_rank_offset(v_info);
        vHiddenSize = v_info->sizes[v_off + 2];
    }

    uint32_t pastSequenceLength = 0;
    if (hasPastKey) {
        auto* pk = LookupShape(value_shapes, inputs[keyValueIsPast ? keyIdx : pastKeyIdx]);
        size_t pk_off = get_rank_offset(pk);
        pastSequenceLength = pk->sizes[pk_off + 2];
    }
    uint32_t totalSequenceLength = kvSequenceLength + pastSequenceLength;

    float scale = adapter.GetAttributeFloat("scale", 0.0f);
    if (scale == 0.0f) scale = 1.0f / std::sqrt(static_cast<float>(headSize));
    float maskFilterValue = adapter.GetAttributeFloat("mask_filter_value", -10000.0f);

    // Mask type detection.
    DML_MULTIHEAD_ATTENTION_MASK_TYPE maskType = DML_MULTIHEAD_ATTENTION_MASK_TYPE_NONE;
    DmlTensorInfo mask_tensor{};
    if (hasMask) {
        auto* mask_info = LookupShape(value_shapes, inputs[maskIdx]);
        if (!mask_info) return std::nullopt;
        uint32_t mask_rank = mask_info->original_rank ? mask_info->original_rank
                           : static_cast<uint32_t>(mask_info->sizes.size());
        if (mask_rank == 1) {
            uint32_t mask_size = mask_info->sizes.back();
            if (mask_size == batchSize) {
                maskType = DML_MULTIHEAD_ATTENTION_MASK_TYPE_KEY_SEQUENCE_LENGTH;
                mask_tensor = MakeTensorInfo({1, batchSize}, mask_info->data_type);
            } else {
                maskType = DML_MULTIHEAD_ATTENTION_MASK_TYPE_KEY_QUERY_SEQUENCE_LENGTH_START_END;
                mask_tensor = MakeTensorInfo(mask_info->sizes, mask_info->data_type);
            }
        } else {
            maskType = DML_MULTIHEAD_ATTENTION_MASK_TYPE_BOOLEAN;
            // Determine actual and desired shapes for broadcast.
            size_t m_off = get_rank_offset(mask_info);
            std::vector<uint32_t> actual_shape, desired_shape;
            if (mask_rank == 2) {
                actual_shape = {batchSize, 1, 1, kvSequenceLength};
                desired_shape = {batchSize, numHeads, sequenceLength, kvSequenceLength};
            } else if (mask_rank == 3) {
                actual_shape = {batchSize, 1, sequenceLength, totalSequenceLength};
                desired_shape = {batchSize, numHeads, sequenceLength, totalSequenceLength};
            } else {
                actual_shape = {batchSize, numHeads, sequenceLength, totalSequenceLength};
                desired_shape = actual_shape;
            }
            // Build broadcast tensor with stride=0 for expanded dims.
            auto strides = ComputePackedStrides(actual_shape);
            for (size_t d = 0; d < 4; ++d) {
                if (actual_shape[d] == 1 && desired_shape[d] > 1)
                    strides[d] = 0;
            }
            uint64_t mask_bytes = ComputeAlignedTotalBytes(actual_shape, mask_info->data_type);
            // Correct mask_bytes for the real data size.
            if (mask_info->total_bytes > 0) mask_bytes = mask_info->total_bytes;
            mask_tensor = MakeTensorInfoWithStrides(desired_shape, strides,
                mask_info->data_type, mask_bytes);
        }
    }

    // Build input_tensors in DML schema order, skipping null slots.
    // Track which DML schema slot each entry corresponds to.
    struct MHAStorage {
        DML_MULTIHEAD_ATTENTION_OPERATOR_DESC desc{};
        // Track positions of each DML input in input_tensor_descs.
        int slot_query = -1, slot_key = -1, slot_value = -1;
        int slot_stacked_kv = -1, slot_stacked_qkv = -1;
        int slot_bias = -1, slot_mask = -1, slot_att_bias = -1;
        int slot_past_key = -1, slot_past_value = -1;
        int out_present_key = -1, out_present_value = -1;
    };
    auto storage = std::make_shared<MHAStorage>();

    TranslatedOp result;
    int idx = 0;
    // push_input records both the ONNX input index (for input_name_reorder) and
    // the DML schema slot (for dml_input_slot_indices), so edge wiring uses the
    // correct ToNodeInputIndex even when schema slots are non-contiguous.
    auto push_input = [&](const DmlTensorInfo& t, size_t onnx_idx, size_t dml_schema_slot) {
        result.input_tensors.push_back(t);
        result.input_buffer_descs.push_back(t.ToBufferDesc());
        result.input_name_reorder.push_back(onnx_idx);
        result.dml_input_slot_indices.push_back(dml_schema_slot);
        return idx++;
    };

    // DML slot 0: QueryTensor
    if (!stackedQkv) {
        auto qt = MakeTensorInfo(q_info->sizes, q_info->data_type);
        storage->slot_query = push_input(qt, queryIdx, 0);
    }

    // DML slot 1: KeyTensor
    if (hasKey) {
        auto* k_info = LookupShape(value_shapes, inputs[keyIdx]);
        auto kt = MakeTensorInfo(k_info->sizes, k_info->data_type);
        storage->slot_key = push_input(kt, keyIdx, 1);
    }

    // DML slot 2: ValueTensor
    if (hasValue) {
        auto* v_info = LookupShape(value_shapes, inputs[valueIdx]);
        auto vt = MakeTensorInfo(v_info->sizes, v_info->data_type);
        storage->slot_value = push_input(vt, valueIdx, 2);
    }

    // DML slot 3: StackedQueryKeyTensor — never used
    // DML slot 4: StackedKeyValueTensor
    if (stackedKv) {
        auto* k_info = LookupShape(value_shapes, inputs[keyIdx]);
        auto skvt = MakeTensorInfo(k_info->sizes, k_info->data_type);
        storage->slot_stacked_kv = push_input(skvt, keyIdx, 4);
    }

    // DML slot 5: StackedQueryKeyValueTensor
    if (stackedQkv) {
        auto sqkvt = MakeTensorInfo(q_info->sizes, q_info->data_type);
        storage->slot_stacked_qkv = push_input(sqkvt, queryIdx, 5);
    }

    // DML slot 6: BiasTensor
    if (hasBias) {
        auto* bias_info = LookupShape(value_shapes, inputs[biasIdx]);
        auto bt = MakeTensorInfo(bias_info->sizes, bias_info->data_type);
        storage->slot_bias = push_input(bt, biasIdx, 6);
    }

    // DML slot 7: MaskTensor
    if (hasMask) {
        storage->slot_mask = push_input(mask_tensor, maskIdx, 7);
    }

    // DML slot 8: RelativePositionBiasTensor
    if (hasAttBias) {
        auto* ab_info = LookupShape(value_shapes, inputs[attBiasIdx]);
        auto abt = MakeTensorInfo(ab_info->sizes, ab_info->data_type);
        storage->slot_att_bias = push_input(abt, attBiasIdx, 8);
    }

    // DML slot 9: PastKeyTensor
    if (hasPastKey) {
        size_t src_idx = keyValueIsPast ? keyIdx : pastKeyIdx;
        auto* pk_info = LookupShape(value_shapes, inputs[src_idx]);
        auto pkt = MakeTensorInfo(pk_info->sizes, pk_info->data_type);
        storage->slot_past_key = push_input(pkt, src_idx, 9);
    }

    // DML slot 10: PastValueTensor
    if (hasPastValue) {
        size_t src_idx = keyValueIsPast ? valueIdx : pastValueIdx;
        auto* pv_info = LookupShape(value_shapes, inputs[src_idx]);
        auto pvt = MakeTensorInfo(pv_info->sizes, pv_info->data_type);
        storage->slot_past_value = push_input(pvt, src_idx, 10);
    }

    result.input_tensor_descs.resize(result.input_tensors.size());

    // Outputs.
    auto* out0_info = LookupShape(value_shapes, outputs[0]);
    if (!out0_info) {
        std::vector<uint32_t> out0_sz = {batchSize, sequenceLength, vHiddenSize};
        auto ot = MakeTensorInfo(PadToMinDims(out0_sz), q_info->data_type);
        result.output_tensors.push_back(ot);
    } else {
        result.output_tensors.push_back(MakeTensorInfo(out0_info->sizes, q_info->data_type));
    }
    result.output_buffer_descs.push_back(result.output_tensors[0].ToBufferDesc());

    if (hasPresentKey) {
        auto* pk_out = LookupShape(value_shapes, outputs[1]);
        auto pkt = pk_out ? MakeTensorInfo(pk_out->sizes, q_info->data_type)
                          : MakeTensorInfo({batchSize, numHeads, totalSequenceLength, headSize}, q_info->data_type);
        result.output_tensors.push_back(pkt);
        result.output_buffer_descs.push_back(pkt.ToBufferDesc());
        storage->out_present_key = 1;
    }
    if (hasPresentValue) {
        auto* pv_out = LookupShape(value_shapes, outputs[2]);
        auto pvt = pv_out ? MakeTensorInfo(pv_out->sizes, q_info->data_type)
                          : MakeTensorInfo({batchSize, numHeads, totalSequenceLength, headSize}, q_info->data_type);
        result.output_tensors.push_back(pvt);
        result.output_buffer_descs.push_back(pvt.ToBufferDesc());
        storage->out_present_value = static_cast<int>(result.output_tensors.size()) - 1;
    }

    result.output_tensor_descs.resize(result.output_tensors.size());
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_MULTIHEAD_ATTENTION, &storage->desc };

    float local_scale = scale, local_mask_filter = maskFilterValue;
    uint32_t local_num_heads = numHeads;
    DML_MULTIHEAD_ATTENTION_MASK_TYPE local_mask_type = maskType;

    result.fixup = [storage, local_scale, local_mask_filter,
                    local_num_heads, local_mask_type](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        auto td = [&](int slot) -> const DML_TENSOR_DESC* {
            return (slot >= 0) ? &self.input_tensor_descs[slot] : nullptr;
        };
        auto otd = [&](int slot) -> const DML_TENSOR_DESC* {
            return (slot >= 0 && slot < static_cast<int>(self.output_tensor_descs.size()))
                   ? &self.output_tensor_descs[slot] : nullptr;
        };
        storage->desc.QueryTensor              = td(storage->slot_query);
        storage->desc.KeyTensor                = td(storage->slot_key);
        storage->desc.ValueTensor              = td(storage->slot_value);
        storage->desc.StackedQueryKeyTensor    = nullptr;
        storage->desc.StackedKeyValueTensor    = td(storage->slot_stacked_kv);
        storage->desc.StackedQueryKeyValueTensor = td(storage->slot_stacked_qkv);
        storage->desc.BiasTensor               = td(storage->slot_bias);
        storage->desc.MaskTensor               = td(storage->slot_mask);
        storage->desc.RelativePositionBiasTensor = td(storage->slot_att_bias);
        storage->desc.PastKeyTensor            = td(storage->slot_past_key);
        storage->desc.PastValueTensor          = td(storage->slot_past_value);
        storage->desc.OutputTensor             = &self.output_tensor_descs[0];
        storage->desc.OutputPresentKeyTensor   = otd(storage->out_present_key);
        storage->desc.OutputPresentValueTensor = otd(storage->out_present_value);
        storage->desc.Scale = local_scale;
        storage->desc.MaskFilterValue = local_mask_filter;
        storage->desc.HeadCount = local_num_heads;
        storage->desc.MaskType = local_mask_type;
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// RotaryEmbedding → Multi-op SubNode decomposition
// ORT ref: DmlOperatorRotaryEmbedding.cpp
// No native DML operator — decomposed into 10-14 DML ops.
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateRotaryEmbedding(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 4 || outputs.empty()) return std::nullopt;

    enum { inputDataIdx=0, posIdsIdx=1, cosCacheIdx=2, sinCacheIdx=3 };

    auto* in_info = LookupShape(value_shapes, inputs[inputDataIdx]);
    auto* pos_info = LookupShape(value_shapes, inputs[posIdsIdx]);
    auto* cos_info = LookupShape(value_shapes, inputs[cosCacheIdx]);
    auto* sin_info = LookupShape(value_shapes, inputs[sinCacheIdx]);
    if (!in_info || !pos_info || !cos_info || !sin_info) return std::nullopt;

    uint32_t in_rank = in_info->original_rank ? in_info->original_rank
                     : static_cast<uint32_t>(in_info->sizes.size());
    uint32_t pos_rank = pos_info->original_rank ? pos_info->original_rank
                      : static_cast<uint32_t>(pos_info->sizes.size());

    bool inputIs4D = (in_rank == 4);
    bool positionIdsIsOffset = (pos_rank == 1);

    OrtNodeAdapter adapter(node, ort_api);
    uint32_t numHeads = static_cast<uint32_t>(adapter.GetAttributeInt("num_heads", 0));
    uint32_t rotaryEmbeddingDim = static_cast<uint32_t>(adapter.GetAttributeInt("rotary_embedding_dim", 0));
    bool interleaved = static_cast<bool>(adapter.GetAttributeInt("interleaved", 0));

    // Extract dimensions from padded 4D shapes.
    size_t in_off = in_info->sizes.size() >= in_rank ? in_info->sizes.size() - in_rank : 0;
    auto& in_sizes = in_info->sizes;

    uint32_t hiddenSize;
    if (inputIs4D) {
        hiddenSize = in_sizes[in_off + 1] * in_sizes[in_off + 3];
    } else {
        hiddenSize = in_sizes.back();
    }

    size_t cos_off = cos_info->sizes.size() >= 2 ? cos_info->sizes.size() - 2 : 0;
    uint32_t cosLastDim = cos_info->sizes.back();

    uint32_t headSize;
    if (numHeads == 0) {
        headSize = cosLastDim * 2;
    } else {
        headSize = hiddenSize / numHeads;
    }

    if (rotaryEmbeddingDim == 0)
        rotaryEmbeddingDim = headSize;

    if (numHeads == 0)
        numHeads = hiddenSize / headSize;

    uint32_t batchSize = inputIs4D ? in_sizes[in_off] : in_sizes[in_off + (in_rank >= 3 ? 0 : 0)];
    uint32_t sequenceLength;
    if (inputIs4D) {
        sequenceLength = in_sizes[in_off + 2];
    } else {
        if (in_rank == 3) {
            batchSize = in_sizes[in_off];
            sequenceLength = in_sizes[in_off + 1];
        } else {
            batchSize = in_sizes[in_off];
            sequenceLength = 1;
        }
    }

    uint32_t halfRotaryDim = rotaryEmbeddingDim / 2;
    bool partialRotary = (headSize != rotaryEmbeddingDim);

    DML_TENSOR_DATA_TYPE data_dtype = in_info->data_type;
    bool is_f16 = (data_dtype == DML_TENSOR_DATA_TYPE_FLOAT16);

    // Storage struct owns all operator descs and intermediate shape data.
    struct RotaryStorage {
        DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC copy_desc{};
        DML_SCALE_BIAS scale_bias{1.0f, 0.0f};
        DML_SPLIT_OPERATOR_DESC split_half_desc{};
        DML_GATHER_OPERATOR_DESC gather_cos_desc{};
        DML_GATHER_OPERATOR_DESC gather_sin_desc{};
        DML_FILL_VALUE_SEQUENCE_OPERATOR_DESC sign_desc{};
        DML_JOIN_OPERATOR_DESC join_swap_desc{};
        DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC mul_cos_desc{};
        DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC mul_sin_desc{};
        DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC mul_sign_desc{};
        DML_ELEMENT_WISE_ADD_OPERATOR_DESC add_desc{};
        // Optional: positionIdsIsOffset
        DML_FILL_VALUE_SEQUENCE_OPERATOR_DESC pos_range_desc{};
        DML_ELEMENT_WISE_ADD_OPERATOR_DESC pos_add_desc{};
        // Optional: partial rotary
        DML_SPLIT_OPERATOR_DESC split_input_desc{};
        DML_JOIN_OPERATOR_DESC join_output_desc{};
        // All intermediate DmlTensorInfos and descriptors.
        DmlTensorInfo partial_in_strided, partial_in_packed, partial_out_strided;
        DmlTensorInfo data_5d, split_half_out, sign_ti;
        DmlTensorInfo gathered_cos_sin, cos_broadcast, sin_broadcast, sign_broadcast;
        DmlTensorInfo joined_data, mul_result;
        DmlTensorInfo pos_range_ti, pos_broadcast, pos_offset_ti;
        DmlTensorInfo full_in_out, split_in_1, split_in_2;
        // Buffer/tensor desc arrays for multi-output/input ops.
        std::vector<DML_BUFFER_TENSOR_DESC> split_half_out_bufs;
        std::vector<DML_TENSOR_DESC> split_half_out_tds;
        std::vector<DML_BUFFER_TENSOR_DESC> join_swap_in_bufs;
        std::vector<DML_TENSOR_DESC> join_swap_in_tds;
        std::vector<DML_BUFFER_TENSOR_DESC> split_in_out_bufs;
        std::vector<DML_TENSOR_DESC> split_in_out_tds;
        std::vector<DML_BUFFER_TENSOR_DESC> join_out_in_bufs;
        std::vector<DML_TENSOR_DESC> join_out_in_tds;
    };
    auto storage = std::make_shared<RotaryStorage>();

    // Shape construction.
    // inputOutputShape: the full input/output shape (4D).
    std::vector<uint32_t> inputOutputShape;
    if (inputIs4D) {
        inputOutputShape = {batchSize, numHeads, sequenceLength, headSize};
    } else {
        inputOutputShape = {batchSize, sequenceLength, numHeads, headSize};
    }

    // partialInputOutputShape: always [B, S, numHeads, rotaryDim] for internal computation.
    std::vector<uint32_t> partialShape = {batchSize, sequenceLength, numHeads, rotaryEmbeddingDim};

    // Strided input for Identity copy (4D input → internal layout).
    if (inputIs4D) {
        std::vector<uint32_t> strides_4d = {
            rotaryEmbeddingDim * numHeads * sequenceLength,
            rotaryEmbeddingDim,
            sequenceLength * rotaryEmbeddingDim,
            1
        };
        uint64_t in_bytes = in_info->total_bytes;
        if (in_bytes == 0) in_bytes = ComputeAlignedTotalBytes(in_info->sizes, data_dtype);
        storage->partial_in_strided = MakeTensorInfoWithStrides(partialShape, strides_4d, data_dtype, in_bytes);
    } else {
        storage->partial_in_strided = MakeTensorInfo(partialShape, data_dtype);
    }
    storage->partial_in_packed = MakeTensorInfo(partialShape, data_dtype);

    // Strided output for Add (writes back to 4D layout if needed).
    if (inputIs4D) {
        std::vector<uint32_t> strides_4d = {
            rotaryEmbeddingDim * numHeads * sequenceLength,
            rotaryEmbeddingDim,
            sequenceLength * rotaryEmbeddingDim,
            1
        };
        uint64_t in_bytes = in_info->total_bytes;
        if (in_bytes == 0) in_bytes = ComputeAlignedTotalBytes(in_info->sizes, data_dtype);
        storage->partial_out_strided = MakeTensorInfoWithStrides(partialShape, strides_4d, data_dtype, in_bytes);
    } else {
        storage->partial_out_strided = storage->partial_in_packed;
    }

    // 5D shapes for rotate_half split/join.
    std::vector<uint32_t> data_5d_shape, split_half_shape;
    uint32_t split_axis;
    if (interleaved) {
        data_5d_shape = {batchSize, sequenceLength, numHeads, halfRotaryDim, 2};
        split_half_shape = {batchSize, sequenceLength, numHeads, halfRotaryDim, 1};
        split_axis = 4;
    } else {
        data_5d_shape = {batchSize, sequenceLength, numHeads, 2, halfRotaryDim};
        split_half_shape = {batchSize, sequenceLength, numHeads, 1, halfRotaryDim};
        split_axis = 3;
    }
    storage->data_5d = MakeTensorInfo(data_5d_shape, data_dtype);
    storage->split_half_out = MakeTensorInfo(split_half_shape, data_dtype);
    storage->joined_data = MakeTensorInfo(data_5d_shape, data_dtype);
    storage->mul_result = MakeTensorInfo(data_5d_shape, data_dtype);

    // Split half output desc arrays (2 outputs).
    storage->split_half_out_bufs = { storage->split_half_out.ToBufferDesc(), storage->split_half_out.ToBufferDesc() };
    storage->split_half_out_tds.resize(2);

    // Join swap input desc arrays (2 inputs — swapped halves).
    storage->join_swap_in_bufs = { storage->split_half_out.ToBufferDesc(), storage->split_half_out.ToBufferDesc() };
    storage->join_swap_in_tds.resize(2);

    // Gathered cos/sin: [1, batchSize, sequenceLength, halfRotaryDim].
    std::vector<uint32_t> gathered_shape = {1, batchSize, sequenceLength, halfRotaryDim};
    storage->gathered_cos_sin = MakeTensorInfo(gathered_shape, data_dtype);

    // Broadcast cos/sin to 5D data shape.
    std::vector<uint32_t> cos_sin_broadcast_shape;
    if (interleaved) {
        cos_sin_broadcast_shape = {batchSize, sequenceLength, 1, halfRotaryDim, 1};
    } else {
        cos_sin_broadcast_shape = {batchSize, sequenceLength, 1, 1, halfRotaryDim};
    }
    auto cos_sin_strides = ComputePackedStrides(cos_sin_broadcast_shape);
    for (size_t d = 0; d < cos_sin_broadcast_shape.size(); ++d) {
        if (cos_sin_broadcast_shape[d] == 1 && data_5d_shape[d] > 1) {
            cos_sin_broadcast_shape[d] = data_5d_shape[d];
            cos_sin_strides[d] = 0;
        }
    }
    uint64_t cos_sin_bytes = ComputeAlignedTotalBytes(gathered_shape, data_dtype);
    storage->cos_broadcast = MakeTensorInfoWithStrides(cos_sin_broadcast_shape, cos_sin_strides, data_dtype, cos_sin_bytes);
    storage->sin_broadcast = storage->cos_broadcast;

    // Sign: [-1, 1] shape [2].
    storage->sign_ti = MakeTensorInfo({2}, data_dtype);

    // Broadcast sign to 5D.
    std::vector<uint32_t> sign_broadcast_actual;
    if (interleaved) {
        sign_broadcast_actual = {1, 1, 1, 1, 2};
    } else {
        sign_broadcast_actual = {1, 1, 1, 2, 1};
    }
    auto sign_strides = ComputePackedStrides(sign_broadcast_actual);
    auto sign_broadcast_shape = data_5d_shape;
    for (size_t d = 0; d < sign_broadcast_actual.size(); ++d) {
        if (sign_broadcast_actual[d] == 1 && data_5d_shape[d] > 1)
            sign_strides[d] = 0;
    }
    uint64_t sign_bytes = ComputeAlignedTotalBytes({2}, data_dtype);
    storage->sign_broadcast = MakeTensorInfoWithStrides(sign_broadcast_shape, sign_strides, data_dtype, sign_bytes);

    // Position IDs shapes (for offset mode).
    if (positionIdsIsOffset) {
        DML_TENSOR_DATA_TYPE pos_dtype = pos_info->data_type;
        storage->pos_range_ti = MakeTensorInfo({1, 1, 1, sequenceLength}, pos_dtype);
        std::vector<uint32_t> offset_shape = {1, 1, batchSize, sequenceLength};
        storage->pos_offset_ti = MakeTensorInfo(offset_shape, pos_dtype);
        // Broadcast position_ids for add.
        std::vector<uint32_t> pos_bcast_shape = offset_shape;
        auto pos_actual = pos_info->sizes;
        while (pos_actual.size() < 4) pos_actual.insert(pos_actual.begin(), 1u);
        auto pos_strides_bcast = ComputePackedStrides(pos_actual);
        for (size_t d = 0; d < 4; ++d) {
            if (pos_actual[d] == 1 && pos_bcast_shape[d] > 1)
                pos_strides_bcast[d] = 0;
        }
        uint64_t pos_bytes = pos_info->total_bytes;
        if (pos_bytes == 0) pos_bytes = ComputeAlignedTotalBytes(pos_info->sizes, pos_dtype);
        storage->pos_broadcast = MakeTensorInfoWithStrides(pos_bcast_shape, pos_strides_bcast, pos_dtype, pos_bytes);
        // Broadcast range for add.
        std::vector<uint32_t> range_bcast_shape = offset_shape;
        std::vector<uint32_t> range_actual = {1, 1, 1, sequenceLength};
        auto range_strides = ComputePackedStrides(range_actual);
        for (size_t d = 0; d < 4; ++d) {
            if (range_actual[d] == 1 && range_bcast_shape[d] > 1)
                range_strides[d] = 0;
        }
        uint64_t range_bytes = ComputeAlignedTotalBytes(range_actual, pos_dtype);
        // Store the broadcasted range for the Add input.
        // The range tensor itself is [1, 1, 1, seqLen], broadcast to [1, 1, B, seqLen].
        storage->pos_range_ti = MakeTensorInfoWithStrides(range_bcast_shape, range_strides, pos_dtype, range_bytes);
    }

    // Partial rotary shapes (4D split/join of input on last axis).
    if (partialRotary) {
        storage->full_in_out = MakeTensorInfo(inputOutputShape, data_dtype);
        std::vector<uint32_t> split1 = inputOutputShape;
        split1.back() = rotaryEmbeddingDim;
        std::vector<uint32_t> split2 = inputOutputShape;
        split2.back() = headSize - rotaryEmbeddingDim;
        storage->split_in_1 = MakeTensorInfo(split1, data_dtype);
        storage->split_in_2 = MakeTensorInfo(split2, data_dtype);
        storage->split_in_out_bufs = { storage->split_in_1.ToBufferDesc(), storage->split_in_2.ToBufferDesc() };
        storage->split_in_out_tds.resize(2);
        storage->join_out_in_bufs = { storage->split_in_1.ToBufferDesc(), storage->split_in_2.ToBufferDesc() };
        storage->join_out_in_tds.resize(2);
    }

    // Build the primary node (Identity copy) and TranslatedOp.
    // Primary = Identity that reshapes input to [B,S,numHeads,rotaryDim].
    TranslatedOp result;

    // All 4 ONNX inputs in input_tensors for consumption tracking.
    // Primary only uses input_data (slot 0); remaining are for sub_node graph_inputs.
    auto pos_tensor = MakeTensorInfo(pos_info->sizes, pos_info->data_type);
    auto cos_tensor = MakeTensorInfo(cos_info->sizes, data_dtype);
    auto sin_tensor = MakeTensorInfo(sin_info->sizes, data_dtype);

    result.input_tensors = { storage->partial_in_strided, pos_tensor, cos_tensor, sin_tensor };
    result.input_buffer_descs = {
        storage->partial_in_strided.ToBufferDesc(), pos_tensor.ToBufferDesc(),
        cos_tensor.ToBufferDesc(), sin_tensor.ToBufferDesc()
    };
    result.input_tensor_descs.resize(4);
    result.primary_input_count = 1;

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    auto out_tensor = out_edge ? MakeTensorInfo(out_edge->sizes, data_dtype)
                               : MakeTensorInfo(in_info->sizes, data_dtype);
    result.output_tensors = { out_tensor };
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);

    // Primary: trivial Identity pass-through on input_data.
    // The real Identity copy (with strides) is a sub_node.
    // We need a primary DML operator so the graph compiler creates a node.
    struct PrimaryIdentityStorage {
        DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC desc{};
    };
    auto primary_storage = std::make_shared<PrimaryIdentityStorage>();
    result.desc_storage = primary_storage;
    result.op_desc = { DML_OPERATOR_ELEMENT_WISE_IDENTITY, &primary_storage->desc };
    result.fixup = [primary_storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        primary_storage->desc.InputTensor = &self.input_tensor_descs[0];
        primary_storage->desc.OutputTensor = &self.output_tensor_descs[0];
        primary_storage->desc.ScaleBias = nullptr;
    };

    // Build sub_nodes. Use dynamic index tracking.
    // Assign node indices based on active paths.
    // Base nodes (always present): Split, GatherCos, GatherSin, SignRange, Join, MulCos, MulSin, MulSign, Add
    // Optional before Gather: PosRange, PosAddOffset
    // Optional around everything: SplitInput (before), JoinOutput (after)

    auto make_sub = [&](const DmlTensorInfo& out_ti) -> SubNode {
        SubNode sn;
        sn.output_tensors = { out_ti };
        sn.output_buffer_descs = { out_ti.ToBufferDesc() };
        sn.output_tensor_descs.resize(1);
        return sn;
    };

    // We'll build sub_nodes in order and track their indices.
    std::vector<SubNode> sub_nodes;

    // Optional: SplitInput for partial rotary.
    int splitInputIdx = -1;
    if (partialRotary) {
        splitInputIdx = static_cast<int>(sub_nodes.size());
        SubNode sn;
        sn.input_tensors = { storage->full_in_out };
        sn.input_buffer_descs = { storage->full_in_out.ToBufferDesc() };
        sn.input_tensor_descs.resize(1);
        sn.output_tensors = { storage->split_in_1, storage->split_in_2 };
        sn.output_buffer_descs = { storage->split_in_1.ToBufferDesc(), storage->split_in_2.ToBufferDesc() };
        sn.output_tensor_descs.resize(2);
        sn.desc_storage = storage;
        sn.op_desc = { DML_OPERATOR_SPLIT, &storage->split_input_desc };
        sn.graph_inputs = { {inputDataIdx, 0} };
        uint32_t local_axis = static_cast<uint32_t>(inputOutputShape.size()) - 1;
        sn.fixup = [storage, local_axis](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->split_in_out_bufs[0] = self.output_buffer_descs[0];
            storage->split_in_out_bufs[1] = self.output_buffer_descs[1];
            storage->split_in_out_tds[0] = { DML_TENSOR_TYPE_BUFFER, &storage->split_in_out_bufs[0] };
            storage->split_in_out_tds[1] = { DML_TENSOR_TYPE_BUFFER, &storage->split_in_out_bufs[1] };
            storage->split_input_desc.InputTensor = &self.input_tensor_descs[0];
            storage->split_input_desc.OutputCount = 2;
            storage->split_input_desc.OutputTensors = storage->split_in_out_tds.data();
            storage->split_input_desc.Axis = local_axis;
        };
        sn.FixupPointers();
        sub_nodes.push_back(std::move(sn));
    }

    // Sub: CopyInput (Identity) — reshapes to [B,S,numHeads,rotaryDim].
    int copyInputIdx = static_cast<int>(sub_nodes.size());
    {
        SubNode sn;
        sn.input_tensors = { storage->partial_in_strided };
        sn.input_buffer_descs = { storage->partial_in_strided.ToBufferDesc() };
        sn.input_tensor_descs.resize(1);
        sn.output_tensors = { storage->partial_in_packed };
        sn.output_buffer_descs = { storage->partial_in_packed.ToBufferDesc() };
        sn.output_tensor_descs.resize(1);
        sn.desc_storage = storage;
        sn.op_desc = { DML_OPERATOR_ELEMENT_WISE_IDENTITY, &storage->copy_desc };
        if (partialRotary) {
            sn.input_from = { {splitInputIdx, 0} };
        } else {
            sn.graph_inputs = { {inputDataIdx, 0} };
        }
        sn.fixup = [storage](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->copy_desc.InputTensor = &self.input_tensor_descs[0];
            storage->copy_desc.OutputTensor = &self.output_tensor_descs[0];
            storage->copy_desc.ScaleBias = &storage->scale_bias;
        };
        sn.FixupPointers();
        sub_nodes.push_back(std::move(sn));
    }

    // Sub: SplitHalf (split into 2 halves for rotate_half).
    int splitHalfIdx = static_cast<int>(sub_nodes.size());
    {
        SubNode sn;
        sn.input_tensors = { storage->data_5d };
        sn.input_buffer_descs = { storage->data_5d.ToBufferDesc() };
        sn.input_tensor_descs.resize(1);
        sn.output_tensors = { storage->split_half_out, storage->split_half_out };
        sn.output_buffer_descs = { storage->split_half_out.ToBufferDesc(), storage->split_half_out.ToBufferDesc() };
        sn.output_tensor_descs.resize(2);
        sn.desc_storage = storage;
        sn.op_desc = { DML_OPERATOR_SPLIT, &storage->split_half_desc };
        sn.input_from = { {copyInputIdx, 0} };
        uint32_t local_split_axis = split_axis;
        sn.fixup = [storage, local_split_axis](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->split_half_out_bufs[0] = self.output_buffer_descs[0];
            storage->split_half_out_bufs[1] = self.output_buffer_descs[1];
            storage->split_half_out_tds[0] = { DML_TENSOR_TYPE_BUFFER, &storage->split_half_out_bufs[0] };
            storage->split_half_out_tds[1] = { DML_TENSOR_TYPE_BUFFER, &storage->split_half_out_bufs[1] };
            storage->split_half_desc.InputTensor = &self.input_tensor_descs[0];
            storage->split_half_desc.OutputCount = 2;
            storage->split_half_desc.OutputTensors = storage->split_half_out_tds.data();
            storage->split_half_desc.Axis = local_split_axis;
        };
        sn.FixupPointers();
        sub_nodes.push_back(std::move(sn));
    }

    // Optional: PosRange + PosAddOffset for positionIdsIsOffset.
    int posAddOffsetIdx = -1;
    if (positionIdsIsOffset) {
        // PosRange: FillValueSequence [0..seqLen-1].
        int posRangeIdx = static_cast<int>(sub_nodes.size());
        {
            SubNode sn;
            sn.output_tensors = { MakeTensorInfo({1, 1, 1, sequenceLength}, pos_info->data_type) };
            sn.output_buffer_descs = { sn.output_tensors[0].ToBufferDesc() };
            sn.output_tensor_descs.resize(1);
            sn.desc_storage = storage;
            sn.op_desc = { DML_OPERATOR_FILL_VALUE_SEQUENCE, &storage->pos_range_desc };
            // No inputs — generator node.
            sn.fixup = [storage](SubNode& self) {
                RebuildSubNodePointers(self);
                storage->pos_range_desc.OutputTensor = &self.output_tensor_descs[0];
                storage->pos_range_desc.ValueDataType = DML_TENSOR_DATA_TYPE_INT64;
                storage->pos_range_desc.ValueStart.Int64 = 0;
                storage->pos_range_desc.ValueDelta.Int64 = 1;
            };
            sn.FixupPointers();
            sub_nodes.push_back(std::move(sn));
        }

        // PosAddOffset: Add(range_broadcast, position_ids_broadcast).
        posAddOffsetIdx = static_cast<int>(sub_nodes.size());
        {
            SubNode sn;
            sn.input_tensors = { storage->pos_range_ti, storage->pos_broadcast };
            sn.input_buffer_descs = { storage->pos_range_ti.ToBufferDesc(), storage->pos_broadcast.ToBufferDesc() };
            sn.input_tensor_descs.resize(2);
            sn.output_tensors = { storage->pos_offset_ti };
            sn.output_buffer_descs = { storage->pos_offset_ti.ToBufferDesc() };
            sn.output_tensor_descs.resize(1);
            sn.desc_storage = storage;
            sn.op_desc = { DML_OPERATOR_ELEMENT_WISE_ADD, &storage->pos_add_desc };
            sn.input_from = { {posRangeIdx, 0} };
            sn.graph_inputs = { {posIdsIdx, 1} };
            sn.fixup = [storage](SubNode& self) {
                RebuildSubNodePointers(self);
                storage->pos_add_desc.ATensor = &self.input_tensor_descs[0];
                storage->pos_add_desc.BTensor = &self.input_tensor_descs[1];
                storage->pos_add_desc.OutputTensor = &self.output_tensor_descs[0];
            };
            sn.FixupPointers();
            sub_nodes.push_back(std::move(sn));
        }
    }

    // Sub: GatherCos.
    int gatherCosIdx = static_cast<int>(sub_nodes.size());
    {
        SubNode sn;
        auto indices_ti = positionIdsIsOffset
            ? storage->pos_offset_ti
            : MakeTensorInfo(pos_info->sizes, pos_info->data_type);
        sn.input_tensors = { MakeTensorInfo(cos_info->sizes, data_dtype), indices_ti };
        sn.input_buffer_descs = { sn.input_tensors[0].ToBufferDesc(), indices_ti.ToBufferDesc() };
        sn.input_tensor_descs.resize(2);
        sn.output_tensors = { storage->gathered_cos_sin };
        sn.output_buffer_descs = { storage->gathered_cos_sin.ToBufferDesc() };
        sn.output_tensor_descs.resize(1);
        sn.desc_storage = storage;
        sn.op_desc = { DML_OPERATOR_GATHER, &storage->gather_cos_desc };
        if (positionIdsIsOffset) {
            sn.input_from = { {-2, 0}, {posAddOffsetIdx, 0} };
            sn.graph_inputs = { {cosCacheIdx, 0} };
        } else {
            sn.graph_inputs = { {cosCacheIdx, 0}, {posIdsIdx, 1} };
        }
        sn.fixup = [storage](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->gather_cos_desc.InputTensor = &self.input_tensor_descs[0];
            storage->gather_cos_desc.IndicesTensor = &self.input_tensor_descs[1];
            storage->gather_cos_desc.OutputTensor = &self.output_tensor_descs[0];
            storage->gather_cos_desc.Axis = 2;
            storage->gather_cos_desc.IndexDimensions = 2;
        };
        sn.FixupPointers();
        sub_nodes.push_back(std::move(sn));
    }

    // Sub: GatherSin.
    int gatherSinIdx = static_cast<int>(sub_nodes.size());
    {
        SubNode sn;
        auto indices_ti = positionIdsIsOffset
            ? storage->pos_offset_ti
            : MakeTensorInfo(pos_info->sizes, pos_info->data_type);
        sn.input_tensors = { MakeTensorInfo(sin_info->sizes, data_dtype), indices_ti };
        sn.input_buffer_descs = { sn.input_tensors[0].ToBufferDesc(), indices_ti.ToBufferDesc() };
        sn.input_tensor_descs.resize(2);
        sn.output_tensors = { storage->gathered_cos_sin };
        sn.output_buffer_descs = { storage->gathered_cos_sin.ToBufferDesc() };
        sn.output_tensor_descs.resize(1);
        sn.desc_storage = storage;
        sn.op_desc = { DML_OPERATOR_GATHER, &storage->gather_sin_desc };
        if (positionIdsIsOffset) {
            sn.input_from = { {-2, 0}, {posAddOffsetIdx, 0} };
            sn.graph_inputs = { {sinCacheIdx, 0} };
        } else {
            sn.graph_inputs = { {sinCacheIdx, 0}, {posIdsIdx, 1} };
        }
        sn.fixup = [storage](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->gather_sin_desc.InputTensor = &self.input_tensor_descs[0];
            storage->gather_sin_desc.IndicesTensor = &self.input_tensor_descs[1];
            storage->gather_sin_desc.OutputTensor = &self.output_tensor_descs[0];
            storage->gather_sin_desc.Axis = 2;
            storage->gather_sin_desc.IndexDimensions = 2;
        };
        sn.FixupPointers();
        sub_nodes.push_back(std::move(sn));
    }

    // Sub: SignRange — FillValueSequence [-1, 1].
    int signRangeIdx = static_cast<int>(sub_nodes.size());
    {
        SubNode sn;
        sn.output_tensors = { storage->sign_ti };
        sn.output_buffer_descs = { storage->sign_ti.ToBufferDesc() };
        sn.output_tensor_descs.resize(1);
        sn.desc_storage = storage;
        sn.op_desc = { DML_OPERATOR_FILL_VALUE_SEQUENCE, &storage->sign_desc };
        bool local_is_f16 = is_f16;
        sn.fixup = [storage, local_is_f16](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->sign_desc.OutputTensor = &self.output_tensor_descs[0];
            if (local_is_f16) {
                storage->sign_desc.ValueDataType = DML_TENSOR_DATA_TYPE_FLOAT16;
                uint16_t neg_one = 0xBC00;  // -1.0 in float16
                uint16_t two = 0x4000;      // 2.0 in float16
                memcpy(storage->sign_desc.ValueStart.Bytes, &neg_one, sizeof(neg_one));
                memcpy(storage->sign_desc.ValueDelta.Bytes, &two, sizeof(two));
            } else {
                storage->sign_desc.ValueDataType = DML_TENSOR_DATA_TYPE_FLOAT32;
                storage->sign_desc.ValueStart.Float32 = -1.0f;
                storage->sign_desc.ValueDelta.Float32 = 2.0f;
            }
        };
        sn.FixupPointers();
        sub_nodes.push_back(std::move(sn));
    }

    // Sub: JoinSwap (swap halves).
    int joinSwapIdx = static_cast<int>(sub_nodes.size());
    {
        SubNode sn;
        sn.input_tensors = { storage->split_half_out, storage->split_half_out };
        sn.input_buffer_descs = { storage->split_half_out.ToBufferDesc(), storage->split_half_out.ToBufferDesc() };
        sn.input_tensor_descs.resize(2);
        sn.output_tensors = { storage->joined_data };
        sn.output_buffer_descs = { storage->joined_data.ToBufferDesc() };
        sn.output_tensor_descs.resize(1);
        sn.desc_storage = storage;
        sn.op_desc = { DML_OPERATOR_JOIN, &storage->join_swap_desc };
        // Swap: slot 0 gets split output 1, slot 1 gets split output 0.
        sn.input_from = { {splitHalfIdx, 1}, {splitHalfIdx, 0} };
        uint32_t local_join_axis = split_axis;
        sn.fixup = [storage, local_join_axis](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->join_swap_in_bufs[0] = self.input_buffer_descs[0];
            storage->join_swap_in_bufs[1] = self.input_buffer_descs[1];
            storage->join_swap_in_tds[0] = { DML_TENSOR_TYPE_BUFFER, &storage->join_swap_in_bufs[0] };
            storage->join_swap_in_tds[1] = { DML_TENSOR_TYPE_BUFFER, &storage->join_swap_in_bufs[1] };
            storage->join_swap_desc.InputCount = 2;
            storage->join_swap_desc.InputTensors = storage->join_swap_in_tds.data();
            storage->join_swap_desc.OutputTensor = &self.output_tensor_descs[0];
            storage->join_swap_desc.Axis = local_join_axis;
        };
        sn.FixupPointers();
        sub_nodes.push_back(std::move(sn));
    }

    // Sub: MulCos — input * cos.
    int mulCosIdx = static_cast<int>(sub_nodes.size());
    {
        SubNode sn;
        sn.input_tensors = { storage->data_5d, storage->cos_broadcast };
        sn.input_buffer_descs = { storage->data_5d.ToBufferDesc(), storage->cos_broadcast.ToBufferDesc() };
        sn.input_tensor_descs.resize(2);
        sn.output_tensors = { storage->mul_result };
        sn.output_buffer_descs = { storage->mul_result.ToBufferDesc() };
        sn.output_tensor_descs.resize(1);
        sn.desc_storage = storage;
        sn.op_desc = { DML_OPERATOR_ELEMENT_WISE_MULTIPLY, &storage->mul_cos_desc };
        sn.input_from = { {copyInputIdx, 0}, {gatherCosIdx, 0} };
        sn.fixup = [storage](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->mul_cos_desc.ATensor = &self.input_tensor_descs[0];
            storage->mul_cos_desc.BTensor = &self.input_tensor_descs[1];
            storage->mul_cos_desc.OutputTensor = &self.output_tensor_descs[0];
        };
        sn.FixupPointers();
        sub_nodes.push_back(std::move(sn));
    }

    // Sub: MulSin — rotated * sin.
    int mulSinIdx = static_cast<int>(sub_nodes.size());
    {
        SubNode sn;
        sn.input_tensors = { storage->joined_data, storage->sin_broadcast };
        sn.input_buffer_descs = { storage->joined_data.ToBufferDesc(), storage->sin_broadcast.ToBufferDesc() };
        sn.input_tensor_descs.resize(2);
        sn.output_tensors = { storage->mul_result };
        sn.output_buffer_descs = { storage->mul_result.ToBufferDesc() };
        sn.output_tensor_descs.resize(1);
        sn.desc_storage = storage;
        sn.op_desc = { DML_OPERATOR_ELEMENT_WISE_MULTIPLY, &storage->mul_sin_desc };
        sn.input_from = { {joinSwapIdx, 0}, {gatherSinIdx, 0} };
        sn.fixup = [storage](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->mul_sin_desc.ATensor = &self.input_tensor_descs[0];
            storage->mul_sin_desc.BTensor = &self.input_tensor_descs[1];
            storage->mul_sin_desc.OutputTensor = &self.output_tensor_descs[0];
        };
        sn.FixupPointers();
        sub_nodes.push_back(std::move(sn));
    }

    // Sub: MulSign — (rotated*sin) * sign.
    int mulSignIdx = static_cast<int>(sub_nodes.size());
    {
        SubNode sn;
        sn.input_tensors = { storage->mul_result, storage->sign_broadcast };
        sn.input_buffer_descs = { storage->mul_result.ToBufferDesc(), storage->sign_broadcast.ToBufferDesc() };
        sn.input_tensor_descs.resize(2);
        sn.output_tensors = { storage->mul_result };
        sn.output_buffer_descs = { storage->mul_result.ToBufferDesc() };
        sn.output_tensor_descs.resize(1);
        sn.desc_storage = storage;
        sn.op_desc = { DML_OPERATOR_ELEMENT_WISE_MULTIPLY, &storage->mul_sign_desc };
        sn.input_from = { {mulSinIdx, 0}, {signRangeIdx, 0} };
        sn.fixup = [storage](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->mul_sign_desc.ATensor = &self.input_tensor_descs[0];
            storage->mul_sign_desc.BTensor = &self.input_tensor_descs[1];
            storage->mul_sign_desc.OutputTensor = &self.output_tensor_descs[0];
        };
        sn.FixupPointers();
        sub_nodes.push_back(std::move(sn));
    }

    // Sub: Add — (input*cos) + (rotated*sin*sign).
    int addIdx = static_cast<int>(sub_nodes.size());
    {
        SubNode sn;
        // Output uses strided desc to write back to original layout if 4D.
        sn.input_tensors = { storage->partial_in_packed, storage->partial_in_packed };
        sn.input_buffer_descs = { storage->partial_in_packed.ToBufferDesc(), storage->partial_in_packed.ToBufferDesc() };
        sn.input_tensor_descs.resize(2);
        sn.output_tensors = { storage->partial_out_strided };
        sn.output_buffer_descs = { storage->partial_out_strided.ToBufferDesc() };
        sn.output_tensor_descs.resize(1);
        sn.desc_storage = storage;
        sn.op_desc = { DML_OPERATOR_ELEMENT_WISE_ADD, &storage->add_desc };
        sn.input_from = { {mulCosIdx, 0}, {mulSignIdx, 0} };
        sn.fixup = [storage](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->add_desc.ATensor = &self.input_tensor_descs[0];
            storage->add_desc.BTensor = &self.input_tensor_descs[1];
            storage->add_desc.OutputTensor = &self.output_tensor_descs[0];
        };
        sn.FixupPointers();
        sub_nodes.push_back(std::move(sn));
    }

    // Optional: JoinOutput for partial rotary.
    if (partialRotary) {
        SubNode sn;
        sn.input_tensors = { storage->split_in_1, storage->split_in_2 };
        sn.input_buffer_descs = { storage->split_in_1.ToBufferDesc(), storage->split_in_2.ToBufferDesc() };
        sn.input_tensor_descs.resize(2);
        sn.output_tensors = { storage->full_in_out };
        sn.output_buffer_descs = { storage->full_in_out.ToBufferDesc() };
        sn.output_tensor_descs.resize(1);
        sn.desc_storage = storage;
        sn.op_desc = { DML_OPERATOR_JOIN, &storage->join_output_desc };
        sn.input_from = { {addIdx, 0}, {splitInputIdx, 1} };
        uint32_t local_axis = static_cast<uint32_t>(inputOutputShape.size()) - 1;
        sn.fixup = [storage, local_axis](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->join_out_in_bufs[0] = self.input_buffer_descs[0];
            storage->join_out_in_bufs[1] = self.input_buffer_descs[1];
            storage->join_out_in_tds[0] = { DML_TENSOR_TYPE_BUFFER, &storage->join_out_in_bufs[0] };
            storage->join_out_in_tds[1] = { DML_TENSOR_TYPE_BUFFER, &storage->join_out_in_bufs[1] };
            storage->join_output_desc.InputCount = 2;
            storage->join_output_desc.InputTensors = storage->join_out_in_tds.data();
            storage->join_output_desc.OutputTensor = &self.output_tensor_descs[0];
            storage->join_output_desc.Axis = local_axis;
        };
        sn.FixupPointers();
        sub_nodes.push_back(std::move(sn));
    }

    result.sub_nodes = std::move(sub_nodes);
    return result;
}

// ---------------------------------------------------------------------------
// LpPool / GlobalLpPool → DML_OPERATOR_LP_POOLING
// ORT ref: DmlOperatorPooling.cpp (DML_OPERATOR_LP_POOLING case)
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateLpPool(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&,
    bool global = false) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    int64_t p_val = adapter.GetAttributeInt("p", 2);

    auto in_sizes = in_info->sizes;
    std::vector<uint32_t> out_sizes;
    uint32_t spatial = static_cast<uint32_t>(in_sizes.size()) - 2;
    if (spatial < 1) spatial = 1;

    std::vector<uint32_t> window_size, strides, start_pad, end_pad;
    if (global) {
        for (uint32_t i = 0; i < spatial; ++i) window_size.push_back(in_sizes[2 + i]);
        strides.assign(spatial, 1u);
        start_pad.assign(spatial, 0u);
        end_pad.assign(spatial, 0u);
    } else {
        auto ks = adapter.GetAttributeInts("kernel_shape");
        for (int64_t v : ks) window_size.push_back(static_cast<uint32_t>(v));
        while (window_size.size() < spatial) window_size.insert(window_size.begin(), 1u);
        auto st = adapter.GetAttributeInts("strides");
        for (int64_t v : st) strides.push_back(static_cast<uint32_t>(v));
        while (strides.size() < spatial) strides.push_back(1u);
        auto pads = adapter.GetAttributeInts("pads");
        if (pads.size() >= 2 * spatial) {
            for (size_t i = 0; i < spatial; ++i) start_pad.push_back(static_cast<uint32_t>(pads[i]));
            for (size_t i = 0; i < spatial; ++i) end_pad.push_back(static_cast<uint32_t>(pads[spatial + i]));
        } else {
            start_pad.assign(spatial, 0u);
            end_pad.assign(spatial, 0u);
        }
    }

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    if (out_edge) {
        out_sizes = out_edge->sizes;
    } else {
        out_sizes = in_sizes;
        for (uint32_t d = 0; d < spatial; ++d)
            out_sizes[2 + d] = (in_sizes[2 + d] + start_pad[d] + end_pad[d] - window_size[d]) / strides[d] + 1;
    }

    auto in_tensor  = MakeTensorInfo(in_sizes,  in_info->data_type);
    auto out_tensor = MakeTensorInfo(out_sizes, in_info->data_type);

    struct LpPoolStorage {
        DML_LP_POOLING_OPERATOR_DESC desc{};
        std::vector<uint32_t> window_size, strides, start_pad, end_pad;
    };
    auto storage = std::make_shared<LpPoolStorage>();
    storage->window_size = window_size;
    storage->strides     = strides;
    storage->start_pad   = start_pad;
    storage->end_pad     = end_pad;
    storage->desc.DimensionCount = spatial;
    storage->desc.WindowSize     = storage->window_size.data();
    storage->desc.Strides        = storage->strides.data();
    storage->desc.StartPadding   = storage->start_pad.data();
    storage->desc.EndPadding     = storage->end_pad.data();
    storage->desc.P              = static_cast<UINT>(p_val);

    TranslatedOp result;
    result.input_tensors = { in_tensor };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs = { in_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_LP_POOLING, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor  = &self.input_tensor_descs[0];
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
        storage->desc.WindowSize   = storage->window_size.data();
        storage->desc.Strides      = storage->strides.data();
        storage->desc.StartPadding = storage->start_pad.data();
        storage->desc.EndPadding   = storage->end_pad.data();
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// Col2Im → DML_OPERATOR_FOLD
// ORT ref: DmlOperatorCol2Im.cpp
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateCol2Im(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>& initializers) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 3 || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    // Read image_shape from initializer (ONNX input 1).
    auto read_int64_initializer = [&](size_t idx) -> std::vector<uint32_t> {
        std::vector<uint32_t> result;
        if (idx >= inputs.size() || inputs[idx].empty()) return result;
        auto it = initializers.find(inputs[idx]);
        if (it == initializers.end() || !it->second) return result;
        void* data = nullptr;
        ort_api.GetTensorMutableData(const_cast<OrtValue*>(it->second), &data);
        if (!data) return result;
        OrtTensorTypeAndShapeInfo* tsi = nullptr;
        ort_api.GetTensorTypeAndShape(const_cast<OrtValue*>(it->second), &tsi);
        size_t cnt = 0;
        ort_api.GetTensorShapeElementCount(tsi, &cnt);
        ort_api.ReleaseTensorTypeAndShapeInfo(tsi);
        auto* vals = static_cast<const int64_t*>(data);
        for (size_t i = 0; i < cnt; ++i)
            result.push_back(static_cast<uint32_t>(vals[i]));
        return result;
    };

    auto image_shape = read_int64_initializer(1);
    auto block_shape = read_int64_initializer(2);
    if (image_shape.empty() || block_shape.empty()) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    uint32_t spatial_dims = static_cast<uint32_t>(block_shape.size());

    auto dil = adapter.GetAttributeInts("dilations");
    std::vector<uint32_t> dilations;
    for (int64_t v : dil) dilations.push_back(static_cast<uint32_t>(v));
    while (dilations.size() < spatial_dims) dilations.push_back(1u);

    auto st = adapter.GetAttributeInts("strides");
    std::vector<uint32_t> strides_vec;
    for (int64_t v : st) strides_vec.push_back(static_cast<uint32_t>(v));
    while (strides_vec.size() < spatial_dims) strides_vec.push_back(1u);

    auto pd = adapter.GetAttributeInts("pads");
    std::vector<uint32_t> start_pad, end_pad;
    if (pd.size() >= 2 * spatial_dims) {
        for (size_t i = 0; i < spatial_dims; ++i) start_pad.push_back(static_cast<uint32_t>(pd[i]));
        for (size_t i = 0; i < spatial_dims; ++i) end_pad.push_back(static_cast<uint32_t>(pd[spatial_dims + i]));
    } else {
        start_pad.assign(spatial_dims, 0u);
        end_pad.assign(spatial_dims, 0u);
    }

    // Col2Im input: [N, C*prod(block_shape), L_1*L_2*...], output: [N, C, image_shape...]
    // C = input_dim[1] / prod(block_shape)
    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    std::vector<uint32_t> out_sizes;
    if (out_edge) {
        out_sizes = out_edge->sizes;
    } else {
        uint32_t orig_rank = in_info->original_rank ? in_info->original_rank
                           : static_cast<uint32_t>(in_info->sizes.size());
        size_t pad_off = in_info->sizes.size() >= orig_rank ? in_info->sizes.size() - orig_rank : 0;
        uint32_t N = in_info->sizes[pad_off];
        uint32_t block_prod = 1;
        for (auto b : block_shape) block_prod *= b;
        uint32_t C = (block_prod > 0) ? in_info->sizes[pad_off + 1] / block_prod : 1u;
        out_sizes.push_back(N);
        out_sizes.push_back(C);
        for (auto s : image_shape) out_sizes.push_back(s);
        out_sizes = PadToMinDims(out_sizes);
    }

    // ORT uses minDimensionCount=3 and pads input/output to the same rank.
    // DML_FOLD requires DimensionCount == InputTensor.DimCount - 2.
    size_t min_dims = std::max<size_t>(3, 2 + spatial_dims);
    auto in_tensor  = MakeTensorInfo(PadToMinDims(in_info->sizes, min_dims), in_info->data_type);
    auto out_tensor = MakeTensorInfo(PadToMinDims(out_sizes, min_dims), in_info->data_type);

    struct Col2ImStorage {
        DML_FOLD_OPERATOR_DESC desc{};
        std::vector<uint32_t> window_sizes, dilations, start_pad, end_pad, strides;
    };
    auto storage = std::make_shared<Col2ImStorage>();
    storage->window_sizes = block_shape;
    storage->dilations    = dilations;
    storage->start_pad    = start_pad;
    storage->end_pad      = end_pad;
    storage->strides      = strides_vec;
    storage->desc.DimensionCount = spatial_dims;
    storage->desc.WindowSizes    = storage->window_sizes.data();
    storage->desc.Dilations      = storage->dilations.data();
    storage->desc.StartPadding   = storage->start_pad.data();
    storage->desc.EndPadding     = storage->end_pad.data();
    storage->desc.Strides        = storage->strides.data();

    TranslatedOp result;
    result.input_tensors = { in_tensor };
    result.output_tensors = { out_tensor };
    result.input_buffer_descs = { in_tensor.ToBufferDesc() };
    result.input_tensor_descs.resize(1);
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_FOLD, &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor  = &self.input_tensor_descs[0];
        storage->desc.OutputTensor = &self.output_tensor_descs[0];
        storage->desc.WindowSizes  = storage->window_sizes.data();
        storage->desc.Dilations    = storage->dilations.data();
        storage->desc.StartPadding = storage->start_pad.data();
        storage->desc.EndPadding   = storage->end_pad.data();
        storage->desc.Strides      = storage->strides.data();
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// QLinearConv → DML_OPERATOR_QUANTIZED_LINEAR_CONVOLUTION
// ORT ref: DmlOperatorQLinearConv.cpp
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateQLinearConv(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 8 || outputs.empty()) return std::nullopt;

    enum { xIdx=0, xScaleIdx=1, xZpIdx=2, wIdx=3, wScaleIdx=4, wZpIdx=5,
           yScaleIdx=6, yZpIdx=7, biasIdx=8 };

    auto valid = [&](size_t idx) {
        return idx < inputs.size() && !inputs[idx].empty() && value_shapes.count(inputs[idx]);
    };

    auto* x_info = LookupShape(value_shapes, inputs[xIdx]);
    auto* w_info = LookupShape(value_shapes, inputs[wIdx]);
    if (!x_info || !w_info) return std::nullopt;

    bool has_x_zp = valid(xZpIdx);
    bool has_w_zp = valid(wZpIdx);
    bool has_bias = valid(biasIdx);

    OrtNodeAdapter adapter(node, ort_api);
    int64_t group = adapter.GetAttributeInt("group", 1);

    auto x_sizes = x_info->sizes;
    auto w_sizes = w_info->sizes;
    auto k = ReadConvKernelArgs(adapter, x_sizes);

    // Resolve auto_pad.
    if (k.auto_pad) {
        for (uint32_t d = 0; d < k.spatial_dim_count; ++d) {
            uint32_t input_length = x_sizes[2 + d];
            uint32_t stride = k.strides[d];
            uint32_t strided_output_length = (input_length + stride - 1) / stride;
            uint32_t kernel_length = 1 + (k.kernel_shape[d] - 1) * k.dilations[d];
            uint32_t length_needed = stride * (strided_output_length - 1) + kernel_length;
            uint32_t padding = (length_needed <= input_length) ? 0 : (length_needed - input_length);
            if (k.auto_pad_same_upper) {
                k.start_padding[d] = padding / 2;
            } else {
                k.start_padding[d] = (padding + 1) / 2;
            }
            k.end_padding[d] = padding - k.start_padding[d];
        }
    }

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    std::vector<uint32_t> out_sizes;
    if (out_edge) {
        out_sizes = out_edge->sizes;
    } else {
        out_sizes = x_sizes;
        out_sizes[1] = w_sizes[0];
        for (uint32_t d = 0; d < k.spatial_dim_count; ++d)
            out_sizes[2 + d] = (x_sizes[2 + d] + k.start_padding[d] + k.end_padding[d] -
                (1 + (k.kernel_shape[d] - 1) * k.dilations[d])) / k.strides[d] + 1;
    }

    struct QLinConvStorage {
        DML_QUANTIZED_LINEAR_CONVOLUTION_OPERATOR_DESC desc{};
        std::vector<uint32_t> strides, dilations, start_pad, end_pad;
    };
    auto storage = std::make_shared<QLinConvStorage>();
    storage->strides   = k.strides;
    storage->dilations = k.dilations;
    storage->start_pad = k.start_padding;
    storage->end_pad   = k.end_padding;

    TranslatedOp result;
    int idx = 0;
    auto push = [&](const DmlTensorInfo& t, size_t onnx_idx) {
        result.input_tensors.push_back(t);
        result.input_buffer_descs.push_back(t.ToBufferDesc());
        result.input_name_reorder.push_back(onnx_idx);
        return idx++;
    };

    // DML order: x, x_scale, x_zp, w, w_scale, w_zp, bias, y_scale, y_zp
    auto x_tensor = MakeTensorInfo(x_sizes, x_info->data_type);
    push(x_tensor, xIdx);

    auto* xs_info = LookupShape(value_shapes, inputs[xScaleIdx]);
    if (!xs_info) return std::nullopt;
    push(MakeTensorInfo(xs_info->sizes, xs_info->data_type), xScaleIdx);

    int slot_x_zp = -1;
    if (has_x_zp) {
        auto* xzp_info = LookupShape(value_shapes, inputs[xZpIdx]);
        if (!xzp_info) return std::nullopt;
        slot_x_zp = push(MakeTensorInfo(xzp_info->sizes, xzp_info->data_type), xZpIdx);
    }

    auto w_tensor = MakeTensorInfo(w_sizes, w_info->data_type);
    push(w_tensor, wIdx);

    auto* ws_info = LookupShape(value_shapes, inputs[wScaleIdx]);
    if (!ws_info) return std::nullopt;
    uint32_t dml_dim_count = static_cast<uint32_t>(x_sizes.size());
    // Per-channel filter scale: place at C-axis (axis 1) with LeftAligned padding.
    // ORT: TensorAxis::C, TensorAxis::LeftAligned → [C_out, 1, 1, ...]
    {
        auto ws_orig_sizes = ws_info->sizes;
        uint32_t ws_orig_rank = ws_info->original_rank ? ws_info->original_rank
                              : static_cast<uint32_t>(ws_orig_sizes.size());
        if (ws_orig_rank == 1) {
            push(MakeTensorInfoAtAxis({ws_orig_sizes.back()}, ws_info->data_type, 0, dml_dim_count), wScaleIdx);
        } else {
            push(MakeTensorInfo(ws_info->sizes, ws_info->data_type), wScaleIdx);
        }
    }

    int slot_w_zp = -1;
    if (has_w_zp) {
        auto* wzp_info = LookupShape(value_shapes, inputs[wZpIdx]);
        if (!wzp_info) return std::nullopt;
        uint32_t wzp_orig_rank = wzp_info->original_rank ? wzp_info->original_rank
                               : static_cast<uint32_t>(wzp_info->sizes.size());
        if (wzp_orig_rank == 1) {
            slot_w_zp = push(MakeTensorInfoAtAxis({wzp_info->sizes.back()}, wzp_info->data_type, 0, dml_dim_count), wZpIdx);
        } else {
            slot_w_zp = push(MakeTensorInfo(wzp_info->sizes, wzp_info->data_type), wZpIdx);
        }
    }

    int slot_bias = -1;
    if (has_bias) {
        auto* b_info = LookupShape(value_shapes, inputs[biasIdx]);
        if (!b_info) return std::nullopt;
        uint32_t b_orig_rank = b_info->original_rank ? b_info->original_rank
                             : static_cast<uint32_t>(b_info->sizes.size());
        if (b_orig_rank == 1) {
            // Broadcast bias [C_out] to C-axis: [C_out, 1, 1, ...] matching ORT.
            slot_bias = push(MakeTensorInfoAtAxis({b_info->sizes.back()}, b_info->data_type, 0, dml_dim_count), biasIdx);
        } else {
            slot_bias = push(MakeTensorInfo(b_info->sizes, b_info->data_type), biasIdx);
        }
    }

    auto* ys_info = LookupShape(value_shapes, inputs[yScaleIdx]);
    if (!ys_info) return std::nullopt;
    push(MakeTensorInfo(ys_info->sizes, ys_info->data_type), yScaleIdx);

    auto* yzp_info = LookupShape(value_shapes, inputs[yZpIdx]);
    if (!yzp_info) return std::nullopt;
    push(MakeTensorInfo(yzp_info->sizes, yzp_info->data_type), yZpIdx);

    auto out_tensor = MakeTensorInfo(out_sizes, yzp_info->data_type);
    result.input_tensor_descs.resize(result.input_tensors.size());
    result.output_tensors = { out_tensor };
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);

    storage->desc.DimensionCount = k.spatial_dim_count;
    storage->desc.Strides        = storage->strides.data();
    storage->desc.Dilations      = storage->dilations.data();
    storage->desc.StartPadding   = storage->start_pad.data();
    storage->desc.EndPadding     = storage->end_pad.data();
    storage->desc.GroupCount     = static_cast<UINT>(group);

    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_QUANTIZED_LINEAR_CONVOLUTION, &storage->desc };

    bool lhxzp = has_x_zp, lhwzp = has_w_zp, lhbias = has_bias;
    int lsxzp = slot_x_zp, lswzp = slot_w_zp, lsbias = slot_bias;
    result.fixup = [storage, lhxzp, lhwzp, lhbias, lsxzp, lswzp, lsbias](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        int i = 0;
        storage->desc.InputTensor          = &self.input_tensor_descs[i++];
        storage->desc.InputScaleTensor     = &self.input_tensor_descs[i++];
        storage->desc.InputZeroPointTensor = lhxzp ? &self.input_tensor_descs[i++] : nullptr;
        storage->desc.FilterTensor         = &self.input_tensor_descs[i++];
        storage->desc.FilterScaleTensor    = &self.input_tensor_descs[i++];
        storage->desc.FilterZeroPointTensor= lhwzp ? &self.input_tensor_descs[i++] : nullptr;
        storage->desc.BiasTensor           = lhbias ? &self.input_tensor_descs[i++] : nullptr;
        storage->desc.OutputScaleTensor    = &self.input_tensor_descs[i++];
        storage->desc.OutputZeroPointTensor= &self.input_tensor_descs[i++];
        storage->desc.OutputTensor         = &self.output_tensor_descs[0];
        storage->desc.Strides              = storage->strides.data();
        storage->desc.Dilations            = storage->dilations.data();
        storage->desc.StartPadding         = storage->start_pad.data();
        storage->desc.EndPadding           = storage->end_pad.data();
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// ConvInteger → DML_OPERATOR_CONVOLUTION_INTEGER
// ORT ref: DmlOperatorConvInteger.cpp
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateConvInteger(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 2 || outputs.empty()) return std::nullopt;

    enum { xIdx=0, wIdx=1, xZpIdx=2, wZpIdx=3 };

    auto valid = [&](size_t idx) {
        return idx < inputs.size() && !inputs[idx].empty() && value_shapes.count(inputs[idx]);
    };

    auto* x_info = LookupShape(value_shapes, inputs[xIdx]);
    auto* w_info = LookupShape(value_shapes, inputs[wIdx]);
    if (!x_info || !w_info) return std::nullopt;

    bool has_x_zp = valid(xZpIdx);
    bool has_w_zp = valid(wZpIdx);

    OrtNodeAdapter adapter(node, ort_api);
    int64_t group = adapter.GetAttributeInt("group", 1);

    auto x_sizes = x_info->sizes;
    auto w_sizes = w_info->sizes;
    auto k = ReadConvKernelArgs(adapter, x_sizes);

    if (k.auto_pad) {
        for (uint32_t d = 0; d < k.spatial_dim_count; ++d) {
            uint32_t input_length = x_sizes[2 + d];
            uint32_t stride = k.strides[d];
            uint32_t strided_output_length = (input_length + stride - 1) / stride;
            uint32_t kernel_length = 1 + (k.kernel_shape[d] - 1) * k.dilations[d];
            uint32_t length_needed = stride * (strided_output_length - 1) + kernel_length;
            uint32_t padding = (length_needed <= input_length) ? 0 : (length_needed - input_length);
            if (k.auto_pad_same_upper) {
                k.start_padding[d] = padding / 2;
            } else {
                k.start_padding[d] = (padding + 1) / 2;
            }
            k.end_padding[d] = padding - k.start_padding[d];
        }
    }

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    std::vector<uint32_t> out_sizes;
    if (out_edge) {
        out_sizes = out_edge->sizes;
    } else {
        out_sizes = x_sizes;
        out_sizes[1] = w_sizes[0];
        for (uint32_t d = 0; d < k.spatial_dim_count; ++d)
            out_sizes[2 + d] = (x_sizes[2 + d] + k.start_padding[d] + k.end_padding[d] -
                (1 + (k.kernel_shape[d] - 1) * k.dilations[d])) / k.strides[d] + 1;
    }

    struct ConvIntStorage {
        DML_CONVOLUTION_INTEGER_OPERATOR_DESC desc{};
        std::vector<uint32_t> strides, dilations, start_pad, end_pad;
    };
    auto storage = std::make_shared<ConvIntStorage>();
    storage->strides   = k.strides;
    storage->dilations = k.dilations;
    storage->start_pad = k.start_padding;
    storage->end_pad   = k.end_padding;

    TranslatedOp result;
    // DML order: x, x_zp, w, w_zp → input_name_reorder = {0, 2, 1, 3}
    auto x_tensor = MakeTensorInfo(x_sizes, x_info->data_type);
    result.input_tensors.push_back(x_tensor);
    result.input_buffer_descs.push_back(x_tensor.ToBufferDesc());
    result.input_name_reorder.push_back(xIdx);

    if (has_x_zp) {
        auto* xzp_info = LookupShape(value_shapes, inputs[xZpIdx]);
        if (!xzp_info) return std::nullopt;
        auto t = MakeTensorInfo(xzp_info->sizes, xzp_info->data_type);
        result.input_tensors.push_back(t);
        result.input_buffer_descs.push_back(t.ToBufferDesc());
        result.input_name_reorder.push_back(xZpIdx);
    }

    auto w_tensor = MakeTensorInfo(w_sizes, w_info->data_type);
    result.input_tensors.push_back(w_tensor);
    result.input_buffer_descs.push_back(w_tensor.ToBufferDesc());
    result.input_name_reorder.push_back(wIdx);

    if (has_w_zp) {
        auto* wzp_info = LookupShape(value_shapes, inputs[wZpIdx]);
        if (!wzp_info) return std::nullopt;
        uint32_t wzp_orig_rank = wzp_info->original_rank ? wzp_info->original_rank
                               : static_cast<uint32_t>(wzp_info->sizes.size());
        uint32_t ci_dml_dim = static_cast<uint32_t>(x_sizes.size());
        DmlTensorInfo t;
        if (wzp_orig_rank == 1) {
            t = MakeTensorInfoAtAxis({wzp_info->sizes.back()}, wzp_info->data_type, 0, ci_dml_dim);
        } else {
            t = MakeTensorInfo(wzp_info->sizes, wzp_info->data_type);
        }
        result.input_tensors.push_back(t);
        result.input_buffer_descs.push_back(t.ToBufferDesc());
        result.input_name_reorder.push_back(wZpIdx);
    }

    auto out_tensor = MakeTensorInfo(out_sizes, DML_TENSOR_DATA_TYPE_INT32);
    result.input_tensor_descs.resize(result.input_tensors.size());
    result.output_tensors = { out_tensor };
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);

    storage->desc.DimensionCount = k.spatial_dim_count;
    storage->desc.Strides        = storage->strides.data();
    storage->desc.Dilations      = storage->dilations.data();
    storage->desc.StartPadding   = storage->start_pad.data();
    storage->desc.EndPadding     = storage->end_pad.data();
    storage->desc.GroupCount     = static_cast<UINT>(group);

    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_CONVOLUTION_INTEGER, &storage->desc };

    bool lhxzp = has_x_zp, lhwzp = has_w_zp;
    result.fixup = [storage, lhxzp, lhwzp](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        int i = 0;
        storage->desc.InputTensor          = &self.input_tensor_descs[i++];
        storage->desc.InputZeroPointTensor = lhxzp ? &self.input_tensor_descs[i++] : nullptr;
        storage->desc.FilterTensor         = &self.input_tensor_descs[i++];
        storage->desc.FilterZeroPointTensor= lhwzp ? &self.input_tensor_descs[i++] : nullptr;
        storage->desc.OutputTensor         = &self.output_tensor_descs[0];
        storage->desc.Strides              = storage->strides.data();
        storage->desc.Dilations            = storage->dilations.data();
        storage->desc.StartPadding         = storage->start_pad.data();
        storage->desc.EndPadding           = storage->end_pad.data();
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// QLinearAveragePool / QLinearGlobalAveragePool
// → DML_OPERATOR_QUANTIZED_LINEAR_AVERAGE_POOLING
// ORT ref: DmlOperatorQLinearAveragePooling.cpp
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateQLinearAveragePool(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&,
    bool global = false) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 5 || outputs.empty()) return std::nullopt;

    enum { inIdx=0, inScaleIdx=1, inZpIdx=2, outScaleIdx=3, outZpIdx=4 };

    auto* in_info = LookupShape(value_shapes, inputs[inIdx]);
    if (!in_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    int64_t count_include_pad = adapter.GetAttributeInt("count_include_pad", 0);
    int64_t channels_last = adapter.GetAttributeInt("channels_last", 0);
    bool isNhwc = (channels_last != 0);

    auto in_sizes = in_info->sizes;
    // NHWC→NCHW permutation for DML.
    if (isNhwc && in_sizes.size() == 4) {
        in_sizes = { in_sizes[0], in_sizes[3], in_sizes[1], in_sizes[2] };
    }
    uint32_t spatial = static_cast<uint32_t>(in_sizes.size()) - 2;
    if (spatial < 1) spatial = 1;

    std::vector<uint32_t> window_size, strides, start_pad, end_pad, dilations;
    if (global) {
        for (uint32_t i = 0; i < spatial; ++i) window_size.push_back(in_sizes[2 + i]);
        strides.assign(spatial, 1u);
        start_pad.assign(spatial, 0u);
        end_pad.assign(spatial, 0u);
        dilations.assign(spatial, 1u);
    } else {
        auto ks = adapter.GetAttributeInts("kernel_shape");
        for (int64_t v : ks) window_size.push_back(static_cast<uint32_t>(v));
        while (window_size.size() < spatial) window_size.insert(window_size.begin(), 1u);
        auto st = adapter.GetAttributeInts("strides");
        for (int64_t v : st) strides.push_back(static_cast<uint32_t>(v));
        while (strides.size() < spatial) strides.push_back(1u);
        auto pads = adapter.GetAttributeInts("pads");
        if (pads.size() >= 2 * spatial) {
            for (size_t i = 0; i < spatial; ++i) start_pad.push_back(static_cast<uint32_t>(pads[i]));
            for (size_t i = 0; i < spatial; ++i) end_pad.push_back(static_cast<uint32_t>(pads[spatial + i]));
        } else {
            start_pad.assign(spatial, 0u);
            end_pad.assign(spatial, 0u);
        }
        auto dil = adapter.GetAttributeInts("dilations");
        for (int64_t v : dil) dilations.push_back(static_cast<uint32_t>(v));
        while (dilations.size() < spatial) dilations.push_back(1u);
    }

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    std::vector<uint32_t> out_sizes;
    if (out_edge) {
        out_sizes = out_edge->sizes;
        if (isNhwc && out_sizes.size() == 4) {
            out_sizes = { out_sizes[0], out_sizes[3], out_sizes[1], out_sizes[2] };
        }
    } else {
        out_sizes = in_sizes;
        for (uint32_t d = 0; d < spatial; ++d) {
            uint32_t eff_k = 1 + (window_size[d] - 1) * dilations[d];
            out_sizes[2 + d] = (in_sizes[2 + d] + start_pad[d] + end_pad[d] - eff_k) / strides[d] + 1;
        }
    }

    auto in_tensor = MakeTensorInfo(in_sizes, in_info->data_type);

    auto* is_info = LookupShape(value_shapes, inputs[inScaleIdx]);
    auto* izp_info = LookupShape(value_shapes, inputs[inZpIdx]);
    auto* os_info = LookupShape(value_shapes, inputs[outScaleIdx]);
    auto* ozp_info = LookupShape(value_shapes, inputs[outZpIdx]);
    if (!is_info || !izp_info || !os_info || !ozp_info) return std::nullopt;

    auto in_scale_tensor = MakeTensorInfo(is_info->sizes, is_info->data_type);
    auto in_zp_tensor    = MakeTensorInfo(izp_info->sizes, izp_info->data_type);
    auto out_scale_tensor= MakeTensorInfo(os_info->sizes, os_info->data_type);
    auto out_zp_tensor   = MakeTensorInfo(ozp_info->sizes, ozp_info->data_type);
    auto out_tensor      = MakeTensorInfo(out_sizes, in_info->data_type);

    struct QLinAvgPoolStorage {
        DML_QUANTIZED_LINEAR_AVERAGE_POOLING_OPERATOR_DESC desc{};
        std::vector<uint32_t> window_size, strides, start_pad, end_pad, dilations;
    };
    auto storage = std::make_shared<QLinAvgPoolStorage>();
    storage->window_size = window_size;
    storage->strides     = strides;
    storage->start_pad   = start_pad;
    storage->end_pad     = end_pad;
    storage->dilations   = dilations;
    storage->desc.DimensionCount  = spatial;
    storage->desc.WindowSize      = storage->window_size.data();
    storage->desc.Strides         = storage->strides.data();
    storage->desc.StartPadding    = storage->start_pad.data();
    storage->desc.EndPadding      = storage->end_pad.data();
    storage->desc.Dilations       = storage->dilations.data();
    storage->desc.IncludePadding  = (count_include_pad != 0);

    TranslatedOp result;
    result.input_tensors = { in_tensor, in_scale_tensor, in_zp_tensor, out_scale_tensor, out_zp_tensor };
    result.output_tensors = { out_tensor };
    for (auto& t : result.input_tensors) result.input_buffer_descs.push_back(t.ToBufferDesc());
    result.input_tensor_descs.resize(result.input_tensors.size());
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);
    result.desc_storage = storage;
    result.op_desc = { static_cast<DML_OPERATOR_TYPE>(DML_OPERATOR_QUANTIZED_LINEAR_AVERAGE_POOLING), &storage->desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->desc.InputTensor          = &self.input_tensor_descs[0];
        storage->desc.InputScaleTensor     = &self.input_tensor_descs[1];
        storage->desc.InputZeroPointTensor = &self.input_tensor_descs[2];
        storage->desc.OutputScaleTensor    = &self.input_tensor_descs[3];
        storage->desc.OutputZeroPointTensor= &self.input_tensor_descs[4];
        storage->desc.OutputTensor         = &self.output_tensor_descs[0];
        storage->desc.WindowSize           = storage->window_size.data();
        storage->desc.Strides              = storage->strides.data();
        storage->desc.StartPadding         = storage->start_pad.data();
        storage->desc.EndPadding           = storage->end_pad.data();
        storage->desc.Dilations            = storage->dilations.data();
    };
    result.FixupPointers();
    return result;
}

// ---------------------------------------------------------------------------
// DynamicQuantizeMatMul → DynamicQuantizeLinear + MatMulIntegerToFloat
// ORT ref: DmlOperatorDynamicQuantizeMatMul.cpp
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateDynamicQuantizeMatMul(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 3 || outputs.empty()) return std::nullopt;

    enum { aIdx=0, bIdx=1, bScaleIdx=2, bZpIdx=3, biasIdx=4 };

    auto valid = [&](size_t idx) {
        return idx < inputs.size() && !inputs[idx].empty() && value_shapes.count(inputs[idx]);
    };

    auto* a_info = LookupShape(value_shapes, inputs[aIdx]);
    auto* b_info = LookupShape(value_shapes, inputs[bIdx]);
    auto* bs_info = LookupShape(value_shapes, inputs[bScaleIdx]);
    if (!a_info || !b_info || !bs_info) return std::nullopt;

    bool has_b_zp = valid(bZpIdx);
    bool has_bias = valid(biasIdx);

    auto a_sizes = a_info->sizes;
    auto b_sizes = b_info->sizes;

    // Output shape: matmul(A, B)
    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    std::vector<uint32_t> out_sizes;
    if (out_edge) {
        out_sizes = out_edge->sizes;
    } else {
        out_sizes = a_sizes;
        out_sizes.back() = b_sizes.back();
    }

    // Broadcast A and B to matching batch dims using stride=0 for broadcast dims.
    std::vector<uint32_t> a_broadcast = a_sizes, b_broadcast = b_sizes;
    while (a_broadcast.size() < out_sizes.size()) a_broadcast.insert(a_broadcast.begin(), 1u);
    while (b_broadcast.size() < out_sizes.size()) b_broadcast.insert(b_broadcast.begin(), 1u);

    // A: broadcast batch dims where A has 1 but output has > 1.
    auto a_strides = ComputePackedStrides(a_broadcast);
    std::vector<uint32_t> a_desired = a_broadcast;
    for (size_t d = 0; d + 2 < a_broadcast.size(); ++d) {
        if (a_broadcast[d] == 1 && out_sizes[d] > 1) {
            a_desired[d] = out_sizes[d];
            a_strides[d] = 0;
        }
    }
    uint64_t a_bytes = ComputeAlignedTotalBytes(a_broadcast, a_info->data_type);
    auto a_tensor = MakeTensorInfoWithStrides(a_desired, a_strides, a_info->data_type, a_bytes);

    // Intermediate tensors for DynamicQuantizeLinear outputs.
    // These are freshly allocated intermediates — use packed strides, not broadcast.
    auto quant_a_tensor = MakeTensorInfo(a_broadcast, b_info->data_type);
    std::vector<uint32_t> scalar_shape(a_broadcast.size(), 1u);
    auto a_scale_tensor = MakeTensorInfo(scalar_shape, DML_TENSOR_DATA_TYPE_FLOAT32);
    auto a_zp_tensor    = MakeTensorInfo(scalar_shape, b_info->data_type);

    // B: broadcast batch dims where B has 1 but output has > 1.
    auto b_strides = ComputePackedStrides(b_broadcast);
    std::vector<uint32_t> b_desired = b_broadcast;
    for (size_t d = 0; d + 2 < b_broadcast.size(); ++d) {
        if (b_broadcast[d] == 1 && out_sizes[d] > 1) {
            b_desired[d] = out_sizes[d];
            b_strides[d] = 0;
        }
    }
    uint64_t b_bytes = ComputeAlignedTotalBytes(b_broadcast, b_info->data_type);
    auto b_tensor = MakeTensorInfoWithStrides(b_desired, b_strides, b_info->data_type, b_bytes);
    auto out_tensor = MakeTensorInfo(out_sizes, DML_TENSOR_DATA_TYPE_FLOAT32);

    struct DynQuantMatMulStorage {
        DML_DYNAMIC_QUANTIZE_LINEAR_OPERATOR_DESC dql_desc{};
        DML_MATRIX_MULTIPLY_INTEGER_TO_FLOAT_OPERATOR_DESC mmitf_desc{};
        DmlTensorInfo quant_a, a_scale, a_zp;
        DML_BUFFER_TENSOR_DESC quant_a_buf{}, a_scale_buf{}, a_zp_buf{};
        DML_TENSOR_DESC quant_a_td{}, a_scale_td{}, a_zp_td{};
    };
    auto storage = std::make_shared<DynQuantMatMulStorage>();
    storage->quant_a = quant_a_tensor;
    storage->a_scale = a_scale_tensor;
    storage->a_zp    = a_zp_tensor;
    storage->quant_a_buf = quant_a_tensor.ToBufferDesc();
    storage->a_scale_buf = a_scale_tensor.ToBufferDesc();
    storage->a_zp_buf    = a_zp_tensor.ToBufferDesc();
    storage->quant_a_td  = { DML_TENSOR_TYPE_BUFFER, &storage->quant_a_buf };
    storage->a_scale_td  = { DML_TENSOR_TYPE_BUFFER, &storage->a_scale_buf };
    storage->a_zp_td     = { DML_TENSOR_TYPE_BUFFER, &storage->a_zp_buf };

    // Primary: DynamicQuantizeLinear(A)
    TranslatedOp result;
    result.input_tensors = { a_tensor };
    result.input_buffer_descs = { a_tensor.ToBufferDesc() };
    result.input_name_reorder = { aIdx };

    // Append remaining ONNX inputs for sub_node graph_inputs wiring.
    result.input_tensors.push_back(b_tensor);
    result.input_buffer_descs.push_back(b_tensor.ToBufferDesc());
    result.input_name_reorder.push_back(bIdx);

    auto bs_tensor = MakeTensorInfo(bs_info->sizes, bs_info->data_type);
    result.input_tensors.push_back(bs_tensor);
    result.input_buffer_descs.push_back(bs_tensor.ToBufferDesc());
    result.input_name_reorder.push_back(bScaleIdx);

    DmlTensorInfo bzp_tensor{}, bias_tensor{};
    if (has_b_zp) {
        auto* bzp_info = LookupShape(value_shapes, inputs[bZpIdx]);
        if (!bzp_info) return std::nullopt;
        bzp_tensor = MakeTensorInfo(bzp_info->sizes, bzp_info->data_type);
        result.input_tensors.push_back(bzp_tensor);
        result.input_buffer_descs.push_back(bzp_tensor.ToBufferDesc());
        result.input_name_reorder.push_back(bZpIdx);
    }
    if (has_bias) {
        auto* bias_info = LookupShape(value_shapes, inputs[biasIdx]);
        if (!bias_info) return std::nullopt;
        bias_tensor = MakeTensorInfo(bias_info->sizes, bias_info->data_type);
        result.input_tensors.push_back(bias_tensor);
        result.input_buffer_descs.push_back(bias_tensor.ToBufferDesc());
        result.input_name_reorder.push_back(biasIdx);
    }

    result.primary_input_count = 1;
    result.input_tensor_descs.resize(result.input_tensors.size());
    result.output_tensors = { out_tensor };
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);

    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_DYNAMIC_QUANTIZE_LINEAR, &storage->dql_desc };
    result.fixup = [storage](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->quant_a_buf.Sizes = storage->quant_a.sizes.data();
        storage->quant_a_buf.Strides = storage->quant_a.strides.empty() ? nullptr : storage->quant_a.strides.data();
        storage->quant_a_td = { DML_TENSOR_TYPE_BUFFER, &storage->quant_a_buf };
        storage->a_scale_buf.Sizes = storage->a_scale.sizes.data();
        storage->a_scale_buf.Strides = storage->a_scale.strides.empty() ? nullptr : storage->a_scale.strides.data();
        storage->a_scale_td = { DML_TENSOR_TYPE_BUFFER, &storage->a_scale_buf };
        storage->a_zp_buf.Sizes = storage->a_zp.sizes.data();
        storage->a_zp_buf.Strides = storage->a_zp.strides.empty() ? nullptr : storage->a_zp.strides.data();
        storage->a_zp_td = { DML_TENSOR_TYPE_BUFFER, &storage->a_zp_buf };
        storage->dql_desc.InputTensor           = &self.input_tensor_descs[0];
        storage->dql_desc.OutputTensor           = &storage->quant_a_td;
        storage->dql_desc.OutputScaleTensor      = &storage->a_scale_td;
        storage->dql_desc.OutputZeroPointTensor  = &storage->a_zp_td;
    };
    result.FixupPointers();

    // Sub_node 0: MatMulIntegerToFloat(quant_A, A_scale, A_zp, B, B_scale, [B_zp], [Bias])
    SubNode mmitf_node;
    mmitf_node.input_tensors = { quant_a_tensor, a_scale_tensor, a_zp_tensor, b_tensor, bs_tensor };
    mmitf_node.input_buffer_descs = { quant_a_tensor.ToBufferDesc(), a_scale_tensor.ToBufferDesc(),
                                       a_zp_tensor.ToBufferDesc(), b_tensor.ToBufferDesc(), bs_tensor.ToBufferDesc() };
    mmitf_node.input_from = { {-1, 0}, {-1, 1}, {-1, 2}, {-2, 0}, {-2, 0} };
    mmitf_node.graph_inputs = { {bIdx, 3}, {bScaleIdx, 4} };

    if (has_b_zp) {
        mmitf_node.input_tensors.push_back(bzp_tensor);
        mmitf_node.input_buffer_descs.push_back(bzp_tensor.ToBufferDesc());
        mmitf_node.input_from.push_back({-2, 0});
        mmitf_node.graph_inputs.push_back({bZpIdx, 5});
    }
    if (has_bias) {
        mmitf_node.input_tensors.push_back(bias_tensor);
        mmitf_node.input_buffer_descs.push_back(bias_tensor.ToBufferDesc());
        mmitf_node.input_from.push_back({-2, 0});
        size_t mmitf_bias_slot = has_b_zp ? 6 : 5;
        mmitf_node.graph_inputs.push_back({biasIdx, mmitf_bias_slot});
    }

    mmitf_node.input_tensor_descs.resize(mmitf_node.input_tensors.size());
    mmitf_node.output_tensors = { out_tensor };
    mmitf_node.output_buffer_descs = { out_tensor.ToBufferDesc() };
    mmitf_node.output_tensor_descs.resize(1);
    mmitf_node.desc_storage = storage;
    mmitf_node.op_desc = { DML_OPERATOR_MATRIX_MULTIPLY_INTEGER_TO_FLOAT, &storage->mmitf_desc };

    bool lhbzp = has_b_zp, lhbias = has_bias;
    mmitf_node.fixup = [storage, lhbzp, lhbias](SubNode& self) {
        RebuildSubNodePointers(self);
        int i = 0;
        storage->mmitf_desc.ATensor         = &self.input_tensor_descs[i++];
        storage->mmitf_desc.AScaleTensor    = &self.input_tensor_descs[i++];
        storage->mmitf_desc.AZeroPointTensor= &self.input_tensor_descs[i++];
        storage->mmitf_desc.BTensor         = &self.input_tensor_descs[i++];
        storage->mmitf_desc.BScaleTensor    = &self.input_tensor_descs[i++];
        storage->mmitf_desc.BZeroPointTensor= lhbzp ? &self.input_tensor_descs[i++] : nullptr;
        storage->mmitf_desc.BiasTensor      = lhbias ? &self.input_tensor_descs[i++] : nullptr;
        storage->mmitf_desc.OutputTensor    = &self.output_tensor_descs[0];
    };
    mmitf_node.FixupPointers();
    result.sub_nodes.push_back(std::move(mmitf_node));

    return result;
}

// ---------------------------------------------------------------------------
// GroupQueryAttention → DML_OPERATOR_MULTIHEAD_ATTENTION1
// ORT ref: DmlOperatorGroupQueryAttention.cpp
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateGroupQueryAttention(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 3 || outputs.size() < 3) return std::nullopt;

    enum { queryIdx=0, keyIdx=1, valueIdx=2, pastKeyIdx=3, pastValueIdx=4, seqLensIdx=5 };

    auto valid = [&](size_t idx) {
        return idx < inputs.size() && !inputs[idx].empty() && value_shapes.count(inputs[idx]);
    };

    auto* q_info = LookupShape(value_shapes, inputs[queryIdx]);
    auto* k_info = LookupShape(value_shapes, inputs[keyIdx]);
    auto* v_info = LookupShape(value_shapes, inputs[valueIdx]);
    if (!q_info || !k_info || !v_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    uint32_t numHeads = static_cast<uint32_t>(adapter.GetAttributeInt("num_heads", 0));
    uint32_t kvNumHeads = static_cast<uint32_t>(adapter.GetAttributeInt("kv_num_heads", 0));
    if (numHeads == 0 || kvNumHeads == 0) return std::nullopt;

    // Q: [B, S, qHidden], K: [B, kvS, kvHidden], V: [B, kvS, kvHidden]
    auto q_sizes = q_info->sizes;
    auto k_sizes = k_info->sizes;
    auto v_sizes = v_info->sizes;

    auto get_rank_offset = [](const DmlTensorInfo* info) -> size_t {
        uint32_t r = info->original_rank ? info->original_rank : static_cast<uint32_t>(info->sizes.size());
        return info->sizes.size() >= r ? info->sizes.size() - r : 0;
    };

    size_t q_off = get_rank_offset(q_info);
    uint32_t batchSize = q_sizes[q_off];
    uint32_t sequenceLength = q_sizes[q_off + 1];
    uint32_t queryHiddenSize = q_sizes[q_off + 2];
    uint32_t queryHeadSize = queryHiddenSize / numHeads;

    size_t k_off = get_rank_offset(k_info);
    uint32_t kvSequenceLength = k_sizes[k_off + 1];
    uint32_t kvHiddenSize = k_sizes[k_off + 2];

    float scale = adapter.GetAttributeFloat("scale", 0.0f);
    if (scale == 0.0f) scale = 1.0f / std::sqrt(static_cast<float>(queryHeadSize));

    bool hasSeqLens = valid(seqLensIdx);

    struct GQAStorage {
        DML_MULTIHEAD_ATTENTION1_OPERATOR_DESC mha1_desc{};
        DML_FILL_VALUE_CONSTANT_OPERATOR_DESC fill_desc{};
        int out_present_key = -1, out_present_value = -1;
    };
    auto storage = std::make_shared<GQAStorage>();

    auto qt = MakeTensorInfo(q_sizes, q_info->data_type);
    auto kt = MakeTensorInfo(k_sizes, k_info->data_type);
    auto vt = MakeTensorInfo(v_sizes, v_info->data_type);

    std::vector<uint32_t> past_seq_shape = { batchSize };
    auto past_seq_tensor = MakeTensorInfo(PadToMinDims(past_seq_shape), DML_TENSOR_DATA_TYPE_INT32);

    float local_scale = scale;
    uint32_t local_num_heads = numHeads, local_kv_heads = kvNumHeads;

    // --- Build the translator based on sequenceLength and fp16 ---
    // For all cases: use FILL_VALUE_CONSTANT primary + MHA1 sub_node.
    // When seqLen==1 and hasSeqLens, wire seqlens input; otherwise generate zeros.
    TranslatedOp result;
    result.input_tensors = { qt, kt, vt };
    result.input_buffer_descs = { qt.ToBufferDesc(), kt.ToBufferDesc(), vt.ToBufferDesc() };
    result.input_name_reorder = { queryIdx, keyIdx, valueIdx };

    if (sequenceLength == 1 && hasSeqLens) {
        auto* sl_info = LookupShape(value_shapes, inputs[seqLensIdx]);
        if (!sl_info) return std::nullopt;
        auto slt = MakeTensorInfo(sl_info->sizes, sl_info->data_type);
        result.input_tensors.push_back(slt);
        result.input_buffer_descs.push_back(slt.ToBufferDesc());
        result.input_name_reorder.push_back(seqLensIdx);
    }

    result.primary_input_count = 0;
    result.input_tensor_descs.resize(result.input_tensors.size());

    // Outputs: output, present_key, present_value.
    auto* out0_info = LookupShape(value_shapes, outputs[0]);
    if (!out0_info) {
        result.output_tensors.push_back(MakeTensorInfo(PadToMinDims({batchSize, sequenceLength, queryHiddenSize}), q_info->data_type));
    } else {
        result.output_tensors.push_back(MakeTensorInfo(out0_info->sizes, q_info->data_type));
    }
    result.output_buffer_descs.push_back(result.output_tensors[0].ToBufferDesc());
    auto* pk_out = (outputs.size() > 1 && !outputs[1].empty()) ? LookupShape(value_shapes, outputs[1]) : nullptr;
    if (pk_out) {
        auto pkt = MakeTensorInfo(pk_out->sizes, q_info->data_type);
        result.output_tensors.push_back(pkt);
        result.output_buffer_descs.push_back(pkt.ToBufferDesc());
        storage->out_present_key = 1;
    }
    auto* pv_out = (outputs.size() > 2 && !outputs[2].empty()) ? LookupShape(value_shapes, outputs[2]) : nullptr;
    if (pv_out) {
        auto pvt = MakeTensorInfo(pv_out->sizes, q_info->data_type);
        result.output_tensors.push_back(pvt);
        result.output_buffer_descs.push_back(pvt.ToBufferDesc());
        storage->out_present_value = static_cast<int>(result.output_tensors.size()) - 1;
    }
    result.output_tensor_descs.resize(result.output_tensors.size());

    // Primary: FILL_VALUE_CONSTANT (zeros) or identity pass-through for seqlens.
    storage->fill_desc.ValueDataType = DML_TENSOR_DATA_TYPE_INT32;
    storage->fill_desc.Value.Int32 = 0;
    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_FILL_VALUE_CONSTANT, &storage->fill_desc };
    result.fixup = [storage, past_seq_tensor](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->fill_desc.OutputTensor = nullptr;
    };
    result.FixupPointers();

    // Determine how PastSequenceLengths reaches MHA1.
    bool seqlens_from_primary = !(sequenceLength == 1 && hasSeqLens);

    // Build the MHA1 sub_node with proper DML schema slot wiring.
    // DML MHA1 has 12 input slots (0-11). We use Q(0), K(1), V(2), PastSeqLens(11).
    // input_from needs 12 entries; slots 3-10 are skip sentinels.
    // The fixup maps local indices to the desc fields by position.
    // For fp16: DML internally handles half-precision via metacommand.
    // ORT adds explicit casts but that requires multi-output sub_node routing
    // which our framework doesn't support. Instead, we pass fp16 directly —
    // DML's MHA1 metacommand handles fp16 internally (same as TranslateMultiHeadAttention).
    {
        SubNode mha_node;
        // 12 input slots matching DML MHA1 schema.
        mha_node.input_tensors.resize(12);
        mha_node.input_buffer_descs.resize(12);
        mha_node.input_tensor_descs.resize(12);
        // Fill Q(0), K(1), V(2) tensor info.
        mha_node.input_tensors[0] = qt;
        mha_node.input_tensors[1] = kt;
        mha_node.input_tensors[2] = vt;
        mha_node.input_tensors[11] = past_seq_tensor;
        mha_node.input_buffer_descs[0] = qt.ToBufferDesc();
        mha_node.input_buffer_descs[1] = kt.ToBufferDesc();
        mha_node.input_buffer_descs[2] = vt.ToBufferDesc();
        mha_node.input_buffer_descs[11] = past_seq_tensor.ToBufferDesc();

        // input_from: Q/K/V from graph_inputs (skip), slots 3-10 skip, slot 11 from primary or graph_inputs.
        mha_node.input_from.resize(12, {-2, 0});
        if (seqlens_from_primary) {
            mha_node.input_from[11] = {-1, 0};
        }
        mha_node.graph_inputs = { {queryIdx, 0}, {keyIdx, 1}, {valueIdx, 2} };
        if (!seqlens_from_primary)
            mha_node.graph_inputs.push_back({seqLensIdx, 11});

        mha_node.output_tensors = result.output_tensors;
        for (auto& ot : mha_node.output_tensors)
            mha_node.output_buffer_descs.push_back(ot.ToBufferDesc());
        mha_node.output_tensor_descs.resize(result.output_tensors.size());
        mha_node.desc_storage = storage;
        mha_node.op_desc = { DML_OPERATOR_MULTIHEAD_ATTENTION1, &storage->mha1_desc };

        mha_node.fixup = [storage, local_scale, local_num_heads, local_kv_heads](SubNode& self) {
            RebuildSubNodePointers(self);
            storage->mha1_desc.QueryTensor              = &self.input_tensor_descs[0];
            storage->mha1_desc.KeyTensor                = &self.input_tensor_descs[1];
            storage->mha1_desc.ValueTensor              = &self.input_tensor_descs[2];
            storage->mha1_desc.StackedQueryKeyTensor    = nullptr;
            storage->mha1_desc.StackedKeyValueTensor    = nullptr;
            storage->mha1_desc.StackedQueryKeyValueTensor = nullptr;
            storage->mha1_desc.BiasTensor               = nullptr;
            storage->mha1_desc.MaskTensor               = nullptr;
            storage->mha1_desc.RelativePositionBiasTensor = nullptr;
            storage->mha1_desc.PastKeyTensor            = nullptr;
            storage->mha1_desc.PastValueTensor          = nullptr;
            storage->mha1_desc.PastSequenceLengthsTensor = &self.input_tensor_descs[11];
            storage->mha1_desc.OutputTensor             = &self.output_tensor_descs[0];
            storage->mha1_desc.OutputPresentKeyTensor   = self.output_tensor_descs.size() > 1 ? &self.output_tensor_descs[1] : nullptr;
            storage->mha1_desc.OutputPresentValueTensor = self.output_tensor_descs.size() > 2 ? &self.output_tensor_descs[2] : nullptr;
            storage->mha1_desc.Scale = local_scale;
            storage->mha1_desc.MaskFilterValue = -10000.0f;
            storage->mha1_desc.QueryHeadCount = local_num_heads;
            storage->mha1_desc.KeyValueHeadCount = local_kv_heads;
            storage->mha1_desc.MaskType = DML_MULTIHEAD_ATTENTION_MASK_TYPE_NONE;
        };
        mha_node.FixupPointers();
        result.sub_nodes.push_back(std::move(mha_node));
    }

    return result;
}

// ---------------------------------------------------------------------------
// QLinearConcat → DequantizeLinear × N + Join + QuantizeLinear
// ORT ref: DmlOperatorQLinearConcat.cpp
// ---------------------------------------------------------------------------

static std::optional<TranslatedOp> TranslateQLinearConcat(
    const OrtApi& ort_api,
    const OrtNode* node,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.size() < 5 || outputs.empty()) return std::nullopt;

    // inputs: y_scale(0), y_zp(1), then N tuples of (x_tensor, x_scale, x_zp)
    uint32_t input_count = static_cast<uint32_t>((inputs.size() - 2) / 3);
    if (input_count < 1 || (inputs.size() - 2) % 3 != 0) return std::nullopt;

    auto* ys_info = LookupShape(value_shapes, inputs[0]);
    auto* yzp_info = LookupShape(value_shapes, inputs[1]);
    if (!ys_info || !yzp_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    int64_t axis = adapter.GetAttributeInt("axis", 0);

    auto* out_edge = LookupShape(value_shapes, outputs[0]);
    if (!out_edge) return std::nullopt;

    // Determine the adjusted DML axis.
    auto out_sizes = out_edge->sizes;
    uint32_t rank = static_cast<uint32_t>(out_sizes.size());
    if (axis < 0) axis += rank;
    uint32_t dml_axis = static_cast<uint32_t>(axis);

    struct QLinConcatStorage {
        std::vector<std::shared_ptr<void>> sub_storages;
        DML_JOIN_OPERATOR_DESC join_desc{};
        DML_ELEMENT_WISE_QUANTIZE_LINEAR_OPERATOR_DESC quant_desc{};
        std::vector<DmlTensorInfo> deq_out_infos;
        std::vector<DML_BUFFER_TENSOR_DESC> deq_out_bufs;
        std::vector<DML_TENSOR_DESC> deq_out_tds;
        std::vector<DML_BUFFER_TENSOR_DESC> join_in_bufs;
        std::vector<DML_TENSOR_DESC> join_in_tds;
        DmlTensorInfo join_out_info;
        DML_BUFFER_TENSOR_DESC join_out_buf{};
        DML_TENSOR_DESC join_out_td{};
        uint32_t axis;
    };
    auto storage = std::make_shared<QLinConcatStorage>();
    storage->axis = dml_axis;

    // Build all input tensors (y_scale, y_zp, then all tuples).
    TranslatedOp result;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].empty() || !value_shapes.count(inputs[i])) return std::nullopt;
        auto* info = LookupShape(value_shapes, inputs[i]);
        if (!info) return std::nullopt;
        auto t = MakeTensorInfo(info->sizes, info->data_type);
        result.input_tensors.push_back(t);
        result.input_buffer_descs.push_back(t.ToBufferDesc());
        result.input_name_reorder.push_back(i);
    }

    // Primary: DequantizeLinear for first tuple (x0_tensor(2), x0_scale(3), x0_zp(4))
    result.primary_input_count = 3;  // x0_tensor, x0_scale, x0_zp

    // But the input_tensors start with y_scale(0), y_zp(1). We need the primary to
    // operate on the first tuple at ONNX indices 2,3,4. The primary gets the first
    // primary_input_count entries from input_tensors. So input_tensors[0..2] should be
    // the first tuple, and y_scale/y_zp should come after.
    // Reorder: put x0 tuple first, then rest.
    {
        // Rebuild input_tensors in order: x0_tuple, x1_tuple, ..., y_scale, y_zp
        std::vector<DmlTensorInfo> reordered_tensors;
        std::vector<DML_BUFFER_TENSOR_DESC> reordered_bufs;
        std::vector<size_t> reordered_names;

        // First: x0 tuple (ONNX 2,3,4)
        for (size_t i = 2; i < 5; ++i) {
            reordered_tensors.push_back(result.input_tensors[i]);
            reordered_bufs.push_back(result.input_buffer_descs[i]);
            reordered_names.push_back(i);
        }
        // Then: remaining tuples (ONNX 5,6,7,...,N*3+1)
        for (uint32_t t = 1; t < input_count; ++t) {
            size_t base = 2 + t * 3;
            for (size_t j = 0; j < 3; ++j) {
                reordered_tensors.push_back(result.input_tensors[base + j]);
                reordered_bufs.push_back(result.input_buffer_descs[base + j]);
                reordered_names.push_back(base + j);
            }
        }
        // Then: y_scale(0), y_zp(1)
        reordered_tensors.push_back(result.input_tensors[0]);
        reordered_bufs.push_back(result.input_buffer_descs[0]);
        reordered_names.push_back(0);
        reordered_tensors.push_back(result.input_tensors[1]);
        reordered_bufs.push_back(result.input_buffer_descs[1]);
        reordered_names.push_back(1);

        result.input_tensors = std::move(reordered_tensors);
        result.input_buffer_descs = std::move(reordered_bufs);
        result.input_name_reorder = std::move(reordered_names);
    }

    result.input_tensor_descs.resize(result.input_tensors.size());

    auto out_tensor = MakeTensorInfo(out_sizes, yzp_info->data_type);
    result.output_tensors = { out_tensor };
    result.output_buffer_descs = { out_tensor.ToBufferDesc() };
    result.output_tensor_descs.resize(1);

    // Primary: DequantizeLinear on x0
    auto* x0_info = LookupShape(value_shapes, inputs[2]);
    auto deq0_out = MakeTensorInfo(x0_info->sizes, DML_TENSOR_DATA_TYPE_FLOAT32);
    storage->deq_out_infos.push_back(deq0_out);
    storage->deq_out_bufs.push_back(deq0_out.ToBufferDesc());
    storage->deq_out_tds.push_back({ DML_TENSOR_TYPE_BUFFER, &storage->deq_out_bufs.back() });

    struct DeqStorage { DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_DESC desc{}; };
    auto deq0_store = std::make_shared<DeqStorage>();
    storage->sub_storages.push_back(deq0_store);

    result.desc_storage = storage;
    result.op_desc = { DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR, &deq0_store->desc };
    result.fixup = [storage, deq0_store](TranslatedOp& self) {
        RebuildTensorDescPointers(self);
        storage->deq_out_bufs[0].Sizes = storage->deq_out_infos[0].sizes.data();
        storage->deq_out_bufs[0].Strides = storage->deq_out_infos[0].strides.empty()
            ? nullptr : storage->deq_out_infos[0].strides.data();
        storage->deq_out_tds[0] = { DML_TENSOR_TYPE_BUFFER, &storage->deq_out_bufs[0] };
        deq0_store->desc.InputTensor    = &self.input_tensor_descs[0];
        deq0_store->desc.ScaleTensor    = &self.input_tensor_descs[1];
        deq0_store->desc.ZeroPointTensor= &self.input_tensor_descs[2];
        deq0_store->desc.OutputTensor   = &storage->deq_out_tds[0];
    };
    result.FixupPointers();

    // Sub_nodes for remaining DequantizeLinear (indices 1..N-1)
    for (uint32_t t = 1; t < input_count; ++t) {
        size_t onnx_base = 2 + t * 3;
        auto* xt_info = LookupShape(value_shapes, inputs[onnx_base]);
        if (!xt_info) return std::nullopt;
        auto deq_out = MakeTensorInfo(xt_info->sizes, DML_TENSOR_DATA_TYPE_FLOAT32);
        storage->deq_out_infos.push_back(deq_out);

        auto deq_store = std::make_shared<DeqStorage>();
        storage->sub_storages.push_back(deq_store);

        SubNode deq_node;
        deq_node.input_tensors.resize(3);
        deq_node.input_buffer_descs.resize(3);
        deq_node.input_tensor_descs.resize(3);
        // All 3 inputs from graph_inputs.
        deq_node.input_from = { {-2, 0}, {-2, 0}, {-2, 0} };
        deq_node.graph_inputs = { {onnx_base, 0}, {onnx_base + 1, 1}, {onnx_base + 2, 2} };

        // Fill input tensor info for desc pointers.
        for (size_t j = 0; j < 3; ++j) {
            auto* ti = LookupShape(value_shapes, inputs[onnx_base + j]);
            if (!ti) return std::nullopt;
            deq_node.input_tensors[j] = MakeTensorInfo(ti->sizes, ti->data_type);
            deq_node.input_buffer_descs[j] = deq_node.input_tensors[j].ToBufferDesc();
        }

        deq_node.output_tensors = { deq_out };
        deq_node.output_buffer_descs = { deq_out.ToBufferDesc() };
        deq_node.output_tensor_descs.resize(1);
        deq_node.desc_storage = deq_store;
        deq_node.op_desc = { DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR, &deq_store->desc };
        deq_node.fixup = [deq_store](SubNode& self) {
            RebuildSubNodePointers(self);
            deq_store->desc.InputTensor    = &self.input_tensor_descs[0];
            deq_store->desc.ScaleTensor    = &self.input_tensor_descs[1];
            deq_store->desc.ZeroPointTensor= &self.input_tensor_descs[2];
            deq_store->desc.OutputTensor   = &self.output_tensor_descs[0];
        };
        deq_node.FixupPointers();
        result.sub_nodes.push_back(std::move(deq_node));
    }

    // Join sub_node: collects all dequantized outputs.
    auto join_out = MakeTensorInfo(out_sizes, DML_TENSOR_DATA_TYPE_FLOAT32);
    storage->join_out_info = join_out;

    SubNode join_node;
    join_node.input_from.push_back({-1, 0}); // From primary (deq 0)
    join_node.input_tensors.push_back(storage->deq_out_infos[0]);
    join_node.input_buffer_descs.push_back(storage->deq_out_infos[0].ToBufferDesc());
    for (uint32_t t = 1; t < input_count; ++t) {
        join_node.input_from.push_back({static_cast<int>(t - 1), 0}); // From sub_node t-1
        join_node.input_tensors.push_back(storage->deq_out_infos[t]);
        join_node.input_buffer_descs.push_back(storage->deq_out_infos[t].ToBufferDesc());
    }
    join_node.input_tensor_descs.resize(input_count);
    join_node.output_tensors = { join_out };
    join_node.output_buffer_descs = { join_out.ToBufferDesc() };
    join_node.output_tensor_descs.resize(1);
    join_node.desc_storage = storage;
    join_node.op_desc = { DML_OPERATOR_JOIN, &storage->join_desc };

    uint32_t local_axis = dml_axis;
    uint32_t local_input_count = input_count;
    join_node.fixup = [storage, local_axis, local_input_count](SubNode& self) {
        RebuildSubNodePointers(self);
        storage->join_in_tds.clear();
        for (size_t i = 0; i < local_input_count; ++i)
            storage->join_in_tds.push_back(self.input_tensor_descs[i]);
        storage->join_desc.InputCount = local_input_count;
        storage->join_desc.InputTensors = storage->join_in_tds.data();
        storage->join_desc.OutputTensor = &self.output_tensor_descs[0];
        storage->join_desc.Axis = local_axis;
    };
    join_node.FixupPointers();
    result.sub_nodes.push_back(std::move(join_node));

    // QuantizeLinear sub_node: quantize joined output with y_scale, y_zp.
    auto quant_store = std::make_shared<DML_ELEMENT_WISE_QUANTIZE_LINEAR_OPERATOR_DESC>();
    storage->sub_storages.push_back(quant_store);

    auto ys_tensor = MakeTensorInfo(ys_info->sizes, ys_info->data_type);
    auto yzp_tensor = MakeTensorInfo(yzp_info->sizes, yzp_info->data_type);

    SubNode quant_node;
    quant_node.input_tensors = { join_out, ys_tensor, yzp_tensor };
    quant_node.input_buffer_descs = { join_out.ToBufferDesc(), ys_tensor.ToBufferDesc(), yzp_tensor.ToBufferDesc() };
    quant_node.input_tensor_descs.resize(3);
    // Input 0: from Join sub_node (index = input_count - 1 in sub_nodes, but Join is the last sub_node before this one)
    int join_sub_idx = static_cast<int>(result.sub_nodes.size()) - 1;
    quant_node.input_from = { {join_sub_idx, 0}, {-2, 0}, {-2, 0} };
    quant_node.graph_inputs = { {0, 1}, {1, 2} }; // y_scale→slot1, y_zp→slot2

    quant_node.output_tensors = { out_tensor };
    quant_node.output_buffer_descs = { out_tensor.ToBufferDesc() };
    quant_node.output_tensor_descs.resize(1);
    quant_node.desc_storage = quant_store;
    quant_node.op_desc = { DML_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR, quant_store.get() };
    quant_node.fixup = [quant_store](SubNode& self) {
        RebuildSubNodePointers(self);
        quant_store->InputTensor    = &self.input_tensor_descs[0];
        quant_store->ScaleTensor    = &self.input_tensor_descs[1];
        quant_store->ZeroPointTensor= &self.input_tensor_descs[2];
        quant_store->OutputTensor   = &self.output_tensor_descs[0];
    };
    quant_node.FixupPointers();
    result.sub_nodes.push_back(std::move(quant_node));

    return result;
}

// ---------------------------------------------------------------------------
// Registry builder
// ---------------------------------------------------------------------------

OpTranslatorRegistry BuildOpTranslatorRegistry() {
    OpTranslatorRegistry registry;

    // --- P0: already done ---
    registry["Add"] = TranslateBinaryElementwise<DML_ELEMENT_WISE_ADD_OPERATOR_DESC, DML_OPERATOR_ELEMENT_WISE_ADD>;
    registry["Mul"] = TranslateBinaryElementwise<DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC, DML_OPERATOR_ELEMENT_WISE_MULTIPLY>;
    registry["Sub"] = TranslateBinaryElementwise<DML_ELEMENT_WISE_SUBTRACT_OPERATOR_DESC, DML_OPERATOR_ELEMENT_WISE_SUBTRACT>;
    registry["Div"] = TranslateBinaryElementwise<DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC, DML_OPERATOR_ELEMENT_WISE_DIVIDE>;

    registry["Relu"]    = TranslateUnaryActivation<DML_ACTIVATION_RELU_OPERATOR_DESC,    DML_OPERATOR_ACTIVATION_RELU>;
    registry["Sigmoid"] = TranslateUnaryActivation<DML_ACTIVATION_SIGMOID_OPERATOR_DESC, DML_OPERATOR_ACTIVATION_SIGMOID>;
    registry["Tanh"]    = TranslateUnaryActivation<DML_ACTIVATION_TANH_OPERATOR_DESC,    DML_OPERATOR_ACTIVATION_TANH>;

    registry["MatMul"]    = TranslateMatMul;
    registry["Softmax"]   = TranslateSoftmax;
    registry["Reshape"]   = TranslateReshape;
    registry["Flatten"]   = TranslateShapeOnly;
    registry["Squeeze"]   = TranslateShapeOnly;
    registry["Unsqueeze"] = TranslateShapeOnly;
    registry["Transpose"] = TranslateTranspose;

    // --- P0: simple unary (no attributes) ---
    registry["Abs"]        = TranslateUnaryActivation<DML_ELEMENT_WISE_ABS_OPERATOR_DESC,           DML_OPERATOR_ELEMENT_WISE_ABS>;
    registry["Ceil"]       = TranslateUnaryActivation<DML_ELEMENT_WISE_CEIL_OPERATOR_DESC,          DML_OPERATOR_ELEMENT_WISE_CEIL>;
    registry["Cos"]        = TranslateUnaryActivation<DML_ELEMENT_WISE_COS_OPERATOR_DESC,           DML_OPERATOR_ELEMENT_WISE_COS>;
    registry["Cosh"]       = TranslateUnaryActivation<DML_ELEMENT_WISE_COSH_OPERATOR_DESC,          DML_OPERATOR_ELEMENT_WISE_COSH>;
    registry["Erf"]        = TranslateUnaryActivation<DML_ELEMENT_WISE_ERF_OPERATOR_DESC,           DML_OPERATOR_ELEMENT_WISE_ERF>;
    registry["Exp"]        = TranslateUnaryActivation<DML_ELEMENT_WISE_EXP_OPERATOR_DESC,           DML_OPERATOR_ELEMENT_WISE_EXP>;
    registry["Floor"]      = TranslateUnaryActivation<DML_ELEMENT_WISE_FLOOR_OPERATOR_DESC,         DML_OPERATOR_ELEMENT_WISE_FLOOR>;
    registry["Log"]        = TranslateUnaryActivation<DML_ELEMENT_WISE_LOG_OPERATOR_DESC,           DML_OPERATOR_ELEMENT_WISE_LOG>;
    registry["Neg"]        = TranslateUnaryActivation<DML_ELEMENT_WISE_NEGATE_OPERATOR_DESC,        DML_OPERATOR_ELEMENT_WISE_NEGATE>;
    registry["Not"]        = TranslateUnaryActivation<DML_ELEMENT_WISE_LOGICAL_NOT_OPERATOR_DESC,   DML_OPERATOR_ELEMENT_WISE_LOGICAL_NOT>;
    registry["Reciprocal"] = TranslateUnaryActivation<DML_ELEMENT_WISE_RECIP_OPERATOR_DESC,         DML_OPERATOR_ELEMENT_WISE_RECIP>;
    registry["Round"]      = TranslateUnaryActivation<DML_ELEMENT_WISE_ROUND_OPERATOR_DESC,         DML_OPERATOR_ELEMENT_WISE_ROUND>;
    registry["Sign"]       = TranslateUnaryActivation<DML_ELEMENT_WISE_SIGN_OPERATOR_DESC,          DML_OPERATOR_ELEMENT_WISE_SIGN>;
    registry["Sin"]        = TranslateUnaryActivation<DML_ELEMENT_WISE_SIN_OPERATOR_DESC,           DML_OPERATOR_ELEMENT_WISE_SIN>;
    registry["Sinh"]       = TranslateUnaryActivation<DML_ELEMENT_WISE_SINH_OPERATOR_DESC,          DML_OPERATOR_ELEMENT_WISE_SINH>;
    registry["Sqrt"]       = TranslateUnaryActivation<DML_ELEMENT_WISE_SQRT_OPERATOR_DESC,          DML_OPERATOR_ELEMENT_WISE_SQRT>;
    registry["Tan"]        = TranslateUnaryActivation<DML_ELEMENT_WISE_TAN_OPERATOR_DESC,           DML_OPERATOR_ELEMENT_WISE_TAN>;
    registry["Acos"]       = TranslateUnaryActivation<DML_ELEMENT_WISE_ACOS_OPERATOR_DESC,          DML_OPERATOR_ELEMENT_WISE_ACOS>;
    registry["Acosh"]      = TranslateUnaryActivation<DML_ELEMENT_WISE_ACOSH_OPERATOR_DESC,         DML_OPERATOR_ELEMENT_WISE_ACOSH>;
    registry["Asin"]       = TranslateUnaryActivation<DML_ELEMENT_WISE_ASIN_OPERATOR_DESC,          DML_OPERATOR_ELEMENT_WISE_ASIN>;
    registry["Asinh"]      = TranslateUnaryActivation<DML_ELEMENT_WISE_ASINH_OPERATOR_DESC,         DML_OPERATOR_ELEMENT_WISE_ASINH>;
    registry["Atan"]       = TranslateUnaryActivation<DML_ELEMENT_WISE_ATAN_OPERATOR_DESC,          DML_OPERATOR_ELEMENT_WISE_ATAN>;
    registry["Atanh"]      = TranslateUnaryActivation<DML_ELEMENT_WISE_ATANH_OPERATOR_DESC,         DML_OPERATOR_ELEMENT_WISE_ATANH>;
    registry["BitwiseNot"] = TranslateUnaryActivation<DML_ELEMENT_WISE_BIT_NOT_OPERATOR_DESC,       DML_OPERATOR_ELEMENT_WISE_BIT_NOT>;
    registry["IsNaN"]      = TranslateUnaryActivation<DML_ELEMENT_WISE_IS_NAN_OPERATOR_DESC,        DML_OPERATOR_ELEMENT_WISE_IS_NAN>;
    registry["Softsign"]   = TranslateUnaryActivation<DML_ACTIVATION_SOFTSIGN_OPERATOR_DESC,        DML_OPERATOR_ACTIVATION_SOFTSIGN>;

    // Dropout = identity at inference
    registry["Dropout"] = TranslateUnaryActivation<DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC, DML_OPERATOR_ELEMENT_WISE_IDENTITY>;

    // --- P0: unary with attributes ---
    registry["LeakyRelu"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>& init) -> std::optional<TranslatedOp> {
        return TranslateUnaryWithAttrs<DML_ACTIVATION_LEAKY_RELU_OPERATOR_DESC, DML_OPERATOR_ACTIVATION_LEAKY_RELU>(
            api, node, vs, init, [](DML_ACTIVATION_LEAKY_RELU_OPERATOR_DESC* d, const OrtNode* n, const OrtApi& a) {
                d->Alpha = OrtNodeAdapter(n, a).GetAttributeFloat("alpha", 0.01f);
            });
    };
    registry["Elu"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>& init) -> std::optional<TranslatedOp> {
        return TranslateUnaryWithAttrs<DML_ACTIVATION_ELU_OPERATOR_DESC, DML_OPERATOR_ACTIVATION_ELU>(
            api, node, vs, init, [](DML_ACTIVATION_ELU_OPERATOR_DESC* d, const OrtNode* n, const OrtApi& a) {
                d->Alpha = OrtNodeAdapter(n, a).GetAttributeFloat("alpha", 1.0f);
            });
    };
    registry["Selu"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>& init) -> std::optional<TranslatedOp> {
        return TranslateUnaryWithAttrs<DML_ACTIVATION_SCALED_ELU_OPERATOR_DESC, DML_OPERATOR_ACTIVATION_SCALED_ELU>(
            api, node, vs, init, [](DML_ACTIVATION_SCALED_ELU_OPERATOR_DESC* d, const OrtNode* n, const OrtApi& a) {
                OrtNodeAdapter ad(n, a);
                d->Alpha = ad.GetAttributeFloat("alpha", 1.6732632f);
                d->Gamma = ad.GetAttributeFloat("gamma", 1.0507010f);
            });
    };
    registry["Celu"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>& init) -> std::optional<TranslatedOp> {
        return TranslateUnaryWithAttrs<DML_ACTIVATION_CELU_OPERATOR_DESC, DML_OPERATOR_ACTIVATION_CELU>(
            api, node, vs, init, [](DML_ACTIVATION_CELU_OPERATOR_DESC* d, const OrtNode* n, const OrtApi& a) {
                d->Alpha = OrtNodeAdapter(n, a).GetAttributeFloat("alpha", 1.0f);
            });
    };
    registry["HardSigmoid"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>& init) -> std::optional<TranslatedOp> {
        return TranslateUnaryWithAttrs<DML_ACTIVATION_HARD_SIGMOID_OPERATOR_DESC, DML_OPERATOR_ACTIVATION_HARD_SIGMOID>(
            api, node, vs, init, [](DML_ACTIVATION_HARD_SIGMOID_OPERATOR_DESC* d, const OrtNode* n, const OrtApi& a) {
                OrtNodeAdapter ad(n, a);
                d->Alpha = ad.GetAttributeFloat("alpha", 0.2f);
                d->Beta  = ad.GetAttributeFloat("beta",  0.5f);
            });
    };
    registry["Softplus"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>& init) -> std::optional<TranslatedOp> {
        return TranslateUnaryWithAttrs<DML_ACTIVATION_SOFTPLUS_OPERATOR_DESC, DML_OPERATOR_ACTIVATION_SOFTPLUS>(
            api, node, vs, init, [](DML_ACTIVATION_SOFTPLUS_OPERATOR_DESC* d, const OrtNode* n, const OrtApi& a) {
                d->Steepness = OrtNodeAdapter(n, a).GetAttributeFloat("steepness", 1.0f);
            });
    };
    registry["ThresholdedRelu"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>& init) -> std::optional<TranslatedOp> {
        return TranslateUnaryWithAttrs<DML_ACTIVATION_THRESHOLDED_RELU_OPERATOR_DESC, DML_OPERATOR_ACTIVATION_THRESHOLDED_RELU>(
            api, node, vs, init, [](DML_ACTIVATION_THRESHOLDED_RELU_OPERATOR_DESC* d, const OrtNode* n, const OrtApi& a) {
                d->Alpha = OrtNodeAdapter(n, a).GetAttributeFloat("alpha", 1.0f);
            });
    };
    registry["Shrink"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>& init) -> std::optional<TranslatedOp> {
        return TranslateUnaryWithAttrs<DML_ACTIVATION_SHRINK_OPERATOR_DESC, DML_OPERATOR_ACTIVATION_SHRINK>(
            api, node, vs, init, [](DML_ACTIVATION_SHRINK_OPERATOR_DESC* d, const OrtNode* n, const OrtApi& a) {
                OrtNodeAdapter ad(n, a);
                d->Bias  = ad.GetAttributeFloat("bias",  0.0f);
                d->Threshold = ad.GetAttributeFloat("lambd", 0.5f);
            });
    };
    registry["ParametricSoftplus"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>& init) -> std::optional<TranslatedOp> {
        return TranslateUnaryWithAttrs<DML_ACTIVATION_PARAMETRIC_SOFTPLUS_OPERATOR_DESC, DML_OPERATOR_ACTIVATION_PARAMETRIC_SOFTPLUS>(
            api, node, vs, init, [](DML_ACTIVATION_PARAMETRIC_SOFTPLUS_OPERATOR_DESC* d, const OrtNode* n, const OrtApi& a) {
                OrtNodeAdapter ad(n, a);
                d->Alpha = ad.GetAttributeFloat("alpha", 1.0f);
                d->Beta  = ad.GetAttributeFloat("beta",  1.0f);
            });
    };
    registry["ScaledTanh"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>& init) -> std::optional<TranslatedOp> {
        return TranslateUnaryWithAttrs<DML_ACTIVATION_SCALED_TANH_OPERATOR_DESC, DML_OPERATOR_ACTIVATION_SCALED_TANH>(
            api, node, vs, init, [](DML_ACTIVATION_SCALED_TANH_OPERATOR_DESC* d, const OrtNode* n, const OrtApi& a) {
                OrtNodeAdapter ad(n, a);
                d->Alpha = ad.GetAttributeFloat("alpha", 1.0f);
                d->Beta  = ad.GetAttributeFloat("beta",  1.0f);
            });
    };

    registry["Cast"]        = TranslateCast;
    registry["IsInf"]       = TranslateIsInf;
    registry["Affine"]      = TranslateAffine;
    registry["ImageScaler"] = TranslateAffine;  // same pattern: scale + per-channel bias (approx)

    // --- P1: binary elementwise ---
    registry["And"]            = TranslateBinaryElementwise<DML_ELEMENT_WISE_LOGICAL_AND_OPERATOR_DESC,                    DML_OPERATOR_ELEMENT_WISE_LOGICAL_AND>;
    registry["Or"]             = TranslateBinaryElementwise<DML_ELEMENT_WISE_LOGICAL_OR_OPERATOR_DESC,                     DML_OPERATOR_ELEMENT_WISE_LOGICAL_OR>;
    registry["Xor"]            = TranslateBinaryElementwise<DML_ELEMENT_WISE_LOGICAL_XOR_OPERATOR_DESC,                    DML_OPERATOR_ELEMENT_WISE_LOGICAL_XOR>;
    registry["BitwiseAnd"]     = TranslateBinaryElementwise<DML_ELEMENT_WISE_BIT_AND_OPERATOR_DESC,                        DML_OPERATOR_ELEMENT_WISE_BIT_AND>;
    registry["BitwiseOr"]      = TranslateBinaryElementwise<DML_ELEMENT_WISE_BIT_OR_OPERATOR_DESC,                         DML_OPERATOR_ELEMENT_WISE_BIT_OR>;
    registry["BitwiseXor"]     = TranslateBinaryElementwise<DML_ELEMENT_WISE_BIT_XOR_OPERATOR_DESC,                        DML_OPERATOR_ELEMENT_WISE_BIT_XOR>;
    registry["Equal"]          = TranslateBinaryElementwise<DML_ELEMENT_WISE_LOGICAL_EQUALS_OPERATOR_DESC,                 DML_OPERATOR_ELEMENT_WISE_LOGICAL_EQUALS>;
    registry["Greater"]        = TranslateBinaryElementwise<DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OPERATOR_DESC,           DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN>;
    registry["GreaterOrEqual"] = TranslateBinaryElementwise<DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL_OPERATOR_DESC,  DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL>;
    registry["Less"]           = TranslateBinaryElementwise<DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OPERATOR_DESC,              DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN>;
    registry["LessOrEqual"]    = TranslateBinaryElementwise<DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL_OPERATOR_DESC,     DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL>;
    registry["Max"]            = TranslateBinaryElementwise<DML_ELEMENT_WISE_MAX_OPERATOR_DESC,                            DML_OPERATOR_ELEMENT_WISE_MAX>;
    registry["Min"]            = TranslateBinaryElementwise<DML_ELEMENT_WISE_MIN_OPERATOR_DESC,                            DML_OPERATOR_ELEMENT_WISE_MIN>;
    // Pow has InputTensor/ExponentTensor, not ATensor/BTensor.
    registry["Pow"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>&) -> std::optional<TranslatedOp> {
        auto inputs  = GetInputNames(api, node);
        auto outputs = GetOutputNames(api, node);
        if (inputs.size() < 2 || outputs.empty()) return std::nullopt;
        auto* a_info = LookupShape(vs, inputs[0]);
        auto* b_info = LookupShape(vs, inputs[1]);
        if (!a_info || !b_info) return std::nullopt;
        auto bc = BroadcastShapes(*a_info, *b_info);
        if (!bc) return std::nullopt;
        auto* out_edge = LookupShape(vs, outputs[0]);
        auto out_info = out_edge ? MakeTensorInfo(out_edge->sizes, a_info->data_type)
                                 : MakeTensorInfo(bc->output_sizes, a_info->data_type);
        struct PowStorage { DML_ELEMENT_WISE_POW_OPERATOR_DESC desc{}; };
        auto storage = std::make_shared<PowStorage>();
        storage->desc.ScaleBias = nullptr;
        TranslatedOp result;
        result.input_tensors  = { bc->a, bc->b };
        result.output_tensors = { out_info };
        result.input_buffer_descs  = { bc->a.ToBufferDesc(), bc->b.ToBufferDesc() };
        result.input_tensor_descs.resize(2);
        result.output_buffer_descs = { out_info.ToBufferDesc() };
        result.output_tensor_descs.resize(1);
        result.desc_storage = storage;
        result.op_desc = { DML_OPERATOR_ELEMENT_WISE_POW, &storage->desc };
        result.fixup = [storage](TranslatedOp& self) {
            RebuildTensorDescPointers(self);
            storage->desc.InputTensor    = &self.input_tensor_descs[0];
            storage->desc.ExponentTensor = &self.input_tensor_descs[1];
            storage->desc.OutputTensor   = &self.output_tensor_descs[0];
        };
        result.FixupPointers();
        return result;
    };
    // PRelu has SlopeTensor not BTensor — use a custom lambda.
    registry["PRelu"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>&) -> std::optional<TranslatedOp> {
        auto inputs  = GetInputNames(api, node);
        auto outputs = GetOutputNames(api, node);
        if (inputs.size() < 2 || outputs.empty()) return std::nullopt;
        auto* x_info     = LookupShape(vs, inputs[0]);
        auto* slope_info = LookupShape(vs, inputs[1]);
        if (!x_info || !slope_info) return std::nullopt;
        auto bc = BroadcastShapes(*x_info, *slope_info);
        if (!bc) return std::nullopt;
        auto* out_edge = LookupShape(vs, outputs[0]);
        auto out_tensor = out_edge ? MakeTensorInfo(out_edge->sizes, x_info->data_type)
                                   : MakeTensorInfo(bc->output_sizes, x_info->data_type);
        struct PReluStorage { DML_ACTIVATION_PARAMETERIZED_RELU_OPERATOR_DESC desc{}; };
        auto storage = std::make_shared<PReluStorage>();
        TranslatedOp result;
        result.input_tensors  = { bc->a, bc->b };
        result.output_tensors = { out_tensor };
        result.input_buffer_descs  = { bc->a.ToBufferDesc(), bc->b.ToBufferDesc() };
        result.input_tensor_descs.resize(2);
        result.output_buffer_descs = { out_tensor.ToBufferDesc() };
        result.output_tensor_descs.resize(1);
        result.desc_storage = storage;
        result.op_desc = { DML_OPERATOR_ACTIVATION_PARAMETERIZED_RELU, &storage->desc };
        result.fixup = [storage](TranslatedOp& self) {
            RebuildTensorDescPointers(self);
            storage->desc.InputTensor  = &self.input_tensor_descs[0];
            storage->desc.SlopeTensor  = &self.input_tensor_descs[1];
            storage->desc.OutputTensor = &self.output_tensor_descs[0];
        };
        result.FixupPointers();
        return result;
    };
    registry["Sum"]            = TranslateSum;
    registry["Mean"]           = TranslateBinaryElementwise<DML_ELEMENT_WISE_MEAN_OPERATOR_DESC,                           DML_OPERATOR_ELEMENT_WISE_MEAN>;
    registry["Where"]          = TranslateWhere;
    registry["Mod"]            = TranslateMod;
    registry["BitShift"]       = TranslateBitShift;

    // --- P2: Reduce ---
    registry["ReduceMean"]       = TranslateReduce<DML_REDUCE_FUNCTION_AVERAGE>;
    registry["ReduceSum"]        = TranslateReduce<DML_REDUCE_FUNCTION_SUM>;
    registry["ReduceMax"]        = TranslateReduce<DML_REDUCE_FUNCTION_MAX>;
    registry["ReduceMin"]        = TranslateReduce<DML_REDUCE_FUNCTION_MIN>;
    registry["ReduceProd"]       = TranslateReduce<DML_REDUCE_FUNCTION_MULTIPLY>;
    registry["ReduceL1"]         = TranslateReduce<DML_REDUCE_FUNCTION_L1>;
    registry["ReduceL2"]         = TranslateReduce<DML_REDUCE_FUNCTION_L2>;
    registry["ReduceLogSum"]     = TranslateReduce<DML_REDUCE_FUNCTION_LOG_SUM>;
    registry["ReduceLogSumExp"]  = TranslateReduce<DML_REDUCE_FUNCTION_LOG_SUM_EXP>;
    registry["ReduceSumSquare"]  = TranslateReduce<DML_REDUCE_FUNCTION_SUM_SQUARE>;
    registry["ArgMax"]    = TranslateArgMaxMin<DML_OPERATOR_ARGMAX, DML_ARGMAX_OPERATOR_DESC>;
    registry["ArgMin"]    = TranslateArgMaxMin<DML_OPERATOR_ARGMIN, DML_ARGMIN_OPERATOR_DESC>;
    registry["Hardmax"]   = TranslateHardmax;
    registry["LogSoftmax"]= TranslateLogSoftmax;

    // --- P3: Conv / Pooling ---
    registry["Conv"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>& init) -> std::optional<TranslatedOp> {
        return TranslateConvImpl(api, node, vs, init, DML_CONVOLUTION_DIRECTION_FORWARD);
    };
    registry["NhwcConv"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>& init) -> std::optional<TranslatedOp> {
        return TranslateConvImpl(api, node, vs, init, DML_CONVOLUTION_DIRECTION_FORWARD, true);
    };
    registry["ConvTranspose"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>& init) -> std::optional<TranslatedOp> {
        return TranslateConvImpl(api, node, vs, init, DML_CONVOLUTION_DIRECTION_BACKWARD);
    };
    registry["ConvTransposeWithDynamicPads"] = registry["ConvTranspose"];
    registry["DmlFusedConv"]          = registry["Conv"];
    registry["DmlFusedConvTranspose"] = registry["ConvTranspose"];

    registry["AveragePool"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>& init) -> std::optional<TranslatedOp> {
        return TranslateAveragePool(api, node, vs, init, false);
    };
    registry["GlobalAveragePool"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>& init) -> std::optional<TranslatedOp> {
        return TranslateAveragePool(api, node, vs, init, true);
    };
    registry["MaxPool"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>& init) -> std::optional<TranslatedOp> {
        return TranslateMaxPool(api, node, vs, init, false);
    };
    registry["GlobalMaxPool"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>& init) -> std::optional<TranslatedOp> {
        return TranslateMaxPool(api, node, vs, init, true);
    };

    // --- P4: Data movement ---
    registry["Slice"]          = TranslateSlice;
    registry["Concat"]         = TranslateConcat;
    registry["Split"]          = TranslateSplit;
    registry["Gather"]         = TranslateGather;
    registry["GatherElements"] = TranslateGatherElements;
    registry["GatherND"]       = TranslateGatherND;
    registry["Pad"]            = TranslatePad;
    registry["DepthToSpace"]   = TranslateDepthToSpace;
    registry["SpaceToDepth"]   = TranslateSpaceToDepth;
    registry["ScatterElements"]= TranslateScatterElements;
    registry["Scatter"]        = TranslateScatterElements;  // older opset name
    registry["ScatterND"]      = TranslateScatterND;
    registry["Tile"]           = TranslateTile;
    registry["ConstantOfShape"]= TranslateConstantOfShape;
    registry["Range"]          = TranslateRange;

    // Expand = broadcast identity (same as Reshape with broadcast strides)
    registry["Expand"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>& init) -> std::optional<TranslatedOp> {
        // Expand input[0] to the shape given by input[1].
        // Implemented as identity with broadcast strides (stride=0 on broadcasted dims).
        auto inputs  = GetInputNames(api, node);
        auto outputs = GetOutputNames(api, node);
        if (inputs.empty() || outputs.empty()) return std::nullopt;
        auto* in_info  = LookupShape(vs, inputs[0]);
        if (!in_info) return std::nullopt;
        // Output shape = broadcast(input, shape_input).
        // shape_input (input[1]) is a constant; its values ARE the target shape.
        // Since shapes are pre-propagated by the graph, try output edge first.
        auto* out_edge = LookupShape(vs, outputs[0]);
        // If output shape isn't available, use the shape initializer values.
        std::vector<uint32_t> computed_out;
        if (!out_edge && inputs.size() > 1 && !inputs[1].empty()) {
            auto it = init.find(inputs[1]);
            if (it != init.end() && it->second) {
                void* data = nullptr;
                api.GetTensorMutableData(const_cast<OrtValue*>(it->second), &data);
                if (data) {
                    OrtTensorTypeAndShapeInfo* tsi = nullptr;
                    api.GetTensorTypeAndShape(const_cast<OrtValue*>(it->second), &tsi);
                    size_t cnt = 0;
                    api.GetTensorShapeElementCount(tsi, &cnt);
                    api.ReleaseTensorTypeAndShapeInfo(tsi);
                    auto* vals = static_cast<const int64_t*>(data);
                    for (size_t d = 0; d < cnt; ++d)
                        computed_out.push_back(static_cast<uint32_t>(std::max<int64_t>(vals[d], 1)));
                }
            }
        }
        if (!out_edge && computed_out.empty()) return std::nullopt;
        auto& in_sz  = in_info->sizes;
        const auto& out_sz = out_edge ? out_edge->sizes : computed_out;
        size_t out_rank = out_sz.size();
        std::vector<uint32_t> strides(out_rank, 1u);
        auto in_strides = ComputePackedStrides(in_sz);
        size_t in_rank  = in_sz.size();
        for (size_t i = 0; i < out_rank; ++i) {
            size_t in_i = i - (out_rank - in_rank);
            if (i < out_rank - in_rank || in_sz[in_i] == 1)
                strides[i] = 0u;
            else
                strides[i] = in_strides[in_i];
        }
        uint64_t in_bytes = ComputeAlignedTotalBytes(in_sz, in_info->data_type);
        auto in_tensor  = MakeTensorInfoWithStrides(out_sz, strides, in_info->data_type, in_bytes);
        in_tensor.sizes   = PadToMinDims(out_sz);
        in_tensor.strides = PadToMinDims(strides);
        auto out_tensor = MakeTensorInfo(out_sz, in_info->data_type);

        auto storage = std::make_shared<DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC>();
        storage->ScaleBias = nullptr;
        TranslatedOp result;
        result.input_tensors  = { in_tensor };
        result.output_tensors = { out_tensor };
        result.input_buffer_descs  = { in_tensor.ToBufferDesc() };
        result.input_tensor_descs.resize(1);
        result.output_buffer_descs = { out_tensor.ToBufferDesc() };
        result.output_tensor_descs.resize(1);
        result.desc_storage = storage;
        result.op_desc = { DML_OPERATOR_ELEMENT_WISE_IDENTITY, storage.get() };
        result.fixup = [storage](TranslatedOp& self) {
            RebuildTensorDescPointers(self);
            storage->InputTensor  = &self.input_tensor_descs[0];
            storage->OutputTensor = &self.output_tensor_descs[0];
        };
        result.FixupPointers();
        return result;
    };

    // --- P5: Normalization ---
    registry["LayerNormalization"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>& init) -> std::optional<TranslatedOp> {
        return TranslateLayerNorm(api, node, vs, init, false);
    };
    registry["SimplifiedLayerNormalization"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>& init) -> std::optional<TranslatedOp> {
        return TranslateLayerNorm(api, node, vs, init, true);
    };
    registry["InstanceNormalization"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>& init) -> std::optional<TranslatedOp> {
        return TranslateLayerNorm(api, node, vs, init, false, 2);
    };
    registry["MeanVarianceNormalization"]    = registry["LayerNormalization"];
    registry["GroupNorm"]                    = TranslateGroupNorm;
    registry["BatchNormalization"]           = TranslateBatchNorm;
    registry["LRN"]                          = TranslateLRN;

    registry["DmlFusedInstanceNormalization"]     = registry["InstanceNormalization"];
    registry["DmlFusedBatchNormalization"]        = TranslateBatchNorm;
    registry["DmlFusedMeanVarianceNormalization"] = registry["LayerNormalization"];

    // --- P6: Gemm / Clip / Quant / Gelu ---
    registry["Gemm"]              = TranslateGemm;
    registry["FusedMatMul"]       = TranslateMatMul;
    registry["FusedMatMulActivation"] = TranslateMatMul;
    registry["DmlFusedMatMul"]    = TranslateMatMul;
    registry["DmlFusedGemm"]      = TranslateGemm;
    registry["DmlFusedAdd"]       = TranslateBinaryElementwise<DML_ELEMENT_WISE_ADD_OPERATOR_DESC, DML_OPERATOR_ELEMENT_WISE_ADD>;
    registry["DmlFusedSum"]       = TranslateBinaryElementwise<DML_ELEMENT_WISE_ADD_OPERATOR_DESC, DML_OPERATOR_ELEMENT_WISE_ADD>;
    registry["BiasAdd"]           = TranslateBinaryElementwise<DML_ELEMENT_WISE_ADD_OPERATOR_DESC, DML_OPERATOR_ELEMENT_WISE_ADD>;

    registry["Resize"]            = TranslateResize;
    registry["Upsample"]          = TranslateResize;
    registry["Clip"]              = TranslateClip;
    registry["QuantizeLinear"]    = TranslateQuantizeLinear;
    registry["DequantizeLinear"]  = TranslateDequantizeLinear;

    registry["Gelu"]              = TranslateGelu;
    registry["BiasGelu"]          = TranslateGelu;  // bias is fused upstream; treat as Gelu

    // Sigmoid-variant activations reuse existing sigmoid template
    registry["QLinearSigmoid"]    = TranslateUnaryActivation<DML_ACTIVATION_SIGMOID_OPERATOR_DESC, DML_OPERATOR_ACTIVATION_SIGMOID>;

    // --- Batch 3: Activation variants ---
    registry["QuickGelu"]         = TranslateQuickGelu;
    registry["FastGelu"]          = TranslateGelu;
    registry["BiasSplitGelu"]     = TranslateBiasSplitGelu;

    // --- Batch 4: Transformer building blocks ---
    registry["SkipLayerNormalization"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>& init) -> std::optional<TranslatedOp> {
        return TranslateSkipLayerNorm(api, node, vs, init, false);
    };
    registry["SkipSimplifiedLayerNormalization"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>& init) -> std::optional<TranslatedOp> {
        return TranslateSkipLayerNorm(api, node, vs, init, true);
    };

    // --- Batch 5: Quantization ---
    registry["DynamicQuantizeLinear"] = TranslateDynamicQuantizeLinear;
    registry["QLinearMatMul"]         = TranslateQLinearMatMul;
    registry["MatMulInteger"]         = TranslateMatMulInteger;
    registry["MatMulIntegerToFloat"]  = TranslateMatMulIntegerToFloat;
    registry["QLinearAdd"]            = TranslateQLinearAdd;

    // --- Batch 6: Quick wins ---
    registry["OneHot"]                = TranslateOneHot;
    registry["Trilu"]                 = TranslateTrilu;
    registry["LpNormalization"]       = TranslateLpNormalization;
    registry["EyeLike"]               = TranslateEyeLike;
    registry["ReverseSequence"]       = TranslateReverseSequence;
    registry["Crop"]                  = TranslateCrop;
    registry["MaxRoiPool"]            = TranslateMaxRoiPool;
    registry["MaxUnpool"]             = TranslateMaxUnpool;

    // --- LLM ops ---
    registry["MultiHeadAttention"]    = TranslateMultiHeadAttention;
    registry["RotaryEmbedding"]       = TranslateRotaryEmbedding;
    registry["MatMulNBits"]           = TranslateMatMulNBits;
    registry["GroupQueryAttention"]    = TranslateGroupQueryAttention;

    // --- BERT ops ---
    registry["Attention"]              = TranslateAttention;
    registry["QAttention"]             = TranslateQAttention;
    registry["EmbedLayerNormalization"] = TranslateEmbedLayerNormalization;

    // --- Remaining pooling ---
    registry["LpPool"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>& init) -> std::optional<TranslatedOp> {
        return TranslateLpPool(api, node, vs, init, false);
    };
    registry["GlobalLpPool"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>& init) -> std::optional<TranslatedOp> {
        return TranslateLpPool(api, node, vs, init, true);
    };

    // --- Remaining conv variants ---
    registry["QLinearConv"]   = TranslateQLinearConv;
    registry["ConvInteger"]   = TranslateConvInteger;

    // --- Quantized pooling ---
    registry["QLinearAveragePool"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>& init) -> std::optional<TranslatedOp> {
        return TranslateQLinearAveragePool(api, node, vs, init, false);
    };
    registry["QLinearGlobalAveragePool"] = [](const OrtApi& api, const OrtNode* node,
        const std::unordered_map<std::string, DmlTensorInfo>& vs,
        const std::unordered_map<std::string, const OrtValue*>& init) -> std::optional<TranslatedOp> {
        return TranslateQLinearAveragePool(api, node, vs, init, true);
    };

    // --- Col2Im ---
    registry["Col2Im"] = TranslateCol2Im;

    // --- Quantized matmul ---
    registry["DynamicQuantizeMatMul"] = TranslateDynamicQuantizeMatMul;

    // --- Quantized concat ---
    registry["QLinearConcat"] = TranslateQLinearConcat;

    return registry;
}

}  // namespace dml_ep
