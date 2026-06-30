// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "dml_op_translators.h"
#include "fusion_utils.h"
#include "ort_node_adapter.h"

#include <algorithm>
#include <cfloat>
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

    // Compute strides for broadcast.
    // DML requires stride=0 for any dimension where the tensor does not vary —
    // either because it's expanding (size 1 → size N) OR because it's a
    // leading padding dim added by PadToMinDims (original_rank tells us how
    // many trailing dims are real; leading dims beyond original_rank are pads).
    auto a_strides = ComputePackedStrides(a_sizes);
    auto b_strides = ComputePackedStrides(b_sizes);

    // Number of leading padding dims added to reach max_rank.
    size_t a_pad = (a_in.original_rank > 0 && max_rank > a_in.original_rank)
                   ? max_rank - a_in.original_rank : 0;
    size_t b_pad = (b_in.original_rank > 0 && max_rank > b_in.original_rank)
                   ? max_rank - b_in.original_rank : 0;

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
    auto out_info = out_edge ? MakeTensorInfo(out_edge->sizes, a_info->data_type)
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
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    int64_t keepdims = adapter.GetAttributeInt("keepdims", 1);
    auto axes_i64 = adapter.GetAttributeInts("axes");

    size_t padded_rank = in_info->sizes.size();
    size_t orig_rank = in_info->original_rank ? in_info->original_rank : padded_rank;
    size_t pad_offset = padded_rank - orig_rank;
    std::vector<uint32_t> axes;
    if (axes_i64.empty()) {
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
        out_sizes[1] = w_sizes[0];
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
    const std::unordered_map<std::string, const OrtValue*>&) {
    auto inputs  = GetInputNames(ort_api, node);
    auto outputs = GetOutputNames(ort_api, node);
    if (inputs.empty() || outputs.empty()) return std::nullopt;

    auto* in_info = LookupShape(value_shapes, inputs[0]);
    if (!in_info) return std::nullopt;

    OrtNodeAdapter adapter(node, ort_api);
    std::string mode_str = adapter.GetAttributeString("mode", "nearest");
    std::string coord_transform = adapter.GetAttributeString("coordinate_transformation_mode", "half_pixel");
    std::string nearest_mode = adapter.GetAttributeString("nearest_mode", "round_prefer_floor");

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
        // Scales input (opset 11+, index 2) — read float values from initializer
        if (!have_output && inputs.size() > 2 && !inputs[2].empty()) {
            auto* scales_info = LookupShape(value_shapes, inputs[2]);
            (void)scales_info; // scales are float values, not a shape — need initializer
            // Compute output from input * scale (round to nearest int)
            for (size_t i = 0; i < rank; ++i)
                out_sizes[i] = static_cast<uint32_t>(std::round(in_info->sizes[i] * scales[i]));
            have_output = true;
        }
        if (!have_output)
            out_sizes = in_info->sizes;
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
    registry["Clip"]              = TranslateClip;
    registry["QuantizeLinear"]    = TranslateQuantizeLinear;
    registry["DequantizeLinear"]  = TranslateDequantizeLinear;

    registry["Gelu"]              = TranslateGelu;
    registry["BiasGelu"]          = TranslateGelu;  // bias is fused upstream; treat as Gelu

    // Sigmoid-variant activations reuse existing sigmoid template
    registry["QLinearSigmoid"]    = TranslateUnaryActivation<DML_ACTIVATION_SIGMOID_OPERATOR_DESC, DML_OPERATOR_ACTIVATION_SIGMOID>;

    return registry;
}

}  // namespace dml_ep
