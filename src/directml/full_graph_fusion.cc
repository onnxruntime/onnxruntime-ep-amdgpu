// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "full_graph_fusion.h"
#include "fusion_utils.h"
#include "ort_node_adapter.h"
#include "dml_execution_provider.h"
#include "DmlExecutionProvider/IExecutionProvider.h"
#include "dml_abi_kernel.h"
#include "core/framework/tensor_type_and_shape.h"  // OrtTensorTypeAndShapeInfo::HasShape()

#include <DirectML.h>
#include <wrl/client.h>
#include <gsl/gsl>
#include <algorithm>
#include <numeric>
#include <unordered_set>

namespace dml_ep {

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// AllGraphInputsStatic
// ---------------------------------------------------------------------------

bool AllGraphInputsStatic(const OrtApi& ort_api, const OrtGraph* graph) {
    size_t num_inputs = 0;
    ort_api.Graph_GetNumInputs(graph, &num_inputs);
    if (num_inputs == 0) return false;

    std::vector<const OrtValueInfo*> input_vis(num_inputs, nullptr);
    ort_api.Graph_GetInputs(graph, input_vis.data(), num_inputs);

    for (size_t i = 0; i < num_inputs; ++i) {
        const OrtValueInfo* vi = input_vis[i];
        if (!vi) return false;
        if (vi->GetTypeInfo() == nullptr || vi->GetTypeInfo()->tensor_type_info == nullptr
            || vi->GetTypeInfo()->tensor_type_info.get() == nullptr) return false;
        if (!vi->GetTypeInfo()->tensor_type_info->HasShape()) return false;

        size_t rank = 0;
        ort_api.GetDimensionsCount(vi->GetTypeInfo()->tensor_type_info.get(), &rank);

        // Scalars (rank 0) are inherently static — 1 element, no dims to check.
        if (rank == 0) continue;

        std::vector<int64_t> dims(rank, -1);
        ort_api.GetDimensions(vi->GetTypeInfo()->tensor_type_info.get(), dims.data(), rank);
        for (int64_t d : dims)
            if (d <= 0) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// AllNodesHaveTranslators
// ---------------------------------------------------------------------------

bool AllNodesHaveTranslators(
    const OrtApi& ort_api,
    const OpTranslatorRegistry& registry,
    const std::vector<const OrtNode*>& nodes) {
    for (const OrtNode* node : nodes) {
        const char* op_type = nullptr;
        ort_api.Node_GetOperatorType(node, &op_type);
        if (!op_type || registry.find(op_type) == registry.end())
            return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Compiled node: holds the translated op and its connectivity info.
// ---------------------------------------------------------------------------

struct CompiledNode {
    std::unique_ptr<TranslatedOp> translated;
    std::vector<std::string>      input_names;
    std::vector<std::string>      output_names;
    std::string                   op_type;
};

// ---------------------------------------------------------------------------
// FullGraph_Compute — single GPU dispatch for the entire graph
// ---------------------------------------------------------------------------

static OrtStatus* ORT_API_CALL FullGraph_Compute(
    OrtNodeComputeInfo* this_ptr,
    void* compute_state,
    OrtKernelContext* kernel_context) noexcept
{
    auto* state = static_cast<FullGraphKernelState*>(compute_state);
    if (!state || !state->provider || !state->ort_api) return nullptr;
    const OrtApi& api = *state->ort_api;

    state->compute_call_count++;

    size_t total_inputs = state->num_runtime_inputs + state->num_initializers;
    std::vector<DML_BUFFER_BINDING> input_buffer_bindings(total_inputs);
    std::vector<DML_BINDING_DESC> input_descs(total_inputs);

    auto get_resource = [&](const OrtValue* value) -> ID3D12Resource* {
        void* raw = nullptr;
        OrtStatus* st = api.GetTensorMutableData(const_cast<OrtValue*>(value), &raw);
        if (st || !raw) { if (st) api.ReleaseStatus(st); return nullptr; }
        return state->provider->DecodeResource(raw);
    };

    // Bind runtime inputs. Iterate subgraph inputs (KernelContext indices) and
    // map to DML graph input indices. Inlined constants have no DML index.
    // OWNED inputs are left as NONE (data already in persistent resource).
    for (size_t sg_i = 0; sg_i < state->num_subgraph_inputs; ++sg_i) {
        if (sg_i >= state->subgraph_to_dml_input.size()) break;
        size_t dml_i = state->subgraph_to_dml_input[sg_i];
        if (dml_i == SIZE_MAX) continue;                        // inlined constant
        if (state->runtime_input_is_owned[dml_i]) continue;    // baked into persistent resource
        if (!state->dml_inputs_with_edges.count(dml_i)) continue; // no edge — leave as NONE

        const OrtValue* input_value = nullptr;
        OrtStatus* st = api.KernelContext_GetInput(kernel_context, sg_i, &input_value);
        if (st || !input_value) {
            if (st) api.ReleaseStatus(st);
            return api.CreateStatus(ORT_FAIL, "Tier0: failed to get runtime input");
        }
        ID3D12Resource* resource = get_resource(input_value);
        if (!resource)
            return api.CreateStatus(ORT_FAIL, "Tier0: null D3D12 resource for input");

        input_buffer_bindings[dml_i] = { resource, 0, state->runtime_input_bytes[dml_i] };
        input_descs[dml_i] = { DML_BINDING_TYPE_BUFFER, &input_buffer_bindings[dml_i] };
    }

    // Allocate and bind outputs.
    std::vector<DML_BUFFER_BINDING> output_buffer_bindings(state->num_outputs);
    std::vector<DML_BINDING_DESC> output_descs(state->num_outputs);

    for (size_t i = 0; i < state->num_outputs; ++i) {
        OrtValue* output_value = nullptr;
        OrtStatus* st = api.KernelContext_GetOutput(
            kernel_context, i,
            state->output_dims[i].data(),
            state->output_dims[i].size(),
            &output_value);
        if (st || !output_value) {
            if (st) api.ReleaseStatus(st);
            return api.CreateStatus(ORT_FAIL, "Tier0: failed to get output");
        }
        ID3D12Resource* resource = get_resource(output_value);
        if (!resource) return api.CreateStatus(ORT_FAIL, "Tier0: null D3D12 resource for output");

        output_buffer_bindings[i] = { resource, 0, state->output_bytes[i] };
        output_descs[i] = { DML_BINDING_TYPE_BUFFER, &output_buffer_bindings[i] };
    }

    const DML_BUFFER_BINDING* persistent = state->persistent_binding
        ? &*state->persistent_binding : nullptr;

    HRESULT hr = state->provider->ExecuteOperator(
        state->compiled_op.Get(),
        persistent,
        gsl::make_span(input_descs),
        gsl::make_span(output_descs));

    if (FAILED(hr))
        return api.CreateStatus(ORT_FAIL, "Tier0: ExecuteOperator failed");

    state->provider->QueueReference(state->compiled_op.Get());
    return nullptr;
}

// ---------------------------------------------------------------------------
// OrtNodeComputeInfo wrapper
// ---------------------------------------------------------------------------

struct FullGraphNodeComputeInfo : OrtNodeComputeInfo {
    std::unique_ptr<FullGraphKernelState> state;

    FullGraphNodeComputeInfo() {
        ort_version_supported = ORT_API_VERSION;
        CreateState = [](OrtNodeComputeInfo* self, OrtNodeComputeContext*, void** out) noexcept -> OrtStatus* {
            *out = static_cast<FullGraphNodeComputeInfo*>(self)->state.get();
            return nullptr;
        };
        Compute = FullGraph_Compute;
        ReleaseState = [](OrtNodeComputeInfo* self, void*) noexcept {
            static_cast<FullGraphNodeComputeInfo*>(self)->state.reset();
        };
    }
};

// ---------------------------------------------------------------------------
// FullGraphFusion::ValidateTier0 — lightweight check from GetCapabilityImpl
//
// Verifies that every node input/output has a fully static shape and a DML-
// supported dtype. This is the precondition Compile relies on — without it,
// translators can't compute output dimensions. Translator/CreateOperator
// failures are caught by Compile, which returns nullptr for a clean fallback.
// ---------------------------------------------------------------------------

bool FullGraphFusion::ValidateTier0(
    const OrtApi&                                            ort_api,
    const std::vector<const OrtNode*>&                       nodes)
{
    // Validate node inputs: every input must have tensor type info with a
    // supported DML data type and all-static dims (scalars at rank 0 are fine).
    for (const OrtNode* node : nodes) {
        if (!node) continue;
        const char* op_type = nullptr;
        ort_api.Node_GetOperatorType(node, &op_type);
        size_t node_num_inputs = 0;
        ort_api.Node_GetNumInputs(node, &node_num_inputs);
        std::vector<const OrtValueInfo*> in_vis(node_num_inputs, nullptr);
        if (node_num_inputs > 0)
            ort_api.Node_GetInputs(node, in_vis.data(), node_num_inputs);
        auto input_names = fusion_utils::GetNodeInputNames(ort_api, node);
        for (size_t k = 0; k < node_num_inputs && k < input_names.size(); ++k) {
            if (in_vis[k] == nullptr || input_names[k].empty()) continue;
            auto* ti = in_vis[k]->GetTypeInfo();
            if (ti == nullptr || ti->tensor_type_info == nullptr) return false;
            const OrtTensorTypeAndShapeInfo* si = ti->tensor_type_info.get();
            if (si == nullptr) return false;
            ONNXTensorElementDataType onnx_dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            ort_api.GetTensorElementType(si, &onnx_dt);
            if (OnnxDtypeToDml(onnx_dt) == DML_TENSOR_DATA_TYPE_UNKNOWN) return false;
            if (!si->HasShape()) return false;
            size_t in_rank = 0;
            ort_api.GetDimensionsCount(si, &in_rank);
            if (in_rank > 0) {
                std::vector<int64_t> in_dims(in_rank, -1);
                ort_api.GetDimensions(si, in_dims.data(), in_rank);
                for (size_t d = 0; d < in_rank; ++d)
                    if (in_dims[d] <= 0) return false;
            }
        }
    }

    // Validate node outputs: same requirements.
    for (const OrtNode* node : nodes) {
        if (!node) continue;
        const char* op_type = nullptr;
        ort_api.Node_GetOperatorType(node, &op_type);
        size_t node_num_outputs = 0;
        ort_api.Node_GetNumOutputs(node, &node_num_outputs);
        std::vector<const OrtValueInfo*> out_vis(node_num_outputs, nullptr);
        if (node_num_outputs > 0)
            ort_api.Node_GetOutputs(node, out_vis.data(), node_num_outputs);
        auto output_names = fusion_utils::GetNodeOutputNames(ort_api, node);
        for (size_t k = 0; k < node_num_outputs && k < output_names.size(); ++k) {
            if (out_vis[k] == nullptr || output_names[k].empty()) continue;
            auto* ti = out_vis[k]->GetTypeInfo();
            if (ti == nullptr || ti->tensor_type_info == nullptr) return false;
            const OrtTensorTypeAndShapeInfo* si = ti->tensor_type_info.get();
            if (si == nullptr) return false;
            if (!si->HasShape()) return false;
            size_t out_rank = 0;
            ort_api.GetDimensionsCount(si, &out_rank);
            if (out_rank > 0) {
                std::vector<int64_t> out_dims(out_rank, -1);
                ort_api.GetDimensions(si, out_dims.data(), out_rank);
                for (size_t d = 0; d < out_rank; ++d)
                    if (out_dims[d] <= 0) return false;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Internal helpers for Compile
// ---------------------------------------------------------------------------

// Read the name string from an OrtValueInfo, releasing any error status.
static std::string ReadValueInfoName(const OrtApi& ort_api, const OrtValueInfo* vi) {
    if (!vi) return {};
    const char* name = nullptr;
    OrtStatus* st = ort_api.GetValueInfoName(vi, &name);
    std::string r = (st || !name) ? std::string{} : std::string(name);
    if (st) ort_api.ReleaseStatus(st);
    return r;
}

// ---------------------------------------------------------------------------
// BuildSubgraphInfo — Steps 1–3 of Compile
//
// Enumerates graph inputs/outputs, merges the initializer maps, and pre-seeds
// value_shapes from graph inputs, initializers, and ORT shape-inference
// results on all node outputs. Returns everything Compile needs to begin
// translation.
// ---------------------------------------------------------------------------

struct SubgraphInfo {
    std::vector<const OrtValueInfo*>                        graph_input_vis;
    std::vector<const OrtValueInfo*>                        graph_output_vis;
    std::unordered_map<std::string, size_t>                 graph_input_map;
    std::unordered_map<std::string, size_t>                 graph_output_map;
    std::unordered_map<std::string, DmlTensorInfo>          value_shapes;
    std::unordered_map<std::string, const OrtValue*>        all_initializers;
};

static SubgraphInfo BuildSubgraphInfo(
    const OrtApi& ort_api,
    const OrtGraph* fused_subgraph,
    const std::unordered_map<std::string, const OrtValue*>& initializers)
{
    SubgraphInfo info;

    // Enumerate graph inputs and outputs.
    size_t num_graph_inputs = 0;
    ort_api.Graph_GetNumInputs(fused_subgraph, &num_graph_inputs);
    info.graph_input_vis.assign(num_graph_inputs, nullptr);
    if (num_graph_inputs > 0)
        ort_api.Graph_GetInputs(fused_subgraph, info.graph_input_vis.data(), num_graph_inputs);

    size_t num_graph_outputs = 0;
    ort_api.Graph_GetNumOutputs(fused_subgraph, &num_graph_outputs);
    info.graph_output_vis.assign(num_graph_outputs, nullptr);
    if (num_graph_outputs > 0)
        ort_api.Graph_GetOutputs(fused_subgraph, info.graph_output_vis.data(), num_graph_outputs);

    // Build name→index maps for fast lookup during edge wiring.
    for (size_t i = 0; i < num_graph_inputs; ++i) {
        std::string name = ReadValueInfoName(ort_api, info.graph_input_vis[i]);
        if (!name.empty()) info.graph_input_map[name] = i;
    }
    for (size_t i = 0; i < num_graph_outputs; ++i) {
        std::string name = ReadValueInfoName(ort_api, info.graph_output_vis[i]);
        if (!name.empty()) info.graph_output_map[name] = i;
    }

    // Merge subgraph-local initializers into the parent initializer map.
    info.all_initializers = initializers;
    {
        size_t num_init = 0;
        ort_api.Graph_GetNumInitializers(fused_subgraph, &num_init);
        if (num_init > 0) {
            std::vector<const OrtValueInfo*> init_vis(num_init, nullptr);
            ort_api.Graph_GetInitializers(fused_subgraph, init_vis.data(), num_init);
            for (const OrtValueInfo* vi : init_vis) {
                if (!vi) continue;
                const char* name = nullptr;
                OrtStatus* st = ort_api.GetValueInfoName(vi, &name);
                if (st || !name) { if (st) ort_api.ReleaseStatus(st); continue; }
                const OrtValue* val = nullptr;
                st = ort_api.ValueInfo_GetInitializerValue(vi, &val);
                if (st) { ort_api.ReleaseStatus(st); continue; }
                if (val) info.all_initializers[name] = val;
            }
        }
    }

    // Seed value_shapes from graph inputs (type info from ORT shape inference).
    for (size_t i = 0; i < num_graph_inputs; ++i) {
        const OrtValueInfo* vi = info.graph_input_vis[i];
        if (!vi) continue;
        std::string name = ReadValueInfoName(ort_api, vi);
        if (name.empty()) continue;

        if (!vi->GetTypeInfo() || !vi->GetTypeInfo()->tensor_type_info) continue;
        const OrtTensorTypeAndShapeInfo* si = vi->GetTypeInfo()->tensor_type_info.get();
        if (!si->HasShape()) continue;

        ONNXTensorElementDataType onnx_dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        ort_api.GetTensorElementType(si, &onnx_dtype);
        DML_TENSOR_DATA_TYPE dml_dtype = OnnxDtypeToDml(onnx_dtype);
        if (dml_dtype == DML_TENSOR_DATA_TYPE_UNKNOWN) continue;

        size_t rank = 0;
        ort_api.GetDimensionsCount(si, &rank);
        std::vector<uint32_t> sizes;
        if (rank > 0) {
            std::vector<int64_t> dims(rank, -1);
            ort_api.GetDimensions(si, dims.data(), rank);
            sizes.resize(rank);
            for (size_t d = 0; d < rank; ++d)
                sizes[d] = static_cast<uint32_t>(dims[d] > 0 ? dims[d] : 1);
        }
        // Scalars (rank 0) get an empty sizes vector; MakeTensorInfo pads to [1,1,1,1].

        auto tensor_info = MakeTensorInfo(sizes, dml_dtype);
        tensor_info.original_rank = static_cast<uint32_t>(rank);
        info.value_shapes[name] = tensor_info;
    }

    // Seed value_shapes from initializers (shape from OrtValue).
    for (const auto& [init_name, init_value] : info.all_initializers) {
        if (!init_value) continue;
        if (info.value_shapes.count(init_name)) continue; // already seeded from graph input

        OrtTensorTypeAndShapeInfo* shape_info = nullptr;
        ort_api.GetTensorTypeAndShape(const_cast<OrtValue*>(init_value), &shape_info);
        if (!shape_info) continue;

        ONNXTensorElementDataType onnx_dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        ort_api.GetTensorElementType(shape_info, &onnx_dtype);
        if (!shape_info->HasShape()) {
            ort_api.ReleaseTensorTypeAndShapeInfo(shape_info);
            continue;
        }
        size_t rank = 0;
        ort_api.GetDimensionsCount(shape_info, &rank);
        std::vector<int64_t> dims(rank, 0);
        if (rank > 0) ort_api.GetDimensions(shape_info, dims.data(), rank);
        ort_api.ReleaseTensorTypeAndShapeInfo(shape_info);

        DML_TENSOR_DATA_TYPE dml_dtype = OnnxDtypeToDml(onnx_dtype);
        if (dml_dtype == DML_TENSOR_DATA_TYPE_UNKNOWN) continue;

        std::vector<uint32_t> sizes(rank);
        for (size_t d = 0; d < rank; ++d)
            sizes[d] = static_cast<uint32_t>(dims[d] > 0 ? dims[d] : 1);

        auto tensor_info = MakeTensorInfo(sizes, dml_dtype);
        tensor_info.original_rank = static_cast<uint32_t>(rank);
        info.value_shapes[init_name] = tensor_info;
    }

    // Pre-seed all node output shapes from ORT shape inference.
    // Tier-0 requires all-static shapes, so every node output should have known dims.
    size_t num_nodes = 0;
    ort_api.Graph_GetNumNodes(fused_subgraph, &num_nodes);
    std::vector<const OrtNode*> subgraph_nodes(num_nodes, nullptr);
    if (num_nodes > 0)
        ort_api.Graph_GetNodes(fused_subgraph, subgraph_nodes.data(), num_nodes);

    for (const OrtNode* node : subgraph_nodes) {
        if (!node) continue;
        auto output_names = fusion_utils::GetNodeOutputNames(ort_api, node);
        size_t node_num_outputs = 0;
        ort_api.Node_GetNumOutputs(node, &node_num_outputs);
        std::vector<const OrtValueInfo*> out_vis(node_num_outputs, nullptr);
        if (node_num_outputs > 0)
            ort_api.Node_GetOutputs(node, out_vis.data(), node_num_outputs);

        for (size_t k = 0; k < node_num_outputs && k < output_names.size(); ++k) {
            if (!out_vis[k] || output_names[k].empty()) continue;
            if (info.value_shapes.count(output_names[k])) continue;
            auto* ti = out_vis[k]->GetTypeInfo();
            if (!ti || !ti->tensor_type_info) continue;
            const OrtTensorTypeAndShapeInfo* si = ti->tensor_type_info.get();
            if (!si->HasShape()) continue;
            ONNXTensorElementDataType onnx_dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            ort_api.GetTensorElementType(si, &onnx_dt);
            DML_TENSOR_DATA_TYPE dml_dt = OnnxDtypeToDml(onnx_dt);
            if (dml_dt == DML_TENSOR_DATA_TYPE_UNKNOWN) continue;
            size_t out_rank = 0;
            ort_api.GetDimensionsCount(si, &out_rank);
            std::vector<uint32_t> out_sizes;
            bool all_static = true;
            if (out_rank > 0) {
                std::vector<int64_t> out_dims(out_rank, -1);
                ort_api.GetDimensions(si, out_dims.data(), out_rank);
                out_sizes.resize(out_rank);
                for (size_t d = 0; d < out_rank; ++d) {
                    if (out_dims[d] <= 0) { all_static = false; break; }
                    out_sizes[d] = static_cast<uint32_t>(out_dims[d]);
                }
            }
            // Scalars (rank 0) are inherently static.
            if (all_static) {
                auto tensor_info = MakeTensorInfo(out_sizes, dml_dt);
                tensor_info.original_rank = static_cast<uint32_t>(out_rank);
                info.value_shapes[output_names[k]] = tensor_info;
            }
        }
    }

    return info;
}

// ---------------------------------------------------------------------------
// BuildDmlInputMap — Step 5 of Compile
//
// Classifies each consumed initializer as either:
//   1. Small constant (< kMaxConstNodeDataSize bytes): embedded as a
//      DML_GRAPH_NODE_TYPE_CONSTANT node with inline data.
//   2. Large initializer (>= kMaxConstNodeDataSize bytes): assigned a DML
//      graph input slot with OWNED_BY_DML. Data uploaded at init time.
//
// kMaxConstNodeDataSize = 8 matches ORT's c_maxConstNodeDataSize threshold.
//
// Returns the DML input index map, subgraph-to-DML mapping, constant node
// data, and the ordered list of Mode-B initializer names.
// ---------------------------------------------------------------------------

static constexpr uint64_t kMaxConstNodeDataSize = 8;

struct ConstantNodeInfo {
    std::string          name;
    std::vector<uint8_t> data;
};

struct DmlInputMapResult {
    std::unordered_map<std::string, size_t> dml_input_map;
    std::vector<size_t>                     subgraph_to_dml_input;
    std::vector<ConstantNodeInfo>           constant_nodes;
    std::unordered_map<std::string, size_t> constant_node_map;
    std::vector<std::string>               ordered_initializer_names;
    size_t                                 total_dml_inputs = 0;
};

static DmlInputMapResult BuildDmlInputMap(
    const OrtApi& ort_api,
    const std::vector<const OrtValueInfo*>& graph_input_vis,
    const std::unordered_set<std::string>& consumed_initializer_names,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>& all_initializers)
{
    DmlInputMapResult result;
    result.subgraph_to_dml_input.assign(graph_input_vis.size(), SIZE_MAX);

    // Pass 1: inline small consumed initializers as DML constant nodes.
    for (const auto& init_name : consumed_initializer_names) {
        auto shape_it = value_shapes.find(init_name);
        uint64_t bytes = (shape_it != value_shapes.end()) ? shape_it->second.total_bytes : 0;
        if (bytes == 0 || bytes >= kMaxConstNodeDataSize) continue;

        auto init_it = all_initializers.find(init_name);
        if (init_it == all_initializers.end() || !init_it->second) continue;

        void* cpu_ptr = nullptr;
        OrtStatus* st = ort_api.GetTensorMutableData(
            const_cast<OrtValue*>(init_it->second), &cpu_ptr);
        if (!st && cpu_ptr) {
            ConstantNodeInfo cni;
            cni.name = init_name;
            cni.data.assign(static_cast<uint8_t*>(cpu_ptr),
                            static_cast<uint8_t*>(cpu_ptr) + bytes);
            result.constant_node_map[init_name] = result.constant_nodes.size();
            result.constant_nodes.push_back(std::move(cni));
        }
        if (st) ort_api.ReleaseStatus(st);
    }

    // Pass 2: assign DML graph input indices starting from subgraph graph inputs.
    //
    // Mode A — ORT exposes both runtime tensors and initializers as subgraph graph
    //   inputs. All appear in graph_input_vis. Large initializers get an index here.
    // Mode B — ORT exposes only runtime tensors as subgraph graph inputs. Initializers
    //   arrive only via all_initializers (added in Pass 3 below).
    size_t dml_input_idx = 0;
    for (size_t i = 0; i < graph_input_vis.size(); ++i) {
        const char* name = nullptr;
        OrtStatus* st = ort_api.GetValueInfoName(graph_input_vis[i], &name);
        std::string n = (st || !name) ? std::string{} : std::string(name);
        if (st) ort_api.ReleaseStatus(st);
        if (n.empty()) continue;
        if (result.constant_node_map.count(n)) continue; // inlined — no slot needed
        result.dml_input_map[n] = dml_input_idx;
        result.subgraph_to_dml_input[i] = dml_input_idx;
        ++dml_input_idx;
    }

    // Pass 3: for Mode B, assign slots for large consumed initializers not already
    // in the subgraph graph input list. ordered_initializer_names records insertion
    // order for the upload loop in UploadInitializers.
    for (const auto& init_name : consumed_initializer_names) {
        if (result.constant_node_map.count(init_name)) continue; // inlined
        if (result.dml_input_map.count(init_name)) continue;     // already assigned (Mode A)
        result.dml_input_map[init_name] = dml_input_idx++;
        result.ordered_initializer_names.push_back(init_name);
    }

    result.total_dml_inputs = dml_input_idx;
    return result;
}

// ---------------------------------------------------------------------------
// UploadInitializers — Step 7 upload portion of Compile
//
// Allocates GPU resources for every OWNED_BY_DML initializer and copies the
// CPU data from all_initializers. Writes DML_BUFFER_BINDINGs into
// init_input_bindings[dml_idx] for use in InitializeOperator.
// Returns false if any allocation or upload fails.
// ---------------------------------------------------------------------------

struct InitBinding {
    ComPtr<ID3D12Resource> gpu_resource;
    ComPtr<IUnknown>       allocator_ref;
    uint64_t               bytes = 0;
};

static bool UploadInitializers(
    const OrtApi& ort_api,
    PluginDmlExecutionProviderImpl* provider,
    const DmlInputMapResult& input_map,
    const std::unordered_set<size_t>& owned_graph_input_indices,
    const std::vector<const OrtValueInfo*>& graph_input_vis,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>& all_initializers,
    std::vector<InitBinding>& const_graph_input_bindings)
{
    auto upload_one = [&](const std::string& name, size_t dml_idx) -> bool {
        if (!owned_graph_input_indices.count(dml_idx)) return true;
        auto init_it = all_initializers.find(name);
        if (init_it == all_initializers.end() || !init_it->second) return true;

        void* cpu_ptr = nullptr;
        OrtStatus* st = ort_api.GetTensorMutableData(
            const_cast<OrtValue*>(init_it->second), &cpu_ptr);
        if (st || !cpu_ptr) { if (st) ort_api.ReleaseStatus(st); return true; }

        OrtTensorTypeAndShapeInfo* tsi = nullptr;
        ort_api.GetTensorTypeAndShape(const_cast<OrtValue*>(init_it->second), &tsi);
        size_t elem_count = 0;
        ONNXTensorElementDataType dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        if (tsi) {
            ort_api.GetTensorShapeElementCount(tsi, &elem_count);
            ort_api.GetTensorElementType(tsi, &dt);
            ort_api.ReleaseTensorTypeAndShapeInfo(tsi);
        }
        uint64_t actual_bytes = static_cast<uint64_t>(elem_count) * DmlDataTypeSize(OnnxDtypeToDml(dt));

        auto shape_it = value_shapes.find(name);
        uint64_t bytes = (shape_it != value_shapes.end()) ? shape_it->second.total_bytes : 0;
        if (bytes == 0) return true;
        uint64_t upload_bytes = std::min(bytes, actual_bytes > 0 ? actual_bytes : bytes);

        InitBinding ib;
        if (FAILED(provider->AllocatePooledResource(
                static_cast<size_t>(bytes), AllocatorRoundingMode::Disabled,
                ib.gpu_resource.GetAddressOf(), ib.allocator_ref.GetAddressOf())))
            return false;
        if (FAILED(provider->UploadToResource(ib.gpu_resource.Get(), cpu_ptr, upload_bytes)))
            return false;
        ib.bytes = bytes;
        const_graph_input_bindings[dml_idx] = std::move(ib);
        return true;
    };

    // Mode A: subgraph graph inputs (includes initializers exposed as graph inputs).
    for (size_t i = 0; i < graph_input_vis.size(); ++i) {
        if (input_map.subgraph_to_dml_input[i] == SIZE_MAX) continue;
        const char* name = nullptr;
        OrtStatus* st = ort_api.GetValueInfoName(graph_input_vis[i], &name);
        std::string n = (st || !name) ? std::string{} : std::string(name);
        if (st) ort_api.ReleaseStatus(st);
        if (!n.empty() && !upload_one(n, input_map.subgraph_to_dml_input[i]))
            return false;
    }

    // Mode B: initializers added as extra DML inputs (not in subgraph graph inputs).
    for (const auto& name : input_map.ordered_initializer_names) {
        auto di_it = input_map.dml_input_map.find(name);
        if (di_it != input_map.dml_input_map.end() && !upload_one(name, di_it->second))
            return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// FullGraphFusion::Compile
// ---------------------------------------------------------------------------

OrtNodeComputeInfo* FullGraphFusion::Compile(
    const OrtApi&                                            ort_api,
    const OrtGraph*                                          fused_subgraph,
    const std::unordered_map<std::string, const OrtValue*>&  initializers,
    PluginDmlExecutionProviderImpl*                          provider)
{
    // -----------------------------------------------------------------------
    // Step 1–3: Enumerate subgraph I/O, merge initializers, pre-seed shapes.
    // -----------------------------------------------------------------------

    SubgraphInfo sg = BuildSubgraphInfo(ort_api, fused_subgraph, initializers);
    const size_t num_graph_inputs  = sg.graph_input_vis.size();
    const size_t num_graph_outputs = sg.graph_output_vis.size();

    // -----------------------------------------------------------------------
    // Step 4: Translate each node.
    // -----------------------------------------------------------------------

    OpTranslatorRegistry registry = BuildOpTranslatorRegistry();

    std::vector<CompiledNode> compiled_nodes;
    compiled_nodes.reserve(64);

    std::unordered_map<std::string, std::pair<size_t, size_t>> value_producer;
    std::unordered_map<std::string, std::string> graph_input_aliases;
    std::unordered_set<std::string> consumed_initializer_names;

    size_t num_nodes = 0;
    ort_api.Graph_GetNumNodes(fused_subgraph, &num_nodes);
    std::vector<const OrtNode*> subgraph_nodes(num_nodes, nullptr);
    if (num_nodes > 0)
        ort_api.Graph_GetNodes(fused_subgraph, subgraph_nodes.data(), num_nodes);

    for (const OrtNode* node : subgraph_nodes) {
        if (!node) continue;

        const char* op_type = nullptr;
        ort_api.Node_GetOperatorType(node, &op_type);
        if (!op_type) return nullptr;

        auto reg_it = registry.find(op_type);
        if (reg_it == registry.end()) return nullptr;

        auto input_names  = fusion_utils::GetNodeInputNames(ort_api, node);
        auto output_names = fusion_utils::GetNodeOutputNames(ort_api, node);

        auto translated = reg_it->second(ort_api, node, sg.value_shapes, sg.all_initializers);
        if (!translated) return nullptr;

        auto translated_ptr = std::make_unique<TranslatedOp>(std::move(*translated));
        translated_ptr->FixupPointers();

        if (translated_ptr->passthrough) {
            // Elide this node from the DML graph. Its outputs alias input[0]'s
            // source, so downstream consumers connect directly to the upstream
            // producer. This handles Reshape (no-op reinterpretation).
            if (!input_names.empty()) {
                const auto& src = input_names[0];
                auto prod_it = value_producer.find(src);
                if (prod_it != value_producer.end()) {
                    for (auto& oname : output_names)
                        value_producer[oname] = prod_it->second;
                } else {
                    // Source is a graph input or initializer — register an alias
                    // so the edge-wiring loop can find it by the output name.
                    for (auto& oname : output_names)
                        graph_input_aliases[oname] = src;
                }
            }
            continue;
        }

        // Track initializer consumption — only for inputs the DML operator uses.
        // Include initializers that are also graph inputs (Tier-0 path) so they
        // can be classified as constant nodes vs OWNED in Step 5.
        size_t dml_input_count = translated_ptr->input_tensors.size();
        for (size_t s = 0; s < input_names.size() && s < dml_input_count; ++s) {
            const auto& in_name = input_names[s];
            if (sg.all_initializers.count(in_name))
                consumed_initializer_names.insert(in_name);
        }

        // Track which DML graph node produces each output value.
        // When sub_nodes exist, the last sub_node is the output producer.
        size_t producer_compiled_idx = compiled_nodes.size();
        for (size_t k = 0; k < output_names.size(); ++k)
            value_producer[output_names[k]] = { producer_compiled_idx, k };

        CompiledNode cn;
        cn.translated = std::move(translated_ptr);
        cn.input_names = std::move(input_names);
        cn.output_names = std::move(output_names);
        cn.op_type = op_type;
        compiled_nodes.push_back(std::move(cn));
    }

    // -----------------------------------------------------------------------
    // Step 5: Mark OWNED_BY_DML initializer inputs.
    //
    // Passthrough aliases (e.g. Reshape output → original initializer) are
    // resolved so bias tensors flowing through elided Reshape nodes are
    // correctly recognized. Small initializers (<= kMaxConstNodeDataSize bytes)
    // are excluded — they become constant graph nodes with embedded data.
    // -----------------------------------------------------------------------

    uint32_t owned_count = 0;
    for (auto& cn : compiled_nodes) {
        size_t dml_input_count = cn.translated->input_tensors.size();
        for (size_t s = 0; s < cn.input_names.size() && s < dml_input_count; ++s) {
            auto resolved = cn.input_names[s];
            while (graph_input_aliases.count(resolved))
                resolved = graph_input_aliases[resolved];
            if (!sg.all_initializers.count(cn.input_names[s]) && !sg.all_initializers.count(resolved))
                continue;

            auto shape_it = sg.value_shapes.find(
                sg.all_initializers.count(resolved) ? resolved : cn.input_names[s]);
            uint64_t bytes = (shape_it != sg.value_shapes.end()) ? shape_it->second.total_bytes : 0;
            if (bytes >= kMaxConstNodeDataSize) {
                cn.translated->input_buffer_descs[s].Flags |= DML_TENSOR_FLAG_OWNED_BY_DML;
                ++owned_count;
            }
        }
        cn.translated->FixupPointers();
    }
    (void)owned_count;

    // -----------------------------------------------------------------------
    // Step 6: Create IDMLOperator for each compiled node.
    // -----------------------------------------------------------------------

    ComPtr<IDMLDevice> dml_device;
    if (FAILED(provider->GetDmlDevice(dml_device.GetAddressOf()))) return nullptr;
    ComPtr<IDMLDevice1> dml_device1;
    if (FAILED(dml_device.As(&dml_device1))) return nullptr;

    for (size_t i = 0; i < compiled_nodes.size(); ++i) {
        auto& cn = compiled_nodes[i];
        HRESULT hr = dml_device->CreateOperator(
            &cn.translated->op_desc, IID_PPV_ARGS(cn.translated->dml_operator.GetAddressOf()));
        if (FAILED(hr)) return nullptr;

        for (size_t s = 0; s < cn.translated->sub_nodes.size(); ++s) {
            auto& sn = cn.translated->sub_nodes[s];
            hr = dml_device->CreateOperator(
                &sn.op_desc, IID_PPV_ARGS(sn.dml_operator.GetAddressOf()));
            if (FAILED(hr)) return nullptr;
        }
    }

    // -----------------------------------------------------------------------
    // Step 7: Build DML graph input map and constant nodes.
    //
    // Initializers reach Compile in two modes depending on ORT partitioning:
    //
    // Mode A — Initializers as fused subgraph graph inputs. ORT lists both
    //   runtime tensors and initializers in Graph_GetInputs (e.g. 52 weights +
    //   1 runtime input = 53 graph inputs). all_initializers is a subset.
    //
    // Mode B — Initializers via all_initializers only. ORT only exposes true
    //   runtime inputs in the subgraph graph input list. Weights arrive only
    //   through the session-level initializers map and are not in graph_input_vis.
    //
    // BuildDmlInputMap handles both modes and assigns compact DML index slots.
    // -----------------------------------------------------------------------

    DmlInputMapResult im = BuildDmlInputMap(
        ort_api, sg.graph_input_vis,
        consumed_initializer_names, sg.value_shapes, sg.all_initializers);

    // Build owned_graph_input_indices: DML graph input indices whose tensors
    // carry DML_TENSOR_FLAG_OWNED_BY_DML. Uses im.dml_input_map so both Mode A
    // and Mode B initializers are covered.
    std::unordered_set<size_t> owned_graph_input_indices;
    for (auto& cn : compiled_nodes) {
        size_t dml_input_count = cn.translated->input_tensors.size();
        for (size_t s = 0; s < cn.input_names.size() && s < dml_input_count; ++s) {
            if (cn.translated->input_buffer_descs[s].Flags & DML_TENSOR_FLAG_OWNED_BY_DML) {
                auto resolved_name = cn.input_names[s];
                while (graph_input_aliases.count(resolved_name))
                    resolved_name = graph_input_aliases[resolved_name];
                auto di_it = im.dml_input_map.find(resolved_name);
                if (di_it != im.dml_input_map.end())
                    owned_graph_input_indices.insert(di_it->second);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 8: Build DML graph node descriptors.
    // dml_node_offset[i] = index in graph_nodes of compiled_nodes[i]'s primary
    // node. Sub_nodes follow immediately after.
    // -----------------------------------------------------------------------

    size_t total_dml_nodes = 0;
    std::vector<size_t> dml_node_offset(compiled_nodes.size());
    for (size_t i = 0; i < compiled_nodes.size(); ++i) {
        dml_node_offset[i] = total_dml_nodes;
        total_dml_nodes += 1 + compiled_nodes[i].translated->sub_nodes.size();
    }

    std::vector<DML_OPERATOR_GRAPH_NODE_DESC> op_node_descs(total_dml_nodes);
    std::vector<DML_GRAPH_NODE_DESC> graph_nodes(total_dml_nodes);
    for (size_t i = 0; i < compiled_nodes.size(); ++i) {
        size_t base = dml_node_offset[i];
        op_node_descs[base] = { compiled_nodes[i].translated->dml_operator.Get(), nullptr };
        graph_nodes[base] = { DML_GRAPH_NODE_TYPE_OPERATOR, &op_node_descs[base] };
        for (size_t s = 0; s < compiled_nodes[i].translated->sub_nodes.size(); ++s) {
            size_t idx = base + 1 + s;
            op_node_descs[idx] = { compiled_nodes[i].translated->sub_nodes[s].dml_operator.Get(), nullptr };
            graph_nodes[idx] = { DML_GRAPH_NODE_TYPE_OPERATOR, &op_node_descs[idx] };
        }
    }

    // Append constant graph nodes for small initializers (inline data).
    std::vector<DML_CONSTANT_DATA_GRAPH_NODE_DESC> const_node_descs(im.constant_nodes.size());
    for (size_t c = 0; c < im.constant_nodes.size(); ++c) {
        const_node_descs[c].Data = im.constant_nodes[c].data.data();
        const_node_descs[c].DataSize = im.constant_nodes[c].data.size();
        graph_nodes.push_back({ DML_GRAPH_NODE_TYPE_CONSTANT, &const_node_descs[c] });
    }
    size_t const_node_base = total_dml_nodes; // operator nodes before constants

    // -----------------------------------------------------------------------
    // Step 9: Wire edges.
    // -----------------------------------------------------------------------

    std::vector<DML_INPUT_GRAPH_EDGE_DESC> input_edge_storage;
    std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediate_edge_storage;
    std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> output_edge_storage;

    for (size_t node_idx = 0; node_idx < compiled_nodes.size(); ++node_idx) {
        const auto& cn = compiled_nodes[node_idx];
        size_t primary_dml_idx = dml_node_offset[node_idx];

        // Only wire edges for as many inputs as the DML operator actually has.
        size_t num_dml_inputs = cn.translated->input_tensors.size();
        for (size_t input_slot = 0; input_slot < cn.input_names.size() && input_slot < num_dml_inputs; ++input_slot) {
            auto name = cn.input_names[input_slot];
            if (name.empty()) continue;

            // Resolve passthrough aliases (e.g. Reshape output → original input).
            while (graph_input_aliases.count(name))
                name = graph_input_aliases[name];

            // Small constant inlined as a constant node.
            auto const_it = im.constant_node_map.find(name);
            if (const_it != im.constant_node_map.end()) {
                DML_INTERMEDIATE_GRAPH_EDGE_DESC edge{};
                edge.FromNodeIndex = static_cast<UINT>(const_node_base + const_it->second);
                edge.FromNodeOutputIndex = 0;
                edge.ToNodeIndex = static_cast<UINT>(primary_dml_idx);
                edge.ToNodeInputIndex = static_cast<UINT>(input_slot);
                intermediate_edge_storage.push_back(edge);
                continue;
            }

            // DML graph input (runtime tensor or large initializer).
            auto dml_in_it = im.dml_input_map.find(name);
            if (dml_in_it != im.dml_input_map.end()) {
                DML_INPUT_GRAPH_EDGE_DESC edge{};
                edge.GraphInputIndex = static_cast<UINT>(dml_in_it->second);
                edge.ToNodeIndex = static_cast<UINT>(primary_dml_idx);
                edge.ToNodeInputIndex = static_cast<UINT>(input_slot);
                input_edge_storage.push_back(edge);
                continue;
            }

            // Produced by a prior node.
            auto prod_it = value_producer.find(name);
            if (prod_it != value_producer.end()) {
                size_t prod_compiled_idx = prod_it->second.first;
                // If the producer has sub_nodes, output comes from the last sub_node.
                size_t prod_dml_idx = dml_node_offset[prod_compiled_idx];
                size_t num_subs = compiled_nodes[prod_compiled_idx].translated->sub_nodes.size();
                if (num_subs > 0)
                    prod_dml_idx += num_subs;

                DML_INTERMEDIATE_GRAPH_EDGE_DESC edge{};
                edge.FromNodeIndex = static_cast<UINT>(prod_dml_idx);
                edge.FromNodeOutputIndex = static_cast<UINT>(prod_it->second.second);
                edge.ToNodeIndex = static_cast<UINT>(primary_dml_idx);
                edge.ToNodeInputIndex = static_cast<UINT>(input_slot);
                intermediate_edge_storage.push_back(edge);
                continue;
            }

            return nullptr; // unresolved input
        }

        // Wire internal edges for sub_nodes.
        for (size_t s = 0; s < cn.translated->sub_nodes.size(); ++s) {
            const auto& sn = cn.translated->sub_nodes[s];
            size_t sn_dml_idx = primary_dml_idx + 1 + s;
            for (size_t inp = 0; inp < sn.input_from.size(); ++inp) {
                auto [src_sub, src_slot] = sn.input_from[inp];
                size_t from_dml_idx = (src_sub < 0)
                    ? primary_dml_idx
                    : primary_dml_idx + 1 + static_cast<size_t>(src_sub);
                DML_INTERMEDIATE_GRAPH_EDGE_DESC edge{};
                edge.FromNodeIndex = static_cast<UINT>(from_dml_idx);
                edge.FromNodeOutputIndex = static_cast<UINT>(src_slot);
                edge.ToNodeIndex = static_cast<UINT>(sn_dml_idx);
                edge.ToNodeInputIndex = static_cast<UINT>(inp);
                intermediate_edge_storage.push_back(edge);
            }
        }
    }

    // Output edges.
    for (const auto& [name, out_idx] : sg.graph_output_map) {
        auto resolved = name;
        while (graph_input_aliases.count(resolved))
            resolved = graph_input_aliases[resolved];
        auto prod_it = value_producer.find(resolved);
        if (prod_it == value_producer.end()) return nullptr;

        size_t prod_compiled_idx = prod_it->second.first;
        size_t prod_dml_idx = dml_node_offset[prod_compiled_idx];
        size_t num_subs = compiled_nodes[prod_compiled_idx].translated->sub_nodes.size();
        if (num_subs > 0)
            prod_dml_idx += num_subs;

        DML_OUTPUT_GRAPH_EDGE_DESC edge{};
        edge.FromNodeIndex = static_cast<UINT>(prod_dml_idx);
        edge.FromNodeOutputIndex = static_cast<UINT>(prod_it->second.second);
        edge.GraphOutputIndex = static_cast<UINT>(out_idx);
        output_edge_storage.push_back(edge);
    }

    // Track which DML graph input indices have actual edges. Inputs with no
    // edges (e.g. shape params consumed at translation time) must not be bound
    // at dispatch — DML rejects bindings for edgeless inputs.
    std::unordered_set<size_t> dml_inputs_with_edges;
    for (const auto& ie : input_edge_storage)
        dml_inputs_with_edges.insert(ie.GraphInputIndex);

    // Wrap storage in typed edge descriptors.
    std::vector<DML_GRAPH_EDGE_DESC> input_edges(input_edge_storage.size());
    for (size_t i = 0; i < input_edge_storage.size(); ++i)
        input_edges[i] = { DML_GRAPH_EDGE_TYPE_INPUT, &input_edge_storage[i] };

    std::vector<DML_GRAPH_EDGE_DESC> intermediate_edges(intermediate_edge_storage.size());
    for (size_t i = 0; i < intermediate_edge_storage.size(); ++i)
        intermediate_edges[i] = { DML_GRAPH_EDGE_TYPE_INTERMEDIATE, &intermediate_edge_storage[i] };

    std::vector<DML_GRAPH_EDGE_DESC> output_edges(output_edge_storage.size());
    for (size_t i = 0; i < output_edge_storage.size(); ++i)
        output_edges[i] = { DML_GRAPH_EDGE_TYPE_OUTPUT, &output_edge_storage[i] };

    // -----------------------------------------------------------------------
    // Step 10: Compile the DML graph.
    // -----------------------------------------------------------------------

    DML_GRAPH_DESC graph_desc{};
    graph_desc.InputCount = static_cast<UINT>(im.total_dml_inputs);
    graph_desc.OutputCount = static_cast<UINT>(num_graph_outputs);
    graph_desc.NodeCount = static_cast<UINT>(graph_nodes.size());
    graph_desc.Nodes = graph_nodes.data();
    graph_desc.InputEdgeCount = static_cast<UINT>(input_edges.size());
    graph_desc.InputEdges = input_edges.data();
    graph_desc.OutputEdgeCount = static_cast<UINT>(output_edges.size());
    graph_desc.OutputEdges = output_edges.data();
    graph_desc.IntermediateEdgeCount = static_cast<UINT>(intermediate_edges.size());
    graph_desc.IntermediateEdges = intermediate_edges.data();

    ComPtr<IDMLCompiledOperator> compiled_op;
    static constexpr size_t kMinNodeCountForDescriptorsVolatile = 5;
    DML_EXECUTION_FLAGS exec_flags = DML_EXECUTION_FLAG_NONE;
    if (compiled_nodes.size() >= kMinNodeCountForDescriptorsVolatile)
        exec_flags |= DML_EXECUTION_FLAG_DESCRIPTORS_VOLATILE;

    HRESULT hr = dml_device1->CompileGraph(
        &graph_desc, exec_flags,
        IID_PPV_ARGS(compiled_op.GetAddressOf()));
    if (FAILED(hr)) return nullptr;

    // -----------------------------------------------------------------------
    // Step 11: Allocate persistent resource and upload initializers.
    // -----------------------------------------------------------------------

    ComPtr<ID3D12Resource> persistent_resource;
    ComPtr<IUnknown> persistent_allocator;
    std::optional<DML_BUFFER_BINDING> persistent_binding;

    auto binding_props = compiled_op->GetBindingProperties();
    UINT64 persistent_size = binding_props.PersistentResourceSize;
    if (persistent_size > 0) {
        if (FAILED(provider->AllocatePooledResource(
                static_cast<size_t>(persistent_size), AllocatorRoundingMode::Disabled,
                persistent_resource.GetAddressOf(),
                persistent_allocator.GetAddressOf())))
            return nullptr;
        persistent_binding = DML_BUFFER_BINDING{
            persistent_resource.Get(), 0, persistent_size };
    }

    std::vector<InitBinding> const_graph_input_bindings(im.total_dml_inputs);
    if (!UploadInitializers(
            ort_api, provider, im, owned_graph_input_indices,
            sg.graph_input_vis, sg.value_shapes, sg.all_initializers,
            const_graph_input_bindings))
        return nullptr;

    // Build init input bindings array indexed by DML graph input index.
    std::vector<DML_BUFFER_BINDING> init_input_bindings(im.total_dml_inputs, DML_BUFFER_BINDING{});
    for (size_t i = 0; i < const_graph_input_bindings.size(); ++i) {
        if (const_graph_input_bindings[i].gpu_resource) {
            auto& ib = const_graph_input_bindings[i];
            init_input_bindings[i] = { ib.gpu_resource.Get(), 0, ib.bytes };
        }
    }

    const DML_BUFFER_BINDING* persistent_ptr =
        persistent_binding ? &*persistent_binding : nullptr;

    if (FAILED(provider->InitializeOperator(
            compiled_op.Get(), persistent_ptr,
            gsl::make_span(init_input_bindings))))
        return nullptr;

    // Flush to surface any deferred GPU errors from InitializeOperator.
    provider->Flush();

    provider->QueueReference(compiled_op.Get());
    if (persistent_allocator)
        provider->QueueReference(persistent_allocator.Get());
    for (auto& ib : const_graph_input_bindings) {
        if (ib.gpu_resource) provider->QueueReference(ib.gpu_resource.Get());
    }
    const_graph_input_bindings.clear();

    // -----------------------------------------------------------------------
    // Step 12: Build kernel state.
    // -----------------------------------------------------------------------

    auto kernel_state = std::make_unique<FullGraphKernelState>();
    kernel_state->provider = provider;
    kernel_state->ort_api = &ort_api;
    kernel_state->compiled_op = std::move(compiled_op);
    kernel_state->persistent_resource = std::move(persistent_resource);
    kernel_state->persistent_allocator = std::move(persistent_allocator);
    kernel_state->persistent_binding = persistent_binding;

    // The Compute path maps ORT KernelContext input indices (subgraph ordering,
    // 0..num_graph_inputs-1) to DML graph input indices (renumbered, excluding
    // inlined constants). subgraph_to_dml_input stores this mapping.
    kernel_state->num_runtime_inputs = im.total_dml_inputs;
    kernel_state->num_subgraph_inputs = num_graph_inputs;
    kernel_state->subgraph_to_dml_input = im.subgraph_to_dml_input;
    kernel_state->dml_inputs_with_edges = dml_inputs_with_edges;
    kernel_state->runtime_input_bytes.resize(im.total_dml_inputs);
    kernel_state->runtime_input_is_owned.resize(im.total_dml_inputs, false);
    kernel_state->num_initializers = 0;

    // Mode A: populate from subgraph graph inputs (runtime + initializers).
    for (size_t i = 0; i < num_graph_inputs; ++i) {
        if (im.subgraph_to_dml_input[i] == SIZE_MAX) continue;
        size_t dml_idx = im.subgraph_to_dml_input[i];
        std::string name = ReadValueInfoName(ort_api, sg.graph_input_vis[i]);
        auto it = sg.value_shapes.find(name);
        kernel_state->runtime_input_bytes[dml_idx] =
            (it != sg.value_shapes.end()) ? it->second.total_bytes : 0;
        if (owned_graph_input_indices.count(dml_idx))
            kernel_state->runtime_input_is_owned[dml_idx] = true;
    }

    // Mode B: populate for initializers added as extra DML inputs.
    // These are always OWNED (large initializers baked into persistent resource).
    for (const auto& name : im.ordered_initializer_names) {
        auto di_it = im.dml_input_map.find(name);
        if (di_it == im.dml_input_map.end()) continue;
        size_t dml_idx = di_it->second;
        auto it = sg.value_shapes.find(name);
        kernel_state->runtime_input_bytes[dml_idx] =
            (it != sg.value_shapes.end()) ? it->second.total_bytes : 0;
        if (owned_graph_input_indices.count(dml_idx))
            kernel_state->runtime_input_is_owned[dml_idx] = true;
    }

    kernel_state->num_outputs = num_graph_outputs;
    kernel_state->output_dims.resize(num_graph_outputs);
    kernel_state->output_bytes.resize(num_graph_outputs);

    for (size_t i = 0; i < num_graph_outputs; ++i) {
        std::string name = ReadValueInfoName(ort_api, sg.graph_output_vis[i]);
        auto it = sg.value_shapes.find(name);
        if (it == sg.value_shapes.end()) continue;

        auto prod_it = value_producer.find(name);
        if (prod_it != value_producer.end()) {
            size_t prod_node = prod_it->second.first;
            size_t prod_slot = prod_it->second.second;
            if (prod_slot < compiled_nodes[prod_node].translated->output_tensors.size()) {
                const auto& out_info = compiled_nodes[prod_node].translated->output_tensors[prod_slot];
                // Strip leading 1s added by PadToMinDims; use the graph output's
                // original rank from shape inference as the authoritative rank.
                const auto& padded = out_info.sizes;
                const OrtValueInfo* out_vi = sg.graph_output_vis[i];
                size_t orig_rank = 0;
                if (out_vi && out_vi->GetTypeInfo() && out_vi->GetTypeInfo()->tensor_type_info
                    && out_vi->GetTypeInfo()->tensor_type_info->HasShape()) {
                    ort_api.GetDimensionsCount(out_vi->GetTypeInfo()->tensor_type_info.get(), &orig_rank);
                }
                if (orig_rank == 0) orig_rank = padded.size();
                size_t skip = padded.size() > orig_rank ? padded.size() - orig_rank : 0;
                kernel_state->output_dims[i].reserve(padded.size() - skip);
                for (size_t d = skip; d < padded.size(); ++d)
                    kernel_state->output_dims[i].push_back(static_cast<int64_t>(padded[d]));
                kernel_state->output_bytes[i] = out_info.total_bytes;
                continue;
            }
        }
        kernel_state->output_bytes[i] = it->second.total_bytes;
    }

    auto* info = new FullGraphNodeComputeInfo();
    info->state = std::move(kernel_state);
    return info;
}

bool FullGraphFusion::TryCompileGraph(
    const OrtApi&                                            ort_api,
    const OrtGraph*                                          main_graph,
    const std::unordered_map<std::string, const OrtValue*>&  initializers,
    PluginDmlExecutionProviderImpl*                          provider)
{
    // Feasibility check: attempt the full Compile pipeline against the main
    // graph. If CompileGraph succeeds, discard the result and return true.
    // If CompileGraph fails, Compile returns nullptr and we return false.
    //
    // The caller (GetCapabilityImpl) uses this to decide whether to claim
    // nodes via AddNodesToFuse. If false, nodes are not claimed and fall
    // through to Tier-1/2 per-node execution without ORT_EP_FAIL.
    //
    // Cost: one full translate + CreateOperator × N + CompileGraph at init.
    // After init this never runs again. The authoritative compile happens
    // in CompileImpl against the fused subgraph (correct input ordering).
    OrtNodeComputeInfo* info = Compile(ort_api, main_graph, initializers, provider);
    if (info != nullptr) {
        delete info; // discard — CompileImpl recompiles against fused subgraph
        return true;
    }
    return false;
}

}  // namespace dml_ep
