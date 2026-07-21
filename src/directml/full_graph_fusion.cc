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
#include "dml_perf_timer.h"

namespace dml_ep {

static void DiagLog([[maybe_unused]] std::string_view msg) noexcept {
#ifdef DML_PERF_PROFILE
    DmlPerfWriteLogImpl(msg);
#endif
}

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

#ifdef DML_PERF_PROFILE
    uint64_t _perf_inf_id = state->compute_call_count;
    uint64_t _perf_t_enter = PerfNowUs();
    uint64_t _perf_t_last = _perf_t_enter;
    DML_PERF_LOG("[PERF] INF#", _perf_inf_id, " FullGraph_Compute ENTER: ", _perf_t_enter, " us\n");
#endif

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

#ifdef DML_PERF_PROFILE
    { uint64_t _t = PerfNowUs(); DML_PERF_LOG("[PERF] INF#", _perf_inf_id, " FullGraph_Compute inputs_bound: ", _t, " us (+", _t - _perf_t_last, ")\n"); _perf_t_last = _t; }
#endif

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

#ifdef DML_PERF_PROFILE
    { uint64_t _t = PerfNowUs(); DML_PERF_LOG("[PERF] INF#", _perf_inf_id, " FullGraph_Compute outputs_bound: ", _t, " us (+", _t - _perf_t_last, ")\n"); _perf_t_last = _t; }
#endif

    const DML_BUFFER_BINDING* persistent = state->persistent_binding
        ? &*state->persistent_binding : nullptr;

    HRESULT hr = state->provider->ExecuteOperator(
        state->compiled_op.Get(),
        persistent,
        gsl::make_span(input_descs),
        gsl::make_span(output_descs));

    if (FAILED(hr))
        return api.CreateStatus(ORT_FAIL, "Tier0: ExecuteOperator failed");

#ifdef DML_PERF_PROFILE
    { uint64_t _t = PerfNowUs(); DML_PERF_LOG("[PERF] INF#", _perf_inf_id, " FullGraph_Compute ExecuteOperator_done: ", _t, " us (+", _t - _perf_t_last, ")\n"); _perf_t_last = _t; }
#endif

    state->provider->QueueReference(state->compiled_op.Get());

#ifdef DML_PERF_PROFILE
    { uint64_t _t = PerfNowUs(); DML_PERF_LOG("[PERF] INF#", _perf_inf_id, " FullGraph_Compute EXIT: ", _t, " us (+", _t - _perf_t_enter, " total)\n"); }
#endif

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
    const std::vector<const OrtNode*>&                       nodes,
    const std::unordered_map<std::string, std::vector<int64_t>>& resolved_shapes)
{
    // Helper: get tensor name from a ValueInfo.
    auto GetViName = [&](const OrtValueInfo* vi) -> std::string {
        if (!vi) return {};
        const char* n = nullptr;
        ort_api.GetValueInfoName(vi, &n);
        return n ? std::string(n) : std::string{};
    };

    // Helper: check resolved_shapes fallback for a VI that ORT reports as dynamic.
    // Returns true if resolved_shapes contains all-static dims for this tensor.
    auto CheckResolvedFallback = [&](const OrtValueInfo* vi) -> bool {
        auto name = GetViName(vi);
        if (name.empty()) return false;
        auto it = resolved_shapes.find(name);
        if (it == resolved_shapes.end()) {
            DiagLog("[ValidateTier0] resolved_shapes MISS: '" + name + "' (map has " + std::to_string(resolved_shapes.size()) + " entries)\n");
            return false;
        }
        for (int64_t d : it->second) if (d < 0) return false;
        DiagLog("[ValidateTier0] resolved_shapes fallback: '" + name + "' accepted\n");
        return true;
    };

    // Helper: get op type for a node (for diagnostics).
    auto GetOpType = [&](const OrtNode* node) -> std::string {
        const char* op = nullptr;
        ort_api.Node_GetOperatorType(node, &op);
        return op ? std::string(op) : "?";
    };

    // Validate dtype support and reject known-dynamic shapes.
    //
    // Consumer-side OrtValueInfo* often lacks shape info entirely (!HasShape()
    // or !HasTypeInfo()) because producer and consumer VIs are different objects.
    // Missing info does NOT mean dynamic — it means ORT didn't propagate it to
    // this particular VI. We skip those tensors and let TryTranslateNodes decide.
    //
    // We only reject when ORT positively reports dynamic dims (HasShape() true
    // AND dims contain -1) and resolved_shapes doesn't cover it.
    for (const OrtNode* node : nodes) {
        if (!node) continue;
        size_t node_num_inputs = 0;
        ort_api.Node_GetNumInputs(node, &node_num_inputs);
        std::vector<const OrtValueInfo*> in_vis(node_num_inputs, nullptr);
        if (node_num_inputs > 0)
            ort_api.Node_GetInputs(node, in_vis.data(), node_num_inputs);
        auto input_names = fusion_utils::GetNodeInputNames(ort_api, node);
        for (size_t k = 0; k < node_num_inputs && k < input_names.size(); ++k) {
            if (in_vis[k] == nullptr || input_names[k].empty()) continue;
            auto* ti = in_vis[k]->GetTypeInfo();
            if (ti == nullptr || ti->tensor_type_info == nullptr) continue;
            const OrtTensorTypeAndShapeInfo* si = ti->tensor_type_info.get();
            if (si == nullptr) continue;
            ONNXTensorElementDataType onnx_dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            ort_api.GetTensorElementType(si, &onnx_dt);
            if (OnnxDtypeToDml(onnx_dt) == DML_TENSOR_DATA_TYPE_UNKNOWN) {
                DiagLog("[ValidateTier0] FAIL input: op=" + GetOpType(node) + " tensor='" + GetViName(in_vis[k]) + "' unsupported dtype=" + std::to_string(static_cast<int>(onnx_dt)) + "\n");
                return false;
            }
            if (!si->HasShape()) continue;
            size_t in_rank = 0;
            ort_api.GetDimensionsCount(si, &in_rank);
            if (in_rank > 0) {
                std::vector<int64_t> in_dims(in_rank, -1);
                ort_api.GetDimensions(si, in_dims.data(), in_rank);
                bool is_empty = false;
                for (size_t d = 0; d < in_rank; ++d)
                    if (in_dims[d] == 0) { is_empty = true; break; }
                if (!is_empty) {
                    bool has_dynamic = false;
                    for (size_t d = 0; d < in_rank; ++d)
                        if (in_dims[d] < 0) { has_dynamic = true; break; }
                    if (has_dynamic && !CheckResolvedFallback(in_vis[k])) {
                        DiagLog("[ValidateTier0] FAIL input: op=" + GetOpType(node) + " tensor='" + GetViName(in_vis[k]) + "' dynamic dims\n");
                        return false;
                    }
                }
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
    const std::unordered_map<std::string, const OrtValue*>& initializers,
    const std::unordered_map<std::string, std::vector<int64_t>>& resolved_shapes = {})
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

    // Enumerate subgraph initializers. These OrtValue pointers come from the fused subgraph
    // and are valid for the lifetime of CompileImpl. The parent initializer map (passed in as
    // `initializers` / m_graphInitializerMap) holds pointers that were valid during
    // GetCapabilityImpl but are freed by ORT when GetCapability returns — do NOT dereference
    // those pointers here. We start all_initializers from the parent map for name-lookup
    // purposes only (e.g. consumed_initializer_names tracking), but overwrite every entry
    // that has a live subgraph pointer before any OrtValue is dereferenced.
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

                // Seed value_shapes from this initializer while we have a fresh, valid OrtValue.
                // Only allocated OrtValues have valid type info (external-data initializers are
                // stored unallocated until lazy-loaded; their shapes come from ORT shape inference).
                if (!val || !val->IsAllocated()) continue;
                if (info.value_shapes.count(name)) continue;

                OrtTensorTypeAndShapeInfo* shape_info = nullptr;
                ort_api.GetTensorTypeAndShape(const_cast<OrtValue*>(val), &shape_info);
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
                info.value_shapes[name] = tensor_info;
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

    // Pre-seed all node output shapes from ORT shape inference.
    // Tier-0 requires all-static shapes, so every node output should have known dims.
    size_t num_nodes = 0;
    ort_api.Graph_GetNumNodes(fused_subgraph, &num_nodes);
    std::vector<const OrtNode*> subgraph_nodes(num_nodes, nullptr);
    if (num_nodes > 0)
        ort_api.Graph_GetNodes(fused_subgraph, subgraph_nodes.data(), num_nodes);

    for (const OrtNode* node : subgraph_nodes) {
        if (!node) continue;
        const char* node_op = nullptr;
        ort_api.Node_GetOperatorType(node, &node_op);
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
            if (!ti || !ti->tensor_type_info) {
                DML_PERF_LOG("[BuildSubgraphInfo] SKIP '", output_names[k], "' op=", (node_op ? node_op : "?"), " reason=no_typeinfo\n");
                continue;
            }
            const OrtTensorTypeAndShapeInfo* si = ti->tensor_type_info.get();
            if (!si->HasShape()) {
                DML_PERF_LOG("[BuildSubgraphInfo] SKIP '", output_names[k], "' op=", (node_op ? node_op : "?"), " reason=no_shape\n");
                continue;
            }
            ONNXTensorElementDataType onnx_dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            ort_api.GetTensorElementType(si, &onnx_dt);
            DML_TENSOR_DATA_TYPE dml_dt = OnnxDtypeToDml(onnx_dt);
            if (dml_dt == DML_TENSOR_DATA_TYPE_UNKNOWN) {
                DML_PERF_LOG("[BuildSubgraphInfo] SKIP '", output_names[k], "' op=", (node_op ? node_op : "?"), " reason=unsupported_dtype\n");
                continue;
            }
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
                if (!all_static) {
                    std::string dims_str = "[";
                    for (size_t d = 0; d < out_rank; ++d) { if (d) dims_str += ","; dims_str += std::to_string(out_dims[d]); }
                    dims_str += "]";
                    DML_PERF_LOG("[BuildSubgraphInfo] SKIP '", output_names[k], "' op=", (node_op ? node_op : "?"), " reason=dynamic_dims ", dims_str, "\n");
                }
            }
            if (all_static) {
                auto tensor_info = MakeTensorInfo(out_sizes, dml_dt);
                tensor_info.original_rank = static_cast<uint32_t>(out_rank);
                info.value_shapes[output_names[k]] = tensor_info;
            }
        }
    }

    // Also seed from node input VIs. Producer and consumer VIs are different
    // objects and may carry different names for the same tensor edge. The output
    // scan above seeds under the producer's name; this scan seeds under the
    // consumer's name so translators (which use GetNodeInputNames) can find it.
    for (const OrtNode* node : subgraph_nodes) {
        if (!node) continue;
        auto input_names_scan = fusion_utils::GetNodeInputNames(ort_api, node);
        size_t node_num_inputs_scan = 0;
        ort_api.Node_GetNumInputs(node, &node_num_inputs_scan);
        std::vector<const OrtValueInfo*> in_vis_scan(node_num_inputs_scan, nullptr);
        if (node_num_inputs_scan > 0)
            ort_api.Node_GetInputs(node, in_vis_scan.data(), node_num_inputs_scan);
        for (size_t k = 0; k < node_num_inputs_scan && k < input_names_scan.size(); ++k) {
            if (input_names_scan[k].empty()) continue;
            if (info.value_shapes.count(input_names_scan[k])) continue;
            if (!in_vis_scan[k]) continue;
            auto* ti = in_vis_scan[k]->GetTypeInfo();
            if (!ti || !ti->tensor_type_info) continue;
            const OrtTensorTypeAndShapeInfo* si = ti->tensor_type_info.get();
            if (!si->HasShape()) continue;
            ONNXTensorElementDataType onnx_dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            ort_api.GetTensorElementType(si, &onnx_dt);
            DML_TENSOR_DATA_TYPE dml_dt = OnnxDtypeToDml(onnx_dt);
            if (dml_dt == DML_TENSOR_DATA_TYPE_UNKNOWN) continue;
            size_t in_rank = 0;
            ort_api.GetDimensionsCount(si, &in_rank);
            std::vector<uint32_t> in_sizes;
            bool all_static = true;
            if (in_rank > 0) {
                std::vector<int64_t> in_dims(in_rank, -1);
                ort_api.GetDimensions(si, in_dims.data(), in_rank);
                in_sizes.resize(in_rank);
                for (size_t d = 0; d < in_rank; ++d) {
                    if (in_dims[d] <= 0) { all_static = false; break; }
                    in_sizes[d] = static_cast<uint32_t>(in_dims[d]);
                }
            }
            if (all_static) {
                auto tensor_info = MakeTensorInfo(in_sizes, dml_dt);
                tensor_info.original_rank = static_cast<uint32_t>(in_rank);
                info.value_shapes[input_names_scan[k]] = tensor_info;
            }
        }
    }

    // Seed value_shapes from resolved_shapes for tensors ORT left dynamic.
    // These were computed in GetCapabilityImpl (e.g. Upsample output shapes).
    // The dtype comes from ORT's type info (which is present even when dims are
    // unknown); only the dims are missing. We look up each node's outputs and
    // graph inputs to find VIs with matching names.
    if (!resolved_shapes.empty()) {
        auto SeedFromVi = [&](const OrtValueInfo* vi, const std::string& name,
                              const std::vector<int64_t>& dims) {
            if (info.value_shapes.count(name)) return;
            if (!vi) return;
            auto* ti = vi->GetTypeInfo();
            if (!ti || !ti->tensor_type_info) return;
            const OrtTensorTypeAndShapeInfo* si = ti->tensor_type_info.get();
            if (!si) return;
            ONNXTensorElementDataType onnx_dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            ort_api.GetTensorElementType(si, &onnx_dt);
            DML_TENSOR_DATA_TYPE dml_dt = OnnxDtypeToDml(onnx_dt);
            if (dml_dt == DML_TENSOR_DATA_TYPE_UNKNOWN) return;
            std::vector<uint32_t> sizes(dims.size());
            for (size_t d = 0; d < dims.size(); ++d)
                sizes[d] = static_cast<uint32_t>(dims[d] > 0 ? dims[d] : 1);
            auto tensor_info = MakeTensorInfo(sizes, dml_dt);
            tensor_info.original_rank = static_cast<uint32_t>(dims.size());
            info.value_shapes[name] = tensor_info;
            DML_PERF_LOG("[BuildSubgraphInfo] seeded from resolved_shapes: '", name, "'\n");
        };

        // Scan graph inputs.
        for (size_t i = 0; i < num_graph_inputs; ++i) {
            std::string name = ReadValueInfoName(ort_api, info.graph_input_vis[i]);
            auto it = resolved_shapes.find(name);
            if (it != resolved_shapes.end())
                SeedFromVi(info.graph_input_vis[i], name, it->second);
        }
        // Scan node outputs.
        for (const OrtNode* node : subgraph_nodes) {
            if (!node) continue;
            auto output_names_local = fusion_utils::GetNodeOutputNames(ort_api, node);
            size_t node_num_outputs_local = 0;
            ort_api.Node_GetNumOutputs(node, &node_num_outputs_local);
            std::vector<const OrtValueInfo*> out_vis_local(node_num_outputs_local, nullptr);
            if (node_num_outputs_local > 0)
                ort_api.Node_GetOutputs(node, out_vis_local.data(), node_num_outputs_local);
            for (size_t k = 0; k < node_num_outputs_local && k < output_names_local.size(); ++k) {
                auto it = resolved_shapes.find(output_names_local[k]);
                if (it != resolved_shapes.end())
                    SeedFromVi(out_vis_local[k], output_names_local[k], it->second);
            }
        }
        // Also scan node inputs — consumer VIs may have dtype but no shape.
        for (const OrtNode* node : subgraph_nodes) {
            if (!node) continue;
            auto input_names_local = fusion_utils::GetNodeInputNames(ort_api, node);
            size_t node_num_inputs_local = 0;
            ort_api.Node_GetNumInputs(node, &node_num_inputs_local);
            std::vector<const OrtValueInfo*> in_vis_local(node_num_inputs_local, nullptr);
            if (node_num_inputs_local > 0)
                ort_api.Node_GetInputs(node, in_vis_local.data(), node_num_inputs_local);
            for (size_t k = 0; k < node_num_inputs_local && k < input_names_local.size(); ++k) {
                if (input_names_local[k].empty()) continue;
                auto it = resolved_shapes.find(input_names_local[k]);
                if (it != resolved_shapes.end())
                    SeedFromVi(in_vis_local[k], input_names_local[k], it->second);
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
    PluginDmlExecutionProviderImpl*                          provider,
    const std::unordered_map<std::string, std::vector<int64_t>>& resolved_shapes)
{
    // -----------------------------------------------------------------------
    // Step 1–3: Enumerate subgraph I/O, merge initializers, pre-seed shapes.
    // -----------------------------------------------------------------------

    SubgraphInfo sg = BuildSubgraphInfo(ort_api, fused_subgraph, initializers, resolved_shapes);
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
    std::unordered_set<std::string> dml_consumed_names;

    size_t num_nodes = 0;
    ort_api.Graph_GetNumNodes(fused_subgraph, &num_nodes);
    std::vector<const OrtNode*> subgraph_nodes(num_nodes, nullptr);
    if (num_nodes > 0)
        ort_api.Graph_GetNodes(fused_subgraph, subgraph_nodes.data(), num_nodes);

    for (const OrtNode* node : subgraph_nodes) {
        if (!node) continue;

        const char* op_type = nullptr;
        ort_api.Node_GetOperatorType(node, &op_type);
        if (!op_type) {
            DML_PERF_LOG("[Compile] FAIL: null op_type\n");
            return nullptr;
        }

        auto reg_it = registry.find(op_type);
        if (reg_it == registry.end()) {
            DML_PERF_LOG("[Compile] FAIL: no translator for op=", op_type, "\n");
            return nullptr;
        }

        auto input_names  = fusion_utils::GetNodeInputNames(ort_api, node);
        auto output_names = fusion_utils::GetNodeOutputNames(ort_api, node);

        DML_PERF_LOG("[Compile] translating op=", op_type,
            "  inputs=[", [&]{ std::string s; for (auto& n : input_names) s += n + ","; return s; }(),
            "]  outputs=[", [&]{ std::string s; for (auto& n : output_names) s += n + ","; return s; }(), "]\n");

        auto translated = reg_it->second(ort_api, node, sg.value_shapes, sg.all_initializers);
        if (!translated) {
            DML_PERF_LOG("[Compile] FAIL: translator returned nullopt for op=", op_type, "\n");
            return nullptr;
        }

        DML_PERF_LOG("[Compile] OK: op=", op_type, "\n");

        // Write-back: translator computed output shapes; seed them for downstream.
        for (size_t k = 0; k < output_names.size() && k < translated->output_tensors.size(); ++k) {
            if (!sg.value_shapes.count(output_names[k])) {
                sg.value_shapes[output_names[k]] = translated->output_tensors[k];
                auto& t = translated->output_tensors[k];
                DML_PERF_LOG("[Compile] write-back '", output_names[k], "' sizes=[");
                for (size_t d = 0; d < t.sizes.size(); ++d) DML_PERF_LOG(d>0?",":"", t.sizes[d]);
                DML_PERF_LOG("] dtype=", static_cast<int>(t.data_type), "\n");
            }
        }

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

        // Track which input names will become actual DML edges. The DML operator
        // only wires input_tensors.size() primary inputs (plus sub_node graph_inputs).
        // Inputs beyond that limit (e.g. Clip's min/max at slots 1,2) are consumed
        // at translation time only and won't have edges in the DML graph.
        size_t dml_input_count = translated_ptr->input_tensors.size();
        size_t npi = translated_ptr->primary_input_count > 0
            ? translated_ptr->primary_input_count : dml_input_count;
        for (size_t s = 0; s < input_names.size() && s < npi; ++s) {
            size_t name_idx = translated_ptr->input_name_reorder.empty()
                ? s : translated_ptr->input_name_reorder[s];
            if (name_idx >= input_names.size()) continue;
            const auto& in_name = input_names[name_idx];
            dml_consumed_names.insert(in_name);
            if (sg.all_initializers.count(in_name))
                consumed_initializer_names.insert(in_name);
        }
        for (const auto& sn : translated_ptr->sub_nodes) {
            for (const auto& [onnx_idx, _] : sn.graph_inputs) {
                if (onnx_idx < input_names.size())
                    dml_consumed_names.insert(input_names[onnx_idx]);
            }
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

    // Strip dead compiled_nodes: nodes whose outputs are never consumed as a
    // DML edge. dml_consumed_names was built during translation with the same
    // input-count limits as edge wiring, so translation-time-only inputs (e.g.
    // Clip's min/max) are already excluded.
    //
    // Algorithm: count how many dml_consumed_names reference each node (via
    // value_producer). Nodes with refcount 0 are dead leaves. Killing a leaf
    // decrements its input producers' refcounts — if a producer hits 0, it
    // cascades. Single O(N) pass via work queue.
    std::vector<bool> node_is_live(compiled_nodes.size(), true);
    {
        for (const auto& [name, _] : sg.graph_output_map)
            dml_consumed_names.insert(name);

        // Refcount: how many names in dml_consumed_names each node produces.
        std::vector<size_t> refcount(compiled_nodes.size(), 0);
        for (const auto& [name, prod] : value_producer) {
            if (dml_consumed_names.count(name))
                ++refcount[prod.first];
        }

        // Collect wired input names per node (for decrementing producers on death).
        std::vector<std::vector<std::string>> wired_inputs(compiled_nodes.size());
        for (size_t i = 0; i < compiled_nodes.size(); ++i) {
            const auto& cn = compiled_nodes[i];
            size_t npi = cn.translated->primary_input_count > 0
                ? cn.translated->primary_input_count
                : cn.translated->input_tensors.size();
            for (size_t s = 0; s < cn.input_names.size() && s < npi; ++s) {
                size_t ni = cn.translated->input_name_reorder.empty()
                    ? s : cn.translated->input_name_reorder[s];
                if (ni < cn.input_names.size())
                    wired_inputs[i].push_back(cn.input_names[ni]);
            }
            for (const auto& sn : cn.translated->sub_nodes) {
                for (const auto& [onnx_idx, _] : sn.graph_inputs) {
                    if (onnx_idx < cn.input_names.size())
                        wired_inputs[i].push_back(cn.input_names[onnx_idx]);
                }
            }
        }

        // Seed work queue with all zero-refcount nodes.
        std::vector<size_t> dead_queue;
        for (size_t i = 0; i < compiled_nodes.size(); ++i) {
            if (refcount[i] == 0)
                dead_queue.push_back(i);
        }

        // Process: mark dead, decrement upstream producers, cascade.
        for (size_t qi = 0; qi < dead_queue.size(); ++qi) {
            size_t di = dead_queue[qi];
            node_is_live[di] = false;
            DML_PERF_LOG("[Compile] dead node: compiled_nodes[", di,
                "] op=", compiled_nodes[di].op_type, "\n");
            for (const auto& inp_name : wired_inputs[di]) {
                auto it = value_producer.find(inp_name);
                if (it != value_producer.end()) {
                    size_t prod_idx = it->second.first;
                    if (node_is_live[prod_idx] && --refcount[prod_idx] == 0)
                        dead_queue.push_back(prod_idx);
                }
            }
        }
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
    for (size_t ci = 0; ci < compiled_nodes.size(); ++ci) {
        if (!node_is_live[ci]) continue;
        auto& cn = compiled_nodes[ci];
        size_t dml_input_count = cn.translated->input_tensors.size();
        for (size_t s = 0; s < cn.input_names.size() && s < dml_input_count; ++s) {
            size_t name_idx = cn.translated->input_name_reorder.empty()
                ? s : cn.translated->input_name_reorder[s];
            if (name_idx >= cn.input_names.size()) continue;
            auto resolved = cn.input_names[name_idx];
            while (graph_input_aliases.count(resolved))
                resolved = graph_input_aliases[resolved];
            if (!sg.all_initializers.count(cn.input_names[name_idx]) && !sg.all_initializers.count(resolved))
                continue;

            auto shape_it = sg.value_shapes.find(
                sg.all_initializers.count(resolved) ? resolved : cn.input_names[name_idx]);
            uint64_t bytes = (shape_it != sg.value_shapes.end()) ? shape_it->second.total_bytes : 0;
            if (bytes >= kMaxConstNodeDataSize) {
                cn.translated->input_buffer_descs[s].Flags |= DML_TENSOR_FLAG_OWNED_BY_DML;
                ++owned_count;
            }
        }
        cn.translated->FixupPointers();
    }
    (void)owned_count;

    // Filter consumed_initializer_names: exclude initializers consumed only by
    // passthrough or dead nodes. Such initializers would create constant nodes
    // with no consumer edge in the DML graph, causing orphaned-node errors.
    {
        std::unordered_set<std::string> live_inputs;
        for (size_t i = 0; i < compiled_nodes.size(); ++i) {
            if (!node_is_live[i]) continue;
            for (auto name : compiled_nodes[i].input_names) {
                while (graph_input_aliases.count(name)) name = graph_input_aliases[name];
                live_inputs.insert(name);
            }
        }
        std::unordered_set<std::string> filtered_consumed;
        for (const auto& name : consumed_initializer_names) {
            if (live_inputs.count(name))
                filtered_consumed.insert(name);
        }
        consumed_initializer_names = std::move(filtered_consumed);
    }

    // -----------------------------------------------------------------------
    // Step 6: Create IDMLOperator for each compiled node.
    // -----------------------------------------------------------------------

    ComPtr<IDMLDevice> dml_device;
    if (FAILED(provider->GetDmlDevice(dml_device.GetAddressOf()))) return nullptr;
    ComPtr<IDMLDevice1> dml_device1;
    if (FAILED(dml_device.As(&dml_device1))) return nullptr;

    for (size_t i = 0; i < compiled_nodes.size(); ++i) {
        if (!node_is_live[i]) continue;
        auto& cn = compiled_nodes[i];
        HRESULT hr = dml_device->CreateOperator(
            &cn.translated->op_desc, IID_PPV_ARGS(cn.translated->dml_operator.GetAddressOf()));
        if (FAILED(hr)) {
            DML_PERF_LOG("[Compile] FAIL: CreateOperator HR=", Hex(static_cast<uint32_t>(hr)),
                " op=", cn.op_type, "\n");
            return nullptr;
        }
        DML_PERF_LOG("[Compile] CreateOperator OK: op=", cn.op_type, "\n");

        for (size_t s = 0; s < cn.translated->sub_nodes.size(); ++s) {
            auto& sn = cn.translated->sub_nodes[s];
            hr = dml_device->CreateOperator(
                &sn.op_desc, IID_PPV_ARGS(sn.dml_operator.GetAddressOf()));
            if (FAILED(hr)) {
                DML_PERF_LOG("[Compile] FAIL: CreateOperator sub_node[", s, "] HR=",
                    Hex(static_cast<uint32_t>(hr)), " op=", cn.op_type, "\n");
                return nullptr;
            }
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
    std::vector<size_t> dml_node_offset(compiled_nodes.size(), SIZE_MAX);
    for (size_t i = 0; i < compiled_nodes.size(); ++i) {
        if (!node_is_live[i]) continue;
        dml_node_offset[i] = total_dml_nodes;
        total_dml_nodes += 1 + compiled_nodes[i].translated->sub_nodes.size();
    }

    std::vector<DML_OPERATOR_GRAPH_NODE_DESC> op_node_descs(total_dml_nodes);
    std::vector<DML_GRAPH_NODE_DESC> graph_nodes(total_dml_nodes);
    for (size_t i = 0; i < compiled_nodes.size(); ++i) {
        if (!node_is_live[i]) continue;
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
        if (!node_is_live[node_idx]) continue;
        const auto& cn = compiled_nodes[node_idx];
        size_t primary_dml_idx = dml_node_offset[node_idx];

        // Only wire edges for as many inputs as the primary DML operator has.
        // When primary_input_count > 0, only that many inputs go to the primary
        // node; remaining inputs are available for sub_node graph_inputs wiring.
        size_t num_primary_inputs = cn.translated->primary_input_count > 0
            ? cn.translated->primary_input_count : cn.translated->input_tensors.size();
        for (size_t input_slot = 0; input_slot < cn.input_names.size() && input_slot < num_primary_inputs; ++input_slot) {
            size_t name_idx = cn.translated->input_name_reorder.empty()
                ? input_slot : cn.translated->input_name_reorder[input_slot];
            if (name_idx >= cn.input_names.size()) continue;
            auto name = cn.input_names[name_idx];
            if (name.empty()) continue;

            // Resolve the DML schema slot for this edge. For dense operators the
            // packed input_slot equals the schema slot. For sparse operators (e.g.
            // MultiHeadAttention with non-contiguous slots) dml_input_slot_indices
            // maps packed position → schema slot so ToNodeInputIndex is correct.
            const size_t dml_schema_slot = cn.translated->dml_input_slot_indices.empty()
                ? input_slot : cn.translated->dml_input_slot_indices[input_slot];

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
                edge.ToNodeInputIndex = static_cast<UINT>(dml_schema_slot);
                intermediate_edge_storage.push_back(edge);
                continue;
            }

            // DML graph input (runtime tensor or large initializer).
            auto dml_in_it = im.dml_input_map.find(name);
            if (dml_in_it != im.dml_input_map.end()) {
                DML_INPUT_GRAPH_EDGE_DESC edge{};
                edge.GraphInputIndex = static_cast<UINT>(dml_in_it->second);
                edge.ToNodeIndex = static_cast<UINT>(primary_dml_idx);
                edge.ToNodeInputIndex = static_cast<UINT>(dml_schema_slot);
                input_edge_storage.push_back(edge);
                continue;
            }

            // Produced by a prior node.
            auto prod_it = value_producer.find(name);
            if (prod_it != value_producer.end()) {
                size_t prod_compiled_idx = prod_it->second.first;
                size_t prod_dml_idx = dml_node_offset[prod_compiled_idx];
                size_t prod_output_slot = prod_it->second.second;
                UINT from_output_index;

                const auto& osrc = compiled_nodes[prod_compiled_idx].translated->output_source;
                if (!osrc.empty() && prod_output_slot < osrc.size()) {
                    auto [src_sub, src_slot] = osrc[prod_output_slot];
                    if (src_sub >= 0)
                        prod_dml_idx += 1 + static_cast<size_t>(src_sub);
                    from_output_index = static_cast<UINT>(src_slot);
                } else {
                    size_t num_subs = compiled_nodes[prod_compiled_idx].translated->sub_nodes.size();
                    if (num_subs > 0)
                        prod_dml_idx += num_subs;
                    from_output_index = static_cast<UINT>(prod_output_slot);
                }

                DML_INTERMEDIATE_GRAPH_EDGE_DESC edge{};
                edge.FromNodeIndex = static_cast<UINT>(prod_dml_idx);
                edge.FromNodeOutputIndex = from_output_index;
                edge.ToNodeIndex = static_cast<UINT>(primary_dml_idx);
                edge.ToNodeInputIndex = static_cast<UINT>(dml_schema_slot);
                intermediate_edge_storage.push_back(edge);
                continue;
            }

            DML_PERF_LOG("[Compile] FAIL: unresolved input '", name,
                "' for op=", cn.op_type, " slot=", input_slot, "\n");
            return nullptr; // unresolved input
        }

        // Wire internal edges for sub_nodes.
        for (size_t s = 0; s < cn.translated->sub_nodes.size(); ++s) {
            const auto& sn = cn.translated->sub_nodes[s];
            size_t sn_dml_idx = primary_dml_idx + 1 + s;
            for (size_t inp = 0; inp < sn.input_from.size(); ++inp) {
                auto [src_sub, src_slot] = sn.input_from[inp];
                if (src_sub < -1) continue;  // skip sentinel: slot wired by graph_inputs
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

            // Wire DML graph inputs directly to this sub_node.
            for (auto& [onnx_idx, to_input] : sn.graph_inputs) {
                if (onnx_idx >= cn.input_names.size()) continue;
                auto gi_name = cn.input_names[onnx_idx];
                while (graph_input_aliases.count(gi_name))
                    gi_name = graph_input_aliases[gi_name];

                auto const_it = im.constant_node_map.find(gi_name);
                if (const_it != im.constant_node_map.end()) {
                    DML_INTERMEDIATE_GRAPH_EDGE_DESC edge{};
                    edge.FromNodeIndex = static_cast<UINT>(const_node_base + const_it->second);
                    edge.FromNodeOutputIndex = 0;
                    edge.ToNodeIndex = static_cast<UINT>(sn_dml_idx);
                    edge.ToNodeInputIndex = static_cast<UINT>(to_input);
                    intermediate_edge_storage.push_back(edge);
                    continue;
                }
                auto dml_in_it = im.dml_input_map.find(gi_name);
                if (dml_in_it != im.dml_input_map.end()) {
                    DML_INPUT_GRAPH_EDGE_DESC edge{};
                    edge.GraphInputIndex = static_cast<UINT>(dml_in_it->second);
                    edge.ToNodeIndex = static_cast<UINT>(sn_dml_idx);
                    edge.ToNodeInputIndex = static_cast<UINT>(to_input);
                    input_edge_storage.push_back(edge);
                }
            }
        }
    }

    // Output edges.
    for (const auto& [name, out_idx] : sg.graph_output_map) {
        auto resolved = name;
        while (graph_input_aliases.count(resolved))
            resolved = graph_input_aliases[resolved];
        auto prod_it = value_producer.find(resolved);
        if (prod_it == value_producer.end()) {
            DML_PERF_LOG("[Compile] FAIL: unresolved output '", name, "' (resolved='", resolved, "')\n");
            return nullptr;
        }

        size_t prod_compiled_idx = prod_it->second.first;
        size_t prod_dml_idx = dml_node_offset[prod_compiled_idx];
        size_t prod_output_slot = prod_it->second.second;
        UINT from_output_index;

        const auto& osrc = compiled_nodes[prod_compiled_idx].translated->output_source;
        if (!osrc.empty() && prod_output_slot < osrc.size()) {
            auto [src_sub, src_slot] = osrc[prod_output_slot];
            if (src_sub >= 0)
                prod_dml_idx += 1 + static_cast<size_t>(src_sub);
            from_output_index = static_cast<UINT>(src_slot);
        } else {
            size_t num_subs = compiled_nodes[prod_compiled_idx].translated->sub_nodes.size();
            if (num_subs > 0)
                prod_dml_idx += num_subs;
            from_output_index = static_cast<UINT>(prod_output_slot);
        }

        DML_OUTPUT_GRAPH_EDGE_DESC edge{};
        edge.FromNodeIndex = static_cast<UINT>(prod_dml_idx);
        edge.FromNodeOutputIndex = from_output_index;
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

    // Match ORT's DmlOperator::GetExecutionFlags(): allow half precision when all
    // tensors in the graph are fp16 (no fp32).  This enables metacommand fast-paths
    // on hardware that supports them.
    {
        bool has_fp16 = false, has_fp32 = false;
        for (size_t i = 0; i < compiled_nodes.size(); ++i) {
            if (!node_is_live[i]) continue;
            const auto& cn = compiled_nodes[i];
            for (const auto& t : cn.translated->input_tensors)
                if (t.data_type == DML_TENSOR_DATA_TYPE_FLOAT16) has_fp16 = true;
                else if (t.data_type == DML_TENSOR_DATA_TYPE_FLOAT32) has_fp32 = true;
            for (const auto& t : cn.translated->output_tensors)
                if (t.data_type == DML_TENSOR_DATA_TYPE_FLOAT16) has_fp16 = true;
                else if (t.data_type == DML_TENSOR_DATA_TYPE_FLOAT32) has_fp32 = true;
        }
        if (has_fp16 && !has_fp32)
            exec_flags |= DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
    }

    DML_PERF_LOG("[Compile] CompileGraph: nodes=", graph_desc.NodeCount,
        " inputs=", graph_desc.InputCount,
        " outputs=", graph_desc.OutputCount,
        " input_edges=", graph_desc.InputEdgeCount,
        " output_edges=", graph_desc.OutputEdgeCount,
        " intermediate_edges=", graph_desc.IntermediateEdgeCount, "\n");

    // Dump all edges so we can find the bad one when CompileGraph returns E_INVALIDARG.
    for (size_t ei = 0; ei < input_edge_storage.size(); ++ei) {
        const auto& e = input_edge_storage[ei];
        DML_PERF_LOG("[Compile] input_edge[", ei, "]: GraphInput=", e.GraphInputIndex,
            " -> node=", e.ToNodeIndex, " input=", e.ToNodeInputIndex, "\n");
    }
    for (size_t ei = 0; ei < output_edge_storage.size(); ++ei) {
        const auto& e = output_edge_storage[ei];
        DML_PERF_LOG("[Compile] output_edge[", ei, "]: node=", e.FromNodeIndex,
            " output=", e.FromNodeOutputIndex, " -> GraphOutput=", e.GraphOutputIndex, "\n");
    }
    for (size_t ei = 0; ei < intermediate_edge_storage.size(); ++ei) {
        const auto& e = intermediate_edge_storage[ei];
        DML_PERF_LOG("[Compile] intermediate_edge[", ei, "]: node=", e.FromNodeIndex,
            " output=", e.FromNodeOutputIndex, " -> node=", e.ToNodeIndex,
            " input=", e.ToNodeInputIndex, "\n");
    }
    for (size_t i = 0; i < compiled_nodes.size(); ++i) {
        if (!node_is_live[i]) continue;
        DML_PERF_LOG("[Compile] node_offset[", i, "]=", dml_node_offset[i],
            " op=", compiled_nodes[i].op_type,
            " sub_nodes=", compiled_nodes[i].translated->sub_nodes.size(), "\n");
    }

    HRESULT hr = dml_device1->CompileGraph(
        &graph_desc, exec_flags,
        IID_PPV_ARGS(compiled_op.GetAddressOf()));
    if (FAILED(hr)) {
        DML_PERF_LOG("[Compile] FAIL: CompileGraph returned HR=0x", Hex(static_cast<uint32_t>(hr)),
            "  nodes=", compiled_nodes.size(), "\n");

        // Drain the D3D12 info queue for DML debug layer validation messages.
        // These appear when DML_CREATE_DEVICE_FLAG_DEBUG is set and describe
        // exactly which tensor, edge, or descriptor is invalid.
        ComPtr<ID3D12Device> d3d12_device_for_iq;
        if (SUCCEEDED(provider->GetD3DDevice(d3d12_device_for_iq.GetAddressOf()))) {
            ComPtr<ID3D12InfoQueue> info_queue;
            if (SUCCEEDED(d3d12_device_for_iq.As(&info_queue))) {
                UINT64 msg_count = info_queue->GetNumStoredMessages();
                DML_PERF_LOG("[Compile] D3D12InfoQueue: ", msg_count, " messages\n");
                for (UINT64 mi = 0; mi < msg_count; ++mi) {
                    SIZE_T msg_len = 0;
                    if (FAILED(info_queue->GetMessage(mi, nullptr, &msg_len))) continue;
                    std::vector<uint8_t> buf(msg_len);
                    auto* msg = reinterpret_cast<D3D12_MESSAGE*>(buf.data());
                    if (FAILED(info_queue->GetMessage(mi, msg, &msg_len))) continue;
                    if (msg->pDescription) {
                        DML_PERF_LOG("[D3D12InfoQueue] ", msg->pDescription, "\n");
                    }
                }
                info_queue->ClearStoredMessages();
            }
        }
        return nullptr;
    }

    // -----------------------------------------------------------------------
    // Step 11: Allocate persistent resource and upload initializers.
    // -----------------------------------------------------------------------

    ComPtr<ID3D12Resource> persistent_resource;
    ComPtr<IUnknown> persistent_allocator;
    std::optional<DML_BUFFER_BINDING> persistent_binding;

    auto binding_props = compiled_op->GetBindingProperties();
    UINT64 persistent_size = binding_props.PersistentResourceSize;
    DML_PERF_LOG("[Compile] BindingProperties: persistent=", persistent_size,
        " temporary=", binding_props.TemporaryResourceSize,
        " descriptors=", binding_props.RequiredDescriptorCount, "\n");
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

    kernel_state->num_outputs     = num_graph_outputs;
    kernel_state->output_dims.resize(num_graph_outputs);
    kernel_state->output_bytes.resize(num_graph_outputs);

    for (size_t i = 0; i < num_graph_outputs; ++i) {
        std::string name = ReadValueInfoName(ort_api, sg.graph_output_vis[i]);
        auto it = sg.value_shapes.find(name);
        if (it == sg.value_shapes.end()) continue;

        // Use ORT shape inference (value_shapes) for output dims and bytes.
        // The producer node's translated output shape can differ from the
        // graph output shape when passthrough nodes (Reshape, Unsqueeze,
        // Squeeze, Flatten) sit between them — value_producer aliases
        // through passthrough nodes, but the graph output shape reflects
        // the final reshaped dimensions.
        const auto& vs = it->second;
        const OrtValueInfo* out_vi = sg.graph_output_vis[i];
        size_t orig_rank = 0;
        if (out_vi && out_vi->GetTypeInfo() && out_vi->GetTypeInfo()->tensor_type_info
            && out_vi->GetTypeInfo()->tensor_type_info->HasShape()) {
            ort_api.GetDimensionsCount(out_vi->GetTypeInfo()->tensor_type_info.get(), &orig_rank);
        }
        if (orig_rank == 0) orig_rank = vs.sizes.size();
        size_t skip = vs.sizes.size() > orig_rank ? vs.sizes.size() - orig_rank : 0;
        kernel_state->output_dims[i].reserve(vs.sizes.size() - skip);
        for (size_t d = skip; d < vs.sizes.size(); ++d)
            kernel_state->output_dims[i].push_back(static_cast<int64_t>(vs.sizes[d]));
        kernel_state->output_bytes[i] = vs.total_bytes;
    }

    auto* info = new FullGraphNodeComputeInfo();
    info->state = std::move(kernel_state);
    return info;
}

bool FullGraphFusion::TryTranslateNodes(
    const OrtApi&                                            ort_api,
    const OrtGraph*                                          main_graph,
    const std::unordered_map<std::string, const OrtValue*>&  initializers,
    const std::vector<const OrtNode*>&                       nodes,
    const std::unordered_map<std::string, std::vector<int64_t>>& resolved_shapes,
    std::unordered_map<std::string, std::vector<int64_t>>*   out_shapes)
{
    SubgraphInfo sg = BuildSubgraphInfo(ort_api, main_graph, initializers, resolved_shapes);

    OpTranslatorRegistry registry = BuildOpTranslatorRegistry();
    bool all_translated = true;

    for (const OrtNode* node : nodes) {
        if (!node) continue;
        const char* op_type = nullptr;
        ort_api.Node_GetOperatorType(node, &op_type);
        if (!op_type) {
            DML_PERF_LOG("[TryTranslateNodes] FAIL: null op_type\n");
            return false;
        }
        auto reg_it = registry.find(op_type);
        if (reg_it == registry.end()) {
            DML_PERF_LOG("[TryTranslateNodes] FAIL: no translator for op=", op_type, "\n");
            return false;
        }
        auto input_names  = fusion_utils::GetNodeInputNames(ort_api, node);
        auto output_names = fusion_utils::GetNodeOutputNames(ort_api, node);
        auto translated = reg_it->second(ort_api, node, sg.value_shapes, sg.all_initializers);
        if (!translated) {
            DML_PERF_LOG("[TryTranslateNodes] FAIL: op=", op_type);
            for (size_t i = 0; i < input_names.size(); ++i) {
                DML_PERF_LOG(" in", i, "='", input_names[i], "'(", sg.value_shapes.count(input_names[i]), ")");
            }
            DML_PERF_LOG(" out0=", (output_names.empty() ? "?" : output_names[0]), "\n");
            all_translated = false;
            continue;  // keep translating for shape collection
        }
        DML_PERF_LOG("[TryTranslateNodes] OK: op=", op_type,
            " out0=", (output_names.empty() ? "?" : output_names[0]), "\n");
        // Write-back: translator computed output shapes; seed them for downstream.
        for (size_t k = 0; k < output_names.size() && k < translated->output_tensors.size(); ++k) {
            if (!sg.value_shapes.count(output_names[k])) {
                sg.value_shapes[output_names[k]] = translated->output_tensors[k];
                auto& t = translated->output_tensors[k];
                DML_PERF_LOG("[TryTranslateNodes] write-back '", output_names[k], "' sizes=[");
                for (size_t d = 0; d < t.sizes.size(); ++d) DML_PERF_LOG(d>0?",":"", t.sizes[d]);
                DML_PERF_LOG("]\n");
            }
        }
    }

    // Export computed shapes so subsequent groups can use them as boundary inputs.
    // value_shapes is keyed by producer-side output names, but downstream groups
    // look up shapes by consumer-side input names (which may differ due to ORT
    // edge renaming). Build a producer→consumer alias map from the main graph,
    // then export under both names.
    if (out_shapes) {
        // Build alias map: for each main-graph node, map its output names to
        // any consumer input names that differ (ORT edge renaming).
        size_t total_nodes = 0;
        ort_api.Graph_GetNumNodes(main_graph, &total_nodes);
        std::vector<const OrtNode*> all_nodes(total_nodes, nullptr);
        if (total_nodes > 0)
            ort_api.Graph_GetNodes(main_graph, all_nodes.data(), total_nodes);

        // Map producer output name → set of consumer input names.
        std::unordered_map<std::string, std::string> producer_output_name;
        for (const OrtNode* n : all_nodes) {
            if (!n) continue;
            auto onames = fusion_utils::GetNodeOutputNames(ort_api, n);
            size_t nout = 0;
            ort_api.Node_GetNumOutputs(n, &nout);
            std::vector<const OrtValueInfo*> ovis(nout, nullptr);
            if (nout > 0) ort_api.Node_GetOutputs(n, ovis.data(), nout);
            for (size_t k = 0; k < nout && k < onames.size(); ++k) {
                if (onames[k].empty() || !ovis[k]) continue;
                producer_output_name[fusion_utils::GetValueInfoName(ort_api, ovis[k])] = onames[k];
            }
        }

        // Export value_shapes under both producer and consumer names.
        auto ExportDims = [&](const std::string& name, const DmlTensorInfo& info) {
            if (out_shapes->count(name)) return;
            std::vector<int64_t> dims(info.sizes.size());
            for (size_t d = 0; d < info.sizes.size(); ++d)
                dims[d] = static_cast<int64_t>(info.sizes[d]);
            (*out_shapes)[name] = std::move(dims);
        };

        for (const auto& [name, info] : sg.value_shapes) {
            ExportDims(name, info);
        }

        // For each main-graph consumer input, if its VI name maps to a producer
        // output that has a shape in value_shapes, export under the consumer name.
        for (const OrtNode* n : all_nodes) {
            if (!n) continue;
            auto in_names = fusion_utils::GetNodeInputNames(ort_api, n);
            size_t nin = 0;
            ort_api.Node_GetNumInputs(n, &nin);
            std::vector<const OrtValueInfo*> ivis(nin, nullptr);
            if (nin > 0) ort_api.Node_GetInputs(n, ivis.data(), nin);
            for (size_t k = 0; k < nin && k < in_names.size(); ++k) {
                if (in_names[k].empty() || out_shapes->count(in_names[k])) continue;
                if (!ivis[k]) continue;
                auto vi_name = fusion_utils::GetValueInfoName(ort_api, ivis[k]);
                // Try: consumer input name is directly in value_shapes.
                auto vs_it = sg.value_shapes.find(in_names[k]);
                if (vs_it != sg.value_shapes.end()) {
                    ExportDims(in_names[k], vs_it->second);
                    continue;
                }
                // Try: VI name → producer output name → value_shapes.
                auto prod_it = producer_output_name.find(vi_name);
                if (prod_it != producer_output_name.end()) {
                    vs_it = sg.value_shapes.find(prod_it->second);
                    if (vs_it != sg.value_shapes.end()) {
                        ExportDims(in_names[k], vs_it->second);
                    }
                }
            }
        }
    }
    return all_translated;
}

bool FullGraphFusion::TryCompilePartition(
    const OrtApi&                                            ort_api,
    const OrtGraph*                                          main_graph,
    const std::unordered_map<std::string, const OrtValue*>&  initializers,
    PluginDmlExecutionProviderImpl*                          provider,
    const std::vector<const OrtNode*>&                       nodes,
    const std::unordered_map<std::string, std::vector<int64_t>>& resolved_shapes)
{
    SubgraphInfo sg = BuildSubgraphInfo(ort_api, main_graph, initializers, resolved_shapes);
    OpTranslatorRegistry registry = BuildOpTranslatorRegistry();

    // Translate nodes (same logic as Compile's step 4).
    std::vector<CompiledNode> compiled_nodes;
    std::unordered_map<std::string, std::pair<size_t, size_t>> value_producer;
    std::unordered_map<std::string, std::string> graph_input_aliases;
    std::unordered_set<std::string> consumed_initializer_names;

    for (const OrtNode* node : nodes) {
        if (!node) continue;
        const char* op_type = nullptr;
        ort_api.Node_GetOperatorType(node, &op_type);
        if (!op_type) return false;
        auto reg_it = registry.find(op_type);
        if (reg_it == registry.end()) return false;
        auto input_names  = fusion_utils::GetNodeInputNames(ort_api, node);
        auto output_names = fusion_utils::GetNodeOutputNames(ort_api, node);
        auto translated = reg_it->second(ort_api, node, sg.value_shapes, sg.all_initializers);
        if (!translated) return false;
        // Write-back: translator computed output shapes; seed them for downstream.
        for (size_t k = 0; k < output_names.size() && k < translated->output_tensors.size(); ++k) {
            if (!sg.value_shapes.count(output_names[k]))
                sg.value_shapes[output_names[k]] = translated->output_tensors[k];
        }
        auto translated_ptr = std::make_unique<TranslatedOp>(std::move(*translated));
        translated_ptr->FixupPointers();
        if (translated_ptr->passthrough) {
            if (!input_names.empty()) {
                const auto& src = input_names[0];
                auto prod_it = value_producer.find(src);
                if (prod_it != value_producer.end()) {
                    for (auto& oname : output_names) value_producer[oname] = prod_it->second;
                } else {
                    for (auto& oname : output_names) graph_input_aliases[oname] = src;
                }
            }
            continue;
        }
        size_t dml_input_count = translated_ptr->input_tensors.size();
        for (size_t s = 0; s < input_names.size() && s < dml_input_count; ++s) {
            size_t name_idx = translated_ptr->input_name_reorder.empty()
                ? s : translated_ptr->input_name_reorder[s];
            if (name_idx >= input_names.size()) continue;
            if (sg.all_initializers.count(input_names[name_idx]))
                consumed_initializer_names.insert(input_names[name_idx]);
        }
        size_t idx = compiled_nodes.size();
        for (size_t k = 0; k < output_names.size(); ++k)
            value_producer[output_names[k]] = {idx, k};
        CompiledNode cn;
        cn.translated = std::move(translated_ptr);
        cn.input_names = std::move(input_names);
        cn.output_names = std::move(output_names);
        cn.op_type = op_type;
        compiled_nodes.push_back(std::move(cn));
    }
    if (compiled_nodes.empty()) return true;

    // CreateOperator.
    ComPtr<IDMLDevice> dml_device;
    if (FAILED(provider->GetDmlDevice(dml_device.GetAddressOf()))) return false;
    ComPtr<IDMLDevice1> dml_device1;
    if (FAILED(dml_device.As(&dml_device1))) return false;
    for (auto& cn : compiled_nodes) {
        if (FAILED(dml_device->CreateOperator(&cn.translated->op_desc,
                IID_PPV_ARGS(cn.translated->dml_operator.GetAddressOf())))) return false;
        for (auto& sn : cn.translated->sub_nodes)
            if (FAILED(dml_device->CreateOperator(&sn.op_desc,
                    IID_PPV_ARGS(sn.dml_operator.GetAddressOf())))) return false;
    }

    // Build graph descriptor (minimal — just enough for CompileGraph validation).
    //
    // TryCompilePartition operates on the main graph, but the DML graph being
    // compiled covers only `nodes` (a partition). Its inputs are the partition
    // boundary: values consumed inside `nodes` but not produced by any node in
    // `nodes`. Using sg.graph_input_vis (the full model's inputs) here would
    // set the wrong InputCount and produce E_INVALIDARG from CompileGraph.
    //
    // Compute boundary inputs: walk each *non-passthrough* partition node's inputs
    // (compiled_nodes only — passthrough nodes have no DML representation), keep
    // names that are not produced inside the partition (value_producer) and not
    // initializers (handled separately via consumed_initializer_names below).
    // boundary_input_names holds alias-resolved names — the same names that the
    // edge-wiring loop will look up in im.dml_input_map after calling
    //   while (graph_input_aliases.count(name)) name = graph_input_aliases[name];
    // So we must key the map on the resolved name, not the raw input name.
    std::vector<std::string> boundary_input_names;
    {
        std::unordered_set<std::string> seen;
        for (const auto& cn : compiled_nodes) {
            size_t npi = cn.translated->primary_input_count > 0
                ? cn.translated->primary_input_count : cn.translated->input_tensors.size();
            for (size_t s = 0; s < cn.input_names.size() && s < npi; ++s) {
                size_t ni = cn.translated->input_name_reorder.empty()
                    ? s : cn.translated->input_name_reorder[s];
                if (ni >= cn.input_names.size()) continue;
                const auto& raw_name = cn.input_names[ni];
                if (raw_name.empty()) continue;
                std::string name = raw_name;
                while (graph_input_aliases.count(name)) name = graph_input_aliases.at(name);
                if (value_producer.count(name)) continue;
                if (sg.all_initializers.count(name)) { consumed_initializer_names.insert(name); continue; }
                if (seen.insert(name).second) boundary_input_names.push_back(name);
            }
        }
    }

    // Build a minimal DmlInputMapResult equivalent for the partition boundary.
    // We replicate BuildDmlInputMap's logic without needing OrtValueInfo* pointers.
    DmlInputMapResult im;
    im.subgraph_to_dml_input.assign(boundary_input_names.size(), SIZE_MAX);
    // Pass 1: inline small consumed initializers as DML constant nodes.
    for (const auto& init_name : consumed_initializer_names) {
        auto shape_it = sg.value_shapes.find(init_name);
        uint64_t bytes = (shape_it != sg.value_shapes.end()) ? shape_it->second.total_bytes : 0;
        if (bytes == 0 || bytes >= kMaxConstNodeDataSize) continue;
        auto init_it = sg.all_initializers.find(init_name);
        if (init_it == sg.all_initializers.end() || !init_it->second) continue;
        void* cpu_ptr = nullptr;
        OrtStatus* st = ort_api.GetTensorMutableData(
            const_cast<OrtValue*>(init_it->second), &cpu_ptr);
        if (!st && cpu_ptr) {
            ConstantNodeInfo cni;
            cni.name = init_name;
            cni.data.assign(static_cast<uint8_t*>(cpu_ptr),
                            static_cast<uint8_t*>(cpu_ptr) + bytes);
            im.constant_node_map[init_name] = im.constant_nodes.size();
            im.constant_nodes.push_back(std::move(cni));
        }
        if (st) ort_api.ReleaseStatus(st);
    }
    // Pass 2: assign DML input indices for partition boundary inputs.
    size_t dml_input_idx = 0;
    for (size_t i = 0; i < boundary_input_names.size(); ++i) {
        const auto& n = boundary_input_names[i];
        if (im.constant_node_map.count(n)) continue; // inlined — no slot
        im.dml_input_map[n] = dml_input_idx;
        im.subgraph_to_dml_input[i] = dml_input_idx;
        ++dml_input_idx;
    }
    // Pass 3: large consumed initializers not in boundary_input_names.
    for (const auto& init_name : consumed_initializer_names) {
        if (im.constant_node_map.count(init_name)) continue;
        if (im.dml_input_map.count(init_name)) continue;
        im.dml_input_map[init_name] = dml_input_idx++;
        im.ordered_initializer_names.push_back(init_name);
    }
    im.total_dml_inputs = dml_input_idx;

    size_t total_dml_nodes = 0;
    std::vector<size_t> dml_node_offset(compiled_nodes.size());
    for (size_t i = 0; i < compiled_nodes.size(); ++i) {
        dml_node_offset[i] = total_dml_nodes;
        total_dml_nodes += 1 + compiled_nodes[i].translated->sub_nodes.size();
    }
    std::vector<DML_OPERATOR_GRAPH_NODE_DESC> op_descs(total_dml_nodes);
    std::vector<DML_GRAPH_NODE_DESC> graph_nodes(total_dml_nodes);
    for (size_t i = 0; i < compiled_nodes.size(); ++i) {
        size_t base = dml_node_offset[i];
        op_descs[base] = {compiled_nodes[i].translated->dml_operator.Get(), nullptr};
        graph_nodes[base] = {DML_GRAPH_NODE_TYPE_OPERATOR, &op_descs[base]};
        for (size_t s = 0; s < compiled_nodes[i].translated->sub_nodes.size(); ++s) {
            op_descs[base+1+s] = {compiled_nodes[i].translated->sub_nodes[s].dml_operator.Get(), nullptr};
            graph_nodes[base+1+s] = {DML_GRAPH_NODE_TYPE_OPERATOR, &op_descs[base+1+s]};
        }
    }
    std::vector<DML_CONSTANT_DATA_GRAPH_NODE_DESC> const_descs(im.constant_nodes.size());
    for (size_t c = 0; c < im.constant_nodes.size(); ++c) {
        const_descs[c].Data = im.constant_nodes[c].data.data();
        const_descs[c].DataSize = im.constant_nodes[c].data.size();
        graph_nodes.push_back({DML_GRAPH_NODE_TYPE_CONSTANT, &const_descs[c]});
    }
    size_t const_node_base = total_dml_nodes;

    // Wire edges (same logic as Compile's step 9, simplified to just detect validity).
    std::vector<DML_INPUT_GRAPH_EDGE_DESC> ie;
    std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> me;
    std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> oe;

    for (size_t ci = 0; ci < compiled_nodes.size(); ++ci) {
        const auto& cn = compiled_nodes[ci];
        size_t pdml = dml_node_offset[ci];
        size_t npi = cn.translated->primary_input_count > 0
            ? cn.translated->primary_input_count : cn.translated->input_tensors.size();
        for (size_t s = 0; s < cn.input_names.size() && s < npi; ++s) {
            size_t ni = cn.translated->input_name_reorder.empty()
                ? s : cn.translated->input_name_reorder[s];
            if (ni >= cn.input_names.size()) continue;
            auto name = cn.input_names[ni];
            if (name.empty()) continue;
            while (graph_input_aliases.count(name)) name = graph_input_aliases[name];
            size_t dml_slot = cn.translated->dml_input_slot_indices.empty()
                ? s : cn.translated->dml_input_slot_indices[s];
            if (im.constant_node_map.count(name)) {
                me.push_back({(UINT)(const_node_base+im.constant_node_map[name]),0,(UINT)pdml,(UINT)dml_slot});
            } else if (im.dml_input_map.count(name)) {
                ie.push_back({(UINT)im.dml_input_map[name],(UINT)pdml,(UINT)dml_slot});
            } else if (value_producer.count(name)) {
                auto [pci, pos] = value_producer[name];
                size_t pdi = dml_node_offset[pci];
                UINT foi;
                const auto& osrc = compiled_nodes[pci].translated->output_source;
                if (!osrc.empty() && pos < osrc.size()) {
                    auto [ss, sl] = osrc[pos];
                    if (ss >= 0) pdi += 1+ss;
                    foi = (UINT)sl;
                } else {
                    if (!compiled_nodes[pci].translated->sub_nodes.empty())
                        pdi += compiled_nodes[pci].translated->sub_nodes.size();
                    foi = (UINT)pos;
                }
                me.push_back({(UINT)pdi,foi,(UINT)pdml,(UINT)dml_slot});
            } else {
                return false;
            }
        }

        // Wire internal edges for sub_nodes (mirrors Compile step 9 sub-node section).
        for (size_t s = 0; s < cn.translated->sub_nodes.size(); ++s) {
            const auto& sn = cn.translated->sub_nodes[s];
            size_t sn_dml_idx = pdml + 1 + s;
            for (size_t inp = 0; inp < sn.input_from.size(); ++inp) {
                auto [src_sub, src_slot] = sn.input_from[inp];
                if (src_sub < -1) continue; // sentinel: handled by graph_inputs
                size_t from_dml_idx = (src_sub < 0)
                    ? pdml
                    : pdml + 1 + static_cast<size_t>(src_sub);
                me.push_back({(UINT)from_dml_idx, (UINT)src_slot,
                              (UINT)sn_dml_idx, (UINT)inp});
            }
            for (auto& [onnx_idx, to_input] : sn.graph_inputs) {
                if (onnx_idx >= cn.input_names.size()) continue;
                auto gi_name = cn.input_names[onnx_idx];
                while (graph_input_aliases.count(gi_name))
                    gi_name = graph_input_aliases[gi_name];
                if (im.constant_node_map.count(gi_name)) {
                    me.push_back({(UINT)(const_node_base + im.constant_node_map[gi_name]),
                                  0, (UINT)sn_dml_idx, (UINT)to_input});
                } else if (im.dml_input_map.count(gi_name)) {
                    ie.push_back({(UINT)im.dml_input_map[gi_name],
                                  (UINT)sn_dml_idx, (UINT)to_input});
                }
            }
        }
    }
    // Wire output edges: every DML node (operator + constant) must have at least
    // one outgoing edge (intermediate or output). Collect all DML node indices
    // that already appear as FromNodeIndex in intermediate edges — those are
    // "wired" and do not need an output edge.  Any DML node not in that set
    // gets a dummy output edge so DML doesn't reject it as orphaned.
    std::unordered_set<UINT> wired_from;
    for (auto& e : me) wired_from.insert(e.FromNodeIndex);

    size_t oi_idx = 0;
    // Output edges for operator nodes.
    for (size_t ci = 0; ci < compiled_nodes.size(); ++ci) {
        for (size_t k = 0; k < compiled_nodes[ci].output_names.size(); ++k) {
            size_t pdi = dml_node_offset[ci];
            UINT foi;
            const auto& osrc = compiled_nodes[ci].translated->output_source;
            if (!osrc.empty() && k < osrc.size()) {
                auto [ss, sl] = osrc[k]; if (ss>=0) pdi+=1+ss; foi=(UINT)sl;
            } else {
                if (!compiled_nodes[ci].translated->sub_nodes.empty())
                    pdi += compiled_nodes[ci].translated->sub_nodes.size();
                foi = (UINT)k;
            }
            if (wired_from.count((UINT)pdi)) continue; // already has an outgoing edge
            oe.push_back({(UINT)pdi, foi, (UINT)oi_idx++});
        }
    }

    // Output edges for constant nodes with no outgoing intermediate edge.
    for (size_t c = 0; c < im.constant_nodes.size(); ++c) {
        UINT const_dml_idx = static_cast<UINT>(const_node_base + c);
        if (wired_from.count(const_dml_idx)) continue;
        oe.push_back({const_dml_idx, 0, (UINT)oi_idx++});
    }

    // Wrap edges.
    std::vector<DML_GRAPH_EDGE_DESC> ie2(ie.size()), me2(me.size()), oe2(oe.size());
    for (size_t i=0;i<ie.size();++i) ie2[i]={DML_GRAPH_EDGE_TYPE_INPUT,&ie[i]};
    for (size_t i=0;i<me.size();++i) me2[i]={DML_GRAPH_EDGE_TYPE_INTERMEDIATE,&me[i]};
    for (size_t i=0;i<oe.size();++i) oe2[i]={DML_GRAPH_EDGE_TYPE_OUTPUT,&oe[i]};

    DML_GRAPH_DESC gd{};
    gd.InputCount = (UINT)im.total_dml_inputs;
    gd.OutputCount = (UINT)oi_idx;
    gd.NodeCount = (UINT)graph_nodes.size();
    gd.Nodes = graph_nodes.data();
    gd.InputEdgeCount = (UINT)ie2.size(); gd.InputEdges = ie2.data();
    gd.OutputEdgeCount = (UINT)oe2.size(); gd.OutputEdges = oe2.data();
    gd.IntermediateEdgeCount = (UINT)me2.size(); gd.IntermediateEdges = me2.data();

    DML_EXECUTION_FLAGS flags = compiled_nodes.size()>=5
        ? DML_EXECUTION_FLAG_DESCRIPTORS_VOLATILE : DML_EXECUTION_FLAG_NONE;
    {
        bool has_fp16 = false, has_fp32 = false;
        for (const auto& cn : compiled_nodes) {
            for (const auto& t : cn.translated->input_tensors)
                if (t.data_type == DML_TENSOR_DATA_TYPE_FLOAT16) has_fp16 = true;
                else if (t.data_type == DML_TENSOR_DATA_TYPE_FLOAT32) has_fp32 = true;
            for (const auto& t : cn.translated->output_tensors)
                if (t.data_type == DML_TENSOR_DATA_TYPE_FLOAT16) has_fp16 = true;
                else if (t.data_type == DML_TENSOR_DATA_TYPE_FLOAT32) has_fp32 = true;
        }
        if (has_fp16 && !has_fp32)
            flags |= DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
    }
    ComPtr<IDMLCompiledOperator> op;
    HRESULT hr = dml_device1->CompileGraph(&gd, flags, IID_PPV_ARGS(op.GetAddressOf()));
    if (FAILED(hr)) {
        char hr_buf[32];
        snprintf(hr_buf, sizeof(hr_buf), "0x%08X", (unsigned)hr);
        DiagLog(std::string("[TryCompilePartition] CompileGraph FAILED HR=") + hr_buf +
            " nodes=" + std::to_string(compiled_nodes.size()) + "\n");

        // Drain D3D12 info queue for DML debug layer validation messages.
        ComPtr<ID3D12Device> d3d12_dev;
        if (SUCCEEDED(provider->GetD3DDevice(d3d12_dev.GetAddressOf()))) {
            ComPtr<ID3D12InfoQueue> iq;
            if (SUCCEEDED(d3d12_dev.As(&iq))) {
                UINT64 n = iq->GetNumStoredMessages();
                DiagLog("[TryCompilePartition] D3D12InfoQueue messages: " + std::to_string(n) + "\n");
                for (UINT64 mi = 0; mi < n; ++mi) {
                    SIZE_T len = 0;
                    if (FAILED(iq->GetMessage(mi, nullptr, &len))) continue;
                    std::vector<uint8_t> buf(len);
                    auto* msg = reinterpret_cast<D3D12_MESSAGE*>(buf.data());
                    if (FAILED(iq->GetMessage(mi, msg, &len))) continue;
                    if (msg->pDescription)
                        DiagLog(std::string("[D3D12InfoQueue] ") + msg->pDescription + "\n");
                }
                iq->ClearStoredMessages();
            } else {
                DiagLog("[TryCompilePartition] ID3D12InfoQueue unavailable"
                    " — DML debug layer not active\n");
            }
        }

        return false;
    }
    return true;
}

bool FullGraphFusion::TryCompileGraph(
    const OrtApi&                                            ort_api,
    const OrtGraph*                                          main_graph,
    const std::unordered_map<std::string, const OrtValue*>&  initializers,
    PluginDmlExecutionProviderImpl*                          provider,
    const std::unordered_map<std::string, std::vector<int64_t>>& resolved_shapes)
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
    OrtNodeComputeInfo* info = Compile(ort_api, main_graph, initializers, provider, resolved_shapes);
    if (info != nullptr) {
        delete info; // discard — CompileImpl recompiles against fused subgraph
        return true;
    }
    return false;
}

}  // namespace dml_ep
