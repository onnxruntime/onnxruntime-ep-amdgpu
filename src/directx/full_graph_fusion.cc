// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "full_graph_fusion.h"
#include "fusion_utils.h"
#include "ort_node_adapter.h"
#include "node_view.h"
#include "dml_execution_provider.h"
#include "DmlExecutionProvider/IExecutionProvider.h"
#include "dml_abi_kernel.h"
#include "core/framework/abi_safe_attr_utils.h"
#include "core/graph/constants.h"                   // kMSDomain et al.
#include "core/graph/contrib_ops/contrib_defs.h"   // RegisterContribSchemas
#include "core/graph/contrib_ops/ms_opset.h"        // OpSet_Microsoft_ver1
#include "onnx/defs/schema.h"                       // OpSchemaRegistry
#include "onnx/shape_inference/implementation.h"    // InferShapes (contrib shape inference)

#include <DirectML.h>
#include <wrl/client.h>
#include <gsl/gsl>
#include <algorithm>
#include <cstring>
#include <mutex>
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

// Deferred (runtime-fusion) first-Compute compile. Builds one CompiledVariant
// from the owned snapshot + concrete runtime input dims. Defined below; forward-
// declared here because FullGraph_Compute (above its definition) calls it.
// Returns false on any failure (caller returns ORT_FAIL — no per-op fallback).
static bool CompileFromSnapshot(
    const OrtApi&                                            ort_api,
    DeferredSubgraph&                                        snap,
    const std::vector<std::vector<int64_t>>&                 runtime_input_dims,
    PluginDmlExecutionProviderImpl*                          provider,
    FullGraphKernelState*                                    kernel_state,
    CompiledVariant&                                         out_variant);

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
        fusion_utils::ValueInfoShape s = fusion_utils::GetValueInfoShape(ort_api, vi);
        if (!s.has_type_info || !s.has_shape) return false;

        // Scalars (rank 0) are inherently static — 1 element, no dims to check.
        if (s.rank == 0) continue;

        for (int64_t d : s.dims)
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

    // -----------------------------------------------------------------------
    // Deferred (runtime-fusion) path: compile lazily on first Compute, keyed by
    // the concrete input-shape signature. On a signature MISS, compile a new
    // variant from the owned snapshot; on HIT, re-point the active variant and
    // replay. Compile failure fails the Run (ORT_FAIL) — no per-op fallback.
    // -----------------------------------------------------------------------
    if (state->deferred) {
        if (!state->snapshot) return api.CreateStatus(ORT_FAIL, "RuntimeFusion: null snapshot");

        // Read concrete runtime input dims from the kernel context, in graph-input
        // order, and hash a signature over them.
        //
        // CRITICAL: KernelContext_GetInput has NO bounds check (custom_ops.cc:
        // GetInputMLValue) — an out-of-range index reads past the input array and
        // crashes. graph_input_names includes INITIALIZERS exposed as graph inputs
        // (Mode A), which are NOT kernel-context inputs; their dims come from the
        // owned snapshot bytes, not the runtime context. So we (a) bound the loop
        // by the actual KernelContext_GetInputCount, and (b) skip any graph input
        // that is a snapshot initializer. GetTensorTypeAndShape can also throw on a
        // non-tensor, so the whole block is wrapped in ORT_TRY (noexcept boundary).
        const size_t num_inputs = state->snapshot->graph_input_names.size();
        std::vector<std::vector<int64_t>> runtime_input_dims(num_inputs);
        size_t sig = 1469598103934665603ull;  // FNV-1a offset
        auto hash_mix = [&](int64_t v) {
            sig ^= static_cast<size_t>(v);
            sig *= 1099511628211ull;
        };

        size_t kctx_input_count = 0;
        api.KernelContext_GetInputCount(kernel_context, &kctx_input_count);
        // kctx_input_count < num_inputs means ORT does not pass initializers as
        // kernel inputs, so indices >= kctx_input_count must not be read from the
        // kernel context (see the i >= kctx_input_count guard below). Equal counts
        // mean ORT passes all graph inputs.
        DML_PERF_LOG("[RuntimeFusion] Compute deferred ENTER: graph_input_names=", num_inputs,
                     " kctx_input_count=", kctx_input_count,
                     " initializers=", state->snapshot->initializer_view.size(), "\n");

        ORT_TRY {
            for (size_t i = 0; i < num_inputs; ++i) {
                const std::string& in_name = state->snapshot->graph_input_names[i];
                // Initializer-backed graph inputs: dims come from the snapshot, not
                // the kernel context. Skip (do not index the context).
                if (state->snapshot->initializer_view.count(in_name)) {
                    DML_PERF_LOG("[RuntimeFusion]   in[", i, "] '", in_name, "' = INITIALIZER (skip)\n");
                    hash_mix(-3); continue;
                }
                // Guard against indexing past the real runtime inputs.
                if (i >= kctx_input_count) {
                    DML_PERF_LOG("[RuntimeFusion]   in[", i, "] '", in_name, "' = OUT_OF_KCTX_RANGE (skip)\n");
                    hash_mix(-5); continue;
                }

                DML_PERF_LOG("[RuntimeFusion]   in[", i, "] '", in_name, "' reading kctx...\n");
                const OrtValue* input_value = nullptr;
                OrtStatus* st = api.KernelContext_GetInput(kernel_context, i, &input_value);
                if (st || !input_value) {
                    if (st) api.ReleaseStatus(st);
                    DML_PERF_LOG("[RuntimeFusion]   in[", i, "] null/err from kctx\n");
                    hash_mix(-7); continue;
                }
                OrtTensorTypeAndShapeInfo* tsi = nullptr;
                if (api.GetTensorTypeAndShape(const_cast<OrtValue*>(input_value), &tsi) == nullptr && tsi) {
                    size_t rank = 0;
                    api.GetDimensionsCount(tsi, &rank);
                    std::vector<int64_t> dims(rank, 0);
                    if (rank > 0) api.GetDimensions(tsi, dims.data(), rank);
                    api.ReleaseTensorTypeAndShapeInfo(tsi);
                    hash_mix(static_cast<int64_t>(rank));
                    for (int64_t d : dims) hash_mix(d);
                    DML_PERF_LOG("[RuntimeFusion]   in[", i, "] rank=", rank, " ok\n");
                    runtime_input_dims[i] = std::move(dims);
                } else {
                    DML_PERF_LOG("[RuntimeFusion]   in[", i, "] not-a-tensor / no shape\n");
                    hash_mix(-9);
                }
            }
        }
        ORT_CATCH(const std::exception& e) {
            ORT_HANDLE_EXCEPTION([&]() {
                DML_PERF_LOG("[RuntimeFusion] signature read threw: ", e.what(), "\n");
            });
            return api.CreateStatus(ORT_FAIL, "RuntimeFusion: signature read failed");
        }
        DML_PERF_LOG("[RuntimeFusion] signature loop DONE, sig=", sig, "\n");

        auto cache_it = state->signature_cache.find(sig);
        if (cache_it != state->signature_cache.end()) {
            DML_PERF_LOG("[RuntimeFusion] sig=", sig, " HIT\n");
            state->ActivateVariant(cache_it->second);
        } else {
            DML_PERF_LOG("[RuntimeFusion] sig=", sig, " MISS — compiling variant\n");
            CompiledVariant variant;
            bool ok = false;
            ORT_TRY {
                ok = CompileFromSnapshot(api, *state->snapshot, runtime_input_dims,
                                         state->provider, state, variant);
            }
            ORT_CATCH(const std::exception& e) {
                ORT_HANDLE_EXCEPTION([&]() {
                    DML_PERF_LOG("[RuntimeFusion] CompileFromSnapshot threw: ", e.what(), "\n");
                });
                ok = false;
            }
            if (!ok)
                return api.CreateStatus(ORT_FAIL, "RuntimeFusion: deferred compile failed");
            auto [it, _] = state->signature_cache.emplace(sig, std::move(variant));
            state->ActivateVariant(it->second);
        }
    }

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
    // Shared-initializer DML slots (deferred path) are bound from state->shared_initializers
    // in the dedicated loop below, NOT from the kernel context. Guard the kernel-context
    // loop against them in case an initializer is ALSO a declared graph input (some ORT
    // graphs expose an initializer as a graph input -> it would get a subgraph_to_dml_input
    // slot here); binding it from KernelContext_GetInput would fetch the wrong/absent value.
    std::unordered_set<size_t> shared_init_dml_slots;
    for (const auto& [dml_i, name] : state->shared_initializer_slots)
        shared_init_dml_slots.insert(dml_i);

    // OWNED inputs are left as NONE (data already in persistent resource).
    for (size_t sg_i = 0; sg_i < state->num_subgraph_inputs; ++sg_i) {
        if (sg_i >= state->subgraph_to_dml_input.size()) break;
        size_t dml_i = state->subgraph_to_dml_input[sg_i];
        if (dml_i == SIZE_MAX) continue;                        // inlined constant
        if (state->runtime_input_is_owned[dml_i]) continue;    // baked into persistent resource
        if (shared_init_dml_slots.count(dml_i)) continue;       // bound from shared_initializers below
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

    // Bind the shared initializer WEIGHTS (deferred path). These DML slots were NOT
    // marked OWNED_BY_DML (so DML did not bake them into the persistent resource), and
    // they are NOT in the kernel context (they're the plugin's own initializers, not
    // ORT runtime inputs). Their data was uploaded ONCE to state->shared_initializers
    // and is bound here every dispatch — one VRAM copy shared across all shape variants,
    // so weights are not duplicated per variant. Skip edge-less slots (DML pruned them
    // -> must stay NONE, same rule as the runtime-input loop above).
    for (const auto& [dml_i, name] : state->shared_initializer_slots) {
        if (dml_i >= input_buffer_bindings.size()) continue;
        if (!state->dml_inputs_with_edges.count(dml_i)) continue;  // no edge -> leave NONE
        auto si_it = state->shared_initializers.find(name);
        if (si_it == state->shared_initializers.end() || !si_it->second.gpu_resource)
            return api.CreateStatus(ORT_FAIL, "RuntimeFusion: missing shared initializer resource");
        input_buffer_bindings[dml_i] = { si_it->second.gpu_resource.Get(), 0, si_it->second.bytes };
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

    // Submit this partition's command list now instead of batching the whole graph
    // into one deferred submission, so the GPU can execute it while the CPU records
    // the next partition. Flush() is non-blocking; per-dispatch UAV barriers and
    // command-queue fence ordering preserve correctness across the split lists.
    state->provider->Flush();

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
        ort_version_supported = NegotiatedOrtApiVersion();
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
    const std::unordered_map<std::string, std::vector<int64_t>>& resolved_shapes,
    const std::unordered_map<std::string, const OrtValue*>& initializers)
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
        if (it->second.empty()) return false;
        for (int64_t d : it->second) if (d <= 0) return false;
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

        // Reject any node with an empty (extent-0) edge DML can't represent in a
        // compiled graph — empty output, or empty non-constant input. Shared with
        // Tier-0 group formation (which uses it to split partitions at the same
        // boundary), so both admission gates agree. Constant-initializer empty
        // inputs (e.g. Resize roi) are exempt.
        if (fusion_utils::NodeHasEmptyEdge(ort_api, node, initializers)) {
            DiagLog("[ValidateTier0] FAIL: op=" + GetOpType(node) + " has empty (extent-0) edge\n");
            return false;
        }

        size_t node_num_inputs = 0;
        ort_api.Node_GetNumInputs(node, &node_num_inputs);
        std::vector<const OrtValueInfo*> in_vis(node_num_inputs, nullptr);
        if (node_num_inputs > 0)
            ort_api.Node_GetInputs(node, in_vis.data(), node_num_inputs);
        auto input_names = fusion_utils::GetNodeInputNames(ort_api, node);
        for (size_t k = 0; k < node_num_inputs && k < input_names.size(); ++k) {
            if (in_vis[k] == nullptr || input_names[k].empty()) continue;
            fusion_utils::ValueInfoShape s = fusion_utils::GetValueInfoShape(ort_api, in_vis[k]);
            if (!s.has_type_info) continue;
            if (OnnxDtypeToDml(s.elem_type) == DML_TENSOR_DATA_TYPE_UNKNOWN) {
                DiagLog("[ValidateTier0] FAIL input: op=" + GetOpType(node) + " tensor='" + GetViName(in_vis[k]) + "' unsupported dtype=" + std::to_string(static_cast<int>(s.elem_type)) + "\n");
                return false;
            }
            if (!s.has_shape) continue;
            if (s.rank > 0) {
                // Empty (extent-0) edges are already handled by NodeHasEmptyEdge
                // above; here we only reject positively-dynamic dims (< 0).
                bool has_dynamic = false;
                for (size_t d = 0; d < s.rank; ++d)
                    if (s.dims[d] < 0) { has_dynamic = true; break; }
                if (has_dynamic && !CheckResolvedFallback(in_vis[k])) {
                    DiagLog("[ValidateTier0] FAIL input: op=" + GetOpType(node) + " tensor='" + GetViName(in_vis[k]) + "' dynamic dims\n");
                    return false;
                }
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// FullGraphFusion::ValidateTier0Structural — shape-blind admission for the
// runtime-fusion (deferred-compile) path. Mirrors ValidateTier0's per-node
// dtype/empty-edge/translator checks but does NOT reject dynamic dims (shapes
// are unknown until first Compute). Adds a data-dependent-shape-op gate: an op
// whose OUTPUT shape is driven by an input tensor's VALUE (not just its shape)
// can only be compiled from a snapshot if that driving input is a resolved
// constant — otherwise its output shape can't be known at deferred-compile time.
// ---------------------------------------------------------------------------

bool FullGraphFusion::ValidateTier0Structural(
    const OrtApi&                                            ort_api,
    const std::vector<const OrtNode*>&                       nodes,
    const std::unordered_map<std::string, std::vector<int64_t>>& resolved_shapes,
    const std::unordered_map<std::string, const OrtValue*>& initializers)
{
    OpTranslatorRegistry registry = BuildOpTranslatorRegistry();

    auto GetViName = [&](const OrtValueInfo* vi) -> std::string {
        if (!vi) return {};
        const char* n = nullptr;
        ort_api.GetValueInfoName(vi, &n);
        return n ? std::string(n) : std::string{};
    };
    auto GetOpType = [&](const OrtNode* node) -> std::string {
        const char* op = nullptr;
        ort_api.Node_GetOperatorType(node, &op);
        return op ? std::string(op) : "?";
    };

    // A tensor name counts as a resolved constant if it is a graph initializer or
    // has a fully-resolved static shape entry. (For shape-driving inputs we need
    // the VALUE to be known; initializers are the concrete case. A resolved_shapes
    // entry is accepted too — GetCapabilityImpl's fixpoint only records a name
    // there when it could evaluate the producing chain, e.g. a Shape->Slice feed.)
    auto IsResolvedConstant = [&](const std::string& name) -> bool {
        if (name.empty()) return false;
        if (initializers.count(name)) return true;
        auto it = resolved_shapes.find(name);
        return it != resolved_shapes.end() && !it->second.empty();
    };

    // Data-dependent-shape ops: map op_type -> the input index whose VALUE drives
    // the output shape. That input must be a resolved constant to admit the node.
    // NonZero is intentionally absent: its output shape depends on tensor CONTENTS
    // at runtime and can never be resolved from a snapshot, so it is always rejected
    // below via the "data-dependent but unlisted driver" fall-through.
    static const std::unordered_map<std::string, size_t> kShapeDrivingInput = {
        {"Reshape",         1},  // shape tensor
        {"Expand",          1},  // shape tensor
        {"Tile",            1},  // repeats tensor
        {"Resize",          3},  // sizes (opset>=11: roi,scales,sizes) — scales(2) also drives
        {"Upsample",        1},  // scales tensor
        {"Range",           0},  // start/limit/delta all drive; require input 0 constant as proxy
        {"ConstantOfShape", 0},  // shape tensor
        {"Split",           1},  // split tensor (opset>=13) — when present
    };
    // Ops whose output shape is purely data-dependent with NO constant driver we
    // can require — never admissible to the deferred path.
    static const std::unordered_set<std::string> kNeverAdmissible = { "NonZero" };

    for (const OrtNode* node : nodes) {
        if (!node) continue;

        const std::string op = GetOpType(node);

        // Must have a translator.
        if (!registry.count(op)) {
            DiagLog("[ValidateTier0Structural] FAIL: no translator for op=" + op + "\n");
            return false;
        }

        if (kNeverAdmissible.count(op)) {
            DiagLog("[ValidateTier0Structural] FAIL: op=" + op + " has purely data-dependent output shape\n");
            return false;
        }

        // No empty (extent-0) edge DML can't represent in a compiled graph.
        if (fusion_utils::NodeHasEmptyEdge(ort_api, node, initializers)) {
            DiagLog("[ValidateTier0Structural] FAIL: op=" + op + " has empty (extent-0) edge\n");
            return false;
        }

        size_t node_num_inputs = 0;
        ort_api.Node_GetNumInputs(node, &node_num_inputs);
        std::vector<const OrtValueInfo*> in_vis(node_num_inputs, nullptr);
        if (node_num_inputs > 0)
            ort_api.Node_GetInputs(node, in_vis.data(), node_num_inputs);
        auto input_names = fusion_utils::GetNodeInputNames(ort_api, node);

        // Dtype support on every typed input — but do NOT reject dynamic dims.
        for (size_t k = 0; k < node_num_inputs && k < input_names.size(); ++k) {
            if (in_vis[k] == nullptr || input_names[k].empty()) continue;
            fusion_utils::ValueInfoShape s = fusion_utils::GetValueInfoShape(ort_api, in_vis[k]);
            if (!s.has_type_info) continue;
            if (OnnxDtypeToDml(s.elem_type) == DML_TENSOR_DATA_TYPE_UNKNOWN) {
                DiagLog("[ValidateTier0Structural] FAIL input: op=" + op + " tensor='" + GetViName(in_vis[k]) + "' unsupported dtype\n");
                return false;
            }
        }

        // Dtype support on outputs.
        size_t node_num_outputs = 0;
        ort_api.Node_GetNumOutputs(node, &node_num_outputs);
        std::vector<const OrtValueInfo*> out_vis(node_num_outputs, nullptr);
        if (node_num_outputs > 0)
            ort_api.Node_GetOutputs(node, out_vis.data(), node_num_outputs);
        for (size_t k = 0; k < node_num_outputs; ++k) {
            if (out_vis[k] == nullptr) continue;
            fusion_utils::ValueInfoShape s = fusion_utils::GetValueInfoShape(ort_api, out_vis[k]);
            if (!s.has_type_info) continue;
            if (OnnxDtypeToDml(s.elem_type) == DML_TENSOR_DATA_TYPE_UNKNOWN) {
                DiagLog("[ValidateTier0Structural] FAIL output: op=" + op + " tensor='" + GetViName(out_vis[k]) + "' unsupported dtype\n");
                return false;
            }
        }

        // Data-dependent-shape op: the shape-driving input must be a resolved constant.
        auto dd_it = kShapeDrivingInput.find(op);
        if (dd_it != kShapeDrivingInput.end()) {
            const size_t drive_idx = dd_it->second;
            // If the driving input is absent (optional, e.g. Split without a split
            // tensor), the output shape comes from an attribute or even split — that
            // is resolvable, so only enforce the constant rule when the input exists.
            if (drive_idx < input_names.size() && !input_names[drive_idx].empty()) {
                if (!IsResolvedConstant(input_names[drive_idx])) {
                    DiagLog("[ValidateTier0Structural] FAIL: op=" + op + " shape-driving input '" +
                            input_names[drive_idx] + "' is not a resolved constant\n");
                    return false;
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
                if (!ort_api.TensorTypeAndShape_HasShape(shape_info)) {
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

        fusion_utils::ValueInfoShape s = fusion_utils::GetValueInfoShape(ort_api, vi);
        if (!s.has_type_info || !s.has_shape) continue;

        DML_TENSOR_DATA_TYPE dml_dtype = OnnxDtypeToDml(s.elem_type);
        if (dml_dtype == DML_TENSOR_DATA_TYPE_UNKNOWN) continue;

        const size_t rank = s.rank;
        std::vector<uint32_t> sizes;
        if (rank > 0) {
            std::vector<int64_t>& dims = s.dims;

            // Recover dynamic boundary-input dims from the Phase-1 resolved shapes.
            bool any_dynamic = false;
            for (size_t d = 0; d < rank; ++d) if (dims[d] <= 0) { any_dynamic = true; break; }
            if (any_dynamic) {
                auto rit = resolved_shapes.find(name);
                if (rit != resolved_shapes.end() && rit->second.size() == rank) {
                    for (size_t d = 0; d < rank; ++d)
                        if (dims[d] <= 0 && rit->second[d] > 0) dims[d] = rit->second[d];
                }
            }

            // If a dim is still dynamic after the fallback, leave the input unseeded
            // rather than coercing to 1. A coerced [1,1,1,3] would let translators
            // produce a bogus static output that propagates a wrong shape downstream;
            // leaving it unseeded makes dependent translators skip so the grouper keeps
            // this boundary out of the fused partition (those nodes run per-op).
            bool still_dynamic = false;
            for (size_t d = 0; d < rank; ++d) if (dims[d] <= 0) { still_dynamic = true; break; }
            if (still_dynamic) {
                DML_PERF_LOG("[BuildSubgraphInfo] SKIP graph-input '", name,
                             "' reason=unresolved_dynamic_input\n");
                continue;
            }

            sizes.resize(rank);
            for (size_t d = 0; d < rank; ++d)
                sizes[d] = static_cast<uint32_t>(dims[d]);
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

        // Capture constant inputs into all_initializers by reading each input's own
        // ValueInfo. Graph_GetInitializers does not enumerate every consumed constant
        // (a Constant-node output is not always a graph initializer), and under
        // drop_constant_initializers=true ORT may release those the EP does not hold —
        // so an uncaptured constant fails edge-wiring with "unresolved input" at
        // compile. ValueInfo_GetInitializerValue resolves the value directly.
        {
            auto in_names = fusion_utils::GetNodeInputNames(ort_api, node);
            size_t nnin = 0;
            ort_api.Node_GetNumInputs(node, &nnin);
            std::vector<const OrtValueInfo*> nin_vis(nnin, nullptr);
            if (nnin > 0) ort_api.Node_GetInputs(node, nin_vis.data(), nnin);
            for (size_t k = 0; k < nnin && k < in_names.size(); ++k) {
                if (!nin_vis[k] || in_names[k].empty()) continue;
                if (info.all_initializers.count(in_names[k])) continue;  // already captured
                const OrtValue* cval = nullptr;
                OrtStatus* cst = ort_api.ValueInfo_GetInitializerValue(nin_vis[k], &cval);
                if (cst) { ort_api.ReleaseStatus(cst); continue; }
                if (cval) {
                    info.all_initializers[in_names[k]] = cval;
                    // Seed value_shapes too — BuildDmlInputMap needs total_bytes to
                    // inline/slot the constant; without it the constant is skipped.
                    if (cval->IsAllocated() && !info.value_shapes.count(in_names[k])) {
                        OrtTensorTypeAndShapeInfo* csi = nullptr;
                        ort_api.GetTensorTypeAndShape(const_cast<OrtValue*>(cval), &csi);
                        if (csi) {
                            ONNXTensorElementDataType cdt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
                            ort_api.GetTensorElementType(csi, &cdt);
                            DML_TENSOR_DATA_TYPE cdml = OnnxDtypeToDml(cdt);
                            if (ort_api.TensorTypeAndShape_HasShape(csi) && cdml != DML_TENSOR_DATA_TYPE_UNKNOWN) {
                                size_t crank = 0;
                                ort_api.GetDimensionsCount(csi, &crank);
                                std::vector<int64_t> cdims(crank, 0);
                                if (crank > 0) ort_api.GetDimensions(csi, cdims.data(), crank);
                                std::vector<uint32_t> csizes(crank);
                                for (size_t d = 0; d < crank; ++d)
                                    csizes[d] = static_cast<uint32_t>(cdims[d] > 0 ? cdims[d] : 1);
                                auto cti = MakeTensorInfo(csizes, cdml);
                                cti.original_rank = static_cast<uint32_t>(crank);
                                info.value_shapes[in_names[k]] = cti;
                            }
                            ort_api.ReleaseTensorTypeAndShapeInfo(csi);
                        }
                    }
                }
            }
        }

        auto output_names = fusion_utils::GetNodeOutputNames(ort_api, node);
        size_t node_num_outputs = 0;
        ort_api.Node_GetNumOutputs(node, &node_num_outputs);
        std::vector<const OrtValueInfo*> out_vis(node_num_outputs, nullptr);
        if (node_num_outputs > 0)
            ort_api.Node_GetOutputs(node, out_vis.data(), node_num_outputs);

        for (size_t k = 0; k < node_num_outputs && k < output_names.size(); ++k) {
            if (!out_vis[k] || output_names[k].empty()) continue;
            if (info.value_shapes.count(output_names[k])) continue;
            fusion_utils::ValueInfoShape s = fusion_utils::GetValueInfoShape(ort_api, out_vis[k]);
            if (!s.has_type_info) {
                DML_PERF_LOG("[BuildSubgraphInfo] SKIP '", output_names[k], "' op=", (node_op ? node_op : "?"), " reason=no_typeinfo\n");
                continue;
            }
            if (!s.has_shape) {
                DML_PERF_LOG("[BuildSubgraphInfo] SKIP '", output_names[k], "' op=", (node_op ? node_op : "?"), " reason=no_shape\n");
                continue;
            }
            DML_TENSOR_DATA_TYPE dml_dt = OnnxDtypeToDml(s.elem_type);
            if (dml_dt == DML_TENSOR_DATA_TYPE_UNKNOWN) {
                DML_PERF_LOG("[BuildSubgraphInfo] SKIP '", output_names[k], "' op=", (node_op ? node_op : "?"), " reason=unsupported_dtype\n");
                continue;
            }
            const size_t out_rank = s.rank;
            std::vector<uint32_t> out_sizes;
            bool all_static = true;
            if (out_rank > 0) {
                std::vector<int64_t>& out_dims = s.dims;
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
            fusion_utils::ValueInfoShape s = fusion_utils::GetValueInfoShape(ort_api, in_vis_scan[k]);
            if (!s.has_type_info || !s.has_shape) continue;
            DML_TENSOR_DATA_TYPE dml_dt = OnnxDtypeToDml(s.elem_type);
            if (dml_dt == DML_TENSOR_DATA_TYPE_UNKNOWN) continue;
            const size_t in_rank = s.rank;
            std::vector<uint32_t> in_sizes;
            bool all_static = true;
            if (in_rank > 0) {
                std::vector<int64_t>& in_dims = s.dims;
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
            // dtype only — dims come from resolved_shapes, so shape presence is
            // not required here (has_type_info is enough to read the element type).
            fusion_utils::ValueInfoShape s = fusion_utils::GetValueInfoShape(ort_api, vi);
            if (!s.has_type_info) return;
            DML_TENSOR_DATA_TYPE dml_dt = OnnxDtypeToDml(s.elem_type);
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
            // Get actual tensor byte count to avoid reading past the ORT tensor.
            OrtTensorTypeAndShapeInfo* tsi = nullptr;
            ort_api.GetTensorTypeAndShape(const_cast<OrtValue*>(init_it->second), &tsi);
            size_t elem_count = 0;
            ONNXTensorElementDataType dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            if (tsi) {
                ort_api.GetTensorShapeElementCount(tsi, &elem_count);
                ort_api.GetTensorElementType(tsi, &dt);
                ort_api.ReleaseTensorTypeAndShapeInfo(tsi);
            }
            uint64_t actual_bytes = elem_count * DmlDataTypeSize(OnnxDtypeToDml(dt));
            // DataSize = min(actualTensorBytes, totalTensorSizeInBytes) to
            // avoid reading past the ORT tensor's allocation.
            uint64_t data_size = (actual_bytes > 0 && actual_bytes < bytes) ? actual_bytes : bytes;

            ConstantNodeInfo cni;
            cni.name = init_name;
            cni.data.resize(static_cast<size_t>(data_size), 0);
            std::memcpy(cni.data.data(), cpu_ptr, static_cast<size_t>(data_size));
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

// Name-based sibling of BuildDmlInputMap for the deferred/snapshot path, which
// has graph-input NAMES (owned) rather than live const OrtValueInfo*. Identical
// logic; the only difference is Pass 2 reads names[i] instead of
// GetValueInfoName(vis[i]). Kept separate (not merged) so the static
// BuildDmlInputMap stays byte-identical during the fork. TODO: converge.
static DmlInputMapResult BuildDmlInputMapByNames(
    const OrtApi& ort_api,
    const std::vector<std::string>& graph_input_names,
    const std::unordered_set<std::string>& consumed_initializer_names,
    const std::unordered_map<std::string, DmlTensorInfo>& value_shapes,
    const std::unordered_map<std::string, const OrtValue*>& all_initializers)
{
    DmlInputMapResult result;
    result.subgraph_to_dml_input.assign(graph_input_names.size(), SIZE_MAX);

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
            OrtTensorTypeAndShapeInfo* tsi = nullptr;
            ort_api.GetTensorTypeAndShape(const_cast<OrtValue*>(init_it->second), &tsi);
            size_t elem_count = 0;
            ONNXTensorElementDataType dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            if (tsi) {
                ort_api.GetTensorShapeElementCount(tsi, &elem_count);
                ort_api.GetTensorElementType(tsi, &dt);
                ort_api.ReleaseTensorTypeAndShapeInfo(tsi);
            }
            uint64_t actual_bytes = elem_count * DmlDataTypeSize(OnnxDtypeToDml(dt));
            uint64_t data_size = (actual_bytes > 0 && actual_bytes < bytes) ? actual_bytes : bytes;

            ConstantNodeInfo cni;
            cni.name = init_name;
            cni.data.resize(static_cast<size_t>(data_size), 0);
            std::memcpy(cni.data.data(), cpu_ptr, static_cast<size_t>(data_size));
            result.constant_node_map[init_name] = result.constant_nodes.size();
            result.constant_nodes.push_back(std::move(cni));
        }
        if (st) ort_api.ReleaseStatus(st);
    }

    // Pass 2: assign DML graph input indices from the graph-input names.
    size_t dml_input_idx = 0;
    for (size_t i = 0; i < graph_input_names.size(); ++i) {
        const std::string& n = graph_input_names[i];
        if (n.empty()) continue;
        if (result.constant_node_map.count(n)) continue;  // inlined
        result.dml_input_map[n] = dml_input_idx;
        result.subgraph_to_dml_input[i] = dml_input_idx;
        ++dml_input_idx;
    }

    // Pass 3: Mode B initializers not in the graph-input list.
    for (const auto& init_name : consumed_initializer_names) {
        if (result.constant_node_map.count(init_name)) continue;
        if (result.dml_input_map.count(init_name)) continue;
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

        InitBinding init_binding;
        if (FAILED(provider->AllocatePooledResource(
                static_cast<size_t>(bytes), AllocatorRoundingMode::Disabled,
                init_binding.gpu_resource.GetAddressOf(), init_binding.allocator_ref.GetAddressOf())))
            return false;
        if (FAILED(provider->UploadToResource(init_binding.gpu_resource.Get(), cpu_ptr, upload_bytes)))
            return false;
        init_binding.bytes = bytes;
        const_graph_input_bindings[dml_idx] = std::move(init_binding);
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

// Name-based sibling of UploadInitializers for the deferred/snapshot path.
// Identical body; Mode A iterates graph-input NAMES instead of live
// const OrtValueInfo*. Kept separate so static UploadInitializers stays
// byte-identical during the fork. TODO: converge.
static bool UploadInitializersByNames(
    const OrtApi& ort_api,
    PluginDmlExecutionProviderImpl* provider,
    const DmlInputMapResult& input_map,
    const std::unordered_set<size_t>& owned_graph_input_indices,
    const std::vector<std::string>& graph_input_names,
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

        InitBinding init_binding;
        if (FAILED(provider->AllocatePooledResource(
                static_cast<size_t>(bytes), AllocatorRoundingMode::Disabled,
                init_binding.gpu_resource.GetAddressOf(), init_binding.allocator_ref.GetAddressOf())))
            return false;
        if (FAILED(provider->UploadToResource(init_binding.gpu_resource.Get(), cpu_ptr, upload_bytes)))
            return false;
        init_binding.bytes = bytes;
        const_graph_input_bindings[dml_idx] = std::move(init_binding);
        return true;
    };

    for (size_t i = 0; i < graph_input_names.size(); ++i) {
        if (input_map.subgraph_to_dml_input[i] == SIZE_MAX) continue;
        const std::string& n = graph_input_names[i];
        if (!n.empty() && !upload_one(n, input_map.subgraph_to_dml_input[i]))
            return false;
    }
    for (const auto& name : input_map.ordered_initializer_names) {
        auto di_it = input_map.dml_input_map.find(name);
        if (di_it != input_map.dml_input_map.end() && !upload_one(name, di_it->second))
            return false;
    }
    return true;
}

// Resolve a name through passthrough aliases and check if the original or
// resolved name is an initializer. Returns the initializer name if found,
// or empty string if not an initializer.
static std::string ResolveToInitializer(
    const std::string& name,
    const std::unordered_map<std::string, std::string>& aliases,
    const std::unordered_map<std::string, const OrtValue*>& initializers) {
    if (initializers.count(name)) return name;
    auto resolved = name;
    while (aliases.count(resolved))
        resolved = aliases.at(resolved);
    if (initializers.count(resolved)) return resolved;
    return {};
}

// Drain the D3D12 info queue for DML debug-layer validation messages. These
// appear when DML_CREATE_DEVICE_FLAG_DEBUG is set and describe exactly which
// tensor, edge, descriptor, or binding parameter is invalid. Gated behind
// DML_PERF_LOG (compiles to nothing without DML_PERF_PROFILE).
static void DrainDmlDebugMessages(PluginDmlExecutionProviderImpl* provider,
                                  const char* stage) {
    ComPtr<ID3D12Device> d3d12_device_for_iq;
    if (SUCCEEDED(provider->GetD3DDevice(d3d12_device_for_iq.GetAddressOf()))) {
        ComPtr<ID3D12InfoQueue> info_queue;
        if (SUCCEEDED(d3d12_device_for_iq.As(&info_queue))) {
            UINT64 msg_count = info_queue->GetNumStoredMessages();
            DML_PERF_LOG("[Compile] D3D12InfoQueue (", stage, "): ", msg_count, " messages\n");
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

    SubgraphInfo subgraph = BuildSubgraphInfo(ort_api, fused_subgraph, initializers, resolved_shapes);
    const size_t num_graph_inputs  = subgraph.graph_input_vis.size();
    const size_t num_graph_outputs = subgraph.graph_output_vis.size();

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

        auto translated = reg_it->second(ort_api, NodeView{ort_api, node}, subgraph.value_shapes, subgraph.all_initializers);
        if (!translated) {
            DML_PERF_LOG("[Compile] FAIL: translator returned nullopt for op=", op_type, "\n");
            return nullptr;
        }

        DML_PERF_LOG("[Compile] OK: op=", op_type, "\n");

        // Write-back: translator computed output shapes; seed them for downstream.
        for (size_t k = 0; k < output_names.size() && k < translated->output_tensors.size(); ++k) {
            if (!subgraph.value_shapes.count(output_names[k])) {
                subgraph.value_shapes[output_names[k]] = translated->output_tensors[k];
            }
        }

        auto translated_ptr = std::make_unique<TranslatedOp>(std::move(*translated));
        translated_ptr->FixupPointers();

        // A passthrough (shape-only) op is normally elided from the DML graph.
        // But if any of its outputs is a partition graph output, that output
        // needs a real DML node to source its outgoing output edge — an elided
        // alias has none, which later fails as "unresolved output". In that
        // case materialize a real ELEMENT_WISE_IDENTITY (as ORT does), so the
        // output is produced by an actual node.
        bool passthrough_is_graph_output = false;
        if (translated_ptr->passthrough) {
            for (const auto& oname : output_names)
                if (subgraph.graph_output_map.count(oname)) { passthrough_is_graph_output = true; break; }
        }

        if (translated_ptr->passthrough && !passthrough_is_graph_output) {
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

        if (passthrough_is_graph_output) {
            // Replace the elided passthrough with a real identity op that copies
            // input[0] to the output. Fall through to the normal compiled-node
            // path so its input edge and output edge are wired like any op.
            DmlTensorInfo out_info = translated_ptr->output_tensors.empty()
                ? DmlTensorInfo{} : translated_ptr->output_tensors[0];
            translated_ptr = std::make_unique<TranslatedOp>(BuildIdentityOp(out_info));
            DML_PERF_LOG("[Compile] passthrough op=", op_type,
                " materialized as IDENTITY (output is a partition output)\n");
        }

        // Track which input names will become actual DML edges. The DML operator
        // only wires input_tensors.size() primary inputs (plus sub_node graph_inputs).
        // Inputs beyond that limit (e.g. Clip's min/max at slots 1,2) are consumed
        // at translation time only and won't have edges in the DML graph.
        size_t dml_input_count = translated_ptr->input_tensors.size();
        size_t num_primary_inputs = translated_ptr->primary_input_count.value_or(dml_input_count);
        for (size_t s = 0; s < input_names.size() && s < num_primary_inputs; ++s) {
            size_t name_idx = translated_ptr->input_name_reorder.empty()
                ? s : translated_ptr->input_name_reorder[s];
            if (name_idx >= input_names.size()) continue;
            const auto& in_name = input_names[name_idx];
            dml_consumed_names.insert(in_name);
            auto init_name = ResolveToInitializer(in_name, graph_input_aliases, subgraph.all_initializers);
            if (!init_name.empty())
                consumed_initializer_names.insert(init_name);
        }
        for (const auto& sn : translated_ptr->sub_nodes) {
            for (const auto& [onnx_idx, _] : sn.graph_inputs) {
                if (onnx_idx < input_names.size()) {
                    dml_consumed_names.insert(input_names[onnx_idx]);
                    auto init_name = ResolveToInitializer(input_names[onnx_idx], graph_input_aliases, subgraph.all_initializers);
                    if (!init_name.empty())
                        consumed_initializer_names.insert(init_name);
                }
            }
        }

        // Track which DML graph node produces each output value.
        // When sub_nodes exist, the last sub_node is the output producer.
        size_t producer_compiled_idx = compiled_nodes.size();
        for (size_t k = 0; k < output_names.size(); ++k)
            value_producer[output_names[k]] = { producer_compiled_idx, k };

        CompiledNode compiled_node;
        compiled_node.translated = std::move(translated_ptr);
        compiled_node.input_names = std::move(input_names);
        compiled_node.output_names = std::move(output_names);
        compiled_node.op_type = op_type;
        compiled_nodes.push_back(std::move(compiled_node));
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
        for (const auto& [name, _] : subgraph.graph_output_map)
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
            const auto& compiled_node = compiled_nodes[i];
            size_t num_primary_inputs = compiled_node.translated->primary_input_count.value_or(
                compiled_node.translated->input_tensors.size());
            for (size_t s = 0; s < compiled_node.input_names.size() && s < num_primary_inputs; ++s) {
                size_t name_idx = compiled_node.translated->input_name_reorder.empty()
                    ? s : compiled_node.translated->input_name_reorder[s];
                if (name_idx < compiled_node.input_names.size())
                    wired_inputs[i].push_back(compiled_node.input_names[name_idx]);
            }
            for (const auto& sn : compiled_node.translated->sub_nodes) {
                for (const auto& [onnx_idx, _] : sn.graph_inputs) {
                    if (onnx_idx < compiled_node.input_names.size())
                        wired_inputs[i].push_back(compiled_node.input_names[onnx_idx]);
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
    for (size_t node_idx = 0; node_idx < compiled_nodes.size(); ++node_idx) {
        if (!node_is_live[node_idx]) continue;
        auto& compiled_node = compiled_nodes[node_idx];
        size_t dml_input_count = compiled_node.translated->input_tensors.size();
        for (size_t s = 0; s < compiled_node.input_names.size() && s < dml_input_count; ++s) {
            size_t name_idx = compiled_node.translated->input_name_reorder.empty()
                ? s : compiled_node.translated->input_name_reorder[s];
            if (name_idx >= compiled_node.input_names.size()) continue;
            auto resolved = compiled_node.input_names[name_idx];
            while (graph_input_aliases.count(resolved))
                resolved = graph_input_aliases[resolved];
            if (!subgraph.all_initializers.count(compiled_node.input_names[name_idx]) && !subgraph.all_initializers.count(resolved))
                continue;

            auto shape_it = subgraph.value_shapes.find(
                subgraph.all_initializers.count(resolved) ? resolved : compiled_node.input_names[name_idx]);
            uint64_t bytes = (shape_it != subgraph.value_shapes.end()) ? shape_it->second.total_bytes : 0;
            if (bytes >= kMaxConstNodeDataSize) {
                compiled_node.translated->input_buffer_descs[s].Flags |= DML_TENSOR_FLAG_OWNED_BY_DML;
                ++owned_count;
            }
        }
        compiled_node.translated->FixupPointers();
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
        auto& compiled_node = compiled_nodes[i];
        HRESULT hr = dml_device->CreateOperator(
            &compiled_node.translated->op_desc, IID_PPV_ARGS(compiled_node.translated->dml_operator.GetAddressOf()));
        if (FAILED(hr)) {
            DML_PERF_LOG("[Compile] FAIL: CreateOperator HR=", Hex(static_cast<uint32_t>(hr)),
                " op=", compiled_node.op_type, "\n");
            return nullptr;
        }
        DML_PERF_LOG("[Compile] CreateOperator OK: op=", compiled_node.op_type, "\n");

        for (size_t s = 0; s < compiled_node.translated->sub_nodes.size(); ++s) {
            auto& sn = compiled_node.translated->sub_nodes[s];
            hr = dml_device->CreateOperator(
                &sn.op_desc, IID_PPV_ARGS(sn.dml_operator.GetAddressOf()));
            if (FAILED(hr)) {
                DML_PERF_LOG("[Compile] FAIL: CreateOperator sub_node[", s, "] HR=",
                    Hex(static_cast<uint32_t>(hr)), " op=", compiled_node.op_type, "\n");
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

    DmlInputMapResult input_map = BuildDmlInputMap(
        ort_api, subgraph.graph_input_vis,
        consumed_initializer_names, subgraph.value_shapes, subgraph.all_initializers);

    // Build owned_graph_input_indices: DML graph input indices whose tensors
    // carry DML_TENSOR_FLAG_OWNED_BY_DML. Uses input_map.dml_input_map so both Mode A
    // and Mode B initializers are covered.
    std::unordered_set<size_t> owned_graph_input_indices;
    for (auto& compiled_node : compiled_nodes) {
        size_t dml_input_count = compiled_node.translated->input_tensors.size();
        for (size_t s = 0; s < compiled_node.input_names.size() && s < dml_input_count; ++s) {
            if (compiled_node.translated->input_buffer_descs[s].Flags & DML_TENSOR_FLAG_OWNED_BY_DML) {
                size_t name_idx = compiled_node.translated->input_name_reorder.empty()
                    ? s : compiled_node.translated->input_name_reorder[s];
                if (name_idx >= compiled_node.input_names.size()) continue;
                auto resolved_name = compiled_node.input_names[name_idx];
                while (graph_input_aliases.count(resolved_name))
                    resolved_name = graph_input_aliases[resolved_name];
                auto di_it = input_map.dml_input_map.find(resolved_name);
                if (di_it != input_map.dml_input_map.end())
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
    std::vector<DML_CONSTANT_DATA_GRAPH_NODE_DESC> const_node_descs(input_map.constant_nodes.size());
    for (size_t c = 0; c < input_map.constant_nodes.size(); ++c) {
        const_node_descs[c].Data = input_map.constant_nodes[c].data.data();
        const_node_descs[c].DataSize = input_map.constant_nodes[c].data.size();
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
        const auto& compiled_node = compiled_nodes[node_idx];
        size_t primary_dml_idx = dml_node_offset[node_idx];

        // Only wire edges for as many inputs as the primary DML operator has.
        // primary_input_count: nullopt = all inputs; N (incl. genuine 0, e.g. GQA's
        // input-less FILL primary) = exactly N. See the field doc in
        // dml_op_translators.h — optional disambiguates explicit-0 from unset.
        size_t num_primary_inputs = compiled_node.translated->primary_input_count.value_or(
            compiled_node.translated->input_tensors.size());
        for (size_t input_slot = 0; input_slot < compiled_node.input_names.size() && input_slot < num_primary_inputs; ++input_slot) {
            size_t name_idx = compiled_node.translated->input_name_reorder.empty()
                ? input_slot : compiled_node.translated->input_name_reorder[input_slot];
            if (name_idx >= compiled_node.input_names.size()) continue;
            auto name = compiled_node.input_names[name_idx];
            if (name.empty()) continue;

            // Resolve the DML schema slot for this edge. For dense operators the
            // packed input_slot equals the schema slot. For sparse operators (e.g.
            // MultiHeadAttention with non-contiguous slots) dml_input_slot_indices
            // maps packed position → schema slot so ToNodeInputIndex is correct.
            const size_t dml_schema_slot = compiled_node.translated->dml_input_slot_indices.empty()
                ? input_slot : compiled_node.translated->dml_input_slot_indices[input_slot];

            // Resolve passthrough aliases (e.g. Reshape output → original input).
            while (graph_input_aliases.count(name))
                name = graph_input_aliases[name];

            // Small constant inlined as a constant node.
            auto const_it = input_map.constant_node_map.find(name);
            if (const_it != input_map.constant_node_map.end()) {
                DML_INTERMEDIATE_GRAPH_EDGE_DESC edge{};
                edge.FromNodeIndex = static_cast<UINT>(const_node_base + const_it->second);
                edge.FromNodeOutputIndex = 0;
                edge.ToNodeIndex = static_cast<UINT>(primary_dml_idx);
                edge.ToNodeInputIndex = static_cast<UINT>(dml_schema_slot);
                intermediate_edge_storage.push_back(edge);
                continue;
            }

            // DML graph input (runtime tensor or large initializer).
            auto dml_in_it = input_map.dml_input_map.find(name);
            if (dml_in_it != input_map.dml_input_map.end()) {
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
                "' for op=", compiled_node.op_type, " slot=", input_slot, "\n");
            return nullptr; // unresolved input
        }

        // Wire internal edges for sub_nodes.
        for (size_t s = 0; s < compiled_node.translated->sub_nodes.size(); ++s) {
            const auto& sn = compiled_node.translated->sub_nodes[s];
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
                if (onnx_idx >= compiled_node.input_names.size()) continue;
                auto gi_name = compiled_node.input_names[onnx_idx];
                while (graph_input_aliases.count(gi_name))
                    gi_name = graph_input_aliases[gi_name];

                auto const_it = input_map.constant_node_map.find(gi_name);
                if (const_it != input_map.constant_node_map.end()) {
                    DML_INTERMEDIATE_GRAPH_EDGE_DESC edge{};
                    edge.FromNodeIndex = static_cast<UINT>(const_node_base + const_it->second);
                    edge.FromNodeOutputIndex = 0;
                    edge.ToNodeIndex = static_cast<UINT>(sn_dml_idx);
                    edge.ToNodeInputIndex = static_cast<UINT>(to_input);
                    intermediate_edge_storage.push_back(edge);
                    continue;
                }
                auto dml_in_it = input_map.dml_input_map.find(gi_name);
                if (dml_in_it != input_map.dml_input_map.end()) {
                    DML_INPUT_GRAPH_EDGE_DESC edge{};
                    edge.GraphInputIndex = static_cast<UINT>(dml_in_it->second);
                    edge.ToNodeIndex = static_cast<UINT>(sn_dml_idx);
                    edge.ToNodeInputIndex = static_cast<UINT>(to_input);
                    input_edge_storage.push_back(edge);
                    continue;
                }
                // Partition-internal producer (e.g. a Resize feeding a concat
                // sub_node dequant). Mirror the primary-input producer logic.
                auto prod_it = value_producer.find(gi_name);
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
                    edge.ToNodeIndex = static_cast<UINT>(sn_dml_idx);
                    edge.ToNodeInputIndex = static_cast<UINT>(to_input);
                    intermediate_edge_storage.push_back(edge);
                }
            }
        }
    }

    // Output edges.
    for (const auto& [name, out_idx] : subgraph.graph_output_map) {
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
    graph_desc.InputCount = static_cast<UINT>(input_map.total_dml_inputs);
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
            const auto& compiled_node = compiled_nodes[i];
            for (const auto& t : compiled_node.translated->input_tensors)
                if (t.data_type == DML_TENSOR_DATA_TYPE_FLOAT16) has_fp16 = true;
                else if (t.data_type == DML_TENSOR_DATA_TYPE_FLOAT32) has_fp32 = true;
            for (const auto& t : compiled_node.translated->output_tensors)
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
        DrainDmlDebugMessages(provider, "CompileGraph");
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

    std::vector<InitBinding> const_graph_input_bindings(input_map.total_dml_inputs);
    if (!UploadInitializers(
            ort_api, provider, input_map, owned_graph_input_indices,
            subgraph.graph_input_vis, subgraph.value_shapes, subgraph.all_initializers,
            const_graph_input_bindings))
        return nullptr;

    // Build init input bindings array indexed by DML graph input index.
    std::vector<DML_BUFFER_BINDING> init_input_bindings(input_map.total_dml_inputs, DML_BUFFER_BINDING{});
    for (size_t i = 0; i < const_graph_input_bindings.size(); ++i) {
        if (const_graph_input_bindings[i].gpu_resource) {
            auto& init_binding = const_graph_input_bindings[i];
            init_input_bindings[i] = { init_binding.gpu_resource.Get(), 0, init_binding.bytes };
        }
    }

    const DML_BUFFER_BINDING* persistent_ptr =
        persistent_binding ? &*persistent_binding : nullptr;

    if (FAILED(provider->InitializeOperator(
            compiled_op.Get(), persistent_ptr,
            gsl::make_span(init_input_bindings)))) {
        DrainDmlDebugMessages(provider, "InitializeOperator");
        return nullptr;
    }

    // Flush to execute pending GPU work. If the device was removed (e.g. due
    // to an invalid binding), GetDeviceRemovedReason throws. Catch it so one
    // bad partition returns nullptr (per-op fallback) instead of aborting.
    ORT_TRY {
        provider->Flush();
        DrainDmlDebugMessages(provider, "Flush");
    }
    ORT_CATCH(const std::exception& e) {
        ORT_HANDLE_EXCEPTION([&]() {
            DML_PERF_LOG("[Compile] FAIL: init/flush threw (device removed?): ", e.what(), "\n");
        });
        DrainDmlDebugMessages(provider, "Flush-threw");
        return nullptr;
    }

    provider->QueueReference(compiled_op.Get());
    if (persistent_allocator)
        provider->QueueReference(persistent_allocator.Get());
    for (auto& init_binding : const_graph_input_bindings) {
        if (init_binding.gpu_resource) provider->QueueReference(init_binding.gpu_resource.Get());
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
    kernel_state->num_runtime_inputs = input_map.total_dml_inputs;
    kernel_state->num_subgraph_inputs = num_graph_inputs;
    kernel_state->subgraph_to_dml_input = input_map.subgraph_to_dml_input;
    kernel_state->dml_inputs_with_edges = dml_inputs_with_edges;
    kernel_state->runtime_input_bytes.resize(input_map.total_dml_inputs);
    kernel_state->runtime_input_is_owned.resize(input_map.total_dml_inputs, false);
    kernel_state->num_initializers = 0;

    // Mode A: populate from subgraph graph inputs (runtime + initializers).
    for (size_t i = 0; i < num_graph_inputs; ++i) {
        if (input_map.subgraph_to_dml_input[i] == SIZE_MAX) continue;
        size_t dml_idx = input_map.subgraph_to_dml_input[i];
        std::string name = ReadValueInfoName(ort_api, subgraph.graph_input_vis[i]);
        auto it = subgraph.value_shapes.find(name);
        kernel_state->runtime_input_bytes[dml_idx] =
            (it != subgraph.value_shapes.end()) ? it->second.total_bytes : 0;
        if (owned_graph_input_indices.count(dml_idx))
            kernel_state->runtime_input_is_owned[dml_idx] = true;
    }

    // Mode B: populate for initializers added as extra DML inputs.
    // These are always OWNED (large initializers baked into persistent resource).
    for (const auto& name : input_map.ordered_initializer_names) {
        auto di_it = input_map.dml_input_map.find(name);
        if (di_it == input_map.dml_input_map.end()) continue;
        size_t dml_idx = di_it->second;
        auto it = subgraph.value_shapes.find(name);
        kernel_state->runtime_input_bytes[dml_idx] =
            (it != subgraph.value_shapes.end()) ? it->second.total_bytes : 0;
        if (owned_graph_input_indices.count(dml_idx))
            kernel_state->runtime_input_is_owned[dml_idx] = true;
    }

    kernel_state->num_outputs     = num_graph_outputs;
    kernel_state->output_dims.resize(num_graph_outputs);
    kernel_state->output_bytes.resize(num_graph_outputs);

    for (size_t i = 0; i < num_graph_outputs; ++i) {
        std::string name = ReadValueInfoName(ort_api, subgraph.graph_output_vis[i]);
        auto it = subgraph.value_shapes.find(name);
        if (it == subgraph.value_shapes.end()) continue;

        // Use ORT shape inference (value_shapes) for output dims and bytes.
        // The producer node's translated output shape can differ from the
        // graph output shape when passthrough nodes (Reshape, Unsqueeze,
        // Squeeze, Flatten) sit between them — value_producer aliases
        // through passthrough nodes, but the graph output shape reflects
        // the final reshaped dimensions.
        const auto& vs = it->second;
        const OrtValueInfo* out_vi = subgraph.graph_output_vis[i];
        // Determine the true (unpadded) output rank. Prefer ORT's shape inference
        // on the graph-output ValueInfo; a rank of 0 there means a genuine scalar
        // and MUST be honored. Only when ORT has no shape info at all do we fall
        // back to the tensor's own original_rank, then to the 4D-padded size.
        //
        // Getting this wrong reports a scalar/low-rank output as the padded rank-4
        // [1,1,1,1], which changes ONNX broadcast semantics for a downstream
        // consumer — e.g. a materialized Squeeze->identity feeding Min/Max, where
        // [1,1,1,1] vs [] injects a spurious dimension and explodes shapes.
        bool rank_known = false;
        size_t orig_rank = 0;
        fusion_utils::ValueInfoShape ovs = fusion_utils::GetValueInfoShape(ort_api, out_vi);
        if (ovs.has_type_info && ovs.has_shape) {
            orig_rank = ovs.rank;
            rank_known = true;  // includes rank==0 (scalar)
        }
        if (!rank_known) orig_rank = vs.original_rank ? vs.original_rank : vs.sizes.size();
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

// ===========================================================================
// Runtime-fusion (deferred-compile) support
// ===========================================================================

DeferredSubgraph::~DeferredSubgraph() {
    if (ort_api) {
        for (auto& [name, val] : owned_initializers) {
            if (val) ort_api->ReleaseValue(val);
        }
    }
    owned_initializers.clear();
    initializer_view.clear();
}

namespace {

// Copy one live initializer OrtValue's bytes into a freshly-allocated owned
// OrtValue (CPU). Returns nullptr on any failure. The owned value's lifetime is
// managed by the DeferredSubgraph (released in its dtor).
OrtValue* CloneInitializerValue(const OrtApi& ort_api, const OrtValue* src) {
    if (!src) return nullptr;

    // Read type + shape of the source.
    OrtTensorTypeAndShapeInfo* tsi = nullptr;
    if (ort_api.GetTensorTypeAndShape(const_cast<OrtValue*>(src), &tsi) != nullptr || !tsi)
        return nullptr;
    ONNXTensorElementDataType dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    ort_api.GetTensorElementType(tsi, &dt);
    size_t rank = 0;
    ort_api.GetDimensionsCount(tsi, &rank);
    std::vector<int64_t> dims(rank, 0);
    if (rank > 0) ort_api.GetDimensions(tsi, dims.data(), rank);
    size_t elem_count = 0;
    ort_api.GetTensorShapeElementCount(tsi, &elem_count);
    ort_api.ReleaseTensorTypeAndShapeInfo(tsi);

    // Source CPU bytes.
    void* src_ptr = nullptr;
    if (ort_api.GetTensorMutableData(const_cast<OrtValue*>(src), &src_ptr) != nullptr || !src_ptr)
        return nullptr;

    // Allocate an owned tensor of the same type/shape and copy bytes in.
    OrtAllocator* allocator = nullptr;
    if (ort_api.GetAllocatorWithDefaultOptions(&allocator) != nullptr || !allocator)
        return nullptr;
    OrtValue* dst = nullptr;
    if (ort_api.CreateTensorAsOrtValue(allocator, dims.data(), dims.size(), dt, &dst) != nullptr || !dst)
        return nullptr;

    void* dst_ptr = nullptr;
    if (ort_api.GetTensorMutableData(dst, &dst_ptr) != nullptr || !dst_ptr) {
        ort_api.ReleaseValue(dst);
        return nullptr;
    }
    const size_t bytes = elem_count * DmlDataTypeSize(OnnxDtypeToDml(dt));
    if (bytes > 0) std::memcpy(dst_ptr, src_ptr, bytes);
    return dst;
}

}  // namespace

// Build the owned DeferredSubgraph snapshot from the LIVE fused subgraph, while
// it is still valid (i.e. before Compile returns). Captures owned node
// snapshots (with pre-materialized attribute protos + original ranks),
// initializer BYTES, base resolved shapes, and graph-input names.
static std::unique_ptr<DeferredSubgraph> BuildDeferredSnapshot(
    const OrtApi&                                            ort_api,
    const OrtGraph*                                          fused_subgraph,
    const std::unordered_map<std::string, const OrtValue*>&  initializers,
    const std::unordered_map<std::string, std::vector<int64_t>>& resolved_shapes,
    PluginDmlExecutionProviderImpl*                          provider,
    FullGraphKernelState*                                    kernel_state)
{
    auto snap = std::make_unique<DeferredSubgraph>();
    snap->ort_api = &ort_api;

    // --- Nodes (topological order) ---
    size_t num_nodes = 0;
    ort_api.Graph_GetNumNodes(fused_subgraph, &num_nodes);
    std::vector<const OrtNode*> nodes(num_nodes, nullptr);
    if (num_nodes > 0)
        ort_api.Graph_GetNodes(fused_subgraph, nodes.data(), num_nodes);

    snap->nodes.reserve(num_nodes);
    for (const OrtNode* node : nodes) {
        if (!node) continue;
        DeferredNode dn;
        ort_api.Node_GetId(node, &dn.id);
        {
            const char* s = nullptr;
            if (ort_api.Node_GetName(node, &s) == nullptr && s) dn.name = s;
            s = nullptr;
            if (ort_api.Node_GetOperatorType(node, &s) == nullptr && s) dn.op_type = s;
            s = nullptr;
            if (ort_api.Node_GetDomain(node, &s) == nullptr && s) dn.domain = s;
        }
        ort_api.Node_GetSinceVersion(node, &dn.since_version);

        // Track the max since_version per domain → opset_imports for InferShapes.
        if (dn.since_version > 0) {
            int& v = snap->opset_imports[dn.domain];
            if (dn.since_version > v) v = dn.since_version;
        }

        dn.input_names  = fusion_utils::GetNodeInputNames(ort_api, node);
        dn.output_names = fusion_utils::GetNodeOutputNames(ort_api, node);

        // Capture original (unpadded) ranks per input/output name, mirroring what
        // rank-sensitive translators (e.g. ScatterND) read off the live node.
        auto capture_ranks = [&](bool is_input) {
            size_t n = 0;
            if (is_input) ort_api.Node_GetNumInputs(node, &n);
            else          ort_api.Node_GetNumOutputs(node, &n);
            std::vector<const OrtValueInfo*> vis(n, nullptr);
            if (n > 0) {
                if (is_input) ort_api.Node_GetInputs(node, vis.data(), n);
                else          ort_api.Node_GetOutputs(node, vis.data(), n);
            }
            const auto& names = is_input ? dn.input_names : dn.output_names;
            for (size_t k = 0; k < n && k < names.size(); ++k) {
                if (!vis[k] || names[k].empty()) continue;
                fusion_utils::ValueInfoShape s = fusion_utils::GetValueInfoShape(ort_api, vis[k]);
                if (s.has_type_info && s.has_shape)
                    dn.original_ranks[names[k]] = s.rank;
            }
        };
        capture_ranks(true);
        capture_ranks(false);

        // Pre-materialize every attribute into an owned AttributeProto (translators
        // read attrs lazily; the borrowed OrtOpAttr* would dangle after Compile).
        size_t num_attrs = 0;
        ort_api.Node_GetNumAttributes(node, &num_attrs);
        if (num_attrs > 0) {
            std::vector<const OrtOpAttr*> attrs(num_attrs, nullptr);
            ort_api.Node_GetAttributes(node, attrs.data(), num_attrs);
            for (const OrtOpAttr* attr : attrs) {
                if (!attr) continue;
                std::string aname = dml_ep::GetOpAttrName(attr, ort_api);
                if (aname.empty()) continue;
                auto proto = dml_ep::BuildPluginAttributeProto(attr, ort_api);
                if (proto)
                    dn.attr_protos[aname] = std::shared_ptr<ONNX_NAMESPACE::AttributeProto>(std::move(proto));
            }
        }

        snap->nodes.push_back(std::move(dn));
    }

    // --- Graph input names (runtime input order) ---
    size_t num_inputs = 0;
    ort_api.Graph_GetNumInputs(fused_subgraph, &num_inputs);
    std::vector<const OrtValueInfo*> in_vis(num_inputs, nullptr);
    if (num_inputs > 0)
        ort_api.Graph_GetInputs(fused_subgraph, in_vis.data(), num_inputs);
    snap->graph_input_names.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        std::string nm = ReadValueInfoName(ort_api, in_vis[i]);
        snap->graph_input_names.push_back(nm);
        // Capture the element type for ONNX shape inference (B2). Borrowed/zero-alloc.
        if (in_vis[i] && !nm.empty()) {
            fusion_utils::ValueInfoShape s = fusion_utils::GetValueInfoShape(ort_api, in_vis[i]);
            if (s.has_type_info && s.elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED)
                snap->graph_input_dtypes[nm] = s.elem_type;
        }
    }

    // ai.onnx opset floor: some nodes report since_version=0 (schema-less/contrib);
    // ensure a usable ONNX opset so InferShapes has a version for standard ops.
    if (snap->opset_imports[""] < 7) snap->opset_imports[""] = 21;
    // com.microsoft contrib ops are opset 1.
    if (snap->opset_imports.count("com.microsoft") == 0)
        snap->opset_imports["com.microsoft"] = 1;

    // --- Initializer bytes: clone every initializer this partition may consume ---
    // CRITICAL: the parent `initializers` / m_graphInitializerMap OrtValue* were
    // valid during GetCapabilityImpl but are FREED by ORT when GetCapability
    // returns — dereferencing them here is a use-after-free (segfault). Source the
    // initializers ONLY from the LIVE fused subgraph (valid for CompileImpl's
    // lifetime), exactly as the static BuildSubgraphInfo does. The parent map is
    // used for NAME lookup only, never dereferenced.
    (void)initializers;  // names only; do not dereference its pointers

    // Large weights (>= kSnapshotDirectUploadBytes) are uploaded to VRAM DIRECTLY from
    // the LIVE OrtValue here at snapshot time, and NOT cloned into host RAM. Uploading
    // straight from the live external-data OrtValue (valid for CompileImpl's lifetime)
    // keeps only ONE copy of the weights resident (the VRAM one), mirroring ORT's
    // incremental CreateResource+RemoveInitializedTensor. Cloning them to host instead
    // would keep a second full-size CPU copy alive until the first-Compute upload,
    // doubling peak host footprint on a UMA box. Small control initializers
    // (< threshold: axes/scales/shape tensors that translators read on every variant
    // compile) are still cloned to host — they are KB, not the concern.
    //
    // A large weight uploaded here is left in the snapshot as: initializer_view[name] =
    // nullptr (dims-only), with dtype+dims recorded in freed_initializer_shapes, and its
    // GPU resource in kernel_state->shared_initializers[name]. This is the same state the
    // per-variant "upload then free CPU copy" block produced after variant 1 — so the
    // existing seed / InferShapes-dims-only / bind-by-name paths all already handle it,
    // and the per-variant upload+free blocks in CompileFromSnapshot become no-ops (guarded
    // by "already uploaded" / "not an owned copy").
    static constexpr uint64_t kSnapshotDirectUploadBytes = 1u << 20;  // 1 MB, matches small-initializer cutoff
    uint64_t direct_upload_bytes_since_drain = 0;
    size_t direct_uploaded = 0;
    uint64_t direct_uploaded_mb = 0;

    auto add_owned = [&](const std::string& name, const OrtValue* src) {
        if (name.empty() || !src || !src->IsAllocated()) return;
        if (snap->owned_initializers.count(name)) return;
        if (snap->initializer_view.count(name)) return;  // already handled (e.g. direct-uploaded)

        // Measure the tensor (dtype/dims/bytes) from the live OrtValue.
        OrtTensorTypeAndShapeInfo* tsi = nullptr;
        if (ort_api.GetTensorTypeAndShape(const_cast<OrtValue*>(src), &tsi) != nullptr || !tsi) {
            // Can't measure — fall back to cloning (keeps old behavior for odd cases).
            OrtValue* owned = CloneInitializerValue(ort_api, src);
            if (owned) { snap->owned_initializers[name] = owned; snap->initializer_view[name] = owned; }
            return;
        }
        ONNXTensorElementDataType dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        ort_api.GetTensorElementType(tsi, &dt);
        size_t rank = 0;
        ort_api.GetDimensionsCount(tsi, &rank);
        std::vector<int64_t> dims(rank, 0);
        if (rank > 0) ort_api.GetDimensions(tsi, dims.data(), rank);
        size_t elem_count = 0;
        ort_api.GetTensorShapeElementCount(tsi, &elem_count);
        ort_api.ReleaseTensorTypeAndShapeInfo(tsi);
        uint64_t bytes = static_cast<uint64_t>(elem_count) * DmlDataTypeSize(OnnxDtypeToDml(dt));

        // Small control initializer, or no provider/state to hold a GPU copy → clone to host.
        if (bytes < kSnapshotDirectUploadBytes || !provider || !kernel_state) {
            OrtValue* owned = CloneInitializerValue(ort_api, src);
            if (owned) { snap->owned_initializers[name] = owned; snap->initializer_view[name] = owned; }
            return;
        }

        // Large weight → upload directly to VRAM from the live OrtValue; no host clone.
        void* cpu_ptr = nullptr;
        OrtStatus* st = ort_api.GetTensorMutableData(const_cast<OrtValue*>(src), &cpu_ptr);
        if (st || !cpu_ptr) {
            if (st) ort_api.ReleaseStatus(st);
            // Fall back to cloning if the bytes aren't directly readable.
            OrtValue* owned = CloneInitializerValue(ort_api, src);
            if (owned) { snap->owned_initializers[name] = owned; snap->initializer_view[name] = owned; }
            return;
        }

        FullGraphKernelState::SharedInitializer si;
        if (FAILED(provider->AllocatePooledResource(
                static_cast<size_t>(bytes), AllocatorRoundingMode::Disabled,
                si.gpu_resource.GetAddressOf(), si.allocator_ref.GetAddressOf()))) {
            OrtValue* owned = CloneInitializerValue(ort_api, src);
            if (owned) { snap->owned_initializers[name] = owned; snap->initializer_view[name] = owned; }
            return;
        }
        if (FAILED(provider->UploadToResource(si.gpu_resource.Get(), cpu_ptr, bytes))) {
            OrtValue* owned = CloneInitializerValue(ort_api, src);
            if (owned) { snap->owned_initializers[name] = owned; snap->initializer_view[name] = owned; }
            return;
        }
        si.bytes = bytes;
        kernel_state->shared_initializers[name] = std::move(si);

        // Record dtype+dims so later value_shapes seeding + InferShapes (dims-only) work
        // without the tensor, and mark the view null (dims-only) — mirrors the post-free state.
        DeferredSubgraph::FreedInitializerInfo fi;
        fi.onnx_dtype = dt;
        fi.dims = std::move(dims);
        snap->freed_initializer_shapes[name] = std::move(fi);
        snap->initializer_view[name] = nullptr;  // NAME present, dims-only (no bytes)

        ++direct_uploaded;
        direct_uploaded_mb += bytes / (1024 * 1024);

        // Bound peak staging: drain the UPLOAD heap every ~512 MB so it doesn't balloon
        // during the burst (same rationale as the old per-variant upload loop).
        direct_upload_bytes_since_drain += bytes;
        if (direct_upload_bytes_since_drain >= (512ull * 1024 * 1024)) {
            provider->WaitForOutstandingWork();
            direct_upload_bytes_since_drain = 0;
        }
    };
    // Fused-subgraph initializers (fresh, live OrtValue*).
    {
        size_t num_init = 0;
        ort_api.Graph_GetNumInitializers(fused_subgraph, &num_init);
        if (num_init > 0) {
            std::vector<const OrtValueInfo*> init_vis(num_init, nullptr);
            ort_api.Graph_GetInitializers(fused_subgraph, init_vis.data(), num_init);
            for (const OrtValueInfo* vi : init_vis) {
                if (!vi) continue;
                std::string n = ReadValueInfoName(ort_api, vi);
                const OrtValue* val = nullptr;
                OrtStatus* st = ort_api.ValueInfo_GetInitializerValue(vi, &val);
                if (st) { ort_api.ReleaseStatus(st); continue; }
                add_owned(n, val);
            }
        }
    }
    // Also pull any fused-subgraph-local initializers exposed on graph inputs.
    for (size_t i = 0; i < num_inputs; ++i) {
        if (!in_vis[i]) continue;
        const OrtValue* cval = nullptr;
        if (ort_api.ValueInfo_GetInitializerValue(in_vis[i], &cval) == nullptr && cval) {
            std::string n = ReadValueInfoName(ort_api, in_vis[i]);
            add_owned(n, cval);
        }
    }

    // Finalize the direct-to-VRAM uploads: complete pending copies and trim the staging
    // heap so it is not held resident during decode. The per-variant upload block in
    // CompileFromSnapshot then short-circuits on shared_initializers.count(name).
    if (direct_uploaded > 0 && provider) {
        provider->WaitForOutstandingWork();
        provider->TrimUploadHeap();
        DML_PERF_LOG("[RuntimeFusion] snapshot direct-uploaded weights: ", direct_uploaded,
                     " tensors, ", direct_uploaded_mb, " MB to VRAM (no host clone)\n");
    }

    DML_PERF_LOG("[RuntimeFusion] snapshot built: nodes=", snap->nodes.size(),
                 " host_initializers=", snap->owned_initializers.size(),
                 " direct_uploaded=", direct_uploaded,
                 " graph_inputs=", snap->graph_input_names.size(), "\n");
    return snap;
}

OrtNodeComputeInfo* FullGraphFusion::CompileDeferred(
    const OrtApi&                                            ort_api,
    const OrtGraph*                                          fused_subgraph,
    const std::unordered_map<std::string, const OrtValue*>&  initializers,
    PluginDmlExecutionProviderImpl*                          provider,
    const std::unordered_map<std::string, std::vector<int64_t>>& resolved_shapes,
    const std::vector<DeferredNode>&                        shape_prep_nodes,
    const std::vector<std::string>&                         shape_prep_roots,
    const std::unordered_map<std::string, std::vector<int64_t>>& shape_prep_inits)
{
    // Create the kernel state FIRST so BuildDeferredSnapshot can upload large weights
    // directly into kernel_state->shared_initializers at snapshot time, avoiding a
    // full-size host clone. Small control initializers are still cloned into the snapshot.
    auto kernel_state = std::make_unique<FullGraphKernelState>();
    kernel_state->provider = provider;
    kernel_state->ort_api  = &ort_api;
    kernel_state->deferred = true;

    auto snap = BuildDeferredSnapshot(ort_api, fused_subgraph, initializers, resolved_shapes,
                                      provider, kernel_state.get());
    if (!snap || snap->nodes.empty()) {
        DML_PERF_LOG("[RuntimeFusion] CompileDeferred FAIL: empty snapshot\n");
        return nullptr;
    }
    // Carry the captured boundary shape-math chain into the snapshot (owned copy).
    snap->shape_prep_nodes = shape_prep_nodes;
    snap->shape_prep_root_inputs = shape_prep_roots;
    snap->shape_prep_initializers = shape_prep_inits;
    DML_PERF_LOG("[RuntimeFusion] CompileDeferred: shape_prep_nodes=", shape_prep_nodes.size(),
                 " roots=", shape_prep_roots.size(), " inits=", shape_prep_inits.size(), "\n");

    kernel_state->snapshot = std::move(snap);

    auto* info = new FullGraphNodeComputeInfo();
    info->state = std::move(kernel_state);
    DML_PERF_LOG("[RuntimeFusion] CompileDeferred OK: snapshot stored, compile deferred to first Compute\n");
    return info;
}

// ---------------------------------------------------------------------------
// FoldShapePrepChain — constant-fold the snapshot's captured shape-math nodes
// (Shape/Concat/Gather/Cast/Slice/Mul/Div/...) from the CONCRETE runtime dims of
// the partition's root inputs (e.g. input_ids). Mirrors the GetCapability value
// evaluator (dml_ep.cc:906-1021) but reads from owned DeferredNodes instead of
// live OrtNode*. Produces a name -> int64 values map for every foldable output.
//
// This recovers data-dependent reshape TARGETS that are only knowable once real
// input dims are present (e.g. the pos_ids Reshape target from Shape(input_ids)
// -> Concat). The results seed the InferShapes GraphProto as constant
// initializers so downstream Reshape/Expand outputs resolve correctly.
// ---------------------------------------------------------------------------
static std::unordered_map<std::string, std::vector<int64_t>> FoldShapePrepChain(
    const OrtApi&                                               ort_api,
    const DeferredSubgraph&                                     snap,
    const std::vector<std::vector<int64_t>>&                    runtime_input_dims)
{
    std::unordered_map<std::string, std::vector<int64_t>> vals;  // value name -> data

    // Seed root inputs with their CONCRETE runtime shape as the "Shape-of" source.
    // We store the DIMS keyed by input name so a Shape node over that input yields
    // these dims. (A Shape op consumes a tensor and emits its dim vector.)
    std::unordered_map<std::string, std::vector<int64_t>> root_dims;
    for (const auto& rn : snap.shape_prep_root_inputs) {
        // Find the runtime dims for this root input by its position in graph_input_names.
        for (size_t i = 0; i < snap.graph_input_names.size() && i < runtime_input_dims.size(); ++i) {
            if (snap.graph_input_names[i] == rn) { root_dims[rn] = runtime_input_dims[i]; break; }
        }
    }

    // Seed constant initializer values the chain consumes (Gather index, Concat/
    // Slice consts) — captured from the full graph at GetCapability. Without these
    // a Gather/Slice on a shape vector cannot fold (its index/bounds are unknown).
    for (const auto& [name, iv] : snap.shape_prep_initializers)
        if (!name.empty() && !iv.empty()) vals[name] = iv;

    auto get = [&](const std::string& n) -> const std::vector<int64_t>* {
        auto it = vals.find(n);
        return it == vals.end() ? nullptr : &it->second;
    };

    // Fixpoint over the captured chain (already topological, but loop to be safe).
    bool changed = true;
    int guard = 0;
    while (changed && guard++ < 16) {
        changed = false;
        for (const auto& dn : snap.shape_prep_nodes) {
            if (dn.output_names.empty() || dn.output_names[0].empty()) continue;
            const std::string& out = dn.output_names[0];
            if (vals.count(out)) continue;  // already folded
            const std::string& op = dn.op_type;

            if (op == "Shape") {
                if (dn.input_names.empty()) continue;
                auto it = root_dims.find(dn.input_names[0]);
                const std::vector<int64_t>* src = (it != root_dims.end()) ? &it->second
                                                 : get(dn.input_names[0]);
                if (src && !src->empty()) { vals[out] = *src; changed = true; }
            } else if (op == "Concat") {
                std::vector<int64_t> o; bool all = true;
                for (const auto& in : dn.input_names) {
                    const auto* v = get(in);
                    if (!v) { all = false; break; }
                    o.insert(o.end(), v->begin(), v->end());
                }
                if (all && !o.empty()) { vals[out] = std::move(o); changed = true; }
            } else if (op == "Cast" || op == "Unsqueeze" || op == "Squeeze" || op == "Identity") {
                // On a shape/dims VECTOR, Cast is a no-op and Unsqueeze/Squeeze only
                // change rank (e.g. [d]→[1,d]); the underlying values are unchanged.
                // For 1-D shape math the value list passes through verbatim.
                if (dn.input_names.empty()) continue;
                const auto* v = get(dn.input_names[0]);
                if (v) { vals[out] = *v; changed = true; }
            } else if (op == "Gather") {
                if (dn.input_names.size() < 2) continue;
                const auto* data = get(dn.input_names[0]);
                const auto* idx  = get(dn.input_names[1]);
                if (data && idx) {
                    std::vector<int64_t> o;
                    for (int64_t ix : *idx) {
                        int64_t i = ix < 0 ? ix + static_cast<int64_t>(data->size()) : ix;
                        if (i >= 0 && i < static_cast<int64_t>(data->size())) o.push_back((*data)[i]);
                    }
                    if (!o.empty()) { vals[out] = std::move(o); changed = true; }
                }
            } else if (op == "Mul" || op == "Div") {
                if (dn.input_names.size() < 2) continue;
                const auto* a = get(dn.input_names[0]);
                const auto* b = get(dn.input_names[1]);
                if (a && b && !a->empty() && !b->empty()) {
                    size_t len = std::max(a->size(), b->size());
                    std::vector<int64_t> o(len);
                    for (size_t i = 0; i < len; ++i) {
                        int64_t av = (*a)[i % a->size()], bv = (*b)[i % b->size()];
                        o[i] = (op == "Mul") ? av * bv : (bv != 0 ? av / bv : 0);
                    }
                    vals[out] = std::move(o); changed = true;
                }
            } else if (op == "Slice") {
                if (dn.input_names.empty()) continue;
                const auto* data = get(dn.input_names[0]);
                if (!data) continue;
                std::vector<int64_t> starts, ends;
                if (dn.input_names.size() >= 3) {
                    if (const auto* s = get(dn.input_names[1])) starts = *s;
                    if (const auto* e = get(dn.input_names[2])) ends = *e;
                }
                if (starts.empty() || ends.empty()) {
                    OrtNodeAdapter adapter(NodeView{dn}, ort_api);
                    auto sa = adapter.GetAttributeInts("starts");
                    auto ea = adapter.GetAttributeInts("ends");
                    if (!sa.empty() && !ea.empty()) { starts = sa; ends = ea; }
                }
                if (!starts.empty() && !ends.empty()) {
                    int64_t s = starts[0], e = ends[0];
                    if (s < 0) s += static_cast<int64_t>(data->size());
                    if (e < 0) e += static_cast<int64_t>(data->size());
                    if (e > static_cast<int64_t>(data->size())) e = static_cast<int64_t>(data->size());
                    std::vector<int64_t> o;
                    for (int64_t i = s; i < e; ++i) if (i >= 0) o.push_back((*data)[i]);
                    if (!o.empty()) { vals[out] = std::move(o); changed = true; }
                }
            }
        }
    }

    for (const auto& [name, v] : vals) {
        std::string s; for (int64_t x : v) s += std::to_string(x) + ",";
        DML_PERF_LOG("[RuntimeFusion] folded shape-value '", name, "' = [", s, "]\n");
    }
    return vals;
}

// ---------------------------------------------------------------------------
// RunSnapshotShapeInference — build an ONNX GraphProto from the owned snapshot
// (nodes + attrs + initializer bytes + graph-input value_info with the CONCRETE
// runtime dims), run onnx::shape_inference::InferShapes over it, and seed
// `value_shapes` with every intermediate shape/dtype the inference resolves.
//
// ONNX shape inference over the snapshot: the plugin has no shape-propagation
// engine of its own, and ~20+ translators READ (never compute) their output
// shape from value_shapes.
// On a dynamic decoder, ORT's build-time inference gives [-1,-1,...], so the
// intermediates are only knowable once concrete input dims arrive at first
// Compute. We reproduce ORT's DmlRuntimeFusedGraphKernel approach: bind concrete
// input dims, then delegate to ONNX's per-operator shape-inference functions.
//
// Best-effort: initializer VALUE bytes are included so data-dependent shape ops
// (Reshape's shape input, etc.) resolve. Anything InferShapes can't resolve is
// left unseeded; the translate loop then fails cleanly (ORT_FAIL at the ABI
// boundary), exactly as before.
// ---------------------------------------------------------------------------
static void RunSnapshotShapeInference(
    const OrtApi&                                   ort_api,
    const DeferredSubgraph&                         snap,
    const std::vector<std::vector<int64_t>>&        runtime_input_dims,
    std::unordered_map<std::string, DmlTensorInfo>& value_shapes)
{
    namespace onnx = ONNX_NAMESPACE;
    onnx::GraphProto g;
    g.set_name("rf_partition");

    // Constant-fold the captured boundary shape-math chain from concrete runtime
    // dims → recovers data-dependent reshape targets (e.g. pos_ids). These folded
    // values are injected as constant initializers below so ONNX InferShapes can
    // resolve the downstream Reshape/Expand output shapes. Replaces the unsound
    // "read the runtime shape-tensor buffer" path (buffer is unpopulated at
    // compile time → 0xCDCD garbage).
    std::unordered_map<std::string, std::vector<int64_t>> folded_values =
        FoldShapePrepChain(ort_api, snap, runtime_input_dims);

    // --- Nodes (already topologically ordered in the snapshot) ---
    for (const auto& dn : snap.nodes) {
        onnx::NodeProto* np = g.add_node();
        np->set_op_type(dn.op_type);
        if (!dn.domain.empty()) np->set_domain(dn.domain);
        if (!dn.name.empty())   np->set_name(dn.name);
        for (const auto& in : dn.input_names)  np->add_input(in);
        for (const auto& on : dn.output_names) np->add_output(on);
        for (const auto& [aname, proto] : dn.attr_protos)
            if (proto) *np->add_attribute() = *proto;   // owned AttributeProto copy
    }

    // --- Initializers: copy dtype + dims. Copy raw bytes ONLY for SMALL initializers
    //     so value-dependent shape inference (Reshape/Expand/Slice/…) can read them.
    //
    // Large weight matrices (the multi-MB/GB MatMul/embedding tensors that make up ~all of
    // a big model's initializer bytes) NEVER drive shape inference — they feed MatMul/Gather
    // as the DATA operand, whose output shape needs only dims. Embedding their full bytes
    // here would build a second full copy of every weight inside this GraphProto, doubling
    // host footprint on a large model. Emitting dims-only for large weights removes that copy.
    //
    // The cutoff is deliberately generous (1 MB): every shape/axes/indices/scalar/norm
    // control tensor is < 1 KB and keeps its bytes; only genuine weight matrices (>> 1 MB)
    // are stripped. Keeping small tensors' bytes is a superset of what the decode path needs
    // (it runs InferShapes after all large weights are already dims-only — see the
    // freed_initializer_shapes branch below), and extra correct initializer data can only
    // help inference.
    static constexpr size_t kShapeInferMaxRawDataBytes = 1u << 20;  // 1 MB
    for (const auto& [name, val] : snap.initializer_view) {
        if (name.empty() || !val) continue;
        OrtTensorTypeAndShapeInfo* tsi = nullptr;
        if (ort_api.GetTensorTypeAndShape(const_cast<OrtValue*>(val), &tsi) != nullptr || !tsi)
            continue;
        ONNXTensorElementDataType dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        ort_api.GetTensorElementType(tsi, &dt);
        size_t rank = 0;
        ort_api.GetDimensionsCount(tsi, &rank);
        std::vector<int64_t> dims(rank, 0);
        if (rank > 0) ort_api.GetDimensions(tsi, dims.data(), rank);
        size_t elem_count = 1;
        for (size_t d = 0; d < rank; ++d) elem_count *= (dims[d] > 0 ? static_cast<size_t>(dims[d]) : 0);
        ort_api.ReleaseTensorTypeAndShapeInfo(tsi);

        onnx::TensorProto* tp = g.add_initializer();
        tp->set_name(name);
        tp->set_data_type(static_cast<int32_t>(dt));   // ONNXTensorElementDataType == TensorProto::DataType numerically
        for (int64_t d : dims) tp->add_dims(d);
        size_t elem_size = 0;
        switch (dt) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: elem_size = 4; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: elem_size = 8; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  elem_size = 2; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    elem_size = 1; break;
            default: elem_size = 0; break;
        }
        // Large weight → dims-only (skip raw bytes). Small control tensor → copy bytes.
        if (elem_size == 0 || elem_count == 0) continue;
        if (elem_count * elem_size >= kShapeInferMaxRawDataBytes) continue;
        const void* src = nullptr;
        if (ort_api.GetTensorData(const_cast<OrtValue*>(val), &src) == nullptr && src)
            tp->set_raw_data(src, elem_count * elem_size);
    }

    // --- Freed weights: emit DIMS-ONLY TensorProtos (no raw bytes). Their CPU copy
    //     was released after upload (FreeUploadedWeightCpuCopies), so they are absent
    //     from initializer_view above — but ONNX InferShapes still needs their dtype +
    //     dims to propagate shapes through MatMul/Gather (without them the whole chain
    //     collapses and every downstream output goes unseeded). Their VALUES never
    //     drive shape inference (only small control initializers do, and those are
    //     never freed), so dims-only is sufficient. Reconstruct the original ONNX dims
    //     from the cached DmlTensorInfo (strip the 4D left-padding via original_rank).
    for (const auto& [name, fi] : snap.freed_initializer_shapes) {
        if (name.empty()) continue;
        // Skip only if a LIVE (non-null) OrtValue is still present — the main loop
        // above already emitted it with real bytes. Freed weights are kept in
        // initializer_view mapped to nullptr, so a plain .count() would wrongly skip
        // them; check the value is actually null/absent before emitting dims-only.
        auto iv_it = snap.initializer_view.find(name);
        if (iv_it != snap.initializer_view.end() && iv_it->second) continue;  // still-live wins
        onnx::TensorProto* tp = g.add_initializer();
        tp->set_name(name);
        tp->set_data_type(static_cast<int32_t>(fi.onnx_dtype));  // ONNXTensorElementDataType == TensorProto::DataType numerically
        for (int64_t d : fi.dims) tp->add_dims(d);
    }

    // --- Folded boundary shape-values as constant initializers. These are the
    //     data-dependent reshape/expand TARGETS (e.g. the pos_ids Reshape target
    //     = Shape(input_ids)->Concat) computed by FoldShapePrepChain from concrete
    //     runtime dims. Injecting them as constants lets ONNX InferShapes resolve
    //     the downstream Reshape/Expand output shapes. Replaces the unsound
    //     read-the-runtime-buffer path (buffer unpopulated at compile time). ---
    // A shape-math node captured for folding may ALSO be a real node inside the
    // partition (the Phase-4 group and the final fusion can diverge). If a folded
    // name is produced by a partition node, do NOT inject it as an initializer —
    // that would double-define the value in the proto. Only inject values that are
    // BOUNDARY inputs (produced outside the partition).
    std::unordered_set<std::string> partition_produced;
    for (const auto& dn : snap.nodes)
        for (const auto& on : dn.output_names) if (!on.empty()) partition_produced.insert(on);
    for (const auto& [name, vals] : folded_values) {
        if (name.empty() || vals.empty()) continue;
        if (snap.initializer_view.count(name)) continue;  // real initializer wins
        if (partition_produced.count(name)) continue;      // produced by a partition node
        onnx::TensorProto* tp = g.add_initializer();
        tp->set_name(name);
        tp->set_data_type(static_cast<int32_t>(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
        tp->add_dims(static_cast<int64_t>(vals.size()));
        for (int64_t v : vals) tp->add_int64_data(v);
    }

    // --- Graph inputs: value_info with CONCRETE runtime dims + captured dtype ---
    auto add_input_vi = [&](const std::string& name, const std::vector<int64_t>& dims,
                            ONNXTensorElementDataType dt) {
        onnx::ValueInfoProto* vip = g.add_input();
        vip->set_name(name);
        auto* tt = vip->mutable_type()->mutable_tensor_type();
        tt->set_elem_type(static_cast<int32_t>(dt));
        auto* shp = tt->mutable_shape();
        for (int64_t d : dims) shp->add_dim()->set_dim_value(d);
    };
    for (size_t i = 0; i < snap.graph_input_names.size(); ++i) {
        const std::string& name = snap.graph_input_names[i];
        if (name.empty()) continue;
        if (snap.initializer_view.count(name)) continue;  // initializer, not a runtime input
        if (folded_values.count(name)) continue;           // injected as constant initializer above
        std::vector<int64_t> dims;
        if (i < runtime_input_dims.size()) dims = runtime_input_dims[i];
        auto dt_it = snap.graph_input_dtypes.find(name);
        ONNXTensorElementDataType dt = (dt_it != snap.graph_input_dtypes.end())
            ? dt_it->second : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        add_input_vi(name, dims, dt);
    }

    // --- Graph outputs: every produced value not consumed internally. ONNX only
    //     fully types values reachable to a declared output, so declare them
    //     (type left blank → inference fills it). Mirrors CompileFromSnapshot's
    //     graph-output derivation. ---
    {
        std::unordered_set<std::string> all_consumed;
        for (const auto& dn : snap.nodes)
            for (const auto& in : dn.input_names) if (!in.empty()) all_consumed.insert(in);
        std::unordered_set<std::string> declared;
        for (const auto& dn : snap.nodes)
            for (const auto& on : dn.output_names)
                if (!on.empty() && !all_consumed.count(on) && declared.insert(on).second)
                    g.add_output()->set_name(on);
    }

    // --- Run ONNX shape inference over the concrete-dim graph ---
    onnx::ShapeInferenceOptions opts;
    opts.error_mode = 0;                 // do not throw on per-op inference failure
    opts.enable_data_propagation = true; // propagate shape values (Shape→Reshape chains)
    onnx::shape_inference::InferShapes(&g, snap.opset_imports,
                                       onnx::OpSchemaRegistry::Instance(), opts);

    // Build a name -> TRUE ONNX rank map from the snapshot's captured original
    // ranks (recorded from the live ValueInfo at GetCapability, before 4D padding).
    // ONNX InferShapes can NORMALIZE a value's rank (e.g. emit a 4D shape for a
    // contrib op whose true rank is 3), which would make rank-branching translators
    // (RotaryEmbedding's inputIs4D, Gather's axis math) take the wrong branch. The
    // static path preserves the true rank (ExportDims, fgf.cc:3758); mirror that
    // here by preferring the captured rank over the InferShapes dim_size.
    std::unordered_map<std::string, size_t> true_rank;
    for (const auto& dn : snap.nodes)
        for (const auto& [nm, r] : dn.original_ranks)
            if (!nm.empty()) true_rank[nm] = r;

    // --- Seed value_shapes from every inferred value_info (and graph outputs) ---
    size_t seeded = 0;
    auto seed_from_vip = [&](const onnx::ValueInfoProto& vip) {
        const std::string& name = vip.name();
        if (name.empty() || value_shapes.count(name)) return;   // keep concrete seeds
        if (!vip.has_type() || !vip.type().has_tensor_type()) return;
        const auto& tt = vip.type().tensor_type();
        if (!tt.has_shape()) return;
        DML_TENSOR_DATA_TYPE dml_dt = OnnxDtypeToDml(
            static_cast<ONNXTensorElementDataType>(tt.elem_type()));
        if (dml_dt == DML_TENSOR_DATA_TYPE_UNKNOWN) return;
        const auto& shp = tt.shape();
        std::vector<uint32_t> sizes(static_cast<size_t>(shp.dim_size()));
        bool all_known = true;
        for (int d = 0; d < shp.dim_size(); ++d) {
            if (shp.dim(d).has_dim_value() && shp.dim(d).dim_value() > 0)
                sizes[d] = static_cast<uint32_t>(shp.dim(d).dim_value());
            else { sizes[d] = 1; all_known = false; }
        }
        if (!all_known) return;   // still-symbolic → leave unseeded (fail clean later)
        auto ti = MakeTensorInfo(sizes, dml_dt);
        // Prefer the captured TRUE rank; fall back to the InferShapes dim_size.
        // Only trust the captured rank if it does not exceed the inferred dim count
        // (a smaller true rank means InferShapes left-padded; strip that padding
        // from the reported original_rank so translators branch correctly).
        auto tr_it = true_rank.find(name);
        size_t inferred_rank = static_cast<size_t>(shp.dim_size());
        size_t orig_rank = (tr_it != true_rank.end() && tr_it->second <= inferred_rank)
            ? tr_it->second : inferred_rank;
        ti.original_rank = static_cast<uint32_t>(orig_rank);
        value_shapes[name] = ti;
        ++seeded;
    };
    for (const auto& vip : g.value_info()) seed_from_vip(vip);
    for (const auto& vip : g.output())     seed_from_vip(vip);

    DML_PERF_LOG("[RuntimeFusion] InferShapes seeded ", seeded,
                 " intermediate shapes (value_info=", g.value_info_size(),
                 " outputs=", g.output_size(), ")\n");
}

// ---------------------------------------------------------------------------
// CompileFromSnapshot — the deferred first-Compute compile.
//
// Forked from Compile (static path left untouched; converge later). Sources all
// nodes/attrs/initializers from the owned DeferredSubgraph instead of a live
// OrtGraph, and seeds graph-input value_shapes from the CONCRETE runtime dims
// observed at first Compute. Produces one CompiledVariant. Name-based throughout
// (no const OrtValueInfo*). Returns false on any failure.
// ---------------------------------------------------------------------------
static bool CompileFromSnapshot(
    const OrtApi&                                            ort_api,
    DeferredSubgraph&                                        snap,
    const std::vector<std::vector<int64_t>>&                 runtime_input_dims,
    PluginDmlExecutionProviderImpl*                          provider,
    FullGraphKernelState*                                    kernel_state,
    CompiledVariant&                                         out_variant)
{
    DML_PERF_LOG("[RuntimeFusion] CompileFromSnapshot ENTER: nodes=", snap.nodes.size(),
                 " initializers=", snap.initializer_view.size(),
                 " graph_inputs=", snap.graph_input_names.size(), "\n");

    // =======================================================================
    // ONE-TIME contrib-schema registration.
    //
    // The com.microsoft ops (GroupQueryAttention, RotaryEmbedding, MatMulNBits,
    // SkipSimplifiedLayerNormalization, QuickGelu, ...) are compiled into the
    // plugin but their schemas are NOT registered on the active path. Register
    // them once here so onnx::shape_inference can resolve them when inferring
    // shapes over the snapshot below. Without this, InferShapes has no contrib
    // schemas and the deferred decoder path fails.
    //
    // The standard ai.onnx schemas register fine (635 core ops, covering
    // everything the decoder uses). In Debug builds an ONNX completeness assert
    // in the fetched onnx schema.cc must be neutralized (it throws 635!=761 on
    // first registry touch for 126 unused ai.onnx.ml ops); Release compiles the
    // assert out. patches/onnx/onnx.patch neutralizes the assert for Debug builds.
    // =======================================================================
    static std::once_flag s_rf_contrib_once;
    std::call_once(s_rf_contrib_once, []() {
        ORT_TRY {
            // The com.microsoft* domains must be added to DomainToVersionRange
            // BEFORE their schemas register, else CheckDomainAndVersionToRegister
            // rejects each schema ("domain is not known by the checker",
            // onnx schema.h:1087-1102). ORT does this in environment.cc:274-292
            // right before RegisterContribSchemas(). Mirror that sequence here.
            auto& dvr = ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance();
            if (dvr.Map().find(onnxruntime::kMSDomain) == dvr.Map().end())
                dvr.AddDomainToVersion(onnxruntime::kMSDomain, 1, 1);
            dvr.AddDomainToVersion(onnxruntime::kMSExperimentalDomain, 1, 1);
            dvr.AddDomainToVersion(onnxruntime::kMSNchwcDomain, 1, 1);
            auto onnx_version = dvr.LastReleaseVersionMap()
                                    .find(ONNX_NAMESPACE::ONNX_DOMAIN)->second;
            dvr.AddDomainToVersion(onnxruntime::kMSInternalNHWCDomain, 1, onnx_version);
            dvr.AddDomainToVersion(onnxruntime::kPytorchAtenDomain, 1, 1);
            dvr.AddDomainToVersion(onnxruntime::kMSDmlDomain, 1, 1);

            // Ops declared via ONNX_MS_OPERATOR_SET_SCHEMA (GroupQueryAttention,
            // RotaryEmbedding, SkipSimplifiedLayerNormalization, ... in
            // bert_defs.cc) register through the OpSet_Microsoft_ver1 opset class,
            // NOT through RegisterContribSchemas()'s inline body. ORT calls both
            // (environment.cc:297 then :301). Register the opset first.
            ONNX_NAMESPACE::RegisterOpSetSchema<onnxruntime::contrib::OpSet_Microsoft_ver1>();
            onnxruntime::contrib::RegisterContribSchemas();
            DML_PERF_LOG("[RuntimeFusion] MS opset + contrib schemas registered\n");
        }
        ORT_CATCH(const std::exception& e) {
            ORT_HANDLE_EXCEPTION([&]() {
                DML_PERF_LOG("[RuntimeFusion] contrib registration threw: ", e.what(), "\n");
            });
        }
    });

    // -----------------------------------------------------------------------
    // Seed value_shapes: initializers (from owned bytes) + graph inputs
    // (concrete runtime dims) + node outputs (filled by translator write-back).
    // -----------------------------------------------------------------------
    std::unordered_map<std::string, DmlTensorInfo> value_shapes;
    const auto& all_initializers = snap.initializer_view;

    // Initializer shapes from the owned OrtValues.
    for (const auto& [name, val] : all_initializers) {
        if (!val) continue;
        OrtTensorTypeAndShapeInfo* tsi = nullptr;
        if (ort_api.GetTensorTypeAndShape(const_cast<OrtValue*>(val), &tsi) != nullptr || !tsi)
            continue;
        ONNXTensorElementDataType dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        ort_api.GetTensorElementType(tsi, &dt);
        size_t rank = 0;
        ort_api.GetDimensionsCount(tsi, &rank);
        std::vector<int64_t> dims(rank, 0);
        if (rank > 0) ort_api.GetDimensions(tsi, dims.data(), rank);
        ort_api.ReleaseTensorTypeAndShapeInfo(tsi);
        DML_TENSOR_DATA_TYPE dml_dt = OnnxDtypeToDml(dt);
        if (dml_dt == DML_TENSOR_DATA_TYPE_UNKNOWN) continue;
        std::vector<uint32_t> sizes(rank);
        for (size_t d = 0; d < rank; ++d) sizes[d] = static_cast<uint32_t>(dims[d] > 0 ? dims[d] : 1);
        auto ti = MakeTensorInfo(sizes, dml_dt);
        ti.original_rank = static_cast<uint32_t>(rank);
        value_shapes[name] = ti;
    }

    // Seed shapes for large weights whose CPU byte-copy was freed after a prior
    // variant's upload (their OrtValue is gone from initializer_view, so the loop
    // above skipped them). Their dtype + dims live in the snapshot's freed cache.
    // Do not overwrite an entry the live loop already produced.
    for (const auto& [name, fi] : snap.freed_initializer_shapes) {
        if (value_shapes.count(name)) continue;
        DML_TENSOR_DATA_TYPE dml_dt = OnnxDtypeToDml(fi.onnx_dtype);
        if (dml_dt == DML_TENSOR_DATA_TYPE_UNKNOWN) continue;
        std::vector<uint32_t> sizes(fi.dims.size());
        for (size_t d = 0; d < fi.dims.size(); ++d)
            sizes[d] = static_cast<uint32_t>(fi.dims[d] > 0 ? fi.dims[d] : 1);
        auto ti = MakeTensorInfo(sizes, dml_dt);
        ti.original_rank = static_cast<uint32_t>(fi.dims.size());
        value_shapes[name] = ti;
    }

    // Graph inputs from concrete runtime dims. dtype comes from the snapshot's
    // captured graph_input_dtypes (real ValueInfo element type from Compile). This
    // is CRITICAL for correctness: integer inputs like input_ids feed Gather as
    // its INDICES tensor, and DML rejects a float indices tensor (E_INVALIDARG). A
    // FLOAT32 default would silently mistype every non-fp32 input. Falls back to
    // FLOAT32 only when no dtype was captured.
    for (size_t i = 0; i < snap.graph_input_names.size() && i < runtime_input_dims.size(); ++i) {
        const std::string& name = snap.graph_input_names[i];
        if (name.empty()) continue;
        if (value_shapes.count(name)) continue;  // already seeded (initializer)
        const auto& dims = runtime_input_dims[i];
        std::vector<uint32_t> sizes(dims.size());
        for (size_t d = 0; d < dims.size(); ++d) sizes[d] = static_cast<uint32_t>(dims[d] > 0 ? dims[d] : 1);
        DML_TENSOR_DATA_TYPE dml_dt = DML_TENSOR_DATA_TYPE_FLOAT32;
        auto dt_it = snap.graph_input_dtypes.find(name);
        if (dt_it != snap.graph_input_dtypes.end()) {
            DML_TENSOR_DATA_TYPE mapped = OnnxDtypeToDml(dt_it->second);
            if (mapped != DML_TENSOR_DATA_TYPE_UNKNOWN) dml_dt = mapped;
        }
        auto ti = MakeTensorInfo(sizes, dml_dt);
        ti.original_rank = static_cast<uint32_t>(dims.size());
        value_shapes[name] = ti;
    }

    // -----------------------------------------------------------------------
    // Seed intermediate shapes via ONNX shape inference. Translators
    // READ (never compute) their output shape from value_shapes; on a dynamic
    // decoder those intermediates are only knowable from concrete runtime dims.
    // Best-effort: unresolved values stay unseeded → clean ORT_FAIL downstream.
    // -----------------------------------------------------------------------
    ORT_TRY {
        RunSnapshotShapeInference(ort_api, snap, runtime_input_dims, value_shapes);
    }
    ORT_CATCH(const std::exception& e) {
        ORT_HANDLE_EXCEPTION([&]() {
            DML_PERF_LOG("[RuntimeFusion] InferShapes threw: ", e.what(), "\n");
        });
    }

    // -----------------------------------------------------------------------
    // Translate each snapshot node (NodeView-backed), replicating Compile's
    // passthrough/alias/producer bookkeeping.
    // -----------------------------------------------------------------------
    OpTranslatorRegistry registry = BuildOpTranslatorRegistry();

    std::vector<CompiledNode> compiled_nodes;
    compiled_nodes.reserve(snap.nodes.size());
    std::unordered_map<std::string, std::pair<size_t, size_t>> value_producer;
    std::unordered_map<std::string, std::string> graph_input_aliases;
    std::unordered_set<std::string> consumed_initializer_names;
    std::unordered_set<std::string> dml_consumed_names;

    // Graph output set + map (owned): the partition outputs are every produced
    // value not consumed internally. We approximate with the last node's outputs
    // plus any value not consumed downstream. Simpler + correct: treat as graph
    // outputs the values that no other snapshot node consumes.
    std::unordered_set<std::string> all_consumed;
    for (const auto& dn : snap.nodes)
        for (const auto& in : dn.input_names) if (!in.empty()) all_consumed.insert(in);
    std::unordered_map<std::string, size_t> graph_output_map;
    {
        size_t out_idx = 0;
        for (const auto& dn : snap.nodes)
            for (const auto& on : dn.output_names)
                if (!on.empty() && !all_consumed.count(on) && !graph_output_map.count(on))
                    graph_output_map[on] = out_idx++;
    }

    for (const auto& dn : snap.nodes) {
        const std::string& op_type = dn.op_type;
        auto reg_it = registry.find(op_type);
        if (reg_it == registry.end()) {
            DML_PERF_LOG("[RuntimeFusion] FAIL: no translator for op=", op_type, "\n");
            return false;
        }

        NodeView view{dn};
        auto input_names  = dn.input_names;
        auto output_names = dn.output_names;

        auto translated = reg_it->second(ort_api, view, value_shapes, all_initializers);
        if (!translated) {
            DML_PERF_LOG("[RuntimeFusion] FAIL: translator nullopt op=", op_type,
                         " node='", dn.name, "'\n");
            auto dims_str = [&](const std::string& n) -> std::string {
                auto it = value_shapes.find(n);
                if (it == value_shapes.end()) return "";
                std::string s = " sizes=[";
                for (auto d : it->second.sizes) s += std::to_string(d) + ",";
                s += "] dt=" + std::to_string(static_cast<int>(it->second.data_type));
                if (it->second.original_rank) s += " rank=" + std::to_string(it->second.original_rank);
                return s;
            };
            for (const auto& in : input_names)
                DML_PERF_LOG("[RuntimeFusion]   in '", in, "' shape=",
                             value_shapes.count(in) ? "SEEDED" : "MISSING",
                             all_initializers.count(in) ? " (init)" : "",
                             dims_str(in).c_str(), "\n");
            for (const auto& on : output_names)
                DML_PERF_LOG("[RuntimeFusion]   out '", on, "' shape=",
                             value_shapes.count(on) ? "SEEDED" : "MISSING",
                             dims_str(on).c_str(), "\n");
            return false;
        }

        // Seed downstream consumers with the translator's computed output shapes.
        // Normally we KEEP an existing seed (a concrete runtime dim or an earlier
        // InferShapes result). GroupQueryAttention is the exception: ONNX InferShapes
        // mis-derives its (com.microsoft) output[0], collapsing the hidden dim to
        // numHeads (e.g. 5120→40). That stale seed would shadow the translator's
        // correct [B,S,queryHiddenSize] and break the downstream weight_only Mul's
        // broadcast. For GQA, OVERWRITE the seed with the translator's authoritative
        // shape (computed from the Q input).
        const bool overwrite_seed = (op_type == "GroupQueryAttention");
        for (size_t k = 0; k < output_names.size() && k < translated->output_tensors.size(); ++k) {
            if (overwrite_seed || !value_shapes.count(output_names[k]))
                value_shapes[output_names[k]] = translated->output_tensors[k];
        }

        auto translated_ptr = std::make_unique<TranslatedOp>(std::move(*translated));
        translated_ptr->FixupPointers();

        bool passthrough_is_graph_output = false;
        if (translated_ptr->passthrough) {
            for (const auto& oname : output_names)
                if (graph_output_map.count(oname)) { passthrough_is_graph_output = true; break; }
        }

        if (translated_ptr->passthrough && !passthrough_is_graph_output) {
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

        if (passthrough_is_graph_output) {
            DmlTensorInfo out_info = translated_ptr->output_tensors.empty()
                ? DmlTensorInfo{} : translated_ptr->output_tensors[0];
            translated_ptr = std::make_unique<TranslatedOp>(BuildIdentityOp(out_info));
        }

        size_t dml_input_count = translated_ptr->input_tensors.size();
        size_t num_primary_inputs = translated_ptr->primary_input_count.value_or(dml_input_count);
        for (size_t s = 0; s < input_names.size() && s < num_primary_inputs; ++s) {
            size_t name_idx = translated_ptr->input_name_reorder.empty()
                ? s : translated_ptr->input_name_reorder[s];
            if (name_idx >= input_names.size()) continue;
            const auto& in_name = input_names[name_idx];
            dml_consumed_names.insert(in_name);
            auto init_name = ResolveToInitializer(in_name, graph_input_aliases, all_initializers);
            if (!init_name.empty()) consumed_initializer_names.insert(init_name);
        }
        for (const auto& sn : translated_ptr->sub_nodes) {
            for (const auto& [onnx_idx, _] : sn.graph_inputs) {
                if (onnx_idx < input_names.size()) {
                    dml_consumed_names.insert(input_names[onnx_idx]);
                    auto init_name = ResolveToInitializer(input_names[onnx_idx], graph_input_aliases, all_initializers);
                    if (!init_name.empty()) consumed_initializer_names.insert(init_name);
                }
            }
        }

        size_t producer_compiled_idx = compiled_nodes.size();
        for (size_t k = 0; k < output_names.size(); ++k)
            value_producer[output_names[k]] = { producer_compiled_idx, k };

        CompiledNode compiled_node;
        compiled_node.translated = std::move(translated_ptr);
        compiled_node.input_names = std::move(input_names);
        compiled_node.output_names = std::move(output_names);
        compiled_node.op_type = op_type;
        compiled_nodes.push_back(std::move(compiled_node));
    }

    // Dead-node strip (identical to static path).
    std::vector<bool> node_is_live(compiled_nodes.size(), true);
    {
        for (const auto& [name, _] : graph_output_map)
            dml_consumed_names.insert(name);
        std::vector<size_t> refcount(compiled_nodes.size(), 0);
        for (const auto& [name, prod] : value_producer)
            if (dml_consumed_names.count(name)) ++refcount[prod.first];
        std::vector<std::vector<std::string>> wired_inputs(compiled_nodes.size());
        for (size_t i = 0; i < compiled_nodes.size(); ++i) {
            const auto& cn = compiled_nodes[i];
            size_t npi = cn.translated->primary_input_count.value_or(cn.translated->input_tensors.size());
            for (size_t s = 0; s < cn.input_names.size() && s < npi; ++s) {
                size_t name_idx = cn.translated->input_name_reorder.empty()
                    ? s : cn.translated->input_name_reorder[s];
                if (name_idx < cn.input_names.size())
                    wired_inputs[i].push_back(cn.input_names[name_idx]);
            }
            for (const auto& sn : cn.translated->sub_nodes)
                for (const auto& [onnx_idx, _] : sn.graph_inputs)
                    if (onnx_idx < cn.input_names.size())
                        wired_inputs[i].push_back(cn.input_names[onnx_idx]);
        }
        std::vector<size_t> dead_queue;
        for (size_t i = 0; i < compiled_nodes.size(); ++i)
            if (refcount[i] == 0) dead_queue.push_back(i);
        for (size_t qi = 0; qi < dead_queue.size(); ++qi) {
            size_t di = dead_queue[qi];
            node_is_live[di] = false;
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

    // Mark OWNED_BY_DML initializers. DML decides owned (bound at init) vs runtime
    // (bound at execute) from the tensor desc AT THE CONSUMING EDGE'S TARGET. So the
    // flag must land on the desc of whichever node actually CONSUMES the initializer:
    //   - primary inputs [0, num_primary_inputs): the primary's input_buffer_descs[s]
    //   - sub_node graph_inputs: the SUB_NODE's input_buffer_descs[to_input]
    // Marking only the primary's desc is WRONG when the initializer is consumed by a
    // sub_node (e.g. RotaryEmbedding's cos_cache/sin_cache go to the Gather SUB_NODEs,
    // NOT the identity primary — the primary declares them but primary_input_count=1
    // never wires them). Result: we'd upload+bind the initializer at init (owned set
    // from the primary's stale flag) but DML compiled the input as runtime (sub_node
    // desc = NONE) → "index N: non-null buffer provided, null expected" → device removal.
    auto is_large_initializer = [&](const std::string& onnx_name) -> bool {
        std::string resolved = onnx_name;
        while (graph_input_aliases.count(resolved)) resolved = graph_input_aliases[resolved];
        // all_initializers keeps freed-weight NAMES (mapped to nullptr), so this
        // name-presence check still classifies them correctly on later variants.
        if (!all_initializers.count(onnx_name) && !all_initializers.count(resolved))
            return false;
        auto shape_it = value_shapes.find(all_initializers.count(resolved) ? resolved : onnx_name);
        uint64_t bytes = (shape_it != value_shapes.end()) ? shape_it->second.total_bytes : 0;
        return bytes >= kMaxConstNodeDataSize;
    };
    // Onnx names of large initializers that would otherwise have been OWNED_BY_DML.
    // Now bound as shared runtime inputs; resolved to DML slots after input_map is built.
    std::unordered_set<std::string> large_initializer_names;
    // VRAM: DO NOT mark large initializers OWNED_BY_DML on the deferred path. Owning bakes
    // the weight into THIS compiled operator's persistent resource, and the deferred path
    // builds one compiled operator per shape variant -> the same (byte-identical) weights
    // would be re-baked into every variant, multiplying VRAM by the variant count
    // (OVER_BUDGET -> PCIe paging -> slowdown). Instead we bind them as ordinary runtime
    // graph inputs from a single shared upload held on the
    // kernel state (see FullGraph_Compute + shared_initializers), exactly as ORT does
    // (its dump shows "0 OWNED_BY_DML tensors", persistent=4 bytes, for this same graph).
    // We still record which onnx initializer names WOULD have been owned so the caller
    // can resolve them to DML slots and set up the shared runtime binding.
    // NOTE: static Compile (fgf.cc ~:1670) is UNTOUCHED — it compiles one variant, so a
    // single owned copy fits budget and keeps DML's weight-repack perf.
    for (size_t node_idx = 0; node_idx < compiled_nodes.size(); ++node_idx) {
        if (!node_is_live[node_idx]) continue;
        auto& cn = compiled_nodes[node_idx];
        // Primary inputs.
        size_t num_primary_inputs = cn.translated->primary_input_count.value_or(
            cn.translated->input_tensors.size());
        size_t dml_input_count = cn.translated->input_tensors.size();
        for (size_t s = 0; s < cn.input_names.size() && s < dml_input_count && s < num_primary_inputs; ++s) {
            size_t name_idx = cn.translated->input_name_reorder.empty()
                ? s : cn.translated->input_name_reorder[s];
            if (name_idx >= cn.input_names.size()) continue;
            if (is_large_initializer(cn.input_names[name_idx]))
                large_initializer_names.insert(cn.input_names[name_idx]);
        }
        // Sub_node graph_inputs — the desc DML actually reads for an initializer that
        // feeds a sub_node (not the primary).
        for (auto& sn : cn.translated->sub_nodes) {
            for (const auto& [onnx_idx, to_input] : sn.graph_inputs) {
                if (onnx_idx >= cn.input_names.size()) continue;
                if (to_input >= sn.input_buffer_descs.size()) continue;
                if (is_large_initializer(cn.input_names[onnx_idx]))
                    large_initializer_names.insert(cn.input_names[onnx_idx]);
            }
        }
        cn.translated->FixupPointers();
    }

    // Filter consumed initializers to live inputs.
    {
        std::unordered_set<std::string> live_inputs;
        for (size_t i = 0; i < compiled_nodes.size(); ++i) {
            if (!node_is_live[i]) continue;
            for (auto name : compiled_nodes[i].input_names) {
                while (graph_input_aliases.count(name)) name = graph_input_aliases[name];
                live_inputs.insert(name);
            }
        }
        std::unordered_set<std::string> filtered;
        for (const auto& name : consumed_initializer_names)
            if (live_inputs.count(name)) filtered.insert(name);
        consumed_initializer_names = std::move(filtered);
    }

    // CreateOperator for each live node.
    ComPtr<IDMLDevice> dml_device;
    if (FAILED(provider->GetDmlDevice(dml_device.GetAddressOf()))) return false;
    ComPtr<IDMLDevice1> dml_device1;
    if (FAILED(dml_device.As(&dml_device1))) return false;

    for (size_t i = 0; i < compiled_nodes.size(); ++i) {
        if (!node_is_live[i]) continue;
        auto& cn = compiled_nodes[i];
        // Re-run fixup immediately before CreateOperator so every op_desc tensor
        // pointer targets the FINAL stable address of this node's desc arrays
        // (idempotent: FixupPointers only re-links pointers).
        cn.translated->FixupPointers();
        HRESULT hr_co = dml_device->CreateOperator(
                &cn.translated->op_desc, IID_PPV_ARGS(cn.translated->dml_operator.GetAddressOf()));
        if (FAILED(hr_co)) {
            const std::string out0 = cn.output_names.empty() ? std::string("?") : cn.output_names[0];
            DML_PERF_LOG("[RuntimeFusion] FAIL: CreateOperator op=", cn.op_type,
                         " HR=", Hex(static_cast<uint32_t>(hr_co)), " out='", out0, "'\n");
            DrainDmlDebugMessages(provider, "RuntimeFusion CreateOperator");
            return false;
        }
        for (size_t s = 0; s < cn.translated->sub_nodes.size(); ++s) {
            auto& sn = cn.translated->sub_nodes[s];
            HRESULT hr_sn = dml_device->CreateOperator(&sn.op_desc, IID_PPV_ARGS(sn.dml_operator.GetAddressOf()));
            if (FAILED(hr_sn)) {
                const std::string out0 = cn.output_names.empty() ? std::string("?") : cn.output_names[0];
                DML_PERF_LOG("[RuntimeFusion] FAIL: CreateOperator sub_node[", s, "] op=", cn.op_type,
                             " opcode=", static_cast<int>(sn.op_desc.Type),
                             " HR=", Hex(static_cast<uint32_t>(hr_sn)), " node-out='", out0, "'\n");
                DrainDmlDebugMessages(provider, "RuntimeFusion CreateOperator sub_node");
                return false;
            }
        }
    }

    // Build DML input map (name-based) + owned indices.
    DmlInputMapResult input_map = BuildDmlInputMapByNames(
        ort_api, snap.graph_input_names, consumed_initializer_names, value_shapes, all_initializers);

    // Resolve each large initializer (no longer owned) to its DML graph-input
    // slot so FullGraph_Compute can bind it from the kernel state's shared upload every
    // dispatch. Dedup by dml_idx (an initializer consumed by several nodes maps to one
    // slot). The onnx name recorded is the ALIAS-RESOLVED name that keys all_initializers
    // (and, at upload time, shared_initializers).
    std::vector<std::pair<size_t, std::string>> shared_initializer_slots;
    {
        std::unordered_set<size_t> seen_slots;
        for (const auto& raw_name : large_initializer_names) {
            std::string resolved_name = raw_name;
            while (graph_input_aliases.count(resolved_name)) resolved_name = graph_input_aliases[resolved_name];
            auto di_it = input_map.dml_input_map.find(resolved_name);
            if (di_it == input_map.dml_input_map.end()) continue;
            if (!seen_slots.insert(di_it->second).second) continue;
            shared_initializer_slots.emplace_back(di_it->second, resolved_name);
        }
    }

    // Collect owned graph-input indices from the OWNED_BY_DML flags. Must scan BOTH
    // primary inputs AND sub_node graph_inputs so the set matches exactly which inputs
    // DML compiled as owned (the flag was placed on the consuming edge's target desc
    // above). If an initializer feeding a sub_node (e.g. RotaryEmbedding cos/sin) were
    // flagged owned on the sub_node desc but omitted here, we'd fail to bind it at init
    // while DML expects it → "null buffer expected" mismatch (or vice versa).
    std::unordered_set<size_t> owned_graph_input_indices;
    auto add_owned = [&](const std::string& onnx_name) {
        std::string resolved_name = onnx_name;
        while (graph_input_aliases.count(resolved_name)) resolved_name = graph_input_aliases[resolved_name];
        auto di_it = input_map.dml_input_map.find(resolved_name);
        if (di_it != input_map.dml_input_map.end())
            owned_graph_input_indices.insert(di_it->second);
    };
    for (auto& cn : compiled_nodes) {
        size_t dml_input_count = cn.translated->input_tensors.size();
        for (size_t s = 0; s < cn.input_names.size() && s < dml_input_count; ++s) {
            if (cn.translated->input_buffer_descs[s].Flags & DML_TENSOR_FLAG_OWNED_BY_DML) {
                size_t name_idx = cn.translated->input_name_reorder.empty()
                    ? s : cn.translated->input_name_reorder[s];
                if (name_idx >= cn.input_names.size()) continue;
                add_owned(cn.input_names[name_idx]);
            }
        }
        for (auto& sn : cn.translated->sub_nodes) {
            for (const auto& [onnx_idx, to_input] : sn.graph_inputs) {
                if (onnx_idx >= cn.input_names.size()) continue;
                if (to_input >= sn.input_buffer_descs.size()) continue;
                if (sn.input_buffer_descs[to_input].Flags & DML_TENSOR_FLAG_OWNED_BY_DML)
                    add_owned(cn.input_names[onnx_idx]);
            }
        }
    }

    // DML node descriptors.
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

    std::vector<DML_CONSTANT_DATA_GRAPH_NODE_DESC> const_node_descs(input_map.constant_nodes.size());
    for (size_t c = 0; c < input_map.constant_nodes.size(); ++c) {
        const_node_descs[c].Data = input_map.constant_nodes[c].data.data();
        const_node_descs[c].DataSize = input_map.constant_nodes[c].data.size();
        graph_nodes.push_back({ DML_GRAPH_NODE_TYPE_CONSTANT, &const_node_descs[c] });
    }
    size_t const_node_base = total_dml_nodes;

    // Wire edges (primary + sub_node graph_inputs + outputs) — identical to static.
    std::vector<DML_INPUT_GRAPH_EDGE_DESC> input_edge_storage;
    std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediate_edge_storage;
    std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> output_edge_storage;

    for (size_t node_idx = 0; node_idx < compiled_nodes.size(); ++node_idx) {
        if (!node_is_live[node_idx]) continue;
        const auto& cn = compiled_nodes[node_idx];
        size_t primary_dml_idx = dml_node_offset[node_idx];
        // nullopt = wire all inputs; N (incl. 0) = wire exactly N. See TranslatedOp doc.
        size_t num_primary_inputs = cn.translated->primary_input_count.value_or(
            cn.translated->input_tensors.size());
        for (size_t input_slot = 0; input_slot < cn.input_names.size() && input_slot < num_primary_inputs; ++input_slot) {
            size_t name_idx = cn.translated->input_name_reorder.empty()
                ? input_slot : cn.translated->input_name_reorder[input_slot];
            if (name_idx >= cn.input_names.size()) continue;
            auto name = cn.input_names[name_idx];
            if (name.empty()) continue;
            const size_t dml_schema_slot = cn.translated->dml_input_slot_indices.empty()
                ? input_slot : cn.translated->dml_input_slot_indices[input_slot];
            while (graph_input_aliases.count(name)) name = graph_input_aliases[name];

            auto const_it = input_map.constant_node_map.find(name);
            if (const_it != input_map.constant_node_map.end()) {
                DML_INTERMEDIATE_GRAPH_EDGE_DESC edge{};
                edge.FromNodeIndex = static_cast<UINT>(const_node_base + const_it->second);
                edge.FromNodeOutputIndex = 0;
                edge.ToNodeIndex = static_cast<UINT>(primary_dml_idx);
                edge.ToNodeInputIndex = static_cast<UINT>(dml_schema_slot);
                intermediate_edge_storage.push_back(edge);
                continue;
            }
            auto dml_in_it = input_map.dml_input_map.find(name);
            if (dml_in_it != input_map.dml_input_map.end()) {
                DML_INPUT_GRAPH_EDGE_DESC edge{};
                edge.GraphInputIndex = static_cast<UINT>(dml_in_it->second);
                edge.ToNodeIndex = static_cast<UINT>(primary_dml_idx);
                edge.ToNodeInputIndex = static_cast<UINT>(dml_schema_slot);
                input_edge_storage.push_back(edge);
                continue;
            }
            auto prod_it = value_producer.find(name);
            if (prod_it != value_producer.end()) {
                size_t prod_compiled_idx = prod_it->second.first;
                size_t prod_dml_idx = dml_node_offset[prod_compiled_idx];
                size_t prod_output_slot = prod_it->second.second;
                UINT from_output_index;
                const auto& osrc = compiled_nodes[prod_compiled_idx].translated->output_source;
                if (!osrc.empty() && prod_output_slot < osrc.size()) {
                    auto [src_sub, src_slot] = osrc[prod_output_slot];
                    if (src_sub >= 0) prod_dml_idx += 1 + static_cast<size_t>(src_sub);
                    from_output_index = static_cast<UINT>(src_slot);
                } else {
                    size_t num_subs = compiled_nodes[prod_compiled_idx].translated->sub_nodes.size();
                    if (num_subs > 0) prod_dml_idx += num_subs;
                    from_output_index = static_cast<UINT>(prod_output_slot);
                }
                DML_INTERMEDIATE_GRAPH_EDGE_DESC edge{};
                edge.FromNodeIndex = static_cast<UINT>(prod_dml_idx);
                edge.FromNodeOutputIndex = from_output_index;
                edge.ToNodeIndex = static_cast<UINT>(primary_dml_idx);
                edge.ToNodeInputIndex = static_cast<UINT>(dml_schema_slot);
                intermediate_edge_storage.push_back(edge);
            }
        }

        // Sub-node graph inputs.
        for (size_t sidx = 0; sidx < cn.translated->sub_nodes.size(); ++sidx) {
            const auto& sn = cn.translated->sub_nodes[sidx];
            size_t sn_dml_idx = primary_dml_idx + 1 + sidx;

            // Wire INTERNAL (intra-node) edges: producer sub/primary output → this
            // sub_node input. Mirrors the static Compile (fgf.cc:1883-1899). WITHOUT
            // this, a multi-sub-node op's internal edge dangles — e.g. MatMulNBits'
            // Gemm B input never connects to the primary Dequantize's FP16 output,
            // so DML sees the raw UINT4 weight → E_INVALIDARG (unsupported GEMM dtype).
            for (size_t inp = 0; inp < sn.input_from.size(); ++inp) {
                auto [src_sub, src_slot] = sn.input_from[inp];
                if (src_sub < -1) continue;  // sentinel: this slot is wired via graph_inputs
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

            for (const auto& [onnx_idx, to_input] : sn.graph_inputs) {
                if (onnx_idx >= cn.input_names.size()) continue;
                auto gi_name = cn.input_names[onnx_idx];
                if (gi_name.empty()) continue;
                while (graph_input_aliases.count(gi_name)) gi_name = graph_input_aliases[gi_name];
                auto const_it = input_map.constant_node_map.find(gi_name);
                if (const_it != input_map.constant_node_map.end()) {
                    DML_INTERMEDIATE_GRAPH_EDGE_DESC edge{};
                    edge.FromNodeIndex = static_cast<UINT>(const_node_base + const_it->second);
                    edge.FromNodeOutputIndex = 0;
                    edge.ToNodeIndex = static_cast<UINT>(sn_dml_idx);
                    edge.ToNodeInputIndex = static_cast<UINT>(to_input);
                    intermediate_edge_storage.push_back(edge);
                    continue;
                }
                auto dml_in_it = input_map.dml_input_map.find(gi_name);
                if (dml_in_it != input_map.dml_input_map.end()) {
                    DML_INPUT_GRAPH_EDGE_DESC edge{};
                    edge.GraphInputIndex = static_cast<UINT>(dml_in_it->second);
                    edge.ToNodeIndex = static_cast<UINT>(sn_dml_idx);
                    edge.ToNodeInputIndex = static_cast<UINT>(to_input);
                    input_edge_storage.push_back(edge);
                    continue;
                }
                auto prod_it = value_producer.find(gi_name);
                if (prod_it != value_producer.end()) {
                    size_t prod_compiled_idx = prod_it->second.first;
                    size_t prod_dml_idx = dml_node_offset[prod_compiled_idx];
                    size_t prod_output_slot = prod_it->second.second;
                    UINT from_output_index;
                    const auto& osrc = compiled_nodes[prod_compiled_idx].translated->output_source;
                    if (!osrc.empty() && prod_output_slot < osrc.size()) {
                        auto [src_sub, src_slot] = osrc[prod_output_slot];
                        if (src_sub >= 0) prod_dml_idx += 1 + static_cast<size_t>(src_sub);
                        from_output_index = static_cast<UINT>(src_slot);
                    } else {
                        size_t num_subs = compiled_nodes[prod_compiled_idx].translated->sub_nodes.size();
                        if (num_subs > 0) prod_dml_idx += num_subs;
                        from_output_index = static_cast<UINT>(prod_output_slot);
                    }
                    DML_INTERMEDIATE_GRAPH_EDGE_DESC edge{};
                    edge.FromNodeIndex = static_cast<UINT>(prod_dml_idx);
                    edge.FromNodeOutputIndex = from_output_index;
                    edge.ToNodeIndex = static_cast<UINT>(sn_dml_idx);
                    edge.ToNodeInputIndex = static_cast<UINT>(to_input);
                    intermediate_edge_storage.push_back(edge);
                }
            }
        }
    }

    // Output edges.
    for (const auto& [name, out_idx] : graph_output_map) {
        auto resolved = name;
        while (graph_input_aliases.count(resolved)) resolved = graph_input_aliases[resolved];
        auto prod_it = value_producer.find(resolved);
        if (prod_it == value_producer.end()) {
            DML_PERF_LOG("[RuntimeFusion] FAIL: unresolved output '", name, "'\n");
            return false;
        }
        size_t prod_compiled_idx = prod_it->second.first;
        size_t prod_dml_idx = dml_node_offset[prod_compiled_idx];
        size_t prod_output_slot = prod_it->second.second;
        UINT from_output_index;
        const auto& osrc = compiled_nodes[prod_compiled_idx].translated->output_source;
        if (!osrc.empty() && prod_output_slot < osrc.size()) {
            auto [src_sub, src_slot] = osrc[prod_output_slot];
            if (src_sub >= 0) prod_dml_idx += 1 + static_cast<size_t>(src_sub);
            from_output_index = static_cast<UINT>(src_slot);
        } else {
            size_t num_subs = compiled_nodes[prod_compiled_idx].translated->sub_nodes.size();
            if (num_subs > 0) prod_dml_idx += num_subs;
            from_output_index = static_cast<UINT>(prod_output_slot);
        }
        DML_OUTPUT_GRAPH_EDGE_DESC edge{};
        edge.FromNodeIndex = static_cast<UINT>(prod_dml_idx);
        edge.FromNodeOutputIndex = from_output_index;
        edge.GraphOutputIndex = static_cast<UINT>(out_idx);
        output_edge_storage.push_back(edge);
    }

    std::unordered_set<size_t> dml_inputs_with_edges;
    for (const auto& ie : input_edge_storage) dml_inputs_with_edges.insert(ie.GraphInputIndex);

    std::vector<DML_GRAPH_EDGE_DESC> input_edges(input_edge_storage.size());
    for (size_t i = 0; i < input_edge_storage.size(); ++i)
        input_edges[i] = { DML_GRAPH_EDGE_TYPE_INPUT, &input_edge_storage[i] };
    std::vector<DML_GRAPH_EDGE_DESC> intermediate_edges(intermediate_edge_storage.size());
    for (size_t i = 0; i < intermediate_edge_storage.size(); ++i)
        intermediate_edges[i] = { DML_GRAPH_EDGE_TYPE_INTERMEDIATE, &intermediate_edge_storage[i] };
    std::vector<DML_GRAPH_EDGE_DESC> output_edges(output_edge_storage.size());
    for (size_t i = 0; i < output_edge_storage.size(); ++i)
        output_edges[i] = { DML_GRAPH_EDGE_TYPE_OUTPUT, &output_edge_storage[i] };

    DML_GRAPH_DESC graph_desc{};
    graph_desc.InputCount = static_cast<UINT>(input_map.total_dml_inputs);
    graph_desc.OutputCount = static_cast<UINT>(graph_output_map.size());
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
    {
        bool has_fp16 = false, has_fp32 = false;
        for (size_t i = 0; i < compiled_nodes.size(); ++i) {
            if (!node_is_live[i]) continue;
            for (const auto& t : compiled_nodes[i].translated->input_tensors)
                if (t.data_type == DML_TENSOR_DATA_TYPE_FLOAT16) has_fp16 = true;
                else if (t.data_type == DML_TENSOR_DATA_TYPE_FLOAT32) has_fp32 = true;
            for (const auto& t : compiled_nodes[i].translated->output_tensors)
                if (t.data_type == DML_TENSOR_DATA_TYPE_FLOAT16) has_fp16 = true;
                else if (t.data_type == DML_TENSOR_DATA_TYPE_FLOAT32) has_fp32 = true;
        }
        if (has_fp16 && !has_fp32)
            exec_flags |= DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
    }

    DML_PERF_LOG("[RuntimeFusion] CompileGraph: nodes=", graph_desc.NodeCount,
        " inputs=", graph_desc.InputCount, " outputs=", graph_desc.OutputCount, "\n");

    HRESULT hr = dml_device1->CompileGraph(&graph_desc, exec_flags, IID_PPV_ARGS(compiled_op.GetAddressOf()));
    if (FAILED(hr)) {
        DML_PERF_LOG("[RuntimeFusion] FAIL: CompileGraph HR=0x", Hex(static_cast<uint32_t>(hr)), "\n");
        DrainDmlDebugMessages(provider, "RuntimeFusion.CompileGraph");
        return false;
    }

    // Upload the large initializer weights ONCE to the kernel state, shared
    // across all shape variants (they are byte-identical). Bound as runtime inputs each
    // dispatch (see FullGraph_Compute), NOT owned/baked into this variant's persistent
    // resource — that is what previously tripled VRAM. On the first variant compile we
    // upload; later variants reuse. Guard on kernel_state (null only if a future caller
    // compiles without a state — then these weights would be unbound, so fail loudly).
    if (!shared_initializer_slots.empty()) {
        if (!kernel_state)
            return false;  // no state to hold shared weights -> would bind NONE -> garbage
        // Per-initializer gating (not all-or-nothing): upload any weight not already on
        // the state. All variants of one graph consume the same weights, so after the
        // first variant this loop is a no-op; but if a later variant references a weight
        // the first did not, it is uploaded then. Each weight is uploaded exactly once.
        // Drain the pooled UPLOAD staging heap every ~kUploadDrainThresholdBytes so it does
        // not balloon during this burst. Each UploadToResource stages a full second copy of
        // the weight in a CPU-visible UPLOAD chunk that is only reclaimable once its copy
        // fence signals; without a mid-burst drain, fences never signal and chunks accumulate
        // to ~4 GiB (on this UMA APU that counts against LOCAL VRAM → OVER_BUDGET → paging).
        // WaitForOutstandingWork() flushes + blocks on the completion fence so the next
        // BeginUploadToGpu's ReclaimAllocations can recycle the drained chunks in place.
        static constexpr uint64_t kUploadDrainThresholdBytes = 512ull * 1024 * 1024;  // 512 MB
        uint64_t bytes_since_drain = 0;

        size_t newly_uploaded = 0;
        for (const auto& [dml_idx, name] : shared_initializer_slots) {
            if (kernel_state->shared_initializers.count(name)) continue;  // already uploaded
            auto init_it = all_initializers.find(name);
            if (init_it == all_initializers.end() || !init_it->second) return false;

            void* cpu_ptr = nullptr;
            OrtStatus* st = ort_api.GetTensorMutableData(
                const_cast<OrtValue*>(init_it->second), &cpu_ptr);
            if (st || !cpu_ptr) { if (st) ort_api.ReleaseStatus(st); return false; }

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
            if (bytes == 0) return false;
            uint64_t upload_bytes = std::min(bytes, actual_bytes > 0 ? actual_bytes : bytes);

            FullGraphKernelState::SharedInitializer si;
            if (FAILED(provider->AllocatePooledResource(
                    static_cast<size_t>(bytes), AllocatorRoundingMode::Disabled,
                    si.gpu_resource.GetAddressOf(), si.allocator_ref.GetAddressOf())))
                return false;
            if (FAILED(provider->UploadToResource(si.gpu_resource.Get(), cpu_ptr, upload_bytes)))
                return false;
            si.bytes = bytes;
            kernel_state->shared_initializers[name] = std::move(si);
            ++newly_uploaded;

            // Bound peak staging: drain once this batch's staged bytes exceed the threshold.
            bytes_since_drain += upload_bytes;
            if (bytes_since_drain >= kUploadDrainThresholdBytes) {
                provider->WaitForOutstandingWork();
                bytes_since_drain = 0;
            }
        }
        if (newly_uploaded > 0) {
            // Final drain + trim: complete any pending copies, then release the residual
            // staging chunks back to the driver so they are not held resident during decode.
            provider->WaitForOutstandingWork();
            provider->TrimUploadHeap();
            DML_PERF_LOG("[RuntimeFusion] shared_initializers uploaded: +", newly_uploaded,
                         " (total=", kernel_state->shared_initializers.size(), ")\n");
        }
    }

    // Persistent resource + upload initializers.
    ComPtr<ID3D12Resource> persistent_resource;
    ComPtr<IUnknown> persistent_allocator;
    std::optional<DML_BUFFER_BINDING> persistent_binding;
    auto binding_props = compiled_op->GetBindingProperties();
    UINT64 persistent_size = binding_props.PersistentResourceSize;
    if (persistent_size > 0) {
        if (FAILED(provider->AllocatePooledResource(
                static_cast<size_t>(persistent_size), AllocatorRoundingMode::Disabled,
                persistent_resource.GetAddressOf(), persistent_allocator.GetAddressOf())))
            return false;
        persistent_binding = DML_BUFFER_BINDING{ persistent_resource.Get(), 0, persistent_size };
    }

    std::vector<InitBinding> const_graph_input_bindings(input_map.total_dml_inputs);
    if (!UploadInitializersByNames(
            ort_api, provider, input_map, owned_graph_input_indices,
            snap.graph_input_names, value_shapes, all_initializers, const_graph_input_bindings))
        return false;

    std::vector<DML_BUFFER_BINDING> init_input_bindings(input_map.total_dml_inputs, DML_BUFFER_BINDING{});
    for (size_t i = 0; i < const_graph_input_bindings.size(); ++i) {
        // A DML graph input with NO consuming edge in the compiled graph MUST be bound
        // as NONE (null) — DML optimized it away, so a real buffer there is rejected
        // ("non-null buffer provided, but a null buffer was expected", causing a
        // device removal at init). This happens when an initializer's only consumer was
        // a node the orphan fixes removed (GQA FILL / RotaryEmbedding placeholder
        // primary). Mirror the runtime Compute binding, which already skips edge-less
        // inputs (see FullGraph_Compute: `if (!dml_inputs_with_edges.count(dml_i))`).
        if (!dml_inputs_with_edges.count(i)) continue;
        if (const_graph_input_bindings[i].gpu_resource) {
            auto& ib = const_graph_input_bindings[i];
            init_input_bindings[i] = { ib.gpu_resource.Get(), 0, ib.bytes };
        }
    }

    // Diagnostic dump: map each DML input index to its value name plus whether it
    // has an edge and a bound buffer. DML rejects a binding with "non-null buffer
    // provided, null expected" (device removal at init) when a bound index lacks a
    // live edge. A bound index NOT in dml_inputs_with_edges means the filter above
    // failed to skip it (set bug); a bound index that IS in the set but still gets
    // rejected means DML dropped the edge during CompileGraph optimization
    // (dead-path input).
    {
        std::vector<std::string> idx_to_name(input_map.total_dml_inputs);
        for (const auto& [nm, di] : input_map.dml_input_map)
            if (di < idx_to_name.size()) idx_to_name[di] = nm;
        DML_PERF_LOG("[RuntimeFusion] init bindings: total=", input_map.total_dml_inputs,
                     " with_edges=", dml_inputs_with_edges.size(),
                     " const_nodes=", input_map.constant_nodes.size(), "\n");
        // Dump every index that is BOUND (non-null) — DML rejects any bound index whose
        // edge it dropped. Print edge-flag so we see which bound index lacks an edge.
        for (size_t i = 0; i < init_input_bindings.size(); ++i) {
            bool bound = init_input_bindings[i].Buffer != nullptr;
            if (!bound) continue;
            bool has_edge = dml_inputs_with_edges.count(i) != 0;
            DML_PERF_LOG("[RuntimeFusion] INIT-BIND idx=", i, " '", idx_to_name[i],
                         "' bound=1 edge=", has_edge ? 1 : 0, "\n");
        }
    }

    const DML_BUFFER_BINDING* persistent_ptr = persistent_binding ? &*persistent_binding : nullptr;
    if (FAILED(provider->InitializeOperator(
            compiled_op.Get(), persistent_ptr, gsl::make_span(init_input_bindings)))) {
        DrainDmlDebugMessages(provider, "RuntimeFusion.InitializeOperator");
        return false;
    }

    ORT_TRY {
        provider->Flush();
        DrainDmlDebugMessages(provider, "RuntimeFusion.Flush");
    }
    ORT_CATCH(const std::exception& e) {
        ORT_HANDLE_EXCEPTION([&]() {
            DML_PERF_LOG("[RuntimeFusion] FAIL: init/flush threw: ", e.what(), "\n");
        });
        return false;
    }

    provider->QueueReference(compiled_op.Get());
    if (persistent_allocator) provider->QueueReference(persistent_allocator.Get());
    for (auto& ib : const_graph_input_bindings)
        if (ib.gpu_resource) provider->QueueReference(ib.gpu_resource.Get());

    // Fill the variant.
    out_variant.compiled_op = std::move(compiled_op);
    out_variant.persistent_resource = std::move(persistent_resource);
    out_variant.persistent_allocator = std::move(persistent_allocator);
    out_variant.persistent_binding = persistent_binding;
    out_variant.num_runtime_inputs = input_map.total_dml_inputs;
    out_variant.num_subgraph_inputs = snap.graph_input_names.size();
    out_variant.subgraph_to_dml_input = input_map.subgraph_to_dml_input;
    out_variant.dml_inputs_with_edges = dml_inputs_with_edges;
    out_variant.runtime_input_bytes.assign(input_map.total_dml_inputs, 0);
    out_variant.runtime_input_is_owned.assign(input_map.total_dml_inputs, false);
    out_variant.num_initializers = 0;
    // DML slots fed by shared (non-owned) initializer weights, so
    // FullGraph_Compute binds them from kernel_state->shared_initializers each dispatch.
    out_variant.shared_initializer_slots = std::move(shared_initializer_slots);

    for (size_t i = 0; i < snap.graph_input_names.size(); ++i) {
        if (input_map.subgraph_to_dml_input[i] == SIZE_MAX) continue;
        size_t dml_idx = input_map.subgraph_to_dml_input[i];
        auto it = value_shapes.find(snap.graph_input_names[i]);
        out_variant.runtime_input_bytes[dml_idx] =
            (it != value_shapes.end()) ? it->second.total_bytes : 0;
        if (owned_graph_input_indices.count(dml_idx))
            out_variant.runtime_input_is_owned[dml_idx] = true;
    }
    for (const auto& name : input_map.ordered_initializer_names) {
        auto di_it = input_map.dml_input_map.find(name);
        if (di_it == input_map.dml_input_map.end()) continue;
        size_t dml_idx = di_it->second;
        auto it = value_shapes.find(name);
        out_variant.runtime_input_bytes[dml_idx] =
            (it != value_shapes.end()) ? it->second.total_bytes : 0;
        if (owned_graph_input_indices.count(dml_idx))
            out_variant.runtime_input_is_owned[dml_idx] = true;
    }

    // Output dims/bytes from graph_output_map ordering.
    out_variant.num_outputs = graph_output_map.size();
    out_variant.output_dims.resize(graph_output_map.size());
    out_variant.output_bytes.resize(graph_output_map.size());
    for (const auto& [name, out_idx] : graph_output_map) {
        auto it = value_shapes.find(name);
        if (it == value_shapes.end()) continue;
        const auto& vs = it->second;
        size_t orig_rank = vs.original_rank ? vs.original_rank : vs.sizes.size();
        size_t skip = vs.sizes.size() > orig_rank ? vs.sizes.size() - orig_rank : 0;
        out_variant.output_dims[out_idx].reserve(vs.sizes.size() - skip);
        for (size_t d = skip; d < vs.sizes.size(); ++d)
            out_variant.output_dims[out_idx].push_back(static_cast<int64_t>(vs.sizes[d]));
        out_variant.output_bytes[out_idx] = vs.total_bytes;
    }

    // Free the host (CPU) byte-copies of the large weights now that they are resident
    // in VRAM (kernel_state->shared_initializers). The snapshot otherwise holds a full
    // ~14 GB second copy of every weight for the whole session (released only in
    // ~DeferredSubgraph); on a UMA APU that host copy plus the VRAM copy exhausts system
    // RAM and leaves no headroom for WDDM to page the over-budget VRAM. ORT frees each
    // initializer right after upload (RemoveInitializedTensor); this mirrors that.
    //
    // Only the shared (large) weights are freed — they are bound from VRAM every dispatch
    // and are re-read from the OrtValue by NOTHING after this point: the upload loop is
    // guarded by "already uploaded", and later-variant compiles get their shape from
    // snap.freed_initializer_shapes (seeded below) instead of the tensor. Small control
    // initializers (Reshape shapes, scales, axes, Range scalars) are NOT freed — some
    // translators read their bytes on every compile.
    // NOTE: iterate out_variant.shared_initializer_slots — the local
    // shared_initializer_slots was std::move'd into out_variant just above.
    if (!out_variant.shared_initializer_slots.empty()) {
        size_t freed = 0;
        uint64_t freed_bytes = 0;
        for (const auto& [dml_i, name] : out_variant.shared_initializer_slots) {
            auto owned_it = snap.owned_initializers.find(name);
            if (owned_it == snap.owned_initializers.end() || !owned_it->second)
                continue;  // not an owned copy (e.g. external-data view) — nothing to free

            // Cache ONNX dtype + original dims BEFORE releasing, so later-variant
            // compiles can rebuild value_shapes and the InferShapes initializer.
            DeferredSubgraph::FreedInitializerInfo fi;
            OrtTensorTypeAndShapeInfo* tsi = nullptr;
            if (ort_api.GetTensorTypeAndShape(owned_it->second, &tsi) == nullptr && tsi) {
                ort_api.GetTensorElementType(tsi, &fi.onnx_dtype);
                size_t rank = 0;
                ort_api.GetDimensionsCount(tsi, &rank);
                fi.dims.resize(rank, 0);
                if (rank > 0) ort_api.GetDimensions(tsi, fi.dims.data(), rank);
                ort_api.ReleaseTensorTypeAndShapeInfo(tsi);
                snap.freed_initializer_shapes[name] = std::move(fi);
            }

            auto vs_it = value_shapes.find(name);
            if (vs_it != value_shapes.end())
                freed_bytes += vs_it->second.total_bytes;
            ort_api.ReleaseValue(owned_it->second);
            snap.owned_initializers.erase(owned_it);
            // Keep the NAME in initializer_view mapped to nullptr rather than erasing:
            // later-variant compiles do many name-presence checks (.count) via
            // all_initializers (ResolveToInitializer, is_large_initializer, input-map
            // classification) that must still recognize this as an initializer. Every
            // site that DEREFERENCES the OrtValue is guarded by `if (!val) continue`
            // and falls back to freed_initializer_shapes for dtype/dims. Setting null
            // (not erasing) preserves the name while releasing the ~14 GB of bytes.
            snap.initializer_view[name] = nullptr;
            ++freed;
        }
        DML_PERF_LOG("[RuntimeFusion] freed CPU weight copies: ", freed,
                     " tensors, ", freed_bytes / (1024 * 1024), " MB host RAM released"
                     " (slots=", out_variant.shared_initializer_slots.size(),
                     " owned=", snap.owned_initializers.size(), ")\n");
    }

    DML_PERF_LOG("[RuntimeFusion] CompileFromSnapshot OK: variant compiled\n");
    return true;
}

bool FullGraphFusion::TryTranslateNodes(
    const OrtApi&                                            ort_api,
    const OrtGraph*                                          main_graph,
    const std::unordered_map<std::string, const OrtValue*>&  initializers,
    const std::vector<const OrtNode*>&                       nodes,
    const std::unordered_map<std::string, std::vector<int64_t>>& resolved_shapes,
    std::unordered_map<std::string, std::vector<int64_t>>*   out_shapes)
{
    SubgraphInfo subgraph = BuildSubgraphInfo(ort_api, main_graph, initializers, resolved_shapes);

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
        auto translated = reg_it->second(ort_api, NodeView{ort_api, node}, subgraph.value_shapes, subgraph.all_initializers);
        if (!translated) {
            DML_PERF_LOG("[TryTranslateNodes] FAIL: op=", op_type);
            for (size_t i = 0; i < input_names.size(); ++i) {
                DML_PERF_LOG(" in", i, "='", input_names[i], "'(", subgraph.value_shapes.count(input_names[i]), ")");
            }
            DML_PERF_LOG(" out0=", (output_names.empty() ? "?" : output_names[0]), "\n");
            all_translated = false;
            continue;  // keep translating for shape collection
        }
        DML_PERF_LOG("[TryTranslateNodes] OK: op=", op_type,
            " out0=", (output_names.empty() ? "?" : output_names[0]), "\n");
        // Write-back: translator computed output shapes; seed them for downstream.
        for (size_t k = 0; k < output_names.size() && k < translated->output_tensors.size(); ++k) {
            if (!subgraph.value_shapes.count(output_names[k])) {
                subgraph.value_shapes[output_names[k]] = translated->output_tensors[k];
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
        // Export the UNPADDED dims: info.sizes is padded to 4D, but original_rank
        // records the true rank. Exporting the padded [1,1,1,1] for a scalar/low-
        // rank tensor would round-trip through resolved_shapes and be re-seeded
        // (SeedFromVi) with original_rank = 4, corrupting a downstream partition's
        // reported output rank the same way the Squeeze->identity scalar did.
        auto ExportDims = [&](const std::string& name, const DmlTensorInfo& info) {
            if (out_shapes->count(name)) return;
            size_t rank = (info.original_rank && info.original_rank <= info.sizes.size())
                ? info.original_rank : info.sizes.size();
            size_t skip = info.sizes.size() - rank;  // strip leading padding 1s
            std::vector<int64_t> dims(rank);
            for (size_t d = 0; d < rank; ++d)
                dims[d] = static_cast<int64_t>(info.sizes[skip + d]);
            (*out_shapes)[name] = std::move(dims);
        };

        for (const auto& [name, info] : subgraph.value_shapes) {
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
                auto vs_it = subgraph.value_shapes.find(in_names[k]);
                if (vs_it != subgraph.value_shapes.end()) {
                    ExportDims(in_names[k], vs_it->second);
                    continue;
                }
                // Try: VI name → producer output name → value_shapes.
                auto prod_it = producer_output_name.find(vi_name);
                if (prod_it != producer_output_name.end()) {
                    vs_it = subgraph.value_shapes.find(prod_it->second);
                    if (vs_it != subgraph.value_shapes.end()) {
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
    SubgraphInfo subgraph = BuildSubgraphInfo(ort_api, main_graph, initializers, resolved_shapes);
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
        if (!op_type) {
            DiagLog("[TryCompilePartition] translate FAIL: null op_type\n");
            return false;
        }
        auto reg_it = registry.find(op_type);
        if (reg_it == registry.end()) {
            DiagLog(std::string("[TryCompilePartition] translate FAIL: no translator for op=") + op_type + "\n");
            return false;
        }
        auto input_names  = fusion_utils::GetNodeInputNames(ort_api, node);
        auto output_names = fusion_utils::GetNodeOutputNames(ort_api, node);
        auto translated = reg_it->second(ort_api, NodeView{ort_api, node}, subgraph.value_shapes, subgraph.all_initializers);
        if (!translated) {
            DiagLog(std::string("[TryCompilePartition] translate FAIL: op=") + op_type +
                " out0=" + (output_names.empty() ? "?" : output_names[0]) + "\n");
            return false;
        }
        // Write-back: translator computed output shapes; seed them for downstream.
        for (size_t k = 0; k < output_names.size() && k < translated->output_tensors.size(); ++k) {
            if (!subgraph.value_shapes.count(output_names[k]))
                subgraph.value_shapes[output_names[k]] = translated->output_tensors[k];
        }
        auto translated_ptr = std::make_unique<TranslatedOp>(std::move(*translated));
        translated_ptr->FixupPointers();
        // Mirror Compile: a passthrough whose output is a partition output must
        // be materialized as a real identity node (an elided alias has no DML
        // node to source the output edge). Keeping this in lockstep with Compile
        // ensures the pre-flight verdict matches the actual compile.
        bool passthrough_is_graph_output = false;
        if (translated_ptr->passthrough) {
            for (const auto& oname : output_names)
                if (subgraph.graph_output_map.count(oname)) { passthrough_is_graph_output = true; break; }
        }
        if (translated_ptr->passthrough && !passthrough_is_graph_output) {
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
        if (passthrough_is_graph_output) {
            DmlTensorInfo out_info = translated_ptr->output_tensors.empty()
                ? DmlTensorInfo{} : translated_ptr->output_tensors[0];
            translated_ptr = std::make_unique<TranslatedOp>(BuildIdentityOp(out_info));
        }
        size_t dml_input_count = translated_ptr->input_tensors.size();
        for (size_t s = 0; s < input_names.size() && s < dml_input_count; ++s) {
            size_t name_idx = translated_ptr->input_name_reorder.empty()
                ? s : translated_ptr->input_name_reorder[s];
            if (name_idx >= input_names.size()) continue;
            if (subgraph.all_initializers.count(input_names[name_idx]))
                consumed_initializer_names.insert(input_names[name_idx]);
        }
        size_t idx = compiled_nodes.size();
        for (size_t k = 0; k < output_names.size(); ++k)
            value_producer[output_names[k]] = {idx, k};
        CompiledNode compiled_node;
        compiled_node.translated = std::move(translated_ptr);
        compiled_node.input_names = std::move(input_names);
        compiled_node.output_names = std::move(output_names);
        compiled_node.op_type = op_type;
        compiled_nodes.push_back(std::move(compiled_node));
    }
    if (compiled_nodes.empty()) return true;

    // CreateOperator.
    ComPtr<IDMLDevice> dml_device;
    if (FAILED(provider->GetDmlDevice(dml_device.GetAddressOf()))) {
        DiagLog("[TryCompilePartition] GetDmlDevice failed\n");
        return false;
    }
    ComPtr<IDMLDevice1> dml_device1;
    if (FAILED(dml_device.As(&dml_device1))) {
        DiagLog("[TryCompilePartition] IDMLDevice1 QI failed\n");
        return false;
    }
    for (auto& compiled_node : compiled_nodes) {
        HRESULT hr_op = dml_device->CreateOperator(&compiled_node.translated->op_desc,
                IID_PPV_ARGS(compiled_node.translated->dml_operator.GetAddressOf()));
        if (FAILED(hr_op)) {
            char hr_buf[32];
            snprintf(hr_buf, sizeof(hr_buf), "0x%08X", (unsigned)hr_op);
            DiagLog(std::string("[TryCompilePartition] CreateOperator FAILED HR=") + hr_buf +
                " op=" + compiled_node.op_type + "\n");
            auto dump_tensors = [](const char* tag, const std::vector<DmlTensorInfo>& tensors) {
                for (size_t ti = 0; ti < tensors.size(); ++ti) {
                    const auto& t = tensors[ti];
                    std::string s = std::string("[TryCompilePartition]   ") + tag + "[" +
                        std::to_string(ti) + "] dtype=" + std::to_string((int)t.data_type) +
                        " rank=" + std::to_string(t.original_rank) + " sizes=[";
                    for (size_t d = 0; d < t.sizes.size(); ++d)
                        s += (d ? "," : "") + std::to_string(t.sizes[d]);
                    s += "] strides=[";
                    for (size_t d = 0; d < t.strides.size(); ++d)
                        s += (d ? "," : "") + std::to_string(t.strides[d]);
                    s += "] total_bytes=" + std::to_string(t.total_bytes) + "\n";
                    DiagLog(s);
                }
            };
            dump_tensors("in",  compiled_node.translated->input_tensors);
            dump_tensors("out", compiled_node.translated->output_tensors);
            return false;
        }
        for (size_t s = 0; s < compiled_node.translated->sub_nodes.size(); ++s) {
            auto& sn = compiled_node.translated->sub_nodes[s];
            HRESULT hr_sn = dml_device->CreateOperator(&sn.op_desc,
                    IID_PPV_ARGS(sn.dml_operator.GetAddressOf()));
            if (FAILED(hr_sn)) {
                char hr_buf[32];
                snprintf(hr_buf, sizeof(hr_buf), "0x%08X", (unsigned)hr_sn);
                DiagLog(std::string("[TryCompilePartition] CreateOperator FAILED (sub_node) HR=") + hr_buf +
                    " op=" + compiled_node.op_type + " sub=" + std::to_string(s) + "\n");
                return false;
            }
        }
    }

    // Build graph descriptor (minimal — just enough for CompileGraph validation).
    //
    // TryCompilePartition operates on the main graph, but the DML graph being
    // compiled covers only `nodes` (a partition). Its inputs are the partition
    // boundary: values consumed inside `nodes` but not produced by any node in
    // `nodes`. Using subgraph.graph_input_vis (the full model's inputs) here would
    // set the wrong InputCount and produce E_INVALIDARG from CompileGraph.
    //
    // Compute boundary inputs: walk each *non-passthrough* partition node's inputs
    // (compiled_nodes only — passthrough nodes have no DML representation), keep
    // names that are not produced inside the partition (value_producer) and not
    // initializers (handled separately via consumed_initializer_names below).
    // boundary_input_names holds alias-resolved names — the same names that the
    // edge-wiring loop will look up in input_map.dml_input_map after calling
    //   while (graph_input_aliases.count(name)) name = graph_input_aliases[name];
    // So we must key the map on the resolved name, not the raw input name.
    std::vector<std::string> boundary_input_names;
    {
        std::unordered_set<std::string> seen;
        for (const auto& compiled_node : compiled_nodes) {
            size_t num_primary_inputs = compiled_node.translated->primary_input_count.value_or(
                compiled_node.translated->input_tensors.size());
            for (size_t s = 0; s < compiled_node.input_names.size() && s < num_primary_inputs; ++s) {
                size_t name_idx = compiled_node.translated->input_name_reorder.empty()
                    ? s : compiled_node.translated->input_name_reorder[s];
                if (name_idx >= compiled_node.input_names.size()) continue;
                const auto& raw_name = compiled_node.input_names[name_idx];
                if (raw_name.empty()) continue;
                std::string name = raw_name;
                while (graph_input_aliases.count(name)) name = graph_input_aliases.at(name);
                if (value_producer.count(name)) continue;
                if (subgraph.all_initializers.count(name)) { consumed_initializer_names.insert(name); continue; }
                if (seen.insert(name).second) boundary_input_names.push_back(name);
            }
        }
    }

    // Build a minimal DmlInputMapResult equivalent for the partition boundary.
    // We replicate BuildDmlInputMap's logic without needing OrtValueInfo* pointers.
    DmlInputMapResult input_map;
    input_map.subgraph_to_dml_input.assign(boundary_input_names.size(), SIZE_MAX);
    // Pass 1: inline small consumed initializers as DML constant nodes.
    for (const auto& init_name : consumed_initializer_names) {
        auto shape_it = subgraph.value_shapes.find(init_name);
        uint64_t bytes = (shape_it != subgraph.value_shapes.end()) ? shape_it->second.total_bytes : 0;
        if (bytes == 0 || bytes >= kMaxConstNodeDataSize) continue;
        auto init_it = subgraph.all_initializers.find(init_name);
        if (init_it == subgraph.all_initializers.end() || !init_it->second) continue;
        void* cpu_ptr = nullptr;
        OrtStatus* st = ort_api.GetTensorMutableData(
            const_cast<OrtValue*>(init_it->second), &cpu_ptr);
        if (!st && cpu_ptr) {
            OrtTensorTypeAndShapeInfo* tsi = nullptr;
            ort_api.GetTensorTypeAndShape(const_cast<OrtValue*>(init_it->second), &tsi);
            size_t elem_count = 0;
            ONNXTensorElementDataType dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            if (tsi) {
                ort_api.GetTensorShapeElementCount(tsi, &elem_count);
                ort_api.GetTensorElementType(tsi, &dt);
                ort_api.ReleaseTensorTypeAndShapeInfo(tsi);
            }
            uint64_t actual_bytes = elem_count * DmlDataTypeSize(OnnxDtypeToDml(dt));
            uint64_t data_size = (actual_bytes > 0 && actual_bytes < bytes) ? actual_bytes : bytes;

            ConstantNodeInfo cni;
            cni.name = init_name;
            cni.data.resize(static_cast<size_t>(data_size), 0);
            std::memcpy(cni.data.data(), cpu_ptr, static_cast<size_t>(data_size));
            input_map.constant_node_map[init_name] = input_map.constant_nodes.size();
            input_map.constant_nodes.push_back(std::move(cni));
        }
        if (st) ort_api.ReleaseStatus(st);
    }
    // Pass 2: assign DML input indices for partition boundary inputs.
    size_t dml_input_idx = 0;
    for (size_t i = 0; i < boundary_input_names.size(); ++i) {
        const auto& n = boundary_input_names[i];
        if (input_map.constant_node_map.count(n)) continue; // inlined — no slot
        input_map.dml_input_map[n] = dml_input_idx;
        input_map.subgraph_to_dml_input[i] = dml_input_idx;
        ++dml_input_idx;
    }
    // Pass 3: large consumed initializers not in boundary_input_names.
    for (const auto& init_name : consumed_initializer_names) {
        if (input_map.constant_node_map.count(init_name)) continue;
        if (input_map.dml_input_map.count(init_name)) continue;
        input_map.dml_input_map[init_name] = dml_input_idx++;
        input_map.ordered_initializer_names.push_back(init_name);
    }
    input_map.total_dml_inputs = dml_input_idx;

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
    std::vector<DML_CONSTANT_DATA_GRAPH_NODE_DESC> const_descs(input_map.constant_nodes.size());
    for (size_t c = 0; c < input_map.constant_nodes.size(); ++c) {
        const_descs[c].Data = input_map.constant_nodes[c].data.data();
        const_descs[c].DataSize = input_map.constant_nodes[c].data.size();
        graph_nodes.push_back({DML_GRAPH_NODE_TYPE_CONSTANT, &const_descs[c]});
    }
    size_t const_node_base = total_dml_nodes;

    // Wire edges (same logic as Compile's step 9, simplified to just detect validity).
    std::vector<DML_INPUT_GRAPH_EDGE_DESC> ie;
    std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> me;
    std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> oe;

    for (size_t node_idx = 0; node_idx < compiled_nodes.size(); ++node_idx) {
        const auto& compiled_node = compiled_nodes[node_idx];
        size_t pdml = dml_node_offset[node_idx];
        size_t num_primary_inputs = compiled_node.translated->primary_input_count.value_or(
            compiled_node.translated->input_tensors.size());
        for (size_t s = 0; s < compiled_node.input_names.size() && s < num_primary_inputs; ++s) {
            size_t name_idx = compiled_node.translated->input_name_reorder.empty()
                ? s : compiled_node.translated->input_name_reorder[s];
            if (name_idx >= compiled_node.input_names.size()) continue;
            auto name = compiled_node.input_names[name_idx];
            if (name.empty()) continue;
            while (graph_input_aliases.count(name)) name = graph_input_aliases[name];
            size_t dml_slot = compiled_node.translated->dml_input_slot_indices.empty()
                ? s : compiled_node.translated->dml_input_slot_indices[s];
            if (input_map.constant_node_map.count(name)) {
                me.push_back({(UINT)(const_node_base+input_map.constant_node_map[name]),0,(UINT)pdml,(UINT)dml_slot});
            } else if (input_map.dml_input_map.count(name)) {
                ie.push_back({(UINT)input_map.dml_input_map[name],(UINT)pdml,(UINT)dml_slot});
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
                DiagLog(std::string("[TryCompilePartition] UNRESOLVED input edge: node_idx=") +
                    std::to_string(node_idx) + " op=" + compiled_node.op_type +
                    " slot=" + std::to_string(s) + " name='" + name + "'\n");
                return false;
            }
        }

        // Wire internal edges for sub_nodes (mirrors Compile step 9 sub-node section).
        for (size_t s = 0; s < compiled_node.translated->sub_nodes.size(); ++s) {
            const auto& sn = compiled_node.translated->sub_nodes[s];
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
                if (onnx_idx >= compiled_node.input_names.size()) continue;
                auto gi_name = compiled_node.input_names[onnx_idx];
                while (graph_input_aliases.count(gi_name))
                    gi_name = graph_input_aliases[gi_name];
                if (input_map.constant_node_map.count(gi_name)) {
                    me.push_back({(UINT)(const_node_base + input_map.constant_node_map[gi_name]),
                                  0, (UINT)sn_dml_idx, (UINT)to_input});
                } else if (input_map.dml_input_map.count(gi_name)) {
                    ie.push_back({(UINT)input_map.dml_input_map[gi_name],
                                  (UINT)sn_dml_idx, (UINT)to_input});
                } else if (value_producer.count(gi_name)) {
                    // Partition-internal producer (e.g. a Resize feeding a concat
                    // sub_node dequant). Mirror the primary-input producer logic.
                    auto [pci, pos] = value_producer[gi_name];
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
                    me.push_back({(UINT)pdi, foi, (UINT)sn_dml_idx, (UINT)to_input});
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
    for (size_t node_idx = 0; node_idx < compiled_nodes.size(); ++node_idx) {
        for (size_t k = 0; k < compiled_nodes[node_idx].output_names.size(); ++k) {
            size_t pdi = dml_node_offset[node_idx];
            UINT foi;
            const auto& osrc = compiled_nodes[node_idx].translated->output_source;
            if (!osrc.empty() && k < osrc.size()) {
                auto [ss, sl] = osrc[k]; if (ss>=0) pdi+=1+ss; foi=(UINT)sl;
            } else {
                if (!compiled_nodes[node_idx].translated->sub_nodes.empty())
                    pdi += compiled_nodes[node_idx].translated->sub_nodes.size();
                foi = (UINT)k;
            }
            if (wired_from.count((UINT)pdi)) continue; // already has an outgoing edge
            oe.push_back({(UINT)pdi, foi, (UINT)oi_idx++});
        }
    }

    // Output edges for constant nodes with no outgoing intermediate edge.
    for (size_t c = 0; c < input_map.constant_nodes.size(); ++c) {
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
    gd.InputCount = (UINT)input_map.total_dml_inputs;
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
        for (const auto& compiled_node : compiled_nodes) {
            for (const auto& t : compiled_node.translated->input_tensors)
                if (t.data_type == DML_TENSOR_DATA_TYPE_FLOAT16) has_fp16 = true;
                else if (t.data_type == DML_TENSOR_DATA_TYPE_FLOAT32) has_fp32 = true;
            for (const auto& t : compiled_node.translated->output_tensors)
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
