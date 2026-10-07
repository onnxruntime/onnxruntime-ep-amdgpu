// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <DirectML.h>
#include <onnxruntime_c_api.h>
#include <wrl/client.h>

#include "dml_op_translators.h"

struct OrtNodeComputeInfo;

namespace dml_ep {

class PluginDmlExecutionProviderImpl;

// ---------------------------------------------------------------------------
// DeferredSubgraph — a fully-owned snapshot of one dynamic partition, captured
// during CompileImpl while the OrtGraph is still alive, so its DML compile can
// run later at first Compute (when concrete shapes are known) after the graph
// has been freed. Holds NO borrowed OrtGraph/OrtNode*/OrtValue* pointers.
// ---------------------------------------------------------------------------
struct DeferredSubgraph {
    // Owned node snapshots in topological order (attrs pre-materialized).
    std::vector<DeferredNode> nodes;

    // Owned initializer tensors: name -> OrtValue* allocated via
    // CreateTensorAsOrtValue with bytes copied in. Released in the destructor.
    // The parallel `initializer_view` map exposes them as const OrtValue* keyed
    // by name, matching the `all_initializers` interface the compile core uses.
    std::unordered_map<std::string, OrtValue*>       owned_initializers;
    std::unordered_map<std::string, const OrtValue*> initializer_view;

    // Shape cache for large weights whose CPU byte-copy has been freed after upload
    // to VRAM. The first variant's compile records each freed initializer's ONNX
    // dtype + dims here (captured from the OrtValue before ReleaseValue); once the
    // tensor is gone, later-variant compiles rebuild both value_shapes AND the ONNX
    // InferShapes GraphProto initializer (dims-only) from this cache. Freeing the
    // ~14 GB of host weight copies after they are resident in VRAM is what keeps the
    // plugin's system-RAM use in line with ORT.
    struct FreedInitializerInfo {
        ONNXTensorElementDataType onnx_dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        std::vector<int64_t>      dims;   // original ONNX dims (un-padded)
    };
    std::unordered_map<std::string, FreedInitializerInfo> freed_initializer_shapes;

    // Names of the partition's graph inputs (runtime inputs), in KernelContext
    // order — used at first Compute to read concrete dims and form the signature.
    std::vector<std::string> graph_input_names;

    // Element type per graph-input name, captured from the live ValueInfo at
    // snapshot time. Feeds the GraphProto input value_info for ONNX shape
    // inference (avoids the FLOAT32 default guess for non-fp32 activations).
    std::unordered_map<std::string, ONNXTensorElementDataType> graph_input_dtypes;

    // ONNX opset per domain ("" = ai.onnx), derived from the max node
    // since_version per domain. Passed to onnx::shape_inference::InferShapes.
    std::unordered_map<std::string, int> opset_imports;

    // CPU-preferred shape-math nodes (Shape/Concat/Gather/Slice/Cast/Mul/Div/...)
    // that feed the partition's BOUNDARY inputs but are EXCLUDED from the fused
    // DML graph. Captured during GetCapability (where they are visible) in
    // topological order. At first Compute they are constant-folded from the
    // concrete runtime dims of `shape_prep_root_inputs` to recover data-dependent
    // reshape targets (e.g. pos_ids). These NEVER enter `nodes` / the DML graph —
    // they are for shape resolution only.
    std::vector<DeferredNode> shape_prep_nodes;

    // Real graph-input names the shape-math chain roots at (e.g. "input_ids").
    // Seeded with concrete runtime dims (as [dim0,dim1,...] values) before folding.
    std::vector<std::string> shape_prep_root_inputs;

    // Integer initializer VALUES consumed by the shape-math chain (e.g. a Gather
    // index [0], a Concat/Slice constant). Captured at GetCapability from the full
    // graph's initializer map (the chain lives OUTSIDE the partition so these are
    // not in initializer_view). Seed the folder so Gather/Slice/Concat resolve.
    std::unordered_map<std::string, std::vector<int64_t>> shape_prep_initializers;

    // Needed to release owned_initializers. Set at build time; not owned.
    const OrtApi* ort_api = nullptr;

    DeferredSubgraph() = default;
    DeferredSubgraph(const DeferredSubgraph&) = delete;
    DeferredSubgraph& operator=(const DeferredSubgraph&) = delete;
    DeferredSubgraph(DeferredSubgraph&&) = default;
    DeferredSubgraph& operator=(DeferredSubgraph&&) = default;
    ~DeferredSubgraph();
};

// Returns true if all graph inputs have fully static shapes (no dynamic dims).
bool AllGraphInputsStatic(const OrtApi& ort_api, const OrtGraph* graph);

// Returns true if the op translator registry has a translator for every
// supported node (non-CPU-preferred, non-control-flow) in the given set.
bool AllNodesHaveTranslators(
    const OrtApi& ort_api,
    const OpTranslatorRegistry& registry,
    const std::vector<const OrtNode*>& nodes);

// ---------------------------------------------------------------------------
// FullGraphFusion — Tier-0 whole-graph compilation
//
// Translates every node in a fused subgraph to a DML_OPERATOR_DESC,
// wires them into a single DML_GRAPH_DESC, and compiles via
// IDMLDevice1::CompileGraph.  The resulting IDMLCompiledOperator
// executes the entire inference graph in one GPU dispatch.
// ---------------------------------------------------------------------------

// One compiled DML variant: everything produced by a single CompileGraph +
// binding-metadata build. The STATIC path fills the mirror fields directly on
// FullGraphKernelState below (a single implicit variant), keeping that path
// byte-identical. The DEFERRED path builds one CompiledVariant per distinct
// input-shape signature at first Compute and caches them.
struct CompiledVariant {
    Microsoft::WRL::ComPtr<IDMLCompiledOperator>         compiled_op;
    Microsoft::WRL::ComPtr<ID3D12Resource>               persistent_resource;
    Microsoft::WRL::ComPtr<IUnknown>                     persistent_allocator;
    std::optional<DML_BUFFER_BINDING>                    persistent_binding;

    size_t                                               num_runtime_inputs = 0;
    size_t                                               num_subgraph_inputs = 0;
    std::vector<size_t>                                  subgraph_to_dml_input;
    std::vector<uint64_t>                                runtime_input_bytes;
    std::vector<bool>                                    runtime_input_is_owned;
    std::unordered_set<size_t>                           dml_inputs_with_edges;

    size_t                                               num_initializers = 0;

    // Deferred path: DML graph-input slots that are fed by a large initializer whose
    // weight lives in FullGraphKernelState::shared_initializers (bound at execute time,
    // not owned/baked). Maps DML input index -> onnx initializer name so FullGraph_Compute
    // can resolve each slot to the shared GPU resource. Empty on the static path.
    std::vector<std::pair<size_t, std::string>>          shared_initializer_slots;

    size_t                                               num_outputs = 0;
    std::vector<std::vector<int64_t>>                    output_dims;
    std::vector<uint64_t>                                output_bytes;
};

struct FullGraphKernelState {
    PluginDmlExecutionProviderImpl*                      provider = nullptr;
    const OrtApi*                                        ort_api = nullptr;

    // --- STATIC path (unchanged): the single active compiled variant, inline. ---
    Microsoft::WRL::ComPtr<IDMLCompiledOperator>         compiled_op;
    Microsoft::WRL::ComPtr<ID3D12Resource>               persistent_resource;
    Microsoft::WRL::ComPtr<IUnknown>                     persistent_allocator;
    std::optional<DML_BUFFER_BINDING>                    persistent_binding;

    size_t                                               num_runtime_inputs = 0;
    size_t                                               num_subgraph_inputs = 0;
    std::vector<size_t>                                  subgraph_to_dml_input;
    std::vector<uint64_t>                                runtime_input_bytes;
    std::vector<bool>                                    runtime_input_is_owned;
    std::unordered_set<size_t>                           dml_inputs_with_edges;

    size_t                                               num_initializers = 0;

    // Mirror of CompiledVariant::shared_initializer_slots for the active variant
    // (deferred path). Empty on the static path.
    std::vector<std::pair<size_t, std::string>>          shared_initializer_slots;

    size_t                                               num_outputs = 0;
    std::vector<std::vector<int64_t>>                    output_dims;
    std::vector<uint64_t>                                output_bytes;

    uint64_t                                             compute_call_count = 0;

    // --- DEFERRED (runtime-fusion) path. Only set when `deferred` is true. ---
    bool                                                 deferred = false;
    std::unique_ptr<DeferredSubgraph>                    snapshot;  // owned partition
    std::unordered_map<size_t, CompiledVariant>          signature_cache;  // sig -> variant
    CompiledVariant*                                     active = nullptr;  // current variant

    // Deferred path: initializer weights uploaded ONCE to GPU and shared across ALL
    // shape variants, bound as runtime inputs each dispatch. Replaces the old
    // per-variant DML_TENSOR_FLAG_OWNED_BY_DML persistent-resource copy, which
    // re-baked the (byte-identical) weights into every variant and tripled VRAM
    // (3 variants x 4.16 GB -> OVER_BUDGET -> PCIe paging -> ~2x slowdown). Now the
    // weights live once here on the kernel state (outlives signature_cache), mirroring
    // ORT's persistent=4-byte binding of weights as ordinary graph inputs.
    struct SharedInitializer {
        Microsoft::WRL::ComPtr<ID3D12Resource> gpu_resource;
        Microsoft::WRL::ComPtr<IUnknown>       allocator_ref;
        uint64_t                               bytes = 0;
    };
    std::unordered_map<std::string, SharedInitializer>   shared_initializers;  // by onnx name

    // Copy a cached/just-built variant into the inline STATIC mirror fields, so
    // FullGraph_Compute's existing binding loop reads the active variant with no
    // branching. Called after a signature HIT or a fresh compile.
    void ActivateVariant(CompiledVariant& v) {
        active = &v;
        compiled_op            = v.compiled_op;
        persistent_resource    = v.persistent_resource;
        persistent_allocator   = v.persistent_allocator;
        persistent_binding     = v.persistent_binding;
        num_runtime_inputs     = v.num_runtime_inputs;
        num_subgraph_inputs    = v.num_subgraph_inputs;
        subgraph_to_dml_input  = v.subgraph_to_dml_input;
        runtime_input_bytes    = v.runtime_input_bytes;
        runtime_input_is_owned = v.runtime_input_is_owned;
        dml_inputs_with_edges  = v.dml_inputs_with_edges;
        num_initializers       = v.num_initializers;
        shared_initializer_slots = v.shared_initializer_slots;
        num_outputs            = v.num_outputs;
        output_dims            = v.output_dims;
        output_bytes           = v.output_bytes;
    }
};

class FullGraphFusion {
public:
    // Check whether Tier-0 full-graph fusion is safe to attempt for these
    // nodes. Verifies every node input and output has tensor type info with a
    // supported DML dtype, rank > 0, and all-static dims. Does not run
    // translators or touch DML.
    //
    // Rejects any node input/output with an empty (extent-0) dimension, since
    // DML cannot represent a zero-sized dim inside a compiled graph (mirrors
    // ORT's GraphPartitioner ContainsEmptyDimensions gate). An empty dim on a
    // constant-initializer input is exempt — it is consumed at compile time and
    // never becomes a runtime DML edge (e.g. Resize's empty roi input).
    static bool ValidateTier0(
        const OrtApi&                                            ort_api,
        const std::vector<const OrtNode*>&                       nodes,
        const std::unordered_map<std::string, std::vector<int64_t>>& resolved_shapes = {},
        const std::unordered_map<std::string, const OrtValue*>& initializers = {});

    // Shape-BLIND admission for the runtime-fusion (deferred-compile) path.
    // Unlike ValidateTier0, this does NOT reject dynamic dims — concrete shapes
    // are unknown until first Compute. It admits a partition on structure alone,
    // mirroring ORT's runtime-fusion membership test. Per node it requires:
    //   - a registered translator for the op type,
    //   - a DML-supported dtype on every typed input/output,
    //   - no empty (extent-0) edge (NodeHasEmptyEdge), and
    //   - for data-dependent-shape ops (Reshape/Expand/Tile/Resize/Upsample/
    //     Range/ConstantOfShape/NonZero/Split), the shape-driving input must be a
    //     resolved constant (present in `initializers` or `resolved_shapes`), since
    //     a non-constant shape driver cannot be compiled from a snapshot.
    // Only called when RuntimeFusionEnabled().
    static bool ValidateTier0Structural(
        const OrtApi&                                            ort_api,
        const std::vector<const OrtNode*>&                       nodes,
        const std::unordered_map<std::string, std::vector<int64_t>>& resolved_shapes = {},
        const std::unordered_map<std::string, const OrtValue*>& initializers = {});

    // Compile a Tier-0 fused subgraph.  Returns an OrtNodeComputeInfo on
    // success, or nullptr if compilation fails (caller falls back to Tier-2/1).
    static OrtNodeComputeInfo* Compile(
        const OrtApi&                                            ort_api,
        const OrtGraph*                                          fused_subgraph,
        const std::unordered_map<std::string, const OrtValue*>&  initializers,
        PluginDmlExecutionProviderImpl*                          provider,
        const std::unordered_map<std::string, std::vector<int64_t>>& resolved_shapes = {});

    // Runtime-fusion (deferred) counterpart of Compile. Instead of compiling DML
    // now, it SNAPSHOTS the dynamic partition into an owned DeferredSubgraph
    // (nodes, initializer bytes, base resolved shapes) while `fused_subgraph` is
    // still alive, and returns an OrtNodeComputeInfo whose kernel state carries
    // that snapshot. The actual DML compile happens lazily at first Compute,
    // keyed by concrete input-shape signature. Returns nullptr on snapshot
    // failure (caller returns ORT_FAIL — no per-op fallback once claimed).
    // Only used when RuntimeFusionEnabled() and the subgraph hash matched
    // m_tier0DynamicGroupHashes.
    static OrtNodeComputeInfo* CompileDeferred(
        const OrtApi&                                            ort_api,
        const OrtGraph*                                          fused_subgraph,
        const std::unordered_map<std::string, const OrtValue*>&  initializers,
        PluginDmlExecutionProviderImpl*                          provider,
        const std::unordered_map<std::string, std::vector<int64_t>>& resolved_shapes = {},
        const std::vector<DeferredNode>&                         shape_prep_nodes = {},
        const std::vector<std::string>&                          shape_prep_roots = {},
        const std::unordered_map<std::string, std::vector<int64_t>>& shape_prep_inits = {});

    // Lightweight feasibility check — runs translation + CreateOperator +
    // CompileGraph against the main graph. Returns true if CompileGraph
    // succeeds, false if it fails (E_INVALIDARG etc.). No GPU memory is
    // allocated, no weights are uploaded, no kernel state is built.
    // Called from GetCapabilityImpl before AddNodesToFuse: if false, nodes
    // are not claimed and fall through to Tier-1/2 without ORT_EP_FAIL.
    // The authoritative compile happens later in CompileImpl against the
    // fused subgraph (correct input ordering, full kernel state).
    static bool TryCompileGraph(
        const OrtApi&                                            ort_api,
        const OrtGraph*                                          main_graph,
        const std::unordered_map<std::string, const OrtValue*>&  initializers,
        PluginDmlExecutionProviderImpl*                          provider,
        const std::unordered_map<std::string, std::vector<int64_t>>& resolved_shapes = {});

    // Translator-only feasibility check for a partition node subset.
    // Runs the translator for each node using the main graph's shape info.
    // Returns false if any translator returns nullopt (unsupported attribute
    // combination, missing input, etc.) without touching DML or GPU resources.
    // Used by GetCapabilityImpl to pre-screen partition groups before claiming.
    static bool TryTranslateNodes(
        const OrtApi&                                            ort_api,
        const OrtGraph*                                          main_graph,
        const std::unordered_map<std::string, const OrtValue*>&  initializers,
        const std::vector<const OrtNode*>&                       nodes,
        const std::unordered_map<std::string, std::vector<int64_t>>& resolved_shapes = {},
        std::unordered_map<std::string, std::vector<int64_t>>*   out_shapes = nullptr);

    // Full DML compilation pre-flight for a partition node subset.
    // Runs the complete Compile pipeline (translate, CreateOperator, CompileGraph)
    // using the main graph's shape info, then discards the result.
    // Catches E_INVALIDARG from CompileGraph (e.g. graph topology errors,
    // DML graph size limits) that TryTranslateNodes cannot catch.
    // Called from GetCapabilityImpl; groups that fail are not claimed and
    // fall through to Tier-1 single-node dispatch.
    static bool TryCompilePartition(
        const OrtApi&                                            ort_api,
        const OrtGraph*                                          main_graph,
        const std::unordered_map<std::string, const OrtValue*>&  initializers,
        PluginDmlExecutionProviderImpl*                          provider,
        const std::vector<const OrtNode*>&                       nodes,
        const std::unordered_map<std::string, std::vector<int64_t>>& resolved_shapes = {});

};

}  // namespace dml_ep
