// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

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

struct FullGraphKernelState {
    PluginDmlExecutionProviderImpl*                      provider = nullptr;
    const OrtApi*                                        ort_api = nullptr;

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

    size_t                                               num_outputs = 0;
    std::vector<std::vector<int64_t>>                    output_dims;
    std::vector<uint64_t>                                output_bytes;

    uint64_t                                             compute_call_count = 0;
};

class FullGraphFusion {
public:
    // Check whether Tier-0 full-graph fusion is safe to attempt for these
    // nodes. Verifies every node input and output has tensor type info with a
    // supported DML dtype, rank > 0, and all-static dims. Does not run
    // translators or touch DML.
    static bool ValidateTier0(
        const OrtApi&                                            ort_api,
        const std::vector<const OrtNode*>&                       nodes);

    // Compile a Tier-0 fused subgraph.  Returns an OrtNodeComputeInfo on
    // success, or nullptr if compilation fails (caller falls back to Tier-2/1).
    static OrtNodeComputeInfo* Compile(
        const OrtApi&                                            ort_api,
        const OrtGraph*                                          fused_subgraph,
        const std::unordered_map<std::string, const OrtValue*>&  initializers,
        PluginDmlExecutionProviderImpl*                          provider);

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
        PluginDmlExecutionProviderImpl*                          provider);
};

}  // namespace dml_ep
