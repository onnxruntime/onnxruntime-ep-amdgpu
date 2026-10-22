// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// NodeView — a single seam that lets the ~150 op translators read a node's
// identity, input/output names, and attributes from EITHER a live ORT graph
// node (const OrtNode* + OrtApi, valid only during Compile) OR an owned
// snapshot captured for the runtime-fusion (deferred-compile) path, whose
// DML compile runs AFTER Compile returns and the OrtGraph is freed.
//
// The static Tier-0 path constructs a live NodeView; behavior is byte-identical
// to passing the raw const OrtNode* it used before. The deferred path
// constructs a snapshot-backed NodeView. Translators change ONLY in signature
// (const OrtNode* -> const NodeView&); their bodies keep calling
// GetInputNames(ort_api, node) / OrtNodeAdapter(node, ort_api) unchanged,
// because those overloads/ctors accept a NodeView.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <onnxruntime_c_api.h>

// AttributeProto — owned attribute storage for the snapshot path.
namespace ONNX_NAMESPACE { class AttributeProto; }

namespace dml_ep {

// ---------------------------------------------------------------------------
// DeferredNode — a fully-owned snapshot of one ONNX node, captured during
// Compile while the live OrtNode is still valid. Holds NO borrowed pointers:
// every string is copied, and every attribute is pre-materialized into an
// owned AttributeProto (translators only read attributes lazily via the
// adapter, so the proto MUST be built up-front while the node is alive).
// ---------------------------------------------------------------------------
struct DeferredNode {
    size_t                    id = 0;
    std::string               name;
    std::string               op_type;
    std::string               domain;
    int                       since_version = 0;

    std::vector<std::string>  input_names;
    std::vector<std::string>  output_names;

    // attr name -> owned proto (pre-built during snapshot). shared_ptr so the
    // DeferredNode stays cheaply movable/copyable inside the snapshot vector.
    std::unordered_map<std::string, std::shared_ptr<ONNX_NAMESPACE::AttributeProto>> attr_protos;

    // value name -> original (pre-4D-pad) rank, captured from live ORT type info.
    // Only needed by rank-sensitive translators (e.g. ScatterND) that today read
    // it off the live node; the snapshot reproduces it here.
    std::unordered_map<std::string, size_t> original_ranks;
};

// ---------------------------------------------------------------------------
// NodeView — non-owning view over EITHER a live node or a DeferredNode.
// Cheap to pass by const ref. Exactly one of the two backings is set.
// ---------------------------------------------------------------------------
class NodeView {
public:
    // Live backing (static Tier-0 path). The OrtNode* is valid for the lifetime
    // of this view (i.e. within Compile). Behavior mirrors the old raw pointer.
    NodeView(const OrtApi& ort_api, const OrtNode* node) noexcept
        : ort_api_(&ort_api), live_node_(node), snapshot_(nullptr) {}

    // Snapshot backing (deferred path). The DeferredNode is owned elsewhere
    // (the DeferredSubgraph) and outlives every use of this view.
    explicit NodeView(const DeferredNode& snapshot) noexcept
        : ort_api_(nullptr), live_node_(nullptr), snapshot_(&snapshot) {}

    bool is_live() const noexcept { return live_node_ != nullptr; }
    bool is_snapshot() const noexcept { return snapshot_ != nullptr; }

    const OrtApi*      ort_api()    const noexcept { return ort_api_; }
    const OrtNode*     live_node()  const noexcept { return live_node_; }
    const DeferredNode* snapshot()  const noexcept { return snapshot_; }

private:
    const OrtApi*       ort_api_;    // non-null iff live
    const OrtNode*      live_node_;  // non-null iff live
    const DeferredNode* snapshot_;   // non-null iff snapshot
};

}  // namespace dml_ep
