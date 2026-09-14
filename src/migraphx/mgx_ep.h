// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <ciso646>
#include <cstdint>
#include <optional>
#include <set>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <migraphx/migraphx.hpp>

#include "common/path_string.h"
#include "common/plugin_ep_utils.h"
#include "common/telemetry.h"
#include "common/murmurhash3.h"

#include "mgx_factory.h"
#include "mgx_info.h"
#include "mgx_utils.h"

namespace mgx_ep {

namespace env_var {
constexpr auto kFP16Enable = "ORT_MIGRAPHX_FP16_ENABLE"sv;
constexpr auto kBF16Enable = "ORT_MIGRAPHX_BF16_ENABLE"sv;
constexpr auto kFP8Enable = "ORT_MIGRAPHX_FP8_ENABLE"sv;
constexpr auto kINT8Enable = "ORT_MIGRAPHX_INT8_ENABLE"sv;
constexpr auto kDisableCompiledModelCaching = "ORT_MIGRAPHX_DISABLE_COMPILED_MODEL_CACHING"sv;
constexpr auto kForceRecompile = "ORT_MIGRAPHX_FORCE_RECOMPILE"sv;
constexpr auto kDumpSubgraphs = "ORT_MIGRAPHX_DUMP_SUBGRAPHS"sv;
constexpr auto kDumpEpContextModel = "ORT_MIGRAPHX_DUMP_EP_CONTEXT_MODEL"sv;
constexpr auto kINT8CalibrationTableName = "ORT_MIGRAPHX_INT8_CALIBRATION_TABLE_NAME"sv;
constexpr auto kCacheDir = "ORT_MIGRAPHX_CACHE_DIR"sv;
constexpr auto kComputeMode = "ORT_MIGRAPHX_COMPUTE_MODE"sv;
constexpr auto kINT8UseNativeCalibrationTable = "ORT_MIGRAPHX_INT8_USE_NATIVE_CALIBRATION_TABLE"sv;
constexpr auto kExhaustiveTune = "ORT_MIGRAPHX_EXHAUSTIVE_TUNE"sv;
constexpr auto kProblemCachePath = "ORT_MIGRAPHX_PROBLEM_CACHE"sv;
constexpr auto kHipGraphEnable = "ORT_MIGRAPHX_HIP_GRAPH_ENABLE"sv;
constexpr auto kMaxDynamicBatch = "ORT_MIGRAPHX_MAX_DYNAMIC_BATCH"sv;
constexpr auto kCompileBatches = "ORT_MIGRAPHX_COMPILE_BATCHES"sv;
constexpr auto kPrecompileAtLoad = "ORT_MIGRAPHX_PRECOMPILE_AT_LOAD"sv;
constexpr auto kCoalesceIO = "ORT_MIGRAPHX_COALESCE_IO"sv;
constexpr auto kMlssUseSpecificOps = "ORT_MIGRAPHX_MLSS_USE_SPECIFIC_OPS"sv;
constexpr auto kCpuControlFlow = "ORT_MIGRAPHX_CPU_CONTROL_FLOW"sv;
constexpr auto kModelArch = "ORT_MIGRAPHX_MODEL_ARCH"sv;
constexpr auto kStaticPadSeq = "ORT_MIGRAPHX_STATIC_PAD_SEQ"sv;
constexpr auto kStaticPadSeqLen = "ORT_MIGRAPHX_STATIC_PAD_SEQ_LEN"sv;
constexpr auto kStaticPadInputs = "ORT_MIGRAPHX_STATIC_PAD_INPUTS"sv;
constexpr auto kStaticPadOutputs = "ORT_MIGRAPHX_STATIC_PAD_OUTPUTS"sv;
// TEMPORARY A/B gate: when set (1/true), restore the legacy unconditional
// per-Compute hipStreamSynchronize instead of relying on stream-ordered copies
// plus the per-Run OnSessionRunEnd sync. Lets the old (full-drain) and new
// (pipelined) behaviors be compared in one build. Remove once the new path is
// confirmed and the drain is deleted for good.
constexpr auto kLegacyComputeSync = "ORT_MIGRAPHX_LEGACY_COMPUTE_SYNC"sv;
// A/B / safety gate: when set (1/true), zero ALL captured graph output buffers
// before every replay (the original behavior) instead of only the read-modify-
// write outputs detected at capture. Lets the pre-replay memset fan-out reduction
// be disabled if a model's outputs are misclassified. Default off (detect + skip).
constexpr auto kForceZeroAllGraphOutputs = "ORT_MIGRAPHX_FORCE_ZERO_ALL_OUTPUTS"sv;
// A/B gate: when set to 0/false, disable lending the EP-owned staging output buffers
// to ORT as output-tensor storage and go back to copying each output into an
// ORT-allocated tensor. Default on wherever the staging path runs.
constexpr auto kBorrowOutputs = "ORT_MIGRAPHX_BORROW_OUTPUTS"sv;
// A/B gate: when set to 0/false, disable the batch-determined shape profile, so every
// Compute call rescans every input's shape instead of deriving them from the
// representative input's batch. Default on.
constexpr auto kBatchShapeProfile = "ORT_MIGRAPHX_BATCH_SHAPE_PROFILE"sv;
// Escape hatch: when set to 1/true, downgrade the compile-time cross-batch op check from
// a hard failure to a warning. Bucketing pads a request up to the compiled batch and the
// pad rows hold stale arena data, so an op that folds axis 0 into its result corrupts the
// rows the caller reads. Only set this if you know padding cannot reach those ops.
constexpr auto kAllowCrossBatchOps = "ORT_MIGRAPHX_ALLOW_CROSS_BATCH_OPS"sv;
}  // namespace env_vars

// EP-owned device staging buffer (pointer-stable across runs so it can be
// safely baked into a captured hipGraph).  Plain hipMalloc/hipFree owned;
// freed centrally in ~ExecutionProvider.
struct StagingBuffer {
    void* data{nullptr};
    std::size_t size_bytes{};
    migraphx::shape shape{};
};

// One compiled bucket's packed input arena (coalesce_io).  Every program input gets a
// 256B-aligned slot sized for THIS bucket's shape, not for max_dynamic_batch, so the
// single H2D that flushes the arena moves exactly the bytes the bucket reads: a batch-1
// call no longer pays a max-batch-sized transfer.  One arena per compiled bucket, each
// pointer-stable for the compute state's lifetime so it is safe to bake into a captured
// hipGraph.  The host side is NOT here: all buckets share the double-buffered pinned
// gather area on ComputeState, which keeps pinned memory bounded by the largest bucket.
struct CoalesceArena {
    void* dev{nullptr};
    std::size_t bytes{};
};

// EP-owned scratch buffer bound to a MIGraphX program's "scratch" parameter.
// Owning it (instead of letting MIGraphX use its internal arena) lets us zero
// it before every capture/replay so kernels start from a deterministic
// baseline.  One per compiled program variant (keyed by shape hash).
struct ScratchBuffer {
    void* data{nullptr};
    std::size_t size_bytes{};
    migraphx::shape shape{};
};

// A captured hipGraph plus the metadata needed to replay it correctly.
struct CapturedHipGraph {
    hipGraph_t graph{nullptr};
    hipGraphExec_t exec{nullptr};
    bool captured{false};

    // MIGraphX outputs not bound to a pre-allocated buffer; their device data
    // is stable across replays and is copied out after each launch.
    struct ExtraOutput {
        std::size_t output_index{};
        std::vector<std::int64_t> ort_shape{};
        void* gpu_data{nullptr};
        std::size_t bytes{};
    };
    std::vector<ExtraOutput> extra_outputs{};

    // Scratch pointer baked into the captured kernels; a mismatch forces re-capture.
    void* captured_scratch_ptr{nullptr};

    // Output buffers (ptr + bytes) that must be zeroed before every replay
    // because some captured kernels read-modify-write their output.
    std::vector<std::pair<void*, std::size_t>> captured_output_zeroes{};
    // When true (item 7), the pre-replay zeroing (scratch + the RMW outputs above)
    // was folded into the captured graph, so replay is a single launch and the
    // out-of-graph memsets are skipped.  Reset on every (re)capture.
    bool zeroing_in_graph{false};

    // ── Direct-bind hipGraph (zero-copy) metadata ────────────────────────────
    // When direct_bind is true the graph was captured against ORT tensor
    // pointers directly (no staging copies), matching the built-in EP's
    // use_direct_hip_graph path.  The captured input/output pointer maps are
    // compared against the current ORT pointers on every replay; a mismatch
    // (e.g. ORT allocator recycled a buffer) forces a re-capture.
    bool direct_bind{false};
    // Captured ORT device pointers in DirectBindCache inputs/outputs order (not a
    // name->ptr map), compared positionally against the current pointers on every
    // replay to detect drift.  Flat vectors avoid the per-call map build + hashing.
    std::vector<void*> captured_input_ptrs{};
    std::vector<void*> captured_output_ptrs{};
};

// One input's resolved staging-copy plan, built once per compiled shape so the
// per-call input copy touches no parameter names, std::strings, or map lookups: it
// just reads the ORT tensor at ort_index and copies into the staging buffer.  For
// the coalesced arena, staging_data is a sub-view at arena_offset in the pinned
// host buffer; otherwise it is a standalone device buffer.  prog_lens/element_size
// are retained for the (non-steady-state) dynamic-batch / seq padding paths.
struct StagingInputBind {
    std::size_t ort_index{};                 // KernelContext input index
    void* staging_data{nullptr};             // staging buffer device ptr (stable until FreeStaging)
    std::size_t arena_offset{};              // byte offset into the pinned host arena (coalesced)
    std::size_t stage_capacity{};            // staging buffer size in bytes (copy clamp)
    std::size_t prog_bytes{};                // full program-shape byte size for this input
    std::size_t row_bytes{};                 // bytes per axis-0 row (prog_bytes / prog_lens[0]); batch-pad math
    std::vector<std::size_t> prog_lens{};    // program-shape lengths (padding math)
    std::size_t element_size{};              // prog_bytes / product(prog_lens) (padding math)
    bool batch_axis{true};                   // graph says axis 0 is the batch (see InputCarriesBatch)
    std::string name{};                      // parameter name (seq-axis lookup, padding path only)
};

// ── Direct-bind ultra-fast binding cache ─────────────────────────────────────
// Precomputed per-parameter binding metadata for the direct-bind path so the hot
// path skips the per-call name lookups, dtype checks, and device-residency probes
// that building the binding from scratch requires.  Mirrors the built-in EP's
// cached_inputs / cached_outputs + a reused program_parameters, but is cached
// per compiled shape (keyed by shape hash) rather than in a single slot, so
// alternating dynamic-batch buckets each keep their binding instead of thrashing.
struct CachedDirectInput {
    std::string name;              // MIGraphX program parameter name
    std::size_t ort_index{};       // KernelContext input index
    migraphx::shape mgx_shape{};   // program-side shape to bind with
};

struct CachedDirectOutput {
    std::string name;                    // MIGraphX "#output_N" parameter name
    std::size_t output_index{};          // ORT output index
    migraphx::shape mgx_shape{};         // program-side shape to bind with
    std::vector<std::int64_t> ort_shape; // ORT output shape (== program shape; no padding)
};

// One shape's direct-bind binding.  The hot path gathers the current ORT pointers
// into `cur_input_ptrs`/`cur_output_ptrs` (reused, no per-call allocation) and
// compares them positionally against the captured graph's pointers.  `params` is
// (re)bound lazily -- only when a capture or eager run is actually needed -- since
// the steady-state replay path uses the captured graph directly and never reads
// `params`.  Used only by the pure direct-bind path (use_direct_hip_graph, i.e.
// hipGraph without coalesce_io); the staging path lends its own pointer-stable
// buffers to ORT instead of binding ORT's pointers into the program.
struct DirectBindCache {
    std::vector<CachedDirectInput> inputs{};
    std::vector<CachedDirectOutput> outputs{};
    migraphx::program_parameters params{};
    // Scratch presence + shape, resolved once when this entry is populated so the
    // hot path allocates/zeros scratch without re-scanning all 200+ program
    // parameter names for a "scratch" entry on every call.
    bool has_scratch{false};
    migraphx::shape scratch_shape{};
    // ORT output indices in `outputs` order; constant per shape, built once.
    std::vector<std::size_t> prog_output_indices{};
    // Current ORT device pointers gathered each call (inputs/outputs order).  Reused
    // across calls so the hot path does no per-call heap allocation.
    std::vector<void*> cur_input_ptrs{};
    std::vector<void*> cur_output_ptrs{};
    // Resolved pointers into the ComputeState maps (captured graph in hip_graph_cache_direct,
    // scratch slot) so steady-state replay does no re-search.  std::unordered_map addresses
    // are stable until the entry is erased: scratch_bufs is never erased mid-session, and the
    // graph map is cleared only by DestroyHipGraphs, which runs with direct_bind_cache.clear()
    // (unbounded-shape path), dropping this dbc.  Null `graph` forces the cold capture path.
    CapturedHipGraph* graph{nullptr};
    ScratchBuffer* scratch_slot{nullptr};
};

// Result of binding staging buffers (and the EP-owned scratch) as program
// parameters for a given compiled shape.  Cached per shape hash in ComputeState:
// staging buffers and scratch are pointer-stable until FreeStaging, so a binding
// built once can be replayed unchanged instead of rebuilt every Compute call.
struct StagingBindResult {
    migraphx::program_parameters params{};
    std::vector<std::size_t> prog_output_indices{};       // ORT output index per bound output
    std::vector<std::string> bound_output_names{};        // staging key per bound output
    std::vector<migraphx::shape> bound_output_shapes{};   // current bucket shape per bound output
    std::vector<std::vector<std::int64_t>> bound_output_ort_shapes{};  // bucket ORT shape (int64) per bound output
    std::vector<std::size_t> bound_output_row_bytes{};    // bytes per axis-0 row per bound output (batch-slice math)
    std::vector<std::size_t> bound_output_bytes{};        // total bucket byte count per bound output (precomputed)
    std::vector<std::size_t> bound_output_capacity{};     // whole staging buffer size per bound output (loan bound)
    std::vector<void*> bound_output_data{};               // staging src ptr per bound output (resolved once)
    std::vector<StagingInputBind> input_copies{};         // flat per-input copy plan (built once)

    // Per-bound-output: set when this call handed ORT the staging buffer itself as the
    // output tensor's storage, so its copy-back is skipped.  Rewritten every call.
    std::vector<char> output_borrowed{};

    // This bucket's packed coalesce arena (null when coalesce_io is off), resolved once
    // at bind so the gather needs no map lookup.  arena_bytes is the bucket's own packed
    // size, which is what the H2D moves.
    void* arena_dev{nullptr};
    std::size_t arena_bytes{};

    // Contiguous slices of the arena, each covering a run of input_copies.  The gather
    // flushes a chunk as soon as it is filled, so the H2D of chunk k overlaps the CPU
    // gather of chunk k+1.  A small arena gets a single chunk: the extra copy launches
    // would cost more than the overlap saves.
    struct GatherChunk {
        std::size_t first_input{};
        std::size_t input_count{};
        std::size_t byte_offset{};
        std::size_t byte_count{};
    };
    std::vector<GatherChunk> gather_chunks{};

    // Resolved pointers into the ComputeState maps (captured graph in hip_graph_cache,
    // scratch slot) so steady-state replay does no re-search.  Same stability invariant as
    // DirectBindCache; the graph map is cleared only by DestroyHipGraphs, which runs with
    // FreeStaging (staging_bind_cache.clear()), dropping this bind.  Null `graph` forces
    // the cold capture path.
    CapturedHipGraph* graph{nullptr};
    ScratchBuffer* scratch_slot{nullptr};
};

// Integer key for the per-shape hot caches: the low 64 bits of the shapes hash.
using ShapeKey = hash::ShapeKey;

struct InputScanEntry {
    std::string name;
    std::size_t ort_index{};
};

// How many fast-path Compute calls may reuse a batch-determined shape profile before one
// full (validating) input scan is forced.  The validation is free -- the full scan
// re-derives the bucket's effective hash anyway and compares it against the memo -- so
// this only trades one rescan per interval for the guarantee that a model whose shapes
// stop tracking the batch is caught instead of silently mis-keyed.
constexpr int kProfileRevalidateCalls = 512;

struct ComputeState {
    std::mutex& mutex;
    int device_id;
    const migraphx::target& t;
    migraphx::onnx_options onnx_options;
    migraphx::program program;
    bool enable_fp16{};
    bool enable_bf16{};
    bool enable_fp8{};
    bool enable_int8{};
    bool int8_calibration_cache_available{};
    bool has_input_shapes{};
    bool dump_subgraphs_{};
    bool exhaustive_tune{};
    std::string mlss_use_specific_ops{};
    const Map<float>& dynamic_ranges;
    Map<size_t> input_name_indices;
    Map<size_t> output_name_indices;
    std::string onnx_string;
    ComputeMode compute_mode{ComputeMode::Balanced};
    fs::path model_path;
    fs::path cache_dir;
    bool disable_compiled_model_caching{};
    bool force_recompile{};
    fs::path external_data_dir;
    std::string mxr_prefix;
    // Ordered read-only problem-cache paths (app override, then DLL-adjacent shipped),
    // JSON-escaped for backend-option delivery.
    std::vector<std::string> problem_cache_paths{};

    // ── Configuration (set at Compile time) ──────────────────────────────────
    bool hip_graph_enable{};
    std::size_t max_dynamic_batch{};
    std::string compile_batches{};

    // Direct-bind (zero-copy) hipGraph: bind ORT tensor pointers straight into
    // the program and capture/replay without staging copies.  Enabled when
    // hipGraph is on and coalesce_io is off (the two are mutually exclusive:
    // coalesce_io deliberately routes inputs through the pinned staging arena).
    // Mirrors the built-in EP's `use_direct_hip_graph = hip_graph && !coalesce`.
    // May be disabled at runtime after repeated pointer drift (see below), after
    // which Compute falls back to the staging hipGraph path.
    bool use_direct_hip_graph{};
    // Consecutive direct-bind re-captures caused by ORT pointer drift.  Once it
    // exceeds kMaxDirectRecaptures the direct path is disabled for the session.
    static constexpr int kMaxDirectRecaptures{3};
    int direct_recapture_count{};

    // Lend the EP-owned staging output buffers to ORT as the output tensors' storage
    // (see hip::ArmOutputAlloc) instead of copying each output into an ORT-allocated
    // tensor.  Works for padded calls too: the staging buffer is sized for the whole
    // compiled bucket while the ORT tensor is reported at the requested batch, so the
    // program keeps writing target_batch rows into a buffer ORT believes is shorter.
    bool borrow_outputs_enable{};

    // ── Direct-bind ultra-fast binding cache (keyed by shape hash) ────────────
    // Populated once per compiled shape; the hot path rebinds ORT pointers into
    // the cached (reused) program_parameters without re-doing name lookups /
    // dtype checks.  Multi-entry so alternating dynamic-batch buckets each keep
    // their binding.  Invalidated per-hash on recompile (mirrors
    // staging_bind_cache) and cleared wholesale on the unbounded-shape path.
    std::unordered_map<ShapeKey, DirectBindCache> direct_bind_cache{};

    // ── Static sequence-length padding (set at Compile time) ──────────────────
    // When static_pad_seq is set, named inputs are padded on their token axis up to
    // static_pad_seq_len before the program runs, and named outputs are sliced back
    // down to the real token length afterwards.  Parsed once from the "name:axis"
    // specs into (parameter name -> token axis) maps.
    bool static_pad_seq{};
    // Fallback target; the pad length normally comes from the attention mask (see Compute).
    std::size_t static_pad_seq_len{};
    bool static_pad_seq_len_from_env{};  // set explicitly via env -> overrides the mask
    Map<int> static_pad_input_axes{};   // input param name -> token axis (inputs keep real names)
    // Outputs are program params named "#output_N", not their ONNX names, so the
    // slice must match on ORT output INDEX, not name.  Resolved from the user's
    // "logits:1" spec via output_name_indices at Compile time.
    std::unordered_map<std::size_t, int> static_pad_output_axes_by_index{};

    // ── Dynamic-batch runtime state ──────────────────────────────────────────
    bool has_dynamic_batch{};
    bool defer_compilation{};
    std::vector<std::size_t> compiled_batch_sizes{};
    // Compiled program variants keyed by shape/batch hash (ShapeKey, so Compute
    // looks one up by the integer key it already computed -- no hex string / find).
    std::unordered_map<ShapeKey, migraphx::program> cached_programs{};

    // ── Coalesced input arena (ORT_MIGRAPHX_COALESCE_IO) ─────────────────────
    // When coalesce_io is set, every input is a slot in its bucket's packed device
    // arena; copying gathers all inputs into a pinned host buffer with the same layout
    // then flushes it with one H2D (chunked, so the DMA overlaps the tail of the
    // gather).  Arenas are per bucket so the transfer is sized to the bucket rather
    // than to max_dynamic_batch.
    bool coalesce_io{};
    bool staging_inputs_coalesced{};
    std::unordered_map<ShapeKey, CoalesceArena> coalesce_arenas{};
    // Shared double-buffered pinned host gather area, grown to the largest arena bound
    // so far.  Calls alternate between the two buffers, and each buffer carries an event
    // recorded after the H2D that drained it, so a gather can never overwrite bytes a
    // previous call's transfer is still reading.  With one inference between reuses the
    // event is already complete, so waiting on it costs nothing in steady state.
    void* in_staging_host[2]{nullptr, nullptr};
    hipEvent_t in_staging_host_done[2]{nullptr, nullptr};
    std::size_t in_staging_host_bytes{};
    unsigned in_staging_host_cur{};
    // Coalesce input residency, stable for a given deployment (a caller such as Triton
    // binds each input to the same memory kind every call).  Determined once and reused
    // instead of rescanning N inputs per inference; reset by FreeStaging.  Pinned host
    // sources DMA straight into the arena (no CPU copy); pageable sources are gathered
    // into the pinned host buffer then flushed with one H2D; a device input disqualifies
    // the coalesced path.
    // Host-vs-device only, matching the built-in EP: all-host inputs are gathered into
    // the pinned staging buffer + one whole-arena H2D; any device input falls back to
    // the per-input staging copy.  (No pinned-vs-pageable split.)
    enum class CoalesceResidency : std::uint8_t {
        kUnknown, kAllHost, kHasDevice };
    CoalesceResidency coalesce_residency{CoalesceResidency::kUnknown};
    // Set when the fused shape-scan already coalesced this call's inputs, so the staging
    // copy is skipped.  Reset at the start of every input scan.
    bool inputs_coalesced_this_call{false};

    // ── hipGraph / staging / scratch runtime state (owned device memory) ──────
    // Staging buffers keyed by MIGraphX program parameter name.
    Map<StagingBuffer> staging_inputs{};
    Map<StagingBuffer> staging_outputs{};
    bool staging_allocated{};
    // Scratch buffers keyed by shape hash.
    std::unordered_map<ShapeKey, ScratchBuffer> scratch_bufs{};
    // Captured graphs keyed by shape hash, split by binding mode so a shape's staging
    // and direct/hybrid captures never evict each other (else a padded and an exact-bucket
    // call for the same bucket thrash the one slot, recapturing on each).
    std::unordered_map<ShapeKey, CapturedHipGraph> hip_graph_cache{};         // staging
    std::unordered_map<ShapeKey, CapturedHipGraph> hip_graph_cache_direct{};  // direct/hybrid
    // Host inputs (e.g. scalar alpha) staged to device before run_async.
    Map<StagingBuffer> cpu_input_upload_bufs{};

    // ── Binding / shape-hash fast-path caches ────────────────────────────────
    // Staging parameter bindings keyed by shape hash (multi-entry, so alternating
    // dynamic-batch buckets each keep their binding).  Invalidated by FreeStaging
    // (staging pointers change) and per-hash on recompile.
    std::unordered_map<ShapeKey, StagingBindResult> staging_bind_cache{};
    // Last call's actual input shapes and their hash, for skipping the shape-compare
    // and rehash loops when the shapes are unchanged from the previous Compute call.
    std::vector<std::int64_t> last_input_shapes{};
    hash::Value last_input_shapes_hash{};
    bool has_last_input_shapes{};
    // Reusable scratch the per-call input-shape scan fills (item 3): on a raw-shape
    // change it is swapped into last_input_shapes (O(1)) so the old buffer becomes next
    // call's scratch -- no per-call heap allocation for the gathered shapes.
    std::vector<std::int64_t> input_shapes_scratch{};
    // Previous call's per-input ranks + batch/seq context, for the effective-hash reuse
    // fast path (item 2): under dynamic batching the raw batch dim changes every call, so
    // the raw-equality `shapes_unchanged` check misses even though the effective (bucketed)
    // shape -- and thus the hash -- is unchanged.  When only the batch value moved (same
    // target bucket, same non-batch dims, seq inactive on both calls) the previous hash is
    // reused without rehashing.  Updated every call so it always tracks the prior call.
    std::vector<std::uint32_t> last_input_ranks{};
    bool last_dyn_active{};
    std::size_t last_dyn_requested_batch{};
    std::size_t last_dyn_target_batch{};
    bool last_seq_active{};
    // Shape key the active `program` was compiled/loaded for, so the shape-changed
    // path matches with one integer compare.  Set on (re)compile or cache swap.
    ShapeKey active_program_shape_key{};
    bool has_active_program_shape_key{};
    // Raw input data pointers gathered during the per-call input-shape scan (by ORT
    // input index), reused by the coalesced input copy.  Rewritten every call.
    std::vector<const void*> cur_input_data{};
    // Per-input dim counts from the same scan (input_name_indices order), letting the
    // shape-changed rehash slice current_input_shapes by offset.  Rewritten every call.
    std::vector<std::uint32_t> cur_input_ranks{};
    // Per-input axis-0 extent (by ORT input index) from the same scan; -1 for a rank-0
    // input or an index the scan skipped.  Lets the batch-pad copy confirm an input is
    // actually batched (actual axis-0 == requested_batch) without a per-call GetShape:
    // the program shape alone is ambiguous when a fixed dim coincidentally equals the
    // target bucket.  Rewritten every call.
    std::vector<std::int64_t> cur_input_axis0{};

    // Flat mirror of input_name_indices (same order), reused by the scan + effective hash.
    std::vector<InputScanEntry> input_scan_order{};
    // Representative input (lowest ORT index, rank > 0) carrying the batch, cached from the
    // last full scan; the steady-state scan reads only its shape to confirm the batch.
    std::size_t batch_repr_index{};
    bool has_batch_repr{};

    // ── Batch-determined shape profile ───────────────────────────────────────
    // When every model input's shape is [axis0, <compile-time-constant dims>], the whole
    // input shape set is a function of the batch alone: reading the representative
    // input's axis-0 determines every other input's shape, so the per-call rescan of all
    // N inputs (an OrtTensorTypeAndShapeInfo allocation each) disappears -- which matters
    // because a Triton-style dynamic batcher changes the batch on nearly every call, and
    // the old fast path only short-circuited an *identical* batch.
    //
    // The profile comes from the graph when its declared shapes prove it, and is
    // otherwise learned at runtime by diffing two full scans taken at different batches.
    // Either way it is self-checking: every full scan re-derives the bucket's effective
    // hash and compares it against the memo below, and a disagreement retires the profile
    // for the session.  `input_axis0_template` is indexed by ORT input index and holds the
    // input's constant axis-0 extent, or -1 when that axis carries the batch.
    bool shapes_batch_determined{};
    bool batch_profile_disabled{};
    std::vector<std::int64_t> input_axis0_template{};
    // Per ORT input index: 1 where the graph declares axis 0 symbolic, i.e. that input
    // really is batched.  Empty when the graph could not tell us, in which case the code
    // falls back to matching the runtime extent against the request -- which misfires on
    // an input whose fixed leading dim happens to equal the batch or the bucket.
    std::vector<char> input_batch_axis{};
    // Calls left before the next forced (validating) full scan; -1 disables revalidation.
    int profile_revalidate_countdown{-1};
    // Effective-shape hash per target bucket, memoized by the full scan so a later switch
    // to that bucket resolves the hash from the representative input alone.
    std::unordered_map<std::size_t, hash::Value> bucket_shape_hashes{};
    // Previous full scan's raw dims / ranks / batch, kept only while learning a profile.
    std::vector<std::int64_t> learn_prev_shapes{};
    std::vector<std::uint32_t> learn_prev_ranks{};
    std::size_t learn_prev_batch{};
    bool learn_have_prev{};

    // Program parameter shapes keyed by shape hash, so each bucket keeps its shapes
    // and the hot path skips the get_parameter_shapes() rebuild. Dropped per-hash on
    // recompile, alongside the binding caches.
    std::unordered_map<ShapeKey, migraphx::program_parameter_shapes> cached_param_shapes{};
};

// Whether input `ort_index`'s axis 0 is the batch axis, per the graph's symbolic dims.
// Falls back to "assume any leading axis could be the batch" when the graph did not
// declare shapes for every input, leaving the runtime extent match to disambiguate.
// Fixed for the life of the session, so it is safe to bake into a cached binding --
// unlike input_axis0_template, which a runtime profile can learn and later revoke.
inline bool InputCarriesBatch(const ComputeState& cs, std::size_t ort_index) {
    return cs.input_batch_axis.empty() ||
        (ort_index < cs.input_batch_axis.size() && cs.input_batch_axis[ort_index] != 0);
}

struct EpContextComputeState {
    std::mutex& mutex;
    int device_id;
    const migraphx::target& t;
    migraphx::program program;
    Map<size_t> input_name_indices;
    Map<size_t> output_name_indices;
};

struct ExecutionProvider : OrtEp, ApiPtrs {
    ExecutionProvider(const ProviderFactory& api_ptrs, std::string_view ep_name,
        Ort::ConstSessionOptions session_options, const Ort::Logger& logger);

    ~ExecutionProvider();

    ComputeState& GetComputeState(const std::string& fused_node_name) {
        const auto it{compute_states_.find(fused_node_name)};
        if (it == compute_states_.end()) {
            throw std::runtime_error{"unknown compute state for the fused node '"
                + fused_node_name + "'"};
        }
        return it->second;
    }
    EpContextComputeState& EpContext_GetComputeState(const std::string& fused_node_name) {
        const auto it{ep_context_compute_states_.find(fused_node_name)};
        if (it == ep_context_compute_states_.end()) {
            throw std::runtime_error{"unknown EPContext compute state for the fused node '"
                + fused_node_name + "'"};
        }
        return it->second;
    }

    void CollectTelemetry(telemetry::BackendData& out) const noexcept;

    // Session-scoped logger (severity resolved once at init). Exposed so the
    // NodeComputeInfo compute path can gate VERBOSE hot-path tracing without an
    // extra per-call C-API logger lookup.
    const Ort::Logger& GetLogger() const noexcept { return logger_; }

private:
    [[nodiscard]] const char* GetName() const noexcept;

    Ort::Status GetCapability(const Ort::ConstGraph& graph,
        OrtEpGraphSupportInfo* graph_support_info) const noexcept;

    Ort::Status Compile(const std::vector<Ort::ConstGraph>& graphs,
        const std::vector<Ort::ConstNode>& fused_nodes,
        gsl::span<OrtNodeComputeInfo*> node_compute_info,
        gsl::span<OrtNode*> ep_context_nodes) noexcept;

    Ort::Status ReleaseNodeComputeInfos(
        gsl::span<OrtNodeComputeInfo*> node_compute_info) noexcept;

    // Ort::Status SetDynamicOptions(const char* const* option_keys, const char* const* option_values, size_t num_options);
    // Ort::Status OnRunStart(const OrtRunOptions* run_options);

    Ort::Status OnRunEnd(const OrtRunOptions* run_options, bool sync_stream) noexcept;
    Ort::Status CreateSyncStreamForDevice(const OrtMemoryDevice* memory_device, OrtSyncStreamImpl** stream);
    // const char* GetCompiledModelCompatibilityInfo(const OrtGraph* graph) const;
    Ort::Status GetKernelRegistry(const OrtKernelRegistry** kernel_registry) const;

    Ort::Status CreateNodeComputeInfoFromGraph(const Ort::ConstGraph& graph, const Ort::ConstNode& fused_node,
        const Map<size_t>& input_name_indices, const Map<size_t>& output_name_indices, const std::string& mxr_prefix,
        OrtNodeComputeInfo*& node_compute_info, OrtNode*& ep_context_node);

    Ort::Status CreateNodeComputeInfoFromCache(const Ort::ConstGraph& graph, const Ort::ConstNode& fused_node,
        const Map<size_t>& input_name_indices, const Map<size_t>& output_name_indices,
        OrtNodeComputeInfo*& node_compute_info);

    // Load-time hipGraph prewarm.  Captures the staging-path hipGraph for every compiled
    // bucket during Compile so the per-bucket warmup+capture cost never lands on a live
    // inference.  Forces the staging path (the only prewarmable one; the direct-bind path
    // bakes in per-request ORT pointers), so the caller only invokes it under coalesce_io, where
    // staging is already the primary path.  Best-effort and no-throw: any failure just
    // leaves the affected buckets to the normal lazy capture path.
    void PrewarmHipGraphs(ComputeState& compute_state) noexcept;
    // Populate problem_cache_paths_ from the app override env var and the DLL-adjacent cache.
    void setup_problem_cache_paths();

    const ProviderFactory& factory_;

    const Ort::Logger logger_;
    ComputeMode compute_mode_{ComputeMode::Balanced};

    migraphx::target t_{"gpu"};

    Map<float> dynamic_ranges_;
    Map<EpContextComputeState> ep_context_compute_states_;
    Map<ComputeState> compute_states_;

    hipDeviceProp_t device_prop_{};

    int device_id_{};
    std::string ep_name_{};
    bool disable_compiled_model_caching_{};
    bool force_recompile_{};
    bool enable_fp16_{};
    bool enable_bf16_{};
    bool enable_fp8_{};
    bool enable_int8_{};
    bool exhaustive_tune_{};
    std::string mlss_use_specific_ops_{};
    // Ordered read-only problem-cache paths (app override, then DLL-adjacent shipped),
    // JSON-escaped for backend-option delivery.
    std::vector<std::string> problem_cache_paths_{};
    std::string model_arch_{};
    bool int8_calibration_cache_available_{};
    bool int8_use_native_calibration_table_{};
    bool dump_subgraphs_{};
    fs::path cache_dir_{};
    std::string int8_calibration_table_name_{};
    fs::path external_data_dir_{};
    std::string compute_capability_{};
    telemetry::BackendData backend_telemetry_{};
    bool context_embed_mode_{};
    bool context_enable_{};
    std::string context_node_name_prefix_{};
    fs::path context_file_path_{};
    fs::path external_initializers_file_name_{};
    bool hip_graph_enable_{};
    std::size_t max_dynamic_batch_{};
    std::string compile_batches_{};
    bool precompile_at_load_{};
    bool coalesce_io_enable_{};
    bool borrow_outputs_enable_{true};
    bool batch_shape_profile_enable_{true};
    bool allow_cross_batch_ops_{};
    bool cpu_control_flow_enable_{};
    bool static_pad_seq_{};
    std::size_t static_pad_seq_len_{};
    bool static_pad_seq_len_from_env_{};  // set explicitly via env -> overrides the mask
    std::string static_pad_inputs_{};
    std::string static_pad_outputs_{};

    // External application-owned compute stream to adopt (see mgx_options.h). Null
    // means the EP creates and owns its own non-blocking stream per device.
    void* user_compute_stream_{};
    bool has_user_compute_stream_{};

    std::mutex mutex_{};
};

struct NodeComputeInfo : OrtNodeComputeInfo {
    explicit NodeComputeInfo(ExecutionProvider& ep)
        : OrtNodeComputeInfo{NegotiatedOrtApiVersion()}, ep_{ep}
    {
        OrtNodeComputeInfo::CreateState = [](OrtNodeComputeInfo* this_,
            OrtNodeComputeContext* compute_context, void** compute_state) noexcept {
            API_CALL_S(NodeComputeInfo, this_, CreateState, compute_context, *compute_state);
        };
        OrtNodeComputeInfo::Compute = [](OrtNodeComputeInfo* this_,
            void* compute_state, OrtKernelContext* kernel_context) noexcept {
            API_CALL_S(NodeComputeInfo, this_, Compute, *static_cast<ComputeState*>(compute_state),
                Ort::KernelContext{kernel_context});
        };
        OrtNodeComputeInfo::ReleaseState = [](OrtNodeComputeInfo* this_, void* compute_state) noexcept {
            API_CALL_V(NodeComputeInfo, this_, ReleaseState, compute_state);
        };
    }

private:
    ExecutionProvider& ep_;

    Ort::Status CreateState(OrtNodeComputeContext* compute_context, void*& compute_state) noexcept;
    Ort::Status Compute(ComputeState& compute_state, const Ort::KernelContext& kernel_context) noexcept;
    void ReleaseState([[maybe_unused]] void* compute_state) noexcept {}
};

struct EpContextNodeComputeInfo : OrtNodeComputeInfo {
    explicit EpContextNodeComputeInfo(ExecutionProvider& ep)
        : OrtNodeComputeInfo{NegotiatedOrtApiVersion()}, ep_{ep}
    {
        OrtNodeComputeInfo::CreateState = [](OrtNodeComputeInfo* this_,
            OrtNodeComputeContext* compute_context, void** compute_state) noexcept {
            API_CALL_S(EpContextNodeComputeInfo, this_, CreateState, compute_context, *compute_state);
        };
        OrtNodeComputeInfo::Compute = [](OrtNodeComputeInfo* this_,
            void* compute_state, OrtKernelContext* kernel_context) noexcept {
            API_CALL_S(EpContextNodeComputeInfo, this_, Compute, *static_cast<EpContextComputeState*>(compute_state),
                Ort::KernelContext{kernel_context});
        };
        OrtNodeComputeInfo::ReleaseState = [](OrtNodeComputeInfo* this_, void* compute_state) noexcept {
            API_CALL_V(EpContextNodeComputeInfo, this_, ReleaseState, compute_state);
        };
    }

private:
    ExecutionProvider& ep_;

    Ort::Status CreateState(OrtNodeComputeContext* compute_context, void*& compute_state) noexcept;
    Ort::Status Compute(EpContextComputeState& compute_state, const Ort::KernelContext& kernel_context) noexcept;
    void ReleaseState([[maybe_unused]] void* compute_state) noexcept {}
};

}  // namespace mgx_ep
