"""
Standalone HIP Graph capture/replay debug script for MIGraphX EP.

Loads an ONNX model via ORT with MIGraphX EP, exercises the warmup -> capture -> replay
sequence with configurable batch sizes, and reports which stage segfaults.
Includes numerical verification: same-input runs through warmup, capture, and replay
must produce matching outputs (within configurable tolerance).

Usage (inside Docker container with ORT + MIGraphX):
    python3 test_hip_graph.py --model /models/feed-gen-rec.model-20251203-113935-d80bb571.2.onnx/0/model.onnx

    # With specific batch sizes:
    python3 test_hip_graph.py --model /path/to/model.onnx --batches 1 4 16 64

    # Without hip graph (baseline sanity check):
    python3 test_hip_graph.py --model /path/to/model.onnx --no-hip-graph

    # With .mxr cache directory:
    python3 test_hip_graph.py --model /path/to/model.onnx --cache-dir /models

    # Interleaved replay across multiple batch sizes (random order):
    python3 test_hip_graph.py --model /path/to/model.onnx --batches 1 4 16 64 --interleaved --rounds 100

    # Limit ORT thread pool to reduce CPU contention on shared machines:
    python3 test_hip_graph.py --model /path/to/model.onnx --threads 16

    # Set a timeout for session creation (compilation):
    python3 test_hip_graph.py --model /path/to/model.onnx --timeout 600

    # Adjust numerical tolerance:
    python3 test_hip_graph.py --model /path/to/model.onnx --rtol 1e-3 --atol 1e-5

    # Skip numerical checks (crash-only testing):
    python3 test_hip_graph.py --model /path/to/model.onnx --no-verify

    # Cross-verify: compare eager vs hipGraph outputs for the same inputs:
    python3 test_hip_graph.py --model /path/to/model.onnx --cross-verify --batches 1 2 4 8

    # Cross-verify + interleaved: same random schedule replayed through both sessions:
    python3 test_hip_graph.py --model /path/to/model.onnx --cross-verify --interleaved \
        --batches 1 2 4 8 --rounds 50 --seed 42
"""

import argparse
import gc
import os
import signal
import sys
import threading
import time
import traceback

import numpy as np


def make_random_input(dtype_str: str, shape: list[int]) -> np.ndarray:
    """Create a random numpy array matching an ORT input's type and shape."""
    type_map = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    np_dtype = type_map.get(dtype_str, np.float32)

    resolved = [max(1, d) if isinstance(d, int) and d > 0 else 1 for d in shape]

    if np.issubdtype(np_dtype, np.integer):
        return np.random.randint(0, 10, size=resolved, dtype=np_dtype)
    if np_dtype == np.bool_:
        return np.random.randint(0, 2, size=resolved).astype(np.bool_)
    return np.random.randn(*resolved).astype(np_dtype)


def _heartbeat(msg: str, stop_event: threading.Event, interval: float = 10.0):
    """Print a periodic heartbeat so long compilations don't look like stalls."""
    t0 = time.perf_counter()
    while not stop_event.wait(interval):
        elapsed = time.perf_counter() - t0
        print(f"  [heartbeat] {msg} ... {elapsed:.0f}s elapsed", flush=True)


def compare_outputs(reference: list[np.ndarray], candidate: list[np.ndarray],
                    ref_label: str, cand_label: str, rtol: float, atol: float) -> bool:
    """Element-wise comparison of two output lists. Returns True if all match."""
    if len(reference) != len(candidate):
        print(f"  [VERIFY FAIL] output count mismatch: "
              f"{ref_label} has {len(reference)}, {cand_label} has {len(candidate)}")
        return False

    all_ok = True
    for i, (ref, cand) in enumerate(zip(reference, candidate)):
        if ref.shape != cand.shape:
            print(f"  [VERIFY FAIL] output[{i}] shape mismatch: "
                  f"{ref_label}={ref.shape} vs {cand_label}={cand.shape}")
            all_ok = False
            continue

        if ref.dtype == np.bool_ or np.issubdtype(ref.dtype, np.integer):
            if np.array_equal(ref, cand):
                print(f"  [VERIFY  OK ] output[{i}] ({ref.dtype}): "
                      f"exact match ({ref_label} vs {cand_label})")
            else:
                mismatches = int(np.sum(ref != cand))
                total = ref.size
                print(f"  [VERIFY FAIL] output[{i}] ({ref.dtype}): "
                      f"{mismatches}/{total} elements differ ({ref_label} vs {cand_label})")
                all_ok = False
        else:
            if np.allclose(ref, cand, rtol=rtol, atol=atol, equal_nan=True):
                max_abs = float(np.max(np.abs(ref - cand))) if ref.size > 0 else 0.0
                print(f"  [VERIFY  OK ] output[{i}] ({ref.dtype}): "
                      f"allclose (max_abs_diff={max_abs:.2e}) ({ref_label} vs {cand_label})")
            else:
                diff = np.abs(ref - cand)
                max_abs = float(np.max(diff)) if ref.size > 0 else 0.0
                mean_abs = float(np.mean(diff)) if ref.size > 0 else 0.0
                num_bad = int(np.sum(~np.isclose(ref, cand, rtol=rtol, atol=atol,
                                                 equal_nan=True)))
                total = ref.size
                print(f"  [VERIFY FAIL] output[{i}] ({ref.dtype}): "
                      f"{num_bad}/{total} elements exceed tol (max_abs={max_abs:.2e}"
                      f", mean_abs={mean_abs:.2e}, rtol={rtol}, atol={atol})"
                      f" ({ref_label} vs {cand_label})")
                all_ok = False

    return all_ok


def _pct(arr, q):
    """Return percentile value, or 0.0 for empty arrays."""
    return float(np.percentile(arr, q)) if len(arr) > 0 else 0.0


def print_latency_comparison(eager_times: dict[int, list[float]],
                             graph_times: dict[int, list[float]]):
    """Print a summary table comparing eager vs hipGraph latencies per batch size."""
    all_bs = sorted(set(eager_times) | set(graph_times))
    if not all_bs:
        return

    sep = "=" * 130
    dash = "-" * 130
    print(f"\n  {sep}")
    print("  LATENCY COMPARISON (ms)  \u2014  excludes warmup iterations")
    print(f"  {sep}")
    print(f"  {'BS':>5s}  {'N':>4s}  "
          f"{'E avg':>7s} {'E p50':>7s} {'E p90':>7s} {'E p95':>7s} {'E p99':>7s}  "
          f"{'G avg':>7s} {'G p50':>7s} {'G p90':>7s} {'G p95':>7s} {'G p99':>7s}  "
          f"{'Speedup':>8s}")
    print(f"  {'-----':>5s}  {'----':>4s}  "
          f"{'-------':>7s} {'-------':>7s} {'-------':>7s} {'-------':>7s} {'-------':>7s}  "
          f"{'-------':>7s} {'-------':>7s} {'-------':>7s} {'-------':>7s} {'-------':>7s}  "
          f"{'--------':>8s}")

    total_eager, total_graph, total_n = 0.0, 0.0, 0
    for bs in all_bs:
        e = eager_times.get(bs, [])
        g = graph_times.get(bs, [])
        n = min(len(e), len(g))
        if n == 0:
            continue
        e_arr = np.array(e[:n])
        g_arr = np.array(g[:n])
        e_avg = float(np.mean(e_arr))
        g_avg = float(np.mean(g_arr))
        speedup = e_avg / g_avg if g_avg > 0 else float('inf')
        total_eager += e_arr.sum()
        total_graph += g_arr.sum()
        total_n += n
        print(f"  {bs:>5d}  {n:>4d}  "
              f"{e_avg:>7.2f} {_pct(e_arr, 50):>7.2f} {_pct(e_arr, 90):>7.2f} "
              f"{_pct(e_arr, 95):>7.2f} {_pct(e_arr, 99):>7.2f}  "
              f"{g_avg:>7.2f} {_pct(g_arr, 50):>7.2f} {_pct(g_arr, 90):>7.2f} "
              f"{_pct(g_arr, 95):>7.2f} {_pct(g_arr, 99):>7.2f}  "
              f"{speedup:>7.2f}x")

    if total_n > 0:
        overall_e = total_eager / total_n
        overall_g = total_graph / total_n
        overall_speedup = overall_e / overall_g if overall_g > 0 else float('inf')
        print(f"  {'-----':>5s}  {'----':>4s}  "
              f"{'-------':>7s} {'-------':>7s} {'-------':>7s} {'-------':>7s} {'-------':>7s}  "
              f"{'-------':>7s} {'-------':>7s} {'-------':>7s} {'-------':>7s} {'-------':>7s}  "
              f"{'--------':>8s}")
        print(f"  {'ALL':>5s}  {total_n:>4d}  "
              f"{overall_e:>7.2f} {'':>7s} {'':>7s} {'':>7s} {'':>7s}  "
              f"{overall_g:>7.2f} {'':>7s} {'':>7s} {'':>7s} {'':>7s}  "
              f"{overall_speedup:>7.2f}x")
    print(f"  {sep}")


def create_session(model_path, hip_graph, cache_dir, compile_batches,
                   intra_threads, inter_threads):
    import onnxruntime as ort

    provider_options = {
        "device_id": "0",
    }
    if hip_graph:
        provider_options["migraphx_hip_graph_enable"] = "1"
    if cache_dir:
        provider_options["migraphx_model_cache_dir"] = cache_dir
    if compile_batches:
        provider_options["migraphx_compile_batches"] = compile_batches

    print(f"[config] Provider options: {provider_options}")

    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 0

    if intra_threads > 0:
        sess_opts.intra_op_num_threads = intra_threads
    if inter_threads > 0:
        sess_opts.inter_op_num_threads = inter_threads

    print(f"[config] intra_op_threads={intra_threads or 'auto'}"
          f", inter_op_threads={inter_threads or 'auto'}")

    sess = ort.InferenceSession(
        model_path,
        sess_options=sess_opts,
        providers=[("MIGraphXExecutionProvider", provider_options)],
    )

    return sess


def create_session_with_timeout(model_path, hip_graph, cache_dir, compile_batches,
                                intra_threads, inter_threads, timeout_s):
    """Create an ORT session with a heartbeat and optional wall-clock timeout."""
    stop_hb = threading.Event()
    hb = threading.Thread(
        target=_heartbeat,
        args=("Session creation (compiling models)", stop_hb, 15.0),
        daemon=True,
    )

    hb.start()

    if timeout_s <= 0:
        try:
            sess = create_session(model_path, hip_graph, cache_dir,
                                  compile_batches, intra_threads, inter_threads)
        finally:
            stop_hb.set()
            hb.join(timeout=2)
        return sess

    result = [None]
    error = [None]

    def _create():
        try:
            result[0] = create_session(model_path, hip_graph, cache_dir,
                                       compile_batches, intra_threads,
                                       inter_threads)
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_create, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    stop_hb.set()
    hb.join(timeout=2)

    if t.is_alive():
        print(f"\nERROR: Session creation timed out after {timeout_s}s")
        print("Likely cause: too many compile-batches for available VRAM/time.")
        print("Try fewer --compile-batches or split into multiple runs.")
        sys.exit(1)
    if error[0]:
        raise error[0]
    return result[0]


def build_feed(sess, batch_size):
    """Build a random input feed dict for the given batch size."""
    feed = {}
    for inp in sess.get_inputs():
        shape = list(inp.shape)
        if shape and (shape[0] is None or isinstance(shape[0], str) or shape[0] == -1):
            shape[0] = batch_size
        feed[inp.name] = make_random_input(inp.type, shape)
    return feed


def run_inference(sess, feed, label):
    """Run a single inference and print timing + output summary.

    Returns (outputs, elapsed_ms) on success, or (None, 0.0) on failure.
    """
    t0 = time.perf_counter()
    try:
        outputs = sess.run(None, feed)
    except Exception as e:
        print(f"  [{label}] EXCEPTION: {e}")
        traceback.print_exc()
        return (None, 0.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    out_summary = ", ".join(f"{o.shape}" for o in outputs)
    print(f"  [{label}] OK  {elapsed_ms:8.2f} ms  outputs: [{out_summary}]")
    return (outputs, elapsed_ms)


def test_batch(sess, batch_size, num_replays, hip_graph, verify=True,
               rtol=1e-3, atol=1e-5):
    """Run the full warmup -> capture -> replay cycle for one batch size.

    Uses the SAME input feed for warmup, capture, and the first replay so their
    outputs can be compared numerically. Subsequent replays use fresh random data
    to stress-test different value ranges.
    """
    phase = "hip_graph" if hip_graph else "eager"
    print(f"\n{'=' * 70}")
    print(f"  Batch size: {batch_size}  |  mode: {phase}  |  replays: {num_replays}")
    print(f"  verify: {verify}  rtol: {rtol}  atol: {atol}")
    print(f"{'=' * 70}")

    feed = build_feed(sess, batch_size)
    input_summary = {k: v.shape for k, v in feed.items()}
    print(f"  Inputs: {len(input_summary)} tensors, batch_dim={batch_size}")

    # --- Run 1: warmup ---
    print("\n  --- Run 1: warmup (triggers compilation if needed) ---")
    r1, _ = run_inference(sess, feed, "warmup")
    if r1 is None:
        print("  FAILED at warmup, aborting batch")
        return False

    # --- Run 2: capture ---
    print("\n  --- Run 2: capture (hipGraph stream capture, same input) ---")
    r2, _ = run_inference(sess, feed, "capture")
    if r2 is None:
        print("  FAILED at capture, aborting batch")
        return False

    verify_pass = 0
    verify_fail = 0

    if verify:
        print("\n  --- Verify: warmup vs capture (same input) ---")
        if compare_outputs(r1, r2, "warmup", "capture", rtol, atol):
            verify_pass += 1
        else:
            verify_fail += 1
            print("  NUMERICAL MISMATCH between warmup and capture")

    # --- Run 3: replay-1 (same input) ---
    print("\n  --- Run 3: replay-1 (hipGraph replay, same input) ---")
    r3, _ = run_inference(sess, feed, "replay-1-same")
    if r3 is None:
        print("  FAILED at replay-1-same, aborting batch")
        return False

    if verify:
        print("\n  --- Verify: warmup vs replay-1 (same input) ---")
        if compare_outputs(r1, r3, "warmup", "replay-1-same", rtol, atol):
            verify_pass += 1
        else:
            verify_fail += 1
            print("  NUMERICAL MISMATCH between warmup and replay-1")

    # --- Remaining replays with new random data ---
    remaining = max(0, num_replays - 1)
    for i in range(remaining):
        new_feed = build_feed(sess, batch_size)
        label = f"replay-{i + 2}"
        print(f"\n  --- Run {4 + i}: replay ({label}, new random input) ---")

        r_ref, _ = run_inference(sess, new_feed, f"{label}-ref")
        if r_ref is None:
            print(f"  FAILED at {label}-ref, aborting batch")
            return False

        r_rep, _ = run_inference(sess, new_feed, f"{label}-rep")
        if r_rep is None:
            print(f"  FAILED at {label}-rep, aborting batch")
            return False

        if verify:
            if compare_outputs(r_ref, r_rep, f"{label}-ref", f"{label}-rep", rtol, atol):
                verify_pass += 1
            else:
                verify_fail += 1
                print(f"  NUMERICAL MISMATCH at {label}")

    total_checks = verify_pass + verify_fail
    if verify and total_checks > 0:
        pct = 100.0 * verify_pass / total_checks
        print(f"\n  Verification: {verify_pass}/{total_checks} PASSED ({pct:.1f}%), "
              f"{verify_fail}/{total_checks} FAILED ({100.0 - pct:.1f}%)")

    all_passed = verify_fail == 0
    print(f"\n  {'All runs passed' if all_passed else 'Some checks FAILED'}"
          f" for batch_size={batch_size}")
    return all_passed


def test_interleaved(sess, batch_sizes, num_rounds, hip_graph, verify=True,
                     rtol=1e-3, atol=1e-5):
    """Prime hipGraphs for all batch sizes, then replay them in random order."""
    phase = "hip_graph" if hip_graph else "eager"
    print(f"\n{'=' * 70}")
    print(f"  INTERLEAVED TEST  |  batches: {batch_sizes}  |  mode: {phase}")
    print(f"  rounds: {num_rounds}  verify: {verify}  rtol: {rtol}  atol: {atol}")
    print(f"{'=' * 70}")

    if len(batch_sizes) < 2:
        print("  Need at least 2 batch sizes for interleaved test")
        return False

    verify_pass = 0
    verify_fail = 0

    # Phase 1: Prime all batch variants
    print(f"\n  --- Phase 1: Priming {len(batch_sizes)} batch variants ---")
    for bs in batch_sizes:
        feed = build_feed(sess, bs)
        r1, _ = run_inference(sess, feed, f"prime-warmup-bs{bs}")
        if r1 is None:
            print(f"  FAILED priming warmup for bs={bs}")
            return False
        r2, _ = run_inference(sess, feed, f"prime-capture-bs{bs}")
        if r2 is None:
            print(f"  FAILED priming capture for bs={bs}")
            return False
        if verify:
            print(f"  --- Verify: warmup vs capture for bs={bs} ---")
            if compare_outputs(r1, r2, f"warmup-bs{bs}", f"capture-bs{bs}", rtol, atol):
                verify_pass += 1
            else:
                verify_fail += 1
                print(f"  NUMERICAL MISMATCH during priming for bs={bs}")
    print(f"  All {len(batch_sizes)} graphs primed"
          f"{' (numerically verified)' if verify else ''}")

    # Phase 2: Interleaved replays
    print(f"\n  --- Phase 2: {num_rounds} interleaved replays ---")
    rng = np.random.default_rng()
    prev_bs = None
    switch_count = 0

    for i in range(num_rounds):
        bs = int(rng.choice(batch_sizes))
        if prev_bs is not None and bs != prev_bs:
            switch_count += 1
        prev_bs = bs

        feed = build_feed(sess, bs)
        r_a, _ = run_inference(sess, feed, f"interleave-{i}-bs{bs}-a")
        if r_a is None:
            print(f"  FAILED at round {i} (bs={bs})")
            return False

        if verify:
            r_b, _ = run_inference(sess, feed, f"interleave-{i}-bs{bs}-b")
            if r_b is None:
                print(f"  FAILED at round {i} (bs={bs})")
                return False
            if compare_outputs(r_a, r_b, f"round-{i}-bs{bs}-a", f"round-{i}-bs{bs}-b",
                               rtol, atol):
                verify_pass += 1
            else:
                verify_fail += 1
                print(f"  NUMERICAL MISMATCH at round {i} verify (bs={bs})")

    total_checks = verify_pass + verify_fail
    if verify and total_checks > 0:
        pct = 100.0 * verify_pass / total_checks
        print(f"\n  Verification: {verify_pass}/{total_checks} PASSED ({pct:.1f}%), "
              f"{verify_fail}/{total_checks} FAILED ({100.0 - pct:.1f}%)")

    all_passed = verify_fail == 0
    print(f"\n  {'All' if all_passed else 'NOT all'} "
          f"{num_rounds} interleaved replays passed ({switch_count} graph switches)")
    return all_passed


def _create_and_warmup_session(model_path, hip_graph, batch_sizes, cache_dir,
                               compile_batches, intra_threads, timeout_s,
                               extra_warmup=4):
    """Helper: create an ORT session and warm up all batch sizes.

    For hipGraph sessions: runs warmup + capture + extra_warmup throwaway
    iterations per batch to stabilize MIGraphX workspace state before any
    comparisons begin.
    Returns the session, or None on failure.
    """
    mode = "HIPGRAPH" if hip_graph else "EAGER"
    print(f"\n  --- Creating {mode} session ---")
    t0 = time.perf_counter()
    sess = create_session_with_timeout(
        model_path, hip_graph, cache_dir, compile_batches,
        intra_threads, 1, timeout_s)
    print(f"  {mode} session created in {(time.perf_counter() - t0) * 1000:.0f} ms")

    print(f"\n  --- Warming up {mode} session ({len(batch_sizes)} batch sizes) ---")
    for bs in batch_sizes:
        feed = build_feed(sess, bs)
        r1, _ = run_inference(sess, feed, f"{mode}-warmup-bs{bs}")
        if r1 is None:
            print(f"  FAILED {mode} warmup for bs={bs}")
            return None
        if hip_graph:
            r2, _ = run_inference(sess, feed, f"{mode}-capture-bs{bs}")
            if r2 is None:
                print(f"  FAILED {mode} capture for bs={bs}")
                return None
            for w in range(extra_warmup):
                rw, _ = run_inference(sess, feed, f"{mode}-warmin-bs{bs}-{w}")
                if rw is None:
                    print(f"  FAILED {mode} warm-in {w} for bs={bs}")
                    return None
    return sess


def test_cross_session(model_path, batch_sizes, num_runs, cache_dir, compile_batches,
                       intra_threads, timeout_s, rtol, atol):
    """Compare eager (no hipGraph) vs hipGraph session outputs for the same inputs.

    Strategy to reduce peak VRAM:
      1. Create eager session, collect reference outputs for all batch sizes, destroy it.
      2. Create hipGraph session, collect its outputs for the same inputs, compare.
    """
    print(f"\n{'#' * 70}")
    print("  CROSS-SESSION VERIFICATION: eager vs hipGraph")
    print(f"{'#' * 70}")
    print(f"  batches: {batch_sizes}")
    print(f"  runs_per_batch: {num_runs}")
    print(f"  rtol: {rtol}  atol: {atol}")

    # Phase 1: Eager session
    sess_eager = _create_and_warmup_session(
        model_path, False, batch_sizes, cache_dir, compile_batches,
        intra_threads, timeout_s)
    if sess_eager is None:
        return False

    eager_times = {bs: [] for bs in batch_sizes}
    graph_times = {bs: [] for bs in batch_sizes}
    reference = {}
    for bs in batch_sizes:
        reference[bs] = []
        for j in range(num_runs):
            feed = build_feed(sess_eager, bs)
            r, ms = run_inference(sess_eager, feed, f"eager-bs{bs}-run{j}")
            if r is None:
                print(f"  FAILED eager run {j} for bs={bs}")
                return False
            reference[bs].append((feed, r))
            eager_times[bs].append(ms)

    print("\n  --- Releasing eager session to free VRAM ---")
    del sess_eager
    gc.collect()

    # Phase 2: hipGraph session
    sess_graph = _create_and_warmup_session(
        model_path, True, batch_sizes, cache_dir, compile_batches,
        intra_threads, timeout_s)
    if sess_graph is None:
        return False

    # Phase 3: Cross-compare (skip run0 per batch — workspace warm-in transient)
    print("\n  --- Cross-comparing outputs (skipping run0 as warm-in) ---")
    verify_pass = 0
    verify_fail = 0

    for bs in batch_sizes:
        eager_outputs = reference[bs]
        for j, (feed, _) in enumerate(eager_outputs):
            label = f"cross-bs{bs}-run{j}"
            graph_outputs, ms = run_inference(sess_graph, feed, f"graph-{label}")
            if graph_outputs is None:
                print(f"  FAILED hipGraph inference at {label}")
                if j > 0:
                    verify_fail += 1
                continue
            if j == 0:
                print(f"  --- Skipping run0 for bs={bs} (warm-in) ---")
                continue
            graph_times[bs].append(ms)
            print(f"  --- Verify: eager vs hipGraph ({label}) ---")
            if compare_outputs(eager_outputs[j][1], graph_outputs,
                               f"eager-{label}", f"graph-{label}", rtol, atol):
                verify_pass += 1
            else:
                verify_fail += 1
                print(f"  CROSS-SESSION MISMATCH at {label}")

    total_checks = verify_pass + verify_fail
    if total_checks > 0:
        pct = 100.0 * verify_pass / total_checks
        print(f"\n  Verification: {verify_pass}/{total_checks} PASSED ({pct:.1f}%), "
              f"{verify_fail}/{total_checks} FAILED ({100.0 - pct:.1f}%)")

    all_passed = verify_fail == 0
    if all_passed:
        print(f"  Cross-session verification PASSED "
              f"({total_checks} comparisons across {len(batch_sizes)} batch sizes)")
    else:
        print("  Cross-session verification FAILED")

    print_latency_comparison(eager_times, graph_times)
    return all_passed


def test_cross_interleaved(model_path, batch_sizes, num_rounds, seed, cache_dir,
                           compile_batches, intra_threads, timeout_s, rtol, atol):
    """Cross-verify with interleaved batch scheduling.

    Uses a deterministic RNG seed so both sessions see identical batch-size
    sequences and identical input data.

    1. Pre-generate the full interleaved schedule: (batch_size, feed) pairs.
    2. Create eager session, run the schedule, store outputs, destroy session.
    3. Create hipGraph session, replay the same schedule, compare pairwise.
    """
    if len(batch_sizes) < 2:
        print("  Need at least 2 batch sizes for interleaved cross-verify")
        return False

    print(f"\n{'#' * 70}")
    print("  CROSS-SESSION INTERLEAVED VERIFICATION: eager vs hipGraph")
    print(f"{'#' * 70}")
    print(f"  batches: {batch_sizes}")
    print(f"  rounds: {num_rounds}  seed: {seed}")
    print(f"  rtol: {rtol}  atol: {atol}")

    rng = np.random.default_rng(seed)
    schedule = [int(rng.choice(batch_sizes)) for _ in range(num_rounds)]
    switch_count = sum(1 for i in range(1, len(schedule)) if schedule[i] != schedule[i - 1])
    print(f"\n  Schedule: {num_rounds} rounds, {switch_count} graph switches")

    # Phase 1: Eager session
    sess_eager = _create_and_warmup_session(
        model_path, False, batch_sizes, cache_dir, compile_batches,
        intra_threads, timeout_s)
    if sess_eager is None:
        return False

    eager_times = {bs: [] for bs in batch_sizes}
    graph_times = {bs: [] for bs in batch_sizes}
    eager_results = []

    print("\n  --- Running schedule through EAGER session ---")
    for i, bs in enumerate(schedule):
        feed = build_feed(sess_eager, bs)
        r, ms = run_inference(sess_eager, feed, f"eager-round{i}-bs{bs}")
        if r is None:
            print(f"  FAILED eager round {i} (bs={bs})")
            return False
        eager_results.append((bs, feed, r))
        eager_times[bs].append(ms)

    print("\n  --- Releasing eager session to free VRAM ---")
    del sess_eager
    gc.collect()

    # Phase 2: hipGraph session
    sess_graph = _create_and_warmup_session(
        model_path, True, batch_sizes, cache_dir, compile_batches,
        intra_threads, timeout_s)
    if sess_graph is None:
        return False

    verify_pass = 0
    verify_fail = 0
    warmed_bs = set()

    print("\n  --- Replaying schedule through HIPGRAPH session & comparing ---")
    for i, (bs, feed, eager_out) in enumerate(eager_results):
        label = f"round{i}-bs{bs}"
        graph_out, ms = run_inference(sess_graph, feed, f"graph-{label}")
        if graph_out is None:
            print(f"  FAILED hipGraph round {i} (bs={bs})")
            if bs in warmed_bs:
                verify_fail += 1
            continue
        if bs not in warmed_bs:
            print(f"  --- Skipping first replay for bs={bs} (warm-in) ---")
            warmed_bs.add(bs)
            continue
        graph_times[bs].append(ms)
        if compare_outputs(eager_out, graph_out, f"eager-{label}", f"graph-{label}",
                           rtol, atol):
            verify_pass += 1
        else:
            verify_fail += 1
            print(f"  CROSS-INTERLEAVED MISMATCH at {label}")

    total_checks = verify_pass + verify_fail
    if total_checks > 0:
        pct = 100.0 * verify_pass / total_checks
        print(f"\n  Verification: {verify_pass}/{total_checks} PASSED ({pct:.1f}%), "
              f"{verify_fail}/{total_checks} FAILED ({100.0 - pct:.1f}%)")

    all_passed = verify_fail == 0
    if all_passed:
        print(f"  Cross-interleaved verification PASSED ({switch_count} graph switches)")
    else:
        print("  Cross-interleaved verification FAILED")

    print_latency_comparison(eager_times, graph_times)
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="HIP Graph capture/replay debug harness")
    parser.add_argument("--model", required=True, help="Path to ONNX model file")
    parser.add_argument("--batches", nargs="+", type=int, default=[1],
                        help="Batch sizes to test (default: 1)")
    parser.add_argument("--replays", type=int, default=5,
                        help="Number of replay runs per batch (default: 5)")
    parser.add_argument("--no-hip-graph", action="store_true",
                        help="Disable hip graph (baseline run)")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="MIGraphX .mxr cache directory")
    parser.add_argument("--compile-batches", type=str, default=None,
                        help="Comma-separated compile batch sizes (e.g. '1,2,4,8,16,32,64,128')")
    parser.add_argument("--interleaved", action="store_true",
                        help="Run interleaved replay test: prime all batch sizes then replay "
                             "them in random order (requires >= 2 batch sizes)")
    parser.add_argument("--rounds", type=int, default=50,
                        help="Number of rounds for interleaved test (default: 50)")
    parser.add_argument("--threads", type=int, default=16,
                        help="ORT intra_op thread pool size (default: 16, 0=auto)")
    parser.add_argument("--timeout", type=int, default=0,
                        help="Session creation timeout in seconds (default: 0=no timeout)")
    parser.add_argument("--rtol", type=float, default=1e-3,
                        help="Relative tolerance for numerical verification (default: 1e-3)")
    parser.add_argument("--atol", type=float, default=1e-5,
                        help="Absolute tolerance for numerical verification (default: 1e-5)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip numerical verification (crash-only testing)")
    parser.add_argument("--cross-verify", action="store_true",
                        help="Create separate eager and hipGraph sessions and compare outputs "
                             "for the same inputs (doubles compilation time). Combine with "
                             "--interleaved for cross-verified interleaving.")
    parser.add_argument("--cross-runs", type=int, default=3,
                        help="Number of runs per batch size during cross-verify (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for deterministic scheduling in cross-interleaved "
                             "mode (default: 42)")

    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)

    hip_graph = not args.no_hip_graph
    verify = not args.no_verify

    print(f"[init] Model:       {args.model}")
    print(f"[init] HIP Graph:   {'ENABLED' if hip_graph else 'DISABLED'}")
    print(f"[init] Batches:     {args.batches}")

    if args.cross_verify and args.interleaved:
        print(f"[init] Mode:        CROSS-VERIFY + INTERLEAVED ({args.rounds} rounds, "
              f"seed={args.seed})")
    elif args.cross_verify:
        print(f"[init] Mode:        CROSS-VERIFY (eager vs hipGraph, "
              f"{args.cross_runs} runs/batch)")
    elif args.interleaved:
        print(f"[init] Mode:        INTERLEAVED ({args.rounds} rounds)")
    else:
        print(f"[init] Replays/bat: {args.replays}")

    print(f"[init] Cache dir:   {args.cache_dir}")
    print(f"[init] Compile bat: {args.compile_batches}")
    print(f"[init] Threads:     {args.threads or 'auto'}")
    print(f"[init] Verify:      {'ON' if verify else 'OFF'}"
          f"{f' (rtol={args.rtol}, atol={args.atol})' if verify else ''}")
    print(f"[init] Timeout:     {args.timeout}s")

    def segfault_handler(signum, frame):
        print(f"\n{'!' * 70}")
        print(f"  SEGFAULT (signal {signum}) caught!")
        print("  Python stack at crash point:")
        traceback.print_stack(frame)
        print(f"{'!' * 70}")
        sys.exit(139)

    signal.signal(signal.SIGSEGV, segfault_handler)

    # Cross-verify + interleaved mode
    if args.cross_verify and args.interleaved:
        ok = test_cross_interleaved(
            args.model, args.batches, args.rounds, args.seed,
            args.cache_dir, args.compile_batches, args.threads, args.timeout,
            args.rtol, args.atol)
        if ok:
            print("\n  Cross-interleaved verify PASSED")
        else:
            print("\n  Cross-interleaved verify FAILED")
            sys.exit(1)
        return

    # Cross-verify mode (non-interleaved)
    if args.cross_verify:
        ok = test_cross_session(
            args.model, args.batches, args.cross_runs,
            args.cache_dir, args.compile_batches, args.threads, args.timeout,
            args.rtol, args.atol)
        if ok:
            print("\n  Cross-verify PASSED")
        else:
            print("\n  Cross-verify FAILED")
            sys.exit(1)
        return

    # Standard / interleaved mode: single session
    print("[init] Creating ORT session with MIGraphX EP...")
    t0 = time.perf_counter()
    sess = create_session_with_timeout(
        args.model, hip_graph, args.cache_dir, args.compile_batches,
        args.threads, 1, args.timeout)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"[init] Session created in {load_ms:.0f} ms")

    print(f"\n[init] Model inputs:  {len(sess.get_inputs())}")
    print(f"[init] Model outputs: {len(sess.get_outputs())}")

    if args.interleaved:
        ok = test_interleaved(sess, args.batches, args.rounds, hip_graph,
                              verify=verify, rtol=args.rtol, atol=args.atol)
        if ok:
            print("\n  Interleaved test PASSED")
        else:
            print("\n  Interleaved test FAILED")
            sys.exit(1)
        return

    results = {}
    for batch in args.batches:
        ok = test_batch(sess, batch, args.replays, hip_graph,
                        verify=verify, rtol=args.rtol, atol=args.atol)
        results[batch] = ok

    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    all_ok = True
    for batch, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  batch={batch:>4d}  {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n  All tests PASSED")
    else:
        print("\n  Some tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
