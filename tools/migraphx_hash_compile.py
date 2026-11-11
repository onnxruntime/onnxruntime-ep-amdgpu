"""
Compile an ONNX model into a MIGraphX EP-named .mxr cache file.

Rather than re-implementing the MIGraphX EP's filename hashing, this script lets
the EP itself do the compile + naming. It loads the MIGraphX plugin EP into ONNX
Runtime, points the EP's cache_dir at an output directory, and lets the EP
compile and write the hashed .mxr. Any .mxr produced this way is byte- and
name-identical to what a deployment session expects, so it is picked up
automatically when the same model is later run with the same GPU arch, MIGraphX
version, OS, input shapes, and hash-affecting EP options. The cache directory
itself is not part of the name, so the files can be copied anywhere.

Filename layout (src/migraphx/mgx_ep.cc):
    <version>-<graph_id>-<arch>-<options>-<mlss_ops>-<shapes>.mxr

    version   MIGraphX version (reversible; decoded in the manifest)
    graph_id  model FILE NAME (not its directory), graph input names/types,
              every node's input names/shapes/types, and the host OS
    arch      gcnArchName of the GPU (e.g. gfx942:sramecc+:xnack-)
    options   fp16/bf16/int8/fp8, exhaustive_tune, compute_mode, and the
              int8/fp8 calibration table name + dynamic ranges when active
    mlss_ops  the effective AMDMLSS op list (explicit option, else the
              per-arch heuristic, minus graphs on the exclusion list)
    shapes    the concrete input shapes the program was compiled for

All but the version are one-way MurmurHash3 values. The EP logs the plain-text
inputs of the graph/arch/options/mlss_ops components as "[mgx-cache]" lines at
INFO level. This script captures those lines and records them in a manifest
next to each .mxr, so the manifest shows exactly what the EP hashed instead of
a Python re-derivation that can drift from the C++.

For each generated cache file the script writes (under <output-dir>/<quant>/):
    <ep_named>.mxr                   (produced by the MIGraphX EP)
    <ep_named>_<md5>_manifest.log    (build info; <md5> is the .mxr's md5sum)

The per-quantization subfolder is only for organization. Precision is part of
the filename hash, so files from different subfolders can be merged into one
deployment cache directory.

Usage (inside a container / env with ORT plugin-EP support + MIGraphX):
    python3 migraphx_hash_compile.py --ep-library /path/to/libmigraphx-ep.so \
        --model /path/to/model.onnx

    # Batch size, quantization, and compute mode:
    python3 migraphx_hash_compile.py --ep-library ... --model model.onnx \
        --batch 8 --quantize fp16 --compute-mode maximum

    # INT8 with a calibration table (staged into the cache dir; see --help):
    python3 migraphx_hash_compile.py --ep-library ... --model model.onnx \
        --quantize fp16 int8 --int8-calib-table calib.flatbuffer

    # Exhaustive tuning (slower compile, faster kernels):
    python3 migraphx_hash_compile.py --ep-library ... --model model.onnx \
        --quantize fp16 --exhaustive-tune

    # Any other EP option the deployment also sets (repeatable):
    python3 migraphx_hash_compile.py --ep-library ... --model model.onnx \
        --ep-option mlss_use_specific_ops=conv

    # Compile, then confirm a fresh deployment session loads the result instead
    # of recompiling. Deployment options default to the compile options; adjust
    # them to match what the deployment actually passes:
    python3 migraphx_hash_compile.py --ep-library ... --model model.onnx \
        --quantize fp16 --verify --deploy-drop-option compile_batches

--verify runs the deployment session in a separate process against a scratch
cache directory that links to the compiled files, so a mismatch never writes
into the output directory. It passes only when the EP looked up cache keys that
name the compiled files and wrote nothing new; on failure it reports which
filename component (options, mlss_ops, shapes, ...) differed. The script exits
with status 2 when verification fails.
"""

import argparse
import contextlib
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

QUANT_CHOICES = ["fp16", "bf16", "int8", "fp8"]
COMPUTE_MODE_CHOICES = ["eager", "balanced", "maximum"]

# Plugin EP provider-option keys (see src/migraphx/mgx_options.h). ORT prefixes
# them with "ep.<lowercase ep name>." when they are passed through
# add_provider_for_devices.
OPT_DEVICE_ID = "device_id"
OPT_CACHE_DIR = "cache_dir"
OPT_FP16 = "fp16_enable"
OPT_BF16 = "bf16_enable"
OPT_FP8 = "fp8_enable"
OPT_INT8 = "int8_enable"
OPT_INT8_CALIB = "int8_calibration_table_name"
OPT_INT8_NATIVE = "int8_use_native_calibration_table"
OPT_EXHAUSTIVE = "exhaustive_tune"
OPT_COMPUTE_MODE = "compute_mode"
OPT_COMPILE_BATCHES = "compile_batches"

# Environment variables the EP reads AFTER the provider options (the env value
# wins), mapped to the .mxr filename component they change. Setting any of
# these while compiling means the deployment environment must set them too.
HASH_AFFECTING_ENV_VARS = {
    "ORT_MIGRAPHX_FP16_ENABLE": "options",
    "ORT_MIGRAPHX_BF16_ENABLE": "options",
    "ORT_MIGRAPHX_FP8_ENABLE": "options",
    "ORT_MIGRAPHX_INT8_ENABLE": "options",
    "ORT_MIGRAPHX_INT8_CALIBRATION_TABLE_NAME": "options",
    "ORT_MIGRAPHX_INT8_USE_NATIVE_CALIBRATION_TABLE": "options",
    "ORT_MIGRAPHX_EXHAUSTIVE_TUNE": "options",
    "ORT_MIGRAPHX_COMPUTE_MODE": "options",
    "ORT_MIGRAPHX_MLSS_USE_SPECIFIC_OPS": "mlss_ops",
    "ORT_MIGRAPHX_MLSS_EXCLUDE_GRAPH_IDS": "mlss_ops",
    "ORT_MIGRAPHX_CPU_CONTROL_FLOW": "graph_id (partitioning)",
    "ORT_MIGRAPHX_MAX_DYNAMIC_BATCH": "shapes",
    "ORT_MIGRAPHX_COMPILE_BATCHES": "shapes",
    "ORT_MIGRAPHX_STATIC_PAD_SEQ": "shapes",
    "ORT_MIGRAPHX_STATIC_PAD_SEQ_LEN": "shapes",
    "ORT_MIGRAPHX_STATIC_PAD_INPUTS": "shapes",
    "ORT_MIGRAPHX_CACHE_DIR": "(output location, not the name)",
}

FILENAME_FIELDS = (
    "version_hex", "graph_id_hash", "gpu_arch_hash",
    "options_hash", "mlss_ops_hash", "shape_hash",
)

# The two INFO lines the EP emits while building an .mxr name (mgx_ep.cc).
CACHE_OPTIONS_LINE = re.compile(
    r"\[mgx-cache\] graph=(?P<graph>\S+) arch=(?P<arch>\S+) "
    r"options_key='(?P<options_key>[^']*)' prefix=(?P<prefix>\S+)")
CACHE_MLSS_LINE = re.compile(
    r"\[mgx-cache\] graph=(?P<graph>\S+) mlss_ops='(?P<mlss_ops>[^']*)' "
    r"prefix=(?P<prefix>\S+)")
ORT_WARNING_LINE = re.compile(r"\[(W|E|F):onnxruntime")

VERIFY_CHILD_FLAG = "--_verify-child"
VERIFY_RESULT_MARKER = "VERIFY_RESULT "
# Unset for the verify session: the first skips the cache lookup, the second
# redirects the EP away from the scratch cache dir, and the third stops a miss
# from being written, which is how a shape mismatch is detected.
VERIFY_UNSET_ENV_VARS = (
    "ORT_MIGRAPHX_FORCE_RECOMPILE",
    "ORT_MIGRAPHX_CACHE_DIR",
    "ORT_MIGRAPHX_DISABLE_COMPILED_MODEL_CACHING",
)


def make_random_input(dtype_str, shape):
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
    resolved = [d if isinstance(d, int) and d > 0 else 1 for d in shape]

    if np.issubdtype(np_dtype, np.integer):
        return np.random.randint(0, 10, size=resolved, dtype=np_dtype)
    if np_dtype == np.bool_:
        return np.random.randint(0, 2, size=resolved).astype(np.bool_)
    return np.random.randn(*resolved).astype(np_dtype)


def resolve_shape(shape, batch_size):
    """Replace a symbolic/dynamic leading dim with the requested batch size."""
    resolved = []
    for i, d in enumerate(shape):
        if i == 0 and (d is None or isinstance(d, str) or d == -1):
            resolved.append(batch_size)
        elif d is None or isinstance(d, str) or d == -1:
            resolved.append(1)
        else:
            resolved.append(int(d))
    return resolved


def build_feed(sess, batch_size):
    """Build a random input feed dict for the given batch size."""
    feed = {}
    shapes = {}
    for inp in sess.get_inputs():
        shape = resolve_shape(list(inp.shape), batch_size)
        shapes[inp.name] = shape
        feed[inp.name] = make_random_input(inp.type, shape)
    return feed, shapes


def get_gpu_arch(device_id):
    """Best-effort lookup of the GPU arch for the console banner.

    The authoritative gcnArchName the EP hashes is taken from its [mgx-cache]
    log line; this is only a fallback when that line is missing.
    """
    try:
        out = subprocess.run(
            ["/opt/rocm/bin/rocminfo"],
            capture_output=True, text=True, timeout=30,
        ).stdout
        archs = [line.split()[-1] for line in out.splitlines()
                 if "Name:" in line and "gfx" in line]
        if archs:
            idx = device_id if device_id < len(archs) else 0
            return archs[idx]
    except Exception as e:  # noqa: BLE001 - best effort only
        return f"unknown (rocminfo failed: {e})"
    return "unknown"


def decode_mxr_filename(filename):
    """Split an EP .mxr filename into its components and decode the version.

    Only the version field is reversible; the other components are one-way
    MurmurHash3 values. Their plain-text inputs come from the EP log instead
    (see match_cache_log_entry).
    """
    stem = filename[:-4] if filename.endswith(".mxr") else filename
    parts = stem.split("-")
    decoded = {"raw_filename": filename, "components": parts}
    if len(parts) != len(FILENAME_FIELDS):
        decoded["warning"] = (
            f"expected {len(FILENAME_FIELDS)} components, got {len(parts)}; "
            "the EP filename layout may have changed")
    for field, value in zip(FILENAME_FIELDS, parts):
        decoded[field] = value
    try:
        v = int(parts[0], 16)
        decoded["migraphx_version"] = (
            f"{(v >> 16) & 0xFFFF}.{(v >> 8) & 0xFF}.{v & 0xFF}"
        )
    except ValueError:
        decoded["migraphx_version"] = "unparsable"
    return decoded


def quant_subdir(quantize):
    """Map the selected quantization level(s) to a subfolder name.

    No quantization -> 'fp32'; a single level -> that level; mixed precision ->
    the levels joined by '_' in canonical (QUANT_CHOICES) order, e.g.
    {fp16, int8} -> 'fp16_int8'.
    """
    if not quantize:
        return "fp32"
    selected = [q for q in QUANT_CHOICES if q in set(quantize)]
    return "_".join(selected)


def parse_ep_option(text):
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got '{text}'")
    return key, value


def stage_calibration_table(calib_path, cache_dir):
    """Copy the calibration table into cache_dir and return the name to pass.

    The EP opens the table at <cache_dir>/<name> and hashes <name> as a plain
    string, so passing a bare file name keeps the hash independent of where the
    table lived on the build machine. A deployment must put the same file in
    its cache_dir and pass the same name.
    """
    name = os.path.basename(calib_path)
    dst = os.path.join(cache_dir, name)
    if os.path.abspath(calib_path) != os.path.abspath(dst):
        shutil.copy2(calib_path, dst)
    return name


def build_provider_options(args, cache_dir, calib_name):
    """Translate CLI flags into MIGraphX plugin EP provider options."""
    quant = set(args.quantize or [])
    opts = {
        OPT_DEVICE_ID: str(args.device_id),
        OPT_CACHE_DIR: cache_dir,
        OPT_FP16: "1" if "fp16" in quant else "0",
        OPT_BF16: "1" if "bf16" in quant else "0",
        OPT_FP8: "1" if "fp8" in quant else "0",
        OPT_INT8: "1" if "int8" in quant else "0",
        OPT_EXHAUSTIVE: "1" if args.exhaustive_tune else "0",
        # Force the requested batch to be precompiled at session init for models
        # with a dynamic batch dim; a run() below covers any remaining cases.
        OPT_COMPILE_BATCHES: str(args.batch),
    }
    if args.compute_mode:
        opts[OPT_COMPUTE_MODE] = args.compute_mode
    if calib_name:
        opts[OPT_INT8_CALIB] = calib_name
        opts[OPT_INT8_NATIVE] = "1" if args.int8_native_calib else "0"
    for key, value in args.ep_option or []:
        opts[key] = value
    return opts


def expected_option_tokens(args):
    """The options_key tokens the requested flags should produce in the EP."""
    quant = set(args.quantize or [])
    tokens = {q for q in QUANT_CHOICES if q in quant}
    if args.exhaustive_tune:
        tokens.add("tune")
    return tokens


def snapshot_migraphx_env():
    return {k: v for k, v in sorted(os.environ.items())
            if k.startswith("ORT_MIGRAPHX_") or k.startswith("MIGRAPHX_")}


@contextlib.contextmanager
def capture_native_stderr():
    """Redirect fd 2 to a temp file so the EP's native log lines can be parsed.

    ORT's default log sink writes to stderr from C++, which Python's sys.stderr
    redirection cannot see; swapping the file descriptor catches both.
    """
    sys.stderr.flush()
    saved_fd = os.dup(2)
    tmp = tempfile.TemporaryFile(mode="w+b")
    os.dup2(tmp.fileno(), 2)
    captured = {"text": ""}
    try:
        yield captured
    finally:
        sys.stderr.flush()
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        tmp.seek(0)
        captured["text"] = tmp.read().decode("utf-8", errors="replace")
        tmp.close()


def parse_cache_log(text):
    """Collect the EP's [mgx-cache] lines: per-graph option keys + MLSS entries."""
    options_by_graph = {}
    for m in CACHE_OPTIONS_LINE.finditer(text):
        options_by_graph[m["graph"]] = m.groupdict()
    mlss_entries = [m.groupdict() for m in CACHE_MLSS_LINE.finditer(text)]
    return options_by_graph, mlss_entries


def match_cache_log_entry(mxr_name, options_by_graph, mlss_entries):
    """Find the logged cache key whose prefix produced this .mxr filename.

    The MLSS line carries the full prefix (everything except the shape hash),
    so the longest one the filename starts with identifies the subgraph.
    """
    best = None
    for entry in mlss_entries:
        if mxr_name.startswith(entry["prefix"]) and (
                best is None or len(entry["prefix"]) > len(best["prefix"])):
            best = entry
    if best is None:
        return None
    options = options_by_graph.get(best["graph"], {})
    return {
        "graph_id": best["graph"],
        "gpu_arch": options.get("arch"),
        "options_key": options.get("options_key"),
        "mlss_ops": best["mlss_ops"],
        "prefix": best["prefix"],
    }


def md5sum(path, chunk_size=1024 * 1024):
    """Return the hex MD5 digest of a file, or None if it can't be read."""
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def list_mxr(cache_dir):
    try:
        return {f for f in os.listdir(cache_dir) if f.endswith(".mxr")}
    except FileNotFoundError:
        return set()


def write_manifest(mxr_path, args, provider_options, input_shapes,
                   output_shapes, gpu_arch, ort_version, ep_key, env_snapshot,
                   verify_status, deploy_options):
    """Write a <name>_<md5>_manifest.log next to the generated .mxr."""
    mxr_dir = os.path.dirname(mxr_path)
    mxr_name = os.path.basename(mxr_path)

    mxr_md5 = md5sum(mxr_path) or "nomd5"
    manifest_name = mxr_name[:-4] + "_" + mxr_md5 + "_manifest.log"
    manifest_path = os.path.join(mxr_dir, manifest_name)

    decoded = decode_mxr_filename(mxr_name)
    file_size = os.path.getsize(mxr_path) if os.path.exists(mxr_path) else 0
    model_filename = os.path.basename(args.model)

    lines = []
    lines.append("MIGraphX EP .mxr compile manifest")
    lines.append("=" * 70)
    lines.append(f"generated          : {datetime.datetime.now().isoformat()}")
    lines.append(f"output .mxr file   : {mxr_name}")
    lines.append(f"full path          : {os.path.abspath(mxr_path)}")
    lines.append(f"file size (bytes)  : {file_size} ({file_size / (1024 * 1024):.2f} MB)")
    lines.append(f"md5sum (.mxr)      : {mxr_md5}")
    lines.append("")
    lines.append("--- Build inputs ---")
    lines.append(f"model path         : {os.path.abspath(args.model)}")
    lines.append(f"model filename     : {model_filename}")
    lines.append("  (NOTE: the graph id hashes the model FILE NAME, not its directory;"
                 " renaming the .onnx changes the hash, moving it does not.)")
    lines.append(f"batch size         : {args.batch}")
    lines.append(f"quantization       : {', '.join(args.quantize) if args.quantize else 'none (fp32)'}")
    lines.append(f"compute mode       : {args.compute_mode or 'EP default (balanced)'}")
    lines.append(f"exhaustive tune    : {bool(args.exhaustive_tune)}")
    if args.int8_calib_table:
        lines.append(f"int8 calib table   : {os.path.abspath(args.int8_calib_table)}")
        lines.append(f"  passed to EP as  : {provider_options.get(OPT_INT8_CALIB)}"
                     " (copied into the cache dir; deploy it there under this name)")
        lines.append(f"int8 native calib  : {bool(args.int8_native_calib)}")
    lines.append(f"ep name            : {args.ep_name}")
    lines.append(f"ep library         : {os.path.abspath(args.ep_library)}")
    lines.append(f"gpu device id      : {args.device_id}")
    lines.append(f"onnxruntime version: {ort_version}")
    lines.append("")
    lines.append("input shapes (resolved at batch):")
    for name, shape in input_shapes.items():
        lines.append(f"  {name}: {shape}")
    if output_shapes:
        lines.append("output shapes:")
        for name, shape in output_shapes.items():
            lines.append(f"  {name}: {shape}")
    lines.append("")
    lines.append("--- Decoded .mxr filename components ---")
    lines.append(f"raw filename       : {decoded['raw_filename']}")
    if "warning" in decoded:
        lines.append(f"WARNING            : {decoded['warning']}")
    lines.append(f"migraphx version   : {decoded['migraphx_version']}"
                 f"  (from field '{decoded.get('version_hex', '?')}')")
    lines.append(f"graph id hash      : {decoded.get('graph_id_hash', 'n/a')}")
    lines.append(f"gpu arch hash      : {decoded.get('gpu_arch_hash', 'n/a')}")
    lines.append(f"options hash       : {decoded.get('options_hash', 'n/a')}")
    lines.append(f"mlss ops hash      : {decoded.get('mlss_ops_hash', 'n/a')}")
    lines.append(f"shape hash         : {decoded.get('shape_hash', 'n/a')}")
    lines.append("  (all but the version are one-way MurmurHash3 values.)")
    lines.append("")
    lines.append("--- Hash inputs reported by the EP ([mgx-cache] log) ---")
    if ep_key:
        lines.append(f"graph id           : {ep_key['graph_id']}")
        lines.append(f"gpu arch           : {ep_key['gpu_arch'] or gpu_arch + ' (rocminfo; EP line missing)'}")
        lines.append(f"options key        : {ep_key['options_key'] if ep_key['options_key'] is not None else 'n/a'}")
        lines.append(f"mlss ops           : {ep_key['mlss_ops'] or '(none)'}")
        lines.append(f"name prefix        : {ep_key['prefix']}")
    else:
        lines.append("  No [mgx-cache] line matched this file. The EP build may predate the")
        lines.append("  cache-key logging, or the file came from an earlier run.")
        lines.append(f"  gpu arch (rocminfo, approx): {gpu_arch}")
    lines.append("")
    lines.append("--- ORT_MIGRAPHX_* / MIGRAPHX_* environment at compile time ---")
    if env_snapshot:
        for key, value in env_snapshot.items():
            affects = HASH_AFFECTING_ENV_VARS.get(key)
            suffix = f"   <- affects {affects}" if affects else ""
            lines.append(f"  {key}={value}{suffix}")
    else:
        lines.append("  (none set)")
    lines.append("")
    lines.append("--- Provider options passed to the EP ---")
    lines.append(json.dumps(provider_options, indent=2))
    lines.append("")
    lines.append("--- Deployment verification (--verify) ---")
    if verify_status is None:
        lines.append("  not run")
    else:
        lines.append(f"status             : {verify_status}")
        lines.append(f"verify batch       : {args.verify_batch or args.batch}")
        lines.append("deployment options (cache_dir pointed at a scratch dir):")
        lines.append(json.dumps(deploy_options, indent=2))
    lines.append("")

    with open(manifest_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return manifest_path


def register_ep(ort, ep_name, ep_library):
    """Register the plugin EP library with ORT and return its OrtEpDevices."""
    ort.register_execution_provider_library(ep_name, os.path.abspath(ep_library))
    devices = [d for d in ort.get_ep_devices() if d.ep_name == ep_name]
    if not devices:
        raise RuntimeError(f"{ep_name} registered but reported no devices. "
                           f"Saw: {[d.ep_name for d in ort.get_ep_devices()]}")
    return devices


def run_model(ort, devices, model, provider_options, batch):
    """Create a session on the EP, run it once at `batch`, and capture the EP log.

    Returns a dict with the captured log text, input/output shapes, session
    creation time, and failure as (stage, message) or None.
    """
    # INFO is the level the EP's [mgx-cache] lines are logged at.
    ort.set_default_logger_severity(1)
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 1
    sess_opts.add_provider_for_devices(devices, provider_options)

    result = {"input_shapes": {}, "output_shapes": {}, "session_ms": None, "failure": None}
    with capture_native_stderr() as ep_log:
        try:
            t0 = time.perf_counter()
            sess = ort.InferenceSession(model, sess_options=sess_opts)
            result["session_ms"] = (time.perf_counter() - t0) * 1000
            # Force any deferred (dynamic-shape) compilation by running once at
            # the requested batch size.
            feed, result["input_shapes"] = build_feed(sess, batch)
            try:
                outputs = sess.run(None, feed)
                for meta, arr in zip(sess.get_outputs(), outputs):
                    result["output_shapes"][meta.name] = list(arr.shape)
            except Exception as e:  # noqa: BLE001
                result["failure"] = ("run", str(e))
            # Release the session so cache writes are flushed before we scan.
            del sess
        except Exception as e:  # noqa: BLE001
            result["failure"] = ("session", str(e))
    result["log"] = ep_log["text"]
    return result


def verify_child(config_path):
    """Entry point of the --verify subprocess: one deployment-style session.

    Runs in its own process so nothing compiled or registered by the compile
    session (in-memory programs, EP state) can satisfy the lookup; the only
    way to avoid a compile is to load the .mxr from disk.
    """
    with open(config_path) as f:
        cfg = json.load(f)
    out = {"failure": None, "new_files": [], "options_by_graph": {},
           "mlss_entries": [], "ep_warnings": [], "log_tail": []}
    try:
        import onnxruntime as ort
        devices = register_ep(ort, cfg["ep_name"], cfg["ep_library"])
    except Exception as e:  # noqa: BLE001
        out["failure"] = ("register", str(e))
        print(VERIFY_RESULT_MARKER + json.dumps(out))
        return
    before = list_mxr(cfg["cache_dir"])
    run = run_model(ort, devices, cfg["model"], cfg["provider_options"], cfg["batch"])
    out["failure"] = run["failure"]
    out["new_files"] = sorted(list_mxr(cfg["cache_dir"]) - before)
    out["options_by_graph"], out["mlss_entries"] = parse_cache_log(run["log"])
    log_lines = run["log"].splitlines()
    out["ep_warnings"] = [line for line in log_lines if ORT_WARNING_LINE.search(line)]
    if run["failure"]:
        out["log_tail"] = log_lines[-40:]
    print(VERIFY_RESULT_MARKER + json.dumps(out))


def link_cache_files(src_dir, dst_dir):
    """Expose every file in src_dir inside dst_dir without copying the data."""
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(dst_dir, name)
        try:
            os.symlink(src, dst)
        except OSError:
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)


def describe_mismatch(new_file, compiled_files):
    """Return the compiled file closest to new_file and the components that differ."""
    new = decode_mxr_filename(new_file)
    closest, closest_diff = None, None
    for compiled in compiled_files:
        old = decode_mxr_filename(compiled)
        diff = [field for field in FILENAME_FIELDS if new.get(field) != old.get(field)]
        if closest_diff is None or len(diff) < len(closest_diff):
            closest, closest_diff = compiled, diff
    return closest, closest_diff or []


def build_deploy_options(args, provider_options):
    """Deployment provider options: the compile options, adjusted by the CLI."""
    opts = {k: v for k, v in provider_options.items() if k != OPT_CACHE_DIR}
    for key in args.deploy_drop_option or []:
        opts.pop(key, None)
    for key, value in args.deploy_option or []:
        opts[key] = value
    return opts


def run_verify(args, deploy_options, cache_dir, compiled_files,
               compile_options_by_graph, compile_mlss_entries):
    """Reopen the model in a fresh process with the deployment options and check
    that the EP loads the compiled .mxr files instead of compiling new ones.

    Returns (passed, {compiled file: status string}).
    """
    verify_batch = args.verify_batch or args.batch
    print(f"\n[verify] Opening a deployment session in a fresh process "
          f"(batch={verify_batch}, options={deploy_options})...")

    env = dict(os.environ)
    for key in VERIFY_UNSET_ENV_VARS:
        if env.pop(key, None) is not None:
            print(f"[verify] Note: {key} is unset for the verify session.")
    # disable_caching only stops saving, never loading, so it does not change
    # what a deployment would hit; forcing it off makes a miss visible.
    child_options = dict(deploy_options)
    if child_options.pop("force_recompile", None) is not None:
        print("[verify] Note: force_recompile is dropped for the verify session.")
    child_options["disable_caching"] = "0"

    with tempfile.TemporaryDirectory(prefix="mxr_verify_") as verify_dir:
        # A miss lands in the scratch dir and is discarded with it. Using a
        # different directory also shows the cache dir is not part of the name.
        link_cache_files(cache_dir, verify_dir)
        child_options[OPT_CACHE_DIR] = verify_dir
        cfg = {
            "model": os.path.abspath(args.model),
            "ep_name": args.ep_name,
            "ep_library": os.path.abspath(args.ep_library),
            "batch": verify_batch,
            "cache_dir": verify_dir,
            "provider_options": child_options,
        }
        cfg_path = os.path.join(verify_dir, "verify_config.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), VERIFY_CHILD_FLAG, cfg_path],
            capture_output=True, text=True, env=env)

    def fail(reason):
        print(f"[verify] FAIL: {reason}")
        return False, {f: f"FAIL: {reason}" for f in compiled_files}

    result = None
    for line in proc.stdout.splitlines():
        if line.startswith(VERIFY_RESULT_MARKER):
            result = json.loads(line[len(VERIFY_RESULT_MARKER):])
    if result is None:
        for line in (proc.stdout + proc.stderr).splitlines()[-40:]:
            print(f"  {line}")
        return fail(f"the verify process exited with code {proc.returncode} without a result")

    for line in result["ep_warnings"]:
        print(f"[verify][ep-log] {line}")
    if result["failure"]:
        stage, message = result["failure"]
        for line in result["log_tail"]:
            print(f"  {line}")
        return fail(f"deployment {stage} failed: {message}")

    mlss_entries = result["mlss_entries"]
    if not mlss_entries:
        return fail("the deployment session logged no [mgx-cache] lines, so the EP compiled "
                    "no subgraph (CPU fallback?) or the EP build lacks cache-key logging")

    new_files = result["new_files"]
    if new_files:
        print(f"[verify] The deployment session compiled {len(new_files)} new .mxr file(s) "
              "instead of loading the cached ones:")
        for new_file in new_files:
            closest, diff = describe_mismatch(new_file, compiled_files)
            print(f"  {new_file}")
            if closest:
                print(f"    closest compiled file: {closest}")
                print(f"    differs in          : {', '.join(diff) or '(nothing)'}")

        compile_graphs = set(compile_options_by_graph)
        deploy_graphs = set(result["options_by_graph"])
        if compile_graphs != deploy_graphs:
            print(f"    graph ids differ: compile {sorted(compile_graphs)} vs deployment "
                  f"{sorted(deploy_graphs)} (model name, input shapes, or partitioning changed)")
        for graph, entry in result["options_by_graph"].items():
            compiled_entry = compile_options_by_graph.get(graph)
            if compiled_entry and compiled_entry["options_key"] != entry["options_key"]:
                print(f"    graph {graph}: options_key compile='{compiled_entry['options_key']}'"
                      f" deployment='{entry['options_key']}'")
        compile_mlss = {e["graph"]: e["mlss_ops"] for e in compile_mlss_entries}
        for entry in mlss_entries:
            if entry["graph"] in compile_mlss and compile_mlss[entry["graph"]] != entry["mlss_ops"]:
                print(f"    graph {entry['graph']}: mlss_ops compile='{compile_mlss[entry['graph']]}'"
                      f" deployment='{entry['mlss_ops']}'")
        return fail("the deployment session recompiled instead of loading the cache")

    unmatched = [e["prefix"] for e in mlss_entries
                 if not any(f.startswith(e["prefix"]) for f in compiled_files)]
    if unmatched:
        return fail(f"the deployment session looked up cache keys {unmatched} that name none "
                    "of the compiled files")

    status = {}
    for compiled in compiled_files:
        if any(compiled.startswith(e["prefix"]) for e in mlss_entries):
            status[compiled] = ("PASS: a fresh deployment session matched this cache key "
                                "and compiled nothing new")
        else:
            status[compiled] = "not looked up by the deployment session"
    print(f"[verify] PASS: the deployment session compiled nothing new; "
          f"{sum(s.startswith('PASS') for s in status.values())} of {len(compiled_files)} "
          "compiled file(s) match its cache keys.")
    return True, status


def main():
    if len(sys.argv) == 3 and sys.argv[1] == VERIFY_CHILD_FLAG:
        verify_child(sys.argv[2])
        return

    parser = argparse.ArgumentParser(
        description="Compile an ONNX model into a MIGraphX EP-named .mxr cache file, "
                    "with a companion build manifest.")
    parser.add_argument("--model", required=True, help="Path to ONNX model file")
    parser.add_argument("--ep-library", required=True,
                        help="Path to the plugin EP library to register with ORT, e.g. "
                             "libmigraphx-ep.so (standalone MIGraphX EP) or libamdgpu-ep.so")
    parser.add_argument("--ep-name", default="MIGraphXExecutionProvider",
                        help="EP name to register the library under (default: "
                             "MIGraphXExecutionProvider; use AMDGPUExecutionProvider "
                             "for the AMDGPU EP)")
    parser.add_argument("--batch", type=int, default=1,
                        help="Batch size to compile for (default: 1)")
    parser.add_argument("--quantize", nargs="+", choices=QUANT_CHOICES, default=None,
                        help=f"Quantization level(s) supported by the EP: {QUANT_CHOICES}. "
                             "Omit for fp32.")
    parser.add_argument("--int8-calib-table", type=str, default=None,
                        help="Path to an INT8/FP8 calibration table (with --quantize int8 or "
                             "fp8). It is copied into the cache dir and passed to the EP by "
                             "file name, because the EP hashes the name as given.")
    parser.add_argument("--int8-native-calib", action="store_true",
                        help="Treat the calibration table as a native MIGraphX table")
    parser.add_argument("--exhaustive-tune", action="store_true",
                        help="Enable exhaustive tuning (slower compile, faster kernels)")
    parser.add_argument("--compute-mode", choices=COMPUTE_MODE_CHOICES, default=None,
                        help="EP compute mode (part of the .mxr hash). Omit for the EP "
                             "default (balanced).")
    parser.add_argument("--ep-option", type=parse_ep_option, action="append",
                        metavar="KEY=VALUE",
                        help="Extra EP provider option, repeatable. Pass anything the "
                             "deployment sets that can change the hash, e.g. "
                             "mlss_use_specific_ops or max_dynamic_batch.")
    parser.add_argument("--device-id", type=int, default=0,
                        help="GPU device id (default: 0)")
    parser.add_argument("--output-dir", type=str, default="./mxr_out",
                        help="Base directory for output. The .mxr + manifest are written to a "
                             "per-quantization subfolder of it (e.g. <output-dir>/fp16), which "
                             "becomes the EP's cache_dir. Default: ./mxr_out")
    parser.add_argument("--verify", action="store_true",
                        help="After compiling, open the model again in a fresh process with "
                             "the deployment options and fail (exit 2) unless the EP loads "
                             "the compiled .mxr files instead of compiling new ones.")
    parser.add_argument("--verify-batch", type=int, default=None,
                        help="Batch size for the verify session (default: --batch)")
    parser.add_argument("--deploy-option", type=parse_ep_option, action="append",
                        metavar="KEY=VALUE",
                        help="With --verify: set or override a provider option for the "
                             "deployment session (repeatable). Deployment options start as "
                             "a copy of the compile options.")
    parser.add_argument("--deploy-drop-option", action="append", metavar="KEY",
                        help="With --verify: remove a compile provider option from the "
                             "deployment session (repeatable), e.g. compile_batches.")

    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)
    if not os.path.isfile(args.ep_library):
        print(f"ERROR: EP library not found: {args.ep_library}")
        sys.exit(1)
    quant = set(args.quantize or [])
    if args.int8_calib_table:
        if not quant & {"int8", "fp8"}:
            print("ERROR: --int8-calib-table requires --quantize int8 or fp8")
            sys.exit(1)
        if not os.path.isfile(args.int8_calib_table):
            print(f"ERROR: Calibration table not found: {args.int8_calib_table}")
            sys.exit(1)

    import onnxruntime as ort

    quant_level = quant_subdir(args.quantize)
    cache_dir = os.path.join(os.path.abspath(args.output_dir), quant_level)
    os.makedirs(cache_dir, exist_ok=True)

    calib_name = (stage_calibration_table(args.int8_calib_table, cache_dir)
                  if args.int8_calib_table else None)
    provider_options = build_provider_options(args, cache_dir, calib_name)
    gpu_arch = get_gpu_arch(args.device_id)
    env_snapshot = snapshot_migraphx_env()

    print(f"[init] Model:          {args.model}")
    print(f"[init] EP:             {args.ep_name} ({args.ep_library})")
    print(f"[init] Batch size:     {args.batch}")
    print(f"[init] Quantization:   {args.quantize or 'none (fp32)'}  ->  subfolder '{quant_level}/'")
    print(f"[init] Compute mode:   {args.compute_mode or 'EP default (balanced)'}")
    print(f"[init] Exhaustive tune: {args.exhaustive_tune}")
    print(f"[init] Output dir:     {cache_dir}")
    print(f"[init] GPU arch:       {gpu_arch}")
    print(f"[init] Provider opts:  {provider_options}")
    for key in env_snapshot:
        if key in HASH_AFFECTING_ENV_VARS:
            print(f"[init] WARNING: {key} is set and overrides the provider options "
                  f"(affects {HASH_AFFECTING_ENV_VARS[key]}); the deployment must match it.")

    try:
        devices = register_ep(ort, args.ep_name, args.ep_library)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}")
        sys.exit(1)

    before = list_mxr(cache_dir)

    print("\n[compile] Creating ORT session with the MIGraphX EP (compiles + writes .mxr);"
          " EP log output is captured until the run finishes...")
    run = run_model(ort, devices, args.model, provider_options, args.batch)
    input_shapes = run["input_shapes"]
    output_shapes = run["output_shapes"]
    failure = run["failure"]

    for line in run["log"].splitlines():
        if ORT_WARNING_LINE.search(line):
            print(f"[ep-log] {line}")

    if failure and failure[0] == "session":
        print(f"\n[compile] ERROR: session creation failed: {failure[1]}")
        tail = run["log"].splitlines()[-40:]
        if tail:
            print("[compile] Last EP log lines:")
            for line in tail:
                print(f"  {line}")
        sys.exit(1)
    print(f"[compile] Session created in {run['session_ms']:.0f} ms")
    if failure:
        print(f"[compile] WARNING: inference failed ({failure[1]}); .mxr may still have been "
              "written during session init.")
    else:
        print(f"[compile] Inference OK at batch={args.batch}")

    options_by_graph, mlss_entries = parse_cache_log(run["log"])
    if not mlss_entries:
        print("[compile] WARNING: no [mgx-cache] lines in the EP log; the manifest cannot "
              "report the hash inputs. Is the EP built from a revision with cache-key logging?")

    expected = expected_option_tokens(args)
    for graph, entry in options_by_graph.items():
        actual = {t for t in entry["options_key"].split(";") if t} & (set(QUANT_CHOICES) | {"tune"})
        if actual != expected:
            print(f"[compile] WARNING: graph {graph}: EP hashed {sorted(actual) or ['fp32']} but "
                  f"the flags requested {sorted(expected) or ['fp32']}. An env var override or a "
                  "calibration table the EP could not read changes the key.")

    after = list_mxr(cache_dir)
    new_files = sorted(after - before)
    ort_version = ort.__version__

    if not new_files:
        # Cache hit: the EP logged the prefixes it looked up, so report the
        # existing files those prefixes name.
        new_files = sorted(f for f in after
                           if match_cache_log_entry(f, options_by_graph, mlss_entries))
        if new_files:
            print("\n[result] No NEW .mxr files were created; this session loaded existing"
                  " ones (cache hit). Refreshing their manifests.")
        else:
            print("\n[result] No .mxr files were created or matched. Check that the EP "
                  "compiled the model (not CPU fallback) and that caching is not disabled.")
            sys.exit(1)

    deploy_options = None
    verify_passed = None
    verify_status = {}
    if args.verify:
        deploy_options = build_deploy_options(args, provider_options)
        verify_passed, verify_status = run_verify(
            args, deploy_options, cache_dir, new_files, options_by_graph, mlss_entries)

    print(f"\n[result] {len(new_files)} .mxr file(s):")
    written = []
    for fname in new_files:
        mxr_path = os.path.join(cache_dir, fname)
        ep_key = match_cache_log_entry(fname, options_by_graph, mlss_entries)
        manifest_path = write_manifest(
            mxr_path, args, provider_options, input_shapes,
            output_shapes, gpu_arch, ort_version, ep_key, env_snapshot,
            verify_status.get(fname), deploy_options)
        written.append((mxr_path, manifest_path))
        print(f"  .mxr      : {mxr_path}")
        print(f"  manifest  : {manifest_path}")
        if fname in verify_status:
            print(f"  verify    : {verify_status[fname]}")

    print("\n[done] Generated the following pairs:")
    for mxr_path, manifest_path in written:
        print(f"  {os.path.basename(mxr_path)}  +  {os.path.basename(manifest_path)}")

    if verify_passed is False:
        print("\n[done] Verification FAILED: a deployment with these options would recompile.")
        sys.exit(2)


if __name__ == "__main__":
    main()
