"""
Compile an ONNX model into a MIGraphX EP-named .mxr cache file.

Rather than re-implementing the MIGraphX EP's filename hashing (MurmurHash3 over
the MIGraphX version, graph identity, GPU arch, and input shapes), this script
lets the EP itself do the compile + naming. It opens an ONNX Runtime session
with the MIGraphX EP, points migraphx_model_cache_dir at an output directory,
and lets the EP compile and write the hashed .mxr. Any .mxr produced this way is
byte- and name-identical to what a deployment session expects, so it will be
picked up automatically when the same model is later run with the same cache
dir, GPU arch, MIGraphX version, shapes, and provider options.

The .mxr filename hash is one-way and cannot be decoded back into the model. So
alongside each generated .mxr this script writes a companion manifest describing
the build inputs that produced the hash (model, batch, quantization, arch,
versions, decoded MIGraphX-version field, and the raw filename components).

Because the EP filename hash does not encode precision, an fp16 and an fp32
compile of the same model would otherwise produce the same .mxr name and
collide. As a workaround this script nests output under a per-quantization
subfolder of --output-dir (fp32, fp16, bf16, fp16_int8, ...), giving each
precision its own migraphx_model_cache_dir.

For each generated cache file the script writes (under <output-dir>/<quant>/):
    <onnxrt_named>.mxr                   (produced by the MIGraphX EP)
    <onnxrt_named>_<md5>_manifest.log    (build info; <md5> is the .mxr's md5sum)

Usage (inside a container / env with ORT + MIGraphX):
    python3 migraphx_hash_compile.py --model /path/to/model.onnx

    # Batch size and quantization:
    python3 migraphx_hash_compile.py --model model.onnx --batch 8 --quantize fp16

    # Multiple quantization levels, custom output dir:
    python3 migraphx_hash_compile.py --model model.onnx --quantize fp16 int8 \
        --int8-calib-table calib.flatbuffer --output-dir ./mxr_out

    # Exhaustive tuning (slower compile, faster kernels):
    python3 migraphx_hash_compile.py --model model.onnx --quantize fp16 --exhaustive-tune
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

QUANT_CHOICES = ["fp16", "bf16", "int8", "fp8"]

# migraphx_* provider-option keys (see migraphx_execution_provider_info.h)
OPT_FP16 = "migraphx_fp16_enable"
OPT_BF16 = "migraphx_bf16_enable"
OPT_FP8 = "migraphx_fp8_enable"
OPT_INT8 = "migraphx_int8_enable"
OPT_INT8_CALIB = "migraphx_int8_calibration_table_name"
OPT_INT8_NATIVE = "migraphx_int8_use_native_calibration_table"
OPT_EXHAUSTIVE = "migraphx_exhaustive_tune"
OPT_CACHE_DIR = "migraphx_model_cache_dir"
OPT_COMPILE_BATCHES = "migraphx_compile_batches"


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
    """Best-effort lookup of the gcnArchName string the EP hashes (e.g. gfx942).

    The EP hashes the exact string from hipGetDeviceProperties. We can only
    approximate it here for the manifest; this is informational, not used to
    name the file (the EP owns the real name).
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

    Filename layout (migraphx_execution_provider.cc):
        <version_hex>-<graph_id>-<gpu_arch_hash>-<shape_hash>.mxr

    Only the version field is reversible; the other three are one-way
    MurmurHash3 values recorded here purely for traceability.
    """
    stem = filename[:-4] if filename.endswith(".mxr") else filename
    parts = stem.split("-")
    decoded = {"raw_filename": filename, "components": parts}
    if len(parts) >= 1:
        try:
            v = int(parts[0], 16)
            decoded["version_hex"] = parts[0]
            decoded["migraphx_version"] = (
                f"{(v >> 16) & 0xFFFF}.{(v >> 8) & 0xFF}.{v & 0xFF}"
            )
        except ValueError:
            decoded["migraphx_version"] = "unpar?able"
    if len(parts) == 4:
        decoded["graph_id_hash"] = parts[1]
        decoded["gpu_arch_hash"] = parts[2]
        decoded["shape_hash"] = parts[3]
    return decoded


def quant_subdir(quantize):
    """Map the selected quantization level(s) to a subfolder name.

    No quantization -> 'fp32'; a single level -> that level; mixed precision ->
    the levels joined by '_' in canonical (QUANT_CHOICES) order, e.g.
    {fp16, int8} -> 'fp16_int8'. This gives each precision its own
    migraphx_model_cache_dir so fp16/fp32/etc. .mxr files never collide (the EP
    filename hash does not encode precision).
    """
    if not quantize:
        return "fp32"
    selected = [q for q in QUANT_CHOICES if q in set(quantize)]
    return "_".join(selected)


def build_provider_options(args, cache_dir):
    """Translate CLI flags into MIGraphX EP provider options."""
    quant = set(args.quantize or [])
    opts = {
        "device_id": str(args.device_id),
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
    if "int8" in quant and args.int8_calib_table:
        opts[OPT_INT8_CALIB] = args.int8_calib_table
        opts[OPT_INT8_NATIVE] = "1" if args.int8_native_calib else "0"
    return opts


def list_mxr(cache_dir):
    try:
        return {f for f in os.listdir(cache_dir) if f.endswith(".mxr")}
    except FileNotFoundError:
        return set()


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


def write_manifest(mxr_path, args, provider_options, input_shapes,
                   output_shapes, gpu_arch, ort_version):
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
    lines.append("--- Build inputs (these determine the EP's filename hash) ---")
    lines.append(f"model path         : {os.path.abspath(args.model)}")
    lines.append(f"model filename     : {model_filename}")
    lines.append("  (NOTE: GenerateGraphId hashes the model FILE NAME; renaming"
                 " the .onnx changes the hash.)")
    lines.append(f"batch size         : {args.batch}")
    lines.append(f"quantization       : {', '.join(args.quantize) if args.quantize else 'none (fp32)'}")
    lines.append(f"exhaustive tune    : {bool(args.exhaustive_tune)}")
    if args.int8_calib_table:
        lines.append(f"int8 calib table   : {args.int8_calib_table}")
        lines.append(f"int8 native calib  : {bool(args.int8_native_calib)}")
    lines.append(f"gpu device id      : {args.device_id}")
    lines.append(f"gpu arch (approx)  : {gpu_arch}")
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
    lines.append(f"raw filename       : {decoded.get('raw_filename')}")
    lines.append(f"migraphx version   : {decoded.get('migraphx_version', 'n/a')}"
                 f"  (from field '{decoded.get('version_hex', '?')}')")
    lines.append(f"graph id hash      : {decoded.get('graph_id_hash', 'n/a')}")
    lines.append(f"gpu arch hash      : {decoded.get('gpu_arch_hash', 'n/a')}")
    lines.append(f"shape hash         : {decoded.get('shape_hash', 'n/a')}")
    lines.append("  (graph id / arch / shape are one-way MurmurHash3 values and"
                 " cannot be inverted.)")
    lines.append("")
    lines.append("--- Provider options passed to ORT ---")
    lines.append(json.dumps(provider_options, indent=2))
    lines.append("")

    with open(manifest_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Compile an ONNX model into a MIGraphX EP-named .mxr cache file, "
                    "with a companion build manifest.")
    parser.add_argument("--model", required=True, help="Path to ONNX model file")
    parser.add_argument("--batch", type=int, default=1,
                        help="Batch size to compile for (default: 1)")
    parser.add_argument("--quantize", nargs="+", choices=QUANT_CHOICES, default=None,
                        help=f"Quantization level(s) supported by the EP: {QUANT_CHOICES}. "
                             "Omit for fp32.")
    parser.add_argument("--int8-calib-table", type=str, default=None,
                        help="Path to an INT8 calibration table (only with --quantize int8)")
    parser.add_argument("--int8-native-calib", action="store_true",
                        help="Treat the INT8 calibration table as a native MIGraphX table")
    parser.add_argument("--exhaustive-tune", action="store_true",
                        help="Enable exhaustive tuning (slower compile, faster kernels)")
    parser.add_argument("--device-id", type=int, default=0,
                        help="GPU device id (default: 0)")
    parser.add_argument("--output-dir", type=str, default="./mxr_out",
                        help="Base directory for output. The .mxr + manifest are written to a "
                             "per-quantization subfolder of it (e.g. <output-dir>/fp16), which "
                             "becomes migraphx_model_cache_dir. Default: ./mxr_out")

    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)
    if args.int8_calib_table and (not args.quantize or "int8" not in args.quantize):
        print("ERROR: --int8-calib-table requires --quantize int8")
        sys.exit(1)

    import onnxruntime as ort

    if "MIGraphXExecutionProvider" not in ort.get_available_providers():
        print("ERROR: MIGraphXExecutionProvider not available in this ORT build.")
        print(f"Available: {ort.get_available_providers()}")
        sys.exit(1)

    # Workaround for the EP filename hash not encoding precision: nest the
    # output under a per-quantization subfolder so different precisions of the
    # same model can never share (and overwrite/load) the same .mxr.
    quant_level = quant_subdir(args.quantize)
    cache_dir = os.path.join(os.path.abspath(args.output_dir), quant_level)
    os.makedirs(cache_dir, exist_ok=True)

    provider_options = build_provider_options(args, cache_dir)
    gpu_arch = get_gpu_arch(args.device_id)

    print(f"[init] Model:          {args.model}")
    print(f"[init] Batch size:     {args.batch}")
    print(f"[init] Quantization:   {args.quantize or 'none (fp32)'}  ->  subfolder '{quant_level}/'")
    print(f"[init] Exhaustive tune: {args.exhaustive_tune}")
    print(f"[init] Output dir:     {cache_dir}")
    print(f"[init] GPU arch:       {gpu_arch}")
    print(f"[init] Provider opts:  {provider_options}")

    before = list_mxr(cache_dir)

    print("\n[compile] Creating ORT session with MIGraphX EP (compiles + writes .mxr)...")
    t0 = time.perf_counter()
    sess_opts = ort.SessionOptions()
    sess = ort.InferenceSession(
        args.model,
        sess_options=sess_opts,
        providers=[("MIGraphXExecutionProvider", provider_options)],
    )
    print(f"[compile] Session created in {(time.perf_counter() - t0) * 1000:.0f} ms")

    # Force any deferred (dynamic-shape) compilation by running once at the
    # requested batch size.
    feed, input_shapes = build_feed(sess, args.batch)
    output_shapes = {}
    print(f"[compile] Running one inference at batch={args.batch} to force compilation...")
    try:
        outputs = sess.run(None, feed)
        for meta, arr in zip(sess.get_outputs(), outputs):
            output_shapes[meta.name] = list(arr.shape)
        print("[compile] Inference OK")
    except Exception as e:  # noqa: BLE001
        print(f"[compile] WARNING: inference failed ({e}); .mxr may still have been "
              "written during session init.")

    # Release the session so cache writes are flushed before we scan.
    del sess

    after = list_mxr(cache_dir)
    new_files = sorted(after - before)

    ort_version = ort.__version__

    if not new_files:
        print("\n[result] No NEW .mxr files were created.")
        print("[result] This usually means a matching cache file already existed"
              " (cache hit) for this model/arch/shape/version.")
        existing = sorted(after)
        if existing:
            print("[result] Writing/refreshing manifests for existing .mxr files"
                  " in the output dir:")
            new_files = existing
        else:
            print("[result] No .mxr files found at all. Check that "
                  "migraphx_model_cache_dir caching is supported by this EP build.")
            sys.exit(1)

    print(f"\n[result] {len(new_files)} .mxr file(s):")
    written = []
    for fname in new_files:
        mxr_path = os.path.join(cache_dir, fname)
        manifest_path = write_manifest(
            mxr_path, args, provider_options, input_shapes,
            output_shapes, gpu_arch, ort_version)
        written.append((mxr_path, manifest_path))
        print(f"  .mxr      : {mxr_path}")
        print(f"  manifest  : {manifest_path}")

    print("\n[done] Generated the following pairs:")
    for mxr_path, manifest_path in written:
        print(f"  {os.path.basename(mxr_path)}  +  {os.path.basename(manifest_path)}")


if __name__ == "__main__":
    main()
