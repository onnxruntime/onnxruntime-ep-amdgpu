#!/usr/bin/env python3
"""
winml_cache_demo.py -- run an ONNX model on the AMD GPU plugin execution
provider from the installed Windows ML EP package, and show whether MIGraphX
.mxr caching was used.

Run it TWICE with the same options:
  1st run  compiles the model  -> mxr_cache=miss
  2nd run  loads the .mxr      -> mxr_cache=hit   (and should be much faster)

The script saves its outputs, so the 2nd run also checks that the cached
program returns the same numbers as the freshly compiled one.

  python winml_cache_demo.py                      # no caching (baseline)
  python winml_cache_demo.py --mode cache         # cache directory  <- normal way
  python winml_cache_demo.py --mode env           # cache via environment variable
  python winml_cache_demo.py --mode epcontext     # EPContext / compiled model

Requirements:
  - the Windows ML AMD GPU EP package installed (MicrosoftCorporationII.WinML.AMD.GPU.EP.2.0)
  - pip install onnxruntime numpy      (onnxruntime 1.24 or newer)
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# The plugin EP. The name we register it under becomes the prefix for provider
# options: "ep.<lowercase name>.<option>". The EP package's manifest declares
# this same name, so using it here matches what Windows ML does.
EP_NAME = "AMDGPUExecutionProvider"
EP_DLL = "amdgpu-ep.dll"

# The EP writes one line per compiled model here, ending in mxr_cache=hit or miss.
TELEMETRY = Path(os.environ["USERPROFILE"]) / "AppData/LocalLow/AMDGPUEP/telemetry.log"


def find_ep_folder():
    """Locate the ExecutionProvider folder of the installed Windows ML AMD GPU EP package.

    Asks Windows where the package lives. C:\\Program Files\\WindowsApps cannot be
    listed without elevation, so searching the filesystem for it does not work.
    """
    command = ("(Get-AppxPackage -Name 'MicrosoftCorporationII.WinML.AMD.GPU.EP*' "
               "| Sort-Object -Property Version | Select-Object -Last 1).InstallLocation")
    try:
        location = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                                  capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        location = ""

    folder = Path(location) / "ExecutionProvider" if location else None
    if not folder or not (folder / EP_DLL).is_file():
        raise SystemExit(
            "Could not find an installed Windows ML AMD GPU EP package.\n"
            "Check it is installed with:\n"
            "    Get-AppxPackage -Name '*WinML.AMD.GPU.EP*'\n"
            "Or pass the folder containing amdgpu-ep.dll with --ep-dir")
    return folder


def read_cache_result():
    """Return 'hit' or 'miss' from the newest telemetry line that reports it."""
    try:
        lines = TELEMETRY.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        for field in line.split():
            if field.startswith("mxr_cache="):
                return field.split("=", 1)[1]
    return None


def make_inputs(session, batch):
    """Random but reproducible inputs matching the model's declared shapes."""
    rng = np.random.default_rng(0)
    dtypes = {"tensor(float)": np.float32, "tensor(float16)": np.float16,
              "tensor(int64)": np.int64, "tensor(int32)": np.int32}
    feeds = {}
    for inp in session.get_inputs():
        # Dynamic dimensions come back as strings or None -- substitute --batch.
        shape = [d if isinstance(d, int) and d > 0 else batch for d in inp.shape]
        dtype = dtypes.get(inp.type, np.float32)
        feeds[inp.name] = rng.standard_normal(shape).astype(dtype)
    return feeds


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(HERE / "resnet50-v2-7.onnx"),
                   help="model to run (default: resnet50-v2-7.onnx next to this script)")
    p.add_argument("--mode", default="none", choices=["none", "cache", "env", "epcontext"],
                   help="caching mechanism to use (default: none)")
    p.add_argument("--out-dir", default=str(HERE / "output"),
                   help="folder for everything this script produces (default: output)")
    p.add_argument("--cache-dir", default=None,
                   help="folder for .mxr files (default: <out-dir>/mxr_cache)")
    p.add_argument("--embed-mode", type=int, default=0, choices=[0, 1],
                   help="EPContext only: 0 keeps the .mxr as a separate file, "
                        "1 embeds it inside the context model (default: 0)")
    p.add_argument("--ep-dir", default=None,
                   help="EP folder (default: auto-detect the installed WinML EP package)")
    p.add_argument("--batch", type=int, default=1, help="value used for dynamic dimensions")
    p.add_argument("--static", action="store_true",
                   help="pin the model's dynamic dimensions to --batch, so models with a "
                        "symbolic batch size can be used with --mode epcontext")
    p.add_argument("--reset", action="store_true",
                   help="clear the cache and saved results, so the next run is a cold one")
    args = p.parse_args()

    model = Path(args.model).resolve()
    if not model.is_file():
        raise SystemExit(f"Model not found: {model}\n"
                         "See the README for how to download the example model, "
                         "or point at your own with --model")

    ep_dir = Path(args.ep_dir).resolve() if args.ep_dir else find_ep_folder()

    # Everything produced by a run lands under one folder so it is easy to look at.
    out_dir = Path(args.out_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else out_dir / "mxr_cache"
    context_dir = out_dir / "context_models"
    results = out_dir / "results"
    # Embedded and non-embedded context models are different files, so keep them apart.
    ctx_model = context_dir / f"{model.stem}_ctx_embed{args.embed_mode}.onnx"
    saved = results / f"{model.stem}_{args.mode}.npz"

    for d in (cache_dir, context_dir, results):
        d.mkdir(parents=True, exist_ok=True)

    if args.reset:
        for f in (list(cache_dir.glob("*.mxr")) + list(context_dir.glob("*"))
                  + list(results.glob("*"))):
            f.unlink()
        print("Cache, context models and saved results cleared "
              "-- the next run will be a cold one.\n")

    print(f"model      {model.name}")
    print(f"EP         {ep_dir}")
    print(f"mode       {args.mode}" + (f"  (embed_mode={args.embed_mode})"
                                       if args.mode == "epcontext" else ""))
    print(f"output     {out_dir}")

    # ------------------------------------------------------------------
    # Load the plugin EP out of the Windows ML package
    # ------------------------------------------------------------------
    # The EP's sibling DLLs (migraphx-backend.dll, amdhip64_7.dll, ...) live in
    # the same folder and are loaded by name, so put it on the DLL search path.
    os.add_dll_directory(str(ep_dir))
    import onnxruntime as ort

    ort.register_execution_provider_library(EP_NAME, str(ep_dir / EP_DLL))
    devices = [d for d in ort.get_ep_devices() if d.ep_name == EP_NAME]
    if not devices:
        raise SystemExit(f"The EP registered but reported no devices. "
                         f"Saw: {[d.ep_name for d in ort.get_ep_devices()]}")

    # ------------------------------------------------------------------
    # Turn caching on -- this is the only part that differs between modes
    # ------------------------------------------------------------------
    session_options = ort.SessionOptions()
    provider_options = {"device_id": "0"}
    model_to_open = model

    if args.static:
        # Many published models leave the batch size symbolic. EPContext rejects
        # those, so pin every symbolic dimension to --batch before the EP sees the
        # graph. The names are read from a short-lived CPU session.
        probe = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        symbolic = sorted({d for i in probe.get_inputs() for d in i.shape if isinstance(d, str)})
        del probe
        for name in symbolic:
            session_options.add_free_dimension_override_by_name(name, args.batch)
        print(f"static     pinned {symbolic} to {args.batch}" if symbolic
              else "static     model already has static input shapes")

    if args.mode == "none":
        # Nothing set. Caching is OFF by default: with no cache directory the EP
        # has nowhere to read or write, so the model is recompiled every run.
        print("           no cache directory set -- the model compiles every time")

    elif args.mode == "cache":
        # The normal way for an application to enable caching. ORT turns this
        # into the session config entry "ep.amdgpuexecutionprovider.cache_dir".
        provider_options["cache_dir"] = str(cache_dir)
        print(f"           cache_dir provider option -> {cache_dir}")

    elif args.mode == "env":
        # Same effect, process-wide. The EP reads this AFTER the provider
        # option, so if both are set the environment variable wins.
        os.environ["ORT_MIGRAPHX_CACHE_DIR"] = str(cache_dir)
        print(f"           ORT_MIGRAPHX_CACHE_DIR -> {cache_dir}")

    elif args.mode == "epcontext":
        # EPContext writes a companion *_ctx.onnx whose EPContext node points at
        # the .mxr. Later runs open that file instead of the original model.
        #
        # cache_dir is set for both writing and reading so the .mxr goes to the
        # cache folder and the context model to its own folder. Without it the EP
        # puts both next to the context file. It must be set on the read side too,
        # because the context model stores a relative path to the .mxr and resolves
        # it against this directory.
        provider_options["cache_dir"] = str(cache_dir)
        if ctx_model.exists():
            model_to_open = ctx_model
            print(f"           opening the context model {ctx_model.name}")
        else:
            session_options.add_session_config_entry("ep.context_enable", "1")
            session_options.add_session_config_entry("ep.context_file_path", str(ctx_model))
            session_options.add_session_config_entry("ep.context_embed_mode", str(args.embed_mode))
            print(f"           ep.context_enable=1 -> will write {ctx_model.name}")
            print("           embed_mode=1 puts the compiled program inside that file;"
                  if args.embed_mode else
                  "           embed_mode=0 keeps the compiled program in a separate .mxr;")
            print("           note: this fails if the model has dynamic input shapes")

    session_options.add_provider_for_devices(devices, provider_options)

    # ------------------------------------------------------------------
    # Create the session, then run it once
    # ------------------------------------------------------------------
    print("\ncreating session ...")
    t0 = time.perf_counter()
    try:
        session = ort.InferenceSession(str(model_to_open), sess_options=session_options)
    except Exception as error:
        if "dynamic input shapes" in str(error):
            raise SystemExit(
                "\nThis model has a dynamic input shape (usually the batch size), and\n"
                "EPContext cannot compile those. Either:\n"
                "  - add --static to pin the dynamic dimensions, or\n"
                "  - use --mode cache, which works with dynamic shapes.") from None
        raise
    create_s = time.perf_counter() - t0

    feeds = make_inputs(session, args.batch)
    print("running inference ...")
    t0 = time.perf_counter()
    outputs = session.run(None, feeds)
    run_s = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    total_s = create_s + run_s
    print()
    print("-" * 58)
    print(f"  session creation   {create_s:8.2f} s")
    print(f"  first inference    {run_s:8.2f} s")
    print(f"  total              {total_s:8.2f} s   <- compare this between runs")
    print("-" * 58)
    # Some models compile during session creation and others during the first
    # inference, so the total is the only number that is fair to compare.

    state = read_cache_result()
    if state == "hit":
        print("  cache              HIT  -- loaded a compiled program from disk")
    elif state == "miss":
        print("  cache              MISS -- compiled the model from scratch")
    else:
        print("  cache              unknown (no telemetry written)")

    # Show what is on disk. With EPContext and no cache_dir the .mxr is written
    # next to the context model, so look in both places.
    print()
    print(f"  files in {out_dir}")
    artefacts = sorted(cache_dir.glob("*.mxr")) + sorted(context_dir.glob("*"))
    if artefacts:
        for f in artefacts:
            print(f"    {f.parent.name}/{f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
    else:
        print("    (no .mxr or context model -- nothing was cached)")

    # ------------------------------------------------------------------
    # Compare with the previous run of this same mode
    # ------------------------------------------------------------------
    print()
    if saved.exists():
        previous = np.load(saved)
        previous_s = float(previous["total_s"])
        previous_outputs = [previous[k] for k in previous.files if k != "total_s"]

        match = len(previous_outputs) == len(outputs) and all(
            a.shape == b.shape and np.allclose(a.astype(np.float64), b.astype(np.float64),
                                               rtol=1e-3, atol=1e-3, equal_nan=True)
            for a, b in zip(previous_outputs, outputs))

        print(f"  previous run       {previous_s:8.2f} s")
        print(f"  this run           {total_s:8.2f} s   ({previous_s / total_s:.1f}x faster)")
        print(f"  outputs            {'match the previous run' if match else 'DIFFER from the previous run'}")
    else:
        print("  No previous run saved for this mode.")
        print("  Run the same command again to see whether the cache gets used.")

    np.savez(saved, total_s=np.float64(total_s),
             **{f"out{i}": o for i, o in enumerate(outputs)})


if __name__ == "__main__":
    main()
