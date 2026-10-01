# MIGraphX `.mxr` caching on the AMD GPU execution provider

Compiling an ONNX model for an AMD GPU takes time. Caching lets the execution
provider save the compiled program to disk as a `.mxr` file and reload it on
later runs instead of recompiling.

`mxr_cache_demo.py` demonstrates each way of enabling it, so you can confirm
caching works in your own environment.

---

## Setup

The Windows ML AMD GPU EP package must be installed. Check with:

```powershell
Get-AppxPackage -Name '*WinML.AMD.GPU.EP*'
```

Then:

```
pip install onnxruntime numpy
```

ONNX Runtime 1.24 or newer is required. The script locates the execution
provider itself by asking Windows where the package is installed, so there is
nothing else to configure.

### The model

The example uses ResNet-50 from the ONNX Model Zoo, which is not included here.
Download it into this folder as `resnet50-v2-7.onnx`:

<https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx>

Any ONNX model works — pass it with `--model`. Note this one has a dynamic batch
size, which matters for `--mode epcontext`; see the notes at the end.

---

## Running it

Run the same command **twice**. The first run compiles the model, the second
should load it from the cache.

```
python mxr_cache_demo.py --mode cache
```

---

## The modes

Each mode enables caching a different way.

| | |
|---|---|
| `--mode none` | No caching. The baseline — the model is recompiled on every run. |
| `--mode cache` | Sets the `cache_dir` provider option. **The normal way for an application to enable caching.** |
| `--mode env` | Sets the `ORT_MIGRAPHX_CACHE_DIR` environment variable. Same effect, but process-wide — useful for testing an application you cannot rebuild. |
| `--mode epcontext` | Writes a pre-compiled companion model (`*_ctx.onnx`) via `ep.context_enable`, and opens that file on later runs. Add `--embed-mode 1` to put the compiled program inside that file instead of keeping it in a separate `.mxr`. |

The equivalent settings in your own code:

```
provider option       cache_dir = <folder>
session config key    ep.amdgpuexecutionprovider.cache_dir = <folder>
environment variable  ORT_MIGRAPHX_CACHE_DIR = <folder>

EPContext             ep.context_enable = 1
                      ep.context_file_path = <file>
                      ep.context_embed_mode = 0    (1 embeds the .mxr in the file)
```

If both the provider option and the environment variable are set, the
environment variable wins.

### Other arguments

| | |
|---|---|
| `--model <path>` | Use a different model. Defaults to `resnet50-v2-7.onnx` in this folder. |
| `--out-dir <path>` | Where everything produced by a run is written. Defaults to `output` in this folder. |
| `--cache-dir <path>` | Where `.mxr` files are written. Defaults to `<out-dir>/mxr_cache`. |
| `--embed-mode 0\|1` | EPContext only. `0` keeps the compiled program in a separate `.mxr`, `1` embeds it in the context model. |
| `--ep-dir <path>` | Use a specific EP build instead of the installed package. |
| `--reset` | Clear the cache, context models and saved results, so the next run is a cold one. |
| `--batch <n>` | Value used for dynamic input dimensions. |
| `--static` | Pin the model's dynamic dimensions to `--batch`. Needed for `--mode epcontext` on a model with a symbolic batch size. |

---

## What a run produces

```
output/
  mxr_cache/         compiled programs (.mxr) for --mode cache and --mode env
  context_models/    *_ctx.onnx and its .mxr for --mode epcontext
  results/           saved outputs, used to compare one run against the next
```

Each run lists what is in there, so you can see the files appear on the first
run and be reused on the second.

**About the inputs.** The script generates random tensors shaped to match the
model it is given — with the default model, that means ResNet-50's input. They
are not real images, so the predictions are meaningless; the point is the
compile and load timing, and checking that a cached program returns the same
values as a freshly compiled one. The same fixed seed is used every run so the
two runs are comparable.

---

## Reading the result

Each run prints:

```
  session creation      ...
  first inference       ...
  total                 ...
  cache              HIT or MISS
```

**`cache HIT` / `MISS`** is reported by the execution provider itself rather
than inferred from timing.

- `MISS` — the model was compiled from scratch. Expected on the first run.
- `HIT` — a compiled program was loaded from disk.

**For a model with dynamic input shapes this indicator always reports `MISS`,
even when the cache was used.** Those models are compiled on the first inference
rather than during session creation, and that path does not update the
indicator. Judge them by the timing instead: if the second run returns in a
fraction of the time and the outputs still match, the cache was used. Passing
`--static` also avoids this, by making the shapes concrete.

**`total`** is the number to compare between runs. Some models compile while
the session is being created and others during the first inference, so session
creation time on its own is not a reliable measure.

From the second run onwards the script also compares against the previous run
and reports the speedup and whether the outputs still match.

---

## Notes

**Caching is off unless a cache directory is set.** There is no default
location. With none configured, the provider silently recompiles every time.

**EPContext requires static input shapes.** Many published models leave the
batch size symbolic, and those fail at session creation. Either add `--static`
to pin the dynamic dimensions, or use `--mode cache`, which works with dynamic
shapes as they are.

**`ep.context_embed_mode = 1` still writes the separate `.mxr`.** The compiled
program is embedded in the context model as intended, but the standalone file is
also left on disk, so the pair takes about twice the space. Expect to clean it up
yourself if you are shipping the context model on its own.

**A cache miss is expected after a change.** The cache key includes the
MIGraphX version, the GPU architecture, the compute mode and the input shapes,
so a miss after updating the EP package or changing any of these is correct.
It should hit again on the following run.

**The older option names are gone.** `migraphx_model_cache_dir`,
`ORT_MIGRAPHX_MODEL_CACHE_PATH`, `ORT_MIGRAPHX_SAVE_COMPILED_MODEL` and
`ORT_MIGRAPHX_LOAD_COMPILED_MODEL` are not recognised and produce no error.
Setting a cache directory is all that is needed — it enables both saving and
loading.
