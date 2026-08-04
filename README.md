# AMD Plugin Execution Providers

> Plugin execution providers for [ONNX Runtime](https://onnxruntime.ai/) targeting AMD GPUs, FPGAs, and MI accelerators.

This repository contains AMD execution providers that implement the ONNX Runtime plugin EP ABI (introduced in ORT 1.24).
Each provider is built as a standalone shared library that ONNX Runtime loads at runtime via the `OrtEpFactory`, `OrtEp`,
`OrtNodeComputeInfo`, `OrtDataTransferImpl`, and related interfaces.

## Execution Providers

| Provider | Target | Directory | Build Flag |
|----------|--------|-----------|------------|
| **AMDGPU** | AMD GPUs (generic ROCm) | `src/amdgpu/` | `--use_amdgpu` |
| **DirectML** | Windows GPU (DirectX 12) | `src/directml/` | `--use_dml` |
| **MIGraphX** | AMD Instinct / Radeon GPUs (ROCm) | `src/migraphx/` | `--use_migraphx` |

## Prerequisites

- **ONNX Runtime** ≥ 1.24 (EP plugin ABI support)
- **CMake** ≥ 4.2
- **Python** ≥ 3.12 (for the build script and wheel packaging)
- A supported **CMake generator**: Ninja (recommended), NMake, Unix Makefiles, or Visual Studio
- Platform-specific SDK depending on the target provider:
  - **ROCm** — for MIGraphX and AMDGPU providers
  - **DirectX 12 / Windows SDK** — for DirectML provider (Windows only)

## Building

The build is driven by `build.bat` (Windows) or `build.sh` (Linux), which wraps `tools/ci_build/build.py`.

### Example: MIGraphX EP

**Windows:**
```cmd
.\build.bat --config Release ^
    --onnxrt_home "<path-to-onnxruntime>" ^
    --use_migraphx ^
    --migraphx_home "<path-to-migraphx>" ^
    --parallel ^
    --build_wheel ^
    --compile_no_warning_as_error
```

**Linux:**
```bash
./build.sh --config Release \
    --onnxrt_home "<path-to-onnxruntime>" \
    --use_migraphx \
    --parallel \
    --build_wheel \
    --compile_no_warning_as_error
```

Replace `<path-to-onnxruntime>` and `<path-to-migraphx>` with the actual install paths on your system.

### End-to-end Linux build (MIGraphX EP)

The `build.sh` example above assumes you already have compatible ONNX Runtime and
AMDMIGraphX installations. If you are starting from source (for example inside a
ROCm container), the
[`scripts/build_migraphx_ep_standalone.sh`](scripts/build_migraphx_ep_standalone.sh)
script automates the complete flow: installing build dependencies, building and
installing AMDMIGraphX, building and installing ONNX Runtime, and finally building
the MIGraphX EP against those installs.

Run it directly:

```bash
./scripts/build_migraphx_ep_standalone.sh \
    --rocm-path /opt/rocm \
    --migraphx-install "$HOME/develop/AMDMIGraphX_install" \
    --onnxrt-install "$HOME/develop/onnxruntime_install"
```

#### Prerequisites

- A ROCm installation (default `/opt/rocm`; `rocminfo` is used to auto-detect the
  GPU target via `GPU_TARGETS`).
- Local checkouts of the three source trees. The script expects them at the
  following paths, overridable via environment variables:

  | Source tree | Default path | Environment variable |
  |-------------|--------------|----------------------|
  | AMDMIGraphX | `/workspace/AMDMIGraphX` | `AMDMIGRAPHX_SRC` |
  | ONNX Runtime | `/onnxruntime` | `ONNXRUNTIME_SRC` |
  | onnxruntime-ep-amdgpu (this repo) | `/workspace/onnxruntime-ep-amdgpu` | `ONNXRUNTIME_EP_SRC` |

#### Configuration

The install prefixes and ROCm path can be set via CLI flags or environment
variables (CLI flags take precedence):

| CLI flag | Environment variable | Default |
|----------|----------------------|---------|
| `--rocm-path` | `ROCM_PATH` | `/opt/rocm` |
| `--migraphx-install` | `AMDMIGRAPHX_INSTALL` | `$HOME/develop/AMDMIGraphX_install` |
| `--onnxrt-install` | `ONNXRUNTIME_INSTALL` | `$HOME/develop/onnxruntime_install` |

#### What the script does

0. **System + Python dependencies.** Installs `curl`, `zip`, `unzip`, `tar`, and
   the Python build tooling (`ninja`, `packaging>=24.2`, `cmake==4.2.3`, and
   [`rbuild`](https://github.com/RadeonOpenCompute/rbuild)). A newer CMake is
   prepended to `PATH`, and `CXXFLAGS` is set to
   `-D__HIP_PLATFORM_AMD__=1 -w`.

1. **Build and install AMDMIGraphX (`develop` branch).** Detects the GPU target
   from `rocminfo`, then builds with `rbuild`:

   ```bash
   rbuild build -d depend -B build \
       -DGPU_TARGETS="$GPU_TARGETS" \
       -DMIGRAPHX_USE_COMPOSABLEKERNEL=OFF
   cmake --install build --prefix "$AMDMIGRAPHX_INSTALL"
   ```

2. **Build and install ONNX Runtime (`v1.27.0` tag).** Checks out the tag and
   cherry-picks an AMD GPU vendor-ID fix that still needs to be upstreamed
   (`38fc6102a2ac126c3a94a01039635a4d40740e76`), then builds the Release
   configuration with the wheel, shared library, and C# bindings enabled before
   installing to `$ONNXRUNTIME_INSTALL`.

3. **Build onnxruntime-ep-amdgpu (`main` branch).** Builds the MIGraphX EP against
   the installs produced above:

   ```bash
   ./build.sh --config Release \
       --cmake_generator Ninja \
       --onnxrt_home "$ONNXRUNTIME_INSTALL" \
       --use_migraphx \
       --migraphx_home "$AMDMIGRAPHX_INSTALL" \
       --compile_no_warning_as_error \
       --parallel 16 \
       --build_dir build.EP.MGX \
       --hip_path "$ROCM_PATH" \
       --build_wheel
   ```

> **Note:** The script runs `apt-get` and `pip3 install` and checks out specific
> branches/tags in each source tree, so it is intended to be run in a disposable
> or container environment (for example, an official ROCm image). Review the
> pinned versions before running it against long-lived working trees.

### Build Options Reference

| Flag | Description |
|------|-------------|
| `--config` | Build configuration: `Debug`, `Release`, `RelWithDebInfo`, `MinSizeRel` |
| `--cmake_extra_defines` | Pass additional CMake definitions (`-DKEY=VALUE`) |
| `--cmake_generator` | CMake generator: `Ninja` (recommended), `Unix Makefiles`, `Visual Studio 17 2022`, etc. |
| `--hip_path` | Path to the HIP SDK installation directory (optional) |
| `--onnxrt_home` | Path to the ONNX Runtime installation (headers + libraries) |
| `--use_amdgpu` | Enable the AMDGPU execution provider (implies `--use_migraphx` and `--use_dml`) |
| `--use_dml` | Enable the DirectML execution provider (Windows only) |
| `--use_migraphx` | Enable the MIGraphX execution provider |
| `--migraphx_home` | Path to the MIGraphX installation |
| `--parallel [N]` | Number of parallel build jobs (0 = auto-detect CPU count) |
| `--build_wheel` | Build a Python wheel package for each enabled EP |
| `--compile_no_warning_as_error` | Do not treat compiler warnings as errors |
| `--use_binskim_compliant_compile_flags` | Enable BinSkim-compliant compile flags (Windows only) |
| `--enable_lto` | Enable Link Time Optimization |
| `--use_cache` | Enable compiler artifacts caching (e.g., ccache) |
| `--build_dir` | Path to the build directory (default build\${configuration}) |

Environment variables `ORT_HOME`, `MIGRAPHX_HOME`, and `HIP_PATH` can be used as alternatives to the corresponding command-line flags.

### Building Python Wheels

When `--build_wheel` is specified alongside an EP flag (e.g., `--use_migraphx`), the build system:

1. Compiles the native EP shared library
2. Reads the EP's `pyproject.toml` (e.g. `src/migraphx/pyproject.toml`)
3. Packages the library and its runtime dependencies into a Python wheel
4. Outputs the `.whl` file to `<build_dir>/<config>/dist/`

### Deploying Wheels

To upload built wheels to a PyPI repository:

```bash
./build.sh --config Release \
    --use_migraphx --migraphx_home "<path-to-migraphx>" \
    --onnxrt_home "<path-to-onnxruntime>" \
    --build_wheel --deploy_wheel \
    --pypi_repo_url "https://your-pypi-server/simple/" \
    --pypi_token "$PYPI_TOKEN"
```


## Project Structure

```
├── cmake/                      # CMake helper modules
├── docs/                       # Design and User documents
├── src/
│   ├── amdgpu/                 # AMDGPU execution provider
│   ├── directml/               # DirectML execution provider (Windows)
│   ├── migraphx/               # MIGraphX execution provider
│   └── shared/                 # Common utilities shared across EPs
├── tools/ci_build/build.py     # Build orchestration script
├── build.bat                   # Build entry points for Windows 
├── build.sh                    # Build entry points for Linux
└── CMakeLists.txt              # Top-level CMake configuration
```

## Contributing

We welcome contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for the issue-reporting and pull-request workflow.

For bugs and feature requests, open a [GitHub Issue](../../issues).

## Security

To report a security vulnerability, **do not open a public GitHub issue**. See [SECURITY.md](SECURITY.md) for our responsible disclosure policy.

## License

See [LICENSE](LICENSE) for details.
