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
