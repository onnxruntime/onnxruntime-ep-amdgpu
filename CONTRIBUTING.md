# Contributing to AMD Plugin Execution Providers

Thanks for your interest in contributing.

## Reporting Issues

Use [GitHub Issues](../../issues) to report bugs or request features. Include a clear description, reproduction steps,
and your environment: OS, ROCm version, GPU architecture (e.g. `gfx942`, `gfx950`, `gfx1100`), and ONNX Runtime version.

## Pull Request Workflow

1. **Fork and branch.** [Fork the repository](../../fork), then create a branch from `main`:
   ```bash
   git clone https://github.com/<your-fork>/onnxruntime-ep-amdgpu.git
   cd onnxruntime-ep-amdgpu
   git checkout -b <your-branch-name>
   ```

2. **Make your change.** Add or update tests as appropriate and ensure documentation reflects any behaviour changes.

3. **Format your code.** C/C++ changes must conform to the repo's `.clang-format`:
   ```bash
   git clang-format origin/main
   ```

4. **Build and test locally** (example for MIGraphX EP):

   **Windows:**
   ```cmd
   build.bat --config Release ^
       --cmake_generator Ninja ^
       --onnxrt_home "<path-to-onnxruntime>" ^
       --use_migraphx ^
       --migraphx_home "<path-to-migraphx>" ^
       --compile_no_warning_as_error ^
       --parallel 30 ^
       --build_wheel ^
       --build_dir build.EP.MGX ^
       --use_binskim_compliant_compile_flags
   ```

   **Linux:**
   ```bash
   ./build.sh --config Release \
       --cmake_generator Ninja \
       --onnxrt_home "<path-to-onnxruntime>" \
       --use_migraphx \
       --migraphx_home "<path-to-migraphx>" \
       --compile_no_warning_as_error \
       --parallel 30 \
       --build_wheel \
       --build_dir build.EP.MGX
   ```

   Replace `<path-to-onnxruntime>` and `<path-to-migraphx>` with the actual install paths on your system. See `CMakeLists.txt`
for all available options and other providers.

5. **Open a Pull Request** against `main`. Describe *what* changed and *why*; link any related issue.

6. **CI and review.** Ensure CI passes and request review from the relevant [CODEOWNERS](.github/CODEOWNERS).

By opening a PR you agree your contribution is licensed under the terms in [LICENSE](LICENSE).

## Coding Standards

- C/C++ style is enforced via `.clang-format` at the repo root.
- Python code should follow [PEP 8](https://peps.python.org/pep-0008/).
- Keep commits focused — one logical change per commit.

## Security

For security issues, do **not** open a public issue — see [SECURITY.md](SECURITY.md).
