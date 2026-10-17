@echo off
REM Canonical Windows DML build of the AMDGPU umbrella for the WebNN EPContext work
REM (branch WebNN_Enablement). The DirectX backend's stock Microsoft DML sources only
REM compile under MSVC cl.exe (their preprocessor accepts token-pasting that the ROCm
REM clang++ GNU driver rejects), so this uses the Visual Studio 2022 generator (build.py
REM runs vcvars64 -> cl.exe) with the clang-only backends (MIGraphX/HIP) OFF. We only
REM need the directx backend + the amdgpu router for WebNN routing.
set "ONNXRT_HOME=C:\opt\onnxruntime-2.2\Release"
set "MIGRAPHX_HOME=C:\opt\migraphx\Release"
python "C:\repos\onnxruntime-ep-amdgpu-muse\tools\ci_build\build.py" ^
  --build_dir "C:\repos\onnxruntime-ep-amdgpu-muse\build_dml_msvc" ^
  --config Release ^
  --use_amdgpu ^
  --parallel ^
  --compile_no_warning_as_error ^
  --cmake_generator "Visual Studio 17 2022" ^
  --cmake_extra_defines USE_MIGRAPHX=OFF USE_HIP=OFF
echo DML_MSVC_BUILD_EXIT:%ERRORLEVEL%
