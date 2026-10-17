@echo off
REM Build ONLY the umbrella amdgpu-ep.dll WITH the USE_DML runtime loader enabled, so it
REM dlopens directx-backend.dll at runtime (gpu_factory.cc:128 #ifdef USE_DML). We build
REM just --target amdgpu-ep under ROCm clang: the umbrella does NOT link directx-ep
REM (it loads it at runtime), so the clang-incompatible DirectML sources are never compiled
REM here. directx-backend.dll comes from the separate MSVC build (_build_directx_msvc.bat).
set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;C:\opt\rocm\bin;%PATH%"
set "MIGRAPHX_HOME=C:\opt\migraphx\Release"
set "ONNXRT_HOME=C:\opt\onnxruntime-2.2\Release"
python "C:\repos\onnxruntime-ep-amdgpu-muse\tools\ci_build\build.py" ^
  --build_dir "C:\repos\onnxruntime-ep-amdgpu-muse\build_umbrella_dml" ^
  --config Release ^
  --use_amdgpu ^
  --parallel ^
  --compile_no_warning_as_error ^
  --cmake_generator Ninja ^
  --hip_path C:/opt/rocm ^
  --target amdgpu-ep ^
  --cmake_extra_defines CMAKE_C_COMPILER=C:/opt/rocm/bin/clang.exe CMAKE_CXX_COMPILER=C:/opt/rocm/bin/clang++.exe GPU_TARGETS=gfx1151 USE_DML=ON USE_HIP=OFF USE_MIGRAPHX=ON
echo UMBRELLA_DML_BUILD_EXIT:%ERRORLEVEL%
