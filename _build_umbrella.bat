@echo off
REM Isolated umbrella (amdgpu-ep.dll) build for the Muse routing e2e.
set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;C:\opt\rocm\bin;%PATH%"
set "MIGRAPHX_HOME=C:\opt\migraphx\Release"
set "ONNXRT_HOME=C:\opt\onnxruntime-2.2\Release"
python "C:\repos\onnxruntime-ep-amdgpu-muse\tools\ci_build\build.py" ^
  --build_dir "C:\repos\onnxruntime-ep-amdgpu-muse\build" ^
  --config Release ^
  --use_amdgpu ^
  --parallel ^
  --compile_no_warning_as_error ^
  --cmake_generator Ninja ^
  --hip_path C:/opt/rocm ^
  --cmake_extra_defines CMAKE_C_COMPILER=C:/opt/rocm/bin/clang.exe CMAKE_CXX_COMPILER=C:/opt/rocm/bin/clang++.exe GPU_TARGETS=gfx1151 USE_DML=OFF USE_HIP=OFF USE_MIGRAPHX=ON
echo UMBRELLA_BUILD_EXIT:%ERRORLEVEL%
