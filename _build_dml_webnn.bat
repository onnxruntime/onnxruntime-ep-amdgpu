@echo off
REM DML (DirectX backend) build of the AMDGPU umbrella for the WebNN EPContext work
REM (branch WebNN_Enablement). USE_DML=ON, MIGraphX/HIP off. Separate build dir so it
REM does not clobber the MIGraphX build/.
set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;C:\opt\rocm\bin;%PATH%"
REM --use_amdgpu forces USE_MIGRAPHX=ON (build.py:776) + USE_DML=ON on Windows, so provide
REM MIGRAPHX_HOME (prebuilt lib) for find_package(migraphx). HIP backend stays off.
set "MIGRAPHX_HOME=C:\opt\migraphx\Release"
set "ONNXRT_HOME=C:\opt\onnxruntime-2.2\Release"
python "C:\repos\onnxruntime-ep-amdgpu-muse\tools\ci_build\build.py" ^
  --build_dir "C:\repos\onnxruntime-ep-amdgpu-muse\build_dml" ^
  --config Release ^
  --use_amdgpu ^
  --parallel ^
  --compile_no_warning_as_error ^
  --cmake_generator Ninja ^
  --hip_path C:/opt/rocm ^
  --cmake_extra_defines CMAKE_C_COMPILER=C:/opt/rocm/bin/clang.exe CMAKE_CXX_COMPILER=C:/opt/rocm/bin/clang++.exe GPU_TARGETS=gfx1151 USE_HIP=OFF
echo DML_BUILD_EXIT:%ERRORLEVEL%
