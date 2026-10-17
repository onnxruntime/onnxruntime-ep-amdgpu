@echo off
REM DML MSVC build of the UPSTREAM-based WebNN port branch (webnn-aot-enablement), fresh dir.
set "ONNXRT_HOME=C:\opt\onnxruntime-2.2\Release"
set "MIGRAPHX_HOME=C:\opt\migraphx\Release"
python "C:\repos\onnxruntime-ep-amdgpu-muse\tools\ci_build\build.py" ^
  --build_dir "C:\repos\onnxruntime-ep-amdgpu-muse\build_webnn_port" ^
  --config Release ^
  --use_amdgpu ^
  --parallel ^
  --compile_no_warning_as_error ^
  --cmake_generator "Visual Studio 17 2022" ^
  --cmake_extra_defines USE_MIGRAPHX=OFF USE_HIP=OFF
echo PORT_BUILD_EXIT:%ERRORLEVEL%
