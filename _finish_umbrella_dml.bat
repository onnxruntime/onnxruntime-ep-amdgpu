@echo off
REM build.py --target emits a malformed "-target" that cmake rejects, so finish the
REM already-configured umbrella tree by invoking cmake --build directly. Build ONLY
REM amdgpu-ep (clang/Ninja) -> it does not depend on directx-ep, so the clang-incompatible
REM DirectML sources are never compiled. vcvars64 gives clang++ the MSVC headers/libs.
set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;C:\opt\rocm\bin;%PATH%"
set "MIGRAPHX_HOME=C:\opt\migraphx\Release"
set "ONNXRT_HOME=C:\opt\onnxruntime-2.2\Release"
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >NUL
cmake --build "C:\repos\onnxruntime-ep-amdgpu-muse\build_umbrella_dml\Release" --config Release --target amdgpu-ep -j 30
echo UMBRELLA_DML_FINISH_EXIT:%ERRORLEVEL%
