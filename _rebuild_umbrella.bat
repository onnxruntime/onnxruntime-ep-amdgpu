@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul
set "MIGRAPHX_HOME=C:\opt\migraphx\Release"
cmake --build "C:\repos\onnxruntime-ep-amdgpu-muse\build_dml_msvc" --config Release --target amdgpu-ep -- -m -nologo -verbosity:minimal
echo EP_BUILD_EXIT:%ERRORLEVEL%
