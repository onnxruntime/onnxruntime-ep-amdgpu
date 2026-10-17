@echo off
REM Focused MSVC build of ONLY the DirectX backend (directx-backend.dll) for the WebNN
REM EPContext work (branch WebNN_Enablement). The umbrella (amdgpu-ep.dll) loads each
REM backend as a SEPARATE runtime DLL via CreateEpFactories (gpu_factory.cc:51,129), so
REM directx-backend.dll can be built independently with MSVC cl.exe (its stock DML sources
REM need the MSVC preprocessor), while migraphx/hip stay on their ROCm-clang build.
REM USE_DML alone gates add_subdirectory(directx) (src/CMakeLists.txt:164) independent of
REM USE_AMDGPU/USE_MIGRAPHX -> only shared+directx configure, no migraphx find_package.
set "SRC=C:\repos\onnxruntime-ep-amdgpu-muse"
set "BLD=C:\repos\onnxruntime-ep-amdgpu-muse\build_directx_msvc\Release"
cmake --compile-no-warning-as-error ^
  -DUSE_DML=ON -DUSE_AMDGPU=OFF -DUSE_MIGRAPHX=OFF -DUSE_HIP=OFF ^
  -DUSE_BINSKIM_COMPLIANT_COMPILE_FLAGS=OFF ^
  -DCMAKE_PREFIX_PATH="C:/opt/onnxruntime-2.2/Release" ^
  -G "Visual Studio 17 2022" -A x64 ^
  -S "%SRC%" -B "%BLD%"
echo CONFIGURE_EXIT:%ERRORLEVEL%
cmake --build "%BLD%" --config Release --target directx-ep -j 30
echo DIRECTX_MSVC_BUILD_EXIT:%ERRORLEVEL%
