# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded$<$<CONFIG:Debug>:Debug>)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_definitions("$<$<COMPILE_LANGUAGE:C,CXX>:ONNX_NAMESPACE=onnx;ONNX_ML;NOMINMAX>")

find_package(onnxruntime REQUIRED)

get_target_property(_onnxruntime_location onnxruntime::onnxruntime LOCATION)
message(STATUS "Found ONNXRuntime: ${_onnxruntime_location}")
message(STATUS "ONNXRuntime version: ${onnxruntime_VERSION}")
file(WRITE "${CMAKE_BINARY_DIR}/onnxruntime_version.txt" "${onnxruntime_VERSION}")

include(FetchContent)

set(CACHE{BUILD_SHARED_LIBS} TYPE BOOL FORCE VALUE OFF)
set(CACHE{ZLIB_USE_STATIC_LIBS} TYPE BOOL FORCE VALUE ON)

FetchContent_Declare(
        fmt
        URL https://github.com/fmtlib/fmt/releases/download/12.1.0/fmt-12.1.0.zip
        DOWNLOAD_NO_PROGRESS TRUE
        SYSTEM
        EXCLUDE_FROM_ALL)

set(CACHE{FMT_INSTALL} TYPE INTERNAL VALUE OFF)
set(CACHE{FMT_DOC} TYPE INTERNAL VALUE OFF)
set(CACHE{FMT_SYSTEM_HEADERS} TYPE INTERNAL VALUE ON)
set(CACHE{FMT_TEST}  TYPE INTERNAL VALUE OFF)

FetchContent_MakeAvailable(fmt)

FetchContent_Declare(
        gsl
        URL https://github.com/microsoft/GSL/archive/refs/tags/v4.2.1.zip
        DOWNLOAD_NO_PROGRESS TRUE
        SYSTEM
        EXCLUDE_FROM_ALL)

set(CACHE{GLS_TEST} TYPE INTERNAL VALUE OFF)
set(CACHE{GLS_INSTALL} TYPE INTERNAL VALUE OFF)

FetchContent_MakeAvailable(gsl)

FetchContent_Declare(
        json
        URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
        DOWNLOAD_NO_PROGRESS TRUE
        SYSTEM
        EXCLUDE_FROM_ALL)

set(CACHE{JSON_BuildTests} TYPE INTERNAL VALUE OFF)
set(CACHE{JSON_SystemInclude} TYPE INTERNAL VALUE ON)

FetchContent_MakeAvailable(json)

FetchContent_Declare(
        zlib
        URL https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.xz
        DOWNLOAD_NO_PROGRESS TRUE
        SYSTEM
        EXCLUDE_FROM_ALL)

set(CACHE{ZLIB_BUILD_TESTING} TYPE INTERNAL VALUE OFF)
set(CACHE{ZLIB_BUILD_EXAMPLES} TYPE INTERNAL VALUE OFF)
set(CACHE{ZLIB_BUILD_SHARED} TYPE INTERNAL VALUE OFF)
set(CACHE{ZLIB_BUILD_MINIZIP} TYPE INTERNAL VALUE OFF)

FetchContent_Declare(
        protobuf
        URL https://github.com/protocolbuffers/protobuf/archive/refs/tags/v34.1.zip
        DOWNLOAD_NO_PROGRESS TRUE
        SYSTEM
        EXCLUDE_FROM_ALL)

set(CACHE{protobuf_BUILD_LIBPROTOC} TYPE INTERNAL VALUE ON)
set(CACHE{protobuf_BUILD_PROTOBUF_BINARIES} TYPE INTERNAL VALUE ON)
set(CACHE{protobuf_BUILD_PROTOC_BINARIES} TYPE INTERNAL VALUE ON)
set(CACHE{protobuf_BUILD_SHARED_LIBS} TYPE INTERNAL VALUE OFF)
set(CACHE{protobuf_BUILD_TESTS} TYPE INTERNAL VALUE OFF)
set(CACHE{protobuf_MSVC_STATIC_RUNTIME} TYPE INTERNAL VALUE ON)

FetchContent_Declare(
        onnx
        URL https://github.com/onnx/onnx/archive/refs/tags/v1.21.0.zip
        DOWNLOAD_NO_PROGRESS TRUE
        SYSTEM
        EXCLUDE_FROM_ALL)

set(CACHE{ONNX_USE_MSVC_STATIC_RUNTIME} TYPE INTERNAL VALUE ON)

FetchContent_Declare(
        flatbuffers
        URL https://github.com/google/flatbuffers/archive/refs/tags/v25.12.19.zip
        DOWNLOAD_NO_PROGRESS TRUE
        SYSTEM
        EXCLUDE_FROM_ALL)

FetchContent_MakeAvailable(zlib flatbuffers protobuf onnx)

FetchContent_Declare(
        range-v3
        URL https://github.com/ericniebler/range-v3/archive/refs/tags/0.12.0.zip
        DOWNLOAD_NO_PROGRESS TRUE
        SYSTEM
        EXCLUDE_FROM_ALL)

set(CACHE{RANGE_V3_TESTS} TYPE INTERNAL VALUE OFF)
set(CACHE{RANGE_V3_EXAMPLES} TYPE INTERNAL VALUE OFF)
set(CACHE{RANGE_V3_DOCS} TYPE INTERNAL VALUE OFF)
set(CACHE{RANGES_BUILD_CALENDAR_EXAMPLE} TYPE INTERNAL VALUE OFF)
set(CACHE{RANGES_DEEP_STL_INTEGRATION} TYPE INTERNAL VALUE ON)

FetchContent_MakeAvailable(range-v3)

if(USE_DML)
    FetchContent_Declare(
            eigen
            URL https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.zip
            DOWNLOAD_NO_PROGRESS TRUE
            SYSTEM
            EXCLUDE_FROM_ALL)

    set(CACHE{EIGEN_BUILD_DOC} TYPE INTERNAL VALUE OFF)
    set(CACHE{EIGEN_BUILD_TESTING} TYPE INTERNAL VALUE OFF)
    set(CACHE{EIGEN_BUILD_PKGCONFIG} TYPE INTERNAL VALUE OFF)
    set(CACHE{BUILD_TESTING} TYPE INTERNAL VALUE OFF)

    FetchContent_MakeAvailable(eigen)

    FetchContent_Declare(
            d3dx12
            URL https://github.com/microsoft/DirectX-Headers/archive/refs/tags/v1.619.1.zip
            DOWNLOAD_NO_PROGRESS TRUE
            SYSTEM
            EXCLUDE_FROM_ALL)

    set(CACHE{DXHEADERS_BUILD_TEST} TYPE INTERNAL VALUE OFF)
    set(CACHE{DXHEADERS_INSTALL} TYPE INTERNAL VALUE OFF)
    set(CACHE{DXHEADERS_BUILD_GOOGLE_TEST} TYPE INTERNAL VALUE OFF)

    FetchContent_MakeAvailable(d3dx12)

    FetchContent_Declare(
            boost
            URL https://github.com/boostorg/boost/releases/download/boost-1.90.0/boost-1.90.0-cmake.zip
            DOWNLOAD_NO_PROGRESS TRUE
            SYSTEM
            EXCLUDE_FROM_ALL)

    set(CACHE{BUILD_SHARED_LIBS} TYPE INTERNAL VALUE OFF)
    set(CACHE{BOOST_INCLUDE_LIBRARIES} TYPE INTERNAL VALUE "mp11")

    FetchContent_MakeAvailable(boost)
endif()

add_subdirectory(shared)

if(USE_AMDGPU)
    add_subdirectory(amdgpu)
endif()
if(USE_DML)
    add_subdirectory(directml)
endif()
if(USE_MIGRAPHX)
    add_subdirectory(migraphx)
endif()
if(USE_HIPEP)
    add_subdirectory(hipep)
endif()
