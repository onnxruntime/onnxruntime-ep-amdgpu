# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 4.2 FATAL_ERROR)

get_property(__generator_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT __generator_is_multi_config AND NOT CMAKE_BUILD_TYPE)
  set(CACHE{CMAKE_BUILD_TYPE} TYPE STRING FORCE VALUE Release)
endif()

cmake_policy(SET CMP0091 NEW)

project(onnxruntime-ep-amdgpu VERSION 1.0.0 LANGUAGES C CXX)
include(cmake/VersionInfo.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(WIN32)
    option(USE_SPECTRE_MITIGATED_LIBRARIES "Use Spectre-mitigated libraries" OFF)
    option(USE_BINSKIM_COMPLIANT_COMPILE_FLAGS "Prepare for BinSkim Binary Analyzer" OFF)
    if(USE_BINSKIM_COMPLIANT_COMPILE_FLAGS)
        message(STATUS "Control Flow Guard enabled")
        if(MSVC AND MSVC_VERSION GREATER_EQUAL 1900)
            add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/guard:cf>)
            add_link_options(/guard:cf /DYNAMICBASE)
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:SHELL:-Xclang -cfguard>")
            add_link_options(-Wl,-guard:cf)
        endif()
    endif()
    if(USE_SPECTRE_MITIGATED_LIBRARIES)
        message(STATUS "Using Spectre-mitigated libraries")
        add_compiler_options($<$<COMPILE_LANGUAGE:C,CXX>:/Qspectre>)
    endif()
endif()

option(USE_AMDGPU "Build the AMD GPU Execution Provider" OFF)
option(USE_DML "Build the DirectML Execution Provider" OFF)
option(USE_MIGRAPHX "Build the MIGraphX Execution Provider" OFF)
option(USE_HIP "Build the HIP (morphizen) backend shim" OFF)

if(USE_AMDGPU)
    set(CACHE{USE_DML} TYPE BOOL FORCE VALUE ON)
    set(CACHE{USE_MIGRAPHX} TYPE BOOL FORCE VALUE ON)
    set(CACHE{USE_HIP} TYPE BOOL FORCE VALUE ON)
endif()

add_subdirectory(src)
