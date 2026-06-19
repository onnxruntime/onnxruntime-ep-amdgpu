# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(plugin-ep-utils STATIC
        common/constants.h
        common/dynamic_library.h
        common/endian.h
        common/enumerate.h
        common/env_var.h
        common/force_inline.h
        common/helper.cc
        common/make_string.h
        common/murmurhash3.cc
        common/murmurhash3.h
        common/ort_graph_to_proto.cc
        common/ort_graph_to_proto.h
        common/parse_string.h
        common/path_string.h
        common/plugin_ep_utils.cc
        common/plugin_ep_utils.h
        common/provider_options.h
        common/provider_options_utils.h
        common/telemetry.cc
        common/telemetry.h)

if(WIN32)
    target_sources(plugin-ep-utils
        PUBLIC
            common/platform/windows/dynamic_library.cc
            common/platform/windows/env_var.cc
            common/platform/windows/env_var.h
            common/platform/windows/telemetry.cc)
else()
    target_sources(plugin-ep-utils
        PUBLIC
            common/platform/linux/dynamic_library.cc
            common/platform/linux/env_var.cc
            common/platform/linux/env_var.h
            common/platform/linux/telemetry.cc)
endif()

target_link_libraries(plugin-ep-utils PUBLIC onnxruntime::onnxruntime
        fmt::fmt Microsoft.GSL::GSL range-v3::range-v3 onnx protobuf::libprotobuf)

target_include_directories(plugin-ep-utils PUBLIC
        ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_CURRENT_LIST_DIR}/common)

if(USE_HIPDNN OR USE_MIGRAPHX OR USE_ROCM)

    add_library(hip-shared STATIC
        hip/allocator.cc
        hip/allocator.h
        hip/data_transfer.cc
        hip/data_transfer.h
        hip/stream_support.cc
        hip/stream_support.h
        hip/utils.cc
        hip/utils.h)

    find_package(hip REQUIRED)
    target_link_libraries(hip-shared PUBLIC plugin-ep-utils hip::host)
    if(WIN32)
        get_property(_amdhip64_location TARGET hip::amdhip64 PROPERTY IMPORTED_LOCATION_RELEASE)
        cmake_path(GET _amdhip64_location PARENT_PATH _hip_path)
        foreach(_file amdhip64_7.dll amd_comgr_3.dll amd_comgr0701.dll amd_comgr0702.dll hiprtc0701.dll hiprtc0702.dll hiprtc-builtins0701.dll hiprtc-builtins0702.dll)
            set(_source "${_hip_path}/${_file}")
            if (EXISTS "${_source}")
                list(APPEND _hip_files "${_source}")
            endif()
        endforeach()
        set(CACHE{HIP_FILES} TYPE INTERNAL VALUE ${_hip_files})
    endif()

endif()
