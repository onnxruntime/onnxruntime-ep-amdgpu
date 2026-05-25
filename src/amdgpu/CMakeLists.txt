# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(amdgpu-ep MODULE
        gpu_allocator.cc
        gpu_allocator.h
        gpu_data_transfer.cc
        gpu_data_transfer.h
        gpu_ep.cc
        gpu_ep.h
        gpu_ep_ctx.cc
        gpu_ep_ctx.h
        gpu_factory.cc
        gpu_factory.h
        gpu_info.cc
        gpu_info.h
        gpu_interop.cc
        gpu_interop.h       
        gpu_stream_support.cc
        gpu_stream_support.h)

target_link_libraries(amdgpu-ep PRIVATE plugin-ep-utils flatbuffers)

target_include_directories(amdgpu-ep PRIVATE
        $<TARGET_PROPERTY:migraphx-ep,INTERFACE_INCLUDE_DIRECTORIES>)

target_include_directories(amdgpu-ep PRIVATE
        $<TARGET_PROPERTY:directml-ep,INTERFACE_INCLUDE_DIRECTORIES>)

if(WIN32)
    set_property(TARGET amdgpu-ep APPEND_STRING
            PROPERTY LINK_FLAGS /DEF:${PROJECT_SOURCE_DIR}/src/shared/symbols.def)
    install(TARGETS amdgpu-ep
            DESTINATION ".")
    install(FILES $<TARGET_PDB_FILE:amdgpu-ep>
            DESTINATION "." OPTIONAL)

else()
    set_property(TARGET amdgpu-ep APPEND_STRING
            PROPERTY LINK_FLAGS "-Xlinker --version-script=${PROJECT_SOURCE_DIR}/src/shared/version_script.lds -Xlinker --gc-sections")
    install(TARGETS amdgpu-ep
            DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

add_version_info(
        TARGET amdgpu-ep
        DESCRIPTION "AMD GPU Execution Provider - ONNX Runtime plugin"
        PRODUCT "GPU Plugin EP")
