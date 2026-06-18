# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

find_package(migraphx REQUIRED)

add_library(migraphx-ep MODULE
        kernels/kernel.h
        kernels/memcpy.cc
        kernels/memcpy.h
        mgx_ep.cc
        mgx_ep.h
        mgx_ep_ctx.cc
        mgx_ep_ctx.h
        mgx_factory.cc
        mgx_factory.h
        mgx_info.cc
        mgx_info.h
        mgx_interop.cc
        mgx_interop.h
        mgx_kernel_reg.cc
        mgx_kernel_reg.h
        mgx_options.h
        mgx_utils.cc
        mgx_utils.h
        ort_trt_int8_cal_table.fbs.h)

target_link_libraries(migraphx-ep PRIVATE migraphx::c hip-shared flatbuffers)

if(WIN32)
    set_property(TARGET migraphx-ep APPEND_STRING
            PROPERTY LINK_FLAGS /DEF:${PROJECT_SOURCE_DIR}/src/shared/symbols.def)

    set(MGX_FILES
        "$<TARGET_FILE_DIR:migraphx::migraphx>/amdmlss.dll"
        "$<TARGET_FILE_DIR:migraphx::migraphx>/amdmlss.pdb"
        "$<TARGET_FILE_DIR:migraphx::migraphx>/migraphx.dll"
        "$<TARGET_FILE_DIR:migraphx::migraphx>/migraphx.pdb"
        "$<TARGET_FILE_DIR:migraphx::migraphx>/migraphx-hiprtc-driver.exe"
        "$<TARGET_FILE_DIR:migraphx::migraphx>/migraphx-hiprtc-driver.pdb"
        "$<TARGET_FILE_DIR:migraphx::migraphx>/migraphx_c.dll"
        "$<TARGET_FILE_DIR:migraphx::migraphx>/migraphx_c.pdb"
        "$<TARGET_FILE_DIR:migraphx::migraphx>/migraphx_device.dll"
        "$<TARGET_FILE_DIR:migraphx::migraphx>/migraphx_device.pdb"
        "$<TARGET_FILE_DIR:migraphx::migraphx>/migraphx_gpu.dll"
        "$<TARGET_FILE_DIR:migraphx::migraphx>/migraphx_gpu.pdb"
        "$<TARGET_FILE_DIR:migraphx::migraphx>/migraphx_onnx.dll"
        "$<TARGET_FILE_DIR:migraphx::migraphx>/migraphx_onnx.pdb"
        "$<TARGET_FILE_DIR:migraphx::migraphx>/migraphx_tf.dll"
        "$<TARGET_FILE_DIR:migraphx::migraphx>/migraphx_tf.pdb")

    foreach(_file ${MGX_FILES} ${HIP_FILES})
        add_custom_command(TARGET migraphx-ep
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/CopyIfExists.cmake "${_file}" "$<TARGET_FILE_DIR:migraphx-ep>")
        cmake_path(GET _file FILENAME _name)
        set(_install_file "$<TARGET_FILE_DIR:migraphx-ep>/${_name}")
        list(APPEND _install_files "${_install_file}")
    endforeach ()

    install(TARGETS migraphx-ep
            DESTINATION ".")

    install(FILES $<TARGET_PDB_FILE:migraphx-ep> ${_install_files}
            DESTINATION "." OPTIONAL)
else()
    set_property(TARGET migraphx-ep APPEND_STRING
            PROPERTY LINK_FLAGS "-Xlinker --version-script=${PROJECT_SOURCE_DIR}/src/shared/version_script.lds -Xlinker --gc-sections")

    install(TARGETS migraphx-ep
            DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(USE_AMDGPU)
    set_target_properties(migraphx-ep PROPERTIES OUTPUT_NAME "migraphx-backend")
    target_include_directories(migraphx-ep INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

add_version_info(
        TARGET migraphx-ep
        DESCRIPTION "MIGraphX Execution Provider - ONNX Runtime plugin"
        PRODUCT "MIGraphX Plugin EP")
