# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

find_package(migraphx REQUIRED)

add_library(migraphx-ep MODULE
        kernels/kernel.h
        kernels/memcpy.cc
        kernels/memcpy.h
        mgx_dynamic_batch.cc
        mgx_dynamic_batch.h
        mgx_ep.cc
        mgx_ep.h
        mgx_ep_ctx.cc
        mgx_ep_ctx.h
        mgx_factory.cc
        mgx_factory.h
        mgx_hip_graph.cc
        mgx_hip_graph.h
        mgx_info.cc
        mgx_info.h
        mgx_precompile.cc
        mgx_precompile.h
        mgx_program_ops.cc
        mgx_program_ops.h
        mgx_interop.cc
        mgx_interop.h
        mgx_kernel_reg.cc
        mgx_kernel_reg.h
        mgx_options.h
        mgx_utils.cc
        mgx_utils.h
        ort_trt_int8_cal_table.fbs.h)

target_link_libraries(migraphx-ep PRIVATE migraphx::c hip-shared flatbuffers)

if(MSVC)
    target_link_options(migraphx-ep PRIVATE "/DELAYLOAD:amdhip64_7.dll")
    target_link_libraries(migraphx-ep PRIVATE delayimp)
endif()

if(WIN32)
    # Compile directly into the DLL so the linker cannot discard DllMain from
    # the plugin-ep-utils static library.
    target_sources(migraphx-ep PRIVATE
            ${PROJECT_SOURCE_DIR}/src/shared/common/platform/windows/dll_main.cc)

    set_property(TARGET migraphx-ep APPEND_STRING
            PROPERTY LINK_FLAGS /DEF:${PROJECT_SOURCE_DIR}/src/migraphx/symbols.def)

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
            PROPERTY LINK_FLAGS "-Xlinker --version-script=${PROJECT_SOURCE_DIR}/src/migraphx/version_script.lds -Xlinker --gc-sections")

    install(TARGETS migraphx-ep
            DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# Optionally ship a read-only problem cache next to the EP: deploy the file at
# MIGRAPHX_EP_PROBLEM_CACHE as problem_cache.json (the name setup_problem_cache_paths
# discovers beside the module at runtime).
set(MIGRAPHX_EP_PROBLEM_CACHE "" CACHE FILEPATH
        "Optional problem-cache JSON to deploy beside migraphx-ep as problem_cache.json")
if(MIGRAPHX_EP_PROBLEM_CACHE)
    if(NOT EXISTS "${MIGRAPHX_EP_PROBLEM_CACHE}")
        message(FATAL_ERROR "MIGRAPHX_EP_PROBLEM_CACHE not found: ${MIGRAPHX_EP_PROBLEM_CACHE}")
    endif()
    add_custom_command(TARGET migraphx-ep POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${MIGRAPHX_EP_PROBLEM_CACHE}"
                "$<TARGET_FILE_DIR:migraphx-ep>/problem_cache.json"
            COMMENT "Deploying problem_cache.json beside migraphx-ep")
    if(WIN32)
        install(FILES "${MIGRAPHX_EP_PROBLEM_CACHE}"
                DESTINATION "." RENAME "problem_cache.json")
    else()
        install(FILES "${MIGRAPHX_EP_PROBLEM_CACHE}"
                DESTINATION ${CMAKE_INSTALL_BINDIR} RENAME "problem_cache.json")
    endif()
endif()

if(USE_AMDGPU)
    set_target_properties(migraphx-ep PROPERTIES OUTPUT_NAME "migraphx-backend")
    target_include_directories(migraphx-ep INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

add_version_info(
        TARGET migraphx-ep
        DESCRIPTION "MIGraphX Execution Provider - ONNX Runtime plugin"
        PRODUCT "MIGraphX Plugin EP")
