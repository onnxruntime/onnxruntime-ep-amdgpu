# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(hip-ep MODULE hip_factory.cc)

target_link_libraries(hip-ep PRIVATE plugin-ep-utils)

if(WIN32)
    # /DEF: is an MSVC linker flag: pass it bare under the MSVC-style driver
    # (cl.exe / clang-cl), but forward it through a GNU-style driver (e.g. ROCm
    # clang++) via -Wl, or it is mistaken for an input file.
    if(MSVC)
        set(_ep_def_flag /DEF:${PROJECT_SOURCE_DIR}/src/shared/symbols.def)
    else()
        set(_ep_def_flag -Wl,/DEF:${PROJECT_SOURCE_DIR}/src/shared/symbols.def)
    endif()
    set_property(TARGET hip-ep APPEND_STRING
            PROPERTY LINK_FLAGS ${_ep_def_flag})

    install(TARGETS hip-ep
            DESTINATION ".")

    install(FILES $<TARGET_PDB_FILE:hip-ep>
            DESTINATION "." OPTIONAL)
else()
    set_property(TARGET hip-ep APPEND_STRING
            PROPERTY LINK_FLAGS "-Xlinker --version-script=${PROJECT_SOURCE_DIR}/src/shared/version_script.lds -Xlinker --gc-sections")

    install(TARGETS hip-ep
            DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()


if (USE_AMDGPU)
    set_target_properties(hip-ep PROPERTIES OUTPUT_NAME "hip-backend")
endif ()

add_version_info(
        TARGET hip-ep
        DESCRIPTION "HIP Execution Provider - ONNX Runtime plugin"
        PRODUCT "HIP Plugin EP")
