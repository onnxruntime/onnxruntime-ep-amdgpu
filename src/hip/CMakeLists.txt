# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(hip-ep MODULE hip_factory.cc)

target_link_libraries(hip-ep PRIVATE plugin-ep-utils)

if(WIN32)
    set_property(TARGET hip-ep APPEND_STRING
            PROPERTY LINK_FLAGS /DEF:${PROJECT_SOURCE_DIR}/src/shared/symbols.def)

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

set_target_properties(hip-ep PROPERTIES OUTPUT_NAME "hip-backend")

add_version_info(
        TARGET hip-ep
        DESCRIPTION "HIP Execution Provider - ONNX Runtime plugin"
        PRODUCT "HIP Plugin EP")
