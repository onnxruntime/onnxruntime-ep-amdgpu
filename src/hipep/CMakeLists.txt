# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(hipep-ep MODULE hipep_factory.cc)

target_link_libraries(hipep-ep PRIVATE plugin-ep-utils)

if(WIN32)
    set_property(TARGET hipep-ep APPEND_STRING
            PROPERTY LINK_FLAGS /DEF:${PROJECT_SOURCE_DIR}/src/shared/symbols.def)

    install(TARGETS hipep-ep
            DESTINATION ".")

    install(FILES $<TARGET_PDB_FILE:hipep-ep>
            DESTINATION "." OPTIONAL)
else()
    set_property(TARGET hipep-ep APPEND_STRING
            PROPERTY LINK_FLAGS "-Xlinker --version-script=${PROJECT_SOURCE_DIR}/src/shared/version_script.lds -Xlinker --gc-sections")

    install(TARGETS hipep-ep
            DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

set_target_properties(hipep-ep PROPERTIES OUTPUT_NAME "hipep-backend")

add_version_info(
        TARGET hipep-ep
        DESCRIPTION "HIP Execution Provider - ONNX Runtime plugin"
        PRODUCT "HIP Plugin EP")
