# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

function(add_version_info)
    # Only process on Windows
    if(NOT WIN32)
        return()
    endif()

    # Parse arguments
    set(options "")
    set(oneValueArgs TARGET DESCRIPTION FILENAME PRODUCT)
    set(multiValueArgs "")
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Validate required arguments
    if(NOT ARG_TARGET)
        message(FATAL_ERROR "add_version_resource: TARGET argument is required")
    endif()

    if(NOT ARG_DESCRIPTION)
        message(FATAL_ERROR "add_version_resource: DESCRIPTION argument is required for ${ARG_TARGET}")
    endif()

    # Get target properties
    get_target_property(TARGET_TYPE ${ARG_TARGET} TYPE)

    # Set component-specific variables
    set(COMPONENT_NAME ${ARG_TARGET})
    set(COMPONENT_DESCRIPTION "${ARG_DESCRIPTION}")
    set(PRODUCT_NAME ${ARG_PRODUCT})

    # Determine output filename
    if(ARG_FILENAME)
        set(COMPONENT_FILENAME "${ARG_FILENAME}")
    else()
        # Get the actual output name of the target
        get_target_property(OUTPUT_NAME ${ARG_TARGET} OUTPUT_NAME)
        if(OUTPUT_NAME)
            set(COMPONENT_FILENAME "${OUTPUT_NAME}")
        else()
            set(COMPONENT_FILENAME "${ARG_TARGET}")
        endif()

        # Add appropriate extension
        if(TARGET_TYPE STREQUAL "SHARED_LIBRARY")
            set(COMPONENT_FILENAME "${COMPONENT_FILENAME}.dll")
        elseif(TARGET_TYPE STREQUAL "EXECUTABLE")
            set(COMPONENT_FILENAME "${COMPONENT_FILENAME}.exe")
        endif()
    endif()

    # Set DLL_BUILD flag for resource compiler
    if(TARGET_TYPE STREQUAL "SHARED_LIBRARY")
        set(DLL_BUILD_FLAG "-DDLL_BUILD")
    else()
        set(DLL_BUILD_FLAG "")
    endif()

    # Configure the version resource file
    set(RC_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${ARG_TARGET}_version.rc")
    configure_file(
        "${PROJECT_SOURCE_DIR}/cmake/version.rc.in"
        "${RC_OUTPUT}"
        @ONLY
    )

    # Add the resource file to the target
    target_sources(${ARG_TARGET} PRIVATE ${RC_OUTPUT})

    # Set resource compiler flags if needed
    if(DLL_BUILD_FLAG)
        set_source_files_properties(${RC_OUTPUT} PROPERTIES
            COMPILE_FLAGS ${DLL_BUILD_FLAG}
        )
    endif()

    message(STATUS "Added version info to ${ARG_TARGET}: ${COMPONENT_DESCRIPTION}")
endfunction()
