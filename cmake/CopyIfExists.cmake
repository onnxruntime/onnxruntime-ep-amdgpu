# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(EXISTS ${CMAKE_ARGV3})
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_ARGV3}" "${CMAKE_ARGV4}")
endif()