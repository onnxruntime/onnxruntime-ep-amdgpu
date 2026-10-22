# Idempotently apply patches/onnx/onnx.patch to the fetched onnx source.
#
# Invoked as the onnx FetchContent PATCH_COMMAND via `cmake -P`. A bare
# `patch -N --forward` returns a non-zero exit when the patch is already applied
# (e.g. on a CMake re-configure that re-runs the patch step without re-fetching),
# which would fail the build. Guard with a reverse dry-run: if the patch already
# reverses cleanly it is already applied, so skip; otherwise apply it. Both paths
# exit 0, so re-configuring is safe.
#
# Required variables (passed with -D):
#   PATCH_EXECUTABLE - path to the patch tool (from find_package(Patch))
#   PATCH_FILE       - absolute path to onnx.patch
#
# FetchContent runs PATCH_COMMAND with the working directory already set to the
# fetched onnx source dir, so the patch (-p1) resolves relative to CMAKE_SOURCE_DIR
# here (which under `cmake -P` is the process working directory).

execute_process(
    COMMAND ${PATCH_EXECUTABLE} -p1 -R --dry-run -f --input=${PATCH_FILE}
    RESULT_VARIABLE _already_applied
    OUTPUT_QUIET ERROR_QUIET)

if(_already_applied EQUAL 0)
    message(STATUS "onnx.patch already applied; skipping")
    return()
endif()

execute_process(
    COMMAND ${PATCH_EXECUTABLE} -p1 -N --input=${PATCH_FILE}
    RESULT_VARIABLE _apply_result)

if(NOT _apply_result EQUAL 0)
    message(FATAL_ERROR "Failed to apply onnx.patch (exit ${_apply_result})")
endif()

message(STATUS "Applied onnx.patch")
