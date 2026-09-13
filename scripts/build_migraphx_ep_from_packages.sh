#!/bin/bash
set -e

ulimit -c unlimited

# ---------------------------------------------------------------------------
# This is a variant of build_migraphx_ep_standalone.sh that AVOIDS building
# AMDMIGraphX and ONNX Runtime from source. Instead it:
#   1. Installs a prebuilt MIGraphX wheel from an AMD pip index.
#   2. Downloads a prebuilt ONNX Runtime C/C++ release tarball ("ORT core":
#      just the plain headers + libonnxruntime.so, no MIGraphX EP baked in).
#   3. Builds onnxruntime-ep-amdgpu (the out-of-tree MIGraphX EP) from source,
#      pointing it at the two prebuilt packages above via --migraphx_home
#      and --onnxrt_home.
#
# This assumes ROCm itself is already installed on the system/image at
# $ROCM_PATH (this script does not install ROCm).
# ---------------------------------------------------------------------------

# Configuration (override via env vars or CLI flags below)
# ---------------------------------------------------------------------------
# $HOME can be unset/empty in some docker/root shells. Bash's "~" expansion
# only falls back to the passwd entry when HOME is truly unset (not just
# empty), so unset it first before re-expanding.
if [ -z "$HOME" ]; then
    unset HOME
    HOME=~
fi

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
ONNXRUNTIME_EP_SRC="${ONNXRUNTIME_EP_SRC:-/workspace/onnxruntime-ep-amdgpu}"

# The MIGraphX pip wheel only ships the runtime .so files + Python bindings;
# it does NOT ship the public C API headers (migraphx.h/migraphx.hpp) or a
# CMake package config. We still need those two small headers to build
# against libmigraphx_c.so, so we grab them (and only them - no building)
# from a checkout of the AMDMIGraphX repo. Point this at an existing
# checkout to avoid a fresh clone (e.g. the one mounted into the container).
AMDMIGRAPHX_SRC="${AMDMIGRAPHX_SRC:-/workspace/AMDMIGraphX}"

AMDMIGRAPHX_INSTALL="${AMDMIGRAPHX_INSTALL:-$HOME/develop/AMDMIGraphX_install}"
ONNXRUNTIME_INSTALL="${ONNXRUNTIME_INSTALL:-$HOME/develop/onnxruntime_install}"

# MIGraphX wheel install
MIGRAPHX_VERSION="${MIGRAPHX_VERSION:-2.17.0+rocm10.0.0}"
MIGRAPHX_WHL_INDEX="${MIGRAPHX_WHL_INDEX:-https://stable.repo.amd.com/rocm/migraphx/whl-next/}"

# ONNX Runtime prebuilt release tarball (plain CPU build; the MIGraphX EP is
# provided by onnxruntime-ep-amdgpu, not by this package).
ONNXRUNTIME_VERSION="${ONNXRUNTIME_VERSION:-1.29.0}"
ONNXRUNTIME_RELEASE_URL="${ONNXRUNTIME_RELEASE_URL:-https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz}"

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --rocm-path PATH             Path to the (already installed) ROCm install (default: $ROCM_PATH)
  --migraphx-install PATH      Where to place/find the MIGraphX install prefix (default: $AMDMIGRAPHX_INSTALL)
  --onnxrt-install PATH        Where to place/find the ONNX Runtime install prefix (default: $ONNXRUNTIME_INSTALL)
  --migraphx-version VER       MIGraphX pip package version to install, e.g. "2.17.0+rocm10.0.0" (default: $MIGRAPHX_VERSION)
  --migraphx-whl-index URL     Pip index URL to install the MIGraphX wheel from (default: $MIGRAPHX_WHL_INDEX)
  --onnxrt-version VER         ONNX Runtime release version tag to download, e.g. "1.29.0" (default: $ONNXRUNTIME_VERSION)
  --onnxrt-release-url URL     Full URL of the ONNX Runtime release tarball to download
                                (default derived from --onnxrt-version: $ONNXRUNTIME_RELEASE_URL)
  -h, --help                   Show this help message

Environment variables ROCM_PATH, AMDMIGRAPHX_INSTALL, ONNXRUNTIME_INSTALL,
MIGRAPHX_VERSION, MIGRAPHX_WHL_INDEX, ONNXRUNTIME_VERSION, and
ONNXRUNTIME_RELEASE_URL can also be used to set these values; CLI flags take
precedence.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --rocm-path)
            ROCM_PATH="$2"
            shift 2
            ;;
        --migraphx-install)
            AMDMIGRAPHX_INSTALL="$2"
            shift 2
            ;;
        --onnxrt-install)
            ONNXRUNTIME_INSTALL="$2"
            shift 2
            ;;
        --migraphx-version)
            MIGRAPHX_VERSION="$2"
            shift 2
            ;;
        --migraphx-whl-index)
            MIGRAPHX_WHL_INDEX="$2"
            shift 2
            ;;
        --onnxrt-version)
            ONNXRUNTIME_VERSION="$2"
            ONNXRUNTIME_RELEASE_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz"
            shift 2
            ;;
        --onnxrt-release-url)
            ONNXRUNTIME_RELEASE_URL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

echo "ROCM_PATH=$ROCM_PATH"
echo "AMDMIGRAPHX_INSTALL=$AMDMIGRAPHX_INSTALL"
echo "ONNXRUNTIME_INSTALL=$ONNXRUNTIME_INSTALL"
echo "MIGRAPHX_VERSION=$MIGRAPHX_VERSION"
echo "MIGRAPHX_WHL_INDEX=$MIGRAPHX_WHL_INDEX"
echo "ONNXRUNTIME_RELEASE_URL=$ONNXRUNTIME_RELEASE_URL"

git config --global --add safe.directory "*"

# ---------------------------------------------------------------------------
# 0. System + Python dependencies
# ---------------------------------------------------------------------------
apt-get update -y
apt-get install -y curl zip unzip tar patchelf git

pip3 install --upgrade ninja "packaging>=24.2" cmake==4.2.3

# Add newer cmake to the path
export PATH="/opt/cmake/bin:$PATH"
export CXXFLAGS="-D__HIP_PLATFORM_AMD__=1 -w"

# ---------------------------------------------------------------------------
# 1. Pull in a prebuilt MIGraphX wheel (instead of building AMDMIGraphX), and
#    assemble a synthetic CMake-findable install prefix around it.
#
# The "migraphx" + "migraphx-libs" pip wheels only contain:
#   - migraphx/migraphx.cpython-*.so        (Python bindings)
#   - migraphx_libs/libmigraphx*.so*         (the actual C/C++ runtime libs)
# They do NOT contain the public C API headers or a migraphx-config.cmake,
# so find_package(migraphx) can't be pointed at the wheel directly. We
# reconstruct the minimal pieces onnxruntime-ep-amdgpu's CMake needs
# (an imported target "migraphx::c") ourselves:
#   - headers: migraphx.h/migraphx.hpp copied from the AMDMIGraphX repo's
#     public C API (small, static files - no build required), plus
#     hand-written config.h/version.h/api/export.h that would otherwise be
#     generated by MIGraphX's own CMake build.
#   - libs: symlinked in from the migraphx-libs wheel install.
#   - migraphx-config.cmake: hand-written, defining migraphx::c pointing at
#     the above.
# ---------------------------------------------------------------------------
rm -rf "$AMDMIGRAPHX_INSTALL"
mkdir -p "$AMDMIGRAPHX_INSTALL/include/migraphx/api" \
         "$AMDMIGRAPHX_INSTALL/lib/cmake/migraphx"

python3 -m pip install --no-cache-dir \
    --index-url "$MIGRAPHX_WHL_INDEX" \
    "migraphx==${MIGRAPHX_VERSION}"

MIGRAPHX_LIBS_DIR="$(python3 -c '
import pathlib, site
roots = set(site.getsitepackages())
try:
    roots.add(site.getusersitepackages())
except Exception:
    pass
for root in roots:
    cand = pathlib.Path(root) / "migraphx_libs"
    if cand.is_dir():
        print(cand)
        break
')"

if [ -z "$MIGRAPHX_LIBS_DIR" ]; then
    echo "ERROR: could not find the migraphx_libs package directory after pip install." >&2
    exit 1
fi
echo "Using MIGraphX runtime libs from: $MIGRAPHX_LIBS_DIR"

# Symlink every runtime lib (skip the Python-only bindings) into our lib dir.
for f in "$MIGRAPHX_LIBS_DIR"/*.so*; do
    base="$(basename "$f")"
    case "$base" in
        libmigraphx_py*.so) continue ;;
    esac
    ln -sf "$f" "$AMDMIGRAPHX_INSTALL/lib/$base"
done
# Unversioned alias so IMPORTED_SONAME / plain -lmigraphx_c lookups work too.
ln -sf libmigraphx_c.so.3 "$AMDMIGRAPHX_INSTALL/lib/libmigraphx_c.so"

# Public C API headers (not shipped in the wheel) - copy the two static
# header files as-is from the AMDMIGraphX repo checkout, no build needed.
MIGRAPHX_HDR_DIR="$AMDMIGRAPHX_SRC/src/api/include/migraphx"
if [ ! -f "$MIGRAPHX_HDR_DIR/migraphx.h" ]; then
    echo "AMDMIGraphX checkout not found/incomplete at $AMDMIGRAPHX_SRC; cloning a shallow copy just for headers." >&2
    rm -rf /tmp/AMDMIGraphX-headers
    git clone --depth 1 https://github.com/ROCm/AMDMIGraphX.git /tmp/AMDMIGraphX-headers
    MIGRAPHX_HDR_DIR="/tmp/AMDMIGraphX-headers/src/api/include/migraphx"
fi
cp "$MIGRAPHX_HDR_DIR/migraphx.h" "$MIGRAPHX_HDR_DIR/migraphx.hpp" \
    "$AMDMIGRAPHX_INSTALL/include/migraphx/"

# config.h: normally generated by MIGraphX's own CMake (@cmakedefine) build;
# both ONNX and TF parsers are present in the wheel (libmigraphx_onnx.so /
# libmigraphx_tf.so), so enable both.
cat > "$AMDMIGRAPHX_INSTALL/include/migraphx/config.h" <<'EOF'
#ifndef MIGRAPHX_GUARD_CONFIG_H
#define MIGRAPHX_GUARD_CONFIG_H

#define MIGRAPHX_ENABLE_ONNX
#define MIGRAPHX_ENABLE_TENSORFLOW

#endif // MIGRAPHX_GUARD_CONFIG_H
EOF

# version.h: normally generated from src/version.h.in; fill in from the
# installed wheel version (MAJOR.MINOR.PATCH[+TWEAK]).
MIGRAPHX_VER_CORE="${MIGRAPHX_VERSION%%+*}"
MIGRAPHX_VER_TWEAK="${MIGRAPHX_VERSION#*+}"
[ "$MIGRAPHX_VER_TWEAK" = "$MIGRAPHX_VERSION" ] && MIGRAPHX_VER_TWEAK=""
MIGRAPHX_VER_MAJOR="$(echo "$MIGRAPHX_VER_CORE" | cut -d. -f1)"
MIGRAPHX_VER_MINOR="$(echo "$MIGRAPHX_VER_CORE" | cut -d. -f2)"
MIGRAPHX_VER_PATCH="$(echo "$MIGRAPHX_VER_CORE" | cut -d. -f3)"
cat > "$AMDMIGRAPHX_INSTALL/include/migraphx/version.h" <<EOF
// clang-format off
#define MIGRAPHX_VERSION_MAJOR ${MIGRAPHX_VER_MAJOR}
#define MIGRAPHX_VERSION_MINOR ${MIGRAPHX_VER_MINOR}
#define MIGRAPHX_VERSION_PATCH ${MIGRAPHX_VER_PATCH}
#define MIGRAPHX_VERSION_TWEAK "${MIGRAPHX_VER_TWEAK}"
#define MIGRAPHX_SO_MAJOR_VERSION           \\
    ${MIGRAPHX_VER_MAJOR} * 1000 * 1000 + \\
    ${MIGRAPHX_VER_MINOR} * 1000 +        \\
    ${MIGRAPHX_VER_PATCH}
// clang-format on
EOF

# api/export.h: normally generated by CMake's generate_export_header() for
# the migraphx_c target. The exact visibility attributes don't matter for a
# *consumer* of the already-built libmigraphx_c.so, so a minimal but valid
# definition is sufficient.
cat > "$AMDMIGRAPHX_INSTALL/include/migraphx/api/export.h" <<'EOF'
#ifndef MIGRAPHX_API_EXPORT_H
#define MIGRAPHX_API_EXPORT_H

#ifdef MIGRAPHX_C_STATIC_DEFINE
#  define MIGRAPHX_C_EXPORT
#  define MIGRAPHX_C_NO_EXPORT
#else
#  ifndef MIGRAPHX_C_EXPORT
#    define MIGRAPHX_C_EXPORT __attribute__((visibility("default")))
#  endif
#  ifndef MIGRAPHX_C_NO_EXPORT
#    define MIGRAPHX_C_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef MIGRAPHX_C_DEPRECATED
#  define MIGRAPHX_C_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef MIGRAPHX_C_DEPRECATED_EXPORT
#  define MIGRAPHX_C_DEPRECATED_EXPORT MIGRAPHX_C_EXPORT MIGRAPHX_C_DEPRECATED
#endif

#ifndef MIGRAPHX_C_DEPRECATED_NO_EXPORT
#  define MIGRAPHX_C_DEPRECATED_NO_EXPORT MIGRAPHX_C_NO_EXPORT MIGRAPHX_C_DEPRECATED
#endif

#endif /* MIGRAPHX_API_EXPORT_H */
EOF

# Hand-written migraphx-config.cmake exposing the migraphx::c imported
# target that onnxruntime-ep-amdgpu's src/migraphx/CMakeLists.txt links
# against.
cat > "$AMDMIGRAPHX_INSTALL/lib/cmake/migraphx/migraphx-config.cmake" <<EOF
if(NOT TARGET migraphx::c)
    add_library(migraphx::c SHARED IMPORTED)
    set_target_properties(migraphx::c PROPERTIES
        IMPORTED_LOCATION "$AMDMIGRAPHX_INSTALL/lib/libmigraphx_c.so.3"
        IMPORTED_SONAME "libmigraphx_c.so.3"
        INTERFACE_INCLUDE_DIRECTORIES "$AMDMIGRAPHX_INSTALL/include"
    )
endif()
EOF

# ---------------------------------------------------------------------------
# 2. Pull in a prebuilt ONNX Runtime core release (instead of building
#    ROCm/onnxruntime from source). This is the plain ONNX Runtime C/C++
#    distributable (include/ + lib/), not a ROCm-specific build; the
#    MIGraphX EP itself comes from onnxruntime-ep-amdgpu below.
# ---------------------------------------------------------------------------
rm -rf "$ONNXRUNTIME_INSTALL"
mkdir -p "$ONNXRUNTIME_INSTALL"

ORT_TARBALL="/tmp/onnxruntime-core.tgz"
curl -fL "$ONNXRUNTIME_RELEASE_URL" -o "$ORT_TARBALL"

ORT_EXTRACT_DIR="$(mktemp -d)"
tar -xzf "$ORT_TARBALL" -C "$ORT_EXTRACT_DIR"

# The release tarball extracts to a single top-level dir
# (e.g. onnxruntime-linux-x64-1.29.0/{include,lib}); flatten that into
# ONNXRUNTIME_INSTALL so it matches the layout build.py expects.
ORT_TOP_DIR="$(find "$ORT_EXTRACT_DIR" -mindepth 1 -maxdepth 1 -type d | head -n1)"
cp -a "$ORT_TOP_DIR"/. "$ONNXRUNTIME_INSTALL"/
rm -rf "$ORT_EXTRACT_DIR" "$ORT_TARBALL"

# Work around a long-standing bug in the official release tarballs: the
# shipped lib/cmake/onnxruntime/onnxruntimeTargets*.cmake hardcodes
# "${_IMPORT_PREFIX}/lib64/libonnxruntime.so.*" and
# "${_IMPORT_PREFIX}/include/onnxruntime", but the tarball actually ships the
# library under plain "lib/" and the headers directly under "include/".
# See https://github.com/microsoft/onnxruntime/issues/22267 and
# https://github.com/microsoft/onnxruntime/issues/23642 (fixed upstream in
# some releases via https://github.com/microsoft/onnxruntime/pull/26104, but
# not all versions/asset variants are fixed, so patch defensively either way).
if [ -d "$ONNXRUNTIME_INSTALL/lib" ] && [ ! -e "$ONNXRUNTIME_INSTALL/lib64" ]; then
    ln -s lib "$ONNXRUNTIME_INSTALL/lib64"
fi
if [ -d "$ONNXRUNTIME_INSTALL/include" ] && [ ! -e "$ONNXRUNTIME_INSTALL/include/onnxruntime" ]; then
    mkdir -p "$ONNXRUNTIME_INSTALL/include/onnxruntime"
    for hdr in "$ONNXRUNTIME_INSTALL"/include/*.h; do
        [ -e "$hdr" ] || continue
        ln -sf "../$(basename "$hdr")" "$ONNXRUNTIME_INSTALL/include/onnxruntime/$(basename "$hdr")"
    done
fi

# Also install the matching Python wheel, best-effort, for convenience when
# running Python-side ONNX Runtime scripts against this environment.
pip3 install --no-cache-dir "onnxruntime==${ONNXRUNTIME_VERSION}" || \
    echo "WARNING: pip install of onnxruntime==${ONNXRUNTIME_VERSION} failed; continuing." >&2

# ---------------------------------------------------------------------------
# 3. Build onnxruntime-ep-amdgpu (main branch) against the packages above
# ---------------------------------------------------------------------------
cd "$ONNXRUNTIME_EP_SRC"
git checkout main

./build.sh --config Release \
    --cmake_generator Ninja \
    --onnxrt_home "$ONNXRUNTIME_INSTALL" \
    --use_migraphx \
    --migraphx_home "$AMDMIGRAPHX_INSTALL" \
    --compile_no_warning_as_error \
    --parallel $(nproc) \
    --build_dir build.EP.MGX \
    --hip_path "$ROCM_PATH" \
    --build_wheel
