#!/bin/bash
set -e

ulimit -c unlimited

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
AMDMIGRAPHX_SRC="${AMDMIGRAPHX_SRC:-/workspace/AMDMIGraphX}"
ONNXRUNTIME_SRC="${ONNXRUNTIME_SRC:-/onnxruntime}"
ONNXRUNTIME_EP_SRC="${ONNXRUNTIME_EP_SRC:-/workspace/onnxruntime-ep-amdgpu}"

AMDMIGRAPHX_INSTALL="${AMDMIGRAPHX_INSTALL:-$HOME/develop/AMDMIGraphX_install}"
ONNXRUNTIME_INSTALL="${ONNXRUNTIME_INSTALL:-$HOME/develop/onnxruntime_install}"

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --rocm-path PATH           Path to the ROCm install (default: $ROCM_PATH)
  --migraphx-install PATH    AMDMIGraphX install prefix (default: $AMDMIGRAPHX_INSTALL)
  --onnxrt-install PATH      ONNX Runtime install prefix (default: $ONNXRUNTIME_INSTALL)
  -h, --help                 Show this help message

Environment variables ROCM_PATH, AMDMIGRAPHX_INSTALL, and ONNXRUNTIME_INSTALL
can also be used to set these values; CLI flags take precedence.
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

git config --global --add safe.directory "*"


# ---------------------------------------------------------------------------
# 0. System + Python dependencies
# ---------------------------------------------------------------------------
apt-get update -y
apt-get install -y curl zip unzip tar

pip3 install --upgrade ninja "packaging>=24.2" cmake==4.2.3
pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz

# Add newer cmake to the path
export PATH="/opt/cmake/bin:$PATH"
export CXXFLAGS="-D__HIP_PLATFORM_AMD__=1 -w"

# ---------------------------------------------------------------------------
# 1. Build + install AMDMIGraphX (develop branch)
# ---------------------------------------------------------------------------
cd "$AMDMIGRAPHX_SRC"
git checkout develop

GPU_TARGETS="$("$ROCM_PATH/bin/rocminfo" | grep -o -m1 'gfx.*')"

rbuild build -d depend -B build \
    -DGPU_TARGETS="$GPU_TARGETS" \
    -DMIGRAPHX_USE_COMPOSABLEKERNEL=OFF

cd build
cmake --install . --prefix "$AMDMIGRAPHX_INSTALL"

# ---------------------------------------------------------------------------
# 2. Build + install ROCm/onnxruntime (v1.24.2 tag)
# ---------------------------------------------------------------------------
cd "$ONNXRUNTIME_SRC"
git checkout v1.29.0
#vendorID fix seen in OnnxRT core that breaks AMDGPUs needs to be upstreamed

./build.sh --config Release \
    --cmake_generator Ninja \
    --cmake_extra_defines HIP_PLATFORM="amd" onnxruntime_USE_COMPOSABLE_KERNEL=OFF \
    --skip_tests \
    --build_wheel \
    --build_shared_lib \
    --parallel \
    --use_binskim_compliant_compile_flags \
    --build_csharp \
    --use_vcpkg \
    --use_vcpkg_ms_internal_asset_cache \
    --client_package_build \
    --allow_running_as_root \
    --enable_pybind \
    --cmake_extra_defines CMAKE_IGNORE_PREFIX_PATH=/usr/local

cd build/Linux/Release
pip3 install dist/*.whl
cmake --install . --prefix "$ONNXRUNTIME_INSTALL"

# ---------------------------------------------------------------------------
# 3. Build onnxruntime-ep-amdgpu (main branch)
# --------------------------------------------------------------------------128-
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
