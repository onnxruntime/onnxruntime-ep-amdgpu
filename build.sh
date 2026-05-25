#!/bin/sh
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Get the path of the script (may be relative)
SRC="$0"
case "$SRC" in
  /*) ;;
  *) SRC="$(pwd)/$SRC";
esac
DIR=$(dirname "$SRC")
DIR=$(cd "$DIR" 2>/dev/null && pwd)

OS="$(uname -s)"
if [ "$OS" != "Linux" ]; then
  echo "Only Linux supported" >&2
  exit 1
fi

python3 $DIR/tools/ci_build/build.py --build_dir $DIR/build/ "$@"