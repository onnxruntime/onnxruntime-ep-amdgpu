:: Copyright (c) Advanced Micro Devices, Inc.
::
:: SPDX-License-Identifier: MIT

@echo off
python "%~dp0\tools\ci_build\build.py" --build_dir "%~dp0\build" %*
