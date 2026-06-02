// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>

#include "path_string.h"
#include "common/plugin_ep_utils.h"

#ifdef _WIN32
#define LIBRARY_PREFIX
#define LIBRARY_SUFFIX ORT_TSTR(".dll")
#else
#define LIBRARY_PREFIX ORT_TSTR("lib")
#define LIBRARY_SUFFIX ORT_TSTR(".so")
#endif

Ort::Status LoadDynamicLibrary(const PathString& path, void** handle);
Ort::Status UnloadDynamicLibrary(void* handle);
Ort::Status GetSymbolFromLibrary(void* handle, std::string_view name, void** symbol);