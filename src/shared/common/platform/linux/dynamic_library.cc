// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/dynamic_library.h"

#include <dlfcn.h>

#include <cstring>
#include <string>

Ort::Status LoadDynamicLibrary(const PathString& path, void** handle) {
    if (handle == nullptr) {
        return MAKE_STATUS(ORT_INVALID_ARGUMENT);
    }
    *handle = ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (*handle == nullptr) {
        const char* const err = ::dlerror();
        return MAKE_STATUS(ORT_FAIL, "LoadDynamicLibrary(): failed to load library: ",
            err != nullptr ? err : "unknown error");
    }
    return STATUS_OK;
}

Ort::Status UnloadDynamicLibrary(void* handle) {
    if (handle == nullptr) {
        return MAKE_STATUS(ORT_INVALID_ARGUMENT);
    }
    if (::dlclose(handle) != 0) {
        const char* const err = ::dlerror();
        return MAKE_STATUS(ORT_FAIL, "dlclose(): failed to unload library: ",
            err != nullptr ? err : "unknown error");
    }
    return STATUS_OK;
}

Ort::Status GetSymbolFromLibrary(void* handle, std::string_view name, void** symbol) {
    if (symbol == nullptr || handle == nullptr || name.empty()) {
        return MAKE_STATUS(ORT_INVALID_ARGUMENT);
    }
    ::dlerror();  // clear any stale error
    *symbol = ::dlsym(handle, std::string{name}.c_str());
    const char* const err = ::dlerror();
    if (err != nullptr) {
        return MAKE_STATUS(ORT_FAIL, "Failed to find symbol '", name, "' in library: ", err);
    }
    if (*symbol == nullptr) {
        return MAKE_STATUS(ORT_FAIL, "Failed to find symbol '", name, "' in library");
    }
    return STATUS_OK;
}
