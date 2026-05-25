// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include "common/dynamic_library.h"

Ort::Status LoadDynamicLibrary(const PathString& path, void** handle) {
    if (handle == nullptr) {
        return MAKE_STATUS(ORT_INVALID_ARGUMENT);
    }
    *handle = LoadLibraryExW(std::filesystem::path{path}.native().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (*handle == nullptr) {
        return MAKE_STATUS(ORT_FAIL, "LoadDynamicLibrary(): failed to load library");
    }
    return STATUS_OK;
}

Ort::Status UnloadDynamicLibrary(void* handle) {
    if (handle == nullptr) {
        return MAKE_STATUS(ORT_INVALID_ARGUMENT);
    }
    if (::FreeLibrary(static_cast<HMODULE>(handle)) == 0) {
        const auto error_code = GetLastError();
        return MAKE_STATUS(ORT_FAIL, "FreeLibrary(): failed to unload library (",
            error_code, ": ", std::system_category().message(error_code), ")");
    }
    return STATUS_OK;
}

Ort::Status GetSymbolFromLibrary(void* handle, std::string_view name, void** symbol) {
    if (symbol == nullptr || handle == nullptr || name.empty()) {
        return MAKE_STATUS(ORT_INVALID_ARGUMENT);
    }
    *symbol = ::GetProcAddress(static_cast<HMODULE>(handle), std::string{name}.c_str());
    if (*symbol == nullptr) {
        const auto error_code{GetLastError()};
        constexpr DWORD bufferLength{128 * 1024};
        std::string s(bufferLength, '\0');
        FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr,
            error_code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), s.data(), 0, nullptr);
        return MAKE_STATUS(ORT_FAIL, "Failed to find symbol '", name, "' in library, error code: ",
            error_code, " \"", s, "\"");
    }
    return STATUS_OK;
}