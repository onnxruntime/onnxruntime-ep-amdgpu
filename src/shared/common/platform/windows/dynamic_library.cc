// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include "common/dynamic_library.h"

namespace {

// Relative names resolve against the directory of the module this code is linked into, so
// sibling DLLs load regardless of the host's search order (e.g. after
// SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS), which drops the module's own folder).
std::filesystem::path ResolveNextToThisModule(const std::filesystem::path& path) {
    if (path.is_absolute()) {
        return path;
    }
    HMODULE self{};
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&ResolveNextToThisModule), &self)) {
        return path;
    }
    std::wstring module_path(MAX_PATH, L'\0');
    for (;;) {
        const DWORD len{GetModuleFileNameW(self, module_path.data(), static_cast<DWORD>(module_path.size()))};
        if (len == 0) {
            return path;
        }
        if (len < module_path.size()) {
            module_path.resize(len);
            break;
        }
        module_path.resize(module_path.size() * 2);
    }
    return std::filesystem::path{module_path}.parent_path() / path;
}

}  // namespace

Ort::Status LoadDynamicLibrary(const PathString& path, void** handle) {
    if (handle == nullptr) {
        return MAKE_STATUS(ORT_INVALID_ARGUMENT);
    }
    *handle = LoadLibraryExW(ResolveNextToThisModule(path).native().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
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