// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <filesystem>

#include "common/dynamic_library.h"

Ort::Status LoadDynamicLibrary(const PathString& path, void** handle) {
    if (handle == nullptr) {
        return MAKE_STATUS(ORT_INVALID_ARGUMENT);
    }
    // Two Windows loader hazards when the umbrella EP is hosted by ONNX Runtime:
    //   1. ORT calls SetDefaultDllDirectories() to harden the DLL search, which makes
    //      LOAD_WITH_ALTERED_SEARCH_PATH illegal (ERROR_INVALID_PARAMETER 87).
    //   2. Backends (migraphx-backend/hip-backend) are co-deployed next to THIS module and pull in
    //      transitive deps (migraphx.dll -> migraphx_gpu.dll -> amdmlss.dll, rocm) from the SAME dir,
    //      which the hardened process search will not find.
    // Fix: resolve a bare filename to a full path in this module's directory, and load with
    // LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR so that directory also resolves the backend's transitive deps
    // (| DEFAULT_DIRS keeps application dir + System32 + AddDllDirectory dirs).
    std::filesystem::path lib{path};
    if (!lib.has_parent_path()) {
        HMODULE self_module{};
        if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCWSTR>(&LoadDynamicLibrary), &self_module) != 0) {
            wchar_t module_path[MAX_PATH]{};
            if (GetModuleFileNameW(self_module, module_path, MAX_PATH) != 0) {
                lib = std::filesystem::path{module_path}.remove_filename() / lib;
            }
        }
    }
    *handle = LoadLibraryExW(lib.native().c_str(), nullptr,
                            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
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