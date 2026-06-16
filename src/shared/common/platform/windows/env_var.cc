// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdlib>

#include "parse_string.h"
#include "platform/windows/env_var.h"

namespace platform {

std::string GetEnvironmentVar(const std::string_view name) {
    // Why getenv() should be avoided on Windows:
    // https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/getenv-wgetenv
    // Instead use the Win32 API: GetEnvironmentVariableA()

    // Max limit of an environment variable on Windows including the null-terminating character
    constexpr DWORD kBufferSize = 32767;
    std::string buffer(kBufferSize, '\0');
    const auto count{::GetEnvironmentVariable(std::string{name}.c_str(), buffer.data(), kBufferSize)};
    if (kBufferSize > count) {
        buffer.resize(count);
        return buffer;
    }
    return {};
}

void SetEnvironmentVar(const std::string_view name, const std::string_view value) {
    // MIGraphX (and other CRT consumers) read via std::getenv, which uses the CRT
    // environment block. On Windows that block is separate from the Win32 block that
    // SetEnvironmentVariable updates, so use _putenv_s to reach getenv. Set both so
    // Win32-based readers stay consistent too. An empty value clears the variable.
    const std::string name_str{name};
    const std::string value_str{value};
    ::_putenv_s(name_str.c_str(), value_str.c_str());
    ::SetEnvironmentVariable(name_str.c_str(), value.empty() ? nullptr : value_str.c_str());
}

}  // namespace platform
