// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

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
    ::SetEnvironmentVariable(std::string{name}.c_str(), std::string{value}.c_str());
}

}  // namespace platform
