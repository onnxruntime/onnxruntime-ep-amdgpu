// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifdef _WIN32

#include <cassert>
#include <string>
#include <utility>

#include <windows.h>

#include "plugin_ep_utils.h"

std::string ToUTF8String(std::wstring_view sv) {
    if (sv.size() >= static_cast<size_t>(std::numeric_limits<int>::max()))
        THROW("length overflow");
    const int src_len = static_cast<int>(sv.size() + 1);
    const int len = WideCharToMultiByte(CP_UTF8, 0, sv.data(), src_len, nullptr, 0, nullptr, nullptr);
    assert(len > 0);
    std::string ret(static_cast<size_t>(len) - 1, '\0');
#pragma warning(disable : 4189)
    const int r = WideCharToMultiByte(CP_UTF8, 0, sv.data(), src_len, (char*)ret.data(), len, nullptr, nullptr);
    assert(len == r);
#pragma warning(default : 4189)
    return ret;
}

std::wstring ToWideString(std::string_view sv) {
    if (sv.size() >= static_cast<size_t>(std::numeric_limits<int>::max()))
        THROW("length overflow");
    const int src_len = static_cast<int>(sv.size() + 1);
    const int len = MultiByteToWideChar(CP_UTF8, 0, sv.data(), src_len, nullptr, 0);
    assert(len > 0);
    std::wstring ret(static_cast<size_t>(len) - 1, '\0');
#pragma warning(disable : 4189)
    const int r = MultiByteToWideChar(CP_UTF8, 0, sv.data(), src_len, (wchar_t*)ret.data(), len);
    assert(len == r);
#pragma warning(default : 4189)
    return ret;
}

#endif  // #ifdef _WIN32
