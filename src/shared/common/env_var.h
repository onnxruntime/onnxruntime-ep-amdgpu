// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <unordered_set>

#if defined(_WIN32)
#include "platform/windows/env_var.h"
#elif defined(__linux__)
#include "platform/linux/env_var.h"
#endif

#include "parse_string.h"

template <typename T>
std::optional<T> ParseEnvironmentVariable(const std::string_view name) {
    auto value{platform::GetEnvironmentVar(name)};
    value = Trim(value, [](const int ch) {
           return ::isspace(ch) == 0 && ch != '"' && ch != '\'' ? 1 : 0;
        });
    if (value.empty()) {
        return std::nullopt;
    }
    T parsed_value;
    ENFORCE(TryParseStringWithClassicLocale(value, parsed_value),
        "Failed to parse environment variable - name: \"", name, "\", value: \"", value, "\"");
    return parsed_value;
}

template <typename T>
T ParseEnvironmentVariableWithDefault(const std::string_view name, const T& default_value) {
    const auto parsed{ParseEnvironmentVariable<T>(name)};
    if (parsed.has_value()) {
        return *parsed;
    }
    return default_value;
}
