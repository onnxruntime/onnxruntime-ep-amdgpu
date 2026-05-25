// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <type_traits>
#include <charconv>
#include <utility>

#include "plugin_ep_utils.h"

template <typename T>
bool TryParseStringWithClassicLocale(std::string_view str, T& value) {
    if constexpr (std::is_integral<T>::value && std::is_unsigned<T>::value) {
        if (!str.empty() && str[0] == '-') {
            return false;
        }
    }

    if (!str.empty() && std::isspace(str[0], std::locale::classic())) {
        return false;
    }

    std::istringstream is{std::string{str}};
    is.imbue(std::locale::classic());
    T parsed_value{};

    const bool parse_successful =
        is >> parsed_value &&
        is.get() == std::istringstream::traits_type::eof();  // don't allow trailing characters
    if (!parse_successful) {
        return false;
    }

    value = std::move(parsed_value);
    return true;
}

inline bool TryParseStringWithClassicLocale(std::string_view str, std::string& value) {
    value = str;
    return true;
}

inline bool TryParseStringWithClassicLocale(std::string_view sv, bool& value) {
    std::string str{sv};
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    if (str == "0" || str == "false" || str == "off" || str == "disable" || str == "disabled") {
        value = false;
        return true;
    }

    if (str == "1" || str == "true" || str == "on" || str == "enable" || str == "enabled") {
        value = true;
        return true;
    }

    return false;
}

template <typename T>
Ort::Status ParseStringWithClassicLocale(std::string_view s, T& value) {
    RETURN_IF_NOT(TryParseStringWithClassicLocale(s, value), "Failed to parse value: \"", value, "\"");
    return STATUS_OK;
}

template <typename T>
Ort::Status ParseStringWithClassicLocale(std::string_view s, std::optional<T>& value) {
    T result{};
    RETURN_IF_NOT(TryParseStringWithClassicLocale(s, result), "Failed to parse value: \"", value, "\"");
    value = result;
    return STATUS_OK;
}

template <typename T>
T ParseStringWithClassicLocale(const std::string_view sv) {
    T value{};
    ORT_THROW_IF_ERROR(ParseStringWithClassicLocale(sv, value));
    return value;
}

inline std::string_view TrimLeft(std::string_view sv, int (*fn)(int) = std::isspace) {
    const auto it{std::find_if(sv.begin(), sv.end(),
        [fn](const int ch) {
            return fn(ch);
        })};
    return sv.substr(it - sv.begin());
}

inline std::string_view TrimRight(std::string_view sv, int (*fn)(int) = std::isspace) {
    const auto it{std::find_if(sv.rbegin(), sv.rend(),
        [fn](int ch) {
            return fn(ch);
        })};
    return sv.substr(0, it.base() - sv.begin());
}

inline std::string_view Trim(std::string_view sv, int (*fn)(int) = std::isspace) {
    return TrimRight(TrimLeft(sv, fn), fn);
}

template <typename T>
T ToInteger(const std::string_view sv) {
    T result = 0;
    if (auto [_, ec] = std::from_chars(sv.data(), sv.data() + sv.length(), result); ec == std::errc{}) {
        return result;
    }
    throw std::runtime_error{"invalid input for conversion to integer"};
}