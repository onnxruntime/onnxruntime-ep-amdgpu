// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>

namespace detail {
template <typename T>
void MakeStringImpl(std::string& s, const T& t) noexcept {
    s += fmt::format("{}", t);
}
inline void MakeStringImpl(std::string&) noexcept {
}
template <typename T, typename... Args>
void MakeStringImpl(std::string& s, const T& t, const Args&... args) noexcept {
    MakeStringImpl(s, t);
    MakeStringImpl(s, args...);
}
template <typename... Args>
std::string MakeStringImpl(const Args&... args) noexcept {
    std::string result;
    MakeStringImpl(result, args...);
    return result;
}
template <typename T>
struct if_char_array_make_ptr {
    using type = T;
};
template <typename T, size_t N>
struct if_char_array_make_ptr<T (&)[N]> {
    using element_type = std::remove_const_t<std::remove_extent_t<T>>;
    using type = std::conditional_t<std::is_same_v<element_type, char>, T*, T (&)[N]>;
};
template <typename T>
using if_char_array_make_ptr_t = typename if_char_array_make_ptr<T>::type;
}  // namespace detail

template <typename... Args>
std::string MakeString(const Args&... args) {
    return detail::MakeStringImpl(detail::if_char_array_make_ptr_t<const Args&>(args)...);
}

inline std::string MakeString(const std::string& s) {
    return s;
}

inline std::string MakeString() {
    return "";
}

inline std::string MakeString(std::string_view sv) {
    return std::string{sv};
}

namespace detail {
template <typename T>
void MakeStringWithClassicLocaleImpl(std::string& s, const T& t) noexcept {
    s += fmt::format(std::locale::classic(), "{}", t);
}
inline void MakeStringWithClassicLocaleImpl(std::string&) noexcept {
}
template <typename T, typename... Args>
void MakeStringWithClassicLocaleImpl(std::string& s, const T& t, const Args&... args) noexcept {
    MakeStringWithClassicLocaleImpl(s, t);
    MakeStringWithClassicLocaleImpl(s, args...);
}
template <typename... Args>
std::string MakeStringWithClassicLocaleImpl(const Args&... args) noexcept {
    std::string result;
    MakeStringImpl(result, args...);
    return result;
}
}  // namespace detail

template <typename... Args>
std::string MakeStringWithClassicLocale(const Args&... args) {
    return detail::MakeStringWithClassicLocaleImpl(detail::if_char_array_make_ptr_t<const Args&>(args)...);
}

inline std::string MakeStringWithClassicLocale(const std::string& s) {
    return s;
}

inline std::string MakeStringWithClassicLocale(const char* s) {
    return s;
}

inline std::string MakeStringWithClassicLocale(std::string_view sv) {
    return std::string{sv};
}