// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <charconv>
#include <cstdint>
#include <cstddef>
#include <gsl/gsl>
#include <string>

namespace hash {

namespace murmur3 {
    // generate 32-bit hash from input and write to 'out'
    void x86_32(const void* key, size_t len, uint32_t seed, void* out);

    // generate 128-bit hash from input and write to 'out'.
    void x86_128(const void* key, size_t len, uint32_t seed, void* out);
}  // namespace murmir3

using Value = std::array<uint32_t, 4>;

template <typename T>
constexpr Value& Hash(Value& v, T t) {
    murmur3::x86_128(t.data(), gsl::narrow_cast<int32_t>(t.size() * sizeof(*t.data())), v.front(), v.data());
    return v;
}

template <>
constexpr Value& Hash(Value& v, const char* t) {
    return Hash(v, std::string_view(t));
}

template <>
inline Value& Hash(Value& v, size_t t) {
    return Hash(v, gsl::span{&t, 1});
}

inline std::string ToHex(const uint64_t v) {
    std::array<char, sizeof v << 1> s{};
    auto [ptr, _] = std::to_chars(s.data(), s.data() + s.size(), v, 16);
    return std::string{s.data(), ptr};
}

inline std::string ToHex(const Value& v) {
    return ToHex(v.at(0) | static_cast<uint64_t>(v.at(1)) << 32);
}

inline std::string ToHex(std::string_view t) {
    Value v{};
    return ToHex(Hash(v, t));
}

}  // namespace hash
