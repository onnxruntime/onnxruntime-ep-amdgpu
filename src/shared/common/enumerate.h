// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <iterator>
#include <tuple>

// Implementation of Python-like 'enumerate' built-in function.
// https://docs.python.org/3/library/functions.html#enumerate
//
// Remark: function's 'start' argument is not implemented, and
//         count starts from 0.

template <typename T,
          typename I = decltype(std::begin(std::declval<T>())),
          typename = decltype(std::end(std::declval<T>()))>
constexpr auto enumerate(T&& t) {
    struct iterator {
        std::size_t index;
        I element;

        bool operator != (iterator const& other) const {
            return element != other.element;
        }
        void operator ++ () {
            ++index;
            ++element;
        }
        auto operator * () const {
            return std::tie(index, *element);
        }
    };

    struct iterable_wrapper {
        T iterable;

        auto begin() {
            return iterator{ 0, std::begin(iterable) };
        }
        auto end() {
            return iterator{ 0, std::end(iterable) };
        }
    };

    return iterable_wrapper{ std::forward<T>(t) };
}