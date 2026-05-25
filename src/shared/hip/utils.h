// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>
#include <type_traits>

#include <hip/hip_runtime_api.h>
#include "plugin_ep_utils.h"

namespace hip {

template <typename ErrorType, bool Throw>
std::conditional_t<Throw, void, Ort::Status> Call(ErrorType retCode, std::string_view exprString,
    std::string_view libName, ErrorType successCode, std::string_view msg, std::string_view file, int line);

}  // namespace hip

inline std::string format_as(hipError_t error) {
    return fmt::format("{}", static_cast<int>(error));
}

#define HIP_CALL(__expr)            \
    ::hip::Call<hipError_t, false>(__expr, #__expr, "HIP", hipSuccess, "", __FILE__, __LINE__)

#define HIP_CALL_THROW(__expr)      \
    ::hip::Call<hipError_t, true >(__expr, #__expr, "HIP", hipSuccess, "", __FILE__, __LINE__)

#define HIP_RETURN_IF_ERROR(fn) RETURN_IF_ERROR(HIP_CALL(fn))