// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "hip/utils.h"

namespace hip {

namespace {
template <typename ErrorType>
std::string_view ErrorString(ErrorType) {
    throw std::logic_error{"not implemented"};
}
template <>
std::string_view ErrorString<hipError_t>(hipError_t error) {
  (void)hipDeviceSynchronize();
  return std::string_view{hipGetErrorString(error)};
}
}  // namespace

template <typename ErrorType, bool Throw>
std::conditional_t<Throw, void, Ort::Status> Call(ErrorType retCode, std::string_view exprString,
    std::string_view libName, ErrorType successCode, std::string_view msg, std::string_view file, int line)
{
  if (retCode != successCode) {
    try {
      int currentHipDevice;
      (void)hipGetDevice(&currentHipDevice);
      (void)hipGetLastError();
      std::string s{fmt::format("{} failure {}: {}; GPU={}; file={}; line={}; expr={}; {}",
          libName, static_cast<int>(retCode), ErrorString(retCode), currentHipDevice, file, line, exprString, msg)};
      if constexpr (Throw) {
          THROW(s);
      } else {
        return Ort::Status{(s.c_str()), ORT_FAIL};
      }
    } catch (const std::exception& e) {
      if constexpr (Throw) {
        THROW(e.what());
      } else {
        return Ort::Status{e};
      }
    }
  }
  if constexpr (!Throw) {
    return {};
  }
}

template Ort::Status Call<hipError_t, false>(hipError_t retCode, std::string_view exprString,
    std::string_view libName, hipError_t successCode, std::string_view msg, std::string_view file, int line);

template void Call<hipError_t, true>(hipError_t retCode, std::string_view exprString,
    std::string_view libName, hipError_t successCode, std::string_view msg, std::string_view file, int line);

}  // namespace hip
