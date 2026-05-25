// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/plugin_ep_utils.h"

namespace mgx_ep {

inline const OrtDataType* GetTensorType(ONNXTensorElementDataType element_type) {
    const OrtDataType* result{};
    Ort::ThrowOnError(Ort::GetEpApi().GetTensorDataType(element_type, &result));
    return result;
}

struct KernelCreateInfo {
    KernelCreateInfo() = default;
    KernelCreateInfo(Ort::KernelDef kernel_def, OrtKernelCreateFunc kernel_create_func, void* kernel_create_func_state)
        : kernel_def_{std::move(kernel_def)}, kernel_create_func_{kernel_create_func}, kernel_create_func_state_{kernel_create_func_state} {}

    Ort::KernelDef kernel_def_{};
    OrtKernelCreateFunc kernel_create_func_{};
    void* kernel_create_func_state_{};
};

using BuildKernelCreateInfoFn = Ort::Status (*)(std::string_view, void*, KernelCreateInfo&);

template <typename T>
Ort::Status BuildKernelCreateInfo(std::string_view ep_name, void* create_func_state, KernelCreateInfo& result);

template <>
inline Ort::Status BuildKernelCreateInfo<void>(std::string_view, void*, KernelCreateInfo& result) {
    result.kernel_def_ = {};
    result.kernel_create_func_ = nullptr;
    result.kernel_create_func_state_ = nullptr;
    return STATUS_OK;
}

#define ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(domain, startver, endver, name) \
    kernel_ep_##name##_##domain##_ver##startver##_##endver

#define ONNX_OPERATOR_KERNEL_CLASS_NAME(domain, version, name) \
    ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(domain, version, version, name)

#define ONNX_OPERATOR_VERSIONED_KERNEL_EX(name, domain, startver, endver, builder, kernel_class)      \
    class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(domain, startver, endver, name);                  \
    template <>                                                                                       \
    Ort::Status BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(domain, startver, endver, name)>( \
        std::string_view ep_name, void* create_kernel_state, KernelCreateInfo& result)                \
    try {                                                                                             \
        Ort::KernelDef kernel_def = builder.SetOperatorType(#name)                                    \
                                           .SetDomain(domain)                                         \
                                           .SetSinceVersion(startver, endver)                         \
                                           .SetExecutionProvider(std::string{ep_name}.c_str())        \
                                           .Build();                                                  \
                                                                                                      \
        auto kernel_create_func = [](void* kernel_create_func_state, const OrtKernelInfo* info,       \
                                     OrtKernelImpl** kernel_out) noexcept -> OrtStatus* {             \
          RETURN_IF(kernel_out == nullptr,                                                            \
                    "OrtKernelCreateFunc received a NULL kernel_out argument");                       \
                                                                                                      \
          *kernel_out = nullptr;                                                                      \
          RETURN_IF_ERROR(kernel_class::Create(info, kernel_create_func_state, *kernel_out));         \
          return STATUS_OK;                                                                           \
        };                                                                                            \
        result = KernelCreateInfo(std::move(kernel_def), kernel_create_func, create_kernel_state);    \
        return STATUS_OK;                                                                             \
    } catch (const Ort::Exception& ex) {                                                              \
        return Ort::Status{ex};                                                                       \
    } catch (const std::exception& ex) {                                                              \
        return Ort::Status{ex.what(), ORT_EP_FAIL};                                                   \
    }

// Defines a function of type BuildKernelCreateInfoFn for a kernel implementation with a start version.
#define ONNX_OPERATOR_KERNEL_EX(name, domain, version, builder, kernel_class) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(name, domain, version, version, builder, kernel_class)

}  // namespace mgx_ep