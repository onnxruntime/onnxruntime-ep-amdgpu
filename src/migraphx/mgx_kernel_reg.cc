// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "kernels/kernel.h"
#include "mgx_kernel_reg.h"

namespace mgx_ep {

static const BuildKernelCreateInfoFn migraphx_build_kernel_create_info_fn[] = {
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kOnnxDomain, 1, MemcpyFromHost)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kOnnxDomain, 1, MemcpyToHost)>
};

size_t GetNumRegisteredKernels() {
    return std::size(migraphx_build_kernel_create_info_fn);
}

Ort::Status CreateKernelRegistry(std::string_view ep_name, void* create_kernel_state,
    OrtKernelRegistry*& out_kernel_registry)
try {
    out_kernel_registry = nullptr;
    Ort::KernelRegistry kernel_registry;
    KernelCreateInfo kernel_create_info;
    for (const auto& build_func : migraphx_build_kernel_create_info_fn) {
        RETURN_IF_ERROR(build_func(ep_name, create_kernel_state, kernel_create_info));
        if (kernel_create_info.kernel_def_ != nullptr) {
            RETURN_IF_ERROR(kernel_registry.AddKernel(kernel_create_info.kernel_def_,
                kernel_create_info.kernel_create_func_, kernel_create_info.kernel_create_func_state_));
        }
    }
    out_kernel_registry = kernel_registry.release();
    return STATUS_OK;
} catch (const Ort::Exception& e) {
    return Ort::Status{e};
} catch (const std::exception& e) {
    return Ort::Status{e.what(), ORT_EP_FAIL};
}

}  // namespace mgx_ep