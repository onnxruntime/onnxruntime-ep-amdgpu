// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <common/plugin_ep_utils.h>

namespace mgx_ep {

struct MemcpyKernel : OrtKernelImpl {
protected:
    MemcpyKernel(const Ort::KernelInfo& kernel_info, void* state)
        : OrtKernelImpl{ORT_API_VERSION}, kernel_info_{kernel_info}, state_{state} {}

    template <typename T>
    static Ort::Status CreateImpl(const OrtKernelInfo* kernel_info, void* state, OrtKernelImpl*& kernel) noexcept;

    template <typename T>
    static void ReleaseImpl(OrtKernelImpl* _this) noexcept;

    const Ort::KernelInfo& kernel_info_;
    void* state_;
};

struct MemcpyFromHost : MemcpyKernel {
private:
    struct PrivateTag {};
    friend struct MemcpyKernel;
public:
    MemcpyFromHost(const Ort::KernelInfo& kernel_info, void* state, PrivateTag)
        : MemcpyKernel{kernel_info, state}
    {
        OrtKernelImpl::Release = [](OrtKernelImpl* _this) noexcept {
            ReleaseImpl<MemcpyFromHost>(_this);
        };
        OrtKernelImpl::Compute = [](OrtKernelImpl* _this, OrtKernelContext* context) noexcept {
            API_CALL_S(MemcpyFromHost, _this, Compute, Ort::KernelContext{context});
        };
    }

    static Ort::Status Create(const OrtKernelInfo* kernel_info, void* state, OrtKernelImpl*& kernel) noexcept {
        return CreateImpl<MemcpyFromHost>(kernel_info, state, kernel);
    }
private:
    Ort::Status Compute(const Ort::KernelContext& context);

};

struct MemcpyToHost : MemcpyKernel {
private:
    struct PrivateTag {};
    friend struct MemcpyKernel;
public:
    MemcpyToHost(const Ort::KernelInfo& kernel_info, void* state, PrivateTag)
        : MemcpyKernel{kernel_info, state}
    {
        OrtKernelImpl::Release = [](OrtKernelImpl* _this) noexcept {
            ReleaseImpl<MemcpyToHost>(_this);
        };
        OrtKernelImpl::Compute = [](OrtKernelImpl* _this, OrtKernelContext* context) noexcept {
            API_CALL_S(MemcpyToHost, _this, Compute, Ort::KernelContext{context});
        };
    }

    static Ort::Status Create(const OrtKernelInfo* kernel_info, void* state, OrtKernelImpl*& kernel) noexcept {
        return CreateImpl<MemcpyToHost>(kernel_info, state, kernel);
    }
private:
    Ort::Status Compute(const Ort::KernelContext& context);
};

}  // namespace mgx_ep::kernel