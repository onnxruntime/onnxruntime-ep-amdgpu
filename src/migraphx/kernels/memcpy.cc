// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/utils.h>
#include <common/constants.h>

#include "memcpy.h"
#include "kernel.h"

namespace mgx_ep {

template <typename T>
Ort::Status MemcpyKernel::CreateImpl(const OrtKernelInfo* kernel_info, void* state, OrtKernelImpl*& kernel) noexcept
try {
    auto p{std::make_unique<T>(Ort::KernelInfo{const_cast<OrtKernelInfo*>(kernel_info)}, state, typename T::PrivateTag{})};
    kernel = p.release();
    return STATUS_OK;
} catch (const Ort::Exception& e) {
    return Ort::Status{e};
} catch (const std::exception& e) {
    return Ort::Status{e.what(), ORT_EP_FAIL};
}

template <typename T>
void MemcpyKernel::ReleaseImpl(OrtKernelImpl* _this) noexcept {
    delete static_cast<T*>(_this);
}

Ort::Status MemcpyFromHost::Compute(const Ort::KernelContext& kernel_context)
try {
    const auto input{kernel_context.GetInput(0)};
    const auto info{input.GetTensorTypeAndShapeInfo()};
    const auto shape{info.GetShape()};
    const auto output{kernel_context.GetOutput(0, shape.data(), shape.size())};
    const void* input_data{};
    RETURN_IF_ERROR(Ort::GetApi().GetTensorData(input, &input_data));
    void* output_data{};
    RETURN_IF_ERROR(Ort::GetApi().GetTensorMutableData(output, &output_data));
    size_t bytes{};
    RETURN_IF_ERROR(Ort::GetApi().GetTensorSizeInBytes(input, &bytes));
    hipStream_t hip_stream{static_cast<hipStream_t>(kernel_context.GetGPUComputeStream())};
    HIP_RETURN_IF_ERROR(hipMemcpyAsync(output_data, input_data, bytes, hipMemcpyHostToDevice, hip_stream));
    return STATUS_OK;
} catch (const Ort::Exception& e) {
    return Ort::Status{e};
} catch (const std::exception& e) {
    return Ort::Status{e.what(), ORT_EP_FAIL};
}

Ort::Status MemcpyToHost::Compute(const Ort::KernelContext& kernel_context)
try {
    const auto input{kernel_context.GetInput(0)};
    const auto info{input.GetTensorTypeAndShapeInfo()};
    const auto shape{info.GetShape()};
    const auto output{kernel_context.GetOutput(0, shape.data(), shape.size())};
    const void* input_data{};
    RETURN_IF_ERROR(Ort::GetApi().GetTensorData(input, &input_data));
    void* output_data{};
    RETURN_IF_ERROR(Ort::GetApi().GetTensorMutableData(output, &output_data));
    size_t bytes{};
    RETURN_IF_ERROR(Ort::GetApi().GetTensorSizeInBytes(input, &bytes));
    hipStream_t hip_stream{static_cast<hipStream_t>(kernel_context.GetGPUComputeStream())};
    HIP_RETURN_IF_ERROR(hipMemcpyAsync(output_data, input_data, bytes, hipMemcpyDeviceToHost, hip_stream));
    return STATUS_OK;
} catch (const Ort::Exception& e) {
    return Ort::Status{e};
} catch (const std::exception& e) {
    return Ort::Status{e.what(), ORT_EP_FAIL};
}

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain, 1,
    (Ort::KernelDefBuilder()
        .SetInputMemType(0, OrtMemType::OrtMemTypeCPUInput)
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ))),
    MemcpyFromHost)

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain, 1,
    (Ort::KernelDefBuilder{}
        .SetOutputMemType(0, OrtMemType::OrtMemTypeCPUOutput)
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2))
        .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ))),
    MemcpyToHost)

}  // namespace mgx_ep