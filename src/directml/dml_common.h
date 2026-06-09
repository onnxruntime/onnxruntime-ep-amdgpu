// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <DirectML.h>

#include "DmlExecutionProvider/inc/MLOperatorAuthor.h"
#include "OperatorAuthorHelper/operator_helper_common.h"

namespace dml_ep {

constexpr int MaximumDimensionCount{DML_TENSOR_DIMENSION_COUNT_MAX1};

DML_TENSOR_DATA_TYPE GetDmlDataTypeFromMlDataType(MLOperatorTensorDataType tensorDataType);
DML_TENSOR_DATA_TYPE GetDmlDataTypeFromMlDataTypeNoThrow(MLOperatorTensorDataType tensorDataType) noexcept;
MLOperatorTensorDataType GetMlDataTypeFromDmlDataType(DML_TENSOR_DATA_TYPE tensorDataType);
size_t ComputeByteSizeFromDimensions(gsl::span<const OperatorHelper::DimensionType> dimensions,
                                     MLOperatorTensorDataType tensorDataType);
size_t ComputeByteSizeFromTensor(const Microsoft::WRL::ComPtr<IMLOperatorTensor>& tensor);
uint32_t GetSupportedDeviceDataTypeMask(const Microsoft::WRL::ComPtr<IDMLDevice>& dmlDevice);
uint32_t GetBitMaskFromIndices(gsl::span<const uint32_t> indices) noexcept;
uint32_t CountLeastSignificantZeros(uint32_t value) noexcept;
void GetDescendingPackedStrides(gsl::span<const uint32_t> sizes, /*out*/ gsl::span<uint32_t> strides) noexcept;

bool IsSigned(DML_TENSOR_DATA_TYPE dataType) noexcept;

template <typename T>
void CastToClampedScalarUnion(DML_TENSOR_DATA_TYPE dataType, T value, DML_SCALAR_UNION* outputValue) {
    switch (dataType) {
    case DML_TENSOR_DATA_TYPE_UINT8:
        outputValue->UInt8 = OperatorHelper::clamp_cast<uint8_t, T>(value);
        break;
    case DML_TENSOR_DATA_TYPE_UINT16:
        outputValue->UInt16 = OperatorHelper::clamp_cast<uint16_t, T>(value);
        break;
    case DML_TENSOR_DATA_TYPE_UINT32:
        outputValue->UInt32 = OperatorHelper::clamp_cast<uint32_t, T>(value);
        break;
    case DML_TENSOR_DATA_TYPE_UINT64:
        outputValue->UInt64 = OperatorHelper::clamp_cast<uint64_t, T>(value);
        break;
    case DML_TENSOR_DATA_TYPE_INT8:
        outputValue->Int8 = OperatorHelper::clamp_cast<int8_t, T>(value);
        break;
    case DML_TENSOR_DATA_TYPE_INT16:
        outputValue->Int16 = OperatorHelper::clamp_cast<int16_t, T>(value);
        break;
    case DML_TENSOR_DATA_TYPE_INT32:
        outputValue->Int32 = OperatorHelper::clamp_cast<int32_t, T>(value);
        break;
    case DML_TENSOR_DATA_TYPE_INT64:
        outputValue->Int64 = OperatorHelper::clamp_cast<int64_t, T>(value);
        break;
    case DML_TENSOR_DATA_TYPE_FLOAT16:
        outputValue->Float32 = OperatorHelper::clamp_cast<float, T>(value);
        break;
    case DML_TENSOR_DATA_TYPE_FLOAT32:
        outputValue->Float32 = OperatorHelper::clamp_cast<float, T>(value);
        break;
    case DML_TENSOR_DATA_TYPE_FLOAT64:
        outputValue->Float64 = OperatorHelper::clamp_cast<double, T>(value);
        break;
    default:
        assert(false);
    }
}
}  // namespace dml_ep
