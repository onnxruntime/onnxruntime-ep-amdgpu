// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>

#include <functional>
#include <map>
#include <memory>
#include <list>
#include <locale>
#include <string>
#include <string_view>
#include <deque>
#include <unordered_set>

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>

#include <range/v3/all.hpp>

#ifdef _WIN32
#include <tchar.h>
#else
#define _T(_s) _s
#endif

#include <gsl/gsl>

using namespace std::literals::string_view_literals;  // NOLINT(build/namespaces_literals)
using ranges::views::zip;

#include "make_string.h"

namespace amd {
constexpr uint32_t VendorId{0x1002};
constexpr auto Vendor{"AMD"};
}

#define MAKE_STATUS(code, ...)                                                  \
    Ort::Status{MakeString(__VA_ARGS__).c_str(), code}

#define RETURN_STATUS(code, ...)                                                \
    return Ort::GetApi().CreateStatus(code, MakeString(__VA_ARGS__).c_str())

#define ORT_MAKE_STATUS(code, ...)                                              \
    ::Ort::Status{::MakeString(__VA_ARGS__).c_str(), code}

#define STATUS_OK                                                               \
    Ort::Status{}

#define ENFORCE(cond, ...)                                                      \
    do {                                                                        \
        if (!(cond)) {                                                          \
            throw std::runtime_error{MakeString(__VA_ARGS__)};                  \
        }                                                                       \
    } while (0)

#define THROW(...)                                                              \
    throw std::runtime_error{MakeString(__VA_ARGS__)}

#define RETURN_IF_STATUS(fn)                                                    \
    do {                                                                        \
        if (OrtStatus* _status{fn}; _status != nullptr) {                       \
            return Ort::Status{_status};                                        \
        }                                                                       \
    } while (0)

#define RETURN_IF_ERROR(fn)                                                     \
    do {                                                                        \
        if (Ort::Status _status{fn}; !_status.IsOK()) {                         \
            return _status;                                                     \
        }                                                                       \
    } while (0)

#define RETURN_IF(cond, ...)                                                    \
    do {                                                                        \
        if ((cond)) {                                                           \
            return MAKE_STATUS(ORT_EP_FAIL, __VA_ARGS__);                       \
        }                                                                       \
    } while (0)

#define RETURN_IF_NOT(cond, ...) RETURN_IF(!(cond), __VA_ARGS__)

#define THROW_IF(cond, ...)                                                     \
    do {                                                                        \
      if ((cond)) {                                                             \
          THROW(__VA_ARGS__);                                                   \
      }                                                                         \
    } while (0)

#define THROW_IF_NOT(cond, ...) THROW_IF(!(cond), __VA_ARGS__)

#define THROW_IF_ERROR(fn)                                                      \
    do {                                                                        \
        if (Ort::Status _status{fn}; !_status.IsOK()) {                         \
            throw std::runtime_error{_status.GetErrorMessage()};                \
        }                                                                       \
    } while (0)

#define RETURN_FALSE_IF_ERROR(fn)                                               \
    do {                                                                        \
        if (Ort::Status _status{fn}; !_status.IsOK()) {                         \
            fmt::println(stderr, "{}", _status.GetErrorMessage(status_))        \
            return false;                                                       \
        }                                                                       \
    } while (0)

#define IGNORE_STATUS(fn)                                                       \
    do {                                                                        \
        const Ort::Status{fn};                                                  \
    } while (0)

#define API_CALL_S(type, p, fn, ...)                                            \
    do {                                                                        \
        if ((p) == nullptr) {                                                   \
            RETURN_STATUS(ORT_INVALID_ARGUMENT, "invalid object pointer");      \
        }                                                                       \
        return (reinterpret_cast<type*>(p)->fn(__VA_ARGS__)).release();         \
    } while (0)

#define API_CALL_T(type, p, fn, defval, ...)                                    \
    do {                                                                        \
        if ((p) == nullptr) {                                                   \
            return defval;                                                      \
        }                                                                       \
        return reinterpret_cast<type*>(p)->fn(__VA_ARGS__);                     \
    } while (0)

#define API_CALL_V(type, p, fn, ...)                                            \
    do {                                                                        \
        if ((p) != nullptr) {                                                   \
            reinterpret_cast<type*>(p)->fn(__VA_ARGS__);                        \
        }                                                                       \
    } while (0)

template <typename Allocator>
using AllocatorPtr = std::unique_ptr<Allocator>;
template <typename Allocator>
using AllocatorMap = std::unordered_map<int, AllocatorPtr<Allocator>>;

template <typename T>
using Map = std::unordered_map<std::string, T>;

using MemoryInfoPtr = std::unique_ptr<OrtMemoryInfo, std::function<void(OrtMemoryInfo*)>>;
using MemoryInfoMap = std::unordered_map<int, MemoryInfoPtr>;

using MemoryDeviceList = std::vector<const OrtMemoryDevice*>;

struct ApiPtrs {
    const OrtApi& ort_api;
    const OrtEpApi& ep_api;
    const OrtModelEditorApi& model_editor_api;
};

template <typename T>
struct DeferRelease {
    DeferRelease(T** object, std::function<void(T*)> release_fn)
        : release_fn_{release_fn}, objects_{object}, count_{1} {}
    DeferRelease(T** objects, size_t count, std::function<void(T*)> release_fn)
        : release_fn_{release_fn}, objects_{objects}, count_(count) {}
    ~DeferRelease() {
        for (auto& p : gsl::span{objects_, count_}) {
            if (p != nullptr) {
                release_fn_(p);
                p = nullptr;
            }
        }
    }
private:
    std::function<void(T*)> release_fn_{};
    T** objects_{};
    size_t count_{};
};

template <ONNXTensorElementDataType Type>
bool IsTensorOfType(const Ort::ConstValueInfo& value_info) {
    const auto type_info{value_info.TypeInfo()};
    return type_info.GetONNXType() == ONNX_TYPE_TENSOR &&
        type_info.GetTensorTypeAndShapeInfo().GetElementType() == Type;
}

inline bool IsInt2Tensor(const Ort::ConstValueInfo& value_info) {
    return IsTensorOfType<ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2>(value_info);
}
inline bool IsInt4Tensor(const Ort::ConstValueInfo& value_info) {
    return IsTensorOfType<ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4>(value_info);
}
inline bool IsInt8Tensor(const Ort::ConstValueInfo& value_info) {
    return IsTensorOfType<ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8>(value_info);
}
inline bool IsInt16Tensor(const Ort::ConstValueInfo& value_info) {
    return IsTensorOfType<ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16>(value_info);
}
inline bool IsInt32Tensor(const Ort::ConstValueInfo& value_info) {
    return IsTensorOfType<ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32>(value_info);
}
inline bool IsInt64Tensor(const Ort::ConstValueInfo& value_info) {
    return IsTensorOfType<ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64>(value_info);
}
inline bool IsUint2Tensor(const Ort::ConstValueInfo& value_info) {
    return IsTensorOfType<ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2>(value_info);
}
inline bool IsUint4Tensor(const Ort::ConstValueInfo& value_info) {
    return IsTensorOfType<ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4>(value_info);
}
inline bool IsUint8Tensor(const Ort::ConstValueInfo& value_info) {
    return IsTensorOfType<ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8>(value_info);
}
inline bool IsUint16Tensor(const Ort::ConstValueInfo& value_info) {
    return IsTensorOfType<ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16>(value_info);
}
inline bool IsUint32Tensor(const Ort::ConstValueInfo& value_info) {
    return IsTensorOfType<ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32>(value_info);
}
inline bool IsUint64Tensor(const Ort::ConstValueInfo& value_info) {
    return IsTensorOfType<ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64>(value_info);
}
inline bool IsFloatTensor(const Ort::ConstValueInfo& value_info) {
    return IsTensorOfType<ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT>(value_info);
}
inline bool IsDoubleTensor(const Ort::ConstValueInfo& value_info) {
    return IsTensorOfType<ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE>(value_info);
}
inline bool IsStringTensor(const Ort::ConstValueInfo& value_info) {
    return IsTensorOfType<ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING>(value_info);
}
inline bool IsBoolTensor(const Ort::ConstValueInfo& value_info) {
    return IsTensorOfType<ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL>(value_info);
}
inline bool IsFloat16Tensor(const Ort::ConstValueInfo& value_info) {
    return IsTensorOfType<ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16>(value_info);
}
inline bool IsComplex64TypeTensor(const Ort::ConstValueInfo& value_info) {
    return IsTensorOfType<ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64>(value_info);
}
inline bool IsComplex128TypeTensor(const Ort::ConstValueInfo& value_info) {
    return IsTensorOfType<ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128>(value_info);
}
inline bool IsBFloat16Tensor(const Ort::ConstValueInfo& value_info) {
    return IsTensorOfType<ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16>(value_info);
}

inline std::optional<std::vector<int64_t>> GetTensorShape(const Ort::ConstValueInfo& value_info) {
    const auto type_info{value_info.TypeInfo()};
    return (type_info.GetONNXType() != ONNX_TYPE_TENSOR) ? std::nullopt :
        std::make_optional(type_info.GetTensorTypeAndShapeInfo().GetShape());
}

std::vector<Ort::ConstNode> GetKahnsVariantTopologicalSortedNodes(const Ort::ConstGraph& graph);

inline std::vector<Ort::ConstValueInfo> GetValueInfos(std::vector<Ort::ConstValueInfo> value_infos) {
    value_infos.erase({ranges::remove_if(value_infos,
        [](const Ort::ConstValueInfo& info) { return info == nullptr; })}, value_infos.end());
    return value_infos;
}

template <> struct fmt::formatter<OrtErrorCode> : formatter<std::string_view> {
    format_context::iterator format(const OrtErrorCode error_code, format_context& ctx) const {
#define ORT_ERROR_CODE(__error_code) \
    case __error_code: name = #__error_code; break;
        std::string_view name{"<UNKNOWN>"};
        switch (error_code) {
            ORT_ERROR_CODE(ORT_OK)
            ORT_ERROR_CODE(ORT_FAIL)
            ORT_ERROR_CODE(ORT_INVALID_ARGUMENT)
            ORT_ERROR_CODE(ORT_NO_SUCHFILE)
            ORT_ERROR_CODE(ORT_NO_MODEL)
            ORT_ERROR_CODE(ORT_ENGINE_ERROR)
            ORT_ERROR_CODE(ORT_RUNTIME_EXCEPTION)
            ORT_ERROR_CODE(ORT_INVALID_PROTOBUF)
            ORT_ERROR_CODE(ORT_MODEL_LOADED)
            ORT_ERROR_CODE(ORT_NOT_IMPLEMENTED)
            ORT_ERROR_CODE(ORT_INVALID_GRAPH)
            ORT_ERROR_CODE(ORT_EP_FAIL)
            ORT_ERROR_CODE(ORT_MODEL_LOAD_CANCELED)
            ORT_ERROR_CODE(ORT_MODEL_REQUIRES_COMPILATION)
            ORT_ERROR_CODE(ORT_NOT_FOUND)
        }
        return formatter<std::string_view>::format(name, ctx);
#undef ORT_ERROR_CODE
    }
};

template <> struct fmt::formatter<OrtMemoryInfoDeviceType> : formatter<std::string_view> {
    format_context::iterator format(const OrtMemoryInfoDeviceType memory_info_device_type, format_context& ctx) const {
#define ORT_MEMORY_INFO_DEVICE_TYPE(__memory_info_device_type) \
    case __memory_info_device_type: name = #__memory_info_device_type; break;
        std::string_view name{"<UNKNOWN>"};
        switch (memory_info_device_type) {
            ORT_MEMORY_INFO_DEVICE_TYPE(OrtMemoryInfoDeviceType_CPU)
            ORT_MEMORY_INFO_DEVICE_TYPE(OrtMemoryInfoDeviceType_GPU)
            ORT_MEMORY_INFO_DEVICE_TYPE(OrtMemoryInfoDeviceType_FPGA)
            ORT_MEMORY_INFO_DEVICE_TYPE(OrtMemoryInfoDeviceType_NPU)
        }
        return formatter<std::string_view>::format(name, ctx);
#undef ORT_MEMORY_INFO_DEVICE_TYPE
    }
};

template <> struct fmt::formatter<OrtMemType> : formatter<std::string_view> {
    format_context::iterator format(const OrtMemType memory_type, format_context& ctx) const {
#define ORT_MEM_TYPE(__memory_type) \
    case __memory_type: name = #__memory_type; break;
        std::string_view name{"<UNKNOWN>"};
        switch (memory_type) {
            ORT_MEM_TYPE(OrtMemTypeCPUInput)
            ORT_MEM_TYPE(OrtMemTypeCPUOutput)
            ORT_MEM_TYPE(OrtMemTypeDefault)
        }
        return formatter<std::string_view>::format(name, ctx);
#undef ORT_MEM_TYPE
    }
};

#include "provider_options.h"
