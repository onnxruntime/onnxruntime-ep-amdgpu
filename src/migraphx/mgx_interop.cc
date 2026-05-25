// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "mgx_interop.h"

#include "common/constants.h"
#include "hip/utils.h"

namespace mgx_ep {

namespace {

bool IsSupportedMemoryHandleType([[maybe_unused]] OrtExternalMemoryHandleType type) noexcept {
#ifdef _WIN32
    return type == ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE
        || type == ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP;
#else
    return false;
#endif
}

hipExternalMemoryHandleType ToHipMemoryHandleType(OrtExternalMemoryHandleType type) noexcept {
    switch (type) {
    case ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE:
        return hipExternalMemoryHandleTypeD3D12Resource;
    case ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP:
        return hipExternalMemoryHandleTypeD3D12Heap;
    }
    return hipExternalMemoryHandleTypeD3D12Resource;
}

bool IsDedicated(OrtExternalMemoryHandleType type) noexcept {
    return type == ORT_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
}

size_t TensorElementSize(ONNXTensorElementDataType type) noexcept {
    switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return sizeof(float);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return sizeof(double);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
        return 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        return sizeof(int16_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return sizeof(int32_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return sizeof(int64_t);
    default:
        return 0;
    }
}

}  // namespace

ExternalMemoryHandle::ExternalMemoryHandle(const OrtExternalMemoryDescriptor& descriptor_in,
    hipExternalMemory_t hip_ext_memory, void* mapped_ptr) noexcept
    : OrtExternalMemoryHandle{}, hip_ext_memory_{hip_ext_memory}, mapped_ptr_{mapped_ptr}
{
    version = ORT_API_VERSION;
    descriptor = descriptor_in;
    Release = ReleaseImpl;
}

void ORT_API_CALL ExternalMemoryHandle::ReleaseImpl(OrtExternalMemoryHandle* handle) noexcept {
    if (handle == nullptr) {
        return;
    }
    auto* derived{static_cast<ExternalMemoryHandle*>(handle)};
    if (derived->hip_ext_memory_ != nullptr) {
        (void)hipDestroyExternalMemory(derived->hip_ext_memory_);
    }
    delete derived;
}

ExternalSemaphoreHandle::ExternalSemaphoreHandle(hipExternalSemaphore_t hip_ext_semaphore) noexcept
    : OrtExternalSemaphoreHandle{}, hip_ext_semaphore_{hip_ext_semaphore}
{
    version = ORT_API_VERSION;
    Release = ReleaseImpl;
}

void ORT_API_CALL ExternalSemaphoreHandle::ReleaseImpl(OrtExternalSemaphoreHandle* handle) noexcept {
    if (handle == nullptr) {
        return;
    }
    auto* derived{static_cast<ExternalSemaphoreHandle*>(handle)};
    if (derived->hip_ext_semaphore_ != nullptr) {
        (void)hipDestroyExternalSemaphore(derived->hip_ext_semaphore_);
    }
    delete derived;
}

ExternalResourceImporter::ExternalResourceImporter(const ApiPtrs& api_ptrs, int device_id)
    : OrtExternalResourceImporterImpl{}, ApiPtrs{api_ptrs}, device_id_{device_id}
{
    ort_version_supported = ORT_API_VERSION;

    CanImportMemory = CanImportMemoryImpl;
    ImportMemory = ImportMemoryImpl;
    ReleaseMemory = ReleaseMemoryImpl;
    CreateTensorFromMemory = CreateTensorFromMemoryImpl;

    CanImportSemaphore = CanImportSemaphoreImpl;
    ImportSemaphore = ImportSemaphoreImpl;
    ReleaseSemaphore = ReleaseSemaphoreImpl;
    WaitSemaphore = WaitSemaphoreImpl;
    SignalSemaphore = SignalSemaphoreImpl;

    Release = ReleaseSelfImpl;
}

bool ORT_API_CALL ExternalResourceImporter::CanImportMemoryImpl(const OrtExternalResourceImporterImpl* /* this_ptr */,
    OrtExternalMemoryHandleType handle_type) noexcept
{
    return IsSupportedMemoryHandleType(handle_type);
}

OrtStatus* ORT_API_CALL ExternalResourceImporter::ImportMemoryImpl(OrtExternalResourceImporterImpl* this_ptr,
    const OrtExternalMemoryDescriptor* desc, OrtExternalMemoryHandle** out_handle) noexcept
{
    auto& impl{*static_cast<ExternalResourceImporter*>(this_ptr)};
    *out_handle = nullptr;

    if (!IsSupportedMemoryHandleType(desc->handle_type)) {
        RETURN_STATUS(ORT_NOT_IMPLEMENTED, "Unsupported external memory handle type for MIGraphX EP: ",
            static_cast<int>(desc->handle_type));
    }

    HIP_RETURN_IF_ERROR(hipSetDevice(impl.device_id_));

    hipExternalMemoryHandleDesc hip_desc{};
    hip_desc.type = ToHipMemoryHandleType(desc->handle_type);
    hip_desc.handle.win32.handle = desc->native_handle;
    hip_desc.handle.win32.name = nullptr;
    hip_desc.size = desc->size_bytes;
    hip_desc.flags = IsDedicated(desc->handle_type) ? hipExternalMemoryDedicated : 0u;

    hipExternalMemory_t hip_ext_memory{};
    HIP_RETURN_IF_ERROR(hipImportExternalMemory(&hip_ext_memory, &hip_desc));

    hipExternalMemoryBufferDesc buffer_desc{};
    buffer_desc.offset = desc->offset_bytes;
    buffer_desc.size = desc->size_bytes - desc->offset_bytes;
    buffer_desc.flags = 0;

    void* mapped_ptr{};
    if (Ort::Status status{HIP_CALL(hipExternalMemoryGetMappedBuffer(&mapped_ptr, hip_ext_memory, &buffer_desc))};
            !status.IsOK()) {
        (void)hipDestroyExternalMemory(hip_ext_memory);
        return status.release();
    }

    *out_handle = std::make_unique<ExternalMemoryHandle>(*desc, hip_ext_memory, mapped_ptr).release();
    return nullptr;
}

void ORT_API_CALL ExternalResourceImporter::ReleaseMemoryImpl(OrtExternalResourceImporterImpl* /* this_ptr */,
    OrtExternalMemoryHandle* handle) noexcept
{
    if (handle == nullptr) {
        return;
    }
    if (handle->Release != nullptr) {
        handle->Release(handle);
        return;
    }
    delete static_cast<ExternalMemoryHandle*>(handle);
}

OrtStatus* ORT_API_CALL ExternalResourceImporter::CreateTensorFromMemoryImpl(OrtExternalResourceImporterImpl* this_ptr,
    const OrtExternalMemoryHandle* mem_handle, const OrtExternalTensorDescriptor* tensor_desc,
    OrtValue** out_tensor) noexcept
{
    auto& impl{*static_cast<ExternalResourceImporter*>(this_ptr)};
    *out_tensor = nullptr;

    const size_t element_size{TensorElementSize(tensor_desc->element_type)};
    if (element_size == 0) {
        RETURN_STATUS(ORT_INVALID_ARGUMENT, "Unsupported tensor element type: ",
            static_cast<int>(tensor_desc->element_type));
    }

    size_t element_count{1};
    for (size_t i{}; i < tensor_desc->rank; ++i) {
        element_count *= static_cast<size_t>(tensor_desc->shape[i]);
    }
    const size_t tensor_bytes{element_count * element_size};

    const auto& derived{*static_cast<const ExternalMemoryHandle*>(mem_handle)};
    const auto& base_desc{derived.descriptor};
    const size_t mapped_bytes{base_desc.size_bytes - base_desc.offset_bytes};
    const size_t available_bytes{mapped_bytes - tensor_desc->offset_bytes};
    if (tensor_bytes > available_bytes) {
        RETURN_STATUS(ORT_INVALID_ARGUMENT,
            "Tensor requires ", tensor_bytes, " bytes but only ", available_bytes,
            " bytes are available in the imported memory");
    }

    void* data_ptr{static_cast<char*>(derived.MappedPtr()) + tensor_desc->offset_bytes};

    constexpr size_t kMemoryAlignment = 0;
    OrtMemoryInfo* memory_info{};
    RETURN_IF_STATUS(impl.ort_api.CreateMemoryInfo_V2(
        "Hip",
        OrtMemoryInfoDeviceType_GPU,
        amd::VendorId,
        impl.device_id_,
        OrtDeviceMemoryType_DEFAULT,
        kMemoryAlignment,
        OrtDeviceAllocator,
        &memory_info));
    DeferRelease<OrtMemoryInfo> mem_info_guard{&memory_info, impl.ort_api.ReleaseMemoryInfo};

    RETURN_IF_STATUS(impl.ort_api.CreateTensorWithDataAsOrtValue(
        memory_info,
        data_ptr,
        tensor_bytes,
        tensor_desc->shape,
        tensor_desc->rank,
        tensor_desc->element_type,
        out_tensor));

    return nullptr;
}

bool ORT_API_CALL ExternalResourceImporter::CanImportSemaphoreImpl(const OrtExternalResourceImporterImpl* /* this_ptr */,
    OrtExternalSemaphoreType /* type */) noexcept
{
    // Semaphore-based synchronization is not implemented for the MIGraphX EP.
    // Scope is limited to buffer sharing only.
    return false;
}

OrtStatus* ORT_API_CALL ExternalResourceImporter::ImportSemaphoreImpl(OrtExternalResourceImporterImpl* /* this_ptr */,
    const OrtExternalSemaphoreDescriptor* /* desc */, OrtExternalSemaphoreHandle** out_handle) noexcept
{
    if (out_handle != nullptr) {
        *out_handle = nullptr;
    }
    RETURN_STATUS(ORT_NOT_IMPLEMENTED,
        "External semaphore import is not implemented by the MIGraphX EP");
}

void ORT_API_CALL ExternalResourceImporter::ReleaseSemaphoreImpl(OrtExternalResourceImporterImpl* /* this_ptr */,
    OrtExternalSemaphoreHandle* handle) noexcept
{
    if (handle == nullptr) {
        return;
    }
    if (handle->Release != nullptr) {
        handle->Release(handle);
        return;
    }
    delete static_cast<ExternalSemaphoreHandle*>(handle);
}

OrtStatus* ORT_API_CALL ExternalResourceImporter::WaitSemaphoreImpl(OrtExternalResourceImporterImpl* /* this_ptr */,
    OrtExternalSemaphoreHandle* /* handle */, OrtSyncStream* /* stream */, uint64_t /* value */) noexcept
{
    RETURN_STATUS(ORT_NOT_IMPLEMENTED,
        "External semaphore wait is not implemented by the MIGraphX EP");
}

OrtStatus* ORT_API_CALL ExternalResourceImporter::SignalSemaphoreImpl(OrtExternalResourceImporterImpl* /* this_ptr */,
    OrtExternalSemaphoreHandle* /* handle */, OrtSyncStream* /* stream */, uint64_t /* value */) noexcept
{
    RETURN_STATUS(ORT_NOT_IMPLEMENTED,
        "External semaphore signal is not implemented by the MIGraphX EP");
}

void ORT_API_CALL ExternalResourceImporter::ReleaseSelfImpl(OrtExternalResourceImporterImpl* this_ptr) noexcept {
    delete static_cast<ExternalResourceImporter*>(this_ptr);
}

}  // namespace mgx_ep
