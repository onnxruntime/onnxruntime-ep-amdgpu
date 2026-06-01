// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License

// dml_execution_provider.h must come before dml_abi_kernel.h so that
// IWinmlExecutionProvider is fully defined when dml_abi_kernel.h's
// AbiSafeKernelContext class (which has Microsoft::WRL::ComPtr<IWinmlExecutionProvider>)
// is parsed by the compiler.
#include "dml_execution_provider.h"
#include "dml_abi_kernel.h"
#include "DmlExecutionProvider/MLOperatorAuthorImpl.h"
#include "core/framework/op_kernel_info.h"
#include <filesystem>
#include <new>
#ifdef DML_PERF_PROFILE
#include <fmt/format.h>
#endif

namespace dml_ep {


// Forward declarations for file-local helpers defined later in this file.
static TensorContent SnapshotConstantInput(IMLOperatorTensor* tensor);
static bool ConstantInputChanged(const TensorContent& last, IMLOperatorTensor* current_tensor);

// ============================================================================
// Profiling - compiled in only when /DDML_PERF_PROFILE is defined.
// ============================================================================

#ifdef DML_PERF_PROFILE

static inline uint64_t NowNs() noexcept {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

static std::mutex& GetPerfMutex() noexcept {
    static std::mutex s_mutex;
    return s_mutex;
}

// Opens (or returns the already-open) log file at <exe_dir>/dml_perf.log.
// Called under the perf mutex; not thread-safe on its own.
static FILE* GetOrOpenPerfLogFile() noexcept {
    static FILE* s_file = nullptr;
    static bool s_tried = false;
    if (s_tried) return s_file;
    s_tried = true;

    std::wstring exe_path;
    DWORD capacity = MAX_PATH;
    while (true) {
        exe_path.resize(capacity);
        DWORD written = GetModuleFileNameW(nullptr, exe_path.data(), capacity);
        if (written == 0) return s_file;
        if (written < capacity) { exe_path.resize(written); break; }
        capacity *= 2;
    }
    std::filesystem::path log_path = std::filesystem::path(exe_path).parent_path() / L"dml_perf.log";

    s_file = _wfopen(log_path.c_str(), L"a");
    if (s_file != nullptr) {
        fmt::println(s_file, "=== DML_PERF session start ===");
        std::fflush(s_file);
    }
    return s_file;
}

void DmlPerfWriteLogImpl(std::string_view msg) noexcept {
    std::lock_guard<std::mutex> lock(GetPerfMutex());
    FILE* f = GetOrOpenPerfLogFile();
    if (f == nullptr) return;
    fmt::print(f, "{}", msg);
    std::fflush(f);
}

void PrintKernelPerfCounters(const DmlAbiKernel& kernel) noexcept {
    if (kernel.perf != nullptr)
    {
        return;
    }
    const auto& p = *kernel.perf;
    uint64_t calls = p.compute_calls.load(std::memory_order_relaxed);
    if (calls == 0) return;
    std::lock_guard<std::mutex> lock(GetPerfMutex());
    FILE* f = GetOrOpenPerfLogFile();
    if (f == nullptr) return;
    fmt::println(f,
        "[DML_PERF] op={:<30} calls={} "
        "ctx_ctor={:6} us  pre_trans={:6} us  kern={:6} us  post_trans={:6} us  "
        "shape_chk={:6} us  lazy_inits={}  shape_changes={}",
        kernel.operator_name,
        calls,
        p.ns_context_construction.load(std::memory_order_relaxed) / 1000 / std::max(calls, 1ull),
        p.ns_transition_pre.load(std::memory_order_relaxed)       / 1000 / std::max(calls, 1ull),
        p.ns_kernel_compute.load(std::memory_order_relaxed)       / 1000 / std::max(calls, 1ull),
        p.ns_transition_post.load(std::memory_order_relaxed)      / 1000 / std::max(calls, 1ull),
        p.ns_shape_change_check.load(std::memory_order_relaxed)   / 1000 / std::max(calls, 1ull),
        p.lazy_inits.load(std::memory_order_relaxed),
        p.shape_changes_detected.load(std::memory_order_relaxed));
    std::fflush(f);
}

// Local helper macros - used only inside DmlAbiKernel_Compute.
// DMLPERF_CTX  : declare the perf pointer for this call.
// DMLPERF_INC  : increment a counter by 1.
// DMLPERF_T0   : record start time into <name>_t0.
// DMLPERF_ADD  : accumulate elapsed ns since <name>_t0 into a counter.
#define DMLPERF_CTX(k)          DmlAbiKernelPerfCounters* _perf = (k)->perf.get()
#define DMLPERF_INC(f)          do { if (_perf != nullptr) _perf->f.fetch_add(1, std::memory_order_relaxed); } while(0)
#define DMLPERF_T0(name)        const uint64_t name##_t0 = (_perf != nullptr) ? NowNs() : 0
#define DMLPERF_ADD(field,name) do { if (_perf != nullptr) _perf->field.fetch_add(NowNs() - name##_t0, std::memory_order_relaxed); } while(0)

#else  // DML_PERF_PROFILE not defined - all macros are no-ops

#define DMLPERF_CTX(k)
#define DMLPERF_INC(f)
#define DMLPERF_T0(name)
#define DMLPERF_ADD(field,name)

#endif // DML_PERF_PROFILE

static MLOperatorTensorDataType ConvertToMLOperatorTensorDataType(ONNXTensorElementDataType onnx_type);

// Pre-fetches a single tensor attribute by name from kernel_info, copying its data into
// a PreFetchedTensorAttr with owned plain bytes. This is the ONE place where the unsafe
// Fetches a single named tensor attribute via the ORT C API (KernelInfoGetAttribute_tensor).
// Returns nullopt if the attribute doesn't exist or isn't a tensor.
// Fully ABI-safe: no protobuf access, no reinterpret_cast of OrtKernelInfo.
static std::optional<PreFetchedTensorAttr>
TryFetchTensorAttribute(const OrtKernelInfo* kernel_info, const OrtApi* ort_api, const char* name) {
    OrtAllocator* allocator = nullptr;
    ort_api->GetAllocatorWithDefaultOptions(&allocator);

    OrtValue* tensor_value = nullptr;
    OrtStatus* status = ort_api->KernelInfoGetAttribute_tensor(kernel_info, name, allocator, &tensor_value);
    if (status != nullptr) {
        ort_api->ReleaseStatus(status);
        return std::nullopt;  // attribute absent or not a tensor type
    }
    if (!tensor_value) return std::nullopt;

    PreFetchedTensorAttr result;
    OrtTensorTypeAndShapeInfo* type_shape_info = nullptr;
    ort_api->GetTensorTypeAndShape(tensor_value, &type_shape_info);

    ONNXTensorElementDataType elem_type{};
    ort_api->GetTensorElementType(type_shape_info, &elem_type);
    result.data_type = ConvertToMLOperatorTensorDataType(elem_type);

    size_t dim_count = 0;
    ort_api->GetDimensionsCount(type_shape_info, &dim_count);
    std::vector<int64_t> dims(dim_count);
    ort_api->GetDimensions(type_shape_info, dims.data(), dim_count);
    for (int64_t d : dims) result.shape.push_back(static_cast<uint32_t>(d));

    size_t elem_count = 0;
    ort_api->GetTensorShapeElementCount(type_shape_info, &elem_count);
    ort_api->ReleaseTensorTypeAndShapeInfo(type_shape_info);

    const void* raw_data = nullptr;
    ort_api->GetTensorData(tensor_value, &raw_data);
    size_t elem_size = 0;
    switch (elem_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   elem_size = 4; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  elem_size = 8; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    elem_size = 1; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: elem_size = 2; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  elem_size = 4; break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:  elem_size = 8; break;
        default: elem_size = 4; break;
    }
    size_t total_bytes = elem_count * elem_size;
    if (raw_data && total_bytes > 0) {
        result.raw_bytes.assign(
            reinterpret_cast<const std::byte*>(raw_data),
            reinterpret_cast<const std::byte*>(raw_data) + total_bytes);
    }

    ort_api->ReleaseValue(tensor_value);
    return result;
}

// Pre-fetches tensor-typed attributes from kernel_info into an ABI-safe cache map.
// Only probes attributes listed in tensor_attribute_names — fully ABI-safe (ORT C API only).
// AttributeValue cannot hold tensor types (GetDefaultAttributes rejects them), so tensor
// attribute names cannot be derived from defaultAttributes and must be provided explicitly.
// In this codebase, only ConstantOfShape has a tensor-typed ONNX attribute ("value").
std::unordered_map<std::string, PreFetchedTensorAttr>
FetchAllTensorAttributes(
    const OrtKernelInfo* kernel_info,
    const OrtApi* ort_api,
    const std::vector<std::string>& tensor_attribute_names) {
    std::unordered_map<std::string, PreFetchedTensorAttr> cache;
    if (!kernel_info || !ort_api) return cache;

    for (const auto& attr_name : tensor_attribute_names) {
        auto fetched = TryFetchTensorAttribute(kernel_info, ort_api, attr_name.c_str());
        if (fetched) {
            cache.emplace(attr_name, std::move(*fetched));
        }
    }
    return cache;
}

// Helper function to convert ONNX tensor element type to ML operator tensor data type
static MLOperatorTensorDataType ConvertToMLOperatorTensorDataType(ONNXTensorElementDataType onnx_type) {
    switch (onnx_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:       return MLOperatorTensorDataType::Float;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:       return MLOperatorTensorDataType::UInt8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:        return MLOperatorTensorDataType::Int8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:      return MLOperatorTensorDataType::UInt16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:       return MLOperatorTensorDataType::Int16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:       return MLOperatorTensorDataType::Int32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:       return MLOperatorTensorDataType::Int64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:      return MLOperatorTensorDataType::String;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:        return MLOperatorTensorDataType::Bool;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:     return MLOperatorTensorDataType::Float16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:      return MLOperatorTensorDataType::Double;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:      return MLOperatorTensorDataType::UInt32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:      return MLOperatorTensorDataType::UInt64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:   return MLOperatorTensorDataType::Complex64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:  return MLOperatorTensorDataType::Complex128;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:    return MLOperatorTensorDataType::TensorProto_DataType_BFLOAT16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN: return MLOperatorTensorDataType::TensorProto_DataType_FLOAT8E4M3FN;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ: return MLOperatorTensorDataType::TensorProto_DataType_FLOAT8E4M3FNUZ;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2: return MLOperatorTensorDataType::TensorProto_DataType_FLOAT8E5M2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ: return MLOperatorTensorDataType::TensorProto_DataType_FLOAT8E5M2FNUZ;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4:       return MLOperatorTensorDataType::UInt4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:        return MLOperatorTensorDataType::Int4;
        default:                                        return MLOperatorTensorDataType::Undefined;
    }
}

// ============================================================================
// AbiSafeTensor - IMLOperatorTensor
// ============================================================================

AbiSafeTensor::AbiSafeTensor(
    const OrtValue* ort_value,
    const OrtApi* ort_api,
    const PluginDmlExecutionProviderImpl* execution_provider,
    bool is_internal_operator)
    : ort_value_(ort_value)
    , ort_api_(ort_api)
    , execution_provider_(execution_provider)
    , is_internal_operator_(is_internal_operator)
{
}

uint32_t AbiSafeTensor::GetDimensionCount() const noexcept {
    if (!ort_value_ || !ort_api_) return 0;

    OrtTensorTypeAndShapeInfo* type_shape_info = nullptr;
    OrtStatus* status = ort_api_->GetTensorTypeAndShape(ort_value_, &type_shape_info);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return 0;
    }

    size_t dim_count = 0;
    status = ort_api_->GetDimensionsCount(type_shape_info, &dim_count);
    ort_api_->ReleaseTensorTypeAndShapeInfo(type_shape_info);

    if (status) {
        ort_api_->ReleaseStatus(status);
        return 0;
    }

    return static_cast<uint32_t>(dim_count);
}

HRESULT AbiSafeTensor::GetShape(uint32_t dimensionCount, uint32_t* dimensions) const noexcept {
    if (!dimensions && dimensionCount > 0) return E_POINTER;  // Allow null for scalar (0-dim) tensors
    if (!ort_value_ || !ort_api_) return E_FAIL;

    OrtTensorTypeAndShapeInfo* type_shape_info = nullptr;
    OrtStatus* status = ort_api_->GetTensorTypeAndShape(ort_value_, &type_shape_info);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    // Get dimensions as int64_t
    size_t dim_count = 0;
    ort_api_->GetDimensionsCount(type_shape_info, &dim_count);

    if (dim_count != dimensionCount) {
        ort_api_->ReleaseTensorTypeAndShapeInfo(type_shape_info);
        return E_INVALIDARG;
    }

    shape_cache_.resize(dim_count);
    status = ort_api_->GetDimensions(type_shape_info, shape_cache_.data(), dim_count);
    ort_api_->ReleaseTensorTypeAndShapeInfo(type_shape_info);

    if (status) {
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    // Convert int64_t to uint32_t
    for (size_t i = 0; i < dim_count; ++i) {
        dimensions[i] = static_cast<uint32_t>(shape_cache_[i]);
    }

    return S_OK;
}

MLOperatorTensorDataType AbiSafeTensor::GetTensorDataType() const noexcept {
    if (!ort_value_ || !ort_api_) return MLOperatorTensorDataType::Undefined;

    OrtTensorTypeAndShapeInfo* type_shape_info = nullptr;
    OrtStatus* status = ort_api_->GetTensorTypeAndShape(ort_value_, &type_shape_info);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return MLOperatorTensorDataType::Undefined;
    }

    ONNXTensorElementDataType elem_type;
    status = ort_api_->GetTensorElementType(type_shape_info, &elem_type);
    ort_api_->ReleaseTensorTypeAndShapeInfo(type_shape_info);

    if (status) {
        ort_api_->ReleaseStatus(status);
        return MLOperatorTensorDataType::Undefined;
    }

    // Map ONNX types to ML types
    switch (elem_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return MLOperatorTensorDataType::Float;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return MLOperatorTensorDataType::UInt8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return MLOperatorTensorDataType::Int8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return MLOperatorTensorDataType::UInt16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return MLOperatorTensorDataType::Int16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return MLOperatorTensorDataType::Int32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return MLOperatorTensorDataType::Int64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: return MLOperatorTensorDataType::String;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return MLOperatorTensorDataType::Bool;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return MLOperatorTensorDataType::Float16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return MLOperatorTensorDataType::Double;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return MLOperatorTensorDataType::UInt32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return MLOperatorTensorDataType::UInt64;
        default: return MLOperatorTensorDataType::Undefined;
    }
}

bool AbiSafeTensor::IsCpuData() const noexcept {
    if (!ort_value_ || !ort_api_) return true; // Default to CPU if we can't check

    const OrtMemoryInfo* mem_info = nullptr;
    OrtStatus* status = ort_api_->GetTensorMemoryInfo(ort_value_, &mem_info);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return true; // Default to CPU on error
    }

    // Check name — "Cpu" means CPU memory
    const char* name = nullptr;
    status = ort_api_->MemoryInfoGetName(mem_info, &name);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return true;
    }
    if (name == nullptr || strcmp(name, "Cpu") == 0) {
        return true;
    }

    // Also treat OrtMemTypeCPUInput/OrtMemTypeCPUOutput as CPU, matching TensorWrapper::IsCpuData
    OrtMemType mem_type = OrtMemTypeDefault;
    status = ort_api_->MemoryInfoGetMemType(mem_info, &mem_type);
    if (!status && (mem_type == OrtMemTypeCPUInput || mem_type == OrtMemTypeCPUOutput)) {
        return true;
    }
    if (status) ort_api_->ReleaseStatus(status);

    return false;
}

bool AbiSafeTensor::IsDataInterface() const noexcept {
    // GPU tensors use data interfaces, CPU tensors don't
    return !IsCpuData();
}

void* AbiSafeTensor::GetData() noexcept {
    if (!ort_value_ || !ort_api_) return nullptr;

    // Only return data pointer for CPU tensors
    if (!IsCpuData()) return nullptr;

    void* data_ptr = nullptr;
    OrtStatus* status = ort_api_->GetTensorMutableData(const_cast<OrtValue*>(ort_value_), &data_ptr);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return nullptr;
    }

    return data_ptr;
}

void AbiSafeTensor::GetDataInterface(IUnknown** dataInterface) noexcept {
    if (!dataInterface) {
        return;
    }
    *dataInterface = nullptr;

    if (!ort_value_ || !ort_api_) {
        return;
    }

    // Verify this is actually a GPU tensor
    bool isCpu = IsCpuData();
    if (isCpu) {
        return;  // CPU tensors don't have data interfaces
    }

    // Get raw data pointer from OrtValue
    // For GPU tensors allocated by the DML allocator, this is actually a PluginDmlAllocationInfo*
    // Note: We need to cast away const because the C API's GetTensorMutableData doesn't have a const variant,
    // but we're only reading the allocation pointer, not modifying the data.
    void* data_ptr = nullptr;
    OrtStatus* status = ort_api_->GetTensorMutableData(const_cast<OrtValue*>(ort_value_), &data_ptr);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return;
    }

    if (!data_ptr) {
        return;
    }

    // GetTensorMutableData returns the raw data pointer (PluginDmlAllocationInfo* cast to void*).
    // We need to go through GetABIDataInterface so external operators receive ID3D12Resource*
    // (not the internal PluginDmlAllocationInfo*). Skipping this translation is the root cause
    // of output corruption: the DML plugin operator casts the returned IUnknown* to ID3D12Resource*
    // and binds wrong GPU memory when it gets PluginDmlAllocationInfo* instead.
    IUnknown* allocation = execution_provider_->GetAllocationFromDataPointer(data_ptr);
    if (!allocation) {
        return;
    }

    // Translate to the ABI object the external operator expects (ID3D12Resource* for non-internal ops).
    execution_provider_->GetABIDataInterface(is_internal_operator_, allocation, dataInterface);
    allocation->Release();  // GetAllocationFromDataPointer already AddRef'd; GetABIDataInterface AddRef's the output
}

// ============================================================================
// AbiSafeKernelContext - IMLOperatorKernelContext
// ============================================================================

AbiSafeKernelContext::AbiSafeKernelContext(
    OrtKernelContext* kernel_context,
    const OrtApi* ort_api,
    const PluginDmlExecutionProviderImpl* execution_provider,
    bool is_internal_operator,
    const std::vector<std::vector<uint32_t>>* inferred_output_shapes,
    IMLOperatorShapeInferrer* shape_inferrer,
    const std::vector<uint32_t>* required_constant_cpu_inputs,
    const AttributeMap* default_attributes,
    const OrtKernelInfo* kernel_info)
    : kernel_context_(kernel_context)
    , ort_api_(ort_api)
    , execution_provider_(execution_provider)
    , is_internal_operator_(is_internal_operator)
    , inferred_output_shapes_(inferred_output_shapes)
    , shape_inferrer_(shape_inferrer)
    , required_constant_cpu_inputs_(required_constant_cpu_inputs)
    , default_attributes_(default_attributes)
    , kernel_info_(kernel_info)
{
    // Get the ABI execution interface and store the winml provider for resource transitions.
    // Mirrors PluginOpKernelContextWrapper constructor which calls GetABIExecutionInterfaceAndInvalidateState
    // and stores m_winmlProvider for use in TransitionResourcesForOperatorIfRequired.
    if (execution_provider_) {
        const_cast<PluginDmlExecutionProviderImpl*>(execution_provider_)->QueryInterface(
            IID_PPV_ARGS(&winml_provider_));

        if (winml_provider_) {
            winml_provider_->GetABIExecutionInterfaceAndInvalidateState(
                is_internal_operator_,
                &abi_execution_object_);
        }
    }
}

void AbiSafeKernelContext::TransitionResourcesForOperatorIfRequired(bool isBeforeOp) {
    // Mirrors PluginOpKernelContextWrapper::TransitionResourcesForOperatorIfRequired.
    // External (non-internal) DML operators require D3D12 resources to be in COMMON state before
    // execution and UAV state after. Without these transitions, GPU reads/writes operate on
    // resources in the wrong state, producing corrupted or stale output.
    if (!winml_provider_ || !winml_provider_->TransitionsRequiredForOperator(is_internal_operator_)) {
        return;
    }

    std::vector<IUnknown*> resourcesToTransition;

    // Collect input resources
    size_t input_count = 0;
    ort_api_->KernelContext_GetInputCount(kernel_context_, &input_count);
    for (size_t i = 0; i < input_count; ++i) {
        const OrtValue* input_value = nullptr;
        OrtStatus* s = ort_api_->KernelContext_GetInput(kernel_context_, i, &input_value);
        if (s) { ort_api_->ReleaseStatus(s); continue; }
        if (!input_value) continue;

        // Check if this is a GPU tensor
        const OrtMemoryInfo* mem_info = nullptr;
        s = ort_api_->GetTensorMemoryInfo(input_value, &mem_info);
        if (s) { ort_api_->ReleaseStatus(s); continue; }

        const char* name = nullptr;
        ort_api_->MemoryInfoGetName(mem_info, &name);
        if (name && strcmp(name, "Cpu") == 0) continue;

        OrtMemType mem_type = OrtMemTypeDefault;
        ort_api_->MemoryInfoGetMemType(mem_info, &mem_type);
        if (mem_type == OrtMemTypeCPUInput || mem_type == OrtMemTypeCPUOutput) continue;

        // Get the D3D12 resource via the ABI translation path
        void* data_ptr = nullptr;
        s = ort_api_->GetTensorMutableData(const_cast<OrtValue*>(input_value), &data_ptr);
        if (s) { ort_api_->ReleaseStatus(s); continue; }
        if (!data_ptr) continue;

        IUnknown* allocation = execution_provider_->GetAllocationFromDataPointer(data_ptr);
        if (!allocation) continue;

        IUnknown* abiResource = nullptr;
        execution_provider_->GetABIDataInterface(is_internal_operator_, allocation, &abiResource);
        allocation->Release();
        if (abiResource) {
            resourcesToTransition.push_back(abiResource);
        }
    }

    // Collect output resources — use output_tensor_cache_ populated by GetOutputTensor calls.
    // We cannot call KernelContext_GetOutput here without a shape (would re-allocate the tensor).
    for (auto& t : output_tensor_cache_) {
        if (t && t->IsDataInterface()) {
            IUnknown* resource = nullptr;
            t->GetDataInterface(&resource);
            if (resource) {
                resourcesToTransition.push_back(resource);
            }
        }
    }

    if (!resourcesToTransition.empty()) {
        winml_provider_->TransitionResourcesForOperator(
            isBeforeOp,
            static_cast<uint32_t>(resourcesToTransition.size()),
            resourcesToTransition.data());
    }

    // Release refs acquired by GetABIDataInterface
    for (IUnknown* r : resourcesToTransition) {
        if (r) r->Release();
    }
}

HRESULT AbiSafeKernelContext::GetInputTensor(uint32_t inputIndex, IMLOperatorTensor** tensor) const noexcept {
    if (!tensor) return E_POINTER;
    *tensor = nullptr;

    if (!kernel_context_ || !ort_api_) return E_FAIL;

    const OrtValue* input_value = nullptr;
    OrtStatus* status = ort_api_->KernelContext_GetInput(kernel_context_, inputIndex, &input_value);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    if (!input_value) {
        // Optional input that doesn't exist
        return S_OK;
    }

    // Create ABI-safe tensor wrapper and cache it to keep it alive
    auto abi_tensor = Microsoft::WRL::Make<AbiSafeTensor>(input_value, ort_api_, execution_provider_, is_internal_operator_);
    tensor_cache_.push_back(abi_tensor);
    *tensor = abi_tensor.Get();
    abi_tensor->AddRef(); // Caller gets a reference

    return S_OK;
}

HRESULT AbiSafeKernelContext::GetOutputTensor(uint32_t outputIndex, IMLOperatorTensor** tensor) noexcept {
    if (!tensor) return E_POINTER;
    *tensor = nullptr;

    if (!kernel_context_ || !ort_api_) return E_FAIL;

    // Use pre-inferred output shapes from graph compilation (source of truth)
    // If there are dynamic dimensions, fill them from runtime input shapes
    if (inferred_output_shapes_ && outputIndex < inferred_output_shapes_->size()) {
        const auto& shape = (*inferred_output_shapes_)[outputIndex];

        if (!shape.empty()) {
            // Check if shape contains dynamic dimensions (-1 cast to uint32_t becomes 0xFFFFFFFF)
            bool has_dynamic = false;
            for (uint32_t dim : shape) {
                if (dim == static_cast<uint32_t>(-1)) {
                    has_dynamic = true;
                    break;
                }
            }

            if (!has_dynamic) {
                // All dimensions are static - use shape as-is
                return GetOutputTensor(outputIndex, static_cast<uint32_t>(shape.size()), shape.data(), tensor);
            } else {
                // Graph shape has dynamic dimensions - fill them from runtime input
                // Assumption: dynamic dimensions match corresponding dimensions from input 0
                const OrtValue* input0 = nullptr;
                OrtStatus* input_status = ort_api_->KernelContext_GetInput(kernel_context_, 0, &input0);
                if (input_status == nullptr && input0 != nullptr) {
                    OrtTensorTypeAndShapeInfo* input_info = nullptr;
                    input_status = ort_api_->GetTensorTypeAndShape(input0, &input_info);
                    if (input_status == nullptr) {
                        size_t input_dim_count = 0;
                        ort_api_->GetDimensionsCount(input_info, &input_dim_count);

                        if (input_dim_count == shape.size()) {
                            std::vector<int64_t> input_dims(input_dim_count);
                            ort_api_->GetDimensions(input_info, input_dims.data(), input_dim_count);

                            // Fill dynamic dimensions from input
                            std::vector<uint32_t> filled_shape(shape.size());

                            for (size_t i = 0; i < shape.size(); ++i) {
                                if (shape[i] == static_cast<uint32_t>(-1)) {
                                    filled_shape[i] = static_cast<uint32_t>(input_dims[i]);
                                } else {
                                    filled_shape[i] = shape[i];
                                }
                            }

                            ort_api_->ReleaseTensorTypeAndShapeInfo(input_info);
                            return GetOutputTensor(outputIndex, static_cast<uint32_t>(filled_shape.size()), filled_shape.data(), tensor);
                        }
                        ort_api_->ReleaseTensorTypeAndShapeInfo(input_info);
                    }
                    if (input_status) ort_api_->ReleaseStatus(input_status);
                } else if (input_status) {
                    ort_api_->ReleaseStatus(input_status);
                }

                // Fallback: couldn't fill from input, let ONNX Runtime infer
                // (will fall through to the end of the function)
            }
        }
    }

    // For internal operators (MemcpyToHost/FromHost), output shape = input 0 shape
    if (is_internal_operator_) {
        const OrtValue* input_value = nullptr;
        OrtStatus* input_status = ort_api_->KernelContext_GetInput(kernel_context_, 0, &input_value);
        if (input_status == nullptr && input_value != nullptr) {
            OrtTensorTypeAndShapeInfo* input_info = nullptr;
            input_status = ort_api_->GetTensorTypeAndShape(input_value, &input_info);
            if (input_status == nullptr) {
                size_t dim_count = 0;
                ort_api_->GetDimensionsCount(input_info, &dim_count);
                std::vector<int64_t> input_dims(dim_count);
                if (dim_count > 0) {
                    ort_api_->GetDimensions(input_info, input_dims.data(), dim_count);
                }
                ort_api_->ReleaseTensorTypeAndShapeInfo(input_info);

                // Convert to uint32_t
                std::vector<uint32_t> output_shape(dim_count);
                for (size_t i = 0; i < dim_count; ++i) {
                    output_shape[i] = static_cast<uint32_t>(input_dims[i]);
                }

                return GetOutputTensor(outputIndex, static_cast<uint32_t>(dim_count), output_shape.data(), tensor);
            } else {
                ort_api_->ReleaseStatus(input_status);
            }
        } else if (input_status) {
            ort_api_->ReleaseStatus(input_status);
        }
    }

    // For operators with shape inferrer (e.g., Reshape), call the inferrer to compute output shapes
    if (shape_inferrer_) {
        auto inference_context = Microsoft::WRL::Make<AbiSafeShapeInferenceContext>(
            kernel_context_,
            ort_api_,
            default_attributes_,
            execution_provider_,
            kernel_info_  // Pass kernel_info so attribute reads use actual node values, not just defaults
        );

        HRESULT hr = shape_inferrer_->InferOutputShapes(inference_context.Get());
        if (SUCCEEDED(hr)) {
            const auto& inferred_shapes = inference_context->GetInferredOutputShapes();
            if (outputIndex < inferred_shapes.size() && !inferred_shapes[outputIndex].empty()) {
                const auto& shape = inferred_shapes[outputIndex];
                return GetOutputTensor(outputIndex, static_cast<uint32_t>(shape.size()), shape.data(), tensor);
            }
        }
    }

    // Get output without specifying shape - let ONNX Runtime infer at runtime
    // This handles: dynamic shapes, empty inferred shapes, or when shape inference is unavailable
    OrtValue* output_value = nullptr;
    OrtStatus* status = ort_api_->KernelContext_GetOutput(kernel_context_, outputIndex, nullptr, 0, &output_value);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    if (!output_value) {
        // Optional output
        return S_OK;
    }

    // Log what shape ONNX Runtime actually created
    OrtTensorTypeAndShapeInfo* output_info = nullptr;
    status = ort_api_->GetTensorTypeAndShape(output_value, &output_info);
    if (!status) {
        size_t dim_count = 0;
        ort_api_->GetDimensionsCount(output_info, &dim_count);
        std::vector<int64_t> dims(dim_count);
        if (dim_count > 0) {
            ort_api_->GetDimensions(output_info, dims.data(), dim_count);
        }

        ort_api_->ReleaseTensorTypeAndShapeInfo(output_info);
    } else {
        ort_api_->ReleaseStatus(status);
    }

    // Create ABI-safe tensor wrapper and cache it for lifetime + post-op resource transitions
    auto abi_tensor = Microsoft::WRL::Make<AbiSafeTensor>(output_value, ort_api_, execution_provider_, is_internal_operator_);

    output_tensor_cache_.push_back(abi_tensor);
    *tensor = abi_tensor.Get();
    abi_tensor->AddRef(); // Caller gets a reference

    return S_OK;
}

HRESULT AbiSafeKernelContext::GetOutputTensor(
    uint32_t outputIndex,
    uint32_t dimensionCount,
    const uint32_t* dimensionSizes,
    IMLOperatorTensor** tensor) noexcept {

    if (!tensor) return E_POINTER;
    if (!dimensionSizes && dimensionCount > 0) return E_POINTER;  // Allow null for scalar tensors
    *tensor = nullptr;

    if (!kernel_context_ || !ort_api_) return E_FAIL;

    // Convert uint32_t dimensions to int64_t
    std::vector<int64_t> shape(dimensionCount);
    for (uint32_t i = 0; i < dimensionCount; ++i) {
        shape[i] = static_cast<int64_t>(dimensionSizes[i]);
    }

    // Get output with specified shape
    OrtValue* output_value = nullptr;
    OrtStatus* status = ort_api_->KernelContext_GetOutput(
        kernel_context_,
        outputIndex,
        shape.data(),
        dimensionCount,
        &output_value);

    if (status) {
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    if (!output_value) {
        // Optional output
        return S_OK;
    }

    // Create ABI-safe tensor wrapper and cache it for lifetime + post-op resource transitions
    auto abi_tensor = Microsoft::WRL::Make<AbiSafeTensor>(output_value, ort_api_, execution_provider_, is_internal_operator_);

    output_tensor_cache_.push_back(abi_tensor);
    *tensor = abi_tensor.Get();
    abi_tensor->AddRef(); // Caller gets a reference

    return S_OK;
}

HRESULT AbiSafeKernelContext::AllocateTemporaryData(size_t size, IUnknown** data) const noexcept {
    if (!data) return E_POINTER;
    *data = nullptr;

    // Allow size == 0 to match ONNX Runtime behavior (allocator may handle it)

    // Create memory info for DirectML execution provider's device allocator
    OrtMemoryInfo* mem_info = nullptr;
    OrtStatus* status = ort_api_->CreateMemoryInfo(
        "DirectMLExecutionProvider", // The execution provider name
        OrtAllocatorType::OrtDeviceAllocator,
        0, // device id
        OrtMemType::OrtMemTypeDefault,
        &mem_info
    );

    if (status) {
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    // Get allocator from kernel context
    OrtAllocator* allocator = nullptr;
    status = ort_api_->KernelContext_GetAllocator(kernel_context_, mem_info, &allocator);
    ort_api_->ReleaseMemoryInfo(mem_info);

    if (status) {
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    if (!allocator) {
        return E_FAIL;
    }

    // Allocate memory
    void* allocated_memory = nullptr;
    status = ort_api_->AllocatorAlloc(allocator, size, &allocated_memory);
    ort_api_->ReleaseAllocator(allocator);

    if (status) {
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    if (!allocated_memory) {
        return E_OUTOFMEMORY;
    }

    // DirectML operators expect raw memory pointers cast to IUnknown*
    // This matches the existing behavior in the DML codebase
    *data = reinterpret_cast<IUnknown*>(allocated_memory);

    return S_OK;
}

void AbiSafeKernelContext::GetExecutionInterface(IUnknown** executionInterface) const noexcept {
    if (!executionInterface) return;

    if (abi_execution_object_) {
        abi_execution_object_.CopyTo(executionInterface);
    } else {
        *executionInterface = nullptr;
    }
}

// IMLOperatorKernelContextPrivate methods
bool AbiSafeKernelContext::IsSequenceInputTensor(uint32_t inputIndex) const noexcept {
    if (!kernel_context_ || !ort_api_) {
        return false;
    }

    // Get the input OrtValue
    const OrtValue* input_value = nullptr;
    OrtStatus* status = ort_api_->KernelContext_GetInput(kernel_context_, inputIndex, &input_value);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return false;
    }

    if (!input_value) {
        return false;  // Optional input not provided
    }

    // Check the type
    ONNXType value_type = ONNX_TYPE_UNKNOWN;
    status = ort_api_->GetValueType(input_value, &value_type);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return false;
    }

    bool is_sequence = (value_type == ONNX_TYPE_SEQUENCE);
    return is_sequence;
}

HRESULT AbiSafeKernelContext::GetSequenceInputInfo(uint32_t inputIndex, uint32_t* inputCount, MLOperatorTensorDataType* dataType) const noexcept {
    if (!inputCount || !dataType) {
        return E_POINTER;
    }
    if (!kernel_context_ || !ort_api_) {
        return E_FAIL;
    }

    // Get the input OrtValue
    const OrtValue* input_value = nullptr;
    OrtStatus* status = ort_api_->KernelContext_GetInput(kernel_context_, inputIndex, &input_value);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    if (!input_value) {
        return E_FAIL;  // Optional input not provided
    }

    // Verify it's a sequence
    ONNXType value_type = ONNX_TYPE_UNKNOWN;
    status = ort_api_->GetValueType(input_value, &value_type);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    if (value_type != ONNX_TYPE_SEQUENCE) {
        return E_INVALIDARG;
    }

    // Get the count of elements in the sequence
    size_t count = 0;
    status = ort_api_->GetValueCount(input_value, &count);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }
    *inputCount = static_cast<uint32_t>(count);

    // Get the data type from the sequence's type info
    OrtTypeInfo* type_info = nullptr;
    status = ort_api_->GetTypeInfo(input_value, &type_info);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    const OrtSequenceTypeInfo* sequence_info = nullptr;
    status = ort_api_->CastTypeInfoToSequenceTypeInfo(type_info, &sequence_info);
    if (status) {
        ort_api_->ReleaseTypeInfo(type_info);
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    OrtTypeInfo* element_type_info = nullptr;
    status = ort_api_->GetSequenceElementType(sequence_info, &element_type_info);
    if (status) {
        ort_api_->ReleaseTypeInfo(type_info);
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    // Get the tensor element data type
    const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
    status = ort_api_->CastTypeInfoToTensorInfo(element_type_info, &tensor_info);
    if (status) {
        ort_api_->ReleaseTypeInfo(element_type_info);
        ort_api_->ReleaseTypeInfo(type_info);
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    ONNXTensorElementDataType element_data_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    status = ort_api_->GetTensorElementType(tensor_info, &element_data_type);
    ort_api_->ReleaseTypeInfo(element_type_info);
    ort_api_->ReleaseTypeInfo(type_info);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    // Convert ONNXTensorElementDataType to MLOperatorTensorDataType
    *dataType = static_cast<MLOperatorTensorDataType>(element_data_type);

    return S_OK;
}

HRESULT AbiSafeKernelContext::GetSequenceInputTensor(uint32_t inputIndex, uint32_t sequenceIndex, IMLOperatorTensor** tensor) const noexcept {
    if (!tensor) {
        return E_POINTER;
    }
    *tensor = nullptr;

    if (!kernel_context_ || !ort_api_) {
        return E_FAIL;
    }

    // Get the input sequence OrtValue
    const OrtValue* input_value = nullptr;
    OrtStatus* status = ort_api_->KernelContext_GetInput(kernel_context_, inputIndex, &input_value);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    if (!input_value) {
        return E_FAIL;
    }

    // Get allocator for extracting sequence element
    OrtAllocator* allocator = nullptr;
    status = ort_api_->GetAllocatorWithDefaultOptions(&allocator);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    // Get the element at sequenceIndex
    OrtValue* element_value = nullptr;
    status = ort_api_->GetValue(input_value, sequenceIndex, allocator, &element_value);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    if (!element_value) {
        return E_FAIL;
    }

    // Create AbiSafeTensor wrapper for the element
    auto abi_tensor = Microsoft::WRL::Make<AbiSafeTensor>(element_value, ort_api_, execution_provider_, is_internal_operator_);
    if (!abi_tensor) {
        ort_api_->ReleaseValue(element_value);
        return E_OUTOFMEMORY;
    }

    *tensor = abi_tensor.Detach();
    // Note: element_value is now owned by the AbiSafeTensor, it will be released when the tensor is released
    return S_OK;
}

HRESULT AbiSafeKernelContext::PrepareSequenceOutput(uint32_t outputIndex, MLOperatorTensorDataType dataType) const noexcept {
    // This method is called to prepare a sequence output
    // In ONNX Runtime, sequence outputs are typically created on-demand when GetSequenceOutputTensor is called
    // So we don't need to do anything here
    return S_OK;
}

HRESULT AbiSafeKernelContext::GetSequenceOutputTensor(uint32_t outputIndex, uint32_t sequenceIndex,
    MLOperatorTensorDataType dataType, uint32_t dimensions, const uint32_t* dimensionSizes,
    bool gpuOutput, IMLOperatorTensor** tensor) const noexcept {
    if (!tensor) {
        return E_POINTER;
    }
    *tensor = nullptr;

    if (!kernel_context_ || !ort_api_) {
        return E_FAIL;
    }

    // Get allocator based on GPU/CPU output
    OrtAllocator* allocator = nullptr;
    OrtStatus* status = nullptr;

    if (gpuOutput) {
        // Get GPU allocator from execution provider
        OrtMemoryInfo* mem_info = nullptr;
        status = ort_api_->CreateMemoryInfo("DML", OrtDeviceAllocator, 0, OrtMemTypeDefault, &mem_info);
        if (status) {
            ort_api_->ReleaseStatus(status);
            return E_FAIL;
        }

        status = ort_api_->KernelContext_GetAllocator(kernel_context_, mem_info, &allocator);
        ort_api_->ReleaseMemoryInfo(mem_info);
        if (status) {
            ort_api_->ReleaseStatus(status);
            return E_FAIL;
        }
    } else {
        // Get CPU allocator
        status = ort_api_->GetAllocatorWithDefaultOptions(&allocator);
        if (status) {
            ort_api_->ReleaseStatus(status);
            return E_FAIL;
        }
    }

    // Convert dimensions to int64_t
    std::vector<int64_t> shape(dimensions);
    for (uint32_t i = 0; i < dimensions; ++i) {
        shape[i] = static_cast<int64_t>(dimensionSizes[i]);
    }

    // Create tensor element
    OrtValue* element_value = nullptr;
    ONNXTensorElementDataType onnx_type = static_cast<ONNXTensorElementDataType>(dataType);
    status = ort_api_->CreateTensorAsOrtValue(allocator, shape.data(), shape.size(), onnx_type, &element_value);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    if (!element_value) {
        return E_FAIL;
    }

    // DEFICIENCY: The C API does not provide incremental sequence building (no "append to sequence" method)
    // The only way to create a sequence is via CreateValue() which requires all elements upfront
    // This means we cannot properly implement GetSequenceOutputTensor in the ABI-safe path
    //
    // The unsafe path uses m_impl->Output<onnxruntime::TensorSeq>() which gives direct access to
    // the C++ TensorSeq object and can call Add() to incrementally build the sequence
    //
    // Workaround: We create the tensor element and return it wrapped in AbiSafeTensor
    // This works for operators that only read sequence elements (like Memcpy reading input sequences)
    // but will NOT work correctly for operators that write/create sequence outputs
    //
    // If sequence output writing is needed, the operator must fall back to the unsafe path
    auto abi_tensor = Microsoft::WRL::Make<AbiSafeTensor>(element_value, ort_api_, execution_provider_, is_internal_operator_);
    if (!abi_tensor) {
        ort_api_->ReleaseValue(element_value);
        return E_OUTOFMEMORY;
    }

    *tensor = abi_tensor.Detach();
    // Note: element_value is now owned by the AbiSafeTensor
    return S_OK;
}

// ============================================================================
// AbiSafeShapeInferenceContext - IMLOperatorShapeInferenceContext
// ============================================================================

AbiSafeShapeInferenceContext::AbiSafeShapeInferenceContext(
    OrtKernelContext* kernel_context,
    const OrtApi* ort_api,
    const AttributeMap* default_attributes,
    const PluginDmlExecutionProviderImpl* execution_provider,
    const OrtKernelInfo* kernel_info)
    : kernel_context_(kernel_context)
    , ort_api_(ort_api)
    , default_attributes_(default_attributes)
    , execution_provider_(execution_provider)
    , kernel_info_(kernel_info)
{
}

HRESULT AbiSafeShapeInferenceContext::GetAttributeElementCount(
    const char* name,
    MLOperatorAttributeType type,
    uint32_t* elementCount) const noexcept {
    if (!name || !elementCount) {
        return E_POINTER;
    }

    // If kernel_info_ is available, get actual attributes from the ONNX node
    // Otherwise fall back to default_attributes_
    if (kernel_info_ && ort_api_) {
        // Try to get attribute from kernel_info_ using ORT API
        OrtStatus* status = nullptr;

        if (type == MLOperatorAttributeType::Int) {
            // For single Int, try to get as scalar first
            int64_t dummy_value;
            status = ort_api_->KernelInfoGetAttribute_int64(kernel_info_, name, &dummy_value);
            if (!status) {
                // Attribute exists as a scalar int
                *elementCount = 1;
                return S_OK;
            }
            if (status) {
                ort_api_->ReleaseStatus(status);
                status = nullptr;
            }
        }
        else if (type == MLOperatorAttributeType::IntArray) {
            // For IntArray, get the array length
            size_t size = 0;
            status = ort_api_->KernelInfoGetAttributeArray_int64(kernel_info_, name, nullptr, &size);
            if (!status) {
                *elementCount = static_cast<uint32_t>(size);
                return S_OK;
            }
            if (status) {
                ort_api_->ReleaseStatus(status);
                status = nullptr;
            }
        }
        else if (type == MLOperatorAttributeType::Float) {
            // For single Float, try to get as scalar
            float dummy_value;
            status = ort_api_->KernelInfoGetAttribute_float(kernel_info_, name, &dummy_value);
            if (!status) {
                *elementCount = 1;
                return S_OK;
            }
            if (status) {
                ort_api_->ReleaseStatus(status);
                status = nullptr;
            }
        }
        else if (type == MLOperatorAttributeType::FloatArray) {
            // For FloatArray, get the array length
            size_t size = 0;
            status = ort_api_->KernelInfoGetAttributeArray_float(kernel_info_, name, nullptr, &size);
            if (!status) {
                *elementCount = static_cast<uint32_t>(size);
                return S_OK;
            }
            if (status) {
                ort_api_->ReleaseStatus(status);
                status = nullptr;
            }
        }
        else if (type == MLOperatorAttributeType::String) {
            // For String, check if it exists by trying to get its length
            size_t size = 0;
            status = ort_api_->KernelInfoGetAttribute_string(kernel_info_, name, nullptr, &size);
            if (!status) {
                *elementCount = 1;
                return S_OK;
            }
            if (status) {
                ort_api_->ReleaseStatus(status);
                status = nullptr;
            }
        }
        // Note: StringArray is not supported by ORT C API (no KernelInfoGetAttributeArray_string)
        // It will fall through to default_attributes_ if needed

        // Fall through to default_attributes_ if ORT API fails
    }

    // Attributes come from default_attributes_ set during operator registration
    // If attribute doesn't exist, return 0 (this is valid - means attribute not set)
    if (!default_attributes_) {
        *elementCount = 0;
        return S_OK;
    }

    auto it = default_attributes_->find(name);
    if (it == default_attributes_->end()) {
        // Attribute not found - this is valid, return 0
        *elementCount = 0;
        return S_OK;
    }

    // Attribute exists - return element count based on type
    const auto& attr_value = it->second;

    try {
        *elementCount = static_cast<uint32_t>(attr_value.ElementCount());
    } catch (...) {
        *elementCount = 0;
    }

    return S_OK;
}

HRESULT AbiSafeShapeInferenceContext::GetAttribute(
    const char* name,
    MLOperatorAttributeType type,
    uint32_t elementCount,
    size_t elementByteSize,
    void* value) const noexcept {
    if (!name || !value) {
        return E_POINTER;
    }

    // If kernel_info_ is available, get actual attributes from the ONNX node
    if (kernel_info_ && ort_api_) {
        OrtStatus* status = nullptr;

        switch (type) {
            case MLOperatorAttributeType::Int:
                if (elementCount == 1 && elementByteSize == sizeof(int64_t)) {
                    status = ort_api_->KernelInfoGetAttribute_int64(kernel_info_, name, static_cast<int64_t*>(value));
                    if (!status) {
                        return S_OK;
                    }
                }
                break;

            case MLOperatorAttributeType::IntArray:
                {
                    size_t count = elementCount;
                    status = ort_api_->KernelInfoGetAttributeArray_int64(kernel_info_, name, static_cast<int64_t*>(value), &count);
                    if (!status) {
                        return S_OK;
                    }
                }
                break;

            case MLOperatorAttributeType::Float:
                if (elementCount == 1 && elementByteSize == sizeof(float)) {
                    status = ort_api_->KernelInfoGetAttribute_float(kernel_info_, name, static_cast<float*>(value));
                    if (!status) {
                        return S_OK;
                    }
                }
                break;

            case MLOperatorAttributeType::FloatArray:
                {
                    size_t count = elementCount;
                    status = ort_api_->KernelInfoGetAttributeArray_float(kernel_info_, name, static_cast<float*>(value), &count);
                    if (!status) {
                        return S_OK;
                    }
                }
                break;

            case MLOperatorAttributeType::String:
                // String attributes need special handling - they're stored in default_attributes
                // because ORT C API doesn't provide direct string attribute access via KernelInfo
                break;

            default:
                break;
        }

        if (status) {
            ort_api_->ReleaseStatus(status);
        }
        // Fall through to default_attributes_ if ORT API fails
    }

    // Fall back to default_attributes_
    if (!default_attributes_) {
        return E_INVALIDARG;
    }

    auto it = default_attributes_->find(name);
    if (it == default_attributes_->end()) {
        return E_INVALIDARG;
    }

    // Use the AttributeValue::GetAttribute method which handles all types
    const auto& attr_value = it->second;

    try {
        attr_value.GetAttribute(type, elementCount, elementByteSize, value);
        return S_OK;
    } catch (const wil::ResultException& ex) {
        return ex.GetErrorCode();
    } catch (...) {
        return E_FAIL;
    }
}

HRESULT AbiSafeShapeInferenceContext::GetStringAttributeElementLength(
    const char* name,
    uint32_t elementIndex,
    uint32_t* attributeElementByteLength) const noexcept {
    if (!name || !attributeElementByteLength) {
        return E_POINTER;
    }

    // Try to get from kernel_info_ first if available
    if (kernel_info_ && ort_api_ && ort_api_->KernelInfoGetAttribute_string && elementIndex == 0) {
        size_t size = 0;
        OrtStatus* status = ort_api_->KernelInfoGetAttribute_string(kernel_info_, name, nullptr, &size);
        if (!status) {
            *attributeElementByteLength = static_cast<uint32_t>(size);
            return S_OK;
        }
        if (status) {
            ort_api_->ReleaseStatus(status);
        }
    }

    // Fall back to default_attributes_
    if (!default_attributes_) {
        return E_INVALIDARG;
    }

    auto it = default_attributes_->find(name);
    if (it == default_attributes_->end()) {
        return E_INVALIDARG;
    }

    const auto& attr_value = it->second;

    try {
        // Use GetStringAttribute to get the string
        const std::string* str = attr_value.GetStringAttribute(name, elementIndex);
        if (!str) {
            return E_INVALIDARG;
        }

        // Return length including null terminator
        *attributeElementByteLength = static_cast<uint32_t>(str->size() + 1);
        return S_OK;
    } catch (...) {
        return E_FAIL;
    }
}

HRESULT AbiSafeShapeInferenceContext::GetStringAttributeElement(
    const char* name,
    uint32_t elementIndex,
    uint32_t attributeElementByteLength,
    char* attributeElement) const noexcept {
    if (!name || !attributeElement) {
        return E_POINTER;
    }

    // Try to get from kernel_info_ first if available
    if (kernel_info_ && ort_api_ && ort_api_->KernelInfoGetAttribute_string && elementIndex == 0) {
        size_t size = attributeElementByteLength;
        OrtStatus* status = ort_api_->KernelInfoGetAttribute_string(kernel_info_, name, attributeElement, &size);
        if (!status) {
            return S_OK;
        }
        if (status) {
            ort_api_->ReleaseStatus(status);
        }
    }

    // Fall back to default_attributes_
    if (!default_attributes_) {
        return E_INVALIDARG;
    }

    auto it = default_attributes_->find(name);
    if (it == default_attributes_->end()) {
        return E_INVALIDARG;
    }

    const auto& attr_value = it->second;

    try {
        // Use GetStringAttribute to get the string
        const std::string* str = attr_value.GetStringAttribute(name, elementIndex);
        if (!str) {
            return E_INVALIDARG;
        }

        // Check buffer size
        if (attributeElementByteLength < str->size() + 1) {
            return E_INVALIDARG;
        }

        // Copy string including null terminator
        strcpy_s(attributeElement, attributeElementByteLength, str->c_str());
        return S_OK;
    } catch (...) {
        return E_FAIL;
    }
}

uint32_t AbiSafeShapeInferenceContext::GetInputCount() const noexcept {
    if (ort_api_ == nullptr) return 0;

    size_t count = 0;

    // Try runtime context first (if available)
    if (kernel_context_ != nullptr) {
        if (Ort::Status status{ort_api_->KernelContext_GetInputCount(kernel_context_, &count)}; status.IsOK()) {
            return static_cast<uint32_t>(count);
        }
    }

    // Fall back to kernel_info at session init
    if (kernel_info_ != nullptr && ort_api_->KernelInfo_GetInputCount) {
        if (Ort::Status status{ort_api_->KernelInfo_GetInputCount(kernel_info_, &count)}; status.IsOK()) {
            return static_cast<uint32_t>(count);
        }
    }

    return 0;
}

uint32_t AbiSafeShapeInferenceContext::GetOutputCount() const noexcept {
    if (ort_api_ == nullptr) return 0;

    size_t count = 0;

    // Try runtime context first (if available)
    if (kernel_context_ != nullptr) {
        if (Ort::Status status{ort_api_->KernelContext_GetOutputCount(kernel_context_, &count)}; status.IsOK()) {
            return static_cast<uint32_t>(count);
        }
    }

    // Fall back to kernel_info at session init
    if (kernel_info_ != nullptr && ort_api_->KernelInfo_GetOutputCount) {
        if (Ort::Status status{ort_api_->KernelInfo_GetOutputCount(kernel_info_, &count)}; status.IsOK()) {
            return static_cast<uint32_t>(count);
        }
    }

    return 0;
}

bool AbiSafeShapeInferenceContext::IsInputValid(uint32_t inputIndex) const noexcept {
    if (ort_api_ == nullptr) return false;

    // Mirror OpNodeInfoWrapper::IsInputValid: index in range AND type/value is non-null.
    if (kernel_context_ != nullptr) {
        const OrtValue* value = nullptr;
        if (Ort::Status status{ort_api_->KernelContext_GetInput(kernel_context_, inputIndex, &value)}; !status.IsOK()) {
            return false;
        }
        return value != nullptr;
    }

    if (kernel_info_ != nullptr && ort_api_->KernelInfo_GetInputTypeInfo) {
        OrtTypeInfo* type_info = nullptr;
        if (Ort::Status status{ort_api_->KernelInfo_GetInputTypeInfo(kernel_info_, inputIndex, &type_info)}; !status.IsOK()) {
            return false;
        }
        bool valid = (type_info != nullptr);
        if (type_info != nullptr) ort_api_->ReleaseTypeInfo(type_info);
        return valid;
    }

    return false;
}

bool AbiSafeShapeInferenceContext::IsOutputValid(uint32_t outputIndex) const noexcept {
    if (ort_api_ == nullptr || kernel_info_ == nullptr) return false;
    OrtTypeInfo* type_info = nullptr;
    if (Ort::Status status{ort_api_->KernelInfo_GetOutputTypeInfo(kernel_info_, outputIndex, &type_info)}; !status.IsOK()) {
        return false;
    }
    bool valid = (type_info != nullptr);
    if (type_info != nullptr) ort_api_->ReleaseTypeInfo(type_info);
    return valid;
}

HRESULT AbiSafeShapeInferenceContext::GetInputEdgeDescription(
    uint32_t inputIndex,
    MLOperatorEdgeDescription* edgeDescription) const noexcept {

    if (edgeDescription == nullptr) return E_POINTER;
    if (ort_api_ == nullptr) return E_FAIL;
    // Note: kernel_context_ can be nullptr at session init - we'll fall back to kernel_info_ below

    OrtTypeInfo* type_info = nullptr;

    // Try runtime context first (if available)
    if (kernel_context_ != nullptr) {
        const OrtValue* input_value = nullptr;
        if (Ort::Status status{ort_api_->KernelContext_GetInput(kernel_context_, inputIndex, &input_value)}; status.IsOK() && input_value != nullptr) {
            Ort::Status type_status{ort_api_->GetTypeInfo(input_value, &type_info)};
            if (!type_status.IsOK()) {
                type_info = nullptr;
            }
        }
    }

    // Fall back to kernel_info if runtime didn't work or isn't available
    if (type_info == nullptr && kernel_info_ != nullptr && ort_api_->KernelInfo_GetInputTypeInfo) {
        if (Ort::Status status{ort_api_->KernelInfo_GetInputTypeInfo(kernel_info_, inputIndex, &type_info)}; !status.IsOK()) {
            return E_INVALIDARG;
        }
    }

    if (type_info == nullptr) {
        return E_INVALIDARG;
    }

    // We have type_info from either runtime or kernel_info
    ONNXType onnx_type;
    if (Ort::Status status{ort_api_->GetOnnxTypeFromTypeInfo(type_info, &onnx_type)}; !status.IsOK()) {
        ort_api_->ReleaseTypeInfo(type_info);
        return E_FAIL;
    }

    if (onnx_type == ONNX_TYPE_TENSOR) {
        edgeDescription->edgeType = MLOperatorEdgeType::Tensor;
        const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
        if (Ort::Status status{ort_api_->CastTypeInfoToTensorInfo(type_info, &tensor_info)}; status.IsOK() && tensor_info != nullptr) {
            ONNXTensorElementDataType elem_type;
            ort_api_->GetTensorElementType(tensor_info, &elem_type);
            edgeDescription->tensorDataType = static_cast<MLOperatorTensorDataType>(elem_type);
        }
    } else {
        edgeDescription->edgeType = MLOperatorEdgeType::Undefined;
    }

    ort_api_->ReleaseTypeInfo(type_info);
    return S_OK;
}

HRESULT AbiSafeShapeInferenceContext::GetInputTensorDimensionCount(
    uint32_t inputIndex,
    uint32_t* dimensionCount) const noexcept {

    if (dimensionCount == nullptr) {
        return E_POINTER;
    }
    if (ort_api_ == nullptr) {
        return E_FAIL;
    }
    // Note: kernel_context_ can be nullptr at session init - we'll fall back to kernel_info_ below

    // Try to get runtime tensor first (only if kernel_context_ is available)
    const OrtValue* input_value = nullptr;
    if (kernel_context_ != nullptr) {
        Ort::Status status{ort_api_->KernelContext_GetInput(kernel_context_, inputIndex, &input_value)};
        if (!status.IsOK()) {
            input_value = nullptr;
        }
    }
    if (input_value != nullptr) {
        // Runtime tensor available
        OrtTensorTypeAndShapeInfo* shape_info = nullptr;
        if (Ort::Status status{ort_api_->GetTensorTypeAndShape(input_value, &shape_info)}; status.IsOK()) {
            size_t dim_count = 0;
            ort_api_->GetDimensionsCount(shape_info, &dim_count);
            *dimensionCount = static_cast<uint32_t>(dim_count);
            ort_api_->ReleaseTensorTypeAndShapeInfo(shape_info);
            return S_OK;
        }
    }

    // Fall back to graph shape from kernel_info (for initializers/constants)
    if (kernel_info_ != nullptr && ort_api_->KernelInfo_GetInputTypeInfo) {
        OrtTypeInfo* type_info = nullptr;
        if (Ort::Status status{ort_api_->KernelInfo_GetInputTypeInfo(kernel_info_, inputIndex, &type_info)}; status.IsOK() && type_info != nullptr) {
            const OrtTensorTypeAndShapeInfo* shape_info = nullptr;
            if (Ort::Status cast_status{ort_api_->CastTypeInfoToTensorInfo(type_info, &shape_info)}; cast_status.IsOK() && shape_info != nullptr) {
                size_t dim_count = 0;
                ort_api_->GetDimensionsCount(shape_info, &dim_count);
                *dimensionCount = static_cast<uint32_t>(dim_count);
                ort_api_->ReleaseTypeInfo(type_info);
                return S_OK;
            }
            ort_api_->ReleaseTypeInfo(type_info);
        }
    }

    return E_INVALIDARG;
}

HRESULT AbiSafeShapeInferenceContext::GetInputTensorShape(
    uint32_t inputIndex,
    uint32_t dimensionCount,
    uint32_t* dimensions) const noexcept {

    if (dimensions == nullptr && dimensionCount > 0) {
        return E_POINTER;
    }
    if (ort_api_ == nullptr) {
        return E_FAIL;
    }

    // Prefer runtime shape when available (mirrors GetInputTensorDimensionCount behavior)
    const OrtValue* input_value = nullptr;
    if (kernel_context_ != nullptr) {
        Ort::Status status{ort_api_->KernelContext_GetInput(kernel_context_, inputIndex, &input_value)};
        if (!status.IsOK()) {
            input_value = nullptr;
        }
    }
    if (input_value != nullptr) {
        OrtTensorTypeAndShapeInfo* shape_info = nullptr;
        if (Ort::Status status{ort_api_->GetTensorTypeAndShape(input_value, &shape_info)}; status.IsOK()) {
            size_t dim_count = 0;
            ort_api_->GetDimensionsCount(shape_info, &dim_count);
            if (dimensionCount == static_cast<uint32_t>(dim_count)) {
                std::vector<int64_t> dims(dim_count);
                ort_api_->GetDimensions(shape_info, dims.data(), dim_count);
                ort_api_->ReleaseTensorTypeAndShapeInfo(shape_info);
                for (size_t i = 0; i < dim_count; ++i) {
                    dimensions[i] = static_cast<uint32_t>(dims[i]);
                }
                return S_OK;
            }
            ort_api_->ReleaseTensorTypeAndShapeInfo(shape_info);
        }
    }

    // Fall back to graph shape from kernel_info (for initializers/constants at session init)
    if (kernel_info_ != nullptr && ort_api_->KernelInfo_GetInputTypeInfo) {
        OrtTypeInfo* type_info = nullptr;
        if (Ort::Status status{ort_api_->KernelInfo_GetInputTypeInfo(kernel_info_, inputIndex, &type_info)}; status.IsOK() && type_info != nullptr) {
            const OrtTensorTypeAndShapeInfo* shape_info = nullptr;
            if (Ort::Status cast_status{ort_api_->CastTypeInfoToTensorInfo(type_info, &shape_info)}; cast_status.IsOK() && shape_info != nullptr) {
                size_t dim_count = 0;
                ort_api_->GetDimensionsCount(shape_info, &dim_count);
                if (dimensionCount == static_cast<uint32_t>(dim_count)) {
                    std::vector<int64_t> dims(dim_count);
                    ort_api_->GetDimensions(shape_info, dims.data(), dim_count);
                    ort_api_->ReleaseTypeInfo(type_info);
                    for (size_t i = 0; i < dim_count; ++i) {
                        dimensions[i] = static_cast<uint32_t>(dims[i]);
                    }
                    return S_OK;
                }
            }
            ort_api_->ReleaseTypeInfo(type_info);
        }
    }

    return E_INVALIDARG;
}

HRESULT AbiSafeShapeInferenceContext::SetOutputTensorShape(
    uint32_t outputIndex,
    uint32_t dimensionCount,
    const uint32_t* dimensions) noexcept {
    if (!dimensions && dimensionCount > 0) return E_POINTER;

    // Resize vector if needed
    if (outputIndex >= inferred_output_shapes_.size()) {
        inferred_output_shapes_.resize(outputIndex + 1);
    }

    // Store the inferred shape
    inferred_output_shapes_[outputIndex].assign(dimensions, dimensions + dimensionCount);

    return S_OK;
}

HRESULT AbiSafeShapeInferenceContext::GetConstantInputTensor(
    uint32_t inputIndex,
    IMLOperatorTensor** tensor) const noexcept {

    if (!tensor) return E_POINTER;
    *tensor = nullptr;

    if (!ort_api_) return E_FAIL;
    // Note: kernel_context_ can be nullptr at session init - we'll fall back to kernel_info_ below

    // Check if already cached
    auto it = constant_tensor_cache_.find(inputIndex);
    if (it != constant_tensor_cache_.end()) {
        *tensor = it->second.Get();
        (*tensor)->AddRef();

        return S_OK;
    }

    const OrtValue* input_value = nullptr;

    // Try runtime context first (if available)
    if (kernel_context_ != nullptr) {
        Ort::Status status{ort_api_->KernelContext_GetInput(kernel_context_, inputIndex, &input_value)};
        DML_PERF_LOG("[ABI_SAFE] ShapeCtx::GetConstantInput[", inputIndex, "]: ctx_status=",
            status.IsOK() ? "OK" : "ERR", "  value=", (void*)input_value, "\n");
        if (!status.IsOK() || input_value == nullptr) {
            input_value = nullptr;
        }
    }

    // Fall back to kernel_info for constant inputs at session init
    if (input_value == nullptr && kernel_info_ != nullptr && ort_api_->KernelInfoGetConstantInput_tensor) {
        // First check if the input index is valid
        uint32_t input_count = GetInputCount();
        if (inputIndex >= input_count) {
            // Input doesn't exist - return S_OK with nullptr (not an error, just not available)
            *tensor = nullptr;
            return S_OK;
        }

        int is_constant = 0;
        if (Ort::Status status{ort_api_->KernelInfoGetConstantInput_tensor(kernel_info_, inputIndex, &is_constant, &input_value)}; !status.IsOK()) {
            // Return S_OK with nullptr rather than error - shape inferencer will handle it
            *tensor = nullptr;
            return S_OK;
        }
        if (is_constant == 0 || input_value == nullptr) {
            // Not a constant input - return S_OK with nullptr to match unsafe path behavior
            // Shape inferencers will check for nullptr and fall back to attributes
            *tensor = nullptr;
            return S_OK;
        }
    }

    if (!input_value) {
        // Input doesn't exist or couldn't be retrieved - return S_OK with nullptr to match unsafe path
        *tensor = nullptr;
        return S_OK;
    }

    // Create ABI-safe tensor wrapper (constant CPU tensors, not GPU — is_internal_operator=false)
    auto abi_tensor = Microsoft::WRL::Make<AbiSafeTensor>(input_value, ort_api_, execution_provider_, false);
    constant_tensor_cache_[inputIndex] = abi_tensor;

    *tensor = abi_tensor.Get();
    (*tensor)->AddRef();

    return S_OK;
}

HRESULT AbiSafeShapeInferenceContext::TryGetConstantInputTensor(
    uint32_t inputIndex,
    IMLOperatorTensor** tensor) const noexcept {
    // Same as GetConstantInputTensor but returns S_FALSE instead of error if not available
    HRESULT hr = GetConstantInputTensor(inputIndex, tensor);
    if (FAILED(hr)) {
        return S_FALSE;
    }
    return hr;
}

HRESULT AbiSafeShapeInferenceContext::GetSequenceInputInfo(
    uint32_t inputIndex,
    uint32_t* inputCount,
    MLOperatorTensorDataType* dataType) const noexcept {
    // Sequence inputs not supported in shape inference context yet
    return E_NOTIMPL;
}

HRESULT AbiSafeShapeInferenceContext::GetSequenceInputTensorDimensionCount(
    uint32_t inputIndex,
    uint32_t sequenceIndex,
    uint32_t* dimensionCount) const noexcept {
    // Sequence inputs not supported in shape inference context yet
    return E_NOTIMPL;
}

HRESULT AbiSafeShapeInferenceContext::GetSequenceInputTensorShape(
    uint32_t inputIndex,
    uint32_t sequenceIndex,
    uint32_t dimensionCount,
    uint32_t* dimensions) const noexcept {
    // Sequence inputs not supported in shape inference context yet
    return E_NOTIMPL;
}

// ============================================================================
// PreFetchedTensorAttrWrapper
// ============================================================================

PreFetchedTensorAttrWrapper::PreFetchedTensorAttrWrapper(
    MLOperatorTensorDataType data_type,
    std::vector<uint32_t> shape,
    std::vector<std::byte> raw_bytes)
    : data_type_(data_type)
    , shape_(std::move(shape))
    , raw_bytes_(std::move(raw_bytes)) {}

uint32_t PreFetchedTensorAttrWrapper::GetDimensionCount() const noexcept {
    return static_cast<uint32_t>(shape_.size());
}

HRESULT PreFetchedTensorAttrWrapper::GetShape(uint32_t dim_count, uint32_t* dims) const noexcept {
    if (!dims || dim_count != static_cast<uint32_t>(shape_.size())) return E_INVALIDARG;
    memcpy(dims, shape_.data(), dim_count * sizeof(uint32_t));
    return S_OK;
}

MLOperatorTensorDataType PreFetchedTensorAttrWrapper::GetTensorDataType() const noexcept {
    return data_type_;
}

bool PreFetchedTensorAttrWrapper::IsCpuData() const noexcept { return true; }
bool PreFetchedTensorAttrWrapper::IsDataInterface() const noexcept { return false; }

void* PreFetchedTensorAttrWrapper::GetData() noexcept {
    return raw_bytes_.data();
}

void PreFetchedTensorAttrWrapper::GetDataInterface(IUnknown** dataInterface) noexcept {
    if (dataInterface) *dataInterface = nullptr;
}

// ============================================================================
// AbiSafeKernelCreationContext - IMLOperatorKernelCreationContext
// ============================================================================

AbiSafeKernelCreationContext::AbiSafeKernelCreationContext(
    const OrtKernelInfo* kernel_info,
    const OrtApi* ort_api,
    const AttributeMap* default_attributes,
    const std::vector<uint32_t>* required_constant_cpu_inputs,
    const PluginDmlExecutionProviderImpl* execution_provider,
    std::unordered_map<uint32_t, Microsoft::WRL::ComPtr<IMLOperatorTensor>>&& constant_tensors,
    OrtKernelContext* runtime_context,
    const char* operator_name,
    bool is_internal_operator,
    bool requires_input_shapes_at_creation,
    std::unordered_map<std::string, PreFetchedTensorAttr> tensor_attribute_cache,
    const EdgeShapes* input_shapes_override)
    : kernel_info_(kernel_info)
    , ort_api_(ort_api)
    , default_attributes_(default_attributes)
    , required_constant_cpu_inputs_(required_constant_cpu_inputs)
    , execution_provider_(execution_provider)
    , constant_tensors_(std::move(constant_tensors))
    , runtime_context_(runtime_context)
    , operator_name_(operator_name)
    , is_internal_operator_(is_internal_operator)
    , requires_input_shapes_at_creation_(requires_input_shapes_at_creation)
    , tensor_attribute_cache_(std::move(tensor_attribute_cache))
    , input_shapes_override_(input_shapes_override)
{
    // Call GetABIExecutionInterfaceAndInvalidateState exactly once here, matching the
    // PluginOpKernelInfoWrapper constructor pattern. GetExecutionInterface then returns
    // the cached object. Calling this function on every GetExecutionInterface() invocation
    // causes repeated GPU resource state invalidations, producing corrupted output.
    if (execution_provider_) {
        auto* provider = const_cast<PluginDmlExecutionProviderImpl*>(execution_provider_);
        Microsoft::WRL::ComPtr<IWinmlExecutionProvider> winml_provider;
        if (SUCCEEDED(provider->QueryInterface(IID_PPV_ARGS(&winml_provider)))) {
            winml_provider->GetABIExecutionInterfaceAndInvalidateState(
                is_internal_operator_, abi_execution_object_.ReleaseAndGetAddressOf());
        } else {
            // Fallback: QI for IUnknown to avoid ambiguous base conversion from multiply-inherited class.
            provider->QueryInterface(IID_PPV_ARGS(&abi_execution_object_));
        }
    }
}

uint32_t AbiSafeKernelCreationContext::GetInputCount() const noexcept {
    size_t count = 0;
    OrtStatus* status = ort_api_->KernelInfo_GetInputCount(kernel_info_, &count);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return 0;
    }
    return static_cast<uint32_t>(count);
}

uint32_t AbiSafeKernelCreationContext::GetOutputCount() const noexcept {
    size_t count = 0;
    OrtStatus* status = ort_api_->KernelInfo_GetOutputCount(kernel_info_, &count);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return 0;
    }
    return static_cast<uint32_t>(count);
}

bool AbiSafeKernelCreationContext::IsInputValid(uint32_t inputIndex) const noexcept {
    // Mirror OpNodeInfoWrapper::IsInputValid: index in range AND type info is non-null.
    // Optional inputs that are absent have no type info and must return false.
    size_t count = 0;
    OrtStatus* status = ort_api_->KernelInfo_GetInputCount(kernel_info_, &count);
    if (status) { ort_api_->ReleaseStatus(status); return false; }
    if (inputIndex >= count) return false;

    OrtTypeInfo* type_info = nullptr;
    status = ort_api_->KernelInfo_GetInputTypeInfo(kernel_info_, inputIndex, &type_info);
    if (status) { ort_api_->ReleaseStatus(status); return false; }
    bool valid = (type_info != nullptr);
    if (type_info) ort_api_->ReleaseTypeInfo(type_info);
    return valid;
}

bool AbiSafeKernelCreationContext::IsOutputValid(uint32_t outputIndex) const noexcept {
    // Mirror OpNodeInfoWrapper::IsOutputValid: index in range AND type info is non-null.
    size_t count = 0;
    OrtStatus* status = ort_api_->KernelInfo_GetOutputCount(kernel_info_, &count);
    if (status) { ort_api_->ReleaseStatus(status); return false; }
    if (outputIndex >= count) return false;

    OrtTypeInfo* type_info = nullptr;
    status = ort_api_->KernelInfo_GetOutputTypeInfo(kernel_info_, outputIndex, &type_info);
    if (status) { ort_api_->ReleaseStatus(status); return false; }
    bool valid = (type_info != nullptr);
    if (type_info) ort_api_->ReleaseTypeInfo(type_info);
    return valid;
}

HRESULT AbiSafeKernelCreationContext::GetInputEdgeDescription(uint32_t inputIndex, MLOperatorEdgeDescription* edgeDesc) const noexcept {
    if (!edgeDesc) {
        return E_POINTER;
    }

    OrtTypeInfo* type_info = nullptr;
    OrtStatus* status = ort_api_->KernelInfo_GetInputTypeInfo(kernel_info_, inputIndex, &type_info);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    // Determine the edge type (tensor, sequence, etc.)
    ONNXType onnx_type = ONNX_TYPE_UNKNOWN;
    status = ort_api_->GetOnnxTypeFromTypeInfo(type_info, &onnx_type);
    if (status) {
        ort_api_->ReleaseTypeInfo(type_info);
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    if (onnx_type == ONNX_TYPE_TENSOR) {
        edgeDesc->edgeType = MLOperatorEdgeType::Tensor;

        // Get tensor data type
        const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
        status = ort_api_->CastTypeInfoToTensorInfo(type_info, &tensor_info);
        if (status) {
            ort_api_->ReleaseTypeInfo(type_info);
            ort_api_->ReleaseStatus(status);
            return E_FAIL;
        }

        ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        status = ort_api_->GetTensorElementType(tensor_info, &elem_type);
        if (status) {
            ort_api_->ReleaseTypeInfo(type_info);
            ort_api_->ReleaseStatus(status);
            return E_FAIL;
        }

        // Convert ONNXTensorElementDataType to MLOperatorTensorDataType
        edgeDesc->tensorDataType = ConvertToMLOperatorTensorDataType(elem_type);
    }
    else if (onnx_type == ONNX_TYPE_SEQUENCE) {
        edgeDesc->edgeType = MLOperatorEdgeType::SequenceTensor;
        // TODO: Get sequence element type if needed
        edgeDesc->tensorDataType = MLOperatorTensorDataType::Undefined;
    }
    else {
        edgeDesc->edgeType = MLOperatorEdgeType::Undefined;
        edgeDesc->tensorDataType = MLOperatorTensorDataType::Undefined;
    }

    ort_api_->ReleaseTypeInfo(type_info);
    return S_OK;
}

HRESULT AbiSafeKernelCreationContext::GetOutputEdgeDescription(uint32_t outputIndex, MLOperatorEdgeDescription* edgeDesc) const noexcept {
    if (!edgeDesc) return E_POINTER;

    OrtTypeInfo* type_info = nullptr;
    OrtStatus* status = ort_api_->KernelInfo_GetOutputTypeInfo(kernel_info_, outputIndex, &type_info);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    // Determine the edge type (tensor, sequence, etc.)
    ONNXType onnx_type = ONNX_TYPE_UNKNOWN;
    status = ort_api_->GetOnnxTypeFromTypeInfo(type_info, &onnx_type);
    if (status) {
        ort_api_->ReleaseTypeInfo(type_info);
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    if (onnx_type == ONNX_TYPE_TENSOR) {
        edgeDesc->edgeType = MLOperatorEdgeType::Tensor;

        // Get tensor data type
        const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
        status = ort_api_->CastTypeInfoToTensorInfo(type_info, &tensor_info);
        if (status) {
            ort_api_->ReleaseTypeInfo(type_info);
            ort_api_->ReleaseStatus(status);
            return E_FAIL;
        }

        ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        status = ort_api_->GetTensorElementType(tensor_info, &elem_type);
        if (status) {
            ort_api_->ReleaseTypeInfo(type_info);
            ort_api_->ReleaseStatus(status);
            return E_FAIL;
        }

        // Convert ONNXTensorElementDataType to MLOperatorTensorDataType
        edgeDesc->tensorDataType = ConvertToMLOperatorTensorDataType(elem_type);
    }
    else if (onnx_type == ONNX_TYPE_SEQUENCE) {
        edgeDesc->edgeType = MLOperatorEdgeType::SequenceTensor;
        // TODO: Get sequence element type if needed
        edgeDesc->tensorDataType = MLOperatorTensorDataType::Undefined;
    }
    else {
        edgeDesc->edgeType = MLOperatorEdgeType::Undefined;
        edgeDesc->tensorDataType = MLOperatorTensorDataType::Undefined;
    }

    ort_api_->ReleaseTypeInfo(type_info);
    return S_OK;
}

bool AbiSafeKernelCreationContext::HasTensorShapeDescription() const noexcept {
    // Mirror old OpKernelInfoWrapper::HasTensorShapeDescription() = m_allowInputShapeQuery
    // which is a static flag from kernel registration, NOT a dynamic shape check.
    // At lazy-init (Compute time) with runtime_context_, shapes are always available.
    if (runtime_context_) {
        return true;
    }
    return requires_input_shapes_at_creation_;
}

HRESULT AbiSafeKernelCreationContext::GetTensorShapeDescription(IMLOperatorTensorShapeDescription** shapeInfo) const noexcept {
    if (!shapeInfo) {
        return E_POINTER;
    }

    *shapeInfo = nullptr;

    if (!HasTensorShapeDescription()) {
        // Dynamic shapes not supported - return E_FAIL to trigger fallback to unsafe path
        return E_FAIL;
    }

    // If shapes are available, return this object as the shape description interface
    Microsoft::WRL::ComPtr<IMLOperatorTensorShapeDescription> ret = const_cast<AbiSafeKernelCreationContext*>(this);
    *shapeInfo = ret.Detach();
    return S_OK;
}

HRESULT AbiSafeKernelCreationContext::GetInputTensorDimensionCount(uint32_t inputIndex, uint32_t* dimensionCount) const noexcept {
    if (!dimensionCount) {
        return E_POINTER;
    }

    // Use captured runtime shapes (mirrors old plugin's m_inputShapesOverride → OpNodeInfoWrapper behavior)
    if (input_shapes_override_) {
        if (inputIndex >= static_cast<uint32_t>(input_shapes_override_->EdgeCount())) {
            return E_INVALIDARG;
        }
        *dimensionCount = static_cast<uint32_t>(input_shapes_override_->GetShape(inputIndex).size());
        return S_OK;
    }

    // Fall back to graph shape information
    OrtTypeInfo* type_info = nullptr;
    OrtStatus* status = ort_api_->KernelInfo_GetInputTypeInfo(kernel_info_, inputIndex, &type_info);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
    status = ort_api_->CastTypeInfoToTensorInfo(type_info, &tensor_info);
    if (status) {
        ort_api_->ReleaseTypeInfo(type_info);
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    size_t dim_count = 0;
    status = ort_api_->GetDimensionsCount(tensor_info, &dim_count);
    if (status) {
        ort_api_->ReleaseTypeInfo(type_info);
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    *dimensionCount = static_cast<uint32_t>(dim_count);
    ort_api_->ReleaseTypeInfo(type_info);
    return S_OK;
}

HRESULT AbiSafeKernelCreationContext::GetInputTensorShape(uint32_t inputIndex, uint32_t dimensionCount, uint32_t* dimensions) const noexcept {
    if (!dimensions && dimensionCount > 0) {
        return E_POINTER;
    }

    // Use captured runtime input shapes (mirrors old plugin's m_inputShapesOverride → OpNodeInfoWrapper behavior)
    if (input_shapes_override_) {
        if (inputIndex >= static_cast<uint32_t>(input_shapes_override_->EdgeCount())) {
            return E_INVALIDARG;
        }
        const auto& shape = input_shapes_override_->GetShape(inputIndex);
        if (dimensionCount != static_cast<uint32_t>(shape.size())) {
            return E_INVALIDARG;
        }
        for (uint32_t i = 0; i < dimensionCount; ++i) {
            dimensions[i] = shape[i];
        }
        return S_OK;
    }

    // Otherwise get from KernelInfo (may have dynamic dimensions)
    OrtTypeInfo* type_info = nullptr;
    OrtStatus* status = ort_api_->KernelInfo_GetInputTypeInfo(kernel_info_, inputIndex, &type_info);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
    status = ort_api_->CastTypeInfoToTensorInfo(type_info, &tensor_info);
    if (status) {
        ort_api_->ReleaseTypeInfo(type_info);
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    size_t dim_count = 0;
    status = ort_api_->GetDimensionsCount(tensor_info, &dim_count);
    if (status) {
        ort_api_->ReleaseTypeInfo(type_info);
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    if (dimensionCount != static_cast<uint32_t>(dim_count)) {
        ort_api_->ReleaseTypeInfo(type_info);
        return E_INVALIDARG;
    }

    std::vector<int64_t> dims(dim_count);
    status = ort_api_->GetDimensions(tensor_info, dims.data(), dim_count);
    if (status) {
        ort_api_->ReleaseTypeInfo(type_info);
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    // Convert int64_t to uint32_t
    // If we encounter dynamic dimensions here, it's a logic error - HasTensorShapeDescription should have returned false
    for (size_t i = 0; i < dim_count; ++i) {
        if (dims[i] < 0) {
            // This should never happen - HasTensorShapeDescription should prevent this
            ort_api_->ReleaseTypeInfo(type_info);
            return E_UNEXPECTED;
        }
        dimensions[i] = static_cast<uint32_t>(dims[i]);
    }

    ort_api_->ReleaseTypeInfo(type_info);
    return S_OK;
}

bool AbiSafeKernelCreationContext::HasOutputShapeDescription() const noexcept {
    // Output shapes are only available when they were precomputed by the shape inferrer.
    // This mirrors OpKernelInfoWrapper::HasOutputShapeDescription() = m_allowOutputShapeQuery
    // which is set to true only when requiresOutputShapesAtCreation (i.e., a shape inferrer exists).
    // Returning HasTensorShapeDescription() here was wrong: it would cause output TensorDescs
    // to be built from graph KernelInfo shapes when no shape inferrer ran, producing wrong strides.
    return !precomputed_output_shapes_.empty();
}

HRESULT AbiSafeKernelCreationContext::GetOutputTensorDimensionCount(uint32_t outputIndex, uint32_t* dimensionCount) const noexcept {
    if (!dimensionCount) {
        return E_POINTER;
    }

    // PRIORITY 1: Check precomputed shapes from shape inferrer
    // These take precedence over graph shapes when available
    if (outputIndex < precomputed_output_shapes_.size() && !precomputed_output_shapes_[outputIndex].empty()) {
        *dimensionCount = static_cast<uint32_t>(precomputed_output_shapes_[outputIndex].size());
        return S_OK;
    }

    // PRIORITY 2: Fall back to graph shape information
    OrtTypeInfo* type_info = nullptr;
    OrtStatus* status = ort_api_->KernelInfo_GetOutputTypeInfo(kernel_info_, outputIndex, &type_info);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
    status = ort_api_->CastTypeInfoToTensorInfo(type_info, &tensor_info);
    if (status) {
        ort_api_->ReleaseTypeInfo(type_info);
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    size_t dim_count = 0;
    status = ort_api_->GetDimensionsCount(tensor_info, &dim_count);
    if (status) {
        ort_api_->ReleaseTypeInfo(type_info);
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    *dimensionCount = static_cast<uint32_t>(dim_count);
    ort_api_->ReleaseTypeInfo(type_info);
    return S_OK;
}

HRESULT AbiSafeKernelCreationContext::GetOutputTensorShape(uint32_t outputIndex, uint32_t dimensionCount, uint32_t* dimensions) const noexcept {
    if (!dimensions && dimensionCount > 0) {
        return E_POINTER;
    }

    // Use pre-computed output shapes from shape inferrer if available.
    // This mirrors PluginOpKernelInfoWrapper::GetOutputTensorShape which returns m_inferredOutputShapes directly.
    if (outputIndex < precomputed_output_shapes_.size() && !precomputed_output_shapes_[outputIndex].empty()) {
        const auto& precomputed_shape = precomputed_output_shapes_[outputIndex];
        if (precomputed_shape.size() == dimensionCount) {
            for (uint32_t i = 0; i < dimensionCount; ++i) {
                dimensions[i] = precomputed_shape[i];
            }
            return S_OK;
        }
    }

    // Fall back to kernel info (for static shapes or when runtime context not available)
    OrtTypeInfo* type_info = nullptr;
    OrtStatus* status = ort_api_->KernelInfo_GetOutputTypeInfo(kernel_info_, outputIndex, &type_info);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
    status = ort_api_->CastTypeInfoToTensorInfo(type_info, &tensor_info);
    if (status) {
        ort_api_->ReleaseTypeInfo(type_info);
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    size_t dim_count = 0;
    status = ort_api_->GetDimensionsCount(tensor_info, &dim_count);
    if (status) {
        ort_api_->ReleaseTypeInfo(type_info);
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    if (dimensionCount != static_cast<uint32_t>(dim_count)) {
        ort_api_->ReleaseTypeInfo(type_info);
        return E_INVALIDARG;
    }

    std::vector<int64_t> dims(dim_count);
    status = ort_api_->GetDimensions(tensor_info, dims.data(), dim_count);
    if (status) {
        ort_api_->ReleaseTypeInfo(type_info);
        ort_api_->ReleaseStatus(status);
        return E_FAIL;
    }

    // Convert int64_t to uint32_t
    // If we encounter dynamic dimensions here, it's a logic error
    // - If runtime_context is available, the PRIMARY APPROACH above should have handled it
    // - If runtime_context is not available, HasTensorShapeDescription should have returned false
    for (size_t i = 0; i < dim_count; ++i) {
        if (dims[i] < 0) {
            // This should never happen
            ort_api_->ReleaseTypeInfo(type_info);
            return E_UNEXPECTED;
        }
        dimensions[i] = static_cast<uint32_t>(dims[i]);
    }

    ort_api_->ReleaseTypeInfo(type_info);
    return S_OK;
}

void AbiSafeKernelCreationContext::GetExecutionInterface(IUnknown** executionInterface) const noexcept {
    if (!executionInterface) return;
    abi_execution_object_.CopyTo(executionInterface);
}

HRESULT AbiSafeKernelCreationContext::GetAttribute(
    const char* name,
    MLOperatorAttributeType type,
    uint32_t elementCount,
    size_t elementByteSize,
    void* value) const noexcept {
    if (!name || !value) return E_POINTER;

    OrtStatus* status = nullptr;

    switch (type) {
        case MLOperatorAttributeType::Float:
            if (elementCount == 1 && elementByteSize == sizeof(float)) {
                status = ort_api_->KernelInfoGetAttribute_float(kernel_info_, name, static_cast<float*>(value));
            }
            break;

        case MLOperatorAttributeType::FloatArray:
            {
                // Array of floats - C API uses size_t*
                size_t count = elementCount;
                status = ort_api_->KernelInfoGetAttributeArray_float(kernel_info_, name, static_cast<float*>(value), &count);
            }
            break;

        case MLOperatorAttributeType::Int:
            if (elementCount == 1 && elementByteSize == sizeof(int64_t)) {
                status = ort_api_->KernelInfoGetAttribute_int64(kernel_info_, name, static_cast<int64_t*>(value));
            }
            break;

        case MLOperatorAttributeType::IntArray:
            {
                // Array of int64s - C API uses size_t*
                size_t count = elementCount;
                status = ort_api_->KernelInfoGetAttributeArray_int64(kernel_info_, name, static_cast<int64_t*>(value), &count);
            }
            break;

        case MLOperatorAttributeType::String:
            if (elementCount == 1) {
                // Single string - C API uses size_t*
                size_t size = elementByteSize;
                status = ort_api_->KernelInfoGetAttribute_string(kernel_info_, name, static_cast<char*>(value), &size);
            }
            break;

        case MLOperatorAttributeType::StringArray:
            // String arrays not directly supported in C API
            return E_NOTIMPL;

        default:
            // Tensor and other unsupported types
            return E_NOTIMPL;
    }

    if (status) {
        ort_api_->ReleaseStatus(status);
        // Attribute absent from node — mirror OpNodeInfoWrapper behavior and check registered defaults.
        if (default_attributes_) {
            auto it = default_attributes_->find(name);
            if (it != default_attributes_->end()) {
                it->second.GetAttribute(type, elementCount, elementByteSize, value);
                return S_OK;
            }
        }
        return E_FAIL;
    }

    return S_OK;
}

HRESULT AbiSafeKernelCreationContext::GetAttributeElementCount(
    const char* name,
    MLOperatorAttributeType type,
    uint32_t* elementCount) const noexcept {

    if (!name || !elementCount) return E_POINTER;

    // Initialize to 0 (attribute not found)
    *elementCount = 0;

    OrtStatus* status = nullptr;

    switch (type) {
        case MLOperatorAttributeType::Float: {
            float dummy;
            status = ort_api_->KernelInfoGetAttribute_float(kernel_info_, name, &dummy);
            if (status == nullptr) {
                *elementCount = 1;
            } else {
                ort_api_->ReleaseStatus(status);
            }
            break;
        }

        case MLOperatorAttributeType::FloatArray: {
            // Query with null to get count
            size_t count = 0;
            status = ort_api_->KernelInfoGetAttributeArray_float(kernel_info_, name, nullptr, &count);
            if (status == nullptr) {
                *elementCount = static_cast<uint32_t>(count);
            } else {
                ort_api_->ReleaseStatus(status);
            }
            break;
        }

        case MLOperatorAttributeType::Int: {
            int64_t dummy;
            status = ort_api_->KernelInfoGetAttribute_int64(kernel_info_, name, &dummy);
            if (status == nullptr) {
                *elementCount = 1;
            } else {
                ort_api_->ReleaseStatus(status);
            }
            break;
        }

        case MLOperatorAttributeType::IntArray: {
            // Query with null to get count
            size_t count = 0;
            status = ort_api_->KernelInfoGetAttributeArray_int64(kernel_info_, name, nullptr, &count);
            if (status == nullptr) {
                *elementCount = static_cast<uint32_t>(count);
            } else {
                ort_api_->ReleaseStatus(status);
            }
            break;
        }

        case MLOperatorAttributeType::String: {
            size_t size = 0;
            status = ort_api_->KernelInfoGetAttribute_string(kernel_info_, name, nullptr, &size);
            if (status == nullptr) {
                *elementCount = 1; // Single string
            } else {
                ort_api_->ReleaseStatus(status);
            }
            break;
        }

        case MLOperatorAttributeType::StringArray:
            // String arrays not directly supported by ORT C API — fall through to default_attributes_
            break;

        default:
            if (type == MLOperatorAttributeTypeTensor) {
                // ORT C API has no tensor attribute accessor; serve from pre-fetched cache
                if (tensor_attribute_cache_.count(name)) {
                    *elementCount = 1;
                    return S_OK;
                }
                // Not in cache — attribute absent or not a tensor type
                break;
            }
            return E_INVALIDARG;
    }

    // Look for a value in the kernel's registered defaults if one was not found
    // This also handles string arrays and tensor attributes which can't be queried through ORT C API
    if (*elementCount == 0 && default_attributes_) {
        auto defaultAttr = default_attributes_->find(name);
        if (defaultAttr != default_attributes_->end()) {
            *elementCount = static_cast<uint32_t>(defaultAttr->second.ElementCount());
        }
    }

    return S_OK;
}

HRESULT AbiSafeKernelCreationContext::GetStringAttributeElementLength(
    const char* name,
    uint32_t elementIndex,
    uint32_t* attributeElementByteSize) const noexcept {

    if (!name || !attributeElementByteSize) return E_POINTER;

    // ORT C API only supports single strings (elementIndex == 0)
    if (elementIndex == 0) {
        size_t size = 0;
        OrtStatus* status = ort_api_->KernelInfoGetAttribute_string(kernel_info_, name, nullptr, &size);
        if (!status) {
            *attributeElementByteSize = static_cast<uint32_t>(size);
            return S_OK;
        }
        ort_api_->ReleaseStatus(status);
    }

    // Fall back to default_attributes_ for string arrays and ORT API failures
    if (!default_attributes_) return E_INVALIDARG;

    auto it = default_attributes_->find(name);
    if (it == default_attributes_->end()) return E_INVALIDARG;

    try {
        const std::string* str = it->second.GetStringAttribute(name, elementIndex);
        if (!str) return E_INVALIDARG;
        *attributeElementByteSize = static_cast<uint32_t>(str->size() + 1);
        return S_OK;
    } catch (...) {
        return E_FAIL;
    }
}

HRESULT AbiSafeKernelCreationContext::GetStringAttributeElement(
    const char* name,
    uint32_t elementIndex,
    uint32_t attributeElementByteSize,
    char* attributeElement) const noexcept {

    if (!name || !attributeElement) return E_POINTER;

    // ORT C API only supports single strings (elementIndex == 0)
    if (elementIndex == 0) {
        size_t size = attributeElementByteSize;
        OrtStatus* status = ort_api_->KernelInfoGetAttribute_string(kernel_info_, name, attributeElement, &size);
        if (!status) {
            return S_OK;
        }
        ort_api_->ReleaseStatus(status);
    }

    // Fall back to default_attributes_ for string arrays and ORT API failures
    if (!default_attributes_) return E_INVALIDARG;

    auto it = default_attributes_->find(name);
    if (it == default_attributes_->end()) return E_INVALIDARG;

    try {
        const std::string* str = it->second.GetStringAttribute(name, elementIndex);
        if (!str) return E_INVALIDARG;
        if (attributeElementByteSize < str->size() + 1) return E_INVALIDARG;
        strcpy_s(attributeElement, attributeElementByteSize, str->c_str());
        return S_OK;
    } catch (...) {
        return E_FAIL;
    }
}

HRESULT AbiSafeKernelCreationContext::GetTensorAttribute(
    const char* name,
    IMLOperatorTensor** tensor) const noexcept {
    if (!name || !tensor) return E_POINTER;
    *tensor = nullptr;

    try {
        auto it = tensor_attribute_cache_.find(name);
        if (it == tensor_attribute_cache_.end()) return E_INVALIDARG;
        const auto& cached = it->second;
        auto wrapper = Microsoft::WRL::Make<PreFetchedTensorAttrWrapper>(
            cached.data_type, cached.shape, cached.raw_bytes);
        *tensor = wrapper.Detach();
        return S_OK;
    } catch (...) {
        return E_FAIL;
    }
}

HRESULT AbiSafeKernelCreationContext::GetConstantInputTensor(
    uint32_t inputIndex,
    IMLOperatorTensor** tensor) const noexcept {
    if (!tensor) {
        return E_POINTER;
    }
    *tensor = nullptr;

    // Check if we have a pre-fetched constant tensor (hybrid approach - ABI-safe)
    auto it = constant_tensors_.find(inputIndex);
    if (it != constant_tensors_.end() && it->second) {
        *tensor = it->second.Get();
        (*tensor)->AddRef();  // Caller will release
        return S_OK;
    }

    // Try to fetch from ORT API (for lazy initialization at Compute time)
    if (kernel_info_ && ort_api_ && ort_api_->KernelInfoGetConstantInput_tensor) {

        int is_constant = 0;
        const OrtValue* ort_value = nullptr;
        OrtStatus* status = ort_api_->KernelInfoGetConstantInput_tensor(kernel_info_, inputIndex, &is_constant, &ort_value);

        if (status) {
            ort_api_->ReleaseStatus(status);
        } else {

            if (is_constant && ort_value) {
                auto safe_tensor = Microsoft::WRL::Make<AbiSafeTensor>(ort_value, ort_api_, execution_provider_, false);
                *tensor = safe_tensor.Detach();
                return S_OK;
            }
        }
    }

    // Check if this input is required to be constant - if so, failure to get it is an error
    bool inputRequiredAsConstant = false;
    if (required_constant_cpu_inputs_) {
        inputRequiredAsConstant = std::find(
            required_constant_cpu_inputs_->begin(),
            required_constant_cpu_inputs_->end(),
            inputIndex) != required_constant_cpu_inputs_->end();
    }


    if (inputRequiredAsConstant) {
        return E_FAIL;  // Required constant not available
    }

    return E_INVALIDARG; // Input not a constant
}

HRESULT AbiSafeKernelCreationContext::TryGetConstantInputTensor(
    uint32_t inputIndex,
    IMLOperatorTensor** tensor) const noexcept {

    if (!tensor) {
        return E_POINTER;
    }
    *tensor = nullptr;

    // Use C API to check if input is constant and get the value
    int is_constant = 0;
    const OrtValue* ort_value = nullptr;
    OrtStatus* status = ort_api_->KernelInfoGetConstantInput_tensor(kernel_info_, inputIndex, &is_constant, &ort_value);

    if (status) {
        ort_api_->ReleaseStatus(status);
        // Return S_OK with null tensor to indicate not available (not an error)
        return S_OK;
    }

    if (!is_constant || !ort_value) {
        // Check if this input is required to be constant
        bool inputRequiredAsConstant = false;
        if (required_constant_cpu_inputs_) {
            inputRequiredAsConstant = std::find(
                required_constant_cpu_inputs_->begin(),
                required_constant_cpu_inputs_->end(),
                inputIndex) != required_constant_cpu_inputs_->end();
        }

        // This shouldn't happen since kernel creation is deferred when required constants are missing
        if (inputRequiredAsConstant) {
            return E_UNEXPECTED;
        }

        // Not a constant input - return S_OK with null tensor (OK for optional inputs)
        return S_OK;
    }

    // Wrap the OrtValue in an AbiSafeTensor (constant CPU initializer — is_internal_operator=false)
    auto safe_tensor = Microsoft::WRL::Make<AbiSafeTensor>(ort_value, ort_api_, execution_provider_, false);
    *tensor = safe_tensor.Detach();

    return S_OK;
}

// IMLOperatorKernelCreationContextNodeWrapperPrivate - Node name methods using C API
uint32_t AbiSafeKernelCreationContext::GetUtf8NameBufferSizeInBytes() const noexcept {
    if (!kernel_info_ || !ort_api_) return 1;

    // Query the size needed for the node name (including null terminator)
    size_t size = 0;
    OrtStatus* status = ort_api_->KernelInfo_GetNodeName(kernel_info_, nullptr, &size);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return 1;
    }

    return static_cast<uint32_t>(size);
}

HRESULT AbiSafeKernelCreationContext::GetUtf8Name(uint32_t bufferSizeInBytes, char* outputName) const noexcept {
    if (bufferSizeInBytes == 0) {
        return E_INVALIDARG;
    }

    if (!outputName || !kernel_info_ || !ort_api_) {
        if (outputName && bufferSizeInBytes > 0) {
            outputName[0] = '\0';
        }
        return outputName ? S_OK : E_POINTER;
    }

    size_t size = bufferSizeInBytes;
    OrtStatus* status = ort_api_->KernelInfo_GetNodeName(kernel_info_, outputName, &size);
    if (status) {
        ort_api_->ReleaseStatus(status);
        // On error, write empty string
        outputName[0] = '\0';
        return S_OK; // Match original behavior - don't fail, just return empty
    }

    return S_OK;
}

uint32_t AbiSafeKernelCreationContext::GetWideNameBufferSizeInBytes() const noexcept {
    if (!kernel_info_ || !ort_api_) {
        return sizeof(wchar_t); // Just null terminator
    }

    // Get UTF-8 name size first
    uint32_t utf8Size = GetUtf8NameBufferSizeInBytes();
    if (utf8Size <= 1) {
        return sizeof(wchar_t); // Empty name, just null terminator
    }

    // Get the actual UTF-8 name
    std::vector<char> utf8Name(utf8Size);
    size_t size = utf8Size;
    OrtStatus* status = ort_api_->KernelInfo_GetNodeName(kernel_info_, utf8Name.data(), &size);
    if (status) {
        ort_api_->ReleaseStatus(status);
        return sizeof(wchar_t);
    }

    // Calculate required wide char buffer size
    int requiredSizeInChars = MultiByteToWideChar(CP_UTF8, 0, utf8Name.data(), static_cast<int>(utf8Size - 1), nullptr, 0);
    if (requiredSizeInChars <= 0) {
        return sizeof(wchar_t);
    }

    // Include null terminator
    return static_cast<uint32_t>((requiredSizeInChars + 1) * sizeof(wchar_t));
}

HRESULT AbiSafeKernelCreationContext::GetWideName(uint32_t bufferSizeInBytes, wchar_t* outputName) const noexcept {
    // Buffer needs to be large enough to at least hold a null terminator
    if (bufferSizeInBytes < sizeof(wchar_t)) {
        return E_INVALIDARG;
    }

    if (!outputName || !kernel_info_ || !ort_api_) {
        if (outputName) {
            outputName[0] = L'\0';
        }
        return outputName ? S_OK : E_POINTER;
    }

    // Get UTF-8 name
    uint32_t utf8Size = GetUtf8NameBufferSizeInBytes();
    std::vector<char> utf8Name(utf8Size);
    size_t size = utf8Size;
    OrtStatus* status = ort_api_->KernelInfo_GetNodeName(kernel_info_, utf8Name.data(), &size);
    if (status || utf8Size <= 1) {
        if (status) ort_api_->ReleaseStatus(status);
        outputName[0] = L'\0';
        return S_OK;
    }

    // Convert to wide char
    uint32_t bufferSizeInChars = bufferSizeInBytes / sizeof(wchar_t);
    int charsCopiedIfSucceeded = MultiByteToWideChar(
        CP_UTF8, 0,
        utf8Name.data(), static_cast<int>(utf8Size - 1), // Don't include null terminator in source
        outputName, bufferSizeInChars
    );

    if (charsCopiedIfSucceeded > 0) {
        // Write null terminator at the end of copied chars
        outputName[charsCopiedIfSucceeded] = L'\0';
        return S_OK;
    }

    // Error occurred
    DWORD lastError = GetLastError();
    if (lastError == ERROR_INSUFFICIENT_BUFFER) {
        // Buffer too small - truncate and add null terminator
        outputName[bufferSizeInChars - 1] = L'\0';
        return S_OK;
    }

    // Other error
    outputName[0] = L'\0';
    return E_FAIL;
}

HRESULT AbiSafeKernelCreationContext::GetExecutionProvider(IUnknown** executionProvider) const noexcept {
    if (!executionProvider) return E_POINTER;
    *executionProvider = nullptr;

    if (!execution_provider_) {
        return S_OK; // No execution provider, return nullptr
    }

    // PluginDmlExecutionProviderImpl implements IWinmlExecutionProvider
    // Query for it and return
    return const_cast<PluginDmlExecutionProviderImpl*>(execution_provider_)->QueryInterface(
        __uuidof(IWinmlExecutionProvider),
        reinterpret_cast<void**>(executionProvider)
    );
}

// ============================================================================
// Kernel Creation and Execution - Pure C API, NO UNSAFE CASTS!
// ============================================================================

// Equivalent to old-version InputTensorShapesDefinedOnNode: returns true only when every
// input tensor has fully concrete (non-symbolic, non-dynamic) dimensions in the ONNX TypeProto
// as seen by OrtKernelInfo. This is the conservative gate that matches the old version's
// deferral behavior for operators whose ONNX proto dims are symbolic (e.g. batch=?).
// Unlike HasTensorShapeDescription, this does NOT return true just because ORT graph inference
// propagated concrete shapes into KernelInfo at session-init time.
static bool AllInputShapesConcreteInProto(const OrtApi* api, const OrtKernelInfo* kernel_info) {
    size_t input_count = 0;
    OrtStatus* s = api->KernelInfo_GetInputCount(kernel_info, &input_count);
    if (s) { api->ReleaseStatus(s); return false; }

    for (size_t i = 0; i < input_count; ++i) {
        OrtTypeInfo* type_info = nullptr;
        s = api->KernelInfo_GetInputTypeInfo(kernel_info, i, &type_info);
        if (s) { api->ReleaseStatus(s); return false; }

        const OrtTensorTypeAndShapeInfo* tsi = nullptr;
        s = api->CastTypeInfoToTensorInfo(type_info, &tsi);
        if (s || !tsi) {
            api->ReleaseTypeInfo(type_info);
            if (s) api->ReleaseStatus(s);
            continue;  // Non-tensor input — skip
        }

        size_t dim_count = 0;
        s = api->GetDimensionsCount(tsi, &dim_count);
        if (s) { api->ReleaseTypeInfo(type_info); api->ReleaseStatus(s); return false; }

        std::vector<int64_t> dims(dim_count);
        if (dim_count > 0) {
            s = api->GetDimensions(tsi, dims.data(), dim_count);
            if (s) { api->ReleaseTypeInfo(type_info); api->ReleaseStatus(s); return false; }
            for (int64_t d : dims) {
                if (d < 0) { api->ReleaseTypeInfo(type_info); return false; }  // Dynamic dim
            }
        }

        // Also reject named symbolic dims (e.g. "batch_size") even when the numeric value is 0+
        std::vector<const char*> sym_dims(dim_count, nullptr);
        s = api->GetSymbolicDimensions(tsi, sym_dims.data(), dim_count);
        if (!s) {
            for (size_t d = 0; d < dim_count; ++d) {
                if (sym_dims[d] && sym_dims[d][0] != '\0') {
                    api->ReleaseTypeInfo(type_info);
                    return false;  // Named symbolic dim — not truly concrete
                }
            }
        } else {
            api->ReleaseStatus(s);  // GetSymbolicDimensions not supported — ignore
        }

        api->ReleaseTypeInfo(type_info);
    }
    return true;
}

OrtStatus* ORT_API_CALL DmlAbiKernel_Create(
    void* kernel_create_state,
    const OrtKernelInfo* kernel_info,
    OrtKernelImpl** kernel_out,
    std::unordered_map<uint32_t, Microsoft::WRL::ComPtr<IMLOperatorTensor>>&& constant_tensors_in
) noexcept {
    if (!kernel_create_state || !kernel_info || !kernel_out) {
        return nullptr;
    }

    try {
        *kernel_out = nullptr;
        auto* state = static_cast<DmlKernelCreationState*>(kernel_create_state);

        // Use pre-fetched constant tensors from caller.
        auto constant_tensors = std::move(constant_tensors_in);

        // Check BEFORE moving whether all required constants were supplied.
        // Mirrors unsafe PluginDmlAbiOpKernel constructor's immediateCreate condition:
        //   immediateCreate = requiredConstantCpuInputsAvailable
        //                  && (!requiresInputShapesAtCreation || InputTensorShapesDefined())
        // Both must be satisfied for immediate (eager) kernel creation. If any required constant
        // is missing (caller passed empty map because KernelInfoGetConstantInput_tensor returned
        // !isConstant for a computed-constant input), defer to lazy init.
        // Example: TopK v10+ needs K at input 1; if K comes from a preceding op rather than an
        // ONNX initializer, the map is empty and we must defer to Compute time.
        bool required_constants_available = true;
        for (uint32_t idx : state->required_constant_cpu_inputs) {
            if (constant_tensors.find(idx) == constant_tensors.end()) {
                required_constants_available = false;
                break;
            }
        }
        size_t constant_tensors_count = constant_tensors.size();  // capture before move for logging

        auto tensor_attr_cache = FetchAllTensorAttributes(kernel_info, state->ort_api, state->tensor_attribute_names);

        // Create ABI-safe kernel creation context with pre-fetched constants and tensor attrs.
        // NOTE: constant_tensors is moved here — all checks on it must happen BEFORE this line.
        auto creation_context = Microsoft::WRL::Make<AbiSafeKernelCreationContext>(
            kernel_info,
            state->ort_api,
            state->default_attributes,
            &state->required_constant_cpu_inputs,
            state->dml_execution_provider,
            std::move(constant_tensors),
            nullptr,  // No runtime context yet
            state->operator_name,
            state->is_internal_operator,
            state->requires_input_shapes_at_creation,
            std::move(tensor_attr_cache)
        );

        // Use the conservative proto-based shape check to decide whether to create the kernel
        // immediately or defer to Compute time (matching old-version InputTensorShapesDefinedOnNode).
        bool shapes_available = AllInputShapesConcreteInProto(state->ort_api, kernel_info);
        // Defer to lazy init when EITHER condition is unmet (mirrors immediateCreate logic):
        //   (a) shapes required at creation but not yet concrete (dynamic input shapes), OR
        //   (b) required constants not all supplied — will be fetched at first Compute via
        //       KernelContext_GetInput (runtime execution frame, covers computed-constant inputs).
        bool needs_lazy_init = !required_constants_available
                             || (state->requires_input_shapes_at_creation && !shapes_available);

        DML_PERF_LOG("[ABI_SAFE] DmlAbiKernel_Create: op=", state->operator_name ? state->operator_name : "?",
            "  requires_input_shapes=", state->requires_input_shapes_at_creation,
            "  shapes_available=", shapes_available,
            "  required_constants_available=", required_constants_available,
            "  needs_lazy_init=", needs_lazy_init,
            "  constant_tensors_passed=", constant_tensors_count, "\n");

        // Create the DML operator kernel using the factory (unless lazy init needed)
        Microsoft::WRL::ComPtr<IMLOperatorKernel> ml_kernel;

        if (!needs_lazy_init) {
            // Run shape inferrer only when the operator requires output shapes at creation.
            // Mirrors unsafe path: InferAndVerifyOutputSizes is called only when requiresOutputShapesAtCreation.
            // shapes_available must also be true (implied when needs_lazy_init=false and shapes were required,
            // but check explicitly in case requiresInputShapesAtCreation=false).
            if (state->shape_inferrer && state->requires_output_shapes_at_creation && shapes_available) {
                // Since needs_lazy_init=false, required_constants_available=true by construction (Gap 1 fix).
                // Re-check here for safety and to mirror the unsafe path's pre-inference constant check.
                bool required_constants_available = true;

                for (uint32_t inputIndex : state->required_constant_cpu_inputs) {
                    int is_constant = 0;
                    const OrtValue* const_value = nullptr;
                    OrtStatus* status = state->ort_api->KernelInfoGetConstantInput_tensor(
                        kernel_info, inputIndex, &is_constant, &const_value);

                    if (status || !is_constant) {
                        // Required constant input is not available
                        required_constants_available = false;

                        if (status) state->ort_api->ReleaseStatus(status);
                        break;
                    }
                    if (status) state->ort_api->ReleaseStatus(status);
                }

                if (required_constants_available) {

                    auto shape_inference_context = Microsoft::WRL::Make<AbiSafeShapeInferenceContext>(
                        nullptr,  // No runtime kernel_context at session init
                        state->ort_api,
                        state->default_attributes,
                        state->dml_execution_provider,
                        kernel_info
                    );


                Microsoft::WRL::ComPtr<IMLOperatorShapeInferenceContext> shape_context_interface;
                HRESULT shape_hr = shape_inference_context.As(&shape_context_interface);


                if (SUCCEEDED(shape_hr)) {

                    shape_hr = state->shape_inferrer->InferOutputShapes(shape_context_interface.Get());

                    if (SUCCEEDED(shape_hr)) {
                        creation_context->SetPrecomputedOutputShapes(shape_inference_context->GetInferredOutputShapes());
                    } else {
                        // Mirrors unsafe path: InferAndVerifyOutputSizes throws on failure when
                        // requiresOutputShapesAtCreation=true. Fail here rather than proceeding with
                        // no output shapes, which would produce incorrect tensor allocations.
                        return state->ort_api->CreateStatus(ORT_FAIL,
                            fmt::format("Eager shape inference failed with HR=0x{:08X} for {}",
                                (unsigned)shape_hr, state->operator_name ? state->operator_name : "unknown").c_str());
                    }
                } else {
                }
                } // if required_constants_available
            }

            Microsoft::WRL::ComPtr<IMLOperatorKernelCreationContext> context_interface;
            HRESULT hr = creation_context.As(&context_interface);
            if (FAILED(hr)) {
                return state->ort_api->CreateStatus(ORT_FAIL, "Failed to query IMLOperatorKernelCreationContext interface");
            }

            hr = state->kernel_factory->CreateKernel(context_interface.Get(), &ml_kernel);

        DML_PERF_LOG("[ABI_SAFE] eager CreateKernel: op=", state->operator_name ? state->operator_name : "?",
            "  HR=", Hex(hr), "  kernel=", (void*)ml_kernel.Get(), "\n");

            if (FAILED(hr) || !ml_kernel) {
                return state->ort_api->CreateStatus(ORT_FAIL,
                    fmt::format("ABI-safe kernel creation failed with HR=0x{:08X} for {}",
                        (unsigned)hr, state->operator_name ? state->operator_name : "unknown").c_str());
            }
        }

        // Allocate memory for OrtKernelImpl + DmlAbiKernel
        size_t total_size = sizeof(OrtKernelImpl) + sizeof(DmlAbiKernel);
        void* memory = ::operator new(total_size, std::nothrow);
        if (!memory) {
            return state->ort_api->CreateStatus(ORT_FAIL, "Failed to allocate kernel memory");
        }

        // Construct OrtKernelImpl
        OrtKernelImpl* impl = new (memory) OrtKernelImpl();
        impl->ort_version_supported = ORT_API_VERSION;
        impl->flags = 0;
        impl->Compute = DmlAbiKernel_Compute;
        impl->Release = DmlAbiKernel_Release;
        impl->PrePackWeight = nullptr;
        impl->SetSharedPrePackedWeight = nullptr;

        // Construct DmlAbiKernel immediately after OrtKernelImpl
        DmlAbiKernel* abi_kernel = new (reinterpret_cast<char*>(memory) + sizeof(OrtKernelImpl)) DmlAbiKernel();
        abi_kernel->ml_operator_kernel = std::move(ml_kernel);
        abi_kernel->ort_api = state->ort_api;
        abi_kernel->dml_execution_provider = state->dml_execution_provider;
        abi_kernel->is_internal_operator = state->is_internal_operator;
        abi_kernel->requires_input_shapes_at_creation = state->requires_input_shapes_at_creation;
        abi_kernel->requires_output_shapes_at_creation = state->requires_output_shapes_at_creation;
        abi_kernel->operator_name = state->operator_name ? state->operator_name : "Unknown";
#ifdef DML_PERF_PROFILE
        abi_kernel->perf = std::make_unique<DmlAbiKernelPerfCounters>();
#endif

        // Store shape inferrer and related data for runtime shape inference
        abi_kernel->shape_inferrer = state->shape_inferrer;
        abi_kernel->required_constant_cpu_inputs = state->required_constant_cpu_inputs;
        abi_kernel->default_attributes = state->default_attributes;
        abi_kernel->tensor_attribute_names = state->tensor_attribute_names;

        // Store lazy initialization state
        abi_kernel->needs_lazy_init = needs_lazy_init;
        if (needs_lazy_init) {
            abi_kernel->kernel_factory = state->kernel_factory;
            abi_kernel->kernel_info = kernel_info;  // Store for lazy init (ORT manages lifetime)
        }

        // Store winml_provider for resource transitions and QueueReference after each Compute call.
        // Mirrors PluginDmlAbiOpKernel which queries m_winmlProvider once in its constructor.
        if (state->dml_execution_provider) {
            auto* provider = const_cast<PluginDmlExecutionProviderImpl*>(state->dml_execution_provider);
            provider->QueryInterface(IID_PPV_ARGS(&abi_kernel->winml_provider));
        }

        // QueueReference after eager kernel creation, matching unsafe path constructor:
        //   if (m_winmlProvider) { m_winmlProvider->QueueReference(m_kernel.Get()); }
        // This ensures any GPU work recorded during CreateKernel (e.g. DML operator compilation or
        // initialization dispatches) is kept alive until the GPU has finished with it.
        // Only for eagerly-created kernels — lazy-init kernels call QueueReference in DmlAbiKernel_Compute.
        if (!needs_lazy_init && abi_kernel->winml_provider && abi_kernel->ml_operator_kernel) {
            abi_kernel->winml_provider->QueueReference(abi_kernel->ml_operator_kernel.Get());
        }

        // Store inferred output shapes for use during Compute.
        // Mirror old version: m_inferredOutputShapes is only populated when requiresOutputShapesAtCreation
        // (i.e., a shape inferrer exists). HasOutputShapeDescription() returns true only in that case.
        // Using HasTensorShapeDescription() here was wrong: it populates inferred_output_shapes with
        // static graph shapes for ops without a shape inferrer, causing GetOutputTensor to allocate
        // output tensors with wrong sizes at Compute time (old version passes nullptr for outputShapes
        // in that case, forcing the kernel to call GetOutputTensor with explicit runtime dims).
        uint32_t output_count = creation_context->GetOutputCount();
        abi_kernel->inferred_output_shapes.resize(output_count);
        for (uint32_t i = 0; i < output_count; ++i) {
            if (creation_context->HasOutputShapeDescription()) {
                uint32_t dim_count = 0;
                HRESULT hr = creation_context->GetOutputTensorDimensionCount(i, &dim_count);
                if (SUCCEEDED(hr) && dim_count > 0) {
                    abi_kernel->inferred_output_shapes[i].resize(dim_count);
                    creation_context->GetOutputTensorShape(i, dim_count, abi_kernel->inferred_output_shapes[i].data());
                }
            }
        }

        // Snapshot constants passed into eager kernel creation for shape-change detection at
        // compute time. Mirrors PluginDmlAbiOpKernel::FillConstantInputs called during the
        // immediate-create path in the constructor. Without this, shape-change recreation in
        // DmlAbiKernel_Compute would have no baseline to compare against and no constants to
        // pass to the recreated kernel for operators like Pad whose pads are not visible via
        // KernelInfoGetConstantInput_tensor or KernelContext_GetInput at compute time.
        if (!needs_lazy_init && !constant_tensors.empty()) {
            size_t input_count = 0;
            state->ort_api->KernelInfo_GetInputCount(kernel_info, &input_count);
            abi_kernel->constant_input_tensor_contents.resize(input_count, TensorContent{});
            for (const auto& kv : constant_tensors) {
                if (kv.first < input_count && kv.second) {
                    abi_kernel->constant_input_tensor_contents[kv.first] =
                        SnapshotConstantInput(kv.second.Get());
                }
            }
        }

        *kernel_out = impl;
        return nullptr;

    } catch (const std::exception& e) {
        auto* state = static_cast<DmlKernelCreationState*>(kernel_create_state);
        return state->ort_api->CreateStatus(ORT_RUNTIME_EXCEPTION, e.what());
    } catch (...) {
        auto* state = static_cast<DmlKernelCreationState*>(kernel_create_state);
        return state->ort_api->CreateStatus(ORT_RUNTIME_EXCEPTION, "Unknown exception during kernel creation");
    }
}

// Snapshots an IMLOperatorTensor's content (shape, type, raw bytes) into a TensorContent struct.
// The tensor must be a CPU tensor (IsCpuData() == true). Mirrors PluginDmlAbiOpKernel::FillConstantInputs.
static TensorContent SnapshotConstantInput(IMLOperatorTensor* tensor) {
    TensorContent content;
    if (!tensor || tensor->IsDataInterface()) {
        // nullptr or GPU tensor — required constant CPU inputs are always CPU tensors,
        // so GPU here means something unexpected; treat as invalid (no snapshot).
        content.isValid = false;
        return content;
    }

    MLOperatorTensor wrapper(tensor);
    content.isValid = true;
    content.shape = wrapper.GetShape();
    content.type = wrapper.GetTensorDataType();
    size_t byte_size = wrapper.GetUnalignedTensorByteSize();
    const auto* src = reinterpret_cast<const std::byte*>(wrapper.GetByteData());
    if (src && byte_size > 0) {
        content.data.assign(src, src + byte_size);
    }
    return content;
}

// Returns true if the current tensor's content differs from the stored snapshot.
// Mirrors PluginDmlAbiOpKernel::RequiredCpuInputChanged (tensor overload).
// NOTE: when validity differs (one null, one non-null), returns false (not changed) — intentional.
// This preserves the upstream behavior at MLOperatorAuthorImpl.h:1168 even though it looks unintuitive.
static bool ConstantInputChanged(const TensorContent& last, IMLOperatorTensor* current_tensor) {
    // A GPU tensor cannot be a required constant CPU input; treat it as invalid (not changed).
    bool current_valid = (current_tensor != nullptr && !current_tensor->IsDataInterface());

    if (last.isValid != current_valid) {
        return false;  // mirrors upstream: validity mismatch → treat as unchanged
    }

    if (last.isValid) {
        MLOperatorTensor wrapper(current_tensor);
        if (last.shape != wrapper.GetShape() ||
            last.type != wrapper.GetTensorDataType() ||
            wrapper.GetUnalignedTensorByteSize() != last.data.size() ||
            (memcmp(last.data.data(), wrapper.GetByteData(), last.data.size()) != 0)) {
            return true;
        }
    }

    return false;
}

OrtStatus* ORT_API_CALL DmlAbiKernel_Compute(
    OrtKernelImpl* this_ptr,
    OrtKernelContext* context
) noexcept {
    if (!this_ptr || !context) {
        return nullptr;
    }

    try {
        // Get the DmlAbiKernel from memory layout
        DmlAbiKernel* kernel = GetDmlAbiKernelFromImpl(this_ptr);

        DMLPERF_CTX(kernel);
        DMLPERF_INC(compute_calls);

        // If lazy initialization is needed, create the kernel now with runtime shapes (ABI-safe)
        if (kernel->needs_lazy_init && !kernel->ml_operator_kernel) {
            DMLPERF_INC(lazy_inits);

            if (!kernel->kernel_info) {
                return kernel->ort_api->CreateStatus(ORT_FAIL,
                    "Lazy initialization failed: kernel_info not stored during creation");
            }

            auto tensor_attr_cache_lazy = FetchAllTensorAttributes(kernel->kernel_info, kernel->ort_api, kernel->tensor_attribute_names);

            // Build runtime input shapes from the actual input tensors.
            // Mirrors PluginDmlAbiOpKernel::Compute which calls GetInputShapes(context) and passes
            // &inputShapes as input_shapes_override_ to AbiSafeKernelCreationContext.
            size_t lazy_input_count = 0;
            kernel->ort_api->KernelContext_GetInputCount(context, &lazy_input_count);

        DML_PERF_LOG("[ABI_SAFE] lazy-init triggered: op=", kernel->operator_name,
            "  ctx_input_count=", lazy_input_count, "\n");

            EdgeShapes runtime_input_shapes(lazy_input_count);
            for (size_t i = 0; i < lazy_input_count; ++i) {
                const OrtValue* input_value = nullptr;
                kernel->ort_api->KernelContext_GetInput(context, i, &input_value);
                if (!input_value) continue;
                OrtTensorTypeAndShapeInfo* shape_info = nullptr;
                kernel->ort_api->GetTensorTypeAndShape(input_value, &shape_info);
                if (!shape_info) continue;
                size_t dim_count = 0;
                kernel->ort_api->GetDimensionsCount(shape_info, &dim_count);
                std::vector<int64_t> dims(dim_count);
                kernel->ort_api->GetDimensions(shape_info, dims.data(), dim_count);
                kernel->ort_api->ReleaseTensorTypeAndShapeInfo(shape_info);
                auto& shape = runtime_input_shapes.GetMutableShape(i);
                shape.resize(dim_count);
                for (size_t d = 0; d < dim_count; ++d) {
                    shape[d] = dims[d] >= 0 ? static_cast<uint32_t>(dims[d]) : 0u;
                }
            }

            // Fetch required constant CPU inputs via C API (mirrors PluginDmlAbiOpKernel lazy-init flow).
            // Without this, operators like Reshape that need constant inputs at creation time will fail.
            std::unordered_map<uint32_t, Microsoft::WRL::ComPtr<IMLOperatorTensor>> lazy_constant_tensors;
            for (uint32_t input_index : kernel->required_constant_cpu_inputs) {
                int is_constant = 0;
                const OrtValue* constant_value = nullptr;
                OrtStatus* get_status = kernel->ort_api->KernelInfoGetConstantInput_tensor(
                    kernel->kernel_info, input_index, &is_constant, &constant_value);
                if (get_status == nullptr && is_constant && constant_value != nullptr) {
                    lazy_constant_tensors[input_index] = Microsoft::WRL::Make<AbiSafeTensor>(
                        constant_value, kernel->ort_api, kernel->dml_execution_provider);
                } else {
                    if (get_status) kernel->ort_api->ReleaseStatus(get_status);
                    const OrtValue* runtime_value = nullptr;
                    OrtStatus* ctx_status = kernel->ort_api->KernelContext_GetInput(
                        context, input_index, &runtime_value);
                    if (ctx_status == nullptr && runtime_value != nullptr) {
                        lazy_constant_tensors[input_index] = Microsoft::WRL::Make<AbiSafeTensor>(
                            runtime_value, kernel->ort_api, kernel->dml_execution_provider);
                    } else if (ctx_status) {
                        kernel->ort_api->ReleaseStatus(ctx_status);
                    }
                }
            }

        DML_PERF_LOG("[ABI_SAFE] lazy-init constants: op=", kernel->operator_name,
            "  resolved=", lazy_constant_tensors.size(),
            " / ", kernel->required_constant_cpu_inputs.size(), " required\n");

            // Snapshot constant tensor contents BEFORE moving lazy_constant_tensors into creation_context.
            // Mirrors PluginDmlAbiOpKernel lazy init: FillConstantInputs is called before CreateKernel.
            // The AbiSafeTensor OrtValue* pointers are valid for the duration of this Compute call.
            // After std::move(lazy_constant_tensors), the map is empty and cannot be iterated.
            std::vector<TensorContent> lazy_constant_snapshot(lazy_input_count, TensorContent{});
            for (uint32_t input_index : kernel->required_constant_cpu_inputs) {
                if (input_index >= lazy_input_count) continue;
                auto it = lazy_constant_tensors.find(input_index);
                if (it != lazy_constant_tensors.end() && it->second) {
                    lazy_constant_snapshot[input_index] = SnapshotConstantInput(it->second.Get());
                }
            }

            // Create context with runtime shapes from actual input tensors.
            // NOTE: lazy_constant_tensors is moved here — snapshot must happen before this line.
            auto creation_context = Microsoft::WRL::Make<AbiSafeKernelCreationContext>(
                kernel->kernel_info,
                kernel->ort_api,
                kernel->default_attributes,
                &kernel->required_constant_cpu_inputs,
                kernel->dml_execution_provider,
                std::move(lazy_constant_tensors),
                context,
                kernel->operator_name.c_str(),
                kernel->is_internal_operator,
                kernel->requires_input_shapes_at_creation,
                std::move(tensor_attr_cache_lazy),
                &runtime_input_shapes
            );

            if (!creation_context->HasTensorShapeDescription()) {
                return kernel->ort_api->CreateStatus(ORT_FAIL,
                    "Lazy initialization failed: shapes still unavailable at Compute time");
            }

            // Invoke shape inferrer only when output shapes are required at creation time.
            // Mirrors unsafe path: if (m_requiresOutputShapesAtCreation) { InferAndVerifyOutputSizes(...); }
            // When requiresOutputShapesAtCreation=false (e.g. Pad, Resize), the shape inferrer is
            // registered for verification only — do NOT call it here, as required constant inputs may
            // be unavailable to the AbiSafeShapeInferenceContext and will cause null-dereference crashes.
            std::vector<std::vector<uint32_t>> shape_inferrer_outputs;
            if (kernel->shape_inferrer && kernel->requires_output_shapes_at_creation) {

                auto inference_context = Microsoft::WRL::Make<AbiSafeShapeInferenceContext>(
                    context,
                    kernel->ort_api,
                    kernel->default_attributes,
                    kernel->dml_execution_provider,
                    kernel->kernel_info  // Pass kernel_info for actual node attributes
                );

                HRESULT shape_hr = kernel->shape_inferrer->InferOutputShapes(inference_context.Get());
                if (SUCCEEDED(shape_hr)) {
                    shape_inferrer_outputs = inference_context->GetInferredOutputShapes();

                    // Store inferred shapes in kernel for later use
                    kernel->inferred_output_shapes = shape_inferrer_outputs;

                    // Set precomputed shapes on creation context so GetOutputTensorShape uses them
                    creation_context->SetPrecomputedOutputShapes(shape_inferrer_outputs);
                }
            }

            Microsoft::WRL::ComPtr<IMLOperatorKernelCreationContext> context_interface;
            HRESULT hr = creation_context.As(&context_interface);
            if (FAILED(hr)) {
                return kernel->ort_api->CreateStatus(ORT_FAIL,
                    "Lazy init: Failed to query creation context interface");
            }


            // Create the actual kernel with runtime shapes
            Microsoft::WRL::ComPtr<IMLOperatorKernel> ml_kernel;
            hr = kernel->kernel_factory->CreateKernel(context_interface.Get(), &ml_kernel);

        DML_PERF_LOG("[ABI_SAFE] lazy-init CreateKernel: op=", kernel->operator_name,
            "  HR=", Hex(hr), "  kernel=", (void*)ml_kernel.Get(), "\n");

            if (FAILED(hr) || !ml_kernel) {
                return kernel->ort_api->CreateStatus(ORT_FAIL,
                    fmt::format("Lazy kernel creation failed with HR=0x{:08X} for {}",
                        (unsigned)hr, kernel->operator_name).c_str());
            }

            // Store the created kernel
            kernel->ml_operator_kernel = std::move(ml_kernel);
            kernel->needs_lazy_init = false;  // Mark as initialized

        DML_PERF_LOG("[ABI_SAFE] lazy-init SUCCESS: op=", kernel->operator_name, "  kernel ready\n");

            kernel->input_shapes_of_kernel_inference = runtime_input_shapes;  // For shape-change detection

            // Store the pre-computed constant snapshot for value-change detection on subsequent calls.
            // The snapshot was taken before lazy_constant_tensors was moved into creation_context
            // (AbiSafeTensor OrtValue* pointers are only valid for this Compute call's duration).
            kernel->constant_input_tensor_contents = std::move(lazy_constant_snapshot);

            // Store concrete output shapes (from runtime context) for use during Compute calls
            // This matches unsafe path behavior: shapes are baked in from first execution
            // Note: DML operators are compiled for specific shapes. If batch size changes, behavior is undefined.
            // This limitation exists in both safe and unsafe paths.

            // If shape inferrer was invoked, inferred_output_shapes already contains the correct shapes
            // Otherwise, retrieve shapes from the creation context only if a shape inferrer exists.
            // Mirror old version: m_inferredOutputShapes is only populated when requiresOutputShapesAtCreation.
            if (shape_inferrer_outputs.empty()) {
                uint32_t creation_context_output_count = creation_context->GetOutputCount();
                kernel->inferred_output_shapes.resize(creation_context_output_count);
                for (uint32_t i = 0; i < creation_context_output_count; ++i) {
                    if (creation_context->HasOutputShapeDescription()) {
                        uint32_t dim_count = 0;
                        hr = creation_context->GetOutputTensorDimensionCount(i, &dim_count);
                        if (SUCCEEDED(hr)) {
                            kernel->inferred_output_shapes[i].resize(dim_count);
                            if (dim_count > 0) {
                                hr = creation_context->GetOutputTensorShape(i, dim_count, kernel->inferred_output_shapes[i].data());
                                if (FAILED(hr)) {
                                    kernel->inferred_output_shapes[i].clear();
                                }
                            }
                        }
                    }
                }
            }
        }

        // Detect input shape changes between calls — mirrors PluginDmlAbiOpKernel::Compute's
        // local_input_shapes != m_inputShapesOfKernelInference check. Only applies when kernel
        // was compiled for specific shapes (requires_input_shapes_at_creation) and lazy init ran.
        DMLPERF_INC(shape_change_checks);
        DMLPERF_T0(shp);
        if (kernel->requires_input_shapes_at_creation &&
            kernel->input_shapes_of_kernel_inference.EdgeCount() > 0 &&
            !kernel->needs_lazy_init) {

            size_t shape_input_count = 0;
            kernel->ort_api->KernelContext_GetInputCount(context, &shape_input_count);
            EdgeShapes current_shapes(shape_input_count);
            for (size_t i = 0; i < shape_input_count; ++i) {
                const OrtValue* input_value = nullptr;
                kernel->ort_api->KernelContext_GetInput(context, i, &input_value);
                if (!input_value) continue;
                OrtTensorTypeAndShapeInfo* shape_info = nullptr;
                kernel->ort_api->GetTensorTypeAndShape(input_value, &shape_info);
                if (!shape_info) continue;
                size_t dim_count = 0;
                kernel->ort_api->GetDimensionsCount(shape_info, &dim_count);
                std::vector<int64_t> dims(dim_count);
                kernel->ort_api->GetDimensions(shape_info, dims.data(), dim_count);
                kernel->ort_api->ReleaseTensorTypeAndShapeInfo(shape_info);
                auto& shape = current_shapes.GetMutableShape(i);
                shape.resize(dim_count);
                for (size_t d = 0; d < dim_count; ++d) {
                    shape[d] = dims[d] >= 0 ? static_cast<uint32_t>(dims[d]) : 0u;
                }
            }

            // Collect current constant input tensors. These are needed for both value-change
            // detection (ConstantInputChanged) and — if a temporary kernel must be created —
            // as inputs to AbiSafeKernelCreationContext. Collecting before the condition check
            // avoids duplicating the fetch logic. Mirrors the unsafe path's constantInputGetter
            // calls inside the shape-change branch.
            std::unordered_map<uint32_t, Microsoft::WRL::ComPtr<IMLOperatorTensor>> tmp_constant_tensors;
            for (uint32_t input_index : kernel->required_constant_cpu_inputs) {
                // Stage 1: KernelInfoGetConstantInput_tensor (works for most operators)
                int is_constant = 0;
                const OrtValue* constant_value = nullptr;
                OrtStatus* get_status = kernel->ort_api->KernelInfoGetConstantInput_tensor(
                    kernel->kernel_info, input_index, &is_constant, &constant_value);
                if (get_status == nullptr && is_constant && constant_value != nullptr) {
                    tmp_constant_tensors[input_index] = Microsoft::WRL::Make<AbiSafeTensor>(
                        constant_value, kernel->ort_api, kernel->dml_execution_provider);
                } else {
                    if (get_status) kernel->ort_api->ReleaseStatus(get_status);
                    // Stage 2: KernelContext_GetInput (dynamically computed constants)
                    const OrtValue* runtime_value = nullptr;
                    OrtStatus* ctx_status = kernel->ort_api->KernelContext_GetInput(
                        context, input_index, &runtime_value);
                    if (ctx_status == nullptr && runtime_value != nullptr) {
                        tmp_constant_tensors[input_index] = Microsoft::WRL::Make<AbiSafeTensor>(
                            runtime_value, kernel->ort_api, kernel->dml_execution_provider);
                    } else {
                        if (ctx_status) kernel->ort_api->ReleaseStatus(ctx_status);
                        // Stage 3: snapshotted contents from kernel creation (e.g. Pad's pads,
                        // which are invisible to KernelInfoGetConstantInput_tensor on the plugin
                        // EP OrtKernelInfo* and to KernelContext_GetInput, but were resolved at
                        // session init via the GetCapability-time initializer cache).
                        if (input_index < kernel->constant_input_tensor_contents.size() &&
                            kernel->constant_input_tensor_contents[input_index].isValid) {
                            tmp_constant_tensors[input_index] = Microsoft::WRL::Make<SnapshotTensor>(
                                kernel->constant_input_tensor_contents[input_index]);
                        }
                    }
                }
            }

            // Check if any required constant input value changed since the last kernel creation.
            // Mirrors PluginDmlAbiOpKernel::Compute's requiredCpuInputsChanged check (line 1097–1114).
            bool required_cpu_inputs_changed = false;
            if (!kernel->constant_input_tensor_contents.empty()) {
                for (uint32_t input_index : kernel->required_constant_cpu_inputs) {
                    if (input_index >= kernel->constant_input_tensor_contents.size()) continue;
                    auto it = tmp_constant_tensors.find(input_index);
                    IMLOperatorTensor* current_tensor = (it != tmp_constant_tensors.end()) ? it->second.Get() : nullptr;
                    if (ConstantInputChanged(kernel->constant_input_tensor_contents[input_index], current_tensor)) {
                        required_cpu_inputs_changed = true;
                        break;
                    }
                }
            }

            if (current_shapes != kernel->input_shapes_of_kernel_inference || required_cpu_inputs_changed) {
                // Shape or constant input value changed — create a temporary kernel for this call,
                // do not replace the stored one. Mirrors unsafe path: local_kernel executed and released.
                // tmp_constant_tensors is already populated above — reuse it directly.
                auto tensor_attr_cache_tmp = FetchAllTensorAttributes(
                    kernel->kernel_info, kernel->ort_api, kernel->tensor_attribute_names);

                auto tmp_creation_context = Microsoft::WRL::Make<AbiSafeKernelCreationContext>(
                    kernel->kernel_info,
                    kernel->ort_api,
                    kernel->default_attributes,
                    &kernel->required_constant_cpu_inputs,
                    kernel->dml_execution_provider,
                    std::move(tmp_constant_tensors),
                    context,
                    kernel->operator_name.c_str(),
                    kernel->is_internal_operator,
                    kernel->requires_input_shapes_at_creation,
                    std::move(tensor_attr_cache_tmp),
                    &current_shapes
                );

                // Run shape inferrer before CreateKernel only when requiresOutputShapesAtCreation=true.
                // Mirrors unsafe path's inferShapesAndCreateKernel: if (m_requiresOutputShapesAtCreation).
                // When false (e.g. Pad), the shape inferrer is for verification only — skip it here.
                std::vector<std::vector<uint32_t>> tmp_output_shapes;
                if (kernel->shape_inferrer && kernel->requires_output_shapes_at_creation) {
                    auto tmp_inference_context = Microsoft::WRL::Make<AbiSafeShapeInferenceContext>(
                        context,
                        kernel->ort_api,
                        kernel->default_attributes,
                        kernel->dml_execution_provider,
                        kernel->kernel_info
                    );
                    HRESULT shape_hr = kernel->shape_inferrer->InferOutputShapes(tmp_inference_context.Get());
                    if (SUCCEEDED(shape_hr)) {
                        tmp_output_shapes = tmp_inference_context->GetInferredOutputShapes();
                        tmp_creation_context->SetPrecomputedOutputShapes(tmp_output_shapes);
                    }
                }

                Microsoft::WRL::ComPtr<IMLOperatorKernelCreationContext> tmp_context_interface;
                if (FAILED(tmp_creation_context.As(&tmp_context_interface))) {
                    return kernel->ort_api->CreateStatus(ORT_FAIL,
                        "Shape-change: failed to query creation context interface");
                }

                Microsoft::WRL::ComPtr<IMLOperatorKernel> tmp_kernel;
                HRESULT tmp_hr = kernel->kernel_factory->CreateKernel(tmp_context_interface.Get(), &tmp_kernel);
                if (FAILED(tmp_hr) || !tmp_kernel) {
                    return kernel->ort_api->CreateStatus(ORT_FAIL,
                        "Shape-change: temporary kernel creation failed");
                }

                // Retrieve output shapes from the temporary creation context if not already inferred above.
                if (tmp_output_shapes.empty()) {
                    uint32_t tmp_output_count = tmp_creation_context->GetOutputCount();
                    tmp_output_shapes.resize(tmp_output_count);
                    for (uint32_t i = 0; i < tmp_output_count; ++i) {
                        if (tmp_creation_context->HasOutputShapeDescription()) {
                            uint32_t dim_count = 0;
                            if (SUCCEEDED(tmp_creation_context->GetOutputTensorDimensionCount(i, &dim_count)) && dim_count > 0) {
                                tmp_output_shapes[i].resize(dim_count);
                                tmp_creation_context->GetOutputTensorShape(i, dim_count, tmp_output_shapes[i].data());
                            }
                        }
                    }
                }

                auto tmp_kernel_context = Microsoft::WRL::Make<AbiSafeKernelContext>(
                    context,
                    kernel->ort_api,
                    kernel->dml_execution_provider,
                    kernel->is_internal_operator,
                    &tmp_output_shapes,
                    kernel->shape_inferrer.Get(),
                    &kernel->required_constant_cpu_inputs,
                    kernel->default_attributes,
                    kernel->kernel_info
                );

                tmp_kernel_context->TransitionResourcesForOperatorIfRequired(true);
                HRESULT tmp_compute_hr = E_FAIL;
                try {
                    tmp_compute_hr = tmp_kernel->Compute(tmp_kernel_context.Get());
                } catch (...) {
                    tmp_kernel_context->TransitionResourcesForOperatorIfRequired(false);
                    return kernel->ort_api->CreateStatus(ORT_FAIL, "Shape-change: compute exception");
                }
                tmp_kernel_context->TransitionResourcesForOperatorIfRequired(false);

                if (FAILED(tmp_compute_hr)) {
                    return kernel->ort_api->CreateStatus(ORT_FAIL, "Shape-change: temporary kernel compute failed");
                }

                if (kernel->winml_provider) {
                    kernel->winml_provider->QueueReference(tmp_kernel.Get());
                }
                DMLPERF_INC(shape_changes_detected);
                DMLPERF_ADD(ns_shape_change_check, shp);
                return nullptr;
            }

            DMLPERF_ADD(ns_shape_change_check, shp);
        }

        // Create ABI-safe kernel context
        // Pass inferred output shapes so output tensors match DirectML's compiled expectations
        // Also pass shape inferrer and related data for runtime shape inference (e.g., Reshape)
        // Pass kernel_info so GetOutputTensor's shape inferrer path reads actual node attributes
        DMLPERF_T0(ctx);
        auto kernel_context = Microsoft::WRL::Make<AbiSafeKernelContext>(
            context,
            kernel->ort_api,
            kernel->dml_execution_provider,
            kernel->is_internal_operator,
            &kernel->inferred_output_shapes,
            kernel->shape_inferrer.Get(),
            &kernel->required_constant_cpu_inputs,
            kernel->default_attributes,
            kernel->kernel_info
        );
        DMLPERF_ADD(ns_context_construction, ctx);

        // Transition D3D12 resources to COMMON state before execution.
        // Mirrors PluginOpKernelContextWrapper constructor which calls TransitionResourcesForOperatorIfRequired(true).
        DMLPERF_T0(pre);
        kernel_context->TransitionResourcesForOperatorIfRequired(true);
        DMLPERF_ADD(ns_transition_pre, pre);

        // Execute the DML operator kernel
        HRESULT hr = E_FAIL;
        DMLPERF_T0(kc);
        try {
            hr = kernel->ml_operator_kernel->Compute(kernel_context.Get());
        } catch (const std::exception& e) {
            DMLPERF_ADD(ns_kernel_compute, kc);
            kernel_context->TransitionResourcesForOperatorIfRequired(false);
            return kernel->ort_api->CreateStatus(ORT_FAIL, e.what());
        } catch (...) {
            DMLPERF_ADD(ns_kernel_compute, kc);
            kernel_context->TransitionResourcesForOperatorIfRequired(false);
            return kernel->ort_api->CreateStatus(ORT_FAIL, "Unknown exception during compute");
        }
        DMLPERF_ADD(ns_kernel_compute, kc);

        // Transition resources back to UAV state after execution.
        // Mirrors PluginOpKernelContextWrapper::Close() which calls TransitionResourcesForOperatorIfRequired(false).
        DMLPERF_T0(post);
        kernel_context->TransitionResourcesForOperatorIfRequired(false);
        DMLPERF_ADD(ns_transition_post, post);

        if (FAILED(hr)) {
            return kernel->ort_api->CreateStatus(ORT_FAIL, "DML operator Compute failed");
        }

        // Keep the kernel alive until scheduled GPU work completes.
        // Mirrors PluginDmlAbiOpKernel::Compute which calls m_winmlProvider->QueueReference(m_kernel.Get()).
        if (kernel->winml_provider) {
            kernel->winml_provider->QueueReference(kernel->ml_operator_kernel.Get());
        }

        return nullptr;

    } catch (const std::exception& e) {
        DmlAbiKernel* kernel = GetDmlAbiKernelFromImpl(this_ptr);
        if (kernel && kernel->ort_api) {
            return kernel->ort_api->CreateStatus(ORT_RUNTIME_EXCEPTION, e.what());
        }
        return nullptr;
    } catch (...) {
        DmlAbiKernel* kernel = GetDmlAbiKernelFromImpl(this_ptr);
        if (kernel && kernel->ort_api) {
            return kernel->ort_api->CreateStatus(ORT_RUNTIME_EXCEPTION, "Unknown exception during compute");
        }
        return nullptr;
    }
}

void ORT_API_CALL DmlAbiKernel_Release(OrtKernelImpl* this_ptr) noexcept {
    if (!this_ptr) return;

    try {
        // Get the DmlAbiKernel and explicitly destroy it
        DmlAbiKernel* kernel = GetDmlAbiKernelFromImpl(this_ptr);
#ifdef DML_PERF_PROFILE
        PrintKernelPerfCounters(*kernel);
#endif
        kernel->~DmlAbiKernel();

        // Destroy the OrtKernelImpl
        this_ptr->~OrtKernelImpl();

        // Free the memory
        ::operator delete(this_ptr);

    } catch (...) {
        // Cannot throw from Release
    }
}

}  // namespace dml_ep
