// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License

// dml_execution_provider.h must come before dml_abi_kernel.h so that
// IWinmlExecutionProvider is fully defined when dml_abi_kernel.h's
// AbiSafeKernelContext class (which has Microsoft::WRL::ComPtr<IWinmlExecutionProvider>)
// is parsed by the compiler.

#include <filesystem>
#include <utility>

#include "common/plugin_ep_utils.h"

#include "dml_execution_provider.h"
#include "dml_plugin_MLOperatorAuthorImpl.h"
#include "dml_abi_kernel.h"

namespace dml_ep {

// Forward declarations for file-local helpers defined later in this file.
// UploadConstantTensorsToGpu is defined before its call sites and needs no forward declaration.
static TensorContent SnapshotConstantInput(IMLOperatorTensor* tensor);
static bool ConstantInputChanged(const TensorContent& last, IMLOperatorTensor* current_tensor);

// Uploads CPU-resident entries from constant_tensors to D3D12_HEAP_TYPE_DEFAULT GPU buffers.
// gpu_resources is resized to hold one slot per input index (sparse — GPU-resident or missing
// entries remain null). Callers must QueueReference each non-null resource after this returns.
static void UploadConstantTensorsToGpu(
    const std::unordered_map<uint32_t, Microsoft::WRL::ComPtr<IMLOperatorTensor>>& constant_tensors,
    const PluginDmlExecutionProviderImpl* provider,
    std::vector<ConstantGpuResource>& gpu_resources)
{
    // Find the max input index so we can size the sparse vector exactly once.
    uint32_t max_index = 0;
    for (const auto& kv : constant_tensors) {
        if (kv.second) max_index = std::max(max_index, kv.first);
    }
    gpu_resources.resize(static_cast<size_t>(max_index) + 1);

    Microsoft::WRL::ComPtr<ID3D12Device> d3d_device;
    ORT_THROW_IF_FAILED(provider->GetD3DDevice(d3d_device.GetAddressOf()));

    for (const auto& kv : constant_tensors) {
        IMLOperatorTensor* ml_tensor = kv.second.Get();
        if (!ml_tensor || !ml_tensor->IsCpuData()) {
            // Null or already GPU-resident — skip (GPU-resident initializers are already in VRAM).
            continue;
        }

        const void* cpu_data = ml_tensor->GetData();
        if (!cpu_data) continue;

        // Compute byte size, shape, dtype via MLOperatorTensor helper.
        MLOperatorTensor wrapper(ml_tensor);
        size_t byte_size = wrapper.GetUnalignedTensorByteSize();
        if (byte_size == 0) continue;

        // Allocate a GPU-exclusive (D3D12_HEAP_TYPE_DEFAULT) buffer.
        // Size is 4-byte aligned — required by D3D12 for UAV buffer resources.
        const uint64_t aligned_size = (static_cast<uint64_t>(byte_size) + 3u) & ~3ull;

        D3D12_HEAP_PROPERTIES heap_props = {};
        heap_props.Type = D3D12_HEAP_TYPE_DEFAULT;

        D3D12_RESOURCE_DESC res_desc = {};
        res_desc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
        res_desc.Width             = aligned_size;
        res_desc.Height            = 1;
        res_desc.DepthOrArraySize  = 1;
        res_desc.MipLevels         = 1;
        res_desc.SampleDesc.Count  = 1;
        res_desc.Layout            = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        res_desc.Flags             = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        Microsoft::WRL::ComPtr<ID3D12Resource> gpu_buf;
        HRESULT hr = d3d_device->CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &res_desc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(gpu_buf.GetAddressOf()));
        if (FAILED(hr) || !gpu_buf) continue;

        // Copy CPU data into the buffer via the provider's pooled upload heap.
        hr = provider->UploadToResource(gpu_buf.Get(), cpu_data, byte_size);
        if (FAILED(hr)) {
            continue;
        }

        ConstantGpuResource& slot = gpu_resources[kv.first];
        slot.resource = std::move(gpu_buf);
        slot.shape    = wrapper.GetShape();
        slot.dtype    = wrapper.GetTensorDataType();
    }
}

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

namespace {

MLOperatorTensorDataType ConvertToMLOperatorTensorDataType(ONNXTensorElementDataType onnx_type) {
    switch (onnx_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return MLOperatorTensorDataType::Float;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return MLOperatorTensorDataType::UInt8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return MLOperatorTensorDataType::Int8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        return MLOperatorTensorDataType::UInt16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        return MLOperatorTensorDataType::Int16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return MLOperatorTensorDataType::Int32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return MLOperatorTensorDataType::Int64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        return MLOperatorTensorDataType::String;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return MLOperatorTensorDataType::Bool;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return MLOperatorTensorDataType::Float16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return MLOperatorTensorDataType::Double;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return MLOperatorTensorDataType::UInt32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        return MLOperatorTensorDataType::UInt64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
        return MLOperatorTensorDataType::Complex64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
        return MLOperatorTensorDataType::Complex128;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        return MLOperatorTensorDataType::TensorProto_DataType_BFLOAT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
        return MLOperatorTensorDataType::TensorProto_DataType_FLOAT8E4M3FN;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
        return MLOperatorTensorDataType::TensorProto_DataType_FLOAT8E4M3FNUZ;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
        return MLOperatorTensorDataType::TensorProto_DataType_FLOAT8E5M2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
        return MLOperatorTensorDataType::TensorProto_DataType_FLOAT8E5M2FNUZ;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4:
        return MLOperatorTensorDataType::UInt4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:
        return MLOperatorTensorDataType::Int4;
    default:
        return MLOperatorTensorDataType::Undefined;
    }
}

// Pre-fetches a single tensor attribute by name from kernel_info, copying its data into
// a PreFetchedTensorAttr with owned plain bytes. This is the ONE place where the unsafe
// Fetches a single named tensor attribute via the ORT C API (KernelInfoGetAttribute_tensor).
// Returns nullopt if the attribute doesn't exist or isn't a tensor.
// Fully ABI-safe: no protobuf access, no reinterpret_cast of OrtKernelInfo.
std::optional<PreFetchedTensorAttr> TryFetchTensorAttribute(const Ort::ConstKernelInfo& kernel_info, const char* name) {
    const auto value{kernel_info.GetTensorAttribute(name, Ort::AllocatorWithDefaultOptions{})};
    if (value == nullptr) {
        return std::nullopt;
    }
    const auto type_info{value.GetTensorTypeAndShapeInfo()};
    const auto shape{type_info.GetShape()};
    const auto elem_type{type_info.GetElementType()};
    size_t elem_size{};
    switch (elem_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        elem_size = 4; break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        elem_size = 8; break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        elem_size = 1; break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        elem_size = 2; break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        elem_size = 4; break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        elem_size = 8; break;
    default:
        elem_size = 4; break;
    }
    const auto data{value.GetTensorData<std::byte>()};
    const size_t total_bytes{type_info.GetElementCount() * elem_size};
    return PreFetchedTensorAttr{
        ConvertToMLOperatorTensorDataType(elem_type),
        {shape.begin(), shape.end()},
        {data, data + total_bytes}};
}

}  // namespace

// Pre-fetches tensor-typed attributes from kernel_info into an ABI-safe cache map.
// Only probes attributes listed in tensor_attribute_names — fully ABI-safe (ORT C API only).
// AttributeValue cannot hold tensor types (GetDefaultAttributes rejects them), so tensor
// attribute names cannot be derived from defaultAttributes and must be provided explicitly.
// In this codebase, only ConstantOfShape has a tensor-typed ONNX attribute ("value").

std::unordered_map<std::string, PreFetchedTensorAttr>
FetchAllTensorAttributes(const OrtKernelInfo* ort_kernel_info, const std::vector<std::string>& attribute_names) {
    const Ort::ConstKernelInfo kernel_info{ort_kernel_info};
    std::unordered_map<std::string, PreFetchedTensorAttr> cache;
    if (kernel_info != nullptr) {
        for (const auto& attr_name : attribute_names) {
            if (auto fetched = TryFetchTensorAttribute(kernel_info, attr_name.c_str())) {
                cache.emplace(attr_name, std::move(*fetched));
            }
        }
    }
    return cache;
}

// ============================================================================
// AbiSafeTensor - IMLOperatorTensor
// ============================================================================

AbiSafeTensor::AbiSafeTensor(const OrtValue* ort_value, const bool is_internal)
    : is_internal_{is_internal}, ort_value_{const_cast<OrtValue*>(ort_value)} {
}

uint32_t AbiSafeTensor::GetDimensionCount() const noexcept
try {
    return ort_value_ == nullptr ? 0 :
        ort_value_.GetTensorTypeAndShapeInfo().GetDimensionsCount();
} catch (const Ort::Exception&) {
    return 0;
}

HRESULT AbiSafeTensor::GetShape(uint32_t dimensionCount, uint32_t* dimensions) const noexcept
try {
    if (dimensions == nullptr && dimensionCount > 0) {
        return E_POINTER;
    }
    if (ort_value_ != nullptr) {
        const auto shape{ort_value_.GetTensorTypeAndShapeInfo().GetShape()};
        if (shape.size() != dimensionCount) {
            return E_INVALIDARG;
        }
        ranges::transform(shape, dimensions,
            [](const int64_t elem) {
                return static_cast<uint32_t>(elem);
            });
    }
    return S_OK;
} catch (const Ort::Exception&) {
    return E_FAIL;
}

MLOperatorTensorDataType AbiSafeTensor::GetTensorDataType() const noexcept
try {
    if (ort_value_ == nullptr) {
        return MLOperatorTensorDataType::Undefined;
    }
    switch (ort_value_.GetTensorTypeAndShapeInfo().GetElementType()) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return MLOperatorTensorDataType::Float;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return MLOperatorTensorDataType::UInt8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return MLOperatorTensorDataType::Int8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        return MLOperatorTensorDataType::UInt16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        return MLOperatorTensorDataType::Int16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return MLOperatorTensorDataType::Int32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return MLOperatorTensorDataType::Int64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        return MLOperatorTensorDataType::String;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return MLOperatorTensorDataType::Bool;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return MLOperatorTensorDataType::Float16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return MLOperatorTensorDataType::Double;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return MLOperatorTensorDataType::UInt32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        return MLOperatorTensorDataType::UInt64;
    default:
        return MLOperatorTensorDataType::Undefined;
    }
} catch (const Ort::Exception&) {
    return MLOperatorTensorDataType::Undefined;
}

bool AbiSafeTensor::IsCpuData() const noexcept {
    if (ort_value_ == nullptr) {
        return true;
    }
    const auto memory_info{ort_value_.GetTensorMemoryInfo()};
    if (memory_info.GetDeviceType() == OrtMemoryInfoDeviceType_CPU) {
        return true;
    }
    const auto memory_type{memory_info.GetMemoryType()};
    return memory_type == OrtMemTypeCPUInput ||
           memory_type == OrtMemTypeCPUOutput;
}

void* AbiSafeTensor::GetData() noexcept {
    if (ort_value_ == nullptr || !IsCpuData()) {
        return nullptr;
    }
    return ort_value_.GetTensorMutableRawData();
}

void AbiSafeTensor::GetDataInterface(IUnknown** dataInterface) noexcept {
    if (dataInterface != nullptr) {
        *dataInterface = nullptr;
        if (ort_value_ != nullptr && !IsCpuData()) {
            if (void* data_ptr{ort_value_.GetTensorMutableRawData()};
                data_ptr != nullptr) {
                const auto allocation_info{static_cast<PluginDmlAllocationInfo*>(data_ptr)};
                if (is_internal_) {
                    *dataInterface = allocation_info;
                } else {
                    *dataInterface = allocation_info->GetResource();
                }
                allocation_info->AddRef();
            }
        }
    }
}

// ============================================================================
// AbiSafeD3D12Tensor - GPU-resident constant wrapper (no OrtValue, no re-upload)
// ============================================================================

AbiSafeD3D12Tensor::AbiSafeD3D12Tensor(
    ID3D12Resource* resource,
    std::vector<uint32_t> shape,
    MLOperatorTensorDataType dtype)
    : resource_(resource)
    , shape_(std::move(shape))
    , dtype_(dtype)
{
}

uint32_t AbiSafeD3D12Tensor::GetDimensionCount() const noexcept {
    return static_cast<uint32_t>(shape_.size());
}

HRESULT AbiSafeD3D12Tensor::GetShape(uint32_t dimensionCount, uint32_t* dimensions) const noexcept {
    if (!dimensions && dimensionCount > 0) return E_POINTER;
    const uint32_t copy_count = std::min(dimensionCount, static_cast<uint32_t>(shape_.size()));
    for (uint32_t i = 0; i < copy_count; ++i) {
        dimensions[i] = shape_[i];
    }
    return S_OK;
}

MLOperatorTensorDataType AbiSafeD3D12Tensor::GetTensorDataType() const noexcept {
    return dtype_;
}

void AbiSafeD3D12Tensor::GetDataInterface(IUnknown** dataInterface) noexcept {
    if (!dataInterface) return;
    *dataInterface = resource_.Get();
    if (*dataInterface) (*dataInterface)->AddRef();
}

// ============================================================================
// AbiSafeKernelContext - IMLOperatorKernelContext
// ============================================================================

AbiSafeKernelContext::AbiSafeKernelContext(
    OrtKernelContext* kernel_context,
    const PluginDmlExecutionProviderImpl* execution_provider,
    const bool is_internal_operator,
    const std::string_view ep_name,
    AttributeMap  default_attributes,
    const std::vector<std::vector<uint32_t>>* inferred_output_shapes,
    IMLOperatorShapeInferrer* shape_inferrer,
    const std::vector<uint32_t>& required_constant_cpu_inputs,
    const OrtKernelInfo* kernel_info,
    const std::vector<ConstantGpuResource>* constant_gpu_resources)
    : kernel_context_{kernel_context}
    , execution_provider_{execution_provider}
    , is_internal_operator_{is_internal_operator}
    , inferred_output_shapes_{inferred_output_shapes}
    , shape_inferrer_{shape_inferrer}
    , required_constant_cpu_inputs_{required_constant_cpu_inputs}
    , default_attributes_{std::move(default_attributes)}
    , kernel_info_{kernel_info}
    , constant_gpu_resources_{constant_gpu_resources}
    , ep_name_{ep_name}
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

void AbiSafeKernelContext::TransitionResourcesForOperatorIfRequired(const bool isBeforeOp) const {
    // Mirrors PluginOpKernelContextWrapper::TransitionResourcesForOperatorIfRequired.
    // External (non-internal) DML operators require D3D12 resources to be in COMMON state before
    // execution and UAV state after. Without these transitions, GPU reads/writes operate on
    // resources in the wrong state, producing corrupted or stale output.
    if (!winml_provider_ || !winml_provider_->TransitionsRequiredForOperator(is_internal_operator_)) {
        return;
    }

    std::vector<IUnknown*> resourcesToTransition;

    // Collect input resources
    const auto input_count{kernel_context_.GetInputCount()};
    for (auto i = 0; i < input_count; ++i) {
        auto value{kernel_context_.GetInput(i)};
        if (value == nullptr) {
            continue;
        }
        const auto memory_info{value.GetTensorMemoryInfo()};
        if (memory_info.GetDeviceType() == OrtMemoryInfoDeviceType_CPU) {
            continue;
        }
        if (const auto memory_type{memory_info.GetMemoryType()};
            memory_type == OrtMemTypeCPUInput ||
            memory_type == OrtMemTypeCPUOutput ||
            memory_type == OrtMemTypeCPU) {
            continue;
        }
        const auto data_ptr{value.GetTensorRawData()};
        if (data_ptr == nullptr) {
            continue;
        }
        const auto allocation_info{
            const_cast<PluginDmlAllocationInfo*>(
                static_cast<const PluginDmlAllocationInfo*>(data_ptr))
        };
        if (allocation_info == nullptr) {
            continue;
        }
        allocation_info->AddRef();
        if (is_internal_operator_) {
            resourcesToTransition.push_back(allocation_info);
        } else {
            resourcesToTransition.push_back(allocation_info->GetResource());
        }
    }

    // Collect output resources — use output_tensor_cache_ populated by GetOutputTensor calls.
    // We cannot call KernelContext_GetOutput here without a shape (would re-allocate the tensor).
    for (auto& t : output_tensor_cache_) {
        if (t != nullptr && t->IsDataInterface()) {
            IUnknown* resource = nullptr;
            t->GetDataInterface(&resource);
            if (resource != nullptr) {
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

    for (IUnknown* r : resourcesToTransition) {
        r->Release();
    }
}

HRESULT AbiSafeKernelContext::GetInputTensor(uint32_t index, IMLOperatorTensor** tensor) const noexcept
try {
    if (tensor == nullptr) {
        return E_POINTER;
    }
    *tensor = nullptr;
    if (kernel_context_.GetOrtKernelContext() == nullptr) {
        return E_FAIL;
    }

    // Fast path: return a pre-uploaded GPU buffer if one was cached for this input index.
    // This is the core of the persistent-resource optimization — no CPU→GPU upload occurs.
    if (constant_gpu_resources_ &&
        inputIndex < constant_gpu_resources_->size() &&
        (*constant_gpu_resources_)[inputIndex].resource) {
        const ConstantGpuResource& cached = (*constant_gpu_resources_)[inputIndex];
        auto gpu_tensor = Microsoft::WRL::Make<AbiSafeD3D12Tensor>(
            cached.resource.Get(), cached.shape, cached.dtype);
        *tensor = gpu_tensor.Detach();
        return S_OK;
    }
    
    if (const auto value{kernel_context_.GetInput(index)}; value != nullptr) {
        const auto abi_tensor{tensor_cache_.emplace_back(
            Microsoft::WRL::Make<AbiSafeTensor>(value, is_internal_operator_))};
        abi_tensor->AddRef();
        *tensor = abi_tensor.Get();
    }
    return S_OK;
} catch (const Ort::Exception&) {
    return E_FAIL;
}

HRESULT AbiSafeKernelContext::GetOutputTensor(uint32_t index, IMLOperatorTensor** tensor) noexcept
try {
    if (tensor == nullptr) {
        return E_POINTER;
    }
    *tensor = nullptr;
    if (kernel_context_.GetOrtKernelContext() == nullptr) {
        return E_FAIL;
    }

    // Use pre-inferred output shapes from graph compilation (source of truth)
    // If there are dynamic dimensions, fill them from runtime input shapes
    if (inferred_output_shapes_ != nullptr && index < inferred_output_shapes_->size()) {
        const auto& shape{(*inferred_output_shapes_)[index]};
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
                return GetOutputTensor(index, static_cast<uint32_t>(shape.size()), shape.data(), tensor);
            }
            // Graph shape has dynamic dimensions - fill them from runtime input
            // Assumption: dynamic dimensions match corresponding dimensions from input 0
            if (const auto input0{kernel_context_.GetInput(0)};
                input0 != nullptr) {
                if (const auto shape0{input0.GetTensorTypeAndShapeInfo().GetShape()};
                    shape0.size() == shape.size()) {
                    std::vector<uint32_t> filled_shape(shape.size());
                    for (size_t i = 0; i < shape.size(); ++i) {
                        if (shape[i] == static_cast<uint32_t>(-1)) {
                            filled_shape[i] = static_cast<uint32_t>(shape0[i]);
                        } else {
                            filled_shape[i] = shape[i];
                        }
                    }
                    return GetOutputTensor(index, static_cast<uint32_t>(filled_shape.size()), filled_shape.data(), tensor);
                }
            }
            // Fallback: couldn't fill from input, let ONNX Runtime infer
            // (will fall through to the end of the function)
        }
    }

    // For internal operators (MemcpyToHost/FromHost), output shape = input 0 shape
    if (is_internal_operator_) {
        if (const auto value{kernel_context_.GetInput(0)};
            value != nullptr) {
            const auto shape{value.GetTensorTypeAndShapeInfo().GetShape()};
            std::vector<uint32_t> output_shape(shape.size());
            ranges::transform(shape, output_shape.begin(),
                [](const int64_t elem) {
                    return static_cast<uint32_t>(elem);
                });
                return GetOutputTensor(index, static_cast<uint32_t>(output_shape.size()),
                    output_shape.data(), tensor);
        }
    }

    // For operators with shape inferrer (e.g., Reshape), call the inferrer to compute output shapes
    if (shape_inferrer_) {
        const auto inference_context{
            Microsoft::WRL::Make<AbiSafeShapeInferenceContext>(
                kernel_context_.GetOrtKernelContext(), default_attributes_, kernel_info_)};

        if (SUCCEEDED(shape_inferrer_->InferOutputShapes(inference_context.Get()))) {
            const auto& inferred_shapes{inference_context->GetInferredOutputShapes()};
            if (index < inferred_shapes.size() && !inferred_shapes[index].empty()) {
                const auto& shape = inferred_shapes[index];
                return GetOutputTensor(index, static_cast<uint32_t>(shape.size()), shape.data(), tensor);
            }
        }
    }

    // Get output without specifying shape - let ONNX Runtime infer at runtime
    // This handles: dynamic shapes, empty inferred shapes, or when shape inference is unavailable
    if (const auto value{kernel_context_.GetOutput(index, nullptr, 0)};
        value != nullptr) {
        // Create ABI-safe tensor wrapper and cache it for lifetime + post-op resource transitions
        const auto abi_tensor{
            output_tensor_cache_.emplace_back(
                Microsoft::WRL::Make<AbiSafeTensor>(value, is_internal_operator_))};
        abi_tensor->AddRef();
        *tensor = abi_tensor.Get();
    }
    return S_OK;
} catch (const Ort::Exception&) {
    return E_FAIL;
}

HRESULT AbiSafeKernelContext::GetOutputTensor(uint32_t index, uint32_t dimensionCount,
    const uint32_t* dimensionSizes, IMLOperatorTensor** tensor) noexcept
try {
    if (tensor == nullptr || (dimensionSizes == nullptr && dimensionCount > 0)) {
        return E_POINTER;
    }
    *tensor = nullptr;
    if (kernel_context_.GetOrtKernelContext() == nullptr) {
        return E_FAIL;
    }

    const std::vector<int64_t> shape{dimensionSizes, dimensionSizes + dimensionCount};
    const auto value{kernel_context_.GetOutput(index, shape.data(), shape.size())};
    if (value != nullptr) {
        const auto abi_tensor{output_tensor_cache_.emplace_back(
            Microsoft::WRL::Make<AbiSafeTensor>(value, is_internal_operator_))};
        abi_tensor->AddRef();
        *tensor = abi_tensor.Get();
    }
    return S_OK;
} catch (const Ort::Exception&) {
    return E_FAIL;
}

HRESULT AbiSafeKernelContext::AllocateTemporaryData(size_t size, IUnknown** data) const noexcept
try {
    if (data == nullptr) {
        return E_POINTER;
    }
    *data = nullptr;
    if (kernel_context_.GetOrtKernelContext() == nullptr) {
        return E_FAIL;
    }

    const Ort::MemoryInfo memory_info{"GPU", OrtMemoryInfoDeviceType_GPU,
        amd::VendorId, 0, OrtDeviceMemoryType_DEFAULT, 0, OrtDeviceAllocator};

    Ort::Allocator allocator{kernel_context_.GetAllocator(*memory_info)};
    if (allocator == nullptr) {
        return E_FAIL;
    }
    const auto memory{allocator.Alloc(size)};
    if (memory == nullptr) {
        return E_OUTOFMEMORY;
    }
    *data = static_cast<IUnknown*>(memory);

    return S_OK;
} catch (const Ort::Exception&) {
    return E_FAIL;
}

void AbiSafeKernelContext::GetExecutionInterface(IUnknown** executionInterface) const noexcept {
    if (executionInterface != nullptr) {
        if (abi_execution_object_) {
            abi_execution_object_.CopyTo(executionInterface);
        } else {
            *executionInterface = nullptr;
        }
    }
}

bool AbiSafeKernelContext::IsSequenceInputTensor(const uint32_t index) const noexcept
try {
    if (kernel_context_.GetOrtKernelContext() == nullptr) {
        return false;
    }
    const auto value {kernel_context_.GetInput(index)};
    return value != nullptr && value.GetTypeInfo().GetONNXType() == ONNX_TYPE_SEQUENCE;
} catch (const Ort::Exception&) {
    return false;
}

HRESULT AbiSafeKernelContext::GetSequenceInputInfo(uint32_t index, uint32_t* inputCount, MLOperatorTensorDataType* dataType) const noexcept
try {
    if (inputCount == nullptr || dataType == nullptr) {
        return E_POINTER;
    }
    *inputCount = 0;
    if (kernel_context_.GetOrtKernelContext() == nullptr) {
        return E_FAIL;
    }

    if (const auto value{kernel_context_.GetInput(index)};
        value != nullptr) {
        const auto type_info{value.GetTypeInfo()};
        if (type_info.GetONNXType() != ONNX_TYPE_SEQUENCE) {
            return E_INVALIDARG;
        }
        size_t count{};
        if (Ort::Status status{Ort::GetApi().GetValueCount(value, &count)};
            !status.IsOK()) {
            return E_FAIL;
        }
        *inputCount = static_cast<uint32_t>(count);
        *dataType = static_cast<MLOperatorTensorDataType>(
            type_info.GetSequenceTypeInfo()
                     .GetSequenceElementType()
                     .GetTensorTypeAndShapeInfo()
                     .GetElementType());
    }
    return S_OK;
} catch (const Ort::Exception&) {
    return E_FAIL;
}

HRESULT AbiSafeKernelContext::GetSequenceInputTensor(uint32_t index, uint32_t sequence_index, IMLOperatorTensor** tensor) const noexcept
try {
    if (tensor == nullptr) {
        return E_POINTER;
    }
    *tensor = nullptr;
    if (kernel_context_.GetOrtKernelContext()) {
        return E_FAIL;
    }
    if (const auto value{kernel_context_.GetInput(index)};
        value != nullptr) {
        Ort::AllocatorWithDefaultOptions allocator{};
        const auto element_value{value.GetValue(index, allocator)};
        if (element_value == nullptr) {
            return E_FAIL;
        }
        auto abi_tensor{
            Microsoft::WRL::Make<AbiSafeTensor>(
                element_value, is_internal_operator_)
        };
        if (abi_tensor == nullptr) {
            return E_OUTOFMEMORY;
        }
        *tensor = abi_tensor.Detach();
    }
    return S_OK;
} catch (const Ort::Exception&) {
    return E_FAIL;
}

HRESULT AbiSafeKernelContext::PrepareSequenceOutput(uint32_t outputIndex, MLOperatorTensorDataType dataType) const noexcept {
    // This method is called to prepare a sequence output
    // In ONNX Runtime, sequence outputs are typically created on-demand when GetSequenceOutputTensor is called
    // So we don't need to do anything here
    return S_OK;
}

HRESULT AbiSafeKernelContext::GetSequenceOutputTensor(uint32_t index, uint32_t sequence_index,
    MLOperatorTensorDataType data_type, uint32_t dimensions, const uint32_t* dimensionSizes, bool gpuOutput,
    IMLOperatorTensor** tensor) const noexcept
try {
    if (tensor == nullptr) {
        return E_POINTER;
    }
    *tensor = nullptr;
    if (kernel_context_.GetOrtKernelContext() == nullptr) {
        return E_FAIL;
    }

    const std::vector<int64_t> shape{dimensionSizes, dimensionSizes + dimensions};
    const Ort::Allocator allocator{gpuOutput ?
        kernel_context_.GetAllocator(*Ort::MemoryInfo{"GPU",
            OrtMemoryInfoDeviceType_GPU, amd::VendorId, 0, OrtDeviceMemoryType_DEFAULT,
            0, OrtDeviceAllocator}) : Ort::AllocatorWithDefaultOptions{}};

    if (const auto element_value{Ort::Value::CreateTensor(allocator, shape.data(),
        shape.size(), static_cast<ONNXTensorElementDataType>(data_type))};
        element_value != nullptr) {
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
        auto abi_tensor{Microsoft::WRL::Make<AbiSafeTensor>(element_value, is_internal_operator_)};
        if (abi_tensor == nullptr) {
            return E_OUTOFMEMORY;
        }
        *tensor = abi_tensor.Detach();
    }
    return S_OK;
} catch (const Ort::Exception&) {
    return E_FAIL;
}

// ============================================================================
// AbiSafeShapeInferenceContext - IMLOperatorShapeInferenceContext
// ============================================================================

AbiSafeShapeInferenceContext::AbiSafeShapeInferenceContext(OrtKernelContext* kernel_context,
    const AttributeMap& default_attributes, const OrtKernelInfo* kernel_info)
    : kernel_context_{kernel_context}, default_attributes_{default_attributes}, kernel_info_{kernel_info} {
}

HRESULT AbiSafeShapeInferenceContext::GetAttributeElementCount(const char* name,
    MLOperatorAttributeType type, uint32_t* elementCount) const noexcept
try {
    if (name == nullptr || elementCount == nullptr) {
        return E_POINTER;
    }
    *elementCount = 0;
    if (kernel_info_ != nullptr) {
        if (type == MLOperatorAttributeType::Int) {
            int64_t value{};
            if (const Ort::Status status{Ort::GetApi().KernelInfoGetAttribute_int64(
                kernel_info_, name, &value)}; status.IsOK()) {
                *elementCount = 1;
                return S_OK;
            }
        } else if (type == MLOperatorAttributeType::IntArray) {
            size_t count{};
            if (const Ort::Status status{Ort::GetApi().KernelInfoGetAttributeArray_int64(
                kernel_info_, name, nullptr, &count)}; status.IsOK()) {
                *elementCount = count;
                return S_OK;
            }
        } else if (type == MLOperatorAttributeType::Float) {
            float value{};
            if (const Ort::Status status{Ort::GetApi().KernelInfoGetAttribute_float(
                kernel_info_, name, &value)}; status.IsOK()) {
                *elementCount = 1;
                return S_OK;
            }
        } else if (type == MLOperatorAttributeType::FloatArray) {
            size_t count{};
            if (const Ort::Status status{Ort::GetApi().KernelInfoGetAttributeArray_float(
                kernel_info_, name, nullptr, &count)}; status.IsOK()) {
                *elementCount = count;
                return S_OK;
            }
        } else if (type == MLOperatorAttributeType::String) {
            size_t count{};
            if (const Ort::Status status{Ort::GetApi().KernelInfoGetAttribute_string(
                kernel_info_, name, nullptr, &count)}; status.IsOK()) {
                *elementCount = 1;
                return S_OK;
            }
        }
    }
    if (const auto it{default_attributes_.find(name)}; it != default_attributes_.end()) {
        *elementCount = static_cast<uint32_t>(it->second.ElementCount());
    }
    return S_OK;
} catch (const Ort::Exception&) {
    return E_FAIL;
} catch (const wil::ResultException& ex) {
    return ex.GetErrorCode();
}

HRESULT AbiSafeShapeInferenceContext::GetAttribute(const char* name, MLOperatorAttributeType type,
    uint32_t elementCount, size_t elementByteSize, void* value) const noexcept
try {
    if (name == nullptr || value == nullptr) {
        return E_POINTER;
    }
    if (kernel_info_ != nullptr) {
        if (type == MLOperatorAttributeType::Int) {
            if (elementCount == 1 && elementByteSize == sizeof(int64_t)) {
                *static_cast<int64_t*>(value) = kernel_info_.GetAttribute<int64_t>(name);
                return S_OK;
            }
        } else if (type == MLOperatorAttributeType::IntArray) {
            size_t count{elementCount};
            const Ort::Status status{Ort::GetApi().KernelInfoGetAttributeArray_int64(
                kernel_info_, name, static_cast<int64_t*>(value), &count)};
            if (status.IsOK()) {
                return S_OK;
            }
        } else if (type == MLOperatorAttributeType::Float) {
            if (elementCount == 1 && elementByteSize == sizeof(float)) {
                *static_cast<float*>(value) = kernel_info_.GetAttribute<float>(name);
                return S_OK;
            }
        } else if (type == MLOperatorAttributeType::FloatArray) {
            size_t count{elementCount};
            Ort::Status status{Ort::GetApi().KernelInfoGetAttributeArray_float(
                kernel_info_, name, static_cast<float*>(value), &count)};
            if (status.IsOK()) {
                return S_OK;
            }
        }
    }
    if (const auto it{default_attributes_.find(name)}; it != default_attributes_.end()) {
        it->second.GetAttribute(type, elementCount, elementByteSize, value);
        return S_OK;
    }
    return E_INVALIDARG;
} catch (const Ort::Exception&) {
    return E_FAIL;
} catch (const wil::ResultException& ex) {
    return ex.GetErrorCode();
}

HRESULT AbiSafeShapeInferenceContext::GetStringAttributeElementLength(
    const char* name, uint32_t index, uint32_t* attributeElementByteLength) const noexcept
try {
    if (name == nullptr || attributeElementByteLength == nullptr) {
        return E_POINTER;
    }
    if (kernel_info_ != nullptr && index == 0) {
        size_t size{};
        if (const Ort::Status status{Ort::GetApi().KernelInfoGetAttribute_string(kernel_info_, name, nullptr, &size)};
            status.IsOK()) {
            *attributeElementByteLength = static_cast<uint32_t>(size);
            return S_OK;
        }
    }
    if (const auto it{default_attributes_.find(name)}; it != default_attributes_.end()) {
        const auto s{it->second.GetStringAttribute(name, index)};
        if (s == nullptr) {
            return E_INVALIDARG;
        }
        // Return length including null terminator
        *attributeElementByteLength = static_cast<uint32_t>(s->length() + 1);
        return S_OK;
    }
    return E_INVALIDARG;
} catch (const Ort::Exception&) {
    return E_FAIL;
} catch (const wil::ResultException& e) {
    return e.GetErrorCode();
}

HRESULT AbiSafeShapeInferenceContext::GetStringAttributeElement(const char* name, uint32_t element_index,
    uint32_t attributeElementByteLength, char* attributeElement) const noexcept
try {
    if (name == nullptr || attributeElement == nullptr) {
        return E_POINTER;
    }
    if (kernel_info_ != nullptr && element_index == 0) {
        size_t size{attributeElementByteLength};
        if (const Ort::Status status{Ort::GetApi().KernelInfoGetAttribute_string(
            kernel_info_, name, attributeElement, &size)}; status.IsOK()) {
            return S_OK;
        }
    }
    if (const auto it{default_attributes_.find(name)}; it != default_attributes_.end()) {
        const auto s{it->second.GetStringAttribute(name, element_index)};
        if (s == nullptr) {
            return E_INVALIDARG;
        }
        if (attributeElementByteLength < s->length() + 1) {
            return E_INVALIDARG;
        }
        s->copy(attributeElement, attributeElementByteLength);
        return S_OK;
    }
    return E_INVALIDARG;
} catch (const Ort::Exception&) {
    return E_FAIL;
} catch (const wil::ResultException& e) {
    return e.GetErrorCode();
}

uint32_t AbiSafeShapeInferenceContext::GetInputCount() const noexcept
try {
    if (kernel_context_.GetOrtKernelContext() != nullptr) {
        return static_cast<uint32_t>(kernel_context_.GetInputCount());
    }
    if (kernel_info_ != nullptr) {
        return static_cast<uint32_t>(kernel_info_.GetInputCount());
    }
    return 0;
} catch (const Ort::Exception&) {
    return 0;
}

uint32_t AbiSafeShapeInferenceContext::GetOutputCount() const noexcept
try {
    if (kernel_context_.GetOrtKernelContext() != nullptr) {
        return static_cast<uint32_t>(kernel_context_.GetOutputCount());
    }
    if (kernel_info_ != nullptr) {
        return static_cast<uint32_t>(kernel_info_.GetOutputCount());
    }
    return 0;
} catch (const Ort::Exception&) {
    return 0;
}

bool AbiSafeShapeInferenceContext::IsInputValid(uint32_t index) const noexcept
try {
    if (kernel_context_.GetOrtKernelContext() != nullptr) {
        return kernel_context_.GetInput(index) != nullptr;
    }
    if (kernel_info_ != nullptr) {
        return kernel_info_.GetInputTypeInfo(index) != nullptr;
    }
    return false;
} catch (const Ort::Exception&) {
    return false;
}

bool AbiSafeShapeInferenceContext::IsOutputValid(uint32_t index) const noexcept {
    return kernel_info_ == nullptr ? false :
        kernel_info_.GetOutputTypeInfo(index) != nullptr;
}

HRESULT AbiSafeShapeInferenceContext::GetInputEdgeDescription(
    uint32_t index, MLOperatorEdgeDescription* description) const noexcept
try {
    if (description == nullptr) {
        return E_POINTER;
    }
    Ort::TypeInfo type_info;
    if (kernel_context_.GetOrtKernelContext() != nullptr) {
        if (const auto value{kernel_context_.GetInput(index)}; value != nullptr) {
            type_info = value.GetTypeInfo();
        }
    } else if (kernel_info_ != nullptr) {
        type_info = kernel_info_.GetInputTypeInfo(index);
    }
    if (type_info == nullptr) {
        return E_INVALIDARG;
    }
    if (type_info.GetONNXType() == ONNX_TYPE_TENSOR) {
        description->edgeType = MLOperatorEdgeType::Tensor;
        description->tensorDataType =  static_cast<MLOperatorTensorDataType>(
            type_info.GetTensorTypeAndShapeInfo().GetElementType());
    } else {
        description->edgeType = MLOperatorEdgeType::Undefined;
    }
    return S_OK;
} catch (const Ort::Exception&) {
    return E_FAIL;
}

HRESULT AbiSafeShapeInferenceContext::GetInputTensorDimensionCount(uint32_t index, uint32_t* count) const noexcept
try {
    if (count == nullptr) {
        return E_POINTER;
    }
    if (kernel_context_.GetOrtKernelContext() != nullptr) {
        if (const auto value{kernel_context_.GetInput(index)};
            value != nullptr) {
            *count = value.GetTensorTypeAndShapeInfo().GetDimensionsCount();
            return S_OK;
        }
    }
    if (kernel_info_ != nullptr) {
        if (const auto type_info{kernel_info_.GetInputTypeInfo(index)};
            type_info != nullptr) {
            *count = type_info.GetTensorTypeAndShapeInfo().GetDimensionsCount();
            return S_OK;
        }
    }
    return E_INVALIDARG;
} catch (const Ort::Exception&) {
    return E_FAIL;
}

HRESULT AbiSafeShapeInferenceContext::GetInputTensorShape(uint32_t index, uint32_t count, uint32_t* dimensions) const noexcept
try {
    if (dimensions == nullptr && count > 0) {
        return E_POINTER;
    }
    if (kernel_context_.GetOrtKernelContext() != nullptr) {
        if (const auto value{kernel_context_.GetInput(index)};
            value != nullptr) {
            const auto shape{value.GetTensorTypeAndShapeInfo().GetShape()};
            ranges::transform(shape, dimensions,
                [](const int64_t elem) {
                    return static_cast<uint32_t>(elem);
            });
            return S_OK;
        }
    }
    if (kernel_info_ != nullptr) {
        if (const auto type_info{kernel_info_.GetInputTypeInfo(index)};
            type_info != nullptr) {
            const auto shape{type_info.GetTensorTypeAndShapeInfo().GetShape()};
            ranges::transform(shape, dimensions,
                [](const int64_t elem) {
                   return static_cast<uint32_t>(elem);
                });
            return S_OK;
        }
    }
    return E_INVALIDARG;
} catch (const Ort::Exception&) {
    return E_FAIL;
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

HRESULT AbiSafeShapeInferenceContext::GetConstantInputTensor(uint32_t index, IMLOperatorTensor** tensor) const noexcept
try {
    if (tensor == nullptr) {
        return E_POINTER;
    }
    *tensor = nullptr;
    if (const auto it{constant_tensor_cache_.find(index)}; it != constant_tensor_cache_.end()) {
        *tensor = it->second.Get();
        (*tensor)->AddRef();
        return S_OK;
    }
    Ort::ConstValue value{};
    if (kernel_context_.GetOrtKernelContext() != nullptr) {
        value = kernel_context_.GetInput(index);
        DML_PERF_LOG("[ABI_SAFE] ShapeCtx::GetConstantInput[", inputIndex, "]: value=", (void*)value, "\n");
    }
    if (value == nullptr && kernel_info_ != nullptr) {
        if (index >= GetInputCount()) {
            return S_OK;
        }
        int is_constant{};
        value = kernel_info_.GetTensorConstantInput(index, &is_constant);
        if (value == nullptr && is_constant == 0) {
            return S_OK;
        }
    }
    if (value == nullptr) {
        return S_OK;
    }

    const auto abi_tensor{Microsoft::WRL::Make<AbiSafeTensor>(value)};
    constant_tensor_cache_[index] = abi_tensor;

    *tensor = abi_tensor.Get();
    (*tensor)->AddRef();

    return S_OK;
} catch (const Ort::Exception&) {
    return E_FAIL;
}

HRESULT AbiSafeShapeInferenceContext::TryGetConstantInputTensor(uint32_t index, IMLOperatorTensor** tensor) const noexcept {
    return GetConstantInputTensor(index, tensor);
}

HRESULT AbiSafeShapeInferenceContext::GetSequenceInputInfo(
    uint32_t index, uint32_t* inputCount, MLOperatorTensorDataType* dataType) const noexcept {
    // Sequence inputs not supported in shape inference context yet
    return E_NOTIMPL;
}

HRESULT AbiSafeShapeInferenceContext::GetSequenceInputTensorDimensionCount(
    uint32_t index, uint32_t sequence_index, uint32_t* dimensionCount) const noexcept {
    // Sequence inputs not supported in shape inference context yet
    return E_NOTIMPL;
}

HRESULT AbiSafeShapeInferenceContext::GetSequenceInputTensorShape(
    uint32_t index, uint32_t sequence_index, uint32_t dimensionCount, uint32_t* dimensions) const noexcept {
    // Sequence inputs not supported in shape inference context yet
    return E_NOTIMPL;
}

// ============================================================================
// PreFetchedTensorAttrWrapper
// ============================================================================

PreFetchedTensorAttrWrapper::PreFetchedTensorAttrWrapper(const MLOperatorTensorDataType data_type,
    std::vector<uint32_t> shape, std::vector<std::byte> raw_bytes)
    : data_type_{data_type}, shape_{std::move(shape)}, raw_bytes_{std::move(raw_bytes)} {
}

uint32_t PreFetchedTensorAttrWrapper::GetDimensionCount() const noexcept {
    return static_cast<uint32_t>(shape_.size());
}

HRESULT PreFetchedTensorAttrWrapper::GetShape(uint32_t dim_count, uint32_t* dims) const noexcept {
    if (dims == nullptr || dim_count != static_cast<uint32_t>(shape_.size())) {
        return E_INVALIDARG;
    }
    memcpy(dims, shape_.data(), dim_count * sizeof(uint32_t));
    return S_OK;
}

MLOperatorTensorDataType PreFetchedTensorAttrWrapper::GetTensorDataType() const noexcept {
    return data_type_;
}

void* PreFetchedTensorAttrWrapper::GetData() noexcept {
    return raw_bytes_.data();
}

void PreFetchedTensorAttrWrapper::GetDataInterface(IUnknown** dataInterface) noexcept {
    if (dataInterface != nullptr) {
        *dataInterface = nullptr;
    }
}

// ============================================================================
// AbiSafeKernelCreationContext - IMLOperatorKernelCreationContext
// ============================================================================

AbiSafeKernelCreationContext::AbiSafeKernelCreationContext(
    const OrtKernelInfo* kernel_info,
    AttributeMap default_attributes,
    const std::vector<uint32_t>& required_constant_cpu_inputs,
    const PluginDmlExecutionProviderImpl* execution_provider,
    std::unordered_map<uint32_t, Microsoft::WRL::ComPtr<IMLOperatorTensor>>&& constant_tensors,
    OrtKernelContext* runtime_context,
    const char* operator_name,
    const bool is_internal_operator,
    const bool requires_input_shapes_at_creation,
    std::unordered_map<std::string, PreFetchedTensorAttr> tensor_attribute_cache,
    const EdgeShapes* input_shapes_override)
    : kernel_info_{kernel_info}
    , default_attributes_{std::move(default_attributes)}
    , required_constant_cpu_inputs_{required_constant_cpu_inputs}
    , execution_provider_{execution_provider}
    , constant_tensors_{std::move(constant_tensors)}
    , runtime_context_{runtime_context}
    , operator_name_{operator_name}
    , is_internal_operator_{is_internal_operator}
    , input_shapes_override_{input_shapes_override}
    , requires_input_shapes_at_creation_{requires_input_shapes_at_creation}
    , tensor_attribute_cache_{std::move(tensor_attribute_cache)}
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

uint32_t AbiSafeKernelCreationContext::GetInputCount() const noexcept
try {
    return static_cast<uint32_t>(kernel_info_.GetInputCount());
} catch (const Ort::Exception&) {
    return 0;
}

uint32_t AbiSafeKernelCreationContext::GetOutputCount() const noexcept
try {
    return static_cast<uint32_t>(kernel_info_.GetOutputCount());
} catch (const Ort::Exception&) {
    return 0;
}

bool AbiSafeKernelCreationContext::IsInputValid(const uint32_t index) const noexcept
try {
    return kernel_info_.GetInputTypeInfo(index) != nullptr;
} catch (const Ort::Exception&) {
    return false;
}

bool AbiSafeKernelCreationContext::IsOutputValid(const uint32_t index) const noexcept
try {
    return kernel_info_.GetOutputTypeInfo(index) != nullptr;
} catch (const Ort::Exception&) {
    return false;
}

HRESULT AbiSafeKernelCreationContext::GetInputEdgeDescription(
    const uint32_t index, MLOperatorEdgeDescription* description) const noexcept
try {
    if (description == nullptr) {
        return E_POINTER;
    }
    const auto type_info{kernel_info_.GetInputTypeInfo(index)};
    if (const auto onnx_type{type_info.GetONNXType()}; onnx_type == ONNX_TYPE_TENSOR) {
        description->edgeType = MLOperatorEdgeType::Tensor;
        description->tensorDataType = ConvertToMLOperatorTensorDataType(
            type_info.GetTensorTypeAndShapeInfo().GetElementType());
    } else if (onnx_type == ONNX_TYPE_SEQUENCE) {
        description->edgeType = MLOperatorEdgeType::SequenceTensor;
        description->tensorDataType = MLOperatorTensorDataType::Undefined;
    } else {
        description->edgeType = MLOperatorEdgeType::Undefined;
        description->tensorDataType = MLOperatorTensorDataType::Undefined;
    }
    return S_OK;
} catch (const Ort::Exception&) {
    return E_FAIL;
}

HRESULT AbiSafeKernelCreationContext::GetOutputEdgeDescription(
    const uint32_t index, MLOperatorEdgeDescription* description) const noexcept
try {
    if (description ==  nullptr) {
        return E_POINTER;
    }
    const auto type_info{kernel_info_.GetOutputTypeInfo(index)};
    if (const auto onnx_type{type_info.GetONNXType()}; onnx_type == ONNX_TYPE_TENSOR) {
        description->edgeType = MLOperatorEdgeType::Tensor;
        description->tensorDataType = ConvertToMLOperatorTensorDataType(
            type_info.GetTensorTypeAndShapeInfo().GetElementType());
    } else if (onnx_type == ONNX_TYPE_SEQUENCE) {
        description->edgeType = MLOperatorEdgeType::SequenceTensor;
        description->tensorDataType = MLOperatorTensorDataType::Undefined;
    } else {
        description->edgeType = MLOperatorEdgeType::Undefined;
        description->tensorDataType = MLOperatorTensorDataType::Undefined;
    }
    return S_OK;
} catch (const Ort::Exception&) {
    return E_FAIL;
}

bool AbiSafeKernelCreationContext::HasTensorShapeDescription() const noexcept {
    // Mirror old OpKernelInfoWrapper::HasTensorShapeDescription() = m_allowInputShapeQuery
    // which is a static flag from kernel registration, NOT a dynamic shape check.
    // At lazy-init (Compute time) with runtime_context_, shapes are always available.
    if (runtime_context_.GetOrtKernelContext() != nullptr) {
        return true;
    }
    return requires_input_shapes_at_creation_;
}

HRESULT AbiSafeKernelCreationContext::GetTensorShapeDescription(IMLOperatorTensorShapeDescription** shapeInfo) const noexcept
try {
    if (shapeInfo == nullptr) {
        return E_POINTER;
    }
    *shapeInfo = nullptr;
    if (!HasTensorShapeDescription()) {
        // Dynamic shapes not supported - return E_FAIL to trigger fallback to unsafe path
        return E_FAIL;
    }
    // If shapes are available, return this object as the shape description interface
    Microsoft::WRL::ComPtr<IMLOperatorTensorShapeDescription> ret{const_cast<AbiSafeKernelCreationContext*>(this)};
    *shapeInfo = ret.Detach();
    return S_OK;
} catch (const Ort::Exception&) {
    return E_FAIL;
}

HRESULT AbiSafeKernelCreationContext::GetInputTensorDimensionCount(uint32_t index, uint32_t* dimensionCount) const noexcept
try {
    if (dimensionCount == nullptr) {
        return E_POINTER;
    }
    if (input_shapes_override_ != nullptr) {
        if (index >= static_cast<uint32_t>(input_shapes_override_->EdgeCount())) {
            return E_INVALIDARG;
        }
        *dimensionCount = static_cast<uint32_t>(input_shapes_override_->GetShape(index).size());
    } else {
        *dimensionCount =
            kernel_info_.GetInputTypeInfo(index)
                        .GetTensorTypeAndShapeInfo()
                        .GetDimensionsCount();
    }
    return S_OK;
} catch (const Ort::Exception&) {
    return E_FAIL;
} catch (const wil::ResultException& e) {
    return e.GetErrorCode();
}

HRESULT AbiSafeKernelCreationContext::GetInputTensorShape(
    uint32_t index, uint32_t dimensionCount, uint32_t* dimensions) const noexcept
try {
    if (dimensions == nullptr && dimensionCount > 0) {
        return E_POINTER;
    }
    if (input_shapes_override_ != nullptr) {
        if (index >= static_cast<uint32_t>(input_shapes_override_->EdgeCount())) {
            return E_INVALIDARG;
        }
        const auto& shape = input_shapes_override_->GetShape(index);
        if (dimensionCount != static_cast<uint32_t>(shape.size())) {
            return E_INVALIDARG;
        }
        ranges::transform(shape, dimensions, [](const int64_t elem) {
                return static_cast<uint32_t>(elem);
            });
    } else {
        const auto shape{kernel_info_.GetInputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape()};
        ranges::transform(shape, dimensions, [](const int64_t elem) {
                return static_cast<uint32_t>(elem);
            });
    }
    return S_OK;
} catch (const Ort::Exception&) {
    return E_FAIL;
} catch (const wil::ResultException& e) {
    return e.GetErrorCode();
}

bool AbiSafeKernelCreationContext::HasOutputShapeDescription() const noexcept {
    // Output shapes are only available when they were precomputed by the shape inferrer.
    // This mirrors OpKernelInfoWrapper::HasOutputShapeDescription() = m_allowOutputShapeQuery
    // which is set to true only when requiresOutputShapesAtCreation (i.e., a shape inferrer exists).
    // Returning HasTensorShapeDescription() here was wrong: it would cause output TensorDescs
    // to be built from graph KernelInfo shapes when no shape inferrer ran, producing wrong strides.
    return !precomputed_output_shapes_.empty();
}

HRESULT AbiSafeKernelCreationContext::GetOutputTensorDimensionCount(uint32_t index, uint32_t* dimensionCount) const noexcept
try {
    if (dimensionCount == nullptr) {
        return E_POINTER;
    }
    if (index < precomputed_output_shapes_.size() && !precomputed_output_shapes_[index].empty()) {
        *dimensionCount = static_cast<uint32_t>(precomputed_output_shapes_[index].size());
    } else {
        *dimensionCount =
            kernel_info_.GetOutputTypeInfo(index)
                        .GetTensorTypeAndShapeInfo()
                        .GetDimensionsCount();
    }
    return S_OK;
} catch (const Ort::Exception&) {
    return E_FAIL;
} catch (const wil::ResultException& e) {
    return e.GetErrorCode();
}

HRESULT AbiSafeKernelCreationContext::GetOutputTensorShape(
    uint32_t index, uint32_t dimensionCount, uint32_t* dimensions) const noexcept
try {
    if (dimensions == nullptr && dimensionCount > 0) {
        return E_POINTER;
    }
    if (index < precomputed_output_shapes_.size() && !precomputed_output_shapes_[index].empty()) {
        if (const auto& precomputed_shape = precomputed_output_shapes_[index]; precomputed_shape.size() == dimensionCount) {
            ranges::transform(precomputed_shape, dimensions, [](const int64_t elem) {
                    return static_cast<uint32_t>(elem);
                });
        }
    } else {
        const auto shape{kernel_info_.GetOutputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape()};
        ranges::transform(shape, dimensions, [](const int64_t elem) {
                return static_cast<uint32_t>(elem);
            });
    }
    return S_OK;
} catch (const Ort::Exception&) {
    return E_FAIL;
}

void AbiSafeKernelCreationContext::GetExecutionInterface(IUnknown** executionInterface) const noexcept {
    if (executionInterface != nullptr) {
        abi_execution_object_.CopyTo(executionInterface);
    }
}

HRESULT AbiSafeKernelCreationContext::GetAttribute(const char* name, MLOperatorAttributeType type,
    uint32_t elementCount, size_t elementByteSize, void* value) const noexcept
try {
    if (name == nullptr || value == nullptr) {
        return E_POINTER;
    }
    if (type == MLOperatorAttributeType::Float) {
        if (elementCount == 1 && elementByteSize == sizeof(float)) {
            *static_cast<float*>(value) = kernel_info_.GetAttribute<float>(name);
            return S_OK;
        }
    } else if (type == MLOperatorAttributeType::FloatArray) {
        size_t count{elementCount};
        if (const Ort::Status status{Ort::GetApi().KernelInfoGetAttributeArray_float(
            kernel_info_, name, static_cast<float*>(value), &count)}; status.IsOK()) {
            return S_OK;
        }
    } else if (type == MLOperatorAttributeType::Int) {
        if (elementCount == 1 && elementByteSize == sizeof(int64_t)) {
            *static_cast<int64_t*>(value) = kernel_info_.GetAttribute<int64_t>(name);
            return S_OK;
        }
    } else if (type == MLOperatorAttributeType::IntArray) {
        size_t count{elementCount};
        if (const Ort::Status status{Ort::GetApi().KernelInfoGetAttributeArray_int64(
            kernel_info_, name, static_cast<int64_t*>(value), &count)}; status.IsOK()) {
            return S_OK;
        }
    } else if (type == MLOperatorAttributeType::String) {
        if (elementCount == 1) {
            size_t size{elementByteSize};
            if (const Ort::Status status{Ort::GetApi().KernelInfoGetAttribute_string(
                kernel_info_, name, static_cast<char*>(value), &size)}; status.IsOK()) {
                return S_OK;
            }
        }
    } else {
        return E_NOTIMPL;
    }
    if (const auto it{default_attributes_.find(name)}; it != default_attributes_.end()) {
        it->second.GetAttribute(type, elementCount, elementByteSize, value);
        return S_OK;
    }
    return E_FAIL;
} catch (const Ort::Exception&) {
    return E_FAIL;
}

HRESULT AbiSafeKernelCreationContext::GetAttributeElementCount(
    const char* name, const MLOperatorAttributeType type, uint32_t* elementCount) const noexcept
try {
    if (name == nullptr || elementCount == nullptr) {
        return E_POINTER;
    }
    *elementCount = 0;
    try {
        if (type == MLOperatorAttributeTypeTensor) {
            if (tensor_attribute_cache_.count(name) == 0) {
                return E_INVALIDARG;
            }
            *elementCount = 1;
        } else if (type == MLOperatorAttributeType::Float) {
            const auto _{kernel_info_.GetAttribute<float>(name)};
            *elementCount = 1;
        } else if (type == MLOperatorAttributeType::FloatArray) {
            size_t count{};
            if (const Ort::Status status{Ort::GetApi().KernelInfoGetAttributeArray_float(
                kernel_info_, name, nullptr, &count)}; status.IsOK()) {
                *elementCount = static_cast<uint32_t>(count);
            }
        } else if (type == MLOperatorAttributeType::Int) {
            const auto _{kernel_info_.GetAttribute<int64_t>(name)};
            *elementCount = 1;
        } else if (type == MLOperatorAttributeType::IntArray) {
            size_t count{};
            if (const Ort::Status status{Ort::GetApi().KernelInfoGetAttributeArray_int64(
                kernel_info_, name, nullptr, &count)}; status.IsOK()) {
                *elementCount = static_cast<uint32_t>(count);
            }
        } else if (type == MLOperatorAttributeType::String) {
            size_t size{};
            if (const Ort::Status status{Ort::GetApi().KernelInfoGetAttribute_string(
                kernel_info_, name, nullptr, &size)}; status.IsOK()) {
                *elementCount = 1;
            }
        }
    } catch (const Ort::Exception&) {
    }

    if (*elementCount == 0) {
        if (const auto it{default_attributes_.find(name)}; it != default_attributes_.end()) {
            *elementCount = static_cast<uint32_t>(it->second.ElementCount());
        }
    }
    return S_OK;
} catch (const wil::ResultException& e) {
    return e.GetErrorCode();
}

HRESULT AbiSafeKernelCreationContext::GetStringAttributeElementLength(
    const char* name, uint32_t index, uint32_t* attributeElementByteSize) const noexcept
try {
    if (name == nullptr || attributeElementByteSize == nullptr) {
        return E_POINTER;
    }
    if (index == 0) {
        size_t size{};
        if (const Ort::Status status{Ort::GetApi().KernelInfoGetAttribute_string(
            kernel_info_, name, nullptr, &size)}; status.IsOK()) {
            *attributeElementByteSize = static_cast<uint32_t>(size);
            return S_OK;
        }
    }
    if (const auto it{default_attributes_.find(name)}; it != default_attributes_.end()) {
        if (const auto s{it->second.GetStringAttribute(name, index)}; s != nullptr) {
            *attributeElementByteSize = static_cast<uint32_t>(s->length() + 1);
            return S_OK;
        }
    }
    return E_INVALIDARG;
} catch (const Ort::Exception&) {
    return E_FAIL;
} catch (const wil::ResultException& e) {
    return e.GetErrorCode();
}

HRESULT AbiSafeKernelCreationContext::GetStringAttributeElement(
    const char* name, uint32_t index, uint32_t attributeElementByteSize, char* attributeElement) const noexcept
try {
    if (name == nullptr || attributeElement == nullptr) {
        return E_POINTER;
    }
    if (index == 0) {
        size_t size{attributeElementByteSize};
        if (Ort::Status status{Ort::GetApi().KernelInfoGetAttribute_string(
            kernel_info_, name, attributeElement, &size)}; status.IsOK()) {
            return S_OK;
        }
    }
    if (!default_attributes_.empty()) {
        if (const auto it{default_attributes_.find(name)}; it != default_attributes_.end()) {
            const auto s{it->second.GetStringAttribute(name, index)};
            if (s != nullptr && attributeElementByteSize >= s->size() + 1) {
                s->copy(attributeElement, attributeElementByteSize);
                return S_OK;
            }
        }
    }
    return E_INVALIDARG;
} catch (const Ort::Exception&) {
    return E_FAIL;
} catch (const wil::ResultException& e) {
    return e.GetErrorCode();
}

HRESULT AbiSafeKernelCreationContext::GetTensorAttribute(const char* name, IMLOperatorTensor** tensor) const noexcept
try {
    if (name == nullptr || tensor == nullptr) {
        return E_POINTER;
    }
    *tensor = nullptr;
    if (const auto it{tensor_attribute_cache_.find(name)}; it != tensor_attribute_cache_.end()) {
        auto wrapper = Microsoft::WRL::Make<PreFetchedTensorAttrWrapper>(
            it->second.data_type, it->second.shape, it->second.raw_bytes);
        *tensor = wrapper.Detach();
        return S_OK;
    }
    return E_INVALIDARG;
} catch (const Ort::Exception&) {
    return E_FAIL;
} catch (const wil::ResultException& e) {
    return e.GetErrorCode();
}

HRESULT AbiSafeKernelCreationContext::GetConstantInputTensor(uint32_t index, IMLOperatorTensor** tensor) const noexcept
try {
    if (tensor == nullptr) {
        return E_POINTER;
    }
    *tensor = nullptr;
    if (const auto it{constant_tensors_.find(index)}; it != constant_tensors_.end() && it->second) {
        *tensor = it->second.Get();
        (*tensor)->AddRef();  // Caller will release
        return S_OK;
    }
    if (kernel_info_ != nullptr) {
        int is_constant{};
        if (const auto value{kernel_info_.GetTensorConstantInput(index, &is_constant)}; is_constant && value != nullptr) {
            auto safe_tensor = Microsoft::WRL::Make<AbiSafeTensor>(value);
            *tensor = safe_tensor.Detach();
            return S_OK;
        }
    }
    return ranges::find(required_constant_cpu_inputs_, index)
        != required_constant_cpu_inputs_.end() ? E_FAIL : E_INVALIDARG;
} catch (const Ort::Exception&) {
    return E_FAIL;
}

HRESULT AbiSafeKernelCreationContext::TryGetConstantInputTensor(uint32_t index, IMLOperatorTensor** tensor) const noexcept
try {
    if (tensor == nullptr) {
        return E_POINTER;
    }
    *tensor = nullptr;
    if (kernel_info_ != nullptr) {
        int is_constant{};
        const auto value{kernel_info_.GetTensorConstantInput(index, &is_constant)};
        if (is_constant == 0 || value == nullptr) {
            return ranges::find(required_constant_cpu_inputs_, index)
                != required_constant_cpu_inputs_.end() ? E_UNEXPECTED : S_OK;
        }
        auto safe_tensor{Microsoft::WRL::Make<AbiSafeTensor>(value)};
        *tensor = safe_tensor.Detach();
    }
    return S_OK;
} catch (const Ort::Exception&) {
    return E_FAIL;
}

uint32_t AbiSafeKernelCreationContext::GetUtf8NameBufferSizeInBytes() const noexcept
try {
    if (kernel_info_ != nullptr) {
        size_t size{};
        if (const Ort::Status status{Ort::GetApi().KernelInfo_GetNodeName(
            kernel_info_, nullptr, &size)}; status.IsOK()) {
            return static_cast<uint32_t>(size);
        }
    }
    return 1;
} catch (const Ort::Exception&) {
    return 1;
}

HRESULT AbiSafeKernelCreationContext::GetUtf8Name(uint32_t bufferSizeInBytes, char* outputName) const noexcept
try {
    if (outputName == nullptr) {
        return E_POINTER;
    }
    *outputName = '\0';
    if (bufferSizeInBytes == 0) {
        return E_INVALIDARG;
    }
    if (kernel_info_ != nullptr) {
        kernel_info_.GetNodeName().copy(outputName, bufferSizeInBytes);
    }
    return S_OK;
} catch (const Ort::Exception&) {
    return E_FAIL;
}

uint32_t AbiSafeKernelCreationContext::GetWideNameBufferSizeInBytes() const noexcept
try {
    if (kernel_info_ == nullptr) {
        return sizeof(wchar_t);
    }
    const auto name{kernel_info_.GetNodeName()};
    if (name.empty()) {
        return sizeof(wchar_t);
    }
    const auto size{MultiByteToWideChar(CP_UTF8, 0, name.data(), static_cast<int>(name.length()), nullptr, 0)};
    return static_cast<uint32_t>(size <= 0 ? sizeof(wchar_t) : (size + 1) * sizeof(wchar_t));
} catch (const Ort::Exception&) {
    return E_FAIL;
}

HRESULT AbiSafeKernelCreationContext::GetWideName(uint32_t bufferSizeInBytes, wchar_t* outputName) const noexcept
try {
    if (outputName == nullptr || kernel_info_ == nullptr) {
        return E_POINTER;
    }
    if (bufferSizeInBytes < sizeof(wchar_t)) {
        return E_INVALIDARG;
    }
    *outputName = L'\0';
    if (const auto name{kernel_info_.GetNodeName()}; !name.empty()) {
        const auto size{name.length() / sizeof(wchar_t)};
        auto copied{MultiByteToWideChar(CP_UTF8, 0, name.data(),
            static_cast<int>(name.length()), outputName, static_cast<int>(size))};
        if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
            copied = std::max(0, copied - 1);
        }
        outputName[copied] = L'\0';
        return S_OK;
    }
    return E_FAIL;
} catch (const Ort::Exception&) {
    return E_FAIL;
}

HRESULT AbiSafeKernelCreationContext::GetExecutionProvider(IUnknown** executionProvider) const noexcept {
    if (executionProvider == nullptr) {
        return E_POINTER;
    }
    *executionProvider = nullptr;
    if (execution_provider_ == nullptr) {
        return S_OK;
    }
    return const_cast<PluginDmlExecutionProviderImpl*>(execution_provider_)->QueryInterface(
        __uuidof(IWinmlExecutionProvider),
        reinterpret_cast<void**>(executionProvider)
    );
}

// ============================================================================
// Kernel Creation and Execution - Pure C API, NO UNSAFE CASTS!
// ============================================================================

// Returns true if all dimensions of an OrtValue's shape are concrete (>= 0).
static bool TensorValueHasConcreteDims(const OrtApi* api, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* tsi = nullptr;
    OrtStatus* s = api->GetTensorTypeAndShape(const_cast<OrtValue*>(value), &tsi);
    if (s || !tsi) { if (s) api->ReleaseStatus(s); return false; }

    size_t dim_count = 0;
    s = api->GetDimensionsCount(tsi, &dim_count);
    if (s) { api->ReleaseTensorTypeAndShapeInfo(tsi); api->ReleaseStatus(s); return false; }

    std::vector<int64_t> dims(dim_count);
    if (dim_count > 0) {
        s = api->GetDimensions(tsi, dims.data(), dim_count);
        if (s) { api->ReleaseTensorTypeAndShapeInfo(tsi); api->ReleaseStatus(s); return false; }
        for (int64_t d : dims) {
            if (d < 0) { api->ReleaseTensorTypeAndShapeInfo(tsi); return false; }
        }
    }

    api->ReleaseTensorTypeAndShapeInfo(tsi);
    return true;
}

// Returns true when all input tensor shapes are known concretely at session-init time.
// For each input, first checks the proto-level shape via KernelInfo_GetInputTypeInfo.
// If that reports a dynamic/symbolic dim, falls back to KernelInfoGetConstantInput_tensor:
// constant/initializer-backed inputs have a materialized OrtValue whose shape is always
// concrete, even when the proto shape is symbolic (e.g. batch=?). Only returns false when
// an input is genuinely non-constant with an unresolved shape — meaning it must be deferred
// to compute time. This matches the effective behavior of the unsafe path's
// InputTensorShapesDefined(), which sees ORT's internally resolved shapes via OpKernelInfo.
static bool AllInputShapesConcreteInProto(const OrtApi* api, const OrtKernelInfo* kernel_info) {
    size_t input_count = 0;
    OrtStatus* s = api->KernelInfo_GetInputCount(kernel_info, &input_count);
    if (s) { api->ReleaseStatus(s); return false; }

    for (size_t i = 0; i < input_count; ++i) {
        OrtTypeInfo* type_info = nullptr;
        s = api->KernelInfo_GetInputTypeInfo(kernel_info, i, &type_info);
        if (s) { api->ReleaseStatus(s); return false; }

        bool proto_shape_concrete = true;

        if (type_info) {
            const OrtTensorTypeAndShapeInfo* tsi = nullptr;
            s = api->CastTypeInfoToTensorInfo(type_info, &tsi);
            if (s) { api->ReleaseTypeInfo(type_info); api->ReleaseStatus(s); return false; }

            if (tsi) {
                size_t dim_count = 0;
                s = api->GetDimensionsCount(tsi, &dim_count);
                if (s) { api->ReleaseTypeInfo(type_info); api->ReleaseStatus(s); return false; }

                if (dim_count > 0) {
                    std::vector<int64_t> dims(dim_count);
                    s = api->GetDimensions(tsi, dims.data(), dim_count);
                    if (s) { api->ReleaseTypeInfo(type_info); api->ReleaseStatus(s); return false; }
                    for (int64_t d : dims) {
                        if (d < 0) { proto_shape_concrete = false; break; }
                    }
                }

                if (proto_shape_concrete) {
                    // Also reject named symbolic dims (e.g. "batch_size")
                    std::vector<const char*> sym_dims(dim_count, nullptr);
                    s = api->GetSymbolicDimensions(tsi, sym_dims.data(), dim_count);
                    if (!s) {
                        for (size_t d = 0; d < dim_count; ++d) {
                            if (sym_dims[d] && sym_dims[d][0] != '\0') {
                                proto_shape_concrete = false;
                                break;
                            }
                        }
                    } else {
                        api->ReleaseStatus(s);
                    }
                }
            }

            api->ReleaseTypeInfo(type_info);
        }

        if (proto_shape_concrete) continue;

        // Proto shape is dynamic/symbolic — check if input is a constant initializer.
        // If so, read the concrete shape from the materialized OrtValue instead.
        int is_constant = 0;
        const OrtValue* constant_value = nullptr;
        s = api->KernelInfoGetConstantInput_tensor(kernel_info, i, &is_constant, &constant_value);
        if (s) { api->ReleaseStatus(s); return false; }

        bool concrete_via_constant = is_constant && constant_value && TensorValueHasConcreteDims(api, constant_value);

        if (concrete_via_constant) {
            continue;  // Shape is concrete via the materialized constant tensor
        }

        return false;  // Genuinely dynamic — must defer to compute time
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

        auto tensor_attr_cache{FetchAllTensorAttributes(kernel_info, state->tensor_attribute_names)};

        // Create ABI-safe kernel creation context with pre-fetched constants and tensor attrs.
        // NOTE: constant_tensors is moved here — all checks on it must happen BEFORE this line.
        const auto creation_context = Microsoft::WRL::Make<AbiSafeKernelCreationContext>(
            kernel_info,
            state->default_attributes,
            state->required_constant_cpu_inputs,
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
                    const auto shape_inference_context{Microsoft::WRL::Make<AbiSafeShapeInferenceContext>(
                        nullptr, state->default_attributes, kernel_info)};

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
                                    static_cast<unsigned>(shape_hr), state->operator_name ? state->operator_name : "unknown").c_str());
                        }
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
                        static_cast<unsigned>(hr), state->operator_name ? state->operator_name : "unknown").c_str());
            }
        }

        // Allocate memory for OrtKernelImpl + DmlAbiKernel
        constexpr size_t total_size{sizeof(OrtKernelImpl) + sizeof(DmlAbiKernel)};
        void* memory{::operator new(total_size, std::nothrow)};
        if (memory == nullptr) {
            return state->ort_api->CreateStatus(ORT_FAIL, "Failed to allocate kernel memory");
        }

        // Construct OrtKernelImpl
        const auto impl = new (memory) OrtKernelImpl();
        impl->ort_version_supported = ORT_API_VERSION;
        impl->flags = 0;
        impl->Compute = DmlAbiKernel_Compute;
        impl->Release = DmlAbiKernel_Release;
        impl->PrePackWeight = nullptr;
        impl->SetSharedPrePackedWeight = nullptr;

        // Construct DmlAbiKernel immediately after OrtKernelImpl
        auto abi_kernel = new (static_cast<char*>(memory) + sizeof(OrtKernelImpl)) DmlAbiKernel();
        abi_kernel->ml_operator_kernel = std::move(ml_kernel);
        abi_kernel->ort_api = state->ort_api;
        abi_kernel->dml_execution_provider = state->dml_execution_provider;
        abi_kernel->is_internal_operator = state->is_internal_operator;
        abi_kernel->ep_name = state->ep_name;
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

        // Pre-upload CPU-resident constant inputs to a D3D12_HEAP_TYPE_DEFAULT GPU buffer.
        // After this point, AbiSafeKernelContext::GetInputTensor() will return an
        // AbiSafeD3D12Tensor backed by constant_gpu_resources[inputIndex] instead of
        // re-fetching and re-uploading from the ORT execution context on every Compute() call.
        if (!needs_lazy_init && !constant_tensors.empty() && state->dml_execution_provider) {
            UploadConstantTensorsToGpu(
                constant_tensors,
                state->dml_execution_provider,
                abi_kernel->constant_gpu_resources);

            uint32_t cached_count = 0;
            for (auto& entry : abi_kernel->constant_gpu_resources) {
                if (entry.resource) {
                    ++cached_count;
                    // QueueReference so the GPU has finished with in-flight upload work before
                    // the resource could be freed (e.g. if the kernel is destroyed early).
                    if (abi_kernel->winml_provider) {
                        abi_kernel->winml_provider->QueueReference(entry.resource.Get());
                    }
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

            auto tensor_attr_cache_lazy{FetchAllTensorAttributes(kernel->kernel_info, kernel->tensor_attribute_names)};

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
                    lazy_constant_tensors[input_index] = Microsoft::WRL::Make<AbiSafeTensor>(constant_value);
                } else {
                    if (get_status) kernel->ort_api->ReleaseStatus(get_status);
                    const OrtValue* runtime_value = nullptr;
                    OrtStatus* ctx_status = kernel->ort_api->KernelContext_GetInput(
                        context, input_index, &runtime_value);
                    if (ctx_status == nullptr && runtime_value != nullptr) {
                        lazy_constant_tensors[input_index] = Microsoft::WRL::Make<AbiSafeTensor>(runtime_value);
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

            // Pre-upload constant inputs to GPU while the tensors are still available
            // (before std::move(lazy_constant_tensors) below). Future Compute() calls then hit
            // the cached D3D12Resource in GetInputTensor() instead of re-uploading.
            if (kernel->dml_execution_provider && !lazy_constant_tensors.empty()) {
                UploadConstantTensorsToGpu(
                    lazy_constant_tensors,
                    kernel->dml_execution_provider,
                    kernel->constant_gpu_resources);

                uint32_t cached_count = 0;
                for (auto& entry : kernel->constant_gpu_resources) {
                    if (entry.resource) {
                        ++cached_count;
                        if (kernel->winml_provider) {
                            kernel->winml_provider->QueueReference(entry.resource.Get());
                        }
                    }
                }

            }

            // Create context with runtime shapes from actual input tensors.
            // NOTE: lazy_constant_tensors is moved here — snapshot must happen before this line.
            auto creation_context = Microsoft::WRL::Make<AbiSafeKernelCreationContext>(
                kernel->kernel_info,
                kernel->default_attributes,
                kernel->required_constant_cpu_inputs,
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
                    context, kernel->default_attributes, kernel->kernel_info);

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

            if (FAILED(hr) || ml_kernel == nullptr) {
                return kernel->ort_api->CreateStatus(ORT_FAIL,
                    fmt::format("Lazy kernel creation failed with HR=0x{:08X} for {}",
                        static_cast<long>(hr), kernel->operator_name).c_str());
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
                    tmp_constant_tensors[input_index] = Microsoft::WRL::Make<AbiSafeTensor>(constant_value);
                } else {
                    if (get_status) kernel->ort_api->ReleaseStatus(get_status);
                    // Stage 2: KernelContext_GetInput (dynamically computed constants)
                    const OrtValue* runtime_value = nullptr;
                    OrtStatus* ctx_status = kernel->ort_api->KernelContext_GetInput(
                        context, input_index, &runtime_value);
                    if (ctx_status == nullptr && runtime_value != nullptr) {
                        tmp_constant_tensors[input_index] = Microsoft::WRL::Make<AbiSafeTensor>(runtime_value);
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
                auto tensor_attr_cache_tmp{FetchAllTensorAttributes(
                    kernel->kernel_info, kernel->tensor_attribute_names)};

                auto tmp_creation_context = Microsoft::WRL::Make<AbiSafeKernelCreationContext>(
                    kernel->kernel_info,
                    kernel->default_attributes,
                    kernel->required_constant_cpu_inputs,
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
                        context, kernel->default_attributes, kernel->kernel_info);
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
                    kernel->dml_execution_provider,
                    kernel->is_internal_operator,
                    kernel->ep_name,
                    kernel->default_attributes,
                    &tmp_output_shapes,
                    kernel->shape_inferrer.Get(),
                    kernel->required_constant_cpu_inputs,
                    kernel->kernel_info,
                    kernel->constant_gpu_resources.empty() ? nullptr : &kernel->constant_gpu_resources
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
            kernel->dml_execution_provider,
            kernel->is_internal_operator,
            kernel->ep_name,
            kernel->default_attributes,
            &kernel->inferred_output_shapes,
            kernel->shape_inferrer.Get(),
            kernel->required_constant_cpu_inputs,
            kernel->kernel_info,
            kernel->constant_gpu_resources.empty() ? nullptr : &kernel->constant_gpu_resources
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
