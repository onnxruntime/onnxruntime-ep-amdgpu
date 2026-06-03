// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <string>
#include <DirectML.h>

#include "dml_ep.h"
#include "dml_factory.h"

namespace dml_ep {

namespace {

enum class DeviceFilter : uint32_t {
    Any [[maybe_unused]] = 0xffffffff,
    Gpu = 1 << 0,
    Npu = 1 << 1,
};

DeviceFilter operator & (DeviceFilter a, DeviceFilter b) {
    return static_cast<DeviceFilter>(static_cast<unsigned int>(a) & static_cast<unsigned int>(b));
}

enum class PerformancePreference {
    Default [[maybe_unused]] = 0,
    HighPerformance [[maybe_unused]] = 1,
    MinimumPower = 2
};


bool IsHardwareAdapter(IDXCoreAdapter* adapter)
{
    bool is_hardware = false;
    THROW_IF_FAILED(adapter->GetProperty(DXCoreAdapterProperty::IsHardware, &is_hardware));
    return is_hardware;
}

bool IsGPU(IDXCoreAdapter* compute_adapter)
{
    // Only considering hardware adapters
    if (!IsHardwareAdapter(compute_adapter))
    {
        return false;
    }
    return compute_adapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS);
}

bool IsNPU(IDXCoreAdapter* compute_adapter)
{
    // Only considering hardware adapters
    if (!IsHardwareAdapter(compute_adapter))
    {
        return false;
    }
    return !(compute_adapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS));
}

enum class DeviceType { GPU, NPU, BadDevice };

DeviceType FilterAdapterTypeQuery(IDXCoreAdapter* adapter, DeviceFilter filter)
{
    auto allow_gpus = (filter & DeviceFilter::Gpu) == DeviceFilter::Gpu;
    if (IsGPU(adapter) && allow_gpus) {
        return DeviceType::GPU;
    }

    auto allow_npus = (filter & DeviceFilter::Npu) == DeviceFilter::Npu;
    if (IsNPU(adapter) && allow_npus) {
        return DeviceType::NPU;
    }

    return DeviceType::BadDevice;
}

// Struct for holding each adapter
struct AdapterInfo
{
    Microsoft::WRL::ComPtr<IDXCoreAdapter> Adapter;
    DeviceType Type; // GPU or NPU
};

Microsoft::WRL::ComPtr<IDXCoreAdapterList> EnumerateDXCoreAdapters(IDXCoreAdapterFactory* adapter_factory) {
    Microsoft::WRL::ComPtr<IDXCoreAdapterList> adapter_list;

    // TODO: use_dxcore_workload_enumeration should be determined by QI
    // When DXCore APIs are available QI for relevant enumeration interfaces
    constexpr bool use_dxcore_workload_enumeration = false;
    if (!use_dxcore_workload_enumeration)
    {
        THROW_IF_FAILED(adapter_factory->CreateAdapterList(1, &DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML, adapter_list.GetAddressOf()));

        if (adapter_list->GetAdapterCount() == 0)
        {
            THROW_IF_FAILED(adapter_factory->CreateAdapterList(1, &DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE, adapter_list.GetAddressOf()));
        }
    }

    return adapter_list;
}

void SortDXCoreAdaptersByPreference(IDXCoreAdapterList* adapter_list, PerformancePreference preference)
{
    if (adapter_list->GetAdapterCount() <= 1)
    {
        return;
    }

    // DML prefers the HighPerformance adapter by default
    std::array<DXCoreAdapterPreference, 1> adapter_list_preferences = {DXCoreAdapterPreference::HighPerformance};

    // If callers specify minimum power change the DXCore sort policy
    // NOTE DXCoreAdapterPrefernce does not apply to mixed adapter lists - only to GPU lists
    if (preference == PerformancePreference::MinimumPower)
    {
        adapter_list_preferences[0] = DXCoreAdapterPreference::MinimumPower;
    }

    THROW_IF_FAILED(adapter_list->Sort(static_cast<uint32_t>(adapter_list_preferences.size()), adapter_list_preferences.data()));
}

std::vector<AdapterInfo> FilterDXCoreAdapters(IDXCoreAdapterList* adapter_list, DeviceFilter filter)
{
    auto adapter_infos = std::vector<AdapterInfo>();
    const uint32_t count = adapter_list->GetAdapterCount();
    for (uint32_t i = 0; i < count; ++i) {
        Microsoft::WRL::ComPtr<IDXCoreAdapter> candidate_adapter;
        THROW_IF_FAILED(adapter_list->GetAdapter(i, candidate_adapter.GetAddressOf()));

        // Add the adapters that are valid based on the device filter (GPU, NPU, or Both)
        auto adapter_type = FilterAdapterTypeQuery(candidate_adapter.Get(), filter);
        if (adapter_type != DeviceType::BadDevice) {
            adapter_infos.push_back(AdapterInfo{candidate_adapter, adapter_type});
        }
    }

    return adapter_infos;
}

void SortHeterogenousDXCoreAdapterList(std::vector<AdapterInfo>& adapter_infos, DeviceFilter filter,
                                              PerformancePreference preference)
{
    if (adapter_infos.size() <= 1)
    {
        return;
    }

    // When considering both GPUs and NPUs sort them by performance preference
    // of Default (Gpus first), HighPerformance (GPUs first), or LowPower (NPUs first)
    auto keep_npus = (filter & DeviceFilter::Npu) == DeviceFilter::Npu;
    auto only_npus = filter == DeviceFilter::Npu;
    if (!keep_npus || only_npus) {
        return;
    }

    struct SortingPolicy {
        // default is false because GPUs are considered higher priority in
        // a mixed adapter environment
        bool npus_first_ = false;

        SortingPolicy(bool npus_first = false) : npus_first_(npus_first) {}

        bool operator()(const AdapterInfo& a, const AdapterInfo& b) {
            return npus_first_ ? a.Type < b.Type : a.Type > b.Type;
        }
    };

    auto npus_first = (preference == PerformancePreference::MinimumPower);
    auto policy = SortingPolicy(npus_first);
    std::sort(adapter_infos.begin(), adapter_infos.end(), policy);
}

D3D12_COMMAND_LIST_TYPE CalculateCommandListType(ID3D12Device* d3d12_device)
{
    D3D12_FEATURE_DATA_FEATURE_LEVELS feature_levels = {};

    D3D_FEATURE_LEVEL feature_levels_list[] = {
        D3D_FEATURE_LEVEL_1_0_GENERIC,
        D3D_FEATURE_LEVEL_1_0_CORE,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_12_0,
        D3D_FEATURE_LEVEL_12_1};

    feature_levels.NumFeatureLevels = ARRAYSIZE(feature_levels_list);
    feature_levels.pFeatureLevelsRequested = feature_levels_list;
    THROW_IF_FAILED(d3d12_device->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &feature_levels, sizeof(feature_levels)));

    auto use_compute_command_list = (feature_levels.MaxSupportedFeatureLevel <= D3D_FEATURE_LEVEL_1_0_CORE);

    if (use_compute_command_list) {
        return D3D12_COMMAND_LIST_TYPE_COMPUTE;
    }

    return D3D12_COMMAND_LIST_TYPE_DIRECT;
}

}  // namespace

ProviderFactory::ProviderFactory(const ApiPtrs& api_ptrs, std::string_view ep_name, const Ort::Logger& default_logger)
    : OrtEpFactory{ORT_API_VERSION},
      ApiPtrs{api_ptrs},
      default_logger_{default_logger},
      ep_name_{ep_name},
      bucketized_buffer_memory_info_{"directML_ep_gpu", OrtMemoryInfoDeviceType_GPU, amd::VendorId,
          0, OrtDeviceMemoryType_DEFAULT, 0, OrtDeviceAllocator},
      cpu_input_allocator_{"directML_ep_cpu", OrtMemoryInfoDeviceType_CPU, amd::VendorId, 0,
          OrtDeviceMemoryType_DEFAULT, 0, OrtDeviceAllocator},
      dml_data_transfer_implementation{std::make_unique<DMLDataTransfer>(api_ptrs)}
{
    GetName = GetNameImpl;
    GetVendor = GetVendorImpl;
    GetVendorId = GetVendorIdImpl;
    GetVersion = GetVersionImpl;

    OrtEpFactory::GetSupportedDevices = [](OrtEpFactory* this_,
            const OrtHardwareDevice* const* devices, size_t num_devices, OrtEpDevice** ep_devices,
            size_t max_ep_devices, size_t* num_ep_devices) noexcept {
        API_CALL_S(ProviderFactory, this_, GetSupportedDevices, {devices, devices + num_devices},
            {ep_devices, ep_devices + max_ep_devices}, *num_ep_devices);
    };

    CreateEp = CreateEpImpl;
    CreateAllocator = CreateAllocatorImpl;
    ReleaseEp = ReleaseEpImpl;

    ReleaseAllocator = ReleaseAllocatorImpl;

    CreateDataTransfer = CreateDataTransferImpl;

    IsStreamAware = IsStreamAwareImpl;
    CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
    GetHardwareDeviceIncompatibilityDetails = GetHardwareDeviceIncompatibilityDetailsImpl;

    CreateExternalResourceImporterForDevice = CreateExternalResourceImporterForDeviceImpl;

    GetNumCustomOpDomains = GetNumCustomOpDomainsImpl;
    GetCustomOpDomains = GetCustomOpDomainsImpl;
}

const char* ORT_API_CALL ProviderFactory::GetNameImpl(const OrtEpFactory* this_ptr) noexcept {
    const auto* factory = static_cast<const ProviderFactory*>(this_ptr);
    return factory->ep_name_.c_str();
}

const char* ORT_API_CALL ProviderFactory::GetVendorImpl(const OrtEpFactory* /* this_ptr */) noexcept {
    return amd::Vendor;
}

uint32_t ORT_API_CALL ProviderFactory::GetVendorIdImpl(const OrtEpFactory* /* this_ptr */) noexcept {
    return amd::VendorId;
}

const char* ORT_API_CALL ProviderFactory::GetVersionImpl(const OrtEpFactory* this_ptr) noexcept {
    const auto* factory = static_cast<const ProviderFactory*>(this_ptr);
    return factory->ep_version_.c_str();
}

Ort::Status ProviderFactory::GetSupportedDevices(const std::vector<Ort::ConstHardwareDevice>& devices,
        const gsl::span<OrtEpDevice*>& ep_devices, size_t& num_ep_devices) noexcept
{
    num_ep_devices = 0;
    for (const auto& device : devices) {
        if (num_ep_devices >= ep_devices.size()) {
            break;
        }
        if (device.VendorId() == amd::VendorId && device.Type() == OrtHardwareDeviceType_GPU) {
            OrtEpDevice* ep_device{};
            RETURN_IF_ERROR(ep_api.CreateEpDevice(this, device, nullptr, nullptr, &ep_device));

            // currently forced to make allocator info here since the ep device creation is
            // separate from the EP creation, but the allocator info is needed to create
            // the EP. In the future we may want to move allocator info creation to EP creation.
            ep_api.EpDevice_AddAllocatorInfo(ep_device, cpu_input_allocator_);
            ep_api.EpDevice_AddAllocatorInfo(ep_device, bucketized_buffer_memory_info_);

            ep_devices[num_ep_devices++] = ep_device;
        }
    }
    return STATUS_OK;
}

OrtStatus* ORT_API_CALL ProviderFactory::CreateEpImpl(OrtEpFactory* this_ptr,
                                                      const OrtHardwareDevice* const* devices,
                                                      const OrtKeyValuePairs* const* ep_metadata,
                                                      size_t num_devices,
                                                      const OrtSessionOptions* session_options,
                                                      const OrtLogger* logger,
                                                      OrtEp** ep) noexcept 
{
    auto* factory = static_cast<ProviderFactory*>(this_ptr);

    factory->ort_api.DisableMemPattern(const_cast<OrtSessionOptions*>(session_options));
    factory->ort_api.SetSessionExecutionMode(const_cast<OrtSessionOptions*>(session_options), ExecutionMode::ORT_SEQUENTIAL);

    *ep = nullptr;
    if (num_devices > 1) {
        return factory->ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
            "DirectML EP supports selection for a single or none devices");
    }

    // Create DML device and execution context for the selected adapter
    Microsoft::WRL::ComPtr<ID3D12Device> d3d12_device = factory->CreateD3d12Device();
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> cmd_queue = factory->CreateCommandQueue(d3d12_device);
    Microsoft::WRL::ComPtr<IDMLDevice> dml_device = factory->CreateDMLDevice(d3d12_device);

    auto context = wil::MakeOrThrow<ExecutionContext>(
        d3d12_device.Get(), 
        dml_device.Get(),
        cmd_queue.Get(),
        false,
        false);

    factory->m_ep = std::make_unique<ExecutionProviderPlugin>(
        static_cast<const ApiPtrs&>(*factory), // use the ApiPtrs base already initialized
        factory->ep_name_,
        d3d12_device.Get(),
        dml_device.Get(),
        context);

    // keep a non-owning raw pointer to the EP for use in the data transfer implementation
    factory->m_ep_raw = factory->m_ep.get();

    // Attach the new EP to the factory-owned data transfer so copies can be serviced.
    if (factory->dml_data_transfer_implementation) {
        factory->dml_data_transfer_implementation->AttachExecutionProvider(
            factory->m_ep_raw->GetInternalExecutionProvider());
        factory->dml_data_transfer_implementation->AttachFactoryEpRef(&factory->m_ep_raw);
    }

    *ep = factory->m_ep.release();
    return nullptr;
}

void ORT_API_CALL ProviderFactory::ReleaseEpImpl(OrtEpFactory* this_ptr, OrtEp* ep) noexcept
{
    auto& factory = *static_cast<ProviderFactory*>(this_ptr);
    if (ep) {
        factory.m_ep_raw = nullptr; // invalidate observer before deletion
        // Clear the factory-owned data transfer's EP reference so the EP's ref count
        // can reach zero and GPU resources are freed when the session ends.
        if (factory.dml_data_transfer_implementation) {
            factory.dml_data_transfer_implementation->AttachExecutionProvider(nullptr);
        }
        ExecutionProviderPlugin* provider = static_cast<ExecutionProviderPlugin*>(ep);
        delete provider;
    }
}

OrtStatus* ORT_API_CALL ProviderFactory::CreateAllocatorImpl(OrtEpFactory* this_ptr, const OrtMemoryInfo* memory_info,
                                                              const OrtKeyValuePairs* allocator_options,
                                                              OrtAllocator** allocator) noexcept {
    auto& factory = *static_cast<ProviderFactory*>(this_ptr);

    factory.IsCpuAllocator(memory_info);
    factory.IsGpuAllocator(memory_info);

    // send passthrough allocator. dml ep allocators defined in dml_ep
    *allocator = std::make_unique<CpuAllocator>(memory_info).release();
    return nullptr;
}

void ORT_API_CALL ProviderFactory::ReleaseAllocatorImpl(OrtEpFactory* /*this*/, OrtAllocator* allocator) noexcept {
    // no-op. The allocators are shared across sessions.
}

OrtStatus* ORT_API_CALL ProviderFactory::CreateDataTransferImpl(OrtEpFactory* this_ptr, OrtDataTransferImpl** data_transfer) noexcept
{
    auto* factory = static_cast<ProviderFactory*>(this_ptr);
    // Return the factory-owned data transfer. The factory manages its lifetime via unique_ptr;
    // ORT's Release() call is intentionally ignored (no-op in ReleaseImpl).
    // The EP reference is attached in CreateEpImpl and cleared in ReleaseEpImpl.
    *data_transfer = factory->dml_data_transfer_implementation.get();
    return nullptr;
}

bool ORT_API_CALL ProviderFactory::IsStreamAwareImpl(const OrtEpFactory* this_ptr) noexcept {
    return false;
}

OrtStatus* ORT_API_CALL ProviderFactory::CreateSyncStreamForDeviceImpl(
    OrtEpFactory* this_ptr,
    const OrtMemoryDevice* memory_device,
    const OrtKeyValuePairs* stream_options,
    OrtSyncStreamImpl** stream) noexcept
{
    auto& factory = *static_cast<ProviderFactory*>(this_ptr);
    *stream = nullptr;

    return nullptr;
}

OrtStatus* ORT_API_CALL ProviderFactory::CreateExternalResourceImporterForDeviceImpl(
    OrtEpFactory* this_ptr,
    const OrtEpDevice* ep_device,
    OrtExternalResourceImporterImpl** out_importer) noexcept
{
    return nullptr;
}

OrtStatus* ORT_API_CALL ProviderFactory::GetHardwareDeviceIncompatibilityDetailsImpl(
    OrtEpFactory* this_ptr,
    const OrtHardwareDevice* hw,
    OrtDeviceEpIncompatibilityDetails* details) noexcept
{
    return nullptr;
}

OrtStatus* ORT_API_CALL ProviderFactory::GetNumCustomOpDomainsImpl(OrtEpFactory* this_ptr,
                                                                 _Out_ size_t* num_domains) noexcept {
    return nullptr;
}

OrtStatus* ORT_API_CALL ProviderFactory::GetCustomOpDomainsImpl(
    OrtEpFactory* this_ptr,
    _Outptr_result_maybenull_ OrtCustomOpDomain** domains,
    _Out_ size_t num_domains) noexcept
{
    return nullptr;
}

typedef HRESULT(WINAPI* PFN_DXCoreCreateAdapterFactory)(REFIID riid, void** ppvFactory);

std::vector<Microsoft::WRL::ComPtr<IDXCoreAdapter>> ProviderFactory::GetAdapters()
{
    // For now assume preference is always prefer max performance and filter for gpu only
    PerformancePreference preference = PerformancePreference::HighPerformance;
    DeviceFilter filter = DeviceFilter::Gpu;

    // Load dxcore.dll. We do this manually so there's not a hard dependency on dxcore which is newer.
    wil::unique_hmodule dxcore_lib{LoadLibraryExW(L"dxcore.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32)};
    if (!dxcore_lib) {
        THROW("Failed to load dxcore.dll. Expected on older Windows version that do not support dxcore.");
    }

    auto pfnDXCoreCreateAdapterFactory = reinterpret_cast<PFN_DXCoreCreateAdapterFactory>(
        GetProcAddress(dxcore_lib.get(), "DXCoreCreateAdapterFactory"));

    if (!pfnDXCoreCreateAdapterFactory) {
        // this isn't expected to fail so ERROR not WARNING
        THROW("Failed to get DXCoreCreateAdapterFactory function address.");
    }

    // Create DXCore Adapter Factory
    Microsoft::WRL::ComPtr<IDXCoreAdapterFactory> adapter_factory;
    if (FAILED(pfnDXCoreCreateAdapterFactory(IID_PPV_ARGS(&adapter_factory)))) {
        THROW("DXCore is not available on this platform. This is expected on older versions of Windows.");
    }

    // Get all DML compatible DXCore adapters
    Microsoft::WRL::ComPtr<IDXCoreAdapterList> adapter_list;
    adapter_list = EnumerateDXCoreAdapters(adapter_factory.Get());

    if (adapter_list->GetAdapterCount() == 0) {
        THROW("No GPUs or NPUs detected.");
    }

    // Sort the adapter list to honor DXCore hardware ordering
    SortDXCoreAdaptersByPreference(adapter_list.Get(), preference);

    // TODO: use_dxcore_workload_enumeration should be determined by QI
    // When DXCore APIs are available QI for relevant enumeration interfaces
    constexpr bool use_dxcore_workload_enumeration = false;

    std::vector<AdapterInfo> adapter_infos;
    if (!use_dxcore_workload_enumeration) {
        // Filter all DXCore adapters to hardware type specified by the device filter
        adapter_infos = FilterDXCoreAdapters(adapter_list.Get(), filter);
        if (adapter_infos.size() == 0) {
            THROW("No devices detected that match the filter criteria.");
        }
    }

    // DXCore Sort ignores NPUs. When both GPUs and NPUs are present, manually sort them.
    SortHeterogenousDXCoreAdapterList(adapter_infos, filter, preference);

    // Extract just the adapters
    auto adapters = std::vector<Microsoft::WRL::ComPtr<IDXCoreAdapter>>(adapter_infos.size());
    std::transform(adapter_infos.begin(), adapter_infos.end(), adapters.begin(), [](auto& a) { return a.Adapter; });

    return adapters;
}

void ProviderFactory::CreateD3DDeviceFromAdapter(IDXCoreAdapter* adapter, Microsoft::WRL::ComPtr<ID3D12Device>& device)
{
    auto feature_level = D3D_FEATURE_LEVEL_11_0;
    if (IsNPU(adapter)) {
        feature_level = D3D_FEATURE_LEVEL_1_0_GENERIC;
    }

    // Create D3D12 Device from DXCore Adapter
    if (feature_level == D3D_FEATURE_LEVEL_1_0_GENERIC) {
        // Attempt to create a D3D_FEATURE_LEVEL_1_0_CORE device first, in case the device supports this
        // feature level and the D3D runtime does not support D3D_FEATURE_LEVEL_1_0_GENERIC
        HRESULT hrUnused = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_1_0_CORE, IID_PPV_ARGS(device.ReleaseAndGetAddressOf()));
    }

    if (!device) {
        THROW_IF_FAILED(D3D12CreateDevice(adapter, feature_level, IID_PPV_ARGS(device.ReleaseAndGetAddressOf())));
    }
}

void ProviderFactory::CreateDMLAndD3DResources()
{
    std::vector<Microsoft::WRL::ComPtr<IDXCoreAdapter>> rankedAdapters = GetAdapters();

    // choose first adapter
    auto adapter = rankedAdapters[0];

    CreateD3DDeviceFromAdapter(adapter.Get(), d3d12_device);

    D3D12_COMMAND_QUEUE_DESC cmd_queue_desc = {};
    cmd_queue_desc.Type = CalculateCommandListType(d3d12_device.Get());
    cmd_queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT;

    THROW_IF_FAILED(d3d12_device->CreateCommandQueue(&cmd_queue_desc, IID_PPV_ARGS(cmd_queue.ReleaseAndGetAddressOf())));

    dml_device = CreateDMLDevice(d3d12_device);
}

Microsoft::WRL::ComPtr<ID3D12Device> ProviderFactory::CreateD3d12Device()
{
    std::vector<Microsoft::WRL::ComPtr<IDXCoreAdapter>> rankedAdapters = GetAdapters();

    // choose first adapter
    auto adapter = rankedAdapters[0];
    Microsoft::WRL::ComPtr<ID3D12Device> d3d12_device;
    CreateD3DDeviceFromAdapter(adapter.Get(), d3d12_device);
    return d3d12_device;
}

Microsoft::WRL::ComPtr<ID3D12CommandQueue> ProviderFactory::CreateCommandQueue(const Microsoft::WRL::ComPtr<ID3D12Device>& device)
{
    D3D12_COMMAND_QUEUE_DESC cmd_queue_desc = {};
    cmd_queue_desc.Type = CalculateCommandListType(device.Get());
    cmd_queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT;

    Microsoft::WRL::ComPtr<ID3D12CommandQueue> cmd_queue;
    THROW_IF_FAILED(device->CreateCommandQueue(&cmd_queue_desc, IID_PPV_ARGS(cmd_queue.ReleaseAndGetAddressOf())));

    return cmd_queue;
}

Microsoft::WRL::ComPtr<IDMLDevice> ProviderFactory::CreateDMLDevice(const Microsoft::WRL::ComPtr<ID3D12Device>& d3d12_device) {
    DML_CREATE_DEVICE_FLAGS flags = DML_CREATE_DEVICE_FLAG_NONE;
    Microsoft::WRL::ComPtr<IDMLDevice> dml_device;
    // In debug builds, enable the DML debug layer if the D3D12 debug layer is also enabled
#if _DEBUG
    Microsoft::WRL::ComPtr<ID3D12DebugDevice> debug_device;
    (void)d3d12_device->QueryInterface(IID_PPV_ARGS(&debug_device)); // ignore failure
    const bool is_d3d12_debug_layer_enabled = (debug_device != nullptr);

    if (is_d3d12_debug_layer_enabled) {
        flags |= DML_CREATE_DEVICE_FLAG_DEBUG;
    }
#endif

    THROW_IF_FAILED(DMLCreateDevice1(d3d12_device.Get(), flags, DML_FEATURE_LEVEL_5_0,
                                     IID_PPV_ARGS(dml_device.ReleaseAndGetAddressOf())));

    return dml_device;
}

bool ProviderFactory::IsCpuAllocator(const OrtMemoryInfo* memory_info) {
    if (!memory_info) {
        return false;
    }

    OrtMemoryInfoDeviceType device_type;
    ort_api.MemoryInfoGetDeviceType(memory_info, &device_type);
    const char* name = nullptr;
    ort_api.MemoryInfoGetName(memory_info, &name);

    return (name != nullptr) && (std::strcmp(name, "directML_ep_cpu") == 0) &&
        (device_type == OrtMemoryInfoDeviceType_CPU);
}

bool ProviderFactory::IsGpuAllocator(const OrtMemoryInfo* memory_info) {
    if (!memory_info) {
        return false;
    }

    OrtMemoryInfoDeviceType device_type;
    ort_api.MemoryInfoGetDeviceType(memory_info, &device_type);
    const char* name = nullptr;
    ort_api.MemoryInfoGetName(memory_info, &name);

    return (name != nullptr) && (std::strcmp(name, "directML_ep_gpu") == 0) &&
        (device_type == OrtMemoryInfoDeviceType_GPU);
}

}  // namespace dml_ep

extern "C" {

OrtStatus* CreateEpFactories(const char* registration_name, const OrtApiBase* ort_api_base,
    const OrtLogger* default_logger, OrtEpFactory** factories, size_t max_factories, size_t* num_factories)
{
    try {
        const OrtApi* ort_api{ort_api_base->GetApi(ORT_API_VERSION)};
        const OrtEpApi* ep_api{ort_api->GetEpApi()};
        const OrtModelEditorApi* model_editor_api{ort_api->GetModelEditorApi()};

        Ort::InitApi(ort_api);

        *num_factories = 1;
        *factories = std::make_unique<dml_ep::ProviderFactory>(ApiPtrs{*ort_api, *ep_api, *model_editor_api},
            registration_name, Ort::Logger{default_logger}).release();

        return nullptr;
    }
    catch (const std::exception& e) {
        RETURN_STATUS(ORT_EP_FAIL, e.what());
    }
}

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
    delete reinterpret_cast<dml_ep::ProviderFactory*>(factory);
    return nullptr;
}

}  // extern "C"
