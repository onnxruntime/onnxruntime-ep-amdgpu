// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "dml_ep.h"
#include "dml_plugin_OperatorRegistration.h"
#include "core/common/inlined_containers.h"
#include "core/common/inlined_containers_fwd.h"
#include "core/framework/fuse_nodes_funcs.h"
#ifdef DML_PERF_PROFILE
#include <cstdio>
#include "dml_abi_kernel.h"
#endif

using namespace Windows::AI::MachineLearning::Adapter;

namespace Dml {

    static void CreateDmlKernelRegistry(
        _In_ const Dml::PluginDmlExecutionProviderImpl* executionProvider,
        _Out_ std::shared_ptr<onnxruntime::KernelRegistry>* registry,
        _Out_ std::shared_ptr<const InternalRegistrationInfoMap>* internalRegInfoMap)
    {
        ComPtr<PluginAbiCustomRegistry> abiRegistry = wil::MakeOrThrow<PluginAbiCustomRegistry>(executionProvider);
        DmlPlugin::RegisterDmlOperators(abiRegistry.Get(), executionProvider);

        assert(abiRegistry->GetRegistries().size() == 1);

        auto customRegistry = *abiRegistry->GetRegistries().begin();
        *registry = customRegistry->GetKernelRegistry();
        *internalRegInfoMap = abiRegistry->GetInternalRegInfoMap();

        DmlPlugin::RegisterCpuOperatorsAsDml(registry->get());
    }
    
ExecutionProviderPlugin::~ExecutionProviderPlugin() {
    m_context->Close();
}

void ExecutionProviderPlugin::Release()
{
    // Match the AddRef() in CreateEpImpl().
    //delete this;
}

ExecutionProviderPlugin::ExecutionProviderPlugin(
    const ApiPtrs& api_ptrs,
    std::string_view name, 
    ID3D12Device* d3d12_device_,
    IDMLDevice* dml_device_,
    ComPtr<PluginDmlExecutionContext> executionContext)
    : OrtEp{ORT_API_VERSION}
    , ApiPtrs{api_ptrs}
    , name_{name}
    , d3d12_device{d3d12_device_}
    , m_dmlDevice{dml_device_}
    , m_context{executionContext}
{
    GetName = GetNameImpl;
    OrtEp::GetCapability = GetCapabilityImpl;
    OrtEp::Compile = CompileImpl;
    ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
    GetPreferredDataLayout = GetPreferredDataLayoutImpl;
    OrtEp::ShouldConvertDataLayoutForOp = ShouldConvertDataLayoutForOpImpl;
    SetDynamicOptions = SetDynamicOptionsImpl;
    OrtEp::OnRunStart = OnRunStartImpl;
    OrtEp::OnRunEnd = OnRunEndImpl;
    CreateAllocator = CreateAllocatorImpl;
    CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;
    OrtEp::GetCompiledModelCompatibilityInfo = GetCompiledModelCompatibilityInfoImpl;
    OrtEp::GetKernelRegistry = GetKernelRegistryImpl;
    IsConcurrentRunSupported = IsConcurrentRunSupportedImpl;

    m_executionProvider = std::make_shared<Dml::PluginDmlExecutionProviderImpl>(
        m_dmlDevice.Get(),
        d3d12_device.Get(),
        m_context.Get(),
        ApiPtrs{api_ptrs},
        m_areMetacommandsEnabled,
        m_graphCaptureEnabled,
        false,
        false);

    m_dataTransfer = std::make_unique<DMLDataTransfer>(ApiPtrs{api_ptrs});
    m_dataTransfer->AttachExecutionProvider(m_executionProvider);

    CreateDmlKernelRegistry(m_executionProvider.get(), &m_kernelRegistry, &m_internalRegInfoMap);
    if(ConvertKernelRegistryToOrtKernelRegistry() != nullptr) {
        throw std::runtime_error("Failed to convert internal kernel registry to OrtKernelRegistry");
    }
}

ONNXTensorElementDataType ExecutionProviderPlugin::GetElementTypeFromMLDataType(onnxruntime::MLDataType ml_type) {
    // Use singleton pointer comparisons — ABI-safe, no protobuf field access.
    using DT = onnxruntime::DataTypeImpl;
    if (ml_type == DT::GetTensorType<float>())                     return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    if (ml_type == DT::GetTensorType<double>())                    return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    if (ml_type == DT::GetTensorType<onnxruntime::MLFloat16>())    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    if (ml_type == DT::GetTensorType<onnxruntime::BFloat16>())     return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    if (ml_type == DT::GetTensorType<int8_t>())                    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    if (ml_type == DT::GetTensorType<int16_t>())                   return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    if (ml_type == DT::GetTensorType<int32_t>())                   return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    if (ml_type == DT::GetTensorType<int64_t>())                   return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    if (ml_type == DT::GetTensorType<uint8_t>())                   return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    if (ml_type == DT::GetTensorType<uint16_t>())                  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    if (ml_type == DT::GetTensorType<uint32_t>())                  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    if (ml_type == DT::GetTensorType<uint64_t>())                  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    if (ml_type == DT::GetTensorType<bool>())                      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    if (ml_type == DT::GetTensorType<std::string>())               return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

OrtStatus* ExecutionProviderPlugin::ConvertKernelRegistryToOrtKernelRegistry()
{
    if (!m_ortKernelRegistry) {
        OrtKernelRegistry* registry = nullptr;
        OrtStatus* st = ep_api.CreateKernelRegistry(&registry);
        if (st != nullptr) {
            return st;
        }

        // Copy kernel entries from the internal registry into the API-created registry
        const auto& kernel_map = m_kernelRegistry->GetKernelCreateMap();
        for (const auto& [key, create_info] : kernel_map) {

            onnxruntime::KernelDef* def = create_info.kernel_def.get();

            // Skip control flow operators (If, Loop, Scan) in ABI-safe path.
            // These are CPU kernels registered under DML EP name only to claim nodes during partitioning.
            // In ABI-safe path, wrapping them in DmlKernelImplAdapter hides the IControlFlowKernel interface,
            // breaking ORT's control flow setup. Let CPU EP handle them directly instead.
            // The subgraphs inside control flow nodes can still contain DML operators.
            const std::string& op_name = def->OpName();
            if (op_name == "If" || op_name == "Loop" || op_name == "Scan") {
                continue;
            }

            // Build an OrtKernelDef from the internal KernelDef properties
            OrtKernelDefBuilder* builder = nullptr;
            st = ep_api.CreateKernelDefBuilder(&builder);
            if (st != nullptr) {
                ep_api.ReleaseKernelRegistry(registry);
                return st;
            }

            st = ep_api.KernelDefBuilder_SetOperatorType(builder, def->OpName().c_str());
            if (st == nullptr) {
                st = ep_api.KernelDefBuilder_SetDomain(builder, def->Domain().c_str());
            }
            if (st == nullptr) {
                int start, end;
                def->SinceVersion(&start, &end);
                st = ep_api.KernelDefBuilder_SetSinceVersion(builder, start, end);
            }
            if (st == nullptr) {
                st = ep_api.KernelDefBuilder_SetExecutionProvider(builder, def->Provider().c_str());
            }

            // for each input memory type
            if (st == nullptr) {
                for (const auto& [index, mem_type] : def->InputMemoryTypeArgs()) {
                    st = ep_api.KernelDefBuilder_SetInputMemType(builder, index, mem_type);
                    if (st != nullptr)
                        break;
                }
            }

            // Copy output memory types
            if (st == nullptr) {
                for (const auto& [index, mem_type] : def->OutputMemoryTypeArgs()) {
                    st = ep_api.KernelDefBuilder_SetOutputMemType(builder, index, mem_type);
                    if (st != nullptr)
                        break;
                }
            }

            // convert MLDataType constraints to OrtDataType and copy them
            if (st == nullptr) {
                for (const auto& [constraint_name, ml_types] : create_info.kernel_def.get()->TypeConstraints()) {
                    std::vector<const OrtDataType*> ort_types;
                    ort_types.reserve(ml_types.size());

                    for (onnxruntime::MLDataType ml_type : ml_types) {
                        ONNXTensorElementDataType elem_type = GetElementTypeFromMLDataType(ml_type);
                        if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
                            continue; // skip unsupported types
                        }

                        const OrtDataType* ort_data_type = nullptr;
                        st = ep_api.GetTensorDataType(elem_type, &ort_data_type);
                        if (st != nullptr)
                            break;

                        ort_types.push_back(ort_data_type);
                    }

                    if (st != nullptr)
                        break;

                    if (!ort_types.empty()) {
                        st = ep_api.KernelDefBuilder_AddTypeConstraint(builder, constraint_name.c_str(),
                                                                       ort_types.data(), ort_types.size());
                        if (st != nullptr)
                            break;
                    }
                }
            }

            OrtKernelDef* ort_kernel_def = nullptr;
            if (st == nullptr) {
                st = ep_api.KernelDefBuilder_Build(builder, &ort_kernel_def);
            }
            ep_api.ReleaseKernelDefBuilder(builder);

            if (st != nullptr) {
                ep_api.ReleaseKernelRegistry(registry);
                return st;
            }

            // Keep the OrtKernelDef alive for the registry's lifetime
            m_ortKernelDefs.push_back(UniqueOrtKernelDef(ort_kernel_def, OrtKernelDefDeleter{&ep_api}));

            // Create a long-lived state that holds kernel creation metadata.
            // This state is passed to DmlKernelCreateFuncAdapter via kernel_create_func_state.
            auto func_state = std::make_unique<KernelCreateFuncState>();
            func_state->ort_api_ptr = &ort_api;
            func_state->operator_name = def->OpName();

            // Versioned key (domain::opname::sinceVersion) — must match what dml_abi_custom_registry.cc
            // writes. Required because ops like Pad, Slice, and Clip change their interface between
            // versions (e.g., inputs vs attributes), so each version needs its own kernelFactory.
            int since_ver_start = 0, since_ver_end = 0;
            def->SinceVersion(&since_ver_start, &since_ver_end);
            std::string regKey = std::string(def->Domain()) + "::" + std::string(def->OpName()) +
                                 "::" + std::to_string(since_ver_start);

            // Check if we have internal registration info for ABI-safe path
            auto reg_info_iter = m_internalRegInfoMap->find(regKey);

            if (reg_info_iter != m_internalRegInfoMap->end()) {
                const auto& reg_info = reg_info_iter->second;

                // Populate ABI-safe kernel creation fields (try this path first)
                func_state->kernel_factory = reg_info->kernelFactory;
                func_state->shape_inferrer = reg_info->shapeInferrer;
                func_state->default_attributes = &reg_info->defaultAttributes;
                // Derive required_constant_cpu_inputs from the versioned KernelDef rather than
                // reg_info, which may reflect a different version's constants. KernelDef is
                // per-registration and authoritative for the specific version being registered.
                for (const auto& [input_index, mem_type] : def->InputMemoryTypeArgs()) {
                    if (mem_type == OrtMemTypeCPUInput) {
                        func_state->required_constant_cpu_inputs.push_back(
                            static_cast<uint32_t>(input_index));
                    }
                }
                func_state->requires_input_shapes_at_creation = reg_info->requiresInputShapesAtCreation;
                func_state->requires_output_shapes_at_creation = reg_info->requiresOutputShapesAtCreation;
                func_state->is_internal_operator = reg_info->isInternalOperator;
                func_state->dml_execution_provider = m_executionProvider.get();
                func_state->ep_plugin = this;

                // Populate tensor attribute names by operator name — these are tensor-typed ONNX attributes
                // that cannot be stored in AttributeMap (which only supports int/float/string).
                // Only ConstantOfShape has a tensor-typed attribute ("value") in this codebase.
                static const std::unordered_map<std::string, std::vector<std::string>> kTensorAttrNames = {
                    {"ConstantOfShape", {"value"}},
                };
                auto tensor_attr_it = kTensorAttrNames.find(std::string(def->OpName()));
                if (tensor_attr_it != kTensorAttrNames.end()) {
                    func_state->tensor_attribute_names = tensor_attr_it->second;
                }
            }

            // ALWAYS store the lambda as fallback (in case ABI-safe fails)
            func_state->kernel_create_fn = const_cast<onnxruntime::KernelCreateFn*>(&create_info.kernel_create_func);

            st = ep_api.KernelRegistry_AddKernel(registry, ort_kernel_def, DmlKernelCreateFuncAdapter, func_state.get());

            if (st != nullptr) {
                ep_api.ReleaseKernelRegistry(registry);
                return st;
            }

            m_kernelCreateFuncStateByOpName[func_state->operator_name] = func_state.get();
            m_kernelCreateFuncStates.push_back(std::move(func_state));
        }

       m_ortKernelRegistry.reset(registry);
    }

    return nullptr;
}

// Adapter functions - Compute using unsafe cast to OpKernelContext
OrtStatus* ORT_API_CALL ExecutionProviderPlugin::DmlKernelImplAdapter_Compute(
    OrtKernelImpl* this_ptr,
    OrtKernelContext* context) noexcept
{
    auto* self = static_cast<DmlKernelImplAdapter*>(this_ptr);

    auto* op_ctx = reinterpret_cast<onnxruntime::OpKernelContext*>(context);

    try {
        auto status = self->internal_kernel->Compute(op_ctx);
        if (!status.IsOK()) {
            return self->ort_api_ptr->CreateStatus(ORT_FAIL, "Kernel compute failed");
        }
        return nullptr;
    } catch (const std::exception& e) {
        return self->ort_api_ptr->CreateStatus(ORT_FAIL, e.what());
    } catch (...) {
        return self->ort_api_ptr->CreateStatus(ORT_FAIL, "Unknown error during kernel compute");
    }
}

void ORT_API_CALL ExecutionProviderPlugin::DmlKernelImplAdapter_Release(OrtKernelImpl* this_ptr) noexcept
{
    delete static_cast<DmlKernelImplAdapter*>(this_ptr);
}

OrtStatus* ExecutionProviderPlugin::DmlKernelCreateFuncAdapter(void* kernel_create_func_state,
                                                                            const OrtKernelInfo* info,
                                                               OrtKernelImpl** kernel_out) noexcept {
    if (!kernel_create_func_state || !info || !kernel_out) {
        return nullptr;
    }

    try {
        *kernel_out = nullptr;
        auto* state = static_cast<KernelCreateFuncState*>(kernel_create_func_state);

        // TRY ABI-SAFE PATH FIRST
        if (state->kernel_factory)
        {
            try {
                // Resolve required constant inputs via two stages:
                // 1. KernelInfoGetConstantInput_tensor — covers statically visible initializers.
                // 2. m_graphInitializerMap — fallback for initializers not visible through the
                //    plugin EP's OrtKernelInfo* wrapper (e.g., inputs not wired as graph edges).
                // If both fail, pass an empty map so DmlAbiKernel_Create defers to lazy init.
                std::unordered_map<uint32_t, Microsoft::WRL::ComPtr<IMLOperatorTensor>> constants_to_pass;
                bool required_constants_available = true;

                for (uint32_t input_index : state->required_constant_cpu_inputs) {
                    const OrtValue* resolved_value = nullptr;

                    // Stage 1: KernelInfoGetConstantInput_tensor
                    int is_constant = 0;
                    const OrtValue* ki_value = nullptr;
                    OrtStatus* ki_st = state->ort_api_ptr->KernelInfoGetConstantInput_tensor(
                        info, input_index, &is_constant, &ki_value);
                    if (ki_st) state->ort_api_ptr->ReleaseStatus(ki_st);
                    if (is_constant && ki_value != nullptr) {
                        resolved_value = ki_value;
                    }

                    // Stage 2: graph initializer map (fallback for optional inputs not visible
                    // as connected node edges, e.g. Pad's pads/constant_value/axes).
                    if (resolved_value == nullptr) {
                        size_t name_len = 0;
                        state->ort_api_ptr->KernelInfo_GetInputName(info, input_index, nullptr, &name_len);
                        if (name_len > 0 && state->ep_plugin) {
                            std::string input_name(name_len, '\0');
                            OrtStatus* name_st = state->ort_api_ptr->KernelInfo_GetInputName(
                                info, input_index, input_name.data(), &name_len);
                            if (!name_st) {
                                auto it = state->ep_plugin->m_graphInitializerMap.find(input_name);
                                if (it != state->ep_plugin->m_graphInitializerMap.end()) {
                                    resolved_value = it->second;
                                }
                            }
                            if (name_st) state->ort_api_ptr->ReleaseStatus(name_st);
                        }
                    }

                    if (resolved_value != nullptr) {
                        constants_to_pass[input_index] = Microsoft::WRL::Make<Dml::AbiSafeTensor>(
                            resolved_value, state->ort_api_ptr, state->dml_execution_provider);
                    } else {
                        // Dynamically computed — defer to lazy init at Compute time.
                        required_constants_available = false;
                        break;
                    }
                }

                Dml::DmlKernelCreationState creation_state;
                creation_state.kernel_factory = state->kernel_factory;
                creation_state.shape_inferrer = state->shape_inferrer;
                creation_state.default_attributes = state->default_attributes;
                creation_state.required_constant_cpu_inputs = state->required_constant_cpu_inputs;
                creation_state.requires_input_shapes_at_creation = state->requires_input_shapes_at_creation;
                creation_state.requires_output_shapes_at_creation = state->requires_output_shapes_at_creation;
                creation_state.is_internal_operator = state->is_internal_operator;
                creation_state.tensor_attribute_names = state->tensor_attribute_names;
                creation_state.dml_execution_provider = state->dml_execution_provider;
                creation_state.ort_api = state->ort_api_ptr;
                creation_state.operator_name = state->operator_name.c_str();

                // Pass the resolved constants (or empty map for lazy-init case).
                // DmlAbiKernel_Create sets needs_lazy_init=true when required_constants_available=false,
                // returning a valid deferred kernel. DmlAbiKernel_Compute then fetches dynamically
                // computed constants from the execution context at first call.
                std::unordered_map<uint32_t, Microsoft::WRL::ComPtr<IMLOperatorTensor>> constants_final;
                if (required_constants_available) {
                    constants_final = std::move(constants_to_pass);
                }

#ifdef DML_PERF_PROFILE
                {
                    char _buf[320];
                    std::snprintf(_buf, sizeof(_buf),
                        "[ABI_SAFE] DmlAbiKernel_Create entry: op=%s  constants_ready=%d  passing=%zu constants\n",
                        state->operator_name.c_str(), (int)required_constants_available, constants_final.size());
                    Dml::DmlPerfWriteLog(_buf);
                }
#endif

                OrtStatus* abi_safe_status = Dml::DmlAbiKernel_Create(
                    &creation_state, info, kernel_out, std::move(constants_final));

                if (abi_safe_status == nullptr && *kernel_out != nullptr) {
#ifdef DML_PERF_PROFILE
                    { char _buf[256]; std::snprintf(_buf, sizeof(_buf), "[ABI_SAFE] success: op=%s  (kernel=%p)\n",
                        state->operator_name.c_str(), (void*)*kernel_out); Dml::DmlPerfWriteLog(_buf); }
                    { char _buf[256]; std::snprintf(_buf, sizeof(_buf), "[DML_PERF] path=safe  op=%s\n",
                        state->operator_name.c_str()); Dml::DmlPerfWriteLog(_buf); }
#endif
                    return nullptr;
                }

#ifdef DML_PERF_PROFILE
                { char _buf[256]; std::snprintf(_buf, sizeof(_buf),
                    "[ABI_SAFE] FAILED: op=%s  status=%p  kernel=%p  -> falling to unsafe\n",
                    state->operator_name.c_str(), (void*)abi_safe_status, (void*)*kernel_out);
                  Dml::DmlPerfWriteLog(_buf); }
#endif
                if (abi_safe_status) {
                    state->ort_api_ptr->ReleaseStatus(abi_safe_status);
                }
                *kernel_out = nullptr;

            } catch (...) {
                *kernel_out = nullptr;
            }
        }

        // FALLBACK: ABI-UNSAFE PATH (when ABI-safe fails or isn't available)
#ifdef DML_PERF_PROFILE
        { char _buf[256]; std::snprintf(_buf, sizeof(_buf),
            "[DML_PERF] path=unsafe op=%s  (safe path absent or failed)\n[ABI_UNSAFE] entry: op=%s\n",
            state->operator_name.c_str(), state->operator_name.c_str()); Dml::DmlPerfWriteLog(_buf); }
#endif

        if (!state->kernel_create_fn) {
            std::string error_msg = "Kernel registration missing both kernel_factory and kernel_create_fn - cannot create kernel";
            return state->ort_api_ptr->CreateStatus(ORT_FAIL, error_msg.c_str());
        }

        // Cast OrtKernelInfo to OpKernelInfo (ABI-UNSAFE!)
        const auto& kernel_info = *reinterpret_cast<const onnxruntime::OpKernelInfo*>(info);

        // Call the lambda - creates PluginDmlAbiOpKernel which can access constant inputs
        // Note: kernel_create_fn expects (FuncManager&, OpKernelInfo&, unique_ptr<OpKernel>&)
        std::unique_ptr<onnxruntime::OpKernel> op_kernel;
        onnxruntime::FuncManager func_manager;

        try {
            auto status = (*state->kernel_create_fn)(func_manager, kernel_info, op_kernel);
            if (!status.IsOK()) {
                return state->ort_api_ptr->CreateStatus(ORT_FAIL, "Kernel creation failed");
            }
        } catch (const std::exception& e) {
            return state->ort_api_ptr->CreateStatus(ORT_FAIL, e.what());
        } catch (...) {
            return state->ort_api_ptr->CreateStatus(ORT_FAIL, "Unknown error during kernel creation");
        }

        if (!op_kernel) {
            return state->ort_api_ptr->CreateStatus(ORT_FAIL, "Kernel creation returned null");
        }

        // Control flow operators (If, Loop, Scan) must NOT be wrapped in adapter
        // They implement IControlFlowKernel which ORT needs to call SetupSubgraphExecutionInfo on
        // Wrapping them hides this interface and causes "not created via OrtEpApi" error
        // These should never reach this point in ABI-safe path (filtered in ConvertKernelRegistryToOrtKernelRegistry)
        // but can reach here in unsafe fallback path
        if (state->operator_name == "If" || state->operator_name == "Loop" || state->operator_name == "Scan") {
            // ERROR: Control flow operators should not be going through EP plugin in the first place
            // They should be handled by CPU EP. This indicates a registration or partitioning issue.
            std::string error_msg = "Control flow operator " + state->operator_name +
                                  " should not be created by DirectML EP - this is a registration error";
            return state->ort_api_ptr->CreateStatus(ORT_FAIL, error_msg.c_str());
        }

        // Wrap the OpKernel in an adapter
        auto* adapter = new (std::nothrow) DmlKernelImplAdapter();
        if (!adapter) {
            return state->ort_api_ptr->CreateStatus(ORT_FAIL, "Failed to allocate kernel adapter");
        }

        adapter->internal_kernel = std::move(op_kernel);
        adapter->ort_api_ptr = state->ort_api_ptr;
        adapter->ort_version_supported = ORT_API_VERSION;
        adapter->flags = 0;
        adapter->Compute = DmlKernelImplAdapter_Compute;
        adapter->Release = DmlKernelImplAdapter_Release;
        adapter->PrePackWeight = nullptr;
        adapter->SetSharedPrePackedWeight = nullptr;

        *kernel_out = adapter;
        return nullptr;  // SUCCESS - unsafe fallback worked

    } catch (const std::exception& e) {
        auto* state = static_cast<KernelCreateFuncState*>(kernel_create_func_state);
        if (state && state->ort_api_ptr) {
            return state->ort_api_ptr->CreateStatus(ORT_RUNTIME_EXCEPTION, e.what());
        }
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}


const char* ORT_API_CALL ExecutionProviderPlugin::GetNameImpl(const OrtEp* this_ptr) noexcept
{
    const auto* ep = static_cast<const ExecutionProviderPlugin*>(this_ptr);
    return ep->name_.c_str();
}

OrtStatus* ORT_API_CALL ExecutionProviderPlugin::GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* graph,
                                                                 OrtEpGraphSupportInfo* graph_support_info) noexcept
{
    auto* ep = static_cast<ExecutionProviderPlugin*>(this_ptr);
    size_t numNodes = 0;
    ep->ort_api.Graph_GetNumNodes(graph, &numNodes);
    std::vector<const OrtNode*> nodesInTopologicalOrder;
    nodesInTopologicalOrder.resize(numNodes);
    ep->ort_api.Graph_GetNodes(graph, nodesInTopologicalOrder.data(), numNodes);

    uint32_t deviceDataTypeMask = ep->GetSupportedDeviceDataTypeMask(); // Each bit corresponds to each DML_TENSOR_DATA_TYPE.

    std::vector<const OrtNode*> tentativeNodes;
    tentativeNodes.reserve(nodesInTopologicalOrder.size());

    for (const OrtNode* node : nodesInTopologicalOrder) {
        const OrtKernelDef* kernel_def = nullptr;
        // kernel lookup
        OrtStatus* st = ep->ep_api.EpGraphSupportInfo_LookUpKernel(graph_support_info, node, &kernel_def);
        if (kernel_def != nullptr) {
            tentativeNodes.push_back(node);
        }
    }

    // Build a flat map of all graph initializers keyed by name, used as a fallback in
    // DmlKernelCreateFuncAdapter when KernelInfoGetConstantInput_tensor cannot see an
    // initializer. Graph_GetInitializers covers all initializers including those not wired
    // as connected node edges. ONNX value names are unique within a graph, so name is an
    // unambiguous key.
    ep->m_graphInitializerMap.clear();
    {
        size_t num_initializers = 0;
        ep->ort_api.Graph_GetNumInitializers(graph, &num_initializers);
        if (num_initializers > 0) {
            std::vector<const OrtValueInfo*> initializer_infos(num_initializers);
            ep->ort_api.Graph_GetInitializers(graph, initializer_infos.data(), num_initializers);
            for (const OrtValueInfo* vi : initializer_infos) {
                if (!vi) continue;
                const char* name_cstr = nullptr;
                OrtStatus* name_st = ep->ort_api.GetValueInfoName(vi, &name_cstr);
                if (name_st || !name_cstr) { if (name_st) ep->ort_api.ReleaseStatus(name_st); continue; }
                const OrtValue* init_value = nullptr;
                OrtStatus* val_st = ep->ort_api.ValueInfo_GetInitializerValue(vi, &init_value);
                if (val_st == nullptr && init_value != nullptr) {
                    ep->m_graphInitializerMap[name_cstr] = init_value;
                }
                if (val_st) ep->ort_api.ReleaseStatus(val_st);
            }
        }
    }

    // Get the list of nodes that should stay on the CPU
    std::unordered_set<size_t> cpuPreferredNodes = ep->GetCpuPreferredNodes(graph, graph_support_info, tentativeNodes);
    std::vector<const OrtNode*> supportedNodes;
    for (const OrtNode* node : nodesInTopologicalOrder) {
        size_t nodeID = 0;
        ep->ort_api.Node_GetId(node, &nodeID);

        const char* op_type = nullptr;
        ep->ort_api.Node_GetOperatorType(node, &op_type);

        bool isSupported = ep->IsNodeSupportedByDml(node, graph_support_info, deviceDataTypeMask);
        bool notCpuPreferred = cpuPreferredNodes.find(nodeID) == cpuPreferredNodes.end();

        if (isSupported && notCpuPreferred)
        {
            ep->ep_api.EpGraphSupportInfo_AddSingleNode(graph_support_info, node);
            supportedNodes.push_back(node);
        }
    }

    ep->m_isGetCapabilityCompleted = true;
    return nullptr;
}

OrtStatus* ORT_API_CALL ExecutionProviderPlugin::CompileImpl(_In_ OrtEp* this_ptr, _In_ const OrtGraph** graphs,
                                    _In_ const OrtNode** fused_nodes, _In_ size_t count,
                                    _Out_writes_all_(count) OrtNodeComputeInfo** node_compute_infos,
                                    _Out_writes_(count) OrtNode** ep_context_nodes) noexcept
{
    auto* ep = static_cast<ExecutionProviderPlugin*>(this_ptr);
    return nullptr;
}

void ORT_API_CALL ExecutionProviderPlugin::ReleaseNodeComputeInfosImpl(
    OrtEp* this_ptr,
    OrtNodeComputeInfo** node_compute_infos,
    size_t num_node_compute_infos) noexcept
{
    auto* ep = static_cast<ExecutionProviderPlugin*>(this_ptr);
    (void)this_ptr;
}

OrtStatus* ORT_API_CALL ExecutionProviderPlugin::GetPreferredDataLayoutImpl(
    _In_ OrtEp* this_ptr,
    _Out_ OrtEpDataLayout* preferred_data_layout) noexcept
{
    auto* ep = static_cast<ExecutionProviderPlugin*>(this_ptr);
    *preferred_data_layout = OrtEpDataLayout_Default;
    return nullptr;
}

OrtStatus* ORT_API_CALL ExecutionProviderPlugin::ShouldConvertDataLayoutForOpImpl(
    _In_ OrtEp* this_ptr,
    _In_z_ const char* domain,
    _In_z_ const char* op_type,
    OrtEpDataLayout target_data_layout,
    _Out_ int* should_convert) noexcept
{
    auto* ep = static_cast<ExecutionProviderPlugin*>(this_ptr);
    return nullptr;
}

OrtStatus* ORT_API_CALL ExecutionProviderPlugin::SetDynamicOptionsImpl(
    _In_ OrtEp* this_ptr,
    const char* const* option_keys,
    const char* const* option_values,
    size_t num_options) noexcept
{
    auto* ep = static_cast<ExecutionProviderPlugin*>(this_ptr);
    return nullptr;
}

OrtStatus* ORT_API_CALL ExecutionProviderPlugin::OnRunStartImpl(
    _In_ OrtEp* this_ptr,
    const OrtRunOptions* run_options) noexcept
{
    auto* ep = static_cast<ExecutionProviderPlugin*>(this_ptr);

    ep->m_executionProvider->OnSessionInitializationEnd();

    ep->m_executionProvider.get()->OnRunStart(*run_options);
    return nullptr;
}

OrtStatus* ORT_API_CALL ExecutionProviderPlugin::OnRunEndImpl(
    _In_ OrtEp* this_ptr,
    const OrtRunOptions* run_options,
    bool sync_stream) noexcept
{
    auto* ep = static_cast<ExecutionProviderPlugin*>(this_ptr);

    ep->m_executionProvider.get()->OnRunEnd();
    return nullptr;
}

OrtStatus* ORT_API_CALL ExecutionProviderPlugin::CreateAllocatorImpl(_In_ OrtEp* this_ptr,
                                                                     const OrtMemoryInfo* memory_info,
                                                                     OrtAllocator** allocator) noexcept
{
    auto& impl = *static_cast<ExecutionProviderPlugin*>(this_ptr);
    *allocator = nullptr;

    bool is_cpu_input_allocator = impl.IsCpuAllocator(memory_info);
    bool is_gpu_input_allocator = impl.IsGpuAllocator(memory_info);

    if (!is_gpu_input_allocator && !is_cpu_input_allocator) {
        return impl.ort_api.CreateStatus(ORT_INVALID_ARGUMENT,
                                         "INTERNAL ERROR! Unknown memory info provided to CreateAllocator. "
                                         "Value did not come directly from an OrtEpDevice returned by this factory.");
    }

    std::vector<OrtAllocator*> allocators = impl.m_executionProvider.get()->CreatePreferredAllocators();

    // is cpu allocator
    if (is_cpu_input_allocator) {
        *allocator = allocators[1];
        return nullptr;
    } else {
        // is gpu allocator
        *allocator = allocators[0];
        return nullptr;
    }

    return nullptr;
}

OrtStatus* ORT_API_CALL ExecutionProviderPlugin::CreateSyncStreamForDeviceImpl(
    _In_ OrtEp* this_ptr,
    _In_ const OrtMemoryDevice* memory_device,
    _Outptr_ OrtSyncStreamImpl** stream) noexcept
{
    const auto& impl = *static_cast<const ExecutionProviderPlugin*>(this_ptr);
    //auto sync_stream = std::make_unique<StreamImpl>(ep->factory_, ep, nullptr);
    return nullptr;
}

const char* ORT_API_CALL ExecutionProviderPlugin::GetCompiledModelCompatibilityInfoImpl(
    _In_ OrtEp* this_ptr,
    _In_ const OrtGraph* graph) noexcept
{
    return nullptr;
}

OrtStatus* ORT_API_CALL ExecutionProviderPlugin::GetKernelRegistryImpl(
    _In_ OrtEp* this_ptr,
    _Outptr_ const OrtKernelRegistry** kernel_registry) noexcept
{
    auto* ep = static_cast<ExecutionProviderPlugin*>(this_ptr);

    *kernel_registry = ep->m_ortKernelRegistry.get();
    return nullptr;
}

OrtStatus* ORT_API_CALL ExecutionProviderPlugin::IsConcurrentRunSupportedImpl(_In_ OrtEp* this_ptr,
                                                                            _Out_ bool* is_supported) noexcept {
    // DML EP supports concurrent runs
    *is_supported = true;
    return nullptr;
}

OrtStatus* ORT_API_CALL ExecutionProviderPlugin::OnSessionInitializationEndImpl(_In_ OrtEp* this_ptr) noexcept {
    auto* ep = static_cast<ExecutionProviderPlugin*>(this_ptr);

    ep->m_executionProvider->OnSessionInitializationEnd();
    return nullptr;
}

bool ExecutionProviderPlugin::IsCpuAllocator(const OrtMemoryInfo* memory_info)
{
    if (!memory_info) {
        return false;
    }

    OrtMemoryInfoDeviceType device_type;
    ort_api.MemoryInfoGetDeviceType(memory_info, &device_type);
    const char* name = nullptr;
    ort_api.MemoryInfoGetName(memory_info, &name);

    return (name != nullptr) && (std::strcmp(name, "directML_ep_cpu") == 0) && (device_type == OrtMemoryInfoDeviceType_CPU);
}

bool ExecutionProviderPlugin::IsGpuAllocator(const OrtMemoryInfo* memory_info)
{
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

uint32_t ExecutionProviderPlugin::GetSupportedDeviceDataTypeMask() const {
    return Dml::GetSupportedDeviceDataTypeMask(m_dmlDevice.Get());
}

std::unordered_set<size_t> ExecutionProviderPlugin::GetCpuPreferredNodes(const OrtGraph* graph,
                                                                         OrtEpGraphSupportInfo* graph_support_info,
                                                   gsl::span<const OrtNode*> tentative_nodes) {
    size_t numNodes = 0;
    ort_api.Graph_GetNumNodes(graph, &numNodes);
    std::vector<const OrtNode*> ordered_nodes;
    ordered_nodes.resize(numNodes);
    ort_api.Graph_GetNodes(graph, ordered_nodes.data(), numNodes);

    std::unordered_map<size_t, const OrtNode*> id_to_node;
    for (const auto* n : ordered_nodes) {
        size_t nid = 0;
        ort_api.Node_GetId(n, &nid);
        id_to_node[nid] = n;
  }

    std::vector<size_t> orderedNodesId;
    orderedNodesId.resize(ordered_nodes.size());
    size_t maxNodeIndex = 1;
    for (int x = 0; x < ordered_nodes.size(); x++)
    {
        ort_api.Node_GetId(ordered_nodes[x], &orderedNodesId[x]);

        if (maxNodeIndex < orderedNodesId[x])
        {
            maxNodeIndex = orderedNodesId[x];
        }
    }

    onnxruntime::InlinedVector<size_t> node_id_to_order_map(maxNodeIndex+1);
    for (size_t id = 0; id < numNodes; id++)
    {
        const size_t& node_id = orderedNodesId[id];
        node_id_to_order_map[node_id] = id;
    }

    // If return false, n1 will be output first; If return true, n2 will be output first
    auto greater_order_comp = [&](const size_t n1, const size_t n2) {
        return node_id_to_order_map[n1] > node_id_to_order_map[n2];
    };

    std::priority_queue<size_t, std::vector<size_t>, decltype(greater_order_comp)> candidates(greater_order_comp);
    
    onnxruntime::InlinedHashSet<const OrtValueInfo*> cpu_output_args;

    onnxruntime::InlinedHashSet<size_t> provider_nodes;
    provider_nodes.reserve(tentative_nodes.size());

    onnxruntime::InlinedHashMap<size_t, const OrtKernelDef*> node_to_kernel;
    node_to_kernel.reserve(tentative_nodes.size());

    std::vector<OrtValueInfo::ConsumerInfo> consumerInfo;

    for (auto& tentativeNode : tentative_nodes)
    {
        size_t node_id = 0;
        ort_api.Node_GetId(tentativeNode, &node_id);
        provider_nodes.insert(node_id);


        const OrtKernelDef* kernel_def = nullptr;
        ep_api.EpGraphSupportInfo_LookUpKernel(graph_support_info, tentativeNode, &kernel_def);

        node_to_kernel.insert({node_id, kernel_def});

        size_t nodeNumOutputs = 0;
        ort_api.Node_GetNumOutputs(tentativeNode, &nodeNumOutputs);

        std::vector<const OrtValueInfo*> valueInfo;
        valueInfo.resize(nodeNumOutputs);
        ort_api.Node_GetOutputs(tentativeNode, valueInfo.data(), nodeNumOutputs);

        for (int x = 0; x < nodeNumOutputs; x++)
        {
            OrtMemType mem_type;
            ep_api.KernelDef_GetOutputMemType(kernel_def, x, &mem_type);

            if (mem_type == OrtMemTypeCPUOutput || mem_type == OrtMemTypeCPUInput)
            {
                cpu_output_args.insert(valueInfo[x]);

                size_t numConsumerInfos = 0;
                valueInfo[x]->GetNumConsumerInfos(numConsumerInfos);

                std::vector<const OrtNode*> consumerNodes;
                std::vector<int64_t> consumerNodeIndices;
                consumerNodes.resize(numConsumerInfos);
                consumerNodeIndices.resize(numConsumerInfos);

                ort_api.ValueInfo_GetValueConsumers(valueInfo[x], consumerNodes.data(),consumerNodeIndices.data(), numConsumerInfos);

                for (int y = 0; y < numConsumerInfos; y++)
                {
                    size_t consumerId = 0;
                    ort_api.Node_GetId(consumerNodes[y], &consumerId);
                    candidates.push(consumerId);
                }
            }
        }
    }

    size_t graphNumInputs = 0;
    std::vector<const OrtValueInfo*> graphInputsValueInfo;

    ort_api.Graph_GetNumInputs(graph, &graphNumInputs);
    graphInputsValueInfo.resize(graphNumInputs);

    ort_api.Graph_GetInputs(graph, graphInputsValueInfo.data(), graphNumInputs);

    onnxruntime::InlinedHashSet<size_t> visited;
    visited.reserve(candidates.size());
    std::unordered_set<size_t> cpu_nodes;
    cpu_nodes.reserve(candidates.size());

  // The algo below is trying to identity a subgraph that only depends on cpu tensors.
    // Usually it is a subgraph that doing shape calculation based on a GPU tensor, then reshape it back.
    // The detail:
    // for each candidate, if one of its input is a cpu tensor and the Non-CPU kernel doesn't mark it as cpu input,
    // force the node to CPU to avoid memory cpu and add its output to the small cpu tensors.
    while (!candidates.empty()) {
        size_t cur = candidates.top();
        candidates.pop();

        auto p = visited.insert(cur);
        if (!p.second)
            continue;

        const OrtNode* candidateNode = nullptr;
        auto it = id_to_node.find(cur);
        if (it != id_to_node.end())
        {
            // If we found the current node in the ordered list, we can use its index.
            candidateNode = it->second;
        }

        if (provider_nodes.find(cur) == provider_nodes.end())
        {
            const char* ep_name = nullptr;
            ort_api.Node_GetEpName(candidateNode, &ep_name);
            // Nodes not in provider_nodes are either have EP assigned or no kernel found on target EP.
            // we assume these nodes will fallback to CPU, so add all direct consumers of all outputs to candidates.
            if (ep_name == nullptr || std::strcmp(ep_name, "") == 0 ||
                std::strcmp(ep_name, "CPUExecutionProvider") == 0)
            {
                size_t nodeNumOutputs = 0;
                std::vector<const OrtValueInfo*> valueInfo;

                ort_api.Node_GetNumOutputs(candidateNode, &nodeNumOutputs);
                valueInfo.resize(nodeNumOutputs);
                ort_api.Node_GetOutputs(candidateNode, valueInfo.data(), nodeNumOutputs);


                for (auto* output : valueInfo) {
                    cpu_output_args.insert(output);
                }
                for (auto it = valueInfo.begin(); it != valueInfo.end(); ++it)
                {
                    size_t numConsumers = 0;
                    ort_api.ValueInfo_GetValueNumConsumers(*it, &numConsumers);

                    std::vector<const OrtNode*> consumerNodes;
                    std::vector<int64_t> consumerNodeIndices;
                    consumerNodes.resize(numConsumers);
                    consumerNodeIndices.resize(numConsumers);

                    ort_api.ValueInfo_GetValueConsumers(*it, consumerNodes.data(), consumerNodeIndices.data(), numConsumers);

                    for (int y = 0; y < numConsumers; y++)
                    {
                        size_t consumerId = 0;
                        ort_api.Node_GetId(consumerNodes[y], &consumerId);
                        candidates.push(consumerId);
                    }
                }
            }
            continue;
        }

        bool place_in_cpu = true;

        size_t nodeNumInputs = 0;
        std::vector<const OrtValueInfo*> valueInfoInputs;
        ort_api.Node_GetNumInputs(candidateNode, &nodeNumInputs);

        valueInfoInputs.resize(nodeNumInputs);
        ort_api.Node_GetInputs(candidateNode, valueInfoInputs.data(), nodeNumInputs);

        for (size_t i = 0; i < valueInfoInputs.size(); ++i)
        {
            auto* input = valueInfoInputs[i];

            ONNXTensorElementDataType datatype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            ort_api.GetTensorElementType(input->GetTypeInfo()->tensor_type_info.get(), &datatype);

            // skip placing on CPU if the data typs is float16 or bfloat16 or
            // float8e4m3fn, float8e4m3fnuz, floate5m2, floate5m2fnuz or float4e2m1
            if (datatype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
                datatype == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 ||
                datatype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN ||
                datatype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ ||
                datatype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2 ||
                datatype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ ||
                datatype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1) {
                place_in_cpu = false;
                break;
            }

            // allow placing on CPU if it's a small initializer or graph input
            if (IsSmallInitializer(graph,input) ||
                std::find(graphInputsValueInfo.begin(), graphInputsValueInfo.end(), input) != graphInputsValueInfo.end()) {
                continue;
            }

            // the input is not a CPU tensor
            if (cpu_output_args.find(input) == cpu_output_args.end()) {
                place_in_cpu = false;
                break;
            }

            OrtMemType mem_type;
            ep_api.KernelDef_GetInputMemType(node_to_kernel[cur], i, &mem_type);
            // input is a CPU tensor, but it's intended to be consumed as CPU input by the target EP
            if (mem_type == OrtMemTypeCPUInput) {
                place_in_cpu = false;
                break;
            }
        }

        size_t nodeNumOutputs = 0;
        std::vector<const OrtValueInfo*> outputNodeDefs;

        ort_api.Node_GetNumOutputs(candidateNode, &nodeNumOutputs);
        outputNodeDefs.resize(nodeNumOutputs);
        ort_api.Node_GetOutputs(candidateNode, outputNodeDefs.data(), nodeNumOutputs);

        if (place_in_cpu) {
            cpu_nodes.insert(cur);

            //LOGS(logger, INFO)
            //    << "ORT optimization- Force fallback to CPU execution for node: " << node->Name()
            //    << " because the CPU execution path is deemed faster than overhead involved with execution "
            //       "on other EPs capable of executing this node";
            for (auto* output : outputNodeDefs) {
                cpu_output_args.insert(output);
            }
            for (auto it = outputNodeDefs.begin(); it != outputNodeDefs.end(); ++it) {
                size_t numConsumers = 0;
                ort_api.ValueInfo_GetValueNumConsumers(*it, &numConsumers);

                std::vector<const OrtNode*> consumerNodes;
                std::vector<int64_t> consumerNodeIndices;
                consumerNodes.resize(numConsumers);
                consumerNodeIndices.resize(numConsumers);

                ort_api.ValueInfo_GetValueConsumers(*it, consumerNodes.data(), consumerNodeIndices.data(),
                                                    numConsumers);

                for (int y = 0; y < numConsumers; y++) {
                    size_t consumerId = 0;
                    ort_api.Node_GetId(consumerNodes[y], &consumerId);
                    candidates.push(consumerId);
                }
            }
        }
    }

    return cpu_nodes;
}

bool ExecutionProviderPlugin::IsNodeSupportedByDml(
    const OrtNode* node,
    OrtEpGraphSupportInfo* graph_support_info,
    uint32_t supportedDeviceDataTypeMask)
{
    const OrtKernelDef* kernel_def = nullptr;
    // kernel lookup
    OrtStatus* st = ep_api.EpGraphSupportInfo_LookUpKernel(graph_support_info, node, &kernel_def);

    if (st != nullptr)
    {
        ort_api.ReleaseStatus(st);
        return false;
    }

    if (kernel_def == nullptr)
    {
        // ORT couldn't find any kernel for this node in the current kernel lookup context
        return false;
    }

    const char* provider = ep_api.KernelDef_GetExecutionProvider(kernel_def);
    if (provider == nullptr || std::strcmp(provider, "DirectMLExecutionProvider") != 0)
    {
        // Not a DML kernel -> won't be in internal map
        return false;
    }

    // Build key from domain and operator name
    const char* op_name = ep_api.KernelDef_GetOperatorType(kernel_def);
    const char* domain = ep_api.KernelDef_GetDomain(kernel_def);
    std::string regKey = std::string(domain) + "::" + std::string(op_name);

    auto regInfoIter = m_internalRegInfoMap->find(regKey);
    std::shared_ptr<InternalRegistrationInfo> internalRegInfo;

    if (regInfoIter != m_internalRegInfoMap->end())
    {
        internalRegInfo = regInfoIter->second;

        if (internalRegInfo->supportQuery) {
            bool queryResult = internalRegInfo->supportQuery(node, ort_api);
            if (!queryResult) {
                return false;
            }
        }
    }

    // Check whether the node uses any data types which are unsupported by the device.
    // Pass nullptr for regInfo during graph partitioning - requiredConstantCpuInputs is for kernel creation, not capability checking
    bool dataTypesSupported = DoesNodeContainSupportedDataTypes(node, nullptr, supportedDeviceDataTypeMask,
                                           m_native16BitShaderOpsSupported);
    if (!dataTypesSupported) {
        return false;
    }

    return true;
}

bool ExecutionProviderPlugin::DoesNodeContainSupportedDataTypes(
    const OrtNode* node,
    const Windows::AI::MachineLearning::Adapter::InternalRegistrationInfo* regInfo,
    uint32_t supportedDeviceDataTypeMask, // Each bit corresponds to each DML_TENSOR_DATA_TYPE.
    bool native16BitShaderOpsSupported)
{
    const char* op_type = nullptr;
    ort_api.Node_GetOperatorType(node, &op_type);

    Dml::OrtNodeAdapter adapter(node, ort_api);
    std::vector<const Dml::OrtValueInfoAdapter*> constantCpuInputs;

    if (regInfo != nullptr) {
        // Collect the list of CPU-bound input tensors, needed when checking 64-bit fallback
        // or for other data types like int-8 which may be supported for CPU inputs but not
        // GPU inputs.
        const auto& inputDefinitions = adapter.GetInputs();
        for (uint32_t i : regInfo->requiredConstantCpuInputs) {
            if (i < inputDefinitions.size()) {
                constantCpuInputs.push_back(inputDefinitions[i].get());
            }
        }
    }

    // Assume data types are supported until proven otherwise.
    bool nodeContainsSupportedDataTypes = true;

    // Callback to check each node's data type against registered operator support.
    auto nodeCallback = [&](const Dml::OrtValueInfoAdapter& valueInfo, bool isInput) -> void {
        // Get the tensor element data type for this node, comparing against what the device actually supports.


        // Reject node if undefined data type or non-tensor, as DML cannot handle it.
        MLOperatorEdgeType edgeType;
        MLOperatorTensorDataType onnxElementType;

        // Check type using OrtValueInfoAdapter
        if (!valueInfo.IsTensor()) {
            // If the model has nodes that use Optional we will arrive here. It's a valid ONNX model but
            // we don't handle Optional.
            nodeContainsSupportedDataTypes = false;
            return;
        }

        edgeType = MLOperatorEdgeType::Tensor;
        onnxElementType = static_cast<MLOperatorTensorDataType>(valueInfo.GetTensorElementType());

        if (onnxElementType == MLOperatorTensorDataType::Float16 && !native16BitShaderOpsSupported &&
            IsCustomOpShader(adapter)) {
            // STFT is a special case since it has a dml ep registered
            // graph transformation that will decompose fp16 STFT into convolution
            // and so it is OK to register for fp16.
            if (strcmp("STFT", adapter.GetOpType().c_str()) != 0) {
                nodeContainsSupportedDataTypes = false;
                return;
            }
        }

        // Allow nodeArgs that are SequenceTensor when they are actually implemented by CPU Kernels.
        if (edgeType == MLOperatorEdgeType::SequenceTensor) {
            if (!IsCpuOnDmlOperator(adapter) && !IsDmlSequenceOperator(adapter)) {
                nodeContainsSupportedDataTypes = false;
            }
            return;
        }

        // Reject node for unknown DML data types.
        DML_TENSOR_DATA_TYPE dmlElementType = GetDmlDataTypeFromMlDataTypeNoThrow(onnxElementType);
        if (dmlElementType == DML_TENSOR_DATA_TYPE_UNKNOWN) {
            nodeContainsSupportedDataTypes = false;
            return;
        }

        // Succeed if the tensor is CPU-bound, as the CPU-side reading code is generic enough
        // to handle multiple types regardless of GPU capability (typically these are just
        // scalars or simple 1D arrays).
        bool isConstantCpuInput = isInput &&
            std::find(constantCpuInputs.begin(), constantCpuInputs.end(), &valueInfo) != constantCpuInputs.end();
        if (isConstantCpuInput) {
            // Leave nodeContainsSupportedDataTypes alone.
            return;
        }

        bool isDataTypeSupported = (1 << dmlElementType) & supportedDeviceDataTypeMask;

        // Reject node if the data type is unsupported by the device.
        if (!isDataTypeSupported) {
            nodeContainsSupportedDataTypes = false;
            return;
        }

    };

    // Check whether the node uses any data types which are unsupported by the device.
    adapter.ForEachDef(nodeCallback);

    return nodeContainsSupportedDataTypes;
}


bool ExecutionProviderPlugin::TryGetTensorDataType(
    const onnxruntime::NodeArg& nodeArg,
    _Out_ MLOperatorEdgeType* edgeType,
    _Out_ MLOperatorTensorDataType* onnxElementType)
{
    *onnxElementType = MLOperatorTensorDataType::Undefined;
    *edgeType = MLOperatorEdgeType::Undefined;

    // nodeArg.Type() returns an interned string like "tensor(float16)" or "seq(tensor(float))".
    // Parse it without touching protobuf objects (TypeAsProto() is unsafe across DLL boundary).
    const std::string* typeStr = nodeArg.Type();
    if (!typeStr || typeStr->empty()) return false;

    std::string_view sv(*typeStr);

    // Map elem-type substring → MLOperatorTensorDataType
    static const std::pair<std::string_view, MLOperatorTensorDataType> kElemTypes[] = {
        {"float16",  MLOperatorTensorDataType::Float16},
        {"bfloat16", static_cast<MLOperatorTensorDataType>(16)},  // ONNX TensorProto_DataType_BFLOAT16
        {"double",   MLOperatorTensorDataType::Double},
        {"float",    MLOperatorTensorDataType::Float},
        {"int64",    MLOperatorTensorDataType::Int64},
        {"int32",    MLOperatorTensorDataType::Int32},
        {"int16",    MLOperatorTensorDataType::Int16},
        {"int8",     MLOperatorTensorDataType::Int8},
        {"uint64",   MLOperatorTensorDataType::UInt64},
        {"uint32",   MLOperatorTensorDataType::UInt32},
        {"uint16",   MLOperatorTensorDataType::UInt16},
        {"uint8",    MLOperatorTensorDataType::UInt8},
        {"bool",     MLOperatorTensorDataType::Bool},
        {"string",   MLOperatorTensorDataType::String},
    };

    bool isTensor   = sv.substr(0, 7) == "tensor(";
    bool isSequence = !isTensor && sv.substr(0, 11) == "seq(tensor(";

    if (!isTensor && !isSequence) return false;

    *edgeType = isTensor ? MLOperatorEdgeType::Tensor : MLOperatorEdgeType::SequenceTensor;

    // Find the elem-type token inside the outermost parens
    auto lparen = sv.find('(');
    if (isSequence) lparen = sv.find('(', lparen + 1); // skip "seq(" → find "tensor("'s '('
    auto rparen  = sv.rfind(')');
    if (lparen == std::string_view::npos || rparen == std::string_view::npos) return false;
    std::string_view inner = sv.substr(lparen + 1, rparen - lparen - 1);
    // inner is e.g. "float16" or "float" or "int64"
    // strip trailing ')' for seq case: "tensor(float16)" → inner="float16)"
    if (auto rp = inner.rfind(')'); rp != std::string_view::npos) inner = inner.substr(0, rp);
    // strip "tensor(" prefix if still present (seq case leaves "tensor(float16")
    if (inner.substr(0, 7) == "tensor(") inner = inner.substr(7);

    for (const auto& [name, dtype] : kElemTypes) {
        if (inner == name) {
            *onnxElementType = dtype;
            return true;
        }
    }

    return false;
}

bool ExecutionProviderPlugin::IsCustomOpShader(const Dml::OrtNodeAdapter& adapter) {
    auto custom_ops = std::array<const char*, 3>{"DFT", "STFT", "GridSample"};

    for (auto& custom_op : custom_ops) {
        if (strcmp(custom_op, adapter.GetOpType().c_str()) == 0) {
            return true;
        }
    }
    return false;
}

bool ExecutionProviderPlugin::IsCpuOnDmlOperator(const Dml::OrtNodeAdapter& adapter) {
    auto cpuOnDmlOperators = std::array<const char*, 9>{
        "SequenceAt",         "SequenceConstruct",  "SequenceEmpty",
        "SequenceLength",     "SequenceErase",      "SequenceInsert",
        "OptionalGetElement", "OptionalHasElement", "If",
    };

    for (auto& cpuOnDmlOperator : cpuOnDmlOperators) {
        if (strcmp(cpuOnDmlOperator, adapter.GetOpType().c_str()) == 0) {
            return true;
        }
    }
    return false;
}

bool ExecutionProviderPlugin::IsDmlSequenceOperator(const Dml::OrtNodeAdapter& adapter) {
    auto sequence_ops = std::array<const char*, 1>{"ConcatFromSequence"};

    for (auto& sequence_op : sequence_ops) {
        if (strcmp(sequence_op, adapter.GetOpType().c_str()) == 0) {
            return true;
        }
    }
    return false;
}

bool ExecutionProviderPlugin::IsSmallInitializer(const OrtGraph* graph, const OrtValueInfo* valueInfo) {

    std::vector<const OrtValue*> init_value;
    init_value.resize(1);
    size_t numInit = 0;
    ort_api.Graph_GetNumInitializers(graph, &numInit);
    //const OrtValue* 
    auto status = ort_api.ValueInfo_GetInitializerValue(valueInfo, init_value.data());
    //auto status = valueInfo->GetInitializerValue(init_value);

    if (status != nullptr || init_value[0] == nullptr) {
        return false; // not an initializer
    }

    size_t dimsCount = 0;
    std::vector<int64_t> dims;
    ort_api.GetDimensionsCount(valueInfo->GetTypeInfo()->tensor_type_info.get(), &dimsCount);
    dims.resize(dimsCount);
    ort_api.GetDimensions(valueInfo->GetTypeInfo()->tensor_type_info.get(), dims.data(), dims.size());

    // Check if "small" enough
    int64_t size = 1;
    for (auto& dim : dims) {
        size *= dim;
    }

    return size <= kSmallInitializerThreshold;
}

void ExecutionProviderPlugin::Flush() const
{
    assert(!m_closed);
    m_context->Flush();
}

std::shared_ptr<Dml::PluginDmlExecutionProviderImpl> ExecutionProviderPlugin::GetInternetalExecutionProvider() {
    return m_executionProvider;
}

bool ExecutionProviderPlugin::IsGetCapabilityCompleted()
{
    return m_isGetCapabilityCompleted;
}

DMLDataTransfer* ExecutionProviderPlugin::GetDataTransfer()
{
    return m_dataTransfer.get();
}

bool ExecutionProviderPlugin::GraphCaptureEnabled() const noexcept { return m_graphCaptureEnabled; }

}  // namespace Dml
