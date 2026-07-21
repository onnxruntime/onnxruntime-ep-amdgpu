// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "dml_ep.h"
#include "dml_plugin_OperatorRegistration.h"
#include "core/common/inlined_containers.h"
#include "core/common/inlined_containers_fwd.h"
#include "core/framework/fuse_nodes_funcs.h"
#include "dml_abi_kernel.h"
#include "ep_fusion_manager.h"
#include "full_graph_fusion.h"
#include "quick_gelu_ep_fusion.h"

namespace dml_ep {

    // ep_name must match the name under which this EP is registered with ORT.
    // Kernels stamped with a different name than the running EP's GetName() will never
    // be dispatched — ORT matches kernel provider strings to EP type strings exactly.
    static void CreateDmlKernelRegistry(
        _In_ const PluginDmlExecutionProviderImpl* executionProvider,
        std::string_view ep_name,
        _Out_ std::shared_ptr<onnxruntime::KernelRegistry>* registry,
        _Out_ std::shared_ptr<const InternalRegistrationInfoMap>* internalRegInfoMap)
    {
        Microsoft::WRL::ComPtr<PluginAbiCustomRegistry> abiRegistry = wil::MakeOrThrow<PluginAbiCustomRegistry>(executionProvider, ep_name);
        RegisterDmlOperators(abiRegistry.Get(), executionProvider);

        assert(abiRegistry->GetRegistries().size() == 1);

        auto customRegistry = *abiRegistry->GetRegistries().begin();
        *registry = customRegistry->GetKernelRegistry();
        *internalRegInfoMap = abiRegistry->GetInternalRegInfoMap();

        RegisterCpuOperatorsAsDml(registry->get(), ep_name);
    }
    
ExecutionProviderPlugin::~ExecutionProviderPlugin() {
    m_context->Close();
}

ExecutionProviderPlugin::ExecutionProviderPlugin(
    const ApiPtrs& api_ptrs,
    std::string_view name, 
    ID3D12Device* d3d12_device_,
    IDMLDevice* dml_device_,
    Microsoft::WRL::ComPtr<ExecutionContext> executionContext)
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

    D3D12_FEATURE_DATA_D3D12_OPTIONS4 featureOptions = {};
    if (SUCCEEDED(d3d12_device->CheckFeatureSupport(
        D3D12_FEATURE_D3D12_OPTIONS4,
        &featureOptions,
        sizeof(featureOptions))))
    {
        m_native16BitShaderOpsSupported = featureOptions.Native16BitShaderOpsSupported;
    }

    m_executionProvider = std::make_shared<PluginDmlExecutionProviderImpl>(
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

    CreateDmlKernelRegistry(m_executionProvider.get(), name_, &m_kernelRegistry, &m_internalRegInfoMap);
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
    if (kernel_registry_ == nullptr) {
        OrtKernelRegistry* registry = nullptr;
        OrtStatus* st = ep_api.CreateKernelRegistry(&registry);
        if (st != nullptr) {
            return st;
        }
        kernel_registry_ = UniqueOrtKernelRegistry(registry, OrtKernelRegistryDeleter{&ep_api});

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

            // Copy input/output aliases (e.g., Reshape output[0] aliases input[0]).
            // Without this, ORT's allocation planner allocates separate output buffers and
            // assigns kReuse based on graph-level shapes. When dynamic dimensions change
            // between Run() calls (e.g., KV cache sequence length growing), the pre-planned
            // buffer is too small. With aliases, the output shares the input's buffer directly.
            if (st == nullptr) {
                const auto& alias_map = def->Alias();
                if (!alias_map.empty()) {
                    std::vector<int> alias_inputs;
                    std::vector<int> alias_outputs;
                    alias_inputs.reserve(alias_map.size());
                    alias_outputs.reserve(alias_map.size());
                    for (const auto& [input_idx, output_idx] : alias_map) {
                        alias_inputs.push_back(input_idx);
                        alias_outputs.push_back(output_idx);
                    }
                    st = ep_api.KernelDefBuilder_AddInputOutputAliases(
                        builder,
                        alias_inputs.data(),
                        alias_outputs.data(),
                        alias_inputs.size());
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
            // Key format must match what PluginAbiCustomRegistry::RegisterOperatorKernel writes.
            // Uses sinceVersion (not sinceVersion+end) because each opset version that changes
            // a kernel's interface (e.g. Pad, Slice, Clip) is registered as a separate entry.
            std::string regKey{std::string{def->Domain()} + "::" + std::string{def->OpName()} +
                               "::" + std::to_string(since_ver_start)};

            // Check if we have internal registration info for ABI-safe path
            auto reg_info_iter = m_internalRegInfoMap->find(regKey);

            if (reg_info_iter != m_internalRegInfoMap->end()) {
                const auto& reg_info = reg_info_iter->second;

                // Populate ABI-safe kernel creation fields (try this path first)
                func_state->kernel_factory = reg_info->kernelFactory;
                func_state->shape_inferrer = reg_info->shapeInferrer;
                func_state->default_attributes = &reg_info->defaultAttributes;
                // Use reg_info->requiredConstantCpuInputs rather than deriving from KernelDef
                // InputMemoryTypeArgs. The KernelDef conflates device placement flags (e.g.
                // OrtMemTypeCPUInput on MemcpyFromHost input 0) with required constants, causing
                // the safe path to incorrectly treat placement-only inputs as required constants.
                // reg_info->requiredConstantCpuInputs is populated only from the explicit
                // requiredConstantCpuInputs arg in RegisterOperatorKernel, matching the unsafe
                // path's constantCpuInputCapture exactly.
                func_state->required_constant_cpu_inputs = reg_info->requiredConstantCpuInputs;
                func_state->requires_input_shapes_at_creation = reg_info->requiresInputShapesAtCreation;
                func_state->requires_output_shapes_at_creation = reg_info->requiresOutputShapesAtCreation;
                func_state->is_internal_operator = reg_info->isInternalOperator;
                func_state->dml_execution_provider = m_executionProvider.get();
                func_state->ep_plugin = this;
                func_state->ep_name = name_;

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

            st = ep_api.KernelRegistry_AddKernel(kernel_registry_.get(), ort_kernel_def, DmlKernelCreateFuncAdapter, func_state.get());

            if (st != nullptr) {
                return st;
            }

            m_kernelCreateFuncStateByOpName[func_state->operator_name] = func_state.get();
            m_kernelCreateFuncStates.push_back(std::move(func_state));
        }
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
                        constants_to_pass[input_index] = Microsoft::WRL::Make<AbiSafeTensor>(
                            resolved_value, state->ort_api_ptr, state->dml_execution_provider);
                    } else {
                        // Dynamically computed — defer to lazy init at Compute time.
                        required_constants_available = false;
                        break;
                    }
                }

                DmlKernelCreationState creation_state;
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
                creation_state.ep_name = state->ep_name;

                // Pass the resolved constants (or empty map for lazy-init case).
                // DmlAbiKernel_Create sets needs_lazy_init=true when required_constants_available=false,
                // returning a valid deferred kernel. DmlAbiKernel_Compute then fetches dynamically
                // computed constants from the execution context at first call.
                std::unordered_map<uint32_t, Microsoft::WRL::ComPtr<IMLOperatorTensor>> constants_final;
                if (required_constants_available) {
                    constants_final = std::move(constants_to_pass);
                }

                DML_PERF_LOG("[ABI_SAFE] DmlAbiKernel_Create entry: op=", state->operator_name,
                    "  constants_ready=", required_constants_available,
                    "  passing=", constants_final.size(), " constants\n");

                OrtStatus* abi_safe_status = DmlAbiKernel_Create(
                    &creation_state, info, kernel_out, std::move(constants_final));

                if (abi_safe_status == nullptr && *kernel_out != nullptr) {
                    DML_PERF_LOG("[ABI_SAFE] success: op=", state->operator_name, "  (kernel=", (void*)*kernel_out, ")\n");
                    DML_PERF_LOG("[DML_PERF] path=safe  op=", state->operator_name, "\n");
                    return nullptr;
                }

                DML_PERF_LOG("[ABI_SAFE] FAILED: op=", state->operator_name,
                    "  status=", (void*)abi_safe_status, "  kernel=", (void*)*kernel_out, "  -> falling to unsafe\n");
                if (abi_safe_status) {
                    state->ort_api_ptr->ReleaseStatus(abi_safe_status);
                }
                *kernel_out = nullptr;

            } catch (...) {
                *kernel_out = nullptr;
            }
        }

        // FALLBACK: ABI-UNSAFE PATH (when ABI-safe fails or isn't available)
        DML_PERF_LOG("[DML_PERF] path=unsafe op=", state->operator_name,
            "  (safe path absent or failed)\n[ABI_UNSAFE] entry: op=", state->operator_name, "\n");

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
        OrtStatus* st = ep->ep_api.EpGraphSupportInfo_LookUpKernel(graph_support_info, node, &kernel_def);
        if (kernel_def != nullptr) {
            tentativeNodes.push_back(node);
        }
        if (st) ep->ort_api.ReleaseStatus(st);
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

    // -----------------------------------------------------------------------
    // Resolve dynamic output shapes for Upsample/Resize nodes and propagate
    // through downstream ops.  ORT's shape inference gives up when Upsample's
    // scales input is computed at runtime (via Shape→Gather→Cast→Slice→Mul),
    // leaving Upsample's output and all downstream nodes with unknown shapes.
    // Since the scales are constant initializers and input shapes are static,
    // we can compute the output shapes ourselves and propagate them so that
    // ValidateTier0 accepts these nodes for Tier-0 graph fusion.
    //
    // resolved_shapes is declared outside the block so it remains accessible
    // when ValidateTier0 is called below (producer/consumer OrtValueInfo*
    // objects are different instances, so SetDimensions on the producer is not
    // visible to ValidateTier0 reading the consumer; the map is the bridge).
    // -----------------------------------------------------------------------
    std::unordered_map<std::string, std::vector<int64_t>> resolved_shapes;
    bool tier0_claimed = false;
    std::unordered_set<size_t> tier0ClaimedNodeIds;
    {

        // Helper: get the name of an OrtValueInfo.
        auto GetViName = [&](const OrtValueInfo* vi) -> std::string {
            if (!vi) return {};
            const char* n = nullptr;
            ep->ort_api.GetValueInfoName(vi, &n);
            return n ? std::string(n) : std::string{};
        };

        // Helper: get static dims from a node's input/output ValueInfo.
        // Returns empty vector if shape is missing or has any dynamic dim.
        // Falls back to resolved_shapes map.
        auto GetStaticDims = [&](const OrtValueInfo* vi) -> std::vector<int64_t> {
            if (!vi) return {};
            // Check ORT metadata first.
            auto* ti = vi->GetTypeInfo();
            if (ti && ti->tensor_type_info) {
                auto* si = ti->tensor_type_info.get();
                if (si && si->HasShape()) {
                    size_t rank = 0;
                    ep->ort_api.GetDimensionsCount(si, &rank);
                    if (rank > 0) {
                        std::vector<int64_t> dims(rank, -1);
                        ep->ort_api.GetDimensions(si, dims.data(), rank);
                        bool all_static = true;
                        for (auto d : dims) if (d < 0) { all_static = false; break; }
                        if (all_static) return dims;
                    }
                }
            }
            // Fallback: check resolved_shapes by name.
            auto name = GetViName(vi);
            if (!name.empty()) {
                auto it = resolved_shapes.find(name);
                if (it != resolved_shapes.end()) return it->second;
            }
            return {};
        };

        // Helper: check if a ValueInfo has missing or dynamic shape.
        auto HasDynamicShape = [&](const OrtValueInfo* vi) -> bool {
            if (!vi) return true;
            // Check resolved_shapes first.
            auto name = GetViName(vi);
            if (!name.empty() && resolved_shapes.count(name)) return false;
            auto* ti = vi->GetTypeInfo();
            if (!ti || !ti->tensor_type_info) return true;
            auto* si = ti->tensor_type_info.get();
            if (!si || !si->HasShape()) return true;
            size_t rank = 0;
            ep->ort_api.GetDimensionsCount(si, &rank);
            if (rank == 0) return false;
            std::vector<int64_t> dims(rank, -1);
            ep->ort_api.GetDimensions(si, dims.data(), rank);
            for (auto d : dims) if (d < 0) return true;
            return false;
        };

        // Helper: mutate a ValueInfo's shape to resolved static dims.
        auto SetResolvedDims = [&](const OrtValueInfo* vi, const std::vector<int64_t>& dims) {
            if (!vi || dims.empty()) return;
            auto* ti = vi->GetTypeInfo();
            if (ti && ti->tensor_type_info) {
                auto* si = const_cast<OrtTensorTypeAndShapeInfo*>(ti->tensor_type_info.get());
                ep->ort_api.SetDimensions(si, dims.data(), dims.size());
            }
            auto name = GetViName(vi);
            if (!name.empty()) {
                resolved_shapes[name] = dims;
                DML_PERF_LOG("[ShapeResolve] SetResolvedDims: '", name, "' = [");
                for (size_t d = 0; d < dims.size(); ++d) DML_PERF_LOG(d>0?",":"", dims[d]);
                DML_PERF_LOG("]\n");
            }
        };

        // Helper: read float data from a constant initializer.
        auto GetInitializerFloats = [&](const std::string& name, size_t expected_count) -> std::vector<float> {
            auto it = ep->m_graphInitializerMap.find(name);
            if (it == ep->m_graphInitializerMap.end()) return {};
            OrtTensorTypeAndShapeInfo* info = nullptr;
            ep->ort_api.GetTensorTypeAndShape(it->second, &info);
            if (!info) return {};
            size_t count = 0;
            ep->ort_api.GetTensorShapeElementCount(info, &count);
            ep->ort_api.ReleaseTensorTypeAndShapeInfo(info);
            if (count != expected_count) return {};
            void* raw = nullptr;
            OrtStatus* st = ep->ort_api.GetTensorMutableData(const_cast<OrtValue*>(it->second), &raw);
            if (st) { ep->ort_api.ReleaseStatus(st); return {}; }
            if (!raw) return {};
            auto* data = static_cast<const float*>(raw);
            return std::vector<float>(data, data + count);
        };

        // Statically evaluate cpuPreferred chain nodes (Shape→Gather→Cast→
        // Slice→Mul→Cast) to compute Upsample/Resize scales, then resolve
        // output shapes and propagate downstream.
        //
        // resolved_values: tensor name → float values (small tensors, typically 4 elements).
        // Stores computed results for cpuPreferred chain nodes.
        std::unordered_map<std::string, std::vector<float>> resolved_values;

        // Seed resolved_values from graph initializers.
        for (auto& [name, val] : ep->m_graphInitializerMap) {
            OrtTensorTypeAndShapeInfo* info = nullptr;
            ep->ort_api.GetTensorTypeAndShape(val, &info);
            if (!info) continue;
            size_t count = 0;
            ep->ort_api.GetTensorShapeElementCount(info, &count);
            ONNXTensorElementDataType dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            ep->ort_api.GetTensorElementType(info, &dt);
            ep->ort_api.ReleaseTensorTypeAndShapeInfo(info);
            if (count == 0 || count > 64) continue;
            void* raw = nullptr;
            OrtStatus* st = ep->ort_api.GetTensorMutableData(const_cast<OrtValue*>(val), &raw);
            if (st) { ep->ort_api.ReleaseStatus(st); continue; }
            if (!raw) continue;
            std::vector<float> fvals(count);
            if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                auto* p = static_cast<const float*>(raw);
                fvals.assign(p, p + count);
            } else if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                auto* p = static_cast<const int64_t*>(raw);
                for (size_t i = 0; i < count; ++i) fvals[i] = static_cast<float>(p[i]);
            } else if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
                auto* p = static_cast<const int32_t*>(raw);
                for (size_t i = 0; i < count; ++i) fvals[i] = static_cast<float>(p[i]);
            } else if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
                auto* p = static_cast<const uint16_t*>(raw);
                for (size_t i = 0; i < count; ++i) {
                    uint16_t h = p[i];
                    uint32_t sign = (h >> 15) & 1;
                    uint32_t exp = (h >> 10) & 0x1F;
                    uint32_t man = h & 0x3FF;
                    uint32_t f;
                    if (exp == 0) f = (sign << 31) | (man ? ((man << 13) | 0x38000000) : 0);
                    else if (exp == 31) f = (sign << 31) | 0x7F800000 | (man << 13);
                    else f = (sign << 31) | ((exp + 112) << 23) | (man << 13);
                    memcpy(&fvals[i], &f, 4);
                }
            } else if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
                auto* p = static_cast<const double*>(raw);
                for (size_t i = 0; i < count; ++i) fvals[i] = static_cast<float>(p[i]);
            } else {
                continue;
            }
            resolved_values[name] = std::move(fvals);
        }

        // Helper: look up resolved values for a tensor name.
        auto GetResolvedValues = [&](const std::string& name) -> const std::vector<float>* {
            auto it = resolved_values.find(name);
            return it != resolved_values.end() ? &it->second : nullptr;
        };

        // Initial Upsample resolution pass is inside the iterative loop below.

    // -----------------------------------------------------------------------
    // Tier-0 multi-partition graph fusion: walk nodes in topological order and
    // build DML partitions using a merge-based algorithm (mirrors ORT's
    // BuildPartitions). Nodes with no translator or that are CPU-preferred are
    // "boundary" nodes: they finalize upstream open partitions. DML-fuseable
    // nodes extend or start partitions, merging any open upstream partitions
    // their inputs come from.  Each resulting partition is claimed via a
    // separate AddNodesToFuse call, enabling BERT/LLM models that contain
    // Shape/Size/GRU/LSTM/RNN boundary ops to still fuse the rest.
    // -----------------------------------------------------------------------

    // Returns true if the registry has an op translator for this node.
    auto HasTranslator = [](const OpTranslatorRegistry& reg,
                            const OrtApi& ort_api,
                            const OrtNode* node) -> bool {
        const char* op_type = nullptr;
        ort_api.Node_GetOperatorType(node, &op_type);
        return op_type && reg.find(op_type) != reg.end();
    };

    ep->m_tier0GroupHashes.clear();

    if (AllGraphInputsStatic(ep->ort_api, graph)) {
        OpTranslatorRegistry tier0_registry = BuildOpTranslatorRegistry();

        // Build partition groups once — graph topology is fixed.
        struct PartitionGroup {
            bool finalized = false;
            bool claimed = false;
            std::vector<const OrtNode*> nodes;
        };
        std::vector<PartitionGroup> groups;
        std::unordered_map<std::string, size_t> output_to_group;

        auto new_group = [&]() -> size_t {
            groups.push_back(PartitionGroup{});
            return groups.size() - 1;
        };

        for (const OrtNode* node : nodesInTopologicalOrder) {
            size_t nid = 0;
            ep->ort_api.Node_GetId(node, &nid);

            auto input_names  = fusion_utils::GetNodeInputNames(ep->ort_api, node);
            auto output_names = fusion_utils::GetNodeOutputNames(ep->ort_api, node);

            bool is_fuseable = !cpuPreferredNodes.count(nid)
                && ep->IsNodeSupportedByDml(node, graph_support_info, deviceDataTypeMask)
                && HasTranslator(tier0_registry, ep->ort_api, node);

            if (!is_fuseable) {
                bool splits = false;
                for (const auto& name : input_names) {
                    auto it = output_to_group.find(name);
                    if (it != output_to_group.end() && !groups[it->second].finalized) {
                        groups[it->second].finalized = true;
                        splits = true;
                    }
                }
                DML_PERF_LOG("[Tier0] BOUNDARY node=", nid, "\n");
                continue;
            }

            std::vector<size_t> open_upstream;
            for (const auto& name : input_names) {
                auto it = output_to_group.find(name);
                if (it == output_to_group.end()) continue;
                size_t gid = it->second;
                if (groups[gid].finalized) continue;
                if (std::find(open_upstream.begin(), open_upstream.end(), gid) == open_upstream.end())
                    open_upstream.push_back(gid);
            }

            size_t target_gid;
            if (open_upstream.empty()) {
                target_gid = new_group();
            } else {
                target_gid = open_upstream[0];
                for (size_t k = 1; k < open_upstream.size(); ++k) {
                    size_t src = open_upstream[k];
                    for (auto* n : groups[src].nodes)
                        groups[target_gid].nodes.push_back(n);
                    groups[src].nodes.clear();
                    groups[src].finalized = true;
                    for (auto& [name, gid] : output_to_group)
                        if (gid == src) gid = target_gid;
                }
            }

            groups[target_gid].nodes.push_back(node);
            for (const auto& name : output_names)
                output_to_group[name] = target_gid;
        }

        // Iterative resolve-and-claim loop.
        // Each iteration: re-run the Upsample shape resolution pass (cheap —
        // resolved_values is cached), then try to claim unclaimed groups.
        // Successful TryTranslateNodes exports shapes to resolved_shapes,
        // enabling subsequent iterations to resolve more Upsamples.
        for (int iter = 0; iter < 10; ++iter) {
            size_t prev_resolved = resolved_shapes.size();
            size_t prev_claimed = tier0ClaimedNodeIds.size();

            // Re-run Upsample resolution with accumulated resolved_shapes.
            // The inner while(changed) loop and resolved_values cache make
            // this cheap on repeat iterations — only newly-resolvable nodes
            // trigger work.
            {
                bool changed = true;
                int pass = 0;
                while (changed) {
                    changed = false;
                    pass++;
                    for (const OrtNode* node : nodesInTopologicalOrder) {
                        const char* op_type = nullptr;
                        ep->ort_api.Node_GetOperatorType(node, &op_type);
                        if (!op_type) continue;

                        size_t num_outputs = 0;
                        ep->ort_api.Node_GetNumOutputs(node, &num_outputs);
                        if (num_outputs == 0) continue;

                        std::vector<const OrtValueInfo*> out_vis(num_outputs, nullptr);
                        ep->ort_api.Node_GetOutputs(node, out_vis.data(), num_outputs);

                        size_t num_inputs = 0;
                        ep->ort_api.Node_GetNumInputs(node, &num_inputs);
                        std::vector<const OrtValueInfo*> in_vis(num_inputs, nullptr);
                        if (num_inputs > 0)
                            ep->ort_api.Node_GetInputs(node, in_vis.data(), num_inputs);

                        auto input_names = fusion_utils::GetNodeInputNames(ep->ort_api, node);
                        auto output_names = fusion_utils::GetNodeOutputNames(ep->ort_api, node);

                        for (size_t i = 0; i < input_names.size() && i < num_inputs; ++i) {
                            if (input_names[i].empty() && in_vis[i]) {
                                const char* vi_name = nullptr;
                                ep->ort_api.GetValueInfoName(in_vis[i], &vi_name);
                                if (vi_name) input_names[i] = vi_name;
                            }
                        }
                        for (size_t i = 0; i < output_names.size() && i < num_outputs; ++i) {
                            if (output_names[i].empty() && out_vis[i]) {
                                const char* vi_name = nullptr;
                                ep->ort_api.GetValueInfoName(out_vis[i], &vi_name);
                                if (vi_name) output_names[i] = vi_name;
                            }
                        }

                        size_t nid = 0;
                        ep->ort_api.Node_GetId(node, &nid);

                        // Evaluate chain nodes.
                        {
                        bool should_evaluate = false;
                        static const std::unordered_set<std::string> chain_ops = {
                            "Cast", "Concat", "Div", "Mul", "Gather", "Slice", "Shape",
                        };
                        bool is_chain_op = chain_ops.count(op_type) != 0;

                        if (!output_names.empty() && !resolved_values.count(output_names[0])) {
                            bool is_shape_op = std::strcmp(op_type, "Shape") == 0;
                            should_evaluate = cpuPreferredNodes.count(nid) || is_shape_op;
                            if (!should_evaluate && num_inputs > 0 && is_chain_op) {
                                bool all_inputs_resolved = true;
                                for (auto& nm : input_names) {
                                    if (nm.empty()) continue;
                                    if (!resolved_values.count(nm)) { all_inputs_resolved = false; break; }
                                }
                                if (all_inputs_resolved) should_evaluate = true;
                            }
                        }

                        if (should_evaluate) {
                            if (std::strcmp(op_type, "Shape") == 0 && num_inputs > 0) {
                                auto in_name = GetViName(in_vis[0]);
                                auto dims = GetStaticDims(in_vis[0]);
                                DML_PERF_LOG("[ShapeResolve] iter=", iter, " Shape: in='", in_name,
                                    "' out='", output_names[0], "' dims=", (dims.empty() ? "EMPTY" : "OK"),
                                    " resolved_shapes_has=", resolved_shapes.count(in_name), "\n");
                                if (!dims.empty()) {
                                    std::vector<float> fvals(dims.size());
                                    for (size_t i = 0; i < dims.size(); ++i) fvals[i] = static_cast<float>(dims[i]);
                                    resolved_values[output_names[0]] = fvals;
                                    changed = true;
                                }
                            } else if (std::strcmp(op_type, "Gather") == 0 && input_names.size() >= 2) {
                                auto* data = GetResolvedValues(input_names[0]);
                                auto* indices = GetResolvedValues(input_names[1]);
                                if (data && indices) {
                                    std::vector<float> out;
                                    for (float idx : *indices) {
                                        int i = static_cast<int>(idx);
                                        if (i < 0) i += static_cast<int>(data->size());
                                        if (i >= 0 && i < static_cast<int>(data->size())) out.push_back((*data)[i]);
                                    }
                                    if (!out.empty()) { resolved_values[output_names[0]] = out; changed = true;}
                                }
                            } else if (std::strcmp(op_type, "Cast") == 0 && input_names.size() >= 1) {
                                auto* vals = GetResolvedValues(input_names[0]);
                                if (vals) { resolved_values[output_names[0]] = *vals; changed = true;}
                            } else if (std::strcmp(op_type, "Slice") == 0 && !input_names.empty()) {
                                auto* data = GetResolvedValues(input_names[0]);
                                const std::vector<float>* starts = nullptr;
                                const std::vector<float>* ends = nullptr;
                                if (input_names.size() >= 3) {
                                    starts = GetResolvedValues(input_names[1]);
                                    ends = GetResolvedValues(input_names[2]);
                                }
                                // Opset 1 Slice: starts/ends are attributes.
                                std::vector<float> attr_starts_v, attr_ends_v;
                                if (!starts || !ends) {
                                    OrtNodeAdapter adapter(node, ep->ort_api);
                                    auto s_attr = adapter.GetAttributeInts("starts");
                                    auto e_attr = adapter.GetAttributeInts("ends");
                                    if (!s_attr.empty() && !e_attr.empty()) {
                                        for (auto v : s_attr) attr_starts_v.push_back(static_cast<float>(v));
                                        for (auto v : e_attr) attr_ends_v.push_back(static_cast<float>(v));
                                        starts = &attr_starts_v;
                                        ends = &attr_ends_v;
                                    }
                                }
                                if (data && starts && ends && !starts->empty() && !ends->empty()) {
                                    int s = static_cast<int>((*starts)[0]);
                                    int e = static_cast<int>((*ends)[0]);
                                    if (s < 0) s += static_cast<int>(data->size());
                                    if (e < 0) e += static_cast<int>(data->size());
                                    if (e > static_cast<int>(data->size())) e = static_cast<int>(data->size());
                                    std::vector<float> out;
                                    for (int i = s; i < e; ++i) out.push_back((*data)[i]);
                                    if (!out.empty()) { resolved_values[output_names[0]] = out; changed = true;}
                                }
                            } else if (std::strcmp(op_type, "Mul") == 0 && input_names.size() >= 2) {
                                auto* a = GetResolvedValues(input_names[0]);
                                auto* b = GetResolvedValues(input_names[1]);
                                if (a && b) {
                                    size_t len = std::max(a->size(), b->size());
                                    std::vector<float> out(len);
                                    for (size_t i = 0; i < len; ++i)
                                        out[i] = (*a)[i % a->size()] * (*b)[i % b->size()];
                                    resolved_values[output_names[0]] = out;
                                    changed = true;
                                }
                            } else if (std::strcmp(op_type, "Div") == 0 && input_names.size() >= 2) {
                                auto* a = GetResolvedValues(input_names[0]);
                                auto* b = GetResolvedValues(input_names[1]);
                                if (a && b) {
                                    size_t len = std::max(a->size(), b->size());
                                    std::vector<float> out(len);
                                    for (size_t i = 0; i < len; ++i) {
                                        float bv = (*b)[i % b->size()];
                                        out[i] = (bv != 0.0f) ? (*a)[i % a->size()] / bv : 0.0f;
                                    }
                                    resolved_values[output_names[0]] = out;
                                    changed = true;
                                }
                            } else if (std::strcmp(op_type, "Concat") == 0) {
                                std::vector<float> out;
                                bool all_resolved = true;
                                for (auto& nm : input_names) {
                                    auto* v = GetResolvedValues(nm);
                                    if (!v) { all_resolved = false; break; }
                                    out.insert(out.end(), v->begin(), v->end());
                                }
                                if (all_resolved && !out.empty()) {
                                    resolved_values[output_names[0]] = out;
                                    changed = true;
                                }
                            }
                            continue;
                        }
                        }

                        // Resolve Upsample/Resize output shapes.
                        if (!HasDynamicShape(out_vis[0])) continue;

                        bool is_upsample = std::strcmp(op_type, "Upsample") == 0;
                        bool is_resize = std::strcmp(op_type, "Resize") == 0;
                        if (!is_upsample && !is_resize) continue;

                        auto in_dims = GetStaticDims(in_vis[0]);
                        if (in_dims.empty()) continue;
                        size_t rank = in_dims.size();

                        std::vector<float> scales;
                        std::vector<int64_t> out_dims;

                        size_t scales_idx = (is_upsample || num_inputs == 2) ? 1 : 2;
                        size_t sizes_idx = (is_resize && num_inputs >= 4) ? 3 : SIZE_MAX;

                        if (is_upsample && num_inputs == 1) {
                            OrtNodeAdapter adapter(node, ep->ort_api);
                            scales = adapter.GetAttributeFloats("scales");
                        }

                        if (scales.empty() && out_dims.empty()) {
                            if (sizes_idx < input_names.size() && !input_names[sizes_idx].empty()) {
                                auto s = GetInitializerFloats(input_names[sizes_idx], rank);
                                if (!s.empty()) {
                                    out_dims.resize(rank);
                                    for (size_t d = 0; d < rank; ++d) out_dims[d] = static_cast<int64_t>(s[d]);
                                }
                            }
                            if (out_dims.empty() && scales_idx < input_names.size() && !input_names[scales_idx].empty()) {
                                scales = GetInitializerFloats(input_names[scales_idx], rank);
                                if (scales.empty()) {
                                    auto* rv = GetResolvedValues(input_names[scales_idx]);
                                    if (rv && rv->size() == rank) scales = *rv;
                                }
                            }
                        }

                        // Backward walk for scales (same as before).
                        if (scales.empty() && out_dims.empty() && scales_idx < num_inputs && in_vis[scales_idx]) {
                            struct PendingEval { const OrtNode* node; std::vector<const OrtValueInfo*> inputs; std::string out_name; };
                            std::vector<PendingEval> eval_stack;
                            const OrtValueInfo* cur_vi = in_vis[scales_idx];
                            for (int hop = 0; hop < 5 && cur_vi; ++hop) {
                                const char* cur_name = nullptr;
                                ep->ort_api.GetValueInfoName(cur_vi, &cur_name);
                                if (cur_name && resolved_values.count(cur_name)) break;
                                const OrtNode* producer = nullptr;
                                for (const OrtNode* n : nodesInTopologicalOrder) {
                                    size_t n_out = 0;
                                    ep->ort_api.Node_GetNumOutputs(n, &n_out);
                                    std::vector<const OrtValueInfo*> n_out_vis(n_out, nullptr);
                                    if (n_out > 0) ep->ort_api.Node_GetOutputs(n, n_out_vis.data(), n_out);
                                    for (size_t o = 0; o < n_out; ++o) {
                                        if (n_out_vis[o] == cur_vi) { producer = n; break; }
                                    }
                                    if (producer) break;
                                }
                                if (!producer) break;
                                const char* prod_op = nullptr;
                                ep->ort_api.Node_GetOperatorType(producer, &prod_op);
                                size_t p_nin = 0;
                                ep->ort_api.Node_GetNumInputs(producer, &p_nin);
                                std::vector<const OrtValueInfo*> p_in_vis(p_nin, nullptr);
                                if (p_nin > 0) ep->ort_api.Node_GetInputs(producer, p_in_vis.data(), p_nin);
                                eval_stack.push_back({producer, p_in_vis, cur_name ? std::string(cur_name) : ""});
                                cur_vi = (p_nin > 0) ? p_in_vis[0] : nullptr;
                            }
                            for (int i = static_cast<int>(eval_stack.size()) - 1; i >= 0; --i) {
                                auto& pe = eval_stack[i];
                                const char* pe_op = nullptr;
                                ep->ort_api.Node_GetOperatorType(pe.node, &pe_op);
                                if (!pe_op) continue;
                                std::vector<std::string> pe_input_names;
                                for (auto* vi : pe.inputs) {
                                    const char* vn = nullptr;
                                    if (vi) ep->ort_api.GetValueInfoName(vi, &vn);
                                    pe_input_names.push_back(vn ? std::string(vn) : "");
                                }
                                if (std::strcmp(pe_op, "Cast") == 0 && !pe_input_names.empty()) {
                                    auto* v = GetResolvedValues(pe_input_names[0]);
                                    if (v) { resolved_values[pe.out_name] = *v; changed = true; }
                                } else if (std::strcmp(pe_op, "Concat") == 0) {
                                    std::vector<float> cat;
                                    bool ok = true;
                                    for (auto& nm : pe_input_names) {
                                        auto* v = GetResolvedValues(nm);
                                        if (!v) { ok = false; break; }
                                        cat.insert(cat.end(), v->begin(), v->end());
                                    }
                                    if (ok && !cat.empty()) { resolved_values[pe.out_name] = cat; changed = true; }
                                } else if (std::strcmp(pe_op, "Div") == 0 && pe_input_names.size() >= 2) {
                                    auto* a = GetResolvedValues(pe_input_names[0]);
                                    auto* b = GetResolvedValues(pe_input_names[1]);
                                    if (a && b) {
                                        size_t len = std::max(a->size(), b->size());
                                        std::vector<float> out(len);
                                        for (size_t d = 0; d < len; ++d) {
                                            float bv = (*b)[d % b->size()];
                                            out[d] = (bv != 0.0f) ? (*a)[d % a->size()] / bv : 0.0f;
                                        }
                                        resolved_values[pe.out_name] = out;
                                        changed = true;
                                    }
                                }
                            }
                            const char* scales_name = nullptr;
                            ep->ort_api.GetValueInfoName(in_vis[scales_idx], &scales_name);
                            if (scales_name) {
                                auto* rv = GetResolvedValues(scales_name);
                                if (rv && rv->size() == rank) scales = *rv;
                            }
                        }

                        if (scales.empty() && out_dims.empty() && is_resize && num_inputs >= 3 &&
                            input_names.size() > 2 && !input_names[2].empty()) {
                            scales = GetInitializerFloats(input_names[2], rank);
                            if (scales.empty()) {
                                auto* rv = GetResolvedValues(input_names[2]);
                                if (rv && rv->size() == rank) scales = *rv;
                            }
                        }

                        if (out_dims.empty() && !scales.empty() && scales.size() == rank) {
                            out_dims.resize(rank);
                            for (size_t d = 0; d < rank; ++d)
                                out_dims[d] = static_cast<int64_t>(in_dims[d] * scales[d]);
                        }

                        if (!out_dims.empty()) {
                            DML_PERF_LOG("[ShapeResolve] iter=", iter, " ", op_type, ": resolved output dims=[");
                            for (size_t d = 0; d < out_dims.size(); ++d)
                                DML_PERF_LOG(d > 0 ? "," : "", out_dims[d]);
                            DML_PERF_LOG("]\n");
                            SetResolvedDims(out_vis[0], out_dims);
                            if (!output_names.empty() && !output_names[0].empty()) {
                                auto upsample_out_name = GetViName(out_vis[0]);
                                if (output_names[0] != upsample_out_name)
                                    resolved_shapes[output_names[0]] = out_dims;
                            }
                            // Propagate through ORT-inserted output Cast.
                            for (const OrtNode* consumer : nodesInTopologicalOrder) {
                                const char* c_op = nullptr;
                                ep->ort_api.Node_GetOperatorType(consumer, &c_op);
                                if (!c_op || std::strcmp(c_op, "Cast") != 0) continue;
                                auto c_inputs = fusion_utils::GetNodeInputNames(ep->ort_api, consumer);
                                if (c_inputs.empty()) continue;
                                auto upsample_out_name = GetViName(out_vis[0]);
                                bool matches = (c_inputs[0] == upsample_out_name);
                                if (!matches && !output_names.empty()) matches = (c_inputs[0] == output_names[0]);
                                if (!matches) {
                                    size_t c_nin = 0;
                                    ep->ort_api.Node_GetNumInputs(consumer, &c_nin);
                                    if (c_nin > 0) {
                                        std::vector<const OrtValueInfo*> c_in(c_nin, nullptr);
                                        ep->ort_api.Node_GetInputs(consumer, c_in.data(), c_nin);
                                        if (GetViName(c_in[0]) == upsample_out_name) matches = true;
                                    }
                                }
                                if (!matches) continue;
                                auto c_outputs = fusion_utils::GetNodeOutputNames(ep->ort_api, consumer);
                                size_t c_nout = 0;
                                ep->ort_api.Node_GetNumOutputs(consumer, &c_nout);
                                std::vector<const OrtValueInfo*> c_out_vis(c_nout, nullptr);
                                if (c_nout > 0) ep->ort_api.Node_GetOutputs(consumer, c_out_vis.data(), c_nout);
                                for (size_t co = 0; co < c_nout && co < c_outputs.size(); ++co) {
                                    if (!c_outputs[co].empty()) resolved_shapes[c_outputs[co]] = out_dims;
                                    auto co_vi_name = GetViName(c_out_vis[co]);
                                    if (!co_vi_name.empty() && co_vi_name != c_outputs[co])
                                        resolved_shapes[co_vi_name] = out_dims;
                                }
                                break;
                            }
                            changed = true;
                        }
                    }
                }
                DML_PERF_LOG("[ShapeResolve] iter=", iter, " completed: resolved_values=",
                    resolved_values.size(), " resolved_shapes=", resolved_shapes.size(), "\n");
            }

            // Try to claim unclaimed groups.
            for (auto& g : groups) {
                if (g.nodes.empty() || g.claimed) continue;

                std::string group_ops;
                for (const OrtNode* n : g.nodes) {
                    const char* ot = nullptr;
                    ep->ort_api.Node_GetOperatorType(n, &ot);
                    if (ot) { group_ops += ot; group_ops += " "; }
                }
                DML_PERF_LOG("[Tier0] iter=", iter, " candidate group (", g.nodes.size(), " nodes): ", group_ops, "\n");

                if (!FullGraphFusion::ValidateTier0(ep->ort_api, g.nodes, resolved_shapes)) {
                    DML_PERF_LOG("[Tier0] SKIP: ValidateTier0 failed\n");
                    continue;
                }
                if (!FullGraphFusion::TryTranslateNodes(ep->ort_api, graph,
                                                        ep->m_graphInitializerMap, g.nodes, resolved_shapes, &resolved_shapes)) {
                    DML_PERF_LOG("[Tier0] SKIP: TryTranslateNodes failed\n");
                    continue;
                }
                if (!FullGraphFusion::TryCompilePartition(ep->ort_api, graph,
                                                          ep->m_graphInitializerMap,
                                                          ep->m_executionProvider.get(), g.nodes, resolved_shapes)) {
                    DML_PERF_LOG("[Tier0] SKIP: TryCompilePartition failed\n");
                    continue;
                }
                DML_PERF_LOG("[Tier0] CLAIM: group passed pre-flight (", g.nodes.size(), " nodes)\n");

                OrtNodeFusionOptions fusion_options{ORT_API_VERSION, true};
                OrtStatus* st = ep->ep_api.EpGraphSupportInfo_AddNodesToFuse(
                    graph_support_info,
                    g.nodes.data(),
                    g.nodes.size(),
                    &fusion_options);
                if (!st) {
                    std::vector<size_t> node_ids;
                    node_ids.reserve(g.nodes.size());
                    for (const OrtNode* n : g.nodes) {
                        size_t nid = 0;
                        ep->ort_api.Node_GetId(n, &nid);
                        node_ids.push_back(nid);
                        tier0ClaimedNodeIds.insert(nid);
                    }
                    std::sort(node_ids.begin(), node_ids.end());
                    ep->m_tier0GroupHashes.insert(HashNodeIds(node_ids));
                    tier0_claimed = true;
                    g.claimed = true;
                }
                if (st) ep->ort_api.ReleaseStatus(st);
            }

            // Stop if no progress.
            bool made_progress = resolved_shapes.size() > prev_resolved
                || tier0ClaimedNodeIds.size() > prev_claimed;
            DML_PERF_LOG("[Tier0] iter=", iter, " resolved_shapes=", resolved_shapes.size(),
                " claimed=", tier0ClaimedNodeIds.size(), " progress=", made_progress, "\n");
            if (!made_progress) break;
        }

    } // end if (AllGraphInputsStatic)
    } // end scope for helpers (GetViName, GetStaticDims, etc.)

    // Update m_resolvedShapes with shapes accumulated from successful
    // TryTranslateNodes calls so CompileImpl can seed BuildSubgraphInfo.
    ep->m_resolvedShapes = resolved_shapes;

    if (!tier0_claimed) {
        // No Tier-0 partitions were claimed (dynamic shapes, or all candidate
        // groups failed ValidateTier0).  Fall through to Tier-2 pattern fusion
        // + Tier-1 single-node claiming.

        // Run graph-level fusions in a single greedy pass (largest pattern wins).
        // Each matched fusion group is submitted to ORT via AddNodesToFuse and
        // routed through CompileImpl().  Claimed node IDs are collected so that
        // the single-node loop below can skip them.
        // Build the anchor index once per session and store it — the IFusionRule
        // objects it owns are referenced by raw pointer in FusionMatch and must
        // outlive m_fusionMap.
        ep->m_anchorIndex = EpFusionManager::BuildAnchorIndex(ep->m_executionProvider->IsMcdmDevice());
        ep->m_fusionMap.clear();

        std::unordered_set<size_t> fusedNodeIds;
        EpFusionManager::ApplyFusions(
            ep->m_anchorIndex,
            ep->ort_api,
            ep->ep_api,
            graph,
            graph_support_info,
            ep->m_graphInitializerMap,
            [ep, graph_support_info](const OrtNode* n, uint32_t mask) {
                return ep->IsNodeSupportedByDml(n, graph_support_info, mask);
            },
            cpuPreferredNodes,
            deviceDataTypeMask,
            fusedNodeIds,
            ep->m_fusionMap);

        std::vector<const OrtNode*> supportedNodes;
        for (const OrtNode* node : nodesInTopologicalOrder) {
            size_t nodeID = 0;
            ep->ort_api.Node_GetId(node, &nodeID);

            // Skip nodes already claimed by a fusion group.
            if (fusedNodeIds.count(nodeID)) continue;

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
    } else {
        // Tier-0 claimed at least one partition. Claim any remaining unclaimed
        // nodes via Tier-2 pattern fusions, then Tier-1 single-node.
        //
        // Tier-2 ApplyFusions scans ALL graph nodes, not just unclaimed ones.
        // Without an exclusion set it will match patterns (e.g. QuickGelu) on
        // nodes already inside a Tier-0 partition, creating overlapping fused
        // groups that crash at inference time. Exclude Tier-0 claimed nodes
        // so ApplyFusions only considers genuinely unclaimed nodes.
        std::unordered_set<size_t> tier0AndCpuExcluded = cpuPreferredNodes;
        tier0AndCpuExcluded.insert(tier0ClaimedNodeIds.begin(),
                                   tier0ClaimedNodeIds.end());

        ep->m_anchorIndex = EpFusionManager::BuildAnchorIndex(ep->m_executionProvider->IsMcdmDevice());
        ep->m_fusionMap.clear();

        std::unordered_set<size_t> fusedNodeIds;
        EpFusionManager::ApplyFusions(
            ep->m_anchorIndex,
            ep->ort_api,
            ep->ep_api,
            graph,
            graph_support_info,
            ep->m_graphInitializerMap,
            [ep, graph_support_info](const OrtNode* n, uint32_t mask) {
                return ep->IsNodeSupportedByDml(n, graph_support_info, mask);
            },
            tier0AndCpuExcluded,
            deviceDataTypeMask,
            fusedNodeIds,
            ep->m_fusionMap);

        for (const OrtNode* node : nodesInTopologicalOrder) {
            size_t nodeID = 0;
            ep->ort_api.Node_GetId(node, &nodeID);
            if (tier0ClaimedNodeIds.count(nodeID)) continue;
            if (fusedNodeIds.count(nodeID)) continue;
            if (cpuPreferredNodes.count(nodeID)) continue;
            if (!ep->IsNodeSupportedByDml(node, graph_support_info, deviceDataTypeMask)) continue;
            ep->ep_api.EpGraphSupportInfo_AddSingleNode(graph_support_info, node);
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

    for (size_t i = 0; i < count; ++i) {
        node_compute_infos[i] = nullptr;
        ep_context_nodes[i]   = nullptr;

        // Tier-0 route: if GetCapabilityImpl claimed one or more fused partitions,
        // check if this subgraph's hash matches any of them and compile via
        // FullGraphFusion.  Multiple partitions are possible when the graph has
        // boundary nodes (Shape, Size, GRU, LSTM, RNN, Memcpy*).
        if (!ep->m_tier0GroupHashes.empty()) {
            size_t num_nodes = 0;
            ep->ort_api.Graph_GetNumNodes(graphs[i], &num_nodes);
            std::vector<const OrtNode*> sg_nodes(num_nodes, nullptr);
            if (num_nodes > 0)
                ep->ort_api.Graph_GetNodes(graphs[i], sg_nodes.data(), num_nodes);
            std::vector<size_t> sg_ids;
            sg_ids.reserve(num_nodes);
            for (const OrtNode* n : sg_nodes) {
                if (n) sg_ids.push_back(fusion_utils::GetNodeId(ep->ort_api, n));
            }
            std::sort(sg_ids.begin(), sg_ids.end());
            size_t sg_hash = HashNodeIds(sg_ids);

            if (ep->m_tier0GroupHashes.count(sg_hash)) {
                DML_PERF_LOG("[CompileImpl] Tier0 compile: subgraph hash match, nodes=", num_nodes, "\n");
                node_compute_infos[i] = FullGraphFusion::Compile(
                    ep->ort_api,
                    graphs[i],
                    ep->m_graphInitializerMap,
                    ep->m_executionProvider.get(),
                    ep->m_resolvedShapes);
                if (node_compute_infos[i]) {
                    DML_PERF_LOG("[CompileImpl] Tier0 compile: SUCCESS\n");
                    continue;
                }
                DML_PERF_LOG("[CompileImpl] Tier0 compile: FAILED (nullptr) — falling through to Tier-2\n");
            }
        }

        // Tier-2 route: pattern-matched fusions.
        node_compute_infos[i] = EpFusionManager::CompileFusion(
            ep->ort_api,
            graphs[i],
            ep->m_graphInitializerMap,
            ep->m_executionProvider.get(),
            ep->m_fusionMap);

        if (!node_compute_infos[i]) {
            // CompileFusion returned null — the matched pattern's Compile failed
            // (e.g. FusedMatMul with no graph inputs in Mode B). Log for diagnosis.
            size_t nm = 0;
            ep->ort_api.Graph_GetNumNodes(graphs[i], &nm);
            std::vector<const OrtNode*> sg_nodes(nm, nullptr);
            if (nm > 0) ep->ort_api.Graph_GetNodes(graphs[i], sg_nodes.data(), nm);
            std::string diag = "CompileImpl: no fusion rule matched (";
            diag += std::to_string(nm) + " nodes: ";
            for (const OrtNode* n : sg_nodes) {
                if (n == nullptr) continue;
                const char* op = nullptr;
                ep->ort_api.Node_GetOperatorType(n, &op);
                size_t nid = 0;
                ep->ort_api.Node_GetId(n, &nid);
                if (op != nullptr) { diag += op; diag += "["; diag += std::to_string(nid); diag += "] "; }
            }
            diag += ")";
            // Return ORT_EP_FAIL so ORT can route this subgraph elsewhere.
            // This is a non-fatal failure — ORT will try CPU or another EP.
            return ep->ort_api.CreateStatus(ORT_EP_FAIL, diag.c_str());
        }
    }

    return nullptr;
}

void ORT_API_CALL ExecutionProviderPlugin::ReleaseNodeComputeInfosImpl(
    OrtEp* this_ptr,
    OrtNodeComputeInfo** node_compute_infos,
    size_t num_node_compute_infos) noexcept
{
    (void)this_ptr;
    for (size_t i = 0; i < num_node_compute_infos; ++i) {
        // Each OrtNodeComputeInfo is a heap-allocated QuickGeluNodeComputeInfo.
        // The destructor releases the compiled state (IDMLCompiledOperator, etc.).
        delete node_compute_infos[i];
        node_compute_infos[i] = nullptr;
    }
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

    *kernel_registry = ep->kernel_registry_.get();
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
    if (memory_info == nullptr) return false;
    OrtMemoryInfoDeviceType device_type;
    ort_api.MemoryInfoGetDeviceType(memory_info, &device_type);
    return device_type == OrtMemoryInfoDeviceType_CPU;
}

bool ExecutionProviderPlugin::IsGpuAllocator(const OrtMemoryInfo* memory_info)
{
    if (memory_info == nullptr) return false;
    OrtMemoryInfoDeviceType device_type;
    ort_api.MemoryInfoGetDeviceType(memory_info, &device_type);
    return device_type == OrtMemoryInfoDeviceType_GPU;
}

uint32_t ExecutionProviderPlugin::GetSupportedDeviceDataTypeMask() const {
    return dml_ep::GetSupportedDeviceDataTypeMask(m_dmlDevice.Get());
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

    // Second seed pass: if a node's output is consumed as a required CPU constant
    // by a downstream EP node (e.g. Reshape's shape input[1], Expand's shape input[1]),
    // that output is a CPU tensor and the producing node should be a CPU candidate.
    // Uses reg_info->requiredConstantCpuInputs rather than KernelDef_GetInputMemType
    // because DML kernels declare required constants through requiredConstantCpuInputs
    // in RegisterOperatorKernel, not via InputMemoryType in the kernel def.
    for (auto& [consumer_id, kernel_def] : node_to_kernel) {
        const OrtNode* consumer_node = id_to_node.count(consumer_id) ? id_to_node[consumer_id] : nullptr;
        if (!consumer_node || !kernel_def) continue;

        // Look up reg_info to get requiredConstantCpuInputs for this kernel.
        const char* op_name = ep_api.KernelDef_GetOperatorType(kernel_def);
        const char* domain  = ep_api.KernelDef_GetDomain(kernel_def);
        int since_ver = 0, until_ver = 0;
        OrtStatus* ver_st = ep_api.KernelDef_GetSinceVersion(kernel_def, &since_ver, &until_ver);
        if (ver_st) { ort_api.ReleaseStatus(ver_st); continue; }
        if (!op_name || !domain) continue;
        std::string regKey = std::string(domain) + "::" + std::string(op_name) +
                             "::" + std::to_string(since_ver);
        auto reg_it = m_internalRegInfoMap->find(regKey);
        if (reg_it == m_internalRegInfoMap->end() || !reg_it->second) continue;
        const auto& required_cpu_inputs = reg_it->second->requiredConstantCpuInputs;
        if (required_cpu_inputs.empty()) continue;

        size_t nodeNumInputs = 0;
        ort_api.Node_GetNumInputs(consumer_node, &nodeNumInputs);
        std::vector<const OrtValueInfo*> inputInfos(nodeNumInputs, nullptr);
        if (nodeNumInputs > 0)
            ort_api.Node_GetInputs(consumer_node, inputInfos.data(), nodeNumInputs);

        for (uint32_t cpu_idx : required_cpu_inputs) {
            if (cpu_idx >= nodeNumInputs || !inputInfos[cpu_idx]) continue;

            // This input is a required CPU constant — seed it and trace back to its producer.
            cpu_output_args.insert(inputInfos[cpu_idx]);

            const OrtNode* producer_node = nullptr;
            size_t producer_output_idx = 0;
            OrtStatus* ps = ort_api.ValueInfo_GetValueProducer(
                inputInfos[cpu_idx], &producer_node, &producer_output_idx);
            if (ps) { ort_api.ReleaseStatus(ps); continue; }
            if (!producer_node) continue;
            size_t producer_id = 0;
            ort_api.Node_GetId(producer_node, &producer_id);
            if (provider_nodes.count(producer_id))
                candidates.push(producer_id);
        }
    }

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

            // Null input = missing optional edge — treat as GPU tensor (don't pull to CPU).
            // Matches ORT's fallback_cpu_capability.cc which skips null NodeArgs naturally.
            if (input == nullptr || input->GetTypeInfo() == nullptr
                || input->GetTypeInfo()->tensor_type_info == nullptr
                || input->GetTypeInfo()->tensor_type_info.get() == nullptr) {
                place_in_cpu = false;
                break;
            }

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
            if (IsSmallInitializer(graph, input) ||
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
    // Compare against the runtime EP name, not a hardcoded string, so kernel lookup works
    // regardless of the name this EP was registered under (e.g. "directml" via amdgpu-ep).
    if (provider == nullptr || std::strcmp(provider, name_.c_str()) != 0)
    {
        // Not a DML kernel -> won't be in internal map
        return false;
    }

    // Build key from domain, operator name, and since-version — must match the format written
    // by ConvertKernelRegistryToOrtKernelRegistry: domain::opname::since_version.
    const char* op_name = ep_api.KernelDef_GetOperatorType(kernel_def);
    const char* domain = ep_api.KernelDef_GetDomain(kernel_def);
    int since_ver = 0, until_ver = 0;
    OrtStatus* ver_st = ep_api.KernelDef_GetSinceVersion(kernel_def, &since_ver, &until_ver);
    if (ver_st) { ort_api.ReleaseStatus(ver_st); since_ver = 0; }
    std::string regKey = std::string(domain) + "::" + std::string(op_name) +
                         "::" + std::to_string(since_ver);

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
    // Pass internalRegInfo so that requiredConstantCpuInputs are excluded from GPU type validation —
    // those inputs (e.g. Resize's sizes/scales/roi) are read on CPU and never need GPU type support.
    // Passing nullptr here was a bug: it caused ops like Resize to be rejected because their
    // int64 CPU-side inputs failed the device type check, matching ORT's ExecutionProvider.cpp:876.
    bool dataTypesSupported = DoesNodeContainSupportedDataTypes(node, internalRegInfo.get(), supportedDeviceDataTypeMask,
                                           m_native16BitShaderOpsSupported);
    if (!dataTypesSupported) {
        return false;
    }

    return true;
}

bool ExecutionProviderPlugin::DoesNodeContainSupportedDataTypes(
    const OrtNode* node,
    const InternalRegistrationInfo* regInfo,
    uint32_t supportedDeviceDataTypeMask, // Each bit corresponds to each DML_TENSOR_DATA_TYPE.
    bool native16BitShaderOpsSupported)
{
    const char* op_type = nullptr;
    ort_api.Node_GetOperatorType(node, &op_type);

    OrtNodeAdapter adapter(node, ort_api);
    std::vector<const OrtValueInfoAdapter*> constantCpuInputs;

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
    auto nodeCallback = [&](const OrtValueInfoAdapter& valueInfo, bool isInput) -> void {
        // Get the tensor element data type for this node, comparing against what the device actually supports.


        // Reject node if undefined data type or non-tensor, as DML cannot handle it.
        MLOperatorEdgeType edgeType;
        MLOperatorTensorDataType onnxElementType;

        // Skip missing optional inputs/outputs — these have no type info.
        // ORT's ForEachDef skips them via NodeArg::Exists(); replicate that here.
        if (!valueInfo.GetTypeInfo()) {
            return;
        }

        // Check type using OrtValueInfoAdapter
        if (!valueInfo.IsTensor()) {
            // If the model has nodes that use Optional we will arrive here. It's a valid ONNX model but
            // we don't handle Optional.
            // Exception: if this is a required CPU constant input that is unconnected (missing optional),
            // skip it rather than rejecting the node.
            bool isMissingConstantCpuInput = isInput &&
                std::find(constantCpuInputs.begin(), constantCpuInputs.end(), &valueInfo) != constantCpuInputs.end();
            if (!isMissingConstantCpuInput) {
                nodeContainsSupportedDataTypes = false;
            }
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

        // Succeed if the tensor is CPU-bound, as the CPU-side reading code is generic enough
        // to handle multiple types regardless of GPU capability (typically these are just
        // scalars or simple 1D arrays like GQA's int64 total_sequence_length).
        bool isConstantCpuInput = isInput &&
            std::find(constantCpuInputs.begin(), constantCpuInputs.end(), &valueInfo) != constantCpuInputs.end();
        if (isConstantCpuInput) {
            return;
        }

        // Reject node for unknown DML data types.
        DML_TENSOR_DATA_TYPE dmlElementType = GetDmlDataTypeFromMlDataTypeNoThrow(onnxElementType);
        if (dmlElementType == DML_TENSOR_DATA_TYPE_UNKNOWN) {
            nodeContainsSupportedDataTypes = false;
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

bool ExecutionProviderPlugin::IsCustomOpShader(const OrtNodeAdapter& adapter) {
    auto custom_ops = std::array<const char*, 3>{"DFT", "STFT", "GridSample"};

    for (auto& custom_op : custom_ops) {
        if (strcmp(custom_op, adapter.GetOpType().c_str()) == 0) {
            return true;
        }
    }
    return false;
}

bool ExecutionProviderPlugin::IsCpuOnDmlOperator(const OrtNodeAdapter& adapter) {
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

bool ExecutionProviderPlugin::IsDmlSequenceOperator(const OrtNodeAdapter& adapter) {
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

std::shared_ptr<PluginDmlExecutionProviderImpl> ExecutionProviderPlugin::GetInternalExecutionProvider() {
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

}  // namespace dml_ep
