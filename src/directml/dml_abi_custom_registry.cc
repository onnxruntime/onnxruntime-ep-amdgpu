// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dml_abi_custom_registry.h"
#include "DmlExecutionProvider/inc/IWinmlExecutionProvider.h"
#include "DmlExecutionProvider/precomp.h"

namespace dml_ep {

PluginAbiCustomRegistry::PluginAbiCustomRegistry() :
    m_kernelRegistry(std::make_shared<onnxruntime::CustomRegistry>()),
    m_internalRegInfoMap(std::make_shared<InternalRegistrationInfoMap>()) {}

PluginAbiCustomRegistry::PluginAbiCustomRegistry(const PluginDmlExecutionProviderImpl* executionProvider) :
    m_kernelRegistry(std::make_shared<onnxruntime::CustomRegistry>()),
    m_internalRegInfoMap(std::make_shared<InternalRegistrationInfoMap>()),
    m_dmlPluginExecutionProvider(executionProvider) {}

#pragma warning(push)
#pragma warning(suppress : 4702)
onnx::OpSchema::FormalParameterOption
PluginAbiCustomRegistry::ConvertFormalParameterOption(MLOperatorParameterOptions options) {
    switch (options) {
    case MLOperatorParameterOptions::Single:
        return onnx::OpSchema::FormalParameterOption::Single;

    case MLOperatorParameterOptions::Optional:
        return onnx::OpSchema::FormalParameterOption::Optional;

    case MLOperatorParameterOptions::Variadic:
        return onnx::OpSchema::FormalParameterOption::Variadic;

    default:
        ORT_THROW_HR(E_NOTIMPL);
        return onnx::OpSchema::FormalParameterOption::Single;
    }
}
#pragma warning(pop)

// Convert edge types from the ABI types to ONNX strings
std::string PluginAbiCustomRegistry::ConvertFormalParameterType(const MLOperatorSchemaEdgeDescription& formalParameter) {
    ML_CHECK_BOOL(formalParameter.typeFormat == MLOperatorSchemaEdgeTypeFormat::Label ||
                  formalParameter.typeFormat == MLOperatorSchemaEdgeTypeFormat::EdgeDescription);

    if (formalParameter.typeFormat == MLOperatorSchemaEdgeTypeFormat::Label) {
        return formalParameter.typeLabel;
    } else {
        return ToTypeString(formalParameter.edgeDescription);
    }
}

// Convert type constraints from the ABI types to ONNX strings
std::vector<std::string> PluginConvertTypeConstraintTypes(const MLOperatorEdgeTypeConstrant& constraint) {
    std::vector<std::string> ret;
    ret.reserve(constraint.allowedTypeCount);

    for (uint32_t i = 0; i < constraint.allowedTypeCount; ++i) {
        ret.emplace_back(ToTypeString(constraint.allowedTypes[i]));
    }

    return ret;
}

// Convert attributes and defaults from the ABI to ONNX schema
void PluginAbiCustomRegistry::SetAttributesAndDefaults(onnx::OpSchema& schema, const MLOperatorSchemaDescription& abiSchema) {
    // Create a map with default attributes
    std::map<std::string, const MLOperatorAttributeNameValue*> defaultAttributes;
    for (uint32_t attributeIndex = 0; attributeIndex < abiSchema.defaultAttributeCount; ++attributeIndex) {
        const MLOperatorAttributeNameValue& defaultAttribute = abiSchema.defaultAttributes[attributeIndex];
        defaultAttributes[defaultAttribute.name] = &defaultAttribute;
    }

    // Set each attribute along with default values, looked up by name, if available
    for (uint32_t attributeIndex = 0; attributeIndex < abiSchema.attributeCount; ++attributeIndex) {
        const MLOperatorAttribute& attribute = abiSchema.attributes[attributeIndex];
        auto defaultVal = defaultAttributes.find(attribute.name);
        if (defaultVal == defaultAttributes.end()) {
            schema.Attr(attribute.name, "", ToProto(attribute.type), attribute.required);
        } else {
            ML_CHECK_BOOL(!attribute.required);
            ML_CHECK_BOOL(attribute.type == defaultVal->second->type);
            uint32_t defaultCount = defaultVal->second->valueCount;

            switch (attribute.type) {
            case MLOperatorAttributeType::Float:
                ML_CHECK_BOOL(defaultCount == 1);
                schema.Attr(attribute.name, "", ToProto(attribute.type), defaultVal->second->floats[0]);
                break;

            case MLOperatorAttributeType::Int:
                ML_CHECK_BOOL(defaultCount == 1);
                schema.Attr(attribute.name, "", ToProto(attribute.type), defaultVal->second->ints[0]);
                break;

            case MLOperatorAttributeType::String:
                ML_CHECK_BOOL(defaultCount == 1);
                schema.Attr(attribute.name, "", ToProto(attribute.type), std::string(defaultVal->second->strings[0]));
                break;

            case MLOperatorAttributeType::FloatArray: {
                std::vector<float> defaultVals(defaultVal->second->floats, defaultVal->second->floats + defaultCount);
                schema.Attr(attribute.name, "", ToProto(attribute.type), defaultVals);
                break;
            }

            case MLOperatorAttributeType::IntArray: {
                std::vector<int64_t> defaultVals(defaultVal->second->ints, defaultVal->second->ints + defaultCount);
                schema.Attr(attribute.name, "", ToProto(attribute.type), defaultVals);
                break;
            }

            case MLOperatorAttributeType::StringArray: {
                std::vector<std::string> defaultVals(defaultVal->second->strings,
                                                     defaultVal->second->strings + defaultCount);
                schema.Attr(attribute.name, "", ToProto(attribute.type), defaultVals);
                break;
            }

#pragma warning(suppress : 4063)
            case MLOperatorAttributeTypeTensor:
                // Tensor is too complex to express a default value. Default checking is done by the operator code.
                __fallthrough;

            default:
                ML_CHECK_BOOL(false);
                break;
            }

            // Remove the default attribute from the map, to later ensure defaults matched attributes
            defaultAttributes.erase(attribute.name);
        }
    }

    ML_CHECK_BOOL(defaultAttributes.empty());
}

// Convert a schema from the ABI to ONNX type
onnx::OpSchema PluginAbiCustomRegistry::ConvertOpSchema(_In_z_ const char* domain,
                                                  const MLOperatorSchemaDescription& abiSchema,
                                                  IMLOperatorTypeInferrer* typeInferrer,
                                                  IMLOperatorShapeInferrer* shapeInferrer) {
    // Set the op schema name, domain, and version
    onnx::OpSchema schema(abiSchema.name, "", 0);
    schema.SetDomain(domain);
    schema.SinceVersion(abiSchema.operatorSetVersionAtLastChange);

    // ONNX fails if using an empty string for edge names, although their names don't
    // matter for us.
    const char* emptyName = " ";

    // Populate inputs
    for (uint32_t inputIndex = 0; inputIndex < abiSchema.inputCount; ++inputIndex) {
        schema.Input(inputIndex, emptyName, "", ConvertFormalParameterType(abiSchema.inputs[inputIndex]),
                     ConvertFormalParameterOption(abiSchema.inputs[inputIndex].options));
    }

    // Populate outputs
    for (uint32_t outputIndex = 0; outputIndex < abiSchema.outputCount; ++outputIndex) {
        schema.Output(outputIndex, emptyName, "", ConvertFormalParameterType(abiSchema.outputs[outputIndex]),
                      ConvertFormalParameterOption(abiSchema.outputs[outputIndex].options));
    }

    // Populate type constraints
    for (uint32_t constraintIndex = 0; constraintIndex < abiSchema.typeConstraintCount; ++constraintIndex) {
        schema.TypeConstraint(abiSchema.typeConstraints[constraintIndex].typeLabel,
                              PluginConvertTypeConstraintTypes(abiSchema.typeConstraints[constraintIndex]), "");
    }

    // Set attribute defaults
    SetAttributesAndDefaults(schema, abiSchema);

    // Set an inferencing method
    if (shapeInferrer || typeInferrer) {
        Microsoft::WRL::ComPtr<IMLOperatorShapeInferrer> shapeInferrerCapture = shapeInferrer;
        Microsoft::WRL::ComPtr<IMLOperatorTypeInferrer> typeInferrerCapture = typeInferrer;

        // PROTOBUF ABI NOTE: This lambda is compiled into the plugin DLL and accesses TypeProto*
        // via onnx::InferenceContext. This is an accepted risk: fixing it would require ORT to provide
        // a C API equivalent for schema inference registration. The InferenceContext interface is from
        // the ONNX library and the TypeProto objects are used read-only.
        std::string schemaName(abiSchema.name);
        schema.TypeAndShapeInferenceFunction([=](onnx::InferenceContext& ctx) {
            // Constant CPU inputs cannot currently be specified through the public ABI for schema registration.
            gsl::span<const uint32_t> requiredConstantCpuInputs;

            onnxruntime::OpNodeProtoHelper<onnx::InferenceContext> nodeInfo(&ctx);
            Microsoft::WRL::ComPtr<MLSchemaInferenceContext> abiContext =
                MLSchemaInferenceContext::Create(&nodeInfo, &ctx, requiredConstantCpuInputs);

            // Do type inference
            if (typeInferrerCapture) {
                ORT_THROW_IF_FAILED(typeInferrerCapture->InferOutputTypes(abiContext.Get()));
            }

            // Do shape inference if all input tensor shapes are known
            bool shapesKnown = shapeInferrerCapture && InputTensorShapesDefinedOnNode(nodeInfo);
            if (shapeInferrerCapture && shapesKnown) {
                ORT_THROW_IF_FAILED(shapeInferrerCapture->InferOutputShapes(abiContext.Get()));
            }

            abiContext->Close();
        });
    }

    return schema;
}

HRESULT STDMETHODCALLTYPE PluginAbiCustomRegistry::RegisterOperatorSetSchema(
    const MLOperatorSetId* opSetId, int baseline_version, const MLOperatorSchemaDescription* const* schema,
    uint32_t schemaCount, _In_opt_ IMLOperatorTypeInferrer* typeInferrer,
    _In_opt_ IMLOperatorShapeInferrer* shapeInferrer) const noexcept {ORT_TRY{std::vector<onnx::OpSchema> schemaVector;
schemaVector.reserve(schemaCount);

// Convert schema to ONNX types and accumulate them in a vector
for (uint32_t i = 0; i < schemaCount; ++i) {
    schemaVector.emplace_back(ConvertOpSchema(opSetId->domain, *schema[i], typeInferrer, shapeInferrer));
}

// Multiple registries are used to avoid having different versions of the same domain in a single
// registry, which Lotus doesn't support.
auto registryKey = std::pair<int, int>(baseline_version, opSetId->version);
auto registryIter = m_customRegistryOpsetVerMap.find(registryKey);
if (registryIter == m_customRegistryOpsetVerMap.end()) {
    m_customRegistryOpsetVerMap[registryKey] = std::make_shared<onnxruntime::CustomRegistry>();
}

        Ort::Status status;

// Register the operator set with Lotus
// TODO - Split apart multiple op-sets with a common domain into multiple registries, as required by Lotus
// for correct lookup (Bug 4662).
THROW_IF_NOT_OK(m_customRegistryOpsetVerMap[registryKey]->RegisterOpSet(schemaVector, opSetId->domain, baseline_version,
                                                                        opSetId->version));

return S_OK;
}  // namespace dml_ep
ORT_CATCH_RETURN
}

// Convert the list of attribute defaults in a kernel registration into a
// map of AttributeValue entries, which own their own memory
AttributeMap PluginAbiCustomRegistry::GetDefaultAttributes(const MLOperatorKernelDescription* opKernel) {
    AttributeMap ret;

    for (uint32_t i = 0; i < opKernel->defaultAttributeCount; ++i) {
        const MLOperatorAttributeNameValue& apiAttr = opKernel->defaultAttributes[i];
        AttributeValue attr;

        attr.type = apiAttr.type;
        switch (apiAttr.type) {
        case MLOperatorAttributeType::Float:
            ML_CHECK_BOOL(apiAttr.valueCount == 1);
            __fallthrough;
        case MLOperatorAttributeType::FloatArray:
            attr.floats.assign(&apiAttr.floats[0], apiAttr.floats + apiAttr.valueCount);
            attr.floats.assign(&apiAttr.floats[0], apiAttr.floats + apiAttr.valueCount);
            break;

        case MLOperatorAttributeType::String:
            ML_CHECK_BOOL(apiAttr.valueCount == 1);
            __fallthrough;
        case MLOperatorAttributeType::StringArray:
            attr.strings.assign(&apiAttr.strings[0], &apiAttr.strings[apiAttr.valueCount]);
            break;

        case MLOperatorAttributeType::Int:
            ML_CHECK_BOOL(apiAttr.valueCount == 1);
            __fallthrough;
        case MLOperatorAttributeType::IntArray:
            attr.ints.assign(&apiAttr.ints[0], &apiAttr.ints[apiAttr.valueCount]);
            break;

#pragma warning(disable : 4063)
        case MLOperatorAttributeTypeTensor:
            // Tensor is too complex to express a default value. Default checking is done by the operator code.
            __fallthrough;

        default:
            ORT_THROW_HR(E_INVALIDARG);
        }

        ret[apiAttr.name] = attr;
    }

    return ret;
}

HRESULT STDMETHODCALLTYPE PluginAbiCustomRegistry::RegisterOperatorKernel(
    const MLOperatorKernelDescription* opKernel, IMLOperatorKernelFactory* operatorKernelFactory,
    _In_opt_ IMLOperatorShapeInferrer* shapeInferrer) const noexcept {
    return RegisterOperatorKernel(opKernel, operatorKernelFactory, shapeInferrer, nullptr, false, false);
}

HRESULT STDMETHODCALLTYPE PluginAbiCustomRegistry::RegisterOperatorKernel(
    const MLOperatorKernelDescription* opKernel, IMLOperatorKernelFactory* operatorKernelFactory,
    _In_opt_ IMLOperatorShapeInferrer* shapeInferrer, _In_opt_ IMLOperatorSupportQueryPrivate* supportQuery,
    bool isInternalOperator, bool supportsGraph, const uint32_t* requiredInputCountForGraph,
    _In_reads_(constantCpuInputCount) const uint32_t* requiredConstantCpuInputs, uint32_t constantCpuInputCount,
    _In_reads_(aliasCount) const std::pair<uint32_t, uint32_t>* aliases, uint32_t aliasCount) const noexcept {
    ORT_TRY {

        // Verify that invalid flags are not passed
        if ((opKernel->options & ~MLOperatorKernelOptions::AllowDynamicInputShapes) != MLOperatorKernelOptions::None) {
            return E_INVALIDARG;
        }

        // Translate flags
        bool requiresInputShapesAtCreation =
            (opKernel->options & MLOperatorKernelOptions::AllowDynamicInputShapes) == MLOperatorKernelOptions::None;
        bool requiresOutputShapesAtCreation = !!shapeInferrer;

        // Verify allowed combinations of flags are used
        if (!requiresInputShapesAtCreation && requiresOutputShapesAtCreation) {
            return E_INVALIDARG;
        }

        const char* providerType = nullptr;
        if (opKernel->executionOptions != 0) {
            return E_INVALIDARG;
        }

        // Might need to change cpu path later if we want to support CPU kernels that aren't just for copying data or
        // shape/size queries, but for now these are the only CPU kernels we have and they can all use the same provider
        // name.
        if (opKernel->executionType == MLOperatorExecutionType::Cpu) {
            providerType = "DirectMLExecutionProvider";
        } else if (opKernel->executionType == MLOperatorExecutionType::D3D12) {
            providerType = "DirectMLExecutionProvider";
        } else {
            return E_INVALIDARG;
        }

        // Set the name, domain, version, and provider
        onnxruntime::KernelDefBuilder builder;
        builder.SetName(opKernel->name);
        builder.SetDomain(opKernel->domain).SinceVersion(opKernel->minimumOperatorSetVersion).Provider(providerType);

        std::string_view name(opKernel->name);
        if (name == "MemcpyToHost") {
            builder.OutputMemoryType(::OrtMemType::OrtMemTypeCPUOutput, 0);
        } else if (name == "MemcpyFromHost") {
            builder.InputMemoryType(::OrtMemType::OrtMemTypeCPUInput, 0);
        } else if (name == "Shape") {
            builder.OutputMemoryType(::OrtMemType::OrtMemTypeCPUInput, 0);
        } else if (name == "Size") {
            builder.OutputMemoryType(::OrtMemType::OrtMemTypeCPUInput, 0);
        }

        std::vector<uint32_t> constantCpuInputCapture;
        constantCpuInputCapture.assign(requiredConstantCpuInputs, requiredConstantCpuInputs + constantCpuInputCount);

        for (uint32_t inputIndex : constantCpuInputCapture) {
            builder.InputMemoryType(::OrtMemType::OrtMemTypeCPUInput, inputIndex);
        }

        for (uint32_t i = 0; i < aliasCount; ++i) {
            builder.Alias(aliases[i].first, aliases[i].second);
        }

        // Set type constraints
        for (uint32_t i = 0; i < opKernel->typeConstraintCount; ++i) {
            std::vector<onnxruntime::MLDataType> types;
            types.reserve(opKernel->typeConstraints[i].allowedTypeCount);

            for (uint32_t j = 0; j < opKernel->typeConstraints[i].allowedTypeCount; ++j) {
                auto edgeType = opKernel->typeConstraints[i].allowedTypes[j].edgeType;
                // TODO - handle non-tensor types
                if (edgeType == MLOperatorEdgeType::Undefined) {
                    ORT_THROW_IF_FAILED(E_NOTIMPL);
                }

                types.push_back(ToMLDataType(edgeType, opKernel->typeConstraints[i].allowedTypes[j].tensorDataType));
            }

            builder.TypeConstraint(opKernel->typeConstraints[i].typeLabel, types);
        }

        Microsoft::WRL::ComPtr<IMLOperatorKernelFactory> kernelFactoryCapture = operatorKernelFactory;
        Microsoft::WRL::ComPtr<IMLOperatorShapeInferrer> shapeInferrerCapture = shapeInferrer;
        AttributeMap defaultAttributesCapture = GetDefaultAttributes(opKernel);

        const dml_ep::PluginDmlExecutionProviderImpl* dmlProviderCapture = m_dmlPluginExecutionProvider;

        auto lotusKernelCreateFn =
            [kernelFactoryCapture, requiresInputShapesAtCreation, requiresOutputShapesAtCreation, isInternalOperator,
             constantCpuInputCapture, shapeInferrerCapture, defaultAttributesCapture,
             dmlProviderCapture](onnxruntime::FuncManager&, const onnxruntime::OpKernelInfo& info,
                                       std::unique_ptr<onnxruntime::OpKernel>& out) -> Ort::Status {
            out = std::make_unique<PluginDmlAbiOpKernel>(
                kernelFactoryCapture.Get(), info, requiresInputShapesAtCreation, requiresOutputShapesAtCreation,
                isInternalOperator, constantCpuInputCapture, shapeInferrerCapture.Get(), &defaultAttributesCapture,
                dmlProviderCapture);
            return STATUS_OK;
        };

        onnxruntime::KernelCreateInfo create_info(builder.Build(), lotusKernelCreateFn);
        onnxruntime::KernelDef* kernelDef = create_info.kernel_def.get();

        if (isInternalOperator) {
            auto regInfo = std::make_shared<InternalRegistrationInfo>();
            regInfo->requiredConstantCpuInputs = constantCpuInputCapture;

            // Store ABI-safe kernel creation information for later use in dml_ep.cc
            regInfo->kernelFactory = kernelFactoryCapture.Get();
            regInfo->shapeInferrer = shapeInferrerCapture.Get();
            regInfo->defaultAttributes = defaultAttributesCapture;  // Copy by value
            regInfo->requiresInputShapesAtCreation = requiresInputShapesAtCreation;
            regInfo->requiresOutputShapesAtCreation = requiresOutputShapesAtCreation;
            regInfo->isInternalOperator = isInternalOperator;

            // Only internal operators support usage in DML graphs
            // DEAD CODE in plugin EP: This factory lambda is only invoked from
            // GraphDescBuilder and GraphPartitioner, which are legacy graph fusion code
            // never called from plugin EP entry points. The ProtoHelperNodeContext(node)
            // construction below is a protobuf ABI violation that is harmless because
            // this code path is unreachable from the plugin.
            if (supportsGraph) {
                GraphNodeFactoryRegistration graphReg;
                graphReg.factory = [kernelFactoryCapture, shapeInferrerCapture, defaultAttributesCapture,
                                    constantCpuInputCapture](
                                       const onnxruntime::Node& node, MLOperatorTensorGetter& constantInputGetter,
                                       const void* executionHandle, const EdgeShapes* inputShapesOverrides,
                                       /*out*/ EdgeShapes* outputShapes,
                                       /*out*/ DmlGraphNodeCreateInfo* graphNodeCreateInfo) {
                    onnxruntime::ProtoHelperNodeContext nodeContext(node);
                    onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext> protoHelper(&nodeContext);

                    // Use the same list of required constant inputs for the shape inferrer and the kernel.
                    PerformInferAndVerifyOutputSizes(node, &defaultAttributesCapture, shapeInferrerCapture.Get(),
                                              constantCpuInputCapture, constantInputGetter, inputShapesOverrides,
                                              *outputShapes);

                    // Create the kernel while allowing input shape and output shape queries according to options
                    Microsoft::WRL::ComPtr<DmlGraphOpKernelInfoWrapper> kernelInfoWrapper =
                        wil::MakeOrThrow<DmlGraphOpKernelInfoWrapper>(&protoHelper, executionHandle, true,
                                                                      inputShapesOverrides, outputShapes,
                                                                      &defaultAttributesCapture, graphNodeCreateInfo,
                                                                      constantCpuInputCapture, constantInputGetter);

                    Microsoft::WRL::ComPtr<IMLOperatorKernel> kernel;
                    ORT_THROW_IF_FAILED(
                        kernelFactoryCapture->CreateKernel(kernelInfoWrapper.Get(), kernel.GetAddressOf()));
                    kernelInfoWrapper->Close();
                };

                if (requiredInputCountForGraph) {
                    graphReg.requiredInputCount = *requiredInputCountForGraph;
                }

                regInfo->graphNodeFactoryRegistration = graphReg;
            }

            if (supportQuery) {
                Microsoft::WRL::ComPtr<IMLOperatorSupportQueryPrivate> supportQueryCapture = supportQuery;

                regInfo->supportQuery = [supportQueryCapture, defaultAttributesCapture](const OrtNode* node, const OrtApi& ort_api) {

                    onnxruntime::AbiSafeProtoHelperNodeContext nodeContext(node, ort_api);

                    onnxruntime::OpNodeProtoHelper<onnxruntime::AbiSafeProtoHelperNodeContext> protoHelper(&nodeContext);

                    Microsoft::WRL::ComPtr<PluginAbiSafeMLSupportQueryContext> supportContext =
                        PluginAbiSafeMLSupportQueryContext::Create(&protoHelper, &defaultAttributesCapture);

                    BOOL bSupported = FALSE;
                    ORT_THROW_IF_FAILED(supportQueryCapture->QuerySupport(supportContext.Get(), &bSupported));
                    return !!bSupported;
                };
            }

            THROW_IF_NOT_OK(m_kernelRegistry->RegisterCustomKernel(create_info));

            // Versioned key (domain::opname::sinceVersion) ensures ConvertKernelRegistryToOrtKernelRegistry
            // gets the correct per-version kernelFactory. Ops like Pad, Slice, and Clip change their
            // interface between versions (e.g., pads moves from attribute to input tensor at v11), so
            // each version must map to its own factory.
            int since_ver_start = 0, since_ver_end = 0;
            kernelDef->SinceVersion(&since_ver_start, &since_ver_end);
            std::string versionedKey = std::string(kernelDef->Domain()) + "::" +
                                       std::string(kernelDef->OpName()) + "::" +
                                       std::to_string(since_ver_start);
            (*m_internalRegInfoMap)[versionedKey] = regInfo;

            // Unversioned key for IsNodeSupportedByDml (supportQuery only, version-agnostic).
            std::string regKey = std::string(kernelDef->Domain()) + "::" + std::string(kernelDef->OpName());
            (*m_internalRegInfoMap)[regKey] = regInfo;
        } else {
            // Currently unsupported for external operators
            if (aliasCount > 0 || supportsGraph || requiredInputCountForGraph) {
                ORT_THROW_HR(E_INVALIDARG);
            }

            //
            // For backward compatibility, this does not propagate errors for external operators
            static_cast<void>(m_kernelRegistry->RegisterCustomKernel(create_info)); // ignore result
        }

        return S_OK;
    }
    ORT_CATCH_RETURN
}

}
