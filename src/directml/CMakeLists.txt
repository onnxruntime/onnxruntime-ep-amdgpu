# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT WIN32)
    message(FATAL_ERROR "DirectML EP is Windows only execution provider")
endif()

set(CMAKE_OBJECT_PATH_MAX 4096)

# ------------------------------------------------------------
# NuGet: Microsoft.AI.DirectML 1.15.4
# ------------------------------------------------------------

set(NUGET_EXE ${CMAKE_BINARY_DIR}/nuget.exe)
set(NUGET_DIR ${CMAKE_BINARY_DIR}/nuget_packages)

if(NOT EXISTS ${NUGET_EXE})
    message(STATUS "Downloading nuget.exe")
    file(DOWNLOAD
        https://dist.nuget.org/win-x86-commandline/latest/nuget.exe
        ${NUGET_EXE}
        SHOW_PROGRESS
    )
endif()

# ---- Helper: reject prerelease versions explicitly ----
function(install_nuget_package NAME VERSION)
    # NuGet prerelease versions always contain '-'
    if(VERSION MATCHES "-")
        message(FATAL_ERROR
            "Prerelease NuGet packages are forbidden.\n"
            "Package: ${NAME}\n"
            "Version: ${VERSION}"
        )
    endif()

    execute_process(
        COMMAND ${NUGET_EXE}
            install ${NAME}
            -Version ${VERSION}
            -OutputDirectory ${NUGET_DIR}
            -ExcludeVersion
        RESULT_VARIABLE _nuget_result
    )

    if(NOT _nuget_result EQUAL 0)
        message(FATAL_ERROR
            "Failed to install NuGet package: ${NAME} ${VERSION}"
        )
    endif()
endfunction()

# ---- Install packages ----
install_nuget_package(Microsoft.AI.DirectML 1.15.4)
install_nuget_package(directxtk12_desktop_2019 2023.4.28.1)
install_nuget_package(Microsoft.Direct3D.D3D12 1.619.1)
install_nuget_package(Microsoft.Windows.ImplementationLibrary 1.0.230411.1)

# ------------------------------------------------------------
# Imported targets
# ------------------------------------------------------------

# ---- Microsoft.AI.DirectML ----
set(DIRECTML_ROOT ${NUGET_DIR}/Microsoft.AI.DirectML)

add_library(nuget_directml SHARED IMPORTED GLOBAL)
set_target_properties(nuget_directml PROPERTIES
    IMPORTED_IMPLIB
        ${DIRECTML_ROOT}/bin/x64-win/DirectML.lib
    IMPORTED_LOCATION
        ${DIRECTML_ROOT}/runtimes/win-x64/native/DirectML.dll
    INTERFACE_INCLUDE_DIRECTORIES
        ${DIRECTML_ROOT}/include
)

# ---- DirectXTK12 ----
set(DIRECTXTK12_ROOT ${NUGET_DIR}/directxtk12_desktop_2019)

add_library(nuget_directxtk12 STATIC IMPORTED GLOBAL)
set_target_properties(nuget_directxtk12 PROPERTIES
    IMPORTED_LOCATION_DEBUG
        ${DIRECTXTK12_ROOT}/native/lib/x64/Debug/DirectXTK12.lib
    IMPORTED_LOCATION_RELEASE
        ${DIRECTXTK12_ROOT}/native/lib/x64/Release/DirectXTK12.lib
    IMPORTED_LOCATION_RELWITHDEBINFO
        ${DIRECTXTK12_ROOT}/native/lib/x64/Release/DirectXTK12.lib
    IMPORTED_LOCATION_MINSIZEREL
        ${DIRECTXTK12_ROOT}/native/lib/x64/Release/DirectXTK12.lib
    INTERFACE_INCLUDE_DIRECTORIES
        ${DIRECTXTK12_ROOT}/include
)

# ---- Direct3D 12 ----
set(D3D12_ROOT ${NUGET_DIR}/Microsoft.Direct3D.D3D12)

add_library(nuget_d3d12 INTERFACE IMPORTED GLOBAL)
set_target_properties(nuget_d3d12 PROPERTIES
    INTERFACE_LINK_LIBRARIES
        d3d12.lib
    INTERFACE_INCLUDE_DIRECTORIES
        ${D3D12_ROOT}/build/native/include
)

# ---- Windows Implementation Library (WIL) ----
set(WIL_ROOT ${NUGET_DIR}/Microsoft.Windows.ImplementationLibrary)

add_library(nuget_wil INTERFACE IMPORTED GLOBAL)
set_target_properties(nuget_wil PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES
        ${WIL_ROOT}/include
)
enable_language(ASM_MASM)

set(MLAS_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/core/mlas/lib)

set(MLAS_ASM_SOURCES
    ${MLAS_SRC_DIR}/amd64/sgemma.asm
    ${MLAS_SRC_DIR}/amd64/cvtfp16a.asm
    ${MLAS_SRC_DIR}/amd64/cvtfp16Avx.asm
    ${MLAS_SRC_DIR}/amd64/SgemmKernelSse2.asm
    ${MLAS_SRC_DIR}/amd64/SgemmKernelAvx.asm
    ${MLAS_SRC_DIR}/amd64/SgemmKernelFma3.asm
    ${MLAS_SRC_DIR}/amd64/SgemmKernelAvx512F.asm
    ${MLAS_SRC_DIR}/amd64/SgemmKernelM1Avx.asm
    ${MLAS_SRC_DIR}/amd64/DgemmKernelSse2.asm
    ${MLAS_SRC_DIR}/amd64/DgemmKernelAvx.asm
    ${MLAS_SRC_DIR}/amd64/DgemmKernelFma3.asm
    ${MLAS_SRC_DIR}/amd64/DgemmKernelAvx512F.asm
    ${MLAS_SRC_DIR}/amd64/SconvKernelSse2.asm
    ${MLAS_SRC_DIR}/amd64/SconvKernelAvx.asm
    ${MLAS_SRC_DIR}/amd64/SconvKernelFma3.asm
    ${MLAS_SRC_DIR}/amd64/SconvKernelAvx512F.asm
    ${MLAS_SRC_DIR}/amd64/SpoolKernelSse2.asm
    ${MLAS_SRC_DIR}/amd64/SpoolKernelAvx.asm
    ${MLAS_SRC_DIR}/amd64/SpoolKernelAvx512F.asm
    ${MLAS_SRC_DIR}/amd64/TransKernelFma3.asm
    ${MLAS_SRC_DIR}/amd64/TransKernelAvx512F.asm
    ${MLAS_SRC_DIR}/amd64/LogisticKernelFma3.asm
    ${MLAS_SRC_DIR}/amd64/TanhKernelFma3.asm
    ${MLAS_SRC_DIR}/amd64/ErfKernelFma3.asm
    ${MLAS_SRC_DIR}/amd64/SoftmaxKernelAvx.asm
    ${MLAS_SRC_DIR}/amd64/SoftmaxKernelAvx512F.asm
    ${MLAS_SRC_DIR}/amd64/QgemmU8S8KernelAmx.asm
    ${MLAS_SRC_DIR}/amd64/QgemmU8S8KernelAvx2.asm
    ${MLAS_SRC_DIR}/amd64/QgemmU8U8KernelAvx2.asm
    ${MLAS_SRC_DIR}/amd64/QgemmU8X8KernelAvx2.asm
    ${MLAS_SRC_DIR}/amd64/QgemmU8X8KernelAvx512Core.asm
    ${MLAS_SRC_DIR}/amd64/QgemvU8S8KernelAvx2.asm
    ${MLAS_SRC_DIR}/amd64/QgemvU8S8KernelAvx512Core.asm
    ${MLAS_SRC_DIR}/amd64/QgemvU8S8KernelAvx512Vnni.asm
    ${MLAS_SRC_DIR}/amd64/QgemvU8S8KernelAvxVnni.asm
    ${MLAS_SRC_DIR}/amd64/ConvSymKernelAvx2.asm
    ${MLAS_SRC_DIR}/amd64/ConvSymKernelAvx512Core.asm
)

set_source_files_properties(${MLAS_ASM_SOURCES} PROPERTIES
        LANGUAGE ASM_MASM
)

# AVX/AVX2/AVX512 intrinsic files need specific compiler flags
set_source_files_properties(
    ${MLAS_SRC_DIR}/intrinsics/avx/min_max_elements.cpp
    PROPERTIES COMPILE_FLAGS "/arch:AVX"
)
set_source_files_properties(
    ${MLAS_SRC_DIR}/intrinsics/avx2/qdwconv_avx2.cpp
    ${MLAS_SRC_DIR}/intrinsics/avx2/qladd_avx2.cpp
    ${MLAS_SRC_DIR}/intrinsics/avx2/saturation_check_avx2.cpp
    ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx2.cpp
    ${MLAS_SRC_DIR}/sqnbitgemm_lut_kernel_avx2.cpp
 ${MLAS_SRC_DIR}/rotary_embedding_kernel_avx2.cpp
    PROPERTIES COMPILE_FLAGS "/arch:AVX2"
)
set_source_files_properties(
    ${MLAS_SRC_DIR}/intrinsics/avx512/quantize_avx512f.cpp
    ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512.cpp
    ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512vnni.cpp
    ${MLAS_SRC_DIR}/q4gemm_avx512.cpp
    PROPERTIES COMPILE_FLAGS "/arch:AVX512"
)

add_library(directml-ep MODULE
    # ---- Top-level directml sources ----
    cpu_allocator.h
    dml_abi_custom_registry.h
    dml_abi_kernel.h
    dml_bucketized_buffer_allocator.h
    dml_client.h
    dml_common.h
    dml_data_transfer.h
    dml_ep.h
    dml_execution_context.h
    dml_execution_provider.h
    dml_factory.h
    dml_info.h
    dml_plugin_command_recorder.h
    dml_plugin_MLOperatorAuthorImpl.h
    dml_plugin_OperatorRegistration.h
    dml_pooled_upload_heap.h
    dml_readback_heap.h
    dml_utils.h
    iallocator_to_ort_allocator_adapter.h
    ort_node_adapter.h
    plugin_dml_AllocationInfo.h
    plugin_DmlDFT.h
    plugin_DmlGridSample.h
    plugin_DmlSTFT.h
    OperatorAuthorHelper/operator_helper_common.h
    ort_node_adapter.cc
    dml_abi_custom_registry.cc
    dml_abi_kernel.cc
    dml_bucketized_buffer_allocator.cc
    dml_common.cc
    dml_data_transfer.cc
    dml_ep.cc
    dml_execution_context.cc
    dml_execution_provider.cc
    dml_factory.cc
    dml_info.cc
    dml_plugin_command_recorder.cc
    dml_plugin_MLOperatorAuthorImpl.cc
    dml_plugin_OperatorRegistration.cc
    dml_pooled_upload_heap.cc
    dml_readback_heap.cc
    dml_utils.cc

    # ---- Graph fusion (ABI-safe, plugin EP only) ----
    fusion_utils.cc
    pattern_matcher.cc
    ep_fusion_manager.cc
    quick_gelu_ep_fusion.cc
    bias_gelu_ep_fusion.cc

    #---- DirectML EP sources ----
    DmlExecutionProvider/ApiTraits.cc
    DmlExecutionProvider/CommandQueue.cc
    DmlExecutionProvider/DescriptorPool.cc
    DmlExecutionProvider/DmlCommittedResourceAllocator.cc
    DmlExecutionProvider/DmlGraphDeserialization.cc
    DmlExecutionProvider/DmlGraphFusionHelper.cc
    DmlExecutionProvider/DmlGraphFusionTransformer.cc
    DmlExecutionProvider/DmlGraphSerialization.cc
    DmlExecutionProvider/DmlRuntimeGraphFusionTransformer.cc
    DmlExecutionProvider/FusedGraphKernel.cc
    DmlExecutionProvider/GraphDescBuilder.cc
    DmlExecutionProvider/GraphPartitioner.cc
    DmlExecutionProvider/GraphTransformer.cc
    DmlExecutionProvider/TensorDesc.cc
    DmlExecutionProvider/MLOperatorAuthorImpl.cc
    DmlExecutionProvider/DmlRuntimeFusedGraphKernel.cc
    OperatorAuthorHelper/OperatorHelper.cc

    #---- DirectML Operator sources ----
    DmlExecutionProvider/Operators/DmlOperator.cc
    DmlExecutionProvider/Operators/DmlOperatorActivation.cc
    DmlExecutionProvider/Operators/DmlOperatorAffine.cc
    DmlExecutionProvider/Operators/DmlOperatorAttention.cc
    DmlExecutionProvider/Operators/DmlOperatorBatchNormalization.cc
    DmlExecutionProvider/Operators/DmlOperatorBiasAdd.cc
    DmlExecutionProvider/Operators/DmlOperatorBiasGelu.cc
    DmlExecutionProvider/Operators/DmlOperatorBiasSplitGelu.cc
    DmlExecutionProvider/Operators/DmlOperatorCast.cc
    DmlExecutionProvider/Operators/DmlOperatorCol2Im.cc
    DmlExecutionProvider/Operators/DmlOperatorConcat.cc
    DmlExecutionProvider/Operators/DmlOperatorConcatFromSequence.cc
    DmlExecutionProvider/Operators/DmlOperatorConstantOfShape.cc
    DmlExecutionProvider/Operators/DmlOperatorConvInteger.cc
    DmlExecutionProvider/Operators/DmlOperatorConvolution.cc
    DmlExecutionProvider/Operators/DmlOperatorCopy.cc
    DmlExecutionProvider/Operators/DmlOperatorCrop.cc
    DmlExecutionProvider/Operators/DmlOperatorCumSum.cc
    DmlExecutionProvider/Operators/DmlOperatorDepthToSpace.cc
    DmlExecutionProvider/Operators/DmlOperatorDynamicQuantizeLinear.cc
    DmlExecutionProvider/Operators/DmlOperatorDynamicQuantizeMatMul.cc
    DmlExecutionProvider/Operators/DmlOperatorEinSum.cc
    DmlExecutionProvider/Operators/DmlOperatorElementWise.cc
    DmlExecutionProvider/Operators/DmlOperatorEmbedLayerNormalization.cc
    DmlExecutionProvider/Operators/DmlOperatorExpand.cc
    DmlExecutionProvider/Operators/DmlOperatorEyeLike.cc
    DmlExecutionProvider/Operators/DmlOperatorFusedMatMul.cc
    DmlExecutionProvider/Operators/DmlOperatorGather.cc
    DmlExecutionProvider/Operators/DmlOperatorGemm.cc
    DmlExecutionProvider/Operators/DmlOperatorGroupNorm.cc
    DmlExecutionProvider/Operators/DmlOperatorGroupQueryAttention.cc
    DmlExecutionProvider/Operators/DmlOperatorInstanceNormalization.cc
    DmlExecutionProvider/Operators/DmlOperatorLayerNormalization.cc
    DmlExecutionProvider/Operators/DmlOperatorLocalResponseNormalization.cc
    DmlExecutionProvider/Operators/DmlOperatorLpNormalization.cc
    DmlExecutionProvider/Operators/DmlOperatorMatMul.cc
    DmlExecutionProvider/Operators/DmlOperatorMatMulInteger.cc
    DmlExecutionProvider/Operators/DmlOperatorMatMulIntegerToFloat.cc
    DmlExecutionProvider/Operators/DmlOperatorMatMulNBits.cc
    DmlExecutionProvider/Operators/DmlOperatorMaxUnpool.cc
    DmlExecutionProvider/Operators/DmlOperatorMeanVarianceNormalization.cc
    DmlExecutionProvider/Operators/DmlOperatorMemcpy.cc
    DmlExecutionProvider/Operators/DmlOperatorMultiHeadAttention.cc
    DmlExecutionProvider/Operators/DmlOperatorNeg.cc
    DmlExecutionProvider/Operators/DmlOperatorNonZero.cc
    DmlExecutionProvider/Operators/DmlOperatorOneHot.cc
    DmlExecutionProvider/Operators/DmlOperatorPadding.cc
    DmlExecutionProvider/Operators/DmlOperatorPooling.cc
    DmlExecutionProvider/Operators/DmlOperatorQAttention.cc
    DmlExecutionProvider/Operators/DmlOperatorQLinearAdd.cc
    DmlExecutionProvider/Operators/DmlOperatorQLinearAveragePooling.cc
    DmlExecutionProvider/Operators/DmlOperatorQLinearConcat.cc
    DmlExecutionProvider/Operators/DmlOperatorQLinearConv.cc
    DmlExecutionProvider/Operators/DmlOperatorQLinearMatMul.cc
    DmlExecutionProvider/Operators/DmlOperatorQLinearSigmoid.cc
    DmlExecutionProvider/Operators/DmlOperatorQuickGelu.cc
    DmlExecutionProvider/Operators/DmlOperatorRange.cc
    DmlExecutionProvider/Operators/DmlOperatorRecurrentNeuralNetwork.cc
    DmlExecutionProvider/Operators/DmlOperatorReduce.cc
    DmlExecutionProvider/Operators/DmlOperatorResize.cc
    DmlExecutionProvider/Operators/DmlOperatorReverseSequence.cc
    DmlExecutionProvider/Operators/DmlOperatorRoiAlign.cc
    DmlExecutionProvider/Operators/DmlOperatorRoiPooling.cc
    DmlExecutionProvider/Operators/DmlOperatorRotaryEmbedding.cc
    DmlExecutionProvider/Operators/DmlOperatorScatter.cc
    DmlExecutionProvider/Operators/DmlOperatorShape.cc
    DmlExecutionProvider/Operators/DmlOperatorSize.cc
    DmlExecutionProvider/Operators/DmlOperatorSkipLayerNormalization.cc
    DmlExecutionProvider/Operators/DmlOperatorSlice.cc
    DmlExecutionProvider/Operators/DmlOperatorSpaceToDepth.cc
    DmlExecutionProvider/Operators/DmlOperatorSplit.cc
    DmlExecutionProvider/Operators/DmlOperatorTile.cc
    DmlExecutionProvider/Operators/DmlOperatorTopk.cc
    DmlExecutionProvider/Operators/DmlOperatorTranspose.cc
    DmlExecutionProvider/Operators/DmlOperatorTrilu.cc
    DmlExecutionProvider/Operators/DmlOperatorValueScale2D.cc
    DmlExecutionProvider/Operators/OperatorUtility.cc

    #---- Core framework sources ----
    core/framework/abi_safe_attr_utils.cc
    core/framework/allocation_planner.cc
    core/framework/allocator.cc
    core/framework/allocator_utils.cc
    core/framework/copy.cc
    core/framework/bfc_arena.cc
    core/framework/config_options.cc
    core/framework/customregistry.cc
    core/framework/data_transfer.cc
    core/framework/data_transfer_manager.cc
    core/framework/data_types.cc
    core/framework/endian_utils.cc
    core/framework/ep_context_options.cc
    core/framework/error_code.cc
    core/framework/execution_frame.cc
    core/framework/execution_provider.cc
    core/framework/execution_steps.cc
    core/framework/ex_lib_loader.cc
    core/framework/external_data_loader_manager.cc
    core/framework/feeds_fetches_manager.cc
    core/framework/kernel_def_builder.cc
    core/framework/kernel_registry.cc
    core/framework/kernel_registry_manager.cc
    core/framework/kernel_type_str_resolver.cc
    core/framework/library_handles.cc
    core/framework/memory_info.cc
    core/framework/mldata_type_utils.cc
    core/framework/node_index_info.cc
    core/framework/node_unit.cc
    core/framework/onnxruntime_map_type_info.cc
    core/framework/onnxruntime_optional_type_info.cc
    core/framework/onnxruntime_sequence_type_info.cc
    core/framework/onnxruntime_typeinfo.cc
    core/framework/op_kernel.cc
    core/framework/op_kernel_info.cc
    core/framework/op_node_proto_helper.cc
    core/framework/ort_value_pattern_planner.cc
    core/framework/prepacked_weights.cc
    core/framework/prepacked_weights_container.cc
    core/framework/resource_accountant.cc
    core/framework/run_options.cc
    core/framework/sequential_executor.cc
    core/framework/session_state.cc
    core/framework/session_state_utils.cc
    core/framework/simple_tensor_allocator.cc
    core/framework/sparse_tensor.cc
    core/framework/sparse_utils.cc
    core/framework/tensor.cc
    core/framework/tensor_allocator.cc
    core/framework/tensor_external_data_info.cc
    core/framework/tensor_shape.cc
    core/framework/tensor_type_and_shape.cc
    core/framework/tensorprotoutils.cc
    core/framework/utils.cc
    core/framework/stream_execution_context.cc
    core/framework/device_stream_collection.cc
    core/framework/stream_handles.cc
    core/framework/transpose_helper.cc

    #---- Core flatbuffers sources ----
    core/flatbuffers/flatbuffers_utils.cc

    #---- Core providers sources ----
    core/providers/cpu/controlflow/if.cc
    core/providers/cpu/controlflow/utils.cc
    core/providers/cpu/optional/optional_ops.cc
    core/providers/cpu/sequence/sequence_ops.cc
    core/providers/cpu/tensor/transpose.cc

    #---- Core platform sources ----
    core/platform/path_lib.cc
    core/platform/env.cc
    core/platform/check_intel.cc
    core/platform/windows/env.cc
    core/platform/windows/hardware_core_enumerator.cc

    #---- Core common sources ----
    core/common/threadpool.cc
    core/common/denormal.cc
    core/common/cpuid_info.cc
    core/common/cpuid_info_vendor.cc
    core/common/cpuid_uarch.cc
    core/common/spin_pause.cc

    #---- Core graph sources ----
    core/graph/graph.cc
    core/graph/graph_viewer.cc
    core/graph/graph_utils.cc
    core/graph/graph_flatbuffers_utils.cc
    core/graph/graph_proto_serializer.cc
    core/graph/model.cc
    core/graph/node_attr_utils.cc
    core/graph/op.cc
    core/graph/op_identifier_utils.cc
    core/graph/schema_registry.cc
    core/graph/function.cc
    core/graph/function_utils.cc
    core/graph/ep_api_types.cc
    core/graph/runtime_optimization_record_container.cc
    core/graph/contrib_ops/contrib_defs.cc
    core/graph/contrib_ops/bert_defs.cc
    core/graph/contrib_ops/collective_defs.cc
    core/graph/contrib_ops/diffusion_defs.cc
    core/graph/contrib_ops/internal_nhwc_onnx_schemas.cc
    core/graph/contrib_ops/nchwc_schema_defs.cc
    core/graph/contrib_ops/nhwc_schema_defs.cc
    core/graph/contrib_ops/onnx_deprecated_operators.cc
    core/graph/contrib_ops/onnx_function_util.cc
    core/graph/contrib_ops/quantization_defs.cc
    core/graph/contrib_ops/range_schema_defs.cc
    core/graph/contrib_ops/shape_inference_functions.cc
    core/graph/contrib_ops/attn_lstm_schema_defs.cc
    core/graph/dml_ops/dml_defs.cc
    core/graph/data_propagation/add_op_data_propagation.cc
    core/graph/data_propagation/custom_data_propagation.cc
    core/graph/data_propagation/div_op_data_propagation.cc
    core/graph/data_propagation/gather_op_data_propagation.cc
    core/graph/data_propagation/mul_op_data_propagation.cc
    core/graph/data_propagation/size_op_data_propagation.cc
    core/graph/data_propagation/squeeze_op_data_propagation.cc
    core/graph/data_propagation/sub_op_data_propagation.cc
    core/graph/data_propagation/unsqueeze_op_data_propagation.cc

    #---- Core optimizer sources ----
    core/optimizer/graph_optimizer_registry.cc
    core/optimizer/graph_transformer.cc
    core/optimizer/graph_transformer_mgr.cc
    core/optimizer/graph_transformer_utils.cc
    core/optimizer/rule_based_graph_transformer.cc
    core/optimizer/utils.cc
    core/optimizer/initializer.cc
    core/optimizer/constant_folding.cc
    core/optimizer/constant_sharing.cc
    core/optimizer/optimizer_execution_frame.cc
    core/optimizer/transpose_optimization/onnx_transpose_optimization.cc
    core/optimizer/transpose_optimization/ort_optimizer_api_impl.cc
    core/optimizer/transpose_optimization/ort_transpose_optimization.cc
    core/optimizer/transpose_optimizer.cc
    core/optimizer/transformer_memcpy.cc
    core/optimizer/insert_cast_transformer.cc
    core/optimizer/layout_transformation/layout_transformation.cc
    core/optimizer/nhwc_transformer.cc
    core/optimizer/nchwc_transformer.cc
    core/optimizer/selection_and_optimization_func.cc
    core/optimizer/selectors_actions/actions.cc
    core/optimizer/selectors_actions/helpers.cc
    core/optimizer/selectors_actions/selector_action_transformer.cc
    core/optimizer/free_dim_override_transformer.cc
    core/optimizer/fuse_initializers_transformer.cc
    core/optimizer/identity_elimination.cc
    core/optimizer/noop_elimination.cc
    core/optimizer/dropout_elimination.cc
    core/optimizer/slice_elimination.cc
    core/optimizer/concat_slice_elimination.cc
    core/optimizer/unsqueeze_elimination.cc
    core/optimizer/expand_elimination.cc
    core/optimizer/pre_shape_node_elimination.cc
    core/optimizer/reshape_fusion.cc
    core/optimizer/conv_activation_fusion.cc
    core/optimizer/conv_add_fusion.cc
    core/optimizer/conv_add_act_fusion.cc
    core/optimizer/conv_bn_fusion.cc
    core/optimizer/conv_mul_fusion.cc
    core/optimizer/gemm_activation_fusion.cc
    core/optimizer/gemm_sum_fusion.cc
    core/optimizer/gemm_transpose_fusion.cc
    core/optimizer/matmul_activation_fusion.cc
    core/optimizer/matmul_add_fusion.cc
    core/optimizer/matmul_bn_fusion.cc
    core/optimizer/matmul_integer_to_float.cc
    core/optimizer/matmul_nbits_fusion.cc
    core/optimizer/matmul_scale_fusion.cc
    core/optimizer/matmul_transpose_fusion.cc
    core/optimizer/gelu_fusion.cc
    core/optimizer/gelu_approximation.cc
    core/optimizer/fast_gelu_fusion.cc
    core/optimizer/quick_gelu_fusion.cc
    core/optimizer/layer_norm_fusion.cc
    core/optimizer/skip_layer_norm_fusion.cc
    core/optimizer/embed_layer_norm_fusion.cc
    core/optimizer/bias_softmax_fusion.cc
    core/optimizer/attention_fusion.cc
    core/optimizer/group_query_attention_fusion.cc
    core/optimizer/relu_clip_fusion.cc
    core/optimizer/div_mul_fusion.cc
    core/optimizer/not_where_fusion.cc
    core/optimizer/gather_fusion.cc
    core/optimizer/isinf_reducesum_fusion.cc
    core/optimizer/label_encoder_fusion.cc
    core/optimizer/pad_fusion.cc
    core/optimizer/propagate_cast_ops.cc
    core/optimizer/stft_decomposition.cc
    core/optimizer/dynamic_quantize_matmul_fusion.cc
    core/optimizer/identical_children_consolidation.cc
    core/optimizer/double_qdq_pairs_remover.cc
    core/optimizer/qdq_transformer/clip_quantizelinear.cc
    core/optimizer/qdq_transformer/constant_folding_dq_node.cc
    core/optimizer/qdq_transformer/ensure_unique_dq_for_node_unit.cc
    core/optimizer/qdq_transformer/qdq_final_cleanup.cc
    core/optimizer/qdq_transformer/qdq_propagation.cc
    core/optimizer/qdq_transformer/qdq_s8_to_u8.cc
    core/optimizer/qdq_transformer/qdq_util.cc
    core/optimizer/qdq_transformer/relu_quantizelinear.cc
    core/optimizer/qdq_transformer/s8_to_u8.cc
    core/optimizer/qdq_transformer/avx2_weight_s8_to_u8.cc
    core/optimizer/qdq_transformer/selectors_actions/qdq_actions.cc
    core/optimizer/qdq_transformer/selectors_actions/qdq_selector_action_transformer.cc
    core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.cc
    core/optimizer/qdq_transformer/selectors_actions/shared/utils.cc
    core/optimizer/qdq_transformer/weight_bias_quantization.cc
    core/optimizer/qdq_transformer/where_dummy_dq.cc
    core/optimizer/compute_optimizer/upstream_gather.cc
    core/optimizer/compute_optimizer/upstream_gather_actors.cc
    core/optimizer/compute_optimizer/upstream_reshape.cc
    core/optimizer/compute_optimizer/upstream_reshape_actors.cc
    core/optimizer/compute_optimizer/upstream_transformer_base.cc
    core/optimizer/cast_elimination.cc
    core/optimizer/cast_chain_elimination.cc
    core/optimizer/bias_dropout_fusion.cc
    core/optimizer/bias_gelu_fusion.cc
    core/optimizer/common_subexpression_elimination.cc

    #---- MLAS sources (inlined) ----
    ${MLAS_SRC_DIR}/platform.cpp
    ${MLAS_SRC_DIR}/threading.cpp
    ${MLAS_SRC_DIR}/sgemm.cpp
    ${MLAS_SRC_DIR}/dgemm.cpp
    ${MLAS_SRC_DIR}/quantize.cpp
    ${MLAS_SRC_DIR}/reorder.cpp
    ${MLAS_SRC_DIR}/snchwc.cpp
    ${MLAS_SRC_DIR}/cast.cpp
    ${MLAS_SRC_DIR}/convolve.cpp
    ${MLAS_SRC_DIR}/pooling.cpp
    ${MLAS_SRC_DIR}/transpose.cpp
    ${MLAS_SRC_DIR}/activate.cpp
    ${MLAS_SRC_DIR}/logistic.cpp
    ${MLAS_SRC_DIR}/tanh.cpp
    ${MLAS_SRC_DIR}/erf.cpp
    ${MLAS_SRC_DIR}/compute.cpp
    ${MLAS_SRC_DIR}/halfgemm.cpp
    ${MLAS_SRC_DIR}/qgemm.cpp
    ${MLAS_SRC_DIR}/qnbitgemm.cpp
    ${MLAS_SRC_DIR}/qdwconv.cpp
    ${MLAS_SRC_DIR}/qdwconv_kernelsize.cpp
    ${MLAS_SRC_DIR}/qpostprocessor.cpp
    ${MLAS_SRC_DIR}/qlmul.cpp
    ${MLAS_SRC_DIR}/qladd.cpp
    ${MLAS_SRC_DIR}/qlgavgpool.cpp
    ${MLAS_SRC_DIR}/qlutgemm.cpp
    ${MLAS_SRC_DIR}/q4gemm.cpp
    ${MLAS_SRC_DIR}/q4_dq.cpp
    ${MLAS_SRC_DIR}/dwconv.cpp
    ${MLAS_SRC_DIR}/eltwise.cpp
    ${MLAS_SRC_DIR}/saturation_check.cpp
    ${MLAS_SRC_DIR}/convsym.cpp
    ${MLAS_SRC_DIR}/dequantize.cpp
    ${MLAS_SRC_DIR}/rotary_embedding.cpp
    ${MLAS_SRC_DIR}/intrinsics/avx/min_max_elements.cpp
    ${MLAS_SRC_DIR}/intrinsics/avx2/qdwconv_avx2.cpp
    ${MLAS_SRC_DIR}/intrinsics/avx2/qladd_avx2.cpp
    ${MLAS_SRC_DIR}/intrinsics/avx2/saturation_check_avx2.cpp
    ${MLAS_SRC_DIR}/intrinsics/avx512/quantize_avx512f.cpp
    ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx2.cpp
    ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512.cpp
    ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512vnni.cpp
    ${MLAS_SRC_DIR}/sqnbitgemm_lut_kernel_avx2.cpp
    ${MLAS_SRC_DIR}/q4gemm_avx512.cpp
    ${MLAS_SRC_DIR}/q4_dq_cli.cpp
    ${MLAS_SRC_DIR}/qgemm_kernel_default.cpp
    ${MLAS_SRC_DIR}/qgemm_kernel_sse.cpp
    ${MLAS_SRC_DIR}/qgemm_kernel_sse41.cpp
    ${MLAS_SRC_DIR}/qgemm_kernel_avx2.cpp
    ${MLAS_SRC_DIR}/qgemm_kernel_amx.cpp
    ${MLAS_SRC_DIR}/rotary_embedding_kernel_avx2.cpp
    ${MLAS_ASM_SOURCES}
)

add_custom_command(TARGET directml-ep POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${DIRECTML_ROOT}/bin/x64-win/DirectML.dll
        $<TARGET_FILE_DIR:directml-ep>
)

add_custom_command(TARGET directml-ep POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${D3D12_ROOT}/build/native/bin/x64/D3D12Core.dll
        $<TARGET_FILE_DIR:directml-ep>
)

target_include_directories(directml-ep PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/External
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/OperatorAuthorHelper
    ${CMAKE_CURRENT_SOURCE_DIR}/core/mlas/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/core/mlas/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/core/providers
)

# Prevent object file name collisions for files with the same basename
# in different directories (e.g. core/platform/env.cc vs core/platform/windows/env.cc)
set_target_properties(directml-ep PROPERTIES OBJECT_PATH_MAX 4096)

# Fix C1041 PDB contention: serialize writes to the shared .pdb file
# and use /Z7 (embedded debug info) to avoid shared PDB file entirely
target_compile_options(directml-ep PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:/FS>)
set_target_properties(directml-ep PROPERTIES MSVC_DEBUG_INFORMATION_FORMAT $<$<CONFIG:Debug,RelWithDebInfo>:Embedded>)

target_link_libraries(directml-ep
    PRIVATE
        plugin-ep-utils
        flatbuffers
        onnx
        onnx_proto
        protobuf::libprotobuf
        Eigen3::Eigen
        nuget_d3d12
        nuget_directxtk12
        nuget_directml
        nuget_wil
        Boost::mp11
        Microsoft::DirectX-Headers
        Microsoft::DirectX-Guids
)

target_compile_definitions(directml-ep
    PRIVATE
        NOGDI
        NOMINMAX
        _USE_MATH_DEFINES
        _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
        USE_DML=1
        DML_TARGET_VERSION_USE_LATEST
)

set_property(TARGET directml-ep APPEND_STRING
        PROPERTY LINK_FLAGS /DEF:${PROJECT_SOURCE_DIR}/src/shared/symbols.def)

install(TARGETS directml-ep
        DESTINATION ".")

install(FILES ${DIRECTML_ROOT}/bin/x64-win/DirectML.dll ${D3D12_ROOT}/build/native/bin/x64/D3D12Core.dll
        DESTINATION ".")

install(FILES $<TARGET_PDB_FILE:directml-ep>
        DESTINATION "." OPTIONAL)

if(USE_AMDGPU)
    set_target_properties(directml-ep PROPERTIES OUTPUT_NAME "directml-backend")
    target_include_directories(directml-ep INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

add_version_info(
     TARGET directml-ep
        DESCRIPTION "DirectML Execution Provider - ONNX Runtime plugin"
        PRODUCT "DirectML Plugin EP")
