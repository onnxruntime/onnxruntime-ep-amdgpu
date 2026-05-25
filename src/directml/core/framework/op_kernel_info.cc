// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/ort_value_name_idx_map.h"
#include "core/framework/fuse_nodes_funcs.h"
#include "core/framework/op_kernel.h"
#include "core/framework/op_kernel_info.h"
#include "core/framework/sequential_execution_plan.h"
#include <onnxruntime_c_api.h>

namespace onnxruntime {

OpKernelInfo::OpKernelInfo(const onnxruntime::Node& node,
                           const KernelDef& kernel_def,
                           const IExecutionProvider& execution_provider,
                           const std::unordered_map<int, OrtValue>& constant_initialized_tensors,
                           const OrtValueNameIdxMap& ort_value_name_idx_map,
                           const DataTransferManager& data_transfer_mgr,
                           const AllocatorMap& allocators,
                           const ConfigOptions& config_options)
    : OpNodeProtoHelper(&proto_helper_context_),
      node_(node),
      kernel_def_(kernel_def),
      execution_provider_(&execution_provider),
      constant_initialized_tensors_(constant_initialized_tensors),
      ort_value_name_idx_map_(ort_value_name_idx_map),
      data_transfer_mgr_(data_transfer_mgr),
      proto_helper_context_(node),
      allocators_(allocators),
      config_options_(config_options) {
}

OpKernelInfo::OpKernelInfo(const OpKernelInfo& other)
    : OpKernelInfo(other.node_, other.kernel_def_, *other.execution_provider_, other.constant_initialized_tensors_,
                   other.ort_value_name_idx_map_, other.data_transfer_mgr_,
                   other.allocators_, other.config_options_) {
}

AllocatorPtr OpKernelInfo::GetAllocator(OrtMemType mem_type) const {
  auto it = allocators_.find(execution_provider_->GetOrtDeviceByMemType(mem_type));
  if (it != allocators_.end()) {
    return it->second;
  }

  return nullptr;
}

const OrtDevice OpKernelInfo::GetDevice(OrtMemType mem_type) const {
  return execution_provider_->GetOrtDeviceByMemType(mem_type);
}

const KernelDef& OpKernelInfo::GetKernelDef() const {
  return kernel_def_;
}

const IExecutionProvider* OpKernelInfo::GetExecutionProvider() const noexcept {
  return execution_provider_;
}

const DataTransferManager& OpKernelInfo::GetDataTransferManager() const noexcept {
  return data_transfer_mgr_;
}

const onnxruntime::Node& OpKernelInfo::node() const noexcept {
  return node_;
}

bool OpKernelInfo::TryGetConstantInput(int input_index, const Tensor** constant_input_value) const {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (!api || input_index < 0) return false;

  auto* ort_info = reinterpret_cast<const OrtKernelInfo*>(this);
  int isConstant = 0;
  const OrtValue* ortValue = nullptr;
  OrtStatus* status = api->KernelInfoGetConstantInput_tensor(ort_info, static_cast<size_t>(input_index), &isConstant, &ortValue);
  if (status) { api->ReleaseStatus(status); return false; }
  if (!isConstant || !ortValue) return false;

  if (!ortValue->IsTensor()) return false;
  *constant_input_value = &ortValue->Get<Tensor>();
  return true;
}

bool OpKernelInfo::TryGetConstantInput(int input_index, const OrtValue** constant_input_value) const {
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (!api || input_index < 0) return false;

  auto* ort_info = reinterpret_cast<const OrtKernelInfo*>(this);
  int isConstant = 0;
  const OrtValue* ortValue = nullptr;
  OrtStatus* status = api->KernelInfoGetConstantInput_tensor(ort_info, static_cast<size_t>(input_index), &isConstant, &ortValue);
  if (status) { api->ReleaseStatus(status); return false; }
  if (!isConstant || !ortValue) return false;

  if (!ortValue->IsTensor()) return false;
  *constant_input_value = ortValue;
  return true;
}

}  // namespace onnxruntime
