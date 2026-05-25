// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>
#include "common/plugin_ep_utils.h"

namespace mgx_ep {

size_t GetNumRegisteredKernels();
Ort::Status CreateKernelRegistry(std::string_view ep_name, void* create_kernel_state, OrtKernelRegistry*& kernel_registry);

}  // namespace mgx_ep
