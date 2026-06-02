// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/plugin_ep_utils.h"

#define INITGUID
#include <dxcore.h>
#undef INITGUID

#include <d3dx12/d3dx12.h>
#include <DirectML.h>

#include <wil/wrl.h>
#include <wrl/implements.h>

namespace dml_ep {
template <typename... T>
using Com = ::Microsoft::WRL::RuntimeClass<::Microsoft::WRL::RuntimeClassFlags<::Microsoft::WRL::ClassicCom>, T...>;

}  // namespace dml_ep