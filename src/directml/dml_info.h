// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "plugin_ep_utils.h"

namespace Dml {

struct Info {
    int device_id{};

    Info() = default;

    explicit Info(const ProviderOptions& provider_options);
};

}  // namespace Dml
