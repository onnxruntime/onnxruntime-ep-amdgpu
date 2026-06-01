// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dml_info.h"
#include "provider_options_utils.h"
#include "parse_string.h"

namespace dml_ep {

namespace provider_option {
constexpr auto kDeviceId = "device_id"sv;
}  // namespace provider_option

Info::Info(const ProviderOptions& provider_options) {
}

}  // namespace dml_ep
