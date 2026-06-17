// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <string_view>

namespace platform {

std::string GetEnvironmentVar(std::string_view name);

void SetEnvironmentVar(std::string_view name, std::string_view value);

}  // namespace platform