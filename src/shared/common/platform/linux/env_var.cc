// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "env_var.h"

#include "parse_string.h"
#include "platform/windows/env_var.h"

namespace platform {
std::string GetEnvironmentVar(const std::string_view name) {
    char* val = getenv(std::string{name}.c_str());
    return val == nullptr ? std::string{} : std::string{val};
}
}  // namespace platform