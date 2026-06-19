// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "common/path_string.h"

namespace telemetry {

inline constexpr bool kUseProgramFiles = false;

// Subdirectory and file name under the chosen base folder.
inline constexpr std::string_view kVendorSubdir = "AMD";
inline constexpr std::string_view kProductSubdir = "GPUEP";
inline constexpr std::string_view kLogFileName = "telemetry.log";

inline constexpr int kSchemaVersion = 1;

struct Record {
    std::optional<std::string> ep_version;
    std::optional<std::string> backend;
    std::optional<std::string> gfx_arch;
    std::optional<std::string> model_name;
    std::optional<bool>        loaded_from_cache;
    std::optional<std::string> parent_process;

    std::string Format() const;
};

namespace detail {
PathString BaseDirectory() noexcept;
}  // namespace detail

PathString LogFilePath() noexcept;

std::string ParentProcessName() noexcept;

std::string CurrentProcessName() noexcept;

void Log(const Record& record) noexcept;

bool AppendLine(const PathString& path, std::string_view line) noexcept;

inline constexpr unsigned kLockTimeoutMs = 10;

}  // namespace telemetry
