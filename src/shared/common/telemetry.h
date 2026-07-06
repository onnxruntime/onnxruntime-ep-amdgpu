// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "common/path_string.h"

struct OrtEp;

namespace telemetry {

inline constexpr std::string_view kVendorSubdir = "AMD";
inline constexpr std::string_view kProductSubdir = "GPUEP";
inline constexpr std::string_view kLogFileName = "telemetry.log";

inline constexpr int kSchemaVersion = 1;

enum class Backend : int {
    Unknown = 0,
    DirectML = 1,
    MIGraphX = 2,
};

std::string_view ToString(Backend backend) noexcept;

struct BackendData {
    Backend backend{Backend::Unknown};
    bool has_gfx_arch{false};
    char gfx_arch[64]{};
    bool has_loaded_from_cache{false};
    bool loaded_from_cache{false};
};

using GetBackendDataFn = bool (*)(const OrtEp* ep, BackendData* out) noexcept;
inline constexpr std::string_view kGetBackendDataSymbol = "GetBackendTelemetry";

class Record {
public:
    Record& SetEpVersion(std::string_view value);
    Record& SetBackend(Backend value) noexcept;
    Record& SetGfxArch(std::string_view value);
    Record& SetModelName(std::string_view value);
    Record& SetLoadedFromCache(bool value) noexcept;
    Record& SetParentProcess(std::string_view value);

    Record& Merge(const BackendData& data);

    [[nodiscard]] Backend GetBackend() const noexcept { return backend_; }

    [[nodiscard]] std::string Format() const;

private:
    std::optional<std::string> ep_version_;
    Backend backend_{Backend::Unknown};
    std::optional<std::string> gfx_arch_;
    std::optional<std::string> model_name_;
    std::optional<bool> loaded_from_cache_;
    std::optional<std::string> parent_process_;
};

struct Config {
    bool enabled{true};
    PathString directory;
};

class Logger {
public:
    explicit Logger(Config config) : config_{std::move(config)} {}

    [[nodiscard]] bool IsEnabled() const noexcept { return config_.enabled; }

    void Write(const Record& record) const;

private:
    Config config_;
};

PathString BaseDirectory();

std::string CurrentProcessName();
std::string ParentProcessName();

void AppendLine(const PathString& path, std::string_view line);

inline constexpr unsigned kLockMaxAttempts = 64;

}  // namespace telemetry
