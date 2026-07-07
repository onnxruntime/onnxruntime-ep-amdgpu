// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

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

namespace env_var {
constexpr auto kDisable = "ORT_AMDGPU_EP_TELEMETRY_DISABLE"sv;
}  // namespace env_var

[[nodiscard]] bool GloballyDisabled();

struct Config {
    bool enabled{false};
    bool file{false};
    PathString directory;
};

class FileWriter {
public:
    FileWriter() = default;
    FileWriter(const FileWriter&) = delete;
    FileWriter& operator=(const FileWriter&) = delete;
    ~FileWriter();

    void Enqueue(PathString path, std::string line);

    void Stop();

private:
    void Run();

    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::pair<PathString, std::string>> queue_;
    bool stop_{false};
    bool started_{false};
    std::thread thread_;
};

class Logger {
public:
    Logger(Config config, FileWriter* writer) noexcept
        : config_{std::move(config)}, writer_{writer} {}

    [[nodiscard]] bool IsEnabled() const {
        return config_.enabled && !GloballyDisabled();
    }

    void Write(const Record& record) const;

private:
    Config config_;
    FileWriter* writer_{};
};

PathString BaseDirectory();

std::string CurrentProcessName();
std::string ParentProcessName();

void AppendLine(const PathString& path, std::string_view line);

inline constexpr unsigned kLockMaxAttempts = 64;

}  // namespace telemetry
