// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/telemetry.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <string>
#include <system_error>
#include <utility>

#include <fmt/chrono.h>
#include <fmt/format.h>

#include "common/env_var.h"

namespace telemetry {

namespace {

std::string Sanitize(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const char c : value) {
        out.push_back((c == '\n' || c == '\r' || c == ' ' || c == '\t') ? '_' : c);
    }
    return out;
}

void AppendField(std::string& line, std::string_view key, const std::optional<std::string>& value) {
    if (value) {
        line += fmt::format(" {}={}", key, Sanitize(*value));
    }
}

std::string IsoUtcNow() {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    return fmt::format("{:%Y-%m-%dT%H:%M:%SZ}", fmt::gmtime(now));
}

}  // namespace

std::string_view ToString(Backend backend) noexcept {
    switch (backend) {
        case Backend::DirectML:
            return "DirectML";
        case Backend::MIGraphX:
            return "MIGraphX";
        case Backend::Hip:
            return "Hip";
        case Backend::Unknown:
            break;
    }
    return "unknown";
}

Record& Record::SetEpVersion(std::string_view value) {
    ep_version_ = std::string{value};
    return *this;
}

Record& Record::SetBackend(Backend value) noexcept {
    backend_ = value;
    return *this;
}

Record& Record::SetGfxArch(std::string_view value) {
    gfx_arch_ = std::string{value};
    return *this;
}

Record& Record::SetModelName(std::string_view value) {
    model_name_ = std::string{value};
    return *this;
}

Record& Record::SetLoadedFromCache(bool value) noexcept {
    loaded_from_cache_ = value;
    return *this;
}

Record& Record::SetParentProcess(std::string_view value) {
    parent_process_ = std::string{value};
    return *this;
}

Record& Record::Merge(const BackendData& data) {
    backend_ = data.backend;
    if (data.has_gfx_arch) {
        size_t len = 0;
        while (len < sizeof(data.gfx_arch) && data.gfx_arch[len] != '\0') {
            ++len;
        }
        gfx_arch_ = std::string{data.gfx_arch, len};
    }
    if (data.has_loaded_from_cache) {
        loaded_from_cache_ = data.loaded_from_cache;
    }
    return *this;
}

std::string Record::Format() const {
    std::string line = fmt::format("{} v={}", IsoUtcNow(), kSchemaVersion);
    AppendField(line, "ep_ver", ep_version_);
    if (backend_ != Backend::Unknown) {
        line += fmt::format(" backend={}", ToString(backend_));
    }
    AppendField(line, "gfx", gfx_arch_);
    AppendField(line, "model", model_name_);
    if (loaded_from_cache_) {
        line += fmt::format(" mxr_cache={}", *loaded_from_cache_ ? "hit" : "miss");
    }
    AppendField(line, "parent", parent_process_);
    line.push_back('\n');
    return line;
}

bool GloballyDisabled() {
    std::string value = platform::GetEnvironmentVar(env_var::kDisable);
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    return value == "1" || value == "true" || value == "on";
}

FileWriter::~FileWriter() {
    Stop();
}

void FileWriter::Enqueue(PathString path, std::string line) {
    if (path.empty() || line.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock{mutex_};
    if (stop_) {
        return;
    }
    if (!started_) {
        started_ = true;
        thread_ = std::thread(&FileWriter::Run, this);
    }
    queue_.emplace_back(std::move(path), std::move(line));
    cv_.notify_one();
}

void FileWriter::Stop() {
    {
        std::lock_guard<std::mutex> lock{mutex_};
        stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) {
        thread_.join();
    }
}

void FileWriter::Run() {
    for (;;) {
        std::deque<std::pair<PathString, std::string>> batch;
        {
            std::unique_lock<std::mutex> lock{mutex_};
            cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) {
                return;
            }
            batch.swap(queue_);
        }
        for (const auto& [path, line] : batch) {
            try {
                const std::filesystem::path fs_path{path};
                if (fs_path.has_parent_path()) {
                    std::error_code ec;
                    std::filesystem::create_directories(fs_path.parent_path(), ec);
                }
                AppendLine(path, line);
            } catch (const std::exception&) {
                // Drop this record; telemetry must never escape the worker
            }
        }
    }
}

void Logger::Write(const Record& record) const {
    if (!config_.enabled || GloballyDisabled()) {
        return;
    }
    // The primary sink (ONNX Runtime telemetry/logging API) is added separately,
    // pending confirmation of which ORT API to use
    if (!config_.file || writer_ == nullptr) {
        return;
    }
    const PathString base = config_.directory.empty() ? BaseDirectory() : config_.directory;
    if (base.empty()) {
        return;
    }
    const std::filesystem::path dir = std::filesystem::path{base} / ToPathString(kAppSubdir);
    const PathString path = (dir / ToPathString(kLogFileName)).native();
    writer_->Enqueue(path, record.Format());
}

}  // namespace telemetry
