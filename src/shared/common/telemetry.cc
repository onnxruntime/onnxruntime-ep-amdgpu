// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/telemetry.h"

#include <chrono>
#include <string>

#include <fmt/chrono.h>
#include <fmt/format.h>

namespace telemetry {

namespace {

// Replace characters that would break the one-record-per-line format. We use
// space as the field separator, so any embedded whitespace/newline is turned
// into '_' to keep a value as a single token.
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
    // Whole-second ISO-8601 UTC, e.g. 2026-06-19T11:38:02Z.
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    return fmt::format("{:%Y-%m-%dT%H:%M:%SZ}", fmt::gmtime(now));
}

}  // namespace

std::string Record::Format() const try {
    std::string line = fmt::format("{} v={}", IsoUtcNow(), kSchemaVersion);
    AppendField(line, "ep_ver", ep_version);
    AppendField(line, "backend", backend);
    AppendField(line, "gfx", gfx_arch);
    AppendField(line, "model", model_name);
    if (loaded_from_cache) {
        line += fmt::format(" mxr_cache={}", *loaded_from_cache ? "hit" : "miss");
    }
    AppendField(line, "parent", parent_process);
    line.push_back('\n');
    return line;
} catch (...) {
    return {};
}

void Log(const Record& record) noexcept try {
    const PathString path = LogFilePath();
    if (path.empty()) {
        return;
    }
    const std::string line = record.Format();
    if (line.empty()) {
        return;
    }
    AppendLine(path, line);
} catch (...) {
    // Telemetry must never propagate failures to the caller.
}

}  // namespace telemetry
