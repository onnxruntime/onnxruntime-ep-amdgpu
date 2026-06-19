// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "common/path_string.h"

// Lightweight, fire-and-forget telemetry for the AMD GPU EP.
//
// A single line is appended per record to a shared log file. The file may be
// written concurrently by many processes and threads, so every append takes an
// exclusive lock on the file, writes, and releases the lock. If the lock cannot
// be acquired within a short timeout, or the file cannot be opened, the record
// is dropped silently: telemetry must never affect the caller. Nothing here
// throws.
//
// Log location (POC): %ProgramData%\AMD\GPUEP\telemetry.log on Windows. This is
// machine-wide and writable by normal user processes. The final location is to
// be confirmed (see kUseProgramFiles below); switch to %ProgramFiles% once the
// installer grants write access there.
namespace telemetry {

// When true, log under %ProgramFiles%\AMD\GPUEP instead of %ProgramData%.
// Kept here so the location decision can flip with a one-line change after the
// installer/ACL question is settled. NOTE: %ProgramFiles% is not writable by
// non-elevated processes unless the installer grants write access via an ACL;
// otherwise appends fail and are silently dropped (by design).
inline constexpr bool kUseProgramFiles = true;

// Subdirectory and file name under the chosen base folder.
inline constexpr std::string_view kVendorSubdir = "AMD";
inline constexpr std::string_view kProductSubdir = "GPUEP";
inline constexpr std::string_view kLogFileName = "telemetry.log";

// Schema version of the formatted record line. Bump when the field set or
// format changes so a consumer can adapt.
inline constexpr int kSchemaVersion = 1;

// One telemetry entry. Every field is optional; absent fields are omitted from
// the formatted line. Strings should already be sanitized by the caller (no
// newlines); Format() defensively strips any that slip through.
struct Record {
    std::optional<std::string> ep_version;        // e.g. "1.0.0"
    std::optional<std::string> backend;           // "MIGraphX" | "DirectML" | ...
    std::optional<std::string> gfx_arch;          // e.g. "gfx1100"
    std::optional<std::string> model_name;        // file name only, not full path
    std::optional<bool>        loaded_from_cache; // true if loaded from MXR cache
    std::optional<std::string> parent_process;    // parent process image name

    // Build a single newline-terminated line:
    //   <ISO-8601-UTC> v=1 ep_ver=... backend=... gfx=... model=... mxr_cache=... parent=...
    // Never throws.
    std::string Format() const;
};

// Absolute path to the telemetry log file, creating the containing directory if
// needed. Returns an empty path on any failure. Platform-specific.
PathString LogFilePath() noexcept;

// Image file name of the parent (launching) process, e.g. "python.exe".
// Returns empty if it cannot be determined. Platform-specific.
std::string ParentProcessName() noexcept;

// Image file name of the current process, e.g. "onnxruntime_test.exe".
// Returns empty if it cannot be determined. Platform-specific.
std::string CurrentProcessName() noexcept;

// Format and append a record to the telemetry log. All failures are swallowed.
// Never throws.
void Log(const Record& record) noexcept;

// Append an already-formatted line (a trailing newline is added if missing) to
// the given log file using an exclusive lock with a short acquire timeout.
// Returns true only if the line was durably written. Never throws. Exposed so
// the locking behaviour can be exercised directly by tests.
bool AppendLine(const PathString& path, std::string_view line) noexcept;

// Acquire-lock timeout budget, in milliseconds (per the spec, ~10ms).
inline constexpr unsigned kLockTimeoutMs = 10;

}  // namespace telemetry
