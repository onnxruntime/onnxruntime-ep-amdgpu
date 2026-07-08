// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <tlhelp32.h>

#include <filesystem>
#include <string>
#include <system_error>
#include <thread>

#include "common/path_string.h"
#include "common/platform/windows/env_var.h"
#include "common/telemetry.h"

namespace fs = std::filesystem;

namespace telemetry {

namespace {

constexpr DWORD kLockOffsetLow = 0xFFFFFFFFu;
constexpr DWORD kLockOffsetHigh = 0x7FFFFFFFu;
constexpr DWORD kLockBytesLow = 1u;
constexpr DWORD kLockBytesHigh = 0u;

// RAII wrapper for a Win32 HANDLE. Handles both failure sentinels: CreateFileW
// yields INVALID_HANDLE_VALUE, OpenProcess yields nullptr.
struct Handle {
    HANDLE h{INVALID_HANDLE_VALUE};

    Handle() = default;
    explicit Handle(HANDLE handle) : h{handle} {}
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    ~Handle() {
        if (IsValid()) {
            ::CloseHandle(h);
        }
    }

    [[nodiscard]] bool IsValid() const noexcept {
        return h != nullptr && h != INVALID_HANDLE_VALUE;
    }
};

std::string ImageNameForPid(DWORD pid) {
    const Handle proc{::OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid)};
    if (!proc.IsValid()) {
        return {};
    }
    wchar_t buffer[MAX_PATH];
    DWORD size = static_cast<DWORD>(std::size(buffer));
    if (!::QueryFullProcessImageNameW(proc.h, 0, buffer, &size)) {
        return {};
    }
    return ToUTF8String(fs::path{std::wstring_view{buffer, size}}.filename().wstring());
}

DWORD ParentPidOf(DWORD pid) {
    const Handle snapshot{::CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)};
    if (!snapshot.IsValid()) {
        return 0;
    }
    PROCESSENTRY32W entry{};
    entry.dwSize = sizeof(entry);
    for (BOOL ok = ::Process32FirstW(snapshot.h, &entry); ok; ok = ::Process32NextW(snapshot.h, &entry)) {
        if (entry.th32ProcessID == pid) {
            return entry.th32ParentProcessID;
        }
    }
    return 0;
}

void AcquireAppendLock(HANDLE file, OVERLAPPED& overlapped) {
    for (unsigned attempt = 1; attempt <= kLockMaxAttempts; ++attempt) {
        if (::LockFileEx(file, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0,
                kLockBytesLow, kLockBytesHigh, &overlapped)) {
            return;
        }
        const DWORD err = ::GetLastError();
        if (err != ERROR_LOCK_VIOLATION && err != ERROR_IO_PENDING) {
            throw std::system_error(static_cast<int>(err), std::system_category(),
                "telemetry: LockFileEx failed");
        }
        std::this_thread::yield();
    }
    throw std::system_error(static_cast<int>(ERROR_LOCK_VIOLATION), std::system_category(),
        "telemetry: log file lock still contended after " + std::to_string(kLockMaxAttempts) + " attempts");
}

}  // namespace

PathString BaseDirectory() {
    // Per-user, low-integrity-writable location: writable without elevation and
    // even from sandboxed / Low-IL host processes (browser renderers, etc.), where
    // machine-wide %ProgramData% is not writable. There is no environment variable
    // for LocalLow, so derive it from the user profile.
    const std::string profile = platform::GetEnvironmentVar("USERPROFILE");
    if (profile.empty()) {
        return {};
    }
    return (fs::path{ToPathString(profile)} / L"AppData" / L"LocalLow").native();
}

std::string CurrentProcessName() {
    std::wstring buffer(MAX_PATH, L'\0');
    for (;;) {
        const DWORD len = ::GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (len == 0) {
            return {};
        }
        if (len < buffer.size()) {
            buffer.resize(len);
            break;
        }
        buffer.resize(buffer.size() * 2);
    }
    return ToUTF8String(fs::path{buffer}.filename().wstring());
}

std::string ParentProcessName() {
    const DWORD parent = ParentPidOf(::GetCurrentProcessId());
    return parent == 0 ? std::string{} : ImageNameForPid(parent);
}

void AppendLine(const PathString& path, std::string_view line) {
    if (path.empty() || line.empty()) {
        return;
    }
    std::string buffer{line};
    if (buffer.back() != '\n') {
        buffer.push_back('\n');
    }

    // Share read/write so other processes can also open the file; the byte-range
    // lock below provides the actual mutual exclusion. OPEN_ALWAYS creates it on
    // first use.
    const Handle file{::CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr)};
    if (!file.IsValid()) {
        throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
            "telemetry: cannot open log file");
    }

    OVERLAPPED overlapped{};
    overlapped.Offset = kLockOffsetLow;
    overlapped.OffsetHigh = kLockOffsetHigh;

    AcquireAppendLock(file.h, overlapped);
    struct Unlock {
        HANDLE h;
        OVERLAPPED* ov;
        ~Unlock() { ::UnlockFileEx(h, 0, kLockBytesLow, kLockBytesHigh, ov); }
    } unlock{file.h, &overlapped};

    // Append at end-of-file under the lock. Seeking + writing is race-free
    // because every writer serializes through the same lock byte.
    if (::SetFilePointer(file.h, 0, nullptr, FILE_END) == INVALID_SET_FILE_POINTER) {
        throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
            "telemetry: seek to end failed");
    }
    DWORD written = 0;
    if (!::WriteFile(file.h, buffer.data(), static_cast<DWORD>(buffer.size()), &written, nullptr)
            || written != buffer.size()) {
        throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
            "telemetry: write failed");
    }
}

}  // namespace telemetry
