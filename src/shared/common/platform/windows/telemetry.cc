// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shlobj.h>
#include <tlhelp32.h>

#include <filesystem>
#include <string>

#include "common/path_string.h"
#include "common/telemetry.h"

namespace fs = std::filesystem;

namespace telemetry {

namespace {

// We gate concurrent appends with a byte-range lock on a single sentinel byte
// placed far beyond any realistic file size. Locking a region that never
// overlaps real data keeps Windows' mandatory locks from blocking a concurrent
// reader (e.g. the telemetry consumer) while a writer holds the lock.
constexpr DWORD kLockOffsetLow = 0xFFFFFFFFu;
constexpr DWORD kLockOffsetHigh = 0x7FFFFFFFu;
constexpr DWORD kLockBytesLow = 1u;
constexpr DWORD kLockBytesHigh = 0u;

// RAII handle wrapper.
struct Handle {
    HANDLE h{INVALID_HANDLE_VALUE};
    ~Handle() {
        if (h != INVALID_HANDLE_VALUE) {
            ::CloseHandle(h);
        }
    }
    explicit operator bool() const { return h != INVALID_HANDLE_VALUE; }
};

PathString KnownFolder(REFKNOWNFOLDERID id) {
    PWSTR raw = nullptr;
    if (FAILED(::SHGetKnownFolderPath(id, KF_FLAG_DEFAULT, nullptr, &raw)) || raw == nullptr) {
        if (raw != nullptr) {
            ::CoTaskMemFree(raw);
        }
        return {};
    }
    PathString path{raw};
    ::CoTaskMemFree(raw);
    return path;
}

std::string ImageNameForPid(DWORD pid) {
    const Handle snapshot{::CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)};
    if (!snapshot) {
        return {};
    }
    PROCESSENTRY32W entry{};
    entry.dwSize = sizeof(entry);
    for (BOOL ok = ::Process32FirstW(snapshot.h, &entry); ok; ok = ::Process32NextW(snapshot.h, &entry)) {
        if (entry.th32ProcessID == pid) {
            return ToUTF8String(std::wstring_view{entry.szExeFile});
        }
    }
    return {};
}

DWORD ParentPidOf(DWORD pid) {
    const Handle snapshot{::CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)};
    if (!snapshot) {
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

}  // namespace

PathString LogFilePath() noexcept try {
    const PathString base = KnownFolder(kUseProgramFiles ? FOLDERID_ProgramFiles : FOLDERID_ProgramData);
    if (base.empty()) {
        return {};
    }
    const fs::path dir = fs::path{base} / ToPathString(kVendorSubdir) / ToPathString(kProductSubdir);
    std::error_code ec;
    fs::create_directories(dir, ec);  // ignore ec: open below fails gracefully if missing
    return (dir / ToPathString(kLogFileName)).native();
} catch (...) {
    return {};
}

std::string CurrentProcessName() noexcept try {
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
} catch (...) {
    return {};
}

std::string ParentProcessName() noexcept try {
    const DWORD parent = ParentPidOf(::GetCurrentProcessId());
    return parent == 0 ? std::string{} : ImageNameForPid(parent);
} catch (...) {
    return {};
}

bool AppendLine(const PathString& path, std::string_view line) noexcept try {
    if (path.empty() || line.empty()) {
        return false;
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
    if (!file) {
        return false;
    }

    // LockFileEx has no native timeout, so poll with LOCKFILE_FAIL_IMMEDIATELY
    // until the ~10ms budget elapses. The OVERLAPPED carries the lock offset.
    OVERLAPPED overlapped{};
    overlapped.Offset = kLockOffsetLow;
    overlapped.OffsetHigh = kLockOffsetHigh;

    bool locked = false;
    const ULONGLONG deadline = ::GetTickCount64() + kLockTimeoutMs;
    do {
        if (::LockFileEx(file.h, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0,
                kLockBytesLow, kLockBytesHigh, &overlapped)) {
            locked = true;
            break;
        }
        if (::GetTickCount64() >= deadline) {
            break;
        }
        ::Sleep(1);
    } while (true);

    if (!locked) {
        return false;  // timed out waiting for the lock; drop the record
    }

    // Append at end-of-file under the lock. Seeking + writing is race-free
    // because every writer serializes through the same lock byte.
    bool ok = false;
    if (::SetFilePointer(file.h, 0, nullptr, FILE_END) != INVALID_SET_FILE_POINTER) {
        DWORD written = 0;
        ok = ::WriteFile(file.h, buffer.data(), static_cast<DWORD>(buffer.size()), &written, nullptr)
             && written == buffer.size();
    }

    ::UnlockFileEx(file.h, 0, kLockBytesLow, kLockBytesHigh, &overlapped);
    return ok;
} catch (...) {
    return false;
}

}  // namespace telemetry
