// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <thread>

#include "common/path_string.h"
#include "common/platform/linux/env_var.h"
#include "common/telemetry.h"

namespace fs = std::filesystem;

namespace telemetry {

namespace {

constexpr off_t kLockOffset = 0x7FFFFFFFFFFFFFFEll;

// RAII wrapper for a file descriptor.
struct Fd {
    int fd{-1};

    Fd() = default;
    explicit Fd(int descriptor) : fd{descriptor} {}
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    ~Fd() {
        if (IsValid()) {
            ::close(fd);
        }
    }

    [[nodiscard]] bool IsValid() const noexcept { return fd >= 0; }
};

std::string ReadComm(const std::string& proc_path) {
    std::ifstream comm{proc_path};
    std::string name;
    std::getline(comm, name);
    return name;
}

void AcquireAppendLock(int fd) {
    struct flock fl{};
    fl.l_type = F_WRLCK;
    fl.l_whence = SEEK_SET;
    fl.l_start = kLockOffset;
    fl.l_len = 1;

    for (unsigned attempt = 1; attempt <= kLockMaxAttempts; ++attempt) {
        if (::fcntl(fd, F_SETLK, &fl) == 0) {
            return;
        }
        const int err = errno;
        if (err != EACCES && err != EAGAIN) {
            throw std::system_error(err, std::generic_category(), "telemetry: fcntl(F_SETLK) failed");
        }
        std::this_thread::yield();
    }
    throw std::system_error(EAGAIN, std::generic_category(),
        "telemetry: log file lock still contended after " + std::to_string(kLockMaxAttempts) + " attempts");
}

}  // namespace

PathString BaseDirectory() {
    // Per-user writable location per the XDG Base Directory Specification:
    // $XDG_STATE_HOME when set, otherwise $HOME/.local/state.
    const std::string xdg_state = platform::GetEnvironmentVar("XDG_STATE_HOME");
    if (!xdg_state.empty()) {
        return ToPathString(xdg_state);
    }
    const std::string home = platform::GetEnvironmentVar("HOME");
    if (home.empty()) {
        return {};
    }
    return (fs::path{ToPathString(home)} / ".local" / "state").native();
}

std::string CurrentProcessName() {
    return ReadComm("/proc/self/comm");
}

std::string ParentProcessName() {
    return ReadComm("/proc/" + std::to_string(::getppid()) + "/comm");
}

void AppendLine(const PathString& path, std::string_view line) {
    if (path.empty() || line.empty()) {
        return;
    }
    std::string buffer{line};
    if (buffer.back() != '\n') {
        buffer.push_back('\n');
    }

    const Fd file{::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644)};
    if (!file.IsValid()) {
        throw std::system_error(errno, std::generic_category(), "telemetry: cannot open log file");
    }

    AcquireAppendLock(file.fd);
    struct Unlock {
        int fd;
        ~Unlock() {
            struct flock fl{};
            fl.l_type = F_UNLCK;
            fl.l_whence = SEEK_SET;
            fl.l_start = kLockOffset;
            fl.l_len = 1;
            ::fcntl(fd, F_SETLK, &fl);
        }
    } unlock{file.fd};

    // O_APPEND makes the write atomically land at end-of-file.
    const ssize_t written = ::write(file.fd, buffer.data(), buffer.size());
    if (written < 0 || static_cast<size_t>(written) != buffer.size()) {
        throw std::system_error(errno, std::generic_category(), "telemetry: write failed");
    }
}

}  // namespace telemetry
