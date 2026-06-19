// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>

#include "common/path_string.h"
#include "common/telemetry.h"

namespace telemetry {

namespace {

constexpr off_t kLockOffset = 0x7FFFFFFFFFFFFFFEll;

struct Fd {
    int fd{-1};
    ~Fd() {
        if (fd >= 0) {
            ::close(fd);
        }
    }
    explicit operator bool() const { return fd >= 0; }
};

}  // namespace

PathString detail::BaseDirectory() noexcept try {
    const char* base_env = std::getenv("AMD_GPUEP_TELEMETRY_DIR");
    return (base_env != nullptr && *base_env != '\0') ? PathString{base_env} : PathString{"/var/log"};
} catch (...) {
    return {};
}

std::string CurrentProcessName() noexcept try {
    std::ifstream comm{"/proc/self/comm"};
    std::string name;
    std::getline(comm, name);
    return name;
} catch (...) {
    return {};
}

std::string ParentProcessName() noexcept try {
    std::ifstream comm{"/proc/" + std::to_string(::getppid()) + "/comm"};
    std::string name;
    std::getline(comm, name);
    return name;
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

    const Fd file{::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644)};
    if (!file) {
        return false;
    }

    struct flock fl{};
    fl.l_type = F_WRLCK;
    fl.l_whence = SEEK_SET;
    fl.l_start = kLockOffset;
    fl.l_len = 1;

    bool locked = false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{kLockTimeoutMs};
    do {
        if (::fcntl(file.fd, F_SETLK, &fl) == 0) {
            locked = true;
            break;
        }
        if (errno != EACCES && errno != EAGAIN) {
            return false;  // a real error, not contention
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    } while (true);

    if (!locked) {
        return false;
    }

    // O_APPEND makes the write atomically land at end-of-file.
    const ssize_t written = ::write(file.fd, buffer.data(), buffer.size());
    const bool ok = written == static_cast<ssize_t>(buffer.size());

    fl.l_type = F_UNLCK;
    ::fcntl(file.fd, F_SETLK, &fl);
    return ok;
} catch (...) {
    return false;
}

}  // namespace telemetry
