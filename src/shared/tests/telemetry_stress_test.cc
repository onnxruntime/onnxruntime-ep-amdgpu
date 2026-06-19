// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Stress test for the telemetry append-with-exclusive-lock path.
//
// It launches several child processes; each child spawns several threads; each
// thread appends a batch of uniquely-tagged lines via telemetry::AppendLine.
// The parent then reads the log back and verifies that:
//   * the total number of lines equals processes * threads * iterations,
//   * every line is well-formed (no torn or interleaved writes),
//   * every (process, thread, sequence) tag appears exactly once.
//
// Usage:
//   telemetry-stress-test                  (run the full parent/child test)
//   telemetry-stress-test <procs> <threads> <iters>
//   telemetry-stress-test --child <id> <threads> <iters> <logpath>   (internal)

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/path_string.h"
#include "common/telemetry.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

constexpr char kChildFlag[] = "--child";

PathString SelfPath() {
#ifdef _WIN32
    std::wstring buffer(MAX_PATH, L'\0');
    for (;;) {
        const DWORD len = ::GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (len == 0) {
            return {};
        }
        if (len < buffer.size()) {
            buffer.resize(len);
            return buffer;
        }
        buffer.resize(buffer.size() * 2);
    }
#else
    std::string buffer(4096, '\0');
    const ssize_t len = ::readlink("/proc/self/exe", buffer.data(), buffer.size());
    if (len <= 0) {
        return {};
    }
    buffer.resize(static_cast<size_t>(len));
    return buffer;
#endif
}

// Each child writes lines of the exact form: "STRESS p=<id> t=<thread> s=<seq>"
std::string MakeTag(int proc_id, int thread_id, int seq) {
    return "STRESS p=" + std::to_string(proc_id) + " t=" + std::to_string(thread_id) +
           " s=" + std::to_string(seq);
}

int RunChild(int proc_id, int threads, int iters, const PathString& log_path) {
    std::vector<std::thread> workers;
    std::atomic<int> failures{0};
    workers.reserve(static_cast<size_t>(threads));
    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&, t]() {
            for (int s = 0; s < iters; ++s) {
                if (!telemetry::AppendLine(log_path, MakeTag(proc_id, t, s))) {
                    failures.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& w : workers) {
        w.join();
    }
    if (failures.load() != 0) {
        std::cerr << "[child " << proc_id << "] " << failures.load()
                  << " append(s) dropped (lock timeout / IO)\n";
    }
    return failures.load() == 0 ? 0 : 1;
}

#ifdef _WIN32
using ProcHandle = HANDLE;

ProcHandle SpawnChild(const PathString& self, int proc_id, int threads, int iters,
                      const PathString& log_path) {
    std::wstring cmd = L"\"" + self + L"\" " L"--child" +
        L" " + std::to_wstring(proc_id) + L" " + std::to_wstring(threads) + L" " +
        std::to_wstring(iters) + L" \"" + log_path + L"\"";
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    std::vector<wchar_t> mutable_cmd(cmd.begin(), cmd.end());
    mutable_cmd.push_back(L'\0');
    if (!::CreateProcessW(nullptr, mutable_cmd.data(), nullptr, nullptr, FALSE, 0,
            nullptr, nullptr, &si, &pi)) {
        return nullptr;
    }
    ::CloseHandle(pi.hThread);
    return pi.hProcess;
}

int WaitChild(ProcHandle h) {
    ::WaitForSingleObject(h, INFINITE);
    DWORD code = 1;
    ::GetExitCodeProcess(h, &code);
    ::CloseHandle(h);
    return static_cast<int>(code);
}
#else
using ProcHandle = pid_t;

ProcHandle SpawnChild(const PathString& self, int proc_id, int threads, int iters,
                      const PathString& log_path) {
    const pid_t pid = ::fork();
    if (pid < 0) {
        return -1;
    }
    if (pid == 0) {
        const std::string id = std::to_string(proc_id);
        const std::string th = std::to_string(threads);
        const std::string it = std::to_string(iters);
        ::execl(self.c_str(), self.c_str(), kChildFlag, id.c_str(), th.c_str(),
                it.c_str(), log_path.c_str(), static_cast<char*>(nullptr));
        std::_Exit(127);  // exec failed
    }
    return pid;
}

int WaitChild(ProcHandle pid) {
    int status = 0;
    ::waitpid(pid, &status, 0);
    return (WIFEXITED(status)) ? WEXITSTATUS(status) : 1;
}
#endif

int Verify(const PathString& log_path, int procs, int threads, int iters) {
    std::ifstream in{std::filesystem::path{log_path}};
    if (!in) {
        std::cerr << "FAIL: cannot open log for verification\n";
        return 1;
    }

    const std::regex pattern{R"(^STRESS p=(\d+) t=(\d+) s=(\d+)$)"};
    std::unordered_map<uint64_t, int> seen;
    size_t total = 0;
    size_t malformed = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        ++total;
        std::smatch m;
        if (!std::regex_match(line, m, pattern)) {
            ++malformed;
            if (malformed <= 5) {
                std::cerr << "  malformed/torn line: \"" << line << "\"\n";
            }
            continue;
        }
        const uint64_t key = (static_cast<uint64_t>(std::stoi(m[1])) << 40) |
                             (static_cast<uint64_t>(std::stoi(m[2])) << 20) |
                             static_cast<uint64_t>(std::stoi(m[3]));
        ++seen[key];
    }

    const size_t expected = static_cast<size_t>(procs) * threads * iters;
    size_t duplicates = 0;
    for (const auto& [key, count] : seen) {
        if (count > 1) {
            ++duplicates;
        }
    }

    std::cout << "lines written : " << total << "\n"
              << "expected       : " << expected << "\n"
              << "unique tags    : " << seen.size() << "\n"
              << "malformed/torn : " << malformed << "\n"
              << "duplicated tags: " << duplicates << "\n";

    const bool ok = total == expected && seen.size() == expected && malformed == 0 && duplicates == 0;
    std::cout << (ok ? "PASS" : "FAIL") << "\n";
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    // Child mode: --child <id> <threads> <iters> <logpath>
    if (argc >= 6 && std::string{argv[1]} == kChildFlag) {
        const int proc_id = std::atoi(argv[2]);
        const int threads = std::atoi(argv[3]);
        const int iters = std::atoi(argv[4]);
        return RunChild(proc_id, threads, iters, ToPathString(argv[5]));
    }

    int procs = 8;
    int threads = 4;
    int iters = 250;
    if (argc >= 4) {
        procs = std::atoi(argv[1]);
        threads = std::atoi(argv[2]);
        iters = std::atoi(argv[3]);
    }

    const PathString self = SelfPath();
    if (self.empty()) {
        std::cerr << "FAIL: cannot resolve own executable path\n";
        return 1;
    }

    const PathString log_path = telemetry::LogFilePath();
    if (log_path.empty()) {
        std::cerr << "FAIL: telemetry::LogFilePath() returned empty (no writable location?)\n";
        return 1;
    }

    // Start from a clean file.
    {
        std::error_code ec;
        std::filesystem::remove(std::filesystem::path{log_path}, ec);
    }

    std::cout << "telemetry stress test\n"
              << "  log      : " << std::filesystem::path{log_path}.string() << "\n"
              << "  processes: " << procs << "\n"
              << "  threads  : " << threads << "\n"
              << "  iters    : " << iters << "\n"
              << "  total    : " << (static_cast<long long>(procs) * threads * iters) << " lines\n\n";

    std::vector<ProcHandle> children;
    children.reserve(static_cast<size_t>(procs));
    for (int p = 0; p < procs; ++p) {
        ProcHandle h = SpawnChild(self, p, threads, iters, log_path);
#ifdef _WIN32
        if (h == nullptr) {
#else
        if (h < 0) {
#endif
            std::cerr << "FAIL: could not spawn child " << p << "\n";
            return 1;
        }
        children.push_back(h);
    }

    int child_failures = 0;
    for (ProcHandle h : children) {
        child_failures += WaitChild(h);
    }
    if (child_failures != 0) {
        std::cerr << "WARNING: " << child_failures
                  << " child process(es) reported dropped appends\n";
    }

    return Verify(log_path, procs, threads, iters);
}
