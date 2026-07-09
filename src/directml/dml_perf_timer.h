// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License

#pragma once

#ifdef DML_PERF_PROFILE
#include <cstdint>
#include "common/make_string.h"

namespace dml_ep {

void DmlPerfWriteLogImpl(std::string_view msg) noexcept;

inline uint64_t PerfNowUs() noexcept {
    LARGE_INTEGER counter, freq;
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&freq);
    return static_cast<uint64_t>(counter.QuadPart * 1000000ULL / freq.QuadPart);
}

}  // namespace dml_ep

#define PERF_TIMER_LOG(...) ::dml_ep::DmlPerfWriteLogImpl(MakeString(__VA_ARGS__))

#else
#define PERF_TIMER_LOG(...)
#endif
