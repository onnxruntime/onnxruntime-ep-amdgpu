// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/common/cpuid_info.h"

#include <iostream>
#include "core/platform/check_intel.h"

#ifdef __linux__
#if (defined(_M_AMD64) || defined(__x86_64__)) && !defined(__ANDROID__)
#include <x86intrin.h>
#endif
#include <unistd.h>
#include <sys/syscall.h>
#if !defined(__NR_getcpu)
#include <asm-generic/unistd.h>
#endif
#endif  // __linux__

#if _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif  // _WIN32

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__)
#include <cpuid.h>
#endif

namespace onnxruntime {

void CPUIDInfo::LogEarlyWarning(std::string_view) {
}

static inline void GetCPUID(int function_id, int data[4]) {  // NOLINT
#if defined(_MSC_VER)
  __cpuid(data, function_id);
#elif defined(__GNUC__)
  __cpuid(function_id, data[0], data[1], data[2], data[3]);
#endif
}

static inline void GetCPUID(int function_id, int sub_leaf, int data[4]) {  // NOLINT
#if defined(_MSC_VER)
  __cpuidex(data, function_id, sub_leaf);
#elif defined(__GNUC__)
  __cpuid_count(function_id, sub_leaf, data[0], data[1], data[2], data[3]);
#endif
}

static inline int XGETBV() {
#if defined(_MSC_VER)
  return static_cast<int>(_xgetbv(0));
#elif defined(__GNUC__)
  int eax, edx;
  __asm__ volatile("xgetbv"
                   : "=a"(eax), "=d"(edx)
                   : "c"(0));
  return eax;
#endif
}

void CPUIDInfo::X86Init() {
  int data[4] = {-1};
  GetCPUID(0, data);

  int num_IDs = data[0];
  if (num_IDs >= 1) {
    GetCPUID(1, data);
    if (data[2] & (1 << 27)) {
      constexpr int AVX_MASK = 0x6;
      constexpr int AVX512_MASK = 0xE6;
      int value = XGETBV();
      bool has_sse2 = (data[3] & (1 << 26));
      has_sse3_ = (data[2] & 0x1);
      has_sse4_1_ = (data[2] & (1 << 19));
      bool has_ssse3 = (data[2] & (1 << 9));
      has_avx_ = has_sse2 && has_ssse3 && (data[2] & (1 << 28)) && ((value & AVX_MASK) == AVX_MASK);
      bool has_avx512 = (value & AVX512_MASK) == AVX512_MASK;
      has_f16c_ = has_avx_ && (data[2] & (1 << 29)) && (data[3] & (1 << 26));

      if (num_IDs >= 7) {
        // This change is made to overcome the issue of __get_cpuid returning all zeros, instead use __get_cpuid_count.
        // Reference: https://stackoverflow.com/questions/46272579/why-does-get-cpuid-return-all-zeros-for-leaf-4
        GetCPUID(7, 0, data);
        const uint32_t max_SubLeaves = data[0];
        has_amx_bf16_ = (data[3] & (1 << 22));
        has_avx2_ = has_avx_ && (data[1] & (1 << 5));
        has_avx512f_ = has_avx512 && (data[1] & (1 << 16));
        // Add check for AVX512 Skylake since tensorization GEMM need intrinsics from avx512bw/avx512dq.
        // avx512_skylake = avx512f | avx512vl | avx512cd | avx512bw | avx512dq
        has_avx512_skylake_ = has_avx512 && (data[1] & ((1 << 16) | (1 << 17) | (1 << 28) | (1 << 30) | (1 << 31)));
        is_hybrid_ = (data[3] & (1 << 15));
        // Check for TPAUSE
        CheckIntelResult check_intel = CheckIntel();
        if (check_intel.is_intel) {
#ifdef __linux__
#if !defined(__ANDROID__)
          has_tpause_ = __builtin_cpu_supports("waitpkg") != 0;
#endif
#else
          has_tpause_ = (data[2] & (1 << 5)) != 0;
#endif
        }
        if (max_SubLeaves >= 1) {
          GetCPUID(7, 1, data);
          has_avx512_bf16_ = has_avx512 && (data[0] & (1 << 5));
        }
      }
    }
  }
}

uint32_t CPUIDInfo::GetCurrentCoreIdx() {
#ifdef _WIN32
  return GetCurrentProcessorNumber();
#elif defined(__linux__)
  uint32_t coreIdx = 0xFFFFFFFF;
  if (syscall(__NR_getcpu, &coreIdx, NULL, NULL) != 0) {
    // failed to detect current core id. give up
    return 0xFFFFFFFF;
  }
  return coreIdx;
#else
  return 0xFFFFFFFF;  // don't know how to get core index
#endif
}

CPUIDInfo::CPUIDInfo() {
  // Note: This should be run after cpuinfo initialization if cpuinfo is enabled.
  VendorInfoInit();
  X86Init();
}
}  // namespace onnxruntime
