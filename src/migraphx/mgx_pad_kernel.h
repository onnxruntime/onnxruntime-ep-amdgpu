// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime_api.h>

// Deliberately free of ONNX Runtime / MIGraphX headers: mgx_pad_kernel.hip is
// compiled by hipcc as device code and must not drag the host-only EP headers
// through the device compiler.  MGX_EP_HAVE_PAD_KERNEL is defined by CMake only
// when the HIP language is available; without it the declaration below is still
// visible but unimplemented, so every use must be guarded by the macro.

namespace mgx_ep {

// One input to place into its arena slot.  `src` holds `real_rows` rows (device
// memory).  The kernel copies those rows to `dst` and replicates the last one
// across rows [real_rows, target_rows).  When real_rows == target_rows the kernel
// is only the copy.
struct PadRowDesc {
    char* dst{nullptr};
    const char* src{nullptr};
    std::uint32_t row_bytes{};
    std::uint32_t real_rows{};
    std::uint32_t target_rows{};
};

// Place every descriptor's real rows into its arena slot and replicate the last
// real row across the pad rows, in ONE kernel launch.  `descs` must point at
// desc_count descriptors in DEVICE memory.  Enqueued on `stream`, so it is ordered
// behind the H2D that filled `src`.  Returns the launch status; the caller decides
// how to escalate it.
hipError_t LaunchPadReplicateRows(const PadRowDesc* descs, unsigned desc_count,
    hipStream_t stream);

}  // namespace mgx_ep
