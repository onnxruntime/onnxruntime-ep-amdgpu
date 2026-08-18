// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

struct OrtApi;
struct OrtApiBase;

// Lowest ORT API version this build can run against. Every ABI struct and API
// call in the plugin stays within this version's surface, so the plugin can be
// built against newer ORT headers yet still load on an older runtime.
inline constexpr uint32_t kMinOrtApiVersion{24};

// ORT API version each optional post-v24 feature first appears in. Gate the wiring
// of an inbound OrtEp callback (or any outbound API call) on
// NegotiatedOrtApiVersion() >= the matching constant so the feature is only used
// when the running runtime actually supports it.
inline constexpr uint32_t kProfilingApiVersion{25};       // OrtEp::CreateProfiler, OrtEpProfilerImpl, OrtEpApi::CreateProfilingEvent/ProfilingEventsContainer_AddEvents
inline constexpr uint32_t kOpSchemaApiVersion{25};        // OrtEpApi::GetOpSchema + OpSchema_*/OpSchemaTypeConstraint_* getters
inline constexpr uint32_t kThreadPoolApiVersion{25};      // SetPerSessionThreadPoolCallbacks + OrtThreadPoolCallbacksConfig
inline constexpr uint32_t kGraphicsInteropApiVersion{25}; // Init/DeinitGraphicsInteropForEpDevice + OrtGraphicsInteropConfig
inline constexpr uint32_t kGraphCaptureApiVersion{26};    // OrtEp::{IsGraphCaptureEnabled,IsGraphCaptured,ReplayGraph,GetGraphCaptureNodeAssignmentPolicy,GetAvailableResource}
inline constexpr uint32_t kSessionInitEndApiVersion{27};  // OrtEp::OnSessionInitializationEnd
inline constexpr uint32_t kDefaultMemoryDeviceApiVersion{27}; // OrtEp::GetDefaultMemoryDevice
inline constexpr uint32_t kReleaseCapturedGraphApiVersion{27}; // OrtEp::ReleaseCapturedGraph

// The ORT API version negotiated at plugin load. NegotiateOrtApi() runs once per
// plugin DLL from CreateEpFactories, before any ABI object is constructed, and
// stores the highest version supported by both this build and the loaded runtime.
// Every OrtEp*/OrtAllocator/... struct stamps this value into ort_version_supported
// so ORT reads only the fields valid for the running runtime. Defaults to
// ORT_API_VERSION until negotiation runs.
uint32_t NegotiatedOrtApiVersion() noexcept;

// Probe GetApi() from ORT_API_VERSION down to min_version; store and return the
// first supported OrtApi*. Returns nullptr (storing 0) if the runtime is older
// than min_version.
const OrtApi* NegotiateOrtApi(const OrtApiBase& ort_api_base, uint32_t min_version) noexcept;
