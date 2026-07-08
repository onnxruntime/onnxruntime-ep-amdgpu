// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <atomic>

#include "common/dynamic_library.h"
#include "common/plugin_ep_utils.h"

namespace {

// The HIP EP ships as a separate prebuilt DLL. hip-backend is a thin
// pass-through: it loads that DLL, forwards CreateEpFactories, and hands back
// its OrtEpFactory unchanged so the parent EP forwards to it directly. This
// presents a repo-built backend parallel to directml/migraphx while keeping the
// HIP EP external. The DLL search directory is set by whoever loads this shim
// (the AMD GPU umbrella, or the executable that registers it).
constexpr auto hipLib{LIBRARY_PREFIX ORT_TSTR("hipgpu") LIBRARY_SUFFIX};

using CreateEpFactories_t = OrtStatus* (*)(const char*, const OrtApiBase*, const OrtLogger*,
    OrtEpFactory**, size_t, size_t*);
using ReleaseEpFactory_t = OrtStatus* (*)(OrtEpFactory*);

// CreateEpFactories may be called more than once (e.g. a real session plus a
// device-init session in OGA). Reference-count the underlying HIP EP DLL so it
// is loaded on the first factory and unloaded only after the last is released.
void* g_hip_module{};
CreateEpFactories_t g_hip_create{};
ReleaseEpFactory_t g_hip_release{};
std::atomic<size_t> g_refcount{0};

}  // namespace

extern "C" {

OrtStatus* CreateEpFactories(const char* registration_name, const OrtApiBase* ort_api_base,
    const OrtLogger* default_logger, OrtEpFactory** factories, size_t max_factories, size_t* num_factories)
{
    try {
        const OrtApi* ort_api{ort_api_base->GetApi(ORT_API_VERSION)};
        Ort::InitApi(ort_api);

        if (g_refcount++ == 0) {
            THROW_IF_ERROR(LoadDynamicLibrary(hipLib, &g_hip_module));
            THROW_IF_ERROR(GetSymbolFromLibrary(g_hip_module, "CreateEpFactories",
                reinterpret_cast<void**>(&g_hip_create)));
            THROW_IF_ERROR(GetSymbolFromLibrary(g_hip_module, "ReleaseEpFactory",
                reinterpret_cast<void**>(&g_hip_release)));
        }

        return g_hip_create(registration_name, ort_api_base, default_logger,
            factories, max_factories, num_factories);
    }
    catch (const std::exception& e) {
        // Roll back the refcount bump so a later retry re-attempts the load.
        --g_refcount;
        RETURN_STATUS(ORT_EP_FAIL, e.what());
    }
}

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
    OrtStatus* status = (g_hip_release != nullptr) ? g_hip_release(factory) : nullptr;
    if (--g_refcount == 0 && g_hip_module != nullptr) {
        UnloadDynamicLibrary(g_hip_module);
        g_hip_module = nullptr;
        g_hip_create = nullptr;
        g_hip_release = nullptr;
    }
    return status;
}

}  // extern "C"
