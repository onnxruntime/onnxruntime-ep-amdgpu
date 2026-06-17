// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/dynamic_library.h"
#include "common/plugin_ep_utils.h"

namespace {

// The HIP EP ships as a separate prebuilt DLL. hipep-backend is a thin
// pass-through: it loads that DLL, forwards CreateEpFactories, and hands back
// its OrtEpFactory unchanged so the parent EP forwards to it directly. This
// presents a repo-built backend parallel to directml/migraphx while keeping the
// HIP EP external. The DLL search directory is set by whoever loads this shim
// (the AMD GPU umbrella, or the executable that registers it).
constexpr auto hipepLib{LIBRARY_PREFIX ORT_TSTR("hipep") LIBRARY_SUFFIX};

using CreateEpFactories_t = OrtStatus* (*)(const char*, const OrtApiBase*, const OrtLogger*,
    OrtEpFactory**, size_t, size_t*);
using ReleaseEpFactory_t = OrtStatus* (*)(OrtEpFactory*);

// hipep-backend only ever creates a single factory, so a single module handle
// plus release pointer is enough.
void* g_hipep_module{};
ReleaseEpFactory_t g_hipep_release{};

}  // namespace

extern "C" {

OrtStatus* CreateEpFactories(const char* registration_name, const OrtApiBase* ort_api_base,
    const OrtLogger* default_logger, OrtEpFactory** factories, size_t max_factories, size_t* num_factories)
{
    try {
        const OrtApi* ort_api{ort_api_base->GetApi(ORT_API_VERSION)};
        Ort::InitApi(ort_api);

        THROW_IF_ERROR(LoadDynamicLibrary(hipepLib, &g_hipep_module));

        CreateEpFactories_t hipep_create{};
        THROW_IF_ERROR(GetSymbolFromLibrary(g_hipep_module, "CreateEpFactories",
            reinterpret_cast<void**>(&hipep_create)));
        THROW_IF_ERROR(GetSymbolFromLibrary(g_hipep_module, "ReleaseEpFactory",
            reinterpret_cast<void**>(&g_hipep_release)));

        return hipep_create(registration_name, ort_api_base, default_logger,
            factories, max_factories, num_factories);
    }
    catch (const std::exception& e) {
        RETURN_STATUS(ORT_EP_FAIL, e.what());
    }
}

OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) {
    OrtStatus* status = (g_hipep_release != nullptr) ? g_hipep_release(factory) : nullptr;
    if (g_hipep_module != nullptr) {
        UnloadDynamicLibrary(g_hipep_module);
        g_hipep_module = nullptr;
    }
    return status;
}

}  // extern "C"
