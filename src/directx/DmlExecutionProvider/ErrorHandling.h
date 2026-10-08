// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdio>
#include <stdexcept>

#include "common/plugin_ep_utils.h"

// Stable marker embedded in the error message of every GPU device-removal (TDR) failure the EP reports
// to ORT. A removed device is permanently dead and cannot be recovered in-process, so the host must
// detect this and restart the process. The host matches on THIS EXACT TOKEN (not on HRESULT text, which
// is WIL-formatted and may vary), so it must stay stable once hosts depend on it.
#define DIRECTX_DEVICE_REMOVED_MARKER "[DIRECTX_DEVICE_REMOVED]"

// True when an HRESULT is a GPU device-removal / TDR reason (DXGI_ERROR_DEVICE_*). Ground-truth check
// used at throw/guard sites instead of pattern-matching message text.
inline bool IsDeviceRemovedHresult(HRESULT hr) noexcept {
    return hr == DXGI_ERROR_DEVICE_REMOVED   // 0x887A0005
        || hr == DXGI_ERROR_DEVICE_HUNG      // 0x887A0006
        || hr == DXGI_ERROR_DEVICE_RESET     // 0x887A0007
        || hr == DXGI_ERROR_DRIVER_INTERNAL_ERROR; // 0x887A0020
}

constexpr HRESULT ErrorCodeToHRESULT(OrtErrorCode error_code) noexcept {
    switch (error_code) {
    case ORT_OK:
        return S_OK;
    case ORT_FAIL:
    case ORT_RUNTIME_EXCEPTION:
    case ORT_ENGINE_ERROR:
        return E_FAIL;
    case ORT_INVALID_ARGUMENT:
        return E_INVALIDARG;
    case ORT_NO_SUCHFILE:
    case ORT_NO_MODEL:
        return HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND);
    case ORT_INVALID_GRAPH:
    case ORT_INVALID_PROTOBUF:
        return HRESULT_FROM_WIN32(ERROR_FILE_CORRUPT);
    case ORT_EP_FAIL:
    case ORT_MODEL_LOADED:
        return HRESULT_FROM_WIN32(ERROR_INTERNAL_ERROR);
    case ORT_NOT_IMPLEMENTED:
        return E_NOTIMPL;
    case ORT_MODEL_LOAD_CANCELED:
        return HRESULT_FROM_WIN32(ERROR_CANCELLED);
    case ORT_MODEL_REQUIRES_COMPILATION:
        return HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED);
    case ORT_NOT_FOUND:
        return HRESULT_FROM_WIN32(ERROR_NOT_FOUND);
    default:
        return E_FAIL;
    }
}

#ifdef ORT_NO_EXCEPTIONS
#define ORT_CATCH_RETURN
#else
#define ORT_CATCH_RETURN CATCH_RETURN()
#endif

#ifdef ORT_NO_EXCEPTIONS
#define ORT_CATCH_GENERIC ORT_CATCH(...) 
#else
#define ORT_CATCH_GENERIC catch(...)
#endif

#ifdef ORT_NO_EXCEPTIONS
#define THROW_IF_NOT_OK(status)   \
    do {                          \
        auto _status = status;    \
        if (!_status.IsOK()) {    \
            ORT_THROW(status);    \
        }                         \
    } while (0);
#else
#define THROW_IF_NOT_OK(status)                                                                                 \
    do {                                                                                                        \
        auto _status = status;                                                                                  \
        if (!_status.IsOK())                                                                                    \
        {                                                                                                       \
             THROW_HR(ErrorCodeToHRESULT(_status.GetErrorCode()));                                              \
        }                                                                                                       \
    } while (0)
#endif

#ifdef ORT_NO_EXCEPTIONS
#define ORT_THROW_IF_FAILED(hr) \
    if(!SUCCEEDED(hr))          \
    {                           \
        ORT_THROW(hr);          \
    }
#else
#define ORT_THROW_IF_FAILED(hr) THROW_IF_FAILED(hr)
#endif

#ifdef ORT_NO_EXCEPTIONS
#define ORT_THROW_LAST_ERROR_IF_NULL(ptr)   \
    if(ptr == nullptr)                      \
    {                                       \
        ORT_THROW(E_POINTER);               \
    }
#else
#define ORT_THROW_LAST_ERROR_IF_NULL(ptr) THROW_LAST_ERROR_IF_NULL(ptr)
#endif

#ifdef ORT_NO_EXCEPTIONS
#define ORT_THROW_HR(hr) ORT_THROW(hr)
#else
#define ORT_THROW_HR(hr) THROW_HR(hr)
#endif

#ifdef ORT_NO_EXCEPTIONS
#define ORT_THROW_HR_IF(hr, condition) ORT_ENFORCE(!(condition), hr)
#else
#define ORT_THROW_HR_IF(hr, condition) THROW_HR_IF(hr, condition)
#endif

#ifdef ORT_NO_EXCEPTIONS
#define ORT_THROW_LAST_ERROR_IF(condition) ORT_ENFORCE(!(condition))
#else
#define ORT_THROW_LAST_ERROR_IF(condition) THROW_LAST_ERROR_IF(condition) 
#endif

#ifdef ORT_NO_EXCEPTIONS
#define ORT_THROW_HR_IF_NULL_MSG(hr, ptr, fmt, ...)     \
    if(ptr == nullptr)                                  \
    {                                                   \
        ORT_THROW(hr);                                  \
    }
#else
#define ORT_THROW_HR_IF_NULL_MSG(hr, ptr, fmt, ...) THROW_HR_IF_NULL_MSG(hr, ptr, fmt, __VA_ARGS__)
#endif

// Throw a device-removal (TDR) failure whose message carries DIRECTX_DEVICE_REMOVED_MARKER and the HRESULT,
// so every noexcept-boundary guard that forwards e.what() propagates the stable marker to the host. Use
// this in place of ORT_THROW_IF_FAILED for a known device-removed HRESULT. A std::runtime_error is thrown
// (not ORT_THROW) so this header needs no dependency on common.h; the existing catch(const std::exception&)
// branches pick it up unchanged.
[[noreturn]] inline void ThrowDeviceRemoved(HRESULT hr) {
    char buf[96];
    std::snprintf(buf, sizeof(buf),
        DIRECTX_DEVICE_REMOVED_MARKER " GPU device removed (TDR), HRESULT 0x%08X",
        static_cast<uint32_t>(hr));
    throw std::runtime_error(buf);
}

// If hr is a device-removed reason, throw the marked exception; otherwise fall back to the normal
// HRESULT throw. Drop-in replacement for ORT_THROW_IF_FAILED at device-removed-reason check sites.
inline void ThrowIfDeviceRemovedOrFailed(HRESULT hr) {
    if (IsDeviceRemovedHresult(hr)) {
        ThrowDeviceRemoved(hr);
    }
    ORT_THROW_IF_FAILED(hr);
}
