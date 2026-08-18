// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifdef _WIN32

#include <windows.h>

#include <google/protobuf/message_lite.h>

// The plugin EPs statically link protobuf and ONNX generated descriptors.
// Protobuf keeps its shutdown registry for process lifetime unless the owner
// explicitly releases it. Clean up this DLL-private protobuf instance when
// FreeLibrary dynamically unloads the EP, which App Verifier checks.
BOOL APIENTRY DllMain(HMODULE /*module*/, DWORD reason, LPVOID reserved)
{
    if (reason == DLL_PROCESS_DETACH && reserved == nullptr)
    {
        ::google::protobuf::ShutdownProtobufLibrary();
    }

    return TRUE;
}

#endif // _WIN32
