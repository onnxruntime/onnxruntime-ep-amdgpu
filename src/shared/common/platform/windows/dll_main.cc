// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifdef _WIN32

#include <Windows.h>

#include <google/protobuf/message_lite.h>

// DllMain for the AMD plugin EP DLLs (migraphx-ep / directx-ep).
//
// Why this file exists
// --------------------
// These EPs statically link their own copy of protobuf (src/CMakeLists.txt fetches
// protobuf with protobuf_BUILD_SHARED_LIBS=OFF) and pull in the generated onnx
// descriptors through plugin-ep-utils / ort_graph_to_proto.cc. Those descriptors are
// registered by C++ static initializers that run at DLL_PROCESS_ATTACH, and protobuf
// deliberately never frees them -- releasing them is what ShutdownProtobufLibrary()
// is for. onnxruntime.dll's own DllMain shuts down *its* copy of protobuf, not ours.
//
// Leaving them allocated is harmless at process exit, but onnxruntime unloads plugin
// EP DLLs at runtime. LoadPluginOrProviderBridge() (core/session/utils.cc) works out
// whether an EP DLL is a legacy provider bridge or a plugin by calling LoadLibrary
// followed by GetProcAddress("GetProvider"). Plugin EPs do not export that symbol, so
// the probe always fails -- and since commit a2cd643d58 (PR microsoft/onnxruntime#28396,
// shipped in onnxruntime 1.27.0) the failed probe calls FreeLibrary. That probe is the
// only holder of the module at that point, so the loader refcount drops to zero and the
// DLL really unloads while it still owns the protobuf descriptor pool. Application
// Verifier reports this as:
//
//     VERIFIER STOP 0000000000000900: A heap allocation was leaked
//
// which is the Microsoft certification failure tracked by SWDEV-601759.
//
// Same approach, and the same reasoning, as:
//   onnxruntime/core/providers/openvino/openvino_provider_dllmain.cc
//   onnxruntime/core/dll/dllmain.cc
BOOL APIENTRY DllMain(HMODULE /*hModule*/,
                      DWORD ul_reason_for_call,
                      LPVOID lpvReserved) {
  switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
      break;
    case DLL_PROCESS_DETACH:
      // Per the Windows docs: "When handling DLL_PROCESS_DETACH, a DLL should free
      // resources such as heap memory only if the DLL is being unloaded dynamically."
      //
      // lpvReserved == nullptr -> somebody called FreeLibrary (dynamic unload). This is
      //                           the onnxruntime probe path described above, and the
      //                           case Application Verifier checks. Clean up.
      // lpvReserved != nullptr -> the process is terminating. The OS reclaims everything
      //                           and other threads may already be gone, so doing work
      //                           here is pointless at best and unsafe at worst. Skip.
      if (lpvReserved == nullptr) {
        ::google::protobuf::ShutdownProtobufLibrary();
      }
      break;
  }

  return TRUE;
}

#endif  // _WIN32
