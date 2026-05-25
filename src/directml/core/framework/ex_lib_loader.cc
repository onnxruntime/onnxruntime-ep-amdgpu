// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT

#include "core/framework/ex_lib_loader.h"
#include "core/platform/env.h"

namespace onnxruntime {
ExLibLoader::~ExLibLoader() {
  for (auto& elem : dso_name_data_map_) {
      ExLibLoader::PreUnloadLibrary(elem.second);
    // unload the DSO
    (void)Env::Default().UnloadDynamicLibrary(elem.second);
  }
}

void* ExLibLoader::GetExLibHandle(const std::string& dso_file_path) const {
  auto it = dso_name_data_map_.find(dso_file_path);
  return it == dso_name_data_map_.end() ? nullptr : it->second;
}

Ort::Status ExLibLoader::LoadExternalLib(const std::string& dso_file_path,
                                            void** handle) {
  auto status = STATUS_OK;
  ORT_TRY {
    if (dso_name_data_map_.count(dso_file_path)) {
      return ORT_MAKE_STATUS(ORT_INVALID_ARGUMENT, "A dso with name ", dso_file_path, " has already been loaded.");
    }

    void* lib_handle = nullptr;
    auto path_str = ToPathString(dso_file_path);
    ORT_RETURN_IF_ERROR(Env::Default().LoadDynamicLibrary(path_str, false, &lib_handle));
    dso_name_data_map_[dso_file_path] = lib_handle;
    *handle = lib_handle;
    return STATUS_OK;
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(ORT_FAIL, "Caught exception while loading custom ops with message: ", ex.what());
    });
  }

  return status;
}

}  // namespace onnxruntime
