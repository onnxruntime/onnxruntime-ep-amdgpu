// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/path_lib.h"

#include <cassert>
#include <array>
#include <algorithm>

#include "core/common/common.h"
#include <PathCch.h>
#pragma comment(lib, "PathCch.lib")

#ifdef _WIN32
namespace onnxruntime {

namespace {

Ort::Status RemoveFileSpec(PWSTR pszPath, size_t cchPath) {
  assert(pszPath != nullptr && pszPath[0] != L'\0');
  // Remove any trailing backslashes
  auto result = PathCchRemoveBackslash(pszPath, cchPath);
  if (result == S_OK || result == S_FALSE) {
    // Remove any trailing filename
    result = PathCchRemoveFileSpec(pszPath, cchPath);
    if (result == S_OK || result == S_FALSE) {
      // If we wind up with an empty string, turn it into '.'
      if (*pszPath == L'\0') {
        pszPath[0] = L'.';
        pszPath[1] = L'\0';
      }
      return STATUS_OK;
    }
  }
  return ORT_MAKE_STATUS(ORT_FAIL, "unexpected failure");
}

}  // namespace

Ort::Status GetDirNameFromFilePath(const std::basic_string<ORTCHAR_T>& s, std::basic_string<ORTCHAR_T>& ret) {
  if (s.empty()) {
    ret = ORT_TSTR(".");
    return STATUS_OK;
  }

  ret = s;

  // Replace slash to backslash since we use PathCchRemoveBackslash
  std::replace(ret.begin(), ret.end(), ORTCHAR_T('/'), ORTCHAR_T('\\'));

  auto st = onnxruntime::RemoveFileSpec(const_cast<wchar_t*>(ret.data()), ret.length() + 1);
  if (!st.IsOK()) {
    std::ostringstream oss;
    oss << "illegal input path:" << ToUTF8String(s) << ". " << st.GetErrorMessage();
    return ORT_MAKE_STATUS(st.GetErrorCode(), oss.str());
  }
  ret.resize(wcslen(ret.c_str()));
  return STATUS_OK;
}

}  // namespace onnxruntime
#else
namespace onnxruntime {

namespace {

inline std::unique_ptr<char[]> StrDup(const std::string& input) {
  auto buf = std::make_unique<char[]>(input.size() + 1);
  strncpy(buf.get(), input.c_str(), input.size());
  buf[input.size()] = 0;
  return buf;
}

}  // namespace

Ort::Status GetDirNameFromFilePath(const std::basic_string<ORTCHAR_T>& input,
                                      std::basic_string<ORTCHAR_T>& output) {
  auto s = StrDup(input);
  output = dirname(s.get());
  return STATUS_OK;
}

std::string GetLastComponent(const std::string& input) {
  auto s = StrDup(input);
  std::string ret = basename(s.get());
  return ret;
}

}  // namespace onnxruntime
#endif
