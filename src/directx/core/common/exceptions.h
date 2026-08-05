// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <exception>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/common/common.h"
#include "core/common/code_location.h"

#include "common/plugin_ep_utils.h"

namespace onnxruntime {

class NotImplementedException : public std::logic_error {
 public:
  explicit NotImplementedException(const char* _Message = "Function not yet implemented") noexcept : std::logic_error(_Message) {};
  explicit NotImplementedException(const std::string& _Message = "Function not yet implemented") noexcept : std::logic_error(_Message) {};
};

class OnnxRuntimeException : public std::exception {
public:
    OnnxRuntimeException(const CodeLocation& location, const std::string& msg) noexcept
        : OnnxRuntimeException(location, nullptr, msg) {
    }

    /**
     Create a new exception that captures the location it was thrown from.
     @param location Location in the source code the exception is being thrown from
     @param msg Message containing additional information about the exception cause.
     @param category Error category
     @param code Error code
  */

  OnnxRuntimeException(const CodeLocation& location,
                       const std::string& message,
                       OrtErrorCode error_code) noexcept
      : OnnxRuntimeException(location, nullptr, message, error_code) {
      }

  /**
     Create a new exception that captures the location it was thrown from.
     The instance will be created with ONNXRUNTIME category and FAIL code.
     @param location Location in the source code the exception is being thrown from
     @param failed_condition Optional string containing the condition that failed.
     e.g. "tensor.Size() == input.Size()". May be nullptr.
     @param msg Message containing additional information about the exception cause.
  */
  OnnxRuntimeException(const CodeLocation& location, const char* failed_condition, const std::string& msg) noexcept
      : OnnxRuntimeException(location, failed_condition, msg, ORT_FAIL) {
      }

    /**
     Create a new exception that captures the location it was thrown from.
     @param location Location in the source code the exception is being thrown from
     @param failed_condition Optional string containing the condition that failed.
     e.g. "tensor.Size() == input.Size()". May be nullptr.
     @param msg Message containing additional information about the exception cause.
     @param error_code Error code
    */
    OnnxRuntimeException(const CodeLocation& location,
                         const char* failed_condition,
                         const std::string& msg,
                         OrtErrorCode error_code) noexcept
        : location_{location}, error_code_{error_code}
    {
        std::ostringstream ss;

        ss << location.ToString(CodeLocation::kFilenameAndPath);  // output full path in case just the filename is ambiguous
        if (failed_condition != nullptr) {
            ss << " " << failed_condition << " was false.";
        }

        ss << " " << msg << "\n";
        what_ = ss.str();
    }

    OrtErrorCode ErrorCode() const noexcept {
        return error_code_;
    }

    const char* what() const noexcept override {
        return what_.c_str();
    }

private:
    const CodeLocation location_;
    const std::vector<std::string> stacktrace_;
    std::string what_;
    OrtErrorCode error_code_;
};

class TypeMismatchException : public std::logic_error {
 public:
  TypeMismatchException() noexcept : logic_error("Type mismatch") {};
};

}  // namespace onnxruntime
