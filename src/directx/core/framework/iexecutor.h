// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "core/framework/ort_value.h"

struct OrtValue;
namespace onnxruntime {
class SessionState;
class TensorShape;

class IExecutor {
 public:
  using CustomAllocator = std::function<Ort::Status(const TensorShape&, const OrtDevice&, OrtValue&, bool& allocated)>;

  virtual ~IExecutor() = default;

  /**
   * The lifetime of 'fetches' is limited by 'session_state'
   */
  Ort::Status Execute(const SessionState& session_state,
                         gsl::span<const int> feed_mlvalue_idxs,
                         gsl::span<const OrtValue> feeds,
                         gsl::span<const int> fetch_mlvalue_idxs,
                         std::vector<OrtValue>& fetches,
                         const Ort::Logger& logger) {
    std::unordered_map<size_t, CustomAllocator> fetch_allocators;
    return Execute(session_state, feed_mlvalue_idxs, feeds, fetch_mlvalue_idxs, fetches, fetch_allocators, logger);
  }

  // TODO: as fetch_allocators is optional, it should be a pointer instead of reference
  virtual Ort::Status Execute(const SessionState& session_state, gsl::span<const int> feed_mlvalue_idxs,
                                 gsl::span<const OrtValue> feeds, gsl::span<const int> fetch_mlvalue_idxs,
                                 std::vector<OrtValue>& fetches,
                                 // optional custom allocators. key is index in fetches
                                 const std::unordered_map<size_t, CustomAllocator>& fetch_allocators,
                                 const Ort::Logger& logger) = 0;
};
}  // namespace onnxruntime
