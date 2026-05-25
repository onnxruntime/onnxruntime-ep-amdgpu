// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/data_transfer.h"

namespace onnxruntime {

// Data transfer manager, which has all functions registered to copy tensors with different location.
// It's not thread-safe.
class DataTransferManager {
 public:
  DataTransferManager() = default;
  // static DataTransferManager& Instance();

  Ort::Status RegisterDataTransfer(std::unique_ptr<IDataTransfer> data_transfer);
  Ort::Status UnregisterDataTransfer(IDataTransfer* data_transfer);

  const IDataTransfer* GetDataTransfer(const OrtDevice& src_device, const OrtDevice& dst_device) const;

  Ort::Status CopyTensor(const Tensor& src, Tensor& dst) const;
  Ort::Status CopyTensorAsync(const Tensor& src, Tensor& dst, Stream& stream) const;
  Ort::Status CopyTensors(const std::vector<IDataTransfer::SrcDstPair>& src_dst_pairs) const;
#if !defined(DISABLE_SPARSE_TENSORS)
  Ort::Status CopySparseTensor(const SparseTensor& src, SparseTensor& dst) const;
  Ort::Status CopySparseTensors(const std::vector<IDataTransfer::SparseSrcDstPair>& src_dst_pairs) const;
#endif

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DataTransferManager);

  // It's assumed that data transfers in this array have no overlap in terms of copying functionality.
  std::vector<std::unique_ptr<IDataTransfer>> datatransfers_;
};
}  // namespace onnxruntime
