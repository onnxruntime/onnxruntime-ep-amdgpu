// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>

#include "common/path_string.h"
#include "common/plugin_ep_utils.h"

namespace fs = std::filesystem;

namespace mgx_ep {

struct PriorityNodeCompare {
    static bool IsHigherPriority(const Ort::ConstNode& n) {
        const auto op_type{n.GetOperatorType()};
        return op_type == "Shape"sv || op_type == "Size"sv;
    }
    bool operator()(const Ort::ConstNode& n1, const Ort::ConstNode& n2) const {
        const bool isN1HighPri{IsHigherPriority(n1)};
        const bool isN2HighPri{IsHigherPriority(n2)};
        if (isN1HighPri != isN2HighPri) {
            return isN2HighPri;
        }
        return n1.GetId() > n2.GetId();
    }
};

inline fs::path GetCachePath(const fs::path& root, std::string_view name) {
    return root.empty() ? root : root / ToPathString(name);
}

bool ReadDynamicRange(const fs::path& path, bool is_native_calibration_table, Map<float>& dynamic_ranges);
float ConvertSinglePrecisionIEEE754ToFloat(unsigned long value);
bool CanEvalNodeArgument(const Ort::ConstGraph& graph, const Ort::ConstNode& node, const std::vector<size_t>& indices);
std::string GenerateGraphId(const Ort::ConstGraph& graph);

}  // namespace mgx_ep