// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <type_traits>
#include <vector>

#include "parse_string.h"
#include "common/plugin_ep_utils.h"

template <typename TEnum>
using EnumNameMapping = std::vector<std::pair<TEnum, std::string>>;

template <typename TEnum>
Ort::Status EnumToName(const EnumNameMapping<TEnum>& mapping, TEnum value, std::string& name) {
  const auto it = std::find_if(
      mapping.begin(), mapping.end(),
      [&value](const std::pair<TEnum, std::string>& entry) {
        return entry.first == value;
      });
  RETURN_IF(it == mapping.end(), "Failed to map enum value to name: ",
      static_cast<std::underlying_type_t<TEnum>>(value));
  name = it->second;
  return STATUS_OK;
}

template <typename TEnum>
std::string EnumToName(const EnumNameMapping<TEnum>& mapping, TEnum value) {
  std::string name;
  THROW_IF_ERROR(EnumToName(mapping, value, name));
  return name;
}

template <typename TEnum>
Ort::Status NameToEnum(const EnumNameMapping<TEnum>& mapping, const std::string& name, TEnum& value) {
  const auto it = std::find_if(
      mapping.begin(), mapping.end(),
      [&name](const std::pair<TEnum, std::string>& entry) {
        return entry.second == name;
      });
  RETURN_IF(
      it == mapping.end(),
      "Failed to map enum name to value: ", name);
  value = it->first;
  return STATUS_OK;
}

template <typename TEnum>
TEnum NameToEnum(const EnumNameMapping<TEnum>& mapping, const std::string& name) {
  TEnum value;
  THROW_IF_ERROR(NameToEnum(mapping, name, value));
  return value;
}

struct ProviderOptionsParser {
    template <typename ValueParserType>
    ProviderOptionsParser& AddValueParser(
        std::string_view name, ValueParserType value_parser) {
        ENFORCE(value_parsers_.emplace(name, ValueParser{value_parser}).second,
            "Provider option \"", name, "\" already has a value parser.");
        return *this;
    }

    template <typename ValueType>
    ProviderOptionsParser& AddAssignmentToReference(
        std::string_view name, ValueType& dest) {
        return AddValueParser(name,
            [&dest](const std::string& value_str) -> Ort::Status {
              return ParseStringWithClassicLocale(value_str, dest);
            });
    }

    template <typename EnumType>
    ProviderOptionsParser& AddAssignmentToEnumReference(
        std::string_view name, const EnumNameMapping<EnumType>& mapping, EnumType& dest) {
        return AddValueParser(name,
            [&mapping, &dest](const std::string& value_str) -> Ort::Status {
              return NameToEnum(mapping, value_str, dest);
            });
    }

    Ort::Status Parse(const ProviderOptions& options) const {
        for (const auto& option : options) {
            const auto& name = option.first;
            const auto& value_str = option.second;
            const auto value_parser_it = value_parsers_.find(name);
            RETURN_IF(value_parser_it == value_parsers_.end(),
                "Unknown provider option: \"", name, "\".");
            const auto parse_status = value_parser_it->second(value_str);
            RETURN_IF_NOT(parse_status == nullptr,
                "Failed to parse provider option \"", name, "\": ", parse_status.GetErrorMessage());
        }
        return STATUS_OK;
    }

private:
    using ValueParser = std::function<Ort::Status(const std::string&)>;
    std::unordered_map<std::string, ValueParser> value_parsers_;
};
