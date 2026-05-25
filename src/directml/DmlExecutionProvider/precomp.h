// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <numeric>
#include <algorithm>
#include <vector>
#include <string>
#include <limits>
#include <memory>
#include <optional>
#include <list>
#include <map>
#include <deque>
#include <chrono>
#include <variant>
#include <cassert>
#include <fstream>
#include <filesystem>

#include <wrl/client.h>
#include <wrl/implements.h>

#include <wil/wrl.h>
#include <wil/result.h>

#include <gsl/gsl>

#ifdef _GAMING_XBOX_SCARLETT
#include <d3d12_xs.h>
#include <d3dx12_xs.h>
#elif defined(_GAMING_XBOX_XBOXONE)
#include <d3d12_x.h>
#include <d3dx12_x.h>
#else // Desktop
#include "d3d12.h"
#include <d3d12sdklayers.h>
#include <directx/d3dx12.h>
#endif
#include "core/common/flatbuffers.h"

#include "GraphicsUnknownHelper.h"

#include <DirectML.h>
#include "core/common/common.h"
#include "ErrorHandling.h"

// DirectML helper libraries
#include "external/DirectMLHelpers/ApiTraits.h"
#include "external/DirectMLHelpers/ApiHelpers.h"
#include "external/DirectMLHelpers/DirectMLSchema.h"
#include "external/DirectMLHelpers/AbstractOperatorDesc.h"
#include "external/DirectMLHelpers/GeneratedSchemaTypes.h"
#include "external/DirectMLHelpers/SchemaHelpers.h"
#include "external/DirectMLHelpers/GeneratedSchemaHelpers.h"
#include "external/DirectMLHelpers/DirectMLX.h"
#include "external/DirectMLHelpers/DmlSerializedGraphDesc.h"
#include "external/DirectMLHelpers/DmlGraphSerialization.h"
#include "external/DirectMLHelpers/DmlGraphDeserialization.h"
#include "external/DirectMLHelpers/DmlGraphHelper.h"

using Microsoft::WRL::ComPtr;

// Windows pollutes the macro space, causing a build break in schema.h.
#undef OPTIONAL

#include "DmlExecutionProvider/inc/DmlExecutionProvider.h"
#include "OperatorAuthorHelper/MLOperatorAuthorHelper.h"
#include "OperatorAuthorHelper/operator_helper_common.h"
#include "dml_execution_provider.h"

#include "dml_common.h"
#include "TensorDesc.h"
#include "DescriptorPool.h"
#include "IExecutionProvider.h"
#include "Utility.h"
