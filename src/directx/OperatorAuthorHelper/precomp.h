// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "DirectML.h"
#include "DmlExecutionProvider/inc/MLOperatorAuthor.h"
#include "OperatorAuthorHelper/MLOperatorAuthorHelper.h"
#include "OperatorAuthorHelper/operator_helper_common.h"
#include <limits>
#include <cassert>
#include <chrono>
#include <vector>
#include <map>
#include <set>
#include <numeric>

#include <wrl/client.h>
#include <wrl/implements.h>

#include <wil/wrl.h>
#include <wil/result.h>
#include <gsl/span>
#include <gsl/gsl>
