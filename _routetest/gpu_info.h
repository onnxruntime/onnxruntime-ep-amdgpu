#pragma once
// Minimal shim: the routing policy depends on gpu_info.h ONLY for the Profile enum
// (per gpu_routing_policy.h's own comment). Order must match the real enum exactly.
enum class Profile { Auto, Eager, Optimized, MIGraphX, DirectX, Hip };
