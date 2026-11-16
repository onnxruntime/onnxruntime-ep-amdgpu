// Standalone unit test of the AMDGPU umbrella EP routing policy WITH the Muse override.
// Compiles the REAL src/amdgpu/gpu_routing_policy.h + gpu_routing_tables.h (our edited copy),
// with a minimal gpu_info.h shim (Profile enum only). USE_HIP is defined so the gfx1151->HIP
// rule is active -- proving our kArchModelBackend override BEATS it for muse_glimmer.
#define USE_HIP 1
#include "gpu_routing_policy.h"
#include <cstdio>
#include <string>
#include <optional>
using namespace gpu_ep;

static const char* nm(Profile p) {
    switch (p) {
        case Profile::Auto: return "Auto";
        case Profile::Eager: return "Eager";
        case Profile::Optimized: return "Optimized";
        case Profile::MIGraphX: return "MIGraphX";
        case Profile::DirectX: return "DirectX";
        case Profile::Hip: return "Hip";
    }
    return "?";
}

static Profile route(const char* gfx, const char* arch) {
    return select_backend(gfx, model_arch_hash(std::optional<std::string>(arch)),
                          /*is_webnn=*/false, Profile::Auto);
}

int main() {
    int fail = 0;
    auto check = [&](const char* gfx, const char* arch, Profile exp) {
        Profile got = route(gfx, arch);
        std::printf("  %-8s %-14s -> %-9s (expect %-9s) %s\n",
                    gfx, arch, nm(got), nm(exp), got == exp ? "OK" : "FAIL");
        if (got != exp) fail++;
    };
    std::printf("USE_HIP build, Auto profile:\n");
    check("gfx1151", "muse_glimmer", Profile::MIGraphX);  // OUR OVERRIDE (beats the gfx1151->HIP rule)
    check("gfx1151", "llama",        Profile::Hip);        // control: generic llama on gfx1151 still HIP
    check("gfx1151", "gptoss",       Profile::Hip);        // control: another LLM on gfx1151 still HIP
    check("gfx1100", "llama",        Profile::MIGraphX);   // gfx11 default (discrete RDNA3)
    check("gfx1100", "muse_glimmer", Profile::MIGraphX);   // gfx11 default (override is gfx1151-only; gfx11 is MIGraphX anyway)
    check("gfx1201", "llama",        Profile::MIGraphX);   // gfx12 (RDNA4) default
    check("gfx1030", "llama",        Profile::DirectX);    // pre-gfx11 default
    std::printf(fail ? "RESULT: %d FAIL\n" : "RESULT: ALL PASS\n", fail);
    return fail;
}
