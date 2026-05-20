/*
 * hip_retain_handle_probe.c -- minimal pure-HIP reproducer for the
 * hipMemRetainAllocationHandle SIGSEGV that breaks CE/SDMA FSDP under
 * NCCL_CUMEM_ENABLE=1 + NCCL_LOCAL_REGISTER=2 +
 * TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true.
 *
 * Theory:
 *   When ROCm's hipMemRetainAllocationHandle is called on a VA that was NOT
 *   created via cuMemCreate (e.g. hipMalloc / cudaMalloc / PyTorch caching
 *   allocator slab), it should return hipErrorInvalidValue. RCCL relies on
 *   that exact behavior and explicitly falls back to cudaIpcGetMemHandle on
 *   failure (rccl/src/transport/p2p.cc:890-921).
 *
 *   The current HIP runtime (libamdhip64.so.7 from ROCm 7.14, build
 *   39213316d2) instead dereferences a NULL sub-slot in its per-allocation
 *   tracker:
 *     mov  0x100(%rax),%rax     ; rax = tracker->cuMemSlot     (non-null)
 *     mov  0xf8(%rax),%r15      ; r15 = cuMemSlot->handle      (NULL -> SEGV)
 *
 * Build:
 *   /opt/rocm-7.14.0/bin/hipcc -O2 hip_retain_handle_probe.c \
 *       -o hip_retain_handle_probe
 *
 * Run modes (each in its own process, since we WILL segfault):
 *   ./hip_retain_handle_probe cumem      -- expect: return success (control)
 *   ./hip_retain_handle_probe hipmalloc  -- expect (theory): SIGSEGV.
 *                                           Per RCCL's API contract should
 *                                           return hipErrorInvalidValue.
 *   ./hip_retain_handle_probe null       -- expect: clean error
 *
 * No RCCL, no PyTorch, no NCCL, no distributed init -- pure HIP.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hip/hip_runtime.h>

/* hipMemRetainAllocationHandle is in hip_runtime_api.h but its prototype is
 * gated behind HIP_VERSION; declare it directly so this builds against any
 * sufficiently recent HIP. */
extern "C" {
hipError_t hipMemRetainAllocationHandle(hipMemGenericAllocationHandle_t* handle,
                                        void* addr);
}

static const char* errstr(hipError_t e) {
    const char* s = hipGetErrorString(e);
    return s ? s : "(null)";
}

static int test_cumem(void) {
    fprintf(stderr, "\n--- mode=cumem (control) ---\n");
    /* Allocate a cuMem-backed slab via the cuMem API. */
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = 0;
    prop.requestedHandleType = hipMemHandleTypePosixFileDescriptor;

    size_t granularity = 0;
    hipError_t e = hipMemGetAllocationGranularity(
        &granularity, &prop, hipMemAllocationGranularityMinimum);
    fprintf(stderr, "hipMemGetAllocationGranularity -> %d (%s), gran=%zu\n",
            e, errstr(e), granularity);
    if (e != hipSuccess) return 1;

    size_t size = ((1u << 20) + granularity - 1) / granularity * granularity;
    hipMemGenericAllocationHandle_t h = 0;
    e = hipMemCreate(&h, size, &prop, 0);
    fprintf(stderr, "hipMemCreate(size=%zu) -> %d (%s), handle=%p\n",
            size, e, errstr(e), (void*)h);
    if (e != hipSuccess) return 1;

    void* ptr = NULL;
    e = hipMemAddressReserve((hipDeviceptr_t*)&ptr, size, 0, NULL, 0);
    fprintf(stderr, "hipMemAddressReserve -> %d (%s), va=%p\n",
            e, errstr(e), ptr);
    if (e != hipSuccess) return 1;

    e = hipMemMap((hipDeviceptr_t)ptr, size, 0, h, 0);
    fprintf(stderr, "hipMemMap -> %d (%s)\n", e, errstr(e));
    if (e != hipSuccess) return 1;

    hipMemAccessDesc access = {};
    access.location.type = hipMemLocationTypeDevice;
    access.location.id = 0;
    access.flags = hipMemAccessFlagsProtReadWrite;
    e = hipMemSetAccess((hipDeviceptr_t)ptr, size, &access, 1);
    fprintf(stderr, "hipMemSetAccess -> %d (%s)\n", e, errstr(e));
    if (e != hipSuccess) return 1;

    hipMemGenericAllocationHandle_t got = 0;
    fprintf(stderr, "calling hipMemRetainAllocationHandle on cuMem VA %p ...\n",
            ptr);
    e = hipMemRetainAllocationHandle(&got, ptr);
    fprintf(stderr, "hipMemRetainAllocationHandle -> %d (%s), handle=%p\n",
            e, errstr(e), (void*)got);
    if (e == hipSuccess) {
        fprintf(stderr, "PASS: cuMem path returned a handle, as expected.\n");
        return 0;
    }
    fprintf(stderr, "UNEXPECTED: cuMem path did not return success.\n");
    return 2;
}

static int test_hipmalloc(void) {
    fprintf(stderr, "\n--- mode=hipmalloc (suspect) ---\n");
    /* The same VA layout an FSDP all-gather buffer has: hipMalloc, no cuMem. */
    void* ptr = NULL;
    size_t size = 1u << 20;
    hipError_t e = hipMalloc(&ptr, size);
    fprintf(stderr, "hipMalloc(%zu) -> %d (%s), va=%p\n",
            size, e, errstr(e), ptr);
    if (e != hipSuccess) return 1;

    /* Touch it so the runtime definitely has the allocation in its tracker. */
    e = hipMemset(ptr, 0xab, size);
    fprintf(stderr, "hipMemset -> %d (%s)\n", e, errstr(e));
    if (e != hipSuccess) return 1;
    e = hipDeviceSynchronize();
    fprintf(stderr, "hipDeviceSynchronize -> %d (%s)\n", e, errstr(e));

    hipMemGenericAllocationHandle_t got = 0;
    fprintf(stderr, "calling hipMemRetainAllocationHandle on hipMalloc VA %p\n",
            ptr);
    fprintf(stderr, "    PER RCCL CONTRACT: must return non-success (will be\n");
    fprintf(stderr, "    handled via legacy cudaIpcGetMemHandle fallback).\n");
    fflush(stderr);

    e = hipMemRetainAllocationHandle(&got, ptr);
    fprintf(stderr, "hipMemRetainAllocationHandle -> %d (%s), handle=%p\n",
            e, errstr(e), (void*)got);
    if (e == hipSuccess) {
        fprintf(stderr, "SURPRISE: returned success on a hipMalloc'd VA. ");
        fprintf(stderr, "Inspect handle=%p\n", (void*)got);
        return 3;
    }
    fprintf(stderr, "OK: returned a clean error on hipMalloc'd VA.\n");
    return 0;
}

static int test_null(void) {
    fprintf(stderr, "\n--- mode=null (sanity) ---\n");
    hipMemGenericAllocationHandle_t got = 0;
    fprintf(stderr, "calling hipMemRetainAllocationHandle(NULL)\n");
    fflush(stderr);
    hipError_t e = hipMemRetainAllocationHandle(&got, NULL);
    fprintf(stderr, "hipMemRetainAllocationHandle(NULL) -> %d (%s)\n",
            e, errstr(e));
    return 0;
}

int main(int argc, char** argv) {
    const char* mode = (argc > 1) ? argv[1] : "hipmalloc";

    int dev = 0;
    hipDeviceProp_t prop;
    hipError_t e = hipSetDevice(dev);
    fprintf(stderr, "hipSetDevice(0) -> %d (%s)\n", e, errstr(e));
    e = hipGetDeviceProperties(&prop, dev);
    fprintf(stderr, "device: %s\n", prop.name);

    if (!strcmp(mode, "cumem"))     return test_cumem();
    if (!strcmp(mode, "hipmalloc")) return test_hipmalloc();
    if (!strcmp(mode, "null"))      return test_null();
    fprintf(stderr, "unknown mode '%s' (want cumem|hipmalloc|null)\n", mode);
    return 64;
}
