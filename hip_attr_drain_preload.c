/*
 * hip_attr_drain_preload.c -- user-space LD_PRELOAD workaround for the
 * "first kernel after ncclMemAlloc fails" bug on ROCm.
 *
 * Background:
 *   RCCL's allocator.cc calls
 *     (void) cuDeviceGetAttribute(&flag,
 *              CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, dev);
 *   inside its cuMem code path (NCCL_CUMEM_ENABLE=1). On ROCm 7.x that
 *   attribute id (128) is not supported -- the call returns
 *   hipErrorInvalidValue AND leaves the same error sitting in the HIP
 *   runtime's per-thread `last_error_` slot. RCCL discards the return
 *   value, so nothing drains the TLS. The next unrelated HIP API entry
 *   (e.g. a kernel launch) reads back that stale error and reports it
 *   as if its own operation failed.
 *
 *   On ROCm the macro CUPFN(x) in rocm-systems/projects/rccl/src/include/
 *   rocmwrap.h expands to a literal `x`, so the call site goes through
 *   the normal global symbol -- which makes it directly interposable
 *   via LD_PRELOAD. No RCCL rebuild required.
 *
 * What this interposer does:
 *   - Wraps  hipDeviceGetAttribute  AND  cuDeviceGetAttribute.
 *   - Forwards to the real implementation (resolved via RTLD_NEXT).
 *   - If the real call returns non-success, calls hipGetLastError() to
 *     drain the TLS error slot so it can't leak into the next HIP call.
 *   - Returns the original return value untouched -- callers see the
 *     same failure they would have, they just don't get the TLS pollution.
 *
 * Build (inside the ROCm container):
 *   gcc -O2 -fPIC -shared hip_attr_drain_preload.c -o libhip_attr_drain.so -ldl
 *
 * Use:
 *   LD_PRELOAD=/path/to/libhip_attr_drain.so  <your program>
 *
 * Optional:
 *   HIP_DRAIN_VERBOSE=1   prints one line per drained call to stderr.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

/* Opaque ABI: hipError_t / CUresult are 32-bit ints, attribute enums are
 * ints, device handles are ints. Using bare int here keeps this .so free
 * of HIP/CUDA header and link dependencies -- it can be built in any
 * container with just gcc + libdl. */
typedef int (*get_attr_fn)(int *, int, int);
typedef int (*get_last_err_fn)(void);

static get_attr_fn      real_hipDeviceGetAttribute = NULL;
static get_attr_fn      real_cuDeviceGetAttribute  = NULL;
static get_last_err_fn  real_hipGetLastError       = NULL;

static int verbose = 0;

__attribute__((constructor))
static void hip_attr_drain_init(void) {
    real_hipDeviceGetAttribute =
        (get_attr_fn)     dlsym(RTLD_NEXT, "hipDeviceGetAttribute");
    real_cuDeviceGetAttribute =
        (get_attr_fn)     dlsym(RTLD_NEXT, "cuDeviceGetAttribute");
    real_hipGetLastError =
        (get_last_err_fn) dlsym(RTLD_NEXT, "hipGetLastError");

    const char *v = getenv("HIP_DRAIN_VERBOSE");
    verbose = (v && *v == '1') ? 1 : 0;

    if (verbose) {
        fprintf(stderr,
            "[hip_attr_drain] loaded. "
            "hipDeviceGetAttribute=%p cuDeviceGetAttribute=%p "
            "hipGetLastError=%p\n",
            (void *)real_hipDeviceGetAttribute,
            (void *)real_cuDeviceGetAttribute,
            (void *)real_hipGetLastError);
    }
}

static inline void maybe_drain(const char *who, int attr, int rc) {
    if (rc == 0) return;   /* hipSuccess / CUDA_SUCCESS == 0 */
    /* Lazy-retry the dlsym: the HIP runtime may have been dlopen()ed
     * by the host process AFTER our constructor ran (e.g. PyTorch loads
     * libamdhip64.so lazily via libtorch_hip). */
    if (!real_hipGetLastError) {
        real_hipGetLastError =
            (get_last_err_fn) dlsym(RTLD_NEXT, "hipGetLastError");
        if (!real_hipGetLastError) {
            real_hipGetLastError =
                (get_last_err_fn) dlsym(RTLD_DEFAULT, "hipGetLastError");
        }
    }
    int drained = -1;
    if (real_hipGetLastError) drained = real_hipGetLastError();
    if (verbose) {
        fprintf(stderr,
            "[hip_attr_drain] %s(attr=%d) -> %d ; drained TLS = %d%s\n",
            who, attr, rc, drained,
            real_hipGetLastError ? "" : "  (WARN: hipGetLastError unresolved)");
    }
}

/* ------------------------------------------------------------------ */
/* hipDeviceGetAttribute -- caught directly by anyone in user code,
 *    including the hip_attr_probe.cpp reproducer.                    */
/* ------------------------------------------------------------------ */
int hipDeviceGetAttribute(int *value, int attrib, int device) {
    if (!real_hipDeviceGetAttribute) {
        /* Constructor may not have run if loader resolved the symbol
         * before our ctor fired; resolve lazily. */
        real_hipDeviceGetAttribute =
            (get_attr_fn) dlsym(RTLD_NEXT, "hipDeviceGetAttribute");
    }
    int rc = real_hipDeviceGetAttribute(value, attrib, device);
    maybe_drain("hipDeviceGetAttribute", attrib, rc);
    return rc;
}

/* ------------------------------------------------------------------ */
/* cuDeviceGetAttribute -- the symbol that RCCL's allocator.cc calls
 *    directly on ROCm (CUPFN(x) expands to `x`). This is the symbol we
 *    actually need to catch to fix the NCCL_CUMEM_ENABLE=1 bug.       */
/* ------------------------------------------------------------------ */
int cuDeviceGetAttribute(int *value, int attrib, int device) {
    if (!real_cuDeviceGetAttribute) {
        real_cuDeviceGetAttribute =
            (get_attr_fn) dlsym(RTLD_NEXT, "cuDeviceGetAttribute");
    }
    int rc = real_cuDeviceGetAttribute(value, attrib, device);
    maybe_drain("cuDeviceGetAttribute", attrib, rc);
    return rc;
}
