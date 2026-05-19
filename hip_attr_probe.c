/* Pure HIP host-API probe -- no RCCL, no PyTorch, no GPU kernels.
 *
 * Probes the one suspect: attribute id 128 ==
 * CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, which RCCL queries
 * (and ignores the return code of) inside ncclMemAlloc's cuMem path.
 * If this single bare HIP call leaves hipErrorInvalidValue in the
 * runtime's per-thread last_error slot, the diagnosis is complete:
 * the next unrelated HIP API entry (e.g. a kernel launch) will surface
 * the stale error as if it were its own failure.
 *
 * Written in C so it compiles with plain gcc against /opt/rocm headers
 * -- no hipcc / clang++ / libstdc++-devel required.
 *
 * Build (inside the ROCm container):
 *   gcc -O0 -D__HIP_PLATFORM_AMD__=1 -I/opt/rocm/include \
 *       hip_attr_probe.c -L/opt/rocm/lib -lamdhip64 -o hip_attr_probe
 * Run:
 *   ./hip_attr_probe
 */
#include <hip/hip_runtime_api.h>
#include <stdio.h>

#define FABRIC_SUPPORTED 128  /* CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED */

int main(void) {
  int device = 0;
  hipError_t e = hipSetDevice(device);
  if (e != hipSuccess) {
    fprintf(stderr, "hipSetDevice failed: %d %s\n",
            (int)e, hipGetErrorString(e));
    return 1;
  }
  hipDeviceProp_t props;
  hipGetDeviceProperties(&props, device);
  printf("device: %s\n", props.name);

  /* Drain any startup errors so we have a clean baseline. */
  hipGetLastError();
  hipError_t base = hipPeekAtLastError();
  printf("baseline peek = %d (%s)\n", (int)base, hipGetErrorString(base));

  /* Match what RCCL does: call the attribute query and discard the return. */
  int dummy = 0;
  printf("\ncalling: (void) hipDeviceGetAttribute(&dummy, %d, %d)\n",
         FABRIC_SUPPORTED, device);
  hipError_t rc = hipDeviceGetAttribute(&dummy,
                                        (hipDeviceAttribute_t)FABRIC_SUPPORTED,
                                        device);
  hipError_t leaked = hipPeekAtLastError();

  printf("  return value    = %d (%s)\n", (int)rc,     hipGetErrorString(rc));
  printf("  leaked into TLS = %d (%s)\n", (int)leaked, hipGetErrorString(leaked));

  if (leaked == hipErrorInvalidValue) {
    printf("\n>>> CONFIRMED: bare HIP hipDeviceGetAttribute(.., 128, ..)\n");
    printf(">>>            leaks hipErrorInvalidValue into TLS last_error.\n");
    printf(">>>            The next HIP API entry will report it as its own failure.\n");
    return 0;
  } else if (leaked == hipSuccess) {
    printf("\n>>> NOT leaked. Bug must be elsewhere.\n");
    return 2;
  } else {
    printf("\n>>> UNEXPECTED leaked value %d (%s)\n",
           (int)leaked, hipGetErrorString(leaked));
    return 3;
  }
}
