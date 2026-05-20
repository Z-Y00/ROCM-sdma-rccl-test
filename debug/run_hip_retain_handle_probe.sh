#!/bin/bash
# Build + run hip_retain_handle_probe.c inside the PyTorch-2.12 image.
# Pure-HIP reproducer for the hipMemRetainAllocationHandle SIGSEGV that
# RCCL's IPC registration path hits when called on a hipMalloc'd buffer
# (the FSDP/regular allocator slab case).
#
# Runs each of three modes in its own container so a SIGSEGV in one
# doesn't poison the next:
#   1) null       -- sanity, expect clean error
#   2) cumem      -- positive control, cuMem-backed VA, expect handle returned
#   3) hipmalloc  -- the suspect path, hipMalloc'd VA, theory says SIGSEGV.
#                    Per RCCL contract should return hipErrorInvalidValue.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${ROCM_BUG_TEST_IMAGE:-lorrisync/therock-main:gfx94X_pytorch2.12_rocm7.14_96bfee1}"

PROBE_B64=$(base64 -w0 "${SCRIPT_DIR}/hip_retain_handle_probe.c")

run_one() {
  local mode="$1"
  echo
  echo "=============================================================="
  echo "MODE: ${mode}"
  echo "=============================================================="
  set +e
  docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    --ipc=host --shm-size=8g \
    -e PROBE_B64="${PROBE_B64}" \
    -e MODE="${mode}" \
    "${IMAGE}" \
    /bin/bash -c '
      set -e
      echo "${PROBE_B64}" | base64 -d > /tmp/hip_retain_handle_probe.c
      ROCM_INC=$(ls -d /opt/rocm-*/include 2>/dev/null | head -1)
      ROCM_INC=${ROCM_INC:-/opt/rocm/include}
      ROCM_LIB=$(ls -d /opt/rocm-*/lib 2>/dev/null | head -1)
      ROCM_LIB=${ROCM_LIB:-/opt/rocm/lib}
      echo "[probe] using ROCM_INC=$ROCM_INC  ROCM_LIB=$ROCM_LIB"
      /opt/rocm-7.14.0/bin/hipcc -O2 -x c++ \
          -D__HIP_PLATFORM_AMD__ -I"$ROCM_INC" -L"$ROCM_LIB" \
          /tmp/hip_retain_handle_probe.c \
          -o /tmp/hip_retain_handle_probe 2>&1 | head -40
      echo "[probe] running mode=${MODE}"
      /tmp/hip_retain_handle_probe "${MODE}"
      echo "[probe] mode=${MODE} exited $?"
    '
  local rc=$?
  set -e
  echo "[probe] container for mode=${mode} exited rc=${rc}"
}

if [ $# -gt 0 ]; then
  run_one "$1"
else
  run_one null
  run_one cumem
  run_one hipmalloc
fi
