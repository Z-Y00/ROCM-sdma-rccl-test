#!/bin/bash
# Compile and run the pure-HIP attribute probe inside the same ROCm container
# used by run_bug_repros.sh. Demonstrates that hipDeviceGetAttribute with
# attribute id 128 (CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED) returns
# hipErrorInvalidValue AND leaks that error into the HIP runtime's per-thread
# last_error slot -- the lowest-level root cause of the
# "first kernel after ncclMemAlloc fails" bug.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_B64=$(base64 -w0 "${SCRIPT_DIR}/hip_attr_probe.c")

IMAGE="${ROCM_BUG_TEST_IMAGE:-registry-sc-harbor.amd.com/framework/therock-main:1384_gfx94X_7.14.0a20260518_centosstream9_py3.12_pytorch_release-2.11_96bfee1}"

echo "=== Image       : ${IMAGE}"
echo "=== Host kernel : $(uname -r)"
echo ""
echo "############################################################"
echo "  PURE-HIP attribute probe  (no RCCL, no PyTorch)"
echo "############################################################"

docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    --ipc=host --shm-size=8g \
    -e SRC_B64="${SRC_B64}" \
    "${IMAGE}" \
    /bin/bash -c '
        set -e
        echo "${SRC_B64}" | base64 -d > /tmp/hip_attr_probe.c
        cd /tmp
        # Pure-C HIP host API: build with plain gcc against /opt/rocm.
        # Avoids hipcc/clang and the libstdc++-devel install dance on
        # the CentOS-Stream-9 image.
        gcc -O0 -D__HIP_PLATFORM_AMD__=1 -I/opt/rocm/include \
            hip_attr_probe.c \
            -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib \
            -o hip_attr_probe
        ./hip_attr_probe
    '
