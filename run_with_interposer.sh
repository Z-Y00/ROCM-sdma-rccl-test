#!/bin/bash
# Build the hip_attr_drain_preload.c interposer inside the container, then
# run both reproducers (the pure-HIP probe and the PyTorch repro) with
# LD_PRELOAD pointing at it. Validates the user-space workaround end-to-end
# WITHOUT touching RCCL.
#
# Expected results when the interposer works:
#   * HIP probe   -> exit code 2  ("NOT leaked")
#   * PyTorch repro (NCCL_CUMEM_ENABLE=1) -> "no bug observed."
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INTERPOSER_B64=$(base64 -w0 "${SCRIPT_DIR}/hip_attr_drain_preload.c")
HIP_PROBE_B64=$(base64 -w0 "${SCRIPT_DIR}/hip_attr_probe.c")
PY_REPRO_B64=$(base64 -w0 "${SCRIPT_DIR}/pytorch_bug_repro.py")

IMAGE="${ROCM_BUG_TEST_IMAGE:-registry-sc-harbor.amd.com/framework/therock-main:1384_gfx94X_7.14.0a20260518_centosstream9_py3.12_pytorch_release-2.11_96bfee1}"

echo "=== Image       : ${IMAGE}"
echo "=== Host kernel : $(uname -r)"

docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    --ipc=host --shm-size=64g \
    -e INTERPOSER_B64="${INTERPOSER_B64}" \
    -e HIP_PROBE_B64="${HIP_PROBE_B64}" \
    -e PY_REPRO_B64="${PY_REPRO_B64}" \
    "${IMAGE}" \
    /bin/bash -c '
        set -e
        export PATH=/opt/rocm/bin:${PATH}

        cd /tmp
        # The probe exits 0 (BUG) or 2 (FIX). Both are "expected", so we
        # turn off set -e for the test section and capture rc explicitly.
        echo "${INTERPOSER_B64}" | base64 -d > hip_attr_drain_preload.c
        echo "${HIP_PROBE_B64}"  | base64 -d > hip_attr_probe.c
        echo "${PY_REPRO_B64}"   | base64 -d > pytorch_bug_repro.py

        echo ""
        echo "############################################################"
        echo "  Build libhip_attr_drain.so"
        echo "############################################################"
        gcc -O2 -fPIC -shared hip_attr_drain_preload.c \
            -o libhip_attr_drain.so -ldl
        ls -l libhip_attr_drain.so
        nm -D libhip_attr_drain.so | grep -E "DeviceGetAttribute|GetLastError" || true

        echo ""
        echo "############################################################"
        echo "  Build hip_attr_probe (pure HIP, host-only, plain gcc)"
        echo "############################################################"
        gcc -O0 -D__HIP_PLATFORM_AMD__=1 -I/opt/rocm/include \
            hip_attr_probe.c \
            -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib \
            -o hip_attr_probe

        set +e

        echo ""
        echo "############################################################"
        echo "  [A] HIP probe BASELINE  (no LD_PRELOAD)  -- expect LEAK"
        echo "############################################################"
        ./hip_attr_probe
        rc_a=$?
        echo ">>> hip_attr_probe baseline exit = ${rc_a}  (0 = leak, 2 = no leak)"

        echo ""
        echo "############################################################"
        echo "  [B] HIP probe WITH INTERPOSER         -- expect NO LEAK"
        echo "############################################################"
        HIP_DRAIN_VERBOSE=1 LD_PRELOAD=/tmp/libhip_attr_drain.so \
            ./hip_attr_probe
        rc_b=$?
        echo ">>> hip_attr_probe interposed exit = ${rc_b}  (0 = leak, 2 = no leak)"

        echo ""
        echo "############################################################"
        echo "  [C] PyTorch repro BASELINE   NCCL_CUMEM_ENABLE=1 -- expect BUG"
        echo "############################################################"
        HSA_NO_SCRATCH_RECLAIM=1 NCCL_CUMEM_ENABLE=1 \
            torchrun --nproc_per_node=1 -- pytorch_bug_repro.py 2>&1 \
            | tee /tmp/baseline.log | tail -25

        echo ""
        echo "############################################################"
        echo "  [D] PyTorch repro WITH INTERPOSER NCCL_CUMEM_ENABLE=1 -- expect FIX"
        echo "############################################################"
        HIP_DRAIN_VERBOSE=1 \
        LD_PRELOAD=/tmp/libhip_attr_drain.so \
        HSA_NO_SCRATCH_RECLAIM=1 NCCL_CUMEM_ENABLE=1 \
            torchrun --nproc_per_node=1 -- pytorch_bug_repro.py 2>&1 \
            | tee /tmp/preload.log | tail -40

        echo ""
        echo "############################################################"
        echo "  VERDICT"
        echo "############################################################"
        baseline_verdict=$(grep -E "^(BUG REPRODUCED|no bug observed)" /tmp/baseline.log | head -1)
        preload_verdict=$(grep -E "^(BUG REPRODUCED|no bug observed)" /tmp/preload.log  | head -1)
        echo "  HIP probe baseline       : ${rc_a} ($([ ${rc_a} = 0 ] && echo LEAK || echo NO_LEAK))"
        echo "  HIP probe interposed     : ${rc_b} ($([ ${rc_b} = 0 ] && echo LEAK || echo NO_LEAK))"
        echo "  PyTorch repro baseline   : ${baseline_verdict}"
        echo "  PyTorch repro interposed : ${preload_verdict}"
    '
