#!/bin/bash
# Run the pure-PyTorch reproducer inside Docker, twice:
#   - NCCL_CUMEM_ENABLE=1
#   - NCCL_CUMEM_ENABLE=0
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_B64=$(base64 -w0 "${SCRIPT_DIR}/pytorch_bug_repro.py")

IMAGE="${ROCM_BUG_TEST_IMAGE:-registry-sc-harbor.amd.com/framework/therock-main:1384_gfx94X_7.14.0a20260518_centosstream9_py3.12_pytorch_release-2.11_96bfee1}"

echo "=== Image       : ${IMAGE}"
echo "=== Host kernel : $(uname -r)"

run_pytorch() {
    local CUMEM=$1
    echo ""
    echo "############################################################"
    echo "  PYTORCH reproducer  (NCCL_CUMEM_ENABLE=${CUMEM})"
    echo "############################################################"
    docker run --rm \
        --device=/dev/kfd --device=/dev/dri --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
        --ipc=host --shm-size=64g \
        -e SRC_B64="${PY_B64}" \
        "${IMAGE}" \
        /bin/bash -c "
            echo \"\${SRC_B64}\" | base64 -d > /tmp/repro.py
            cd /tmp
            HSA_NO_SCRATCH_RECLAIM=1 NCCL_CUMEM_ENABLE=${CUMEM} \
                torchrun --nproc_per_node=1 -- repro.py 2>&1
        "
}

run_pytorch 1
run_pytorch 0
