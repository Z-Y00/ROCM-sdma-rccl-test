#!/bin/bash
# Run the Copy Engine (CE) Collectives test inside the Docker container.
# Usage: ./run_ce_test.sh [NUM_GPUS]
#
# Ref: https://docs.pytorch.org/docs/2.11/symmetric_memory.html#copy-engine-collectives

URL="registry-sc-harbor.amd.com/framework/therock-main"
TAG="1347_gfx94X_7.13.0a20260506_ubuntu22.04_py3.11_pytorch_release-2.10_1a27007"
TAG="1347_gfx94X_7.13.0a20260506_ubuntu24.04_py3.12_pytorch_release-2.11_443606e"
# Custom PyTorch 2.12 image built from ROCm/pytorch release/2.12
IMAGE="therock-main:gfx94X_pytorch2.12_rocm7.13"

NUM_GPUS=${1:-8}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_CONTENT=$(base64 -w0 "${SCRIPT_DIR}/ce_collectives_test.py")

docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    --ipc=host --shm-size=64g \
    ${IMAGE} /bin/bash -c "\
        echo '${SCRIPT_CONTENT}' | base64 -d > /tmp/ce_collectives_test.py && \
        cd /tmp &&  \
        HSA_NO_SCRATCH_RECLAIM=1 \
        NCCL_DEBUG=INFO \
        NCCL_WIN_ENABLE=1 \
        NCCL_CUMEM_ENABLE=1 \
        torchrun --nproc_per_node=${NUM_GPUS} -- ce_collectives_test.py" > log.ce_collectives_test 2>&1

cat log.ce_collectives_test | grep "OK"
