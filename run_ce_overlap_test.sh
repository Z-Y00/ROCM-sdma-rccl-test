#!/bin/bash
# Run CE AllGather + GEMM overlap benchmark in Docker.
# Profiler traces are copied out to ./profile_output/
# Usage: ./run_ce_overlap_test.sh [NUM_GPUS] [M] [N] [K]

URL="registry-sc-harbor.amd.com/framework/therock-main"
TAG="1347_gfx94X_7.13.0a20260506_ubuntu24.04_py3.12_pytorch_release-2.11_443606e"
IMAGE="${URL}:${TAG}"

NUM_GPUS=${1:-8}
M=${2:-4096}
N=${3:-4096}
K=${4:-40960}

CONTAINER_NAME="ce_overlap_$$"
PROFILE_DIR="./profile_output"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_CONTENT=$(base64 -w0 "${SCRIPT_DIR}/ce_overlap_gemm_test.py")

# Run without --rm so we can docker cp the traces out
docker run --name ${CONTAINER_NAME} \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    --ipc=host --shm-size=64g \
    ${IMAGE} /bin/bash -c "\
        echo '${SCRIPT_CONTENT}' | base64 -d > /tmp/ce_overlap_gemm_test.py && \
        cd /tmp && \
        HSA_NO_SCRATCH_RECLAIM=1 \
        NCCL_DEBUG=INFO \
        NCCL_CUMEM_ENABLE=1 \
        NCCL_WIN_ENABLE=1 \
        torchrun --nproc_per_node=${NUM_GPUS} -- ce_overlap_gemm_test.py \
            --m ${M} --n ${N} --k ${K} \
            --numel 1048576 \
            --warmup 5 --iters 20 \
            --dtype float16 \
            --profile-dir /tmp/profile_traces"

# Copy traces to host
rm -rf "${PROFILE_DIR}"
mkdir -p "${PROFILE_DIR}"
docker cp ${CONTAINER_NAME}:/tmp/profile_traces/. "${PROFILE_DIR}/" 2>/dev/null \
    && echo "Profiler traces copied to ${PROFILE_DIR}/" \
    || echo "No profiler traces found (test may have failed before profiling)"

# Cleanup container
docker rm ${CONTAINER_NAME} > /dev/null 2>&1
