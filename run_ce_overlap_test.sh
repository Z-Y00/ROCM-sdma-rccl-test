#!/bin/bash
# Run CE AllGather + GEMM overlap benchmark in Docker.
# Collects both PyTorch profiler traces and rocprofv3 traces (captures SDMA memcpy).
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

docker run --name ${CONTAINER_NAME} \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    --ipc=host --shm-size=64g \
    ${IMAGE} /bin/bash -c "
        echo '${SCRIPT_CONTENT}' | base64 -d > /tmp/ce_overlap_gemm_test.py
        cd /tmp

        # ── Run 1: Benchmark + PyTorch profiler ──
        echo '========== Run 1: Benchmark + PyTorch profiler =========='
        HSA_NO_SCRATCH_RECLAIM=1 \
        NCCL_DEBUG=INFO \
        NCCL_CUMEM_ENABLE=1 \
        NCCL_WIN_ENABLE=1 \
        torchrun --nproc_per_node=${NUM_GPUS} -- ce_overlap_gemm_test.py \
            --m ${M} --n ${N} --k ${K} \
            --numel 10485760 \
            --warmup 5 --iters 20 \
            --dtype float16 \
            --profile-dir /tmp/profile_traces/torch

        # ── Run 2: rocprofv3 to capture SDMA / copy engine activity ──
        echo ''
        echo '========== Run 2: rocprofv3 (SDMA + kernel + copy traces) =========='
        HSA_NO_SCRATCH_RECLAIM=1 \
        NCCL_CUMEM_ENABLE=1 \
        NCCL_WIN_ENABLE=1 \
        rocprofv3 \
            --kernel-trace \
            --memory-copy-trace \
            --hsa-trace \
            --rccl-trace \
            --output-format pftrace csv \
            --output-directory /tmp/profile_traces/rocprof \
            -- \
        torchrun --nproc_per_node=${NUM_GPUS} -- ce_overlap_gemm_test.py \
            --m ${M} --n ${N} --k ${K} \
            --numel 10485760 \
            --warmup 3 --iters 5 \
            --dtype float16
    "

# Copy all traces to host
rm -rf "${PROFILE_DIR}"
mkdir -p "${PROFILE_DIR}"
docker cp ${CONTAINER_NAME}:/tmp/profile_traces/. "${PROFILE_DIR}/" 2>/dev/null \
    && echo "All traces copied to ${PROFILE_DIR}/" \
    || echo "No traces found (test may have failed)"

echo ""
echo "Output structure:"
find "${PROFILE_DIR}" -type f 2>/dev/null | head -30
echo ""
echo "Perfetto traces (.pftrace) can be opened at https://ui.perfetto.dev"
echo "PyTorch traces (.json) can be opened in chrome://tracing or TensorBoard"

# Cleanup container
docker rm ${CONTAINER_NAME} > /dev/null 2>&1
