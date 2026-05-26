#!/bin/bash
# Op-level AG bandwidth comparison for dist.all_gather_into_tensor on 8x MI300X.
#
# Two modes back-to-back in the same process:
#   symm_ag    -- cuMem-backed buffers via symm_mem.empty -> SDMA path
#   regular_ag -- caching-allocator buffers via torch.empty -> CU-kernel path
#
# Default payload matches one Llama-3 70B FSDP=8 per-layer AG:
#   per-rank input  = 219_088_896 B = 209.5 MiB bf16
#   AG output       = 8 * 219_088_896 = 1.63 GiB
#
# Knobs:
#   INPUT_BYTES   per-rank input bytes (default 219088896 = ~209.5 MiB bf16)
#   WARMUP        warmup iters per mode (default 5)
#   TIMED         timed iters per mode  (default 30)
#   MODES         comma-list (default "symm_ag,regular_ag")
#   NPROC         ranks (default 8)
#   ROCM_BUG_TEST_IMAGE   docker image (default lorrisync therock-main pytorch2.12)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${ROCM_BUG_TEST_IMAGE:-lorrisync/therock-main:gfx94X_pytorch2.12_rocm7.14_96bfee1}"
NPROC="${NPROC:-8}"
INPUT_BYTES="${INPUT_BYTES:-219088896}"
WARMUP="${WARMUP:-5}"
TIMED="${TIMED:-30}"
MODES="${MODES:-symm_ag,regular_ag}"

PROBE_B64=$(base64 -w0 "${SCRIPT_DIR}/ag_bw_bench.py")
INTERPOSER_B64=$(base64 -w0 "${SCRIPT_DIR}/hip_attr_drain_preload.c")

echo "=== Image       : ${IMAGE}"
echo "=== NPROC       : ${NPROC}"
echo "=== input_bytes : ${INPUT_BYTES} per rank"
echo "=== modes       : ${MODES}"
echo "=== warmup/timed: ${WARMUP}/${TIMED}"

docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    --ipc=host --shm-size=64g \
    --network=host \
    -e PROBE_B64="${PROBE_B64}" \
    -e INTERPOSER_B64="${INTERPOSER_B64}" \
    -e NPROC="${NPROC}" \
    -e INPUT_BYTES="${INPUT_BYTES}" \
    -e WARMUP="${WARMUP}" \
    -e TIMED="${TIMED}" \
    -e MODES="${MODES}" \
    -e HSA_SDMA_LINEAR_B2B="${HSA_SDMA_LINEAR_B2B:-0}" \
    "${IMAGE}" \
    /bin/bash -c '
        set -e
        echo "${PROBE_B64}"      | base64 -d > /tmp/ag_bw_bench.py
        echo "${INTERPOSER_B64}" | base64 -d > /tmp/hip_attr_drain_preload.c
        gcc -O2 -fPIC -shared /tmp/hip_attr_drain_preload.c \
            -o /tmp/libhip_attr_drain.so -ldl

        export LD_PRELOAD=/tmp/libhip_attr_drain.so
        export HSA_NO_SCRATCH_RECLAIM=1
        export HSA_SDMA_LINEAR_B2B="${HSA_SDMA_LINEAR_B2B:-0}"
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
        export OMP_NUM_THREADS=8
        export MASTER_ADDR=127.0.0.1
        export MASTER_PORT=29591
        export NCCL_CTA_POLICY=2
        export NCCL_CUMEM_ENABLE=1
        export NCCL_LOCAL_REGISTER=0
        export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true
        export NCCL_SOCKET_IFNAME=lo
        export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

        torchrun --nproc_per_node="${NPROC}" --nnodes=1 --node_rank=0 \
            --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
            /tmp/ag_bw_bench.py \
            --input-bytes "${INPUT_BYTES}" \
            --warmup "${WARMUP}" --timed "${TIMED}" \
            --modes "${MODES}"
    '
