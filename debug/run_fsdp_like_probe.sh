#!/bin/bash
# Run fsdp_like_ag_probe.py in the PyTorch-2.12 image.
#
# This is the fast reproducer/comparator for the Torchtitan FSDP crash:
#   mode=symm_ag       explicit symm_mem tensors (bench-like)
#   mode=regular_ag    regular tensors + allocator hook (FSDP-like collective)
#   mode=fsdp_forward  tiny FSDP2 model (Torchtitan-like Python stack)
#
# Examples:
#   ./run_fsdp_like_probe.sh symm_ag
#   ./run_fsdp_like_probe.sh regular_ag
#   ./run_fsdp_like_probe.sh fsdp_forward
#   ./run_fsdp_like_probe.sh all
#
# Env:
#   CE_MODE=1 (default)  zero-CTA/CE env
#   CE_MODE=0            ring/default RCCL env
#   NPROC=8              ranks
#   NUMEL=16777216       AG elements per rank
#   HIDDEN=4096 LAYERS=2 SEQ=64 BATCH=1
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-all}"
IMAGE="${ROCM_BUG_TEST_IMAGE:-lorrisync/therock-main:gfx94X_pytorch2.12_rocm7.14_96bfee1}"
NPROC="${NPROC:-8}"
CE_MODE="${CE_MODE:-1}"
NUMEL="${NUMEL:-16777216}"
HIDDEN="${HIDDEN:-4096}"
LAYERS="${LAYERS:-2}"
SEQ="${SEQ:-64}"
BATCH="${BATCH:-1}"

PROBE_B64=$(base64 -w0 "${SCRIPT_DIR}/fsdp_like_ag_probe.py")
INTERPOSER_B64=$(base64 -w0 "${SCRIPT_DIR}/hip_attr_drain_preload.c")

echo "=== Image    : ${IMAGE}"
echo "=== Mode     : ${MODE}"
echo "=== NPROC    : ${NPROC}"
echo "=== CE_MODE  : ${CE_MODE}"
echo "=== NUMEL    : ${NUMEL}"

docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    --ipc=host --shm-size=64g \
    --network=host \
    -e PROBE_B64="${PROBE_B64}" \
    -e INTERPOSER_B64="${INTERPOSER_B64}" \
    -e NPROC="${NPROC}" \
    -e MODE="${MODE}" \
    -e CE_MODE="${CE_MODE}" \
    -e NUMEL="${NUMEL}" \
    -e HIDDEN="${HIDDEN}" \
    -e LAYERS="${LAYERS}" \
    -e SEQ="${SEQ}" \
    -e BATCH="${BATCH}" \
    -e NCCL_DEBUG="${NCCL_DEBUG:-}" \
    -e NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-}" \
    "${IMAGE}" \
    /bin/bash -c '
        set -e
        echo "${PROBE_B64}" | base64 -d > /tmp/fsdp_like_ag_probe.py
        echo "${INTERPOSER_B64}" | base64 -d > /tmp/hip_attr_drain_preload.c
        gcc -O2 -fPIC -shared /tmp/hip_attr_drain_preload.c \
            -o /tmp/libhip_attr_drain.so -ldl

        export LD_PRELOAD=/tmp/libhip_attr_drain.so
        export HSA_NO_SCRATCH_RECLAIM=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
        export OMP_NUM_THREADS=8
        : "${NCCL_DEBUG:=WARN}"; export NCCL_DEBUG
        [ -n "${NCCL_DEBUG_SUBSYS}" ] && export NCCL_DEBUG_SUBSYS
        export MASTER_ADDR=127.0.0.1
        export MASTER_PORT=29591

        if [ "${CE_MODE}" = "1" ]; then
            export NCCL_CTA_POLICY=2
            export NCCL_CUMEM_ENABLE=1
            export NCCL_LOCAL_REGISTER=2
            export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true
            echo "CE env enabled"
        else
            unset NCCL_CTA_POLICY NCCL_LOCAL_REGISTER \
                  TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK
            export NCCL_CUMEM_ENABLE=1
            echo "CE env disabled (ring/default RCCL)"
        fi

        torchrun --nproc_per_node="${NPROC}" --nnodes=1 --node_rank=0 \
            --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
            /tmp/fsdp_like_ag_probe.py \
            --mode "${MODE}" --numel "${NUMEL}" \
            --hidden "${HIDDEN}" --layers "${LAYERS}" \
            --seq "${SEQ}" --batch "${BATCH}"
    '
