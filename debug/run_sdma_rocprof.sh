#!/bin/bash
# Definitive SDMA-usage verification using rocprofv3.
#
# For each of two configurations, runs debug/fsdp_like_ag_probe.py in
# fsdp_forward mode (the tiny FSDP2 model that matches the Torchtitan
# stack) under `rocprofv3 --hsa-amd-trace`, dumping per-rank CSV traces.
# We then count hsa_amd_memory_async_batch_copy (the SDMA dispatch API)
# vs hsa_amd_memory_async_copy (general / non-SDMA) per rank.
#
# Configurations:
#   ce_localreg0 : CTA=ZERO via PG opts + CUMEM_ENABLE=1
#                  + LOCAL_REGISTER=0 + allocator hook
#                  --> RCCL chooses P2P/CUMEM channels (SDMA-eligible)
#   true_ring    : same, but NCCL_CUMEM_ENABLE=0
#                  --> RCCL chooses P2P/IPC channels (CU-driven copies, no SDMA)
#
# Expected: ce_localreg0 shows many batch_copy calls per rank,
#           true_ring shows ~zero.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${ROCM_BUG_TEST_IMAGE:-lorrisync/therock-main:gfx94X_pytorch2.12_rocm7.14_96bfee1}"
NPROC="${NPROC:-8}"
HIDDEN="${HIDDEN:-4096}"
LAYERS="${LAYERS:-4}"
SEQ="${SEQ:-256}"
BATCH="${BATCH:-1}"
ITERS="${ITERS:-3}"
NUMEL="${NUMEL:-16777216}"     # 32 MB bf16 per rank, 256 MB total AG payload
MODE="${MODE:-regular_ag}"     # regular_ag (cudaMalloc'd buffer, no LOCAL_REGISTER crash since we set it 0)
OUT_HOST="${OUT_HOST:-${SCRIPT_DIR}/rocprof_out_$(date +%s)}"
mkdir -p "$OUT_HOST"

PROBE_B64=$(base64 -w0 "${SCRIPT_DIR}/fsdp_like_ag_probe.py")
INTERPOSER_B64=$(base64 -w0 "${SCRIPT_DIR}/hip_attr_drain_preload.c")

run_one() {
  local tag="$1"           # e.g. ce_localreg0 or true_ring
  local cumem="$2"         # 1 or 0
  echo
  echo "=============================================================="
  echo "  CONFIG: ${tag}   (NCCL_CUMEM_ENABLE=${cumem})"
  echo "=============================================================="
  local cname="sdma_rocprof_${tag}_$$"
  docker rm -f "$cname" >/dev/null 2>&1 || true
  set +e
  docker run --name "$cname" \
    --device=/dev/kfd --device=/dev/dri --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined --privileged \
    --ipc=host --shm-size=64g --network=host \
    -e PROBE_B64="$PROBE_B64" \
    -e INTERPOSER_B64="$INTERPOSER_B64" \
    -e NPROC="$NPROC" -e HIDDEN="$HIDDEN" -e LAYERS="$LAYERS" \
    -e SEQ="$SEQ" -e BATCH="$BATCH" -e ITERS="$ITERS" \
    -e NUMEL="$NUMEL" -e MODE="$MODE" \
    -e CUMEM="$cumem" -e TAG="$tag" \
    "$IMAGE" \
    /bin/bash -c '
        set -e
        echo "${PROBE_B64}"     | base64 -d > /tmp/fsdp_like_ag_probe.py
        echo "${INTERPOSER_B64}" | base64 -d > /tmp/hip_attr_drain_preload.c
        gcc -O2 -fPIC -shared /tmp/hip_attr_drain_preload.c \
            -o /tmp/libhip_attr_drain.so -ldl

        mkdir -p /workspace/outputs
        export LD_PRELOAD=/tmp/libhip_attr_drain.so
        export HSA_NO_SCRATCH_RECLAIM=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
        export OMP_NUM_THREADS=8
        export MASTER_ADDR=127.0.0.1
        export MASTER_PORT=29591
        # FSDP/Torchtitan-style CE env, with LOCAL_REGISTER=0 (avoid the
        # hipMemRetainAllocationHandle crash).
        export NCCL_CTA_POLICY=2
        export NCCL_CUMEM_ENABLE="${CUMEM}"
        export NCCL_LOCAL_REGISTER=0
        export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true
        env | grep -E "^(NCCL_|TORCH_NCCL_|LD_PRELOAD)" | sort

        # Each torchrun child writes its rocprof output to a per-rank dir.
        # rocprofv3 supports %p (pid) in -d; we use a stable per-rank dir via
        # a wrapper script that picks up LOCAL_RANK.
        cat > /tmp/rocprof_wrap.sh <<EOF
#!/bin/bash
set -e
RANK=\${LOCAL_RANK:-\${OMPI_COMM_WORLD_LOCAL_RANK:-0}}
mkdir -p /workspace/outputs/rocprof_'"${tag}"'/rank\${RANK}
exec /opt/venv/bin/rocprofv3 \
    --hsa-amd-trace --memory-copy-trace --stats \
    -d /workspace/outputs/rocprof_'"${tag}"'/rank\${RANK} \
    -o trace \
    --output-format csv \
    -- "\$@"
EOF
        chmod +x /tmp/rocprof_wrap.sh

        echo "=== launching ${TAG} (8 ranks, mode=${MODE}, numel=${NUMEL}, iters=${ITERS}) ==="
        torchrun --nproc_per_node=${NPROC} --nnodes=1 --node_rank=0 \
            --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
            --no-python /tmp/rocprof_wrap.sh \
            python3 /tmp/fsdp_like_ag_probe.py \
              --mode ${MODE} --iters ${ITERS} --numel ${NUMEL} \
              --hidden ${HIDDEN} --layers ${LAYERS} \
              --seq ${SEQ} --batch ${BATCH}

        echo "=== rocprof outputs ==="
        ls -R /workspace/outputs/rocprof_'"${tag}"' | head -50
    '
  local rc=$?
  set -e
  echo "[host] ${tag} rc=${rc}"
  docker cp "${cname}:/workspace/outputs/rocprof_${tag}/." "${OUT_HOST}/rocprof_${tag}/" || true
  docker rm -f "$cname" >/dev/null 2>&1 || true
}

run_one ce_localreg0 1
run_one true_ring    0

echo
echo "=============================================================="
echo "  OUTPUTS extracted to: ${OUT_HOST}"
echo "=============================================================="
ls -R "${OUT_HOST}" | head -60
