#!/bin/bash
# Sweep MODE x NCCL_LOCAL_REGISTER and count hsa_amd_memory_async_batch_copy
# per rank under rocprofv3, to figure out exactly what's required for FSDP
# to use the bench's SDMA dispatch path.
#
# Modes (in fsdp_like_ag_probe.py):
#   symm_ag        - symm_mem.empty + rendezvous + dist.all_gather_into_tensor
#                    (bench-equivalent; known to use SDMA)
#   regular_ag     - torch.full + dist.all_gather_into_tensor
#                    (FSDP buffer shape; suspected NOT on SDMA path)
#   staged_symm_ag - regular shard staged through symm_mem buffer
#                    (does the AG itself on symm_mem; should be SDMA)
#   fsdp_forward   - tiny FSDP2 model with fully_shard
#                    (the actual FSDP stack we care about)
#
# LOCAL_REG values:
#   0  - workaround for the hipMemRetainAllocationHandle crash
#   2  - real CE config (crashes for regular_ag / fsdp_forward on this build)
#
# Discriminator: hsa_amd_memory_async_batch_copy call count per rank.
# Non-zero => the AG is going through __amd_rocclr_batchMemOp.kd => SDMA.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${ROCM_BUG_TEST_IMAGE:-lorrisync/therock-main:gfx94X_pytorch2.12_rocm7.14_96bfee1}"
NPROC="${NPROC:-8}"
HIDDEN="${HIDDEN:-4096}"
LAYERS="${LAYERS:-4}"
SEQ="${SEQ:-256}"
BATCH="${BATCH:-1}"
ITERS="${ITERS:-3}"
NUMEL="${NUMEL:-16777216}"
OUT_HOST="${OUT_HOST:-${SCRIPT_DIR}/sdma_sweep_out}"
rm -rf "$OUT_HOST"
mkdir -p "$OUT_HOST"

# Modes x LOCAL_REGISTER values to sweep
SWEEP="${SWEEP:-symm_ag:0 symm_ag:2 regular_ag:0 staged_symm_ag:0 fsdp_forward:0}"

PROBE_B64=$(base64 -w0 "${SCRIPT_DIR}/fsdp_like_ag_probe.py")
INTERPOSER_B64=$(base64 -w0 "${SCRIPT_DIR}/hip_attr_drain_preload.c")

run_one() {
  local mode="$1"
  local lreg="$2"
  local tag="${mode}_lr${lreg}"
  echo
  echo "=============================================================="
  echo "  CELL: mode=${mode}  NCCL_LOCAL_REGISTER=${lreg}"
  echo "=============================================================="
  local cname="sdma_sweep_${tag}_$$"
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
    -e NUMEL="$NUMEL" -e MODE="$mode" -e LREG="$lreg" -e TAG="$tag" \
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

        # Full CE env, varying only NCCL_LOCAL_REGISTER per cell
        export NCCL_CTA_POLICY=2
        export NCCL_CUMEM_ENABLE=1
        export NCCL_LOCAL_REGISTER="${LREG}"
        export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true
        env | grep -E "^(NCCL_|TORCH_NCCL_|LD_PRELOAD)" | sort

        cat > /tmp/rocprof_wrap.sh <<EOF
#!/bin/bash
set -e
RANK=\${LOCAL_RANK:-0}
mkdir -p /workspace/outputs/'"${tag}"'/rank\${RANK}
exec /opt/venv/bin/rocprofv3 \
    --hsa-amd-trace --stats \
    -d /workspace/outputs/'"${tag}"'/rank\${RANK} \
    -o trace --output-format csv \
    -- "\$@"
EOF
        chmod +x /tmp/rocprof_wrap.sh

        echo "=== launching ${TAG} (8 ranks, mode=${MODE}, lr=${LREG}, numel=${NUMEL}) ==="
        torchrun --nproc_per_node=${NPROC} --nnodes=1 --node_rank=0 \
            --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
            --no-python /tmp/rocprof_wrap.sh \
            python3 /tmp/fsdp_like_ag_probe.py \
              --mode ${MODE} --iters ${ITERS} --numel ${NUMEL} \
              --hidden ${HIDDEN} --layers ${LAYERS} \
              --seq ${SEQ} --batch ${BATCH} \
              || echo "[cell] python exit=$?"
    '
  local rc=$?
  set -e
  echo "[host] ${tag} container_rc=${rc}"
  docker cp "${cname}:/workspace/outputs/${tag}/." "${OUT_HOST}/${tag}/" 2>/dev/null || true
  docker rm -f "$cname" >/dev/null 2>&1 || true
}

for spec in $SWEEP; do
  mode=${spec%%:*}
  lreg=${spec##*:}
  run_one "$mode" "$lreg"
done

echo
echo "=============================================================="
echo "  Results extracted to: ${OUT_HOST}"
echo "=============================================================="
ls -1 "$OUT_HOST"
