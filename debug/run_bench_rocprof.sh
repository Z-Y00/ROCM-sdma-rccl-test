#!/bin/bash
# Run the bench (bench_ar_gemm.py --mode sdma) wrapped in rocprofv3
# --hsa-amd-trace. Expectation: since the PyTorch trace already shows
# __amd_rocclr_batchMemOp.kd (the GPU stub for the rocclr batched copy
# dispatch), rocprof should show non-zero hsa_amd_memory_async_batch_copy
# calls -- which is the canonical SDMA dispatch API.
#
# Compare-against: rocprof_out/rocprof_ce_localreg0 (FSDP-like regular_ag)
# showed ZERO batch_copy and ZERO __amd_rocclr_batchMemOp -- if bench shows
# non-zero here, that's definitive evidence that FSDP is NOT on the SDMA
# path despite the env knobs.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE="${ROCM_BUG_TEST_IMAGE:-lorrisync/therock-main:gfx94X_pytorch2.12_rocm7.14_96bfee1}"
NPROC="${NPROC:-8}"
OUT_HOST="${OUT_HOST:-${SCRIPT_DIR}/bench_rocprof_out}"
rm -rf "$OUT_HOST"
mkdir -p "$OUT_HOST"

INTERPOSER_B64=$(base64 -w0 "${REPO_DIR}/debug/hip_attr_drain_preload.c")
COMMON_B64=$(base64 -w0 "${REPO_DIR}/bench/bench_common.py")
AR_B64=$(base64 -w0 "${REPO_DIR}/bench/bench_ar_gemm.py")

CNAME="bench_rocprof_$$"
docker rm -f "$CNAME" >/dev/null 2>&1 || true
docker run --name "$CNAME" \
    --device=/dev/kfd --device=/dev/dri --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined --privileged \
    --ipc=host --shm-size=64g --network=host \
    -e INTERPOSER_B64="${INTERPOSER_B64}" \
    -e COMMON_B64="${COMMON_B64}" \
    -e AR_B64="${AR_B64}" \
    -e NPROC="${NPROC}" \
    "$IMAGE" \
    /bin/bash -c '
        set -e
        cd /tmp
        echo "${INTERPOSER_B64}" | base64 -d > hip_attr_drain_preload.c
        echo "${COMMON_B64}"     | base64 -d > bench_common.py
        echo "${AR_B64}"         | base64 -d > bench_ar_gemm.py
        gcc -O2 -fPIC -shared hip_attr_drain_preload.c -o libhip_attr_drain.so -ldl
        mkdir -p /workspace/outputs

        export LD_PRELOAD=/tmp/libhip_attr_drain.so
        export HSA_NO_SCRATCH_RECLAIM=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
        export OMP_NUM_THREADS=8
        export MASTER_ADDR=127.0.0.1
        export MASTER_PORT=29593
        export NCCL_CTA_POLICY=2
        export NCCL_CUMEM_ENABLE=1
        export NCCL_LOCAL_REGISTER=2
        export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true

        cat > /tmp/rocprof_wrap.sh <<EOF
#!/bin/bash
set -e
RANK=\${LOCAL_RANK:-0}
mkdir -p /workspace/outputs/rocprof_bench/rank\${RANK}
exec /opt/venv/bin/rocprofv3 \
    --hsa-amd-trace --memory-copy-trace --stats \
    -d /workspace/outputs/rocprof_bench/rank\${RANK} \
    -o trace \
    --output-format csv \
    -- "\$@"
EOF
        chmod +x /tmp/rocprof_wrap.sh

        echo "=== launching bench_ar_gemm.py --mode sdma under rocprofv3 ==="
        torchrun --nproc_per_node=${NPROC} --nnodes=1 --node_rank=0 \
            --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
            --no-python /tmp/rocprof_wrap.sh \
            python3 /tmp/bench_ar_gemm.py --mode sdma --warmup 2 --timed 3
        ls -R /workspace/outputs/rocprof_bench | head -30
    '
RC=$?
docker cp "${CNAME}:/workspace/outputs/rocprof_bench/." "${OUT_HOST}/" || true
docker rm -f "$CNAME" >/dev/null 2>&1 || true
echo "[host] bench rocprof rc=${RC}"
echo "[host] outputs in: ${OUT_HOST}"
