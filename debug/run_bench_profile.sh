#!/bin/bash
# Run the bench (bench_ar_gemm.py --mode sdma) with PyTorch profiling on,
# dump per-rank chrome trace JSON. Workload: symm_mem.empty + rendezvous +
# dist.all_gather_into_tensor -- the canonical CE/SDMA path. We then compare
# its trace structure to the torchtitan outputs_ce_localreg0 traces to see
# if FSDP+LOCAL_REGISTER=0 uses the same kernels/markers.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${ROCM_BUG_TEST_IMAGE:-lorrisync/therock-main:gfx94X_pytorch2.12_rocm7.14_96bfee1}"
NPROC="${NPROC:-8}"
WARMUP="${WARMUP:-2}"
TIMED="${TIMED:-3}"
PROFILE_ITERS="${PROFILE_ITERS:-5}"
TRACE_OUT="${TRACE_OUT:-${SCRIPT_DIR}/bench_traces}"
mkdir -p "${TRACE_OUT}"
rm -f "${TRACE_OUT}"/*.json 2>/dev/null || true

INTERPOSER_B64=$(base64 -w0 "${REPO_DIR}/debug/hip_attr_drain_preload.c")
COMMON_B64=$(base64 -w0 "${REPO_DIR}/bench/bench_common.py")
AR_B64=$(base64 -w0 "${REPO_DIR}/bench/bench_ar_gemm.py")

CNAME="bench_prof_$$"
docker rm -f "${CNAME}" >/dev/null 2>&1 || true

docker run --name "${CNAME}" \
    --device=/dev/kfd --device=/dev/dri --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined --privileged \
    --ipc=host --shm-size=64g --network=host \
    -e INTERPOSER_B64="${INTERPOSER_B64}" \
    -e COMMON_B64="${COMMON_B64}" \
    -e AR_B64="${AR_B64}" \
    -e NPROC="${NPROC}" -e WARMUP="${WARMUP}" -e TIMED="${TIMED}" \
    -e PROFILE_ITERS="${PROFILE_ITERS}" \
    "${IMAGE}" \
    /bin/bash -c '
        set -e
        cd /tmp
        echo "${INTERPOSER_B64}" | base64 -d > hip_attr_drain_preload.c
        echo "${COMMON_B64}"     | base64 -d > bench_common.py
        echo "${AR_B64}"         | base64 -d > bench_ar_gemm.py
        gcc -O2 -fPIC -shared hip_attr_drain_preload.c -o libhip_attr_drain.so -ldl
        mkdir -p /traces

        export LD_PRELOAD=/tmp/libhip_attr_drain.so
        export HSA_NO_SCRATCH_RECLAIM=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
        export OMP_NUM_THREADS=8
        export MASTER_ADDR=127.0.0.1
        export MASTER_PORT=29592
        # Full CE/SDMA env -- bench uses symm_mem which needs cuMem.
        export NCCL_CTA_POLICY=2
        export NCCL_CUMEM_ENABLE=1
        export NCCL_LOCAL_REGISTER=2
        export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true
        export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

        echo "=== launching bench_ar_gemm.py --mode sdma with profile ==="
        torchrun --nproc_per_node=${NPROC} --nnodes=1 --node_rank=0 \
            --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
            bench_ar_gemm.py --mode sdma \
            --warmup ${WARMUP} --timed ${TIMED} \
            --profile-dir /traces --profile-iters ${PROFILE_ITERS}
        ls -la /traces
    '
RC=$?
echo "[host] bench profile rc=${RC}"
docker cp "${CNAME}:/traces/." "${TRACE_OUT}/" || true
docker rm -f "${CNAME}" >/dev/null 2>&1 || true
echo "Traces extracted to: ${TRACE_OUT}"
ls -la "${TRACE_OUT}"
