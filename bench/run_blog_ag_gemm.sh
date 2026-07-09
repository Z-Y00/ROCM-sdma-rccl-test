#!/usr/bin/env bash
# Blog benchmark: AG+GEMM co-run, SDMA (copy-engine) AG vs CU-driven AG.
# Runs bench_ag_gemm.py twice on the same shapes:
#   pass 1 (sdma): CE env -> RCCL dispatches AG on the SDMA copy engines
#   pass 2 (cu)  : default env -> RCCL runs AG as a CU-resident kernel
# The delta in overlap efficiency / hidden% shows how much the SDMA path
# frees CUs for the concurrent GEMM.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEBUG_DIR="${REPO_DIR}/debug"
OUT="${SCRIPT_DIR}/blog_results"
mkdir -p "${OUT}"

IMAGE="${ROCM_BUG_TEST_IMAGE:-rocm/primus:v26.4}"
NPROC="${NPROC:-8}"
WARMUP="${WARMUP:-5}"
TIMED="${TIMED:-30}"
GEMM_ITERS="${GEMM_ITERS:-8}"
COMM_ITERS="${COMM_ITERS:-1}"
TRACE="${TRACE:-1}"          # 1 = dump per-pass Perfetto traces (rank 0)

INTERPOSER_B64=$(base64 -w0 "${DEBUG_DIR}/hip_attr_drain_preload.c")
COMMON_B64=$(base64 -w0 "${SCRIPT_DIR}/bench_common.py")
AG_B64=$(base64 -w0 "${SCRIPT_DIR}/bench_ag_gemm.py")

CNAME="blog_ag_gemm_$$"
docker rm -f "${CNAME}" >/dev/null 2>&1 || true

docker run --name "${CNAME}" \
    --device=/dev/kfd --device=/dev/dri --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined --privileged --ipc=host --shm-size=64g --network=host \
    -e INTERPOSER_B64="${INTERPOSER_B64}" -e COMMON_B64="${COMMON_B64}" -e AG_B64="${AG_B64}" \
    -e NPROC="${NPROC}" -e WARMUP="${WARMUP}" -e TIMED="${TIMED}" \
    -e GEMM_ITERS="${GEMM_ITERS}" -e COMM_ITERS="${COMM_ITERS}" \
    -e TRACE="${TRACE}" \
    "${IMAGE}" \
    /bin/bash -c '
        set -e
        export PATH=/opt/rocm/bin:${PATH}
        cd /tmp
        echo "${INTERPOSER_B64}" | base64 -d > hip_attr_drain_preload.c
        echo "${COMMON_B64}"     | base64 -d > bench_common.py
        echo "${AG_B64}"         | base64 -d > bench_ag_gemm.py
        gcc -O2 -fPIC -shared hip_attr_drain_preload.c -o libhip_attr_drain.so -ldl
        export LD_PRELOAD=/tmp/libhip_attr_drain.so
        export HSA_NO_SCRATCH_RECLAIM=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
        export MASTER_ADDR=127.0.0.1
        export HSA_SDMA_LINEAR_B2B=0
        D="--nproc_per_node=${NPROC} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1"
        if [ "${TRACE}" = "1" ]; then
            export BENCH_TRACE_DIR=/tmp/traces
            mkdir -p /tmp/traces
        fi

        echo "@@@PASS=sdma"
        export BENCH_TAG=sdma
        export NCCL_CTA_POLICY=2 NCCL_CUMEM_ENABLE=1 NCCL_LOCAL_REGISTER=0 \
               TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true
        torchrun ${D} --master_port=12378 bench_ag_gemm.py --mode sdma \
            --warmup ${WARMUP} --timed ${TIMED} --gemm-iters ${GEMM_ITERS} --comm-iters ${COMM_ITERS}

        echo "@@@PASS=cu"
        export BENCH_TAG=cu
        unset NCCL_CTA_POLICY NCCL_LOCAL_REGISTER TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK
        export NCCL_CUMEM_ENABLE=1
        torchrun ${D} --master_port=12379 bench_ag_gemm.py --mode cu \
            --warmup ${WARMUP} --timed ${TIMED} --gemm-iters ${GEMM_ITERS} --comm-iters ${COMM_ITERS}
    ' 2>&1 | tee "${OUT}/ag_gemm_corun.log"

if [ "${TRACE}" = "1" ]; then
    docker cp "${CNAME}:/tmp/traces" "${OUT}/" >/dev/null 2>&1 \
        && echo "Traces: ${OUT}/traces/  (open the .json files in https://ui.perfetto.dev)"
fi
docker rm -f "${CNAME}" >/dev/null 2>&1 || true
echo "Saved: ${OUT}/ag_gemm_corun.log"
