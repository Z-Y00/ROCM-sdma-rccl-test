#!/bin/bash
# Run the AG+GEMM and AR+GEMM overlap benchmarks inside the ROCm container,
# with the hip_attr_drain LD_PRELOAD interposer already in place so we don't
# hit the FABRIC_SUPPORTED TLS-leak bug on the first cuMem-path allocation.
#
# Usage:
#   ./run_bench.sh             # runs both benches
#   ./run_bench.sh ag          # only the AG+GEMM bench
#   ./run_bench.sh ar          # only the AR+GEMM bench
#
# Overrides:
#   ROCM_BUG_TEST_IMAGE  -- override the container image
#   NPROC                -- world size  (default 8)
#   WARMUP, TIMED        -- bench iter counts
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

WHICH="${1:-both}"
NPROC="${NPROC:-8}"
WARMUP="${WARMUP:-5}"
TIMED="${TIMED:-20}"
GEMM_ITERS="${GEMM_ITERS:-8}"
COMM_ITERS="${COMM_ITERS:-1}"

IMAGE="${ROCM_BUG_TEST_IMAGE:-registry-sc-harbor.amd.com/framework/therock-main:1384_gfx94X_7.14.0a20260518_centosstream9_py3.12_pytorch_release-2.11_96bfee1}"

INTERPOSER_B64=$(base64 -w0 "${REPO_DIR}/hip_attr_drain_preload.c")
COMMON_B64=$(base64 -w0 "${SCRIPT_DIR}/bench_common.py")
AG_B64=$(base64     -w0 "${SCRIPT_DIR}/bench_ag_gemm.py")
AR_B64=$(base64     -w0 "${SCRIPT_DIR}/bench_ar_gemm.py")

echo "=== Image       : ${IMAGE}"
echo "=== Host kernel : $(uname -r)"
echo "=== World size  : ${NPROC}    warmup=${WARMUP}  timed=${TIMED}"
echo "=== Which       : ${WHICH}"

docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    --ipc=host --shm-size=64g \
    -e INTERPOSER_B64="${INTERPOSER_B64}" \
    -e COMMON_B64="${COMMON_B64}" \
    -e AG_B64="${AG_B64}" \
    -e AR_B64="${AR_B64}" \
    -e WHICH="${WHICH}" \
    -e NPROC="${NPROC}" \
    -e WARMUP="${WARMUP}" \
    -e TIMED="${TIMED}" \
    -e GEMM_ITERS="${GEMM_ITERS}" \
    -e COMM_ITERS="${COMM_ITERS}" \
    "${IMAGE}" \
    /bin/bash -c '
        set -e
        export PATH=/opt/rocm/bin:${PATH}

        cd /tmp
        echo "${INTERPOSER_B64}" | base64 -d > hip_attr_drain_preload.c
        echo "${COMMON_B64}"     | base64 -d > bench_common.py
        echo "${AG_B64}"         | base64 -d > bench_ag_gemm.py
        echo "${AR_B64}"         | base64 -d > bench_ar_gemm.py

        echo ""
        echo "############################################################"
        echo "  Build libhip_attr_drain.so"
        echo "############################################################"
        gcc -O2 -fPIC -shared hip_attr_drain_preload.c \
            -o libhip_attr_drain.so -ldl
        ls -l libhip_attr_drain.so

        # ---- env shared by all passes ----
        export LD_PRELOAD=/tmp/libhip_attr_drain.so
        export HSA_NO_SCRATCH_RECLAIM=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
        export MASTER_ADDR=127.0.0.1

        # CE/SDMA-mode env (forces zero CTAs so AG / RS land on CE).
        ce_env() {
            export NCCL_CTA_POLICY=2          # NCCL_CTA_POLICY_ZERO
            export NCCL_CUMEM_ENABLE=1
            export NCCL_LOCAL_REGISTER=2
            export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true
        }
        # Default-RCCL env (ring path; required for dist.all_reduce reference).
        ref_env() {
            unset NCCL_CTA_POLICY
            unset NCCL_LOCAL_REGISTER
            unset TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK
            export NCCL_CUMEM_ENABLE=1
        }

        dist_args() {
            local port=$1
            echo "--nproc_per_node=${NPROC} --nnodes=1 --node_rank=0
                  --master_addr=${MASTER_ADDR} --master_port=${port}"
        }

        cd /tmp

        if [ "${WHICH}" = "ag"   ] || [ "${WHICH}" = "both" ]; then
            echo ""
            echo "############################################################"
            echo "  AG + GEMM overlap   (CE-mode env)"
            echo "############################################################"
            ce_env
            torchrun $(dist_args 12378) bench_ag_gemm.py \
                --warmup ${WARMUP} --timed ${TIMED} \
                --gemm-iters ${GEMM_ITERS} --comm-iters ${COMM_ITERS}
        fi

        if [ "${WHICH}" = "ar"   ] || [ "${WHICH}" = "both" ]; then
            echo ""
            echo "############################################################"
            echo "  AR + GEMM overlap   pass 1/2:  SDMA   (CE-mode env)"
            echo "############################################################"
            ce_env
            torchrun $(dist_args 12379) bench_ar_gemm.py \
                --mode sdma --warmup ${WARMUP} --timed ${TIMED} \
                --gemm-iters ${GEMM_ITERS} --comm-iters ${COMM_ITERS}

            echo ""
            echo "############################################################"
            echo "  AR + GEMM overlap   pass 2/2:  REF    (default RCCL env)"
            echo "############################################################"
            ref_env
            torchrun $(dist_args 12380) bench_ar_gemm.py \
                --mode ref  --warmup ${WARMUP} --timed ${TIMED} \
                --gemm-iters ${GEMM_ITERS} --comm-iters ${COMM_ITERS}
        fi
    '
