#!/bin/bash
# Profile the overlap path of both AR modes (sdma + ref) and dump
# per-rank Chrome traces to ./traces/.  Also prints a kernel-time
# table on rank 0 so you can read the topline without opening the trace.
#
# Usage:
#   ./run_trace.sh             # both modes, default shapes, small iter counts
#
# Overrides:
#   NPROC=8 WARMUP=2 TIMED=3 PROFILE_ITERS=5
#   TRACE_OUT=/path/to/traces  (default: $PWD/traces)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEBUG_DIR="${REPO_DIR}/debug"

NPROC="${NPROC:-8}"
WARMUP="${WARMUP:-2}"
TIMED="${TIMED:-3}"
PROFILE_ITERS="${PROFILE_ITERS:-5}"
TRACE_OUT="${TRACE_OUT:-${SCRIPT_DIR}/traces}"

IMAGE="${ROCM_BUG_TEST_IMAGE:-registry-sc-harbor.amd.com/framework/therock-main:1384_gfx94X_7.14.0a20260518_centosstream9_py3.12_pytorch_release-2.11_96bfee1}"

INTERPOSER_B64=$(base64 -w0 "${DEBUG_DIR}/hip_attr_drain_preload.c")
COMMON_B64=$(base64 -w0 "${SCRIPT_DIR}/bench_common.py")
AR_B64=$(base64     -w0 "${SCRIPT_DIR}/bench_ar_gemm.py")

mkdir -p "${TRACE_OUT}"
rm -f "${TRACE_OUT}"/ar_*.json 2>/dev/null || true

# Snap-docker can't reliably bind-mount paths under /apps or NFS-backed
# /home, so we use docker cp instead: run a named container, dump traces to
# an in-container path, then docker cp them out and rm the container.
CNAME="sdma_bench_trace_$$"

echo "=== Image       : ${IMAGE}"
echo "=== Host kernel : $(uname -r)"
echo "=== World size  : ${NPROC}   warmup=${WARMUP}  timed=${TIMED}  profile_iters=${PROFILE_ITERS}"
echo "=== Trace out   : ${TRACE_OUT}    (extracted via docker cp)"

# Cleanup leftover container from prior failed run, if any
docker rm -f "${CNAME}" >/dev/null 2>&1 || true

# Note: no --rm and no -v for /traces. We extract after the run.
docker run --name "${CNAME}" \
    --device=/dev/kfd --device=/dev/dri --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    --ipc=host --shm-size=64g \
    -e INTERPOSER_B64="${INTERPOSER_B64}" \
    -e COMMON_B64="${COMMON_B64}" \
    -e AR_B64="${AR_B64}" \
    -e NPROC="${NPROC}" \
    -e WARMUP="${WARMUP}" \
    -e TIMED="${TIMED}" \
    -e PROFILE_ITERS="${PROFILE_ITERS}" \
    "${IMAGE}" \
    /bin/bash -c '
        set -e
        export PATH=/opt/rocm/bin:${PATH}
        mkdir -p /traces

        cd /tmp
        echo "${INTERPOSER_B64}" | base64 -d > hip_attr_drain_preload.c
        echo "${COMMON_B64}"     | base64 -d > bench_common.py
        echo "${AR_B64}"         | base64 -d > bench_ar_gemm.py

        echo ""
        echo "############################################################"
        echo "  Build libhip_attr_drain.so"
        echo "############################################################"
        gcc -O2 -fPIC -shared hip_attr_drain_preload.c \
            -o libhip_attr_drain.so -ldl

        # Shared env
        export LD_PRELOAD=/tmp/libhip_attr_drain.so
        export HSA_NO_SCRATCH_RECLAIM=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
        export MASTER_ADDR=127.0.0.1

        # CE-mode env (sdma path)
        ce_env() {
            export NCCL_CTA_POLICY=2
            export NCCL_CUMEM_ENABLE=1
            export NCCL_LOCAL_REGISTER=2
            export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true
        }
        ref_env() {
            unset NCCL_CTA_POLICY
            unset NCCL_LOCAL_REGISTER
            unset TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK
            export NCCL_CUMEM_ENABLE=1
        }

        dist_args() {
            echo "--nproc_per_node=${NPROC} --nnodes=1 --node_rank=0
                  --master_addr=${MASTER_ADDR} --master_port=$1"
        }

        echo ""
        echo "############################################################"
        echo "  AR + GEMM   trace pass 1/2  SDMA  (CE-mode env)"
        echo "############################################################"
        ce_env
        torchrun $(dist_args 12381) bench_ar_gemm.py \
            --mode sdma --warmup ${WARMUP} --timed ${TIMED} \
            --profile-dir /traces --profile-iters ${PROFILE_ITERS}

        echo ""
        echo "############################################################"
        echo "  AR + GEMM   trace pass 2/2  REF   (default RCCL env)"
        echo "############################################################"
        ref_env
        torchrun $(dist_args 12382) bench_ar_gemm.py \
            --mode ref --warmup ${WARMUP} --timed ${TIMED} \
            --profile-dir /traces --profile-iters ${PROFILE_ITERS}

        echo ""
        echo "############################################################"
        echo "  Traces written inside container"
        echo "############################################################"
        ls -la /traces
    '
RC=$?

echo ""
echo "=== Extracting traces with docker cp ==="
if [ ${RC} -eq 0 ]; then
    docker cp "${CNAME}:/traces/." "${TRACE_OUT}/"
fi
docker rm -f "${CNAME}" >/dev/null 2>&1 || true

if [ ${RC} -ne 0 ]; then
    echo "Container exited non-zero (${RC}); not extracting traces." >&2
    exit ${RC}
fi

echo ""
echo "Done. Traces (chrome trace JSON, openable in chrome://tracing or ui.perfetto.dev):"
ls -la "${TRACE_OUT}"
