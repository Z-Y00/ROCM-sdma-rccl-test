#!/bin/bash
# Run a Llama-3 70B torchtitan training job on single-node 8x MI300X,
# inside the PyTorch-2.12 container produced by ../docker/build_pytorch212.sh.
#
# What this does (in-container):
#   1. Builds the hip_attr_drain LD_PRELOAD interposer from ../debug/.
#   2. git clone torchtitan @ a pinned ref and pip-install its requirements.
#   3. Downloads the Llama-3 tokenizer from a public HuggingFace mirror
#      (unsloth/Meta-Llama-3.1-70B-Instruct -- no HF_TOKEN required).
#      Skip with TOKENIZER_SKIP=1 to use the debug-model flavor.
#   4. torchrun --nproc_per_node=8 -m torchtitan.train against the toml
#      config in ./configs/.
#
# Env knobs (host):
#   ROCM_BUG_TEST_IMAGE   container image (default: pushed PyTorch 2.12 image)
#   NPROC                 GPUs per node (default 8)
#   STEPS                 training steps (default 100)
#   SEQ_LEN               sequence length (default 8192)
#   BATCH_SIZE            per-GPU micro-batch (default 1)
#   FSDP                  data_parallel_shard_degree (default = NPROC)
#   TP                    tensor_parallel_degree (default 1)
#   CONFIG                relative path to the toml (default configs/llama3_70b_mi300x_8gpu.toml)
#   TORCHTITAN_REF        git ref to check out (default: main)
#   TOKENIZER_REPO        HuggingFace repo for tokenizer (default: public mirror
#                         unsloth/Meta-Llama-3.1-70B-Instruct, no auth needed)
#   HF_TOKEN              optional; only needed if you switch TOKENIZER_REPO to
#                         a gated repo (e.g. meta-llama/Meta-Llama-3-70B)
#   TOKENIZER_SKIP=1      use the debugmodel flavor; skips HF download
#   CE_MODE               1 (default) = CE-eligible env
#                          (NCCL_CTA_POLICY=2, NCCL_CUMEM_ENABLE=1,
#                           NCCL_LOCAL_REGISTER=0,
#                           TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true).
#                         0 = stock RCCL env (none of those set).
#                         NOTE on NCCL_LOCAL_REGISTER=0 (not the usual 2):
#                           avoids the hipMemRetainAllocationHandle SIGSEGV on
#                           hipMalloc-backed FSDP buffers in libamdhip64.so.7
#                           build 39213316d2 (see ../README.md section (1b)
#                           and ../debug/ROOT_CAUSE_hipMemRetainAllocationHandle.md).
#                         CAVEAT: rocprof verification (debug/run_bench_rocprof.sh
#                           vs debug/run_sdma_rocprof.sh) shows that with
#                           PyTorch caching-allocator buffers, FSDP does NOT
#                           actually go through SDMA even with this env --
#                           the bytes move via ncclDevKernel_Generic_2 on the
#                           CUs (cuMem IPC is used only for address mapping,
#                           not for hsa_amd_memory_async_batch_copy dispatch).
#                           Getting FSDP onto the bench's SDMA path requires
#                           symm_mem-backed AG buffers (see debug/fsdp_like_ag_probe.py
#                           staged_symm_ag mode for prototype).
#   PROFILE               1 = enable torch.profiler captures
#                         (writes per-rank chrome traces to outputs/profile_trace).
#                         0 (default) = no profiling.
#   PROFILE_FREQ          steps between captures when PROFILE=1 (default 5)
#   OUTPUTS_HOST          host dir to docker-cp outputs to (default ./outputs_run)
#
# Usage:
#   ./run_train.sh                                          # default 100-step run
#   STEPS=20 SEQ_LEN=4096 ./run_train.sh                    # short bring-up
#   TOKENIZER_SKIP=1 STEPS=5 ./run_train.sh                 # smoke, no HF at all
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEBUG_DIR="${REPO_DIR}/debug"

IMAGE="${ROCM_BUG_TEST_IMAGE:-lorrisync/therock-main:gfx94X_pytorch2.12_rocm7.14_96bfee1}"
NPROC="${NPROC:-8}"
STEPS="${STEPS:-100}"
SEQ_LEN="${SEQ_LEN:-4096}"
BATCH_SIZE="${BATCH_SIZE:-1}"
FSDP="${FSDP:-${NPROC}}"
TP="${TP:-1}"
CONFIG="${CONFIG:-configs/llama3_70b_mi300x_8gpu.toml}"
TORCHTITAN_REF="${TORCHTITAN_REF:-v0.2.0}"
TOKENIZER_REPO="${TOKENIZER_REPO:-unsloth/Meta-Llama-3.1-70B-Instruct}"
OUTPUTS_HOST="${OUTPUTS_HOST:-${SCRIPT_DIR}/outputs_run}"
TOKENIZER_SKIP="${TOKENIZER_SKIP:-0}"
HF_TOKEN="${HF_TOKEN:-}"
CE_MODE="${CE_MODE:-1}"
PROFILE="${PROFILE:-0}"
PROFILE_FREQ="${PROFILE_FREQ:-5}"

# Inline the source files (snap-docker can't bind-mount /apps; we extract
# with docker cp at the end, same pattern as bench/run_trace.sh).
INTERPOSER_B64=$(base64 -w0 "${DEBUG_DIR}/hip_attr_drain_preload.c")
CONFIG_B64=$(base64 -w0 "${SCRIPT_DIR}/${CONFIG}")

CNAME="sdma_titan_train_$$"
mkdir -p "${OUTPUTS_HOST}"

echo "=== Image           : ${IMAGE}"
echo "=== Host kernel     : $(uname -r)"
echo "=== Hostname        : $(hostname)"
echo "=== Node GPUs       : ${NPROC}    FSDP=${FSDP}  TP=${TP}"
echo "=== Steps / seq / bs: ${STEPS} / ${SEQ_LEN} / ${BATCH_SIZE}"
echo "=== Config          : ${CONFIG}"
echo "=== torchtitan ref  : ${TORCHTITAN_REF}"
echo "=== CE_MODE         : ${CE_MODE}    PROFILE=${PROFILE}"
echo "=== outputs (host)  : ${OUTPUTS_HOST}    (extracted via docker cp)"
echo "=== Container name  : ${CNAME}"

docker rm -f "${CNAME}" >/dev/null 2>&1 || true

docker run --name "${CNAME}" \
    --device=/dev/kfd --device=/dev/dri --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    --ipc=host --shm-size=64g \
    --network=host \
    -e INTERPOSER_B64="${INTERPOSER_B64}" \
    -e CONFIG_B64="${CONFIG_B64}" \
    -e NPROC="${NPROC}" \
    -e STEPS="${STEPS}" \
    -e SEQ_LEN="${SEQ_LEN}" \
    -e BATCH_SIZE="${BATCH_SIZE}" \
    -e FSDP="${FSDP}" \
    -e TP="${TP}" \
    -e CONFIG_NAME="$(basename "${CONFIG}")" \
    -e TORCHTITAN_REF="${TORCHTITAN_REF}" \
    -e TOKENIZER_REPO="${TOKENIZER_REPO}" \
    -e TOKENIZER_SKIP="${TOKENIZER_SKIP}" \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e CE_MODE="${CE_MODE}" \
    -e PROFILE="${PROFILE}" \
    -e PROFILE_FREQ="${PROFILE_FREQ}" \
    -e NCCL_DEBUG="${NCCL_DEBUG:-}" \
    -e NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-}" \
    "${IMAGE}" \
    /bin/bash -c '
        set -e
        export PATH=/opt/rocm/bin:${PATH}

        echo ""
        echo "############################################################"
        echo "  [1/4] Build LD_PRELOAD interposer"
        echo "############################################################"
        mkdir -p /workspace /workspace/outputs
        echo "${INTERPOSER_B64}" | base64 -d > /tmp/hip_attr_drain_preload.c
        gcc -O2 -fPIC -shared /tmp/hip_attr_drain_preload.c \
            -o /tmp/libhip_attr_drain.so -ldl
        ls -l /tmp/libhip_attr_drain.so

        echo ""
        echo "############################################################"
        echo "  [2/4] Fetch torchtitan @ ${TORCHTITAN_REF}"
        echo "############################################################"
        cd /workspace
        git clone --depth 1 --branch "${TORCHTITAN_REF}" \
            https://github.com/pytorch/torchtitan.git \
            || git clone https://github.com/pytorch/torchtitan.git
        cd /workspace/torchtitan
        if [ "${TORCHTITAN_REF}" != "main" ]; then
            git fetch origin "${TORCHTITAN_REF}" && git checkout "${TORCHTITAN_REF}"
        fi
        echo "torchtitan HEAD: $(git log -1 --format=%H) $(git log -1 --format=%s)"
        pip install -e . --no-deps                # install torchtitan itself
        pip install -r requirements.txt           # its python deps (tomli, datasets, etc.)
        # blobfile is required by the Llama-3 tokenizer
        pip install blobfile sentencepiece tiktoken

        echo ""
        echo "############################################################"
        echo "  [3/4] Tokenizer setup"
        echo "############################################################"
        # torchtitan v0.2.0+ uses --model.hf_assets_path pointing at a HF-style
        # directory (tokenizer.json / tokenizer_config.json / special_tokens_map.json).
        # Even the debug flavor needs SOMETHING here -- torchtitan ships a test
        # tokenizer at tests/assets/tokenizer that we can reuse for smoke runs.
        ASSETS_DIR="/workspace/torchtitan/assets/hf/llama3"
        TEST_ASSETS_DIR="/workspace/torchtitan/tests/assets/tokenizer"
        if [ "${TOKENIZER_SKIP}" = "1" ]; then
            echo "TOKENIZER_SKIP=1 -- will run flavor=debugmodel with the built-in"
            echo "                    test tokenizer at ${TEST_ASSETS_DIR}"
            if [ ! -d "${TEST_ASSETS_DIR}" ]; then
                echo "ERROR: ${TEST_ASSETS_DIR} not in this torchtitan ref" >&2
                exit 2
            fi
            FLAVOR_OVERRIDE="--model.flavor=debugmodel --model.hf_assets_path=${TEST_ASSETS_DIR}"
        else
            echo "Downloading tokenizer from public mirror: ${TOKENIZER_REPO}"
            # Public repo (e.g. unsloth/Meta-Llama-3.1-70B-Instruct) needs no
            # HF_TOKEN; pass it through if set so this also works for gated
            # repos (meta-llama/...) without further edits.
            [ -n "${HF_TOKEN}" ] && export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
            mkdir -p "${ASSETS_DIR}"
            python3 - <<EOF
import os, sys
from huggingface_hub import snapshot_download
repo  = os.environ["TOKENIZER_REPO"]
token = os.environ.get("HF_TOKEN") or None
dest  = "${ASSETS_DIR}"
# Pull all tokenizer-related files plus minimal model metadata.
snapshot_download(
    repo,
    allow_patterns=[
        "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "tokenizer.model",
        "original/tokenizer.model",
        "config.json", "generation_config.json",
    ],
    local_dir=dest, token=token,
)
print(f"OK: pulled tokenizer assets from {repo} -> {dest}")
EOF
            ls -la "${ASSETS_DIR}" 2>/dev/null
            FLAVOR_OVERRIDE="--model.hf_assets_path=${ASSETS_DIR}"
        fi

        echo ""
        echo "############################################################"
        echo "  [4/4] torchrun training"
        echo "############################################################"
        echo "${CONFIG_B64}" | base64 -d > /workspace/torchtitan/run.toml

        # Always-on env (interposer is harmless on the ring path, so safe).
        export LD_PRELOAD=/tmp/libhip_attr_drain.so
        export HSA_NO_SCRATCH_RECLAIM=1
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
        export OMP_NUM_THREADS=8
        export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
        export MASTER_ADDR=127.0.0.1
        export MASTER_PORT=29500
        export PYTORCH_ROCM_ARCH=gfx942

        if [ "${CE_MODE}" = "1" ]; then
            echo "CE_MODE=1 -- enabling CE-eligible env (cuMem IPC channels)"
            export NCCL_CTA_POLICY=2          # NCCL_CTA_POLICY_ZERO (request CE)
            export NCCL_CUMEM_ENABLE=1
            # NCCL_LOCAL_REGISTER=0 (not 2!) -- avoids the
            # hipMemRetainAllocationHandle SIGSEGV on hipMalloc-backed FSDP
            # buffers in libamdhip64.so.7 build 39213316d2. RCCL still picks
            # P2P/CUMEM channels (cuMem-backed peer IPC for addressing) but
            # the actual data-moving GPU kernel is ncclDevKernel_Generic_2,
            # NOT __amd_rocclr_batchMemOp/hsa_amd_memory_async_batch_copy.
            # So this is NOT yet on the real SDMA dispatch path the bench
            # uses; see README (1b) for details and the staged_symm_ag prototype.
            export NCCL_LOCAL_REGISTER=0
            export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true
        else
            echo "CE_MODE=0 -- stock RCCL ring path (no CTA_POLICY=ZERO)"
            unset NCCL_CTA_POLICY NCCL_LOCAL_REGISTER \
                  TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK
            export NCCL_CUMEM_ENABLE=1   # leave this set so allocator path is consistent
        fi

        PROF_OVERRIDE=""
        if [ "${PROFILE}" = "1" ]; then
            echo "PROFILE=1 -- enabling torch.profiler captures every ${PROFILE_FREQ} steps"
            # tyro treats bool flags as present/absent; no =true.
            PROF_OVERRIDE="--profiling.enable-profiling --profiling.profile-freq=${PROFILE_FREQ}"
        fi

        cd /workspace/torchtitan
        torchrun \
            --nproc_per_node=${NPROC} --nnodes=1 --node_rank=0 \
            --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
            -m torchtitan.train \
            --job.config_file=/workspace/torchtitan/run.toml \
            --training.steps=${STEPS} \
            --training.seq_len=${SEQ_LEN} \
            --training.local_batch_size=${BATCH_SIZE} \
            --parallelism.data_parallel_shard_degree=${FSDP} \
            --parallelism.tensor_parallel_degree=${TP} \
            ${PROF_OVERRIDE} \
            ${FLAVOR_OVERRIDE} \
            2>&1 | tee /workspace/outputs/train.log
    '
RC=$?

echo ""
echo "=== Extracting outputs with docker cp ==="
if [ ${RC} -eq 0 ]; then
    docker cp "${CNAME}:/workspace/outputs/." "${OUTPUTS_HOST}/" || true
fi
docker rm -f "${CNAME}" >/dev/null 2>&1 || true

if [ ${RC} -ne 0 ]; then
    echo "Training container exited non-zero (${RC})." >&2
    exit ${RC}
fi

echo ""
echo "Done. Outputs in: ${OUTPUTS_HOST}"
ls -la "${OUTPUTS_HOST}"
