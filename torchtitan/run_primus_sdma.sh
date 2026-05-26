#!/bin/bash
# Launch torchtitan inside our verified-bug image (lorrisync/therock-main:...
# build 39213316d2) via Primus, with the sdma_symm_mem_collectives patch
# enabled. The patch wraps every fully_shard() so FSDP's AG/RS buffers come
# from symm_mem (cuMem) and RCCL dispatches the collective on the SDMA path
# (__amd_rocclr_batchMemOp / hsa_amd_memory_async_batch_copy).
#
# What we do on the host:
#   1. Build libhip_attr_drain.so (interposer for the unrelated
#      cuDeviceGetAttribute TLS-leak bug).
#   2. Snapshot-download the *public* unsloth Llama-3.1 70B (or 8B for smoke)
#      tokenizer / config files to a host dir -- meta-llama is gated so we
#      bypass HF_TOKEN requirements by mirroring through unsloth.
#
# Inside the container:
#   3. pip install Primus dependencies (the image has PyTorch 2.12 + ROCm;
#      Primus itself is pure-Python and tiny).
#   4. Set the CE env we verified works on this build:
#        NCCL_CTA_POLICY=2 NCCL_CUMEM_ENABLE=1
#        NCCL_LOCAL_REGISTER=0
#        TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true
#        LD_PRELOAD=libhip_attr_drain.so
#   5. Run primus-cli direct -- train pretrain --config <SDMA yaml>
#
# Usage:
#   ./run_primus_sdma.sh                            # 70B BF16 SDMA, 5 steps (default)
#   SCALE=8b ./run_primus_sdma.sh                   # 8B smoke (mock_data, no HF)
#   STEPS=20 SCALE=70b ./run_primus_sdma.sh         # longer 70B run
#   ROCM_BUG_TEST_IMAGE=lorrisync/...:other ./run_primus_sdma.sh
#
# Outputs are copied from the container to ${OUTPUTS_HOST}/.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEBUG_DIR="${REPO_DIR}/debug"
PRIMUS_DIR="${REPO_DIR}/primus"

# Auto-init the Primus submodule if the user cloned this repo without
# --recurse-submodules. Single-command reproduction:
#   git clone https://github.com/Z-Y00/ROCM-sdma-rccl-test.git && \
#     cd ROCM-sdma-rccl-test && ./torchtitan/run_primus_sdma.sh
if [ ! -f "${PRIMUS_DIR}/runner/primus-cli" ]; then
    echo "=== Primus submodule not initialized; running 'git submodule update --init --recursive primus'"
    (cd "${REPO_DIR}" && git submodule update --init --recursive primus)
fi

IMAGE="${ROCM_BUG_TEST_IMAGE:-lorrisync/therock-main:gfx94X_pytorch2.12_rocm7.14_96bfee1}"
NPROC="${NPROC:-8}"
SCALE="${SCALE:-70b}"          # 70b or 8b
STEPS="${STEPS:-5}"
TOKENIZER_REPO="${TOKENIZER_REPO:-unsloth/Meta-Llama-3.1-70B-Instruct}"   # public mirror; no HF_TOKEN
HF_TOKEN="${HF_TOKEN:-}"
OUTPUTS_HOST="${OUTPUTS_HOST:-${SCRIPT_DIR}/outputs_primus_sdma_${SCALE}}"
mkdir -p "${OUTPUTS_HOST}"

# Stage tokenizer to a host dir we mount into the container.
TOKENIZER_HOST_DIR="${SCRIPT_DIR}/.primus_tokenizer_cache/${SCALE}"
mkdir -p "${TOKENIZER_HOST_DIR}"

# SDMA_MODE=on (default) selects the SDMA-enabled yaml; SDMA_MODE=off
# selects the CE-baseline yaml that runs through the same Primus stack
# but with our sdma_symm_mem_collectives patch *disabled*. Useful for
# perf-A/B comparisons (with profiling on, the chrome traces let us
# diff the per-AG breakdown).
SDMA_MODE="${SDMA_MODE:-on}"

case "${SCALE}/${SDMA_MODE}" in
    70b/on)
        CONFIG="examples/torchtitan/configs/MI300X/llama3.1_70B-BF16-SDMA-pretrain.yaml"
        ASSETS_IN_CTR="/workspace/llama3_70b_assets"
        ;;
    70b/off)
        CONFIG="examples/torchtitan/configs/MI300X/llama3.1_70B-BF16-CE-baseline-pretrain.yaml"
        ASSETS_IN_CTR="/workspace/llama3_70b_assets"
        ;;
    8b/on)
        CONFIG="examples/torchtitan/configs/MI300X/llama3.1_8B-BF16-SDMA-pretrain.yaml"
        ASSETS_IN_CTR="/workspace/llama3_8b_assets"
        ;;
    *)
        echo "Unknown SCALE=${SCALE} / SDMA_MODE=${SDMA_MODE}" >&2
        echo "Valid combos: 70b/on, 70b/off, 8b/on" >&2
        exit 2
        ;;
esac

CNAME="primus_sdma_${SCALE}_$$"
docker rm -f "${CNAME}" >/dev/null 2>&1 || true

echo "=== Image             : ${IMAGE}"
echo "=== Host kernel       : $(uname -r)"
echo "=== Scale / steps     : ${SCALE} / ${STEPS}"
echo "=== Config            : ${CONFIG}"
echo "=== Tokenizer (host)  : ${TOKENIZER_HOST_DIR} (from public mirror ${TOKENIZER_REPO})"
echo "=== SDMA_MODE         : ${SDMA_MODE}"
echo "=== Outputs (host)    : ${OUTPUTS_HOST}"
echo "=== Primus repo (host): ${PRIMUS_DIR}"

# Base64 the interposer source so the container can build it without a bind
# mount (snap-docker can't reliably bind /apps).
INTERPOSER_B64=$(base64 -w0 "${DEBUG_DIR}/hip_attr_drain_preload.c")

docker run --name "${CNAME}" \
    --device=/dev/kfd --device=/dev/dri --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    --ipc=host --shm-size=64g \
    --network=host \
    -v "${PRIMUS_DIR}:/workspace/primus" \
    -v "${TOKENIZER_HOST_DIR}:${ASSETS_IN_CTR}" \
    -e INTERPOSER_B64="${INTERPOSER_B64}" \
    -e NPROC="${NPROC}" \
    -e STEPS="${STEPS}" \
    -e SCALE="${SCALE}" \
    -e CONFIG_REL="${CONFIG}" \
    -e ASSETS_IN_CTR="${ASSETS_IN_CTR}" \
    -e TOKENIZER_REPO="${TOKENIZER_REPO}" \
    -e HF_TOKEN="${HF_TOKEN}" \
    -e HSA_SDMA_LINEAR_B2B="${HSA_SDMA_LINEAR_B2B:-0}" \
    "${IMAGE}" \
    /bin/bash -c '
        set -e
        export PATH=/opt/rocm/bin:${PATH}

        echo ""
        echo "############################################################"
        echo "  [1/4] Build LD_PRELOAD interposer"
        echo "############################################################"
        echo "${INTERPOSER_B64}" | base64 -d > /tmp/hip_attr_drain_preload.c
        gcc -O2 -fPIC -shared /tmp/hip_attr_drain_preload.c \
            -o /tmp/libhip_attr_drain.so -ldl
        ls -l /tmp/libhip_attr_drain.so

        echo ""
        echo "############################################################"
        echo "  [2/4] Install Primus deps + init torchtitan submodule"
        echo "############################################################"
        # Primus repo is mounted at /workspace/primus (host -> container).
        cd /workspace/primus
        # Init torchtitan submodule if not present.
        if [ ! -f third_party/torchtitan/torchtitan/train.py ]; then
            git submodule update --init --depth 1 third_party/torchtitan
        fi
        # Install the core Primus deps (only what the torchtitan trainer
        # touches). Skip megatron / maxtext / large frameworks to keep the
        # bring-up fast. Anything Primus imports unconditionally at trainer
        # startup time goes in this list.
        pip install --no-cache-dir -q \
            loguru tyro "tomli>=2.0" pyyaml typing_extensions \
            "datasets>=3.6.0" "torchdata>=0.8.0" \
            blobfile sentencepiece tiktoken huggingface_hub \
            pyrsmi plotext expecttest
        # Make torchtitan + Primus importable.
        export PYTHONPATH="/workspace/primus:/workspace/primus/third_party/torchtitan:${PYTHONPATH:-}"

        echo ""
        echo "############################################################"
        echo "  [3/4] Stage public tokenizer assets -> ${ASSETS_IN_CTR}"
        echo "############################################################"
        # Bind-mounted from host so this persists across runs.
        if [ -z "$(ls -A ${ASSETS_IN_CTR} 2>/dev/null)" ]; then
            python3 - <<EOF
import os
from huggingface_hub import snapshot_download
repo  = os.environ["TOKENIZER_REPO"]
dest  = os.environ["ASSETS_IN_CTR"]
token = os.environ.get("HF_TOKEN") or None
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
print(f"[tokenizer] staged {repo} -> {dest}")
EOF
        else
            echo "[tokenizer] reusing cached assets in ${ASSETS_IN_CTR}"
        fi
        ls -la "${ASSETS_IN_CTR}" | head -15

        echo ""
        echo "############################################################"
        echo "  [4/4] Launch Primus -> torchtitan with SDMA patch"
        echo "############################################################"
        # CE env (FSDP forces cta_policy=ZERO via PG opts anyway, but we set
        # the env for completeness). LOCAL_REGISTER=0 sidesteps the unrelated
        # hipMemRetainAllocationHandle SIGSEGV on this build.
        export LD_PRELOAD=/tmp/libhip_attr_drain.so
        export HSA_NO_SCRATCH_RECLAIM=1
        # Critical for SDMA throughput on this build: with the default
        # (HSA_SDMA_LINEAR_B2B=1) the SDMA dispatch is throttled to
        # ~48 GB/s busbw on a 209 MiB AG -> 6.4x slower than the
        # CU-driven path. Setting =0 unlocks the full ~323 GB/s xGMI
        # ceiling and makes the SDMA path bandwidth-equivalent (and
        # slightly faster: 4.75 vs 4.93 ms in debug/run_ag_bw_bench.sh).
        export HSA_SDMA_LINEAR_B2B="${HSA_SDMA_LINEAR_B2B:-0}"
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
        export OMP_NUM_THREADS=8
        export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
        export MASTER_ADDR=127.0.0.1
        export MASTER_PORT=29500
        export PYTORCH_ROCM_ARCH=gfx942
        export NCCL_CTA_POLICY=2
        export NCCL_CUMEM_ENABLE=1
        export NCCL_LOCAL_REGISTER=0
        export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true
        # Single-node rendezvous over loopback (Primus auto-detects an
        # interface from `hostname -I`, which can pick something not present
        # inside the container -- force lo for our local docker test).
        export NCCL_SOCKET_IFNAME=lo
        export GLOO_SOCKET_IFNAME=lo
        export PRIMUS_HF_ASSETS_PATH="${ASSETS_IN_CTR}"
        env | grep -E "^(NCCL_|TORCH_NCCL_|HSA_|LD_PRELOAD|PRIMUS_)" | sort

        mkdir -p /workspace/outputs
        cd /workspace/primus
        # primus-cli direct == run in current shell (we are already inside the
        # container). It auto-detects MI300X and torchruns the CLI main.
        bash runner/primus-cli direct \
             --env NCCL_CTA_POLICY=2 \
             --env NCCL_CUMEM_ENABLE=1 \
             --env NCCL_LOCAL_REGISTER=0 \
             --env TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true \
             --env NCCL_SOCKET_IFNAME=lo \
             --env GLOO_SOCKET_IFNAME=lo \
             --env LD_PRELOAD=/tmp/libhip_attr_drain.so \
             -- train pretrain --config "${CONFIG_REL}" \
            2>&1 | tee /workspace/outputs/train.log
    '
RC=$?

echo ""
echo "=== Extracting outputs with docker cp ==="
docker cp "${CNAME}:/workspace/outputs/." "${OUTPUTS_HOST}/" 2>/dev/null || true
# Primus dumps profile traces under ${dump_folder}/profile_traces/iteration_*;
# both the trainer dump_folder and Primus workspace dirs are inside the container.
docker cp "${CNAME}:/workspace/primus/output/." "${OUTPUTS_HOST}/primus_output/" 2>/dev/null || true
docker cp "${CNAME}:/workspace/primus/outputs/." "${OUTPUTS_HOST}/torchtitan_outputs/" 2>/dev/null || true
docker rm -f "${CNAME}" >/dev/null 2>&1 || true

if [ ${RC} -ne 0 ]; then
    echo "Training exited non-zero (${RC})." >&2
    exit ${RC}
fi

echo ""
echo "Done. Outputs in: ${OUTPUTS_HOST}"
ls -la "${OUTPUTS_HOST}"
