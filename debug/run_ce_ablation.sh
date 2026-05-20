#!/bin/bash
# Ablation harness for the FSDP-like CE all-gather crash.
#
# Drives debug/fsdp_like_ag_probe.py in regular_ag mode with one toggle at a
# time so we can find which env knob is REQUIRED for the SIGSEGV in
#   librccl.so.1 -> hipMemRetainAllocationHandle
#
# The four CE-mode knobs we toggle:
#   NCCL_CTA_POLICY=2           (zero-CTA: send/recv kernels use no SMs)
#   NCCL_CUMEM_ENABLE=1         (cuMem allocator: VA reserve + cuMemCreate)
#   NCCL_LOCAL_REGISTER=2       (auto-register all user comm buffers)
#   TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true
#                               (PyTorch caching-allocator hook -> ncclCommRegister)
#
# Usage:
#   ./run_ce_ablation.sh                # run every preset
#   ./run_ce_ablation.sh <preset_name>  # run one preset (see PRESETS below)
#
# Env overrides:
#   ROCM_BUG_TEST_IMAGE  default: lorrisync/therock-main:gfx94X_pytorch2.12_rocm7.14_96bfee1
#   NPROC                default: 8
#   NUMEL                default: 1048576
#   MODE                 default: regular_ag (also accepts symm_ag, fsdp_forward, all)
#   NCCL_DEBUG           default: WARN (set to INFO for verbose)
#   NCCL_DEBUG_SUBSYS    default: (unset)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${ROCM_BUG_TEST_IMAGE:-lorrisync/therock-main:gfx94X_pytorch2.12_rocm7.14_96bfee1}"
NPROC="${NPROC:-8}"
NUMEL="${NUMEL:-1048576}"
MODE="${MODE:-regular_ag}"
NCCL_DEBUG_OUT="${NCCL_DEBUG:-WARN}"
NCCL_DEBUG_SUBSYS_OUT="${NCCL_DEBUG_SUBSYS:-}"

# Each preset is a SPACE-SEPARATED list of "K=V" pairs that get exported in the
# container before torchrun. The names are stable so we can grep results by
# preset name later.
# Important: RCCL defaults NCCL_LOCAL_REGISTER=1 (enabled). To DISABLE
# registration we must EXPLICITLY set NCCL_LOCAL_REGISTER=0. Likewise we
# explicitly set TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=false to
# disable the allocator hook (else PyTorch may default it on too).
declare -A PRESETS=(
  [baseline_default]="NCCL_LOCAL_REGISTER=0 TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=false"
  [ce_full]="NCCL_CTA_POLICY=2 NCCL_CUMEM_ENABLE=1 NCCL_LOCAL_REGISTER=2 TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true"
  # one knob removed at a time from ce_full
  [no_cta_policy]="NCCL_CUMEM_ENABLE=1 NCCL_LOCAL_REGISTER=2 TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true"
  [no_cumem]="NCCL_CTA_POLICY=2 NCCL_CUMEM_ENABLE=0 NCCL_LOCAL_REGISTER=2 TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true"
  [no_local_register]="NCCL_CTA_POLICY=2 NCCL_CUMEM_ENABLE=1 NCCL_LOCAL_REGISTER=0 TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true"
  [no_allocator_hook]="NCCL_CTA_POLICY=2 NCCL_CUMEM_ENABLE=1 NCCL_LOCAL_REGISTER=2 TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=false"
  [no_reg_at_all]="NCCL_CTA_POLICY=2 NCCL_CUMEM_ENABLE=1 NCCL_LOCAL_REGISTER=0 TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=false"
  # one knob ENABLED at a time on top of registration OFF
  [only_cta_policy]="NCCL_CTA_POLICY=2 NCCL_LOCAL_REGISTER=0 TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=false"
  [only_cumem]="NCCL_CUMEM_ENABLE=1 NCCL_LOCAL_REGISTER=0 TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=false"
  [only_local_register]="NCCL_LOCAL_REGISTER=2 TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=false"
  [only_allocator_hook]="NCCL_LOCAL_REGISTER=0 TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true"
  # interesting combos
  [cumem_plus_register]="NCCL_CUMEM_ENABLE=1 NCCL_LOCAL_REGISTER=2 TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=false"
  [cumem_plus_hook]="NCCL_CUMEM_ENABLE=1 NCCL_LOCAL_REGISTER=0 TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true"
  [cta_plus_cumem]="NCCL_CTA_POLICY=2 NCCL_CUMEM_ENABLE=1 NCCL_LOCAL_REGISTER=0 TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=false"
)
ORDER=(
  baseline_default
  ce_full
  no_cta_policy
  no_cumem
  no_local_register
  no_allocator_hook
  no_reg_at_all
  only_cta_policy
  only_cumem
  only_local_register
  only_allocator_hook
  cumem_plus_register
  cumem_plus_hook
  cta_plus_cumem
)

PROBE_B64=$(base64 -w0 "${SCRIPT_DIR}/fsdp_like_ag_probe.py")
INTERPOSER_B64=$(base64 -w0 "${SCRIPT_DIR}/hip_attr_drain_preload.c")

run_one() {
  local name="$1"
  local envs="$2"
  echo
  echo "=============================================================="
  echo "PRESET   : ${name}"
  echo "ENV ADD  : ${envs:-<none>}"
  echo "MODE     : ${MODE}"
  echo "NPROC    : ${NPROC}"
  echo "NUMEL    : ${NUMEL}"
  echo "=============================================================="

  local extra_export=""
  if [ -n "$envs" ]; then
    for kv in $envs; do
      extra_export+="export ${kv}; "
    done
  fi

  set +e
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
      -e NUMEL="${NUMEL}" \
      -e EXTRA_EXPORTS="${extra_export}" \
      -e NCCL_DEBUG="${NCCL_DEBUG_OUT}" \
      -e NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS_OUT}" \
      "${IMAGE}" \
      /bin/bash -c '
          set -e
          echo "${PROBE_B64}"     | base64 -d > /tmp/fsdp_like_ag_probe.py
          echo "${INTERPOSER_B64}" | base64 -d > /tmp/hip_attr_drain_preload.c
          gcc -O2 -fPIC -shared /tmp/hip_attr_drain_preload.c \
              -o /tmp/libhip_attr_drain.so -ldl

          export LD_PRELOAD=/tmp/libhip_attr_drain.so
          export HSA_NO_SCRATCH_RECLAIM=1
          export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
          export OMP_NUM_THREADS=8
          export MASTER_ADDR=127.0.0.1
          export MASTER_PORT=29591
          # NCCL_DEBUG / NCCL_DEBUG_SUBSYS come in via -e from the host.
          eval "${EXTRA_EXPORTS}"
          echo "[ablation] env after applying preset:"
          env | grep -E "^(NCCL_|TORCH_NCCL_|HSA_|LD_PRELOAD)" | sort
          torchrun --nproc_per_node="${NPROC}" --nnodes=1 --node_rank=0 \
              --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
              /tmp/fsdp_like_ag_probe.py \
              --mode "${MODE}" --numel "${NUMEL}"
      '
  local rc=$?
  set -e
  echo "[ablation] preset=${name} exit=${rc}"
}

if [ $# -gt 0 ]; then
  TARGET="$1"
  if [ -z "${PRESETS[$TARGET]+_}" ]; then
    echo "Unknown preset: $TARGET"
    echo "Known: ${ORDER[*]}"
    exit 2
  fi
  run_one "$TARGET" "${PRESETS[$TARGET]}"
else
  for name in "${ORDER[@]}"; do
    run_one "$name" "${PRESETS[$name]}"
  done
fi
