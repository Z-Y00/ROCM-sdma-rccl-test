#!/usr/bin/env bash
# PyTorch/RCCL AllGather bandwidth sweep.
#
# We use PyTorch as the rccl-test harness because it can select the real SDMA
# discriminator: buffer provenance.
#   symm_ag    : symm_mem.empty buffers -> RCCL SDMA/copy-engine dispatch
#   regular_ag : torch.empty buffers    -> RCCL CU-resident kernel path
#
# Outputs:
#   bench/blog_results/rccl_tests_ag.log
#   bench/blog_results/rccl_tests_ag.csv
#   bench/blog_results/rccl_tests_ag.png
#
# Knobs:
#   NPROC=8 MIN_BYTES=1024 MAX_BYTES=1073741824 FACTOR=2 DTYPE=bf16
#   WARMUP=5 TIMED=30 MODES=symm_ag,regular_ag
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT="${SCRIPT_DIR}/blog_results"
mkdir -p "${OUT}"

IMAGE="rocm/primus:v26.4"
NPROC="${NPROC:-8}"
MIN_BYTES="${MIN_BYTES:-1024}"
MAX_BYTES="${MAX_BYTES:-1073741824}"
FACTOR="${FACTOR:-2}"
SIZES="${SIZES:-}"
DTYPE="${DTYPE:-bf16}"
WARMUP="${WARMUP:-5}"
TIMED="${TIMED:-30}"
MODES="${MODES:-symm_ag,regular_ag}"
RESULT_TAG="${RESULT_TAG:-}"
PLOT_MIN_BYTES="${PLOT_MIN_BYTES:-0}"

if [[ -n "${RESULT_TAG}" ]]; then
    LOG="${OUT}/rccl_tests_ag_${RESULT_TAG}.log"
    CSV="${OUT}/rccl_tests_ag_${RESULT_TAG}.csv"
    PNG="${OUT}/rccl_tests_ag_${RESULT_TAG}.png"
else
    LOG="${OUT}/rccl_tests_ag.log"
    CSV="${OUT}/rccl_tests_ag.csv"
    PNG="${OUT}/rccl_tests_ag.png"
fi
CSV_BASENAME="$(basename "${CSV}")"

PROBE_B64="$(base64 -w0 "${SCRIPT_DIR}/ag_bw_sweep.py")"
INTERPOSER_B64="$(base64 -w0 "${REPO_DIR}/debug/hip_attr_drain_preload.c")"

echo "=== Image       : ${IMAGE}"
echo "=== NPROC       : ${NPROC}"
if [[ -n "${SIZES}" ]]; then
    echo "=== sizes       : ${SIZES}"
else
    echo "=== sweep       : ${MIN_BYTES} .. ${MAX_BYTES} bytes, factor=${FACTOR}, dtype=${DTYPE}"
fi
echo "=== modes       : ${MODES}"
echo "=== linear_b2b  : 0"
echo "=== warmup/timed: ${WARMUP}/${TIMED}"
echo "=== output      : ${LOG}"

docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined --privileged --ipc=host --shm-size=64g --network=host \
    -v "${OUT}:/outputs" \
    -e PROBE_B64="${PROBE_B64}" \
    -e INTERPOSER_B64="${INTERPOSER_B64}" \
    -e NPROC="${NPROC}" \
    -e MIN_BYTES="${MIN_BYTES}" \
    -e MAX_BYTES="${MAX_BYTES}" \
    -e FACTOR="${FACTOR}" \
    -e SIZES="${SIZES}" \
    -e DTYPE="${DTYPE}" \
    -e WARMUP="${WARMUP}" \
    -e TIMED="${TIMED}" \
    -e MODES="${MODES}" \
    -e CSV_BASENAME="${CSV_BASENAME}" \
    "${IMAGE}" \
    /bin/bash -lc '
        set -e
        echo "${PROBE_B64}"      | base64 -d > /tmp/ag_bw_sweep.py
        echo "${INTERPOSER_B64}" | base64 -d > /tmp/hip_attr_drain_preload.c
        gcc -O2 -fPIC -shared /tmp/hip_attr_drain_preload.c \
            -o /tmp/libhip_attr_drain.so -ldl

        export LD_PRELOAD=/tmp/libhip_attr_drain.so
        export HSA_NO_SCRATCH_RECLAIM=1
        export HSA_SDMA_LINEAR_B2B=0
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
        export OMP_NUM_THREADS=8
        export MASTER_ADDR=127.0.0.1
        export MASTER_PORT=29591
        export NCCL_CTA_POLICY=2
        export NCCL_CUMEM_ENABLE=1
        export NCCL_LOCAL_REGISTER=0
        export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true
        export NCCL_SOCKET_IFNAME=lo
        export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
        SIZE_ARGS=(--min-bytes "${MIN_BYTES}" --max-bytes "${MAX_BYTES}" --factor "${FACTOR}")
        if [ -n "${SIZES}" ]; then
            SIZE_ARGS=(--sizes "${SIZES}")
        fi

        torchrun --nproc_per_node="${NPROC}" --nnodes=1 --node_rank=0 \
            --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
            /tmp/ag_bw_sweep.py \
            "${SIZE_ARGS[@]}" \
            --dtype "${DTYPE}" \
            --warmup "${WARMUP}" \
            --timed "${TIMED}" \
            --modes "${MODES}" \
            --csv "/outputs/${CSV_BASENAME}"
    ' 2>&1 | tee "${LOG}"

python3 - "${CSV}" "${PNG}" "${PLOT_MIN_BYTES}" <<'PY'
import csv
import sys

csv_path, png_path, plot_min_bytes = sys.argv[1:4]
plot_min_bytes = int(plot_min_bytes)
rows = [r for r in csv.DictReader(open(csv_path)) if int(r["total_bytes"]) >= plot_min_bytes]
if not rows:
    raise SystemExit(f"No rows in {csv_path}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def fmt_size(num_bytes):
    units = ((1 << 30, "GB"), (1 << 20, "MB"), (1 << 10, "KB"))
    for scale, suffix in units:
        if num_bytes >= scale:
            value = num_bytes / scale
            return f"{int(value) if value.is_integer() else value:g}{suffix}"
    return f"{num_bytes}B"

series = {}
for row in rows:
    total = int(row["total_bytes"])
    for mode in ("symm_ag", "regular_ag"):
        key = f"{mode}_median_busbw_GBps"
        if key in row and row[key]:
            series.setdefault(mode, []).append((total, float(row[key])))

labels = {
    "symm_ag": "SDMA copy-engine (symm_mem)",
    "regular_ag": "CU kernel (regular tensor)",
}
colors = {"symm_ag": "#d62728", "regular_ag": "#1f77b4"}

all_sizes = sorted({int(row["total_bytes"]) for row in rows})
x_pos = {size: i for i, size in enumerate(all_sizes)}

fig, ax = plt.subplots(figsize=(10, 5.5))
for mode, data in series.items():
    data = sorted(data)
    ax.plot(
        [x_pos[b] for b, _ in data],
        [bw for _, bw in data],
        marker="o",
        linewidth=1.8,
        markersize=4,
        label=labels.get(mode, mode),
        color=colors.get(mode),
    )

ax.set_xticks(range(len(all_sizes)))
ax.set_xticklabels([fmt_size(size) for size in all_sizes], rotation=30, ha="right")
ax.set_xlabel("AllGather output size (total gathered buffer)")
ax.set_ylabel("Median bus bandwidth (GB/s)")
ax.set_title("PyTorch/RCCL all_gather_into_tensor: SDMA vs CU raw bandwidth (median of timed iters)")
ax.grid(True, axis="y", linestyle=":", alpha=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(png_path, dpi=150)
print(f"CSV: {csv_path}")
print(f"PNG: {png_path}")
PY

echo "Saved: ${LOG}"
