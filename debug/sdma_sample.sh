#!/bin/bash
# Sample SDMA usage from rocm-smi for a fixed period.
#
# Usage:  ./sdma_sample.sh <out_dir> <duration_sec> [interval_sec]
#
# Writes:
#   <out>/showpids.tsv       per-process SDMA-USED snapshots (tab-separated)
#                            cols: t_unix, pid, name, gpu, vram, sdma_used, cu_occ
#   <out>/xgmi.tsv           per-GPU xGMI bytes snapshots
#                            cols: t_unix, gpu, xgmi_read_kb_total, xgmi_write_kb_total
#   <out>/metrics.tsv        per-GPU aggregate metric snapshots
#                            cols: t_unix, gpu, gfx_busy_acc, mem_busy_acc

set -e
OUT="${1:-/tmp/sdma_$$}"
DUR="${2:-120}"
INT="${3:-0.5}"
mkdir -p "$OUT"

PIDS_TSV="$OUT/showpids.tsv"
XGMI_TSV="$OUT/xgmi.tsv"
MET_TSV="$OUT/metrics.tsv"
> "$PIDS_TSV" ; > "$XGMI_TSV" ; > "$MET_TSV"

echo "[sdma_sample] writing to $OUT for ${DUR}s every ${INT}s"

end=$(awk -v d="$DUR" 'BEGIN { print systime()+d }')
while :; do
  now=$(date +%s.%3N)
  # --showpids: cols PID NAME GPU(s) VRAM-USED SDMA-USED CU-OCCUPANCY
  rocm-smi --showpids 2>/dev/null \
    | awk -v t="$now" '/^[0-9]+/ { print t"\t"$1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6 }' \
    >> "$PIDS_TSV"
  # --showmetrics has xgmi_read_data_acc / xgmi_write_data_acc per GPU
  rocm-smi --showmetrics 2>/dev/null \
    | awk -v t="$now" '
        /^GPU\[[0-9]+\]/ {
          g=$1; gsub(/[^0-9]/,"",g)
          if ($0 ~ /xgmi_read_data_acc/) {
            v=$0; sub(/.*: */,"",v); print t"\t"g"\tread\t"v >> "'"$XGMI_TSV"'"
          }
          if ($0 ~ /xgmi_write_data_acc/) {
            v=$0; sub(/.*: */,"",v); print t"\t"g"\twrite\t"v >> "'"$XGMI_TSV"'"
          }
          if ($0 ~ /gfx_activity_acc/) {
            v=$0; sub(/.*: */,"",v); print t"\t"g"\tgfx_acc\t"v >> "'"$MET_TSV"'"
          }
          if ($0 ~ /mem_activity_acc/) {
            v=$0; sub(/.*: */,"",v); print t"\t"g"\tmem_acc\t"v >> "'"$MET_TSV"'"
          }
        }'
  sleep "$INT"
  # stop after duration
  now_int=$(date +%s)
  if [ "$now_int" -ge "${end%.*}" ]; then break; fi
done
echo "[sdma_sample] done"
