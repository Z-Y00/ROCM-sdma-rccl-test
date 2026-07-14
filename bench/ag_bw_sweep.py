#!/usr/bin/env python3
"""AllGather raw-bandwidth sweep: SDMA (copy-engine) vs CU (kernel) path.

Sweeps the total AllGather message size from a few KB up to 1 GiB (powers of
two) and, at each size, times dist.all_gather_into_tensor for two buffer
provenances in the SAME process / process group:

  symm_ag    : cuMem-backed symm_mem buffers  -> RCCL SDMA copy-engine dispatch
  regular_ag : torch.empty caching-allocator  -> RCCL CU-resident kernel

The buffer type (not the env) is what selects the dispatch path; the PG is
created with NCCL_CTA_POLICY_ZERO so the symm path can land on the CE.

Reports NCCL-tests-style busbw = size * (N-1)/N / time and writes:
  * a CSV  (--csv)
  * a PNG figure of busbw vs message size (--png), one curve per mode.

Launch via bench/run_ag_bw_sweep.sh (torchrun --nproc_per_node 8).
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--min-bytes", type=int, default=1 << 10,
                   help="min TOTAL allgather size (default 1 KiB)")
    p.add_argument("--max-bytes", type=int, default=1 << 30,
                   help="max TOTAL allgather size (default 1 GiB)")
    p.add_argument("--factor", type=int, default=2, help="size multiplier per step")
    p.add_argument("--sizes", type=str, default="",
                   help="comma-separated TOTAL allgather sizes in bytes; "
                        "if set, overrides --min-bytes/--max-bytes/--factor")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--timed", type=int, default=30)
    p.add_argument("--modes", type=str, default="symm_ag,regular_ag")
    p.add_argument("--csv", type=str, default="")
    p.add_argument("--png", type=str, default="")
    return p.parse_args()


def init_dist():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    opts = dist.ProcessGroupNCCL.Options()
    opts.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
    dist.init_process_group("nccl", rank=rank, world_size=world,
                            device_id=device, pg_options=opts,
                            timeout=timedelta(minutes=10))
    symm_mem.set_backend("NCCL")
    return rank, world, device


def dtype_from_arg(s: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]


def symm_empty(numel, dtype, device, group_name):
    bytes_per = torch.empty((), dtype=dtype).element_size()
    n_f32 = (numel * bytes_per + 3) // 4
    raw = symm_mem.empty(n_f32, device=device, dtype=torch.float32)
    symm_mem.rendezvous(raw, group=group_name)
    return raw.view(torch.uint8)[: numel * bytes_per].view(dtype)


def time_mode(mode, args, device, rank, world, group, group_name,
              numel_per_rank, dtype):
    output_numel = numel_per_rank * world
    bytes_per_el = torch.empty((), dtype=dtype).element_size()
    output_bytes = output_numel * bytes_per_el

    if mode == "symm_ag":
        x = symm_empty(numel_per_rank, dtype, device, group_name)
        y = symm_empty(output_numel,   dtype, device, group_name)
    else:
        x = torch.empty(numel_per_rank, dtype=dtype, device=device)
        y = torch.empty(output_numel,   dtype=dtype, device=device)
    x.fill_(rank + 1)

    def one_iter():
        dist.all_gather_into_tensor(y, x, group=group, async_op=True).wait()

    for _ in range(args.warmup):
        one_iter()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)

    per_iter_ms = []
    for _ in range(args.timed):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        one_iter()
        e.record()
        torch.cuda.synchronize(device)
        per_iter_ms.append(s.elapsed_time(e))

    local = torch.tensor(per_iter_ms, dtype=torch.float64, device=device)
    dist.all_reduce(local, op=dist.ReduceOp.MAX)  # slowest rank per iter
    max_per_iter = local.tolist()
    med_ms = statistics.median(max_per_iter)
    mean_ms = statistics.fmean(max_per_iter)

    algbw = output_bytes / (med_ms * 1e-3) / 1e9
    busbw = (output_bytes * (world - 1) / world) / (med_ms * 1e-3) / 1e9
    mean_algbw = output_bytes / (mean_ms * 1e-3) / 1e9
    mean_busbw = (output_bytes * (world - 1) / world) / (mean_ms * 1e-3) / 1e9
    return med_ms, mean_ms, algbw, busbw, mean_algbw, mean_busbw, output_bytes


def make_figure(png_path, sizes_bytes, series, world, dtype_str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pretty = {"symm_ag": "SDMA (copy-engine)", "regular_ag": "CU kernel (ring)"}
    color = {"symm_ag": "#d62728", "regular_ag": "#1f77b4"}
    x = [b / (1 << 20) for b in sizes_bytes]  # MiB

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for mode, busbws in series.items():
        ax.plot(x, busbws, marker="o", ms=4, lw=1.8,
                label=pretty.get(mode, mode), color=color.get(mode))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("AllGather message size (MiB, total gathered buffer)")
    ax.set_ylabel("Bus bandwidth (GB/s)")
    ax.set_title(f"RCCL AllGather raw bandwidth on {world}x MI300X ({dtype_str})\n"
                 f"SDMA copy-engine vs CU-resident kernel")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    print(f"[figure] wrote {png_path}", flush=True)


def main():
    args = parse_args()
    dtype = dtype_from_arg(args.dtype)
    rank, world, device = init_dist()
    group = dist.group.WORLD
    group_name = group.group_name
    bytes_per_el = torch.empty((), dtype=dtype).element_size()

    # Build the size ladder (total AG buffer bytes), each a multiple of
    # world * elem_size so it divides evenly across ranks.
    gran = world * bytes_per_el
    if args.sizes:
        raw_sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    else:
        raw_sizes = []
        s = args.min_bytes
        while s <= args.max_bytes:
            raw_sizes.append(s)
            s *= args.factor
    sizes = []
    for raw_size in raw_sizes:
        aligned = max(gran, (raw_size // gran) * gran)
        if aligned not in sizes:
            sizes.append(aligned)

    modes = [m.strip() for m in args.modes.split(",")]
    series = {m: [] for m in modes}
    rows = []

    if rank == 0:
        print(f"\n[init] world={world} dtype={args.dtype} "
              f"sizes={len(sizes)} ({sizes[0]}..{sizes[-1]} B) "
              f"warmup={args.warmup} timed={args.timed}\n", flush=True)
        hdr = f"{'total_bytes':>13} {'per_rank_MiB':>12}"
        for m in modes:
            hdr += f" {m+'_busbw':>18}"
        print(hdr, flush=True)

    for total_bytes in sizes:
        numel_per_rank = (total_bytes // bytes_per_el) // world
        line = {"total_bytes": total_bytes,
                "per_rank_bytes": numel_per_rank * bytes_per_el}
        pr = f"{total_bytes:>13} {numel_per_rank*bytes_per_el/(1<<20):>12.4f}"
        for m in modes:
            med, mean_ms, algbw, busbw, mean_algbw, mean_busbw, obytes = time_mode(
                m, args, device, rank, world, group, group_name,
                numel_per_rank, dtype)
            series[m].append(busbw)
            line[f"{m}_median_ms"] = med
            line[f"{m}_mean_ms"] = mean_ms
            line[f"{m}_median_algbw_GBps"] = algbw
            line[f"{m}_median_busbw_GBps"] = busbw
            line[f"{m}_mean_algbw_GBps"] = mean_algbw
            line[f"{m}_mean_busbw_GBps"] = mean_busbw
            pr += f" {busbw:>18.1f}"
        rows.append(line)
        if rank == 0:
            print(pr, flush=True)

    if rank == 0:
        if args.csv:
            with open(args.csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"[csv] wrote {args.csv}", flush=True)
        if args.png:
            make_figure(args.png, sizes, series, world, args.dtype)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
