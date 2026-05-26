#!/usr/bin/env python3
"""
Op-level bandwidth bench for dist.all_gather_into_tensor on 8x MI300X.

Compares two buffer provenances at a fixed payload (matched to the
70B/FSDP=8 production AG: per-rank input ~220 MB bf16, AG output ~1.76 GB):

  symm_ag     : output (and input view-of-output) allocated via
                symm_mem.empty + symm_mem.rendezvous -> cuMem-backed VA
                -> RCCL takes the rocclr SDMA dispatch path
                (__amd_rocclr_batchMemOp.kd / hsa_amd_memory_async_batch_copy).

  regular_ag  : output allocated via torch.empty (caching allocator,
                hipMalloc-backed) -> RCCL falls back to its CU-driven
                ncclDevKernel_Generic_2 path.

Per-iter timing uses cuda events; we max-reduce across ranks so the
reported ms is the actual collective wallclock seen by the slowest rank.

Bandwidth model: an AllGather of input_bytes per rank gathered into N
ranks of output is conventionally reported as
  algbw  = output_bytes / time          (algorithmic bytes/s; the "useful" rate)
  busbw  = output_bytes * (N-1)/N / time
                                         (bus bytes/s; matches NCCL-tests, the
                                         rate the xGMI/NIC links actually run at,
                                         since each rank effectively transmits
                                         (N-1)/N of the output once)

Launch via debug/run_ag_bw_bench.sh.
"""

from __future__ import annotations

import argparse
import os
import statistics
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-bytes", type=int, default=219_088_896,
                   help="per-rank AG input bytes (default 219 088 896 B = 209.5 MiB "
                        "bf16 = roughly the per-rank per-layer Llama-3 70B AG)")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--timed", type=int, default=30)
    p.add_argument("--modes", type=str, default="symm_ag,regular_ag",
                   help="comma-separated; both/either of symm_ag,regular_ag")
    return p.parse_args()


def init_dist():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    opts = dist.ProcessGroupNCCL.Options()
    opts.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
    dist.init_process_group(
        "nccl", rank=rank, world_size=world,
        device_id=device, pg_options=opts,
        timeout=timedelta(minutes=10),
    )
    symm_mem.set_backend("NCCL")
    return rank, local_rank, world, device


def dtype_from_arg(s: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]


def symm_empty(numel: int, dtype: torch.dtype, device: torch.device,
               group_name: str) -> torch.Tensor:
    """Allocate a cuMem-backed symm_mem tensor of `numel` elements at `dtype`,
    pre-rendezvous'd. symm_mem.empty returns float32, so we view via uint8."""
    bytes_per = torch.empty((), dtype=dtype).element_size()
    n_f32 = (numel * bytes_per + 3) // 4
    raw = symm_mem.empty(n_f32, device=device, dtype=torch.float32)
    symm_mem.rendezvous(raw, group=group_name)
    return raw.view(torch.uint8)[: numel * bytes_per].view(dtype)


def time_mode(mode: str, args, device, rank, world, group, group_name,
              numel_per_rank: int, dtype: torch.dtype) -> tuple[float, float]:
    """Returns (median_ms, min_ms) max-reduced across ranks."""
    output_numel = numel_per_rank * world
    bytes_per_el = torch.empty((), dtype=dtype).element_size()
    input_bytes  = numel_per_rank * bytes_per_el
    output_bytes = output_numel * bytes_per_el

    if mode == "symm_ag":
        x = symm_empty(numel_per_rank, dtype, device, group_name)
        y = symm_empty(output_numel,   dtype, device, group_name)
    elif mode == "regular_ag":
        x = torch.full((numel_per_rank,), rank + 1, dtype=dtype, device=device)
        y = torch.empty((output_numel,),  dtype=dtype, device=device)
    else:
        raise ValueError(mode)

    x.fill_(rank + 1)  # deterministic, also forces real allocation

    def one_iter():
        work = dist.all_gather_into_tensor(y, x, group=group, async_op=True)
        work.wait()

    # warmup
    for _ in range(args.warmup):
        one_iter()
    torch.cuda.synchronize(device)
    dist.barrier(group=group)

    # timed
    per_iter_ms: list[float] = []
    for _ in range(args.timed):
        evs = torch.cuda.Event(enable_timing=True)
        eve = torch.cuda.Event(enable_timing=True)
        evs.record()
        one_iter()
        eve.record()
        torch.cuda.synchronize(device)
        per_iter_ms.append(evs.elapsed_time(eve))

    # max across ranks per-iter (= slowest rank's time; what comm actually took)
    local = torch.tensor(per_iter_ms, dtype=torch.float64, device=device)
    out   = torch.empty_like(local)
    dist.all_reduce(local, op=dist.ReduceOp.MAX)
    max_per_iter = local.tolist()

    median_ms = statistics.median(max_per_iter)
    min_ms    = min(max_per_iter)
    mean_ms   = statistics.fmean(max_per_iter)

    # NCCL-tests style busbw: output_bytes * (N-1)/N / time
    algbw_GBps = output_bytes / (median_ms * 1e-3) / 1e9
    busbw_GBps = (output_bytes * (world - 1) / world) / (median_ms * 1e-3) / 1e9
    # Per-rank xGMI egress: each rank sends (N-1)/N of input_bytes to peers
    egress_per_rank_bytes = input_bytes * (world - 1)
    egress_GBps = egress_per_rank_bytes / (median_ms * 1e-3) / 1e9

    if rank == 0:
        print(
            f"  {mode:>11s}  iters={args.timed:>3d}  "
            f"size: in={input_bytes/1024/1024:>7.1f} MiB/rank, "
            f"out={output_bytes/1024/1024/1024:>5.2f} GiB    "
            f"min={min_ms:>7.3f} ms  med={median_ms:>7.3f} ms  mean={mean_ms:>7.3f} ms    "
            f"algbw={algbw_GBps:>6.1f} GB/s  busbw={busbw_GBps:>6.1f} GB/s  "
            f"egress/rank={egress_GBps:>6.1f} GB/s",
            flush=True,
        )

    return median_ms, min_ms


def main():
    args = parse_args()
    dtype = dtype_from_arg(args.dtype)
    rank, _, world, device = init_dist()
    group = dist.group.WORLD
    group_name = group.group_name

    bytes_per_el = torch.empty((), dtype=dtype).element_size()
    numel_per_rank = args.input_bytes // bytes_per_el
    if rank == 0:
        print(
            f"\n[init] world={world}  dtype={args.dtype}  "
            f"input_bytes/rank={args.input_bytes}  numel/rank={numel_per_rank}  "
            f"output_bytes={numel_per_rank*world*bytes_per_el}  "
            f"warmup={args.warmup} timed={args.timed}",
            flush=True,
        )
        print(f"\n--- mode comparison (median/min ms max-reduced across {world} ranks) ---")

    for mode in args.modes.split(","):
        time_mode(mode.strip(), args, device, rank, world, group, group_name,
                  numel_per_rank, dtype)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
