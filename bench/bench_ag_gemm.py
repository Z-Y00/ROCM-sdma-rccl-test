"""
AllGather + GEMM overlap benchmark.

Models a workload pattern where compute is a long-running loop and one
(or a few) collectives overlap with PART of that loop -- e.g. one TP
AllGather hidden inside several layers' worth of matmuls.

For each Llama-70B-style GEMM shape (M,N,K) we measure three timings:
  * gemm_loop   -- N back-to-back A[M,K] @ B[K,N] on the compute stream
  * comm_loop   -- K back-to-back all_gather_into_tensor on the comm stream
                   (CE/SDMA path)
  * overlap     -- both issued together: N GEMMs and K AGs run concurrently

Reported (per iter = per N-GEMM + K-AG block, ms = max across ranks):
  gemm_loop_ms   total time for the N-GEMM loop alone
  comm_loop_ms   total time for the K-AG loop alone
  overlap_ms     total time for the joint case
  hidden_ms      gemm_loop_ms - (overlap_ms - comm_loop_ms)
                  -- how much of comm was successfully hidden by compute,
                     clamped to [0, comm_loop_ms]
  hidden_pct     hidden_ms / comm_loop_ms
  eff            max(gemm_loop_ms, comm_loop_ms) / overlap_ms
                  (1.0 = perfect overlap; <1 = SM contention or serialization)

Launch: see ./run_bench.sh -- in short:
  torchrun --nproc_per_node=8 bench_ag_gemm.py --gemm-iters 8 --comm-iters 1
"""
from __future__ import annotations
import argparse

import torch
import torch.distributed as dist

from bench_common import (
    Bench, DistCtx, LLAMA70B_SHAPES, Shape,
    gather_max_ms, init_distributed, make_streams, print_table, symm_empty,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--timed", type=int, default=20)
    p.add_argument(
        "--gemm-iters", type=int, default=8,
        help="GEMMs per compute-loop block (the continuous loop)",
    )
    p.add_argument(
        "--comm-iters", type=int, default=1,
        help="AGs per comm-loop block (overlap with PART of the GEMM loop)",
    )
    p.add_argument(
        "--dtype", choices=("bf16", "fp16"), default="bf16",
        help="dtype for tensors + GEMM accumulation",
    )
    p.add_argument(
        "--shape", type=str, default="",
        help="Override shape, format 'name:M,N,K' (else sweep llama-70B FFN-up/down)",
    )
    return p.parse_args()


def to_torch_dtype(s: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[s]


# --------------------------------------------------------------------------
# One shape: build tensors, run three timings, return a results row.
# --------------------------------------------------------------------------
def run_shape(shape: Shape, dtype: torch.dtype, ctx: DistCtx, bench: Bench,
              gemm_iters: int, comm_iters: int) -> dict:
    device = ctx.device
    W = ctx.world_size
    compute_s, comm_s = make_streams(device)

    # GEMM operands.
    A = torch.randn(shape.M, shape.K, dtype=dtype, device=device)
    B = torch.randn(shape.K, shape.N, dtype=dtype, device=device)

    # AllGather payload: row-parallel-style. Each rank holds K/W rows of [M],
    # AG to a full [M, K]-shaped column block.
    if shape.K % W != 0:
        raise ValueError(
            f"K={shape.K} must be divisible by world_size={W} for AG decomp"
        )
    n_in = shape.M * (shape.K // W)
    n_out = shape.M * shape.K
    x = symm_empty(n_in, dtype, ctx)
    y = symm_empty(n_out, dtype, ctx)
    x.fill_(float(ctx.rank))  # so we can sanity-check the gather

    # Sanity: one AG to fault-in everything before timing.
    w = dist.all_gather_into_tensor(y, x, async_op=True)
    w.wait()
    torch.cuda.synchronize(device)

    # ---- iter fns: each "iter" is one (N-gemm + K-ag) block -------------
    def gemm_loop() -> None:
        compute_s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(compute_s):
            for _ in range(gemm_iters):
                _ = A @ B
        torch.cuda.current_stream(device).wait_stream(compute_s)

    def comm_loop() -> None:
        comm_s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(comm_s):
            for _ in range(comm_iters):
                wk = dist.all_gather_into_tensor(y, x, async_op=True)
                wk.wait()
        torch.cuda.current_stream(device).wait_stream(comm_s)

    def overlap_iter() -> None:
        compute_s.wait_stream(torch.cuda.current_stream(device))
        comm_s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(comm_s):
            for _ in range(comm_iters):
                wk = dist.all_gather_into_tensor(y, x, async_op=True)
                wk.wait()
        with torch.cuda.stream(compute_s):
            for _ in range(gemm_iters):
                _ = A @ B
        torch.cuda.current_stream(device).wait_stream(comm_s)
        torch.cuda.current_stream(device).wait_stream(compute_s)

    # ---- time them ----------------------------------------------------
    gemm_loop_ms = gather_max_ms(bench.run("gemm_loop", gemm_loop),  ctx)
    comm_loop_ms = gather_max_ms(bench.run("comm_loop", comm_loop),  ctx)
    over_ms      = gather_max_ms(bench.run("overlap",   overlap_iter), ctx)

    # ---- derived metrics ----------------------------------------------
    flops_per_gemm = 2 * shape.M * shape.N * shape.K
    flops_total    = gemm_iters * flops_per_gemm
    bytes_per_el   = torch.empty((), dtype=dtype).element_size()
    ag_busy_per    = (W - 1) * n_in * bytes_per_el      # AG bus-busy bytes per rank
    ag_busy_total  = comm_iters * ag_busy_per

    gemm_tflops = flops_total   / (gemm_loop_ms * 1e-3) / 1e12
    ag_gbs      = ag_busy_total / (comm_loop_ms * 1e-3) / 1e9

    # How much of comm was hidden by the GEMM loop?
    #   hidden_ms = comm_loop_ms - (overlap_ms - gemm_loop_ms)
    # clamped to [0, comm_loop_ms].
    hidden_ms  = max(0.0, min(comm_loop_ms, comm_loop_ms - (over_ms - gemm_loop_ms)))
    hidden_pct = (hidden_ms / comm_loop_ms * 100.0) if comm_loop_ms > 0 else 0.0
    eff        = max(gemm_loop_ms, comm_loop_ms) / over_ms  # 1.0 = perfect

    return {
        "shape":       shape.name,
        "M,N,K":       f"{shape.M},{shape.N},{shape.K}",
        "G/C":         f"{gemm_iters}/{comm_iters}",
        "gemm_ms":     f"{gemm_loop_ms:8.3f}",
        "TFLOPs":      f"{gemm_tflops:6.1f}",
        "comm_ms":     f"{comm_loop_ms:8.3f}",
        "ag_GB/s":     f"{ag_gbs:6.1f}",
        "overlap_ms":  f"{over_ms:8.3f}",
        "hidden_ms":   f"{hidden_ms:7.3f}",
        "hidden_%":    f"{hidden_pct:5.1f}",
        "eff":         f"{eff:5.2f}",
    }


def main() -> None:
    args = parse_args()
    ctx = init_distributed()
    dtype = to_torch_dtype(args.dtype)
    bench = Bench(ctx.device, warmup=args.warmup, timed=args.timed)

    if args.shape:
        name, mnk = args.shape.split(":")
        M, N, K = (int(v) for v in mnk.split(","))
        shapes = [Shape(name, M, N, K)]
    else:
        shapes = LLAMA70B_SHAPES

    if ctx.rank == 0:
        print(
            f"\n=== AG + GEMM overlap  (world={ctx.world_size}, dtype={args.dtype}, "
            f"warmup={args.warmup}, timed={args.timed}, "
            f"gemm_iters={args.gemm_iters}, comm_iters={args.comm_iters}) ===",
            flush=True,
        )

    rows = [run_shape(s, dtype, ctx, bench, args.gemm_iters, args.comm_iters)
            for s in shapes]
    print_table(
        rows,
        headers=("shape", "M,N,K", "G/C",
                 "gemm_ms", "TFLOPs", "comm_ms", "ag_GB/s",
                 "overlap_ms", "hidden_ms", "hidden_%", "eff"),
        rank=ctx.rank,
    )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
