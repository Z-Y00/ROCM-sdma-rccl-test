"""
AllReduce + GEMM overlap benchmark.

Models a workload pattern where compute is a long-running loop and one
(or a few) AllReduce overlaps with PART of that loop. Per-shape we time
three blocks of size {gemm_iters GEMMs, comm_iters ARs}:
  * gemm_loop   -- N GEMMs back-to-back on the compute stream
  * comm_loop   -- K ARs back-to-back on the comm stream
  * overlap     -- both issued together
and report `hidden_ms`/`hidden_%` (how much of comm got hidden by the
GEMM loop) and `eff = max(gemm, comm) / overlap` (1.0 = perfect overlap).

Two implementations of "AllReduce", each timed alone and overlapped with a
Llama-70B-style matmul:

  * --mode sdma  (default, requires CE-mode env: NCCL_CTA_POLICY=2 etc.)
      AR(x)  ==  all_gather_into_tensor(out=[W*N], in=x) on SDMA
                 .view(W,N).sum(dim=0)                    on the SMs
      Allocates x and the gather buffer via symmetric memory so the AG
      lands on the Copy Engines and doesn't steal SMs from the GEMM.

  * --mode ref   (requires DEFAULT RCCL env, no CTA_POLICY=0)
      AR(x)  ==  dist.all_reduce(x, op=SUM)               on the SMs
      Tensors are regular (non-symm). Baseline ring/RCCL AllReduce.

Why two modes:
  CE-mode (CTA_POLICY=ZERO) disables SM allocation for collectives. AG
  and RS have a CE-backed implementation; AR does not. Calling
  dist.all_reduce in CE-mode fails inside RCCL with
  HSA_STATUS_ERROR_NOT_INITIALIZED. So we run each mode in its own
  torchrun with the matching env; see ./run_bench.sh for the wrapper.

Each pass per shape times:
  * gemm_only
  * ar_only         (sdma or ref, per mode)
  * overlap         (gemm + ar)
  * eff = max(gemm, ar) / overlap

Numerics check (sdma mode only): we fill the input with a deterministic
pattern so the expected sum is known analytically; no reliance on
dist.all_reduce.
"""
from __future__ import annotations
import argparse

import torch
import torch.distributed as dist

from bench_common import (
    Bench, DistCtx, LLAMA70B_SHAPES, Shape,
    gather_max_ms, init_distributed, make_streams, print_table,
    profile_overlap, symm_empty,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("sdma", "ref"), default="sdma",
                   help="sdma = AG-on-CE + local sum;  ref = dist.all_reduce")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--timed", type=int, default=20)
    p.add_argument("--gemm-iters", type=int, default=8,
                   help="GEMMs per compute-loop block (the continuous loop)")
    p.add_argument("--comm-iters", type=int, default=1,
                   help="ARs per comm-loop block (overlap with PART of GEMM loop)")
    p.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--shape", type=str, default="",
                   help="Override, format 'name:M,N,K'")
    p.add_argument("--ar-elems", type=int, default=0,
                   help="Override AR vector length (default: M*N capped at 256MB)")
    p.add_argument("--profile-dir", type=str, default="",
                   help="If set, profile the overlap path and dump chrome "
                        "trace per rank to {dir}/ar_{mode}_{shape}_rank*.json")
    p.add_argument("--profile-iters", type=int, default=5,
                   help="iters captured under torch.profiler (after warmup)")
    return p.parse_args()


def to_torch_dtype(s: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[s]


# --------------------------------------------------------------------------
# Per-shape run, sdma mode.
# --------------------------------------------------------------------------
def run_shape_sdma(shape: Shape, dtype: torch.dtype, ctx: DistCtx, bench: Bench,
                   ar_elems_override: int, gemm_iters: int, comm_iters: int,
                   profile_dir: str = "", profile_iters: int = 5) -> dict:
    device = ctx.device
    W = ctx.world_size
    compute_s, comm_s = make_streams(device)

    A = torch.randn(shape.M, shape.K, dtype=dtype, device=device)
    B = torch.randn(shape.K, shape.N, dtype=dtype, device=device)

    N = ar_elems_override if ar_elems_override > 0 else (shape.M * shape.N)
    max_n = (256 << 20) // torch.empty((), dtype=dtype).element_size()
    if N > max_n:
        N = max_n

    x = symm_empty(N, dtype, ctx)
    gathered = symm_empty(W * N, dtype, ctx)

    # Deterministic input: x[i] = (rank + 1). Expected sum-across-ranks
    # at every position = sum_{r=0..W-1} (r+1) = W*(W+1)/2.
    fill_val = float(ctx.rank + 1)
    expected = float(W * (W + 1) // 2)
    x.fill_(fill_val)

    # Warm fault-in.
    _ = (A @ B)
    wk = dist.all_gather_into_tensor(gathered, x, async_op=True); wk.wait()
    torch.cuda.synchronize(device)

    # ---- numerics check (analytical, no dist.all_reduce) -----------------
    sdma_out = gathered.view(W, N).sum(dim=0)
    abs_err = (sdma_out.float() - expected).abs().max().item()
    ok = abs_err <= max(1e-2, expected * 5e-2)
    okt = torch.tensor([1 if ok else 0], device=device, dtype=torch.int32)
    dist.all_reduce(okt, op=dist.ReduceOp.MIN)
    if ctx.rank == 0:
        print(f"[{shape.name}] numerics_check  ok={bool(okt.item())}  "
              f"max_abs_err={abs_err:.3e}  expected={expected}",
              flush=True)
    # NOTE: dist.all_reduce above used int32 -- 4 bytes, single element,
    # tiny enough to succeed even in CE-mode env if RCCL falls back. If
    # this becomes an issue we can switch to a hand-rolled all_gather+OR.

    # Restore x for timing (the sum kernel ran on `gathered`, x is intact).
    x.fill_(fill_val)

    # ---- iter fns: each "iter" is one (N-gemm + K-ar) block --------------
    def gemm_loop() -> None:
        compute_s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(compute_s):
            for _ in range(gemm_iters):
                _ = A @ B
        torch.cuda.current_stream(device).wait_stream(compute_s)

    def sdma_ar_loop() -> None:
        comm_s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(comm_s):
            for _ in range(comm_iters):
                wk = dist.all_gather_into_tensor(gathered, x, async_op=True)
                wk.wait()
                _ = gathered.view(W, N).sum(dim=0)
        torch.cuda.current_stream(device).wait_stream(comm_s)

    def overlap_iter() -> None:
        compute_s.wait_stream(torch.cuda.current_stream(device))
        comm_s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(comm_s):
            for _ in range(comm_iters):
                wk = dist.all_gather_into_tensor(gathered, x, async_op=True)
                wk.wait()
                _ = gathered.view(W, N).sum(dim=0)
        with torch.cuda.stream(compute_s):
            for _ in range(gemm_iters):
                _ = A @ B
        torch.cuda.current_stream(device).wait_stream(comm_s)
        torch.cuda.current_stream(device).wait_stream(compute_s)

    gemm_loop_ms = gather_max_ms(bench.run("gemm_loop", gemm_loop),    ctx)
    sdma_ar_ms   = gather_max_ms(bench.run("sdma_ar",   sdma_ar_loop), ctx)
    over_ms      = gather_max_ms(bench.run("overlap",   overlap_iter), ctx)

    profile_overlap(
        overlap_iter, ctx,
        tag=f"ar_sdma_{shape.name}",
        profile_dir=profile_dir,
        n_iters=profile_iters,
    )

    flops_total = gemm_iters * 2 * shape.M * shape.N * shape.K
    gemm_tflops = flops_total / (gemm_loop_ms * 1e-3) / 1e12
    bytes_per_el = torch.empty((), dtype=dtype).element_size()
    busy = comm_iters * (W - 1) * N * bytes_per_el     # AG bus-busy bytes/rank
    ar_gbs = busy / (sdma_ar_ms * 1e-3) / 1e9

    hidden_ms  = max(0.0, min(sdma_ar_ms, sdma_ar_ms - (over_ms - gemm_loop_ms)))
    hidden_pct = (hidden_ms / sdma_ar_ms * 100.0) if sdma_ar_ms > 0 else 0.0
    eff = max(gemm_loop_ms, sdma_ar_ms) / over_ms

    return {
        "mode":        "sdma",
        "shape":       shape.name,
        "N":           f"{N}",
        "G/C":         f"{gemm_iters}/{comm_iters}",
        "gemm_ms":     f"{gemm_loop_ms:7.3f}",
        "TFLOPs":      f"{gemm_tflops:6.1f}",
        "ar_ms":       f"{sdma_ar_ms:7.3f}",
        "ar_GB/s":     f"{ar_gbs:6.1f}",
        "overlap_ms":  f"{over_ms:7.3f}",
        "hidden_ms":   f"{hidden_ms:7.3f}",
        "hidden_%":    f"{hidden_pct:5.1f}",
        "eff":         f"{eff:5.2f}",
        "num_ok":      "OK" if bool(okt.item()) else "FAIL",
    }


# --------------------------------------------------------------------------
# Per-shape run, ref mode (regular tensors, ring AR).
# --------------------------------------------------------------------------
def run_shape_ref(shape: Shape, dtype: torch.dtype, ctx: DistCtx, bench: Bench,
                  ar_elems_override: int, gemm_iters: int, comm_iters: int,
                  profile_dir: str = "", profile_iters: int = 5) -> dict:
    device = ctx.device
    W = ctx.world_size
    compute_s, comm_s = make_streams(device)

    A = torch.randn(shape.M, shape.K, dtype=dtype, device=device)
    B = torch.randn(shape.K, shape.N, dtype=dtype, device=device)

    N = ar_elems_override if ar_elems_override > 0 else (shape.M * shape.N)
    max_n = (256 << 20) // torch.empty((), dtype=dtype).element_size()
    if N > max_n:
        N = max_n

    x_initial = torch.full((N,), float(ctx.rank + 1), dtype=dtype, device=device)
    x = x_initial.clone()

    # Warm.
    _ = (A @ B)
    dist.all_reduce(x.clone(), op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)

    def gemm_loop() -> None:
        compute_s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(compute_s):
            for _ in range(gemm_iters):
                _ = A @ B
        torch.cuda.current_stream(device).wait_stream(compute_s)

    def ref_ar_loop() -> None:
        comm_s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(comm_s):
            for _ in range(comm_iters):
                wk = dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=True)
                wk.wait()
                x.copy_(x_initial)
        torch.cuda.current_stream(device).wait_stream(comm_s)

    def overlap_iter() -> None:
        compute_s.wait_stream(torch.cuda.current_stream(device))
        comm_s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(comm_s):
            for _ in range(comm_iters):
                wk = dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=True)
                wk.wait()
                x.copy_(x_initial)
        with torch.cuda.stream(compute_s):
            for _ in range(gemm_iters):
                _ = A @ B
        torch.cuda.current_stream(device).wait_stream(comm_s)
        torch.cuda.current_stream(device).wait_stream(compute_s)

    gemm_loop_ms = gather_max_ms(bench.run("gemm_loop", gemm_loop),    ctx)
    ref_ar_ms    = gather_max_ms(bench.run("ref_ar",    ref_ar_loop),  ctx)
    over_ms      = gather_max_ms(bench.run("overlap",   overlap_iter), ctx)

    profile_overlap(
        overlap_iter, ctx,
        tag=f"ar_ref_{shape.name}",
        profile_dir=profile_dir,
        n_iters=profile_iters,
    )

    flops_total = gemm_iters * 2 * shape.M * shape.N * shape.K
    gemm_tflops = flops_total / (gemm_loop_ms * 1e-3) / 1e12
    bytes_per_el = torch.empty((), dtype=dtype).element_size()
    busy = comm_iters * 2 * (W - 1) * N * bytes_per_el     # ring AR bus-busy
    ar_gbs = busy / (ref_ar_ms * 1e-3) / 1e9

    hidden_ms  = max(0.0, min(ref_ar_ms, ref_ar_ms - (over_ms - gemm_loop_ms)))
    hidden_pct = (hidden_ms / ref_ar_ms * 100.0) if ref_ar_ms > 0 else 0.0
    eff = max(gemm_loop_ms, ref_ar_ms) / over_ms

    return {
        "mode":        "ref",
        "shape":       shape.name,
        "N":           f"{N}",
        "G/C":         f"{gemm_iters}/{comm_iters}",
        "gemm_ms":     f"{gemm_loop_ms:7.3f}",
        "TFLOPs":      f"{gemm_tflops:6.1f}",
        "ar_ms":       f"{ref_ar_ms:7.3f}",
        "ar_GB/s":     f"{ar_gbs:6.1f}",
        "overlap_ms":  f"{over_ms:7.3f}",
        "hidden_ms":   f"{hidden_ms:7.3f}",
        "hidden_%":    f"{hidden_pct:5.1f}",
        "eff":         f"{eff:5.2f}",
        "num_ok":      "-",
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
            f"\n=== AR + GEMM overlap  [mode={args.mode}]  "
            f"(world={ctx.world_size}, dtype={args.dtype}, "
            f"warmup={args.warmup}, timed={args.timed}, "
            f"gemm_iters={args.gemm_iters}, comm_iters={args.comm_iters}) ===",
            flush=True,
        )

    runner = run_shape_sdma if args.mode == "sdma" else run_shape_ref
    rows = [runner(s, dtype, ctx, bench, args.ar_elems,
                   args.gemm_iters, args.comm_iters,
                   profile_dir=args.profile_dir,
                   profile_iters=args.profile_iters)
            for s in shapes]

    print_table(
        rows,
        headers=("mode", "shape", "N", "G/C",
                 "gemm_ms", "TFLOPs", "ar_ms", "ar_GB/s",
                 "overlap_ms", "hidden_ms", "hidden_%", "eff", "num_ok"),
        rank=ctx.rank,
    )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
