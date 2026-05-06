"""
CE AllGather + GEMM overlap benchmark.

Tests whether Copy Engine AllGather can truly overlap with GEMM
when they run on separate streams (CE on comm_stream, GEMM on compute_stream).

Note: only AllGather is supported by the CE path in RCCL.
      ReduceScatter has no CE implementation (see ce_coll.cc).

Launch with:
    torchrun --nproc_per_node=<NUM_GPUS> ce_overlap_gemm_test.py [options]
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--numel", type=int, default=1024 * 1024)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--dtype", type=str, default="float16",
                    choices=["float16", "bfloat16", "float32"])
    p.add_argument("--profile-dir", type=str, default="",
                    help="Directory for torch profiler traces (empty = skip profiling)")
    return p.parse_args()


DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def bytes_to_gb(b):
    return b / (1024 ** 3)


def main():
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", local_rank)

    # ── Step 1: Allocate GEMM tensors BEFORE any distributed / symm_mem init ──
    m, n, k = args.m, args.n, args.k
    a = torch.randn(m, k, dtype=dtype, device=device)
    b = torch.randn(k, n, dtype=dtype, device=device)
    compute_stream = torch.cuda.Stream(device=device)
    comm_stream = torch.cuda.Stream(device=device)

    # Warm up GEMM to make sure rocBLAS is initialized
    with torch.cuda.stream(compute_stream):
        torch.mm(a, b)
    compute_stream.synchronize()
    log(rank, "GEMM pre-init OK")

    # ── Step 2: Initialize process group with zero-CTA for CE collectives ──
    opts = dist.ProcessGroupNCCL.Options()
    opts.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
    dist.init_process_group(backend="nccl", pg_options=opts, device_id=device)
    log(rank, "init_process_group OK")

    # ── Step 3: Set up symmetric memory ──
    symm_mem.set_backend("NCCL")
    group_name = dist.group.WORLD.group_name

    numel = args.numel
    small = symm_mem.empty(numel, device=device)
    large = symm_mem.empty(numel * world_size, device=device)
    symm_mem.rendezvous(small, group=group_name)
    symm_mem.rendezvous(large, group=group_name)
    torch.cuda.synchronize(device)
    log(rank, "symm_mem rendezvous OK")

    # ── Step 4: Test if GEMM still works after symm_mem setup ──
    try:
        with torch.cuda.stream(compute_stream):
            torch.mm(a, b)
        compute_stream.synchronize()
        log(rank, "GEMM after symm_mem: OK")
        gemm_works = True
    except Exception as e:
        log(rank, f"GEMM after symm_mem: FAILED ({e})")
        gemm_works = False

    # ── Step 5: Test CE AllGather ──
    work = dist.all_gather_into_tensor(large, small, async_op=True)
    work.wait()
    torch.cuda.synchronize(device)
    log(rank, "CE AllGather OK")

    if not gemm_works:
        log(rank, "\nGEMM fails after symm_mem setup on ROCm — cannot run overlap benchmark.")
        log(rank, "Use overlap_comm_gemm.py (regular tensors, no CE) as a workaround.")
        dist.destroy_process_group()
        return

    # ── Step 6: Benchmarks ──
    coll_bytes = large.numel() * large.element_size()

    log(rank, "=" * 72)
    log(rank, "CE AllGather + GEMM Overlap Benchmark")
    log(rank, "=" * 72)
    log(rank, f"  World size:   {world_size}")
    log(rank, f"  GEMM:         M={m} N={n} K={k}  dtype={args.dtype}")
    log(rank, f"  Collectives:  numel/rank={numel:,}  total={numel*world_size:,}")
    log(rank, f"  Warmup={args.warmup}  Iters={args.iters}")
    log(rank, "=" * 72)

    # Baseline: GEMM only
    for _ in range(args.warmup):
        with torch.cuda.stream(compute_stream):
            torch.mm(a, b)
    compute_stream.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record(compute_stream)
    for _ in range(args.iters):
        with torch.cuda.stream(compute_stream):
            torch.mm(a, b)
    e.record(compute_stream)
    e.synchronize()
    gemm_ms = s.elapsed_time(e) / args.iters

    # Baseline: CE AllGather only
    for _ in range(args.warmup):
        with torch.cuda.stream(comm_stream):
            w = dist.all_gather_into_tensor(large, small, async_op=True)
            w.wait()
    comm_stream.synchronize()

    s.record(comm_stream)
    for _ in range(args.iters):
        with torch.cuda.stream(comm_stream):
            w = dist.all_gather_into_tensor(large, small, async_op=True)
            w.wait()
    e.record(comm_stream)
    e.synchronize()
    ag_ms = s.elapsed_time(e) / args.iters

    gemm_tflops = 2.0 * m * n * k / (gemm_ms / 1e3) / 1e12
    ag_bw = bytes_to_gb(coll_bytes) / (ag_ms / 1e3)

    log(rank, f"\n{'─' * 72}")
    log(rank, "STANDALONE BASELINES")
    log(rank, f"{'─' * 72}")
    log(rank, f"  GEMM:            {gemm_ms:.3f} ms  |  {gemm_tflops:.2f} TFLOPS")
    log(rank, f"  CE AllGather:    {ag_ms:.3f} ms  |  {ag_bw:.2f} GB/s")

    # ── Overlap: CE AllGather + GEMM ──
    for _ in range(args.warmup):
        with torch.cuda.stream(comm_stream):
            w = dist.all_gather_into_tensor(large, small, async_op=True)
            w.wait()
        with torch.cuda.stream(compute_stream):
            torch.mm(a, b)
        compute_stream.synchronize()
        comm_stream.synchronize()

    s_comp = torch.cuda.Event(enable_timing=True)
    e_comp = torch.cuda.Event(enable_timing=True)
    s_comm = torch.cuda.Event(enable_timing=True)
    e_comm = torch.cuda.Event(enable_timing=True)

    s_comp.record(compute_stream)
    s_comm.record(comm_stream)
    for _ in range(args.iters):
        with torch.cuda.stream(comm_stream):
            w = dist.all_gather_into_tensor(large, small, async_op=True)
            w.wait()
        with torch.cuda.stream(compute_stream):
            torch.mm(a, b)
    e_comp.record(compute_stream)
    e_comm.record(comm_stream)
    e_comp.synchronize()
    e_comm.synchronize()

    ag_comp_ms = s_comp.elapsed_time(e_comp) / args.iters
    ag_comm_ms = s_comm.elapsed_time(e_comm) / args.iters
    ag_wall = max(ag_comp_ms, ag_comm_ms)
    ag_seq = gemm_ms + ag_ms
    ag_speedup = ag_seq / ag_wall
    ag_eff = max(0.0, 1.0 - ag_wall / ag_seq) * 100

    log(rank, f"\n{'─' * 72}")
    log(rank, "OVERLAPPED: CE AllGather + GEMM")
    log(rank, f"{'─' * 72}")
    log(rank, f"  Wall time:       {ag_wall:.3f} ms")
    log(rank, f"  Compute leg:     {ag_comp_ms:.3f} ms")
    log(rank, f"  Comm leg:        {ag_comm_ms:.3f} ms")
    log(rank, f"  Sequential sum:  {ag_seq:.3f} ms")
    log(rank, f"  Speedup:         {ag_speedup:.2f}x")
    log(rank, f"  Overlap eff:     {ag_eff:.1f}%")

    log(rank, f"\n{'─' * 72}")

    # ── Profiler pass ──
    if args.profile_dir:
        os.makedirs(args.profile_dir, exist_ok=True)
        log(rank, f"\nRunning profiler pass → {args.profile_dir}/")
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=2, active=3, repeat=1),
            on_trace_ready=tensorboard_trace_handler(args.profile_dir),
            record_shapes=True,
            with_stack=True,
        ) as prof:
            for _ in range(6):  # wait(1) + warmup(2) + active(3)
                with torch.cuda.stream(compute_stream):
                    torch.mm(a, b)
                with torch.cuda.stream(comm_stream):
                    w = dist.all_gather_into_tensor(large, small, async_op=True)
                    w.wait()
                prof.step()
            compute_stream.synchronize()
            comm_stream.synchronize()
        if rank == 0:
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
        log(rank, f"Traces saved to {args.profile_dir}/")

    torch.cuda.synchronize(device)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
