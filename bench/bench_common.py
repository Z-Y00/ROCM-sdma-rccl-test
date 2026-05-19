"""
Shared helpers for the AG+GEMM and AR+GEMM overlap benchmarks.

  * init_distributed -- bring up an NCCL PG with CE-friendly options
                        and rendezvous a symmetric-memory group.
  * make_streams     -- compute stream + comm stream.
  * Bench            -- cuda-event-based warmup+timed loop.
  * gather_max_ms    -- aggregate per-rank ms across all ranks.
  * print_table      -- markdown-style rank-0 results table.

Designed to be torchrun-launched (env-vars set by torchrun). All comm
defaults to the Copy-Engine path: NCCL_CTA_POLICY_ZERO + symmetric
memory + async_op=True. The runner script is responsible for setting
LD_PRELOAD to libhip_attr_drain.so so we don't hit the FABRIC_SUPPORTED
TLS-leak bug on the first cuMem-path allocation.
"""
from __future__ import annotations
import faulthandler
import os
import statistics
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Iterable, Sequence

# Print a Python traceback on SIGSEGV / SIGABRT etc. Without this, torchrun
# reports only "exitcode -11" with no signal context. Cheap; no runtime cost
# unless we actually crash.
faulthandler.enable()

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.profiler import ProfilerActivity, profile


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------
@dataclass
class DistCtx:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    group_name: str


def init_distributed(timeout_minutes: int = 10) -> DistCtx:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # CE collectives + symmetric memory: zero-CTA + cuMem-backed alloc.
    opts = dist.ProcessGroupNCCL.Options()
    opts.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO

    dist.init_process_group(
        backend="nccl",
        pg_options=opts,
        device_id=device,
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=timeout_minutes),
    )

    symm_mem.set_backend("NCCL")
    group_name = dist.group.WORLD.group_name
    # enable_symm_mem_for_group emits a FutureWarning saying it's no longer
    # required; the call is a no-op on recent PyTorch. We skip it entirely
    # to keep stdout clean.

    if rank == 0:
        print(
            f"[init] world_size={world_size}  device={device}  "
            f"torch={torch.__version__}  hip={getattr(torch.version, 'hip', None)}",
            flush=True,
        )

    return DistCtx(rank, local_rank, world_size, device, group_name)


def make_streams(device: torch.device) -> tuple[torch.cuda.Stream, torch.cuda.Stream]:
    """compute stream + comm stream, both on `device`."""
    compute_s = torch.cuda.Stream(device=device, priority=-1)  # higher priority
    comm_s = torch.cuda.Stream(device=device, priority=0)
    return compute_s, comm_s


# ---------------------------------------------------------------------------
# Symmetric-memory allocation helper
# ---------------------------------------------------------------------------
def symm_empty(numel: int, dtype: torch.dtype, ctx: DistCtx) -> torch.Tensor:
    """Allocate a symm_mem tensor of `numel` elements at `dtype`, rendezvous'd."""
    # symm_mem.empty returns float32 today; reinterpret via .view + .to is wrong
    # for symm-mem because it'd realloc. Allocate enough float32 elements and
    # then view as the target dtype if dtype is also 4 bytes; for bf16 we
    # over-allocate by 2x and view.
    bytes_per = torch.empty((), dtype=dtype).element_size()
    f32_per = 4
    if bytes_per > f32_per:
        # caller asked for fp64 etc; round up
        n_f32 = numel * bytes_per // f32_per + (1 if (numel * bytes_per) % f32_per else 0)
    else:
        n_f32 = (numel * bytes_per + f32_per - 1) // f32_per
    raw = symm_mem.empty(n_f32, device=ctx.device, dtype=torch.float32)
    # rendezvous BEFORE the view (rendezvous keys off the underlying storage)
    symm_mem.rendezvous(raw, group=ctx.group_name)
    # reinterpret as the requested dtype
    t = raw.view(torch.uint8)[: numel * bytes_per].view(dtype)
    return t


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------
class Bench:
    """cuda-event-based warmup + timed loop."""

    def __init__(self, device: torch.device, warmup: int = 5, timed: int = 20):
        self.device = device
        self.warmup = warmup
        self.timed = timed

    def run(self, name: str, iter_fn: Callable[[], None]) -> float:
        """Run iter_fn `warmup` then `timed` times; return mean ms per iter."""
        # warmup
        for _ in range(self.warmup):
            iter_fn()
        torch.cuda.synchronize(self.device)

        # timed
        per_iter_ms: list[float] = []
        for _ in range(self.timed):
            ev_s = torch.cuda.Event(enable_timing=True)
            ev_e = torch.cuda.Event(enable_timing=True)
            ev_s.record()
            iter_fn()
            ev_e.record()
            torch.cuda.synchronize(self.device)
            per_iter_ms.append(ev_s.elapsed_time(ev_e))

        mean = statistics.fmean(per_iter_ms)
        return mean


# ---------------------------------------------------------------------------
# Cross-rank aggregation + table printing
# ---------------------------------------------------------------------------
def gather_max_ms(local_ms: float, ctx: DistCtx) -> float:
    """Aggregate per-rank ms to a single 'wallclock' = max-across-ranks."""
    t = torch.tensor([local_ms], device=ctx.device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def print_table(rows: Sequence[dict], headers: Sequence[str], rank: int) -> None:
    if rank != 0:
        return
    widths = {h: max(len(h), *(len(f"{r.get(h, ''):}") for r in rows)) for h in headers}
    sep = "| " + " | ".join("-" * widths[h] for h in headers) + " |"
    head = "| " + " | ".join(h.ljust(widths[h]) for h in headers) + " |"
    print()
    print(head)
    print(sep)
    for r in rows:
        cells = []
        for h in headers:
            v = r.get(h, "")
            cells.append(str(v).ljust(widths[h]))
        print("| " + " | ".join(cells) + " |")
    print(flush=True)


# ---------------------------------------------------------------------------
# Profiling helper
# ---------------------------------------------------------------------------
def profile_overlap(
    iter_fn: Callable[[], None],
    ctx: DistCtx,
    tag: str,
    profile_dir: str,
    n_iters: int = 5,
    warmup: int = 2,
) -> None:
    """Profile `n_iters` of iter_fn after warmup; write chrome trace per rank
    to {profile_dir}/{tag}_rank{rank}.json and print a kernel-summary table
    on rank 0. No-op if profile_dir is empty."""
    if not profile_dir:
        return
    os.makedirs(profile_dir, exist_ok=True)

    for _ in range(warmup):
        iter_fn()
    torch.cuda.synchronize(ctx.device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(n_iters):
            iter_fn()
        torch.cuda.synchronize(ctx.device)

    out = os.path.join(profile_dir, f"{tag}_rank{ctx.rank}.json")
    prof.export_chrome_trace(out)

    # Rank 0 prints a kernel-summary table so the user can read it in the log
    # without opening the chrome trace. Other ranks just write their trace.
    if ctx.rank == 0:
        try:
            tbl = prof.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=20,
            )
        except Exception:
            tbl = prof.key_averages().table(
                sort_by="self_cpu_time_total",
                row_limit=20,
            )
        print(f"\n[profile:{tag}] top events (rank 0):", flush=True)
        print(tbl, flush=True)
        print(f"[profile:{tag}] chrome trace -> {out}", flush=True)


# ---------------------------------------------------------------------------
# Shape config -- llama-70B TP-style FFN-up + FFN-down (one node 8x MI300X)
# ---------------------------------------------------------------------------
@dataclass
class Shape:
    name: str
    M: int
    N: int
    K: int


LLAMA70B_SHAPES: list[Shape] = [
    Shape("llama70b_ffn_up",   M=8192, N=28672, K=8192),
    Shape("llama70b_ffn_down", M=8192, N=8192,  K=28672),
]
