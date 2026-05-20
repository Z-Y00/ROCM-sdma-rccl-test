#!/usr/bin/env python3
"""
Fast probe for the difference between the isolated CE/SDMA bench path and the
Torchtitan/FSDP path.

Modes:
  symm_ag
    Matches bench/bench_ag_gemm.py:
      symm_mem.empty -> symm_mem.rendezvous -> dist.all_gather_into_tensor
    Both input and output are explicit symmetric-memory tensors.

  regular_ag
    Same collective shape, but with regular torch.empty/full tensors. This is
    closer to FSDP's call-site: FSDP owns normal parameter/all-gather buffers and
    depends on the tensor-register allocator hook to make CE collectives legal.

  staged_symm_ag
    FSDP-like regular source/destination tensors, but the collective itself uses
    explicit symmetric-memory staging buffers:
      regular shard -> symm input -> all_gather_into_tensor -> symm output
        -> regular destination
    This prototypes replacing allocator-hook registration with explicit
    symm_mem.empty + rendezvous at the all-gather buffer boundary.

  fsdp_forward
    A tiny FSDP2 model using torch.distributed.fsdp.fully_shard. This triggers
    the same PyTorch stack as Torchtitan:
      _fully_shard/_fsdp_collectives.foreach_all_gather
        -> dist.all_gather_into_tensor
    The AG output buffer is allocated via the DefaultAllGather alloc mixin
    (torch.empty -> hipMalloc), so RCCL routes the collective through
    ncclDevKernel_Generic_2 -- *not* the SDMA path.

  fsdp_forward_symm
    Same tiny FSDP2 model, but with module.set_custom_all_gather(SymmMemAllGather(...))
    and module.set_custom_reduce_scatter(SymmMemReduceScatter(...)) wired in
    after fully_shard. SymmMemAllGather.allocate uses a symm_mem mempool,
    so the AG output is cuMem-backed and the collective lands on
    __amd_rocclr_batchMemOp.kd / hsa_amd_memory_async_batch_copy (real SDMA).

Run under CE-mode env to compare:
  NCCL_CTA_POLICY=2 NCCL_CUMEM_ENABLE=1 NCCL_LOCAL_REGISTER=2 \\
  TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true \\
  torchrun --nproc_per_node=8 debug/fsdp_like_ag_probe.py --mode all
"""

from __future__ import annotations

import argparse
import faulthandler
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch import nn

faulthandler.enable()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=("symm_ag", "regular_ag", "staged_symm_ag",
                 "fsdp_forward", "fsdp_forward_symm", "all"),
        default="all",
    )
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--numel", type=int, default=1 << 24)  # 32 MB bf16 per rank
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    return p.parse_args()


def dtype_from_arg(s: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]


def init_dist() -> tuple[int, int, int, torch.device, str]:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    opts = dist.ProcessGroupNCCL.Options()
    opts.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world,
        device_id=device,
        pg_options=opts,
        timeout=timedelta(minutes=10),
    )
    symm_mem.set_backend("NCCL")
    group_name = dist.group.WORLD.group_name
    if rank == 0:
        print(
            f"[init] world={world} torch={torch.__version__} "
            f"hip={getattr(torch.version, 'hip', None)}",
            flush=True,
        )
    return rank, local_rank, world, device, group_name


def symm_empty(numel: int, dtype: torch.dtype, device: torch.device, group: str) -> torch.Tensor:
    bytes_per = torch.empty((), dtype=dtype).element_size()
    n_f32 = (numel * bytes_per + 3) // 4
    raw = symm_mem.empty(n_f32, device=device, dtype=torch.float32)
    symm_mem.rendezvous(raw, group=group)
    return raw.view(torch.uint8)[: numel * bytes_per].view(dtype)


def time_block(name: str, fn, device: torch.device, rank: int, iters: int) -> None:
    dist.barrier()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if rank == 0:
        print(f"\n[{name}] begin", flush=True)
    start.record()
    for i in range(iters):
        if rank == 0:
            print(f"[{name}] iter {i}", flush=True)
        fn()
    end.record()
    torch.cuda.synchronize(device)
    ms = torch.tensor([start.elapsed_time(end)], dtype=torch.float64, device=device)
    dist.all_reduce(ms, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"[{name}] PASS  max_rank_ms={ms.item():.3f}", flush=True)


def run_symm_ag(args, dtype, rank, world, device, group_name) -> None:
    x = symm_empty(args.numel, dtype, device, group_name)
    y = symm_empty(args.numel * world, dtype, device, group_name)
    x.fill_(rank + 1)

    def step():
        work = dist.all_gather_into_tensor(y, x, async_op=True)
        work.wait()

    time_block("symm_ag", step, device, rank, args.iters)


def run_regular_ag(args, dtype, rank, world, device, _group_name) -> None:
    x = torch.full((args.numel,), rank + 1, dtype=dtype, device=device)
    y = torch.empty((args.numel * world,), dtype=dtype, device=device)

    def step():
        work = dist.all_gather_into_tensor(y, x, async_op=True)
        work.wait()

    time_block("regular_ag", step, device, rank, args.iters)


def run_staged_symm_ag(args, dtype, rank, world, device, group_name) -> None:
    regular_x = torch.full((args.numel,), rank + 1, dtype=dtype, device=device)
    regular_y = torch.empty((args.numel * world,), dtype=dtype, device=device)
    symm_x = symm_empty(args.numel, dtype, device, group_name)
    symm_y = symm_empty(args.numel * world, dtype, device, group_name)

    def step():
        symm_x.copy_(regular_x)
        work = dist.all_gather_into_tensor(symm_y, symm_x, async_op=True)
        work.wait()
        regular_y.copy_(symm_y)

    time_block("staged_symm_ag", step, device, rank, args.iters)


class TinyBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden * 4, bias=False),
            nn.SiLU(),
            nn.Linear(hidden * 4, hidden, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_fsdp_model(args, dtype, device):
    layers = [TinyBlock(args.hidden).to(device=device, dtype=dtype) for _ in range(args.layers)]
    model = nn.Sequential(*layers)
    from torch.distributed.fsdp import fully_shard
    for layer in model:
        fully_shard(layer)
    fully_shard(model)
    return model


def run_fsdp_forward(args, dtype, rank, _world, device, _group_name) -> None:
    model = _build_fsdp_model(args, dtype, device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    inp = torch.randn(args.batch, args.seq, args.hidden, device=device, dtype=dtype)

    def step():
        optim.zero_grad(set_to_none=True)
        out = model(inp)
        loss = out.float().square().mean()
        loss.backward()
        optim.step()

    time_block("fsdp_forward", step, device, rank, args.iters)


def run_fsdp_forward_symm(args, dtype, rank, _world, device, _group_name) -> None:
    """FSDP2 model, but switched to symm_mem-backed AG/RS buffers via the
    public set_custom_all_gather / set_custom_reduce_scatter APIs. This is
    the supported way (in torch 2.12+) to route FSDP collectives onto the
    bench's CE/SDMA path -- the AG output is allocated from a symm_mem
    mempool, so RCCL lands on __amd_rocclr_batchMemOp.kd /
    hsa_amd_memory_async_batch_copy."""
    model = _build_fsdp_model(args, dtype, device)

    from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
        SymmMemAllGather,
        SymmMemReduceScatter,
    )

    group = dist.group.WORLD
    # set_custom_all_gather targets a single param group; for our tiny
    # nn.Sequential the outer fully_shard creates one group, and each
    # inner fully_shard creates one more. We have to set per-module so
    # every AG flows through SymmMemAllGather.
    def _attach(mod):
        try:
            mod.set_custom_all_gather(SymmMemAllGather(group))
            mod.set_custom_reduce_scatter(SymmMemReduceScatter(group))
        except (AttributeError, ValueError) as e:
            # nested children that aren't fully_shard'd or don't have a single
            # param group fall through silently
            if rank == 0:
                print(f"[fsdp_forward_symm] skip {type(mod).__name__}: {e}", flush=True)

    _attach(model)
    for layer in model:
        _attach(layer)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    inp = torch.randn(args.batch, args.seq, args.hidden, device=device, dtype=dtype)

    def step():
        optim.zero_grad(set_to_none=True)
        out = model(inp)
        loss = out.float().square().mean()
        loss.backward()
        optim.step()

    time_block("fsdp_forward_symm", step, device, rank, args.iters)


def main() -> None:
    args = parse_args()
    dtype = dtype_from_arg(args.dtype)
    rank, _local_rank, world, device, group = init_dist()

    modes = (
        ("symm_ag", run_symm_ag),
        ("regular_ag", run_regular_ag),
        ("staged_symm_ag", run_staged_symm_ag),
        ("fsdp_forward", run_fsdp_forward),
        ("fsdp_forward_symm", run_fsdp_forward_symm),
    )
    for name, fn in modes:
        if args.mode in ("all", name):
            fn(args, dtype, rank, world, device, group)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
