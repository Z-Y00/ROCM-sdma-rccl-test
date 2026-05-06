"""
Copy Engine (CE) Collectives test.

Based on: https://docs.pytorch.org/docs/2.11/symmetric_memory.html#copy-engine-collectives

Launch with:
    torchrun --nproc_per_node=<NUM_GPUS> ce_collectives_test.py
"""

import os
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

# Initialize process group with zero-CTA policy for CE collectives
opts = dist.ProcessGroupNCCL.Options()
opts.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
device = torch.device("cuda", local_rank)
dist.init_process_group(backend="nccl", pg_options=opts, device_id=device)
if rank == 0:
    print("init_process_group OK")

# Set up symmetric memory with NCCL backend
symm_mem.set_backend("NCCL")
group_name = dist.group.WORLD.group_name
if rank == 0:
    print("set_backend OK")

# Allocate tensors using symmetric memory
numel = 1024 * 1024
inp = symm_mem.empty(numel, device=device)
out = symm_mem.empty(numel * world_size, device=device)
if rank == 0:
    print(f"symm_mem.empty OK  inp={inp.shape} out={out.shape}")

# Register tensors for symmetric memory operations
symm_mem.rendezvous(inp, group=group_name)
symm_mem.rendezvous(out, group=group_name)
if rank == 0:
    print("rendezvous OK")

# Perform collective operation using copy engines
# This now runs on DMA engines instead of SMs
work = dist.all_gather_into_tensor(out, inp, async_op=True)
work.wait()
torch.cuda.synchronize(device)

if rank == 0:
    print(f"all_gather_into_tensor OK  world_size={world_size}  numel={numel}")

dist.destroy_process_group()
