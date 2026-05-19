#!/usr/bin/env python3
"""
ROCm bug reproducer

the first HIP kernel launch after `symm_mem.empty(...)` 
(which internally calls rccl) fails. But the 2nd launch passes. 

Verified trigger:
  * NCCL_CUMEM_ENABLE=1  (forces RCCL down the cuMem path)

Verified on:
  * AMD Instinct MI300X, 8x visible, ROCm 7.13.26176
  * Linux kernel 6.5.0-45-generic (Ubuntu 22.04 HWE)
  * Stock TheRock image (PyTorch 2.11.0+rocm7.13) AND custom 2.12 build

Run:
  NCCL_CUMEM_ENABLE=1 torchrun --nproc_per_node=1 pytorch_bug_repro.py
  NCCL_CUMEM_ENABLE=0 torchrun --nproc_per_node=1 pytorch_bug_repro.py   # control
"""
import os, sys, platform
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Init NCCL PG. (Required: triggers RCCL's cuMem PFN loading.)
    opts = dist.ProcessGroupNCCL.Options()
    opts.config.cta_policy = dist.ProcessGroupNCCL.NCCL_CTA_POLICY_ZERO
    dist.init_process_group(backend="nccl", pg_options=opts, device_id=device)
    symm_mem.set_backend("NCCL")

    print("==== ROCm symm_mem + first-kernel-launch bug reproducer ====")
    print(f"  device              : cuda:{local_rank}  "
          f"({torch.cuda.get_device_name(local_rank)})")
    print(f"  torch.version       : {torch.__version__}")
    print(f"  torch.version.hip   : {getattr(torch.version, 'hip', None)}")
    print(f"  Linux kernel        : {platform.release()}")
    print(f"  NCCL_CUMEM_ENABLE   : {os.environ.get('NCCL_CUMEM_ENABLE', '<unset>')}"
          f"   (verified trigger: '=1')")
    print()

    numel = 1 << 20  # 4 MB float32

    # --- pre-warm: regular tensor, regular kernel, drain runtime state ---
    ref0 = torch.empty(numel, device=device, dtype=torch.float32).fill_(0.5)
    torch.cuda.synchronize()
    print(f"phase 1  pre-warm regular kernel                  PASS")

    # --- allocate via NCCL symm_mem (calls ncclMemAlloc -> cuMem path) ---
    t = symm_mem.empty(numel, device=device, dtype=torch.float32)
    print(f"phase 2  symm_mem.empty(4 MB)                     PASS  "
          f"(ptr=0x{t.data_ptr():016x})")

    # --- phase 3: FIRST kernel after the cuMem allocation ---------------
    ref1 = torch.empty(numel, device=device, dtype=torch.float32)
    try:
        ref1.fill_(3.0)
        torch.cuda.synchronize()
        print("phase 3  ref1.fill_(3.0) [1st kernel after alloc]  PASS")
        first_failed = False
    except Exception as e:
        try: torch.cuda.synchronize()
        except Exception: pass
        msg = str(e).splitlines()[0]
        print(f"phase 3  ref1.fill_(3.0) [1st kernel after alloc]  FAIL  "
              f"({msg[:80]})")
        first_failed = True

    # --- phase 4: SECOND kernel — should always pass per the established
    #              "first-shot" pattern -----------------------------------
    ref2 = torch.empty(numel, device=device, dtype=torch.float32)
    try:
        ref2.fill_(4.0)
        torch.cuda.synchronize()
        print("phase 4  ref2.fill_(4.0) [2nd kernel]             PASS")
        second_passed = True
    except Exception as e:
        msg = str(e).splitlines()[0]
        print(f"phase 4  ref2.fill_(4.0) [2nd kernel]             FAIL  "
              f"({msg[:80]})")
        second_passed = False

    dist.destroy_process_group()
    print()
    if first_failed and second_passed:
        print("BUG REPRODUCED.")
    elif not first_failed and second_passed:
        print("no bug observed.")
    else:
        print(f"UNEXPECTED pattern: phase3={'FAIL' if first_failed else 'PASS'}, "
              f"phase4={'PASS' if second_passed else 'FAIL'}")
    sys.exit(0)  # always 0 so torchrun doesn't append child-failure noise


if __name__ == "__main__":
    main()
