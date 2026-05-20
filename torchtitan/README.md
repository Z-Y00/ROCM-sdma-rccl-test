# torchtitan / Llama-3 70B on 8x MI300X

Single-node Llama-3 70B training bring-up on top of the PyTorch 2.12 image
built by `../docker/build_pytorch212.sh` (also available pre-built at
`lorrisync/therock-main:gfx94X_pytorch2.12_rocm7.14_96bfee1`).

## What's here

| | |
|---|---|
| `run_train.sh` | builds the LD_PRELOAD interposer in-container, clones torchtitan, downloads the tokenizer, launches `torchrun --nproc_per_node=8 -m torchtitan.train` |
| `configs/llama3_70b_mi300x_8gpu.toml` | starting toml: FSDP=8, TP=1, seq=8192, batch=1, full activation checkpointing |

Outputs are dropped into a named container at `/workspace/outputs` and
extracted with `docker cp` to `./outputs_run/` on the host (snap-docker
can't bind-mount the NFS-backed home).

## Prerequisites

- 8x MI300X visible (`/dev/kfd`, `/dev/dri`)
- The PyTorch 2.12 image pulled locally: `docker pull lorrisync/therock-main:gfx94X_pytorch2.12_rocm7.14_96bfee1`
- Network access from the container to HuggingFace (no auth required by default
  — we pull the tokenizer from the public mirror
  **unsloth/Meta-Llama-3.1-70B-Instruct**).

## Quick start

```bash
# Default 100-step run on 8 GPUs, public tokenizer, no HF token needed:
./run_train.sh

# Short bring-up:
STEPS=20 SEQ_LEN=4096 ./run_train.sh

# Smoke without any HF download at all (uses torchtitan's debug model):
TOKENIZER_SKIP=1 STEPS=5 ./run_train.sh

# Use the gated official tokenizer instead of the unsloth mirror:
TOKENIZER_REPO=meta-llama/Meta-Llama-3-70B HF_TOKEN=hf_... ./run_train.sh
```

## Knobs (host env)

| var | default | meaning |
|---|---|---|
| `ROCM_BUG_TEST_IMAGE` | `lorrisync/therock-main:...96bfee1` | container image |
| `NPROC` | 8 | GPUs per node |
| `STEPS` | 100 | training steps |
| `SEQ_LEN` | 8192 | sequence length |
| `BATCH_SIZE` | 1 | per-GPU micro-batch |
| `FSDP` | `${NPROC}` | `training.data_parallel_shard_degree` |
| `TP` | 1 | `training.tensor_parallel_degree` |
| `CONFIG` | `configs/llama3_70b_mi300x_8gpu.toml` | base toml |
| `TORCHTITAN_REF` | `main` | git ref to check out (pin once green) |
| `TOKENIZER_REPO` | `unsloth/Meta-Llama-3.1-70B-Instruct` | HF repo to pull tokenizer from (public; no auth) |
| `HF_TOKEN` | _empty_ | only needed if you switch `TOKENIZER_REPO` to a gated repo |
| `TOKENIZER_SKIP` | 0 | set to 1 to use `flavor=debugmodel` and skip HF download |
| `OUTPUTS_HOST` | `./outputs_run` | where `docker cp` extracts outputs to |

## What the script bakes in

- `LD_PRELOAD=libhip_attr_drain.so` (the interposer from `../debug/`), so
  the RCCL `cuMem` path doesn't break the first kernel launch on a
  patched-but-not-yet-released RCCL.
- CE-collectives env: `NCCL_CTA_POLICY=2`, `NCCL_CUMEM_ENABLE=1`,
  `NCCL_LOCAL_REGISTER=2`, `TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true`
  -- so FSDP all-gather / reduce-scatter land on the SDMA copy engines,
  same path as the `bench/` overlap experiments. See
  [`../README.md` section (1b)](../README.md) for what each of the four
  knobs actually does, source-truth references, and ablation matrix.
- `HSA_NO_SCRATCH_RECLAIM=1`, `PYTORCH_ROCM_ARCH=gfx942`, `OMP_NUM_THREADS=8`.

### Heads-up: `hipMemRetainAllocationHandle` SIGSEGV with the default CE env

On ROCm 7.14 (`libamdhip64.so.7` build `39213316d2`) the FSDP first AllGather
**crashes inside the HIP runtime** when the four CE knobs above are all on,
because RCCL's IPC auto-registration calls `hipMemRetainAllocationHandle`
on PyTorch's `hipMalloc`-backed caching-allocator slabs and the runtime
NULL-derefs instead of returning `hipErrorInvalidValue`. Stack:

```
hipMemRetainAllocationHandle  [libamdhip64.so.7]
  ipcRegisterBuffer / ncclIpcLocalRegisterBuffer / ncclRegisterCollBuffers
  → ncclAllGather_impl
  → ProcessGroupNCCL::_allgather_base
  → torch.distributed.fsdp._fully_shard._fsdp_collectives.foreach_all_gather
```

Until the HIP runtime fix lands, set **`NCCL_LOCAL_REGISTER=0`** in the
torchtitan env. This is what `CE_MODE=1` bakes in here. FSDP trains
end-to-end (Llama-3 70B, 5 steps, ~23.6 % MFU at step 5).

**Important caveat — verified via rocprof:** this env workaround keeps
the crash away and keeps RCCL on `P2P/CUMEM` channels for peer address
translation, but it does **not** by itself get FSDP onto the bench's
SDMA dispatch path. With `cudaMalloc`-backed AG buffers, the bytes move
inside `ncclDevKernel_Generic_2` on the CUs;
`hsa_amd_memory_async_batch_copy` is called **zero** times (vs 22 in
the bench).

**To actually put FSDP on the SDMA path** call
`module.set_custom_all_gather(SymmMemAllGather(group))` (and the RS
counterpart) after `fully_shard()`. `SymmMemAllGather` is built into
PyTorch 2.12 (`torch.distributed.fsdp._fully_shard._fsdp_collectives`)
and allocates the AG output buffer from a `symm_mem` mempool, which
lands the collective on `__amd_rocclr_batchMemOp.kd` /
`hsa_amd_memory_async_batch_copy` — i.e. real SDMA. Reference
implementation: `../debug/fsdp_like_ag_probe.py --mode fsdp_forward_symm`
(verified 192 `batch_copy` calls across 8 ranks vs 0 in the unmodified
`fsdp_forward` mode). Wiring it into this `run_train.sh` /
torchtitan is the next bring-up step — see
[`../README.md` section (1b)](../README.md) / "Path to real SDMA in FSDP"
and
[`../debug/ROOT_CAUSE_hipMemRetainAllocationHandle.md`](../debug/ROOT_CAUSE_hipMemRetainAllocationHandle.md)
for the full evidence chain.

## When the node is free, sanity-check sequence

1. Smoke without any HF call, debug model, 5 steps, single GPU:
   ```bash
   NPROC=1 TOKENIZER_SKIP=1 STEPS=5 ./run_train.sh
   ```
2. Smoke with the public tokenizer, 70B, 8 GPUs, 5 steps:
   ```bash
   STEPS=5 ./run_train.sh
   ```
3. Profile a few steps under `torch.profiler` (todo: extend the runner
   to flip `[profiling].enable_profiling = true` and `docker cp` the
   trace artifacts out).

## Tunables for memory

If 70B + seq=8192 + activation-checkpoint=full doesn't fit:

- drop `seq_len` to 4096 or 2048
- set `TP=2 FSDP=4` (cuts per-rank weight footprint in half)
- in the toml, set `[float8]   enable_fsdp_float8_all_gather = true`
  (requires the rocm/pytorch float8 path, may need tweaking on MI300X)
