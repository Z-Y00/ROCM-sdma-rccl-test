# sdma_rccl_pytorch

Top-level workspace for two pieces of work on AMD MI300X. Layout:

```
debug/        interposer fix + bug-investigation reproducers
bench/        AG+GEMM and AR+GEMM overlap benches
docker/       image-build infrastructure (PyTorch 2.12 on top of TheRock 2.11 base)
torchtitan/   Llama-3 70B torchtitan training on 8x MI300X
```

1. **Interposer fix** (`debug/`) for the "first kernel after `ncclMemAlloc`
   fails with `hipErrorInvalidValue`" bug on RCCL's `NCCL_CUMEM_ENABLE=1` path.
2. **SDMA-based comm/compute overlap benchmarks** (`bench/`) for
   AllGather+GEMM and AllReduce+GEMM. (Sibling sub-project
   `rocm_sdma_comm_compute_overlap/` is its own git repo, not tracked here.)
3. **PyTorch-2.12 build recipe** (`docker/`) for stacking PyTorch 2.12 on top
   of the TheRock ROCm 7.14 / 2.11 base image; gives a notably better
   torch.profiler trace where SDMA copy-engine kernels show up directly.
4. **Llama-3 70B torchtitan training** (`torchtitan/`) on top of (3), with
   the LD_PRELOAD interposer and CE-collective env wired in. Script-only
   for now (node busy); see `torchtitan/README.md` for prereqs and knobs.

## Test images

The interposer + the bench harness are validated against TheRock PyTorch
2.11 images. Set `ROCM_BUG_TEST_IMAGE` to override.

```
registry-sc-harbor.amd.com/framework/therock-main:1384_gfx94X_7.14.0a20260518_centosstream9_py3.12_pytorch_release-2.11_96bfee1
registry-sc-harbor.amd.com/framework/therock-main:1384_gfx94X_7.14.0a20260518_ubuntu24.04_py3.14_pytorch_release-2.11_96bfee1
```

## (1) Interposer fix

All artifacts live under `debug/`.

`debug/hip_attr_drain_preload.c` is a tiny `LD_PRELOAD` shim that wraps
`hipDeviceGetAttribute` / `cuDeviceGetAttribute` and drains the per-thread
`last_error` slot when the call fails. Built and validated end-to-end by
`debug/run_with_interposer.sh`:

| stage | result |
|---|---|
| HIP probe baseline | leaks `hipErrorInvalidValue` into TLS |
| HIP probe + `LD_PRELOAD` | TLS clean |
| PyTorch repro baseline (`NCCL_CUMEM_ENABLE=1`) | first kernel after `symm_mem.empty()` FAILS |
| PyTorch repro + `LD_PRELOAD` | no bug observed |

Files:

| | |
|---|---|
| `debug/hip_attr_drain_preload.c` | the `.so` source (gcc + libdl, no HIP deps) |
| `debug/hip_attr_probe.c` | minimal pure-HIP demonstrator of the TLS leak |
| `debug/pytorch_bug_repro.py` | minimal pure-PyTorch end-to-end reproducer |
| `debug/run_with_interposer.sh` | builds the `.so` in-container and runs both as A/B/C/D |
| `debug/run_hip_attr_probe.sh` | standalone "does this image have the leak?" check |
| `debug/run_bug_repros.sh` | standalone "does this image trigger the PyTorch bug?" check |

User-facing one-liner workaround (no RCCL rebuild required):
```bash
gcc -O2 -fPIC -shared debug/hip_attr_drain_preload.c -o libhip_attr_drain.so -ldl
LD_PRELOAD=$PWD/libhip_attr_drain.so NCCL_CUMEM_ENABLE=1 torchrun ... your_script.py
```

### Root cause

The failure is a three-layer interaction between RCCL, the HIP runtime,
and PyTorch's launch-error checker. Each layer is doing something
reasonable in isolation; together they manufacture a phantom "first
kernel after `ncclMemAlloc` fails with `invalid argument`."

**Layer 1 — RCCL probes an attribute that doesn't exist on ROCm.**

In RCCL's cuMem allocator, when `NCCL_CUMEM_ENABLE=1` (or its default
value on builds that auto-enable it), every `ncclMemAlloc` runs through
this preflight:

```c
// rocm-systems/projects/rccl/src/allocator.cc @ commit 6d9918b, line 40
(void) CUPFN(cuDeviceGetAttribute(
    &flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, currentDev));
```
([source on GitHub](https://github.com/ROCm/rocm-systems/blob/6d9918ba4da49f1650a08f045a043bb811960b1d/projects/rccl/src/allocator.cc#L40))

Two ROCm-specific details make this innocuous-looking line dangerous:

- `CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED` (enum value `128`)
  is an NVLink-fabric / MNNVL feature added in CUDA 12.4. ROCm's HIP
  runtime does not recognize that enum value, so the call returns
  `hipErrorInvalidValue` (1).
- `CUPFN(x)` on the AMD platform is defined in
  `rocm-systems/projects/rccl/src/include/rocmwrap.h` as a literal
  expansion — `#define CUPFN(symbol) symbol` — so the call goes through
  the normal global symbol `cuDeviceGetAttribute` (which on ROCm is just
  an alias for `hipDeviceGetAttribute` in `libamdhip64.so`). On NVIDIA
  the same macro expands to `pfn_##symbol`, a function pointer loaded
  via `cudaGetDriverEntryPoint`.
- RCCL discards the return value (`(void) ...`). The probe is treated as
  best-effort: "if fabric is supported, set a flag; otherwise carry on."

So far so good — except for what the HIP runtime does on its way out of
the failing call.

**Layer 2 — the HIP runtime leaks the error into per-thread TLS.**

Every HIP host-API entry point updates the per-thread "last error" slot
(`last_error_`) on the way out. When `hipDeviceGetAttribute` is called
with an unknown enum it takes the failure branch, returns
`hipErrorInvalidValue`, *and* writes that same `hipErrorInvalidValue`
into TLS — without anyone subsequently draining it. The pure-HIP probe
(`hip_attr_probe.c`) demonstrates this in isolation:

```
device: AMD Instinct MI300X
baseline peek = 0 (no error)

calling: (void) hipDeviceGetAttribute(&dummy, 128, 0)
  return value    = 1 (invalid argument)
  leaked into TLS = 1 (invalid argument)
```

Once the slot is polluted, every subsequent HIP entry on that thread
that *reads* the slot will see the stale error and attribute it to its
own operation. Critically, this includes the post-launch error check.

**Layer 3 — PyTorch reads the slot after the next kernel launch.**

PyTorch wraps every kernel launch with `C10_CUDA_KERNEL_LAUNCH_CHECK`,
which calls `cudaGetLastError()` and raises a Python exception if the
result is non-success. On ROCm that resolves to `hipGetLastError()`,
which reads-and-clears the same TLS slot RCCL just polluted.

So the timeline on the user's main thread is:

```
phase 1  pre-warm kernel                      PASS   (TLS clean)
phase 2  symm_mem.empty(4 MB)
           -> ncclMemAlloc
              -> cuDeviceGetAttribute(.., 128, ..) -> returns 1, leaks 1
           (TLS now contains hipErrorInvalidValue)
phase 3  ref1.fill_(3.0)                       <-- kernel launch fine,
           -> C10_CUDA_KERNEL_LAUNCH_CHECK         but post-launch check
              -> hipGetLastError() -> 1            reads the stale 1
                 raises "CUDA error: invalid argument"
phase 4  ref2.fill_(4.0)                      PASS   (TLS was cleared by
                                                     phase-3 hipGetLastError)
```

That is *exactly* the symptom in `pytorch_bug_repro.py`: first kernel
after `symm_mem.empty()` FAILS, second kernel PASSES, and nothing the
user wrote was wrong. The same shape works fine without
`NCCL_CUMEM_ENABLE` because `ncclMemAlloc` then doesn't take the cuMem
preflight path.

**Layer 4 — the fix.**

Two equivalent fixes; both drain the TLS so it can't propagate:

- **RCCL-side** (durable): in the failure branch of the attribute query,
  call `cudaGetLastError()` once to clear the slot. This is what the
  upstream patch on the `rccl-fabric-error-drain` branch does and it
  fits in three lines around the existing call.
- **User-side** (this repo): the LD_PRELOAD interposer wraps both
  `hipDeviceGetAttribute` and `cuDeviceGetAttribute`. On any non-success
  return it calls `hipGetLastError()` to drain the slot, then returns
  the original return value unchanged so caller behavior is identical.
  Because `CUPFN(x) == x` on AMD, the interposer catches RCCL's call
  without any RCCL rebuild.

**Why the bug is specifically about the attribute query, not query failures in general.**

Strictly speaking the HIP-runtime behavior of "set TLS on failure, even
for query APIs that have an out-parameter return path" is the deeper
defect — a tighter fix would be either (a) classify query APIs so they
don't leak into TLS, or (b) make ROCm recognize the unknown enum and
fail cleanly without touching `last_error_`. But RCCL's discard of the
return code is the *amplifier*: had RCCL checked the result and called
`cudaGetLastError()` (as is conventional for "best-effort" probes
elsewhere in the same file), the leak would never propagate. The
upstream RCCL patch fixes the amplifier; the HIP runtime change would
fix the source. The interposer fixes the amplifier at the symbol level
so any other caller that makes the same mistake is also covered.

## (1b) CE/SDMA env knobs + `hipMemRetainAllocationHandle` SIGSEGV

The CE/SDMA collective path is controlled by four knobs that the bench and
the torchtitan runner both set. Each does a distinct thing. Mis-combining
them tips RCCL into a HIP-runtime bug that crashes
`dist.all_gather_into_tensor` (and every FSDP step, since FSDP's first op
is an AllGather).

### Knob reference

| env var | what it controls | default | required for | source-truth |
|---|---|---|---|---|
| `NCCL_CTA_POLICY=2` | Launch collectives with **zero CTAs** (NCCL_CTA_POLICY_ZERO). With zero CTAs the data movement falls off the SMs and lands on the **copy engines (CE/SDMA)**. | unset (=full CTA) | CE/SDMA AG/RS; bench overlap | RCCL `init.cc` / `ProcessGroupNCCL::Options.config.cta_policy`. **Note:** PyTorch passes `pg_options.config.cta_policy = NCCL_CTA_POLICY_ZERO` directly to `init_process_group` in our probes and in Torchtitan, so the env var is overridden at PG-construction time. |
| `NCCL_CUMEM_ENABLE=1` | RCCL's **internal staging buffers** are allocated via the cuMem API (`hipMemCreate` + `hipMemMap`) instead of `hipMalloc`. Required so the staging buffers can be IPC-exported as cuMem handles, which is what the P2P/CUMEM transport needs to do CE-direct GPU↔GPU transfers. | unset (=0) | the P2P/CUMEM transport that CE/SDMA rides on; also `ncclMemAlloc` / `symm_mem.empty` | RCCL `allocator.cc`; bench `bench/run_bench.sh` `ce_env()` |
| `NCCL_LOCAL_REGISTER=2` | RCCL **auto-registers user send/recv buffers for direct IPC** at each collective enqueue. Effect: the CE collective DMAs the user buffer **directly** into the remote peer's user buffer, skipping one CE memcpy on each side that otherwise stages through RCCL's internal cuMem buffer. **Pure perf optimization; not required for CE/SDMA to work.** | `1` (enabled) | none — opt-in fast path | RCCL `coll_reg.cc` (top-level `ncclParamLocalRegister` gate); RCCL `transport/p2p.cc` `ipcRegisterBuffer` (the function that crashes — see bug below) |
| `TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true` | PyTorch caching allocator calls `ncclCommRegister` on every new slab, populating RCCL's `ncclReg` table so the LOCAL_REGISTER fast-path can find user buffers. Without it, RCCL's `ncclRegFind` returns NULL for FSDP's `torch.empty`-backed buffers and the registration path is a no-op. | unset (=false) | the LOCAL_REGISTER fast-path | PyTorch `c10d/ProcessGroupNCCL.cpp` |

### What goes wrong: `hipMemRetainAllocationHandle` SIGSEGV

**`libamdhip64.so.7` (ROCm 7.14, build `39213316d2`) segfaults inside
`hipMemRetainAllocationHandle` when called on a VA that wasn't created via
`hipMemCreate`/`cuMemCreate`** (i.e. anything from `hipMalloc` or PyTorch's
caching allocator). The function is supposed to return `hipErrorInvalidValue`
on such VAs; RCCL explicitly handles that return code by falling back to
legacy `cudaIpcGetMemHandle` (`rocm-systems/projects/rccl/src/transport/p2p.cc:890`):

```c
// Get the mem handle for that buffer. It may have been allocated through
// cudaMalloc in which case we'll get the CUDA legacy mem handle, or through cuMem*.
if (ncclCuMemEnable()) {
  CUmemGenericAllocationHandle handle;
  if (CUPFN(cuMemRetainAllocationHandle(&handle, baseAddr)) != CUDA_SUCCESS) {
    // if cuMem* export fails, retry legacy export
    if (comm->directMode || !ncclParamLegacyCudaRegister()) goto fail;
    CUDACHECKGOTO(cudaIpcGetMemHandle(&ipcInfo.ipcDesc.devIpc, baseAddr), ret, fail);
    ...
```

The fallback never runs because the call crashes instead of returning. The
faulting instruction in `libamdhip64.so.7+0x466ce3` is a second-level NULL
deref inside the runtime's per-allocation tracker:

```
call <ROCm allocation-tracker lookup>    ; rax = tracker entry (non-null for hipMalloc'd VAs)
test %rax,%rax / je <ret_invalid>        ; passes
mov  0x100(%rax),%rax                    ; rax = tracker->cuMemSlot (non-null on this build)
mov  0xf8(%rax),%r15                     ; r15 = cuMemSlot->handle  <-- NULL deref, SIGSEGV
```

Independent pure-HIP reproducer (no RCCL, no PyTorch, no distributed): see
`debug/hip_retain_handle_probe.c`. Output:

```
mode=null      hipMemRetainAllocationHandle(NULL)        -> 1 (invalid argument)   ✓
mode=cumem     hipMemCreate + hipMemMap + retain         -> 0 (success), handle=...  ✓
mode=hipmalloc hipMalloc + hipMemset + retain            -> SIGSEGV (core dumped)   ✗
```

### Minimal trigger combo and ablation matrix

In RCCL terms the crash needs all three of:

1. `NCCL_LOCAL_REGISTER != 0` — opens the gate at the top of `ncclRegisterCollBuffers`.
2. `TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true` — so `ncclRegFind` actually finds the user buffer.
3. `NCCL_CUMEM_ENABLE=1` — so `ipcRegisterBuffer` takes the cuMem branch.

…plus a user buffer that's `hipMalloc`'d, not cuMem-backed. The bench
(`bench_ag_gemm.py`) doesn't crash because its buffers come from
`symm_mem.empty → ncclMemAlloc → cuMemCreate`. FSDP / Torchtitan / our
`regular_ag` probe crash because their buffers come from the PyTorch
caching allocator (plain `hipMalloc`).

Verified by ablation (`debug/run_ce_ablation.sh`, `regular_ag` mode,
8 ranks on MI300X, kernel 6.5.0-45):

| preset | CTA_POLICY | CUMEM_ENABLE | LOCAL_REGISTER | ALLOCATOR_HOOK | result |
|---|---|---|---|---|---|
| `baseline_default` | – | – | 0 | false | PASS |
| `no_reg_at_all` | 2 | 1 | 0 | false | PASS |
| `no_local_register` | 2 | 1 | **0** | true | PASS |
| `no_allocator_hook` | 2 | 1 | 2 | **false** | PASS |
| `no_cumem` | 2 | **0** | 2 | true | PASS |
| `only_local_register` | – | – | 2 | false | PASS |
| `only_allocator_hook` | – | – | 0 | true | PASS |
| `cumem_plus_hook` | – | 1 | 0 | true | PASS |
| `ce_full` | 2 | 1 | 2 | true | **CRASH** |
| `no_cta_policy` | – | 1 | 2 | true | **CRASH** |

`NCCL_CTA_POLICY` is irrelevant to the crash (PyTorch passes
`cta_policy=NCCL_CTA_POLICY_ZERO` via PG opts in any case).

### Workaround

Set **`NCCL_LOCAL_REGISTER=0`** in the FSDP / Torchtitan environment, keeping
everything else CE-mode. Verified that this avoids the
`ipcRegisterBuffer → hipMemRetainAllocationHandle` call path entirely
(the top-of-function gate in `ncclRegisterCollBuffers` closes).

FSDP trains end-to-end (see `torchtitan/outputs_ce_localreg0/`,
Llama-3 70B, 5 steps, MFU ~23.6 % at step 5).

Other single-knob disablements that also avoid the crash (with different
trade-offs): turning off `TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK`,
or turning off `NCCL_CUMEM_ENABLE` (this breaks `ncclMemAlloc` /
`symm_mem` and the bench, so not viable for the bench but is fine for
plain FSDP).

### Verified via rocprof: this workaround does NOT put FSDP on the real SDMA path

Per-rank `rocprofv3 --hsa-amd-trace` comparison (see
`debug/run_bench_rocprof.sh` and `debug/run_sdma_rocprof.sh`):

| API | bench (`symm_mem` AG) | FSDP CE + LOCAL_REGISTER=0 | FSDP TRUE RING (`NCCL_CUMEM_ENABLE=0`) |
|---|---|---|---|
| **`hsa_amd_memory_async_batch_copy`** | **22** (21.6 ms) | **0** | **0** |
| `hsa_amd_memory_async_copy_on_engine` | 1 (init only) | 1 (init only) | 1 (init only) |
| `hsa_amd_signal_create` | 33,185 | 31,000 | 420 |
| `hsa_amd_vmem_map` / `_unmap` | 622 / 622 | 532 / 532 | 0 / 0 |
| `hsa_amd_vmem_export/import_shareable_handle` | 238 / 238 | 168 / 168 | 0 / 0 |
| `hsa_amd_ipc_memory_create/attach/detach` | 0 / 0 / 0 | 0 / 0 / 0 | 168 / 168 / 168 |
| GPU kernel doing the bytes | **`__amd_rocclr_batchMemOp.kd`** | `ncclDevKernel_Generic_2` | `ncclDevKernel_Generic_2` |

`hsa_amd_memory_async_batch_copy` is rocclr's canonical SDMA dispatch
API (its GPU-side stub kernel is `__amd_rocclr_batchMemOp.kd`). The
**bench** uses it 22 times across its 8 AG iterations. **FSDP uses it
zero times** in either CE or TRUE RING configuration — the bytes move
inside `ncclDevKernel_Generic_2`, which is RCCL's own generic kernel
running on CUs and reading/writing peer-mapped VAs directly. The cuMem
`vmem_*` plumbing FSDP sets up is used for **peer address translation**,
not for **SDMA dispatch**. NCCL's `Channel XX/0 ... via P2P/CUMEMCUMEM`
init line therefore only attests to the cuMem transport setup, **not** to
SDMA usage for the actual data movement.

This contradicts what an earlier write-up in this README claimed
(it implied "CE/SDMA stays active under LOCAL_REGISTER=0"). The
workaround keeps FSDP working and on the cuMem IPC transport, but the
copy-engine dispatch path used by the bench is **not yet reachable from
FSDP** with `cudaMalloc`-backed AG buffers. The path forward is to
allocate FSDP's AG send/recv buffers via `symm_mem.empty` (cuMem-backed),
which is what the `staged_symm_ag` mode in
`debug/fsdp_like_ag_probe.py` prototypes.

Three follow-up artifacts contain the raw evidence:

| | |
|---|---|
| `debug/run_bench_rocprof.sh` | runs `bench_ar_gemm.py --mode sdma` under `rocprofv3 --hsa-amd-trace`; outputs `debug/bench_rocprof_out/rank{0..7}/trace_hsa_api_stats.csv` |
| `debug/run_sdma_rocprof.sh` | runs `fsdp_like_ag_probe.py --mode regular_ag` under the same rocprof harness, in both `ce_localreg0` and `true_ring` configurations |
| `debug/run_sdma_sweep.sh` | sweep harness used to pin the discriminator: `MODE x NCCL_LOCAL_REGISTER` matrix under rocprof, see next subsection |
| `debug/run_bench_profile.sh` | runs the bench under `torch.profiler` for chrome-trace inspection; the diff to `torchtitan/outputs_ce_localreg0/profile_trace/iteration_5/rank0_trace.json` shows `__amd_rocclr_batchMemOp.kd` only in the bench |

### Productized integration via Primus (this repo's submodule)

The end-to-end Primus integration of the SDMA enabler lives on the
`feature/sdma-symm-mem-fsdp` branch of [`primus/`](./primus) (a
submodule of this repo, pointing at <git@github.com:Z-Y00/Primus.git>).
Two additions there:

| | |
|---|---|
| `primus/backends/torchtitan/patches/sdma_symm_mem_collectives.py` | Registers a Primus patch (`torchtitan.fsdp.sdma_symm_mem_collectives`) that runs at trainer `setup` phase. It wraps `torch.distributed.fsdp.fully_shard` so every fully_shard'd module automatically gets `set_custom_all_gather(SymmMemAllGather(group))` and `set_custom_reduce_scatter(SymmMemReduceScatter(group))` attached. Activated via `primus_sdma.enable_symm_mem_collectives: true` in the experiment YAML. Carries `fully_shard.state` across the wrapper (otherwise nested fully_shard calls AttributeError on `fully_shard.state(modules[0])`). Skips multi-param-group modules gracefully. |
| `examples/torchtitan/configs/MI300X/llama3.1_{8B,70B}-BF16-SDMA-pretrain.yaml` | Experiment YAMLs that flip the patch on, point `model.hf_assets_path` at a runner-provided dir (`PRIMUS_HF_ASSETS_PATH`) for the public `unsloth/Meta-Llama-3.1-70B-Instruct` tokenizer mirror, and drop the `primus_turbo` model converter for a minimal SDMA bring-up. |
| `torchtitan/run_primus_sdma.sh` (this repo) | Runner that launches the lorrisync image, mounts the Primus submodule, pip-installs the minimal Primus deps, stages the public unsloth tokenizer to a host dir, exports the CE env (`NCCL_CTA_POLICY=2 NCCL_CUMEM_ENABLE=1 NCCL_LOCAL_REGISTER=0 TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true LD_PRELOAD=libhip_attr_drain.so NCCL_SOCKET_IFNAME=lo`), and runs `primus-cli direct -- train pretrain --config <8B\|70B>-SDMA-pretrain.yaml`. Knobs: `SCALE=8b\|70b`, `STEPS=N`, `TOKENIZER_REPO=...`. |

Smoke results on 8x MI300X, this build (5-step mock_data smoke):

| scale | step 1 (warmup) | steady state (step 5) | memory |
|---|---|---|---|
| Llama-3.1 8B  | tps 120 / mfu 0.4 % | **tps 2960 / mfu 10.99 %** | 27.27 GiB |
| Llama-3.1 70B | tps 138 / mfu 4.79 % | **tps 595  / mfu 20.55 %** | 164.89 GiB |

(For comparison, the same 70B workload on the *default* `DefaultAllGather`
path through this repo's `torchtitan/run_train.sh` lands at step 5
tps 682 / mfu 23.6 %. The SDMA path is slightly slower at this
small-batch, full-ACK config because comm is already well-hidden by
recompute, and the SymmMem mempool adds a first-step warmup cost. The
SDMA path wins on comm-bound workloads where the rocclr SDMA dispatch
keeps the CUs entirely free; the SDMA-vs-CU choice was independently
verified via rocprof in the section above.)

### Path to real SDMA in FSDP: `SymmMemAllGather`

The root question is now: **what makes RCCL choose the SDMA dispatch path
vs the on-CU generic kernel?** Sweep matrix (`debug/run_sdma_sweep.sh`,
8 ranks × 3 iters, NUMEL = 32 MB bf16/rank), counting
`hsa_amd_memory_async_batch_copy` across all 8 ranks:

| cell (mode × NCCL_LOCAL_REGISTER) | AG output buffer | `batch_copy` calls | SDMA? |
|---|---|---|---|
| `symm_ag` × 0 | `symm_mem.empty` | **24** (3/rank) | **YES** |
| `symm_ag` × 2 | `symm_mem.empty` | **24** | **YES** |
| `staged_symm_ag` × 0 | regular → `symm_mem` staging → AG | **24** | **YES** |
| `regular_ag` × 0 | `torch.full` (`hipMalloc`) | 0 | NO |
| `fsdp_forward` × 0 | FSDP default (`torch.empty` via DefaultAllGather) | 0 | NO |
| `fsdp_forward_symm` × 0 | FSDP via `SymmMemAllGather` (`symm_mem.empty`) | **192** (~24/rank) | **YES** |

**Two clean conclusions:**

1. **The discriminator is buffer provenance, not the env knob.** `symm_ag`
   fires `batch_copy` 24 times whether `NCCL_LOCAL_REGISTER=0` or `=2`.
   The rocclr SDMA dispatch path triggers purely on "is the user buffer
   cuMem-backed?". `NCCL_LOCAL_REGISTER` is orthogonal (it controls the
   IPC auto-registration that hits the unrelated `hipMemRetainAllocationHandle`
   crash; see above).
2. **The fix is one PyTorch call: `module.set_custom_all_gather(SymmMemAllGather(group))`**
   (and the analogous `set_custom_reduce_scatter`). PyTorch 2.12 already
   ships `SymmMemAllGather` in `torch.distributed.fsdp._fully_shard._fsdp_collectives`
   — its `allocate()` uses `symm_mem.get_mem_pool(device)` so the AG
   output buffer is cuMem-backed. The source comment literally explains
   what we observed:
   ```
   # Calling regular all-gather would already cause libraries like NCCL to
   # use its optimized all-gather implementation for symmetric memory:
   #   - Copy Engine All-Gather (when zero-CTA policy is enabled)
   #   - Symmetric Kernel All-Gather (when zero-CTA policy is not enabled)
   ```

The exact alloc site that needed to change is
`pytorch/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py:351`:
```python
all_gather_output = all_gather_comm.allocate(   # default: DefaultAllGather -> torch.empty (hipMalloc)
    (all_gather_input_numel * world_size,), dtype=dtype, device=device,
)
```
After `fully_shard()`, wiring in the SDMA-eligible alloc looks like:
```python
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    SymmMemAllGather, SymmMemReduceScatter,
)
group = dist.group.WORLD
for m in (model, *model):                       # outer + each per-layer FSDP unit
    m.set_custom_all_gather(SymmMemAllGather(group))
    m.set_custom_reduce_scatter(SymmMemReduceScatter(group))
```

See `debug/fsdp_like_ag_probe.py --mode fsdp_forward_symm` for the
working reference. Under rocprof it produces `192 hsa_amd_memory_async_batch_copy`
calls (3 iters × 8 layers × 8 ranks) vs `0` for the unmodified
`fsdp_forward` mode.

Known caveats with the `SymmMemAllGather` switch:
- **Teardown noise.** With `SymmMemAllGather` enabled, every rank emits
  `c10::DistBackendError: NCCL communicator was aborted on rank N`
  during `destroy_process_group`. Forward / backward / step all run
  correctly; the abort happens after the last step. Likely the symm_mem
  mempool is not drained before PG destroy. Functional-only; needs a
  fix.
- **`hsa_amd_memory_async_copy_on_engine` stays at 552/rank in both
  modes** — those are FSDP's `all_gather_copy_in/out` boundary memcpys
  (param shard packing/unpacking). They were already on the SDMA engine
  via the runtime's per-op dispatch; they are independent of the AG
  output buffer change.

### Fix proposals (in order of where the fix belongs)

1. **PyTorch FSDP wiring (the actual fix to use SDMA in FSDP):** call
   `module.set_custom_all_gather(SymmMemAllGather(group))` and
   `module.set_custom_reduce_scatter(SymmMemReduceScatter(group))` on
   every fully_shard'd module after construction. Already in PyTorch
   2.12; just needs to be opted into by Torchtitan. Independent of the
   `hipMemRetainAllocationHandle` crash, and works with
   `NCCL_LOCAL_REGISTER=0`.
2. **HIP runtime (still worth fixing):** add a NULL check on the
   per-allocation cuMem-handle sub-slot in
   `hipMemRetainAllocationHandle`; return `hipErrorInvalidValue` instead
   of dereferencing NULL when the VA wasn't created via `cuMemCreate`.
   Matches CUDA driver behavior and unblocks RCCL's existing legacy-IPC
   fallback, which would let `NCCL_LOCAL_REGISTER=2` be safe again.
3. **RCCL workaround:** probe with `hipPointerGetAttribute` before
   calling `cuMemRetainAllocationHandle`; skip the cuMem branch when the
   buffer isn't cuMem-backed.
4. **User-side LD_PRELOAD shim:** same shape as
   `debug/hip_attr_drain_preload.c`; wrap `hipMemRetainAllocationHandle`
   so non-cuMem VAs return `hipErrorInvalidValue` without entering the
   runtime. No RCCL/PyTorch rebuilds required.
5. **User env workaround (avoids crash, does not enable SDMA):**
   `NCCL_LOCAL_REGISTER=0`. Use as a default unless/until (1) lands;
   does not by itself put the AG on the SDMA path.

### Files

| | |
|---|---|
| `debug/fsdp_like_ag_probe.py` | Five-mode AG reproducer: `symm_ag` (cuMem buffers, works + SDMA), `regular_ag` (caching-allocator buffers, crashes under LOCAL_REG=2, no SDMA), `staged_symm_ag` (regular→symm staging, gets SDMA), `fsdp_forward` (tiny FSDP2 model, matches Torchtitan stack, no SDMA), **`fsdp_forward_symm` (FSDP2 + `set_custom_all_gather(SymmMemAllGather(...))`, gets SDMA)** |
| `debug/run_fsdp_like_probe.sh` | One-shot container runner for the probe; forwards `NCCL_DEBUG[_SUBSYS]` |
| `debug/run_ce_ablation.sh` | Drives the probe with named env presets to ablate each CE knob independently. Explicit `NCCL_LOCAL_REGISTER=0` / `…ALLOCATOR_HOOK=false` in off-presets because RCCL defaults `LOCAL_REGISTER=1` and unsetting is not enough |
| `debug/run_sdma_sweep.sh` | sweep harness used to pin the SDMA discriminator: runs each probe mode (and each NCCL_LOCAL_REGISTER value) under `rocprofv3 --hsa-amd-trace`, then counts `hsa_amd_memory_async_batch_copy` per cell |
| `debug/run_sdma_rocprof.sh` | rocprof harness for the original ce_localreg0-vs-true_ring comparison (regular_ag mode) |
| `debug/run_bench_rocprof.sh` | rocprof harness for `bench_ar_gemm.py --mode sdma` (the known-good SDMA path) |
| `debug/run_bench_profile.sh` | torch.profiler chrome-trace harness for the bench; the diff to a torchtitan trace shows `__amd_rocclr_batchMemOp.kd` only in the bench |
| `debug/hip_retain_handle_probe.c` | Pure-HIP, no-RCCL, no-PyTorch reproducer. Three modes: `null` (clean error), `cumem` (success), `hipmalloc` (SIGSEGV) |
| `debug/run_hip_retain_handle_probe.sh` | Builds + runs the pure-HIP probe in three sibling containers (each mode in its own process so SIGSEGV in one doesn't poison the next) |
| `debug/ROOT_CAUSE_hipMemRetainAllocationHandle.md` | Full write-up: symptom, crash decode, RCCL call chain w/ source quote, ablation matrix, pure-HIP repro transcript, fix proposals |

## (2) Comm/compute overlap benchmarks

Lives in `bench/`. Two scripts, one shared runner:

| | |
|---|---|
| `bench/bench_common.py` | distributed init, cuda-event timing harness, table printer, shape config (Llama-70B FFN-up + FFN-down) |
| `bench/bench_ag_gemm.py` | AllGather + GEMM overlap on the SDMA/CE path |
| `bench/bench_ar_gemm.py` | AllReduce + GEMM overlap, two implementations (`--mode sdma` = AG-on-CE + local reduce; `--mode ref` = `dist.all_reduce` ring) |
| `bench/run_bench.sh` | Builds `libhip_attr_drain.so`, sets CE-mode vs default-RCCL env, torchruns the right scripts. `./run_bench.sh {ag\|ar\|both}` -- overrides via env: `NPROC`, `WARMUP`, `TIMED`, `GEMM_ITERS`, `COMM_ITERS` |
| `bench/run_trace.sh` | Same as above but profiles the overlap path with `torch.profiler` and `docker cp`s per-rank Chrome traces to `./traces/` |

The AR bench runs as two torchruns with different env: CE-mode for the
SDMA path, default-RCCL env for the reference. (Reason: `NCCL_CTA_POLICY=2`
disables ring `AllReduce` because there are no SMs available to run it;
CE collectives cover AG/RS only.)

### Workload pattern

Each timing case in the bench is a **block of `G` back-to-back GEMMs**
issued on the compute stream concurrently with **`K` collectives** on
the comm stream. Defaults: `G=8`, `K=1` — one collective overlapped
with eight matmuls' worth of compute. This matches how comm actually
shows up in production: a periodic event hidden inside many layers of
ongoing compute, rather than a one-shot matmul-vs-collective race.

Per case we report:

| column | meaning |
|---|---|
| `gemm_ms` | wall time for the G-GEMM loop alone |
| `comm_ms` (`ar_ms`) | wall time for the K-collective loop alone |
| `overlap_ms` | wall time for both loops issued together |
| `hidden_ms` | how much of `comm_ms` got hidden by the GEMM loop, clamped to `[0, comm_ms]` |
| `hidden_%` | `hidden_ms / comm_ms` |
| `eff` | `max(gemm_ms, comm_ms) / overlap_ms` — 1.0 = perfect overlap |

### Smoke results (single-node 8x MI300X, bf16, G=8, K=1, warmup=5, timed=20)

AllGather + GEMM:

| shape | gemm_ms | comm_ms | overlap_ms | hidden_ms | hidden_% | eff |
|---|---|---|---|---|---|---|
| llama70b_ffn_up   (M=8192, N=28672, K=8192)  | 51.30 | 2.54 | 51.95 | 1.89 | 74.3 | 0.99 |
| llama70b_ffn_down (M=8192, N=8192,  K=28672) | 49.59 | 8.62 | 50.22 | 7.99 | 92.7 | 0.99 |

AllReduce + GEMM, both modes (N = M*N elems, capped 256 MB bf16):

| mode | shape    | gemm_ms | ar_ms  | overlap_ms | hidden_ms | hidden_% | eff |
|------|----------|---------|--------|------------|-----------|----------|-----|
| sdma | ffn_up   | 51.65   | 39.45  | 53.80      | 37.31     | 94.6     | 0.96 |
| sdma | ffn_down | 49.94   | 19.79  | 51.15      | 18.58     | 93.9     | 0.98 |
| ref  | ffn_up   | 52.26   |  1.75  | 53.47      |  0.54     | 30.7     | 0.98 |
| ref  | ffn_down | 50.45   |  0.93  | 51.55      |  0.00     |  0.0     | 0.98 |

**Headline.** In the realistic "comm-hidden-in-compute" regime, the
SDMA AllReduce hides 94-95% of its work inside the 8-GEMM loop and adds
only ~2 ms of wallclock to a ~52 ms compute block, even though the
comm itself is 20-40 ms in isolation. The ring AllReduce is so fast
(< 2 ms) that it adds barely anything regardless of how you schedule
it, so the wallclock-overhead delta between the two implementations is
only **~1 ms over 52 ms**, not the 5× ratio the standalone numbers
suggest. Both ARs are essentially "free" against an 8-GEMM background;
the SDMA path is the more interesting result because it shows that even
a 40 ms collective can disappear under enough compute, at the cost of
spending the CE engines while the ring AR uses the full xGMI fabric
through the SMs.

For AllGather, the CE path is even more dramatic: 2.5-8.6 ms of comm
hidden inside 50 ms of compute with `eff = 0.99` and `hidden_% ≥ 74%`
across both shapes.

Useful next experiments:
- Tune `G/C` to find the regime where the SDMA path stops fitting in
  the compute window (e.g. `G=2 K=1` for the AR case will reveal the
  raw comm-vs-comm gap again).
- Sweep payload size to find the CE-vs-ring crossover for raw AR cost.
- Chunk + pipeline the SDMA AG over multiple comm streams to lift the
  per-rank ~47 GB/s ceiling.
- Compare against reduce-scatter + all-gather (the standard FSDP decomp)
  via CE to see if that closes the per-iter raw-cost gap.
