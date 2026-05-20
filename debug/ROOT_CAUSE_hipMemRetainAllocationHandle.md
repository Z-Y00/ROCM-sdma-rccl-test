# Root cause: `hipMemRetainAllocationHandle` SIGSEGV under CE/SDMA + FSDP

**TL;DR.** The HIP runtime in `libamdhip64.so.7` (ROCm 7.14, build
`39213316d2`) segfaults inside `hipMemRetainAllocationHandle` when called
on a VA that was allocated via `hipMalloc`/PyTorch caching allocator (i.e.
**not** through the cuMem API). The function is documented and used by RCCL
as a probe: RCCL **expects** non-success on cuMem-incompatible VAs and falls
back to legacy CUDA IPC. Instead the runtime dereferences a NULL per-allocation
cuMem-handle slot and crashes. The fault is purely in `libamdhip64`; RCCL and
PyTorch are doing the right thing.

Repro is independent of NCCL, PyTorch, Torchtitan, FSDP, kernel version, or
distributed init. See `debug/hip_retain_handle_probe.c` for the 10-line
demonstrator.

---

## 1. Observed symptom

Single-node 8x MI300X, full CE env:

```
NCCL_CTA_POLICY=2 NCCL_CUMEM_ENABLE=1 NCCL_LOCAL_REGISTER=2 \
TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true \
torchrun --nproc_per_node=8 debug/fsdp_like_ag_probe.py --mode fsdp_forward
```

All 8 ranks SIGSEGV on the **first** forward AllGather:

```
Fatal Python error: Segmentation fault
File ".../distributed_c10d.py", line 4316 in all_gather_into_tensor
File ".../_fully_shard/_fsdp_collectives.py", line 364 in foreach_all_gather
File ".../_fully_shard/_fsdp_param_group.py", line 372 in unshard
...
C stack:
  hipMemRetainAllocationHandle+0x1b           [libamdhip64.so.7]
  +0x466ce3                                   [libamdhip64.so.7]
  +0x5f6ad4                                   [librocprofiler-sdk.so.1]
  +0x14e606f5  ipcRegisterBuffer              [librccl.so.1]
  +0x14e603eb  ncclIpcLocalRegisterBuffer     [librccl.so.1]
  +0x14e32155  ncclRegisterCollBuffers        [librccl.so.1]
  +0x14d5a667  ncclTasksRegAndEnqueue         [librccl.so.1]
  +0x14d6cbb0  groupLaunch                    [librccl.so.1]
  +0x14d6b08e  ncclGroupEndInternal           [librccl.so.1]
  +0x14d667c2  ncclEnqueueCheck               [librccl.so.1]
  ncclAllGather_impl+0xbb9
  c10d::ProcessGroupNCCL::_allgather_base
```

Same crash from the `regular_ag` mode (which uses ordinary
`torch.empty`/`torch.full` tensors), proving the trigger has nothing to do
with FSDP itself — it is the **buffer type passed to
`dist.all_gather_into_tensor`** that matters.

The `symm_ag` mode (which routes through
`symm_mem.empty -> symm_mem.rendezvous -> dist.all_gather_into_tensor`)
**passes**, because the buffer was allocated via `ncclMemAlloc -> cuMemCreate`
and is a real cuMem-backed VA.

---

## 2. Crash decode

`addr2line` on the RCCL offsets gives the symbolicated chain above. The
fault itself is in `hipMemRetainAllocationHandle+0x1b`, which is just
this:

```
hipMemRetainAllocationHandle:
  push %rbp
  push %r14
  push %rbx
  mov  %rsi,%rbx               ; rbx = arg2 (dptr)
  mov  %rdi,%r14               ; r14 = arg1 (out handle**)
  call <get-dispatcher>        ; dispatcher object -> rax
  mov  %r14,%rdi
  mov  %rbx,%rsi
  call *0x808(%rax)            ; vtable dispatch into per-device impl
```

The per-device implementation (libamdhip64.so.7 around `+0x466cc0`) is:

```
  call <ROCm allocation-tracker lookup>     ; returns rax = tracker entry or NULL
  test %rax,%rax / je <ret_invalid>         ; not NULL when dptr is valid
  mov  0x100(%rax),%rax                     ; rax = tracker->cuMemSlot (non-null on this build)
  mov  0xf8(%rax),%r15                      ; r15 = cuMemSlot->handle  <-- SIGSEGV (NULL deref)
```

For a `hipMalloc`'d VA the tracker entry exists, and the cuMem-slot pointer
at `+0x100` is non-null (it's a per-allocation metadata struct), but the
handle pointer inside it at `+0xf8` is NULL because the allocation was never
created through `hipMemCreate`. The next load faults.

The intended behavior is to return `hipErrorInvalidValue` cleanly. This is
how the CUDA driver implements `cuMemRetainAllocationHandle` on legacy VAs.

---

## 3. Why RCCL hits this on FSDP but not on `bench_ag_gemm.py`

The RCCL caller is the **per-collective IPC auto-registration** path
(`rocm-systems/projects/rccl/src/transport/p2p.cc`, line 890):

```cpp
// Get the mem handle for that buffer. It may have been allocated through
// cudaMalloc in which case we'll get the CUDA legacy mem handle, or through cuMem*.
if (ncclCuMemEnable()) {
#if ROCM_VERSION >= 70000
    CUmemGenericAllocationHandle handle;
    if (CUPFN(cuMemRetainAllocationHandle(&handle, baseAddr)) != CUDA_SUCCESS) {
        // if cuMem* export fails, retry legacy export
        if (comm->directMode || !ncclParamLegacyCudaRegister()) goto fail;
        CUDACHECKGOTO(cudaIpcGetMemHandle(&ipcInfo.ipcDesc.devIpc, baseAddr), ret, fail);
        ipcInfo.legacyIpcCap = true;
        ...
```

RCCL is **correct here**. The author explicitly comments "It may have been
allocated through cudaMalloc... or through cuMem*" and has a graceful
legacy-IPC fallback for the `!= CUDA_SUCCESS` case. The fallback never
executes because the call crashes instead of returning an error.

To reach this code path, three things must be true at the time the AG is
enqueued:

1. **`ncclRegisterCollBuffers` is entered at all.** The top of that function
   in `rocm-systems/projects/rccl/src/register/coll_reg.cc` is:
   ```cpp
   if (!(ncclParamLocalRegister() ||
         (comm->planner.persistent && ncclParamGraphRegister()))) goto exit;
   ```
   So `NCCL_LOCAL_REGISTER != 0` (default: `1`) is needed.

2. **`ncclRegFind` finds the buffer.** The RING branch (which is what 8-rank
   single-node AG uses on AMD — there is no NVLS) does:
   ```cpp
   NCCLCHECK(ncclRegFind(comm, info->recvbuff, recvbuffSize, &recvRegRecord));
   if (recvRegRecord == NULL && !persistent_graph_register) goto exit;
   NCCLCHECK(ncclRegFind(comm, info->sendbuff, sendbuffSize, &sendRegRecord));
   if (sendRegRecord == NULL && !persistent_graph_register) goto exit;
   ```
   So the buffer must already be in RCCL's `ncclReg` table. PyTorch's
   `TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true` hooks the caching
   allocator and calls `ncclCommRegister` on every new slab, which
   populates the table.

3. **The cuMem branch is taken.** `ipcRegisterBuffer` checks
   `if (ncclCuMemEnable())` before calling `cuMemRetainAllocationHandle`.
   `NCCL_CUMEM_ENABLE=1` is required.

Bench (`bench/bench_ag_gemm.py`) only triggers conditions 1 and 3 — its
buffers come from `symm_mem.empty -> ncclMemAlloc -> cuMemCreate`, so when
`cuMemRetainAllocationHandle` is called on them, the per-allocation
cuMem-handle slot **is** populated, the function returns successfully, and
RCCL's IPC export works as designed. FSDP's buffers come from the PyTorch
caching allocator (regular `hipMalloc` slabs), and condition 1+2+3 align
to send them into the broken HIP runtime function.

---

## 4. Ablation matrix (regular_ag, 8 ranks, kernel 6.5.0-45)

```
preset                 NCCL_CTA_POLICY  NCCL_CUMEM_ENABLE  NCCL_LOCAL_REGISTER  TORCH_NCCL_..._ALLOCATOR_HOOK  result
---------------------  ---------------  -----------------  -------------------  -----------------------------  ------
baseline_default       -                -                  0                    false                          PASS
no_reg_at_all          2                1                  0                    false                          PASS
no_local_register      2                1                  0  (default flipped) true                           PASS
no_allocator_hook      2                1                  2                    false                          PASS
no_cumem               2                0                  2                    true                           PASS
only_local_register    -                -                  2                    false                          PASS
only_allocator_hook    -                -                  0                    true                           PASS
cumem_plus_hook        -                1                  0                    true                           PASS
ce_full                2                1                  2                    true                           CRASH
no_cta_policy          -                1                  2                    true                           CRASH
```

Minimal sufficient trigger: `NCCL_CUMEM_ENABLE=1` + `NCCL_LOCAL_REGISTER=2`
+ `TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true`. `NCCL_CTA_POLICY=2`
is irrelevant to the crash (it only affects which kernel runs the payload).

---

## 5. Independent confirmation: pure-HIP probe

`debug/hip_retain_handle_probe.c` removes RCCL and PyTorch from the
equation entirely. Three modes, one process each:

```
$ ./debug/run_hip_retain_handle_probe.sh

--- mode=null (sanity) ---
hipMemRetainAllocationHandle(NULL)                       -> 1 (invalid argument)

--- mode=cumem (control) ---
hipMemCreate + hipMemAddressReserve + hipMemMap + hipMemSetAccess succeed
hipMemRetainAllocationHandle on cuMem VA 0x7c1458500000 -> 0 (no error), handle=0x...

--- mode=hipmalloc (suspect) ---
hipMalloc(1048576)                                      -> 0 (no error), va=0x7aadcf000000
hipMemset / hipDeviceSynchronize                        -> 0 (no error)
hipMemRetainAllocationHandle on hipMalloc VA 0x7aadcf000000
                                                        -> Segmentation fault (core dumped)
```

This is the entire root cause, isolated from every other component. The
clean error on `NULL` proves the function has a validation path; the SIGSEGV
on a `hipMalloc`'d VA proves the validation does not cover the
"`hipMalloc`-not-`hipMemCreate`" case.

---

## 6. Fixes (in order of where the fix belongs)

### 6a. HIP runtime (the real fix, upstream)

Add a NULL check on the per-allocation cuMem-handle slot in
`hipMemRetainAllocationHandle`'s dispatcher implementation. Pseudo-code:

```c
hipError_t hipMemRetainAllocationHandle(handle**, void* dptr) {
    auto entry = tracker_lookup(dptr);
    if (!entry) return hipErrorInvalidValue;
    auto cuMemSlot = entry->cuMemSlot;            // offset +0x100
    if (!cuMemSlot || !cuMemSlot->handle)         // offset +0xf8 inside that
        return hipErrorInvalidValue;
    *handle = retain(cuMemSlot->handle);
    return hipSuccess;
}
```

This is contract-correct, matches the CUDA driver's behavior for the
analogous case, and unblocks RCCL's pre-existing fallback to legacy
`cudaIpcGetMemHandle`.

### 6b. RCCL workaround (one-line)

Before calling `cuMemRetainAllocationHandle`, probe the VA with
`hipPointerGetAttribute(&memType, HIP_POINTER_ATTRIBUTE_MEMORY_TYPE, dptr)`
or `cuMemGetAccess`, and skip the cuMem branch if the buffer is not
cuMem-backed. Falls back to the existing legacy IPC path with no behavior
change for cuMem-backed buffers.

### 6c. User-side LD_PRELOAD shim (immediate, no rebuilds)

Same shape as the existing `debug/hip_attr_drain_preload.c` (which wraps
`hipDeviceGetAttribute` / `cuDeviceGetAttribute`). Add a wrapper around
`hipMemRetainAllocationHandle` and `cuMemRetainAllocationHandle` that
first checks the VA is cuMem-backed; on non-cuMem VAs, return
`hipErrorInvalidValue` without entering the runtime. Detection mechanism:

- `hipPointerGetAttribute(HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR, va)`
  returns the same `va` for the base of every `hipMalloc`/`hipMemMap`
  allocation; comparing this to a cuMem-known range table is one approach,
  but more cheaply we can just try the call and intercept the SIGSEGV with
  a per-thread `sigsetjmp`/`siglongjmp` guard. Practical; ugly but works.
- Cleaner: keep a process-local set of VAs we have seen come back from
  `hipMemCreate` and only allow the real call through for those. Requires
  also intercepting `hipMemCreate`/`hipMemMap`.

### 6d. User-side env-only workaround

Pick any one of these (any one alone breaks the trigger combo):

| change | what you lose |
|---|---|
| `NCCL_LOCAL_REGISTER=0` | per-collective IPC user-buffer registration optimization |
| `TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=false` | PyTorch -> RCCL slab registration; FSDP buffers won't be registered |
| `NCCL_CUMEM_ENABLE=0` | RCCL's cuMem allocator (breaks `ncclMemAlloc`, so `symm_mem.empty` won't work; bench breaks too) |

For Torchtitan/FSDP the cheapest workaround is `NCCL_LOCAL_REGISTER=0`.
That keeps `NCCL_CTA_POLICY=2` + `NCCL_CUMEM_ENABLE=1` + the allocator hook
on, so `ncclMemAlloc` (and `symm_mem`) keep working, while the
per-collective IPC registration of user buffers is skipped. Confirmed
`regular_ag` PASS in 25 ms with this preset
(`debug/run_ce_ablation.sh no_local_register`).

### 6e. PyTorch FSDP-side patch (long term)

Replace FSDP's all-gather buffer allocator with a cuMem-backed allocator
analogous to `symm_mem.empty`. The `staged_symm_ag` mode in
`debug/fsdp_like_ag_probe.py` is a stepping stone: stage regular shards
through cuMem-backed buffers immediately before the collective. The
"real" fix is to skip the staging copy and have FSDP allocate the AG
output (and the per-rank shards if possible) as cuMem-backed up front.

---

## 7. Files

| file | purpose |
|---|---|
| `debug/fsdp_like_ag_probe.py` | three-mode reproducer: `symm_ag` (works), `regular_ag` (crashes), `fsdp_forward` (crashes with Torchtitan stack), `staged_symm_ag` (workaround prototype) |
| `debug/run_fsdp_like_probe.sh` | one-shot container runner (forwards `NCCL_DEBUG[_SUBSYS]`) |
| `debug/run_ce_ablation.sh` | drives the probe with named env presets to ablate the four CE knobs independently |
| `debug/hip_retain_handle_probe.c` | pure-HIP reproducer (`null` / `cumem` / `hipmalloc` modes); no RCCL, no PyTorch |
| `debug/run_hip_retain_handle_probe.sh` | builds + runs the pure-HIP probe in three sibling containers (each mode in its own process so SIGSEGV in one doesn't poison the next) |
| `debug/hip_attr_drain_preload.c` | the existing LD_PRELOAD shim for the `cuDeviceGetAttribute(FABRIC_SUPPORTED)` TLS-leak bug (separate issue, unrelated mechanism) |
