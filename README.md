# sdma_rccl_pytorch

Top-level workspace for two pieces of work on AMD MI300X. Layout:

```
debug/    interposer fix + bug-investigation reproducers
bench/    AG+GEMM and AR+GEMM overlap benches
docker/   image-build infrastructure (PyTorch 2.12 on top of TheRock 2.11 base)
```

1. **Interposer fix** (`debug/`) for the "first kernel after `ncclMemAlloc`
   fails with `hipErrorInvalidValue`" bug on RCCL's `NCCL_CUMEM_ENABLE=1` path.
2. **SDMA-based comm/compute overlap benchmarks** (`bench/`) for
   AllGather+GEMM and AllReduce+GEMM. (Sibling sub-project
   `rocm_sdma_comm_compute_overlap/` is its own git repo, not tracked here.)
3. **PyTorch-2.12 build recipe** (`docker/`) for stacking PyTorch 2.12 on top
   of the TheRock ROCm 7.14 / 2.11 base image; gives a notably better
   torch.profiler trace where SDMA copy-engine kernels show up directly.

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
