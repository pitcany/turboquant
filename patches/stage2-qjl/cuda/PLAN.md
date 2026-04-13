# CUDA kernel plan — follow-up commit

Scope: port `ggml-tq-paper.c` to CUDA for sm_89 (RTX 4090) and sm_120
(RTX 5090). CPU path stays as the validation oracle and fallback.

> **Branch note (`claude/swap-haar-to-wht-cuda`)**: this document was
> originally written for the paper-faithful Haar rotation Π stored as a
> dense 32·d²-float matrix per head-dim. The CPU path has since been
> swapped to a Randomized Hadamard Transform — see
> `ggml-tq-paper.c` and the WHT-variant summary in
> `../BYTE_LAYOUT.md`. The CUDA kernels have now also been ported to the
> same WHT variant (see the updated "Kernel 1" and "Kernel 2" sections
> below); sections describing the d×d Π matmul and its constant-memory
> budget are preserved as historical context for diff-review clarity.

## Files to add

```
patches/stage2-qjl/cuda/
├── tqp-quantize.cu          // quantize_row on-device (per-block parallel)
├── tqp-prepare-query.cu     // Sq = S·q once per query token
├── tqp-vec-dot.cu           // (query, K-block) → score
└── tqp-kernels.cuh          // shared helpers (bitplane unpack, fp16 conv)
```

Plus wiring:
- `cuda/ggml-cuda.cu` dispatch table entries for `GGML_TYPE_TQ4P_D128/D256`.
- `ggml/src/CMakeLists.txt` CUDA target picks up the new `.cu` files.

## Constants in `__constant__` memory (WHT variant)

After the RHT swap, the per-layer rotation data drops from a d×d matrix to
a d-sized ±1 sign vector, so all 32 layers fit comfortably in the 64 KB
constant cache (Ada) for both d=128 and d=256. The S matrix is still too
large for `__constant__` and lives in device global memory.

| Symbol | Size (d=128) | Size (d=256) | Cache |
|---|---|---|---|
| `c_tqp_sigma_d{D}[32][D]` | 16 KB | 32 KB | `__constant__`, fits 64K cache |
| `d_tqp_s_d{D}` (32 × d² floats) | 2 MB | 8 MB | device global + L2 |
| `c_tqp_centroids_d{D}` | 32 B | 32 B | `__constant__`, trivial |
| `c_tqp_boundaries_d{D}` | 28 B | 28 B | `__constant__`, trivial |

### Pre-WHT layout (historical)

| Symbol | Size (d=128) | Size (d=256) | Fits const cache? |
|---|---|---|---|
| `tqp_pi_d{D}` | 32 KB | 128 KB | d=128 yes (Ada 64K/Blackwell 128K); d=256 spills to L2 |
| `tqp_s_d{D}` | 32 KB | 128 KB | same |

d=256 spilling for Ada was tolerable — access pattern is broadcast-heavy and
L2 is well-cached. Predicted ~15% throughput loss vs. d=128 at the time.
Under the RHT, this spill disappears entirely on the rotation side.

## Kernel 1: `tqp_quantize_kernel_d{D}` (WHT variant)

**Grid:** one CTA per block to quantize. `D` threads per CTA (one per coord).

**Steps per CTA:**
1. Load `D` input floats into shared memory.
2. Compute `orig_norm` via a serial reduction in thread 0 (fine for this
   small `D`; a warp-level reduction would also work).
3. `x_unit[tid] = x[tid] / orig_norm`, kept in *both* shared memory and a
   per-thread register (the register survives the destructive WHTs).
4. Forward RHT: `smem[tid] = σ[tid] · x_unit[tid]`, then in-place FWHT on
   `smem` (`log₂(D)` butterfly steps, `log₂(D)` `__syncthreads`), then
   `x_rot[tid] = smem[tid] / √D`.
5. `idx[tid] = bucketize(x_rot[tid], boundaries)`.
6. Inverse RHT: `smem[tid] = centroids[idx[tid]]`, in-place FWHT, then
   `x_hat_unit[tid] = σ[tid] · smem[tid] / √D`.
7. `residual[tid] = x_unit_reg - x_hat_unit`, written back to `smem`;
   `‖residual‖` reduced in thread 0.
8. `proj[tid] = S[tid, :] · residual`: d-wide FMA loop, S read from
   device memory via `__ldg`.
9. Warp-ballot for sign packing; lane 0 of each warp writes four
   `uint32_t`s (or eight for d=256) of `qjl_signs`.
10. Thread 0 writes `orig_norm`, `res_d` (fp16).
11. Bitplane-pack indices via `__ballot_sync` on the low/mid/high bits of
    `idx` — one warp contributes three bytes per 8 coords.

**Register pressure:** one extra `float` (the persistent `x_unit_reg` +
the per-thread `σ[tid]` slot) vs. the pre-WHT kernel. Both are spills of
values that used to sit in constant cache; occupancy is effectively
unchanged.

**Flop count:** `O(D log D)` for the two WHTs (a big win over `O(D²)`
GEMV). The `S · residual` matmul is still `O(D²)` and dominates.

## Kernel 2: `tqp_prepare_query_kernel_d{D}` (WHT variant)

**Grid:** one CTA per (layer × query token). `D` threads.

**Steps:**
1. Load `q` into shared memory; each thread also holds its own `q[tid]`
   in a register.
2. `Sq[tid] = Σ_j S[tid, j] · q[j]` — unchanged from pre-WHT path.
3. After the S matmul finishes, reuse `q_smem` for the RHT:
   `q_smem[tid] = σ[tid] · q[tid]`, in-place FWHT, then
   `q_rot[tid] = q_smem[tid] / √D`.
4. Write `Sq` and `q_rot` to their per-(layer × token) global buffers.

Called once per token per layer. `q_rot` is the same size as `Sq`
(`D × sizeof(float)` per token), well under 1 KB at d=256.

## Kernel 3: `tqp_vec_dot_kernel_d{D}`

**Grid:** one CTA per (query, K-chunk) pair, where K-chunk covers a
cache-line-aligned span of K positions. 32 threads (one warp) per CTA
works well for d=128, 64 threads for d=256.

**Steps:**
1. Each thread loads one K block's `idx` and `signs` via bitplane unpack.
2. Warp loads `orig_norm`, `res_d` (one thread), broadcast.
3. Stage 1: `term1 = Σ_i q_rot[i] · centroids[idx[i]]`, where `q_rot`
   comes from a precomputed per-query buffer (upstream rotation kernel),
   or recomputed on-the-fly from `q` and constant-memory Π if the fork's
   attention path doesn't separate q-rotation. Warp-sum reduction.
4. Stage 2: `term2 = res_d · sqrt_pi_over_2 / d · Σ_i Sq[i] · (signs[i] ? -1 : 1)`.
   `Sq` comes from the prepare_query output buffer. Warp-sum reduction.
5. Return `orig_norm · (term1 + term2)`.

**Amortization:** `Sq` is reused across all K positions in a single
attention pass. For N cached keys, Stage 2 adds `O(d)` flops per key and
`O(d²)` setup, vs. Stage 1's `O(d)` per key — so Stage 2 is <1% overhead
at N > 100.

## Performance targets

Measured on `TQ3_0` (the fork's existing type) as baseline. Pre-WHT
numbers (from the original plan) kept for diff context; WHT numbers are
rough estimates pending on-hardware measurement.

| Config | 4090 decode | 5090 decode |
|---|---|---|
| TQ3_0 (baseline) | 100% | 100% |
| TQ4P_D128, pre-WHT (Haar Π matmul) | ~85% | ~90% |
| TQ4P_D256, pre-WHT (Haar Π matmul) | ~70% | ~80% |
| TQ4P_D128, WHT (this commit) | ≥ pre-WHT; bounded below by S matmul | same |
| TQ4P_D256, WHT (this commit) | notably faster than pre-WHT at d=256 (no Π const-cache spill) | same |

The rotation side is now `O(D log D)` instead of `O(D²)`, but the overall
kernel is bounded below by the remaining `S · residual` / `S · q`
matmuls, which are unchanged. Expect a solid win at d=256 where Π used
to spill the Ada constant cache; modest win at d=128.

## Build integration

`build_ollama_tq.sh --stage2-cuda` (new flag in follow-up): appends the
`.cu` sources to `ggml/src/CMakeLists.txt` and sets
`CMAKE_CUDA_ARCHITECTURES="89;120"` so both GPUs get kernels in one build.

## Explicit non-goals of the CUDA commit

- No tensor-core (wmma) usage. d=128 / d=256 matvecs are below efficient TC
  tile size; plain FMA via CUDA cores is equivalent and simpler.
- No sm_90 (Hopper) paths — wgmma/TMA are unavailable on Ada and irrelevant
  for this data size on Blackwell.
- No quantized query (int8) path — keeping `vec_dot_type = GGML_TYPE_F32`
  matches the CPU draft. Adding Q8_K query support is a second CUDA
  optimization commit.
- No multi-GPU-specific code — llama.cpp's `--tensor-split` + per-device
  constant memory replication handle this for free.

## Validation for the CUDA commit

The same byte-exact tests in `python/test_c_vs_python.py` will run against
the CUDA implementation by switching the ctypes dlopen target. Adds:
`test_cuda_vs_cpu.py` that quantizes on both and asserts byte-identical
output and inner-product agreement within fp32 accumulation noise
(<1e-4). If the CUDA path drifts from the CPU reference, the test fails
and the deviation is visible at the block level.

## Perf polish (item 5) — NOT APPLICABLE

Profiled on RTX 4090 (sm_89), CUDA 12.9, d=128 WHT path. Full ncu HW
counters unavailable (RmProfilingAdminOnly=1), so analysis is based on
static instruction counts and CUDA-event wall-clock timing.

Cost breakdown of `tqp_quantize_kernel_d128<WHT>`:

| Phase | FMAs/thread | % of total | Notes |
|---|---|---|---|
| S·residual GEMV | 128 | 87.7% | 128 `__ldg` from device global per thread |
| FWHT (2x) | 7 each | 9.6% | 7 butterfly iters, 2-way bank conflict at h=32,64 |
| Norm reductions (2x) | 128 each (tid==0) | 1.4% | Serial on thread 0, all others idle |
| Bucketize + ballot | ~1 | 1.4% | Negligible |

The S·residual GEMV dominates at ~88% of total work. The two serial
norm reductions (||x|| and ||residual||) add ~256 serial FMAs on thread 0
while 127 threads idle — but this is 1.4% of the total instruction budget
and invisible next to the 16,384-FMA GEMV. The FWHT butterfly hits 2-way
shared-memory bank conflicts at h>=32 (XOR stride aliases to the same
32-bank column), adding ~4 extra transactions per thread vs ~128 global
loads in the GEMV — also noise.

Warp-shuffle reduction for the norms and a skewed butterfly for the FWHT
would both be correct optimizations in isolation, but their combined
speedup is bounded by Amdahl's law to < 11% of kernel time — and the
kernel itself is not on the critical path of a typical inference loop
(quantize runs once per K-cache write; vec_dot runs once per query×key
pair and is ~1000x cheaper per block).
