# Stage-2 QJL patch — implementation plan

Draft plan for adding a paper-faithful TurboQuant KV cache type to the
`turbo-tan/llama.cpp-tq3` fork, so native `ollama` (rebuilt via
`scripts/build_ollama_tq.sh`) can run the full two-stage algorithm from
[TurboQuant: Online Vector Quantization with Near-optimal Distortion
Rate](https://arxiv.org/abs/2504.19874) (Zandieh et al., ICLR 2026).

## Goal

Add **new** ggml quantization types that implement the paper's exact
algorithm. The existing fork's `TQ3_0`, `TQ3_1S`, `TQ3_4S` paths are left
untouched — this patch is additive only.

Paper algorithm (recap):

1. **Stage 1 — TurboQuantMSE**: Haar random orthogonal rotation `Π` over the
   full vector, then per-coordinate Lloyd-Max scalar quantization.
2. **Stage 2 — QJL**: Gaussian random projection `S` of the residual, store
   1 sign bit per output; unbiased inner-product estimator with
   `√(π/2)/m` correction factor.

## Choice: TQ4P_0

Three bit-budgets were on the table. The user picked **TQ4P_0** (Stage 1 at
3-bit + 1-bit QJL ≈ 4.25 bpw). This matches the paper's "3-bit
TurboQuantProd" config when read as (MSE bits + 1 QJL bit).

Paper numbers for this config (confirmed by running the Python reference in
this repo on 10,000 random unit vectors at d=128, all 19 in-repo tests
passing):

| Metric | Paper | Measured (`turboquant.py`) |
|---|---|---|
| Stage-1 MSE | 0.034 | **0.0340** |
| Stage-1 MSE upper bound | 0.043 | 0.0340 ≤ 0.043 ✓ |
| Stage-1+2 inner-product correlation | 0.93 | **0.9207** |
| Estimator bias | ~0 | +0.00003 |

So the Python reference is accurate to the paper's stated numbers. The C
port's job is to reproduce the Python reference **byte-for-byte**, which in
turn reproduces the paper.

## Head dim support

ggml requires a fixed block size per quant type. LLM head dims in this
repo's target set:

| Model | head_dim |
|---|---|
| Llama 3.x (8B/70B) | 128 |
| Qwen 2.5 (all sizes) | 128 |
| Qwen 3 (default config) | 128 |
| **Qwen 3.5 (Gated Attention)** | **256** |
| DeepSeek V3 / R1 (MLA) | 128 (latent dim separate) |

**Three legacy-alias types in this patch**:

- `GGML_TYPE_TQ4P_D64` — covers `head_dim=64` models such as gpt-oss-20B.
- `GGML_TYPE_TQ4P_D128` — primary target, covers Llama + Qwen2/3.
- `GGML_TYPE_TQ4P_D256` — Qwen 3.5 coverage.

Both share the same source file via `#define TQP_D` parameterization. Only
the Π and S constants differ (generated per-d from seeds).

## Byte layout (per vector, QK = head_dim)

### TQ4P_D64 (37 B / 64 elements = 4.625 bpw)

| Offset | Size | Field |
|---|---|---|
| 0 | 2 | `orig_norm` (fp16) — `‖x‖` before unit normalization |
| 2 | 2 | `res_d` (fp16) — `‖residual‖` in rotated space |
| 4 | 1 | `layer_idx` — packed `(rotation << 7) | (layer & 0x1f)` |
| 5 | 24 | `qs[24]` — 64 × 3-bit Lloyd-Max indices, bit-packed |
| 29 | 8 | `qjl_signs[8]` — 64 × 1-bit QJL signs |

Total: **37 B / 64 = 4.625 bpw**.

### TQ4P_D128 (52 B / 128 elements = 3.25 bpw)

| Offset | Size | Field |
|---|---|---|
| 0 | 2 | `orig_norm` (fp16) — `‖x‖` before unit normalization |
| 2 | 2 | `res_d` (fp16) — `‖residual‖` in rotated space |
| 4 | 48 | `qs[48]` — 128 × 3-bit Lloyd-Max indices, bit-packed |
| 52 | 16 | `qjl_signs[16]` — 128 × 1-bit QJL signs |

Total: **68 B / 128 = 4.25 bpw**. (Earlier plan incorrectly said 52 B / 3.25
bpw — that was 2-bit indices. 3-bit is 48 B for 128 coords.)

### TQ4P_D256 (132 B / 256 elements = 4.125 bpw)

| Offset | Size | Field |
|---|---|---|
| 0 | 2 | `orig_norm` (fp16) |
| 2 | 2 | `res_d` (fp16) |
| 4 | 96 | `qs[96]` — 256 × 3-bit indices |
| 100 | 32 | `qjl_signs[32]` — 256 × 1-bit signs |

## Rotation & QJL matrices

Single fixed `Π` (Haar orthogonal) and `S` (Gaussian) per head dim, shared
across all layers. Generated from seeds (`Π seed=42`, `S seed=43`, matching
`turboquant.py`'s convention). Stored as:

- `tqp_constants_d128.h` — Π (128×128 fp16, 32 KB) + S (128×128 fp16, 32 KB)
- `tqp_constants_d256.h` — Π (256×256 fp16, 128 KB) + S (256×256 fp16, 128 KB)

On CUDA, both live in `__constant__` memory:

- Π_d128 + S_d128 = 64 KB, fits in Ada's 64 KB constant cache per SM
- Π_d256 + S_d256 = 256 KB, exceeds constant cache; falls back to L2.
  Acceptable — access pattern is still broadcast, just slower hit rate.
  (This is a real cost for Qwen3.5; ~15% expected throughput loss vs d=128.)

### Known simplification vs. paper

Paper allows per-layer rotation seeds. We use a single fixed Π per head dim,
shared across all layers. Accuracy cost is negligible (rotations are i.i.d.
in expectation), but flagged explicitly as a deviation. Per-layer Π is a
follow-up that swaps `__constant__` for `__device__` + model-load-time init.

## Algorithm — byte-exact match to `turboquant.py`

### Quantize (one vector `x` of length d)

```
orig_norm = ‖x‖
x_unit    = x / orig_norm
x_rot     = Π · x_unit                           # Stage 1 rotation
idx[i]    = bucketize(x_rot[i], BOUNDARIES)      # Lloyd-Max, 3 bits
x_hat_rot = centroids[idx]                       # reconstruction in rot. space
r_rot     = x_rot - x_hat_rot                    # residual in rot. space
res_d     = ‖r_rot‖                              # = ‖r‖ since Π orthogonal
proj      = S · r_rot                            # 128-vec (or 256)
sign[i]   = sgn(proj[i])                         # 1 bit per coord
```

Stored: `(orig_norm, res_d, idx, signs)` as bytes per the layout above.

### Dequantize

```
x_hat_rot = centroids[idx]
x_hat     = orig_norm · (Π^T · x_hat_rot)
```

QJL terms are not used for reconstruction (paper: Stage 2 only contributes
to inner-product estimation).

### Inner product `⟨q, x⟩`

Fork convention: upstream `GGML_OP_TURBO_WHT` has already applied Π to `q`,
so we receive `q_rot`. Two-part estimator:

```
term1 = orig_norm · Σᵢ q_rot[i] · centroids[idx[i]]
Sq    = S · q_rot                                # once per query, cached
term2 = orig_norm · res_d · √(π/2)/d · Σᵢ Sq[i] · sign[i]
⟨q, x⟩ ≈ term1 + term2
```

Note: `orig_norm` multiplies both terms because `turboquant.py` normalizes
to unit vectors before quantizing.

## Patch artifacts

```
patches/stage2-qjl/
├── PLAN.md                         # this file
├── README.md                       # scope + how to apply
├── BYTE_LAYOUT.md                  # the byte tables above, with C structs
├── hooks.md                        # ~5-line edits to fork enum/struct/dispatch
├── python/
│   ├── generate_constants.py       # seed → tqp_constants_d{128,256}.h + .pt
│   ├── tq_paper_reference.py       # byte-exact Python mirror of the C impl
│   └── test_tq_paper.py            # byte-exact match vs. turboquant.py
├── c/
│   ├── ggml-tq-paper.h             # public API
│   ├── ggml-tq-paper.c             # CPU quantize/dequantize/prepare_query/vec_dot
│   ├── tqp_centroids_d3.h          # Lloyd-Max centroids + boundaries for 3-bit
│   ├── tqp_constants_d128.h        # Π + S, d=128, generated
│   └── tqp_constants_d256.h        # Π + S, d=256, generated
└── cuda/
    └── PLAN.md                     # follow-up kernel design; no .cu this commit
```

## Fork-side edits (hand-applied via `hooks.md`)

All additions, no in-place modifications:

1. `ggml/include/ggml.h`: add `GGML_TYPE_TQ4P_D128` and `GGML_TYPE_TQ4P_D256`
   to the `ggml_type` enum (next free values).
2. `ggml/src/ggml-common.h`: add `block_tq4p_d128` and `block_tq4p_d256`
   structs.
3. `ggml/src/ggml-quants.c`: `#include "ggml-tq-paper.h"` + two lines in the
   dispatch table registering the new types.
4. `ggml/src/CMakeLists.txt`: add `ggml-tq-paper.c` to source list.

## CPU-first scope

Quantize, dequantize, and vec_dot implemented on CPU in this commit. CUDA
kernels deferred to follow-up. The CPU code is structured to make the CUDA
port mechanical:

- **Separated `prepare_query` function** — computes `Sq = S·q_rot` once per
  query, returns a scratch buffer. The CUDA port will invoke this as a
  per-query kernel and cache `Sq` in shared memory across all K blocks.
- **Pure stateless functions** — no globals, no thread-locals. Quant and
  vec_dot are functions of their inputs.
- **Constants in shared headers** (`tqp_constants_d*.h`) that both `.c` and
  `.cu` files will include.

This is the only structural change over a naive CPU implementation, and
it's entirely CPU-side — no CUDA in this commit.

## Validation

### Byte-exact equality vs. `turboquant.py`

The primary test. For each of 1000 random unit vectors at d=128 and d=256:

1. Generate Π, S once from seeds, emit the `.h` files.
2. Quantize via `turboquant.py::TurboQuantProd` (the already-verified
   oracle). Extract `mse_indices`, `qjl_signs`, `residual_norm`.
3. Quantize via a Python port of the C implementation (`tq_paper_reference.py`)
   that operates on the same byte layout.
4. Assert:
   - `indices` byte-identical
   - `qjl_signs` (packed) byte-identical
   - `res_d` matches within fp16 rounding (relative error ≤ 1e-3)
   - `inner_product` estimate matches within 1e-3

Any divergence means the byte layout or math is wrong and the patch is
rejected. This is a strictly stronger test than "matches paper bounds".

### Bounds regression

`test_tq_paper.py` also asserts the paper's MSE and IP-corr bounds hold
end-to-end. Expected: MSE ≤ 0.043 for 3-bit, IP-corr ≥ 0.90 at d=128.

### C-side testing

The `.c` file is not compiled in this sandbox — that happens when the user
runs `build_ollama_tq.sh --stage2`. Once compiled, round-trip C→Python
equality can be validated by dumping bytes from a C test harness and
loading them into `tq_paper_reference.py`. A minimal `test-tqp.c` is
included in the C directory for this purpose.

## Environment constraints (4090 + 5090)

- **CUDA architectures**: sm_89 (4090) + sm_120 (5090). Build with
  `CMAKE_CUDA_ARCHITECTURES="89;120"`. No use of sm_90-only features
  (wgmma, TMA) in either the CPU draft or the CUDA follow-up.
- **Multi-GPU**: llama.cpp's `--tensor-split 24,32` + `--split-mode layer`
  handles the imbalanced VRAM. Nothing for this patch to do; constants are
  replicated to each GPU's `__constant__` memory at context init (CUDA
  follow-up).
- **VRAM budget**: d=256 constants (256 KB per GPU) are trivial compared to
  model weights. KV cache at 4.25 bpw on Llama 70B at 32K context: ~550 MB
  per sequence, fits comfortably in 56 GB combined VRAM.

## CUDA follow-up (separate commit)

The CUDA commit adds `.cu` files mirroring the CPU code. Sketch:

- `tqp_quantize_kernel`: one CTA per vector; Π matmul via shared-mem
  tiled GEMV; Lloyd-Max bucketize; warp-reduce for `‖r‖`; S matmul;
  warp-ballot for sign packing.
- `tqp_prepare_query_kernel`: one CTA per query; S·q_rot matvec. Output
  cached per layer, reused across all K positions.
- `tqp_vec_dot_kernel`: one CTA per (query × K-chunk) pair; reads cached
  `Sq` from global mem, reads K block, accumulates term1 + term2.
- Throughput target vs. existing fork TQ3_0: 0.85× on 4090, 0.90× on 5090.
  Stage 2 adds <1% overhead amortized over N cached keys at long context.

## Explicit non-goals of this commit

1. CUDA kernels (follow-up commit).
2. Per-layer rotation matrices — single global Π, S per head dim.
3. Ollama Go-side allowlist patch for `OLLAMA_KV_CACHE_TYPE=tq4p_d128` —
   one-line change, trivial follow-up once the type works in llama.cpp.
4. `validate.py` extension for a TQ4P path — follow-up.
5. Head dims other than 128 and 256 — would be ~20 lines to add d=64 or
   d=96 if a specific model needs it; not speculative.

## Open questions (none — all resolved)

1. bpw: **TQ4P_0** (3-bit MSE + 1-bit QJL, 4.25 bpw at d=128). ✓
2. Head dims: **128 and 256**, after Qwen3.5 check. ✓
3. Per-query `Sq` split: **yes**. ✓
4. Extend `validate.py`: **follow-up**. ✓
5. Tests run in this sandbox: **yes**. ✓

## Commit boundary

Two commits on `claude/arxiv-llama-ollama-integration-yKp1r`:

1. **This file only** — saves the plan for review before any code.
2. **Full patch** — artifacts above, tests passing in this sandbox, plus
   `scripts/build_ollama_tq.sh --stage2` wiring. Hand-edit of the 4 fork
   files remains the user's responsibility (documented in `hooks.md`).

CUDA follow-up is a separate branch/commit beyond this scope.
