# TQ4P — TurboQuant-ish KV quantization for ollama's ggml

Two new ggml block quantization types, derived from TurboQuant
(ICLR 2026, [arXiv 2504.19874](https://arxiv.org/abs/2504.19874)),
additive to whichever ggml tree you drop it into (tested against ollama's
vendored `ml/backend/ggml/ggml/`). Nothing existing is modified in place.

| Type | head_dim | Covers | Block size | bpw |
|---|---|---|---|---|
| `GGML_TYPE_TQ4P_D128` | 128 | Llama 3.x, Qwen 2.5, Qwen 3 | 69 B / 128 | 4.3125 |
| `GGML_TYPE_TQ4P_D256` | 256 | Qwen 3.5 gated attention | 133 B / 256 | ~4.16 |

Algorithm = Randomized Hadamard Transform Π = (1/√d) · H · diag(σ)
(replacing the paper's Haar rotation on this branch) + 3-bit Lloyd-Max
per coord + 1-bit Gaussian QJL on residual. Unbiased inner-product
estimator with `√(π/2)/d` correction. One byte of each block holds the
per-layer index (32 layers supported) so Π and S are per-layer without
changing the wire format.

## Branch note: WHT swap

This branch (`claude/swap-haar-to-wht-*`) intentionally diverges from the
paper. The Haar rotation Π is replaced with the Randomized Hadamard
Transform; per-layer storage drops from `d²` floats (Π) to `d` floats (σ)
and apply cost drops from O(d²) to O(d log d). Still orthogonal, so the
paper's distortion bounds still hold for random unit vector inputs, but
the stored indices / QJL signs no longer match `turboquant.py::TurboQuantProd`
byte-for-byte.

## Validation status

The patch is **byte-exactly** self-consistent between the C and Python
reference implementations:

| Check | Result |
|---|---|
| C `quantize_row_*` output byte-identical to Python reference | ✓ |
| C `dequantize_row_*` matches Python reference (max diff < 1e-4) | ✓ |
| C `vec_dot` matches Python reference (max diff < 5e-4) | ✓ |
| Paper MSE bound (≤ 0.043 per vector for 3-bit) | ✓ |
| Paper IP correlation (≥ 0.85 for 3-bit + 1-bit QJL) | ✓ |

Reproduce locally:

```bash
cd patches/stage2-qjl/python
python3 generate_constants.py                    # regenerates Π, S, centroids
gcc -O2 -fPIC -shared -o ../c/libggml_tq_paper.so ../c/ggml-tq-paper.c
python3 -m pytest test_tq_paper.py test_c_vs_python.py -v
```

## How it gets applied

```bash
scripts/build_ollama_tq.sh           # full: clone ollama, patch ggml, build
scripts/build_ollama_tq.sh --rebuild  # reapply + rebuild only
```

What the build script does on the ollama tree:

1. Runs `generate_constants.py` so the `.h` files are current.
2. Copies `c/ggml-tq-paper.{c,h}` + `c/tqp_{constants,centroids}_*.h`
   into `ollama/ml/backend/ggml/ggml/src/`.
3. If `nvcc` is on `PATH`, also copies `cuda/tqp-*.{cu,cuh}` into
   `ollama/ml/backend/ggml/ggml/src/ggml-cuda/` and sets
   `CMAKE_CUDA_ARCHITECTURES=89;120` (4090 + 5090).
4. Runs [`apply_hooks.sh`](apply_hooks.sh) — 5 additive C-side edits:
   new enum values, block struct defs, two dispatch table entries,
   `CMakeLists.txt` source addition, and (when CUDA kernels are present)
   a `ggml_cuda_op_tqp_vec_dot` early-return inside `ggml_cuda_mul_mat`.
   All idempotent via a `tq4p` marker.
5. Runs [`apply_go_plumbing.sh`](apply_go_plumbing.sh) and
   [`../../scripts/patch_ollama_kv_types.sh`](../../scripts/patch_ollama_kv_types.sh) —
   widens ollama's KV-cache-type allowlist and threads
   `tq4p_d128` / `tq4p_d256` through four Go sites (`ml/backend.go`,
   `ml/backend/ggml/ggml.go`, `runner/ollamarunner/cache.go`,
   `llama/llama.go`) so the cache-type string maps end-to-end to the
   correct GGML enum instead of silently falling back to f16.
6. Runs `go generate && go build` on ollama.

See [`hooks.md`](hooks.md) for the exact edits, in case you prefer to
apply them by hand or need to adjust for an ollama layout change.
[`../../scripts/smoke_test_tq4p.sh`](../../scripts/smoke_test_tq4p.sh)
runs an end-to-end smoke test against the built binary.

## Layout

```
patches/stage2-qjl/
├── PLAN.md                            # design doc (earlier commit)
├── README.md                          # this file
├── BYTE_LAYOUT.md                     # the 69 B / 133 B block specs
├── hooks.md                           # the 5 fork edits you apply by hand
├── apply_hooks.sh                     # idempotent applier for C-side hooks
├── apply_go_plumbing.sh               # idempotent applier for Go-side DType plumbing
├── python/
│   ├── generate_constants.py          # seed → .h + .pt
│   ├── tq_paper_reference.py          # byte-level Python mirror
│   ├── test_tq_paper.py               # self-consistency + paper-adjacent bounds
│   ├── test_c_vs_python.py            # C-vs-Python byte-exact (via ctypes)
│   └── test_cuda_vs_cpu.py            # CUDA-vs-CPU parity test (needs nvcc)
├── c/
│   ├── ggml-tq-paper.h                # public API
│   ├── ggml-tq-paper.c                # CPU quant/dequant/prepare_query/vec_dot
│   ├── tqp_centroids_d{128,256}.h     # Lloyd-Max codebook (generated)
│   └── tqp_constants_d{128,256}.h     # σ + S per-layer (generated)
└── cuda/
    ├── PLAN.md                        # kernel design (WHT variant)
    ├── CUDA_IMPL_PLAN.md              # historical impl notes
    ├── CMakeLists.txt                 # standalone .so build
    ├── tqp-kernels.cuh                # shared device helpers + FWHT
    ├── tqp-constants-cuda.cuh         # __constant__ σ + device-mem S init
    ├── tqp-quantize.cu                # quantize kernel
    ├── tqp-prepare-query.cu           # Sq + q_rot kernel
    └── tqp-vec-dot.cu                 # per-(query, K-block) inner-product kernel
```

`c/tqp_constants.pt` is a torch state dict used by the Python reference to
load the same bits the C headers contain. `c/libggml_tq_paper.so` and
`__pycache__/` are build artifacts, gitignored.

## Scope & non-goals

- **CUDA kernels land alongside the CPU path** in `cuda/`, wired up via
  hook 5 in `apply_hooks.sh`. The CPU path remains the validation oracle
  and is ~5–10× slower than the fork's existing `TQ3_0` vec_dot.
- **Ollama wiring is end-to-end**: KV-cache-type allowlist
  (`scripts/patch_ollama_kv_types.sh`) + four-site Go DType plumbing
  (`apply_go_plumbing.sh`) + CUDA dispatch hook. `OLLAMA_KV_CACHE_TYPE=tq4p_d128`
  and `OLLAMA_KV_CACHE_TYPE=tq4p_d256` both resolve to the real GGML
  enum instead of falling back to f16.
- **Per-layer σ and S** (CPU) — the block header carries a `layer_idx`
  byte so the 32 pre-generated per-layer constants (seeds 42+i / 43+i)
  are selected at quant and dot-product time. The CUDA kernels in this
  branch currently still use layer 0 only; threading `layer_idx` through
  the CUDA block struct is a follow-up.
- **`validate.py` has a TQ4P path** that drives the C library via
  ctypes for attention-score cosine-sim checks (see the `TQ4P (C)` mode
  in the module docstring).
- **Still open**: on-GPU validation of the WHT CUDA kernels (the sandbox
  has no `nvcc`; algorithmic correctness is asserted only indirectly via
  the CPU tests). `scripts/smoke_test_tq4p.sh` is the fastest way to
  shake out the CUDA build on a real 4090/5090.

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) — Zandieh et al., ICLR 2026
- Paper reference Python: [`turboquant.py`](../../turboquant.py), [`lloyd_max.py`](../../lloyd_max.py)
- Target ggml tree: ollama's vendored `ml/backend/ggml/ggml/`, or any
  compatible llama.cpp ggml tree (hooks auto-adapt to the `GGML_TYPE_COUNT`
  value in use).
