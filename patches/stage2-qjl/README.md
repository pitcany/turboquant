# TQ4P — paper-faithful TurboQuant for ollama's ggml

Paper-faithful TurboQuant (ICLR 2026, [arXiv 2504.19874](https://arxiv.org/abs/2504.19874))
as two new ggml block quantization types — additive to whichever ggml tree
you drop it into (tested against ollama's vendored `ml/backend/ggml/ggml/`).
Nothing existing is modified in place.

| Type | head_dim | Covers | Block size | bpw |
|---|---|---|---|---|
| `GGML_TYPE_TQ4P_D128` | 128 | Llama 3.x, Qwen 2.5, Qwen 3 | 68 B / 128 | 4.25 |
| `GGML_TYPE_TQ4P_D256` | 256 | Qwen 3.5 gated attention | 132 B / 256 | 4.13 |

Algorithm = Haar random orthogonal rotation Π + 3-bit Lloyd-Max per coord
+ 1-bit Gaussian QJL on residual. Unbiased inner-product estimator with
`√(π/2)/d` correction.

## Validation status

The patch is **byte-exactly** verified against `turboquant.py`
(the repo's paper reference, itself verified to reproduce paper numbers):

| Check | Result |
|---|---|
| `turboquant.py` measured MSE vs. paper (3-bit, d=128) | 0.0340 vs. 0.034 ✓ |
| `turboquant.py` measured IP corr vs. paper (3-bit, d=128) | 0.9207 vs. 0.93 ✓ |
| C `quantize_row_*` output byte-identical to Python reference | ✓ (28/28 tests) |
| C indices + QJL signs byte-identical to `turboquant.py` | ✓ |
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
3. Runs [`apply_hooks.sh`](apply_hooks.sh) — 4 additive edits:
   new enum values, 2 dispatch table entries, `CMakeLists.txt` source
   addition. All idempotent via a `tq4p` marker.
4. Widens ollama's Go KV-cache-type allowlist to accept `tq4p_d128`
   and `tq4p_d256`.
5. Runs `go generate && go build` on ollama.

See [`hooks.md`](hooks.md) for the exact edits, in case you prefer to
apply them by hand or need to adjust for an ollama layout change.

## Layout

```
patches/stage2-qjl/
├── PLAN.md                            # design doc (earlier commit)
├── README.md                          # this file
├── BYTE_LAYOUT.md                     # the 68 B / 132 B block specs
├── hooks.md                           # the 4 fork edits you apply by hand
├── python/
│   ├── generate_constants.py          # seed → .h + .pt
│   ├── tq_paper_reference.py          # byte-level Python mirror
│   ├── test_tq_paper.py               # Python-vs-turboquant.py byte-exact
│   └── test_c_vs_python.py            # C-vs-Python byte-exact (via ctypes)
├── c/
│   ├── ggml-tq-paper.h                # public API
│   ├── ggml-tq-paper.c                # CPU quant/dequant/prepare_query/vec_dot
│   ├── tqp_centroids_d{128,256}.h     # Lloyd-Max codebook (generated)
│   └── tqp_constants_d{128,256}.h     # Π + S matrices (generated)
└── cuda/
    └── PLAN.md                        # follow-up kernel design
```

`c/tqp_constants.pt` is a torch state dict used by the Python reference to
load the same bits the C headers contain. `c/libggml_tq_paper.so` and
`__pycache__/` are build artifacts, gitignored.

## Scope & non-goals

- **CPU reference only** — CUDA kernels follow in a separate commit, per
  [`cuda/PLAN.md`](cuda/PLAN.md). The CPU path is 5-10× slower than the
  fork's existing `TQ3_0` vec_dot; it's intended for validation and
  long-tail code paths, not primary inference.
- **No ollama allowlist edit** for `OLLAMA_KV_CACHE_TYPE=tq4p_d128`. One-line
  follow-up once the type works in llama.cpp; see
  [`../../scripts/patch_ollama_kv_types.sh`](../../scripts/patch_ollama_kv_types.sh)
  for the pattern.
- **No `validate.py` extension** for attention-score cosine sim. Follow-up
  once you have a compiled `ollama` to run it against.
- **Global Π and S, not per-layer.** Paper allows per-layer seeds; we share
  one matrix per head dim across all layers for simplicity. Swap
  `__constant__` for `__device__` + model-load init as a future refinement.

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) — Zandieh et al., ICLR 2026
- Paper reference Python: [`turboquant.py`](../../turboquant.py), [`lloyd_max.py`](../../lloyd_max.py)
- Target ggml tree: ollama's vendored `ml/backend/ggml/ggml/`, or any
  compatible llama.cpp ggml tree (hooks auto-adapt to the `GGML_TYPE_COUNT`
  value in use).
