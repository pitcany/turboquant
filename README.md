# TurboQuant

A from-scratch implementation of [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026), Google's two-stage vector quantization algorithm for compressing LLM key-value caches — integrated natively into Ollama via patched ggml.

## What It Does

TurboQuant compresses the KV cache to ~3 bits per element (down from 16-bit fp16), enabling **5x longer context windows** on the same GPU hardware. Your existing GGUF models work unchanged — TurboQuant only rewrites the runtime KV cache, it does not touch weights on disk.

## Architecture

### Core Algorithm (`turboquant.py`)

Two-stage compression applied to each KV vector:

1. **Stage 1 — PolarQuant** (2-bit): Random orthogonal rotation + per-coordinate Lloyd-Max quantization. The rotation makes coordinates follow a predictable distribution, enabling optimal scalar quantization.

2. **Stage 2 — QJL** (1-bit): Quantized Johnson-Lindenstrauss projection on the residual. Stores only the sign of each projection, making inner product estimates mathematically unbiased.

Combined: 3 bits per element, ~4x memory reduction, with near-zero impact on attention accuracy.

### Native Ollama Integration (`patches/stage2-qjl/`)

TurboQuant is integrated directly into Ollama's vendored ggml layer as two new quantization types: `tq4p_d128` (head_dim 128) and `tq4p_d256` (head_dim 256, for Qwen 3.5). The integration is additive — no upstream files are replaced, only extended.

The C implementation in `patches/stage2-qjl/c/` is paper-faithful and byte-exact against the Python reference. See [patches/stage2-qjl/PLAN.md](patches/stage2-qjl/PLAN.md) for the algorithm details and [docs/OLLAMA_NATIVE.md](docs/OLLAMA_NATIVE.md) for the full workflow.

### Compressed Byte Layout

For `head_dim=128` with 2-bit MSE keys + 1-bit QJL + 3-bit MSE values:

```
[0..31]    key MSE indices   (128x2 bit = 32 bytes)
[32..47]   key QJL signs     (128x1 bit = 16 bytes)
[48..49]   key residual norm (float16)
[50..51]   key original norm (float16)
[52..115]  val MSE indices   (128x4 bit = 64 bytes)
[116..117] val original norm (float16)
───────    118 bytes total per token per KV head
```

vs 512 bytes for fp16 (128 dims x 2 bytes x 2 for K+V) = **4.3x compression**.

## Quick Start

### Prerequisites

- Linux (tested on Ubuntu 22.04+); macOS untested
- Go 1.22+, CMake 3.21+, GCC 11+
- For GPU: CUDA 12 toolkit (`nvcc`), an NVIDIA GPU with compute capability ≥ 7.5
  - CUDA arch defaults to `89;120` (RTX 4090 + 5090); override with `CUDA_ARCHS=86` etc.

### Build

```bash
scripts/build_ollama_tq.sh
```

This clones ollama under `$HOME/.local/src/ollama-tq/`, patches its vendored ggml with the TQ4P sources, widens the Go KV-cache-type allowlist, and builds. CUDA is on by default (`CUDA=0` for CPU-only).

### Run

```bash
# Stop system ollama if running
systemctl --user stop ollama 2>/dev/null || pkill -x ollama || true

# Llama 3.x, Qwen 2.5, Qwen 3 — head_dim 128
OLLAMA_KV_CACHE_TYPE=tq4p_d128 OLLAMA_FLASH_ATTENTION=1 \
    ~/.local/src/ollama-tq/ollama/ollama serve

# In another shell
ollama run qwen2.5-coder:32b
```

For Qwen 3.5 (head_dim 256), use `OLLAMA_KV_CACHE_TYPE=tq4p_d256`.

### Picking a Rotation: WHT vs Haar

TurboQuant's Stage-1 random orthogonal rotation can be either a Randomized Hadamard Transform (WHT, default) or a dense Haar matrix (paper-faithful). Both have identical distortion bounds; WHT is `O(d log d)` while Haar is `O(d²)`.

```bash
# Default: fast WHT (Randomized Hadamard Transform)
OLLAMA_TQP_ROTATION=wht  ollama serve

# Paper-exact dense Haar rotation (~10x slower per block)
OLLAMA_TQP_ROTATION=haar ollama serve
```

Resolution precedence (highest first): per-call explicit (`layer_byte` bit 6) > per-thread (`tqp_set_thread_rotation` C API) > `OLLAMA_TQP_ROTATION` env > compile-time WHT.

### Inspecting a Quantized File

After ollama writes a TQ4P-quantized GGUF, you can verify the metadata and per-block rotation distribution:

```bash
python3 scripts/tq4p_inspect.py ~/.ollama/models/blobs/sha256-...
```

Reports the `tq4p.default_rotation` GGUF KV and a histogram of bit-7 (rotation) over the first TQ4P-quantized tensor.

### Iterate

```bash
# Edit patches/stage2-qjl/c/*.{c,h}, then rebuild
scripts/build_ollama_tq.sh --rebuild
```

See [docs/OLLAMA_NATIVE.md](docs/OLLAMA_NATIVE.md) for the full workflow, rollback instructions, and troubleshooting.

## Benchmarks

Measured on an RTX 4090, median of 3 runs per configuration, ~2K-token prompt for prefill + 128-token decode. Produced by [`scripts/bench_prefill_tq4p.sh`](scripts/bench_prefill_tq4p.sh).

| Model | Prefill overhead vs f16 | Decode overhead | Decode tok/s |
|---|---:|---:|---:|
| qwen2.5:3b      | +13.6% | **-29%** | 283 |
| llama3.1:8b     |  +9.4% | **-22%** | 195 |
| llama3.3:70b    |  +1.4% |  **-9%** |  26 |

**KV cache**: 16 bpw → 4.25 bpw (3.76x compression) for `tq4p_d128`; ~3.84x for `tq4p_d256`.

**Key finding**: TQ4P overhead shrinks with model size — it's essentially free on 70B and decode is actually *faster* than f16 across all tested sizes. On large models decode is memory-bandwidth-bound, so moving 4.25 bpw through the bus beats moving 16 bpw by more than the extra quantize math costs.

Reproduce on your own hardware:

```bash
scripts/bench_prefill_tq4p.sh                       # auto-picks smallest model
scripts/bench_prefill_tq4p.sh --model llama3.1:8b   # specific model
scripts/bench_prefill_tq4p.sh --runs 5              # more samples per config
```

The benchmark prepends a unique per-run tag to defeat ollama's prefix-KV cache and uses the median of N runs to filter cold-start outliers.

## Validation

### Synthetic Tests (`tests/test_core.py`, `tests/test_stability.py`)

Validates the core algorithm against theoretical bounds on random unit vectors:

| Bits | Measured MSE | Paper's Upper Bound | Inner Product Correlation |
|------|-------------|---------------------|--------------------------|
| 2-bit | 0.116 | 0.170 | 0.80 |
| 3-bit | 0.034 | 0.043 | 0.93 |
| 4-bit | 0.009 | 0.011 | 0.98 |

```bash
python3 -m pytest tests/ -v
```

### C Library Validation (`patches/stage2-qjl/c/`)

Byte-exact cross-check of the C implementation against `turboquant.py`:

```bash
cd patches/stage2-qjl/c && make test
```

### Real Model Validation (`validate.py`)

Tests TQ4P C-library attention fidelity on actual KV cache data:

```bash
python validate.py          # Needs CUDA GPU
```

## Project Structure

```
turboquant/
├── turboquant.py              # Core: TurboQuantMSE, TurboQuantProd, TurboQuantKVCache
├── lloyd_max.py               # Lloyd-Max optimal scalar quantizer
├── compressors.py             # Production compressors for real model tensors
├── ollama_resolver.py         # Ollama GGUF resolver + TQ metadata export
├── validate.py                # C-library attention fidelity validation
├── patches/stage2-qjl/
│   ├── c/                     # TQ4P C implementation (ggml integration)
│   ├── cuda/                  # CUDA kernel implementation (in progress)
│   ├── python/                # Python reference + cross-check tests
│   ├── PLAN.md                # Algorithm design and validation strategy
│   ├── BYTE_LAYOUT.md         # Wire format specification
│   └── apply_hooks.sh         # Idempotent ggml patching
├── scripts/
│   ├── build_ollama_tq.sh     # One-step ollama build with TQ4P
│   ├── patch_ollama_kv_types.sh  # Go KV-type allowlist patch
│   └── smoke_test_tq4p.sh     # End-to-end smoke test
├── docs/
│   └── OLLAMA_NATIVE.md       # Full workflow, rollback, troubleshooting
├── tests/
│   ├── test_core.py           # Core algorithm unit tests
│   ├── test_ollama_resolver.py # Ollama resolver tests
│   └── test_stability.py     # Numerical stability tests
├── autoresearch/              # Automated research experiment scripts
└── requirements.txt           # Python dependencies
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- scipy (codebook computation)
- Go toolchain (for building ollama)
- CUDA toolkit + nvcc (for GPU support)

```bash
pip install -r requirements.txt
```

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- [QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead](https://arxiv.org/abs/2406.03482)
- [PolarQuant: Quantizing KV Caches with Polar Transformation](https://arxiv.org/abs/2502.02617)

## License

MIT
