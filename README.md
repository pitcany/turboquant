# TurboQuant

A from-scratch PyTorch implementation of [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026), Google's two-stage vector quantization algorithm for compressing LLM key-value caches — plus a vLLM plugin that makes it work in production serving.

## What It Does

TurboQuant compresses the KV cache to ~3 bits per element (down from 16-bit fp16), enabling **5x longer context windows** on the same GPU hardware. On a dual-GPU setup (RTX 4090 + RTX 5090), this means running Llama 3.3 70B at 32K context instead of 12K, or Qwen 2.5 72B at 16K instead of 6K.

| Model | Without TurboQuant | With TurboQuant |
|-------|-------------------|-----------------|
| Llama 3.3 70B (AWQ) | ~12K context | **32K context** |
| Qwen 2.5 72B (AWQ) | ~6K context | **16K context** |

## Architecture

### Core Algorithm (`turboquant.py`)

Two-stage compression applied to each KV vector:

1. **Stage 1 — PolarQuant** (2-bit): Random orthogonal rotation + per-coordinate Lloyd-Max quantization. The rotation makes coordinates follow a predictable distribution, enabling optimal scalar quantization.

2. **Stage 2 — QJL** (1-bit): Quantized Johnson-Lindenstrauss projection on the residual. Stores only the sign of each projection, making inner product estimates mathematically unbiased.

Combined: 3 bits per element, ~4x memory reduction, with near-zero impact on attention accuracy.

### vLLM Plugin (`vllm_plugin/`)

Registers as a vLLM general plugin via the `vllm.general_plugins` entry point. Provides two attention backends:

- **Pure TurboQuant** (`attention.py`): Computes attention scores directly on compressed data using the asymmetric estimator — keys are never fully dequantized for scoring. Uses custom Triton kernels for decode with split-KV parallelism.

- **Hybrid TQ+SDPA** (`attention_hybrid.py`): Same compressed KV storage, but dequantizes on-the-fly and passes to `torch.nn.functional.scaled_dot_product_attention` (which dispatches to FlashAttention-style fused kernels). Trades the asymmetric estimator for optimized attention compute.

Both backends share the same compressed byte layout and storage path — the only difference is how attention scores are computed at decode time.

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

## Usage

### Harbor Ollama GGUF Models

For GGUF models managed by Harbor's Ollama service, pull the model with Harbor
and serve it directly with vLLM + TurboQuant:

```bash
cd /home/yannik/AI/turboquant

harbor ollama pull qwen2.5-coder:32b
./serve_ollama_tq.sh qwen2.5-coder:32b
```

This path uses Harbor for model management and cache location, but does not
patch Harbor's vLLM Docker image. The launcher resolves the GGUF blob from
Harbor's Ollama cache, reads GGUF metadata, exports the matching `TQ_*`
settings, and starts vLLM on `http://127.0.0.1:8003/v1`.

The resolver checks model storage in this order:

1. `OLLAMA_MODELS`
2. `HARBOR_OLLAMA_CACHE/models`
3. `$HARBOR_HOME/.env` -> `HARBOR_OLLAMA_CACHE/models`
4. `~/.ollama/models`

Examples:

```bash
# Single GPU, default port 8003
./serve_ollama_tq.sh qwen2.5-coder:32b

# Tensor parallel over both GPUs
./serve_ollama_tq.sh llama3.3 --tp 2

# Hybrid TurboQuant storage + SDPA compute
./serve_ollama_tq.sh qwen2.5-coder:32b --hybrid
```

If Harbor's Ollama container is holding GPU memory, stop it before starting
vLLM:

```bash
harbor down ollama
```

Then test the OpenAI-compatible endpoint:

```bash
curl -s http://127.0.0.1:8003/v1/models | python3 -m json.tool
```

### Serving with vLLM

Install the plugin into your vLLM environment:

```bash
pip install -e /path/to/turboquant
```

The editable install is required for the flat-module layout used by
`turboquant.py`, `lloyd_max.py`, and `compressors.py`. The vLLM plugin now
assumes those modules are available through the environment rather than
manually rewriting `sys.path`.

Start vLLM with the TurboQuant attention backend:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model casperhansen/llama-3.3-70b-instruct-awq \
    --quantization awq_marlin \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --attention-backend CUSTOM \
    --enforce-eager \
    --served-model-name llama-3.3-70b-turboquant
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TQ_USE_TRITON` | `0` | Enable Triton kernels for decode |
| `TQ_NUM_KV_SPLITS` | `8` | Split-KV parallelism for decode |
| `TQ_HYBRID` | `0` | Use hybrid TQ+SDPA backend instead of pure TQ |
| `TQ_B_MSE` | `2` | MSE quantization bits (Stage 1) |
| `TQ_B_QJL` | `1` | QJL bits (Stage 2) |

### systemd / gpu-models Integration

For persistent serving via systemd, see [local-llm-stack](https://github.com/pitcany/local-llm-stack) which provides `gpu-models.sh` — a GPU model backend switcher with TurboQuant presets:

```bash
gpu-models vllm llama-3.3-70b-tq       # Pure TQ, Llama 70B, 32K ctx
gpu-models vllm llama-3.3-70b-hybrid   # Hybrid TQ+SDPA, 32K ctx
gpu-models vllm qwen2.5-72b-tq         # Pure TQ, Qwen 72B, 16K ctx
gpu-models vllm qwen2.5-72b-hybrid     # Hybrid TQ+SDPA, 16K ctx
```

The systemd service auto-installs the plugin via `ExecStartPre` and manages `TQ_*` env vars.

## Benchmarking

### Inference Speed Comparison

`benchmark_tq_comparison.py` cycles through model configs (standard, TQ, hybrid) with controlled variables, restarting vLLM between each. Measures tokens/sec across short, medium, and long prompts.

```bash
python benchmark_tq_comparison.py                        # Run all configs
python benchmark_tq_comparison.py --configs llama70b-turboquant-32k llama70b-hybrid-32k  # Specific configs
python benchmark_tq_comparison.py --requests 5 --max-tokens 512   # More samples
```

Results are saved to `benchmark_tq_results.tsv`.

### Other Benchmarks

- `benchmark_openai.py` — Benchmark a running vLLM server via the OpenAI-compatible API
- `benchmark_decode.py` — Low-level decode kernel benchmarks

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

### Real Model Validation (`validate.py`)

Tests on actual KV cache data from Qwen2.5-3B-Instruct:

| Config | Compression | Cosine Sim (attention scores) | Top-1 Match |
|--------|-------------|------------------------------|-------------|
| TQ 4-bit | 3.8x | 0.9986 | 88% |
| TQ 3-bit | 5.0x | 0.9954 | 82% |
| TQ 2-bit | 7.3x | 0.9875 | 66% |

3-bit is the practical sweet spot: 5x compression with 99.5% attention fidelity.

```bash
python validate.py          # Needs CUDA GPU, downloads Qwen2.5-3B (~2GB)
python validate_vllm.py     # Validates plugin integration with vLLM
```

## Project Structure

```
turboquant/
├── turboquant.py              # Core: TurboQuantMSE, TurboQuantProd, TurboQuantKVCache
├── lloyd_max.py               # Lloyd-Max optimal scalar quantizer
├── compressors.py             # Production compressors for real model tensors
├── ollama_resolver.py          # Harbor/Ollama GGUF resolver + TQ metadata export
├── serve_ollama_tq.sh          # Harbor Ollama GGUF launcher for vLLM + TurboQuant
├── vllm_plugin/
│   ├── platform.py            # vLLM general plugin registration
│   ├── attention.py           # Pure TQ attention backend (Triton decode)
│   ├── attention_hybrid.py    # Hybrid TQ storage + SDPA compute backend
│   ├── triton_kernels.py      # Triton + torch decode kernels (split-KV)
│   ├── triton_wrapper.py      # Decode dispatch wrapper
│   ├── config.py              # TurboQuant configuration
│   └── kv_spec.py             # Compressed KV cache spec for vLLM
├── benchmark_tq_comparison.py # Controlled A/B benchmark (standard vs TQ vs hybrid)
├── benchmark_openai.py        # OpenAI API benchmark client
├── benchmark_decode.py        # Low-level decode kernel benchmark
├── validate.py                # Real model attention validation
├── validate_vllm.py           # vLLM plugin integration validation
├── autoresearch/              # Automated research experiment scripts
├── setup.py                   # Editable install with vllm.general_plugins entry point
└── requirements.txt           # Python dependencies
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- scipy (codebook computation)
- pytest (test runner)
- vLLM 0.6+ (for serving)
- triton (optional, for optimized decode kernels)

```bash
pip install -r requirements.txt
pip install -e .  # Install plugin for vLLM
```

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- [QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead](https://arxiv.org/abs/2406.03482)
- [PolarQuant: Quantizing KV Caches with Polar Transformation](https://arxiv.org/abs/2502.02617)

## License

MIT
