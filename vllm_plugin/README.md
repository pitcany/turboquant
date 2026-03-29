# TurboQuant vLLM Plugin

3-bit KV cache compression for vLLM 0.17.1, combining **PolarQuant** (MSE-optimal scalar quantization) with **QJL** (1-bit Quantized Johnson-Lindenstrauss residual correction).

Achieves **4.3x KV cache memory reduction** with near-zero accuracy loss, enabling significantly longer context lengths on the same hardware.

> **Reference:** [TurboQuant paper (arxiv 2504.19874)](https://arxiv.org/abs/2504.19874)

---

## What It Does

TurboQuant compresses the transformer KV cache from **16 bits/channel (FP16) to ~3 bits/channel**. Keys use a 2-bit MSE quantizer + 1-bit QJL correction for unbiased attention scores. Values use a 3-bit MSE quantizer (errors average out under softmax).

| Stage | Key bits | Value bits | Purpose |
|-------|----------|------------|---------|
| PolarQuant (MSE) | 2 | 3 | Scalar quantization after random rotation |
| QJL | 1 | — | Unbiased residual correction for inner products |
| Norms | ~0.25 | ~0.12 | FP16 norms for denormalization |
| **Total** | **~3.25** | **~3.12** | **4.3x compression vs FP16** |

**Practical impact:** Llama 3.3 70B on RTX 4090 + RTX 5090 goes from 8K to 32K context.

---

## Installation

```bash
cd ~/ai/turboquant
~/miniconda3/envs/vllm-serve/bin/pip install -e .
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.0
- vLLM 0.17.1
- CUDA-capable GPU

---

## Usage

```bash
vllm serve <model> --attention-backend CUSTOM [--enforce-eager]
```

The plugin auto-registers via vLLM's `vllm.general_plugins` entry point. No manual imports needed.

See [SERVING.md](../SERVING.md) for complete examples with environment variables and multi-GPU flags.

### Configuration

All parameters via `TQ_*` environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TQ_B_MSE` | 2 | PolarQuant bits per coordinate |
| `TQ_B_QJL` | 1 | QJL bits per coordinate |

---

## Architecture

```
vllm_plugin/
├── __init__.py      # Package exports
├── platform.py      # Plugin registration + KV cache spec patching
├── kv_spec.py       # TurboQuantSpec (compressed page_size_bytes)
├── attention.py     # TurboQuantAttentionBackend + TurboQuantAttentionImpl
├── config.py        # TurboQuantConfig dataclass with env-var overrides
└── README.md        # This file

Parent modules (from turboquant-vllm root):
├── turboquant.py    # Core algorithm: TurboQuantProd, TurboQuantMSE
├── lloyd_max.py     # Lloyd-Max codebook solver
├── compressors.py   # Asymmetric inner product compressors
└── setup.py         # Entry point registration
```

### How It Integrates with vLLM 0.17.1

1. **Plugin registration** (`platform.py`): On startup, `register_turboquant()` is called via the `vllm.general_plugins` entry point. It:
   - Registers `TurboQuantAttentionBackend` as `AttentionBackendEnum.CUSTOM`
   - Monkey-patches `Attention.get_kv_cache_spec` to return `TurboQuantSpec` (compressed page sizes)
   - Registers `TurboQuantSpec` in the KV cache manager's spec-to-manager map

2. **Memory allocation** (`kv_spec.py`): `TurboQuantSpec` overrides `real_page_size_bytes` so vLLM's block allocator sizes each KV cache page at ~4.3x less memory than FP16. This is what enables longer context.

3. **Cache tensor layout** (`attention.py`): `get_kv_cache_shape()` returns `(num_blocks, block_size, num_kv_heads, 59)` where 59 FP16 elements = 118 bytes of bit-packed compressed data per token per head:
   - Key MSE indices: 32 bytes (128 coords x 2 bits, packed)
   - Key QJL signs: 16 bytes (128 coords x 1 bit, packed)
   - Key residual norm: 2 bytes (FP16)
   - Key original norm: 2 bytes (FP16)
   - Value MSE indices: 64 bytes (128 coords x 4 bits, nibble-packed)
   - Value norm: 2 bytes (FP16)

4. **Attention computation** (`attention.py`): `TurboQuantAttentionImpl.forward()`:
   - Compresses new K,V tokens and writes packed bytes to `kv_cache`
   - Gathers compressed data for each request via `block_table`
   - Unpacks and computes attention using the **asymmetric estimator** (keys are never fully dequantized)
   - Dequantizes values for the softmax-weighted sum

### Attention Flow

**Prefill** (multiple query tokens):
1. Compress all K,V → write to cache via `slot_mapping`
2. Gather all compressed tokens for the request
3. For each KV head: unpack, dequantize key MSE, compute asymmetric scores with causal mask, dequantize values, softmax-weighted sum
4. GQA: batch query heads sharing a KV head

**Decode** (single new token per request):
1. Compress new K,V → write to cache
2. Gather full history from cache
3. Compute attention (no causal mask needed for single query token)

### Asymmetric Score Estimator

For a query `q` and compressed key `k` (normalized to unit norm before compression):

```
score(q, k) = ||k|| * [<q, k_mse> + ||r|| * sqrt(pi/2)/m * <S@q, sign(S@r)>] / sqrt(d)
```

Where:
- `k_mse` = Lloyd-Max MSE reconstruction of the normalized key
- `r` = quantization residual (k_normalized - k_mse)
- `S` = random Gaussian projection matrix (QJL)
- `sign(S@r)` = 1-bit QJL signs stored in the cache
- `||k||` = original key norm stored alongside

This estimator is **mathematically unbiased** — the expected score equals the true inner product.

---

## Known Limitations

- **Pure PyTorch**: No Triton/CUDA kernels yet. Inference is slower than FlashAttention. Suitable for throughput-insensitive use cases where longer context matters more than speed.
- **`--enforce-eager` required**: CUDA graphs and `torch.compile` are not supported (custom Python attention loop).
- **Batch size**: Works with any batch size but per-request Python loops limit throughput.
- **No prefix caching**: Compressed cache doesn't support vLLM's prefix caching optimization.

---

## License

MIT
