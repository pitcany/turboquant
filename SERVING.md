# Serving Models with TurboQuant on vLLM

## Prerequisites

- vLLM 0.17.1 installed in the `vllm-serve` conda environment
- TurboQuant plugin installed in editable mode:
  ```bash
  cd ~/ai/turboquant
  ~/miniconda3/envs/vllm-serve/bin/pip install -e .
  ```
- NVIDIA GPUs (tested with RTX 4090 + RTX 5090)

## Quick Start

### Harbor Ollama GGUF Models

Use this path when models are pulled through Harbor's Ollama service:

```bash
cd /home/yannik/AI/turboquant

# Pull or update the GGUF-backed Ollama model through Harbor
harbor ollama pull qwen2.5-coder:32b

# Serve the Harbor-managed Ollama model with vLLM + TurboQuant
./serve_ollama_tq.sh qwen2.5-coder:32b
```

`serve_ollama_tq.sh` does not patch Harbor's vLLM Docker image. It uses Harbor
for the Ollama model cache, then launches vLLM directly from the local
`vllm-serve` conda environment.

Model storage resolution order:

1. `OLLAMA_MODELS`
2. `HARBOR_OLLAMA_CACHE/models`
3. `$HARBOR_HOME/.env` -> `HARBOR_OLLAMA_CACHE/models`
4. `~/.ollama/models`

Harbor defaults to `HARBOR_OLLAMA_CACHE="~/.ollama"`, so host Ollama and
Harbor Ollama may share the same physical store. Custom Harbor caches are also
supported:

```bash
harbor config set ollama.cache /path/to/custom/ollama-cache
```

Useful Harbor model commands:

```bash
harbor ollama pull qwen2.5-coder:32b
harbor ollama list
harbor ollama rm qwen2.5-coder:32b
```

Common variants:

```bash
# Llama 3.3 over both GPUs
./serve_ollama_tq.sh llama3.3 --tp 2

# Hybrid TurboQuant compressed storage + SDPA compute
./serve_ollama_tq.sh qwen2.5-coder:32b --hybrid

# Override port or context length
PORT=8005 ./serve_ollama_tq.sh qwen2.5-coder:32b --max-len 16384
```

If Harbor's Ollama service is using GPU memory, stop it before starting vLLM:

```bash
harbor down ollama
```

Verify the running vLLM endpoint from another terminal:

```bash
curl -s http://127.0.0.1:8003/v1/models | python3 -m json.tool

curl -s http://127.0.0.1:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder:32b-turboquant",
    "messages": [{"role": "user", "content": "Write a Python function to reverse a linked list."}],
    "max_tokens": 200,
    "temperature": 0
  }' | python3 -m json.tool
```

Inspect the resolved GGUF path and generated TurboQuant settings without
starting vLLM:

```bash
/home/yannik/miniconda3/envs/vllm-serve/bin/python ollama_resolver.py qwen2.5-coder:32b
```

### Manual Start: Llama 3.3 70B AWQ (dual GPU, TP=2, 32K context)

```bash
LD_LIBRARY_PATH=~/miniconda3/envs/vllm-serve/lib:$LD_LIBRARY_PATH \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=0,1 \
NCCL_P2P_DISABLE=1 \
TQ_USE_TRITON=1 \
TQ_NUM_KV_SPLITS=8 \
~/miniconda3/envs/vllm-serve/bin/python -m vllm.entrypoints.openai.api_server \
  --model casperhansen/llama-3.3-70b-instruct-awq \
  --served-model-name llama-3.3-70b-turboquant \
  --port 8003 \
  --quantization awq_marlin \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --attention-backend CUSTOM \
  --disable-custom-all-reduce \
  --enforce-eager \
  --trust-remote-code
```

Without TurboQuant, max context is ~8K. With TurboQuant compression (4.3x), this extends to 32K.

### Single GPU: Qwen 2.5 3B

```bash
LD_LIBRARY_PATH=~/miniconda3/envs/vllm-serve/lib:$LD_LIBRARY_PATH \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
TQ_USE_TRITON=1 \
~/miniconda3/envs/vllm-serve/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --port 8003 \
  --attention-backend CUSTOM \
  --enforce-eager
```

## Decode Path: PyTorch vs Triton

TurboQuant has two decode backends controlled by `TQ_USE_TRITON`:

| `TQ_USE_TRITON` | Decode path | Performance | Notes |
|------------------|-------------|-------------|-------|
| `0` (default)    | PyTorch einsum | ~7.6 tok/s | Always works, no Triton dependency |
| `1`              | Triton split-KV kernels | ~9.0 tok/s | **Recommended.** +19% throughput, tighter latency |

Prefill (q_len > 1) always uses the PyTorch path regardless of this setting.

### Validated Benchmark (2025-03-29)

Llama 3.3 70B AWQ, TP=2, RTX 4090 + RTX 5090, max_tokens=128:

| Metric | PyTorch (TQ_USE_TRITON=0) | Triton (TQ_USE_TRITON=1) |
|--------|---------------------------|--------------------------|
| Mean tok/s | 7.58 | **9.00** |
| Peak tok/s | 7.95 | **9.07** |
| Mean latency | 16.66s | **14.23s** |
| Latency spread | 7.28–7.95 | **8.90–9.07** |

## Environment Variables

These must be set **before** the server command:

| Variable | Required | Purpose |
|----------|----------|---------|
| `LD_LIBRARY_PATH=~/miniconda3/envs/vllm-serve/lib:$LD_LIBRARY_PATH` | Yes | Fixes `CXXABI_1.3.15` missing symbol (conda libstdc++ vs system) |
| `CUDA_DEVICE_ORDER=PCI_BUS_ID` | Yes (multi-GPU) | Consistent GPU ordering when mixing architectures |
| `NCCL_P2P_DISABLE=1` | Yes (multi-GPU) | Required for RTX 4090 + 5090 (different architectures lack P2P) |

## vLLM Flags

| Flag | Purpose |
|------|---------|
| `--attention-backend CUSTOM` | Routes attention through TurboQuant (registered as the CUSTOM backend) |
| `--quantization awq_marlin` | AWQ weight quantization with Marlin kernels (faster than plain AWQ) |
| `--tensor-parallel-size 2` | Split model across both GPUs |
| `--disable-custom-all-reduce` | Required for mixed-architecture GPUs (no P2P support) |
| `--enforce-eager` | Disables CUDA graphs and torch.compile (needed since TurboQuant uses custom Python attention) |
| `--max-model-len N` | Max context length (limited by VRAM) |
| `--gpu-memory-utilization N` | Fraction of GPU memory for model + KV cache (default 0.92) |

## TurboQuant Configuration

TurboQuant parameters are set via `TQ_*` environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TQ_USE_TRITON` | `0` | `1` to use Triton decode kernels (recommended) |
| `TQ_NUM_KV_SPLITS` | `8` | Number of KV sequence splits for Triton decode |
| `TQ_B_MSE` | `2` | Bits per coordinate for PolarQuant MSE stage |
| `TQ_B_QJL` | `1` | Bits per coordinate for QJL residual correction |
| `TQ_NUM_LAYERS` | `32` | Auto-detected from model |
| `TQ_NUM_HEADS` | `32` | Auto-detected from model |
| `TQ_NUM_KV_HEADS` | `32` | Auto-detected from model |
| `TQ_HEAD_DIM` | `128` | Auto-detected from model |
| `TQ_GGUF_PATH` | unset | Optional GGUF path used to auto-populate model shape defaults |

The defaults (2-bit MSE + 1-bit QJL = 3 bits total) provide ~4.3x KV cache compression.

## Helper Script Configuration

`serve_ollama_tq.sh` accepts Harbor/Ollama model names and resolves them to
GGUF blobs:

```bash
./serve_ollama_tq.sh qwen2.5-coder:32b
./serve_ollama_tq.sh llama3.3 --tp 2
./serve_ollama_tq.sh qwen2.5-coder:32b --hybrid
```

Useful overrides:

```bash
HARBOR_OLLAMA_CACHE=/path/to/harbor/ollama-cache ./serve_ollama_tq.sh qwen2.5-coder:32b
CUDA_VISIBLE_DEVICES_VALUE=0 ./serve_ollama_tq.sh qwen2.5-coder:32b
SERVED_MODEL_NAME=qwen-coder-tq ./serve_ollama_tq.sh qwen2.5-coder:32b
```

## Troubleshooting

### GPU memory: "Free memory on device cuda:0 ... less than desired"
Something else is using GPU memory. Check with:
```bash
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv
```
Kill the offending process, or lower `GPU_MEMORY_UTILIZATION`.

### Worker killed silently during first inference
The model + KV cache + temporary attention tensors exceed GPU memory. The 70B AWQ model uses ~18.5 GiB per GPU. With `--gpu-memory-utilization 0.92` on a 24 GB card, there's minimal headroom. Solutions:
- Ensure no other processes are using the GPU
- Lower `--gpu-memory-utilization` to 0.88 (reduces KV cache but more stable)
- Reduce `--max-model-len`

### `CXXABI_1.3.15 not found`
The conda environment has the right libstdc++ but the system one is loaded first.
Fix: prefix with `LD_LIBRARY_PATH=~/miniconda3/envs/vllm-serve/lib:$LD_LIBRARY_PATH`.

### NCCL hangs with TP=2
Mixed GPU architectures (e.g., 4090 Ada + 5090 Blackwell) need `NCCL_P2P_DISABLE=1`
and `--disable-custom-all-reduce`. Without these, NCCL collective operations hang.

### `Unknown attention backend: 'TURBOQUANT'`
Use `--attention-backend CUSTOM`, not `--attention-backend turboquant`.
The plugin registers itself under the `CUSTOM` enum slot.

### Ollama holding GPU memory
Stop ollama before serving: `sudo systemctl disable --now ollama.service`
Or: `sudo kill -9 $(pgrep -f 'ollama serve')`

## Testing the Server

```bash
# Check model is loaded
curl -s http://127.0.0.1:8003/v1/models | python3 -m json.tool

# Chat completion
curl -s http://127.0.0.1:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.3-70b-turboquant",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }' | python3 -m json.tool
```

## OpenClaw Integration

The model is configured in `~/.openclaw/openclaw.json` under `models.providers.vllm-turboquant`:

```json
{
  "vllm-turboquant": {
    "baseUrl": "http://127.0.0.1:8003/v1",
    "apiKey": "vllm-dummy-key",
    "api": "openai-completions",
    "models": [
      {
        "id": "llama-3.3-70b-turboquant",
        "name": "llama-3.3-70b-turboquant"
      }
    ]
  }
}
```

Reference in openclaw as: `vllm-turboquant/llama-3.3-70b-turboquant`

## Hardware Tested

- **GPU 0**: NVIDIA GeForce RTX 4090 (24 GB, Ada Lovelace, compute 8.9)
- **GPU 1**: NVIDIA GeForce RTX 5090 (32 GB, Blackwell, compute 12.0)
- **NCCL**: 2.27.5 (P2P disabled, shared memory transport)
- **Driver**: 575.57.08, CUDA 12.9
