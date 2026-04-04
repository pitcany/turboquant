# TurboQuant + Harbor vLLM

Run TurboQuant KV cache compression inside [Harbor](https://github.com/av/harbor)'s vLLM service. This gives you ~4.3x KV cache compression, extending context from ~8K to 32K on the same hardware.

## Quick Start

```bash
# Enable TurboQuant (backs up originals, patches Dockerfile, rebuilds image)
bash harbor-apply.sh

# Start the server
harbor up vllm --detach

# Verify it's running
harbor logs vllm
curl -s http://127.0.0.1:33911/v1/models | python3 -m json.tool
```

## Removing TurboQuant

```bash
# Revert to stock Harbor vLLM (restores originals, rebuilds image)
bash harbor-revert.sh

# Restart
harbor up vllm --detach
```

## What the Scripts Do

### `harbor-apply.sh`

1. Backs up `~/.harbor/services/vllm/Dockerfile` and `override.env` (first run only)
2. Copies TurboQuant source into the Docker build context
3. Patches the Dockerfile to `pip install` TurboQuant
4. Sets TurboQuant environment variables (`TQ_USE_TRITON=1`, `TQ_NUM_KV_SPLITS=8`, etc.)
5. Configures Harbor: `--attention-backend CUSTOM`, `--enforce-eager`, AWQ quantization, TP=2, 32K context
6. Rebuilds the vLLM Docker image

### `harbor-revert.sh`

1. Restores the original Dockerfile and override.env from backups
2. Removes the TurboQuant source copy from the build context
3. Resets Harbor config to stock defaults (`FLASH_ATTN`, 8K context)
4. Rebuilds the vLLM Docker image

## Managing Models

```bash
# Set a model (Hugging Face model ID)
harbor vllm model casperhansen/llama-3.3-70b-instruct-awq

# Remove current model
harbor vllm model rm

# Restart after changing model
harbor down vllm && harbor up vllm --detach
```

For TurboQuant, use **AWQ-quantized** models (since the config uses `--quantization awq_marlin`):

| Model | Hugging Face ID | TP |
|---|---|---|
| Llama 3.3 70B | `casperhansen/llama-3.3-70b-instruct-awq` | 2 |
| Qwen 2.5 72B | `Qwen/Qwen2.5-72B-Instruct-AWQ` | 2 |
| Llama 3.1 8B | `casperhansen/llama-3.1-8b-instruct-awq` | 1 |
| Mistral 7B | `TheBloke/Mistral-7B-Instruct-v0.2-AWQ` | 1 |

For non-AWQ (FP16) models, update the vLLM args to drop `--quantization`:

```bash
harbor vllm model Qwen/Qwen2.5-3B-Instruct
harbor vllm args '--dtype float16 --tensor-parallel-size 1 --max-model-len 16384 --gpu-memory-utilization 0.92 --attention-backend CUSTOM --enforce-eager --trust-remote-code'
```

## Useful Commands

```bash
harbor up vllm --detach    # start in background
harbor down vllm           # stop
harbor logs vllm           # view logs
harbor ps                  # list running services
harbor build vllm          # rebuild image (after code changes)
```

## Running Without TurboQuant

To temporarily disable TurboQuant without a full revert, switch the attention backend:

```bash
harbor config set vllm.attention_backend FLASH_ATTN
harbor down vllm && harbor up vllm --detach
```

To re-enable:

```bash
harbor config set vllm.attention_backend CUSTOM
harbor down vllm && harbor up vllm --detach
```

## Updating TurboQuant

After changing TurboQuant source code, re-copy and rebuild:

```bash
bash harbor-apply.sh
harbor down vllm && harbor up vllm --detach
```

## Compatibility

TurboQuant works with standard decoder-only transformer models with power-of-2 `head_dim` (128 is typical). It does **not** support:

- Sliding window attention models (some Mistral/Gemma variants)
- Encoder-decoder models (T5, BART)
- Non-power-of-2 head dimensions

There is no automatic fallback. Incompatible models will fail at startup. For those models, use `FLASH_ATTN` instead of `CUSTOM`.

## Hardware

Tested with:

- **GPU 0**: NVIDIA GeForce RTX 4090 (24 GB)
- **GPU 1**: NVIDIA GeForce RTX 5090 (32 GB)
- Harbor vLLM port: **33911**
