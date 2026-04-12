# Native Ollama with TurboQuant

Run `ollama` with a TurboQuant KV cache — no vLLM, no Harbor, no Python
serving layer. This path rebuilds `ollama` against the
[`turbo-tan/llama.cpp-tq3`](https://github.com/turbo-tan/llama.cpp-tq3) fork
of llama.cpp, which adds the `tq3_0` KV cache type (3.5-bit TurboQuant
quantization, CUDA kernels) from Zandieh et al.'s [TurboQuant: Online Vector
Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)
(ICLR 2026).

This is an end-run around upstream: neither `ollama` nor `llama.cpp` has
merged TurboQuant as of April 2026. The build script swaps the vendored
llama.cpp and widens ollama's KV cache type allowlist so
`OLLAMA_KV_CACHE_TYPE=tq3_0` is accepted.

## One-time build

```bash
scripts/build_ollama_tq.sh
```

This clones, patches, and compiles everything under
`$HOME/.local/src/ollama-tq/` (override with `WORKDIR=...`):

```
~/.local/src/ollama-tq/
├── llama.cpp-tq3/        # editable fork clone
└── ollama/
    ├── llama/llama.cpp   -> ../../llama.cpp-tq3   (symlink)
    └── ollama            # patched binary
```

CUDA is on by default (`CUDA=0` for CPU-only). `nvcc` must be in `PATH`.

## Run

```bash
# Stop any existing ollama daemon first
systemctl --user stop ollama 2>/dev/null || pkill -x ollama || true

# Use the rebuilt binary with the TurboQuant cache type
OLLAMA_KV_CACHE_TYPE=tq3_0 \
OLLAMA_FLASH_ATTENTION=1 \
~/.local/src/ollama-tq/ollama/ollama serve

# In another shell
ollama run qwen2.5-coder:32b
```

Your existing pulled GGUFs under `~/.ollama/models` work unchanged —
TurboQuant only rewrites the runtime KV cache; it does not touch weights on
disk.

`OLLAMA_FLASH_ATTENTION=1` is recommended: the fork's TQ kernels are wired
into the flash-attention path.

## Making changes to the fork

The fork lives at `~/.local/src/ollama-tq/llama.cpp-tq3/` as a normal git
clone. Edit it however you like — add your own branch, modify kernels in
`ggml/src/ggml-cuda/`, tweak the `tq3_0` codebook — then rebuild:

```bash
scripts/build_ollama_tq.sh --rebuild
```

`--rebuild` skips the clone step and just re-runs `go generate` + `go build`
in the ollama tree. Because ollama's `llama/llama.cpp` is a symlink into your
fork clone, your edits are picked up automatically.

To pin a specific branch or commit of the fork:

```bash
LLAMACPP_REF=my-tq-experiments scripts/build_ollama_tq.sh
```

## Cross-checking against this repo's reference

`turboquant.py` and `lloyd_max.py` in the repo root are a PyTorch reference
implementation of the same algorithm. They're useful for validating fork
changes: dump a batch of KV vectors from the fork's C implementation,
round-trip them through `TurboQuantMSE` in Python, and compare MSE. See
`tests/test_core.py` for the MSE/correlation bounds the paper specifies:

| Bits | Expected MSE | Inner Product Corr |
|------|-------------|--------------------|
| 2-bit | ≤ 0.170 | ~0.80 |
| 3-bit | ≤ 0.043 | ~0.93 |
| 4-bit | ≤ 0.011 | ~0.98 |

## Rolling back

```bash
# Restore original vendored llama.cpp in the ollama tree
cd ~/.local/src/ollama-tq/ollama
rm llama/llama.cpp
mv llama/llama.cpp.orig llama/llama.cpp

# Or just delete the whole workdir and use distro ollama
rm -rf ~/.local/src/ollama-tq
```

The `patch_ollama_kv_types.sh` allowlist edit is marked with the comment
`// turboquant: tq3_0 allowlisted` in whichever Go file holds the validator,
so you can `git -C ~/.local/src/ollama-tq/ollama diff` to inspect it or
`git checkout` to revert.

## Troubleshooting

**`OLLAMA_KV_CACHE_TYPE=tq3_0` silently falls back to f16.** Check logs for
`unsupported KV cache type`. The patch script failed to find the allowlist —
run `bash scripts/patch_ollama_kv_types.sh ~/.local/src/ollama-tq/ollama` by
hand and follow the error output; the validator may have moved in a newer
ollama release.

**CUDA build errors in the fork.** The fork requires a CUDA toolkit matching
your driver. Check `nvcc --version` and the compute capability of your GPUs
(RTX 4090 = `sm_89`, RTX 5090 = `sm_120`). Override with
`CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=89;120"` in the environment before
calling `build_ollama_tq.sh --rebuild`.

**Two ollama binaries competing.** Distro ollama (`/usr/local/bin/ollama`)
and the patched one will fight over `:11434`. Always stop the system service
before running the rebuilt binary.

## Why not `serve_ollama_tq.sh`?

That script (at the repo root) routes Ollama-managed GGUFs through vLLM for
serving. This doc is the opposite: keep ollama doing the serving, just make
it use TurboQuant underneath. The two paths are independent — pick one per
workflow.
