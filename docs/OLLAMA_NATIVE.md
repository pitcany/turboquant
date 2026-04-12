# Native Ollama with TurboQuant (TQ4P)

Run `ollama` with the paper-faithful TurboQuant KV cache — no vLLM, no
Harbor, no Python serving layer. `scripts/build_ollama_tq.sh` patches
ollama's own vendored ggml in place with two new quantization types,
`tq4p_d128` and `tq4p_d256`, implementing Zandieh et al.'s [TurboQuant:
Online Vector Quantization with Near-optimal Distortion
Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026).

Neither ollama nor upstream llama.cpp has merged TurboQuant as of April 2026.
This patch set is an end-run around that — a thin, additive layer on top of
ollama's own sources.

## One-time build

```bash
scripts/build_ollama_tq.sh
```

This clones ollama under `$HOME/.local/src/ollama-tq/` (override with
`WORKDIR=...`), patches `ml/backend/ggml/ggml/src/` with the TQ4P sources,
applies four additive edits (enum, two dispatch tables, CMakeLists), widens
ollama's Go KV-cache-type allowlist to accept `tq4p_d128` and `tq4p_d256`,
then runs `go generate` and `go build`.

CUDA is on by default (`CUDA=0` for CPU-only). `nvcc` must be in `PATH`.

## Run

```bash
systemctl --user stop ollama 2>/dev/null || pkill -x ollama || true

# Llama 3.x, Qwen 2.5, Qwen 3 — head_dim 128
OLLAMA_KV_CACHE_TYPE=tq4p_d128 \
OLLAMA_FLASH_ATTENTION=1 \
~/.local/src/ollama-tq/ollama/ollama serve

# Qwen 3.5 (Gated Attention) — head_dim 256
OLLAMA_KV_CACHE_TYPE=tq4p_d256 \
OLLAMA_FLASH_ATTENTION=1 \
~/.local/src/ollama-tq/ollama/ollama serve

# In another shell
ollama run qwen2.5-coder:32b
```

Your pulled GGUFs under `~/.ollama/models` work unchanged — TurboQuant
only rewrites the runtime KV cache; it does not touch weights on disk.

## Iterating on TQ4P

The TQ4P sources live at `patches/stage2-qjl/c/` in this repo. Edits there
are picked up on the next build:

```bash
scripts/build_ollama_tq.sh --rebuild
```

`--rebuild` skips the clone step, re-copies `patches/stage2-qjl/c/*.{c,h}`
into ollama's `ml/backend/ggml/ggml/src/`, re-runs `apply_hooks.sh`
(idempotent), then rebuilds ollama.

## What the build script actually does

1. Clones (or updates) ollama into `$WORKDIR/ollama`.
2. Recovers from earlier symlink-based builds if detected: if
   `ollama/llama/llama.cpp` is a symlink and `llama.cpp.orig` exists, it
   restores the original tree. (An older version of this script replaced
   the tree wholesale; that broke ollama's build because the replacement
   was missing Go package stubs. The new flow doesn't touch
   `llama/llama.cpp` at all.)
3. Regenerates Π, S, and Lloyd-Max centroids from seeds.
4. Copies the TQ4P `.c`/`.h` files into `ml/backend/ggml/ggml/src/`.
5. Applies four hooks via `patches/stage2-qjl/apply_hooks.sh`:
   - new enum values in `include/ggml.h`
   - new entries in `src/ggml.c` `type_traits` table
   - new entries in `src/ggml-cpu/ggml-cpu.c` `type_traits_cpu` table
   - `ggml-tq-paper.c` added to `src/CMakeLists.txt` ggml-base sources
6. Widens ollama's Go KV-cache-type allowlist
   (`scripts/patch_ollama_kv_types.sh`, idempotent).
7. Runs `go generate ./... && go build .` in the ollama tree.

Ollama's own `llama/llama.cpp/` is **not touched** — that's the llama.cpp
C++ layer, separate from ggml, and ollama maintains Go package stubs
inside it that the build depends on.

## Validation (before running a model)

End-to-end byte-exact tests for the TQ4P C sources:

```bash
cd patches/stage2-qjl/c && make test
```

This compiles the C as a shared library, loads it via ctypes, and
cross-checks 30 byte-exact assertions against `turboquant.py` (the
paper-accurate Python reference). All 30 must pass before the patch
gets applied.

## Rolling back

```bash
cd ~/.local/src/ollama-tq/ollama

# Revert the hook edits and the allowlist patch
git checkout -- ml/backend/ggml/ggml/include/ggml.h \
                ml/backend/ggml/ggml/src/ggml.c \
                ml/backend/ggml/ggml/src/ggml-cpu/ggml-cpu.c \
                ml/backend/ggml/ggml/src/CMakeLists.txt \
                $(grep -rl "turboquant:" --include='*.go' .)

# Remove the added source files
rm ml/backend/ggml/ggml/src/ggml-tq-paper.{c,h} \
   ml/backend/ggml/ggml/src/tqp_{centroids,constants}_d{128,256}.h

# Rebuild
go generate ./... && go build -o ollama .

# Or nuclear: delete the workdir and reinstall ollama from your distro
rm -rf ~/.local/src/ollama-tq
```

Everything the patch adds is marked with the `tq4p` prefix (or a
`// turboquant:` comment for the Go allowlist), so you can verify the
rollback with `grep -r tq4p ~/.local/src/ollama-tq/ollama`.

## Troubleshooting

**`OLLAMA_KV_CACHE_TYPE=tq4p_d128` silently falls back to f16.** Check logs
for `unsupported KV cache type`. The allowlist patch failed to find the
validator — run `bash scripts/patch_ollama_kv_types.sh
~/.local/src/ollama-tq/ollama` manually and follow the error output.

**`ollama ggml tree not found at ml/backend/ggml/ggml`**. ollama's repo
layout changed. Check the current path to `include/ggml.h` inside ollama
and update the `GGML=` line in `build_ollama_tq.sh`.

**CUDA build errors.** CUDA toolkit must match your driver.
`nvcc --version`. For RTX 4090 + RTX 5090 set
`CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=89;120"` before `--rebuild`.

**Two ollama binaries competing.** Distro `/usr/local/bin/ollama` and the
rebuilt one will both try to bind `:11434`. Stop the system service before
running the rebuilt binary.

**Hook apply claims "already patched" but the type is missing.** The
idempotency marker (`tq4p`) is present but a file got reverted. Remove the
marker line and rerun.

## Note on the old turbo-tan fork approach

Earlier revisions of this script replaced ollama's `llama/llama.cpp` with
a symlink to `turbo-tan/llama.cpp-tq3` (a non-paper-faithful TurboQuant
implementation). That approach was abandoned because:

1. The fork is *not* paper-faithful — it uses a Walsh-Hadamard rotation
   instead of Haar, has no QJL stage, and operates per-32-block instead of
   per-head. Details in `patches/stage2-qjl/PLAN.md`.
2. Replacing `llama/llama.cpp` removed ollama's required Go package stubs,
   breaking the build.
3. Our TQ4P patch doesn't need anything from that fork — it's paper-faithful
   from scratch and lives entirely in `patches/stage2-qjl/`.

If you specifically want the fork's (non-paper) `tq3_0` type, check it out
manually and apply it alongside — but that's opt-in and unsupported by
this script.
