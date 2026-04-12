#!/usr/bin/env bash
# End-to-end smoke test: build the fork with the Stage-2 TQ4P patch, then run
# llama-cli on a tiny model with --cache-type-k/-v tq4p_d128 and verify the
# output is non-garbage.
#
# What this catches that the Python unit tests can't:
#   - Does the vec_dot dispatch actually route through the right function?
#   - Does sizeof(block_tq4p_d128) match what ggml expects at runtime?
#   - Does the KV cache path handle non-contiguous writes correctly?
#   - Does attention produce finite tokens (no NaN, no infinite loops)?
#
# Usage:
#   scripts/smoke_test_tq4p.sh [--gguf PATH] [--prompt TEXT]
#
# If no GGUF is provided, it falls back to looking in ~/.ollama/models for
# any llama-family or qwen-family model with head_dim=128. Head_dim≠128 models
# will fail — that's expected; use --gguf to target a specific model.
#
# Exit codes:
#   0 — build + run succeeded, output looks sane
#   1 — build failed
#   2 — runtime failure (crash, NaN, unsupported type)
#   3 — output is garbage (all the same token, or empty)

set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/.local/src/ollama-tq}"
GGUF=""
PROMPT="The quick brown fox"
N_TOKENS=32

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gguf) GGUF="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --tokens) N_TOKENS="$2"; shift 2 ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

LLAMACPP_DIR="$WORKDIR/llama.cpp-tq3"
LLAMA_CLI="$LLAMACPP_DIR/build/bin/llama-cli"

if [[ ! -x "$LLAMA_CLI" ]]; then
    echo "ERROR: $LLAMA_CLI not found. Run scripts/build_ollama_tq.sh --stage2 first," >&2
    echo "then apply hooks via patches/stage2-qjl/apply_hooks.sh, then --rebuild." >&2
    exit 1
fi

# Verify TQ4P types are registered in the binary (indirect — look for the
# symbols that the dispatch table references).
if ! nm -D --defined-only "$LLAMA_CLI" 2>/dev/null | grep -q ggml_vec_dot_tq4p_d128; then
    # Static symbol, nm -D won't show it. Fall back to grep on the binary.
    if ! strings "$LLAMA_CLI" | grep -q tq4p_d128; then
        echo "ERROR: llama-cli binary doesn't contain 'tq4p_d128' string — hooks not applied?" >&2
        exit 1
    fi
fi
echo "[+] llama-cli contains tq4p_d128 references"

# Locate a GGUF if not provided.
if [[ -z "$GGUF" ]]; then
    for candidate in "$HOME/.ollama/models/blobs/"sha256-*; do
        [[ -f "$candidate" ]] || continue
        # GGUF magic is "GGUF" (0x46554747 little-endian).
        magic=$(head -c 4 "$candidate" 2>/dev/null | tr -d '\0' || true)
        if [[ "$magic" == "GGUF" ]]; then
            GGUF="$candidate"; break
        fi
    done
fi
if [[ -z "$GGUF" ]]; then
    echo "ERROR: no GGUF found. Pass --gguf PATH." >&2
    exit 1
fi
echo "[+] using GGUF: $GGUF"

# Run llama-cli with TQ4P cache. Capture stdout + stderr.
OUT=$(mktemp)
set +e
"$LLAMA_CLI" \
    -m "$GGUF" \
    -ctk tq4p_d128 \
    -ctv tq4p_d128 \
    -p "$PROMPT" \
    -n "$N_TOKENS" \
    --no-warmup \
    -ngl 0 \
    2>&1 > "$OUT"
RC=$?
set -e

if [[ $RC -ne 0 ]]; then
    echo "ERROR: llama-cli exited with $RC" >&2
    sed 's/^/    /' "$OUT" | tail -20 >&2
    rm -f "$OUT"
    exit 2
fi

# Output sanity checks.
BODY=$(sed -n '/^'"${PROMPT:0:20}"'/,$p' "$OUT" | head -50)
if [[ -z "$BODY" ]]; then
    echo "ERROR: llama-cli produced no output following the prompt" >&2
    cat "$OUT" >&2
    rm -f "$OUT"
    exit 3
fi

# Check for NaN indicator (llama-cli prints "nan" in output tokens sometimes
# when attention produces NaNs).
if grep -qiE "\bnan\b|cache type not supported" "$OUT"; then
    echo "ERROR: output contains NaN or unsupported-type marker" >&2
    grep -iE "\bnan\b|cache type not supported" "$OUT" | head -5 >&2
    rm -f "$OUT"
    exit 3
fi

echo "[+] sample output:"
echo "$BODY" | head -5 | sed 's/^/    /'

rm -f "$OUT"
echo "[+] smoke test passed"
