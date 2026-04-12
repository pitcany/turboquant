#!/usr/bin/env bash
# Build a TurboQuant-enabled ollama by swapping the vendored llama.cpp for
# turbo-tan/llama.cpp-tq3 (TurboQuant 3-bit KV cache, CUDA kernels).
#
# The layout is designed for iteration: both the ollama source tree and the
# llama.cpp-tq3 fork live as editable git clones under $WORKDIR. The fork is
# symlinked into ollama/llama/llama.cpp so edits to the fork are picked up on
# the next ollama rebuild without re-running this script end-to-end.
#
# Usage:
#   scripts/build_ollama_tq.sh                 # full build
#   scripts/build_ollama_tq.sh --rebuild       # skip clones, just rebuild
#   scripts/build_ollama_tq.sh --stage2        # copy Stage-2 QJL C sources
#                                              # into the fork clone; prints
#                                              # the 4 hand-edits to apply.
#                                              # See patches/stage2-qjl/hooks.md
#   WORKDIR=/path bash scripts/build_ollama_tq.sh
#
# Env:
#   WORKDIR           default: $HOME/.local/src/ollama-tq
#   LLAMACPP_FORK     default: https://github.com/turbo-tan/llama.cpp-tq3
#   LLAMACPP_REF      default: (fork default branch)
#   OLLAMA_REPO       default: https://github.com/ollama/ollama
#   OLLAMA_REF        default: (ollama default branch)
#   CUDA              default: 1  (set 0 to build CPU-only)

set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/.local/src/ollama-tq}"
LLAMACPP_FORK="${LLAMACPP_FORK:-https://github.com/turbo-tan/llama.cpp-tq3}"
LLAMACPP_REF="${LLAMACPP_REF:-}"
OLLAMA_REPO="${OLLAMA_REPO:-https://github.com/ollama/ollama}"
OLLAMA_REF="${OLLAMA_REF:-}"
CUDA="${CUDA:-1}"

REBUILD_ONLY=0
STAGE2=0
for arg in "$@"; do
    case "$arg" in
        --rebuild) REBUILD_ONLY=1 ;;
        --stage2)  STAGE2=1 ;;
        -h|--help)
            sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_SCRIPT="$SCRIPT_DIR/patch_ollama_kv_types.sh"

mkdir -p "$WORKDIR"
cd "$WORKDIR"

LLAMACPP_DIR="$WORKDIR/llama.cpp-tq3"
OLLAMA_DIR="$WORKDIR/ollama"

clone_or_update() {
    local repo="$1" dir="$2" ref="$3"
    if [[ -d "$dir/.git" ]]; then
        echo "[=] updating $dir"
        git -C "$dir" fetch --all --tags --prune
    else
        echo "[+] cloning $repo -> $dir"
        git clone "$repo" "$dir"
    fi
    if [[ -n "$ref" ]]; then
        git -C "$dir" checkout "$ref"
    fi
}

if [[ $REBUILD_ONLY -eq 0 ]]; then
    clone_or_update "$LLAMACPP_FORK" "$LLAMACPP_DIR" "$LLAMACPP_REF"
    clone_or_update "$OLLAMA_REPO"  "$OLLAMA_DIR"   "$OLLAMA_REF"
fi

# Swap the vendored llama.cpp for our fork via symlink. Re-link even on
# --rebuild so a freshly-cloned tree gets wired up.
VENDOR="$OLLAMA_DIR/llama/llama.cpp"
if [[ -e "$VENDOR" && ! -L "$VENDOR" ]]; then
    echo "[=] moving vendored llama.cpp aside: $VENDOR -> ${VENDOR}.orig"
    mv "$VENDOR" "${VENDOR}.orig"
fi
if [[ -L "$VENDOR" ]]; then
    rm "$VENDOR"
fi
echo "[+] symlinking $LLAMACPP_DIR -> $VENDOR"
ln -s "$LLAMACPP_DIR" "$VENDOR"

# Patch the ollama KV-cache-type allowlist so OLLAMA_KV_CACHE_TYPE=tq3_0 is
# accepted by the runner. Idempotent.
bash "$PATCH_SCRIPT" "$OLLAMA_DIR"

# Stage-2 QJL patch: copy paper-faithful C source into the fork's ggml/src
# and print the hand-edits the user has to apply before rebuilding.
if [[ $STAGE2 -eq 1 ]]; then
    STAGE2_DIR="$(cd -- "$SCRIPT_DIR/../patches/stage2-qjl" && pwd)"
    GGML_SRC="$LLAMACPP_DIR/ggml/src"
    if [[ ! -d "$GGML_SRC" ]]; then
        echo "ERROR: $GGML_SRC not found. Is the fork cloned?" >&2
        exit 1
    fi

    echo "[+] stage2: regenerating constants (Π, S, centroids)"
    (cd "$STAGE2_DIR/python" && python3 generate_constants.py)

    echo "[+] stage2: copying C sources into $GGML_SRC/"
    cp -v "$STAGE2_DIR/c/ggml-tq-paper.h"          "$GGML_SRC/"
    cp -v "$STAGE2_DIR/c/ggml-tq-paper.c"          "$GGML_SRC/"
    cp -v "$STAGE2_DIR/c/tqp_centroids_d128.h"     "$GGML_SRC/"
    cp -v "$STAGE2_DIR/c/tqp_centroids_d256.h"     "$GGML_SRC/"
    cp -v "$STAGE2_DIR/c/tqp_constants_d128.h"     "$GGML_SRC/"
    cp -v "$STAGE2_DIR/c/tqp_constants_d256.h"     "$GGML_SRC/"

    cat <<HOOKS_BANNER

===================================================================
Stage-2 sources copied. You now need to apply 4 hand-edits to the
fork (adds enum values, block structs, dispatch entries, CMake source).

See: $STAGE2_DIR/hooks.md

After applying the hand-edits, rebuild with:
    $0 --rebuild
===================================================================
HOOKS_BANNER
    exit 0
fi

# Build. ollama's CMake preset for CUDA expects nvcc in PATH.
cd "$OLLAMA_DIR"
if [[ "$CUDA" = "1" ]]; then
    export OLLAMA_CUSTOM_CPU_DEFS="${OLLAMA_CUSTOM_CPU_DEFS:-}"
    export CMAKE_ARGS="${CMAKE_ARGS:-} -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release"
fi

echo "[+] go generate ./..."
go generate ./...

echo "[+] go build ."
go build -o ollama .

echo
echo "Built: $OLLAMA_DIR/ollama"
echo
echo "Run with:"
echo "    OLLAMA_KV_CACHE_TYPE=tq3_0 OLLAMA_FLASH_ATTENTION=1 $OLLAMA_DIR/ollama serve"
echo
echo "Edit the fork at: $LLAMACPP_DIR"
echo "After fork edits, rebuild with: $0 --rebuild"
