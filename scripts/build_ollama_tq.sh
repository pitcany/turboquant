#!/usr/bin/env bash
# Build ollama with the TurboQuant paper-faithful TQ4P KV cache.
#
# Patches ollama's OWN vendored ggml in place. No llama.cpp fork is cloned
# or symlinked.
#
# Three additive patches:
#   1. TQ4P ggml quant types, dropped into  ml/backend/ggml/ggml/src/
#      and registered via 4 hooks in that same tree.
#   2. ollama Go KV-cache-type allowlist widened to accept "tq4p_d128" and
#      "tq4p_d256".
#   3. ollama Go plumbing: switch-case entries in 4 Go files so the
#      cache-type strings map to the correct GGML enum values end-to-end.
#
# Usage:
#   scripts/build_ollama_tq.sh                 # full build
#   scripts/build_ollama_tq.sh --rebuild       # skip clone, reapply patches,
#                                              # rebuild ollama
#   WORKDIR=/path bash scripts/build_ollama_tq.sh
#
# Env:
#   WORKDIR           default: $HOME/.local/src/ollama-tq
#   OLLAMA_REPO       default: https://github.com/ollama/ollama
#   OLLAMA_REF        default: (repo default branch)
#   CUDA              default: 1  (set 0 for CPU-only)

set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/.local/src/ollama-tq}"
OLLAMA_REPO="${OLLAMA_REPO:-https://github.com/ollama/ollama}"
OLLAMA_REF="${OLLAMA_REF:-}"
CUDA="${CUDA:-1}"

REBUILD_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --rebuild) REBUILD_ONLY=1 ;;
        -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PATCH_ALLOWLIST="$SCRIPT_DIR/patch_ollama_kv_types.sh"
STAGE2_DIR="$REPO_ROOT/patches/stage2-qjl"
APPLY_HOOKS="$STAGE2_DIR/apply_hooks.sh"
APPLY_GO_PLUMBING="$STAGE2_DIR/apply_go_plumbing.sh"

mkdir -p "$WORKDIR"
cd "$WORKDIR"

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
    clone_or_update "$OLLAMA_REPO" "$OLLAMA_DIR" "$OLLAMA_REF"
fi

# ---- Recovery from earlier symlink-based builds ------------------------------
# Older versions of this script replaced ollama/llama/llama.cpp with a symlink
# to a turbo-tan fork and preserved the original at llama.cpp.orig. That broke
# ollama's build (missing Go package stubs under llama.cpp/{common,src,...}).
# Restore the original tree if we detect that state.
VENDOR="$OLLAMA_DIR/llama/llama.cpp"
VENDOR_ORIG="${VENDOR}.orig"
if [[ -L "$VENDOR" ]]; then
    if [[ -d "$VENDOR_ORIG" ]]; then
        echo "[=] restoring ollama's vendored llama.cpp from ${VENDOR_ORIG}"
        rm "$VENDOR"
        mv "$VENDOR_ORIG" "$VENDOR"
    else
        echo "ERROR: $VENDOR is a symlink but $VENDOR_ORIG is missing." >&2
        echo "  Delete $OLLAMA_DIR and rerun without --rebuild to re-clone." >&2
        exit 1
    fi
fi

# Locate ollama's ggml tree. Path has been stable (ml/backend/ggml/ggml) since
# ollama's Go rewrite; warn loudly if it moves.
GGML="$OLLAMA_DIR/ml/backend/ggml/ggml"
if [[ ! -f "$GGML/include/ggml.h" ]]; then
    echo "ERROR: ollama ggml tree not found at $GGML" >&2
    echo "  ollama may have restructured. Check its repo layout and update" >&2
    echo "  this script's GGML= path." >&2
    exit 1
fi

# ---- Apply TQ4P patch --------------------------------------------------------

echo "[+] regenerating TQ4P constants (Π, S, Lloyd-Max)"
(cd "$STAGE2_DIR/python" && python3 generate_constants.py --out-c ../c --out-pt ../c >/dev/null)

GGML_SRC="$GGML/src"
echo "[+] copying TQ4P sources into $GGML_SRC/"
for f in ggml-tq-paper.h ggml-tq-paper.c \
         tqp_centroids_d128.h tqp_centroids_d256.h \
         tqp_constants_d128.h tqp_constants_d256.h; do
    cp "$STAGE2_DIR/c/$f" "$GGML_SRC/"
done

echo "[+] applying TQ4P hooks"
bash "$APPLY_HOOKS" "$GGML"

# ---- Widen ollama Go allowlist -----------------------------------------------

bash "$PATCH_ALLOWLIST" "$OLLAMA_DIR"

# ---- Patch Go switch statements so tq4p types resolve end-to-end ------------

bash "$APPLY_GO_PLUMBING" "$OLLAMA_DIR"

# ---- Build -------------------------------------------------------------------

cd "$OLLAMA_DIR"
if [[ "$CUDA" = "1" ]]; then
    export CMAKE_ARGS="${CMAKE_ARGS:-} -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release"
fi

echo "[+] go generate ./..."
go generate ./...

echo "[+] go build ."
go build -o ollama .

cat <<DONE

Built: $OLLAMA_DIR/ollama

Run with:
    OLLAMA_KV_CACHE_TYPE=tq4p_d128 OLLAMA_FLASH_ATTENTION=1 $OLLAMA_DIR/ollama serve

For Qwen 3.5 (head_dim=256):
    OLLAMA_KV_CACHE_TYPE=tq4p_d256 OLLAMA_FLASH_ATTENTION=1 $OLLAMA_DIR/ollama serve

The TQ4P sources live at patches/stage2-qjl/c/ in this repo and are copied
into $GGML_SRC on every run. Edit them in the repo, then rerun with --rebuild.
DONE
