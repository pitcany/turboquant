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
#   CUDA              default: auto-detect ("1" if nvcc is on PATH, "0" otherwise)

set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/.local/src/ollama-tq}"
OLLAMA_REPO="${OLLAMA_REPO:-https://github.com/ollama/ollama}"
OLLAMA_REF="${OLLAMA_REF:-}"
# Auto-detect CUDA: default on if `nvcc` is on PATH, off otherwise. User can
# override with CUDA=1 or CUDA=0 in the environment.
if [[ -z "${CUDA:-}" ]]; then
    if command -v nvcc >/dev/null 2>&1; then CUDA=1; else CUDA=0; fi
fi
echo "[=] CUDA=$CUDA (nvcc $(command -v nvcc >/dev/null 2>&1 && echo found || echo 'not found'))"

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

if [[ "$CUDA" = "1" ]]; then
    GGML_CUDA="$GGML/src/ggml-cuda"
    if [[ -d "$GGML_CUDA" ]]; then
        echo "[+] copying TQ4P CUDA kernels into $GGML_CUDA/"
        for f in tqp-quantize.cu tqp-prepare-query.cu tqp-vec-dot.cu \
                 tqp-set-rows.cu tqp-kernels.cuh tqp-constants-cuda.cuh; do
            cp "$STAGE2_DIR/cuda/$f" "$GGML_CUDA/"
        done
        for f in tqp_constants_d128.h tqp_constants_d256.h \
                 tqp_centroids_d128.h tqp_centroids_d256.h \
                 ggml-tq-paper.h; do
            cp "$STAGE2_DIR/c/$f" "$GGML_CUDA/"
        done
    else
        echo "[!] $GGML_CUDA not found; skipping CUDA kernel copy"
    fi
fi

echo "[+] applying TQ4P hooks"
bash "$APPLY_HOOKS" "$GGML"

# ---- Widen ollama Go allowlist -----------------------------------------------

bash "$PATCH_ALLOWLIST" "$OLLAMA_DIR"

# ---- Patch Go switch statements so tq4p types resolve end-to-end ------------

bash "$APPLY_GO_PLUMBING" "$OLLAMA_DIR"

# ---- Build shared libs via cmake ---------------------------------------------
#
# ollama discovers GPU backends at runtime by loading shared libraries from
# a directory next to the binary. Without this step the binary is CPU-only:
# GPU discovery finds 0 devices and inference falls back to CPU.
#
# cmake presets (defined in ollama's CMakePresets.json):
#   CPU     → libggml-base.so, libggml-cpu-*.so
#   CUDA 12 → cuda_v12/libggml-cuda.so  (requires nvcc)

cd "$OLLAMA_DIR"

# Vulkan requires glslc which is often not installed; skip it since we only
# need CPU + CUDA.
CMAKE_EXTRA="-DCMAKE_DISABLE_FIND_PACKAGE_Vulkan=TRUE"

NPROC=$(nproc 2>/dev/null || echo 4)

echo "[+] cmake: configuring CPU preset"
cmake --preset 'CPU' $CMAKE_EXTRA 2>&1 | tail -3

echo "[+] cmake: building CPU shared libs"
cmake --build --preset 'CPU' -- -j"$NPROC" 2>&1 | tail -3

if [[ "$CUDA" = "1" ]]; then
    # Detect CUDA major version for the right preset. Under `set -euo
    # pipefail`, grep returning exit 1 on no-match would abort the script
    # and skip the `:-12` fallback below, so trap its exit with `|| true`.
    CUDA_VER=$(nvcc --version 2>/dev/null | { grep -oP 'release \K[0-9]+' || true; } | head -1)
    CUDA_PRESET="CUDA ${CUDA_VER:-12}"
    CUDA_ARCHS="${CUDA_ARCHS:-89;120}"

    echo "[+] cmake: configuring $CUDA_PRESET preset (archs: $CUDA_ARCHS)"
    cmake --preset "$CUDA_PRESET" $CMAKE_EXTRA \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" 2>&1 | tail -3

    echo "[+] cmake: building CUDA shared libs"
    cmake --build --preset "$CUDA_PRESET" -- -j"$NPROC" 2>&1 | tail -3
fi

# Copy built libs next to the binary so ollama's runner discovery finds them.
echo "[+] installing shared libs"
LIB_OUT="$OLLAMA_DIR/build/lib/ollama"
if [[ -d "$LIB_OUT" ]]; then
    # CPU libs go next to binary. Trailing `*` after `.so` catches
    # versioned variants (`libggml-cpu.so.1`, etc.) — ollama's cmake can
    # emit SONAME'd shared libraries, and `cp -a` would otherwise copy
    # the unversioned symlink without its target, leaving a dangling
    # link that the dynamic linker silently fails to load.
    for f in "$LIB_OUT"/libggml-base.so* "$LIB_OUT"/libggml-cpu*.so*; do
        [[ -f "$f" ]] && cp -a "$f" "$OLLAMA_DIR/"
    done
    # CUDA lib goes into cuda_v{N}/ subdir for runner discovery
    if [[ -f "$LIB_OUT/libggml-cuda.so" ]]; then
        CUDA_SUBDIR="cuda_v${CUDA_VER:-12}"
        mkdir -p "$OLLAMA_DIR/$CUDA_SUBDIR"
        cp -a "$LIB_OUT/libggml-cuda.so" "$OLLAMA_DIR/$CUDA_SUBDIR/"
        echo "  → $CUDA_SUBDIR/libggml-cuda.so"
    fi
else
    echo "[!] warning: cmake build dir $LIB_OUT not found"
fi

# ---- Build Go binary ---------------------------------------------------------

echo "[+] go generate ./..."
go generate ./...

echo "[+] go build ."
go build -o ollama .

cat <<DONE

Built: $OLLAMA_DIR/ollama

Shared libs installed next to binary:
$(ls "$OLLAMA_DIR"/*.so* "$OLLAMA_DIR"/cuda_v*/*.so 2>/dev/null | sed 's|.*/||;s/^/  /')

Run with:
    OLLAMA_KV_CACHE_TYPE=tq4p_d128 OLLAMA_FLASH_ATTENTION=0 $OLLAMA_DIR/ollama serve

For Qwen 3.5 (head_dim=256):
    OLLAMA_KV_CACHE_TYPE=tq4p_d256 OLLAMA_FLASH_ATTENTION=0 $OLLAMA_DIR/ollama serve

Note: flash attention must be disabled with TQ4P. The fattn kernels have
hardcoded type combinations that don't include TQ4P — enabling it forces
every attention layer to fall back to CPU. The MUL_MAT path handles TQ4P
at full GPU speed.

The TQ4P sources live at patches/stage2-qjl/c/ in this repo and are copied
into $GGML_SRC on every run. Edit them in the repo, then rerun with --rebuild.
DONE
