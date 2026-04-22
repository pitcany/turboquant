#!/usr/bin/env bash
# Benchmark: estimate KV-cache memory in MB/token by comparing two context
# sizes against the rebuilt ollama binary.
#
# This validates the Go-side VRAM scheduler math added in apply_go_plumbing.sh:
#   MB/token = (KV_large_ctx - KV_small_ctx) / (large_ctx - small_ctx)
#
# Expected for tq4p_d128-backed Llama models: roughly 0.08 MB/token.
#
# Usage:
#   scripts/bench_kv_memory.sh
#   scripts/bench_kv_memory.sh --model llama3.1:8b
#   scripts/bench_kv_memory.sh --small-ctx 2048 --large-ctx 8192

set -euo pipefail

OLLAMA_BIN="${OLLAMA_BIN:-$HOME/.local/src/ollama-tq/ollama/ollama}"
CACHE_TYPE="${CACHE_TYPE:-tq4p_d128}"
MODEL=""
SMALL_CTX=2048
LARGE_CTX=8192
PORT=11434
TIMEOUT=30
EXPECTED_MB_PER_TOKEN="${EXPECTED_MB_PER_TOKEN:-0.08}"
MAX_MB_PER_TOKEN="${MAX_MB_PER_TOKEN:-0.15}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="${2:?--model requires a value}"; shift 2 ;;
        --model=*)     MODEL="${1#*=}"; shift ;;
        --cache-type)  CACHE_TYPE="${2:?--cache-type requires a value}"; shift 2 ;;
        --cache-type=*) CACHE_TYPE="${1#*=}"; shift ;;
        --small-ctx)   SMALL_CTX="${2:?--small-ctx requires a value}"; shift 2 ;;
        --small-ctx=*) SMALL_CTX="${1#*=}"; shift ;;
        --large-ctx)   LARGE_CTX="${2:?--large-ctx requires a value}"; shift 2 ;;
        --large-ctx=*) LARGE_CTX="${1#*=}"; shift ;;
        -h|--help)     sed -n '2,18p' "$0"; exit 0 ;;
        *)             echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -x "$OLLAMA_BIN" ]]; then
    echo "ERROR: ollama binary not found at $OLLAMA_BIN" >&2
    echo "  Run scripts/build_ollama_tq.sh first." >&2
    exit 1
fi

if ! nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found — GPU memory benchmark requires a GPU" >&2
    exit 1
fi

if (( LARGE_CTX <= SMALL_CTX )); then
    echo "ERROR: --large-ctx must be greater than --small-ctx" >&2
    exit 1
fi

OLLAMA_PID=""
DISTRO_WAS_RUNNING=0
PAYLOAD_FILE="$(mktemp)"
LOG_FILE="/tmp/ollama-kv-memory.log"

cleanup() {
    if [[ -n "$OLLAMA_PID" ]]; then
        kill "$OLLAMA_PID" 2>/dev/null || true
        wait "$OLLAMA_PID" 2>/dev/null || true
    fi
    rm -f "$PAYLOAD_FILE"
    if [[ "$DISTRO_WAS_RUNNING" -eq 1 ]]; then
        echo "[=] restarting distro ollama daemon"
        if command -v systemctl &>/dev/null && systemctl is-enabled --quiet ollama 2>/dev/null; then
            systemctl start ollama 2>/dev/null || true
        fi
    fi
}
trap cleanup EXIT

wait_for_server() {
    for _ in $(seq 1 "$TIMEOUT"); do
        if curl -sf "http://localhost:${PORT}/api/tags" &>/dev/null; then
            return 0
        fi
        if [[ -n "$OLLAMA_PID" ]] && ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
            echo "ERROR: ollama exited during startup" >&2
            tail -40 "$LOG_FILE" >&2 || true
            return 1
        fi
        sleep 1
    done
    echo "ERROR: ollama did not respond on :$PORT within ${TIMEOUT}s" >&2
    return 1
}

start_ollama() {
    if pgrep -x ollama &>/dev/null; then
        DISTRO_WAS_RUNNING=1
        echo "[=] stopping existing ollama daemon"
        pkill -x ollama 2>/dev/null || true
        sleep 2
    fi

    OLLAMA_KV_CACHE_TYPE="$CACHE_TYPE" \
    OLLAMA_FLASH_ATTENTION=1 \
    OLLAMA_DEBUG=1 \
    OLLAMA_LOG_LEVEL=debug \
        "$OLLAMA_BIN" serve > "$LOG_FILE" 2>&1 &
    OLLAMA_PID=$!

    wait_for_server
}

discover_model() {
    local model_list
    model_list=$("$OLLAMA_BIN" list 2>/dev/null || true)
    if [[ -n "$model_list" ]]; then
        MODEL=$(echo "$model_list" | tail -n +2 \
            | grep -viE 'embed|rerank' \
            | awk '$3+0 > 0 {
                mb = ($4 == "GB") ? $3 * 1024 : $3
                printf "%012.1f %s\n", mb, $1
            }' \
            | sort -n | head -1 | awk '{print $2}')
    fi
}

run_context_load() {
    local ctx="$1"

    python3 - "$MODEL" "$ctx" "$PAYLOAD_FILE" <<'PY'
import json
import sys

payload = {
    "model": sys.argv[1],
    "prompt": "Reply with one short sentence about KV cache compression.",
    "stream": False,
    "options": {
        "num_ctx": int(sys.argv[2]),
        "num_predict": 8,
    },
}
with open(sys.argv[3], "w", encoding="utf-8") as handle:
    json.dump(payload, handle)
PY

    curl -sf --max-time 180 "http://localhost:${PORT}/api/generate" \
        -d @"$PAYLOAD_FILE" >/dev/null
}

read_kv_cache_mb() {
    python3 - "$LOG_FILE" <<'PY'
import re
import sys

log_path = sys.argv[1]
pattern = re.compile(r'msg="kv cache".*size="([0-9.]+)\s+(KiB|MiB|GiB)"')
units = {"KiB": 1 / 1024, "MiB": 1, "GiB": 1024}

groups = []
current = []

with open(log_path, encoding="utf-8", errors="replace") as handle:
    for raw_line in handle:
        line = raw_line.rstrip("\n")
        match = pattern.search(line)
        if match:
            current.append(float(match.group(1)) * units[match.group(2)])
            continue
        if current:
            groups.append(current)
            current = []

if current:
    groups.append(current)

if not groups:
    sys.exit(1)

print(f"{sum(groups[-1]):.6f}")
PY
}

if [[ -z "$MODEL" ]]; then
    start_ollama
    discover_model
    if [[ -z "$MODEL" ]]; then
        echo "ERROR: no generative local models found. Pull one first:" >&2
        echo "  ollama pull qwen2.5:3b" >&2
        exit 1
    fi
fi

echo "[+] binary: $OLLAMA_BIN"
echo "[+] model: $MODEL"
echo "[+] cache type: $CACHE_TYPE"
echo "[+] contexts: $SMALL_CTX -> $LARGE_CTX"
echo "[+] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

if [[ -z "$OLLAMA_PID" ]]; then
    start_ollama
fi

echo "[.] loading model at num_ctx=$SMALL_CTX"
run_context_load "$SMALL_CTX"
sleep 2
KV_SMALL_MB="$(read_kv_cache_mb)"
echo "[+] KV cache @ $SMALL_CTX ctx: ${KV_SMALL_MB} MB"

echo "[.] loading model at num_ctx=$LARGE_CTX"
run_context_load "$LARGE_CTX"
sleep 2
KV_LARGE_MB="$(read_kv_cache_mb)"
echo "[+] KV cache @ $LARGE_CTX ctx: ${KV_LARGE_MB} MB"

RESULT="$(python3 - "$KV_SMALL_MB" "$KV_LARGE_MB" "$SMALL_CTX" "$LARGE_CTX" "$EXPECTED_MB_PER_TOKEN" "$MAX_MB_PER_TOKEN" <<'PY'
import sys

small_mb = float(sys.argv[1])
large_mb = float(sys.argv[2])
small_ctx = int(sys.argv[3])
large_ctx = int(sys.argv[4])
expected = float(sys.argv[5])
limit = float(sys.argv[6])

delta_mb = large_mb - small_mb
delta_ctx = large_ctx - small_ctx
mb_per_token = delta_mb / delta_ctx
passed = delta_mb > 0 and mb_per_token >= expected * 0.5 and mb_per_token <= limit
status = "PASS" if passed else "FAIL"

print(f"{status} {delta_mb:.6f} {mb_per_token:.6f} {expected:.6f} {limit:.6f}")
PY
)"

STATUS="$(echo "$RESULT" | awk '{print $1}')"
DELTA_MB="$(echo "$RESULT" | awk '{print $2}')"
MB_PER_TOKEN="$(echo "$RESULT" | awk '{print $3}')"

echo
echo "KV memory delta: ${DELTA_MB} MB"
echo "Measured MB/token: ${MB_PER_TOKEN}"
echo "Expected MB/token: ${EXPECTED_MB_PER_TOKEN} (PASS threshold <= ${MAX_MB_PER_TOKEN})"

if grep -q "TQ4P: KV cache memory estimate" "$LOG_FILE" 2>/dev/null; then
    echo "[+] debug log:"
    grep "TQ4P: KV cache memory estimate" "$LOG_FILE" | tail -1
fi

echo
if [[ "$STATUS" == "PASS" ]]; then
    echo "PASS: KV scheduler estimate is in the expected TQ4P range."
    exit 0
fi

echo "FAIL: KV scheduler estimate is still too high."
echo "      A broken build typically measures around 0.6 MB/token instead of ~0.08."
exit 1
