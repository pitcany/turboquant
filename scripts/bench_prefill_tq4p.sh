#!/usr/bin/env bash
# Benchmark: measure prefill speedup of TQ4P KV cache vs f16 baseline.
#
# Runs the same prompt through the rebuilt ollama binary twice — once with
# OLLAMA_KV_CACHE_TYPE=tq4p_d128 and once with f16 — and compares the
# prompt_eval_duration (prefill time) reported by /api/generate.
#
# This closes the unchecked test-plan item from PR #13:
#   "Prefill speedup measurement requires offloading layers to GPU"
#
# Prerequisites:
#   - Built binary at $HOME/.local/src/ollama-tq/ollama/ollama
#   - A small model pulled (qwen2.5:3b or llama3.1:8b)
#   - GPU available (RTX 4090 or 5090)
#
# Usage:
#   scripts/bench_prefill_tq4p.sh
#   scripts/bench_prefill_tq4p.sh --model llama3.1:8b
#   scripts/bench_prefill_tq4p.sh --runs 5

set -euo pipefail

OLLAMA_BIN="${OLLAMA_BIN:-$HOME/.local/src/ollama-tq/ollama/ollama}"
MODEL=""
RUNS=3
PORT=11434
TIMEOUT=30

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)   MODEL="${2:?--model requires a value}"; shift 2 ;;
        --model=*) MODEL="${1#*=}"; shift ;;
        --runs)    RUNS="${2:?--runs requires a value}"; shift 2 ;;
        --runs=*)  RUNS="${1#*=}"; shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *)         echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── Preflight ──────────────────────────────────────────────────────────

if [[ ! -x "$OLLAMA_BIN" ]]; then
    echo "ERROR: ollama binary not found at $OLLAMA_BIN" >&2
    echo "  Run scripts/build_ollama_tq.sh first." >&2
    exit 1
fi

if ! nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found — GPU required for this benchmark" >&2
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null | head -1)
echo "[+] GPU: $GPU_INFO"
echo "[+] binary: $OLLAMA_BIN"

# ── Prompt generation ──────────────────────────────────────────────────
# Generate a ~2K token prompt for a meaningful prefill measurement.
# Repeat a paragraph to reach target length without needing external files.

PARA="The TurboQuant algorithm applies a two-stage compression pipeline to key-value cache vectors in transformer attention. Stage one rotates the vector by a fixed Haar orthogonal matrix and applies Lloyd-Max scalar quantization at three bits per coordinate. Stage two projects the residual through a Gaussian random matrix and stores one sign bit per output dimension. The combined estimator achieves an inner-product correlation of 0.93 at 4.25 bits per weight, matching the theoretical distortion-rate bound derived in Zandieh et al. (ICLR 2026). This compression reduces KV cache memory by approximately 4x compared to fp16 storage while maintaining attention score fidelity above 0.99 cosine similarity on needle-in-haystack retrieval tasks."

PROMPT=""
for _ in $(seq 1 14); do
    PROMPT="${PROMPT}${PARA} "
done

echo "[+] prompt length: ~$(echo "$PROMPT" | wc -w) words"

# ── Helper functions ───────────────────────────────────────────────────

OLLAMA_PID=""
DISTRO_WAS_RUNNING=0

cleanup() {
    if [[ -n "$OLLAMA_PID" ]]; then
        kill "$OLLAMA_PID" 2>/dev/null || true
        wait "$OLLAMA_PID" 2>/dev/null || true
        OLLAMA_PID=""
    fi
}

restore_distro() {
    if [[ "$DISTRO_WAS_RUNNING" -eq 1 ]]; then
        echo "[=] restarting distro ollama daemon"
        if command -v systemctl &>/dev/null && systemctl is-enabled --quiet ollama 2>/dev/null; then
            systemctl start ollama 2>/dev/null || true
        fi
    fi
}

trap 'cleanup; restore_distro' EXIT

start_ollama() {
    local kv_type="$1"
    local log="$2"

    cleanup

    OLLAMA_KV_CACHE_TYPE="$kv_type" \
    OLLAMA_FLASH_ATTENTION=1 \
    OLLAMA_DEBUG=1 \
        "$OLLAMA_BIN" serve > "$log" 2>&1 &
    OLLAMA_PID=$!

    # Wait for server.
    for _ in $(seq 1 "$TIMEOUT"); do
        if curl -sf "http://localhost:${PORT}/api/tags" &>/dev/null; then
            return 0
        fi
        if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
            echo "ERROR: ollama died during startup (kv_type=$kv_type)" >&2
            tail -20 "$log" >&2
            return 1
        fi
        sleep 1
    done
    echo "ERROR: ollama did not respond within ${TIMEOUT}s" >&2
    return 1
}

PROMPT_JSON_FILE=$(mktemp)
trap 'cleanup; restore_distro; rm -f "$PROMPT_JSON_FILE"' EXIT

run_inference() {
    # Returns: prompt_eval_duration eval_duration prompt_eval_count eval_count (nanoseconds)
    local model="$1"
    local prompt="$2"

    # Build JSON payload safely via python to handle escaping.
    python3 -c "
import json, sys
payload = {'model': sys.argv[1], 'prompt': sys.argv[2],
           'stream': False, 'options': {'num_predict': 1}}
json.dump(payload, open(sys.argv[3], 'w'))
" "$model" "$prompt" "$PROMPT_JSON_FILE" 2>/dev/null || return 1

    local resp
    resp=$(curl -sf --max-time 300 \
        "http://localhost:${PORT}/api/generate" \
        -d @"$PROMPT_JSON_FILE" \
        2>/dev/null) || return 1

    echo "$resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d.get('prompt_eval_duration', 0),
      d.get('eval_duration', 0),
      d.get('prompt_eval_count', 0),
      d.get('eval_count', 0))
" 2>/dev/null
}

# ── Kill existing ollama ───────────────────────────────────────────────

if pgrep -x ollama &>/dev/null; then
    DISTRO_WAS_RUNNING=1
    echo "[=] stopping existing ollama daemon"
    pkill -x ollama 2>/dev/null || true
    sleep 2
fi

# ── Discover model ─────────────────────────────────────────────────────

if [[ -z "$MODEL" ]]; then
    # Start ollama briefly to list models, then kill it.
    start_ollama f16 /tmp/ollama-bench-discovery.log || exit 1

    MODEL_LIST=$("$OLLAMA_BIN" list 2>/dev/null || true)
    if [[ -n "$MODEL_LIST" ]]; then
        MODEL=$(echo "$MODEL_LIST" | tail -n +2 \
            | grep -viE 'embed|rerank' \
            | awk '$3+0 > 0 {
                mb = ($4 == "GB") ? $3 * 1024 : $3;
                printf "%012.1f %s\n", mb, $1
            }' \
            | sort -n | head -1 | awk '{print $2}')
    fi
    cleanup

    if [[ -z "$MODEL" ]]; then
        echo "ERROR: no generative local models found. Pull one first:" >&2
        echo "  ollama pull qwen2.5:3b" >&2
        exit 1
    fi
fi
echo "[+] model: $MODEL"
echo "[+] runs per config: $RUNS"
echo

# ── Benchmark loop ─────────────────────────────────────────────────────

declare -a TQ4P_TIMES F16_TIMES

for kv_type in tq4p_d128 f16; do
    LOG="/tmp/ollama-bench-${kv_type}.log"

    echo "━━━ KV cache type: $kv_type ━━━"
    start_ollama "$kv_type" "$LOG" || exit 1

    # Warmup: load the model once so subsequent runs measure steady-state.
    echo -n "[.] warmup "
    run_inference "$MODEL" "warmup" >/dev/null 2>&1 || true
    echo "done"

    for i in $(seq 1 "$RUNS"); do
        result=$(run_inference "$MODEL" "$PROMPT" 2>/dev/null || true)
        if [[ -z "$result" ]]; then
            echo "  run $i: FAILED" >&2
            continue
        fi

        read -r prefill_ns decode_ns prompt_tokens gen_tokens <<< "$result"
        prefill_ms=$((prefill_ns / 1000000))
        tokens_per_sec=$(python3 -c "print(f'{$prompt_tokens / ($prefill_ns / 1e9):.1f}')" 2>/dev/null)

        echo "  run $i: prefill=${prefill_ms}ms  tokens=${prompt_tokens}  speed=${tokens_per_sec} tok/s"

        if [[ "$kv_type" == "tq4p_d128" ]]; then
            TQ4P_TIMES+=("$prefill_ns")
        else
            F16_TIMES+=("$prefill_ns")
        fi
    done

    # Verify KV cache type in log for tq4p runs.
    if [[ "$kv_type" == "tq4p_d128" ]]; then
        if grep -q "KvCacheType:tq4p_d128" "$LOG" 2>/dev/null; then
            echo "  [+] log confirms tq4p_d128 active"
        elif grep -qiE "tq4p" "$LOG" 2>/dev/null; then
            echo "  [+] log mentions tq4p"
        else
            echo "  [!] WARNING: could not confirm tq4p in log"
        fi
    fi

    cleanup
    echo
done

# ── Results ────────────────────────────────────────────────────────────

if [[ ${#TQ4P_TIMES[@]} -eq 0 ]] || [[ ${#F16_TIMES[@]} -eq 0 ]]; then
    echo "ERROR: not enough successful runs to compare" >&2
    exit 1
fi

python3 -c "
import sys

tq4p = [${TQ4P_TIMES[*]// /,}]
f16  = [${F16_TIMES[*]// /,}]

tq4p_avg = sum(tq4p) / len(tq4p)
f16_avg  = sum(f16) / len(f16)

tq4p_ms = tq4p_avg / 1e6
f16_ms  = f16_avg / 1e6

if f16_avg > 0:
    speedup = f16_avg / tq4p_avg
    overhead = (tq4p_avg - f16_avg) / f16_avg * 100
else:
    speedup = float('inf')
    overhead = float('inf')

print()
print('┌─────────────────────────────────────────────────────┐')
print('│           PREFILL BENCHMARK RESULTS                 │')
print('├─────────────────────────────────────────────────────┤')
print(f'│  f16 (baseline):   {f16_ms:8.1f} ms  ({len(f16)} runs)          │')
print(f'│  tq4p_d128:        {tq4p_ms:8.1f} ms  ({len(tq4p)} runs)          │')
print('├─────────────────────────────────────────────────────┤')
if speedup >= 1:
    print(f'│  Speedup:          {speedup:8.2f}x                       │')
else:
    print(f'│  Overhead:         {overhead:+7.1f}%                        │')
print('│                                                     │')
if speedup >= 0.95:
    print('│  ✓ TQ4P prefill within expected range               │')
else:
    print('│  ✗ TQ4P prefill slower than expected                 │')
print('└─────────────────────────────────────────────────────┘')
print()
print('Notes:')
print('  - TQ4P quantizes KV cache to 4.25 bpw (vs 16 for f16)')
print('  - Prefill overhead from on-device quantize kernel is')
print('    expected to be <5% — amortized over N cached keys')
print('  - Memory savings: ~3.8x KV cache reduction')
"
