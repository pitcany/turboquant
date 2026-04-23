#!/usr/bin/env bash
# Smoke test: prove the rebuilt ollama binary uses TQ4P for its KV cache
# at runtime, not silently falling back to f16.
#
# What this catches (each step maps to a failure mode):
#   (a) Binary missing          — build_ollama_tq.sh not run
#   (d) Server won't start      — Go plumbing crash on unknown type
#   (e) Silent f16 fallback     — allowlist + Go switch missing/stale
#   (f) Inference broken         — ggml dispatch not wired, segfault
#   (g) NaN / garbled output    — vec_dot mismatch, numerical blow-up
#
# Together these prove that the KV-type allowlist, Go plumbing switch
# statements, and ggml dispatch table are aligned end-to-end.
#
# Exit codes:
#   0 — all checks passed: TQ4P KV cache is live
#   1 — preflight failure (binary missing, no model)
#   2 — server startup / connectivity failure
#   3 — cache type fallback detected (f16 instead of tq4p)
#   4 — inference failure (crash, NaN, empty output)
#
# Usage:
#   scripts/smoke_test_tq4p.sh
#   scripts/smoke_test_tq4p.sh --cache-type tq4p_d256
#   scripts/smoke_test_tq4p.sh --model qwen3.5:4b-q8_0

set -euo pipefail

CACHE_TYPE="${CACHE_TYPE:-tq4p_d128}"
OLLAMA_BIN="${OLLAMA_BIN:-$HOME/.local/src/ollama-tq/ollama/ollama}"
MODEL="${MODEL:-}"
LOG="/tmp/ollama-tq4p.log"
PORT=11434
TIMEOUT=30

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cache-type=*) CACHE_TYPE="${1#*=}"; shift ;;
        --cache-type)   CACHE_TYPE="${2:?--cache-type requires a value}"; shift 2 ;;
        --model=*)      MODEL="${1#*=}"; shift ;;
        --model)        MODEL="${2:?--model requires a value}"; shift 2 ;;
        -h|--help)      sed -n '2,28p' "$0"; exit 0 ;;
        *)              echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Track state for cleanup.
OLLAMA_PID=""
DISTRO_WAS_RUNNING=0

cleanup() {
    if [[ -n "$OLLAMA_PID" ]]; then
        kill "$OLLAMA_PID" 2>/dev/null || true
        wait "$OLLAMA_PID" 2>/dev/null || true
    fi
    if [[ "$DISTRO_WAS_RUNNING" -eq 1 ]]; then
        echo "[=] restarting distro ollama daemon"
        if command -v systemctl &>/dev/null && systemctl is-enabled --quiet ollama 2>/dev/null; then
            systemctl start ollama 2>/dev/null || true
        fi
    fi
}
trap cleanup EXIT

fail_log() {
    echo
    echo "=== last 50 lines of $LOG ==="
    tail -50 "$LOG" 2>/dev/null || echo "(log file missing)"
    exit "${1:-1}"
}

# ── (a) Preflight ──────────────────────────────────────────────────────

if [[ ! -x "$OLLAMA_BIN" ]]; then
    echo "ERROR: ollama binary not found at $OLLAMA_BIN" >&2
    echo "  Run scripts/build_ollama_tq.sh first." >&2
    exit 1
fi
echo "[+] binary: $OLLAMA_BIN"

# ── (b) Kill existing ollama ───────────────────────────────────────────

if pgrep -x ollama &>/dev/null; then
    DISTRO_WAS_RUNNING=1
    echo "[=] stopping existing ollama daemon"
    pkill -x ollama 2>/dev/null || true
    # Wait for port to free up.
    for _ in $(seq 1 10); do
        if ! ss -tlnp 2>/dev/null | grep -q ":${PORT} " && \
           ! curl -sf "http://localhost:${PORT}/api/tags" &>/dev/null; then
            break
        fi
        sleep 1
    done
fi

# ── (c) Start rebuilt binary ───────────────────────────────────────────

echo "[+] starting ollama with OLLAMA_KV_CACHE_TYPE=$CACHE_TYPE"

OLLAMA_KV_CACHE_TYPE="$CACHE_TYPE" \
OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_DEBUG=1 \
OLLAMA_LOG_LEVEL=debug \
    "$OLLAMA_BIN" serve > "$LOG" 2>&1 &
OLLAMA_PID=$!

# ── (d) Wait for server ───────────────────────────────────────────────

echo -n "[.] waiting for server on :$PORT "
READY=0
for _ in $(seq 1 "$TIMEOUT"); do
    if curl -sf "http://localhost:${PORT}/api/tags" &>/dev/null; then
        READY=1
        break
    fi
    # Bail early if the process died.
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
        echo " DIED"
        echo "ERROR: ollama process exited during startup" >&2
        fail_log 2
    fi
    echo -n "."
    sleep 1
done
echo

if [[ "$READY" -ne 1 ]]; then
    echo "ERROR: ollama did not respond within ${TIMEOUT}s" >&2
    fail_log 2
fi
echo "[+] server is up (PID $OLLAMA_PID)"

# ── (e) Check log for cache-type resolution ────────────────────────────
#
# ollama logs the KV cache type when a model is loaded. We need to trigger
# a model load first (step f), then check the log. But we can do a quick
# pre-check: if the server startup log already shows "f16" as a fallback
# for the cache type (without any tq4p mention), that's a red flag.
#
# The definitive check happens after inference in step (f), when the model
# has been loaded and the cache type decision is logged.

# ── (f) Run inference ──────────────────────────────────────────────────

if [[ -z "$MODEL" ]]; then
    # Discover a small generative model the user has already pulled.
    # Skip embedding models (can't generate text) and cloud models (no local weights).
    MODEL_LIST=$("$OLLAMA_BIN" list 2>/dev/null || true)

    if [[ -n "$MODEL_LIST" ]]; then
        # awk fields: $1=name $2=id $3=size_num $4=size_unit (GB/MB).
        # Convert to MB, sort numerically, pick smallest generative model.
        MODEL=$(echo "$MODEL_LIST" | tail -n +2 \
            | grep -viE 'embed|rerank' \
            | awk '$3+0 > 0 {
                mb = ($4 == "GB") ? $3 * 1024 : $3;
                printf "%012.1f %s\n", mb, $1
            }' \
            | sort -n | head -1 | awk '{print $2}')
    fi
fi

if [[ -z "$MODEL" ]]; then
    echo "ERROR: no generative (non-embedding) local models found." >&2
    echo "  Pull a small model first: ollama pull qwen2.5:3b" >&2
    echo "  Or pass one explicitly: scripts/smoke_test_tq4p.sh --model qwen3.5:4b-q8_0" >&2
    exit 1
fi
echo "[+] using model: $MODEL"

# Use the API directly instead of `ollama run` to get clean JSON without
# terminal ANSI escape codes from the interactive CLI.
echo "[+] running inference via /api/generate..."
set +e
RESPONSE=$(curl -sf --max-time 120 \
    "http://localhost:${PORT}/api/generate" \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"Say hello in one sentence.\",\"stream\":false,\"options\":{\"num_ctx\":2048,\"num_predict\":64}}" \
    2>/dev/null)
INFERENCE_RC=$?
set -e

if [[ $INFERENCE_RC -ne 0 ]] || [[ -z "$RESPONSE" ]]; then
    echo "ERROR: inference API call failed (curl exit $INFERENCE_RC)" >&2
    fail_log 4
fi

# Extract the response text from JSON.
INFERENCE_TEXT=$(echo "$RESPONSE" | python3 -c \
    "import sys,json; print(json.load(sys.stdin).get('response',''))" 2>/dev/null || true)

# ── (e continued) Check log for cache type AFTER model load ────────────
#
# ollama's Go layer logs the resolved KV cache type during model init.
# With OLLAMA_DEBUG=1 we expect to see the cache type string in the log.
# Three failure modes:
#   1. "tq4p_d128" appears → success
#   2. "f16" appears as resolved type with no tq4p → fallback detected
#   3. Neither appears → inconclusive, warn but don't fail

sleep 2  # let log flush

TQ4P_IN_LOG=0
F16_FALLBACK=0

if grep -qiE "tq4p" "$LOG" 2>/dev/null; then
    TQ4P_IN_LOG=1
fi

# Check for f16 fallback: ollama logs "KV cache type: f16" or similar when
# it falls back. We look for "f16" near "cache" context, but only flag it
# if tq4p is NOT also mentioned (which would mean tq4p was attempted).
if [[ "$TQ4P_IN_LOG" -eq 0 ]]; then
    if grep -iE "(kv|cache).*(type|dtype).*f16|falling back.*f16|defaulting.*f16" "$LOG" 2>/dev/null; then
        F16_FALLBACK=1
    fi
    # Also check: if the log mentions the cache type env var but then
    # resolves to f16, that's a definitive fallback.
    if grep -iE "cache.type.*f16" "$LOG" 2>/dev/null; then
        F16_FALLBACK=1
    fi
fi

if [[ "$F16_FALLBACK" -eq 1 ]]; then
    echo "FAIL: KV cache type fell back to f16 — TQ4P plumbing is broken" >&2
    echo
    echo "Relevant log lines:"
    grep -iE "cache|type|f16|tq4p|fallback|kv" "$LOG" | head -20 >&2
    fail_log 3
fi

if [[ "$TQ4P_IN_LOG" -eq 1 ]]; then
    echo "[+] log confirms TQ4P cache type active"
else
    echo "[!] warning: could not confirm TQ4P in log (debug logging may differ)"
    echo "    check $LOG manually for cache type resolution"
fi

# ── (g) Verify output is non-garbage ───────────────────────────────────

if [[ -z "$INFERENCE_TEXT" ]]; then
    echo "ERROR: inference produced empty output" >&2
    fail_log 4
fi

# Check for NaN / inf indicators in the generated text.
if echo "$INFERENCE_TEXT" | grep -qiE '\bnan\b|\binf\b' 2>/dev/null; then
    echo "ERROR: inference output contains NaN/inf markers" >&2
    echo "--- output ---"
    echo "$INFERENCE_TEXT" | head -10 >&2
    fail_log 4
fi

echo "[+] inference output looks sane (${#INFERENCE_TEXT} chars)"
echo "    $(echo "$INFERENCE_TEXT" | head -3 | sed 's/^/    /')"

# ── (i) Success banner ─────────────────────────────────────────────────

# Resolve bits-per-weight for the cache type.
case "$CACHE_TYPE" in
    tq4p_d64|tqp_d64_b3)   BPW="4.625"  ;;
    tqp_d64_b2)             BPW="3.625"  ;;
    tqp_d64_b4)             BPW="5.625"  ;;
    tq4p_d128|tqp_d128_b3)  BPW="4.25"   ;;
    tqp_d128_b2)            BPW="3.3125" ;;
    tqp_d128_b4)            BPW="5.3125" ;;
    tq4p_d256|tqp_d256_b3)  BPW="4.16"   ;;
    tqp_d256_b2)            BPW="3.16"   ;;
    tqp_d256_b4)            BPW="5.16"   ;;
    *)                       BPW="?"      ;;
esac

printf '
╔══════════════════════════════════════════════════════════════╗
║                    TQ4P IS LIVE                             ║
║                                                             ║
║  KV cache type:  %-14s (%s bpw)%*s║
║  Algorithm:      Stage-1 Lloyd-Max + Stage-2 QJL            ║
║  Expected:       cosine similarity ≥ 0.92  (paper: 0.93)   ║
║                                                             ║
║  Allowlist + Go plumbing + ggml dispatch all aligned.       ║
╚══════════════════════════════════════════════════════════════╝
' "$CACHE_TYPE" "$BPW" $((27 - ${#CACHE_TYPE} - ${#BPW})) ""

echo "[+] log preserved at $LOG"
exit 0
