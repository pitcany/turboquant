#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SERVER_PID=""
PYTHON_BIN="${PYTHON_BIN:-/home/yannik/miniconda3/envs/vllm-serve/bin/python}"
PIP_BIN="${PIP_BIN:-/home/yannik/miniconda3/envs/vllm-serve/bin/pip}"

MODEL="${MODEL:-casperhansen/llama-3.3-70b-instruct-awq}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-llama-3.3-70b-turboquant}"
PORT="${PORT:-8003}"
HOST="${HOST:-0.0.0.0}"
QUANTIZATION="${QUANTIZATION:-awq_marlin}"
DTYPE="${DTYPE:-float16}"
TP_SIZE="${TP_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0,1}"
TQ_USE_TRITON_VALUE="${TQ_USE_TRITON_VALUE:-1}"
TQ_NUM_KV_SPLITS_VALUE="${TQ_NUM_KV_SPLITS_VALUE:-8}"
REQUESTS="${REQUESTS:-5}"
WARMUP="${WARMUP:-1}"
MAX_TOKENS="${MAX_TOKENS:-128}"
PROMPT="${PROMPT:-Summarize recursion in 3 bullet points.}"
LOG_FILE="${LOG_FILE:-/tmp/turboquant-vllm-${PORT}.log}"

usage() {
  cat <<'EOF'
Usage:
  bash serve_turboquant_vllm.sh serve
  bash serve_turboquant_vllm.sh smoke
  bash serve_turboquant_vllm.sh bench
  bash serve_turboquant_vllm.sh all

Modes:
  serve  Install plugin deps, export TurboQuant env, and run vLLM in foreground.
  smoke  Hit /v1/models and /v1/chat/completions on the configured server.
  bench  Run benchmark_openai.py against the configured server.
  all    Start the server in background, wait for readiness, run smoke + bench, then stop it.

Environment overrides:
  MODEL, SERVED_MODEL_NAME, PORT, HOST, QUANTIZATION, DTYPE, TP_SIZE
  MAX_MODEL_LEN, GPU_MEMORY_UTILIZATION, CUDA_VISIBLE_DEVICES_VALUE
  TQ_USE_TRITON_VALUE, TQ_NUM_KV_SPLITS_VALUE
  REQUESTS, WARMUP, MAX_TOKENS, PROMPT, LOG_FILE
EOF
}

export_common_env() {
  export LD_LIBRARY_PATH="/home/yannik/miniconda3/envs/vllm-serve/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export PATH="/home/yannik/miniconda3/envs/vllm-serve/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin${PATH:+:${PATH}}"
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
  export NCCL_P2P_DISABLE=1
  export TQ_USE_TRITON="${TQ_USE_TRITON_VALUE}"
  export TQ_NUM_KV_SPLITS="${TQ_NUM_KV_SPLITS_VALUE}"
}

install_plugin() {
  "${PIP_BIN}" install -e "${ROOT_DIR}"
}

serve_cmd() {
  "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --quantization "${QUANTIZATION}" \
    --dtype "${DTYPE}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --disable-custom-all-reduce \
    --attention-backend CUSTOM \
    --enforce-eager \
    --host "${HOST}" \
    --port "${PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --trust-remote-code
}

wait_for_server() {
  local base_url="http://127.0.0.1:${PORT}/v1/models"
  local tries=120
  for ((i=1; i<=tries; i++)); do
    if curl -sf --max-time 2 "${base_url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "server did not become ready; check ${LOG_FILE}" >&2
  return 1
}

smoke_test() {
  curl -s "http://127.0.0.1:${PORT}/v1/models" | python3 -m json.tool
  curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(python3 - <<PY
import json
print(json.dumps({
  "model": "${SERVED_MODEL_NAME}",
  "messages": [{"role": "user", "content": ${PROMPT@Q}}],
  "max_tokens": ${MAX_TOKENS},
  "temperature": 0.0,
}))
PY
)" | python3 -m json.tool
}

run_bench() {
  "${PYTHON_BIN}" "${ROOT_DIR}/benchmark_openai.py" \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --model "${SERVED_MODEL_NAME}" \
    --prompt "${PROMPT}" \
    --max-tokens "${MAX_TOKENS}" \
    --warmup "${WARMUP}" \
    --requests "${REQUESTS}"
}

run_server_foreground() {
  install_plugin
  export_common_env
  serve_cmd
}

run_all() {
  install_plugin
  export_common_env

  : > "${LOG_FILE}"
  serve_cmd >"${LOG_FILE}" 2>&1 &
  _SERVER_PID=$!
  trap '[ -n "${_SERVER_PID}" ] && kill "${_SERVER_PID}" >/dev/null 2>&1 || true' EXIT

  wait_for_server
  smoke_test
  run_bench

  kill "${_SERVER_PID}" >/dev/null 2>&1 || true
  wait "${_SERVER_PID}" || true
  trap - EXIT
}

main() {
  local mode="${1:-serve}"
  case "${mode}" in
    serve)
      run_server_foreground
      ;;
    smoke)
      smoke_test
      ;;
    bench)
      run_bench
      ;;
    all)
      run_all
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      echo "unknown mode: ${mode}" >&2
      usage
      exit 2
      ;;
  esac
}

main "${1:-serve}"
