#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/yannik/miniconda3/envs/vllm-serve/bin/python}"
PIP_BIN="${PIP_BIN:-/home/yannik/miniconda3/envs/vllm-serve/bin/pip}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8003}"
DTYPE="${DTYPE:-float16}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
TQ_USE_TRITON_VALUE="${TQ_USE_TRITON_VALUE:-1}"
TQ_NUM_KV_SPLITS_VALUE="${TQ_NUM_KV_SPLITS_VALUE:-8}"
INSTALL_PLUGIN="${INSTALL_PLUGIN:-1}"
HYBRID="${HYBRID:-0}"
TOKENIZER="${TOKENIZER:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MODEL_REF=""

usage() {
  cat <<'EOF'
Usage:
  ./serve_ollama_tq.sh <ollama-model> [--tp N] [--hybrid] [--tokenizer HF_ID] [--max-len N]

Examples:
  harbor ollama pull qwen2.5-coder:32b
  ./serve_ollama_tq.sh qwen2.5-coder:32b
  ./serve_ollama_tq.sh llama3.3 --tp 2
  ./serve_ollama_tq.sh qwen2.5-coder:32b --hybrid

Model storage:
  Uses OLLAMA_MODELS when set. Otherwise reads HARBOR_OLLAMA_CACHE or
  $HARBOR_HOME/.env, then falls back to ~/.ollama/models.

Environment overrides:
  PYTHON_BIN, PIP_BIN, HOST, PORT, DTYPE, TP_SIZE, MAX_MODEL_LEN
  GPU_MEMORY_UTILIZATION, CUDA_VISIBLE_DEVICES_VALUE, TOKENIZER
  TQ_USE_TRITON_VALUE, TQ_NUM_KV_SPLITS_VALUE, INSTALL_PLUGIN, HYBRID
  OLLAMA_MODELS, HARBOR_OLLAMA_CACHE, HARBOR_HOME
EOF
}

parse_args() {
  while (($#)); do
    case "$1" in
      --tp|--tensor-parallel-size)
        TP_SIZE="${2:?missing value for $1}"
        shift 2
        ;;
      --hybrid)
        HYBRID=1
        shift
        ;;
      --tokenizer)
        TOKENIZER="${2:?missing value for --tokenizer}"
        shift 2
        ;;
      --max-len|--max-model-len)
        MAX_MODEL_LEN="${2:?missing value for $1}"
        shift 2
        ;;
      --host)
        HOST="${2:?missing value for --host}"
        shift 2
        ;;
      --port)
        PORT="${2:?missing value for --port}"
        shift 2
        ;;
      -h|--help|help)
        usage
        exit 0
        ;;
      -*)
        echo "unknown option: $1" >&2
        usage
        exit 2
        ;;
      *)
        if [[ -n "${MODEL_REF}" ]]; then
          echo "unexpected extra argument: $1" >&2
          usage
          exit 2
        fi
        MODEL_REF="$1"
        shift
        ;;
    esac
  done

  if [[ -z "${MODEL_REF}" ]]; then
    usage
    exit 2
  fi
}

install_plugin() {
  if [[ "${INSTALL_PLUGIN}" == "1" ]]; then
    "${PIP_BIN}" install -e "${ROOT_DIR}"
  fi
}

export_common_env() {
  export LD_LIBRARY_PATH="/home/yannik/miniconda3/envs/vllm-serve/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export PATH="/home/yannik/miniconda3/envs/vllm-serve/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin${PATH:+:${PATH}}"
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export NCCL_P2P_DISABLE=1
  export TQ_USE_TRITON="${TQ_USE_TRITON_VALUE}"
  export TQ_NUM_KV_SPLITS="${TQ_NUM_KV_SPLITS_VALUE}"
  export TQ_PATCH_KV=1
  export TQ_HYBRID="${HYBRID}"

  if [[ -z "${CUDA_VISIBLE_DEVICES_VALUE+x}" ]]; then
    if ((TP_SIZE > 1)); then
      CUDA_VISIBLE_DEVICES_VALUE="0,1"
    else
      CUDA_VISIBLE_DEVICES_VALUE="1"
    fi
  fi
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
}

resolve_ollama_model() {
  local resolved_json
  resolved_json="$("${PYTHON_BIN}" "${ROOT_DIR}/ollama_resolver.py" "${MODEL_REF}")"

  GGUF_PATH="$(
    "${PYTHON_BIN}" -c 'import json, sys; print(json.load(sys.stdin)["gguf_path"])' \
      <<<"${resolved_json}"
  )"
  TOKENIZER_HINT="$(
    "${PYTHON_BIN}" -c 'import json, sys; print(json.load(sys.stdin).get("tokenizer") or "")' \
      <<<"${resolved_json}"
  )"

  while IFS='=' read -r key value; do
    if [[ -n "${key}" && -z "${!key+x}" ]]; then
      export "${key}=${value}"
    fi
  done < <(
    "${PYTHON_BIN}" -c '
import json
import sys

data = json.load(sys.stdin)
for key, value in data["tq_env"].items():
    print(f"{key}={value}")
' <<<"${resolved_json}"
  )

  export TQ_GGUF_PATH="${GGUF_PATH}"
  TOKENIZER="${TOKENIZER:-${TOKENIZER_HINT}}"
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-${TQ_MAX_SEQ_LEN:-32768}}"
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_REF}-turboquant}"
}

serve_cmd() {
  local tokenizer_args=()
  if [[ -n "${TOKENIZER}" ]]; then
    tokenizer_args=(--tokenizer "${TOKENIZER}")
  fi

  echo "Serving ${MODEL_REF} from ${GGUF_PATH}"
  echo "TurboQuant config: layers=${TQ_NUM_LAYERS:-?} heads=${TQ_NUM_HEADS:-?} kv_heads=${TQ_NUM_KV_HEADS:-?} head_dim=${TQ_HEAD_DIM:-?} max_len=${MAX_MODEL_LEN}"

  "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
    --model "${GGUF_PATH}" \
    --load-format gguf \
    "${tokenizer_args[@]}" \
    --attention-backend CUSTOM \
    --enforce-eager \
    --dtype "${DTYPE}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --disable-custom-all-reduce \
    --host "${HOST}" \
    --port "${PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --trust-remote-code
}

main() {
  parse_args "$@"
  install_plugin
  export_common_env
  resolve_ollama_model
  serve_cmd
}

main "$@"
