#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_BIN="${PIP_BIN:-pip}"

MODE="${1:-turboquant}"
MODEL="${MODEL:-casperhansen/llama-3.3-70b-instruct-awq}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-llama33-70b-awq-turboquant}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
TP_SIZE="${TP_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0,1}"
DTYPE="${DTYPE:-float16}"
QUANTIZATION="${QUANTIZATION:-awq_marlin}"

usage() {
  cat <<'EOF'
Usage:
  bash run_llama70b_tq.sh baseline
  bash run_llama70b_tq.sh turboquant

Environment overrides:
  PYTHON_BIN, PIP_BIN, MODEL, SERVED_MODEL_NAME, PORT, HOST, TP_SIZE
  MAX_MODEL_LEN, GPU_MEMORY_UTILIZATION, CUDA_VISIBLE_DEVICES_VALUE
  DTYPE, QUANTIZATION
EOF
}

install_plugin() {
  "${PIP_BIN}" install -e "${ROOT_DIR}"
}

export_common_env() {
  export LD_LIBRARY_PATH="$(dirname "$(${PYTHON_BIN} -c 'import sys; print(sys.executable)')")/../lib:${LD_LIBRARY_PATH:-}"
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
  export NCCL_P2P_DISABLE=1
}

export_turboquant_env() {
  export TQ_HYBRID=0
  export TQ_B_MSE=2
  export TQ_B_QJL=1
  export TQ_USE_TRITON=1
  export TQ_NUM_KV_SPLITS=8
  export TQ_PATCH_KV=1
}

run_server() {
  local attention_args=()
  if [[ "${MODE}" == "turboquant" ]]; then
    attention_args=(--attention-backend CUSTOM)
  fi

  "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --quantization "${QUANTIZATION}" \
    --dtype "${DTYPE}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --disable-custom-all-reduce \
    --host "${HOST}" \
    --port "${PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --trust-remote-code \
    --enforce-eager \
    "${attention_args[@]}"
}

main() {
  case "${MODE}" in
    baseline)
      install_plugin
      export_common_env
      run_server
      ;;
    turboquant)
      install_plugin
      export_common_env
      export_turboquant_env
      run_server
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      echo "unknown mode: ${MODE}" >&2
      usage
      exit 2
      ;;
  esac
}

main
