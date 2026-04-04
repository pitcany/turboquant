#!/usr/bin/env bash
# Revert Harbor vLLM to stock (pre-TurboQuant) configuration.
#
# Usage:
#   bash harbor-revert.sh
#
# This restores:
#   - The original Dockerfile and override.env
#   - Harbor config (attention backend, extra args)
#   - Removes the TurboQuant source copy from the build context
set -euo pipefail

HARBOR_VLLM_DIR="${HOME}/.harbor/services/vllm"

echo "Restoring original Dockerfile..."
if [[ -f "${HARBOR_VLLM_DIR}/Dockerfile.pre-turboquant" ]]; then
  cp "${HARBOR_VLLM_DIR}/Dockerfile.pre-turboquant" "${HARBOR_VLLM_DIR}/Dockerfile"
else
  echo "  WARNING: backup not found, skipping"
fi

echo "Restoring original override.env..."
if [[ -f "${HARBOR_VLLM_DIR}/override.env.pre-turboquant" ]]; then
  cp "${HARBOR_VLLM_DIR}/override.env.pre-turboquant" "${HARBOR_VLLM_DIR}/override.env"
else
  echo "  WARNING: backup not found, skipping"
fi

echo "Removing TurboQuant source from build context..."
rm -rf "${HARBOR_VLLM_DIR}/turboquant"

echo "Restoring Harbor config..."
harbor config set vllm.attention_backend FLASH_ATTN
harbor vllm args '--tensor-parallel-size 2 --max-model-len 8192 --enforce-eager --gpu-memory-utilization 0.94'

echo "Rebuilding vLLM image without TurboQuant..."
harbor build vllm

echo ""
echo "Done. Harbor vLLM is back to stock configuration."
echo "Run 'harbor up vllm' to start the standard vLLM server."
