#!/usr/bin/env bash
# Apply TurboQuant to Harbor vLLM configuration.
#
# Usage:
#   bash harbor-apply.sh
#
# This will:
#   - Back up original Dockerfile and override.env (if not already backed up)
#   - Copy TurboQuant source into the Harbor vLLM build context
#   - Patch the Dockerfile to install TurboQuant
#   - Set TurboQuant environment variables
#   - Configure Harbor vLLM settings
#   - Rebuild the vLLM image
set -euo pipefail

TQ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARBOR_VLLM_DIR="${HOME}/.harbor/services/vllm"

# Back up originals (only if no backup exists yet)
if [[ ! -f "${HARBOR_VLLM_DIR}/Dockerfile.pre-turboquant" ]]; then
  echo "Backing up original Dockerfile..."
  cp "${HARBOR_VLLM_DIR}/Dockerfile" "${HARBOR_VLLM_DIR}/Dockerfile.pre-turboquant"
fi
if [[ ! -f "${HARBOR_VLLM_DIR}/override.env.pre-turboquant" ]]; then
  echo "Backing up original override.env..."
  cp "${HARBOR_VLLM_DIR}/override.env" "${HARBOR_VLLM_DIR}/override.env.pre-turboquant"
fi

echo "Copying TurboQuant source into build context..."
mkdir -p "${HARBOR_VLLM_DIR}/turboquant/vllm_plugin"
cp "${TQ_DIR}"/{setup.py,turboquant.py,lloyd_max.py,compressors.py,__init__.py} \
   "${HARBOR_VLLM_DIR}/turboquant/"
cp "${TQ_DIR}"/vllm_plugin/*.py \
   "${HARBOR_VLLM_DIR}/turboquant/vllm_plugin/"

echo "Patching Dockerfile..."
cat > "${HARBOR_VLLM_DIR}/Dockerfile" <<'DOCKERFILE'
ARG HARBOR_VLLM_VERSION=latest
ARG HARBOR_VLLM_IMAGE=vllm/vllm-openai

FROM ${HARBOR_VLLM_IMAGE}:${HARBOR_VLLM_VERSION}

# Install:
# - bitsandbytes for additional quantization support
RUN pip install bitsandbytes

# TurboQuant KV cache compression plugin
COPY turboquant/ /opt/turboquant/
RUN pip install -e /opt/turboquant
DOCKERFILE

echo "Setting TurboQuant environment variables..."
cat > "${HARBOR_VLLM_DIR}/override.env" <<'ENV'
# TurboQuant configuration
TQ_USE_TRITON=1
TQ_NUM_KV_SPLITS=8

# Required for mixed-architecture multi-GPU (e.g. RTX 4090 + RTX 5090)
NCCL_P2P_DISABLE=1
CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV

echo "Configuring Harbor..."
harbor config set vllm.attention_backend CUSTOM
harbor vllm args '--quantization awq_marlin --dtype float16 --tensor-parallel-size 2 --max-model-len 32768 --gpu-memory-utilization 0.92 --disable-custom-all-reduce --enforce-eager --trust-remote-code'

echo "Rebuilding vLLM image with TurboQuant..."
harbor build vllm

echo ""
echo "Done. Run 'harbor up vllm' to start with TurboQuant."
