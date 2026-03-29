"""
TurboQuant vLLM Plugin — 3-bit KV Cache Compression

TurboQuant combines PolarQuant (MSE-optimal scalar quantization) with
QJL (1-bit Quantized Johnson-Lindenstrauss residual correction) to
achieve ~4x KV cache compression with near-zero accuracy loss.

Reference: https://arxiv.org/abs/2504.19874
"""

__version__ = "0.1.0"
__author__ = "TurboQuant Contributors"
__license__ = "MIT"

from vllm_plugin.config import TurboQuantConfig

__all__ = [
    "__version__",
    "TurboQuantConfig",
]
