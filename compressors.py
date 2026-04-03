"""
TurboQuant KV cache compressors built on the core quantizer classes.

These wrappers preserve the higher-level tensor APIs used by validation and
research scripts while delegating codebook construction and quantization math
to ``TurboQuantMSE`` and ``TurboQuantProd``.
"""

from __future__ import annotations

import math

import torch

from turboquant import TurboQuantMSE, TurboQuantProd

_NORM_EPS = 1e-8


class TurboQuantCompressorV2:
    """
    Compressor that stores compressed representations and supports direct inner
    product computation without full decompression.
    """

    def __init__(self, head_dim: int, bits: int, seed: int, device: str = "cpu") -> None:
        self.head_dim = head_dim
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.device = device

        self.quantizer = TurboQuantProd(head_dim, bits, seed=seed, device=device)
        self.Pi = self.quantizer.mse.Pi
        self.centroids = self.quantizer.mse.centroids
        self.S = self.quantizer.S

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict[str, torch.Tensor | tuple[int, ...]]:
        """
        Compress states: (batch, heads, seq, head_dim) -> compressed dict.
        Stores everything needed for asymmetric inner product computation.
        """
        batch, heads, seq_len, dim = states.shape
        flat = states.reshape(-1, dim).float()

        vec_norms = torch.norm(flat, dim=-1).clamp_min(_NORM_EPS)
        flat_unit = flat / vec_norms.unsqueeze(-1)
        compressed = self.quantizer.quantize(flat_unit)

        k_mse = self.quantizer.dequantize(compressed) * vec_norms.unsqueeze(-1)
        residual_norm = compressed["residual_norm"] * vec_norms

        return {
            "k_mse": k_mse.to(torch.float16).reshape(batch, heads, seq_len, dim),
            "qjl_signs": compressed["qjl_signs"].reshape(batch, heads, seq_len, self.quantizer.qjl_dim),
            "residual_norm": residual_norm.to(torch.float16).reshape(batch, heads, seq_len),
            "shape": (batch, heads, seq_len, dim),
        }

    @torch.no_grad()
    def asymmetric_attention_scores(
        self,
        queries: torch.Tensor,
        compressed: dict[str, torch.Tensor | tuple[int, ...]],
    ) -> torch.Tensor:
        """
        Compute attention scores <Q, K> directly from compressed K.

        Returns:
            scores: (batch, heads, seq_q, seq_k)
        """
        k_mse = compressed["k_mse"].float()
        signs = compressed["qjl_signs"].float()
        residual_norm = compressed["residual_norm"].float()

        term1 = torch.matmul(queries.float(), k_mse.transpose(-2, -1))
        q_projected = torch.matmul(queries.float(), self.S.T)
        qjl_ip = torch.matmul(q_projected, signs.transpose(-2, -1))

        correction_scale = math.sqrt(math.pi / 2) / self.S.shape[0]
        term2 = correction_scale * qjl_ip * residual_norm.unsqueeze(-2)

        return term1 + term2


class TurboQuantCompressorMSE:
    """MSE-only compressor for values or MSE-only key ablations."""

    def __init__(self, head_dim: int, bits: int, seed: int, device: str = "cpu") -> None:
        self.head_dim = head_dim
        self.bits = bits
        self.device = device

        self.quantizer = TurboQuantMSE(head_dim, bits, seed=seed, device=device)
        self.Pi = self.quantizer.Pi
        self.centroids = self.quantizer.centroids

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict[str, torch.Tensor | tuple[int, ...]]:
        batch, heads, seq_len, dim = states.shape
        flat = states.reshape(-1, dim).float()
        vec_norms = torch.norm(flat, dim=-1).clamp_min(_NORM_EPS)
        flat_unit = flat / vec_norms.unsqueeze(-1)
        indices = self.quantizer.quantize(flat_unit)

        return {
            "indices": indices,
            "vec_norms": vec_norms.to(torch.float16),
            "shape": (batch, heads, seq_len, dim),
        }

    @torch.no_grad()
    def decompress(self, compressed: dict[str, torch.Tensor | tuple[int, ...]]) -> torch.Tensor:
        batch, heads, seq_len, dim = compressed["shape"]
        indices = compressed["indices"].long()
        reconstructed = self.quantizer.dequantize(indices)
        vec_norms = compressed["vec_norms"].float().unsqueeze(-1)
        return (reconstructed * vec_norms).reshape(batch, heads, seq_len, dim)
