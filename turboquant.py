"""
TurboQuant: Two-stage vector quantization with near-optimal distortion.

Stage 1 (MSE): Random rotation + per-coordinate Lloyd-Max quantization
Stage 2 (QJL): 1-bit Quantized Johnson-Lindenstrauss on residuals for unbiased inner products

Reference: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (ICLR 2026)
"""

import torch
import torch.nn as nn
import math
from typing import Optional

from lloyd_max import LloydMaxCodebook

_NORM_EPS = 1e-8


def generate_rotation_matrix(d: int, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    """
    Generate a random orthogonal rotation matrix via QR decomposition of Gaussian matrix.
    This is the Haar-distributed random rotation used in TurboQuant.
    """
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    # Generate random Gaussian matrix and QR decompose
    G = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(G)
    # Ensure proper rotation (det = +1) by fixing sign ambiguity in QR
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device)


def generate_qjl_matrix(d: int, m: Optional[int] = None, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    """
    Generate the random projection matrix S for QJL.
    S has i.i.d. N(0,1) entries, shape (m, d).
    Default m = d (same dimensionality).
    """
    if m is None:
        m = d
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    S = torch.randn(m, d, generator=gen)
    return S.to(device)


class TurboQuantMSE(nn.Module):
    """
    Stage 1: MSE-optimal quantizer.
    Randomly rotates, then applies per-coordinate Lloyd-Max quantization.
    """

    def __init__(self, d: int, bits: int, seed: int = 42, device: str = "cpu") -> None:
        super().__init__()
        self.d = d
        self.bits = bits
        self.device = device

        # Precompute rotation matrix
        self.register_buffer("Pi", generate_rotation_matrix(d, seed=seed, device=device))

        # Precompute Lloyd-Max codebook
        self.codebook = LloydMaxCodebook(d, bits)
        self.register_buffer("centroids", self.codebook.centroids.to(device))
        self.register_buffer("boundaries", self.codebook.boundaries.to(device))

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random rotation: y = Pi @ x."""
        # x: (batch, d) or (d,)
        return x @ self.Pi.T

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        """Undo rotation: x = Pi^T @ y."""
        return y @ self.Pi

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize vectors to codebook indices. Returns integer indices."""
        y = self.rotate(x)
        return torch.bucketize(y, self.boundaries.to(y.device))

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """Dequantize indices back to vectors."""
        y_hat = self.centroids[indices]  # (..., d)
        return self.unrotate(y_hat)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full quantize-dequantize cycle.
        Returns: (reconstructed_x, indices)
        """
        indices = self.quantize(x)
        x_hat = self.dequantize(indices)
        return x_hat, indices


class TurboQuantProd(nn.Module):
    """
    Stage 1 + Stage 2: Unbiased inner product quantizer.
    Uses (b-1)-bit MSE quantizer + 1-bit QJL on residuals.

    Total storage per vector: (b-1)*d bits for MSE indices + d bits for QJL signs + 16 bits for residual norm
    Effective: ~b bits per dimension (the QJL bit replaces one MSE bit)
    """

    def __init__(self, d: int, bits: int, qjl_dim: Optional[int] = None, seed: int = 42, device: str = "cpu") -> None:
        """
        Args:
            d: vector dimension
            bits: total bit budget per coordinate (MSE uses bits-1, QJL uses 1)
            qjl_dim: projection dimension for QJL (default = d)
            seed: random seed for reproducibility
            device: torch device
        """
        super().__init__()
        self.d = d
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.qjl_dim = qjl_dim or d
        self.device = device

        # Stage 1: MSE quantizer with (bits-1) bits
        self.mse = TurboQuantMSE(d, self.mse_bits, seed=seed, device=device)

        # Stage 2: QJL projection matrix
        self.register_buffer("S", generate_qjl_matrix(d, m=self.qjl_dim, seed=seed + 1, device=device))

    def quantize(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Full TurboQuant quantization.

        Returns dict with:
            - 'mse_indices': (batch, d) int tensor, MSE codebook indices
            - 'qjl_signs': (batch, qjl_dim) sign bits of QJL-projected residual
            - 'residual_norm': (batch,) L2 norm of residual
        """
        # Stage 1: MSE quantize
        x_hat, mse_indices = self.mse(x)

        # Compute residual
        residual = x - x_hat
        residual_norm = torch.norm(residual, dim=-1, keepdim=True).clamp_min(_NORM_EPS)

        # Stage 2: QJL - project residual and take sign
        projected = residual @ self.S.T  # (batch, qjl_dim)
        qjl_signs = torch.sign(projected)  # (batch, qjl_dim)
        qjl_signs[qjl_signs == 0] = 1.0  # map zeros to +1

        return {
            "mse_indices": mse_indices,
            "qjl_signs": qjl_signs,
            "residual_norm": residual_norm.squeeze(-1),
        }

    def dequantize(self, compressed: dict[str, torch.Tensor]) -> torch.Tensor:
        """Dequantize MSE component (for reconstruction)."""
        return self.mse.dequantize(compressed["mse_indices"])

    def inner_product(self, y: torch.Tensor, compressed: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute unbiased inner product estimate: <y, x> using compressed representation of x.

        The estimator is:
            <y, x_mse> + ||r|| * sqrt(pi/2) / m * <S @ y, qjl_signs>

        Args:
            y: query vectors (batch, d) or (d,)
            compressed: dict from quantize()

        Returns:
            Estimated inner products (batch,)
        """
        # Term 1: inner product with MSE reconstruction
        x_mse = self.mse.dequantize(compressed["mse_indices"])
        term1 = (y * x_mse).sum(dim=-1)

        # Term 2: QJL correction
        # Project query with same S matrix (but don't quantize query)
        y_projected = y @ self.S.T  # (batch, qjl_dim)
        qjl_ip = (y_projected * compressed["qjl_signs"]).sum(dim=-1)

        m = self.qjl_dim
        correction_scale = math.sqrt(math.pi / 2) / m
        term2 = compressed["residual_norm"] * correction_scale * qjl_ip

        return term1 + term2

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Quantize input vectors."""
        return self.quantize(x)


class TurboQuantKVCache:
    """
    KV cache wrapper that uses TurboQuant to compress keys and values.
    Drop-in replacement concept for a standard KV cache.
    """

    def __init__(self, d_key: int, d_value: int, bits: int = 3, seed: int = 42, device: str = "cpu") -> None:
        self.d_key = d_key
        self.d_value = d_value
        self.bits = bits
        self.device = device

        # Use TurboQuantProd for keys (need inner products for attention)
        self.key_quantizer = TurboQuantProd(d_key, bits, seed=seed, device=device)
        # Use TurboQuantMSE for values (need MSE reconstruction, not inner products)
        self.value_quantizer = TurboQuantMSE(d_value, bits, seed=seed + 100, device=device)

        # Storage
        self.key_cache = []    # list of compressed key dicts
        self.value_cache = []  # list of (indices,) tuples

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """
        Append new key-value pairs to cache.
        keys: (batch, seq_len, d_key) or (seq_len, d_key)
        values: (batch, seq_len, d_value) or (seq_len, d_value)
        """
        orig_shape = keys.shape
        flat_keys = keys.reshape(-1, self.d_key)
        flat_values = values.reshape(-1, self.d_value)

        key_norms = torch.norm(flat_keys, dim=-1).clamp_min(_NORM_EPS)
        value_norms = torch.norm(flat_values, dim=-1).clamp_min(_NORM_EPS)
        key_units = flat_keys / key_norms.unsqueeze(-1)
        value_units = flat_values / value_norms.unsqueeze(-1)

        compressed_keys = self.key_quantizer.quantize(key_units)
        value_indices = self.value_quantizer.quantize(value_units)

        self.key_cache.append({
            "mse_indices": compressed_keys["mse_indices"],
            "qjl_signs": compressed_keys["qjl_signs"],
            "residual_norm": compressed_keys["residual_norm"],
            "key_norms": key_norms,
            "shape": orig_shape,
        })
        self.value_cache.append({
            "indices": value_indices,
            "value_norms": value_norms,
            "shape": values.shape,
        })

    def attention_scores(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Compute attention scores between queries and all cached keys.
        Uses unbiased inner product estimation via TurboQuant.

        queries: (batch, d_key) or (d_key,)
        Returns: scores for each cached position
        """
        scores = []
        for cached in self.key_cache:
            s = self.key_quantizer.inner_product(queries, cached) * cached["key_norms"]
            scores.append(s)
        if scores:
            return torch.cat(scores, dim=-1)
        return torch.empty(0, device=queries.device, dtype=queries.dtype)

    def get_values(self) -> torch.Tensor:
        """Reconstruct all cached values."""
        values = []
        for cached in self.value_cache:
            v = self.value_quantizer.dequantize(cached["indices"])
            v = v * cached["value_norms"].unsqueeze(-1)
            values.append(v)
        if values:
            return torch.cat(values, dim=0)
        return torch.empty(0, self.d_value, device=self.value_quantizer.Pi.device)

    def memory_usage_bits(self) -> dict[str, float]:
        """Estimate memory usage in bits."""
        n_keys = sum(c["mse_indices"].numel() for c in self.key_cache) if self.key_cache else 0
        n_qjl = sum(c["qjl_signs"].numel() for c in self.key_cache) if self.key_cache else 0
        n_residual_norms = sum(c["residual_norm"].numel() for c in self.key_cache) if self.key_cache else 0
        n_key_norms = sum(c["key_norms"].numel() for c in self.key_cache) if self.key_cache else 0
        n_values = sum(c["indices"].numel() for c in self.value_cache) if self.value_cache else 0
        n_value_norms = sum(c["value_norms"].numel() for c in self.value_cache) if self.value_cache else 0

        key_bits = (
            n_keys * self.key_quantizer.mse_bits
            + n_qjl
            + n_residual_norms * 16
            + n_key_norms * 16
        )
        value_bits = n_values * self.bits + n_value_norms * 16
        fp16_equivalent = (n_keys + n_values) * 16  # what fp16 would cost

        return {
            "key_bits": key_bits,
            "value_bits": value_bits,
            "total_bits": key_bits + value_bits,
            "fp16_bits": fp16_equivalent,
            "compression_ratio": fp16_equivalent / (key_bits + value_bits) if (key_bits + value_bits) > 0 else 0,
        }

    def __len__(self) -> int:
        return sum(c["mse_indices"].shape[0] for c in self.key_cache) if self.key_cache else 0
