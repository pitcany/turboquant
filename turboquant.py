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

try:
    from .lloyd_max import LloydMaxCodebook
except ImportError:  # Support direct module imports from the repo root.
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


def generate_sign_vector(d: int, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    """Generate a random ±1 sign vector for the randomized Hadamard transform."""
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    return (2 * torch.randint(0, 2, (d,), generator=gen) - 1).float().to(device)


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform along the last dimension.

    Returns a new tensor (not in-place).
    Operates on the unnormalized transform (multiply by 1/sqrt(d) after).
    Requires the last dimension to be a power of 2.

    Uses vectorized reshape+stack butterfly steps — each step is one
    batched tensor operation instead of a Python loop over index pairs.
    """
    d = x.shape[-1]
    batch_shape = x.shape[:-1]
    h = 1
    while h < d:
        x = x.view(*batch_shape, d // (2 * h), 2, h)
        a = x[..., 0, :]
        b = x[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2)
        h <<= 1
    return x.view(*batch_shape, d)


def wht_rotate(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Apply randomized Hadamard rotation: y = (1/sqrt(d)) * WHT(sigma * x)."""
    d = x.shape[-1]
    y = x * sigma
    y = fwht(y)
    return y * (1.0 / math.sqrt(d))


def wht_unrotate(y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Inverse randomized Hadamard rotation: x = sigma * (1/sqrt(d)) * WHT(y)."""
    d = y.shape[-1]
    x = fwht(y)
    return sigma * x * (1.0 / math.sqrt(d))


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

    Supports two rotation modes:
      - "haar": Dense random orthogonal matrix (O(d²) per vector)
      - "wht":  Randomized Walsh-Hadamard Transform (O(d log d) per vector)
    """

    def __init__(self, d: int, bits: int, seed: int = 42, device: str = "cpu",
                 rotation: str = "haar") -> None:
        super().__init__()
        self.d = d
        self.bits = bits
        self.device = device
        self.rotation = rotation

        if rotation == "haar":
            self.register_buffer("Pi", generate_rotation_matrix(d, seed=seed, device=device))
        elif rotation == "wht":
            self.register_buffer("sigma", generate_sign_vector(d, seed=seed, device=device))
            # Materialize the WHT rotation matrix so downstream consumers
            # (e.g. vLLM plugin) can use Pi / Pi.T for rotation/unrotation.
            wht_pi_t = wht_rotate(torch.eye(d, device=device), self.sigma)
            self.register_buffer("Pi", wht_pi_t.T.contiguous())
        else:
            raise ValueError(f"rotation must be 'haar' or 'wht', got {rotation!r}")

        # Precompute Lloyd-Max codebook
        self.codebook = LloydMaxCodebook(d, bits)
        self.register_buffer("centroids", self.codebook.centroids.to(device))
        self.register_buffer("boundaries", self.codebook.boundaries.to(device))

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random rotation."""
        if self.rotation == "wht":
            return wht_rotate(x, self.sigma)
        return x @ self.Pi.T

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        """Undo rotation."""
        if self.rotation == "wht":
            return wht_unrotate(y, self.sigma)
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

    def __init__(self, d: int, bits: int, qjl_dim: Optional[int] = None, seed: int = 42,
                 device: str = "cpu", rotation: str = "haar") -> None:
        """
        Args:
            d: vector dimension
            bits: total bit budget per coordinate (MSE uses bits-1, QJL uses 1)
            qjl_dim: projection dimension for QJL (default = d)
            seed: random seed for reproducibility
            device: torch device
            rotation: "haar" (dense, O(d²)) or "wht" (Walsh-Hadamard, O(d log d))
        """
        super().__init__()
        self.d = d
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.qjl_dim = qjl_dim or d
        self.device = device
        self.rotation = rotation

        # Stage 1: MSE quantizer with (bits-1) bits
        self.mse = TurboQuantMSE(d, self.mse_bits, seed=seed, device=device, rotation=rotation)

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
        m = self.qjl_dim
        correction_scale = math.sqrt(math.pi / 2) / m

        # Term 1: inner product with MSE reconstruction
        x_mse = self.mse.dequantize(compressed["mse_indices"])
        y_nd = y.unsqueeze(0) if y.ndim == 1 else y
        x_mse_nd = x_mse.unsqueeze(0) if x_mse.ndim == 1 else x_mse
        qjl_signs = compressed["qjl_signs"]
        qjl_signs_nd = qjl_signs.unsqueeze(0) if qjl_signs.ndim == 1 else qjl_signs
        residual_norm = compressed["residual_norm"]
        residual_norm_nd = residual_norm.reshape(1) if residual_norm.ndim == 0 else residual_norm

        # Shape of result is determined by input ndim only, not runtime sizes:
        #   y 1D + x 1D -> scalar
        #   y 1D + x 2D -> (M,)
        #   y 2D + x 1D -> (N,)
        #   y 2D + x 2D -> (N, M)
        # Using matmul unconditionally avoids the previous shape-dependent
        # dispatch that silently returned (N,) (diagonal) when N == M.
        term1 = torch.matmul(y_nd, x_mse_nd.T)                       # (N, M)
        y_projected = torch.matmul(y_nd, self.S.T)                   # (N, qjl_dim)
        qjl_ip = torch.matmul(y_projected, qjl_signs_nd.T)           # (N, M)
        term2 = residual_norm_nd.unsqueeze(0) * correction_scale * qjl_ip
        result = term1 + term2                                       # (N, M)

        if y.ndim == 1 and x_mse.ndim == 1:
            return result.reshape(())
        if y.ndim == 1:
            return result.squeeze(0)
        if x_mse.ndim == 1:
            return result.squeeze(1)
        return result

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
