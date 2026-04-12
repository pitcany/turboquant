"""
Byte-exact Python mirror of the C TurboQuant paper implementation.

This file operates on the same byte layouts, same Π, same S, and same Lloyd-Max
centroids as the C code in ../c/ggml-tq-paper.c, so it serves as the oracle for
byte-exact equality tests (see test_tq_paper.py).

Byte layout TQ4P_D128 (68 bytes / 128 elements = 4.25 bpw):
    offset  size  field
    0       2     orig_norm (fp16)
    2       2     res_d     (fp16)
    4       48    qs        (3-bit × 128, bitplane-packed per 8 coords)
    52      16    qjl_signs (1 bit × 128; bit set = negative)

3-bit bitplane packing: for each group of 8 consecutive coordinates,
emit 3 bytes:
    byte[0] bit_i = (idx_i >> 0) & 1     (low bits)
    byte[1] bit_i = (idx_i >> 1) & 1     (mid bits)
    byte[2] bit_i = (idx_i >> 2) & 1     (high bits)
This is the standard ggml bitplane pattern; easy to unpack via bitwise ops
without division, and CUDA-friendly (per-warp bit broadcasts).

Layout TQ4P_D256 is the same shape, scaled to 132 bytes / 256 elements.
"""

from __future__ import annotations

import math
import pathlib
import struct
from dataclasses import dataclass
from typing import Tuple

import torch

_HERE = pathlib.Path(__file__).resolve().parent
_CONSTANTS_PT = _HERE.parent / "c" / "tqp_constants.pt"

_SQRT_PI_OVER_2 = math.sqrt(math.pi / 2.0)


@dataclass(frozen=True)
class TQPConstants:
    d: int
    pi: torch.Tensor          # (d, d) fp32
    s: torch.Tensor           # (d, d) fp32
    centroids: torch.Tensor   # (8,)   fp32
    boundaries: torch.Tensor  # (7,)   fp32


def load_constants(d: int) -> TQPConstants:
    state = torch.load(_CONSTANTS_PT, weights_only=True)
    return TQPConstants(
        d=d,
        pi=state[f"pi_d{d}"].float(),
        s=state[f"s_d{d}"].float(),
        centroids=state[f"centroids_d{d}"].float(),
        boundaries=state[f"boundaries_d{d}"].float(),
    )


def block_size(d: int) -> int:
    # 4 bytes of norms + 3-bit indices + 1-bit signs
    return 4 + (d * 3) // 8 + d // 8


# ---------- bit packing helpers (byte-exact, CUDA-compatible) ----------

def _pack_indices_bitplane(indices: torch.Tensor) -> bytes:
    """indices: int tensor, shape (d,), values in [0, 8). Output: d*3/8 bytes."""
    d = indices.numel()
    assert d % 8 == 0
    out = bytearray((d * 3) // 8)
    idx = indices.tolist()
    for g in range(d // 8):
        lo = mid = hi = 0
        for i in range(8):
            v = idx[g * 8 + i]
            lo  |= ((v >> 0) & 1) << i
            mid |= ((v >> 1) & 1) << i
            hi  |= ((v >> 2) & 1) << i
        out[g * 3 + 0] = lo
        out[g * 3 + 1] = mid
        out[g * 3 + 2] = hi
    return bytes(out)


def _unpack_indices_bitplane(buf: bytes, d: int) -> torch.Tensor:
    out = torch.zeros(d, dtype=torch.int64)
    for g in range(d // 8):
        lo, mid, hi = buf[g * 3], buf[g * 3 + 1], buf[g * 3 + 2]
        for i in range(8):
            out[g * 8 + i] = ((lo >> i) & 1) | (((mid >> i) & 1) << 1) | (((hi >> i) & 1) << 2)
    return out


def _pack_signs(signs: torch.Tensor) -> bytes:
    """signs: +1 / -1 tensor, shape (d,). Output: d/8 bytes, bit=1 means negative."""
    d = signs.numel()
    assert d % 8 == 0
    out = bytearray(d // 8)
    s = signs.tolist()
    for i in range(d):
        if s[i] < 0:
            out[i // 8] |= 1 << (i % 8)
    return bytes(out)


def _unpack_signs(buf: bytes, d: int) -> torch.Tensor:
    out = torch.ones(d, dtype=torch.float32)
    for i in range(d):
        if (buf[i // 8] >> (i % 8)) & 1:
            out[i] = -1.0
    return out


# ---------- fp16 round-trip for orig_norm and res_d ----------

def _fp16_round(x: float) -> float:
    # Match what ggml_half / __half will store. Use torch's fp16 conversion.
    return torch.tensor([x], dtype=torch.float32).to(torch.float16).to(torch.float32).item()


# ---------- quantize / dequantize / inner_product ----------

def quantize_block(x: torch.Tensor, c: TQPConstants) -> bytes:
    """
    x: (d,) fp32. Returns block_size(d) bytes matching the C layout.

    QJL is applied to the *original-space* residual (x_unit - Pi^T · centroids),
    matching turboquant.py::TurboQuantProd exactly. ‖residual‖ is the same in
    either space (Π orthogonal), but the sign pattern differs, so we must
    un-rotate the reconstruction before projecting with S.
    """
    d = c.d
    assert x.shape == (d,)
    x = x.float()

    orig_norm = x.norm().clamp_min(1e-8).item()
    x_unit = x / orig_norm

    x_rot = c.pi @ x_unit                                    # Stage 1 rotation
    indices = torch.bucketize(x_rot, c.boundaries)            # (d,) in [0, 8)
    x_hat_rot = c.centroids[indices]                          # rotated reconstruction
    x_hat_unit = c.pi.T @ x_hat_rot                           # un-rotated reconstruction
    residual = x_unit - x_hat_unit                            # original-space residual
    res_d = residual.norm().item()

    proj = c.s @ residual                                     # QJL in original space
    signs = torch.where(proj < 0, torch.tensor(-1.0), torch.tensor(1.0))

    buf = bytearray(block_size(d))
    # orig_norm (fp16)
    buf[0:2] = struct.pack('<e', _fp16_round(orig_norm))
    # res_d (fp16)
    buf[2:4] = struct.pack('<e', _fp16_round(res_d))
    # indices (3-bit bitplane)
    qs_bytes = _pack_indices_bitplane(indices)
    buf[4 : 4 + len(qs_bytes)] = qs_bytes
    # qjl signs (1 bit each)
    signs_off = 4 + len(qs_bytes)
    buf[signs_off : signs_off + d // 8] = _pack_signs(signs)
    return bytes(buf)


def dequantize_block(blk: bytes, c: TQPConstants) -> torch.Tensor:
    """Returns (d,) fp32 reconstruction (Stage-1 only, QJL ignored for dequant)."""
    d = c.d
    orig_norm = struct.unpack('<e', blk[0:2])[0]
    qs = blk[4 : 4 + (d * 3) // 8]
    indices = _unpack_indices_bitplane(qs, d)
    x_hat_rot = c.centroids[indices]
    x_hat_unit = c.pi.T @ x_hat_rot
    return orig_norm * x_hat_unit


def inner_product(q: torch.Tensor, blk: bytes, c: TQPConstants) -> float:
    """Estimate <q, x> given unrotated query q and quantized block.

    Matches turboquant.py::TurboQuantProd.inner_product exactly:
        term1 = <q, x_hat_unit>  (x_hat_unit scaled by orig_norm)
        term2 = orig_norm · res_d · √(π/2)/d · <S·q, signs>
    where S operates in the original (unrotated) space.
    """
    d = c.d
    q = q.float()
    orig_norm = float(struct.unpack('<e', blk[0:2])[0])
    res_d     = float(struct.unpack('<e', blk[2:4])[0])
    qs = blk[4 : 4 + (d * 3) // 8]
    indices = _unpack_indices_bitplane(qs, d)
    signs = _unpack_signs(blk[4 + (d * 3) // 8 : 4 + (d * 3) // 8 + d // 8], d)

    # Stage 1: <q, orig_norm · Pi^T · centroids[idx]> = orig_norm · <Pi·q, centroids[idx]>
    q_rot = c.pi @ q
    term1 = orig_norm * (q_rot * c.centroids[indices]).sum().item()

    # Stage 2: QJL in original space
    sq = c.s @ q
    correction = _SQRT_PI_OVER_2 / d
    term2 = orig_norm * res_d * correction * (sq * signs).sum().item()
    return term1 + term2
