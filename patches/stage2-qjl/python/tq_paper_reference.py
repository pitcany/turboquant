"""
Byte-exact Python mirror of the C TurboQuant(-ish) implementation.

This file operates on the same byte layouts, same per-layer σ_i, Π_i, S_i,
and Lloyd-Max centroids as the C code in ../c/ggml-tq-paper.c, so it serves
as the oracle for byte-exact equality tests (see test_c_vs_python.py).

Rotation mode is user-selectable per block:
    TQP_ROT_WHT  (0): Π_i = (1/√d) · H · diag(σ_i) — RHT, O(d log d).
    TQP_ROT_HAAR (1): Π_i is a dense d×d Haar random orthogonal — paper-exact.

Selection is packed in the high bit of the block's layer_idx byte:
    layer_byte = (rotation << 7) | (layer_idx & 0x1f)

Per-layer constants:
    σ_i:   seed = 42 + layer_idx   (Rademacher sign vector for RHT)
    Π_i:   seed = 42 + layer_idx   (Haar rotation — matches turboquant.py)
    S_i:   seed = 43 + layer_idx

Byte layout TQ4P_D128 (69 bytes / 128 elements = 4.3125 bpw):
    offset  size  field
    0       2     orig_norm (fp16)
    2       2     res_d     (fp16)
    4       1     layer_idx (uint8)
    5       48    qs        (3-bit x 128, bitplane-packed per 8 coords)
    53      16    qjl_signs (1 bit x 128; bit set = negative)

3-bit bitplane packing: for each group of 8 consecutive coordinates,
emit 3 bytes:
    byte[0] bit_i = (idx_i >> 0) & 1     (low bits)
    byte[1] bit_i = (idx_i >> 1) & 1     (mid bits)
    byte[2] bit_i = (idx_i >> 2) & 1     (high bits)
This is the standard ggml bitplane pattern; easy to unpack via bitwise ops
without division, and CUDA-friendly (per-warp bit broadcasts).

Layout TQ4P_D256 is the same shape, scaled to 133 bytes / 256 elements.
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

MAX_LAYERS = 32

# Rotation modes, packed in the high bit of the layer byte.
TQP_ROT_WHT  = 0
TQP_ROT_HAAR = 1


BIT6_EXPLICIT = 1 << 6


def layer_byte(layer_idx: int, rotation: int) -> int:
    """Pack (layer_idx, rotation) into an *explicit* quantize-call byte.

    Sets bit 6 so the rotation in bit 7 is used as-is, bypassing the
    runtime rotation resolver.  Matches the C macro TQP_LAYER_BYTE.
    """
    assert 0 <= layer_idx < 32
    assert rotation in (TQP_ROT_WHT, TQP_ROT_HAAR)
    return BIT6_EXPLICIT | ((rotation & 1) << 7) | (layer_idx & 0x1f)


def stored_byte(layer_idx: int, rotation: int) -> int:
    """Pack (layer_idx, rotation) for block storage (bit 6 = 0).

    Matches the C macro TQP_STORED_BYTE.
    """
    assert 0 <= layer_idx < 32
    assert rotation in (TQP_ROT_WHT, TQP_ROT_HAAR)
    return ((rotation & 1) << 7) | (layer_idx & 0x1f)


def extract_layer(byte: int) -> int:
    return byte & 0x1f


def extract_rotation(byte: int) -> int:
    return (byte >> 7) & 1


def extract_explicit(byte: int) -> int:
    return (byte >> 6) & 1


# ---------- Runtime rotation resolver ----------
#
# Python mirror of the C runtime rotation selector.  Uses module-level
# globals (process default) and threading.local (thread default).  The
# env var OLLAMA_TQP_ROTATION is read once at first resolve call.

import os
import threading

_ROT_UNSET = 0xFF

_process_rotation: int = _ROT_UNSET
_env_read: bool = False
_env_lock = threading.Lock()
_thread_local = threading.local()


def _ensure_env_read() -> None:
    global _process_rotation, _env_read
    if _env_read:
        return
    with _env_lock:
        if _env_read:
            return
        val = os.environ.get("OLLAMA_TQP_ROTATION", "")
        if val.lower().startswith("h"):
            _process_rotation = TQP_ROT_HAAR
        elif val.lower().startswith("w"):
            _process_rotation = TQP_ROT_WHT
        _env_read = True


def set_default_rotation(rot: int) -> None:
    """Set the process-wide default rotation (0, 1, or 0xff to clear)."""
    global _process_rotation
    _ensure_env_read()
    _process_rotation = rot


def set_thread_rotation(rot: int) -> None:
    """Set per-thread rotation override (0, 1, or 0xff to clear)."""
    _thread_local.rotation = rot


def clear_thread_rotation() -> None:
    """Clear the per-thread rotation override."""
    _thread_local.rotation = _ROT_UNSET


def resolve_rotation(lb: int) -> int:
    """Apply the three-tier precedence chain to a layer_byte.

    Returns a byte with bit 6 cleared and bit 7 set to the resolved
    rotation.  Matches the C function tqp_resolve_rotation().
    """
    _ensure_env_read()

    # Per-call explicit: bit 6 = 1
    if lb & BIT6_EXPLICIT:
        return lb & ~BIT6_EXPLICIT

    # Cascade: thread-local > process default > compile-time WHT
    thread_rot = getattr(_thread_local, "rotation", _ROT_UNSET)
    if thread_rot != _ROT_UNSET:
        rot = thread_rot & 1
    elif _process_rotation != _ROT_UNSET:
        rot = _process_rotation & 1
    else:
        rot = TQP_ROT_WHT

    return ((rot & 1) << 7) | (lb & 0x1f)


@dataclass(frozen=True)
class TQPConstants:
    d: int
    sigma: torch.Tensor       # (MAX_LAYERS, d) ±1 fp32 — RHT sign vector
    pi: torch.Tensor          # (MAX_LAYERS, d, d) fp32 — Haar rotation
    s: torch.Tensor           # (MAX_LAYERS, d, d) fp32 — QJL matrix
    centroids: torch.Tensor   # (8,)   fp32
    boundaries: torch.Tensor  # (7,)   fp32


def load_constants(d: int) -> TQPConstants:
    state = torch.load(_CONSTANTS_PT, weights_only=True)
    sigma = state[f"sigma_d{d}"].float()
    pi = state[f"pi_d{d}"].float()
    s = state[f"s_d{d}"].float()
    assert sigma.shape == (MAX_LAYERS, d), f"Expected sigma shape ({MAX_LAYERS}, {d}), got {sigma.shape}"
    assert pi.shape == (MAX_LAYERS, d, d), f"Expected pi shape ({MAX_LAYERS}, {d}, {d}), got {pi.shape}"
    assert s.shape == (MAX_LAYERS, d, d), f"Expected s shape ({MAX_LAYERS}, {d}, {d}), got {s.shape}"
    return TQPConstants(
        d=d,
        sigma=sigma,
        pi=pi,
        s=s,
        centroids=state[f"centroids_d{d}"].float(),
        boundaries=state[f"boundaries_d{d}"].float(),
    )


# ---------- Randomized Hadamard Transform ----------

def _wht_inplace(x: torch.Tensor) -> torch.Tensor:
    """In-place fast Walsh-Hadamard transform (natural ordering), unnormalized.
    Divide by sqrt(d) afterwards for the orthogonal version.
    """
    d = x.numel()
    assert (d & (d - 1)) == 0, "d must be a power of 2"
    h = 1
    while h < d:
        for i in range(0, d, h << 1):
            a = x[i : i + h].clone()
            b = x[i + h : i + 2 * h].clone()
            x[i : i + h]         = a + b
            x[i + h : i + 2 * h] = a - b
        h <<= 1
    return x


def rht_apply(sigma: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """y = Π · v = (1/sqrt(d)) · H · diag(σ) · v."""
    d = v.numel()
    y = (sigma * v).clone()
    _wht_inplace(y)
    return y / math.sqrt(d)


def rht_apply_t(sigma: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """y = Πᵀ · v = (1/sqrt(d)) · diag(σ) · H · v (H is symmetric)."""
    d = v.numel()
    y = v.clone()
    _wht_inplace(y)
    return sigma * y / math.sqrt(d)


def rot_apply(rotation: int, sigma: torch.Tensor, pi: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply Π · v for the requested rotation mode."""
    if rotation == TQP_ROT_HAAR:
        return pi @ v
    return rht_apply(sigma, v)


def rot_apply_t(rotation: int, sigma: torch.Tensor, pi: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply Πᵀ · v for the requested rotation mode."""
    if rotation == TQP_ROT_HAAR:
        return pi.T @ v
    return rht_apply_t(sigma, v)


def block_size(d: int) -> int:
    # 4 bytes of norms + 1 byte layer_idx + 3-bit indices + 1-bit signs
    return 4 + 1 + (d * 3) // 8 + d // 8


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


# ---------- byte offsets into the block ----------

def _qs_offset() -> int:
    return 5  # orig_norm(2) + res_d(2) + layer_idx(1)


def _signs_offset(d: int) -> int:
    return _qs_offset() + (d * 3) // 8


# ---------- quantize / dequantize / inner_product ----------

def quantize_block(x: torch.Tensor, c: TQPConstants,
                   layer_idx: int = 0, rotation: int = TQP_ROT_WHT) -> bytes:
    """
    x: (d,) fp32. Returns block_size(d) bytes matching the C layout.

    layer_idx in [0, 31] picks σ_i / Π_i / S_i. rotation in {TQP_ROT_WHT,
    TQP_ROT_HAAR} selects which rotation to apply. Both are recorded in
    the block's layer_idx byte.
    """
    d = c.d
    assert x.shape == (d,)
    assert 0 <= layer_idx < MAX_LAYERS
    assert rotation in (TQP_ROT_WHT, TQP_ROT_HAAR)
    x = x.float()

    sigma = c.sigma[layer_idx]   # (d,)
    pi = c.pi[layer_idx]         # (d, d)
    s = c.s[layer_idx]           # (d, d)

    orig_norm = x.norm().clamp_min(1e-8).item()
    x_unit = x / orig_norm

    x_rot = rot_apply(rotation, sigma, pi, x_unit)             # Stage 1 rotation
    indices = torch.bucketize(x_rot, c.boundaries)             # (d,) in [0, 8)
    x_hat_rot = c.centroids[indices]                            # rotated reconstruction
    x_hat_unit = rot_apply_t(rotation, sigma, pi, x_hat_rot)    # un-rotated reconstruction
    residual = x_unit - x_hat_unit                              # original-space residual
    res_d = residual.norm().item()

    proj = s @ residual                                         # QJL in original space
    signs = torch.where(proj < 0, torch.tensor(-1.0), torch.tensor(1.0))

    buf = bytearray(block_size(d))
    buf[0:2] = struct.pack('<e', _fp16_round(orig_norm))
    buf[2:4] = struct.pack('<e', _fp16_round(res_d))
    buf[4] = stored_byte(layer_idx, rotation)
    qs_off = _qs_offset()
    qs_bytes = _pack_indices_bitplane(indices)
    buf[qs_off : qs_off + len(qs_bytes)] = qs_bytes
    signs_off = _signs_offset(d)
    buf[signs_off : signs_off + d // 8] = _pack_signs(signs)
    return bytes(buf)


def dequantize_block(blk: bytes, c: TQPConstants) -> torch.Tensor:
    """Returns (d,) fp32 reconstruction (Stage-1 only, QJL ignored for dequant).

    Reads both layer_idx and rotation from the block's packed header byte.
    """
    d = c.d
    orig_norm = struct.unpack('<e', blk[0:2])[0]
    layer_byte_val = blk[4]
    layer_idx = extract_layer(layer_byte_val)
    rotation = extract_rotation(layer_byte_val)
    assert 0 <= layer_idx < MAX_LAYERS
    sigma = c.sigma[layer_idx]
    pi = c.pi[layer_idx]

    qs_off = _qs_offset()
    qs = blk[qs_off : qs_off + (d * 3) // 8]
    indices = _unpack_indices_bitplane(qs, d)
    x_hat_rot = c.centroids[indices]
    x_hat_unit = rot_apply_t(rotation, sigma, pi, x_hat_rot)
    return orig_norm * x_hat_unit


def inner_product(q: torch.Tensor, blk: bytes, c: TQPConstants) -> float:
    """Estimate <q, x> given unrotated query q and quantized block.

    Reads both layer_idx and rotation from the block's packed header byte.
    """
    d = c.d
    q = q.float()
    orig_norm = float(struct.unpack('<e', blk[0:2])[0])
    res_d     = float(struct.unpack('<e', blk[2:4])[0])
    layer_byte_val = blk[4]
    layer_idx = extract_layer(layer_byte_val)
    rotation = extract_rotation(layer_byte_val)
    assert 0 <= layer_idx < MAX_LAYERS

    sigma = c.sigma[layer_idx]
    pi = c.pi[layer_idx]
    s = c.s[layer_idx]

    qs_off = _qs_offset()
    qs = blk[qs_off : qs_off + (d * 3) // 8]
    indices = _unpack_indices_bitplane(qs, d)

    signs_off = _signs_offset(d)
    signs = _unpack_signs(blk[signs_off : signs_off + d // 8], d)

    # Stage 1: <q, orig_norm . Πᵀ . centroids[idx]> = orig_norm . <Π.q, centroids[idx]>
    q_rot = rot_apply(rotation, sigma, pi, q)
    term1 = orig_norm * (q_rot * c.centroids[indices]).sum().item()

    # Stage 2: QJL in original space
    sq = s @ q
    correction = _SQRT_PI_OVER_2 / d
    term2 = orig_norm * res_d * correction * (sq * signs).sum().item()
    return term1 + term2
