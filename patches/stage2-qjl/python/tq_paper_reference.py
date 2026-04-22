"""
Byte-exact Python mirror of the configurable-bit-width C TQP implementation.

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

Byte layout TQP_D{d}_B{bits}:
    offset  size             field
    0       2                orig_norm (fp16)
    2       2                res_d     (fp16)
    4       1                layer_idx (uint8)
    5       d*bits/8         qs        (bits x d, bitplane-packed per 8 coords)
    ...     d/8              qjl_signs (1 bit x d; bit set = negative)

Bitplane packing: for each group of 8 consecutive coordinates, emit `bits`
bytes where byte[p].bit_i = (idx_i >> p) & 1 for p in 0..bits-1.
"""

from __future__ import annotations

import math
import os
import pathlib
import struct
import threading
from dataclasses import dataclass

import torch

_HERE = pathlib.Path(__file__).resolve().parent
_CONSTANTS_PT = _HERE.parent / "c" / "tqp_constants.pt"

_SQRT_PI_OVER_2 = math.sqrt(math.pi / 2.0)

MAX_LAYERS = 32

# Rotation modes, packed in the high bit of the layer byte.
TQP_ROT_WHT = 0
TQP_ROT_HAAR = 1

BIT6_EXPLICIT = 1 << 6


def layer_byte(layer_idx: int, rotation: int) -> int:
    """Pack (layer_idx, rotation) into an *explicit* quantize-call byte."""
    assert 0 <= layer_idx < 32
    assert rotation in (TQP_ROT_WHT, TQP_ROT_HAAR)
    return BIT6_EXPLICIT | ((rotation & 1) << 7) | (layer_idx & 0x1F)


def stored_byte(layer_idx: int, rotation: int) -> int:
    """Pack (layer_idx, rotation) for block storage (bit 6 = 0)."""
    assert 0 <= layer_idx < 32
    assert rotation in (TQP_ROT_WHT, TQP_ROT_HAAR)
    return ((rotation & 1) << 7) | (layer_idx & 0x1F)


def extract_layer(byte: int) -> int:
    return byte & 0x1F


def extract_rotation(byte: int) -> int:
    return (byte >> 7) & 1


def extract_explicit(byte: int) -> int:
    return (byte >> 6) & 1


# ---------- Runtime rotation resolver ----------

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
    """Apply the three-tier precedence chain to a layer_byte."""
    _ensure_env_read()
    if lb & BIT6_EXPLICIT:
        return lb & ~BIT6_EXPLICIT

    thread_rot = getattr(_thread_local, "rotation", _ROT_UNSET)
    if thread_rot != _ROT_UNSET:
        rot = thread_rot & 1
    elif _process_rotation != _ROT_UNSET:
        rot = _process_rotation & 1
    else:
        rot = TQP_ROT_WHT

    return ((rot & 1) << 7) | (lb & 0x1F)


@dataclass(frozen=True)
class TQPConstants:
    d: int
    bits: int
    sigma: torch.Tensor       # (MAX_LAYERS, d) ±1 fp32 — RHT sign vector
    pi: torch.Tensor          # (MAX_LAYERS, d, d) fp32 — Haar rotation
    s: torch.Tensor           # (MAX_LAYERS, d, d) fp32 — QJL matrix
    centroids: torch.Tensor   # (2^bits,) fp32
    boundaries: torch.Tensor  # (2^bits - 1,) fp32


def load_constants(d: int, bits: int) -> TQPConstants:
    state = torch.load(_CONSTANTS_PT, weights_only=True)
    sigma = state[f"sigma_d{d}"].float()
    pi = state[f"pi_d{d}"].float()
    s = state[f"s_d{d}"].float()
    centroids_key = f"centroids_d{d}_b{bits}"
    boundaries_key = f"boundaries_d{d}_b{bits}"
    if centroids_key not in state and bits == 3:
        centroids_key = f"centroids_d{d}"
        boundaries_key = f"boundaries_d{d}"
    assert sigma.shape == (MAX_LAYERS, d), f"Expected sigma shape ({MAX_LAYERS}, {d}), got {sigma.shape}"
    assert pi.shape == (MAX_LAYERS, d, d), f"Expected pi shape ({MAX_LAYERS}, {d}, {d}), got {pi.shape}"
    assert s.shape == (MAX_LAYERS, d, d), f"Expected s shape ({MAX_LAYERS}, {d}, {d}), got {s.shape}"
    centroids = state[centroids_key].float()
    boundaries = state[boundaries_key].float()
    expected_bins = 1 << bits
    assert centroids.shape == (expected_bins,), f"Expected {expected_bins} centroids, got {centroids.shape}"
    assert boundaries.shape == (expected_bins - 1,), f"Expected {expected_bins - 1} boundaries, got {boundaries.shape}"
    return TQPConstants(
        d=d,
        bits=bits,
        sigma=sigma,
        pi=pi,
        s=s,
        centroids=centroids,
        boundaries=boundaries,
    )


# ---------- Randomized Hadamard Transform ----------

def _wht_inplace(x: torch.Tensor) -> torch.Tensor:
    """In-place fast Walsh-Hadamard transform (natural ordering), unnormalized."""
    d = x.numel()
    assert (d & (d - 1)) == 0, "d must be a power of 2"
    h = 1
    while h < d:
        for i in range(0, d, h << 1):
            a = x[i : i + h].clone()
            b = x[i + h : i + 2 * h].clone()
            x[i : i + h] = a + b
            x[i + h : i + 2 * h] = a - b
        h <<= 1
    return x


def rht_apply(sigma: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    d = v.numel()
    y = (sigma * v).clone()
    _wht_inplace(y)
    return y / math.sqrt(d)


def rht_apply_t(sigma: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    d = v.numel()
    y = v.clone()
    _wht_inplace(y)
    return sigma * y / math.sqrt(d)


def rot_apply(rotation: int, sigma: torch.Tensor, pi: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if rotation == TQP_ROT_HAAR:
        return pi @ v
    return rht_apply(sigma, v)


def rot_apply_t(rotation: int, sigma: torch.Tensor, pi: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if rotation == TQP_ROT_HAAR:
        return pi.T @ v
    return rht_apply_t(sigma, v)


def block_size(d: int, bits: int) -> int:
    return 4 + 1 + (d * bits) // 8 + d // 8


def _qs_size(d: int, bits: int) -> int:
    return (d * bits) // 8


# ---------- bit packing helpers (byte-exact, CUDA-compatible) ----------

def _pack_indices_bitplane(indices: torch.Tensor, bits: int) -> bytes:
    """indices: int tensor, shape (d,), values in [0, 2^bits)."""
    d = indices.numel()
    assert d % 8 == 0
    out = bytearray(_qs_size(d, bits))
    idx = indices.tolist()
    for g in range(d // 8):
        for plane in range(bits):
            packed = 0
            for i in range(8):
                packed |= ((idx[g * 8 + i] >> plane) & 1) << i
            out[g * bits + plane] = packed
    return bytes(out)


def _unpack_indices_bitplane(buf: bytes, d: int, bits: int) -> torch.Tensor:
    out = torch.zeros(d, dtype=torch.int64)
    for g in range(d // 8):
        for i in range(8):
            value = 0
            for plane in range(bits):
                value |= ((buf[g * bits + plane] >> i) & 1) << plane
            out[g * 8 + i] = value
    return out


def _pack_signs(signs: torch.Tensor) -> bytes:
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
    return torch.tensor([x], dtype=torch.float32).to(torch.float16).to(torch.float32).item()


# ---------- byte offsets into the block ----------

def _qs_offset() -> int:
    return 5


def _signs_offset(d: int, bits: int) -> int:
    return _qs_offset() + _qs_size(d, bits)


# ---------- quantize / dequantize / inner_product ----------

def quantize_block(
    x: torch.Tensor,
    c: TQPConstants,
    layer_idx: int = 0,
    rotation: int = TQP_ROT_WHT,
) -> bytes:
    d = c.d
    bits = c.bits
    assert x.shape == (d,)
    assert 0 <= layer_idx < MAX_LAYERS
    assert rotation in (TQP_ROT_WHT, TQP_ROT_HAAR)
    x = x.float()

    sigma = c.sigma[layer_idx]
    pi = c.pi[layer_idx]
    s = c.s[layer_idx]

    orig_norm = x.norm().clamp_min(1e-8).item()
    x_unit = x / orig_norm

    x_rot = rot_apply(rotation, sigma, pi, x_unit)
    indices = torch.bucketize(x_rot, c.boundaries)
    x_hat_rot = c.centroids[indices]
    x_hat_unit = rot_apply_t(rotation, sigma, pi, x_hat_rot)
    residual = x_unit - x_hat_unit
    res_d = residual.norm().item()

    proj = s @ residual
    signs = torch.where(proj < 0, torch.tensor(-1.0), torch.tensor(1.0))

    buf = bytearray(block_size(d, bits))
    buf[0:2] = struct.pack("<e", _fp16_round(orig_norm))
    buf[2:4] = struct.pack("<e", _fp16_round(res_d))
    buf[4] = stored_byte(layer_idx, rotation)
    qs_off = _qs_offset()
    qs_bytes = _pack_indices_bitplane(indices, bits)
    buf[qs_off : qs_off + len(qs_bytes)] = qs_bytes
    signs_off = _signs_offset(d, bits)
    buf[signs_off : signs_off + d // 8] = _pack_signs(signs)
    return bytes(buf)


def dequantize_block(blk: bytes, c: TQPConstants) -> torch.Tensor:
    d = c.d
    bits = c.bits
    orig_norm = struct.unpack("<e", blk[0:2])[0]
    layer_byte_val = blk[4]
    layer_idx = extract_layer(layer_byte_val)
    rotation = extract_rotation(layer_byte_val)
    assert 0 <= layer_idx < MAX_LAYERS
    sigma = c.sigma[layer_idx]
    pi = c.pi[layer_idx]

    qs_off = _qs_offset()
    qs = blk[qs_off : qs_off + _qs_size(d, bits)]
    indices = _unpack_indices_bitplane(qs, d, bits)
    x_hat_rot = c.centroids[indices]
    x_hat_unit = rot_apply_t(rotation, sigma, pi, x_hat_rot)
    return orig_norm * x_hat_unit


def inner_product(q: torch.Tensor, blk: bytes, c: TQPConstants) -> float:
    d = c.d
    bits = c.bits
    q = q.float()
    orig_norm = float(struct.unpack("<e", blk[0:2])[0])
    res_d = float(struct.unpack("<e", blk[2:4])[0])
    layer_byte_val = blk[4]
    layer_idx = extract_layer(layer_byte_val)
    rotation = extract_rotation(layer_byte_val)
    assert 0 <= layer_idx < MAX_LAYERS

    sigma = c.sigma[layer_idx]
    pi = c.pi[layer_idx]
    s = c.s[layer_idx]

    qs_off = _qs_offset()
    qs = blk[qs_off : qs_off + _qs_size(d, bits)]
    indices = _unpack_indices_bitplane(qs, d, bits)

    signs_off = _signs_offset(d, bits)
    signs = _unpack_signs(blk[signs_off : signs_off + d // 8], d)

    q_rot = rot_apply(rotation, sigma, pi, q)
    term1 = orig_norm * (q_rot * c.centroids[indices]).sum().item()

    sq = s @ q
    correction = _SQRT_PI_OVER_2 / d
    term2 = orig_norm * res_d * correction * (sq * signs).sum().item()
    return term1 + term2
