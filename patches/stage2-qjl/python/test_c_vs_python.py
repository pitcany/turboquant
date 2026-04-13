"""
Call the compiled C implementation via ctypes and assert byte-exact equality
with the Python reference, across both rotation modes (WHT and Haar) and
a spread of layer indices.

Build the shared library first:
    cd ../c && gcc -O2 -fPIC -shared -o libggml_tq_paper.so ggml-tq-paper.c

Then:
    pytest test_c_vs_python.py
"""

from __future__ import annotations

import ctypes
import pathlib
import struct
import sys

import pytest
import torch

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
_LIB = _HERE.parent / "c" / "libggml_tq_paper.so"

sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_HERE))
import tq_paper_reference as ref


TEST_LAYERS = [0, 1, 7, 15, 31]
ROTATIONS = [ref.TQP_ROT_WHT, ref.TQP_ROT_HAAR]
ROTATION_IDS = {ref.TQP_ROT_WHT: "wht", ref.TQP_ROT_HAAR: "haar"}


# ---------- Load library and declare ctypes signatures ----------

lib = ctypes.CDLL(str(_LIB))

# Block struct shapes (sizeof): d64=37, d128=69, d256=133.
class block_tq4p_d64(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("orig_norm", ctypes.c_uint16),
        ("res_d",     ctypes.c_uint16),
        ("layer_idx", ctypes.c_uint8),
        ("qs",        ctypes.c_uint8 * 24),
        ("qjl_signs", ctypes.c_uint8 * 8),
    ]

class block_tq4p_d128(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("orig_norm", ctypes.c_uint16),
        ("res_d",     ctypes.c_uint16),
        ("layer_idx", ctypes.c_uint8),
        ("qs",        ctypes.c_uint8 * 48),
        ("qjl_signs", ctypes.c_uint8 * 16),
    ]

class block_tq4p_d256(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("orig_norm", ctypes.c_uint16),
        ("res_d",     ctypes.c_uint16),
        ("layer_idx", ctypes.c_uint8),
        ("qs",        ctypes.c_uint8 * 96),
        ("qjl_signs", ctypes.c_uint8 * 32),
    ]

assert ctypes.sizeof(block_tq4p_d64) == 37
assert ctypes.sizeof(block_tq4p_d128) == 69
assert ctypes.sizeof(block_tq4p_d256) == 133

# d=64 ctypes declarations
lib.ggml_quantize_row_tq4p_d64.restype = None
lib.ggml_quantize_row_tq4p_d64.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(block_tq4p_d64),
    ctypes.c_int64,
    ctypes.c_uint8,
]
lib.ggml_dequantize_row_tq4p_d64.restype = None
lib.ggml_dequantize_row_tq4p_d64.argtypes = [
    ctypes.POINTER(block_tq4p_d64),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int64,
]
lib.ggml_tqp_vec_dot_block_d64.restype = ctypes.c_float
lib.ggml_tqp_vec_dot_block_d64.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(block_tq4p_d64),
]
lib.ggml_tqp_prepare_query_d64.restype = None
lib.ggml_tqp_prepare_query_d64.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_uint8,
]
lib.ggml_vec_dot_tq4p_d64_f32.restype = None
lib.ggml_vec_dot_tq4p_d64_f32.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
]
lib.ggml_vec_dot_tq4p_d64_q8k.restype = None
lib.ggml_vec_dot_tq4p_d64_q8k.argtypes = lib.ggml_vec_dot_tq4p_d64_f32.argtypes

# quantize_row now takes layer_idx as 4th arg
lib.ggml_quantize_row_tq4p_d128.restype = None
lib.ggml_quantize_row_tq4p_d128.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(block_tq4p_d128),
    ctypes.c_int64,
    ctypes.c_uint8,
]

lib.ggml_quantize_row_tq4p_d256.restype = None
lib.ggml_quantize_row_tq4p_d256.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(block_tq4p_d256),
    ctypes.c_int64,
    ctypes.c_uint8,
]

lib.ggml_dequantize_row_tq4p_d128.restype = None
lib.ggml_dequantize_row_tq4p_d128.argtypes = [
    ctypes.POINTER(block_tq4p_d128),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int64,
]

lib.ggml_dequantize_row_tq4p_d256.restype = None
lib.ggml_dequantize_row_tq4p_d256.argtypes = [
    ctypes.POINTER(block_tq4p_d256),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int64,
]

lib.ggml_tqp_vec_dot_block_d128.restype = ctypes.c_float
lib.ggml_tqp_vec_dot_block_d128.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(block_tq4p_d128),
]

lib.ggml_tqp_vec_dot_block_d256.restype = ctypes.c_float
lib.ggml_tqp_vec_dot_block_d256.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(block_tq4p_d256),
]

# prepare_query now takes layer_idx as 3rd arg
lib.ggml_tqp_prepare_query_d128.restype = None
lib.ggml_tqp_prepare_query_d128.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_uint8,
]

lib.ggml_tqp_prepare_query_d256.restype = None
lib.ggml_tqp_prepare_query_d256.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_uint8,
]

# The ggml dispatch wrappers use ggml's vec_dot signature (unchanged).
lib.ggml_vec_dot_tq4p_d128_f32.restype = None
lib.ggml_vec_dot_tq4p_d128_f32.argtypes = [
    ctypes.c_int,                            # n
    ctypes.POINTER(ctypes.c_float),          # s
    ctypes.c_size_t,                         # bs
    ctypes.c_void_p,                         # vx (K blocks)
    ctypes.c_size_t,                         # bx
    ctypes.c_void_p,                         # vy (query)
    ctypes.c_size_t,                         # by
    ctypes.c_int,                            # nrc
]

lib.ggml_vec_dot_tq4p_d256_f32.restype = None
lib.ggml_vec_dot_tq4p_d256_f32.argtypes = lib.ggml_vec_dot_tq4p_d128_f32.argtypes

# Q8_K query path — same ggml vec_dot signature, vy points to Q8_K blocks.
lib.ggml_vec_dot_tq4p_d128_q8k.restype = None
lib.ggml_vec_dot_tq4p_d128_q8k.argtypes = lib.ggml_vec_dot_tq4p_d128_f32.argtypes

lib.ggml_vec_dot_tq4p_d256_q8k.restype = None
lib.ggml_vec_dot_tq4p_d256_q8k.argtypes = lib.ggml_vec_dot_tq4p_d128_f32.argtypes


# Q8_K block struct (matches ggml-common.h / block_q8k_compat in ggml-tq-paper.h).
QK_Q8K = 256

class block_q8k_compat(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("d",     ctypes.c_float),
        ("qs",    ctypes.c_int8 * QK_Q8K),
        ("bsums", ctypes.c_int16 * (QK_Q8K // 16)),
    ]

assert ctypes.sizeof(block_q8k_compat) == 292


# ---------- Fixtures ----------

@pytest.fixture(scope="module", params=[64, 128, 256])
def d(request):
    return request.param


@pytest.fixture(scope="module")
def constants(d):
    return ref.load_constants(d)


@pytest.fixture(scope="module", params=TEST_LAYERS)
def layer_idx(request):
    return request.param


@pytest.fixture(scope="module", params=ROTATIONS, ids=lambda r: ROTATION_IDS[r])
def rotation(request):
    return request.param


@pytest.fixture(scope="module")
def vectors(d):
    g = torch.Generator().manual_seed(54321)
    x = torch.randn(50, d, generator=g)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


_BLOCK_TYPES = {
    64:  (block_tq4p_d64,  lib.ggml_quantize_row_tq4p_d64),
    128: (block_tq4p_d128, lib.ggml_quantize_row_tq4p_d128),
    256: (block_tq4p_d256, lib.ggml_quantize_row_tq4p_d256),
}

_DEQUANT_FNS = {
    64:  lib.ggml_dequantize_row_tq4p_d64,
    128: lib.ggml_dequantize_row_tq4p_d128,
    256: lib.ggml_dequantize_row_tq4p_d256,
}

_PREPARE_QUERY_FNS = {
    64:  lib.ggml_tqp_prepare_query_d64,
    128: lib.ggml_tqp_prepare_query_d128,
    256: lib.ggml_tqp_prepare_query_d256,
}

_VEC_DOT_BLOCK_FNS = {
    64:  lib.ggml_tqp_vec_dot_block_d64,
    128: lib.ggml_tqp_vec_dot_block_d128,
    256: lib.ggml_tqp_vec_dot_block_d256,
}

_VEC_DOT_F32_FNS = {
    64:  lib.ggml_vec_dot_tq4p_d64_f32,
    128: lib.ggml_vec_dot_tq4p_d128_f32,
    256: lib.ggml_vec_dot_tq4p_d256_f32,
}

_VEC_DOT_Q8K_FNS = {
    64:  lib.ggml_vec_dot_tq4p_d64_q8k,
    128: lib.ggml_vec_dot_tq4p_d128_q8k,
    256: lib.ggml_vec_dot_tq4p_d256_q8k,
}

_BLK_SIZES = {64: 37, 128: 69, 256: 133}


def _c_quantize(d, x_np, layer_idx, rotation):
    layer_byte = ref.layer_byte(layer_idx, rotation)
    blk_cls, quantize_fn = _BLOCK_TYPES[d]
    blk = blk_cls()
    quantize_fn(
        x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(blk), d, layer_byte,
    )
    return bytes(blk)


# ---------- Tests ----------

def test_byte_exact_quantize(d, constants, vectors, layer_idx, rotation):
    """C quantize output must be byte-identical to the Python reference."""
    for i, x in enumerate(vectors):
        x_np = x.float().numpy().copy()
        c_bytes = _c_quantize(d, x_np, layer_idx, rotation)
        py_bytes = ref.quantize_block(x, constants, layer_idx=layer_idx, rotation=rotation)
        if c_bytes != py_bytes:
            diffs = [j for j in range(len(c_bytes)) if c_bytes[j] != py_bytes[j]]
            pytest.fail(f"byte mismatch vec {i} d={d} layer={layer_idx} rot={ROTATION_IDS[rotation]}: "
                        f"{len(diffs)} bytes differ, first offset {diffs[0]} "
                        f"(C={c_bytes[diffs[0]]:02x}, Py={py_bytes[diffs[0]]:02x})")


def test_layer_byte_in_c_block(d, vectors, layer_idx, rotation):
    """C block must store the correct packed layer_idx + rotation at offset 4."""
    x_np = vectors[0].float().numpy().copy()
    c_bytes = _c_quantize(d, x_np, layer_idx, rotation)
    assert ref.extract_layer(c_bytes[4]) == layer_idx
    assert ref.extract_rotation(c_bytes[4]) == rotation


def test_c_dequantize_matches_python(d, constants, vectors, layer_idx, rotation):
    """C dequantize round-trip must match Python dequantize."""
    blk_cls = _BLOCK_TYPES[d][0]
    dequant_fn = _DEQUANT_FNS[d]
    for x in vectors:
        py_blk = ref.quantize_block(x, constants, layer_idx=layer_idx, rotation=rotation)
        blk = blk_cls.from_buffer_copy(py_blk)
        out = (ctypes.c_float * d)()
        dequant_fn(ctypes.byref(blk), out, d)
        c_out = torch.tensor(list(out))
        py_out = ref.dequantize_block(py_blk, constants)
        max_diff = (c_out - py_out).abs().max().item()
        assert max_diff < 1e-4, f"dequantize max diff {max_diff} layer {layer_idx} rot {ROTATION_IDS[rotation]}"


def test_c_vec_dot_matches_python(d, constants, vectors, layer_idx, rotation):
    """C inner-product estimator must match Python within fp32 accumulation noise."""
    keys    = vectors[:25]
    queries = vectors[25:50]

    blk_cls = _BLOCK_TYPES[d][0]
    prepare_fn = _PREPARE_QUERY_FNS[d]
    vec_dot_fn = _VEC_DOT_BLOCK_FNS[d]

    layer_byte_val = ref.layer_byte(layer_idx, rotation)
    max_abs = 0.0
    for q in queries:
        q_np = q.float().numpy().copy()
        Sq = (ctypes.c_float * d)()
        prepare_fn(q_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), Sq, layer_byte_val)

        for k in keys:
            py_blk = ref.quantize_block(k, constants, layer_idx=layer_idx, rotation=rotation)
            blk = blk_cls.from_buffer_copy(py_blk)
            c_ip = vec_dot_fn(q_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                              Sq, ctypes.byref(blk))
            py_ip = ref.inner_product(q, py_blk, constants)
            max_abs = max(max_abs, abs(c_ip - py_ip))
    assert max_abs < 5e-4, f"C vs Python IP max diff {max_abs:.2e} layer {layer_idx} rot {ROTATION_IDS[rotation]}"


def test_ggml_dispatch_wrapper_matches_block_api(d, constants, vectors, layer_idx, rotation):
    """Regression test: the ggml vec_dot wrapper must follow ggml's arg convention."""
    keys = vectors[:5]
    queries = vectors[5:10]

    key_bytes = b"".join(ref.quantize_block(k, constants, layer_idx=layer_idx, rotation=rotation) for k in keys)
    n = d * len(keys)
    KeyBuf = (ctypes.c_uint8 * len(key_bytes)).from_buffer_copy(key_bytes)
    blk_size = _BLK_SIZES[d]
    vec_dot_f32 = _VEC_DOT_F32_FNS[d]

    for q in queries:
        q_flat = (ctypes.c_float * n)()
        for ki in range(len(keys)):
            q_np = q.float().numpy()
            for j in range(d):
                q_flat[ki * d + j] = float(q_np[j])

        out = ctypes.c_float(0.0)
        vec_dot_f32(
            n, ctypes.byref(out), ctypes.sizeof(ctypes.c_float),
            ctypes.cast(KeyBuf, ctypes.c_void_p), blk_size,
            ctypes.cast(q_flat, ctypes.c_void_p), ctypes.sizeof(ctypes.c_float),
            1,
        )
        wrapper_score = out.value

        expected = sum(
            ref.inner_product(q, ref.quantize_block(k, constants, layer_idx=layer_idx, rotation=rotation), constants)
            for k in keys
        )
        assert abs(wrapper_score - expected) < 1e-2, \
            f"dispatch wrapper returned {wrapper_score}, expected {expected} layer {layer_idx} rot {ROTATION_IDS[rotation]}"


# ---------- Q8_K query path helpers ----------

def _quantize_fp32_to_q8k(fp32_values):
    """Quantize a flat fp32 array (length multiple of 256) to Q8_K blocks.

    Mimics ggml's quantize_row_q8_K: per-block absmax scale,
    round-to-nearest int8. Returns a list of block_q8k_compat ctypes structs.
    """
    import numpy as np
    arr = np.asarray(fp32_values, dtype=np.float32)
    assert arr.size % QK_Q8K == 0
    nb = arr.size // QK_Q8K
    blocks = []
    for b in range(nb):
        chunk = arr[b * QK_Q8K : (b + 1) * QK_Q8K]
        amax = float(np.abs(chunk).max())
        d = amax / 127.0 if amax > 0 else 1.0
        inv_d = 1.0 / d
        blk = block_q8k_compat()
        blk.d = d
        for i in range(QK_Q8K):
            v = round(chunk[i] * inv_d)
            v = max(-128, min(127, v))
            blk.qs[i] = v
        # bsums (not used by our vec_dot, but fill for correctness)
        for g in range(QK_Q8K // 16):
            blk.bsums[g] = sum(blk.qs[g * 16 + j] for j in range(16))
        blocks.append(blk)
    return blocks


def test_q8k_dispatch_matches_fp32(d, constants, vectors, layer_idx, rotation):
    """Q8_K query path must match fp32 path within Q8_K quantization noise.

    We construct the same attention scenario (5 keys, 5 queries) and verify
    the Q8_K dispatch wrapper produces results close to the fp32 wrapper.
    The tolerance is wider because Q8_K quantization of the query introduces
    ~0.4% relative error, but absolute scores should stay within 0.05.
    """
    import numpy as np

    # n must be a multiple of 256 for the Q8_K path.
    # Use enough keys so total elements = lcm(d, 256).
    n_keys = QK_Q8K // d  # 2 for d=128, 1 for d=256
    keys = vectors[:n_keys]
    queries = vectors[n_keys : n_keys + 5]

    key_bytes = b"".join(
        ref.quantize_block(k, constants, layer_idx=layer_idx, rotation=rotation) for k in keys
    )
    n = d * n_keys
    assert n % QK_Q8K == 0
    KeyBuf = (ctypes.c_uint8 * len(key_bytes)).from_buffer_copy(key_bytes)

    blk_size = _BLK_SIZES[d]
    vec_dot_f32 = _VEC_DOT_F32_FNS[d]
    vec_dot_q8k = _VEC_DOT_Q8K_FNS[d]

    max_abs_diff = 0.0
    for q in queries:
        q_np = q.float().numpy()

        # Build fp32 query buffer (replicate query per key block, same as fp32 test)
        q_flat = (ctypes.c_float * n)()
        for ki in range(n_keys):
            for j in range(d):
                q_flat[ki * d + j] = float(q_np[j])

        # fp32 reference
        out_f32 = ctypes.c_float(0.0)
        vec_dot_f32(
            n, ctypes.byref(out_f32), ctypes.sizeof(ctypes.c_float),
            ctypes.cast(KeyBuf, ctypes.c_void_p), blk_size,
            ctypes.cast(q_flat, ctypes.c_void_p), ctypes.sizeof(ctypes.c_float),
            1,
        )

        # Quantize query to Q8_K blocks
        q8k_blocks = _quantize_fp32_to_q8k(list(q_flat))
        Q8kArr = block_q8k_compat * len(q8k_blocks)
        q8k_buf = Q8kArr(*q8k_blocks)

        # Q8_K query path
        out_q8k = ctypes.c_float(0.0)
        vec_dot_q8k(
            n, ctypes.byref(out_q8k), ctypes.sizeof(ctypes.c_float),
            ctypes.cast(KeyBuf, ctypes.c_void_p), blk_size,
            ctypes.cast(q8k_buf, ctypes.c_void_p),
            ctypes.sizeof(block_q8k_compat),
            1,
        )

        diff = abs(out_q8k.value - out_f32.value)
        max_abs_diff = max(max_abs_diff, diff)

    # Q8_K introduces ~0.4% relative quantization error on the query.
    # For unit-normalized vectors with scores near ±1, 0.05 abs tolerance
    # is generous but appropriate.
    assert max_abs_diff < 0.05, (
        f"Q8_K vs fp32 max diff {max_abs_diff:.4f} exceeds tolerance "
        f"(d={d}, layer={layer_idx}, rot={ROTATION_IDS[rotation]})"
    )
