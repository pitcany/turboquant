"""
Call the compiled C implementation via ctypes and assert byte-exact equality
with the Python reference (which is already tested byte-exact against
turboquant.py in test_tq_paper.py).

This is the critical cross-check. If it passes, the C code will reproduce
the paper's results when integrated into llama.cpp — no interpretation
errors in the port.

Per-layer verification: tests multiple layer indices to confirm the C code
indexes into the correct per-layer Pi/S arrays.

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


# Layer indices to test: layer 0 (backward compat), plus a spread.
TEST_LAYERS = [0, 1, 7, 15, 31]


# ---------- Load library and declare ctypes signatures ----------

lib = ctypes.CDLL(str(_LIB))

# Block struct shapes (sizeof): d128=69, d256=133.
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

assert ctypes.sizeof(block_tq4p_d128) == 69
assert ctypes.sizeof(block_tq4p_d256) == 133

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


# ---------- Fixtures ----------

@pytest.fixture(scope="module", params=[128, 256])
def d(request):
    return request.param


@pytest.fixture(scope="module")
def constants(d):
    return ref.load_constants(d)


@pytest.fixture(scope="module", params=TEST_LAYERS)
def layer_idx(request):
    return request.param


@pytest.fixture(scope="module")
def vectors(d):
    g = torch.Generator().manual_seed(54321)
    x = torch.randn(50, d, generator=g)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _c_quantize(d, x_np, layer_idx):
    if d == 128:
        blk = block_tq4p_d128()
        lib.ggml_quantize_row_tq4p_d128(
            x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(blk), d, layer_idx,
        )
    else:
        blk = block_tq4p_d256()
        lib.ggml_quantize_row_tq4p_d256(
            x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(blk), d, layer_idx,
        )
    return bytes(blk)


# ---------- Tests ----------

def test_byte_exact_quantize(d, constants, vectors, layer_idx):
    """C quantize output must be byte-identical to the Python reference."""
    import numpy as np
    for i, x in enumerate(vectors):
        x_np = x.float().numpy().copy()
        c_bytes = _c_quantize(d, x_np, layer_idx)
        py_bytes = ref.quantize_block(x, constants, layer_idx=layer_idx)
        if c_bytes != py_bytes:
            diffs = [j for j in range(len(c_bytes)) if c_bytes[j] != py_bytes[j]]
            pytest.fail(f"byte mismatch on vec {i} (d={d}, layer={layer_idx}): "
                        f"{len(diffs)} bytes differ, first at offset {diffs[0]} "
                        f"(C={c_bytes[diffs[0]]:02x}, Py={py_bytes[diffs[0]]:02x})")


def test_layer_idx_in_c_block(d, vectors, layer_idx):
    """C block must store the correct layer_idx at offset 4."""
    import numpy as np
    x_np = vectors[0].float().numpy().copy()
    c_bytes = _c_quantize(d, x_np, layer_idx)
    assert c_bytes[4] == layer_idx, f"Expected layer_idx={layer_idx} at offset 4, got {c_bytes[4]}"


def test_c_dequantize_matches_python(d, constants, vectors, layer_idx):
    """C dequantize round-trip must match Python dequantize."""
    import numpy as np
    for x in vectors:
        py_blk = ref.quantize_block(x, constants, layer_idx=layer_idx)
        if d == 128:
            blk = block_tq4p_d128.from_buffer_copy(py_blk)
            out = (ctypes.c_float * d)()
            lib.ggml_dequantize_row_tq4p_d128(ctypes.byref(blk), out, d)
        else:
            blk = block_tq4p_d256.from_buffer_copy(py_blk)
            out = (ctypes.c_float * d)()
            lib.ggml_dequantize_row_tq4p_d256(ctypes.byref(blk), out, d)
        c_out = torch.tensor(list(out))
        py_out = ref.dequantize_block(py_blk, constants)
        max_diff = (c_out - py_out).abs().max().item()
        assert max_diff < 1e-4, f"dequantize max diff {max_diff} at layer {layer_idx}"


def test_c_vec_dot_matches_python(d, constants, vectors, layer_idx):
    """C inner-product estimator must match Python within fp32 accumulation noise."""
    import numpy as np
    keys    = vectors[:25]
    queries = vectors[25:50]

    max_abs = 0.0
    for q in queries:
        q_np = q.float().numpy().copy()
        Sq = (ctypes.c_float * d)()
        if d == 128:
            lib.ggml_tqp_prepare_query_d128(
                q_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), Sq, layer_idx)
        else:
            lib.ggml_tqp_prepare_query_d256(
                q_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), Sq, layer_idx)

        for k in keys:
            py_blk = ref.quantize_block(k, constants, layer_idx=layer_idx)
            if d == 128:
                blk = block_tq4p_d128.from_buffer_copy(py_blk)
                c_ip = lib.ggml_tqp_vec_dot_block_d128(
                    q_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    Sq, ctypes.byref(blk))
            else:
                blk = block_tq4p_d256.from_buffer_copy(py_blk)
                c_ip = lib.ggml_tqp_vec_dot_block_d256(
                    q_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    Sq, ctypes.byref(blk))
            py_ip = ref.inner_product(q, py_blk, constants)
            max_abs = max(max_abs, abs(c_ip - py_ip))
    assert max_abs < 5e-4, f"C vs Python IP max diff {max_abs:.2e} at layer {layer_idx}"


def test_ggml_dispatch_wrapper_matches_block_api(d, constants, vectors, layer_idx):
    """Regression test: the ggml vec_dot wrapper must follow ggml's arg convention."""
    import numpy as np
    keys = vectors[:5]
    queries = vectors[5:10]

    key_bytes = b"".join(ref.quantize_block(k, constants, layer_idx=layer_idx) for k in keys)
    n = d * len(keys)
    KeyBuf = (ctypes.c_uint8 * len(key_bytes)).from_buffer_copy(key_bytes)

    for q in queries:
        q_flat = (ctypes.c_float * n)()
        for ki in range(len(keys)):
            q_np = q.float().numpy()
            for j in range(d):
                q_flat[ki * d + j] = float(q_np[j])

        out = ctypes.c_float(0.0)
        blk_size = 69 if d == 128 else 133
        if d == 128:
            lib.ggml_vec_dot_tq4p_d128_f32(
                n, ctypes.byref(out), ctypes.sizeof(ctypes.c_float),
                ctypes.cast(KeyBuf, ctypes.c_void_p), blk_size,
                ctypes.cast(q_flat, ctypes.c_void_p), ctypes.sizeof(ctypes.c_float),
                1,
            )
        else:
            lib.ggml_vec_dot_tq4p_d256_f32(
                n, ctypes.byref(out), ctypes.sizeof(ctypes.c_float),
                ctypes.cast(KeyBuf, ctypes.c_void_p), blk_size,
                ctypes.cast(q_flat, ctypes.c_void_p), ctypes.sizeof(ctypes.c_float),
                1,
            )
        wrapper_score = out.value

        expected = sum(
            ref.inner_product(q, ref.quantize_block(k, constants, layer_idx=layer_idx), constants)
            for k in keys
        )
        assert abs(wrapper_score - expected) < 1e-2, \
            f"dispatch wrapper returned {wrapper_score}, expected {expected} at layer {layer_idx}"


def test_c_matches_turboquant_paper_oracle(d, constants, vectors, layer_idx):
    """End-to-end: C output must also match turboquant.py (transitive via Python ref)."""
    from turboquant import TurboQuantProd
    qp = TurboQuantProd(d, bits=4, seed=42 + layer_idx)
    oracle = qp.quantize(vectors)

    qs_off = ref._qs_offset()
    signs_off = ref._signs_offset(d)

    for i, x in enumerate(vectors):
        x_np = x.float().numpy().copy()
        c_bytes = _c_quantize(d, x_np, layer_idx)
        # Extract C indices
        qs = c_bytes[qs_off : qs_off + (d * 3) // 8]
        signs = c_bytes[signs_off : signs_off + d // 8]
        c_idx = ref._unpack_indices_bitplane(qs, d)
        c_signs_pm = ref._unpack_signs(signs, d)
        c_signs_bits = (c_signs_pm < 0).to(torch.uint8)
        oracle_signs_bits = (oracle["qjl_signs"][i] < 0).to(torch.uint8)

        assert torch.equal(c_idx, oracle["mse_indices"][i]), \
            f"vec {i} idx mismatch at layer {layer_idx}"
        assert torch.equal(c_signs_bits, oracle_signs_bits), \
            f"vec {i} signs mismatch at layer {layer_idx}"
