"""
Call the compiled C implementation via ctypes and assert byte-exact equality
with the Python reference (which is already tested byte-exact against
turboquant.py in test_tq_paper.py).

This is the critical cross-check. If it passes, the C code will reproduce
the paper's results when integrated into llama.cpp — no interpretation
errors in the port.

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
_LIB = _HERE.parent / "c" / "libggml_tq_paper.so"

sys.path.insert(0, str(_HERE))
import tq_paper_reference as ref


# ---------- Load library and declare ctypes signatures ----------

lib = ctypes.CDLL(str(_LIB))

# Block struct shapes (sizeof): d128=68, d256=132.
class block_tq4p_d128(ctypes.Structure):
    _fields_ = [
        ("orig_norm", ctypes.c_uint16),
        ("res_d",     ctypes.c_uint16),
        ("qs",        ctypes.c_uint8 * 48),
        ("qjl_signs", ctypes.c_uint8 * 16),
    ]

class block_tq4p_d256(ctypes.Structure):
    _fields_ = [
        ("orig_norm", ctypes.c_uint16),
        ("res_d",     ctypes.c_uint16),
        ("qs",        ctypes.c_uint8 * 96),
        ("qjl_signs", ctypes.c_uint8 * 32),
    ]

assert ctypes.sizeof(block_tq4p_d128) == 68
assert ctypes.sizeof(block_tq4p_d256) == 132

lib.ggml_quantize_row_tq4p_d128.restype = None
lib.ggml_quantize_row_tq4p_d128.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(block_tq4p_d128),
    ctypes.c_int64,
]

lib.ggml_quantize_row_tq4p_d256.restype = None
lib.ggml_quantize_row_tq4p_d256.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(block_tq4p_d256),
    ctypes.c_int64,
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

lib.ggml_tqp_prepare_query_d128.restype = None
lib.ggml_tqp_prepare_query_d128.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]

lib.ggml_tqp_prepare_query_d256.restype = None
lib.ggml_tqp_prepare_query_d256.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]


# ---------- Fixtures ----------

@pytest.fixture(scope="module", params=[128, 256])
def d(request):
    return request.param


@pytest.fixture(scope="module")
def constants(d):
    return ref.load_constants(d)


@pytest.fixture(scope="module")
def vectors(d):
    g = torch.Generator().manual_seed(54321)
    x = torch.randn(50, d, generator=g)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _c_quantize(d, x_np):
    if d == 128:
        blk = block_tq4p_d128()
        lib.ggml_quantize_row_tq4p_d128(
            x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(blk), d,
        )
    else:
        blk = block_tq4p_d256()
        lib.ggml_quantize_row_tq4p_d256(
            x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(blk), d,
        )
    return bytes(blk)


# ---------- Tests ----------

def test_byte_exact_quantize(d, constants, vectors):
    """C quantize output must be byte-identical to the Python reference."""
    import numpy as np
    for i, x in enumerate(vectors):
        x_np = x.float().numpy().copy()
        c_bytes = _c_quantize(d, x_np)
        py_bytes = ref.quantize_block(x, constants)
        if c_bytes != py_bytes:
            # Diagnose where.
            diffs = [i for i in range(len(c_bytes)) if c_bytes[i] != py_bytes[i]]
            pytest.fail(f"byte mismatch on vec {i} (d={d}): "
                        f"{len(diffs)} bytes differ, first at offset {diffs[0]} "
                        f"(C={c_bytes[diffs[0]]:02x}, Py={py_bytes[diffs[0]]:02x})")


def test_c_dequantize_matches_python(d, constants, vectors):
    """C dequantize round-trip must match Python dequantize."""
    import numpy as np
    for x in vectors:
        py_blk = ref.quantize_block(x, constants)
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
        assert max_diff < 1e-4, f"dequantize max diff {max_diff}"


def test_c_vec_dot_matches_python(d, constants, vectors):
    """C inner-product estimator must match Python within fp32 accumulation noise."""
    import numpy as np
    keys    = vectors[:25]
    queries = vectors[25:50]

    max_abs = 0.0
    for q in queries:
        # Prepare Sq in C
        q_np = q.float().numpy().copy()
        Sq = (ctypes.c_float * d)()
        if d == 128:
            lib.ggml_tqp_prepare_query_d128(q_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), Sq)
        else:
            lib.ggml_tqp_prepare_query_d256(q_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), Sq)

        for k in keys:
            py_blk = ref.quantize_block(k, constants)
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
    # C uses float32 accumulation, Python uses torch fp32 — differences of
    # order 1e-5 are typical. Anything >1e-3 indicates a real bug.
    assert max_abs < 5e-4, f"C vs Python IP max diff {max_abs:.2e}"


def test_c_matches_turboquant_paper_oracle(d, constants, vectors):
    """End-to-end: C output must also match turboquant.py (transitive via Python ref)."""
    from turboquant import TurboQuantProd
    qp = TurboQuantProd(d, bits=4, seed=42)
    oracle = qp.quantize(vectors)

    for i, x in enumerate(vectors):
        x_np = x.float().numpy().copy()
        c_bytes = _c_quantize(d, x_np)
        # Extract C indices
        if d == 128:
            qs = c_bytes[4:52]; signs = c_bytes[52:68]
        else:
            qs = c_bytes[4:100]; signs = c_bytes[100:132]
        c_idx = ref._unpack_indices_bitplane(qs, d)
        c_signs_pm = ref._unpack_signs(signs, d)
        c_signs_bits = (c_signs_pm < 0).to(torch.uint8)
        oracle_signs_bits = (oracle["qjl_signs"][i] < 0).to(torch.uint8)

        assert torch.equal(c_idx, oracle["mse_indices"][i]), f"vec {i} idx mismatch"
        assert torch.equal(c_signs_bits, oracle_signs_bits), f"vec {i} signs mismatch"
