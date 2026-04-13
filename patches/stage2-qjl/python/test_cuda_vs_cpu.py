"""
CUDA implementation cross-checks against the C CPU reference.

Build first:
    cd patches/stage2-qjl/c
    gcc -O2 -fPIC -shared -o libggml_tq_paper.so ggml-tq-paper.c -lm
    cd ../cuda
    cmake -S . -B build
    cmake --build build -j

Then:
    cd patches/stage2-qjl/python
    pytest test_cuda_vs_cpu.py -v

Per-layer verification: tests iterate over several layer indices in [0, 31]
to confirm the CUDA kernels index into the correct per-layer Π_i / S_i
matrices. A prior revision of these kernels silently used layer 0 for every
block; the parametrization below would have caught that.
"""

from __future__ import annotations

import ctypes
import pathlib
import sys

import pytest
import torch

_HERE = pathlib.Path(__file__).resolve().parent
_CPU_LIB = _HERE.parent / "c" / "libggml_tq_paper.so"
_CUDA_LIB = _HERE.parent / "cuda" / "build" / "libggml_tq_paper_cuda.so"

if not _CPU_LIB.exists():
    pytest.skip(f"CPU test library not built: {_CPU_LIB}", allow_module_level=True)
if not _CUDA_LIB.exists():
    pytest.skip(f"CUDA test library not built: {_CUDA_LIB}", allow_module_level=True)

sys.path.insert(0, str(_HERE))
import tq_paper_reference as ref


# Layer indices to test: layer 0 (backward compat), plus a spread.
# Must match TEST_LAYERS in test_tq_paper.py so CPU↔CUDA↔oracle all agree.
TEST_LAYERS = [0, 1, 7, 15, 31]


class block_tq4p_d128(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("orig_norm", ctypes.c_uint16),
        ("res_d", ctypes.c_uint16),
        ("layer_idx", ctypes.c_uint8),
        ("qs", ctypes.c_uint8 * 48),
        ("qjl_signs", ctypes.c_uint8 * 16),
    ]


class block_tq4p_d256(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("orig_norm", ctypes.c_uint16),
        ("res_d", ctypes.c_uint16),
        ("layer_idx", ctypes.c_uint8),
        ("qs", ctypes.c_uint8 * 96),
        ("qjl_signs", ctypes.c_uint8 * 32),
    ]


assert ctypes.sizeof(block_tq4p_d128) == 69
assert ctypes.sizeof(block_tq4p_d256) == 133

cpu = ctypes.CDLL(str(_CPU_LIB))
cuda = ctypes.CDLL(str(_CUDA_LIB))

cuda.tqp_cuda_device_count.restype = ctypes.c_int
_CUDA_DEVICE_COUNT = cuda.tqp_cuda_device_count()
if _CUDA_DEVICE_COUNT <= 0:
    pytest.skip(
        f"CUDA runtime has no usable device, cudaGetDeviceCount returned {_CUDA_DEVICE_COUNT}",
        allow_module_level=True,
    )

# ---- CPU signatures (match ggml-tq-paper.h as of commit a6b8e87) ----

cpu.ggml_quantize_row_tq4p_d128.restype = None
cpu.ggml_quantize_row_tq4p_d128.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(block_tq4p_d128),
    ctypes.c_int64,
    ctypes.c_uint8,
]
cpu.ggml_quantize_row_tq4p_d256.restype = None
cpu.ggml_quantize_row_tq4p_d256.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(block_tq4p_d256),
    ctypes.c_int64,
    ctypes.c_uint8,
]
cpu.ggml_tqp_prepare_query_d128.restype = None
cpu.ggml_tqp_prepare_query_d128.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_uint8,
]
cpu.ggml_tqp_prepare_query_d256.restype = None
cpu.ggml_tqp_prepare_query_d256.argtypes = cpu.ggml_tqp_prepare_query_d128.argtypes
cpu.ggml_tqp_vec_dot_block_d128.restype = ctypes.c_float
cpu.ggml_tqp_vec_dot_block_d128.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(block_tq4p_d128),
]
cpu.ggml_tqp_vec_dot_block_d256.restype = ctypes.c_float
cpu.ggml_tqp_vec_dot_block_d256.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(block_tq4p_d256),
]

# ---- CUDA signatures (layer_idx threaded through quantize + prepare_query) ----

cuda.tqp_cuda_quantize_row_d128.restype = ctypes.c_int
cuda.tqp_cuda_quantize_row_d128.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_uint8,
]
cuda.tqp_cuda_quantize_row_d256.restype = ctypes.c_int
cuda.tqp_cuda_quantize_row_d256.argtypes = cuda.tqp_cuda_quantize_row_d128.argtypes
cuda.tqp_cuda_prepare_query_d128.restype = ctypes.c_int
cuda.tqp_cuda_prepare_query_d128.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_uint8,
]
cuda.tqp_cuda_prepare_query_d256.restype = ctypes.c_int
cuda.tqp_cuda_prepare_query_d256.argtypes = cuda.tqp_cuda_prepare_query_d128.argtypes

# vec_dot helpers read layer_idx from block->layer_idx; no explicit arg needed.
cuda.tqp_cuda_vec_dot_block_d128.restype = ctypes.c_float
cuda.tqp_cuda_vec_dot_block_d128.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(block_tq4p_d128),
]
cuda.tqp_cuda_vec_dot_block_d256.restype = ctypes.c_float
cuda.tqp_cuda_vec_dot_block_d256.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(block_tq4p_d256),
]
cuda.tqp_cuda_vec_dot_row_d128.restype = ctypes.c_int
cuda.tqp_cuda_vec_dot_row_d128.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(block_tq4p_d128),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int64,
]
cuda.tqp_cuda_vec_dot_row_d256.restype = ctypes.c_int
cuda.tqp_cuda_vec_dot_row_d256.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(block_tq4p_d256),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int64,
]


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


def _as_float_ptr(x):
    return x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _cpu_quantize(d: int, x_np, layer_idx: int):
    if d == 128:
        blk = block_tq4p_d128()
        cpu.ggml_quantize_row_tq4p_d128(_as_float_ptr(x_np), ctypes.byref(blk), d, layer_idx)
    else:
        blk = block_tq4p_d256()
        cpu.ggml_quantize_row_tq4p_d256(_as_float_ptr(x_np), ctypes.byref(blk), d, layer_idx)
    return blk


def _cuda_quantize(d: int, x_np, layer_idx: int):
    if d == 128:
        blk = block_tq4p_d128()
        err = cuda.tqp_cuda_quantize_row_d128(_as_float_ptr(x_np), ctypes.byref(blk), d, layer_idx)
    else:
        blk = block_tq4p_d256()
        err = cuda.tqp_cuda_quantize_row_d256(_as_float_ptr(x_np), ctypes.byref(blk), d, layer_idx)
    assert err == 0
    return blk


def _block_bytes(blk) -> bytes:
    return bytes(memoryview(blk))


def test_byte_identical_quantize(d, vectors, layer_idx):
    for i, x in enumerate(vectors):
        x_np = x.float().numpy().copy()
        cpu_bytes = _block_bytes(_cpu_quantize(d, x_np, layer_idx))
        cuda_bytes = _block_bytes(_cuda_quantize(d, x_np, layer_idx))
        if cuda_bytes != cpu_bytes:
            diffs = [j for j in range(len(cpu_bytes)) if cuda_bytes[j] != cpu_bytes[j]]
            fp16_fields_close = abs(int.from_bytes(cuda_bytes[0:2], "little") - int.from_bytes(cpu_bytes[0:2], "little")) <= 1
            fp16_fields_close &= abs(int.from_bytes(cuda_bytes[2:4], "little") - int.from_bytes(cpu_bytes[2:4], "little")) <= 1
            layer_equal = cuda_bytes[4] == cpu_bytes[4]
            bit_fields_equal = cuda_bytes[5:] == cpu_bytes[5:]
            pytest.fail(
                f"byte mismatch on vec {i} d={d} layer={layer_idx}: {len(diffs)} bytes differ, "
                f"first offset {diffs[0]}, fp16_within_ulp={fp16_fields_close}, "
                f"layer_idx_equal={layer_equal}, qs_signs_exact={bit_fields_equal}"
            )


def test_cuda_writes_layer_idx(d, vectors, layer_idx):
    """CUDA-quantized blocks must store the layer_idx byte at offset 4."""
    x_np = vectors[0].float().numpy().copy()
    blk = _cuda_quantize(d, x_np, layer_idx)
    assert blk.layer_idx == layer_idx, f"expected layer_idx={layer_idx}, got {blk.layer_idx}"


def test_cuda_per_layer_produces_different_bytes(d, vectors):
    """Same input, different layer_idx → different qs / qjl_signs bytes.
    If CUDA silently used layer 0 for everything, these would be identical."""
    x_np = vectors[0].float().numpy().copy()
    blk0 = _cuda_quantize(d, x_np, 0)
    blk_hi = _cuda_quantize(d, x_np, TEST_LAYERS[-1])
    b0 = _block_bytes(blk0)
    bh = _block_bytes(blk_hi)
    # Skip the layer_idx byte at offset 4; require qs or qjl_signs to differ.
    assert b0[5:] != bh[5:], (
        f"qs/qjl_signs identical across layer_idx=0 and {TEST_LAYERS[-1]} at d={d}; "
        f"CUDA is likely using a single global Π for all layers"
    )


def test_prepare_query_agreement(d, constants, vectors, layer_idx):
    for q in vectors[25:35]:
        q_np = q.float().numpy().copy()
        cpu_sq = (ctypes.c_float * d)()
        cuda_sq = (ctypes.c_float * d)()
        cuda_q_rot = (ctypes.c_float * d)()

        if d == 128:
            cpu.ggml_tqp_prepare_query_d128(_as_float_ptr(q_np), cpu_sq, layer_idx)
            err = cuda.tqp_cuda_prepare_query_d128(_as_float_ptr(q_np), cuda_sq, cuda_q_rot, layer_idx)
        else:
            cpu.ggml_tqp_prepare_query_d256(_as_float_ptr(q_np), cpu_sq, layer_idx)
            err = cuda.tqp_cuda_prepare_query_d256(_as_float_ptr(q_np), cuda_sq, cuda_q_rot, layer_idx)

        assert err == 0
        cpu_sq_t = torch.tensor(list(cpu_sq))
        cuda_sq_t = torch.tensor(list(cuda_sq))
        cuda_q_rot_t = torch.tensor(list(cuda_q_rot))
        py_q_rot_t = constants.pi[layer_idx] @ q.float()

        assert (cuda_sq_t - cpu_sq_t).abs().max().item() < 1e-5
        assert (cuda_q_rot_t - py_q_rot_t).abs().max().item() < 1e-5


def test_vec_dot_agreement(d, vectors, layer_idx):
    keys = vectors[:25]
    queries = vectors[25:50]
    max_abs = 0.0

    for q in queries:
        q_np = q.float().numpy().copy()
        for k in keys:
            k_np = k.float().numpy().copy()
            blk = _cpu_quantize(d, k_np, layer_idx)

            sq = (ctypes.c_float * d)()
            if d == 128:
                cpu.ggml_tqp_prepare_query_d128(_as_float_ptr(q_np), sq, layer_idx)
                cpu_ip = cpu.ggml_tqp_vec_dot_block_d128(_as_float_ptr(q_np), sq, ctypes.byref(blk))
                cuda_ip = cuda.tqp_cuda_vec_dot_block_d128(_as_float_ptr(q_np), ctypes.byref(blk))
            else:
                cpu.ggml_tqp_prepare_query_d256(_as_float_ptr(q_np), sq, layer_idx)
                cpu_ip = cpu.ggml_tqp_vec_dot_block_d256(_as_float_ptr(q_np), sq, ctypes.byref(blk))
                cuda_ip = cuda.tqp_cuda_vec_dot_block_d256(_as_float_ptr(q_np), ctypes.byref(blk))
            max_abs = max(max_abs, abs(cpu_ip - cuda_ip))

    assert max_abs < 1e-4, f"CUDA vs CPU vec_dot max diff {max_abs:.2e} at layer {layer_idx}"


def test_dispatch_wrapper_matches_block_api(d, vectors, layer_idx):
    keys = vectors[:5]
    q = vectors[5]
    q_np = q.float().numpy().copy()

    if d == 128:
        BlockArray = block_tq4p_d128 * len(keys)
        blocks = BlockArray(*[_cpu_quantize(d, k.float().numpy().copy(), layer_idx) for k in keys])
        out = (ctypes.c_float * len(keys))()
        err = cuda.tqp_cuda_vec_dot_row_d128(_as_float_ptr(q_np), blocks, out, len(keys))
        block_scores = [
            cuda.tqp_cuda_vec_dot_block_d128(_as_float_ptr(q_np), ctypes.byref(blocks[i]))
            for i in range(len(keys))
        ]
    else:
        BlockArray = block_tq4p_d256 * len(keys)
        blocks = BlockArray(*[_cpu_quantize(d, k.float().numpy().copy(), layer_idx) for k in keys])
        out = (ctypes.c_float * len(keys))()
        err = cuda.tqp_cuda_vec_dot_row_d256(_as_float_ptr(q_np), blocks, out, len(keys))
        block_scores = [
            cuda.tqp_cuda_vec_dot_block_d256(_as_float_ptr(q_np), ctypes.byref(blocks[i]))
            for i in range(len(keys))
        ]

    assert err == 0
    row_scores = list(out)
    assert max(abs(a - b) for a, b in zip(row_scores, block_scores)) < 1e-6
