"""
CUDA implementation cross-checks against the C CPU reference.

Build first:
    cd patches/stage2-qjl/c && make libggml_tq_paper.so
    cd ../cuda && cmake -S . -B build && cmake --build build -j

Then:
    cd ../python && pytest test_cuda_vs_cpu.py -q
"""

from __future__ import annotations

import ctypes
import pathlib
import struct
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


BITS = [2, 3, 4]
BIT_IDS = {2: "b2", 3: "b3", 4: "b4"}
TEST_LAYERS = [0, 1, 7, 15, 31]
ROTATIONS = [ref.TQP_ROT_WHT, ref.TQP_ROT_HAAR]
ROTATION_IDS = {ref.TQP_ROT_WHT: "wht", ref.TQP_ROT_HAAR: "haar"}
Q8_QK = 256


def _block_cls(d: int, bits: int):
    qs_bytes = (d * bits) // 8
    signs_bytes = d // 8

    class Block(ctypes.Structure):
        _pack_ = 1
        _fields_ = [
            ("orig_norm", ctypes.c_uint16),
            ("res_d", ctypes.c_uint16),
            ("layer_idx", ctypes.c_uint8),
            ("qs", ctypes.c_uint8 * qs_bytes),
            ("qjl_signs", ctypes.c_uint8 * signs_bytes),
        ]

    Block.__name__ = f"block_tqp_d{d}_b{bits}"
    return Block


BLOCK_CLASSES = {(d, bits): _block_cls(d, bits) for d in (64, 128, 256) for bits in BITS}


class block_q8k_compat(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("d", ctypes.c_float),
        ("qs", ctypes.c_int8 * Q8_QK),
        ("bsums", ctypes.c_int16 * (Q8_QK // 16)),
    ]


assert ctypes.sizeof(block_q8k_compat) == 292

cpu = ctypes.CDLL(str(_CPU_LIB))
cuda = ctypes.CDLL(str(_CUDA_LIB))

cuda.tqp_cuda_device_count.restype = ctypes.c_int
_CUDA_DEVICE_COUNT = cuda.tqp_cuda_device_count()
if _CUDA_DEVICE_COUNT <= 0:
    pytest.skip(f"CUDA runtime has no usable device, cudaGetDeviceCount returned {_CUDA_DEVICE_COUNT}", allow_module_level=True)


def _declare(lib, name: str, restype, argtypes):
    fn = getattr(lib, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


def _cpu_name(kind: str, d: int, bits: int) -> str:
    if kind == "quantize":
        return f"ggml_quantize_row_tqp_d{d}_b{bits}"
    if kind == "dequantize":
        return f"ggml_dequantize_row_tqp_d{d}_b{bits}"
    if kind == "prepare":
        return f"ggml_tqp_prepare_query_d{d}_b{bits}"
    if kind == "vec_dot_block":
        return f"ggml_tqp_vec_dot_block_d{d}_b{bits}"
    if kind == "vec_dot_q8k":
        return f"ggml_vec_dot_tqp_d{d}_b{bits}_q8k"
    raise ValueError(kind)


def _cpu_legacy_name(kind: str, d: int) -> str:
    if kind == "quantize":
        return f"ggml_quantize_row_tq4p_d{d}"
    if kind == "dequantize":
        return f"ggml_dequantize_row_tq4p_d{d}"
    if kind == "prepare":
        return f"ggml_tqp_prepare_query_d{d}"
    if kind == "vec_dot_block":
        return f"ggml_tqp_vec_dot_block_d{d}"
    if kind == "vec_dot_q8k":
        return f"ggml_vec_dot_tq4p_d{d}_q8k"
    raise ValueError(kind)


def _cuda_name(kind: str, d: int, bits: int) -> str:
    if kind == "quantize":
        return f"tqp_cuda_quantize_row_d{d}_b{bits}"
    if kind == "dequantize_f32":
        return f"tqp_cuda_dequantize_row_d{d}_b{bits}_f32"
    if kind == "dequantize_f16":
        return f"tqp_cuda_dequantize_row_d{d}_b{bits}_f16"
    if kind == "prepare":
        return f"tqp_cuda_prepare_query_d{d}_b{bits}"
    if kind == "vec_dot_block":
        return f"tqp_cuda_vec_dot_block_d{d}_b{bits}"
    if kind == "vec_dot_row":
        return f"tqp_cuda_vec_dot_row_d{d}_b{bits}"
    if kind == "vec_dot_q8k":
        return f"tqp_cuda_vec_dot_q8k_d{d}_b{bits}"
    raise ValueError(kind)


def _cuda_legacy_name(kind: str, d: int) -> str:
    if kind == "quantize":
        return f"tqp_cuda_quantize_row_d{d}"
    if kind == "dequantize_f32":
        return f"tqp_cuda_dequantize_row_d{d}_f32"
    if kind == "dequantize_f16":
        return f"tqp_cuda_dequantize_row_d{d}_f16"
    if kind == "prepare":
        return f"tqp_cuda_prepare_query_d{d}"
    if kind == "vec_dot_block":
        return f"tqp_cuda_vec_dot_block_d{d}"
    if kind == "vec_dot_row":
        return f"tqp_cuda_vec_dot_row_d{d}"
    if kind == "vec_dot_q8k":
        return f"tqp_cuda_vec_dot_q8k_d{d}"
    raise ValueError(kind)


CPU = {}
CUDA = {}

for d in (64, 128, 256):
    for bits in BITS:
        block_cls = BLOCK_CLASSES[(d, bits)]
        CPU[(d, bits, "quantize")] = _declare(
            cpu,
            _cpu_name("quantize", d, bits),
            None,
            [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(block_cls), ctypes.c_int64, ctypes.c_uint8],
        )
        CPU[(d, bits, "dequantize")] = _declare(
            cpu,
            _cpu_name("dequantize", d, bits),
            None,
            [ctypes.POINTER(block_cls), ctypes.POINTER(ctypes.c_float), ctypes.c_int64],
        )
        CPU[(d, bits, "prepare")] = _declare(
            cpu,
            _cpu_name("prepare", d, bits),
            None,
            [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_uint8],
        )
        CPU[(d, bits, "vec_dot_block")] = _declare(
            cpu,
            _cpu_name("vec_dot_block", d, bits),
            ctypes.c_float,
            [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(block_cls)],
        )
        CPU[(d, bits, "vec_dot_q8k")] = _declare(
            cpu,
            _cpu_name("vec_dot_q8k", d, bits),
            None,
            [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ],
        )

        CUDA[(d, bits, "quantize")] = _declare(
            cuda,
            _cuda_name("quantize", d, bits),
            ctypes.c_int,
            [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int64, ctypes.c_uint8],
        )
        CUDA[(d, bits, "dequantize_f32")] = _declare(
            cuda,
            _cuda_name("dequantize_f32", d, bits),
            ctypes.c_int,
            [ctypes.POINTER(block_cls), ctypes.POINTER(ctypes.c_float), ctypes.c_int64],
        )
        CUDA[(d, bits, "dequantize_f16")] = _declare(
            cuda,
            _cuda_name("dequantize_f16", d, bits),
            ctypes.c_int,
            [ctypes.POINTER(block_cls), ctypes.POINTER(ctypes.c_uint16), ctypes.c_int64],
        )
        CUDA[(d, bits, "prepare")] = _declare(
            cuda,
            _cuda_name("prepare", d, bits),
            ctypes.c_int,
            [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_uint8],
        )
        CUDA[(d, bits, "vec_dot_block")] = _declare(
            cuda,
            _cuda_name("vec_dot_block", d, bits),
            ctypes.c_float,
            [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(block_cls)],
        )
        CUDA[(d, bits, "vec_dot_row")] = _declare(
            cuda,
            _cuda_name("vec_dot_row", d, bits),
            ctypes.c_int,
            [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(block_cls), ctypes.POINTER(ctypes.c_float), ctypes.c_int64],
        )
        CUDA[(d, bits, "vec_dot_q8k")] = _declare(
            cuda,
            _cuda_name("vec_dot_q8k", d, bits),
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.POINTER(block_cls), ctypes.POINTER(ctypes.c_float), ctypes.c_int64],
        )

for d in (64, 128, 256):
    legacy_block = BLOCK_CLASSES[(d, 3)]
    CPU[(d, 3, "legacy_quantize")] = _declare(
        cpu,
        _cpu_legacy_name("quantize", d),
        None,
        [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(legacy_block), ctypes.c_int64, ctypes.c_uint8],
    )
    CPU[(d, 3, "legacy_dequantize")] = _declare(
        cpu,
        _cpu_legacy_name("dequantize", d),
        None,
        [ctypes.POINTER(legacy_block), ctypes.POINTER(ctypes.c_float), ctypes.c_int64],
    )
    CPU[(d, 3, "legacy_prepare")] = _declare(
        cpu,
        _cpu_legacy_name("prepare", d),
        None,
        [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_uint8],
    )
    CPU[(d, 3, "legacy_vec_dot_block")] = _declare(
        cpu,
        _cpu_legacy_name("vec_dot_block", d),
        ctypes.c_float,
        [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(legacy_block)],
    )
    CPU[(d, 3, "legacy_vec_dot_q8k")] = _declare(
        cpu,
        _cpu_legacy_name("vec_dot_q8k", d),
        None,
        [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ],
    )

    CUDA[(d, 3, "legacy_quantize")] = _declare(
        cuda,
        _cuda_legacy_name("quantize", d),
        ctypes.c_int,
        [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int64, ctypes.c_uint8],
    )
    CUDA[(d, 3, "legacy_dequantize_f32")] = _declare(
        cuda,
        _cuda_legacy_name("dequantize_f32", d),
        ctypes.c_int,
        [ctypes.POINTER(legacy_block), ctypes.POINTER(ctypes.c_float), ctypes.c_int64],
    )
    CUDA[(d, 3, "legacy_dequantize_f16")] = _declare(
        cuda,
        _cuda_legacy_name("dequantize_f16", d),
        ctypes.c_int,
        [ctypes.POINTER(legacy_block), ctypes.POINTER(ctypes.c_uint16), ctypes.c_int64],
    )
    CUDA[(d, 3, "legacy_prepare")] = _declare(
        cuda,
        _cuda_legacy_name("prepare", d),
        ctypes.c_int,
        [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_uint8],
    )
    CUDA[(d, 3, "legacy_vec_dot_block")] = _declare(
        cuda,
        _cuda_legacy_name("vec_dot_block", d),
        ctypes.c_float,
        [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(legacy_block)],
    )
    CUDA[(d, 3, "legacy_vec_dot_row")] = _declare(
        cuda,
        _cuda_legacy_name("vec_dot_row", d),
        ctypes.c_int,
        [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(legacy_block), ctypes.POINTER(ctypes.c_float), ctypes.c_int64],
    )
    CUDA[(d, 3, "legacy_vec_dot_q8k")] = _declare(
        cuda,
        _cuda_legacy_name("vec_dot_q8k", d),
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.POINTER(legacy_block), ctypes.POINTER(ctypes.c_float), ctypes.c_int64],
    )


@pytest.fixture(scope="module", params=[64, 128, 256])
def d(request):
    return request.param


@pytest.fixture(scope="module", params=BITS, ids=lambda b: BIT_IDS[b])
def bits(request):
    return request.param


@pytest.fixture(scope="module")
def constants(d, bits):
    return ref.load_constants(d, bits)


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


def _as_float_ptr(x):
    return x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _block_bytes(blk) -> bytes:
    return bytes(memoryview(blk))


def _cpu_quantize(d: int, bits: int, x_np, layer_byte: int):
    blk = BLOCK_CLASSES[(d, bits)]()
    CPU[(d, bits, "quantize")](_as_float_ptr(x_np), ctypes.byref(blk), d, layer_byte)
    return blk


def _cuda_quantize(d: int, bits: int, x_np, layer_byte: int):
    blk = BLOCK_CLASSES[(d, bits)]()
    err = CUDA[(d, bits, "quantize")](_as_float_ptr(x_np), ctypes.byref(blk), d, layer_byte)
    assert err == 0
    return blk


def _cpu_dequantize(d: int, bits: int, blk):
    out = (ctypes.c_float * d)()
    CPU[(d, bits, "dequantize")](ctypes.byref(blk), out, d)
    return torch.tensor(list(out))


def _cuda_dequantize_f32(d: int, bits: int, blk):
    out = (ctypes.c_float * d)()
    err = CUDA[(d, bits, "dequantize_f32")](ctypes.byref(blk), out, d)
    assert err == 0
    return torch.tensor(list(out))


def _cuda_dequantize_f16(d: int, bits: int, blk):
    out = (ctypes.c_uint16 * d)()
    err = CUDA[(d, bits, "dequantize_f16")](ctypes.byref(blk), out, d)
    assert err == 0
    return torch.tensor([
        struct.unpack("<e", int(raw).to_bytes(2, "little"))[0]
        for raw in out
    ])


def _quantize_q8k(x_fp32):
    x = x_fp32.flatten()
    n_blocks = (len(x) + Q8_QK - 1) // Q8_QK
    blocks = (block_q8k_compat * n_blocks)()
    for b in range(n_blocks):
        chunk = x[b * Q8_QK : (b + 1) * Q8_QK]
        amax = max(abs(float(v)) for v in chunk) if len(chunk) else 0.0
        d = amax / 127.0 if amax > 0 else 1.0
        blocks[b].d = d
        for i, v in enumerate(chunk):
            blocks[b].qs[i] = max(-128, min(127, round(float(v) / d)))
    return blocks


@pytest.mark.parametrize("bits", [3], ids=["b3"])
def test_legacy_b3_symbols_match_new_api(d, vectors, layer_idx, rotation, bits):
    x_np = vectors[0].float().numpy().copy()
    layer_byte = ref.layer_byte(layer_idx, rotation)

    cpu_new = BLOCK_CLASSES[(d, bits)]()
    cpu_old = BLOCK_CLASSES[(d, bits)]()
    CPU[(d, bits, "quantize")](_as_float_ptr(x_np), ctypes.byref(cpu_new), d, layer_byte)
    CPU[(d, bits, "legacy_quantize")](_as_float_ptr(x_np), ctypes.byref(cpu_old), d, layer_byte)
    assert _block_bytes(cpu_new) == _block_bytes(cpu_old)

    cuda_new = BLOCK_CLASSES[(d, bits)]()
    cuda_old = BLOCK_CLASSES[(d, bits)]()
    assert CUDA[(d, bits, "quantize")](_as_float_ptr(x_np), ctypes.byref(cuda_new), d, layer_byte) == 0
    assert CUDA[(d, bits, "legacy_quantize")](_as_float_ptr(x_np), ctypes.byref(cuda_old), d, layer_byte) == 0
    assert _block_bytes(cuda_new) == _block_bytes(cuda_old)


@pytest.mark.parametrize("bits", [3], ids=["b3"])
def test_byte_identical_quantize_b3(d, vectors, layer_idx, rotation, bits):
    layer_byte = ref.layer_byte(layer_idx, rotation)
    for i, x in enumerate(vectors[:10]):
        x_np = x.float().numpy().copy()
        cpu_bytes = _block_bytes(_cpu_quantize(d, bits, x_np, layer_byte))
        cuda_bytes = _block_bytes(_cuda_quantize(d, bits, x_np, layer_byte))
        if cuda_bytes != cpu_bytes:
            diffs = [j for j in range(len(cpu_bytes)) if cuda_bytes[j] != cpu_bytes[j]]
            pytest.fail(
                f"byte mismatch vec {i} d={d} bits={bits} layer={layer_idx} rot={ROTATION_IDS[rotation]}: "
                f"{len(diffs)} bytes differ, first offset {diffs[0]}"
            )


def test_layer_byte_stored_in_block(d, bits, vectors, layer_idx, rotation):
    layer_byte = ref.layer_byte(layer_idx, rotation)
    x_np = vectors[0].float().numpy().copy()
    for blk in (
        _cpu_quantize(d, bits, x_np, layer_byte),
        _cuda_quantize(d, bits, x_np, layer_byte),
    ):
        assert ref.extract_layer(blk.layer_idx) == layer_idx
        assert ref.extract_rotation(blk.layer_idx) == rotation


def test_dequantize_agreement_f32(d, bits, vectors, layer_idx, rotation):
    layer_byte = ref.layer_byte(layer_idx, rotation)
    max_abs = 0.0
    for x in vectors[:10]:
        blk = _cpu_quantize(d, bits, x.float().numpy().copy(), layer_byte)
        cpu_out = _cpu_dequantize(d, bits, blk)
        cuda_out = _cuda_dequantize_f32(d, bits, blk)
        max_abs = max(max_abs, (cuda_out - cpu_out).abs().max().item())

    assert max_abs < 1e-5


def test_dequantize_agreement_f16(d, bits, vectors, layer_idx, rotation):
    layer_byte = ref.layer_byte(layer_idx, rotation)
    max_abs = 0.0
    for x in vectors[:10]:
        blk = _cpu_quantize(d, bits, x.float().numpy().copy(), layer_byte)
        cpu_out = _cpu_dequantize(d, bits, blk).half().float()
        cuda_out = _cuda_dequantize_f16(d, bits, blk).float()
        max_abs = max(max_abs, (cuda_out - cpu_out).abs().max().item())

    assert max_abs == 0.0


def test_prepare_query_agreement(d, bits, constants, vectors, layer_idx, rotation):
    layer_byte = ref.layer_byte(layer_idx, rotation)
    for q in vectors[25:35]:
        q_np = q.float().numpy().copy()
        cpu_sq = (ctypes.c_float * d)()
        cuda_sq = (ctypes.c_float * d)()
        cuda_q_rot = (ctypes.c_float * d)()

        CPU[(d, bits, "prepare")](_as_float_ptr(q_np), cpu_sq, layer_byte)
        err = CUDA[(d, bits, "prepare")](_as_float_ptr(q_np), cuda_sq, cuda_q_rot, layer_byte)
        assert err == 0

        cpu_sq_t = torch.tensor(list(cpu_sq))
        cuda_sq_t = torch.tensor(list(cuda_sq))
        cuda_q_rot_t = torch.tensor(list(cuda_q_rot))
        py_q_rot_t = ref.rot_apply(rotation, constants.sigma[layer_idx], constants.pi[layer_idx], q.float())

        assert (cuda_sq_t - cpu_sq_t).abs().max().item() < 1e-5
        assert (cuda_q_rot_t - py_q_rot_t).abs().max().item() < 1e-4


def test_vec_dot_agreement(d, bits, vectors, layer_idx, rotation):
    keys = vectors[:10]
    queries = vectors[10:20]
    layer_byte = ref.layer_byte(layer_idx, rotation)
    max_abs = 0.0

    for q in queries:
        q_np = q.float().numpy().copy()
        for k in keys:
            blk = _cpu_quantize(d, bits, k.float().numpy().copy(), layer_byte)
            sq = (ctypes.c_float * d)()
            CPU[(d, bits, "prepare")](_as_float_ptr(q_np), sq, layer_byte)
            cpu_ip = CPU[(d, bits, "vec_dot_block")](_as_float_ptr(q_np), sq, ctypes.byref(blk))
            cuda_ip = CUDA[(d, bits, "vec_dot_block")](_as_float_ptr(q_np), ctypes.byref(blk))
            max_abs = max(max_abs, abs(cpu_ip - cuda_ip))

    assert max_abs < 1e-4


def test_dispatch_wrapper_matches_block_api(d, bits, vectors, layer_idx, rotation):
    keys = vectors[:5]
    q_np = vectors[5].float().numpy().copy()
    layer_byte = ref.layer_byte(layer_idx, rotation)

    block_cls = BLOCK_CLASSES[(d, bits)]
    BlockArray = block_cls * len(keys)
    blocks = BlockArray(*[_cpu_quantize(d, bits, k.float().numpy().copy(), layer_byte) for k in keys])
    out = (ctypes.c_float * len(keys))()

    err = CUDA[(d, bits, "vec_dot_row")](_as_float_ptr(q_np), blocks, out, len(keys))
    assert err == 0

    block_scores = [
        CUDA[(d, bits, "vec_dot_block")](_as_float_ptr(q_np), ctypes.byref(blocks[i]))
        for i in range(len(keys))
    ]
    assert max(abs(a - b) for a, b in zip(list(out), block_scores)) < 1e-6


def test_q8k_cuda_vs_cpu(d, bits, vectors, layer_idx, rotation):
    n_tqp_per_q8k = Q8_QK // d
    keys = vectors[: n_tqp_per_q8k * 4]
    query_vec = torch.cat([vectors[20 + i] for i in range(n_tqp_per_q8k)], dim=0)
    layer_byte = ref.layer_byte(layer_idx, rotation)

    block_cls = BLOCK_CLASSES[(d, bits)]
    BlockArray = block_cls * len(keys)
    blocks = BlockArray(*[_cpu_quantize(d, bits, k.float().numpy().copy(), layer_byte) for k in keys])
    q8k = _quantize_q8k(query_vec.float().numpy())

    for start in range(0, len(keys), n_tqp_per_q8k):
        block_slice = ctypes.cast(
            ctypes.byref(blocks, start * ctypes.sizeof(block_cls)),
            ctypes.POINTER(block_cls),
        )
        cuda_out = (ctypes.c_float * n_tqp_per_q8k)()
        cpu_out = ctypes.c_float(0.0)

        CPU[(d, bits, "vec_dot_q8k")](
            Q8_QK,
            ctypes.byref(cpu_out),
            ctypes.sizeof(ctypes.c_float),
            block_slice,
            ctypes.sizeof(block_cls),
            q8k,
            ctypes.sizeof(block_q8k_compat),
            1,
        )
        err = CUDA[(d, bits, "vec_dot_q8k")](q8k, block_slice, cuda_out, n_tqp_per_q8k)
        assert err == 0

        cuda_total = sum(cuda_out)
        assert abs(cpu_out.value - cuda_total) < 1e-4
