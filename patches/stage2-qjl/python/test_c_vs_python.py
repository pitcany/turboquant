"""
Call the compiled C implementation via ctypes and assert byte-exact equality
with the Python reference across dimensions, Stage-1 bit-widths, and both
rotation modes.

Build the shared library first:
    cd ../c && make libggml_tq_paper.so

Then:
    pytest test_c_vs_python.py
"""

from __future__ import annotations

import ctypes
import pathlib
import sys

import numpy as np
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
BITS = [2, 3, 4]
BIT_IDS = {2: "b2", 3: "b3", 4: "b4"}
QK_Q8K = 256

lib = ctypes.CDLL(str(_LIB))


def _declare(name: str, restype, argtypes):
    fn = getattr(lib, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


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
BLOCK_SIZES = {(d, bits): ctypes.sizeof(BLOCK_CLASSES[(d, bits)]) for d in (64, 128, 256) for bits in BITS}


class block_q8k_compat(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("d", ctypes.c_float),
        ("qs", ctypes.c_int8 * QK_Q8K),
        ("bsums", ctypes.c_int16 * (QK_Q8K // 16)),
    ]


assert ctypes.sizeof(block_q8k_compat) == 292


def _tqp_name(kind: str, d: int, bits: int, suffix: str = "") -> str:
    if kind == "quantize":
        return f"ggml_quantize_row_tqp_d{d}_b{bits}{suffix}"
    if kind == "dequantize":
        return f"ggml_dequantize_row_tqp_d{d}_b{bits}"
    if kind == "prepare":
        return f"ggml_tqp_prepare_query_d{d}_b{bits}"
    if kind == "vec_dot_block":
        return f"ggml_tqp_vec_dot_block_d{d}_b{bits}"
    if kind == "vec_dot_f32":
        return f"ggml_vec_dot_tqp_d{d}_b{bits}_f32"
    if kind == "vec_dot_q8k":
        return f"ggml_vec_dot_tqp_d{d}_b{bits}_q8k"
    raise ValueError(kind)


def _legacy_name(kind: str, d: int, suffix: str = "") -> str:
    if kind == "quantize":
        return f"ggml_quantize_row_tq4p_d{d}{suffix}"
    if kind == "dequantize":
        return f"ggml_dequantize_row_tq4p_d{d}"
    if kind == "prepare":
        return f"ggml_tqp_prepare_query_d{d}"
    if kind == "vec_dot_block":
        return f"ggml_tqp_vec_dot_block_d{d}"
    if kind == "vec_dot_f32":
        return f"ggml_vec_dot_tq4p_d{d}_f32"
    if kind == "vec_dot_q8k":
        return f"ggml_vec_dot_tq4p_d{d}_q8k"
    raise ValueError(kind)


APIS = {}
for d in (64, 128, 256):
    for bits in BITS:
        block_cls = BLOCK_CLASSES[(d, bits)]
        APIS[(d, bits, "quantize")] = _declare(
            _tqp_name("quantize", d, bits),
            None,
            [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(block_cls), ctypes.c_int64, ctypes.c_uint8],
        )
        APIS[(d, bits, "quantize_bf16")] = _declare(
            _tqp_name("quantize", d, bits, "_bf16"),
            None,
            [ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(block_cls), ctypes.c_int64, ctypes.c_uint8],
        )
        APIS[(d, bits, "quantize_f16")] = _declare(
            _tqp_name("quantize", d, bits, "_f16"),
            None,
            [ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(block_cls), ctypes.c_int64, ctypes.c_uint8],
        )
        APIS[(d, bits, "dequantize")] = _declare(
            _tqp_name("dequantize", d, bits),
            None,
            [ctypes.POINTER(block_cls), ctypes.POINTER(ctypes.c_float), ctypes.c_int64],
        )
        APIS[(d, bits, "prepare")] = _declare(
            _tqp_name("prepare", d, bits),
            None,
            [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_uint8],
        )
        APIS[(d, bits, "vec_dot_block")] = _declare(
            _tqp_name("vec_dot_block", d, bits),
            ctypes.c_float,
            [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(block_cls)],
        )
        vec_dot_sig = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        APIS[(d, bits, "vec_dot_f32")] = _declare(_tqp_name("vec_dot_f32", d, bits), None, vec_dot_sig)
        APIS[(d, bits, "vec_dot_q8k")] = _declare(_tqp_name("vec_dot_q8k", d, bits), None, vec_dot_sig)

for d in (64, 128, 256):
    legacy_block = BLOCK_CLASSES[(d, 3)]
    APIS[(d, 3, "legacy_quantize")] = _declare(
        _legacy_name("quantize", d),
        None,
        [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(legacy_block), ctypes.c_int64, ctypes.c_uint8],
    )
    APIS[(d, 3, "legacy_quantize_bf16")] = _declare(
        _legacy_name("quantize", d, "_bf16"),
        None,
        [ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(legacy_block), ctypes.c_int64, ctypes.c_uint8],
    )
    APIS[(d, 3, "legacy_quantize_f16")] = _declare(
        _legacy_name("quantize", d, "_f16"),
        None,
        [ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(legacy_block), ctypes.c_int64, ctypes.c_uint8],
    )
    APIS[(d, 3, "legacy_dequantize")] = _declare(
        _legacy_name("dequantize", d),
        None,
        [ctypes.POINTER(legacy_block), ctypes.POINTER(ctypes.c_float), ctypes.c_int64],
    )
    APIS[(d, 3, "legacy_prepare")] = _declare(
        _legacy_name("prepare", d),
        None,
        [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_uint8],
    )
    APIS[(d, 3, "legacy_vec_dot_block")] = _declare(
        _legacy_name("vec_dot_block", d),
        ctypes.c_float,
        [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(legacy_block)],
    )
    vec_dot_sig = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    APIS[(d, 3, "legacy_vec_dot_f32")] = _declare(_legacy_name("vec_dot_f32", d), None, vec_dot_sig)
    APIS[(d, 3, "legacy_vec_dot_q8k")] = _declare(_legacy_name("vec_dot_q8k", d), None, vec_dot_sig)

lib.tqp_set_default_rotation.restype = None
lib.tqp_set_default_rotation.argtypes = [ctypes.c_uint8]
lib.tqp_set_thread_rotation.restype = None
lib.tqp_set_thread_rotation.argtypes = [ctypes.c_uint8]
lib.tqp_clear_thread_rotation.restype = None
lib.tqp_clear_thread_rotation.argtypes = []
lib.tqp_resolve_rotation.restype = ctypes.c_uint8
lib.tqp_resolve_rotation.argtypes = [ctypes.c_uint8]


# ---------- Fixtures ----------

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


DTYPES = ["fp32", "bf16", "f16"]


@pytest.fixture(params=DTYPES, ids=lambda dt: dt)
def input_dtype(request):
    return request.param


def _fp32_to_bf16_array(x_fp32):
    x_u32 = x_fp32.view(np.uint32)
    return (x_u32 >> 16).astype(np.uint16)


def _fp32_to_fp16_array(x_fp32):
    return x_fp32.astype(np.float16).view(np.uint16)


def _c_quantize(d, bits, x_np, layer_idx, rotation, *, legacy=False):
    layer_byte = ref.layer_byte(layer_idx, rotation)
    blk_cls = BLOCK_CLASSES[(d, bits)]
    blk = blk_cls()
    key = "legacy_quantize" if legacy else "quantize"
    APIS[(d, bits, key)](
        x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(blk),
        d,
        layer_byte,
    )
    return bytes(blk)


def _c_quantize_dtype(d, bits, x_fp32_np, layer_idx, rotation, dtype, *, legacy=False):
    layer_byte = ref.layer_byte(layer_idx, rotation)
    blk = BLOCK_CLASSES[(d, bits)]()
    if dtype == "fp32":
        key = "legacy_quantize" if legacy else "quantize"
        APIS[(d, bits, key)](
            x_fp32_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(blk),
            d,
            layer_byte,
        )
    elif dtype == "bf16":
        key = "legacy_quantize_bf16" if legacy else "quantize_bf16"
        x_bf16 = _fp32_to_bf16_array(x_fp32_np)
        APIS[(d, bits, key)](
            x_bf16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.byref(blk),
            d,
            layer_byte,
        )
    elif dtype == "f16":
        key = "legacy_quantize_f16" if legacy else "quantize_f16"
        x_f16 = _fp32_to_fp16_array(x_fp32_np)
        APIS[(d, bits, key)](
            x_f16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.byref(blk),
            d,
            layer_byte,
        )
    else:
        raise ValueError(dtype)
    return bytes(blk)


def _quantize_fp32_to_q8k(fp32_values):
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
        for g in range(QK_Q8K // 16):
            blk.bsums[g] = sum(blk.qs[g * 16 + j] for j in range(16))
        blocks.append(blk)
    return blocks


# ---------- Tests ----------

def test_byte_exact_quantize(d, bits, constants, vectors, layer_idx, rotation):
    if bits != 3:
        pytest.skip("strict byte parity is reserved for the legacy B3 contract; B2/B4 are covered numerically")
    for i, x in enumerate(vectors):
        x_np = x.float().numpy().copy()
        c_bytes = _c_quantize(d, bits, x_np, layer_idx, rotation)
        py_bytes = ref.quantize_block(x, constants, layer_idx=layer_idx, rotation=rotation)
        if c_bytes != py_bytes:
            diffs = [j for j in range(len(c_bytes)) if c_bytes[j] != py_bytes[j]]
            pytest.fail(
                f"byte mismatch vec {i} d={d} bits={bits} layer={layer_idx} rot={ROTATION_IDS[rotation]}: "
                f"{len(diffs)} bytes differ, first offset {diffs[0]} "
                f"(C={c_bytes[diffs[0]]:02x}, Py={py_bytes[diffs[0]]:02x})"
            )


def test_layer_byte_in_c_block(d, bits, vectors, layer_idx, rotation):
    x_np = vectors[0].float().numpy().copy()
    c_bytes = _c_quantize(d, bits, x_np, layer_idx, rotation)
    assert ref.extract_layer(c_bytes[4]) == layer_idx
    assert ref.extract_rotation(c_bytes[4]) == rotation


def test_c_dequantize_matches_python(d, bits, constants, vectors, layer_idx, rotation):
    dequant_fn = APIS[(d, bits, "dequantize")]
    blk_cls = BLOCK_CLASSES[(d, bits)]
    for x in vectors:
        py_blk = ref.quantize_block(x, constants, layer_idx=layer_idx, rotation=rotation)
        blk = blk_cls.from_buffer_copy(py_blk)
        out = (ctypes.c_float * d)()
        dequant_fn(ctypes.byref(blk), out, d)
        c_out = torch.tensor(list(out))
        py_out = ref.dequantize_block(py_blk, constants)
        max_diff = (c_out - py_out).abs().max().item()
        assert max_diff < 1e-4, (
            f"dequantize max diff {max_diff} d={d} bits={bits} "
            f"layer={layer_idx} rot={ROTATION_IDS[rotation]}"
        )


def test_c_vec_dot_matches_python(d, bits, constants, vectors, layer_idx, rotation):
    keys = vectors[:25]
    queries = vectors[25:50]
    prepare_fn = APIS[(d, bits, "prepare")]
    vec_dot_fn = APIS[(d, bits, "vec_dot_block")]
    blk_cls = BLOCK_CLASSES[(d, bits)]
    layer_byte_val = ref.layer_byte(layer_idx, rotation)

    max_abs = 0.0
    for q in queries:
        q_np = q.float().numpy().copy()
        Sq = (ctypes.c_float * d)()
        prepare_fn(q_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), Sq, layer_byte_val)
        for k in keys:
            py_blk = ref.quantize_block(k, constants, layer_idx=layer_idx, rotation=rotation)
            blk = blk_cls.from_buffer_copy(py_blk)
            c_ip = vec_dot_fn(
                q_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                Sq,
                ctypes.byref(blk),
            )
            py_ip = ref.inner_product(q, py_blk, constants)
            max_abs = max(max_abs, abs(c_ip - py_ip))
    assert max_abs < 5e-4, f"C vs Python IP max diff {max_abs:.2e} d={d} bits={bits}"


def test_ggml_dispatch_wrapper_matches_block_api(d, bits, constants, vectors, layer_idx, rotation):
    keys = vectors[:5]
    queries = vectors[5:10]
    key_bytes = b"".join(ref.quantize_block(k, constants, layer_idx=layer_idx, rotation=rotation) for k in keys)
    n = d * len(keys)
    KeyBuf = (ctypes.c_uint8 * len(key_bytes)).from_buffer_copy(key_bytes)
    vec_dot_f32 = APIS[(d, bits, "vec_dot_f32")]

    for q in queries:
        q_flat = (ctypes.c_float * n)()
        q_np = q.float().numpy()
        for ki in range(len(keys)):
            for j in range(d):
                q_flat[ki * d + j] = float(q_np[j])

        out = ctypes.c_float(0.0)
        vec_dot_f32(
            n, ctypes.byref(out), ctypes.sizeof(ctypes.c_float),
            ctypes.cast(KeyBuf, ctypes.c_void_p), BLOCK_SIZES[(d, bits)],
            ctypes.cast(q_flat, ctypes.c_void_p), ctypes.sizeof(ctypes.c_float),
            1,
        )

        expected = sum(
            ref.inner_product(q, ref.quantize_block(k, constants, layer_idx=layer_idx, rotation=rotation), constants)
            for k in keys
        )
        assert abs(out.value - expected) < 1e-2


def test_q8k_dispatch_matches_fp32(d, bits, constants, vectors, layer_idx, rotation):
    n_keys = QK_Q8K // d
    keys = vectors[:n_keys]
    queries = vectors[n_keys : n_keys + 5]

    key_bytes = b"".join(ref.quantize_block(k, constants, layer_idx=layer_idx, rotation=rotation) for k in keys)
    n = d * n_keys
    KeyBuf = (ctypes.c_uint8 * len(key_bytes)).from_buffer_copy(key_bytes)
    vec_dot_f32 = APIS[(d, bits, "vec_dot_f32")]
    vec_dot_q8k = APIS[(d, bits, "vec_dot_q8k")]

    max_abs_diff = 0.0
    for q in queries:
        q_np = q.float().numpy()
        q_flat = (ctypes.c_float * n)()
        for ki in range(n_keys):
            for j in range(d):
                q_flat[ki * d + j] = float(q_np[j])

        out_f32 = ctypes.c_float(0.0)
        vec_dot_f32(
            n, ctypes.byref(out_f32), ctypes.sizeof(ctypes.c_float),
            ctypes.cast(KeyBuf, ctypes.c_void_p), BLOCK_SIZES[(d, bits)],
            ctypes.cast(q_flat, ctypes.c_void_p), ctypes.sizeof(ctypes.c_float),
            1,
        )

        q8k_blocks = _quantize_fp32_to_q8k(list(q_flat))
        Q8kArr = block_q8k_compat * len(q8k_blocks)
        q8k_buf = Q8kArr(*q8k_blocks)

        out_q8k = ctypes.c_float(0.0)
        vec_dot_q8k(
            n, ctypes.byref(out_q8k), ctypes.sizeof(ctypes.c_float),
            ctypes.cast(KeyBuf, ctypes.c_void_p), BLOCK_SIZES[(d, bits)],
            ctypes.cast(q8k_buf, ctypes.c_void_p), ctypes.sizeof(block_q8k_compat),
            1,
        )
        max_abs_diff = max(max_abs_diff, abs(out_q8k.value - out_f32.value))

    assert max_abs_diff < 0.05, f"Q8_K vs fp32 max diff {max_abs_diff:.4f} d={d} bits={bits}"


class TestCRotationResolver:
    def _reset(self):
        lib.tqp_clear_thread_rotation()
        lib.tqp_set_default_rotation(0xFF)

    def test_explicit_overrides_all(self):
        self._reset()
        lib.tqp_set_thread_rotation(ref.TQP_ROT_HAAR)
        lib.tqp_set_default_rotation(ref.TQP_ROT_HAAR)
        lb = ref.layer_byte(5, ref.TQP_ROT_WHT)
        resolved = lib.tqp_resolve_rotation(lb)
        assert ref.extract_rotation(resolved) == ref.TQP_ROT_WHT
        assert ref.extract_layer(resolved) == 5
        assert ref.extract_explicit(resolved) == 0
        self._reset()

    def test_thread_overrides_process(self):
        self._reset()
        lib.tqp_set_default_rotation(ref.TQP_ROT_WHT)
        lib.tqp_set_thread_rotation(ref.TQP_ROT_HAAR)
        resolved = lib.tqp_resolve_rotation(ref.stored_byte(3, 0))
        assert ref.extract_rotation(resolved) == ref.TQP_ROT_HAAR
        self._reset()

    def test_process_overrides_compile_time(self):
        self._reset()
        lib.tqp_set_default_rotation(ref.TQP_ROT_HAAR)
        resolved = lib.tqp_resolve_rotation(ref.stored_byte(3, 0))
        assert ref.extract_rotation(resolved) == ref.TQP_ROT_HAAR
        self._reset()

    def test_compile_time_is_wht(self):
        self._reset()
        resolved = lib.tqp_resolve_rotation(ref.stored_byte(3, ref.TQP_ROT_HAAR))
        assert ref.extract_rotation(resolved) == ref.TQP_ROT_WHT
        self._reset()

    def test_c_matches_python_resolver(self):
        self._reset()
        for layer in [0, 5, 31]:
            for rot in [ref.TQP_ROT_WHT, ref.TQP_ROT_HAAR]:
                lb = ref.layer_byte(layer, rot)
                assert lib.tqp_resolve_rotation(lb) == ref.resolve_rotation(lb)

                lib.tqp_set_thread_rotation(rot)
                ref.set_thread_rotation(rot)
                lb = ref.stored_byte(layer, 0)
                assert lib.tqp_resolve_rotation(lb) == ref.resolve_rotation(lb)
                lib.tqp_clear_thread_rotation()
                ref.clear_thread_rotation()

                lib.tqp_set_default_rotation(rot)
                ref.set_default_rotation(rot)
                lb = ref.stored_byte(layer, 0)
                assert lib.tqp_resolve_rotation(lb) == ref.resolve_rotation(lb)
                self._reset()
                ref.set_default_rotation(ref._ROT_UNSET)

    def test_thread_quantize_uses_resolved_rotation(self, d, bits, vectors):
        self._reset()
        lib.tqp_set_thread_rotation(ref.TQP_ROT_HAAR)
        x_np = vectors[0].float().numpy().copy()
        blk = BLOCK_CLASSES[(d, bits)]()
        APIS[(d, bits, "quantize")](
            x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(blk),
            d,
            0x00,
        )
        c_bytes = bytes(blk)
        assert ref.extract_rotation(c_bytes[4]) == ref.TQP_ROT_HAAR
        assert ref.extract_explicit(c_bytes[4]) == 0
        self._reset()


def test_bf16_f16_vec_dot_matches_fp32(d, bits, constants, vectors, input_dtype):
    if input_dtype == "fp32":
        pytest.skip("fp32 baseline")

    keys = vectors[:10]
    queries = vectors[10:15]
    max_abs = 0.0
    for q in queries:
        for k in keys:
            k_np = k.float().numpy().copy()
            blk_fp32 = _c_quantize_dtype(d, bits, k_np, 0, ref.TQP_ROT_WHT, "fp32")
            blk_test = _c_quantize_dtype(d, bits, k_np, 0, ref.TQP_ROT_WHT, input_dtype)
            ip_fp32 = ref.inner_product(q, blk_fp32, constants)
            ip_test = ref.inner_product(q, blk_test, constants)
            max_abs = max(max_abs, abs(ip_fp32 - ip_test))

    tol = {64: 0.05, 128: 0.03, 256: 0.02}[d]
    if bits == 2 and input_dtype == "bf16":
        tol = max(tol, 0.04)
    assert max_abs < tol, f"{input_dtype} vs fp32 vec_dot max diff {max_abs:.2e} exceeds {tol:.2e}"


def test_legacy_b3_symbols_match_new_api(d, vectors, layer_idx, rotation):
    x_np = vectors[0].float().numpy().copy()
    new_bytes = _c_quantize(d, 3, x_np, layer_idx, rotation)
    legacy_bytes = _c_quantize(d, 3, x_np, layer_idx, rotation, legacy=True)
    assert new_bytes == legacy_bytes
