"""
Comprehensive CUDA GPU tests for TurboQuant.

Tests the Python quantizer on CUDA tensors (both GPUs), validates numerical
accuracy of the full pipeline, exposes multi-GPU issues, and stress-tests
edge cases.

Requirements:
    - PyTorch with CUDA support
    - At least one NVIDIA GPU
    - For multi-GPU tests: 2+ GPUs (e.g. RTX 4090 + RTX 5090)

Run:
    pytest tests/test_cuda_gpu.py -v
"""

from __future__ import annotations

import math
import ctypes
import pathlib
import struct
import sys

import pytest
import torch

from lloyd_max import LloydMaxCodebook
from turboquant import (
    TurboQuantKVCache,
    TurboQuantMSE,
    TurboQuantProd,
    generate_rotation_matrix,
    generate_qjl_matrix,
)
from compressors import TurboQuantCompressorV2, TurboQuantCompressorMSE

# ---------------------------------------------------------------------------
# Skip entire module if no CUDA device
# ---------------------------------------------------------------------------

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

DEVICE_COUNT = torch.cuda.device_count()
MULTI_GPU = DEVICE_COUNT >= 2

# Discover GPU names for test IDs
GPU_NAMES = {i: torch.cuda.get_device_name(i) for i in range(DEVICE_COUNT)}


def _devices():
    """Return list of CUDA device strings."""
    return [f"cuda:{i}" for i in range(DEVICE_COUNT)]


def _make_unit_vectors(
    batch: int, dim: int, seed: int, device: str = "cpu"
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    vecs = torch.randn(batch, dim, generator=gen)
    vecs = vecs / vecs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return vecs.to(device)


# =====================================================================
# 1. Python quantizer on each GPU — basic round-trip
# =====================================================================


class TestMSEQuantizerOnGPU:
    """TurboQuantMSE round-trip on each available GPU."""

    @pytest.fixture(params=_devices(), ids=lambda d: GPU_NAMES.get(int(d.split(":")[1]), d))
    def device(self, request):
        return request.param

    @pytest.mark.parametrize("dim", [64, 128, 256])
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_mse_roundtrip_on_gpu(self, device, dim, bits):
        quantizer = TurboQuantMSE(dim, bits, seed=42, device=device)
        vecs = _make_unit_vectors(128, dim, seed=7, device=device)

        reconstructed, indices = quantizer(vecs)

        assert reconstructed.device.type == "cuda"
        assert indices.device.type == "cuda"
        assert reconstructed.shape == vecs.shape
        assert torch.isfinite(reconstructed).all()

        # MSE should be bounded
        mse = ((vecs - reconstructed) ** 2).sum(dim=-1).mean().item()
        theoretical = math.sqrt(3) * math.pi / 2 * (1 / (4**bits))
        assert mse <= theoretical * 2.0, f"MSE {mse:.4f} exceeds 2x theoretical {theoretical:.4f}"

    @pytest.mark.parametrize("dim", [128, 256])
    def test_gpu_matches_cpu(self, device, dim):
        """GPU quantize output must be identical to CPU."""
        seed, bits = 42, 3
        cpu_q = TurboQuantMSE(dim, bits, seed=seed, device="cpu")
        gpu_q = TurboQuantMSE(dim, bits, seed=seed, device=device)
        vecs_cpu = _make_unit_vectors(64, dim, seed=99)
        vecs_gpu = vecs_cpu.to(device)

        _, cpu_idx = cpu_q(vecs_cpu)
        _, gpu_idx = gpu_q(vecs_gpu)

        assert torch.equal(cpu_idx, gpu_idx.cpu()), "GPU indices differ from CPU"


# =====================================================================
# 2. TurboQuantProd on each GPU — inner product accuracy
# =====================================================================


class TestProdQuantizerOnGPU:
    """TurboQuantProd quantize + inner product on each GPU."""

    @pytest.fixture(params=_devices(), ids=lambda d: GPU_NAMES.get(int(d.split(":")[1]), d))
    def device(self, request):
        return request.param

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_prod_inner_product_bias_on_gpu(self, device, bits):
        dim = 128
        quantizer = TurboQuantProd(dim, bits, seed=42, device=device)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(200 + bits)

        x = _make_unit_vectors(1000, dim, seed=200 + bits, device=device)
        y = _make_unit_vectors(1000, dim, seed=300 + bits, device=device)

        compressed = quantizer.quantize(x)
        # inner_product returns (N, M); take diagonal for pairwise y_i, x_i.
        estimated = quantizer.inner_product(y, compressed).diagonal()
        true_ip = (x * y).sum(dim=-1)

        bias = (estimated - true_ip).mean().item()
        assert abs(bias) < 0.02, f"Bias {bias:.4f} on {device}"

        # Correlation should be strong
        corr = torch.corrcoef(torch.stack([true_ip, estimated]))[0, 1].item()
        assert corr > 0.80, f"Correlation {corr:.3f} too low on {device}"

    def test_prod_quantize_shapes_on_gpu(self, device):
        dim, bits = 128, 3
        quantizer = TurboQuantProd(dim, bits, seed=42, device=device)
        x = _make_unit_vectors(16, dim, seed=42, device=device)

        compressed = quantizer.quantize(x)
        assert compressed["mse_indices"].shape == (16, dim)
        assert compressed["qjl_signs"].shape == (16, dim)
        assert compressed["residual_norm"].shape == (16,)
        assert compressed["mse_indices"].device.type == "cuda"

    def test_prod_gpu_matches_cpu(self, device):
        dim, bits, seed = 128, 3, 42
        cpu_q = TurboQuantProd(dim, bits, seed=seed, device="cpu")
        gpu_q = TurboQuantProd(dim, bits, seed=seed, device=device)
        vecs = _make_unit_vectors(32, dim, seed=77)

        cpu_c = cpu_q.quantize(vecs)
        gpu_c = gpu_q.quantize(vecs.to(device))

        assert torch.equal(cpu_c["mse_indices"], gpu_c["mse_indices"].cpu())
        assert torch.allclose(
            cpu_c["qjl_signs"], gpu_c["qjl_signs"].cpu(), atol=0, rtol=0
        )


# =====================================================================
# 3. KV Cache on GPU
# =====================================================================


class TestKVCacheOnGPU:
    """TurboQuantKVCache on CUDA devices."""

    @pytest.fixture(params=_devices(), ids=lambda d: GPU_NAMES.get(int(d.split(":")[1]), d))
    def device(self, request):
        return request.param

    def test_kv_cache_append_and_score(self, device):
        dim = 64
        cache = TurboQuantKVCache(
            d_key=dim, d_value=dim, bits=3, seed=42, device=device
        )
        gen = torch.Generator(device="cpu")
        gen.manual_seed(42)

        keys = torch.randn(8, dim, generator=gen).to(device)
        values = torch.randn(8, dim, generator=gen).to(device)
        cache.append(keys, values)

        queries = torch.randn(1, dim, generator=gen).to(device)
        scores = cache.attention_scores(queries)
        # 2D query -> 2D (N, M) score matrix.
        assert scores.shape == (1, 8)
        assert torch.isfinite(scores).all()

        vals = cache.get_values()
        assert vals.shape == (8, dim)
        assert torch.isfinite(vals).all()

    def test_kv_cache_multiple_appends(self, device):
        dim = 64
        cache = TurboQuantKVCache(
            d_key=dim, d_value=dim, bits=3, seed=42, device=device
        )
        gen = torch.Generator(device="cpu")
        gen.manual_seed(99)

        for _ in range(5):
            k = torch.randn(4, dim, generator=gen).to(device)
            v = torch.randn(4, dim, generator=gen).to(device)
            cache.append(k, v)

        assert len(cache) == 20
        scores = cache.attention_scores(torch.randn(1, dim).to(device))
        # 2D query -> 2D (N, M) score matrix.
        assert scores.shape == (1, 20)

    def test_kv_cache_attention_quality(self, device):
        """Quantized attention scores should correlate with fp32 attention."""
        dim = 128
        n_keys = 64
        cache = TurboQuantKVCache(
            d_key=dim, d_value=dim, bits=3, seed=42, device=device
        )

        keys = _make_unit_vectors(n_keys, dim, seed=50, device=device) * 2.0
        values = torch.randn(n_keys, dim).to(device)
        cache.append(keys, values)

        query = torch.randn(1, dim).to(device)

        # 2D query -> (1, n_keys); squeeze for comparison with fp32 reference.
        tq_scores = cache.attention_scores(query).squeeze(0)
        fp32_scores = (query @ keys.T).squeeze(0)

        corr = torch.corrcoef(torch.stack([fp32_scores, tq_scores]))[0, 1].item()
        assert corr > 0.85, f"Attention score correlation {corr:.3f} too low"


# =====================================================================
# 4. Compressor V2 on GPU
# =====================================================================


class TestCompressorV2OnGPU:
    @pytest.fixture(params=_devices(), ids=lambda d: GPU_NAMES.get(int(d.split(":")[1]), d))
    def device(self, request):
        return request.param

    def test_compress_decompress_attention(self, device):
        batch, heads, seq_len, dim = 1, 4, 32, 128
        comp = TurboQuantCompressorV2(dim, bits=3, seed=42, device=device)

        keys = torch.randn(batch, heads, seq_len, dim).to(device)
        queries = torch.randn(batch, heads, 1, dim).to(device)

        compressed = comp.compress(keys)
        tq_scores = comp.asymmetric_attention_scores(queries, compressed)
        fp32_scores = torch.matmul(queries.float(), keys.float().transpose(-2, -1))

        assert tq_scores.shape == fp32_scores.shape

        # Per-head correlation check
        for h in range(heads):
            corr = torch.corrcoef(
                torch.stack([
                    fp32_scores[0, h, 0].cpu(),
                    tq_scores[0, h, 0].cpu(),
                ])
            )[0, 1].item()
            assert corr > 0.80, f"Head {h} correlation {corr:.3f} on {device}"

    def test_default_storage_is_fp32(self, device):
        """Bug #5 fix: default storage_dtype is now fp32, eliminating the
        fp16 truncation noise that previously degraded k_mse precision."""
        batch, heads, seq_len, dim = 1, 2, 16, 128
        comp = TurboQuantCompressorV2(dim, bits=3, seed=42, device=device)

        keys = torch.randn(batch, heads, seq_len, dim).to(device)
        compressed = comp.compress(keys)

        assert compressed["k_mse"].dtype == torch.float32, "default should be fp32"
        assert compressed["residual_norm"].dtype == torch.float32

    def test_fp16_storage_opt_in(self, device):
        """Opting into fp16 storage still works and introduces bounded error."""
        batch, heads, seq_len, dim = 1, 2, 16, 128
        comp = TurboQuantCompressorV2(
            dim, bits=3, seed=42, device=device, storage_dtype=torch.float16,
        )

        keys = torch.randn(batch, heads, seq_len, dim).to(device)
        compressed = comp.compress(keys)

        k_mse_fp16 = compressed["k_mse"]
        assert k_mse_fp16.dtype == torch.float16

        # Recompute in fp32 to measure the fp16 round-trip error
        flat = keys.reshape(-1, dim).float()
        norms = flat.norm(dim=-1).clamp_min(1e-8)
        flat_unit = flat / norms.unsqueeze(-1)
        c = comp.quantizer.quantize(flat_unit)
        k_mse_fp32 = comp.quantizer.dequantize(c) * norms.unsqueeze(-1)
        k_mse_fp32 = k_mse_fp32.reshape(batch, heads, seq_len, dim)

        fp16_error = (k_mse_fp16.float() - k_mse_fp32).abs().max().item()
        assert fp16_error < 0.1, f"fp16 error {fp16_error} unexpectedly large"


# =====================================================================
# 5. Multi-GPU tests (only run if 2+ GPUs available)
# =====================================================================


@pytest.mark.skipif(not MULTI_GPU, reason="Need 2+ GPUs for multi-GPU tests")
class TestMultiGPU:
    """Tests that exercise multiple GPUs simultaneously."""

    def test_python_quantizer_cross_device_consistency(self):
        """Same quantizer parameters on different GPUs must produce identical indices."""
        dim, bits, seed = 128, 3, 42
        vecs = _make_unit_vectors(64, dim, seed=55)

        results = {}
        for i in range(DEVICE_COUNT):
            dev = f"cuda:{i}"
            q = TurboQuantMSE(dim, bits, seed=seed, device=dev)
            _, idx = q(vecs.to(dev))
            results[dev] = idx.cpu()

        devices = list(results.keys())
        for i in range(1, len(devices)):
            assert torch.equal(results[devices[0]], results[devices[i]]), (
                f"Indices differ between {devices[0]} ({GPU_NAMES[0]}) "
                f"and {devices[i]} ({GPU_NAMES[i]})"
            )

    def test_python_prod_cross_device_consistency(self):
        """TurboQuantProd inner products must agree across GPUs."""
        dim, bits, seed = 128, 3, 42
        x = _make_unit_vectors(100, dim, seed=10)
        y = _make_unit_vectors(100, dim, seed=20)

        ips = {}
        for i in range(DEVICE_COUNT):
            dev = f"cuda:{i}"
            q = TurboQuantProd(dim, bits, seed=seed, device=dev)
            c = q.quantize(x.to(dev))
            ip = q.inner_product(y.to(dev), c)
            ips[dev] = ip.cpu()

        devices = list(ips.keys())
        for i in range(1, len(devices)):
            assert torch.allclose(ips[devices[0]], ips[devices[i]], atol=1e-5), (
                f"Inner products differ between {devices[0]} and {devices[i]}: "
                f"max diff = {(ips[devices[0]] - ips[devices[i]]).abs().max():.2e}"
            )

    def test_kv_cache_on_each_gpu_separately(self):
        """KV cache should work independently on each GPU."""
        dim = 64
        for i in range(DEVICE_COUNT):
            dev = f"cuda:{i}"
            cache = TurboQuantKVCache(
                d_key=dim, d_value=dim, bits=3, seed=42, device=dev
            )
            keys = torch.randn(8, dim).to(dev)
            values = torch.randn(8, dim).to(dev)
            cache.append(keys, values)

            query = torch.randn(1, dim).to(dev)
            scores = cache.attention_scores(query)
            assert scores.device == torch.device(dev)
            assert torch.isfinite(scores).all(), f"Non-finite scores on {dev}"


# =====================================================================
# 6. Multi-GPU CUDA library tests (C/CUDA via ctypes)
# =====================================================================

_CUDA_LIB = pathlib.Path(__file__).resolve().parents[1] / "patches" / "stage2-qjl" / "cuda" / "build" / "libggml_tq_paper_cuda.so"
_CPU_LIB = pathlib.Path(__file__).resolve().parents[1] / "patches" / "stage2-qjl" / "c" / "libggml_tq_paper.so"

HAS_CUDA_LIB = _CUDA_LIB.exists()
HAS_CPU_LIB = _CPU_LIB.exists()


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


@pytest.mark.skipif(not HAS_CUDA_LIB, reason="CUDA library not built")
@pytest.mark.skipif(not HAS_CPU_LIB, reason="CPU library not built")
@pytest.mark.skipif(not MULTI_GPU, reason="Need 2+ GPUs for multi-GPU CUDA tests")
class TestMultiGPUCudaLibrary:
    """Test the CUDA library's behavior when switching between GPUs."""

    @pytest.fixture(autouse=True)
    def _load_cuda_lib(self):
        self.cpu = ctypes.CDLL(str(_CPU_LIB))
        self.cpu.ggml_quantize_row_tq4p_d128.restype = None
        self.cpu.ggml_quantize_row_tq4p_d128.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(block_tq4p_d128),
            ctypes.c_int64,
            ctypes.c_uint8,
        ]
        self.cuda = ctypes.CDLL(str(_CUDA_LIB))
        self.cuda.tqp_cuda_device_count.restype = ctypes.c_int
        self.cuda.tqp_cuda_quantize_row_d128.restype = ctypes.c_int
        self.cuda.tqp_cuda_quantize_row_d128.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_uint8,
        ]
        self.cuda.tqp_cuda_vec_dot_block_d128.restype = ctypes.c_float
        self.cuda.tqp_cuda_vec_dot_block_d128.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(block_tq4p_d128),
        ]

    def _as_float_ptr(self, x):
        return x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def test_cuda_init_is_per_device(self):
        dev_count = self.cuda.tqp_cuda_device_count()
        assert dev_count >= 2, f"Expected 2+ devices, got {dev_count}"

        x = torch.randn(128).numpy().copy()
        layer_byte = (1 << 6) | (1 << 7) | 0  # explicit, layer 0, Haar

        cpu_blk = block_tq4p_d128()
        self.cpu.ggml_quantize_row_tq4p_d128(
            self._as_float_ptr(x), ctypes.byref(cpu_blk), 128, layer_byte
        )
        expected = ctypes.string_at(ctypes.byref(cpu_blk), ctypes.sizeof(cpu_blk))

        torch.cuda.set_device(0)
        blk0 = block_tq4p_d128()
        err = self.cuda.tqp_cuda_quantize_row_d128(
            self._as_float_ptr(x), ctypes.byref(blk0), 128, layer_byte
        )
        assert err == 0, f"quantize on device 0 failed with error {err}"
        assert ctypes.string_at(ctypes.byref(blk0), ctypes.sizeof(blk0)) == expected

        torch.cuda.set_device(1)
        blk1 = block_tq4p_d128()
        err = self.cuda.tqp_cuda_quantize_row_d128(
            self._as_float_ptr(x), ctypes.byref(blk1), 128, layer_byte
        )
        assert err == 0, f"quantize on device 1 failed with error {err}"
        assert ctypes.string_at(ctypes.byref(blk1), ctypes.sizeof(blk1)) == expected

        # Restore device 0
        torch.cuda.set_device(0)


# =====================================================================
# 7. Edge cases & stress tests on GPU
# =====================================================================


class TestGPUEdgeCases:
    @pytest.fixture
    def device(self):
        return "cuda:0"

    def test_zero_vectors_on_gpu(self, device):
        quantizer = TurboQuantProd(d=128, bits=3, seed=42, device=device)
        zeros = torch.zeros(4, 128, device=device)
        compressed = quantizer.quantize(zeros)

        assert torch.isfinite(compressed["residual_norm"]).all()
        assert torch.isfinite(compressed["qjl_signs"]).all()

        recon = quantizer.dequantize(compressed)
        assert torch.isfinite(recon).all()

    def test_tiny_norm_vectors_on_gpu(self, device):
        quantizer = TurboQuantProd(d=128, bits=3, seed=42, device=device)
        tiny = torch.randn(4, 128, device=device) * 1e-15
        compressed = quantizer.quantize(tiny)
        ip = quantizer.inner_product(tiny, compressed)
        assert torch.isfinite(ip).all()

    def test_large_norm_vectors_on_gpu(self, device):
        quantizer = TurboQuantProd(d=128, bits=3, seed=42, device=device)
        big = torch.randn(4, 128, device=device) * 1e6
        big = big / big.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        compressed = quantizer.quantize(big)
        ip = quantizer.inner_product(big, compressed)
        assert torch.isfinite(ip).all()

    def test_single_vector_on_gpu(self, device):
        quantizer = TurboQuantProd(d=128, bits=3, seed=42, device=device)
        x = _make_unit_vectors(1, 128, seed=42, device=device)
        compressed = quantizer.quantize(x)
        ip = quantizer.inner_product(x, compressed)
        # 2D x (1, d) dot 2D x (1, d) -> (1, 1).
        assert ip.shape == (1, 1)
        # Self inner product of unit vector should be close to 1.0
        assert abs(ip.item() - 1.0) < 0.3, f"Self-IP {ip.item():.4f} too far from 1.0"

    def test_large_batch_stress(self, device):
        """Stress test with large batch to verify GPU memory handling."""
        dim, bits = 128, 3
        quantizer = TurboQuantMSE(dim, bits, seed=42, device=device)
        # 10K vectors: should fit comfortably on any modern GPU
        vecs = _make_unit_vectors(10000, dim, seed=777, device=device)
        reconstructed, indices = quantizer(vecs)

        assert reconstructed.shape == vecs.shape
        assert torch.isfinite(reconstructed).all()

        mse = ((vecs - reconstructed) ** 2).sum(dim=-1).mean().item()
        theoretical = math.sqrt(3) * math.pi / 2 * (1 / (4**bits))
        assert mse <= theoretical * 1.5

    def test_deterministic_across_runs(self, device):
        """Same seed must produce identical results."""
        dim, bits, seed = 128, 3, 42
        vecs = _make_unit_vectors(32, dim, seed=99, device=device)

        results = []
        for _ in range(3):
            q = TurboQuantProd(dim, bits, seed=seed, device=device)
            c = q.quantize(vecs)
            results.append(c["mse_indices"].cpu())

        for i in range(1, len(results)):
            assert torch.equal(results[0], results[i])


# =====================================================================
# 8. Bug #4: inner_product batching limitation
# =====================================================================


class TestInnerProductBatchingBug:
    """Expose Bug #4: TurboQuantProd.inner_product uses element-wise
    multiplication, so it only works when n_queries == n_keys or one is 1.
    It CANNOT compute a full (n_queries, n_keys) attention matrix."""

    def test_equal_sizes_returns_full_matrix(self):
        """Bug #4 fix: N == M no longer shortcuts to (N,) diagonal."""
        dim, bits = 128, 3
        q = TurboQuantProd(dim, bits, seed=42, device="cpu")
        x = _make_unit_vectors(10, dim, seed=1)
        y = _make_unit_vectors(10, dim, seed=2)

        compressed = q.quantize(x)
        ip = q.inner_product(y, compressed)
        assert ip.shape == (10, 10), f"Expected (10, 10), got {ip.shape}"

    def test_broadcast_1_query_many_keys(self):
        """Bug #4 fix: 2D (1, d) query returns (1, M), not (M,)."""
        dim, bits = 128, 3
        q = TurboQuantProd(dim, bits, seed=42, device="cpu")
        x = _make_unit_vectors(10, dim, seed=1)
        y = _make_unit_vectors(1, dim, seed=2)

        compressed = q.quantize(x)
        ip = q.inner_product(y, compressed)
        assert ip.shape == (1, 10), f"Expected (1, 10), got {ip.shape}"

    def test_many_queries_many_keys(self):
        """Bug #4 fix: n_queries != n_keys now returns (n_queries, n_keys)."""
        dim, bits = 128, 3
        q = TurboQuantProd(dim, bits, seed=42, device="cpu")
        x = _make_unit_vectors(10, dim, seed=1)
        y = _make_unit_vectors(5, dim, seed=2)

        compressed = q.quantize(x)
        ip = q.inner_product(y, compressed)
        assert ip.shape == (5, 10), f"Expected (5, 10), got {ip.shape}"


# =====================================================================
# 9. GPU-specific numerical validation
# =====================================================================


class TestGPUNumericalValidation:
    """Validate numerical properties that matter for real inference."""

    @pytest.fixture
    def device(self):
        return "cuda:0"

    @pytest.mark.parametrize("dim", [128, 256])
    def test_rotation_preserves_norm_on_gpu(self, device, dim):
        q = TurboQuantMSE(dim, bits=3, seed=42, device=device)
        vecs = _make_unit_vectors(100, dim, seed=42, device=device)

        rotated = q.rotate(vecs)
        norms_before = vecs.norm(dim=-1)
        norms_after = rotated.norm(dim=-1)

        assert torch.allclose(norms_before, norms_after, atol=1e-4), (
            f"Rotation changed norms: max diff = "
            f"{(norms_before - norms_after).abs().max().item():.2e}"
        )

    @pytest.mark.parametrize("dim", [128, 256])
    def test_rotation_inverse_on_gpu(self, device, dim):
        q = TurboQuantMSE(dim, bits=3, seed=42, device=device)
        vecs = _make_unit_vectors(50, dim, seed=42, device=device)

        recovered = q.unrotate(q.rotate(vecs))
        assert torch.allclose(vecs, recovered, atol=1e-4), (
            f"Rotation inverse failed: max diff = "
            f"{(vecs - recovered).abs().max().item():.2e}"
        )

    def test_qjl_estimator_is_unbiased_on_gpu(self, device):
        """QJL inner product estimator should be unbiased (mean error ~ 0)."""
        dim, bits = 128, 3
        n_samples = 5000
        q = TurboQuantProd(dim, bits, seed=42, device=device)

        x = _make_unit_vectors(n_samples, dim, seed=10, device=device)
        y = _make_unit_vectors(n_samples, dim, seed=20, device=device)

        compressed = q.quantize(x)
        # inner_product returns (N, N); take diagonal for pairwise y_i, x_i.
        estimated = q.inner_product(y, compressed).diagonal()
        true_ip = (x * y).sum(dim=-1)

        bias = (estimated - true_ip).mean().item()
        assert abs(bias) < 0.005, f"QJL estimator bias {bias:.4f} on GPU"

    def test_codebook_symmetry_preserved_on_gpu(self, device):
        """Lloyd-Max codebook should be symmetric on GPU."""
        for dim in [64, 128, 256]:
            cb = LloydMaxCodebook(dim, bits=3)
            centroids_gpu = cb.centroids.to(device)
            # Centroids should sum to ~0 (symmetric distribution)
            assert centroids_gpu.sum().abs().item() < 0.01

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_compression_ratio_on_gpu(self, device, bits):
        """Verify reported compression ratio is reasonable."""
        dim = 128
        cache = TurboQuantKVCache(
            d_key=dim, d_value=dim, bits=bits, seed=42, device=device
        )
        keys = torch.randn(64, dim).to(device)
        values = torch.randn(64, dim).to(device)
        cache.append(keys, values)

        usage = cache.memory_usage_bits()
        ratio = usage["compression_ratio"]
        # At 3 bits, we expect ~4-5x compression vs fp16
        assert ratio > 1.0, f"Compression ratio {ratio:.2f} is not compressing"
        assert ratio < 20.0, f"Compression ratio {ratio:.2f} is implausibly high"


# =====================================================================
# 10. Compressor MSE-only on GPU
# =====================================================================


class TestCompressorMSEOnGPU:
    @pytest.fixture(params=_devices(), ids=lambda d: GPU_NAMES.get(int(d.split(":")[1]), d))
    def device(self, request):
        return request.param

    def test_compress_decompress_roundtrip(self, device):
        batch, heads, seq_len, dim = 1, 4, 16, 128
        comp = TurboQuantCompressorMSE(dim, bits=3, seed=42, device=device)

        values = torch.randn(batch, heads, seq_len, dim).to(device)
        compressed = comp.compress(values)
        reconstructed = comp.decompress(compressed)

        assert reconstructed.shape == values.shape

        # Normalized MSE should be bounded
        mse = ((values - reconstructed) ** 2).sum(dim=-1).mean().item()
        # Very loose bound since values are unnormalized
        assert mse < 10.0, f"MSE {mse:.4f} seems too high"
        assert torch.isfinite(reconstructed).all()
