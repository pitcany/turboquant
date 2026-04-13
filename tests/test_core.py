import math

import pytest
import torch

from compressors import TurboQuantCompressorMSE, TurboQuantCompressorV2
from lloyd_max import LloydMaxCodebook
from turboquant import TurboQuantKVCache, TurboQuantMSE, TurboQuantProd


@pytest.mark.parametrize(("dim", "bits"), [(64, 1), (128, 2), (256, 3)])
def test_lloyd_max_codebook_is_symmetric(dim: int, bits: int) -> None:
    codebook = LloydMaxCodebook(dim, bits)

    assert codebook.n_levels == 2 ** bits
    assert torch.all(torch.diff(codebook.centroids) > 0)
    assert torch.all(torch.diff(codebook.boundaries) > 0)
    assert codebook.centroids.sum().abs().item() < 0.01


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_turboquant_mse_respects_theoretical_bound(bits: int) -> None:
    dim = 128
    quantizer = TurboQuantMSE(dim, bits, seed=42, device="cpu")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7 + bits)
    vectors = torch.randn(512, dim, generator=generator)
    vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True).clamp_min(1e-8)

    reconstructed, _ = quantizer(vectors)
    mse = ((vectors - reconstructed) ** 2).sum(dim=-1).mean().item()
    theoretical_bound = math.sqrt(3) * math.pi / 2 * (1 / (4 ** bits))

    assert mse <= theoretical_bound * 1.5


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_turboquant_prod_bias_stays_small(bits: int) -> None:
    dim = 128
    quantizer = TurboQuantProd(dim, bits, seed=42, device="cpu")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(100 + bits)
    x = torch.randn(1500, dim, generator=generator)
    y = torch.randn(1500, dim, generator=generator)
    x = x / torch.norm(x, dim=-1, keepdim=True).clamp_min(1e-8)
    y = y / torch.norm(y, dim=-1, keepdim=True).clamp_min(1e-8)

    true_ip = (x * y).sum(dim=-1)
    # inner_product returns the full (N, M) score matrix; take the diagonal
    # to get pairwise y_i, x_i scores for this test.
    estimated_ip = quantizer.inner_product(y, quantizer.quantize(x)).diagonal()

    bias = (estimated_ip - true_ip).mean().item()
    assert abs(bias) < 0.01


def test_mse_only_inner_products_are_more_biased_than_prod() -> None:
    dim = 128
    bits = 3
    generator = torch.Generator(device="cpu")
    generator.manual_seed(314)
    x = torch.randn(1500, dim, generator=generator)
    x = x / torch.norm(x, dim=-1, keepdim=True).clamp_min(1e-8)
    y = x.clone()

    prod_quantizer = TurboQuantProd(dim, bits, seed=42, device="cpu")
    mse_quantizer = TurboQuantMSE(dim, bits, seed=42, device="cpu")

    true_ip = (x * y).sum(dim=-1)
    # Full (N, M) score matrix; take the diagonal for pairwise y_i, x_i.
    prod_ip = prod_quantizer.inner_product(y, prod_quantizer.quantize(x)).diagonal()
    mse_ip = (mse_quantizer.dequantize(mse_quantizer.quantize(x)) * y).sum(dim=-1)

    prod_bias = (prod_ip - true_ip).mean().abs().item()
    mse_bias = (mse_ip - true_ip).mean().abs().item()

    assert prod_bias < mse_bias


def test_inner_product_matrix_shape() -> None:
    dim, bits = 128, 3
    generator = torch.Generator(device="cpu")
    generator.manual_seed(401)
    quantizer = TurboQuantProd(dim, bits, seed=42, device="cpu")
    keys = torch.randn(10, dim, generator=generator)
    queries = torch.randn(5, dim, generator=generator)
    keys = keys / torch.norm(keys, dim=-1, keepdim=True).clamp_min(1e-8)
    queries = queries / torch.norm(queries, dim=-1, keepdim=True).clamp_min(1e-8)

    estimated = quantizer.inner_product(queries, quantizer.quantize(keys))

    assert estimated.shape == (5, 10)


def test_inner_product_matrix_matches_loop() -> None:
    dim, bits = 128, 3
    generator = torch.Generator(device="cpu")
    generator.manual_seed(402)
    quantizer = TurboQuantProd(dim, bits, seed=42, device="cpu")
    keys = torch.randn(10, dim, generator=generator)
    queries = torch.randn(5, dim, generator=generator)
    keys = keys / torch.norm(keys, dim=-1, keepdim=True).clamp_min(1e-8)
    queries = queries / torch.norm(queries, dim=-1, keepdim=True).clamp_min(1e-8)
    compressed = quantizer.quantize(keys)

    estimated = quantizer.inner_product(queries, compressed)
    expected = torch.stack([
        quantizer.inner_product(query, compressed)
        for query in queries
    ])

    assert torch.allclose(estimated, expected)


def test_inner_product_matrix_unbiased() -> None:
    dim, bits = 128, 3
    generator = torch.Generator(device="cpu")
    generator.manual_seed(403)
    quantizer = TurboQuantProd(dim, bits, seed=42, device="cpu")
    keys = torch.randn(40, dim, generator=generator)
    queries = torch.randn(25, dim, generator=generator)
    keys = keys / torch.norm(keys, dim=-1, keepdim=True).clamp_min(1e-8)
    queries = queries / torch.norm(queries, dim=-1, keepdim=True).clamp_min(1e-8)

    estimated = quantizer.inner_product(queries, quantizer.quantize(keys))
    true_ip = torch.matmul(queries, keys.T)
    bias = (estimated - true_ip).mean().item()

    assert abs(bias) < 0.01


def test_bucketize_matches_nearest_centroid_assignment() -> None:
    codebook = LloydMaxCodebook(d=128, bits=3)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(29)
    values = torch.randn(4096, generator=generator) / math.sqrt(128)

    diffs = (values.unsqueeze(-1) - codebook.centroids).abs()
    nearest = diffs.argmin(dim=-1)

    assert torch.equal(codebook.quantize(values), nearest)


def test_kv_cache_normalizes_before_quantization_and_restores_scale() -> None:
    dim = 32
    cache = TurboQuantKVCache(d_key=dim, d_value=dim, bits=4, seed=23, device="cpu")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(211)

    keys = torch.randn(6, dim, generator=generator) * torch.linspace(0.5, 3.0, 6).unsqueeze(-1)
    values = torch.randn(6, dim, generator=generator) * torch.linspace(1.5, 4.0, 6).unsqueeze(-1)
    queries = torch.randn(1, dim, generator=generator)

    cache.append(keys, values)

    key_norms = torch.norm(keys, dim=-1)
    value_norms = torch.norm(values, dim=-1)
    key_units = keys / key_norms.unsqueeze(-1).clamp_min(1e-8)
    value_units = values / value_norms.unsqueeze(-1).clamp_min(1e-8)

    expected_keys = cache.key_quantizer.quantize(key_units)
    expected_value_indices = cache.value_quantizer.quantize(value_units)

    cached_keys = cache.key_cache[0]
    cached_values = cache.value_cache[0]

    assert torch.equal(cached_keys["mse_indices"], expected_keys["mse_indices"])
    assert torch.equal(cached_keys["qjl_signs"], expected_keys["qjl_signs"])
    assert torch.allclose(cached_keys["residual_norm"], expected_keys["residual_norm"])
    assert torch.allclose(cached_keys["key_norms"], key_norms)
    assert torch.equal(cached_values["indices"], expected_value_indices)
    assert torch.allclose(cached_values["value_norms"], value_norms)

    expected_scores = cache.key_quantizer.inner_product(queries, expected_keys) * key_norms
    assert torch.allclose(cache.attention_scores(queries), expected_scores)

    expected_values = cache.value_quantizer.dequantize(expected_value_indices) * value_norms.unsqueeze(-1)
    assert torch.allclose(cache.get_values(), expected_values)

    usage = cache.memory_usage_bits()
    num_vectors = keys.shape[0]
    key_bits = (
        keys.numel() * cache.key_quantizer.mse_bits
        + expected_keys["qjl_signs"].numel()
        + num_vectors * 16
        + num_vectors * 16
    )
    value_bits = values.numel() * cache.bits + num_vectors * 16

    assert usage["key_bits"] == key_bits
    assert usage["value_bits"] == value_bits
    assert usage["total_bits"] == key_bits + value_bits


def test_compressor_v2_defaults_to_fp32_storage() -> None:
    batch, heads, seq_len, dim = 1, 2, 8, 128
    generator = torch.Generator(device="cpu")
    generator.manual_seed(501)
    compressor = TurboQuantCompressorV2(dim, bits=3, seed=42, device="cpu")
    keys = torch.randn(batch, heads, seq_len, dim, generator=generator)

    compressed = compressor.compress(keys)

    assert compressed["k_mse"].dtype == torch.float32
    assert compressed["residual_norm"].dtype == torch.float32


def test_compressor_v2_default_storage_matches_fp32_recompute() -> None:
    batch, heads, seq_len, dim = 1, 2, 8, 128
    generator = torch.Generator(device="cpu")
    generator.manual_seed(504)
    compressor = TurboQuantCompressorV2(dim, bits=3, seed=42, device="cpu")
    keys = torch.randn(batch, heads, seq_len, dim, generator=generator)

    compressed = compressor.compress(keys)
    flat = keys.reshape(-1, dim).float()
    vec_norms = torch.norm(flat, dim=-1).clamp_min(1e-8)
    flat_unit = flat / vec_norms.unsqueeze(-1)
    raw_compressed = compressor.quantizer.quantize(flat_unit)
    expected_k_mse = (
        compressor.quantizer.dequantize(raw_compressed)
        * vec_norms.unsqueeze(-1)
    ).reshape(batch, heads, seq_len, dim)
    expected_residual_norm = (
        raw_compressed["residual_norm"] * vec_norms
    ).reshape(batch, heads, seq_len)

    assert torch.allclose(compressed["k_mse"], expected_k_mse, atol=0, rtol=0)
    assert torch.allclose(compressed["residual_norm"], expected_residual_norm, atol=0, rtol=0)


def test_compressor_v2_can_store_fp16_for_legacy_size() -> None:
    batch, heads, seq_len, dim = 1, 2, 8, 128
    generator = torch.Generator(device="cpu")
    generator.manual_seed(502)
    compressor = TurboQuantCompressorV2(
        dim,
        bits=3,
        seed=42,
        device="cpu",
        storage_dtype=torch.float16,
    )
    keys = torch.randn(batch, heads, seq_len, dim, generator=generator)

    compressed = compressor.compress(keys)

    assert compressed["k_mse"].dtype == torch.float16
    assert compressed["residual_norm"].dtype == torch.float16


def test_compressor_v2_attention_scores_are_consistent_across_storage_dtypes() -> None:
    batch, heads, seq_len, dim = 1, 2, 8, 128
    generator = torch.Generator(device="cpu")
    generator.manual_seed(505)
    keys = torch.randn(batch, heads, seq_len, dim, generator=generator)
    queries = torch.randn(batch, heads, 3, dim, generator=generator)
    fp32_compressor = TurboQuantCompressorV2(dim, bits=3, seed=42, device="cpu")
    fp16_compressor = TurboQuantCompressorV2(
        dim,
        bits=3,
        seed=42,
        device="cpu",
        storage_dtype=torch.float16,
    )

    fp32_scores = fp32_compressor.asymmetric_attention_scores(
        queries,
        fp32_compressor.compress(keys),
    )
    fp16_scores = fp16_compressor.asymmetric_attention_scores(
        queries,
        fp16_compressor.compress(keys),
    )

    assert torch.allclose(fp16_scores, fp32_scores, atol=1e-2, rtol=1e-3)


def test_compressor_mse_defaults_to_fp32_storage() -> None:
    batch, heads, seq_len, dim = 1, 2, 8, 128
    generator = torch.Generator(device="cpu")
    generator.manual_seed(503)
    compressor = TurboQuantCompressorMSE(dim, bits=3, seed=42, device="cpu")
    values = torch.randn(batch, heads, seq_len, dim, generator=generator)

    compressed = compressor.compress(values)

    assert compressed["vec_norms"].dtype == torch.float32


def test_compressor_rejects_invalid_storage_dtype() -> None:
    with pytest.raises(ValueError, match="storage_dtype"):
        TurboQuantCompressorV2(128, bits=3, seed=42, storage_dtype=torch.int8)

    with pytest.raises(ValueError, match="storage_dtype"):
        TurboQuantCompressorMSE(128, bits=3, seed=42, storage_dtype=torch.int8)
