import math

import torch

from lloyd_max import LloydMaxCodebook, beta_pdf
from turboquant import TurboQuantKVCache, TurboQuantMSE, TurboQuantProd


def test_zero_vector_quantization_stays_finite() -> None:
    quantizer = TurboQuantProd(d=32, bits=3, seed=5, device="cpu")
    compressed = quantizer.quantize(torch.zeros(1, 32))
    reconstructed = quantizer.dequantize(compressed)

    assert torch.isfinite(compressed["residual_norm"]).all()
    assert torch.isfinite(compressed["qjl_signs"]).all()
    assert torch.isfinite(reconstructed).all()


def test_tiny_norm_quantization_stays_stable() -> None:
    quantizer = TurboQuantProd(d=32, bits=3, seed=13, device="cpu")
    vector = torch.randn(1, 32) * 1e-10
    compressed = quantizer.quantize(vector)
    score = quantizer.inner_product(vector, compressed)

    assert torch.isfinite(compressed["residual_norm"]).all()
    assert torch.isfinite(score).all()


def test_large_dimension_codebook_has_no_nan_centroids() -> None:
    codebook = LloydMaxCodebook(d=1024, bits=2)

    assert torch.isfinite(codebook.centroids).all()
    assert torch.isfinite(codebook.boundaries).all()
    assert math.isfinite(codebook.distortion)


def test_beta_pdf_is_finite_near_boundary() -> None:
    value = beta_pdf(0.9999, 128)

    assert math.isfinite(value)
    assert value >= 0.0


def test_single_vector_batch_shapes_work() -> None:
    mse_quantizer = TurboQuantMSE(d=32, bits=3, seed=19, device="cpu")
    prod_quantizer = TurboQuantProd(d=32, bits=3, seed=29, device="cpu")
    vector = torch.randn(1, 32)
    vector = vector / torch.norm(vector, dim=-1, keepdim=True).clamp_min(1e-8)

    reconstructed, indices = mse_quantizer(vector)
    compressed = prod_quantizer.quantize(vector)
    estimated_ip = prod_quantizer.inner_product(vector, compressed)

    assert reconstructed.shape == (1, 32)
    assert indices.shape == (1, 32)
    assert compressed["mse_indices"].shape == (1, 32)
    assert estimated_ip.shape == (1,)


def test_empty_cache_attention_scores_returns_empty_tensor() -> None:
    cache = TurboQuantKVCache(d_key=32, d_value=32, bits=3, seed=31, device="cpu")
    scores = cache.attention_scores(torch.randn(1, 32))

    assert scores.numel() == 0
    assert scores.shape == (0,)
