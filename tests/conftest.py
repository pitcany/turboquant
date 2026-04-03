import pytest
import torch

from turboquant import TurboQuantKVCache, TurboQuantMSE, TurboQuantProd


def _make_unit_vectors(batch: int, dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    vectors = torch.randn(batch, dim, generator=generator)
    return vectors / torch.norm(vectors, dim=-1, keepdim=True).clamp_min(1e-8)


@pytest.fixture
def unit_vectors() -> torch.Tensor:
    return _make_unit_vectors(batch=256, dim=64, seed=123)


@pytest.fixture
def mse_quantizer() -> TurboQuantMSE:
    return TurboQuantMSE(d=64, bits=3, seed=11, device="cpu")


@pytest.fixture
def prod_quantizer() -> TurboQuantProd:
    return TurboQuantProd(d=64, bits=3, seed=17, device="cpu")


@pytest.fixture
def kv_cache() -> TurboQuantKVCache:
    return TurboQuantKVCache(d_key=32, d_value=32, bits=4, seed=23, device="cpu")
