import math
import unittest

import torch
import torch.nn.functional as F

from turboquant import TurboQuantMSE, TurboQuantProd
from vllm_plugin.attention import _CompressedLayout
from vllm_plugin.triton_wrapper import (
    prerotate_queries,
    turboquant_decode_attention,
    turboquant_decode_attention_pytorch,
)


def _build_fixture(
    *,
    seq_len: int = 19,
    q_len: int = 1,
    num_kv_heads: int = 2,
    heads_per_kv: int = 2,
    head_dim: int = 16,
    device: str = "cpu",
):
    torch.manual_seed(7)
    num_heads = num_kv_heads * heads_per_kv
    layout = _CompressedLayout(head_dim, key_mse_bits=2, key_qjl_bits=1,
                               val_mse_bits=3)

    key_q = TurboQuantProd(head_dim, 3, seed=11, device=device)
    val_q = TurboQuantMSE(head_dim, 3, seed=23, device=device)

    keys = torch.randn(seq_len, num_kv_heads, head_dim, device=device)
    values = torch.randn(seq_len, num_kv_heads, head_dim, device=device)
    queries = torch.randn(q_len, num_heads, head_dim, device=device)

    k_norm = torch.norm(keys, dim=-1)
    k_unit = keys / (k_norm.unsqueeze(-1) + 1e-8)
    v_norm = torch.norm(values, dim=-1)
    v_unit = values / (v_norm.unsqueeze(-1) + 1e-8)

    key_comp = key_q.quantize(k_unit.reshape(-1, head_dim))
    val_idx = val_q.quantize(v_unit.reshape(-1, head_dim))
    packed = layout.pack(
        key_comp["mse_indices"],
        key_comp["qjl_signs"],
        key_comp["residual_norm"],
        k_norm.reshape(-1),
        val_idx,
        v_norm.reshape(-1),
    )
    comp_bytes = packed.reshape(seq_len, num_kv_heads, layout.total_bytes)

    return {
        "queries": queries,
        "comp_bytes": comp_bytes,
        "layout": layout,
        "key_q": key_q,
        "val_q": val_q,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "heads_per_kv": heads_per_kv,
        "head_dim": head_dim,
    }


def _reference_attention(
    queries: torch.Tensor,
    comp_bytes: torch.Tensor,
    layout: _CompressedLayout,
    key_q: TurboQuantProd,
    val_q: TurboQuantMSE,
    heads_per_kv: int,
    *,
    causal: bool,
    pos_offset: int,
) -> torch.Tensor:
    seq_len, num_kv_heads, head_dim = comp_bytes.shape[0], comp_bytes.shape[1], layout.head_dim
    q_len = queries.shape[0]
    flat_bytes = comp_bytes.reshape(seq_len * num_kv_heads, layout.total_bytes)
    (km_idx, k_signs, k_rnorm,
     k_norm, vm_idx, v_norm) = layout.unpack(flat_bytes)
    km_idx = km_idx.reshape(seq_len, num_kv_heads, head_dim)
    k_signs = k_signs.reshape(seq_len, num_kv_heads, head_dim)
    k_rnorm = k_rnorm.reshape(seq_len, num_kv_heads)
    k_norm = k_norm.reshape(seq_len, num_kv_heads)
    vm_idx = vm_idx.reshape(seq_len, num_kv_heads, head_dim)
    v_norm = v_norm.reshape(seq_len, num_kv_heads)

    k_rot = key_q.mse.centroids[km_idx.reshape(-1, head_dim)]
    k_mse = (k_rot @ key_q.mse.Pi).reshape(seq_len, num_kv_heads, head_dim)
    v_rot = val_q.centroids[vm_idx.reshape(-1, head_dim)]
    v_full = (v_rot @ val_q.Pi).reshape(seq_len, num_kv_heads, head_dim)
    v_full = v_full * v_norm.unsqueeze(-1)

    q_view = queries.reshape(q_len, num_kv_heads, heads_per_kv, head_dim)
    t1 = torch.einsum("qghd,sgd->qghs", q_view, k_mse)
    q_proj = q_view @ key_q.S.T
    t2_raw = torch.einsum("qghd,sgd->qghs", q_proj, k_signs)
    corr = math.sqrt(math.pi / 2.0) / key_q.qjl_dim
    t2 = corr * t2_raw * k_rnorm.permute(1, 0)[None, :, None, :]
    scores = (t1 + t2) * k_norm.permute(1, 0)[None, :, None, :]
    scores = scores / math.sqrt(head_dim)

    if causal and q_len > 1:
        pos = torch.arange(seq_len, device=queries.device)
        qpos = torch.arange(q_len, device=queries.device) + pos_offset
        mask = pos[None, :] > qpos[:, None]
        scores.masked_fill_(mask[:, None, None, :], float("-inf"))

    weights = F.softmax(scores.float(), dim=-1)
    out = torch.einsum("qghs,sgd->qghd", weights, v_full)
    return out.reshape(q_len, num_kv_heads * heads_per_kv, head_dim)


class TritonWrapperTests(unittest.TestCase):
    def test_prerotate_queries_matches_manual_projection(self) -> None:
        fx = _build_fixture(q_len=3)
        queries = fx["queries"].reshape(3, fx["num_kv_heads"], fx["heads_per_kv"], fx["head_dim"])
        q_rot, q_sketch = prerotate_queries(
            queries,
            fx["key_q"].mse.Pi.T,
            fx["key_q"].S.T,
        )
        expected_rot = queries @ fx["key_q"].mse.Pi.T
        expected_sketch = queries @ fx["key_q"].S.T
        self.assertTrue(torch.allclose(q_rot, expected_rot, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(q_sketch, expected_sketch, atol=1e-6, rtol=1e-6))

    def test_layout_roundtrip_is_exact(self) -> None:
        fx = _build_fixture()
        flat_bytes = fx["comp_bytes"].reshape(-1, fx["layout"].total_bytes)
        unpacked = fx["layout"].unpack(flat_bytes)
        repacked = fx["layout"].pack(*unpacked)
        self.assertTrue(torch.equal(flat_bytes, repacked))

    def test_pytorch_decode_matches_reference(self) -> None:
        fx = _build_fixture(q_len=4, seq_len=19)
        expected = _reference_attention(
            fx["queries"],
            fx["comp_bytes"],
            fx["layout"],
            fx["key_q"],
            fx["val_q"],
            fx["heads_per_kv"],
            causal=True,
            pos_offset=15,
        )
        actual = turboquant_decode_attention_pytorch(
            fx["queries"],
            fx["comp_bytes"],
            fx["layout"],
            key_centroids=fx["key_q"].mse.centroids,
            val_centroids=fx["val_q"].centroids,
            key_pi=fx["key_q"].mse.Pi,
            key_pi_t=fx["key_q"].mse.Pi.T,
            val_pi=fx["val_q"].Pi,
            s_t=fx["key_q"].S.T,
            heads_per_kv=fx["heads_per_kv"],
            qjl_dim=fx["key_q"].qjl_dim,
            sm_scale=1.0 / math.sqrt(fx["head_dim"]),
            causal=True,
            pos_offset=15,
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-4, rtol=1e-4))

    def test_split_decode_matches_reference(self) -> None:
        fx = _build_fixture(q_len=1, seq_len=21)
        expected = _reference_attention(
            fx["queries"],
            fx["comp_bytes"],
            fx["layout"],
            fx["key_q"],
            fx["val_q"],
            fx["heads_per_kv"],
            causal=True,
            pos_offset=20,
        )
        actual = turboquant_decode_attention(
            fx["queries"],
            fx["comp_bytes"],
            fx["layout"],
            key_centroids=fx["key_q"].mse.centroids,
            val_centroids=fx["val_q"].centroids,
            key_pi=fx["key_q"].mse.Pi,
            key_pi_t=fx["key_q"].mse.Pi.T,
            val_pi=fx["val_q"].Pi,
            s_t=fx["key_q"].S.T,
            heads_per_kv=fx["heads_per_kv"],
            qjl_dim=fx["key_q"].qjl_dim,
            sm_scale=1.0 / math.sqrt(fx["head_dim"]),
            causal=True,
            pos_offset=20,
            num_kv_splits=4,
            use_triton=False,
        )
        cosine = F.cosine_similarity(
            actual.reshape(1, -1).float(),
            expected.reshape(1, -1).float(),
        ).item()
        self.assertGreaterEqual(cosine, 0.999)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Triton kernel test")
    def test_triton_decode_matches_reference_on_cuda(self) -> None:
        fx = _build_fixture(q_len=1, seq_len=21, device="cuda")
        expected = _reference_attention(
            fx["queries"],
            fx["comp_bytes"],
            fx["layout"],
            fx["key_q"],
            fx["val_q"],
            fx["heads_per_kv"],
            causal=True,
            pos_offset=20,
        )
        actual = turboquant_decode_attention(
            fx["queries"],
            fx["comp_bytes"],
            fx["layout"],
            key_centroids=fx["key_q"].mse.centroids,
            val_centroids=fx["val_q"].centroids,
            key_pi=fx["key_q"].mse.Pi,
            key_pi_t=fx["key_q"].mse.Pi.T,
            val_pi=fx["val_q"].Pi,
            s_t=fx["key_q"].S.T,
            heads_per_kv=fx["heads_per_kv"],
            qjl_dim=fx["key_q"].qjl_dim,
            sm_scale=1.0 / math.sqrt(fx["head_dim"]),
            causal=True,
            pos_offset=20,
            num_kv_splits=4,
            use_triton=True,
        )
        self.assertTrue(torch.allclose(actual, expected, atol=2e-3, rtol=2e-3))


if __name__ == "__main__":
    unittest.main()
