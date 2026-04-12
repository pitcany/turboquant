"""
TurboQuant validation: compare attention scores on real model data.
Single forward pass, no monkey-patching. Just math.

Measures:
  - TQ 2/3/4-bit: Python TurboQuant compressors (existing)
  - TQ4P (C):     paper-faithful C implementation via ctypes (Stage 1+2)
"""

import ctypes
import math
import os
import pathlib
import struct
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

# Allow running as `python validate.py` from within the package directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from compressors import TurboQuantCompressorV2, TurboQuantCompressorMSE

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
NEEDLE = "The secret project code name is AURORA-7749."
QUESTION = "What is the secret project code name?"

FILLER = """The quarterly financial review meeting covered several topics including
budget allocations for the upcoming fiscal year, departmental spending reports, and projected
revenue streams from various business units. The committee discussed infrastructure upgrades
planned for the western regional offices and noted that maintenance schedules should be
coordinated with the facilities management team. Several action items were assigned to team
leads for follow-up before the next meeting cycle.\n\n"""


# ---------------------------------------------------------------------------
# TQ4P C library loading
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parent
_C_DIR = _REPO_ROOT / "patches" / "stage2-qjl" / "c"
_LIB_PATH = _C_DIR / "libggml_tq_paper.so"


def _ensure_tq4p_lib() -> ctypes.CDLL:
    """Build the shared library if missing, then load it."""
    if not _LIB_PATH.exists():
        print("[+] building libggml_tq_paper.so ...", flush=True)
        result = subprocess.run(
            ["make", "-C", str(_C_DIR), "libggml_tq_paper.so"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(
                f"Failed to build {_LIB_PATH}. Run 'make -C {_C_DIR}' manually."
            )
        print("[+] built successfully", flush=True)
    return ctypes.CDLL(str(_LIB_PATH))


# Block structs -- mirror test_c_vs_python.py exactly.
class block_tq4p_d128(ctypes.Structure):
    _fields_ = [
        ("orig_norm", ctypes.c_uint16),
        ("res_d", ctypes.c_uint16),
        ("qs", ctypes.c_uint8 * 48),
        ("qjl_signs", ctypes.c_uint8 * 16),
    ]


class block_tq4p_d256(ctypes.Structure):
    _fields_ = [
        ("orig_norm", ctypes.c_uint16),
        ("res_d", ctypes.c_uint16),
        ("qs", ctypes.c_uint8 * 96),
        ("qjl_signs", ctypes.c_uint8 * 32),
    ]


_BLK_CLS = {128: block_tq4p_d128, 256: block_tq4p_d256}
_BLK_SIZE = {128: 68, 256: 132}


def _setup_lib_signatures(lib: ctypes.CDLL) -> None:
    """Declare ctypes argtypes/restypes for the C functions we call."""
    for suffix, blk_cls in [("d128", block_tq4p_d128), ("d256", block_tq4p_d256)]:
        # quantize
        fn = getattr(lib, f"ggml_quantize_row_tq4p_{suffix}")
        fn.restype = None
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(blk_cls),
            ctypes.c_int64,
        ]
        # prepare_query
        fn = getattr(lib, f"ggml_tqp_prepare_query_{suffix}")
        fn.restype = None
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        # vec_dot_block
        fn = getattr(lib, f"ggml_tqp_vec_dot_block_{suffix}")
        fn.restype = ctypes.c_float
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(blk_cls),
        ]


# ---------------------------------------------------------------------------
# TQ4P compress / attention via C
# ---------------------------------------------------------------------------

def tq4p_c_compress(
    lib: ctypes.CDLL,
    keys: torch.Tensor,
    d: int,
) -> list[bytes]:
    """
    Quantize each head-vector in *keys* via the C library.

    Args:
        keys: (B, H, S, D) fp32 tensor (on CPU).
        d: head_dim (128 or 256).

    Returns:
        List of packed block bytes, one per (b, h, s) vector.
        Length = B * H * S.
    """
    B, H, S, D = keys.shape
    assert D == d
    blk_cls = _BLK_CLS[d]
    quantize_fn = getattr(lib, f"ggml_quantize_row_tq4p_d{d}")
    blocks: list[bytes] = []

    flat = keys.reshape(-1, D).contiguous().float()
    for i in range(flat.shape[0]):
        vec = flat[i].numpy().copy()
        blk = blk_cls()
        quantize_fn(
            vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(blk),
            ctypes.c_int64(D),
        )
        blocks.append(bytes(blk))

    return blocks


def tq4p_c_attention_scores(
    lib: ctypes.CDLL,
    queries: torch.Tensor,
    key_blocks: list[bytes],
    d: int,
    n_keys: int,
) -> torch.Tensor:
    """
    Compute attention scores <Q, K> using the C inner-product estimator.

    Args:
        queries: (B, H, 1, D) fp32 CPU tensor (last-token queries).
        key_blocks: packed block bytes from tq4p_c_compress, length B*H*S.
        d: head_dim.
        n_keys: S (sequence length).

    Returns:
        scores: (B, H, 1, n_keys) fp32 tensor.
    """
    B, H, Sq, D = queries.shape
    assert Sq == 1 and D == d
    blk_cls = _BLK_CLS[d]
    prepare_fn = getattr(lib, f"ggml_tqp_prepare_query_d{d}")
    dot_fn = getattr(lib, f"ggml_tqp_vec_dot_block_d{d}")

    scores = torch.zeros(B, H, 1, n_keys)

    for b in range(B):
        for h in range(H):
            q_vec = queries[b, h, 0].contiguous().float().numpy().copy()
            q_ptr = q_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

            # Prepare Sq = S * q once per query
            Sq_buf = (ctypes.c_float * D)()
            prepare_fn(q_ptr, Sq_buf)

            for s in range(n_keys):
                blk_idx = (b * H + h) * n_keys + s
                blk = blk_cls.from_buffer_copy(key_blocks[blk_idx])
                score = dot_fn(q_ptr, Sq_buf, ctypes.byref(blk))
                scores[b, h, 0, s] = score

    return scores


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(tokenizer, target_tokens: int = 2048, needle_pos: float = 0.5) -> str:
    filler_len = len(tokenizer.encode(FILLER))
    n_reps = max(1, target_tokens // filler_len)
    needle_idx = int(n_reps * needle_pos)
    parts = []
    for i in range(n_reps):
        if i == needle_idx:
            parts.append(f"\n--- Memo ---\n{NEEDLE}\n--- End ---\n\n")
        parts.append(FILLER)
    haystack = "".join(parts)
    return f"<|im_start|>user\n{haystack}\nQuestion: {QUESTION}<|im_end|>\n<|im_start|>assistant\n"


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def _find_needle(tokenizer, input_ids: torch.Tensor) -> int | None:
    """Find the token position of the needle phrase in input_ids."""
    needle_phrase = "AURORA-7749"
    needle_tokens = tokenizer.encode(needle_phrase, add_special_tokens=False)
    input_ids_list = input_ids[0].tolist()
    for i in range(len(input_ids_list) - len(needle_tokens) + 1):
        if input_ids_list[i : i + len(needle_tokens)] == needle_tokens:
            return i
    # Partial match fallback
    for width in range(len(needle_tokens), 0, -1):
        sub = needle_tokens[:width]
        for i in range(len(input_ids_list) - width + 1):
            if input_ids_list[i : i + width] == sub:
                return i
    return None


def _score_metrics(
    real_scores: torch.Tensor,
    approx_scores: torch.Tensor,
    needle_start: int | None,
) -> dict:
    """Compute per-head metrics between real and approximate attention scores."""
    H = real_scores.shape[1]
    cosine_sims = []
    top1_matches = 0
    top5_matches = 0
    needle_rank_sum = 0

    for h in range(H):
        rs = real_scores[0, h]
        ts = approx_scores[0, h]

        cos = F.cosine_similarity(rs.unsqueeze(0), ts.unsqueeze(0)).item()
        cosine_sims.append(cos)

        real_top1 = rs.argmax().item()
        tq_top1 = ts.argmax().item()
        if real_top1 == tq_top1:
            top1_matches += 1

        tq_top5 = ts.topk(5).indices.tolist()
        if real_top1 in tq_top5:
            top5_matches += 1

        if needle_start is not None:
            needle_rank = (ts.argsort(descending=True) == needle_start).nonzero()
            if len(needle_rank) > 0:
                needle_rank_sum += needle_rank[0].item()

    return {
        "cosine_sims": cosine_sims,
        "top1_matches": top1_matches,
        "top5_matches": top5_matches,
        "needle_rank_sum": needle_rank_sum,
        "n_checks": H,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load TQ4P C library
    lib = _ensure_tq4p_lib()
    _setup_lib_signatures(lib)

    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        ),
        device_map={"": "cuda:1"},
        dtype=torch.float16,
    )
    model.eval()
    print(f"Loaded. GPU: {torch.cuda.memory_allocated() // 1024 // 1024} MB\n")

    for target_tokens in [2048, 4096, 8192]:
        prompt = build_prompt(tokenizer, target_tokens)
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=target_tokens + 256
        ).to("cuda:1")
        seq_len = inputs["input_ids"].shape[1]
        needle_start = _find_needle(tokenizer, inputs["input_ids"])

        print(f"{'=' * 78}")
        print(f"Context: {seq_len} tokens | Needle at token {needle_start}")
        print(f"{'=' * 78}")

        # Forward pass -- capture KV cache
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True, output_attentions=False)
        cache = outputs.past_key_values

        n_layers = len(cache.layers)
        head_dim = cache.layers[0].keys.shape[-1]
        num_kv_heads = cache.layers[0].keys.shape[1]

        # ---- Python TurboQuant compressors (existing: 2/3/4-bit) ----

        for bits in [2, 3, 4]:
            total_compressed_bytes = 0
            total_uncompressed_bytes = 0
            all_metrics: list[dict] = []

            for layer_idx in range(n_layers):
                keys = cache.layers[layer_idx].keys
                values = cache.layers[layer_idx].values
                B, H, S, D = keys.shape

                key_comp = TurboQuantCompressorV2(D, bits, seed=layer_idx * 1000, device="cuda:1")
                val_comp = TurboQuantCompressorMSE(D, bits, seed=layer_idx * 1000 + 500, device="cuda:1")

                compressed_k = key_comp.compress(keys)
                compressed_v = val_comp.compress(values)

                # Memory accounting
                n_key_vecs = B * H * S
                mse_bits = max(bits - 1, 1)
                k_bits = n_key_vecs * D * mse_bits
                k_bits += n_key_vecs * D * 1
                k_bits += n_key_vecs * 16
                k_bits += n_key_vecs * 16
                v_bits = n_key_vecs * D * bits
                v_bits += n_key_vecs * 16

                total_compressed_bytes += (k_bits + v_bits) / 8
                total_uncompressed_bytes += (keys.numel() + values.numel()) * 2

                query = keys[:, :, -1:, :]
                real_scores = torch.matmul(
                    query.float(), keys.float().transpose(-2, -1)
                ).squeeze(-2)
                tq_scores = key_comp.asymmetric_attention_scores(
                    query, compressed_k
                ).squeeze(-2)

                metrics = _score_metrics(real_scores, tq_scores, needle_start)
                all_metrics.append(metrics)

            _print_summary(f"TQ-{bits}bit", all_metrics,
                           total_compressed_bytes, total_uncompressed_bytes,
                           needle_start)

        # ---- TQ4P (C implementation) ----

        # Only run for d=128 or d=256 (the two supported block sizes).
        if head_dim in (128, 256):
            total_compressed_bytes = 0
            total_uncompressed_bytes = 0
            all_metrics = []

            for layer_idx in range(n_layers):
                keys = cache.layers[layer_idx].keys
                B, H, S, D = keys.shape

                # Move to CPU fp32 for C calls
                keys_cpu = keys.cpu().float()

                # Compress via C
                key_blocks = tq4p_c_compress(lib, keys_cpu, D)

                # Memory accounting: block_size bytes per vector (keys only,
                # no separate value compression in this measurement).
                n_key_vecs = B * H * S
                total_compressed_bytes += n_key_vecs * _BLK_SIZE[D]
                total_uncompressed_bytes += keys.numel() * 2  # fp16 baseline

                # Attention scores for last-token query
                query_cpu = keys_cpu[:, :, -1:, :]
                real_scores = torch.matmul(
                    query_cpu, keys_cpu.transpose(-2, -1)
                ).squeeze(-2)

                tq4p_scores = tq4p_c_attention_scores(
                    lib, query_cpu, key_blocks, D, S
                ).squeeze(-2)

                metrics = _score_metrics(real_scores, tq4p_scores, needle_start)
                all_metrics.append(metrics)

            _print_summary(f"TQ4P-C (d={head_dim})", all_metrics,
                           total_compressed_bytes, total_uncompressed_bytes,
                           needle_start)
        else:
            print(f"\n  TQ4P-C: skipped (head_dim={head_dim} not in {{128, 256}})")

        print()

    print("=" * 78)
    print("DONE")
    print("=" * 78)


def _print_summary(
    label: str,
    all_metrics: list[dict],
    total_compressed_bytes: float,
    total_uncompressed_bytes: float,
    needle_start: int | None,
) -> None:
    """Print a summary row for one compressor config."""
    cosine_sims = []
    top1_matches = 0
    top5_matches = 0
    needle_rank_sum = 0
    n_checks = 0

    for m in all_metrics:
        cosine_sims.extend(m["cosine_sims"])
        top1_matches += m["top1_matches"]
        top5_matches += m["top5_matches"]
        needle_rank_sum += m["needle_rank_sum"]
        n_checks += m["n_checks"]

    ratio = total_uncompressed_bytes / total_compressed_bytes
    avg_cos = sum(cosine_sims) / len(cosine_sims)
    top1_pct = 100 * top1_matches / n_checks
    top5_pct = 100 * top5_matches / n_checks
    avg_needle_rank = needle_rank_sum / n_checks if needle_start else -1

    print(f"\n  {label}:")
    print(f"    Compression:       {ratio:.1f}x  ({total_compressed_bytes / 1024 / 1024:.1f} MB vs {total_uncompressed_bytes / 1024 / 1024:.1f} MB)")
    print(f"    Score cosine sim:  {avg_cos:.6f}  (1.0 = perfect)")
    print(f"    Top-1 match:       {top1_pct:.1f}%  ({top1_matches}/{n_checks} heads)")
    print(f"    Top-5 match:       {top5_pct:.1f}%  ({top5_matches}/{n_checks} heads)")
    if needle_start is not None:
        print(f"    Avg needle rank:   {avg_needle_rank:.1f}  (lower = better, 0 = top)")


if __name__ == "__main__":
    main()
