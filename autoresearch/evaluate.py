#!/usr/bin/env python3
"""
TurboQuant autoresearch: Evaluation harness.

DO NOT MODIFY THIS FILE. It is the ground truth evaluator.
The agent modifies policy.py; this file reads it and evaluates.

Outputs grep-able metrics to stdout:
    composite_score:    0.8523
    avg_cosine_sim:     0.9912
    avg_top1_match:     0.9800
    compression_ratio:  4.3200
    peak_memory_mb:     12.34

Usage:
    python evaluate.py              # run evaluation
    python evaluate.py --quick      # fast mode (fewer seq lengths)
"""

import sys
import os
import time
import math
import importlib

import torch
import torch.nn.functional as F

# Add parent dir for turboquant imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compressors import TurboQuantCompressorV2, TurboQuantCompressorMSE


def load_policy():
    """Import the current policy from policy.py."""
    spec = importlib.util.spec_from_file_location(
        "policy",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.py"),
    )
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    return policy


def evaluate_layer(
    layer_idx: int,
    seq_len: int,
    policy,
    device: str = "cpu",
) -> dict:
    """
    Evaluate a single layer's quantization quality.

    Returns dict with: cosine_sim, top1_match, top5_match, compression_ratio, memory_bytes
    """
    cfg = policy.get_layer_config(layer_idx)
    head_dim = policy.HEAD_DIM
    n_heads = policy.NUM_HEADS
    n_queries = policy.N_QUERIES

    # Deterministic data generation (generator must be on CPU, then move tensors)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(policy.SEED + layer_idx * 1000 + seq_len)

    # Generate on CPU with seeded generator, then move to device
    keys = torch.randn(1, n_heads, seq_len, head_dim, generator=gen).to(device)
    values = torch.randn(1, n_heads, seq_len, head_dim, generator=gen).to(device)
    queries = torch.randn(1, n_heads, n_queries, head_dim, generator=gen).to(device)

    # Full-precision attention scores (ground truth)
    true_scores = torch.matmul(queries.float(), keys.float().transpose(-2, -1))
    true_scores = true_scores / math.sqrt(head_dim)

    # Compress keys
    key_bits = cfg["key_b_mse"] + (1 if cfg["key_qjl_enabled"] else 0)
    key_seed = layer_idx * 1000
    qjl_dim = int(head_dim * cfg["qjl_dim_ratio"]) if cfg["key_qjl_enabled"] else head_dim

    if cfg["key_qjl_enabled"]:
        key_compressor = TurboQuantCompressorV2(
            head_dim=head_dim,
            bits=key_bits,
            seed=key_seed,
            device=device,
        )
        compressed_keys = key_compressor.compress(keys)
        est_scores = key_compressor.asymmetric_attention_scores(queries, compressed_keys)
        est_scores = est_scores / math.sqrt(head_dim)
    else:
        # MSE-only for keys (no QJL correction)
        key_compressor = TurboQuantCompressorMSE(
            head_dim=head_dim,
            bits=cfg["key_b_mse"],
            seed=key_seed,
            device=device,
        )
        compressed_keys = key_compressor.compress(keys)
        k_recon = key_compressor.decompress(compressed_keys)
        est_scores = torch.matmul(queries.float(), k_recon.float().transpose(-2, -1))
        est_scores = est_scores / math.sqrt(head_dim)

    # Compress values
    val_seed = layer_idx * 1000 + 500
    val_compressor = TurboQuantCompressorMSE(
        head_dim=head_dim,
        bits=cfg["val_b_mse"],
        seed=val_seed,
        device=device,
    )

    # --- Metrics ---

    # 1. Cosine similarity of attention score vectors
    true_flat = true_scores.reshape(-1, seq_len)
    est_flat = est_scores.reshape(-1, seq_len)
    cosine_sim = F.cosine_similarity(true_flat, est_flat, dim=-1).mean().item()

    # 2. Top-1 match (does the estimated top key match the true top key?)
    true_top1 = true_scores.argmax(dim=-1)
    est_top1 = est_scores.argmax(dim=-1)
    top1_match = (true_top1 == est_top1).float().mean().item()

    # 3. Top-5 match (is the true top-1 in the estimated top-5?)
    est_top5 = est_scores.topk(min(5, seq_len), dim=-1).indices
    true_top1_expanded = true_top1.unsqueeze(-1).expand_as(est_top5)
    top5_match = (est_top5 == true_top1_expanded).any(dim=-1).float().mean().item()

    # 4. Compression ratio
    # FP16: 2 bytes per element, keys + values
    fp16_bytes = 2 * seq_len * head_dim * n_heads * 2  # keys + values

    # Compressed keys
    if cfg["key_qjl_enabled"]:
        # MSE indices: key_b_mse bits per coord + QJL signs: 1 bit per coord + norms: 4 bytes per vector
        key_bytes = seq_len * n_heads * (
            math.ceil(head_dim * cfg["key_b_mse"] / 8)  # MSE indices
            + math.ceil(head_dim * cfg["qjl_dim_ratio"] / 8)  # QJL signs
            + 4  # residual norm (fp16) + original norm (fp16)
        )
    else:
        key_bytes = seq_len * n_heads * (
            math.ceil(head_dim * cfg["key_b_mse"] / 8) + 2  # indices + norm
        )

    # Compressed values: val_b_mse bits per coord + norm
    val_bytes = seq_len * n_heads * (
        math.ceil(head_dim * cfg["val_b_mse"] / 8) + 2  # indices + norm
    )

    compression_ratio = fp16_bytes / (key_bytes + val_bytes)

    return {
        "cosine_sim": cosine_sim,
        "top1_match": top1_match,
        "top5_match": top5_match,
        "compression_ratio": compression_ratio,
        "compressed_bytes": key_bytes + val_bytes,
        "fp16_bytes": fp16_bytes,
    }


def main():
    quick = "--quick" in sys.argv
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = load_policy()

    seq_lengths = [512] if quick else policy.SEQ_LENGTHS
    num_layers = 28  # Qwen2.5-3B

    # Evaluate a representative subset of layers
    test_layers = [0, 3, 7, 13, 20, 27]  # early, mid, late

    t_start = time.time()

    all_cosine = []
    all_top1 = []
    all_top5 = []
    all_compression = []

    for layer_idx in test_layers:
        for seq_len in seq_lengths:
            result = evaluate_layer(layer_idx, seq_len, policy, device=device)
            all_cosine.append(result["cosine_sim"])
            all_top1.append(result["top1_match"])
            all_top5.append(result["top5_match"])
            all_compression.append(result["compression_ratio"])

    avg_cosine = sum(all_cosine) / len(all_cosine)
    avg_top1 = sum(all_top1) / len(all_top1)
    avg_top5 = sum(all_top5) / len(all_top5)
    avg_compression = sum(all_compression) / len(all_compression)
    min_cosine = min(all_cosine)

    # Composite score: accuracy * compression
    # Penalize if cosine drops below 0.95 (hard threshold)
    accuracy_score = avg_cosine * avg_top1
    if min_cosine < 0.90:
        accuracy_score *= 0.5  # heavy penalty for any layer dropping below 0.90
    composite = accuracy_score * avg_compression

    elapsed = time.time() - t_start
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 if device == "cuda" else 0

    # Policy summary
    cfg_default = policy.get_layer_config(0)
    n_overrides = len(policy.LAYER_OVERRIDES)

    # Output in grep-able format (matches autoresearch convention)
    print("---")
    print(f"composite_score:    {composite:.6f}")
    print(f"avg_cosine_sim:     {avg_cosine:.6f}")
    print(f"min_cosine_sim:     {min_cosine:.6f}")
    print(f"avg_top1_match:     {avg_top1:.6f}")
    print(f"avg_top5_match:     {avg_top5:.6f}")
    print(f"compression_ratio:  {avg_compression:.4f}")
    print(f"eval_seconds:       {elapsed:.1f}")
    print(f"peak_memory_mb:     {peak_mem:.1f}")
    print(f"key_b_mse:          {cfg_default['key_b_mse']}")
    print(f"val_b_mse:          {cfg_default['val_b_mse']}")
    print(f"qjl_enabled:        {cfg_default['key_qjl_enabled']}")
    print(f"qjl_dim_ratio:      {cfg_default['qjl_dim_ratio']}")
    print(f"layer_overrides:    {n_overrides}")


if __name__ == "__main__":
    main()
