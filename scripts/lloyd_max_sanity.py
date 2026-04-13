"""
Lloyd-Max codebook sanity check for TQ4P under non-isotropic activations.

Phase A: synthetic distributions (no model weights).
Phase B: real K/V activations from Qwen2.5-3B-Instruct.

For each distribution, samples are rotated via both WHT and Haar, then
quantized with the Gaussian-approx 3-bit Lloyd-Max codebook. Reports
per-coord post-rotation stats and measured MSE vs the Gaussian-approx bound.

Pass criterion: MSE within 1.5x of the Gaussian-approx bound.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from typing import Dict, List, Tuple

import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "patches" / "stage2-qjl" / "python"))

from lloyd_max import LloydMaxCodebook
import tq_paper_reference as ref


DIMS = [128, 256]
BITS = 3
THRESHOLD = 1.5
TEST_LAYER = 0


def rotation_label(rot: int) -> str:
    return "WHT" if rot == ref.TQP_ROT_WHT else "Haar"


def quantize_and_mse(
    x_unit: torch.Tensor,
    constants: ref.TQPConstants,
    layer_idx: int,
    rotation: int,
) -> Tuple[float, torch.Tensor]:
    """Rotate, quantize, back-rotate, compute per-vector MSE.

    Returns (mean_mse, rotated_coords) where rotated_coords is (N, d).
    """
    sigma = constants.sigma[layer_idx]
    pi = constants.pi[layer_idx]
    centroids = constants.centroids
    boundaries = constants.boundaries

    rotated = torch.stack([
        ref.rot_apply(rotation, sigma, pi, v) for v in x_unit
    ])

    indices = torch.bucketize(rotated, boundaries)
    x_hat_rot = centroids[indices]

    back_rotated = torch.stack([
        ref.rot_apply_t(rotation, sigma, pi, v) for v in x_hat_rot
    ])

    mse_per_vec = ((x_unit - back_rotated) ** 2).sum(dim=-1)
    return mse_per_vec.mean().item(), rotated


def make_unit(x: torch.Tensor) -> torch.Tensor:
    """Normalize rows to unit vectors."""
    norms = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return x / norms


def generate_synthetic_distributions(d: int, n: int = 10000) -> Dict[str, torch.Tensor]:
    """Generate synthetic test distributions as (n, d) tensors."""
    g = torch.Generator().manual_seed(12345)
    dists: Dict[str, torch.Tensor] = {}

    # 1. iid N(0, 1/d) baseline
    dists["iid_gaussian"] = torch.randn(n, d, generator=g) / math.sqrt(d)

    # 2. Outlier channels: 4 channels with ~10x variance
    x = torch.randn(n, d, generator=g) / math.sqrt(d)
    outlier_idx = [0, d // 4, d // 2, 3 * d // 4]
    for idx in outlier_idx:
        x[:, idx] *= 10.0
    dists["outlier_channels"] = x

    # 3. Heavy-tailed: Student-t, df=3, scaled to match variance ~1/d
    normal = torch.randn(n, d, generator=g)
    chi2_samples = (
        torch.randn(n, 1, generator=g) ** 2
        + torch.randn(n, 1, generator=g) ** 2
        + torch.randn(n, 1, generator=g) ** 2
    )
    student_t = normal / torch.sqrt(chi2_samples / 3.0)
    # Student-t(3) has var = df/(df-2) = 3. Scale to var ~ 1/d.
    dists["student_t_df3"] = student_t / math.sqrt(3.0 * d)

    return dists


def run_phase_a() -> List[dict]:
    """Phase A: synthetic distributions, no model weights."""
    print("=" * 70)
    print("Phase A: Synthetic Distributions")
    print("=" * 70)

    results = []
    for d in DIMS:
        constants = ref.load_constants(d)
        codebook = LloydMaxCodebook(d, BITS, use_exact=False)
        gaussian_bound = codebook.distortion * d

        dists = generate_synthetic_distributions(d)
        for name, raw in dists.items():
            x_unit = make_unit(raw)

            for rot in [ref.TQP_ROT_WHT, ref.TQP_ROT_HAAR]:
                mse, rotated = quantize_and_mse(x_unit, constants, TEST_LAYER, rot)
                ratio = mse / gaussian_bound if gaussian_bound > 0 else float("inf")
                passed = ratio <= THRESHOLD

                rot_mean = rotated.mean().item()
                rot_std = rotated.std().item()
                centered = (rotated - rotated.mean()) / rotated.std()
                rot_kurt = (centered ** 4).mean().item() - 3.0

                row = {
                    "phase": "A",
                    "d": d,
                    "dist": name,
                    "rotation": rotation_label(rot),
                    "mse": mse,
                    "bound": gaussian_bound,
                    "ratio": ratio,
                    "pass": passed,
                    "rot_mean": rot_mean,
                    "rot_std": rot_std,
                    "rot_kurtosis": rot_kurt,
                }
                results.append(row)

                status = "PASS" if passed else "FAIL"
                print(
                    f"  d={d:3d} | {name:20s} | {rotation_label(rot):4s} | "
                    f"MSE={mse:.6f} | bound={gaussian_bound:.6f} | "
                    f"ratio={ratio:.3f} | {status}"
                )
                print(
                    f"         post-rot stats: mean={rot_mean:+.4e} "
                    f"std={rot_std:.4e} kurtosis={rot_kurt:+.3f}"
                )

    return results


def run_phase_b(model_name: str, n_prompts: int = 8) -> List[dict]:
    """Phase B: real activations from a transformer model."""
    print()
    print("=" * 70)
    print(f"Phase B: Real Activations ({model_name})")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    target_layers = [0, num_layers // 2, num_layers - 1]
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "In quantum computing, a qubit can be in a superposition of states.",
        "Machine learning models require large amounts of training data.",
        "The Fourier transform converts a signal from time domain to frequency domain.",
        "Gradient descent is an iterative optimization algorithm.",
        "Neural networks consist of layers of interconnected neurons.",
        "The halting problem is undecidable in general.",
        "Attention mechanisms allow models to focus on relevant input tokens.",
    ][:n_prompts]

    print(f"  head_dim={head_dim}, num_layers={num_layers}, target_layers={target_layers}")
    print(f"  Running {len(prompts)} prompts...")

    k_activations: Dict[int, List[torch.Tensor]] = {l: [] for l in target_layers}
    v_activations: Dict[int, List[torch.Tensor]] = {l: [] for l in target_layers}

    hooks = []

    for layer_idx in target_layers:
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn

        def make_k_hook(lidx: int):
            def fn(module, inp, out):
                bsz, seq_len, _ = out.shape
                num_kv_heads = out.shape[-1] // head_dim
                reshaped = out.reshape(bsz, seq_len, num_kv_heads, head_dim)
                k_activations[lidx].append(reshaped.detach().reshape(-1, head_dim))
            return fn

        def make_v_hook(lidx: int):
            def fn(module, inp, out):
                bsz, seq_len, _ = out.shape
                num_kv_heads = out.shape[-1] // head_dim
                reshaped = out.reshape(bsz, seq_len, num_kv_heads, head_dim)
                v_activations[lidx].append(reshaped.detach().reshape(-1, head_dim))
            return fn

        hooks.append(attn.k_proj.register_forward_hook(make_k_hook(layer_idx)))
        hooks.append(attn.v_proj.register_forward_hook(make_v_hook(layer_idx)))

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            model(**inputs)

    for h in hooks:
        h.remove()

    results = []
    matched_d = [d for d in DIMS if d == head_dim]

    if not matched_d:
        print(f"  head_dim={head_dim} does not match DIMS={DIMS}")
        print("  Collecting raw activation stats (no quantization):")
        for layer_idx in target_layers:
            for kv_label, activations in [("K", k_activations), ("V", v_activations)]:
                if not activations[layer_idx]:
                    continue
                raw = torch.cat(activations[layer_idx], dim=0)
                norms = raw.norm(dim=-1)
                per_ch_var = raw.var(dim=0)
                print(
                    f"  {kv_label}_layer{layer_idx:2d}: shape={raw.shape}, "
                    f"norm mean={norms.mean():.3f} std={norms.std():.3f}, "
                    f"per-ch var range=[{per_ch_var.min():.4f}, {per_ch_var.max():.4f}], "
                    f"var ratio={per_ch_var.max()/per_ch_var.min():.1f}x"
                )
        return results

    d = matched_d[0]
    constants = ref.load_constants(d)
    codebook = LloydMaxCodebook(d, BITS, use_exact=False)
    gaussian_bound = codebook.distortion * d

    for layer_idx in target_layers:
        for kv_label, activations in [("K", k_activations), ("V", v_activations)]:
            if not activations[layer_idx]:
                continue

            raw = torch.cat(activations[layer_idx], dim=0)
            x_unit = make_unit(raw)
            if x_unit.shape[0] > 10000:
                perm = torch.randperm(x_unit.shape[0])[:10000]
                x_unit = x_unit[perm]

            for rot in [ref.TQP_ROT_WHT, ref.TQP_ROT_HAAR]:
                mse, rotated = quantize_and_mse(x_unit, constants, layer_idx % 32, rot)
                ratio = mse / gaussian_bound if gaussian_bound > 0 else float("inf")
                passed = ratio <= THRESHOLD

                rot_std = rotated.std().item()
                centered = (rotated - rotated.mean()) / rotated.std()
                rot_kurt = (centered ** 4).mean().item() - 3.0

                row = {
                    "phase": "B",
                    "d": d,
                    "dist": f"{kv_label}_layer{layer_idx}",
                    "rotation": rotation_label(rot),
                    "mse": mse,
                    "bound": gaussian_bound,
                    "ratio": ratio,
                    "pass": passed,
                    "rot_mean": rotated.mean().item(),
                    "rot_std": rot_std,
                    "rot_kurtosis": rot_kurt,
                    "n_vectors": x_unit.shape[0],
                }
                results.append(row)

                status = "PASS" if passed else "FAIL"
                print(
                    f"  d={d:3d} | {kv_label}_layer{layer_idx:2d} | {rotation_label(rot):4s} | "
                    f"MSE={mse:.6f} | bound={gaussian_bound:.6f} | "
                    f"ratio={ratio:.3f} | {status} | n={x_unit.shape[0]}"
                )

    return results


def print_summary(results: List[dict]) -> bool:
    """Print summary table and return True if all pass."""
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    header = (
        f"{'Phase':5s} | {'d':>3s} | {'Distribution':20s} | {'Rot':4s} | "
        f"{'MSE':>10s} | {'Bound':>10s} | {'Ratio':>6s} | {'Pass':4s}"
    )
    print(header)
    print("-" * len(header))

    all_pass = True
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        if not r["pass"]:
            all_pass = False
        print(
            f"{r['phase']:5s} | {r['d']:3d} | {r['dist']:20s} | {r['rotation']:4s} | "
            f"{r['mse']:10.6f} | {r['bound']:10.6f} | {r['ratio']:6.3f} | {status}"
        )

    print()
    if all_pass:
        print("ALL PASS: MSE within 1.5x of Gaussian-approx bound.")
    else:
        failed = [r for r in results if not r["pass"]]
        print(f"FAILURES: {len(failed)} test(s) exceeded 1.5x threshold.")
        for r in failed:
            print(f"  - {r['dist']} d={r['d']} {r['rotation']}: ratio={r['ratio']:.3f}")

    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description="Lloyd-Max codebook sanity check for TQ4P"
    )
    parser.add_argument(
        "--skip-phase-b", action="store_true", help="Skip Phase B (real activations)"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HF model for Phase B",
    )
    parser.add_argument(
        "--n-prompts", type=int, default=8, help="Number of prompts for Phase B"
    )
    args = parser.parse_args()

    results = run_phase_a()

    if not args.skip_phase_b:
        results.extend(run_phase_b(args.model, args.n_prompts))

    all_pass = print_summary(results)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
