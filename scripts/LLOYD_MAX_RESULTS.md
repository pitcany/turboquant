# Lloyd-Max Codebook Sanity Results

Model: Qwen/Qwen2.5-3B-Instruct (head_dim=128, 36 layers)
Gaussian-approx bound (3-bit, d=128): 0.034548
Gaussian-approx bound (3-bit, d=256): 0.034548

> **Note:** the Phase B numbers below were produced with the initial
> version of `scripts/lloyd_max_sanity.py` that silently wrapped
> `layer_idx % 32` when indexing into the 32-layer TQ4P constants. That
> paired layer-35 activations with layer-3 constants (both rotation σ
> and Haar Π) while labelling the row "Layer 35" — the MSE ratio is
> still a valid codebook-vs-rotation signal but the row label is
> misleading. The script now clamps `target_layers` to
> `min(num_layers, ref.MAX_LAYERS)`, so a rerun on the same model will
> report the ceiling layer (31) instead of 35. Rerun pending.

## Phase A: Synthetic Distributions

| d   | Distribution    | Rotation | MSE      | Ratio |
|-----|-----------------|----------|----------|-------|
| 128 | iid_gaussian    | WHT      | 0.033917 | 0.982 |
| 128 | iid_gaussian    | Haar     | 0.034010 | 0.984 |
| 128 | outlier_ch (10x)| WHT      | 0.030705 | 0.889 |
| 128 | outlier_ch (10x)| Haar     | 0.033500 | 0.970 |
| 128 | student_t(df=3) | WHT      | 0.033957 | 0.983 |
| 128 | student_t(df=3) | Haar     | 0.033919 | 0.982 |
| 256 | iid_gaussian    | WHT      | 0.034282 | 0.992 |
| 256 | iid_gaussian    | Haar     | 0.034239 | 0.991 |
| 256 | outlier_ch (10x)| WHT      | 0.031628 | 0.915 |
| 256 | outlier_ch (10x)| Haar     | 0.034977 | 1.012 |
| 256 | student_t(df=3) | WHT      | 0.034201 | 0.990 |
| 256 | student_t(df=3) | Haar     | 0.034288 | 0.992 |

## Phase B: Real Activations (Qwen2.5-3B-Instruct)

| Layer | K/V | Rotation | MSE      | Ratio |
|-------|-----|----------|----------|-------|
| 0     | K   | WHT      | 0.030092 | 0.871 |
| 0     | K   | Haar     | 0.031578 | 0.914 |
| 0     | V   | WHT      | 0.032526 | 0.941 |
| 0     | V   | Haar     | 0.033859 | 0.980 |
| 18    | K   | WHT      | 0.030401 | 0.880 |
| 18    | K   | Haar     | 0.034627 | 1.002 |
| 18    | V   | WHT      | 0.032858 | 0.951 |
| 18    | V   | Haar     | 0.033557 | 0.971 |
| 35    | K   | WHT      | 0.036357 | 1.052 |
| 35    | K   | Haar     | 0.033737 | 0.977 |
| 35    | V   | WHT      | 0.033818 | 0.979 |
| 35    | V   | Haar     | 0.034012 | 0.985 |

All ratios within 1.5x threshold. No re-solved Lloyd-Max needed.
