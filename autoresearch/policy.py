"""
TurboQuant autoresearch: Quantization policy.

THIS IS THE FILE THE AGENT MODIFIES.
Everything here is fair game: bit-widths, QJL settings, per-layer overrides.
The evaluation harness (evaluate.py) is READ-ONLY.

The goal: maximize the composite score = accuracy * compression_ratio.
Higher is better. Accuracy must stay above 0.95 cosine similarity.
"""

# ---------------------------------------------------------------------------
# Global defaults (apply to all layers unless overridden)
# ---------------------------------------------------------------------------

KEY_B_MSE = 3           # MSE bits for keys (1-4). Lower = more compression, less accuracy.
VAL_B_MSE = 1           # MSE bits for values (1-4). Values tolerate lower bits (softmax averaging).
KEY_QJL_ENABLED = False # Whether to use 1-bit QJL correction on keys.
QJL_DIM_RATIO = 0.25    # QJL projection dim as fraction of head_dim (0.25, 0.5, 0.75, 1.0).

# ---------------------------------------------------------------------------
# Per-layer overrides (sparse — only specify layers that differ from default)
# Format: {layer_index: {"key_b_mse": N, "val_b_mse": N, ...}}
# Qwen2.5-3B has 28 layers (indices 0-27).
#
# Example: give more bits to early and final layers (they matter more for
# attention patterns), compress middle layers more aggressively:
#
#   LAYER_OVERRIDES = {
#       0: {"key_b_mse": 3},
#       1: {"key_b_mse": 3},
#       26: {"key_b_mse": 3},
#       27: {"key_b_mse": 3},
#   }
# ---------------------------------------------------------------------------

LAYER_OVERRIDES = {}

# ---------------------------------------------------------------------------
# Evaluation settings (agent can tune these too)
# ---------------------------------------------------------------------------

SEQ_LENGTHS = [512]  # Sequence lengths to test. More = slower but more robust.
NUM_HEADS = 8              # Number of KV heads to simulate.
HEAD_DIM = 128             # Head dimension (must match model).
N_QUERIES = 2              # Number of query vectors per test.
SEED = 42


# ---------------------------------------------------------------------------
# Helper: resolve per-layer config
# ---------------------------------------------------------------------------

def get_layer_config(layer_idx: int) -> dict:
    """Get the effective config for a specific layer."""
    config = {
        "key_b_mse": KEY_B_MSE,
        "val_b_mse": VAL_B_MSE,
        "key_qjl_enabled": KEY_QJL_ENABLED,
        "qjl_dim_ratio": QJL_DIM_RATIO,
    }
    if layer_idx in LAYER_OVERRIDES:
        config.update(LAYER_OVERRIDES[layer_idx])
    return config
