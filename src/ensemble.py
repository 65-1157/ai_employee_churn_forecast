"""
ensemble.py

Combines predictions from multiple models (e.g. an existing champion and a
newly retrained challenger) using inverse-variance weighting, with a
statistical significance gate and a minimum weight floor -- so a noisy new
model is discounted proportionally to its actual uncertainty, not just
zeroed out after one bad evaluation cycle.
"""

import json
import numpy as np
from pathlib import Path

WEIGHTS_PATH = Path("models/registry/ensemble_weights.json")


def compute_inverse_variance_weights(model_evals: dict) -> dict:
    raw_weights = {}
    for model_name, evals in model_evals.items():
        variance = evals["auc"]["std"] ** 2
        raw_weights[model_name] = 1.0 / max(variance, 1e-6)

    total = sum(raw_weights.values())
    return {k: v / total for k, v in raw_weights.items()}


def apply_minimum_weight_floor(weights: dict, floor: float = 0.10) -> dict:
    adjusted = {k: max(v, floor) for k, v in weights.items()}
    total = sum(adjusted.values())
    return {k: v / total for k, v in adjusted.items()}


def significance_gate(model_evals: dict, challenger: str, champion: str, margin: float = 0.0) -> bool:
    """True only if the challenger's CI doesn't overlap the champion's -- a confirmed
    improvement, not just a point-estimate difference that could be noise."""
    c_auc = model_evals[challenger]["auc"]
    champ_auc = model_evals[champion]["auc"]
    return c_auc["ci_low"] > (champ_auc["ci_high"] - margin)


def combine_predictions(predictions: dict, weights: dict) -> np.ndarray:
    combined = np.zeros_like(next(iter(predictions.values())), dtype=float)
    for model_name, preds in predictions.items():
        combined += weights.get(model_name, 0) * np.array(preds)
    return combined


def save_weights(weights: dict, meta: dict | None = None):
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_PATH, "w") as f:
        json.dump({"weights": weights, "meta": meta or {}}, f, indent=2)


def load_weights() -> dict:
    if not WEIGHTS_PATH.exists():
        return {}
    with open(WEIGHTS_PATH) as f:
        return json.load(f)
