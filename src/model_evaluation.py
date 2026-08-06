"""
model_evaluation.py

Evaluates candidate models on a shared holdout set -- bootstrapped confidence
intervals, not just a point estimate -- so weighting decisions in ensemble.py
are grounded in uncertainty, not noise.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss


def bootstrap_metric(y_true, y_pred, metric_fn, n_boot: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_t, y_p = y_true[idx], y_pred[idx]
        if len(np.unique(y_t)) < 2:
            continue
        scores.append(metric_fn(y_t, y_p))

    if not scores:
        point = float(metric_fn(y_true, y_pred))
        return {"point": point, "ci_low": point, "ci_high": point, "std": 0.0}

    return {
        "point": float(metric_fn(y_true, y_pred)),
        "ci_low": float(np.percentile(scores, 2.5)),
        "ci_high": float(np.percentile(scores, 97.5)),
        "std": float(np.std(scores)),
    }


def evaluate_model_on_shared_holdout(y_true, y_pred_proba, fn_cost_multiplier: float = 2.0) -> dict:
    """
    Cost-aware + calibration-aware evaluation, not just AUC. Run every candidate
    model on the SAME holdout set so comparisons are fair.
    """
    auc = bootstrap_metric(y_true, y_pred_proba, roc_auc_score)
    brier = bootstrap_metric(y_true, y_pred_proba, brier_score_loss)

    sample_weight = np.where(np.array(y_true) == 1, fn_cost_multiplier, 1.0)
    weighted_ll = log_loss(y_true, y_pred_proba, sample_weight=sample_weight)

    return {"auc": auc, "brier": brier, "weighted_log_loss": weighted_ll}
