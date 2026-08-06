"""
explain.py

Per-individual explainability. Given a single employee's raw record, returns
the top N factors driving their risk score, in a plain-language format
suitable for a non-technical HR reader.
"""

import joblib
import pandas as pd

from feature_engineering import build_features

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def explain_employee(employee_row: pd.DataFrame, model_path: str = "models/churn_model.pkl",
                      top_n: int = 3) -> list[str]:
    """
    employee_row: a single-row (or small) DataFrame slice from the raw/snapshot data,
    e.g. snapshot_df[snapshot_df["EmployeeNumber"] == some_id]
    """
    bundle = joblib.load(model_path)
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    df, _ = build_features(employee_row)
    X = df.reindex(columns=feature_cols, fill_value=0)

    if not HAS_SHAP:
        # Fallback: global feature importance, same for every employee -- less useful,
        # but keeps the pipeline running if shap isn't installed.
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        top = importances.sort_values(ascending=False).head(top_n)
        return [f"{feat} (global importance {val:.2f})" for feat, val in top.items()]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # GradientBoostingClassifier -> shap_values is a single array (not a list per class)
    row_values = shap_values[0] if hasattr(shap_values, "__len__") and len(shap_values.shape) == 2 else shap_values

    contrib = pd.Series(row_values, index=feature_cols).sort_values(key=abs, ascending=False)
    top_factors = contrib.head(top_n)
    return [
        f"{feat} ({'+' if val > 0 else '-'}{abs(val):.2f} impact)"
        for feat, val in top_factors.items()
    ]


if __name__ == "__main__":
    from data_loader import load_raw_data, simulate_weekly_snapshot
    from datetime import datetime

    week = datetime.utcnow().isocalendar()[1]
    raw = load_raw_data()
    snap = simulate_weekly_snapshot(raw, week_seed=week)
    example = snap.iloc[[0]]
    print(f"Employee {example['EmployeeNumber'].values[0]}:")
    for factor in explain_employee(example):
        print(" -", factor)
