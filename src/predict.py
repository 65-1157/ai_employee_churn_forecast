"""
predict.py

Scores each individual employee for the current weekly snapshot, using the
model bundle saved by train_model.py ({"model": ..., "feature_cols": ...}).
"""

import pandas as pd
import joblib

from feature_engineering import build_features


def score_employees(snapshot_df: pd.DataFrame, model_path: str = "models/churn_model.pkl") -> pd.DataFrame:
    bundle = joblib.load(model_path)
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    df, _ = build_features(snapshot_df)
    X = df.reindex(columns=feature_cols, fill_value=0)

    df["risk_score"] = model.predict_proba(X)[:, 1]
    df["risk_tier"] = pd.cut(
        df["risk_score"], bins=[0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"]
    )

    keep_cols = ["EmployeeNumber", "risk_score", "risk_tier"]
    extra_cols = [c for c in snapshot_df.columns if c not in keep_cols and c in df.columns]
    return df[keep_cols + extra_cols]


if __name__ == "__main__":
    from data_loader import load_raw_data, simulate_weekly_snapshot
    from datetime import datetime

    week = datetime.utcnow().isocalendar()[1]
    raw = load_raw_data()
    snap = simulate_weekly_snapshot(raw, week_seed=week)
    scored = score_employees(snap)
    print(scored.sort_values("risk_score", ascending=False).head(10))
