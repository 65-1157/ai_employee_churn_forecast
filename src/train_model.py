"""
train_model.py

Trains the baseline champion model. Not run every week -- scoring runs
weekly (see predict.py, to be added next), but retraining runs only when
enough new labeled data has accumulated or a review flag calls for it.
"""

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

from data_loader import load_raw_data
from feature_engineering import build_features

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False


def train(model_out_path: str = "models/churn_model.pkl") -> dict:
    df = load_raw_data()
    df["OverTime_Flag"] = (df["OverTime"] == "Yes").astype(int)
    df, feature_cols = build_features(df)

    X = df[feature_cols].fillna(0)
    y = (df["Attrition"] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    if HAS_SMOTE:
        X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
    else:
        X_res, y_res = X_train, y_train

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_res, y_res)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)

    report = classification_report(y_test, preds, output_dict=True)
    auc = roc_auc_score(y_test, probs)

    print(classification_report(y_test, preds))
    print(f"ROC-AUC: {auc:.3f}")

    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_out_path)

    return {"auc": auc, "report": report, "feature_cols": feature_cols}


if __name__ == "__main__":
    train()
