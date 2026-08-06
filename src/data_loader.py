"""
data_loader.py

Loads the base IBM HR Attrition dataset and simulates a weekly snapshot.
Designed to run identically on Colab or any other environment -- the
dataset is pulled from a public GitHub-hosted CSV mirror, so no Kaggle
authentication is required.

In production, `load_raw_data()` is the function that gets replaced with
a real pull from the client's HRIS / questionnaire export.
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Public mirror of the IBM HR Analytics Attrition & Performance dataset.
# Swap this for the client's real data source when available.
DATA_URL = (
    "https://raw.githubusercontent.com/nelson-wu/employee-attrition-ml/"
    "master/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)
LOCAL_RAW_PATH = "data/raw/ibm_hr_attrition.csv"


def load_raw_data(use_local: bool = True) -> pd.DataFrame:
    """
    Loads the base dataset. Tries local file first (faster, avoids repeated
    downloads on Colab reruns); falls back to the GitHub mirror if not found.
    """
    if use_local:
        try:
            df = pd.read_csv(LOCAL_RAW_PATH)
            df.columns = [c.strip() for c in df.columns]
            return df
        except FileNotFoundError:
            pass

    df = pd.read_csv(DATA_URL)
    df.columns = [c.strip() for c in df.columns]
    return df


def simulate_weekly_snapshot(df: pd.DataFrame, week_seed: int) -> pd.DataFrame:
    """
    MVP-only: the IBM dataset is static (one-time snapshot), so there is no
    real time dimension to it. To demonstrate the weekly pipeline behavior,
    we deterministically nudge a few volatile fields (satisfaction, overtime)
    based on the week number, simulating natural week-to-week drift.

    This function is the one to delete/replace once a real weekly data feed
    exists -- everything downstream (features, DQ checks, scoring) should
    keep working unchanged.
    """
    rng = np.random.default_rng(week_seed)
    snap = df.copy()

    # Nudge satisfaction-type fields by -1/0/+1, staying within valid 1-4 range
    for col in ["JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance"]:
        if col in snap.columns:
            noise = rng.integers(-1, 2, size=len(snap))
            snap[col] = (snap[col] + noise).clip(1, 4)

    snap["OverTime_Flag"] = (snap["OverTime"] == "Yes").astype(int)
    snap["snapshot_date"] = datetime.utcnow().date().isoformat()
    snap["week_number"] = week_seed

    return snap


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {len(df)} employee records, {df.shape[1]} columns.")
    print(df[["EmployeeNumber", "Age", "Attrition", "JobSatisfaction"]].head())
