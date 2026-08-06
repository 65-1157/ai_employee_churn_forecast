"""
data_quality.py

Data Quality radar + gate. Runs BEFORE scoring. Answers: "can we trust this
week's data enough to score employees on it?" Produces a 0-100 score per
dimension plus an overall PASS / WARN / FAIL gate.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from scipy.stats import ks_2samp

VALID_RANGES = {
    "Age": (18, 70),
    "MonthlyIncome": (1000, 50000),
    "JobSatisfaction": (1, 4),
    "EnvironmentSatisfaction": (1, 4),
    "WorkLifeBalance": (1, 4),
    "RelationshipSatisfaction": (1, 4),
    "YearsAtCompany": (0, 50),
}

DRIFT_FIELDS = ["JobSatisfaction", "MonthlyIncome", "YearsAtCompany", "EnvironmentSatisfaction"]


def check_completeness(df: pd.DataFrame) -> dict:
    null_pct = df.isnull().mean() * 100
    score = 100 - null_pct.mean()
    worst_fields = null_pct[null_pct > 0].sort_values(ascending=False).to_dict()
    return {"score": round(float(score), 1), "worst_fields": worst_fields}


def check_validity(df: pd.DataFrame) -> dict:
    violations = {}
    total_checks, total_violations = 0, 0
    for field, (low, high) in VALID_RANGES.items():
        if field not in df.columns:
            continue
        out_of_range = int(((df[field] < low) | (df[field] > high)).sum())
        total_checks += len(df)
        total_violations += out_of_range
        if out_of_range > 0:
            violations[field] = out_of_range
    score = 100 * (1 - total_violations / max(total_checks, 1))
    return {"score": round(float(score), 1), "violations": violations}


def check_consistency(df: pd.DataFrame) -> dict:
    issues = {}
    if {"YearsAtCompany", "TotalWorkingYears"}.issubset(df.columns):
        bad = int((df["YearsAtCompany"] > df["TotalWorkingYears"]).sum())
        if bad > 0:
            issues["YearsAtCompany_gt_TotalWorkingYears"] = bad
    if {"YearsSinceLastPromotion", "YearsAtCompany"}.issubset(df.columns):
        bad = int((df["YearsSinceLastPromotion"] > df["YearsAtCompany"]).sum())
        if bad > 0:
            issues["PromotionYears_gt_TenureYears"] = bad
    score = 100 - (sum(issues.values()) / len(df) * 100 if len(df) else 0)
    return {"score": round(max(float(score), 0), 1), "issues": issues}


def check_uniqueness(df: pd.DataFrame) -> dict:
    if "EmployeeNumber" not in df.columns:
        return {"score": 0.0, "duplicates": "EmployeeNumber column missing"}
    dupes = int(df["EmployeeNumber"].duplicated().sum())
    score = 100 * (1 - dupes / max(len(df), 1))
    return {"score": round(float(score), 1), "duplicate_count": dupes}


def check_freshness(df: pd.DataFrame, expected_date: str) -> dict:
    if "snapshot_date" not in df.columns:
        return {"score": 0.0, "detail": "no snapshot_date field"}
    actual_dates = df["snapshot_date"].unique().tolist()
    on_time = expected_date in actual_dates
    return {"score": 100.0 if on_time else 0.0, "expected": expected_date, "found": actual_dates}


def check_drift(df: pd.DataFrame, baseline: dict | None) -> dict:
    if not baseline:
        return {"score": 100.0, "detail": "no baseline yet -- first run", "fields": {}}

    drift_results, scores = {}, []
    for field in DRIFT_FIELDS:
        if field not in df.columns or field not in baseline:
            continue
        baseline_vals = np.array(baseline[field])
        current_vals = df[field].dropna().values
        if len(current_vals) < 5:
            continue
        stat, p_value = ks_2samp(baseline_vals, current_vals)
        drifted = p_value < 0.05
        drift_results[field] = {
            "ks_stat": round(float(stat), 3),
            "p_value": round(float(p_value), 4),
            "drifted": bool(drifted),
        }
        scores.append(0 if drifted else 100)

    overall = float(np.mean(scores)) if scores else 100.0
    return {"score": round(overall, 1), "fields": drift_results}


def run_data_quality_checks(
    df: pd.DataFrame,
    week_num: int,
    expected_date: str,
    baseline: dict | None = None,
) -> dict:
    results = {
        "week": week_num,
        "run_at": datetime.utcnow().isoformat(),
        "n_records": len(df),
        "completeness": check_completeness(df),
        "validity": check_validity(df),
        "consistency": check_consistency(df),
        "uniqueness": check_uniqueness(df),
        "freshness": check_freshness(df, expected_date),
        "drift": check_drift(df, baseline),
    }

    dims = ["completeness", "validity", "consistency", "uniqueness", "freshness", "drift"]
    overall_score = float(np.mean([results[d]["score"] for d in dims]))
    results["overall_score"] = round(overall_score, 1)

    if overall_score >= 90:
        results["gate_status"] = "PASS"
    elif overall_score >= 70:
        results["gate_status"] = "WARN"
    else:
        results["gate_status"] = "FAIL"

    return results


def build_baseline(df: pd.DataFrame) -> dict:
    """Snapshot of 'normal' distributions, used for drift comparison in later weeks."""
    return {field: df[field].dropna().tolist() for field in DRIFT_FIELDS if field in df.columns}


def save_report(results: dict, week_num: int, out_dir: str = "outputs/data_quality") -> str:
    path = f"{out_dir}/dq_report_week{week_num}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path
