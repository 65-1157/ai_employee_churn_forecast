"""
review_trigger.py

Decides whether a human needs to review this week's pipeline run.
- Objective triggers: a single, clear, citable cause (DQ FAIL, confirmed
  model regression, drift, schema coverage collapse) -- fires immediately.
- Ambiguous signals: no single clean cause, but sub-threshold symptoms
  accumulate -- produces a RANKED list of hypotheses to investigate,
  not a false verdict.
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path

OBJECTIVE_DRIFT_FIELDS_MIN = 2
SCHEMA_COVERAGE_DROP_THRESHOLD = 0.30
LABEL_RATE_Z_THRESHOLD = 3.0
AMBIGUOUS_CONCERN_THRESHOLD = 0.35


def check_objective_triggers(dq_history: list[dict], eval_history: list[dict]) -> list[dict]:
    triggers = []
    if not dq_history:
        return triggers

    latest_dq = dq_history[-1]

    if latest_dq["gate_status"] == "FAIL":
        triggers.append({
            "trigger": "dq_gate_fail",
            "severity": "high",
            "confidence": "objective",
            "evidence": f"Data quality gate FAILED this week (score {latest_dq['overall_score']}).",
            "recommended_action": "Halt scoring, inspect upstream data source before any model review.",
        })

    recent = dq_history[-3:]
    if len(recent) == 3 and all(r["gate_status"] == "WARN" for r in recent):
        triggers.append({
            "trigger": "dq_gate_warn_sustained",
            "severity": "medium",
            "confidence": "objective",
            "evidence": "Data quality has been WARN for 3 consecutive weeks.",
            "recommended_action": "Investigate the specific DQ dimension that's persistently weak.",
        })

    drifted_fields = [f for f, r in latest_dq.get("drift", {}).get("fields", {}).items() if r.get("drifted")]
    if len(drifted_fields) >= OBJECTIVE_DRIFT_FIELDS_MIN:
        triggers.append({
            "trigger": "drift_detected",
            "severity": "medium",
            "confidence": "objective",
            "evidence": f"{len(drifted_fields)} features drifted this week: {drifted_fields}.",
            "recommended_action": "Confirm whether this reflects a real behavior shift or a data pipeline change.",
        })

    if eval_history and eval_history[-1].get("significant_regression", False):
        triggers.append({
            "trigger": "significant_performance_drop",
            "severity": "high",
            "confidence": "objective",
            "evidence": "Challenger model's confidence interval does not overlap champion's -- confirmed regression.",
            "recommended_action": "Roll back to champion model weighting; investigate challenger's training data.",
        })

    return triggers


def compute_ambiguous_signals(dq_history: list[dict], eval_history: list[dict]) -> dict:
    signals = []

    scores = [r["overall_score"] for r in dq_history[-8:]]
    if len(scores) >= 4:
        trend = float(np.polyfit(range(len(scores)), scores, 1)[0])
        if -3 < trend < -0.5:
            signals.append({
                "hypothesis": "Slow, sub-threshold data quality erosion",
                "strength": min(abs(trend) / 3, 1.0),
                "evidence": f"DQ score trending down at {trend:.2f} pts/week over last {len(scores)} weeks.",
                "check_first": "Review completeness/validity sub-scores individually before assuming a global cause.",
                "effort": "low",
            })

    briers = [e.get("brier", {}).get("point") for e in eval_history[-6:] if e.get("brier")]
    if len(briers) >= 4 and briers[-1] - min(briers[:-1]) > 0.02:
        signals.append({
            "hypothesis": "Model still ranks correctly but probabilities are miscalibrating",
            "strength": 0.6,
            "evidence": f"Brier score rose from {min(briers[:-1]):.3f} to {briers[-1]:.3f} while AUC stayed stable.",
            "check_first": "Recalibrate (Platt scaling / isotonic) before assuming a full retrain is needed.",
            "effort": "low",
        })

    if eval_history:
        latest_eval = eval_history[-1]
        if not latest_eval.get("significant_regression", False):
            auc_gap = latest_eval.get("champion_auc_point", 0) - latest_eval.get("challenger_auc_point", 0)
            if 0.01 < auc_gap < 0.04:
                signals.append({
                    "hypothesis": "New model may be genuinely weaker, but sample size too small to confirm",
                    "strength": 0.4,
                    "evidence": f"AUC gap of {auc_gap:.3f} exists but confidence intervals still overlap.",
                    "check_first": "Let 2-3 more evaluation cycles accumulate before acting.",
                    "effort": "low (wait and re-check)",
                })

    signals.sort(key=lambda s: s["strength"], reverse=True)
    concern = round(sum(s["strength"] for s in signals) / max(len(signals), 1), 2) if signals else 0.0
    return {"composite_concern_score": concern, "n_signals": len(signals), "ranked_hypotheses": signals}


def run_review_assessment(dq_history: list[dict], eval_history: list[dict], week_num: int,
                           out_dir: str = "outputs/review_flags") -> dict:
    objective = check_objective_triggers(dq_history, eval_history)

    result = {"week": week_num, "run_at": datetime.utcnow().isoformat()}

    if objective:
        result.update({"review_needed": True, "review_type": "OBJECTIVE",
                        "triggers": objective, "ambiguous_analysis": None})
    else:
        ambiguous = compute_ambiguous_signals(dq_history, eval_history)
        needed = ambiguous["composite_concern_score"] > AMBIGUOUS_CONCERN_THRESHOLD
        result.update({
            "review_needed": needed,
            "review_type": "AMBIGUOUS" if needed else "NONE",
            "triggers": [],
            "ambiguous_analysis": ambiguous,
        })

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = f"{out_dir}/review_week{week_num}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def log_review_outcome(week_num: int, reviewer: str, finding: str, resolution: str,
                        was_actionable: bool, log_path: str = "logs/review_log.csv"):
    import pandas as pd

    entry = pd.DataFrame([{
        "week": week_num,
        "logged_at": datetime.utcnow().isoformat(),
        "reviewer": reviewer,
        "finding": finding,
        "resolution": resolution,
        "was_actionable": was_actionable,
    }])
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(log_path).exists():
        entry.to_csv(log_path, mode="a", header=False, index=False)
    else:
        entry.to_csv(log_path, index=False)
