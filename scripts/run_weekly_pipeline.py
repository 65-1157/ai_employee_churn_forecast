"""
run_weekly_pipeline.py

End-to-end weekly orchestrator. Order mirrors the pipeline diagram:

  ingest -> source routing -> DQ gate -> features -> score -> explain ->
  report -> review trigger

Run manually (`python scripts/run_weekly_pipeline.py`) or via
.github/workflows/weekly_pipeline.yml on a schedule.
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import load_raw_data, simulate_weekly_snapshot
from source_router import route_input
from data_quality import run_data_quality_checks, build_baseline, save_report as save_dq_report
from feature_engineering import build_features
from train_model import train
from predict import score_employees
from explain import explain_employee
from report_generator import generate_weekly_report, save_report as save_weekly_report
from review_trigger import run_review_assessment

MODEL_PATH = "models/churn_model.pkl"
ALL_CANONICAL_FIELDS = [
    "job_satisfaction_norm", "overtime_flag", "work_life_balance_norm",
    "environment_satisfaction_norm", "relationship_satisfaction_norm", "manager_support_norm",
]


def _load_history(out_dir: str, prefix: str) -> list[dict]:
    """Loads all prior weekly JSON reports from a directory, sorted by week number."""
    history = []
    for path in sorted(Path(out_dir).glob(f"{prefix}week*.json")):
        with open(path) as f:
            history.append(json.load(f))
    return history


def main():
    week_num = datetime.utcnow().isocalendar()[1]
    expected_date = datetime.utcnow().date().isoformat()
    print(f"=== Weekly pipeline run — week {week_num} ===")

    # --- 1. Ingest + weekly snapshot ---
    raw_df = load_raw_data()
    snapshot_df = simulate_weekly_snapshot(raw_df, week_seed=week_num)

    # --- 2. Source routing (single- vs multi-source) ---
    incoming_batches = [snapshot_df]  # single source for this MVP; extend when a 2nd feed exists
    routed_df, routing_decision = route_input(incoming_batches, ALL_CANONICAL_FIELDS)
    Path("outputs/routing").mkdir(parents=True, exist_ok=True)
    with open(f"outputs/routing/routing_week{week_num}.json", "w") as f:
        json.dump(routing_decision.__dict__, f, indent=2)
    print(f"Routing mode: {routing_decision.mode} ({routing_decision.reason})")

    # --- 3. Data Quality gate ---
    baseline = build_baseline(raw_df)
    dq_results = run_data_quality_checks(snapshot_df, week_num, expected_date, baseline)
    save_dq_report(dq_results, week_num)
    print(f"Data Quality: {dq_results['overall_score']} ({dq_results['gate_status']})")

    if dq_results["gate_status"] == "FAIL":
        print("Data quality FAILED. Halting pipeline before scoring.")
        return
    elif dq_results["gate_status"] == "WARN":
        print("Data quality WARNING — proceeding, but flag this run for review.")

    # --- 4. Train (only if no model exists yet -- otherwise this is a retrain trigger decision) ---
    if not Path(MODEL_PATH).exists():
        print("No existing model found -- training baseline model.")
        train(model_out_path=MODEL_PATH)

    # --- 5. Score individuals ---
    scored = score_employees(snapshot_df, model_path=MODEL_PATH)
    Path("outputs/scores").mkdir(parents=True, exist_ok=True)
    scored.to_csv(f"outputs/scores/scores_week{week_num}.csv", index=False)

    high_risk = scored[scored["risk_tier"] == "High"]
    print(f"Scored {len(scored)} employees — {len(high_risk)} flagged High risk.")

    # --- 6. Explain flagged individuals ---
    explanations = {}
    for _, row in high_risk.iterrows():
        emp_id = row["EmployeeNumber"]
        employee_row = snapshot_df[snapshot_df["EmployeeNumber"] == emp_id]
        try:
            explanations[emp_id] = explain_employee(employee_row, model_path=MODEL_PATH)
        except Exception as e:
            explanations[emp_id] = [f"(explanation unavailable: {e})"]

    # --- 7. Report ---
    report_text = generate_weekly_report(scored, explanations)
    save_weekly_report(report_text, week_num)

    # --- 8. Review trigger ---
    dq_history = _load_history("outputs/data_quality", "dq_report_")
    eval_history = []  # populated once model_evaluation.py + ensemble.py are wired into a retrain step
    review_result = run_review_assessment(dq_history, eval_history, week_num)
    print(f"Review status: {review_result['review_type']} (needed: {review_result['review_needed']})")

    print(f"=== Week {week_num} pipeline complete ===")


if __name__ == "__main__":
    main()
