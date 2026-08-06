"""
report_generator.py

Turns weekly scores + per-employee explanations into a readable markdown
report -- the artifact that makes model output feel like an insight rather
than a spreadsheet.
"""

import pandas as pd
from datetime import datetime


def generate_weekly_report(scored_df: pd.DataFrame, explanations: dict) -> str:
    date_str = datetime.utcnow().date().isoformat()
    high_risk = scored_df[scored_df["risk_tier"] == "High"].sort_values("risk_score", ascending=False)

    lines = [f"# Weekly Churn Risk Report — {date_str}\n"]
    lines.append(f"Total employees scored: {len(scored_df)}")
    lines.append(
        f"High risk: {len(high_risk)} | "
        f"Medium: {(scored_df['risk_tier'] == 'Medium').sum()} | "
        f"Low: {(scored_df['risk_tier'] == 'Low').sum()}\n"
    )

    lines.append("## Individuals flagged High Risk\n")
    for _, row in high_risk.iterrows():
        emp_id = row["EmployeeNumber"]
        factors = explanations.get(emp_id, [])
        lines.append(f"- **Employee {emp_id}** — score {row['risk_score']:.2f}")
        for f in factors:
            lines.append(f"    - {f}")

    return "\n".join(lines)


def save_report(report_text: str, week_num: int, out_dir: str = "outputs/reports") -> str:
    from pathlib import Path
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = f"{out_dir}/report_week{week_num}.md"
    with open(path, "w") as f:
        f.write(report_text)
    return path
