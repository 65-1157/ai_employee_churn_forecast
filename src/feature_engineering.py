"""
feature_engineering.py

Builds interpretable, individual-level features from the IBM HR dataset
(or its canonical-schema equivalent, once the schema module is wired in).
Kept deliberately simple and explainable -- HR stakeholders need to trust
*why* an employee is flagged, not just trust a black-box score.
"""

import pandas as pd

CATEGORICAL = ["Department", "JobRole", "MaritalStatus", "BusinessTravel", "Gender", "EducationField"]

NUMERIC = [
    "Age", "MonthlyIncome", "DistanceFromHome", "YearsAtCompany",
    "YearsSinceLastPromotion", "TotalWorkingYears", "NumCompaniesWorked",
    "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance",
    "RelationshipSatisfaction", "JobInvolvement", "StockOptionLevel",
    "PercentSalaryHike", "TrainingTimesLastYear", "OverTime_Flag",
]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns (dataframe_with_features, list_of_feature_column_names).
    Safe to call on a full dataset or a single-employee slice.
    """
    df = df.copy()

    if "OverTime_Flag" not in df.columns and "OverTime" in df.columns:
        df["OverTime_Flag"] = (df["OverTime"] == "Yes").astype(int)

    # Engineered ratios -- these tend to carry more signal than raw counts
    df["TenureRatio"] = df["YearsAtCompany"] / df["TotalWorkingYears"].replace(0, 1)
    df["PromotionStagnation"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
    df["IncomePerYearWorked"] = df["MonthlyIncome"] / df["TotalWorkingYears"].replace(0, 1)

    engineered = ["TenureRatio", "PromotionStagnation", "IncomePerYearWorked"]

    present_categorical = [c for c in CATEGORICAL if c in df.columns]
    df = pd.get_dummies(df, columns=present_categorical, drop_first=True)

    present_numeric = [c for c in NUMERIC if c in df.columns]
    dummy_cols = [
        c for c in df.columns
        if any(c.startswith(cat + "_") for cat in present_categorical)
    ]

    feature_cols = present_numeric + engineered + dummy_cols
    return df, feature_cols


if __name__ == "__main__":
    from data_loader import load_raw_data
    raw = load_raw_data()
    featured, cols = build_features(raw)
    print(f"Built {len(cols)} features from {len(raw)} records.")
    print(cols[:10], "...")
