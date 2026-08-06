"""
schema_mapper.py

Translates a raw batch (in whichever schema version it arrived in) into the
stable canonical schema, so old and new questionnaire structures can be
merged without discarding history.
"""

import pandas as pd
import json

MAPPING_RULES_PATH = "schema_registry/mapping_rules.json"

TRANSFORMS = {
    "scale_1_4_to_0_1": lambda x: (pd.to_numeric(x, errors="coerce") - 1) / 3,
    "scale_1_5_to_0_1": lambda x: (pd.to_numeric(x, errors="coerce") - 1) / 4,
    "yesno_to_bool": lambda x: (x == "Yes").astype(int),
    "hours_to_bool_threshold_10": lambda x: (pd.to_numeric(x, errors="coerce") > 10).astype(int),
}


def load_mapping_rules() -> dict:
    with open(MAPPING_RULES_PATH) as f:
        return json.load(f)


def map_to_canonical(df: pd.DataFrame, schema_version: str) -> pd.DataFrame:
    rules = load_mapping_rules()[f"{schema_version}_to_canonical"]
    canonical = pd.DataFrame(index=df.index)

    for raw_field, rule in rules.items():
        if raw_field not in df.columns:
            continue
        transform_fn = TRANSFORMS[rule["transform"]]
        canonical[rule["canonical"]] = transform_fn(df[raw_field])

    canonical["schema_version"] = schema_version
    canonical["EmployeeNumber"] = df.get("EmployeeNumber", df.index)
    if "snapshot_date" in df.columns:
        canonical["snapshot_date"] = df["snapshot_date"]
    return canonical


def align_to_full_canonical_schema(canonical_df: pd.DataFrame, all_canonical_fields: list[str]) -> pd.DataFrame:
    """
    Ensures every batch has the same columns, regardless of which schema version
    it came from. Missing fields become NaN, with a companion '_was_collected'
    flag so downstream consumers can tell "not asked yet" from "answered 0".
    """
    df = canonical_df.copy()
    for field in all_canonical_fields:
        was_collected_col = f"{field}_was_collected"
        if field not in df.columns:
            df[field] = pd.NA
            df[was_collected_col] = 0
        else:
            df[was_collected_col] = df[field].notna().astype(int)
    return df


if __name__ == "__main__":
    print("Mapping rules:", list(load_mapping_rules().keys()))
