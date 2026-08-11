"""
schema_mapper.py

Translates a raw batch (in whichever schema version it arrived in) into the
stable canonical schema, so old and new questionnaire structures can be
merged without discarding history.
"""

import pandas as pd
import json

MAPPING_RULES_PATH = "schema_registry/mapping_rules.json"

def _ordinal_map(mapping: dict):
    """Factory for a transform that looks up each raw categorical answer in
    an explicit ordering dict -- used for the AI-questionnaire's multi-choice
    fields, which aren't simple 1-4 numeric scales like the original schema."""
    return lambda x: x.map(mapping)

TRANSFORMS = {
    "scale_1_4_to_0_1": lambda x: (pd.to_numeric(x, errors="coerce") - 1) / 3,
    "scale_1_5_to_0_1": lambda x: (pd.to_numeric(x, errors="coerce") - 1) / 4,
    "yesno_to_bool": lambda x: (x == "Yes").astype(int),
    "hours_to_bool_threshold_10": lambda x: (pd.to_numeric(x, errors="coerce") > 10).astype(int),
    # AI-questionnaire ordinal transforms (schema_v3) -- each maps that
    # question's specific answer options to an explicit 0/0.5/1 ordering.
    "remote_capable_to_0_1": _ordinal_map({"Not remote-capable": 0.0, "Hybrid": 0.5, "Fully remote-capable": 1.0}),
    "ai_usage_freq_to_0_1": _ordinal_map({"Never/Rarely": 0.0, "Frequently": 0.5, "Daily": 1.0}),
    "ai_reliance_to_0_1": _ordinal_map({"Autonomous, no reliance": 0.0, "Somewhat, depends on task": 0.5, "Heavily reliant": 1.0}),
    "three_level_yes_maybe_no_to_0_1": _ordinal_map({"No": 0.0, "Maybe": 0.5, "Yes": 1.0}),
    "three_level_never_maybe_yes_to_0_1": _ordinal_map({"Never": 0.0, "Maybe, it depends": 0.5, "Yes, definitely": 1.0}),
    "three_level_never_sometimes_yes_to_0_1": _ordinal_map({"Never": 0.0, "Sometimes": 0.5, "Yes": 1.0}),
    "impact_human_connections_to_0_1": _ordinal_map({"No, not at all": 0.0, "Somewhat": 0.5, "Yes, significantly": 1.0}),
    # "Not applicable" is intentionally left OUT of this mapping -- pandas'
    # .map() returns NaN for values not present in the dict, which correctly
    # treats it as missing rather than forcing it into a false ordinal midpoint.
    "change_passivity_to_0_1": _ordinal_map({"No, I keep updating myself": 0.0, "Yes, easier to stay passive": 1.0}),
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
