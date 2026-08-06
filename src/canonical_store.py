"""
canonical_store.py

Append-only store for canonical-schema records. Nothing from an earlier
questionnaire version is ever discarded when the schema evolves -- it's
appended, tagged with its schema_version, and aligned to the same columns.
"""

import pandas as pd
from pathlib import Path

STORE_PATH = Path("data/canonical/canonical_dataset.parquet")


def append_to_canonical_store(new_batch: pd.DataFrame) -> pd.DataFrame:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if STORE_PATH.exists():
        existing = pd.read_parquet(STORE_PATH)
        combined = pd.concat([existing, new_batch], ignore_index=True)
        # De-dupe on employee + week, not employee overall -- we want history preserved.
        dedupe_cols = [c for c in ["EmployeeNumber", "snapshot_date"] if c in combined.columns]
        if dedupe_cols:
            combined = combined.drop_duplicates(subset=dedupe_cols, keep="last")
    else:
        combined = new_batch

    combined.to_parquet(STORE_PATH, index=False)
    return combined


def load_canonical_store() -> pd.DataFrame:
    return pd.read_parquet(STORE_PATH) if STORE_PATH.exists() else pd.DataFrame()
