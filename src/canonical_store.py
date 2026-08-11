"""
canonical_store.py

Append-only store for canonical-schema records. Nothing from an earlier
questionnaire version is ever discarded when the schema evolves -- it's
appended, tagged with its schema_version, and aligned to the same columns.
"""

import pandas as pd
from pathlib import Path

DEFAULT_STORE_PATH = Path("data/canonical/canonical_dataset.parquet")


def append_to_canonical_store(new_batch: pd.DataFrame, store_path: Path = DEFAULT_STORE_PATH) -> pd.DataFrame:
    """
    store_path defaults to the core churn-feature canonical store (unchanged
    behavior for all existing callers). Pass a different path to route a
    batch into a SEPARATE store instead -- e.g. quarantining a questionnaire
    wave that shouldn't be merged into the churn model's feature set.
    """
    store_path = Path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    if store_path.exists():
        existing = pd.read_parquet(store_path)
        combined = pd.concat([existing, new_batch], ignore_index=True)
        # De-dupe on employee + week, not employee overall -- we want history preserved.
        dedupe_cols = [c for c in ["EmployeeNumber", "snapshot_date"] if c in combined.columns]
        if dedupe_cols:
            combined = combined.drop_duplicates(subset=dedupe_cols, keep="last")
    else:
        combined = new_batch

    combined.to_parquet(store_path, index=False)
    return combined


def load_canonical_store(store_path: Path = DEFAULT_STORE_PATH) -> pd.DataFrame:
    store_path = Path(store_path)
    return pd.read_parquet(store_path) if store_path.exists() else pd.DataFrame()
