"""
source_router.py

Before invoking the schema-mapping/merge machinery, checks whether it's
actually needed this run:
  - one batch, one schema version  -> direct path, no merge overhead
  - multiple batches / schema versions -> full canonical merge pipeline
"""

import pandas as pd
from dataclasses import dataclass

from schema_registry import detect_schema_version
from schema_mapper import map_to_canonical, align_to_full_canonical_schema
from canonical_store import append_to_canonical_store


@dataclass
class RoutingDecision:
    mode: str
    sources_detected: list
    reason: str
    action_taken: str = ""
    used_merge: bool = False


def detect_input_sources(batches: list[pd.DataFrame]) -> list[str]:
    versions = set()
    for batch in batches:
        versions.add(detect_schema_version(batch.columns.tolist()))
    return sorted(versions)


def route_input(batches: list[pd.DataFrame], all_canonical_fields: list[str]) -> tuple[pd.DataFrame, RoutingDecision]:
    sources = detect_input_sources(batches)

    if len(batches) == 1 and len(sources) == 1:
        decision = RoutingDecision(
            mode="single_source",
            sources_detected=sources,
            reason="Only one input batch and one schema version detected this run.",
            action_taken="Passed through directly, no merge/reconciliation needed.",
            used_merge=False,
        )
        result_df = batches[0].copy()
        result_df["schema_version"] = sources[0]
        result_df["ingestion_mode"] = "single_source"
        return result_df, decision

    canonical_batches = []
    for batch in batches:
        version = detect_schema_version(batch.columns.tolist())
        canon = map_to_canonical(batch, version)
        canon = align_to_full_canonical_schema(canon, all_canonical_fields)
        canonical_batches.append(canon)

    combined_batch = pd.concat(canonical_batches, ignore_index=True)
    full_store = append_to_canonical_store(combined_batch)

    decision = RoutingDecision(
        mode="multi_source",
        sources_detected=sources,
        reason=f"Detected {len(batches)} batch(es) spanning {len(sources)} schema version(s): {sources}.",
        action_taken="Routed through schema_mapper + canonical_store merge pipeline.",
        used_merge=True,
    )
    return full_store, decision
