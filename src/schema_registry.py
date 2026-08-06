"""
schema_registry.py

Declares each questionnaire version explicitly (schema_registry/schema_*.json)
and detects which version an incoming batch's columns best match.
"""

import json
from pathlib import Path

REGISTRY_DIR = Path("schema_registry")


def load_schema(version: str) -> dict:
    path = REGISTRY_DIR / f"schema_{version}.json"
    with open(path) as f:
        return json.load(f)


def list_known_versions() -> list[str]:
    return sorted(p.stem.replace("schema_", "") for p in REGISTRY_DIR.glob("schema_*.json"))


def detect_schema_version(df_columns: list[str]) -> str:
    """
    Matches incoming columns against known schema versions by field-name overlap.
    Picks the version with the highest overlap. In production this could instead
    read an explicit version tag sent alongside the data.
    """
    best_version, best_score = None, -1
    for version in list_known_versions():
        schema = load_schema(version)
        expected = set(schema["fields"].keys())
        score = len(expected.intersection(df_columns))
        if score > best_score:
            best_version, best_score = version, score
    return best_version


if __name__ == "__main__":
    print("Known schema versions:", list_known_versions())
