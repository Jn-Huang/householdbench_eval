"""Strict public-table schemas and canonical HouseholdBench metadata."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.utils.task_registry import TASK_REGISTRY_PATH, TABLE_SCHEMA_FIELDS, load_task_registry


METADATA_COLUMNS = ("id", "time", "release_date")
ISO_DATE_PATTERN = re.compile(r"^[0-9]{4}-(0[1-9]|1[0-2])-([0-2][0-9]|3[01])$")


@lru_cache(maxsize=1)
def load_table_schema(path: Path = TASK_REGISTRY_PATH) -> dict[str, dict[str, Any]]:
    """Return only ordered schema fields, keeping model definitions free of labels."""
    return {
        task_id: {field: specification[field] for field in TABLE_SCHEMA_FIELDS}
        for task_id, specification in load_task_registry(path).items()
    }


def public_table_columns(
    task_id: str, *, schema_path: Path = TASK_REGISTRY_PATH
) -> list[str]:
    """Return the exact ordered public columns for one task."""
    schema = load_table_schema(schema_path)
    if task_id not in schema:
        raise KeyError(f"Unknown HouseholdBench table schema: {task_id}")
    return [*METADATA_COLUMNS, *schema[task_id]["targets"], *schema[task_id]["prompt_predictors"]]


def canonical_public_metadata(frame: pd.DataFrame, *, task_id: str) -> pd.DataFrame:
    """Validate public keys already constructed by the task, before sampling."""
    require_columns(frame, list(METADATA_COLUMNS), label=task_id)
    metadata = frame.loc[:, list(METADATA_COLUMNS)].astype("string")
    for column in METADATA_COLUMNS:
        metadata[column] = metadata[column].str.strip()
        if metadata[column].isna().any() or metadata[column].eq("").any():
            raise ValueError(f"{task_id}: {column} contains missing or empty values.")
    for column in ("time", "release_date"):
        invalid = ~metadata[column].str.fullmatch(ISO_DATE_PATTERN)
        parsed = pd.to_datetime(metadata[column], format="%Y-%m-%d", errors="coerce")
        if invalid.any() or parsed.isna().any():
            raise ValueError(f"{task_id}: {column} must contain valid ISO calendar dates.")
    if metadata.duplicated(["id", "time"]).any():
        raise ValueError(f"{task_id}: duplicate (id, time) keys.")
    return metadata


def canonicalize_public_table(
    frame: pd.DataFrame,
    *,
    task_id: str,
    schema_path: Path = TASK_REGISTRY_PATH,
) -> pd.DataFrame:
    """Validate explicit task metadata and project to the ordered public columns."""
    columns = public_table_columns(task_id, schema_path=schema_path)
    require_columns(frame, columns, label=task_id)
    public = frame.loc[:, columns].copy()
    public[list(METADATA_COLUMNS)] = canonical_public_metadata(public, task_id=task_id)
    return public


def write_public_table(
    frame: pd.DataFrame,
    *,
    task_id: str,
    public_path: Path,
    schema_path: Path = TASK_REGISTRY_PATH,
) -> pd.DataFrame:
    """Validate and write the sole strict public table for one task."""
    if frame.empty:
        raise ValueError(f"{task_id}: cannot publish an empty task table.")
    public = canonicalize_public_table(
        frame.reset_index(drop=True),
        task_id=task_id,
        schema_path=schema_path,
    )
    public_path.parent.mkdir(parents=True, exist_ok=True)
    public.to_csv(public_path, index=False)
    recorded = pd.read_csv(public_path, nrows=0)
    expected = public_table_columns(task_id, schema_path=schema_path)
    if list(recorded.columns) != expected:
        raise RuntimeError(f"{task_id}: recorded strict table header differs from its schema.")
    return public


def validate_public_table_schema(
    frame: pd.DataFrame,
    *,
    task_id: str,
    schema_path: Path = TASK_REGISTRY_PATH,
) -> None:
    """Require exact public columns, canonical metadata, and a unique task-local key."""
    expected = public_table_columns(task_id, schema_path=schema_path)
    actual = list(frame.columns)
    if actual != expected:
        raise ValueError(f"{task_id}: unexpected public schema. Expected {expected}; got {actual}.")
    canonical_public_metadata(frame, task_id=task_id)


def require_columns(df: pd.DataFrame, columns: list[str], *, label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise RuntimeError(f"{label} is missing required columns: {missing}")


def read_public_table(
    path: Path,
    *,
    task_id: str,
    schema_path: Path = TASK_REGISTRY_PATH,
    float_precision: str | None = None,
) -> pd.DataFrame:
    """Read public CSVs with explicit types; preserve identifiers and reject invalid values.

    Metadata and targets are required. The schema lists unconditional required
    predictors; conditional history and survey-branch checks remain in renderers.
    """
    spec = load_table_schema(schema_path)[task_id]
    text_columns = [*METADATA_COLUMNS, *spec["categorical_predictors"]]
    if spec["response"]["family"] == "categorical":
        text_columns += spec["targets"]
    dtypes = {
        column: "string" if column in text_columns else "float64"
        for column in public_table_columns(task_id, schema_path=schema_path)
    }
    table = pd.read_csv(path, dtype=dtypes, low_memory=False, float_precision=float_precision)
    validate_public_table_schema(table, task_id=task_id, schema_path=schema_path)
    if table.empty:
        raise ValueError(f"{task_id}: public table is empty.")
    required = [*METADATA_COLUMNS, *spec["targets"], *spec["required_predictors"]]
    for column in required:
        if table[column].isna().any() or (
            column in text_columns and table[column].str.strip().eq("").any()
        ):
            raise ValueError(f"{task_id}: missing required values in {column}.")
    for column, dtype in dtypes.items():
        if dtype == "float64" and table[column].isin([float("inf"), float("-inf")]).any():
            raise ValueError(f"{task_id}: nonfinite values in {column}.")
    if spec["response"]["family"] == "categorical":
        for column, domain in spec["response"]["categories"].items():
            if not table[column].isin(domain).all():
                raise ValueError(f"{task_id}: invalid target categories in {column}.")
    return table
