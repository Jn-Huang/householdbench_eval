"""Shared mechanics for Michigan task table builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from scripts.utils.macro_context import CORE_MACRO_COLUMNS, attach_macro_context


def iso_date_or_none(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def load_microdata(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Michigan microdata not found: {path}")
    df = pd.read_parquet(path, columns=columns)
    df["survey_date"] = pd.to_datetime(df["survey_date"], errors="coerce")
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    if df["release_date"].isna().any():
        raise RuntimeError("Michigan microdata contains missing or invalid release_date values.")
    if df["release_date_source"].isna().any() or df["release_date_source"].str.strip().eq("").any():
        raise RuntimeError("Michigan microdata contains missing release_date_source values.")
    return df.sort_values(["survey_date", "CASEID"], kind="mergesort").reset_index(drop=True)


def attach_macro_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["survey_year"] = out["survey_date"].dt.year.astype("Int64")
    out["survey_quarter"] = out["survey_date"].dt.quarter.astype("Int64")
    out["survey_q_index"] = out["survey_year"] * 4 + out["survey_quarter"]
    return attach_macro_context(
        out,
        origin_q_index_col="survey_q_index",
        required_columns=CORE_MACRO_COLUMNS,
    )


def validate_code_domains(
    df: pd.DataFrame,
    *,
    code_domains: dict[str, set[int]],
    columns: list[str],
    task_slug: str,
) -> None:
    for column in columns:
        allowed = code_domains[column]
        values = pd.to_numeric(df[column], errors="coerce")
        bad = values.isna() | ~values.astype("Int64").isin(allowed)
        if bad.any():
            examples = sorted(values.loc[bad].dropna().unique().tolist())[:10]
            raise RuntimeError(f"{task_slug}: {column} has missing or unmapped required codes: {examples}")
