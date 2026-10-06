"""Shared mechanics for PSID task table builders."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd


PSID_WAVES = list(range(1968, 1998)) + list(range(1999, 2022, 2))
WAVE_SET = set(PSID_WAVES)
WAVE_ORDER = {year: idx for idx, year in enumerate(PSID_WAVES)}
NEXT_WAVE = {PSID_WAVES[idx]: PSID_WAVES[idx + 1] for idx in range(len(PSID_WAVES) - 1)}
PREV_WAVE = {PSID_WAVES[idx]: PSID_WAVES[idx - 1] for idx in range(1, len(PSID_WAVES))}
PSID_WEALTH_HISTORY_VALUE_SPECS = [
    ("finc_tot_nd", "Household income", "dollar"),
    ("wlth_tot_net_nd", "Total net worth", "dollar"),
    ("wlth_savi_net_nd", "Savings", "dollar"),
    ("wlth_inve_net_nd", "Financial investments", "dollar"),
    ("wlth_home_net_nd", "Home equity", "dollar"),
    ("wlth_odeb_net_nd", "Other debt", "dollar"),
    ("home_stat", "Housing status", "label"),
]

def safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def safe_int(value: Any) -> int | None:
    out = safe_float(value)
    if out is None:
        return None
    rounded = int(round(out))
    if abs(out - rounded) > 1e-6:
        return None
    return rounded


def is_missing(value: Any) -> bool:
    return value is None or pd.isna(value)


def iso_date(value: Any) -> str:
    if is_missing(value):
        raise RuntimeError(f"Invalid date value: {value!r}")
    return pd.Timestamp(value).date().isoformat()


def load_family_panel(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"PSID input not found: {path}")
    family = pd.read_parquet(
        path,
        columns=columns,
        filters=[("response", "=", 0), ("is_reference_person", "=", True)],
    )
    family = family.loc[family["year"].isin(WAVE_SET)].copy()
    family["id"] = pd.to_numeric(family["id"], errors="coerce").astype("Int64")
    family["year"] = pd.to_numeric(family["year"], errors="coerce").astype("Int64")
    if family["id"].isna().any() or family["year"].isna().any():
        raise RuntimeError("Family panel contains missing id/year after filtering.")
    family["release_date"] = pd.to_datetime(family["release_date"], errors="coerce")
    if family["release_date"].isna().any():
        raise RuntimeError("Family panel contains missing release_date values.")
    if family["release_date_source"].isna().any() or family["release_date_source"].str.strip().eq("").any():
        raise RuntimeError("Family panel contains missing release_date_source values.")
    family["wave_index"] = family["year"].map(WAVE_ORDER)
    if family["wave_index"].isna().any():
        raise RuntimeError("Family panel includes years outside the expected PSID wave schedule.")
    return family.sort_values(["id", "wave_index"], kind="mergesort").reset_index(drop=True)


def load_person_panel(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"PSID input not found: {path}")
    person = pd.read_parquet(
        path,
        columns=columns,
        filters=[("response", "=", 0), ("sample_person", "=", True)],
    )
    person = person.loc[person["year"].isin(WAVE_SET)].copy()
    person["id"] = pd.to_numeric(person["id"], errors="coerce").astype("Int64")
    person["year"] = pd.to_numeric(person["year"], errors="coerce").astype("Int64")
    if person["id"].isna().any() or person["year"].isna().any():
        raise RuntimeError("Person panel contains missing id/year after filtering.")
    person["release_date"] = pd.to_datetime(person["release_date"], errors="coerce")
    if person["release_date"].isna().any():
        raise RuntimeError("Person panel contains missing release_date values.")
    if person["release_date_source"].isna().any() or person["release_date_source"].str.strip().eq("").any():
        raise RuntimeError("Person panel contains missing release_date_source values.")
    person["wave_index"] = person["year"].map(WAVE_ORDER)
    if person["wave_index"].isna().any():
        raise RuntimeError("Person panel includes years outside the expected PSID wave schedule.")
    return person.sort_values(["id", "wave_index"], kind="mergesort").reset_index(drop=True)


def contiguous_next(records: list[dict[str, Any]], idx: int) -> dict[str, Any] | None:
    if idx + 1 >= len(records):
        return None
    current_year = safe_int(records[idx]["year"])
    next_year = safe_int(records[idx + 1]["year"])
    if current_year is None or next_year is None:
        return None
    if NEXT_WAVE.get(current_year) == next_year:
        return records[idx + 1]
    return None


def contiguous_history(records: list[dict[str, Any]], idx: int, max_depth: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    expected_prev = PREV_WAVE.get(safe_int(records[idx]["year"]))
    cursor = idx - 1
    while cursor >= 0 and expected_prev is not None and len(out) < max_depth:
        candidate_year = safe_int(records[cursor]["year"])
        if candidate_year != expected_prev:
            break
        out.append(records[cursor])
        expected_prev = PREV_WAVE.get(candidate_year)
        cursor -= 1
    return list(reversed(out))


def signed_int_change(current: Any, previous: Any) -> int | None:
    current_value = safe_float(current)
    previous_value = safe_float(previous)
    if current_value is None or previous_value is None:
        return None
    return int(round(current_value - previous_value))


def validate_code_domains(df: pd.DataFrame, code_domains: dict[str, set[int]], task_slug: str) -> None:
    for column, allowed in code_domains.items():
        values = pd.to_numeric(df[column], errors="coerce")
        bad = values.isna() | ~values.astype("Int64").isin(allowed)
        if bad.any():
            examples = sorted(values.loc[bad].dropna().unique().tolist())[:10]
            raise RuntimeError(f"{task_slug}: {column} has missing or unmapped required codes: {examples}")
