#!/usr/bin/env python
"""Build annual state unemployment rates from BLS LAUS raw files."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.source_inputs import validate_source_inputs

RAW_DIR = Path("data/raw/macro/bls_la")
INTERMEDIATE_DIR = Path("data/intermediate")
OUT_DIR = Path("output/preprocessing/bls_laus/build")

DATA_FILE = RAW_DIR / "la.data.2.AllStatesU.txt"

OUT_PATH = INTERMEDIATE_DIR / "bls_laus_state_annual.parquet"
SERIES_AUDIT_PATH = OUT_DIR / "series_selection.csv"
COVERAGE_AUDIT_PATH = OUT_DIR / "coverage.csv"

VALID_US_STATE_CODES = {
    1,
    2,
    4,
    5,
    6,
    8,
    9,
    10,
    11,
    12,
    13,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    53,
    54,
    55,
    56,
}


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing expected LAUS file: {path}")
    return pd.read_csv(path, sep="\t", dtype=str)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    for col in out.columns:
        out[col] = out[col].map(lambda value: value.strip() if isinstance(value, str) else value)
    return out


if __name__ == "__main__":
    validate_source_inputs("bls_laus")
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Select the documented state unemployment series.
    selected = pd.read_csv(Path(__file__).resolve().parent / "mappings/bls_laus_series.csv", dtype=str)
    selected["statefip"] = selected["statefip"].astype(int)

    # 2. Construct and check the annual state panel.
    raw = clean_columns(read_tsv(DATA_FILE))
    need_cols = {"series_id", "year", "period", "value"}
    if not need_cols.issubset(raw.columns):
        raise SystemExit(f"Unexpected LAUS data schema in {DATA_FILE}; expected {sorted(need_cols)}")

    data = raw.loc[raw["series_id"].isin(selected["series_id"]) & raw["period"].eq("M13")].copy()
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["year", "value"]).copy()
    data["year"] = data["year"].astype(int)
    data = data.rename(columns={"value": "unemployment_rate"})

    panel = data.merge(
        selected[["series_id", "statefip", "state_name"]],
        on="series_id",
        how="left",
        validate="many_to_one",
    )
    if panel[["statefip", "state_name"]].isna().any().any():
        raise SystemExit("Selected LAUS data rows failed to map back to state metadata.")

    if panel.duplicated(["statefip", "year"]).any():
        raise SystemExit("Duplicate state-year rows found in annual LAUS unemployment panel.")

    panel = panel[["statefip", "state_name", "year", "unemployment_rate"]].sort_values(
        ["statefip", "year"]
    )
    if panel["year"].min() != 1976:
        raise SystemExit(f"Expected LAUS annual state coverage to start in 1976, found {panel['year'].min()}.")
    if panel["year"].max() != 2025:
        raise SystemExit(f"Expected LAUS annual state coverage to end in 2025, found {panel['year'].max()}.")
    if set(panel["statefip"].unique()) != VALID_US_STATE_CODES:
        raise SystemExit("Final LAUS annual panel does not match the Census valid state/DC set.")
    expected_state_years = pd.MultiIndex.from_product(
        [sorted(VALID_US_STATE_CODES), range(1976, 2026)],
        names=["statefip", "year"],
    )
    observed_state_years = pd.MultiIndex.from_frame(panel[["statefip", "year"]])
    if not observed_state_years.equals(expected_state_years):
        missing = expected_state_years.difference(observed_state_years).tolist()[:10]
        extra = observed_state_years.difference(expected_state_years).tolist()[:10]
        raise SystemExit(
            "LAUS annual state panel is not the complete 51-state/DC 1976-2025 grid; "
            f"first missing={missing}, first extra={extra}."
        )
    panel = panel.reset_index(drop=True)

    # 3. Record coverage and write the constructed products.
    coverage = (
        panel.groupby(["statefip", "state_name"], as_index=False)
        .agg(min_year=("year", "min"), max_year=("year", "max"), n_years=("year", "size"))
        .sort_values("statefip")
        .reset_index(drop=True)
    )

    selected.to_csv(SERIES_AUDIT_PATH, index=False)
    coverage.to_csv(COVERAGE_AUDIT_PATH, index=False)
    panel.to_parquet(OUT_PATH, index=False)

    print(f"Wrote BLS LAUS state annual panel: {OUT_PATH}")
    print(f"Wrote LAUS series audit: {SERIES_AUDIT_PATH}")
    print(f"Wrote LAUS coverage audit: {COVERAGE_AUDIT_PATH}")
    print(
        f"Rows: {len(panel):,}; states: {panel['statefip'].nunique()}; "
        f"years: {panel['year'].min()}-{panel['year'].max()}"
    )
