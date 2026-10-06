#!/usr/bin/env python
"""Build annual state personal-income panel from BEA SAINC1."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.source_inputs import validate_source_inputs

RAW_DIR = Path("data/raw/macro/bea_sainc")
INTERMEDIATE_DIR = Path("data/intermediate")
OUT_DIR = Path("output/preprocessing/bea_sainc/build")

RAW_FILE = RAW_DIR / "SAINC1__ALL_AREAS_1929_2025.csv"
OUT_PATH = INTERMEDIATE_DIR / "bea_sainc_state_annual.parquet"
US_OUT_PATH = INTERMEDIATE_DIR / "bea_sainc_us_annual.parquet"
SERIES_AUDIT_PATH = OUT_DIR / "retained_series.csv"
GEOGRAPHY_AUDIT_PATH = OUT_DIR / "geography_coverage.csv"
US_COVERAGE_AUDIT_PATH = OUT_DIR / "us_coverage.csv"

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

LINECODE_RENAMES = {
    1: "personal_income_millions",
    2: "population",
    3: "pcpi_nominal",
}


def year_columns(data: pd.DataFrame) -> list[str]:
    cols = [col for col in data.columns if col.isdigit() and len(col) == 4]
    if not cols:
        raise SystemExit("SAINC1 file has no year columns.")
    return cols


if __name__ == "__main__":
    validate_source_inputs("bea_sainc")
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Read the source and normalise its fields.
    if not RAW_FILE.exists():
        raise SystemExit(f"Missing expected BEA SAINC file: {RAW_FILE}")
    raw = pd.read_csv(RAW_FILE, dtype=str)

    raw = raw.copy()
    raw.columns = [str(col).strip() for col in raw.columns]
    for col in raw.columns:
        raw[col] = raw[col].map(lambda value: value.strip() if isinstance(value, str) else value)

    needed = {
        "GeoFIPS",
        "GeoName",
        "Region",
        "TableName",
        "LineCode",
        "Description",
        "Unit",
    }
    if not needed.issubset(raw.columns):
        raise SystemExit(f"Unexpected BEA schema in {RAW_FILE}; expected columns {sorted(needed)}")

    base = raw.copy()
    base["GeoFIPS"] = base["GeoFIPS"].str.replace('"', "", regex=False)
    base["GeoName"] = base["GeoName"].str.replace("*", "", regex=False).str.strip()
    base["geofips_int"] = pd.to_numeric(base["GeoFIPS"], errors="coerce")
    base["statefip"] = (base["geofips_int"] // 1000).astype("Int64")
    linecodes = pd.to_numeric(base["LineCode"], errors="coerce").astype("Int64")
    base["LineCode"] = linecodes

    # 2. Select the state and national observations.
    keep_states = (
        base["statefip"].isin(sorted(VALID_US_STATE_CODES))
        & base["geofips_int"].mod(1000).eq(0)
        & base["LineCode"].isin(sorted(LINECODE_RENAMES))
    )
    state_data = base.loc[keep_states].copy()
    if state_data.empty:
        raise SystemExit("No state/DC SAINC1 rows survived geography and line-code filters.")

    if set(state_data["LineCode"].dropna().unique().tolist()) != set(LINECODE_RENAMES):
        raise SystemExit("Retained SAINC1 state line codes do not match the expected {1,2,3} set.")

    if state_data["statefip"].nunique() != len(VALID_US_STATE_CODES):
        raise SystemExit(
            f"Expected {len(VALID_US_STATE_CODES)} state/DC geographies in SAINC1, found {state_data['statefip'].nunique()}."
        )

    us_data = base.loc[
        base["GeoFIPS"].eq("00000") & base["LineCode"].isin(sorted(LINECODE_RENAMES))
    ].copy()
    if us_data.empty:
        raise SystemExit("No U.S. aggregate SAINC1 rows survived GeoFIPS==00000 filtering.")
    if set(us_data["LineCode"].dropna().unique().tolist()) != set(LINECODE_RENAMES):
        raise SystemExit("Retained SAINC1 U.S. line codes do not match the expected {1,2,3} set.")

    # 3. Reshape the state panel and calculate income growth.
    long_df = state_data.melt(
        id_vars=["statefip", "GeoName", "LineCode", "Description", "Unit"],
        value_vars=year_columns(state_data),
        var_name="year",
        value_name="value",
    )
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["year", "value"]).copy()
    long_df["year"] = long_df["year"].astype(int)
    long_df["LineCode"] = long_df["LineCode"].astype(int)

    panel = (
        long_df.pivot_table(
            index=["statefip", "GeoName", "year"],
            columns="LineCode",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
        .rename(columns={"GeoName": "state_name", **LINECODE_RENAMES})
    )

    expected = {"statefip", "state_name", "year", *LINECODE_RENAMES.values()}
    if not expected.issubset(panel.columns):
        raise SystemExit(f"SAINC1 reshape failed; expected columns {sorted(expected)}.")

    if panel.duplicated(["statefip", "year"]).any():
        raise SystemExit("Duplicate BEA state-year rows found after SAINC1 reshape.")

    panel = panel.sort_values(["statefip", "year"]).reset_index(drop=True)
    panel["pcpi_growth_1y"] = (
        panel.groupby("statefip")["pcpi_nominal"].pct_change().mul(100.0)
    )

    # 4. Construct the corresponding national series.
    long_df = us_data.melt(
        id_vars=["GeoName", "LineCode", "Description", "Unit"],
        value_vars=year_columns(us_data),
        var_name="year",
        value_name="value",
    )
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["year", "value"]).copy()
    long_df["year"] = long_df["year"].astype(int)
    long_df["LineCode"] = long_df["LineCode"].astype(int)

    pivot = (
        long_df.pivot_table(
            index=["GeoName", "year"],
            columns="LineCode",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
        .rename(columns={"GeoName": "geo_name", **LINECODE_RENAMES})
        .sort_values("year")
        .reset_index(drop=True)
    )
    needed = {"year", "pcpi_nominal", "personal_income_millions", "population"}
    if not needed.issubset(pivot.columns):
        raise SystemExit(f"SAINC1 U.S. reshape failed; expected columns {sorted(needed)}.")

    pivot["us_pcpi_growth_1y"] = pivot["pcpi_nominal"].pct_change().mul(100.0)
    us_panel = pivot[
        [
            "year",
            "pcpi_nominal",
            "us_pcpi_growth_1y",
        ]
    ].rename(columns={"pcpi_nominal": "us_pcpi_nominal"})

    # 5. Record coverage and write the constructed products.
    series_audit = (
        state_data[["statefip", "GeoName", "LineCode", "Description", "Unit"]]
        .drop_duplicates()
        .rename(columns={"GeoName": "state_name", "LineCode": "line_code"})
        .sort_values(["statefip", "line_code"])
        .reset_index(drop=True)
    )

    geography_audit = (
        panel.groupby(["statefip", "state_name"], as_index=False)
        .agg(min_year=("year", "min"), max_year=("year", "max"), n_years=("year", "size"))
        .sort_values("statefip")
        .reset_index(drop=True)
    )

    us_coverage_audit = pd.DataFrame(
        [
            {
                "geo_name": "United States",
                "min_year": int(us_panel["year"].min()),
                "max_year": int(us_panel["year"].max()),
                "n_years": int(len(us_panel)),
            }
        ]
    )

    series_audit.to_csv(SERIES_AUDIT_PATH, index=False)
    geography_audit.to_csv(GEOGRAPHY_AUDIT_PATH, index=False)
    us_coverage_audit.to_csv(US_COVERAGE_AUDIT_PATH, index=False)
    panel.to_parquet(OUT_PATH, index=False)
    us_panel.to_parquet(US_OUT_PATH, index=False)

    print(f"Wrote BEA SAINC state annual panel: {OUT_PATH}")
    print(f"Wrote BEA SAINC U.S. annual panel: {US_OUT_PATH}")
    print(f"Wrote BEA retained-series audit: {SERIES_AUDIT_PATH}")
    print(f"Wrote BEA geography coverage audit: {GEOGRAPHY_AUDIT_PATH}")
    print(f"Wrote BEA U.S. coverage audit: {US_COVERAGE_AUDIT_PATH}")
    print(
        f"Rows: {len(panel):,}; states: {panel['statefip'].nunique()}; "
        f"years: {panel['year'].min()}-{panel['year'].max()}"
    )
