#!/usr/bin/env python
"""Merge state unemployment, income, and house-price panels for Census prompts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

INTERMEDIATE_ROOT = Path("data/intermediate")
RAW_DIR = Path("data/raw/macro/fred-qd")
OUT_DIR = Path("output/preprocessing/state_macro_panel/build")

BLS_PATH = INTERMEDIATE_ROOT / "bls_laus_state_annual.parquet"
BEA_PATH = INTERMEDIATE_ROOT / "bea_sainc_state_annual.parquet"
BEA_US_PATH = INTERMEDIATE_ROOT / "bea_sainc_us_annual.parquet"
FHFA_PATH = INTERMEDIATE_ROOT / "fhfa_hpi_state_annual.parquet"
FHFA_US_PATH = INTERMEDIATE_ROOT / "fhfa_hpi_us_annual.parquet"

OUT_PATH = INTERMEDIATE_ROOT / "state_macro_panel.parquet"
SOURCE_COVERAGE_AUDIT_PATH = OUT_DIR / "source_coverage.csv"
STATE_MATCH_AUDIT_PATH = OUT_DIR / "state_match_audit.csv"
NATIONAL_COVERAGE_AUDIT_PATH = OUT_DIR / "national_coverage.csv"

MIGRATION_ORIGIN_YEARS = {1985, 1995}

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


def read_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing expected cleaned state-macro input: {path}")
    return pd.read_parquet(path)


def resolve_fred_qd_file(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(path.glob("*-QD.csv"))
        if candidates:
            return candidates[-1]
        raise SystemExit(f"No *-QD.csv files found in folder: {path}")
    raise SystemExit(f"FRED-QD path not found: {path}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. Read the constructed state and national panels.
    bls = read_panel(BLS_PATH)
    bea = read_panel(BEA_PATH)
    bea_us = read_panel(BEA_US_PATH)
    fhfa = read_panel(FHFA_PATH)
    fhfa_us = read_panel(FHFA_US_PATH)

    # 2. Construct annual national unemployment.
    raw = pd.read_csv(resolve_fred_qd_file(RAW_DIR))
    raw = raw.iloc[2:].copy()
    raw["date"] = pd.to_datetime(raw["sasdate"], format="%m/%d/%Y", errors="coerce")
    raw["UNRATE"] = pd.to_numeric(raw["UNRATE"], errors="coerce")
    raw = raw.dropna(subset=["date", "UNRATE"]).copy()
    raw["year"] = raw["date"].dt.year.astype(int)
    annual = (
        raw.groupby("year", as_index=False)["UNRATE"]
        .mean()
        .rename(columns={"UNRATE": "us_unemployment_rate"})
        .sort_values("year")
        .reset_index(drop=True)
    )
    fred_us = annual

    # 3. Match sources on the documented state-year keys.
    bls_keep = bls[["statefip", "state_name", "year", "unemployment_rate"]].copy()
    bea_keep = bea[
        ["statefip", "state_name", "year", "pcpi_nominal", "pcpi_growth_1y"]
    ].copy()
    fhfa_keep = fhfa[
        ["statefip", "state_abbrev", "state_name", "year", "hpi_all_transactions_annual", "hpi_growth_1y"]
    ].copy()
    bea_us_keep = bea_us[["year", "us_pcpi_nominal", "us_pcpi_growth_1y"]].copy()
    fhfa_us_keep = fhfa_us[
        ["year", "us_hpi_all_transactions_annual", "us_hpi_growth_1y"]
    ].copy()

    merged = (
        bls_keep.merge(
            bea_keep,
            on=["statefip", "state_name", "year"],
            how="inner",
            validate="one_to_one",
        )
        .merge(
            fhfa_keep,
            on=["statefip", "state_name", "year"],
            how="inner",
            validate="one_to_one",
        )
        .merge(
            bea_us_keep,
            on="year",
            how="left",
            validate="many_to_one",
        )
        .merge(
            fhfa_us_keep,
            on="year",
            how="left",
            validate="many_to_one",
        )
        .merge(
            fred_us,
            on="year",
            how="left",
            validate="many_to_one",
        )
        .sort_values(["statefip", "year"])
        .reset_index(drop=True)
    )

    if merged.duplicated(["statefip", "year"]).any():
        raise SystemExit("Duplicate state-year rows found in merged state macro panel.")
    if set(pd.to_numeric(merged["statefip"], errors="coerce").dropna().astype(int).unique()) != VALID_US_STATE_CODES:
        raise SystemExit("Merged state macro panel does not match the Census valid state/DC set.")

    needed = [
        "unemployment_rate",
        "pcpi_nominal",
        "pcpi_growth_1y",
        "hpi_all_transactions_annual",
        "hpi_growth_1y",
        "us_unemployment_rate",
        "us_pcpi_nominal",
        "us_pcpi_growth_1y",
        "us_hpi_all_transactions_annual",
        "us_hpi_growth_1y",
    ]
    if merged[needed].isna().any().any():
        missing_cols = [col for col in needed if merged[col].isna().any()]
        raise SystemExit(f"Merged state macro panel still has missing values in: {missing_cols}")

    final = merged[
        [
            "statefip",
            "state_abbrev",
            "state_name",
            "year",
            "unemployment_rate",
            "pcpi_nominal",
            "pcpi_growth_1y",
            "hpi_all_transactions_annual",
            "hpi_growth_1y",
            "us_unemployment_rate",
            "us_pcpi_nominal",
            "us_pcpi_growth_1y",
            "us_hpi_all_transactions_annual",
            "us_hpi_growth_1y",
        ]
    ].copy()

    # 4. Record source and matching coverage.
    rows = []
    for source_name, df in [
        ("bls_laus", bls_keep),
        ("bea_sainc", bea_keep),
        ("bea_sainc_us", bea_us_keep),
        ("fhfa_hpi", fhfa_keep),
        ("fhfa_hpi_us", fhfa_us_keep),
        ("fred_qd_unrate", fred_us),
        ("merged_panel", final),
    ]:
        rows.append(
            {
                "source": source_name,
                "min_year": int(df["year"].min()),
                "max_year": int(df["year"].max()),
                "n_rows": int(len(df)),
                "n_states": int(df["statefip"].nunique()) if "statefip" in df.columns else 1,
            }
        )
    source_coverage = pd.DataFrame(rows)

    rows = []
    expected = sorted(VALID_US_STATE_CODES)
    for source_name, df in [
        ("bls_laus", bls_keep),
        ("bea_sainc", bea_keep),
        ("fhfa_hpi", fhfa_keep),
        ("merged_panel", final),
    ]:
        observed = sorted(int(code) for code in pd.to_numeric(df["statefip"], errors="coerce").dropna().astype(int).unique())
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        rows.append(
            {
                "source": source_name,
                "matches_census_state_set": not missing and not extra,
                "missing_statefip": ",".join(str(code) for code in missing),
                "extra_statefip": ",".join(str(code) for code in extra),
            }
        )
    state_match = pd.DataFrame(rows)

    rows = []
    origin_subset = final.loc[final["year"].isin(sorted(MIGRATION_ORIGIN_YEARS))].copy()
    national_cols = [
        "us_unemployment_rate",
        "us_pcpi_nominal",
        "us_pcpi_growth_1y",
        "us_hpi_all_transactions_annual",
        "us_hpi_growth_1y",
    ]
    for col in national_cols:
        series = pd.to_numeric(final[col], errors="coerce")
        origin_series = pd.to_numeric(origin_subset[col], errors="coerce")
        rows.append(
            {
                "column": col,
                "min_year_nonmissing": int(final.loc[series.notna(), "year"].min()),
                "max_year_nonmissing": int(final.loc[series.notna(), "year"].max()),
                "n_missing_full_panel": int(series.isna().sum()),
                "n_missing_migration_origin_years": int(origin_series.isna().sum()),
            }
        )
    national_coverage = pd.DataFrame(rows)

    final.to_parquet(OUT_PATH, index=False)
    source_coverage.to_csv(SOURCE_COVERAGE_AUDIT_PATH, index=False)
    state_match.to_csv(STATE_MATCH_AUDIT_PATH, index=False)
    national_coverage.to_csv(NATIONAL_COVERAGE_AUDIT_PATH, index=False)

    print(f"Wrote merged state macro panel: {OUT_PATH}")
    print(f"Wrote source coverage audit: {SOURCE_COVERAGE_AUDIT_PATH}")
    print(f"Wrote state match audit: {STATE_MATCH_AUDIT_PATH}")
    print(f"Wrote national coverage audit: {NATIONAL_COVERAGE_AUDIT_PATH}")
    print(
        f"Rows: {len(final):,}; states: {final['statefip'].nunique()}; "
        f"years: {final['year'].min()}-{final['year'].max()}"
    )
