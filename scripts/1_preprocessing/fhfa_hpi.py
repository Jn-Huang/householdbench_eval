#!/usr/bin/env python
"""Build annual state house-price growth panel from FHFA HPI."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.source_inputs import validate_source_inputs

RAW_DIR = Path("data/raw/macro/fhfa_hpi")
INTERMEDIATE_DIR = Path("data/intermediate")
OUT_DIR = Path("output/preprocessing/fhfa_hpi/build")

RAW_FILE = RAW_DIR / "hpi_master.csv"
OUT_PATH = INTERMEDIATE_DIR / "fhfa_hpi_state_annual.parquet"
US_OUT_PATH = INTERMEDIATE_DIR / "fhfa_hpi_us_annual.parquet"
SELECTION_AUDIT_PATH = OUT_DIR / "selection_audit.csv"
COVERAGE_AUDIT_PATH = OUT_DIR / "coverage.csv"
US_COVERAGE_AUDIT_PATH = OUT_DIR / "us_coverage.csv"

STATE_ABBREV_TO_INFO = {
    "AL": (1, "Alabama"),
    "AK": (2, "Alaska"),
    "AZ": (4, "Arizona"),
    "AR": (5, "Arkansas"),
    "CA": (6, "California"),
    "CO": (8, "Colorado"),
    "CT": (9, "Connecticut"),
    "DE": (10, "Delaware"),
    "DC": (11, "District of Columbia"),
    "FL": (12, "Florida"),
    "GA": (13, "Georgia"),
    "HI": (15, "Hawaii"),
    "ID": (16, "Idaho"),
    "IL": (17, "Illinois"),
    "IN": (18, "Indiana"),
    "IA": (19, "Iowa"),
    "KS": (20, "Kansas"),
    "KY": (21, "Kentucky"),
    "LA": (22, "Louisiana"),
    "ME": (23, "Maine"),
    "MD": (24, "Maryland"),
    "MA": (25, "Massachusetts"),
    "MI": (26, "Michigan"),
    "MN": (27, "Minnesota"),
    "MS": (28, "Mississippi"),
    "MO": (29, "Missouri"),
    "MT": (30, "Montana"),
    "NE": (31, "Nebraska"),
    "NV": (32, "Nevada"),
    "NH": (33, "New Hampshire"),
    "NJ": (34, "New Jersey"),
    "NM": (35, "New Mexico"),
    "NY": (36, "New York"),
    "NC": (37, "North Carolina"),
    "ND": (38, "North Dakota"),
    "OH": (39, "Ohio"),
    "OK": (40, "Oklahoma"),
    "OR": (41, "Oregon"),
    "PA": (42, "Pennsylvania"),
    "RI": (44, "Rhode Island"),
    "SC": (45, "South Carolina"),
    "SD": (46, "South Dakota"),
    "TN": (47, "Tennessee"),
    "TX": (48, "Texas"),
    "UT": (49, "Utah"),
    "VT": (50, "Vermont"),
    "VA": (51, "Virginia"),
    "WA": (53, "Washington"),
    "WV": (54, "West Virginia"),
    "WI": (55, "Wisconsin"),
    "WY": (56, "Wyoming"),
}


if __name__ == "__main__":
    validate_source_inputs("fhfa_hpi")
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Read the source and normalise its fields.
    if not RAW_FILE.exists():
        raise SystemExit(f"Missing expected FHFA file: {RAW_FILE}")
    raw = pd.read_csv(RAW_FILE, dtype=str)

    raw = raw.copy()
    raw.columns = [str(col).strip() for col in raw.columns]
    for col in raw.columns:
        raw[col] = raw[col].map(lambda value: value.strip() if isinstance(value, str) else value)

    # 2. Record the selected source series.
    audit = (
        raw.groupby(["level", "hpi_type", "hpi_flavor", "frequency"], as_index=False)
        .agg(
            n_rows=("place_id", "size"),
            n_places=("place_id", "nunique"),
            min_year=("yr", lambda s: pd.to_numeric(s, errors="coerce").min()),
            max_year=("yr", lambda s: pd.to_numeric(s, errors="coerce").max()),
        )
        .sort_values(["level", "hpi_type", "hpi_flavor", "frequency"])
        .reset_index(drop=True)
    )
    audit["chosen"] = (
        audit["level"].eq("State")
        & audit["hpi_type"].eq("traditional")
        & audit["hpi_flavor"].eq("all-transactions")
        & audit["frequency"].eq("quarterly")
    ) | (
        audit["level"].eq("USA or Census Division")
        & audit["hpi_type"].eq("traditional")
        & audit["hpi_flavor"].eq("all-transactions")
        & audit["frequency"].eq("quarterly")
    )
    audit["selection_reason"] = ""
    audit.loc[
        audit["level"].eq("State")
        & audit["hpi_type"].eq("traditional")
        & audit["hpi_flavor"].eq("all-transactions")
        & audit["frequency"].eq("quarterly"),
        "selection_reason",
    ] = "Chosen because it is the only state-level FHFA series with pre-1991 coverage needed for 1985 origin-year matching."
    audit.loc[
        audit["level"].eq("USA or Census Division")
        & audit["hpi_type"].eq("traditional")
        & audit["hpi_flavor"].eq("all-transactions")
        & audit["frequency"].eq("quarterly"),
        "selection_reason",
    ] = "Includes the U.S. aggregate and division-level all-transactions quarterly HPI family used to recover the comparable national series."
    selection_audit = audit

    # 3. Construct state annual indices from complete quarterly observations.
    data = raw.loc[
        raw["level"].eq("State")
        & raw["hpi_type"].eq("traditional")
        & raw["hpi_flavor"].eq("all-transactions")
        & raw["frequency"].eq("quarterly")
    ].copy()
    if data.empty:
        raise SystemExit("No FHFA rows matched the selected state all-transactions quarterly specification.")

    data["yr"] = pd.to_numeric(data["yr"], errors="coerce")
    data["period"] = pd.to_numeric(data["period"], errors="coerce")
    data["index_nsa"] = pd.to_numeric(data["index_nsa"], errors="coerce")
    data = data.dropna(subset=["yr", "period", "index_nsa"]).copy()
    data["yr"] = data["yr"].astype(int)
    data["period"] = data["period"].astype(int)
    data["state_abbrev"] = data["place_id"].str.upper()
    data["statefip"] = data["state_abbrev"].map(lambda code: STATE_ABBREV_TO_INFO.get(code, (None, None))[0])
    data["state_name"] = data["state_abbrev"].map(lambda code: STATE_ABBREV_TO_INFO.get(code, (None, None))[1])

    if data[["statefip", "state_name"]].isna().any().any():
        bad = data.loc[data["statefip"].isna(), "place_id"].drop_duplicates().tolist()
        raise SystemExit(f"Failed to map FHFA state abbreviations to Census state FIPS: {bad[:5]}")

    quarterly_counts = (
        data.groupby(["statefip", "yr"], as_index=False)
        .agg(n_quarters=("period", "nunique"))
        .sort_values(["statefip", "yr"])
        .reset_index(drop=True)
    )
    complete = quarterly_counts.loc[quarterly_counts["n_quarters"].eq(4), ["statefip", "yr"]]
    data = data.merge(complete, on=["statefip", "yr"], how="inner", validate="many_to_one")

    annual = (
        data.groupby(["statefip", "state_abbrev", "state_name", "yr"], as_index=False)["index_nsa"]
        .mean()
        .rename(columns={"yr": "year", "index_nsa": "hpi_all_transactions_annual"})
        .sort_values(["statefip", "year"])
        .reset_index(drop=True)
    )
    if annual.duplicated(["statefip", "year"]).any():
        raise SystemExit("Duplicate FHFA state-year rows found in the annual HPI panel.")

    annual["hpi_growth_1y"] = (
        annual.groupby("statefip")["hpi_all_transactions_annual"].pct_change().mul(100.0)
    )
    if annual["statefip"].nunique() != len(STATE_ABBREV_TO_INFO):
        raise SystemExit(
            f"Expected {len(STATE_ABBREV_TO_INFO)} state/DC geographies in FHFA panel, found {annual['statefip'].nunique()}."
        )
    panel = annual

    # 4. Construct the corresponding national series.
    data = raw.loc[
        raw["place_id"].eq("USA")
        & raw["level"].eq("USA or Census Division")
        & raw["hpi_type"].eq("traditional")
        & raw["hpi_flavor"].eq("all-transactions")
        & raw["frequency"].eq("quarterly")
    ].copy()
    if data.empty:
        raise SystemExit("No FHFA rows matched the selected U.S. all-transactions quarterly specification.")

    data["yr"] = pd.to_numeric(data["yr"], errors="coerce")
    data["period"] = pd.to_numeric(data["period"], errors="coerce")
    data["index_nsa"] = pd.to_numeric(data["index_nsa"], errors="coerce")
    data = data.dropna(subset=["yr", "period", "index_nsa"]).copy()
    data["yr"] = data["yr"].astype(int)
    data["period"] = data["period"].astype(int)

    quarter_counts = (
        data.groupby("yr", as_index=False)
        .agg(n_quarters=("period", "nunique"))
        .sort_values("yr")
        .reset_index(drop=True)
    )
    complete = quarter_counts.loc[quarter_counts["n_quarters"].eq(4), ["yr"]]
    data = data.merge(complete, on="yr", how="inner", validate="many_to_one")

    annual = (
        data.groupby("yr", as_index=False)["index_nsa"]
        .mean()
        .rename(columns={"yr": "year", "index_nsa": "us_hpi_all_transactions_annual"})
        .sort_values("year")
        .reset_index(drop=True)
    )
    if annual.duplicated(["year"]).any():
        raise SystemExit("Duplicate FHFA U.S.-year rows found in the annual HPI panel.")
    annual["us_hpi_growth_1y"] = annual["us_hpi_all_transactions_annual"].pct_change().mul(100.0)
    us_panel = annual

    # 5. Record coverage and write the constructed products.
    coverage = (
        panel.groupby(["statefip", "state_abbrev", "state_name"], as_index=False)
        .agg(min_year=("year", "min"), max_year=("year", "max"), n_years=("year", "size"))
        .sort_values("statefip")
        .reset_index(drop=True)
    )

    us_coverage = pd.DataFrame(
        [
            {
                "geo_name": "United States",
                "min_year": int(us_panel["year"].min()),
                "max_year": int(us_panel["year"].max()),
                "n_years": int(len(us_panel)),
            }
        ]
    )

    selection_audit.to_csv(SELECTION_AUDIT_PATH, index=False)
    coverage.to_csv(COVERAGE_AUDIT_PATH, index=False)
    us_coverage.to_csv(US_COVERAGE_AUDIT_PATH, index=False)
    panel.to_parquet(OUT_PATH, index=False)
    us_panel.to_parquet(US_OUT_PATH, index=False)

    print(f"Wrote FHFA HPI state annual panel: {OUT_PATH}")
    print(f"Wrote FHFA HPI U.S. annual panel: {US_OUT_PATH}")
    print(f"Wrote FHFA selection audit: {SELECTION_AUDIT_PATH}")
    print(f"Wrote FHFA coverage audit: {COVERAGE_AUDIT_PATH}")
    print(f"Wrote FHFA U.S. coverage audit: {US_COVERAGE_AUDIT_PATH}")
    print(
        f"Rows: {len(panel):,}; states: {panel['statefip'].nunique()}; "
        f"years: {panel['year'].min()}-{panel['year'].max()}"
    )
