#!/usr/bin/env python
"""Build CPI major-group index panel (aggregate + 8 major groups)."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.source_inputs import validate_source_inputs

RAW_DIR = Path("data/raw/macro/cpi")
INTERMEDIATE_DIR = Path("data/intermediate")
OUT_DIR = Path("output/preprocessing/cpi/build")

OUT_PATH = INTERMEDIATE_DIR / "cpi_major_groups.parquet"
SERIES_AUDIT_PATH = OUT_DIR / "cpi_series_selection.csv"
CROSSWALK_PATH = OUT_DIR / "cex_cpi_coarse_crosswalk.csv"


# Flat files that contain the target aggregate + major groups.
DATA_FILES = [
    RAW_DIR / "cu.data.1.AllItems.txt",
    RAW_DIR / "cu.data.11.USFoodBeverage.txt",
    RAW_DIR / "cu.data.12.USHousing.txt",
    RAW_DIR / "cu.data.13.USApparel.txt",
    RAW_DIR / "cu.data.14.USTransportation.txt",
    RAW_DIR / "cu.data.15.USMedical.txt",
    RAW_DIR / "cu.data.16.USRecreation.txt",
    RAW_DIR / "cu.data.17.USEducationAndCommunication.txt",
    RAW_DIR / "cu.data.18.USOtherGoodsAndServices.txt",
]


# Coarse CEX-to-CPI map requested by user (non-UCC-level approximation).
COARSE_CEX_CPI_CROSSWALK = [
    ("cons_total", "cpi_all_items", "Approximate aggregate index anchor."),
    ("cons_parker_total", "cpi_all_items", "Approximate aggregate index anchor for Parker-style total expenditure."),
    ("cons_food", "cpi_food_beverages", "Direct broad-group match."),
    ("cons_alcohol", "cpi_food_beverages", "Alcohol sits inside food/beverage CPI major group."),
    ("cons_housing", "cpi_housing", "Direct broad-group match. The task keeps utilities inside the broad housing aggregate."),
    ("cons_apparel", "cpi_apparel", "Direct broad-group match."),
    ("cons_transport", "cpi_transportation", "Direct broad-group match."),
    ("cons_health", "cpi_medical_care", "Direct broad-group match."),
    ("cons_entertainment", "cpi_recreation", "Entertainment/recreation broad alignment."),
    ("cons_read", "cpi_recreation", "Reading materials sit inside the recreation CPI hierarchy."),
    ("cons_education", "cpi_education_communication", "Education in CEX maps to education+communication bundle in CPI."),
    ("cons_personal_care", "cpi_other_goods_services", "Personal care is a key component of other goods/services."),
    ("cons_tobacco", "cpi_other_goods_services", "Tobacco/smoking supplies are placed under other goods/services."),
    ("cons_misc", "cpi_other_goods_services", "Miscellaneous spending is assigned to other goods/services in the coarse task mapping."),
    ("cons_cash_contrib", "", "No direct CPI price-index counterpart."),
]


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    for col in out.columns:
        out[col] = out[col].astype(str).str.strip()
    return out


if __name__ == "__main__":
    validate_source_inputs("cpi")
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Select the documented CPI series.
    selected = pd.read_csv(Path(__file__).resolve().parent / "mappings/cpi_series.csv", dtype=str)
    selected.to_csv(SERIES_AUDIT_PATH, index=False)

    # 2. Read the monthly observations for those series.
    series_ids = set(selected["series_id"].tolist())
    chunks = []

    for path in DATA_FILES:
        if not path.exists():
            raise SystemExit(f"Missing expected CPI data file: {path}")
        df = clean_columns(read_tsv(path))
        need_cols = {"series_id", "year", "period", "value"}
        if not need_cols.issubset(df.columns):
            raise SystemExit(f"Unexpected schema in {path}; expected columns {sorted(need_cols)}")
        sub = df[df["series_id"].isin(series_ids)][["series_id", "year", "period", "value"]].copy()
        if not sub.empty:
            chunks.append(sub)

    if not chunks:
        raise SystemExit("No CPI rows found for selected major-group series.")

    out = pd.concat(chunks, ignore_index=True)
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out = out[out["period"].str.match(r"^M(0[1-9]|1[0-2])$")].copy()
    out["month"] = pd.to_numeric(out["period"].str[1:], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["year", "month", "value"]).copy()

    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)
    out["date"] = pd.to_datetime(
        out["year"].astype(str) + "-" + out["month"].astype(str).str.zfill(2) + "-01",
        errors="coerce",
    )
    out = out.dropna(subset=["date"]).copy()

    out = out.merge(
        selected[["series_id", "target_name", "item_name", "item_code"]],
        on="series_id",
        how="left",
    )
    raw = out

    # 3. Construct the complete monthly panel.
    wide = raw.pivot_table(index="date", columns="target_name", values="value", aggfunc="first")
    wide = wide.sort_index().reset_index()

    # Build a complete monthly index over the selected data window.
    full_dates = pd.DataFrame({"date": pd.date_range(wide["date"].min(), wide["date"].max(), freq="MS")})
    wide = full_dates.merge(wide, on="date", how="left")

    wide["year"] = wide["date"].dt.year.astype(int)
    wide["month"] = wide["date"].dt.month.astype(int)
    wide["quarter"] = ((wide["month"] - 1) // 3 + 1).astype(int)
    wide["year_quarter"] = wide["year"].astype(str) + "Q" + wide["quarter"].astype(str)

    ordered = [
        "date",
        "year",
        "month",
        "quarter",
        "year_quarter",
        "cpi_all_items",
        "cpi_food_beverages",
        "cpi_housing",
        "cpi_apparel",
        "cpi_transportation",
        "cpi_medical_care",
        "cpi_recreation",
        "cpi_education_communication",
        "cpi_other_goods_services",
    ]

    # Ensure all expected columns exist even if one target series is absent.
    for col in ordered:
        if col not in wide.columns:
            wide[col] = pd.NA

    final = wide[ordered].copy()
    final.to_parquet(OUT_PATH, index=False)

    # 4. Write the expenditure-to-CPI crosswalk.
    crosswalk = pd.DataFrame(
        COARSE_CEX_CPI_CROSSWALK,
        columns=["cex_variable", "mapped_cpi_series", "notes"],
    )
    crosswalk.to_csv(CROSSWALK_PATH, index=False)

    print(f"Wrote CPI panel: {OUT_PATH}")
    print(f"Wrote series selection audit: {SERIES_AUDIT_PATH}")
    print(f"Wrote coarse crosswalk: {CROSSWALK_PATH}")
    print(f"Rows: {len(final):,}; date range: {final['date'].min()} to {final['date'].max()}")
    print("Selected series IDs:")
    print(selected[["target_name", "series_id", "begin_year", "begin_period", "end_year", "end_period"]].to_string(index=False))
    print(f"Crosswalk rows: {len(crosswalk)}")
