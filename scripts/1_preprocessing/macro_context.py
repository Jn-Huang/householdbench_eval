#!/usr/bin/env python
"""Build the canonical quarterly macro context used by HouseholdBench tasks."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.source_inputs import validate_source_inputs


FRED_QD_PATH = Path("data/raw/macro/fred-qd/2026-07-QD.csv")
CPI_PATH = Path("data/intermediate/cpi_major_groups.parquet")
OUTPUT_PATH = Path("data/intermediate/householdbench_macro_context.parquet")
OUTPUT_DIR = Path("output/preprocessing/macro_context/build")
SOURCE_MANIFEST_PATH = OUTPUT_DIR / "source_manifest.csv"
SERIES_COVERAGE_PATH = OUTPUT_DIR / "series_coverage.csv"

FRED_LEVEL_COLUMNS = [
    "GDPC1",
    "UNRATE",
    "CPIAUCSL",
    "FEDFUNDS",
    "USSTHPI",
    "S&P 500",
    "MORTGAGE30US",
]
CPI_LEVEL_COLUMNS = [
    "cpi_food_beverages",
    "cpi_housing",
    "cpi_apparel",
    "cpi_transportation",
    "cpi_medical_care",
    "cpi_recreation",
    "cpi_education_communication",
    "cpi_other_goods_services",
]
GROWTH_SERIES = {
    "gdp_growth_qoq": "GDPC1",
    "headline_cpi_qoq": "CPIAUCSL",
    "cpi_food_beverages_qoq": "cpi_food_beverages",
    "cpi_housing_qoq": "cpi_housing",
    "cpi_apparel_qoq": "cpi_apparel",
    "cpi_transportation_qoq": "cpi_transportation",
    "cpi_medical_care_qoq": "cpi_medical_care",
    "cpi_recreation_qoq": "cpi_recreation",
    "cpi_education_communication_qoq": "cpi_education_communication",
    "cpi_other_goods_services_qoq": "cpi_other_goods_services",
    "house_price_growth_qoq": "USSTHPI",
    "sp500_growth_qoq": "S&P 500",
}
RATE_SERIES = {
    "unemployment_rate": "UNRATE",
    "fedfunds_rate": "FEDFUNDS",
    "mortgage30us_rate": "MORTGAGE30US",
}
CORE_COLUMNS = [
    "gdp_growth_qoq_lag1",
    "gdp_growth_qoq_lag2",
    "gdp_growth_qoq_lag3",
    "gdp_growth_qoq_lag4",
    "gdp_growth_qoq_5y_compound",
    "unemployment_rate_lag1",
    "unemployment_rate_lag2",
    "unemployment_rate_lag3",
    "unemployment_rate_lag4",
    "unemployment_rate_5y_mean",
    "headline_cpi_qoq_lag1",
    "headline_cpi_qoq_lag2",
    "headline_cpi_qoq_lag3",
    "headline_cpi_qoq_lag4",
    "headline_cpi_qoq_5y_compound",
    "fedfunds_rate_lag1",
    "fedfunds_rate_lag2",
    "fedfunds_rate_lag3",
    "fedfunds_rate_lag4",
    "fedfunds_rate_5y_mean",
]


fred_path = FRED_QD_PATH
validate_source_inputs("macro_context")

if not fred_path.is_file():
    raise FileNotFoundError(f"Required July 2026 FRED-QD vintage is missing: {fred_path}")
if not CPI_PATH.is_file():
    raise SystemExit(f"CPI intermediate not found: {CPI_PATH}")

fred = pd.read_csv(fred_path, usecols=["sasdate", *FRED_LEVEL_COLUMNS])
fred["date"] = pd.to_datetime(fred["sasdate"], format="%m/%d/%Y", errors="coerce")
fred = fred.dropna(subset=["date"]).copy()
for column in FRED_LEVEL_COLUMNS:
    fred[column] = pd.to_numeric(fred[column], errors="coerce")
fred["year"] = fred["date"].dt.year.astype(int)
fred["quarter"] = fred["date"].dt.quarter.astype(int)
fred["q_index"] = fred["year"] * 4 + fred["quarter"]
fred = fred.sort_values("q_index", kind="mergesort").reset_index(drop=True)
if fred["q_index"].duplicated().any():
    raise RuntimeError("FRED-QD contains duplicate quarters.")
if not fred["q_index"].diff().dropna().eq(1).all():
    raise RuntimeError("FRED-QD contains a gap in quarterly coverage.")

cpi = pd.read_parquet(CPI_PATH, columns=["date", "year", "quarter", *CPI_LEVEL_COLUMNS])
cpi["date"] = pd.to_datetime(cpi["date"], errors="coerce")
if cpi["date"].isna().any() or cpi["date"].duplicated().any():
    raise RuntimeError("CPI intermediate must be unique by valid month.")
for column in CPI_LEVEL_COLUMNS:
    cpi[column] = pd.to_numeric(cpi[column], errors="coerce")
cpi = cpi.sort_values("date", kind="mergesort").reset_index(drop=True)
if not cpi["date"].dt.to_period("M").astype(int).diff().dropna().eq(1).all():
    raise RuntimeError("CPI intermediate contains a gap in monthly coverage.")

quarterly_means = cpi.groupby(["year", "quarter"], as_index=False)[CPI_LEVEL_COLUMNS].mean()
quarterly_counts = cpi.groupby(["year", "quarter"], as_index=False)[CPI_LEVEL_COLUMNS].count()
quarterly_cpi = quarterly_means.copy()
for column in CPI_LEVEL_COLUMNS:
    quarterly_cpi.loc[quarterly_counts[column].ne(3), column] = pd.NA
quarterly_cpi["q_index"] = quarterly_cpi["year"].astype(int) * 4 + quarterly_cpi["quarter"].astype(int)

levels = fred[["date", "year", "quarter", "q_index", *FRED_LEVEL_COLUMNS]].merge(
    quarterly_cpi[["q_index", *CPI_LEVEL_COLUMNS]],
    on="q_index",
    how="left",
    validate="one_to_one",
)

out = pd.DataFrame(
    {
        "eligible_origin_q_index": levels["q_index"] + 1,
        "macro_reference_q_index": levels["q_index"],
        "macro_reference_year": levels["year"],
        "macro_reference_quarter": levels["quarter"],
        "macro_reference_date": levels["date"],
    }
)

for prefix, level_column in GROWTH_SERIES.items():
    level = levels[level_column]
    growth = level.pct_change(fill_method=None) * 100.0
    for lag in range(1, 5):
        out[f"{prefix}_lag{lag}"] = growth.shift(lag - 1)
    full_window = growth.rolling(window=20, min_periods=20).count().eq(20)
    compound = ((level / level.shift(20)).pow(1.0 / 20.0) - 1.0) * 100.0
    out[f"{prefix}_5y_compound"] = compound.where(full_window)

for prefix, level_column in RATE_SERIES.items():
    level = levels[level_column]
    for lag in range(1, 5):
        out[f"{prefix}_lag{lag}"] = level.shift(lag - 1)
    out[f"{prefix}_5y_mean"] = level.rolling(window=20, min_periods=20).mean()

out = out.loc[out[CORE_COLUMNS].notna().all(axis=1)].copy()
out = out.sort_values("eligible_origin_q_index", kind="mergesort").reset_index(drop=True)
if out["eligible_origin_q_index"].duplicated().any():
    raise RuntimeError("Canonical macro context is not unique by eligible origin quarter.")

ordered_columns = [
    "eligible_origin_q_index",
    "macro_reference_q_index",
    "macro_reference_year",
    "macro_reference_quarter",
    "macro_reference_date",
]
for prefix in GROWTH_SERIES:
    ordered_columns.extend([f"{prefix}_lag{lag}" for lag in range(1, 5)])
    ordered_columns.append(f"{prefix}_5y_compound")
for prefix in RATE_SERIES:
    ordered_columns.extend([f"{prefix}_lag{lag}" for lag in range(1, 5)])
    ordered_columns.append(f"{prefix}_5y_mean")
out = out[ordered_columns]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
out.to_parquet(OUTPUT_PATH, index=False)

pd.DataFrame(
    [
        {
            "input": "fred_qd",
            "path": str(fred_path),
            "rows": len(fred),
            "min_date": fred["date"].min().date().isoformat(),
            "max_date": fred["date"].max().date().isoformat(),
        },
        {
            "input": "cpi_major_groups",
            "path": str(CPI_PATH),
            "rows": len(cpi),
            "min_date": cpi["date"].min().date().isoformat(),
            "max_date": cpi["date"].max().date().isoformat(),
        },
    ]
).to_csv(SOURCE_MANIFEST_PATH, index=False)

coverage_rows = []
for column in ordered_columns[5:]:
    available = out.loc[out[column].notna(), ["macro_reference_date", column]]
    coverage_rows.append(
        {
            "column": column,
            "n_nonmissing": len(available),
            "first_reference_date": "" if available.empty else available["macro_reference_date"].min().date().isoformat(),
            "last_reference_date": "" if available.empty else available["macro_reference_date"].max().date().isoformat(),
        }
    )
pd.DataFrame(coverage_rows).to_csv(SERIES_COVERAGE_PATH, index=False)

print(f"Wrote {OUTPUT_PATH} with {len(out):,} origin-quarter rows and {len(out.columns):,} columns.")
print(f"FRED-QD source: {fred_path}")
