"""Load, attach, and render the precomputed HouseholdBench macro context."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MACRO_CONTEXT_PATH = PROJECT_ROOT / "data/intermediate/householdbench_macro_context.parquet"

CORE_MACRO_COLUMNS = [
    "macro_reference_q_index",
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
PUBLIC_CORE_MACRO_COLUMNS = [
    column for column in CORE_MACRO_COLUMNS if column != "macro_reference_q_index"
]

CATEGORY_PREFIXES = [
    ("cpi_food_beverages_qoq", "Food and beverages"),
    ("cpi_housing_qoq", "Housing"),
    ("cpi_apparel_qoq", "Apparel"),
    ("cpi_transportation_qoq", "Transportation"),
    ("cpi_medical_care_qoq", "Medical care"),
    ("cpi_recreation_qoq", "Recreation"),
    ("cpi_education_communication_qoq", "Education and communication"),
    ("cpi_other_goods_services_qoq", "Other goods and services"),
]
CATEGORY_CPI_COLUMNS = [
    column
    for prefix, _ in CATEGORY_PREFIXES
    for column in [*[f"{prefix}_lag{lag}" for lag in range(1, 5)], f"{prefix}_5y_compound"]
]

PSID_ASSET_PREFIXES = [
    ("house_price_growth_qoq", "U.S. house-price growth", "growth"),
    ("sp500_growth_qoq", "S&P 500 growth", "growth"),
    ("mortgage30us_rate", "30-year mortgage rate", "rate"),
]
PSID_ASSET_COLUMNS = [
    column
    for prefix, _, kind in PSID_ASSET_PREFIXES
    for column in [
        *[f"{prefix}_lag{lag}" for lag in range(1, 5)],
        f"{prefix}_5y_{'compound' if kind == 'growth' else 'mean'}",
    ]
]
MORTGAGE_CONTEXT_COLUMNS = ["mortgage30us_rate_lag1"]


def load_macro_context(columns: list[str]) -> pd.DataFrame:
    if not MACRO_CONTEXT_PATH.is_file():
        raise RuntimeError(
            f"Macro context not found: {MACRO_CONTEXT_PATH}. "
            "Run scripts/1_preprocessing/macro_context.py first."
        )
    requested = list(dict.fromkeys(["eligible_origin_q_index", *columns]))
    available = pd.read_parquet(MACRO_CONTEXT_PATH).columns.tolist()
    missing = [column for column in requested if column not in available]
    if missing:
        raise RuntimeError(f"Macro context is missing requested columns: {missing}")
    context = pd.read_parquet(MACRO_CONTEXT_PATH, columns=requested)
    if context["eligible_origin_q_index"].duplicated().any():
        raise RuntimeError("Macro context is not unique by eligible origin quarter.")
    return context


def attach_macro_context(
    df: pd.DataFrame,
    *,
    origin_q_index_col: str,
    required_columns: list[str],
    optional_columns: list[str] | None = None,
) -> pd.DataFrame:
    if origin_q_index_col not in df.columns:
        raise RuntimeError(f"Origin-quarter column not found: {origin_q_index_col}")
    optional = [] if optional_columns is None else optional_columns
    columns = list(dict.fromkeys([*required_columns, *optional]))
    collisions = sorted(set(columns) & set(df.columns))
    if collisions:
        raise RuntimeError(f"Macro attachment would overwrite columns: {collisions}")
    context = load_macro_context(columns)
    order_column = "_macro_attachment_order"
    if order_column in df.columns:
        raise RuntimeError(f"Reserved temporary column already exists: {order_column}")
    out = df.copy()
    out[order_column] = range(len(out))
    out = out.merge(
        context,
        left_on=origin_q_index_col,
        right_on="eligible_origin_q_index",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if len(out) != len(df):
        raise RuntimeError("Macro attachment changed the row count.")
    out = out.sort_values(order_column, kind="mergesort").drop(columns=[order_column])
    out.index = df.index
    if out[required_columns].isna().any().any():
        counts = out[required_columns].isna().sum()
        raise RuntimeError(
            f"Required macro fields are missing: {counts[counts.gt(0)].to_dict()}"
        )
    return out


def _value(row: Any, column: str) -> float:
    value = row[column]
    if pd.isna(value):
        raise RuntimeError(f"Cannot render missing macro value: {column}")
    return float(value)


def _path(row: Any, prefix: str) -> str:
    values = [_value(row, f"{prefix}_lag{lag}") for lag in [4, 3, 2, 1]]
    return f"{values[0]:.2f}%, {values[1]:.2f}%, {values[2]:.2f}%, and {values[3]:.2f}%"


def render_core_macro_paragraph(row: Any) -> str:
    return (
        "You are also aware of recent national economic conditions. Over the four most recently "
        "completed calendar quarters, listed from oldest to most recent, U.S. real GDP changed by "
        f"{_path(row, 'gdp_growth_qoq')} quarter over quarter; the unemployment rate was "
        f"{_path(row, 'unemployment_rate')}; overall consumer prices changed by "
        f"{_path(row, 'headline_cpi_qoq')} quarter over quarter; and the effective federal funds rate was "
        f"{_path(row, 'fedfunds_rate')}. Over the five years ending in the most recently completed "
        f"quarter, compound average quarterly real GDP growth was {_value(row, 'gdp_growth_qoq_5y_compound'):.2f}%, "
        f"the average unemployment rate was {_value(row, 'unemployment_rate_5y_mean'):.2f}%, compound average "
        f"quarterly CPI inflation was {_value(row, 'headline_cpi_qoq_5y_compound'):.2f}%, and the average effective "
        f"federal funds rate was {_value(row, 'fedfunds_rate_5y_mean'):.2f}%."
    )




def render_category_cpi_paragraph(row: Any) -> str | None:
    m4_columns = [f"{prefix}_lag{lag}" for prefix, _ in CATEGORY_PREFIXES for lag in range(1, 5)]
    if any(pd.isna(row[column]) for column in m4_columns):
        return None
    paths = [f"{_path(row, prefix)} quarter over quarter for {label.lower()}" for prefix, label in CATEGORY_PREFIXES]
    first = (
        "You are also aware of changes in prices for specific spending categories. Over the same four "
        "quarters, listed from oldest to most recent, prices changed by "
        + "; ".join(paths[:-1])
        + f"; and {paths[-1]}."
    )
    l5_columns = [f"{prefix}_5y_compound" for prefix, _ in CATEGORY_PREFIXES]
    if any(pd.isna(row[column]) for column in l5_columns):
        return first
    references = [f"{_value(row, f'{prefix}_5y_compound'):.2f}% for {label.lower()}" for prefix, label in CATEGORY_PREFIXES]
    return first + " Over the same five-year period, compound average quarterly inflation was " + ", ".join(references[:-1]) + f", and {references[-1]}."




def render_psid_asset_paragraph(row: Any) -> str:
    first = (
        "You are also aware of recent financial and housing-market conditions. Over the same four completed "
        "quarters, listed from oldest to most recent, U.S. house prices changed by "
        f"{_path(row, 'house_price_growth_qoq')} quarter over quarter; the S&P 500 changed by "
        f"{_path(row, 'sp500_growth_qoq')} quarter over quarter; and the average 30-year mortgage rate was "
        f"{_path(row, 'mortgage30us_rate')}."
    )
    if pd.isna(row["house_price_growth_qoq_5y_compound"]):
        return first
    return (
        first
        + f" Over the same five-year period, compound average quarterly house-price growth was {_value(row, 'house_price_growth_qoq_5y_compound'):.2f}%, "
        f"compound average quarterly S&P 500 growth was {_value(row, 'sp500_growth_qoq_5y_compound'):.2f}%, and the average 30-year "
        f"mortgage rate was {_value(row, 'mortgage30us_rate_5y_mean'):.2f}%."
    )
