#!/usr/bin/env python
"""Build the source-faithful financing-experiment handoff.

This stage checks source integrity and the documented calculator formula. Paper-estimation screens and paper-result replications are
deliberately outside the benchmark pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.source_inputs import validate_source_inputs

from scripts.utils.io import sha256_file
TASK_ID = "house_sce_financing"
REPLICATION_DIR = PROJECT_ROOT / "data/raw/micro/sce/financing_fuster_zafar_2021/ReplicationMaterial"
RAW_PATH = REPLICATION_DIR / "data/raw/FZdata_raw.dta"
INTERMEDIATE_PATH = PROJECT_ROOT / "data/intermediate/house_sce_financing.parquet"
OUTPUT_DIR = PROJECT_ROOT / "output/preprocessing/sce_financing"

SOURCE_SEQUENCE_COLUMNS = [
    "hypo_q1", "hypo_q2", "hypo_q3", "hypo_q4", "hypo_d2", "hypo_d3", "hypo_d4"
]
CALCULATOR_TERM_MONTHS = 360
CALCULATOR_DEDUCTIBLE_INTEREST_SHARE = 0.95
CALCULATOR_TAX_RATE = 0.34
CALCULATOR_NONMORTGAGE_ANNUAL_RATE = 0.0335
CALCULATOR_FORMULA_VERSION = "fz_questionnaire_released_code_v1"
INCOME_BANDS = {
    1: "less than $10,000",
    2: "$10,000 to $20,000",
    3: "$20,000 to $30,000",
    4: "$30,000 to $40,000",
    5: "$40,000 to $50,000",
    6: "$50,000 to $60,000",
    7: "$60,000 to $75,000",
    8: "$75,000 to $100,000",
    9: "$100,000 to $150,000",
    10: "$150,000 to $200,000",
    11: "$200,000 or more",
}
CREDIT_SCORE_BANDS = {
    1: "below 620",
    2: "620 to 679",
    3: "680 to 719",
    4: "720 to 760",
    5: "above 760",
    6: "does not know",
}


def calculator_payment(
    purchase_price: pd.Series,
    down_payment: pd.Series,
    comparable_home_value: pd.Series,
    annual_rate_percent: pd.Series,
) -> pd.Series:
    monthly_rate = annual_rate_percent / 1200
    mortgage_balance = purchase_price - down_payment
    annuity_factor = monthly_rate * (1 + monthly_rate) ** CALCULATOR_TERM_MONTHS / (
        (1 + monthly_rate) ** CALCULATOR_TERM_MONTHS - 1
    )
    unrounded = mortgage_balance * (
        annuity_factor
        - CALCULATOR_DEDUCTIBLE_INTEREST_SHARE * CALCULATOR_TAX_RATE * monthly_rate
    ) + comparable_home_value * CALCULATOR_NONMORTGAGE_ANNUAL_RATE / 12
    return np.floor(unrounded + 0.5)


def principal_and_interest_per_100k(rate_percent: pd.Series) -> pd.Series:
    monthly_rate = rate_percent / 1200
    unrounded = 100_000 * monthly_rate * (1 + monthly_rate) ** CALCULATOR_TERM_MONTHS / (
        (1 + monthly_rate) ** CALCULATOR_TERM_MONTHS - 1
    )
    return np.floor(unrounded + 0.5)


def liquid_savings_band(value: float) -> str:
    if value == 0:
        return "no liquid savings"
    if value < 5_000:
        return "less than $5,000"
    if value < 30_000:
        return "$5,000 to $30,000"
    if value < 100_000:
        return "$30,000 to $100,000"
    if value < 500_000:
        return "$100,000 to $500,000"
    return "$500,000 or more"


def debt_band(value: float) -> str:
    if value == 0:
        return "no non-housing debt"
    if value < 1_000:
        return "less than $1,000"
    if value < 5_000:
        return "$1,000 to $5,000"
    if value < 30_000:
        return "$5,000 to $30,000"
    return "$30,000 or more"


validate_source_inputs("sce_financing")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INTERMEDIATE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Fuster and Zafar (2021), released questionnaire and text_statements.do:
# 360-month annuity; deduct 95% of interest at a 34% tax rate; add annual
# maintenance, property tax and insurance of 3.35% of comparable-home value.
# The constants above and the nearest-dollar rule below preserve that formula.
raw_sha256 = sha256_file(RAW_PATH)
raw = pd.read_stata(RAW_PATH, convert_categoricals=False)
required_columns = set(
    SOURCE_SEQUENCE_COLUMNS
    + [
        "userid", "hypoversion", "homevalue", "age", "q4", "q16_6",
        "liquid_savings", "non_house_debt", "college", "male", "q38",
        "hypo_d1", "qtime_hypo_q1", "qtime_hypo_q2", "qtime_hypo_q3",
        "qtime_hypo_q4", "hypo_mpayment_q1", "hypo_mpayment_q2",
        "hypo_mpayment_q3", "hypo_mpayment_q4", "total_numeracy", "q13",
        "prob_move_3yr", "zip_good_inv", "residence_children_under18",
        "residence_children_over18", "hh_income", "house_value", "house_debt",
        "q14", "region", "sce_acs_weights",
    ]
)
missing_columns = sorted(required_columns - set(raw.columns))
if missing_columns:
    raise RuntimeError(f"Raw experiment data are missing required columns: {missing_columns}")
if len(raw) != 1211 or raw["userid"].isna().any() or raw["userid"].duplicated().any():
    raise RuntimeError("Expected 1,211 unique respondent records in FZdata_raw.dta.")

numeric_columns = [
    *SOURCE_SEQUENCE_COLUMNS,
    "hypo_d1",
    "hypo_mpayment_q1",
    "homevalue",
    "age",
    "q38",
    "q4",
    "house_value",
    "house_debt",
    "liquid_savings",
    "q16_6",
    "non_house_debt",
    "q14",
    "hh_income",
    "college",
    "male",
    "residence_children_under18",
    "residence_children_over18",
    "total_numeracy",
    "q13",
    "prob_move_3yr",
    "zip_good_inv",
    "sce_acs_weights",
]
for column in numeric_columns:
    raw[column] = pd.to_numeric(raw[column], errors="coerce")

raw["source_sequence_complete"] = raw[SOURCE_SEQUENCE_COLUMNS].notna().all(axis=1)
raw["calculator_formula_version"] = CALCULATOR_FORMULA_VERSION
raw["calculator_term_months"] = CALCULATOR_TERM_MONTHS
raw["calculator_deductible_interest_share"] = CALCULATOR_DEDUCTIBLE_INTEREST_SHARE
raw["calculator_tax_rate"] = CALCULATOR_TAX_RATE
raw["calculator_nonmortgage_annual_rate"] = CALCULATOR_NONMORTGAGE_ANNUAL_RATE
raw["calculator_nonmortgage_value_basis"] = "comparable_home_value"
raw["target_unit"] = "nominal_usd"
raw["source_file_sha256"] = raw_sha256
raw = raw.copy()

initial_rate = pd.Series(
    np.where(raw["hypoversion"].eq(1), 4.5, np.where(raw["hypoversion"].eq(2), 6.5, np.nan)),
    index=raw.index,
)
alternative_rate = pd.Series(
    np.where(initial_rate.eq(4.5), 6.5, np.where(initial_rate.eq(6.5), 4.5, np.nan)),
    index=raw.index,
)
raw["initial_rate"] = initial_rate
raw["alternative_rate"] = alternative_rate
raw["rate_assignment"] = pd.Series(
    np.where(
        initial_rate.eq(4.5),
        "4.5_percent_first",
        np.where(initial_rate.eq(6.5), "6.5_percent_first", pd.NA),
    ),
    index=raw.index,
    dtype="string",
)
raw["initial_wtp"] = raw["hypo_q1"]
raw["initial_down_payment"] = np.floor(0.20 * raw["initial_wtp"] + 0.5)
raw["initial_effective_monthly_payment"] = raw["hypo_mpayment_q1"]
raw["monthly_principal_interest_per_100k_initial_rate"] = principal_and_interest_per_100k(
    initial_rate
)
raw["monthly_principal_interest_per_100k_alternative_rate"] = principal_and_interest_per_100k(
    alternative_rate
)
raw["calculator_rounding_convention"] = (
    "nearest_whole_dollar_after_combining_all_components"
)
raw["calculator_nonmortgage_component_scope"] = (
    "maintenance_property_taxes_homeowners_insurance_combined"
)
raw["calculator_nonmortgage_component_disaggregation"] = "not_provided_by_frozen_source"
raw["wtp_flexible_original_rate"] = raw["hypo_q2"]
raw["down_payment_flexible_original_rate"] = raw["hypo_d2"]
raw["wtp_flexible_alternative_rate"] = raw["hypo_q3"]
raw["down_payment_flexible_alternative_rate"] = raw["hypo_d3"]
raw["wtp_after_cash_inheritance"] = raw["hypo_q4"]
raw["down_payment_after_cash_inheritance"] = raw["hypo_d4"]
raw["owner"] = raw["q4"].eq(1)
raw["renter"] = raw["q4"].eq(0)
raw["comparable_home_value"] = raw["homevalue"]
raw["current_home_value"] = np.where(raw["owner"], raw["house_value"], 0.0)
raw["housing_debt"] = np.where(raw["owner"], raw["house_debt"], 0.0)
raw["liquid_savings_usd"] = np.where(
    raw["q16_6"].eq(1), 0.0, raw["liquid_savings"] * 1_000.0
)
raw["income_band"] = raw["hh_income"].round().astype("Int64").map(INCOME_BANDS)
raw["credit_score_band"] = raw["q14"].round().astype("Int64").map(CREDIT_SCORE_BANDS)
raw["liquid_savings_band"] = raw["liquid_savings_usd"].map(
    lambda value: liquid_savings_band(float(value))
    if pd.notna(value) and value >= 0
    else pd.NA
)
raw["non_housing_debt_band"] = raw["non_house_debt"].map(
    lambda value: debt_band(float(value)) if pd.notna(value) and value >= 0 else pd.NA
)
raw["tenure"] = pd.Series(
    np.where(raw["owner"], "owner", np.where(raw["renter"], "renter", pd.NA)),
    index=raw.index,
    dtype="string",
)
raw["sex"] = pd.Series(
    np.where(raw["male"].eq(1), "man", np.where(raw["male"].eq(0), "woman", pd.NA)),
    index=raw.index,
    dtype="string",
)
raw["education"] = pd.Series(
    np.where(
        raw["college"].eq(1),
        "college degree or higher",
        np.where(raw["college"].eq(0), "less than a college degree", pd.NA),
    ),
    index=raw.index,
    dtype="string",
)
raw["married"] = raw["q38"].eq(1).where(raw["q38"].notna())
raw["numeracy_score"] = raw["total_numeracy"].astype("Int64")
raw["willingness_to_take_risks"] = raw["q13"].astype("Int64")
raw["probability_move_three_years"] = raw["prob_move_3yr"]
raw["property_good_investment"] = pd.Series(
    np.where(raw["zip_good_inv"].eq(1), "yes", np.where(raw["zip_good_inv"].eq(0), "no", pd.NA)),
    index=raw.index,
    dtype="string",
)
raw["cash_inheritance"] = 100_000.0
raw["scenario_order"] = (
    "initial_fixed_20pct; flexible_original_rate; flexible_alternative_rate; inheritance"
)
raw["weight"] = raw["sce_acs_weights"]
calculator_rows = []
for question, rate, down_payment in [
    (1, initial_rate, raw["hypo_d1"]),
    (2, initial_rate, raw["hypo_d2"]),
    (3, alternative_rate, raw["hypo_d3"]),
    (4, alternative_rate, raw["hypo_d4"]),
]:
    price = pd.to_numeric(raw[f"hypo_q{question}"], errors="coerce")
    down_payment = pd.to_numeric(down_payment, errors="coerce")
    stored = pd.to_numeric(raw[f"hypo_mpayment_q{question}"], errors="coerce")
    required_minimum = np.floor(0.05 * price + 0.5)
    down_payment_rule = (
        (down_payment - 0.20 * price).abs().le(0.01)
        if question == 1
        else down_payment.ge(required_minimum)
    )
    valid_inputs = (
        price.gt(0)
        & down_payment_rule
        & down_payment.le(price)
        & raw["homevalue"].gt(0)
        & rate.notna()
        & stored.notna()
    )
    reproduced = calculator_payment(price, down_payment, raw["homevalue"], rate)
    difference = reproduced - stored
    raw[f"calculator_reproduced_q{question}"] = reproduced
    raw[f"calculator_difference_q{question}"] = difference
    raw[f"calculator_valid_inputs_q{question}"] = valid_inputs
    valid_count = int(valid_inputs.sum())
    exact_matches = int(difference.loc[valid_inputs].eq(0).sum())
    maximum_error = float(difference.loc[valid_inputs].abs().max())
    calculator_rows.append(
        {
            "task_id": TASK_ID,
            "scenario": f"q{question}",
            "valid_source_rows": valid_count,
            "exact_matches": exact_matches,
            "maximum_absolute_error_usd": maximum_error,
            "required_maximum_absolute_error_usd": 0.0,
            "formula_version": CALCULATOR_FORMULA_VERSION,
            "status": "pass" if exact_matches == valid_count and maximum_error == 0 else "fail",
        }
    )
calculator_audit = pd.DataFrame(calculator_rows)
if not calculator_audit["status"].eq("pass").all():
    raise RuntimeError("The recovered calculator formula does not reproduce stored valid inputs.")
calculator_audit.to_csv(OUTPUT_DIR / "00_calculator_numerical_audit.csv", index=False)

raw.to_parquet(INTERMEDIATE_PATH, index=False)
manifest_rows = [
    {
        "task_id": TASK_ID,
        "source_file": str(path.relative_to(PROJECT_ROOT)),
        "sha256": expected,
        "status": "pass",
    }
    for path, expected in [(RAW_PATH, raw_sha256)]
]
manifest_rows.append(
    {
        "task_id": TASK_ID,
        "source_file": str(INTERMEDIATE_PATH.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(INTERMEDIATE_PATH),
        "status": "pass",
    }
)
pd.DataFrame(manifest_rows).to_csv(OUTPUT_DIR / "00_source_manifest.csv", index=False)
pd.DataFrame(
    [
        {
            "task_id": TASK_ID,
            "raw_respondents": len(raw),
            "experiment_assigned_respondents": int(raw["hypoversion"].isin([1, 2]).sum()),
            "complete_initial_and_six_target_answers": int(raw["source_sequence_complete"].sum()),
            "status": "pass",
        }
    ]
).to_csv(OUTPUT_DIR / "00_handoff_summary.csv", index=False)

print(
    f"{TASK_ID}: wrote {len(raw):,} source respondent rows and verified the "
    "questionnaire and calculator contracts."
)
