#!/usr/bin/env python
"""Construct the Fuster--Zafar housing-financing stated-response task table."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.sample_accounting import filter_count

from scripts.utils.table_schema import (
    require_columns,
    write_public_table,
)
from scripts.utils.macro_context import CORE_MACRO_COLUMNS, attach_macro_context
from scripts.utils.sampling import sample_task_records


TASK_ID = "house_sce_financing"
SEED = 42
MAX_PROMPTS = 500_000
SURVEY_YEAR = 2014
SURVEY_MONTH = 2
SURVEY_DATE = pd.Timestamp("2014-02-01")
HANDOFF_PATH = PROJECT_ROOT / "data/intermediate/house_sce_financing.parquet"
RELEASE_CROSSWALK_PATH = PROJECT_ROOT / "data/raw/release_dates/release_sce.csv"
TABULAR_PATH = PROJECT_ROOT / "data/householdbench/tabular/house_sce_financing.csv"
OUTPUT_DIR = PROJECT_ROOT / "output/householdbench/tasks/house_sce_financing"
SAMPLING_DIR = PROJECT_ROOT / "data" / "intermediate" / "householdbench" / TASK_ID
PREPROCESSING_OUTPUT_DIR = PROJECT_ROOT / "output/preprocessing/sce_financing"

STEP_COLUMNS = [
    "task_id",
    "step",
    "step_label",
    "rows_before",
    "rows_after",
    "rows_dropped",
    "consumer_units_after",
    "consumer_unit_weeks_after",
    "consumer_unit_days_after",
]
DROP_DETAIL_COLUMNS = [
    "task_id",
    "step",
    "detail_order",
    "filter_id",
    "pass_condition",
    "fail_count",
]
OUTLIER_COLUMNS = [
    "task_id",
    "variable",
    "role",
    "period",
    "n_reference",
    "p01",
    "p99",
    "n_at_or_below_p01",
    "n_at_or_above_p99",
]

TARGET_COLUMNS = [
    "wtp_flexible_original_rate",
    "down_payment_flexible_original_rate",
    "wtp_flexible_alternative_rate",
    "down_payment_flexible_alternative_rate",
    "wtp_after_cash_inheritance",
    "down_payment_after_cash_inheritance",
]
REQUIRED_SOURCE_COLUMNS = [
    "userid",
    "hypoversion",
    "initial_rate",
    "alternative_rate",
    "rate_assignment",
    "initial_wtp",
    "initial_down_payment",
    "initial_effective_monthly_payment",
    *TARGET_COLUMNS,
    "monthly_principal_interest_per_100k_initial_rate",
    "monthly_principal_interest_per_100k_alternative_rate",
    "comparable_home_value",
    "age",
    "residence_children_under18",
    "residence_children_over18",
    "sex",
    "education",
    "married",
    "income_band",
    "region",
    "tenure",
    "owner",
    "current_home_value",
    "housing_debt",
    "liquid_savings_usd",
    "liquid_savings_band",
    "non_house_debt",
    "non_housing_debt_band",
    "credit_score_band",
    "numeracy_score",
    "willingness_to_take_risks",
    "probability_move_three_years",
    "property_good_investment",
    "cash_inheritance",
    "scenario_order",
    "weight",
    "source_sequence_complete",
    "calculator_formula_version",
    "calculator_term_months",
    "calculator_deductible_interest_share",
    "calculator_tax_rate",
    "calculator_nonmortgage_annual_rate",
    "calculator_nonmortgage_value_basis",
    "source_file_sha256",
]


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def direct_dollar_minimum(price: pd.Series) -> pd.Series:
    """Replicate JavaScript Math.round(0.05 * price) for non-negative dollars."""
    return np.floor(0.05 * price + 0.5)


def add_step(rows: list[dict[str, object]], step: int, label: str, before: int, after: int) -> None:
    rows.append(
        {
            "task_id": TASK_ID,
            "step": step,
            "step_label": label,
            "rows_before": int(before),
            "rows_after": int(after),
            "rows_dropped": int(before - after),
            "consumer_units_after": int(after),
            "consumer_unit_weeks_after": int(after),
            "consumer_unit_days_after": int(after),
        }
    )


def add_drop_detail(
    rows: list[dict[str, object]],
    step: int,
    detail_order: int,
    filter_id: str,
    pass_condition: str,
    mask: pd.Series,
) -> None:
    rows.append(
        filter_count(
            TASK_ID, step, detail_order,
            filter_id, pass_condition, int((~mask.fillna(False)).sum()),
        )
    )


for required_path in [
    HANDOFF_PATH,
    RELEASE_CROSSWALK_PATH,
    PREPROCESSING_OUTPUT_DIR / "00_source_manifest.csv",
]:
    if not required_path.is_file():
        raise RuntimeError(f"Required source material is missing: {required_path}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABULAR_PATH.parent.mkdir(parents=True, exist_ok=True)

step_rows: list[dict[str, object]] = []
drop_detail_rows: list[dict[str, object]] = []

# 1. Start from the one-row-per-respondent preprocessing handoff.
raw = pd.read_parquet(HANDOFF_PATH)
require_columns(raw, REQUIRED_SOURCE_COLUMNS, label=str(HANDOFF_PATH))
if raw["userid"].isna().any() or raw["userid"].duplicated().any():
    raise RuntimeError("The preprocessing handoff must contain one nonmissing unique userid per record.")
if len(raw) != 1211 or int(raw["hypoversion"].isin([1, 2]).sum()) != 1200:
    raise RuntimeError("The preprocessing handoff does not reproduce the source and assigned experiment universe.")
raw["respondent_id"] = "fz_" + numeric(raw, "userid").astype("Int64").astype(str)
raw["group_id"] = raw["respondent_id"]
raw["source_record_id"] = raw["respondent_id"]
raw["preprocessing_validated"] = 1
add_drop_detail(
    drop_detail_rows, 1, 1, "validated_financing_respondents",
    "start from every unique respondent in the validated financing-experiment intermediate",
    pd.Series(True, index=raw.index),
)
add_step(step_rows, 1, "Start from source observations", len(raw), len(raw))

# 2. Attach released timing and public macro information without dropping observations.
release_crosswalk = pd.read_csv(RELEASE_CROSSWALK_PATH)
require_columns(
    release_crosswalk,
    ["module", "survey_year", "survey_month", "release_date", "source", "comments"],
    label=str(RELEASE_CROSSWALK_PATH),
)
release_row = release_crosswalk.loc[
    release_crosswalk["module"].eq("housing")
    & release_crosswalk["survey_year"].eq(SURVEY_YEAR)
    & release_crosswalk["survey_month"].eq(SURVEY_MONTH)
].copy()
if len(release_row) != 1:
    raise RuntimeError("Expected exactly one conservative SCE housing release-date row for February 2014.")
release_date = pd.to_datetime(release_row.iloc[0]["release_date"], errors="coerce")
if pd.isna(release_date):
    raise RuntimeError("The selected SCE housing release date is invalid.")
pool = raw.copy()
pool["survey_year"] = SURVEY_YEAR
pool["survey_month"] = SURVEY_MONTH
pool["survey_date"] = SURVEY_DATE
pool["release_date"] = release_date
pool["release_date_source"] = "SCE housing 18-month conservative proxy"
pool["release_date_rule"] = str(release_row.iloc[0]["comments"])
pool["macro_origin_q_index"] = SURVEY_YEAR * 4 + 1
before = len(pool)
pool = attach_macro_context(
    pool,
    origin_q_index_col="macro_origin_q_index",
    required_columns=CORE_MACRO_COLUMNS,
)
if len(pool) != before:
    raise RuntimeError("Macro attachment changed the source row count.")
pool["release_date_merge"] = 1
pool["macro_context_merge"] = 1
add_drop_detail(
    drop_detail_rows, 2, 1, "release_and_macro_context_attached",
    "attach the February 2014 SCE release date and the common pre-survey macro block to every respondent",
    pd.Series(True, index=pool.index),
)
add_step(step_rows, 2, "Attach additional sources", before, len(pool))

# 3. This cross-sectional task has no leads or lags to construct.
before = len(pool)
add_drop_detail(
    drop_detail_rows, 3, 1, "cross_section_has_no_leads_or_lags",
    "retain every respondent because this same-session experiment requires no lead or lag construction",
    pd.Series(True, index=pool.index),
)
add_step(step_rows, 3, "Construct leads and lags", before, len(pool))

# 4. The source file is already the financing-experiment universe.
before = len(pool)
add_drop_detail(
    drop_detail_rows,
    4,
    1,
    "source_financing_experiment_universe",
    "all rows in the source experiment file are in universe; assignment nonresponse is handled by target missingness",
    pd.Series(True, index=pool.index),
)
add_step(step_rows, 4, "Restrict universe", before, len(pool))

# 5. Require the six direct numeric targets.
before = len(pool)
direct_target_observed = pool[TARGET_COLUMNS].notna().all(axis=1)
add_drop_detail(
    drop_detail_rows,
    5,
    1,
    "all_six_scored_answers_observed",
    "all six later WTP and down-payment answers used as scored targets are observed",
    direct_target_observed,
)
pool = pool.loc[direct_target_observed].copy()
pool["direct_target_observed"] = True
predictor_reference_pool = pool.copy()
add_step(step_rows, 5, "Require observed targets", before, len(pool))

# 6. Require only information available before the three later scored scenarios.
before = len(pool)
owner_housing_context = (~pool["owner"]) | (
    pool[["current_home_value", "housing_debt"]].notna().all(axis=1)
    & pool["current_home_value"].gt(0)
    & pool["housing_debt"].ge(0)
)
predictor_requirements = [
    ("initial_wtp_observed", "the initial stated maximum home price is observed and positive", pool["initial_wtp"].gt(0)),
    ("valid_randomized_rate_assignment", "the preprocessed initial and alternative mortgage rates are the two experiment rates", pool["initial_rate"].isin([4.5, 6.5]) & pool["alternative_rate"].isin([4.5, 6.5])),
    ("initial_calculator_output", "the initial calculator-implied effective monthly payment is observed and positive", pool["initial_effective_monthly_payment"].gt(0)),
    ("comparable_home_value", "the scenario's comparable-home value is observed and positive", pool["comparable_home_value"].gt(0)),
    ("demographic_context", "age, sex, education, marital status, children, income band, and region are observed", pool[["age", "sex", "education", "married", "residence_children_under18", "residence_children_over18", "income_band", "region"]].notna().all(axis=1)),
    ("tenure_and_owner_housing_context", "tenure is owner or renter; owners have current home value and housing debt", pool["tenure"].isin(["owner", "renter"]) & owner_housing_context),
    ("financial_context", "liquid-savings band, non-housing-debt band, and credit-score band are observed", pool[["liquid_savings_band", "non_housing_debt_band", "credit_score_band"]].notna().all(axis=1)),
    ("numeracy_and_preferences", "numeracy, willingness to take risks, three-year moving probability, and ZIP-code investment view are observed", pool[["numeracy_score", "willingness_to_take_risks", "probability_move_three_years", "property_good_investment"]].notna().all(axis=1)),
    ("release_and_macro_context", "release date and all pre-anchor macro fields are observed", pool["release_date"].notna() & pool[CORE_MACRO_COLUMNS].notna().all(axis=1)),
]
predictor_complete = pd.Series(True, index=pool.index)
for order, (filter_id, condition, mask) in enumerate(predictor_requirements, start=1):
    add_drop_detail(drop_detail_rows, 6, order, filter_id, condition, mask)
    predictor_complete &= mask.fillna(False)
predictor_reference_pool["approved_predictor_complete"] = predictor_complete
pool = pool.loc[predictor_complete].copy()
pool["predictor_complete"] = True
add_step(step_rows, 6, "Require observed predictors", before, len(pool))

# 7. Enforce the questionnaire's direct dollar constraints and source-supported logical validity.
before = len(pool)
positive_wtp = pool[["initial_wtp", "wtp_flexible_original_rate", "wtp_flexible_alternative_rate", "wtp_after_cash_inheritance"]].gt(0).all(axis=1)
q2_minimum = direct_dollar_minimum(pool["wtp_flexible_original_rate"])
q3_minimum = direct_dollar_minimum(pool["wtp_flexible_alternative_rate"])
q4_minimum = direct_dollar_minimum(pool["wtp_after_cash_inheritance"])
minimum_down_payment = (
    pool["down_payment_flexible_original_rate"].ge(q2_minimum)
    & pool["down_payment_flexible_alternative_rate"].ge(q3_minimum)
    & pool["down_payment_after_cash_inheritance"].ge(q4_minimum)
)
down_payment_not_above_price = (
    pool["down_payment_flexible_original_rate"].le(pool["wtp_flexible_original_rate"])
    & pool["down_payment_flexible_alternative_rate"].le(pool["wtp_flexible_alternative_rate"])
    & pool["down_payment_after_cash_inheritance"].le(pool["wtp_after_cash_inheritance"])
)
valid_context_ranges = (
    pool["age"].between(18, 100)
    & pool["residence_children_under18"].ge(0)
    & pool["residence_children_over18"].ge(0)
    & pool["non_house_debt"].ge(0)
    & pool["liquid_savings_usd"].ge(0)
    & pool["numeracy_score"].between(0, 5)
    & pool["willingness_to_take_risks"].between(1, 10)
    & pool["probability_move_three_years"].between(0, 100)
    & pool["property_good_investment"].isin(["yes", "no"])
)
logical_requirements = [
    ("positive_direct_wtp", "all direct WTP answers are strictly positive dollars", positive_wtp),
    ("five_percent_minimum_down_payment", "each direct later down payment is at least JavaScript-rounded 5 percent of its stated price", minimum_down_payment),
    ("down_payment_not_above_purchase_price", "each direct later down payment is no greater than its stated price", down_payment_not_above_price),
    ("context_ranges", "age is 18--100 and children, financial, numeracy, preference, moving-probability, and investment-view fields satisfy source-supported ranges", valid_context_ranges),
]
logical_valid = pd.Series(True, index=pool.index)
for order, (filter_id, condition, mask) in enumerate(logical_requirements, start=1):
    add_drop_detail(drop_detail_rows, 7, order, filter_id, condition, mask)
    logical_valid &= mask.fillna(False)
pool = pool.loc[logical_valid].copy()
pool["logical_valid"] = True
add_step(step_rows, 7, "Apply logical and validity filters", before, len(pool))

# 8. Apply the standard strict-interior 1st/99th-percentile rule to unbounded levels.
before = len(pool)
pool["selection_period"] = pool["survey_date"].dt.strftime("%Y-%m")
trim_columns = {
    "initial_wtp": "predictor",
    "wtp_flexible_original_rate": "target",
    "down_payment_flexible_original_rate": "target",
    "wtp_flexible_alternative_rate": "target",
    "down_payment_flexible_alternative_rate": "target",
    "wtp_after_cash_inheritance": "target",
    "down_payment_after_cash_inheritance": "target",
    "comparable_home_value": "predictor",
}
outlier_rows: list[dict[str, object]] = []
outlier_fail = pd.Series(False, index=pool.index)
outlier_trigger_masks: dict[str, pd.Series] = {}
for column, role in trim_columns.items():
    values = numeric(pool, column)
    if values.isna().any() or values.le(0).any():
        raise RuntimeError(f"{column} must be complete and positive before strict-interior trimming.")
    p01 = float(values.quantile(0.01, interpolation="linear"))
    p99 = float(values.quantile(0.99, interpolation="linear"))
    if not p01 < p99:
        raise RuntimeError(f"Degenerate strict-interior cutoffs for {column}.")
    below_or_equal = values.le(p01)
    above_or_equal = values.ge(p99)
    triggered = below_or_equal | above_or_equal
    outlier_trigger_masks[column] = triggered
    outlier_fail |= triggered
    outlier_rows.append(
        {
            "task_id": TASK_ID,
            "variable": column,
            "role": role,
            "period": "2014-02",
            "n_reference": int(len(values)),
            "p01": p01,
            "p99": p99,
            "n_at_or_below_p01": int(below_or_equal.sum()),
            "n_at_or_above_p99": int(above_or_equal.sum()),
        }
    )
add_drop_detail(
    drop_detail_rows,
    8,
    1,
    "strict_interior_1st_99th_percentile",
    "all selected unbounded levels lie strictly inside their February 2014 1st/99th-percentile cutoffs",
    ~outlier_fail,
)
trigger_rows = []
for row_index in pool.index[outlier_fail]:
    triggered_variables = [
        column for column in trim_columns if bool(outlier_trigger_masks[column].loc[row_index])
    ]
    trigger_rows.append(
        {
            "task_id": TASK_ID,
            "respondent_id": pool.at[row_index, "respondent_id"],
            "selection_period": pool.at[row_index, "selection_period"],
            "trigger_count": len(triggered_variables),
            "trigger_variables": ";".join(triggered_variables),
            **{
                f"trigger_{column}": int(column in triggered_variables)
                for column in trim_columns
            },
        }
    )
trigger_audit = pd.DataFrame(trigger_rows).sort_values("respondent_id", kind="mergesort")
if trigger_audit.empty or trigger_audit["respondent_id"].duplicated().any():
    raise RuntimeError("The simultaneous p1/p99 trigger audit is empty or duplicates respondents.")
trigger_audit.to_csv(OUTPUT_DIR / "01_outlier_trigger_list.csv", index=False)
pool = pool.loc[~outlier_fail].copy()
pool["outlier_retained"] = True
add_step(step_rows, 8, "Apply outlier trimming", before, len(pool))

# 9. Select deterministically after all restrictions. One record per person preserves the split group.
eligible_rows = len(pool)
if not pool["respondent_id"].is_unique or not pool["group_id"].eq(pool["respondent_id"]).all():
    raise RuntimeError("The task requires one record and one split group per respondent.")
# Public keys are fixed before sampling; source keys retain their sort types.
pool["id"] = pool["respondent_id"].astype("string")
pool["time"] = pd.to_datetime(pool["survey_date"], errors="raise").dt.strftime("%Y-%m-%d")
pool["release_date"] = pd.to_datetime(pool["release_date"], errors="raise").dt.strftime("%Y-%m-%d")
selected, _, sampling_summary = sample_task_records(
    pool,
    task_id=TASK_ID,
    output_dir=SAMPLING_DIR,
    output_sort_cols=["survey_date", "respondent_id"],
)
if selected.empty:
    raise RuntimeError("No selected records remain after deterministic selection.")
add_drop_detail(
    drop_detail_rows, 9, 1, "deterministic_export_cap",
    "apply the registered deterministic sampling order and export all eligible respondents because the sample is below the task cap",
    pd.Series(True, index=pool.index),
)
add_step(step_rows, 9, "Export sample", eligible_rows, len(selected))

selected["target_unit"] = "nominal_usd"

table_columns = [
    "respondent_id",
    "group_id",
    "source_record_id",
    "survey_date",
    "survey_year",
    "survey_month",
    "selection_period",
    "release_date",
    "release_date_source",
    "release_date_rule",
    "source_file_sha256",
    "release_date_merge",
    "macro_context_merge",
    "preprocessing_validated",
    "source_sequence_complete",
    "direct_target_observed",
    "predictor_complete",
    "logical_valid",
    "outlier_retained",
    "rate_assignment",
    "scenario_order",
    "initial_rate",
    "alternative_rate",
    "cash_inheritance",
    "initial_wtp",
    "initial_down_payment",
    "initial_effective_monthly_payment",
    "calculator_formula_version",
    "calculator_term_months",
    "calculator_deductible_interest_share",
    "calculator_tax_rate",
    "calculator_nonmortgage_annual_rate",
    "calculator_nonmortgage_value_basis",
    "calculator_rounding_convention",
    "calculator_nonmortgage_component_scope",
    "calculator_nonmortgage_component_disaggregation",
    "monthly_principal_interest_per_100k_initial_rate",
    "monthly_principal_interest_per_100k_alternative_rate",
    "comparable_home_value",
    "tenure",
    "current_home_value",
    "housing_debt",
    "age",
    "sex",
    "education",
    "married",
    "residence_children_under18",
    "residence_children_over18",
    "income_band",
    "region",
    "liquid_savings_usd",
    "liquid_savings_band",
    "non_house_debt",
    "non_housing_debt_band",
    "credit_score_band",
    "numeracy_score",
    "willingness_to_take_risks",
    "probability_move_three_years",
    "property_good_investment",
    "weight",
    *CORE_MACRO_COLUMNS,
    *TARGET_COLUMNS,
    "target_unit",
]
require_columns(selected, table_columns, label="selected construction frame")
table = selected[table_columns].copy()
if table[TARGET_COLUMNS].isna().any().any() or table[TARGET_COLUMNS].le(0).any().any():
    raise RuntimeError("The final public table contains invalid direct numeric targets.")
if not table["respondent_id"].is_unique:
    raise RuntimeError("The final public table must have one row per respondent.")
for date_column in ["survey_date", "release_date"]:
    table[date_column] = pd.to_datetime(table[date_column], errors="raise").dt.date.astype(str)
table["id"] = table["respondent_id"]
table["time"] = table["survey_date"]
write_public_table(table, task_id=TASK_ID, public_path=TABULAR_PATH)

source_crosswalk_rows = [
    ("respondent_id", "userid", "prefix the local numeric identifier with fz_", "identifier", "source record", "string"),
    ("initial_rate", "hypoversion", "version 1 maps to 4.5; version 2 maps to 6.5", "predictor", "assigned before initial answer", "percent"),
    ("alternative_rate", "hypoversion", "switch the assigned initial rate by exactly two percentage points", "scenario", "fixed later sequence", "percent"),
    ("initial_wtp", "hypo_q1", "direct initial final answer", "predictor", "observed before all six targets", "nominal USD"),
    ("initial_down_payment", "hypo_q1", "nearest-dollar 20 percent of initial WTP", "calculated predictor", "calculated from initial answer only", "nominal USD"),
    ("initial_effective_monthly_payment", "hypo_mpayment_q1", "stored calculator-displayed effective total monthly payment", "predictor", "observed before all six targets", "nominal USD per month"),
    ("wtp_flexible_original_rate", "hypo_q2", "direct final answer", "target 1", "later scenario 1", "nominal USD"),
    ("down_payment_flexible_original_rate", "hypo_d2", "direct final answer", "target 2", "later scenario 1", "nominal USD"),
    ("wtp_flexible_alternative_rate", "hypo_q3", "direct final answer", "target 3", "later scenario 2", "nominal USD"),
    ("down_payment_flexible_alternative_rate", "hypo_d3", "direct final answer", "target 4", "later scenario 2", "nominal USD"),
    ("wtp_after_cash_inheritance", "hypo_q4", "direct final answer", "target 5", "later scenario 3", "nominal USD"),
    ("down_payment_after_cash_inheritance", "hypo_d4", "direct final answer", "target 6", "later scenario 3", "nominal USD"),
    ("comparable_home_value", "homevalue", "source-rounded comparable-home value", "predictor", "known before initial answer", "nominal USD"),
    ("age", "age", "retain exact source age after the benchmark's 18--100 validity screen", "predictor", "pre-module household fact", "years"),
    ("sex", "male", "1 maps to man; 0 maps to woman", "predictor", "pre-module household fact", "category"),
    ("education", "college", "1 maps to college degree or higher; 0 maps to less than a college degree", "predictor", "pre-module household fact", "category"),
    ("married", "q38", "1 maps to married; other valid code maps to not married", "predictor", "pre-module household fact", "category"),
    ("children", "residence_children_under18; residence_children_over18", "render both resident-child counts in natural language", "predictor", "pre-module household fact", "counts"),
    ("income_band", "hh_income", "map the eleven direct source categories to natural-language dollar bands", "predictor", "pre-module household fact", "category"),
    ("region", "region", "retain source region label", "predictor", "pre-module household fact", "category"),
    ("tenure", "q4", "1 maps to owner; 0 maps to renter", "predictor and prompt subtemplate", "pre-module household fact", "category"),
    ("current_home_value", "house_value", "retain for owners; omit from renter prompt", "predictor", "pre-module household fact", "nominal USD"),
    ("housing_debt", "house_debt", "retain outstanding housing-related debt for owners; omit from renter prompt", "predictor", "pre-module household fact", "nominal USD"),
    ("liquid_savings_band", "liquid_savings; q16_6", "set no-liquid-wealth responses to zero and map to frozen bands", "predictor", "pre-module household fact", "category"),
    ("non_housing_debt_band", "non_house_debt", "map source level to frozen bands", "predictor", "pre-module household fact", "category"),
    ("credit_score_band", "q14", "map the six direct categories to natural-language bands", "predictor", "pre-module household fact", "category"),
    ("numeracy_score", "total_numeracy", "retain the direct count of correct numeracy answers", "predictor", "pre-module household fact", "0--5 correct"),
    ("willingness_to_take_risks", "q13", "retain the direct willingness-to-take-risks response", "predictor", "pre-module household fact", "1--10 scale"),
    ("probability_move_three_years", "prob_move_3yr", "retain the direct probability of moving within three years", "predictor", "pre-module household fact", "percent"),
    ("property_good_investment", "zip_good_inv", "1 maps to yes and 0 maps to no", "predictor", "pre-module household fact", "category"),
    ("cash_inheritance", "FZsurveyquestions.pdf", "fixed scenario amount of 100000", "scenario", "later scenario 3, not a response", "nominal USD"),
    ("macro_context", "data/intermediate/householdbench_macro_context.parquet", "attach the four completed quarters and five-year references using the frozen pre-module quarter", "predictor", "public information before the module", "percent"),
    ("weight", "sce_acs_weights", "retain public respondent weight for construction descriptives", "non-model-facing metadata", "source record", "weight"),
]
pd.DataFrame(
    source_crosswalk_rows,
    columns=[
        "analysis_variable", "source_field", "transformation", "role", "timing", "unit"
    ],
).assign(task_id=TASK_ID)[
    ["task_id", "analysis_variable", "source_field", "transformation", "role", "timing", "unit"]
].to_csv(OUTPUT_DIR / "01_source_to_analysis_crosswalk.csv", index=False)


def coverage(mask: pd.Series, universe: pd.Series | None = None) -> tuple[int, int, float]:
    relevant = pd.Series(True, index=predictor_reference_pool.index) if universe is None else universe
    denominator = int(relevant.sum())
    numerator = int((mask.fillna(False) & relevant).sum())
    return numerator, denominator, numerator / denominator


predictor_specs = [
    ("age", "age", predictor_reference_pool["age"].notna(), None, "Age", "years", "require source-valid value", "included", ""),
    ("sex", "male", predictor_reference_pool["sex"].notna(), None, "Sex", "category", "require source-valid value", "included", ""),
    ("education", "college", predictor_reference_pool["education"].notna(), None, "Highest completed education", "category", "require source-valid value", "included", ""),
    ("married", "q38", predictor_reference_pool["married"].notna(), None, "Marital status", "category", "require source-valid value", "included", "partner status is not separately observed"),
    ("children", "residence_children_under18; residence_children_over18", predictor_reference_pool[["residence_children_under18", "residence_children_over18"]].notna().all(axis=1), None, "Children living with you", "counts", "require both counts", "included", ""),
    ("income_band", "hh_income", predictor_reference_pool["income_band"].notna(), None, "Household income", "category", "require source-valid band", "included", ""),
    ("region", "region", predictor_reference_pool["region"].notna(), None, "Region", "category", "require source-valid value", "included", ""),
    ("tenure", "q4", predictor_reference_pool["tenure"].isin(["owner", "renter"]), None, "Housing tenure", "category", "require owner or renter", "included", ""),
    ("comparable_home_value", "homevalue", predictor_reference_pool["comparable_home_value"].gt(0), None, "Typical comparable-home value", "nominal USD", "require positive value", "included", ""),
    ("current_home_value", "house_value", predictor_reference_pool["current_home_value"].gt(0), predictor_reference_pool["owner"], "Current home value", "nominal USD", "require for owners; omit for renters", "included", "owner-only field"),
    ("housing_debt", "house_debt", predictor_reference_pool["housing_debt"].ge(0), predictor_reference_pool["owner"], "Outstanding housing-related debt", "nominal USD", "require for owners; omit for renters", "included", "source does not isolate mortgage balance"),
    ("liquid_savings_band", "liquid_savings; q16_6", predictor_reference_pool["liquid_savings_band"].notna(), None, "Liquid savings", "category", "require source-valid band", "included", ""),
    ("non_housing_debt_band", "non_house_debt", predictor_reference_pool["non_housing_debt_band"].notna(), None, "Non-housing debt", "category", "require source-valid band", "included", ""),
    ("credit_score_band", "q14", predictor_reference_pool["credit_score_band"].notna(), None, "Credit situation", "category", "require source-valid category", "included", ""),
    ("numeracy_score", "total_numeracy", predictor_reference_pool["numeracy_score"].notna(), None, "Numeracy score", "0--5 correct", "require direct response", "included", ""),
    ("willingness_to_take_risks", "q13", predictor_reference_pool["willingness_to_take_risks"].notna(), None, "Willingness to take risks", "1--10 scale", "require direct response", "included", ""),
    ("probability_move_three_years", "prob_move_3yr", predictor_reference_pool["probability_move_three_years"].notna(), None, "Probability of moving within three years", "percent", "require direct response", "included", ""),
    ("property_good_investment", "zip_good_inv", predictor_reference_pool["property_good_investment"].notna(), None, "Property in ZIP code is a good investment", "category", "require direct response", "included", ""),
    ("initial_rate", "hypoversion", predictor_reference_pool["initial_rate"].isin([4.5, 6.5]), None, "Initial 30-year mortgage rate", "percent", "require valid preprocessed assignment", "included", ""),
    ("initial_wtp", "hypo_q1", predictor_reference_pool["initial_wtp"].gt(0), None, "Your stated maximum price", "nominal USD", "require direct initial answer", "included", ""),
    ("initial_down_payment", "hypo_q1", predictor_reference_pool["initial_down_payment"].ge(0), None, "Calculated 20 percent down payment", "nominal USD", "calculate from initial answer", "included", "not a direct answer"),
    ("initial_effective_monthly_payment", "hypo_mpayment_q1", predictor_reference_pool["initial_effective_monthly_payment"].gt(0), None, "Calculator-displayed effective total monthly payment", "nominal USD per month", "require stored positive calculator output", "included", "calculator output, not a direct answer"),
]
predictor_manifest_rows = []
for (
    model_field, source_field, valid_mask, universe_mask, prompt_label, unit,
    missing_rule, decision, note,
) in predictor_specs:
    valid_rows, relevant_rows, valid_share = coverage(valid_mask, universe_mask)
    predictor_manifest_rows.append(
        {
            "task_id": TASK_ID,
            "model_field": model_field,
            "source_field": source_field,
            "transformation": next(
                row[2] for row in source_crosswalk_rows if row[0] == model_field
            ) if any(row[0] == model_field for row in source_crosswalk_rows) else "direct recode documented in constructor",
            "timing": "known before the six held-out later answers",
            "unit": unit,
            "missing_value_rule": missing_rule,
            "exact_prompt_label": prompt_label,
            "coverage_universe": "owners" if universe_mask is not None else "complete target sample",
            "source_valid_rows": valid_rows,
            "relevant_rows": relevant_rows,
            "source_valid_share": valid_share,
            "decision": decision,
            "note": note,
        }
    )
for model_field, prompt_label, note in [
    ("household_size", "Household size", "not present in the frozen strategic respondent file; not inferred from marital status and children"),
    ("employment_status", "Employment status", "not present in the frozen strategic respondent file"),
    ("current_rent", "Current rent", "not present in the frozen strategic respondent file"),
    ("current_mortgage_rate", "Current mortgage rate", "not present in the frozen strategic respondent file"),
]:
    predictor_manifest_rows.append(
        {
            "task_id": TASK_ID,
            "model_field": model_field,
            "source_field": "not available",
            "transformation": "none",
            "timing": "not applicable",
            "unit": "not available",
            "missing_value_rule": "omit from the user prompt",
            "exact_prompt_label": prompt_label,
            "coverage_universe": "complete target sample",
            "source_valid_rows": 0,
            "relevant_rows": len(predictor_reference_pool),
            "source_valid_share": 0.0,
            "decision": "omitted_source_unavailable",
            "note": note,
        }
    )
pd.DataFrame(predictor_manifest_rows).to_csv(
    OUTPUT_DIR / "01_predictor_manifest.csv", index=False
)

prefield = predictor_reference_pool
postfield = prefield.loc[prefield["approved_predictor_complete"]]
composition_fields = ["rate_assignment", "tenure", "sex", "education", "married"]
composition_shifts = []
for column in composition_fields:
    pre_shares = prefield[column].value_counts(normalize=True, dropna=False)
    post_shares = postfield[column].value_counts(normalize=True, dropna=False)
    categories = pre_shares.index.union(post_shares.index)
    for category in categories:
        composition_shifts.append(
            abs(float(pre_shares.get(category, 0.0)) - float(post_shares.get(category, 0.0)))
        )
predictor_completeness_audit = pd.DataFrame(
    [
        {
            "task_id": TASK_ID,
            "otherwise_eligible_rows": len(prefield),
            "complete_approved_predictor_rows": len(postfield),
            "rows_removed": len(prefield) - len(postfield),
            "share_removed": 1 - len(postfield) / len(prefield),
            "maximum_core_composition_shift": max(composition_shifts),
            "maximum_allowed_share_removed": 0.10,
            "maximum_allowed_composition_shift": 0.05,
            "status": "pass" if (1 - len(postfield) / len(prefield) <= 0.10 and max(composition_shifts) <= 0.05) else "fail",
        }
    ]
)
if not predictor_completeness_audit["status"].eq("pass").all():
    raise RuntimeError("The approved predictor completeness rule failed.")
predictor_completeness_audit.to_csv(
    OUTPUT_DIR / "01_predictor_completeness_audit.csv", index=False
)

calculator_assumptions = pd.DataFrame(
    [
        ("mortgage_term_and_type", "30-year fixed-rate mortgage with 360 monthly payments", "FZsurveyquestions.pdf and text_statements.do", "pass"),
        ("principal_and_interest", "fixed monthly annuity payment on purchase price less down payment", "text_statements.do", "pass"),
        ("tax_treatment", "subtract estimated tax savings using 95 percent of mortgage interest and a 34 percent tax rate", "text_statements.do and FZusercost.xlsx B3", "pass"),
        ("maintenance_property_tax_insurance", "one combined annual amount equal to 3.35 percent of comparable-home value; no separate component rates are supplied", "FZsurveyquestions.pdf, text_statements.do, and FZusercost.xlsx B5", "pass"),
        ("rounding", "round the final effective total monthly payment to the nearest whole dollar", "questionnaire JavaScript and released Stata code", "pass"),
        ("worked_principal_interest_examples", "$507 per month per $100,000 at 4.5 percent and $632 at 6.5 percent", "360-month annuity implied by the frozen calculator contract", "pass"),
    ],
    columns=["assumption", "source_exact_description", "evidence", "status"],
)
calculator_assumptions.insert(0, "task_id", TASK_ID)
calculator_assumptions.to_csv(OUTPUT_DIR / "01_calculator_assumptions_audit.csv", index=False)

step_frame = pd.DataFrame(step_rows, columns=STEP_COLUMNS)
step_frame[["task_id", "step", "step_label", "rows_before", "rows_after", "rows_dropped"]].to_csv(
    OUTPUT_DIR / "01_sample_construction_steps.csv", index=False
)
step_frame.to_csv(OUTPUT_DIR / "01_sample_construction_entity_counts.csv", index=False)
pd.DataFrame(drop_detail_rows, columns=DROP_DETAIL_COLUMNS).to_csv(
    OUTPUT_DIR / "01_sample_construction_drop_details.csv", index=False
)
pd.DataFrame(outlier_rows, columns=OUTLIER_COLUMNS).to_csv(
    OUTPUT_DIR / "01_outlier_cutoffs.csv", index=False
)

scenario_order = selected.groupby(["initial_rate", "alternative_rate"], dropna=False).size().reset_index(name="records")
scenario_order["scenario_order"] = "Q1 fixed 20 percent; Q2 flexible original rate; Q3 flexible alternative rate; Q4 inheritance"
scenario_order["valid_two_point_switch"] = (scenario_order["initial_rate"] - scenario_order["alternative_rate"]).abs().eq(2.0)
if not scenario_order["valid_two_point_switch"].all() or set(scenario_order["initial_rate"]) != {4.5, 6.5}:
    raise RuntimeError("The random assignment does not yield the required two-point rate switch.")
scenario_order.to_csv(OUTPUT_DIR / "01_experiment_scenario_order_audit.csv", index=False)

bound_rows = []
for label, price, down_payment in [
    ("flexible_original_rate", "wtp_flexible_original_rate", "down_payment_flexible_original_rate"),
    ("flexible_alternative_rate", "wtp_flexible_alternative_rate", "down_payment_flexible_alternative_rate"),
    ("cash_inheritance", "wtp_after_cash_inheritance", "down_payment_after_cash_inheritance"),
]:
    minimum = direct_dollar_minimum(selected[price])
    bound_rows.append(
        {
            "task_id": TASK_ID,
            "scenario": label,
            "records": int(len(selected)),
            "nonpositive_wtp": int(selected[price].le(0).sum()),
            "down_payment_below_rounded_five_percent_minimum": int(selected[down_payment].lt(minimum).sum()),
            "down_payment_above_price": int(selected[down_payment].gt(selected[price]).sum()),
            "target_unit": "nominal_usd",
        }
    )
bound_audit = pd.DataFrame(bound_rows)
if bound_audit[["nonpositive_wtp", "down_payment_below_rounded_five_percent_minimum", "down_payment_above_price"]].to_numpy().sum() != 0:
    raise RuntimeError("The selected table contains a direct-response bound violation.")
bound_audit.to_csv(OUTPUT_DIR / "01_experiment_response_bounds_audit.csv", index=False)

preprocessing_manifest = pd.read_csv(PREPROCESSING_OUTPUT_DIR / "00_source_manifest.csv")
require_columns(
    preprocessing_manifest,
    ["task_id", "source_file", "sha256", "status"],
    label="preprocessing source manifest",
)
if (
    preprocessing_manifest.empty
    or not preprocessing_manifest["task_id"].eq(TASK_ID).all()
    or not preprocessing_manifest["status"].eq("pass").all()
):
    raise RuntimeError("The preprocessing source manifest is empty, belongs to another task, or contains a failed source.")

lineage_metadata = {
    "FZdata_raw.dta": (
        "source_microdata",
        "one row per local userid; direct WTP/down-payment and hidden-payment fields",
    ),
    "FZsurveyquestions.pdf": (
        "exact_questionnaire",
        "routing, direct dollar fields, calculator availability and component wording",
    ),
    "FZusercost.xlsx": (
        "user_cost_workbook",
        "cells B3:B7 and formula cells versioned in the preprocessing audit",
    ),
    "text_statements.do": (
        "released_payment_code",
        "360-month formula, 95 percent deductible share, 34 percent tax rate, and 3.35 percent annual nonmortgage cost",
    ),
    "house_sce_financing.parquet": (
        "preprocessing_handoff",
        "one frozen row per respondent; task constructor does not read raw microdata directly",
    ),
}
lineage_rows = []
for record in preprocessing_manifest.itertuples(index=False):
    source_name = Path(record.source_file).name
    if source_name not in lineage_metadata:
        raise RuntimeError(f"Unexpected source in the preprocessing manifest: {record.source_file}")
    source_role, evidence = lineage_metadata[source_name]
    lineage_rows.append(
        {
            "task_id": TASK_ID,
            "source_role": source_role,
            "source_file": record.source_file,
            "sha256": record.sha256,
            "evidence": evidence,
            "status": record.status,
        }
    )
pd.DataFrame(lineage_rows).to_csv(OUTPUT_DIR / "01_source_lineage.csv", index=False)
pd.DataFrame(
    [
        {
            "task_id": TASK_ID,
            "group_id_definition": "respondent_id",
            "selected_rows": int(len(table)),
            "unique_groups": int(table["group_id"].nunique()),
            "groups_with_more_than_one_row": int((table.groupby("group_id").size() > 1).sum()),
            "proposed_split_rule": "keep all records with the same group_id in the same downstream split",
        }
    ]
).to_csv(OUTPUT_DIR / "01_group_split_audit.csv", index=False)

print(
    f"{TASK_ID}: wrote {len(table):,} selected rows from {eligible_rows:,} eligible rows; "
)
