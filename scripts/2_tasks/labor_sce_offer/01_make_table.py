#!/usr/bin/env python
"""Build the tabular CSV for the SCE first job-offer decision task."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.sample_accounting import construction_step, filter_count

from scripts.utils.table_schema import (
    canonicalize_public_table,
    require_columns,
)

from scripts.utils.macro_context import CORE_MACRO_COLUMNS, attach_macro_context
from scripts.utils.sampling import sample_task_records


TASK_ID = 'labor_sce_offer'
INPUT_PATH = PROJECT_ROOT / "data" / "intermediate" / "sce_labor_market.parquet"
CORE_INPUT_PATH = PROJECT_ROOT / "data" / "intermediate" / "sce_core.parquet"
MIN_VALID_ANNUAL_AMOUNT = 500.0
TABULAR_PATH = PROJECT_ROOT / "data" / "householdbench" / "tabular" / f"{TASK_ID}.csv"
DIAGNOSTICS_DIR = PROJECT_ROOT / "output" / "householdbench" / "tasks" / TASK_ID
SAMPLING_DIR = PROJECT_ROOT / "data" / "intermediate" / "householdbench" / TASK_ID
STEP_DIAGNOSTICS_PATH = DIAGNOSTICS_DIR / "01_sample_construction_steps.csv"
DROP_DETAIL_PATH = DIAGNOSTICS_DIR / "01_sample_construction_drop_details.csv"
OUTLIER_CUTOFFS_PATH = DIAGNOSTICS_DIR / "01_outlier_cutoffs.csv"
STEP_COLUMNS = [
    "task_id",
    "step",
    "step_label",
    "rows_before",
    "rows_after",
    "rows_dropped",
]
DROP_DETAIL_COLUMNS = [
    "task_id",
    "step",
    "detail_order",
    "filter_id",
    "pass_condition",
    "fail_count",
]
OUTLIER_CUTOFF_COLUMNS = [
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
CORE_CONTEXT_COLUMNS = ['respondent_id',
 'survey_year',
 'survey_month',
 'age_years',
 '_NUM_CAT',
 '_REGION_CAT',
 '_EDU_CAT',
 '_HH_INC_DETAILED',
 'Q1',
 'Q2',
 'Q9_mean',
 'Q9c_mean',
 'Q10_1',
 'Q10_2',
 'Q10_3',
 'Q10_4',
 'Q10_5']

CORE_CONTEXT_LABEL_OUTPUT_COLUMNS = {
    "_NUM_CAT": "_NUM_CAT__label",
    "_REGION_CAT": "_REGION_CAT__label",
    "_EDU_CAT": "_EDU_CAT__label",
    "_HH_INC_DETAILED": "_HH_INC_DETAILED__label",
    "Q1": "Q1__label",
    "Q2": "Q2__label",
    "Q10_1": "Q10_1__label",
    "Q10_2": "Q10_2__label",
    "Q10_3": "Q10_3__label",
    "Q10_4": "Q10_4__label",
    "Q10_5": "Q10_5__label",
}
CORE_CONTEXT_READ_COLUMNS = list(dict.fromkeys(CORE_CONTEXT_COLUMNS + list(CORE_CONTEXT_LABEL_OUTPUT_COLUMNS.values())))

COMMON_PROMPT_CONTEXT_COLS = ['age_years',
 '_NUM_CAT',
 '_REGION_CAT',
 '_EDU_CAT',
 '_HH_INC_DETAILED',
 *CORE_MACRO_COLUMNS]

BASE_INPUT_COLUMNS = ['respondent_id',
 'survey_year',
 'survey_month',
 'survey_date',
 'release_date',
 'release_date_source']

TASK_INPUT_COLUMNS = ['nl1', 'nl2a_1', 'nl2b_1', 'nl3_1', 'reservation_wage_annual_full_time']

TASK_OUTPUT_COLUMNS = ['nl1',
 'nl2a_1',
 'nl2b_1',
 'nl3_1',
 'reservation_wage_annual_full_time',
 'first_offer_decision']

TASK_LABEL_OUTPUT_COLUMNS = {
    "nl2b_1": "nl2b_1__label",
    "nl3_1": "nl3_1__label",
}


step_rows = []
drop_detail_rows = []


# 1. Start from pre-processed main source.
if not INPUT_PATH.is_file():
    raise RuntimeError(f"Missing cleaned SCE labor-market file: {INPUT_PATH}")
sce = pd.read_parquet(INPUT_PATH)
require_columns(sce, BASE_INPUT_COLUMNS + TASK_INPUT_COLUMNS + list(TASK_LABEL_OUTPUT_COLUMNS.values()), label=str(INPUT_PATH))
sce["survey_date"] = pd.to_datetime(sce["survey_date"], errors="coerce")
sce["release_date"] = pd.to_datetime(sce["release_date"], errors="coerce")
if sce["survey_date"].isna().any():
    raise RuntimeError(f"{INPUT_PATH} contains missing or invalid survey_date values.")
reference_keys = ["respondent_id", "survey_year", "survey_month"]
reference = sce.loc[sce[reference_keys].notna().all(axis=1), reference_keys].copy()
if reference.duplicated(reference_keys).any():
    raise RuntimeError(f"{TASK_ID}: source-wide reference is not unique by respondent and survey month.")
sce["_trim_first_offer_annual_salary"] = pd.to_numeric(sce["nl2a_1"], errors="coerce")
sce["_trim_reservation_wage_annual_full_time"] = pd.to_numeric(
    sce["reservation_wage_annual_full_time"],
    errors="coerce",
)
for source_column, numeric_column in {
    "nl2a_1": "_trim_first_offer_annual_salary",
    "reservation_wage_annual_full_time": "_trim_reservation_wage_annual_full_time",
}.items():
    invalid_numeric = sce[source_column].notna() & sce[numeric_column].isna()
    if invalid_numeric.any():
        examples = sce.loc[
            invalid_numeric,
            ["respondent_id", "survey_year", "survey_month", source_column],
        ].head(20)
        raise RuntimeError(
            f"{TASK_ID}: nonnumeric observed values in {source_column}: {examples.to_dict('records')}"
        )

salary_reference = sce.loc[
    sce[reference_keys].notna().all(axis=1) & sce["_trim_first_offer_annual_salary"].notna(),
    reference_keys + ["_trim_first_offer_annual_salary"],
].copy()
salary_summary = (
    salary_reference.groupby(["survey_year", "survey_month"])["_trim_first_offer_annual_salary"]
    .agg(n_reference="size", p01=lambda values: values.quantile(0.01), p99=lambda values: values.quantile(0.99))
    .reset_index()
)
if salary_summary.empty or salary_summary[["p01", "p99"]].isna().any().any():
    raise RuntimeError(f"{TASK_ID}: missing source-wide monthly first-offer salary cutoffs.")
degenerate = salary_summary["p01"].eq(salary_summary["p99"])
if degenerate.any():
    bad = salary_summary.loc[degenerate, ["survey_year", "survey_month", "n_reference", "p01"]]
    raise RuntimeError(f"{TASK_ID}: degenerate monthly first-offer salary cutoffs: {bad.to_dict('records')}")
salary_summary["period"] = (
    salary_summary["survey_year"].astype(int).astype(str)
    + "-"
    + salary_summary["survey_month"].astype(int).astype(str).str.zfill(2)
)
salary_reference = salary_reference.merge(
    salary_summary, on=["survey_year", "survey_month"], how="left", validate="many_to_one"
)
salary_lower_counts = (
    salary_reference["_trim_first_offer_annual_salary"].le(salary_reference["p01"])
    .groupby([salary_reference["survey_year"], salary_reference["survey_month"]])
    .sum().rename("n_at_or_below_p01").reset_index()
)
salary_upper_counts = (
    salary_reference["_trim_first_offer_annual_salary"].ge(salary_reference["p99"])
    .groupby([salary_reference["survey_year"], salary_reference["survey_month"]])
    .sum().rename("n_at_or_above_p99").reset_index()
)
salary_audit = salary_summary.merge(
    salary_lower_counts, on=["survey_year", "survey_month"], validate="one_to_one"
).merge(salary_upper_counts, on=["survey_year", "survey_month"], validate="one_to_one")
salary_audit["task_id"] = TASK_ID
salary_audit["variable"] = "nl2a_1"
salary_audit["role"] = "predictor"
sce = sce.merge(
    salary_summary[["survey_year", "survey_month", "p01", "p99"]].rename(
        columns={"p01": "_trim_first_offer_annual_salary_p01", "p99": "_trim_first_offer_annual_salary_p99"}
    ),
    on=["survey_year", "survey_month"], how="left", validate="many_to_one",
)

reservation_reference = sce.loc[
    sce[reference_keys].notna().all(axis=1)
    & sce["_trim_reservation_wage_annual_full_time"].notna(),
    reference_keys + ["_trim_reservation_wage_annual_full_time"],
].copy()
reservation_summary = (
    reservation_reference.groupby(["survey_year", "survey_month"])["_trim_reservation_wage_annual_full_time"]
    .agg(n_reference="size", p01=lambda values: values.quantile(0.01), p99=lambda values: values.quantile(0.99))
    .reset_index()
)
if reservation_summary.empty or reservation_summary[["p01", "p99"]].isna().any().any():
    raise RuntimeError(f"{TASK_ID}: missing source-wide monthly reservation-wage cutoffs.")
degenerate = reservation_summary["p01"].eq(reservation_summary["p99"])
if degenerate.any():
    bad = reservation_summary.loc[degenerate, ["survey_year", "survey_month", "n_reference", "p01"]]
    raise RuntimeError(f"{TASK_ID}: degenerate monthly reservation-wage cutoffs: {bad.to_dict('records')}")
reservation_summary["period"] = (
    reservation_summary["survey_year"].astype(int).astype(str)
    + "-" + reservation_summary["survey_month"].astype(int).astype(str).str.zfill(2)
)
reservation_reference = reservation_reference.merge(
    reservation_summary,
    on=["survey_year", "survey_month"],
    how="left", validate="many_to_one",
)
reservation_lower_counts = (
    reservation_reference["_trim_reservation_wage_annual_full_time"].le(reservation_reference["p01"])
    .groupby([reservation_reference["survey_year"], reservation_reference["survey_month"]])
    .sum().rename("n_at_or_below_p01").reset_index()
)
reservation_upper_counts = (
    reservation_reference["_trim_reservation_wage_annual_full_time"].ge(reservation_reference["p99"])
    .groupby([reservation_reference["survey_year"], reservation_reference["survey_month"]])
    .sum().rename("n_at_or_above_p99").reset_index()
)
reservation_audit = reservation_summary.merge(
    reservation_lower_counts, on=["survey_year", "survey_month"], validate="one_to_one"
).merge(reservation_upper_counts, on=["survey_year", "survey_month"], validate="one_to_one")
reservation_audit["task_id"] = TASK_ID
reservation_audit["variable"] = "reservation_wage_annual_full_time"
reservation_audit["role"] = "predictor"
sce = sce.merge(
    reservation_summary[["survey_year", "survey_month", "p01", "p99"]].rename(
        columns={
            "p01": "_trim_reservation_wage_annual_full_time_p01",
            "p99": "_trim_reservation_wage_annual_full_time_p99",
        }
    ),
    on=["survey_year", "survey_month"], how="left", validate="many_to_one",
)
cutoff_audit = pd.concat([salary_audit, reservation_audit], ignore_index=True)
step_rows.append(
    construction_step(
        TASK_ID, 1, "Start from source observations",
        int(len(sce)), int(len(sce)), 0,
    )
)


# 2. Attach additional sources.
if not CORE_INPUT_PATH.is_file():
    raise RuntimeError(f"Missing SCE Core context file: {CORE_INPUT_PATH}")
core_context = pd.read_parquet(CORE_INPUT_PATH, columns=CORE_CONTEXT_READ_COLUMNS)
core_id_missing = core_context[["respondent_id", "survey_year", "survey_month"]].isna().sum()
if core_id_missing.any():
    raise RuntimeError(f"SCE Core context has missing identifiers: {core_id_missing[core_id_missing > 0].to_dict()}")
if core_context.duplicated(["respondent_id", "survey_year", "survey_month"]).any():
    raise RuntimeError("SCE Core context has duplicate respondent-month rows.")

rows_before = len(sce)
pool = sce.merge(core_context, on=["respondent_id", "survey_year", "survey_month"], how="left")
pool["survey_q_index"] = pool["survey_date"].dt.year * 4 + pool["survey_date"].dt.quarter
pool = attach_macro_context(
    pool,
    origin_q_index_col="survey_q_index",
    required_columns=CORE_MACRO_COLUMNS,
)
if len(pool) != rows_before:
    raise RuntimeError(f"{TASK_ID}: context attachment changed the row count.")
step_rows.append(
    construction_step(
        TASK_ID, 2, "Attach additional sources",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)


# 3. Construct leads and lags.
rows_before = len(pool)
for column, valid_codes in {"nl3_1": {1, 2, 3, 4}, "nl2b_1": {1, 2}}.items():
    values = pd.to_numeric(pool[column], errors="coerce").dropna()
    observed = set(values.round().astype(int).unique())
    unexpected = sorted(observed - valid_codes)
    if unexpected:
        raise RuntimeError(f"{TASK_ID}: unexpected codes in {column}: {unexpected}")
nl1 = pd.to_numeric(pool["nl1"], errors="coerce")
nl3 = pd.to_numeric(pool["nl3_1"], errors="coerce")
pool["first_offer_decision"] = pool["nl3_1__label"].astype("string")
step_rows.append(
    construction_step(
        TASK_ID, 3, "Construct leads and lags",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)


# 4. Restrict universe.
rows_before = len(pool)
offer_universe = nl1.ge(1)
drop_detail_rows.append(
    filter_count(
        TASK_ID, 4, 1,
        "at_least_one_job_offer", 'pd.to_numeric(pool["nl1"], errors="coerce").ge(1)', int((~offer_universe).sum()),
    )
)
pool = pool.loc[offer_universe].copy()
nl3 = pd.to_numeric(pool["nl3_1"], errors="coerce")
step_rows.append(
    construction_step(
        TASK_ID, 4, "Restrict universe",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)


# 5. Require observed target(s).
rows_before = len(pool)
target_observed = (
    nl3.isin([1, 2, 3])
    & pool["first_offer_decision"].isin(["accepted", "rejected"])
    & pool["first_offer_decision"].notna()
)
drop_detail_rows.append(
    filter_count(
        TASK_ID, 5, 1,
        "first_offer_decision_accepted_or_rejected", 'pd.to_numeric(pool["nl3_1"], errors="coerce").isin([1, 2, 3]) & '
            'pool["first_offer_decision"].isin(["accepted", "rejected"]) & '
            'pool["first_offer_decision"].notna()', int((~target_observed).sum()),
    )
)
pool = pool.loc[target_observed].copy()
step_rows.append(
    construction_step(
        TASK_ID, 5, "Require observed targets",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)


# 6. Require observed predictors.
rows_before = len(pool)
condition_masks = []
detail_order = 1
for column in ["release_date", "_HH_INC_DETAILED__label", *COMMON_PROMPT_CONTEXT_COLS]:
    pass_condition = f'pool["{column}"].notna()'
    pass_mask = pool[column].notna()
    condition_masks.append(pass_mask)
    drop_detail_rows.append(
        filter_count(
            TASK_ID, 6, detail_order,
            f"predictor_{column}_not_missing", pass_condition, int((~pass_mask).sum()),
        )
    )
    detail_order += 1
predictors_observed = pd.Series(True, index=pool.index)
for pass_mask in condition_masks:
    predictors_observed &= pass_mask
pool = pool.loc[predictors_observed].copy()
step_rows.append(
    construction_step(
        TASK_ID, 6, "Require observed predictors",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)


# 7. Apply logical and validity filters.
rows_before = len(pool)
age_pass_mask = pool["age_years"].between(18, 100, inclusive="both")
salary_floor_pass = (
    pool["_trim_first_offer_annual_salary"].isna()
    | pool["_trim_first_offer_annual_salary"].ge(MIN_VALID_ANNUAL_AMOUNT)
)
reservation_floor_pass = (
    pool["_trim_reservation_wage_annual_full_time"].isna()
    | pool["_trim_reservation_wage_annual_full_time"].ge(MIN_VALID_ANNUAL_AMOUNT)
)
validity_conditions = [
    (
        "age_years_between_18_and_100",
        'pool["age_years"].between(18, 100, inclusive="both")',
        age_pass_mask,
    ),
    (
        "valid_first_offer_annual_salary_at_least_500",
        (
            'pool["_trim_first_offer_annual_salary"].isna() | '
            'pool["_trim_first_offer_annual_salary"].ge(500.0)'
        ),
        salary_floor_pass,
    ),
    (
        "valid_reservation_wage_annual_full_time_at_least_500",
        (
            'pool["_trim_reservation_wage_annual_full_time"].isna() | '
            'pool["_trim_reservation_wage_annual_full_time"].ge(500.0)'
        ),
        reservation_floor_pass,
    ),
]
validity_mask = pd.Series(True, index=pool.index)
for detail_order, (filter_id, pass_condition, pass_mask) in enumerate(validity_conditions, start=1):
    validity_mask &= pass_mask
    drop_detail_rows.append(
        filter_count(
            TASK_ID, 7, detail_order,
            filter_id, pass_condition, int((~pass_mask).sum()),
        )
    )
pool = pool.loc[validity_mask].copy()
step_rows.append(
    construction_step(
        TASK_ID, 7, "Apply logical and validity filters",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)
if pool.empty:
    raise RuntimeError(f"{TASK_ID} has zero eligible rows.")


# 8. Apply outlier trimming.
rows_before = len(pool)
salary_observed = pool["_trim_first_offer_annual_salary"].notna()
salary_missing_cutoff = salary_observed & pool[
    ["_trim_first_offer_annual_salary_p01", "_trim_first_offer_annual_salary_p99"]
].isna().any(axis=1)
if salary_missing_cutoff.any():
    missing_periods = pool.loc[salary_missing_cutoff, ["survey_year", "survey_month"]].drop_duplicates()
    raise RuntimeError(f"{TASK_ID}: observed first-offer salaries lack monthly cutoffs: {missing_periods.to_dict('records')}")
salary_inside = (
    pool["_trim_first_offer_annual_salary"].gt(pool["_trim_first_offer_annual_salary_p01"])
    & pool["_trim_first_offer_annual_salary"].lt(pool["_trim_first_offer_annual_salary_p99"])
)
salary_pass = pool["_trim_first_offer_annual_salary"].isna() | salary_inside

reservation_observed = pool["_trim_reservation_wage_annual_full_time"].notna()
reservation_missing_cutoff = reservation_observed & pool[
    ["_trim_reservation_wage_annual_full_time_p01", "_trim_reservation_wage_annual_full_time_p99"]
].isna().any(axis=1)
if reservation_missing_cutoff.any():
    missing_periods = pool.loc[reservation_missing_cutoff, ["survey_year", "survey_month"]].drop_duplicates()
    raise RuntimeError(f"{TASK_ID}: observed reservation wages lack monthly cutoffs: {missing_periods.to_dict('records')}")
reservation_inside = (
    pool["_trim_reservation_wage_annual_full_time"].gt(pool["_trim_reservation_wage_annual_full_time_p01"])
    & pool["_trim_reservation_wage_annual_full_time"].lt(pool["_trim_reservation_wage_annual_full_time_p99"])
)
reservation_pass = pool["_trim_reservation_wage_annual_full_time"].isna() | reservation_inside
trim_conditions = [
    (
        "trim_first_offer_annual_salary_predictor_strict_inside_month_p01_p99",
        (
            'pool["_trim_first_offer_annual_salary"].isna() | '
            '(pool["_trim_first_offer_annual_salary"].gt(pool["_trim_first_offer_annual_salary_p01"]) '
            '& pool["_trim_first_offer_annual_salary"].lt(pool["_trim_first_offer_annual_salary_p99"]))'
        ),
        salary_pass,
    ),
    (
        "trim_reservation_wage_annual_full_time_predictor_strict_inside_month_p01_p99",
        (
            'pool["_trim_reservation_wage_annual_full_time"].isna() | '
            '(pool["_trim_reservation_wage_annual_full_time"].gt('
            'pool["_trim_reservation_wage_annual_full_time_p01"]) '
            '& pool["_trim_reservation_wage_annual_full_time"].lt('
            'pool["_trim_reservation_wage_annual_full_time_p99"]))'
        ),
        reservation_pass,
    ),
]
trim_mask = pd.Series(True, index=pool.index)
for detail_order, (filter_id, pass_condition, pass_mask) in enumerate(trim_conditions, start=1):
    trim_mask &= pass_mask
    drop_detail_rows.append(
        filter_count(
            TASK_ID, 8, detail_order,
            filter_id, pass_condition, int((~pass_mask).sum()),
        )
    )
pool = pool.loc[trim_mask].copy()
step_rows.append(
    construction_step(
        TASK_ID, 8, "Apply outlier trimming",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)
if pool.empty:
    raise RuntimeError(f"{TASK_ID} has zero eligible rows after outlier trimming.")

rows_before_export = len(pool)
pool["selection_period"] = [
    f"{int(year):04d}-{int(month):02d}"
    for year, month in zip(pool["survey_year"], pool["survey_month"], strict=True)
]
# Public keys are fixed before sampling; source keys retain their sort types.
pool["id"] = pool["respondent_id"].astype("string")
pool["time"] = pd.to_datetime(pool["survey_date"], errors="raise").dt.strftime("%Y-%m-%d")
pool["release_date"] = pd.to_datetime(pool["release_date"], errors="raise").dt.strftime("%Y-%m-%d")
sample, sampling_registry, sampling_summary = sample_task_records(
    pool,
    task_id=TASK_ID,
    output_dir=SAMPLING_DIR,
    output_sort_cols=["survey_year", "survey_month", "respondent_id"],
)

sample["subject_id"] = sample["respondent_id"].astype(str)
output_columns = list(
    dict.fromkeys(
        [
            "subject_id",
            "selection_period",
            "respondent_id",
            "survey_year",
            "survey_month",
            "survey_date",
            "release_date",
            "release_date_source",
        ]
        + [column for column in CORE_CONTEXT_COLUMNS if column not in {"respondent_id", "survey_year", "survey_month"}]
        + COMMON_PROMPT_CONTEXT_COLS
        + TASK_OUTPUT_COLUMNS
    )
)

for output_column, label_column in CORE_CONTEXT_LABEL_OUTPUT_COLUMNS.items():
    sample[output_column] = sample[label_column].astype("string")
for output_column, label_column in TASK_LABEL_OUTPUT_COLUMNS.items():
    if output_column in sample.columns:
        if label_column not in sample.columns:
            raise RuntimeError(f"{TASK_ID} output needs missing label column {label_column} for {output_column}.")
        sample[output_column] = sample[label_column].astype("string")

require_columns(sample, output_columns, label=f"{TASK_ID} output")
required_selected_columns = ['subject_id', 'respondent_id', 'survey_year', 'survey_month', 'survey_date', 'release_date', 'release_date_source', 'age_years', '_NUM_CAT', '_REGION_CAT', '_EDU_CAT', '_HH_INC_DETAILED', *CORE_MACRO_COLUMNS, 'nl1', 'nl3_1', 'first_offer_decision']
missing_selected = sample[required_selected_columns].isna().sum()
missing_selected = missing_selected[missing_selected.gt(0)]
if not missing_selected.empty:
    raise RuntimeError(f"{TASK_ID} selected sample has missing required values: {missing_selected.to_dict()}")
for date_col in ["survey_date", "release_date"]:
    sample[date_col] = pd.to_datetime(sample[date_col], errors="coerce")
    if sample[date_col].isna().any():
        raise RuntimeError(f"{TASK_ID} selected sample has missing or invalid {date_col} values.")
    sample[date_col] = sample[date_col].dt.date.astype(str)

TABULAR_PATH.parent.mkdir(parents=True, exist_ok=True)
canonicalize_public_table(sample, task_id=TASK_ID).to_csv(TABULAR_PATH, index=False)
step_rows.append(
    construction_step(
        TASK_ID, 9, "Export sample",
        int(rows_before_export), int(len(sample)), int(rows_before_export) - int(len(sample)),
    )
)

DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(step_rows, columns=STEP_COLUMNS).to_csv(STEP_DIAGNOSTICS_PATH, index=False)
pd.DataFrame(drop_detail_rows, columns=DROP_DETAIL_COLUMNS).to_csv(DROP_DETAIL_PATH, index=False)
cutoff_audit.loc[:, OUTLIER_CUTOFF_COLUMNS].to_csv(OUTLIER_CUTOFFS_PATH, index=False)

print(
    f"{TASK_ID}: eligible={len(pool):,} sampled={len(sample):,} "
    f"tabular={TABULAR_PATH}"
)
