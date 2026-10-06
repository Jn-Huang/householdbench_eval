#!/usr/bin/env python
"""Build the tabular CSV for the SCE reservation-wage and preferred-hours task."""

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


TASK_ID = 'labor_sce_reswage'
INPUT_PATH = PROJECT_ROOT / "data" / "intermediate" / "sce_job_search.parquet"
CORE_INPUT_PATH = PROJECT_ROOT / "data" / "intermediate" / "sce_core.parquet"
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

TASK_INPUT_COLUMNS = [ 'hhsize',
 'numchildren_under18',
 'l2_num_jobs',
 'l7_days_spent_searching',
 'l8_months_no_work__clean',
 'l8_months_no_work__status',
 'l10_cps_job_usual_hrs',
 'l11_cps_job_earn_hrly',
 'l11_cps_job_earn_wkly',
 'l11_cps_job_earn_ann',
 'l22_last_job_usual_hrs',
 'l24_last_job_earn_hrly',
 'l24_last_job_earn_wkly',
 'l24_last_job_earn_ann',
 'rw2h_reserv_wage',
 'rw1a_desired_hours']

TASK_OUTPUT_COLUMNS = [ 'hhsize',
 'numchildren_under18',
 'l2_num_jobs',
 'l7_days_spent_searching',
 'months_since_paid_work',
 'never_had_paid_job',
 'l10_cps_job_usual_hrs',
 'l11_cps_job_earn_hrly',
 'l11_cps_job_earn_wkly',
 'l11_cps_job_earn_ann',
 'current_job_hourly_equivalent_pay',
 'current_job_branch_valid',
 'l22_last_job_usual_hrs',
 'l24_last_job_earn_hrly',
 'l24_last_job_earn_wkly',
 'l24_last_job_earn_ann',
 'last_job_hourly_equivalent_pay',
 'last_job_branch_valid',
 'selected_job_branch',
 'rw2h_reserv_wage',
 'reservation_wage_hourly',
 'rw1a_desired_hours',
 'preferred_weekly_hours']

TASK_LABEL_OUTPUT_COLUMNS = {}


step_rows = []
drop_detail_rows = []


def construct_hourly_equivalent_pay(
    frame: pd.DataFrame,
    *,
    hourly_column: str,
    weekly_column: str,
    annual_column: str,
    hours_column: str,
) -> pd.Series:
    hours = pd.to_numeric(frame[hours_column], errors="coerce")
    valid_hours = hours.gt(0) & hours.le(168)
    hourly = pd.to_numeric(frame[hourly_column], errors="coerce")
    weekly = pd.to_numeric(frame[weekly_column], errors="coerce")
    annual = pd.to_numeric(frame[annual_column], errors="coerce")
    result = pd.Series(float("nan"), index=frame.index, dtype="float64")
    use_hourly = valid_hours & hourly.gt(0)
    result.loc[use_hourly] = hourly.loc[use_hourly]
    use_weekly = valid_hours & result.isna() & weekly.gt(0)
    result.loc[use_weekly] = weekly.loc[use_weekly] / hours.loc[use_weekly]
    use_annual = valid_hours & result.isna() & annual.gt(0)
    result.loc[use_annual] = annual.loc[use_annual] / (52 * hours.loc[use_annual])
    return result


# 1. Start from pre-processed main source.
if not INPUT_PATH.is_file():
    raise RuntimeError(f"Missing cleaned SCE job-search file: {INPUT_PATH}")
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
sce["_trim_current_job_hourly_equivalent_pay"] = construct_hourly_equivalent_pay(
    sce,
    hourly_column="l11_cps_job_earn_hrly",
    weekly_column="l11_cps_job_earn_wkly",
    annual_column="l11_cps_job_earn_ann",
    hours_column="l10_cps_job_usual_hrs",
)
sce["_trim_last_job_hourly_equivalent_pay"] = construct_hourly_equivalent_pay(
    sce,
    hourly_column="l24_last_job_earn_hrly",
    weekly_column="l24_last_job_earn_wkly",
    annual_column="l24_last_job_earn_ann",
    hours_column="l22_last_job_usual_hrs",
)
trim_source_columns = {
    "reservation_wage_hourly": "rw2h_reserv_wage",
    "current_job_hourly_equivalent_pay": "_trim_current_job_hourly_equivalent_pay",
    "last_job_hourly_equivalent_pay": "_trim_last_job_hourly_equivalent_pay",
}
cutoff_audit_frames = []
for variable, source_column in trim_source_columns.items():
    task_column = f"_trim_{variable}"
    if task_column != source_column:
        sce[task_column] = pd.to_numeric(sce[source_column], errors="coerce")
    variable_reference = sce.loc[
        sce[reference_keys].notna().all(axis=1) & sce[task_column].notna(),
        reference_keys + [task_column],
    ].copy()
    variable_summary = (
        variable_reference.groupby(["survey_year", "survey_month"])[task_column]
        .agg(n_reference="size", p01=lambda values: values.quantile(0.01), p99=lambda values: values.quantile(0.99))
        .reset_index()
    )
    if variable_summary.empty or variable_summary[["p01", "p99"]].isna().any().any():
        raise RuntimeError(f"{TASK_ID}: missing source-wide monthly cutoffs for {variable}.")
    degenerate = variable_summary["p01"].eq(variable_summary["p99"])
    if degenerate.any():
        bad = variable_summary.loc[degenerate, ["survey_year", "survey_month", "n_reference", "p01"]]
        raise RuntimeError(f"{TASK_ID}: degenerate monthly {variable} cutoffs: {bad.to_dict('records')}")
    variable_summary["period"] = (
        variable_summary["survey_year"].astype(int).astype(str)
        + "-"
        + variable_summary["survey_month"].astype(int).astype(str).str.zfill(2)
    )
    variable_reference = variable_reference.merge(
        variable_summary, on=["survey_year", "survey_month"], how="left", validate="many_to_one"
    )
    lower_counts = (
        variable_reference[task_column].le(variable_reference["p01"])
        .groupby([variable_reference["survey_year"], variable_reference["survey_month"]])
        .sum()
        .rename("n_at_or_below_p01")
        .reset_index()
    )
    upper_counts = (
        variable_reference[task_column].ge(variable_reference["p99"])
        .groupby([variable_reference["survey_year"], variable_reference["survey_month"]])
        .sum()
        .rename("n_at_or_above_p99")
        .reset_index()
    )
    variable_audit = variable_summary.merge(
        lower_counts, on=["survey_year", "survey_month"], validate="one_to_one"
    ).merge(upper_counts, on=["survey_year", "survey_month"], validate="one_to_one")
    variable_audit["task_id"] = TASK_ID
    variable_audit["variable"] = variable
    variable_audit["role"] = "target" if variable == "reservation_wage_hourly" else "predictor"
    cutoff_audit_frames.append(variable_audit)
    sce = sce.merge(
        variable_summary[["survey_year", "survey_month", "p01", "p99"]].rename(
            columns={"p01": f"{task_column}_p01", "p99": f"{task_column}_p99"}
        ),
        on=["survey_year", "survey_month"],
        how="left",
        validate="many_to_one",
    )
cutoff_audit = pd.concat(cutoff_audit_frames, ignore_index=True)
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
pool["reservation_wage_hourly"] = pd.to_numeric(pool["rw2h_reserv_wage"], errors="coerce")
pool["preferred_weekly_hours"] = pd.to_numeric(pool["rw1a_desired_hours"], errors="coerce")
pool["current_job_hourly_equivalent_pay"] = construct_hourly_equivalent_pay(
    pool,
    hourly_column="l11_cps_job_earn_hrly",
    weekly_column="l11_cps_job_earn_wkly",
    annual_column="l11_cps_job_earn_ann",
    hours_column="l10_cps_job_usual_hrs",
)
pool["last_job_hourly_equivalent_pay"] = construct_hourly_equivalent_pay(
    pool,
    hourly_column="l24_last_job_earn_hrly",
    weekly_column="l24_last_job_earn_wkly",
    annual_column="l24_last_job_earn_ann",
    hours_column="l22_last_job_usual_hrs",
)
pool["current_job_branch_valid"] = (
    pool["current_job_hourly_equivalent_pay"].notna()
    & pd.to_numeric(pool["l10_cps_job_usual_hrs"], errors="coerce").gt(0)
    & pd.to_numeric(pool["l10_cps_job_usual_hrs"], errors="coerce").le(168)
)
pool["last_job_branch_valid"] = (
    pool["last_job_hourly_equivalent_pay"].notna()
    & pd.to_numeric(pool["l22_last_job_usual_hrs"], errors="coerce").gt(0)
    & pd.to_numeric(pool["l22_last_job_usual_hrs"], errors="coerce").le(168)
)
pool["months_since_paid_work"] = pd.to_numeric(pool["l8_months_no_work__clean"], errors="coerce")
months_status = pool["l8_months_no_work__status"].astype("string")
pool["never_had_paid_job"] = pd.Series(pd.NA, index=pool.index, dtype="Int64")
pool.loc[months_status.notna(), "never_had_paid_job"] = 0
pool.loc[months_status.eq("never_had_paid_job"), "never_had_paid_job"] = 1
if (pool["never_had_paid_job"].eq(1) & pool["months_since_paid_work"].notna()).any():
    raise RuntimeError(f"{TASK_ID}: never-worked indicator coexists with a numeric workless duration.")
if (months_status.eq("valid") & pool["months_since_paid_work"].isna()).any():
    raise RuntimeError(f"{TASK_ID}: valid months-not-working rows are missing the cleaned duration.")
target_columns = [
    "reservation_wage_hourly",
    "preferred_weekly_hours",
]
step_rows.append(
    construction_step(
        TASK_ID, 3, "Construct leads and lags",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)


# 4. Restrict universe.
rows_before = len(pool)
step_rows.append(
    construction_step(
        TASK_ID, 4, "Restrict universe",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)


# 5. Require observed target(s).
rows_before = len(pool)
condition_masks = []
detail_order = 1
for column in target_columns:
    pass_condition = f'pool["{column}"].notna()'
    pass_mask = pool[column].notna()
    condition_masks.append(pass_mask)
    drop_detail_rows.append(
        filter_count(
            TASK_ID, 5, detail_order,
            f"target_{column}_not_missing", pass_condition, int((~pass_mask).sum()),
        )
    )
    detail_order += 1
target_observed = pd.Series(True, index=pool.index)
for pass_mask in condition_masks:
    target_observed &= pass_mask
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
complete_raw_branch = pool["current_job_branch_valid"] | pool["last_job_branch_valid"]
condition_masks.append(complete_raw_branch)
drop_detail_rows.append(
    filter_count(
        TASK_ID, 6, detail_order,
        "predictor_complete_current_or_last_job_pay_hours_branch", 'pool["current_job_branch_valid"] | pool["last_job_branch_valid"]', int((~complete_raw_branch).sum()),
    )
)
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
condition_masks = []
detail_order = 1
quality_conditions = [
    (
        "reservation_wage_hourly_positive",
        'pool["reservation_wage_hourly"].gt(0)',
        pool["reservation_wage_hourly"].gt(0),
    ),
    (
        "preferred_weekly_hours_between_0_and_168",
        'pool["preferred_weekly_hours"].gt(0) & pool["preferred_weekly_hours"].le(168)',
        pool["preferred_weekly_hours"].gt(0) & pool["preferred_weekly_hours"].le(168),
    ),
]
for filter_id, pass_condition, pass_mask in quality_conditions:
    condition_masks.append(pass_mask)
    drop_detail_rows.append(
        filter_count(
            TASK_ID, 7, detail_order,
            filter_id, pass_condition, int((~pass_mask).sum()),
        )
    )
    detail_order += 1
quality_mask = pd.Series(True, index=pool.index)
for pass_mask in condition_masks:
    quality_mask &= pass_mask
pool = pool.loc[quality_mask].copy()
age_pass_mask = pool["age_years"].between(18, 100, inclusive="both")
drop_detail_rows.append(
    filter_count(
        TASK_ID, 7, 1 + sum(row["step"] == 7 for row in drop_detail_rows),
        "age_years_between_18_and_100", 'pool["age_years"].between(18, 100, inclusive="both")', int((~age_pass_mask).sum()),
    )
)
pool = pool.loc[age_pass_mask].copy()
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
for variable in trim_source_columns:
    task_column = f"_trim_{variable}"
    observed = pool[task_column].notna()
    missing_cutoff = observed & pool[[f"{task_column}_p01", f"{task_column}_p99"]].isna().any(axis=1)
    if missing_cutoff.any():
        missing_periods = pool.loc[missing_cutoff, ["survey_year", "survey_month"]].drop_duplicates()
        raise RuntimeError(f"{TASK_ID}: observed {variable} values lack monthly cutoffs: {missing_periods.to_dict('records')}")

reservation_wage_inside = pool["_trim_reservation_wage_hourly"].gt(
    pool["_trim_reservation_wage_hourly_p01"]
) & pool["_trim_reservation_wage_hourly"].lt(pool["_trim_reservation_wage_hourly_p99"])
current_job_inside = pool["_trim_current_job_hourly_equivalent_pay"].gt(
    pool["_trim_current_job_hourly_equivalent_pay_p01"]
) & pool["_trim_current_job_hourly_equivalent_pay"].lt(
    pool["_trim_current_job_hourly_equivalent_pay_p99"]
)
last_job_inside = pool["_trim_last_job_hourly_equivalent_pay"].gt(
    pool["_trim_last_job_hourly_equivalent_pay_p01"]
) & pool["_trim_last_job_hourly_equivalent_pay"].lt(
    pool["_trim_last_job_hourly_equivalent_pay_p99"]
)
current_job_valid_after_trim = pool["current_job_branch_valid"] & current_job_inside
last_job_valid_after_trim = pool["last_job_branch_valid"] & last_job_inside
complete_branch_after_trim = current_job_valid_after_trim | last_job_valid_after_trim
trim_conditions = [
    (
        "trim_reservation_wage_hourly_strict_inside_month_p01_p99",
        'pool["_trim_reservation_wage_hourly"].gt(pool["_trim_reservation_wage_hourly_p01"]) & '
        'pool["_trim_reservation_wage_hourly"].lt(pool["_trim_reservation_wage_hourly_p99"])',
        reservation_wage_inside,
    ),
    (
        "trim_complete_current_or_last_job_pay_hours_branch",
        "current_job_valid_after_trim | last_job_valid_after_trim",
        complete_branch_after_trim,
    ),
]
for detail_order, (filter_id, pass_condition, pass_mask) in enumerate(trim_conditions, start=1):
    drop_detail_rows.append(
        filter_count(
            TASK_ID, 8, detail_order,
            filter_id, pass_condition, int((~pass_mask).sum()),
        )
    )
trim_mask = reservation_wage_inside & complete_branch_after_trim
pool = pool.loc[trim_mask].copy()
current_job_valid_after_trim = current_job_valid_after_trim.loc[pool.index]
last_job_valid_after_trim = last_job_valid_after_trim.loc[pool.index]
pool["selected_job_branch"] = "last"
pool.loc[current_job_valid_after_trim, "selected_job_branch"] = "current"
pool["current_job_branch_valid"] = current_job_valid_after_trim
pool["last_job_branch_valid"] = last_job_valid_after_trim
pool.loc[~current_job_valid_after_trim, ["current_job_hourly_equivalent_pay", "l10_cps_job_usual_hrs"]] = pd.NA
pool.loc[~last_job_valid_after_trim, ["last_job_hourly_equivalent_pay", "l22_last_job_usual_hrs"]] = pd.NA
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
required_selected_columns = ['subject_id', 'respondent_id', 'survey_year', 'survey_month', 'survey_date', 'release_date', 'release_date_source', 'age_years', '_NUM_CAT', '_REGION_CAT', '_EDU_CAT', '_HH_INC_DETAILED', *CORE_MACRO_COLUMNS, 'reservation_wage_hourly', 'preferred_weekly_hours', 'selected_job_branch']
missing_selected = sample[required_selected_columns].isna().sum()
missing_selected = missing_selected[missing_selected.gt(0)]
if not missing_selected.empty:
    raise RuntimeError(f"{TASK_ID} selected sample has missing required values: {missing_selected.to_dict()}")
if not sample["reservation_wage_hourly"].equals(pd.to_numeric(sample["rw2h_reserv_wage"], errors="coerce")):
    raise RuntimeError(f"{TASK_ID}: reservation_wage_hourly no longer matches publisher rw2h_reserv_wage.")
selected_current = sample["selected_job_branch"].eq("current")
selected_last = sample["selected_job_branch"].eq("last")
if not (selected_current | selected_last).all():
    raise RuntimeError(f"{TASK_ID}: selected job branch must be current or last.")
if sample.loc[selected_current, ["current_job_hourly_equivalent_pay", "l10_cps_job_usual_hrs"]].isna().any().any():
    raise RuntimeError(f"{TASK_ID}: selected current-job branches lack matched pay and hours.")
if sample.loc[selected_last, ["last_job_hourly_equivalent_pay", "l22_last_job_usual_hrs"]].isna().any().any():
    raise RuntimeError(f"{TASK_ID}: selected last-job branches lack matched pay and hours.")
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
