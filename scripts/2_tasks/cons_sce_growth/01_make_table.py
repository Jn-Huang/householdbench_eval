#!/usr/bin/env python
"""Build the tabular CSV for the SCE household-spending growth task."""

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


TASK_ID = 'cons_sce_growth'
INPUT_PATH = PROJECT_ROOT / "data" / "intermediate" / "sce_household_spending.parquet"
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
REQUIRED_PROMPT_CONTEXT_COLS = [
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
    'Q10_5',
    'k2e',
    *CORE_MACRO_COLUMNS,
]

BASE_INPUT_COLUMNS = ['respondent_id',
 'survey_year',
 'survey_month',
 'survey_date',
 'release_date',
 'release_date_source']

TASK_INPUT_COLUMNS = ['k2e', 'qsp1', 'qsp2__clean']

TASK_OUTPUT_COLUMNS = ['k2e', 'qsp1', 'qsp2__clean', 'signed_spending_growth_percent']

TASK_LABEL_OUTPUT_COLUMNS = {
    "k2e": "k2e__label",
    "qsp1": "qsp1__label",
}

step_rows = []
drop_detail_rows = []


# 1. Start from pre-processed main source.
if not INPUT_PATH.is_file():
    raise RuntimeError(f"Missing cleaned SCE household-spending file: {INPUT_PATH}")
sce = pd.read_parquet(INPUT_PATH)
require_columns(sce, BASE_INPUT_COLUMNS + TASK_INPUT_COLUMNS + list(TASK_LABEL_OUTPUT_COLUMNS.values()), label=str(INPUT_PATH))
sce["survey_date"] = pd.to_datetime(sce["survey_date"], errors="coerce")
sce["release_date"] = pd.to_datetime(sce["release_date"], errors="coerce")
if sce["survey_date"].isna().any():
    raise RuntimeError(f"{INPUT_PATH} contains missing or invalid survey_date values.")
reference_keys = ["respondent_id", "survey_year", "survey_month"]
reference = sce.loc[sce[reference_keys].notna().all(axis=1), reference_keys + ["qsp1", "qsp2__clean"]].copy()
if reference.duplicated(reference_keys).any():
    raise RuntimeError(f"{TASK_ID}: source-wide reference is not unique by respondent and survey month.")
reference["signed_spending_growth_percent"] = pd.to_numeric(reference["qsp2__clean"], errors="coerce")
direction = pd.to_numeric(reference["qsp1"], errors="coerce")
reference.loc[direction.eq(2), "signed_spending_growth_percent"] *= -1
reference.loc[~direction.isin([1, 2]), "signed_spending_growth_percent"] = pd.NA
reference_values = reference.dropna(subset=["signed_spending_growth_percent"]).copy()
reference_summary = (
    reference_values.groupby(["survey_year", "survey_month"])["signed_spending_growth_percent"]
    .agg(n_reference="size", p01=lambda values: values.quantile(0.01), p99=lambda values: values.quantile(0.99))
    .reset_index()
)
if reference_summary.empty or reference_summary[["p01", "p99"]].isna().any().any():
    raise RuntimeError(f"{TASK_ID}: missing source-wide monthly spending-growth cutoffs.")
degenerate = reference_summary["p01"].eq(reference_summary["p99"])
if degenerate.any():
    bad = reference_summary.loc[degenerate, ["survey_year", "survey_month", "n_reference", "p01"]]
    raise RuntimeError(f"{TASK_ID}: degenerate monthly spending-growth cutoffs: {bad.to_dict('records')}")
reference_summary["period"] = (
    reference_summary["survey_year"].astype(int).astype(str)
    + "-"
    + reference_summary["survey_month"].astype(int).astype(str).str.zfill(2)
)
reference_values = reference_values.merge(
    reference_summary,
    on=["survey_year", "survey_month"],
    how="left",
    validate="many_to_one",
)
cutoff_audit = reference_summary.copy()
lower_counts = (
    reference_values["signed_spending_growth_percent"].le(reference_values["p01"])
    .groupby([reference_values["survey_year"], reference_values["survey_month"]])
    .sum()
    .rename("n_at_or_below_p01")
    .reset_index()
)
upper_counts = (
    reference_values["signed_spending_growth_percent"].ge(reference_values["p99"])
    .groupby([reference_values["survey_year"], reference_values["survey_month"]])
    .sum()
    .rename("n_at_or_above_p99")
    .reset_index()
)
cutoff_audit = cutoff_audit.merge(lower_counts, on=["survey_year", "survey_month"], validate="one_to_one")
cutoff_audit = cutoff_audit.merge(upper_counts, on=["survey_year", "survey_month"], validate="one_to_one")
cutoff_audit["task_id"] = TASK_ID
cutoff_audit["variable"] = "signed_spending_growth_percent"
cutoff_audit["role"] = "target"
sce = sce.merge(
    reference_summary[["survey_year", "survey_month", "p01", "p99"]].rename(
        columns={"p01": "signed_spending_growth_percent_p01", "p99": "signed_spending_growth_percent_p99"}
    ),
    on=["survey_year", "survey_month"],
    how="left",
    validate="many_to_one",
)
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
for column, valid_codes in {"qsp1": {1, 2}, "k2e": {1, 2}}.items():
    values = pd.to_numeric(pool[column], errors="coerce").dropna()
    observed = set(values.round().astype(int).unique())
    unexpected = sorted(observed - valid_codes)
    if unexpected:
        raise RuntimeError(f"{TASK_ID}: unexpected codes in {column}: {unexpected}")
pool["signed_spending_growth_percent"] = pd.to_numeric(pool["qsp2__clean"], errors="coerce")
pool.loc[pd.to_numeric(pool["qsp1"], errors="coerce").eq(2), "signed_spending_growth_percent"] *= -1
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
pass_condition = 'pd.to_numeric(pool["qsp1"], errors="coerce").isin([1, 2])'
pass_mask = pd.to_numeric(pool["qsp1"], errors="coerce").isin([1, 2])
condition_masks.append(pass_mask)
drop_detail_rows.append(
    filter_count(
        TASK_ID, 5, detail_order,
        "spending_growth_direction_valid", pass_condition, int((~pass_mask).sum()),
    )
)
detail_order += 1
pass_condition = 'pool["signed_spending_growth_percent"].notna()'
pass_mask = pool["signed_spending_growth_percent"].notna()
condition_masks.append(pass_mask)
drop_detail_rows.append(
    filter_count(
        TASK_ID, 5, detail_order,
        "spending_growth_magnitude_not_missing", pass_condition, int((~pass_mask).sum()),
    )
)
valid_target = pd.Series(True, index=pool.index)
for pass_mask in condition_masks:
    valid_target &= pass_mask
pool = pool.loc[valid_target].copy()
step_rows.append(
    construction_step(
        TASK_ID, 5, "Require observed targets",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)
# 6. Require observed predictors.
rows_before = len(pool)
required_predictor_columns = ["release_date", "_HH_INC_DETAILED__label"] + REQUIRED_PROMPT_CONTEXT_COLS
condition_masks = []
detail_order = 1
for column in required_predictor_columns:
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


# 8. Apply outlier trimming.
rows_before = len(pool)
if pool[["signed_spending_growth_percent_p01", "signed_spending_growth_percent_p99"]].isna().any().any():
    missing_periods = pool.loc[
        pool[["signed_spending_growth_percent_p01", "signed_spending_growth_percent_p99"]].isna().any(axis=1),
        ["survey_year", "survey_month"],
    ].drop_duplicates()
    raise RuntimeError(f"{TASK_ID}: eligible rows lack monthly spending-growth cutoffs: {missing_periods.to_dict('records')}")
pass_condition = (
    'pool["signed_spending_growth_percent"].gt(pool["signed_spending_growth_percent_p01"]) '
    '& pool["signed_spending_growth_percent"].lt(pool["signed_spending_growth_percent_p99"])'
)
pass_mask = (
    pool["signed_spending_growth_percent"].gt(pool["signed_spending_growth_percent_p01"])
    & pool["signed_spending_growth_percent"].lt(pool["signed_spending_growth_percent_p99"])
)
drop_detail_rows.append(
    filter_count(
        TASK_ID, 8, 1,
        "trim_signed_spending_growth_percent_target_strict_inside_month_p01_p99", pass_condition, int((~pass_mask).sum()),
    )
)
pool = pool.loc[pass_mask].copy()
step_rows.append(
    construction_step(
        TASK_ID, 8, "Apply outlier trimming",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)
if pool.empty:
    raise RuntimeError(f"{TASK_ID} has zero eligible rows.")

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

rows_before_export = len(pool)

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
required_selected_columns = [
    'subject_id',
    'respondent_id',
    'survey_year',
    'survey_month',
    'survey_date',
    'release_date',
    'release_date_source',
    *REQUIRED_PROMPT_CONTEXT_COLS,
    'qsp1',
    'qsp2__clean',
    'signed_spending_growth_percent',
]
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
