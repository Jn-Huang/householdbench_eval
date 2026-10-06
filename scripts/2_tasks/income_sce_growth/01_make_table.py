#!/usr/bin/env python
"""Build the HouseholdBench tabular file for SCE household income growth."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.sample_accounting import construction_step, filter_count

from scripts.utils.table_schema import canonicalize_public_table

from scripts.utils.macro_context import CORE_MACRO_COLUMNS, attach_macro_context
from scripts.utils.sampling import sample_task_records


OUTPUT_SLUG = 'income_sce_growth'
INPUT_PATH = PROJECT_ROOT / "data" / "intermediate" / "sce_core.parquet"
TABULAR_PATH = PROJECT_ROOT / "data" / "householdbench" / "tabular" / f"{OUTPUT_SLUG}.csv"
DIAGNOSTICS_DIR = PROJECT_ROOT / "output" / "householdbench" / "tasks" / OUTPUT_SLUG
SAMPLING_DIR = PROJECT_ROOT / "data" / "intermediate" / "householdbench" / OUTPUT_SLUG
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

SCE_COLUMNS = ['source',
 'module',
 'release_id',
 'release_date',
 'release_date_source',
 'respondent_id',
 'userid',
 'survey_year',
 'survey_month',
 'survey_date',
 'panel_month',
 'weight_core',
 'age_years',
 '_NUM_CAT',
 '_REGION_CAT',
 '_EDU_CAT',
 '_HH_INC_DETAILED',
 '_STATE',
 'Q1',
 'Q2',
 'Q3',
 'Q4new__clean',
 'Q5new__clean',
 'Q6new__clean',
 'Q9_mean',
 'Q9_iqr',
 'Q9__task_valid',
 'Q9c_mean',
 'Q9c_iqr',
 'Q9c__task_valid',
 'Q10_1',
 'Q10_2',
 'Q10_3',
 'Q10_4',
 'Q10_5',
 'Q11',
 'Q12new',
 'Q13new__clean',
 'Q14new__clean',
 'Q15',
 'Q16',
 'Q17new__clean',
 'Q18new__clean',
 'Q19',
 'Q22new__clean',
 'Q25v2',
 'Q25v2part2',
 'Q30new__clean',
 'Q45b',
 'D1',
 'D6']

CORE_LABEL_OUTPUT_COLUMNS = {
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
    "Q12new": "Q12new__label",
    "Q15": "Q15__label",
    "Q45b": "Q45b__label",
}
SCE_READ_COLUMNS = list(dict.fromkeys(SCE_COLUMNS + list(CORE_LABEL_OUTPUT_COLUMNS.values())))

REQUIRED_CONTEXT_COLUMNS = ['age_years',
 '_NUM_CAT',
 '_REGION_CAT',
 '_EDU_CAT',
 '_HH_INC_DETAILED',
 'panel_month',
 *CORE_MACRO_COLUMNS]

TARGET_COLUMNS = ['Q25v2part2']

PROBABILITY_COLUMNS = []

OUTPUT_COLUMNS = ['subject_id',
 'selection_period',
 'respondent_id',
 'survey_date',
 'release_date',
 'release_date_source',
 'panel_month',
 'age_years',
 '_NUM_CAT',
 '_REGION_CAT',
 '_EDU_CAT',
 '_HH_INC_DETAILED',
 'Q1',
 'Q2',
 'Q45b',
 'Q10_1',
 'Q10_2',
 'Q10_3',
 'Q10_4',
 'Q10_5',
 'Q11',
 'Q12new',
 'Q15',
 'Q16',
 'Q19',
 *CORE_MACRO_COLUMNS,
 'Q25v2part2']


step_rows = []
drop_detail_rows = []


# 1. Start from pre-processed main source.
if not INPUT_PATH.is_file():
    raise SystemExit(f"SCE Core parquet not found: {INPUT_PATH}")
sce = pd.read_parquet(INPUT_PATH, columns=SCE_READ_COLUMNS)
if not sce["module"].eq("core").all():
    raise SystemExit("Input contains non-Core SCE rows.")
sce["survey_date"] = pd.to_datetime(sce["survey_date"], errors="coerce")
sce["release_date"] = pd.to_datetime(sce["release_date"], errors="coerce")
if sce["survey_date"].isna().any():
    raise SystemExit("SCE Core input contains missing or invalid survey_date values.")
if sce["release_date"].isna().any():
    raise SystemExit("SCE Core input contains missing or invalid release_date values.")
sce = sce.sort_values(["survey_date", "respondent_id"], kind="mergesort").reset_index(drop=True)
reference_keys = ["respondent_id", "survey_year", "survey_month"]
reference = sce.loc[
    sce[reference_keys].notna().all(axis=1), reference_keys + ["Q25v2part2"]
].copy()
if reference.duplicated(reference_keys).any():
    raise SystemExit(f"{OUTPUT_SLUG}: source-wide reference is not unique by respondent and survey month.")
reference["Q25v2part2"] = pd.to_numeric(reference["Q25v2part2"], errors="coerce")
reference = reference.dropna(subset=["Q25v2part2"]).copy()
cutoff_summary = (
    reference.groupby(["survey_year", "survey_month"])["Q25v2part2"]
    .agg(n_reference="size", p01=lambda values: values.quantile(0.01), p99=lambda values: values.quantile(0.99))
    .reset_index()
)
if cutoff_summary.empty or cutoff_summary[["p01", "p99"]].isna().any().any():
    raise SystemExit(f"{OUTPUT_SLUG}: missing source-wide monthly Q25v2part2 cutoffs.")
degenerate = cutoff_summary["p01"].eq(cutoff_summary["p99"])
if degenerate.any():
    bad = cutoff_summary.loc[degenerate, ["survey_year", "survey_month", "n_reference", "p01"]]
    raise SystemExit(f"{OUTPUT_SLUG}: degenerate monthly Q25v2part2 cutoffs: {bad.to_dict('records')}")
cutoff_summary["period"] = (
    cutoff_summary["survey_year"].astype(int).astype(str)
    + "-"
    + cutoff_summary["survey_month"].astype(int).astype(str).str.zfill(2)
)
reference = reference.merge(
    cutoff_summary, on=["survey_year", "survey_month"], how="left", validate="many_to_one"
)
lower_counts = (
    reference["Q25v2part2"].le(reference["p01"])
    .groupby([reference["survey_year"], reference["survey_month"]])
    .sum()
    .rename("n_at_or_below_p01")
    .reset_index()
)
upper_counts = (
    reference["Q25v2part2"].ge(reference["p99"])
    .groupby([reference["survey_year"], reference["survey_month"]])
    .sum()
    .rename("n_at_or_above_p99")
    .reset_index()
)
cutoff_audit = cutoff_summary.merge(
    lower_counts, on=["survey_year", "survey_month"], validate="one_to_one"
).merge(upper_counts, on=["survey_year", "survey_month"], validate="one_to_one")
cutoff_audit["task_id"] = OUTPUT_SLUG
cutoff_audit["variable"] = "Q25v2part2"
cutoff_audit["role"] = "target"
sce = sce.merge(
    cutoff_summary[["survey_year", "survey_month", "p01", "p99"]].rename(
        columns={"p01": "Q25v2part2_p01", "p99": "Q25v2part2_p99"}
    ),
    on=["survey_year", "survey_month"],
    how="left",
    validate="many_to_one",
)
step_rows.append(
    construction_step(
        OUTPUT_SLUG, 1, "Start from source observations",
        int(len(sce)), int(len(sce)), 0,
    )
)


# 2. Attach additional sources.
rows_before = len(sce)
sce["survey_q_index"] = sce["survey_date"].dt.year * 4 + sce["survey_date"].dt.quarter
sce = attach_macro_context(
    sce,
    origin_q_index_col="survey_q_index",
    required_columns=CORE_MACRO_COLUMNS,
)
if len(sce) != rows_before:
    raise SystemExit(f"{OUTPUT_SLUG}: macro attachment changed the row count.")
step_rows.append(
    construction_step(
        OUTPUT_SLUG, 2, "Attach additional sources",
        int(rows_before), int(len(sce)), int(rows_before) - int(len(sce)),
    )
)


# 3. Construct leads and lags.
rows_before = len(sce)
target_values = pd.to_numeric(sce["Q25v2part2"], errors="coerce")
step_rows.append(
    construction_step(
        OUTPUT_SLUG, 3, "Construct leads and lags",
        int(rows_before), int(len(sce)), int(rows_before) - int(len(sce)),
    )
)


# 4. Restrict universe.
rows_before = len(sce)
step_rows.append(
    construction_step(
        OUTPUT_SLUG, 4, "Restrict universe",
        int(rows_before), int(len(sce)), int(rows_before) - int(len(sce)),
    )
)


# 5. Require observed target(s).
rows_before = len(sce)
condition_masks = []
detail_order = 1
for column in TARGET_COLUMNS:
    pass_condition = f'sce["{column}"].notna()'
    pass_mask = sce[column].notna()
    condition_masks.append(pass_mask)
    drop_detail_rows.append(
        filter_count(
            OUTPUT_SLUG, 5, detail_order,
            f"target_{column}_not_missing", pass_condition, int((~pass_mask).sum()),
        )
    )
    detail_order += 1
pass_condition = 'pd.to_numeric(sce["Q25v2part2"], errors="coerce").notna()'
pass_mask = target_values.notna()
condition_masks.append(pass_mask)
drop_detail_rows.append(
    filter_count(
        OUTPUT_SLUG, 5, detail_order,
        "target_Q25v2part2_numeric", pass_condition, int((~pass_mask).sum()),
    )
)
target_observed = pd.Series(True, index=sce.index)
for pass_mask in condition_masks:
    target_observed &= pass_mask
sce = sce.loc[target_observed].copy()
target_values = pd.to_numeric(sce["Q25v2part2"], errors="coerce")
step_rows.append(
    construction_step(
        OUTPUT_SLUG, 5, "Require observed targets",
        int(rows_before), int(len(sce)), int(rows_before) - int(len(sce)),
    )
)


# 6. Require observed predictors.
rows_before = len(sce)
condition_masks = []
detail_order = 1
for column in ["_HH_INC_DETAILED__label", *REQUIRED_CONTEXT_COLUMNS]:
    pass_condition = f'sce["{column}"].notna()'
    pass_mask = sce[column].notna()
    condition_masks.append(pass_mask)
    drop_detail_rows.append(
        filter_count(
            OUTPUT_SLUG, 6, detail_order,
            f"predictor_{column}_not_missing", pass_condition, int((~pass_mask).sum()),
        )
    )
    detail_order += 1
predictors_observed = pd.Series(True, index=sce.index)
for pass_mask in condition_masks:
    predictors_observed &= pass_mask
sce = sce.loc[predictors_observed].copy()
target_values = pd.to_numeric(sce["Q25v2part2"], errors="coerce")
step_rows.append(
    construction_step(
        OUTPUT_SLUG, 6, "Require observed predictors",
        int(rows_before), int(len(sce)), int(rows_before) - int(len(sce)),
    )
)


# 7. Apply logical and validity filters.
rows_before = len(sce)
pool = sce.copy()
age_pass_mask = pool["age_years"].between(18, 100, inclusive="both")
drop_detail_rows.append(
    filter_count(
        OUTPUT_SLUG, 7, 1 + sum(row["step"] == 7 for row in drop_detail_rows),
        "age_years_between_18_and_100", 'pool["age_years"].between(18, 100, inclusive="both")', int((~age_pass_mask).sum()),
    )
)
pool = pool.loc[age_pass_mask].copy()
step_rows.append(
    construction_step(
        OUTPUT_SLUG, 7, "Apply logical and validity filters",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)


# 8. Apply outlier trimming.
rows_before = len(pool)
if pool[["Q25v2part2_p01", "Q25v2part2_p99"]].isna().any().any():
    missing_periods = pool.loc[
        pool[["Q25v2part2_p01", "Q25v2part2_p99"]].isna().any(axis=1),
        ["survey_year", "survey_month"],
    ].drop_duplicates()
    raise SystemExit(f"{OUTPUT_SLUG}: eligible rows lack monthly Q25v2part2 cutoffs: {missing_periods.to_dict('records')}")
pass_condition = 'pool["Q25v2part2"].gt(pool["Q25v2part2_p01"]) & pool["Q25v2part2"].lt(pool["Q25v2part2_p99"])'
trim_mask = pool["Q25v2part2"].gt(pool["Q25v2part2_p01"]) & pool["Q25v2part2"].lt(pool["Q25v2part2_p99"])
drop_detail_rows.append(
    filter_count(
        OUTPUT_SLUG, 8, 1,
        "trim_Q25v2part2_target_strict_inside_month_p01_p99", pass_condition, int((~trim_mask).sum()),
    )
)
pool = pool.loc[trim_mask].copy()
step_rows.append(
    construction_step(
        OUTPUT_SLUG, 8, "Apply outlier trimming",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)
if pool.empty:
    raise SystemExit(f"{OUTPUT_SLUG} has no eligible rows.")

pool["selection_period"] = pd.to_datetime(pool["survey_date"]).dt.strftime("%Y-%m")
rows_before_export = len(pool)
# Public keys are fixed before sampling; source keys retain their sort types.
pool["id"] = pool["respondent_id"].astype("string")
pool["time"] = pd.to_datetime(pool["survey_date"], errors="raise").dt.strftime("%Y-%m-%d")
pool["release_date"] = pd.to_datetime(pool["release_date"], errors="raise").dt.strftime("%Y-%m-%d")
sample, sampling_registry, sampling_summary = sample_task_records(
    pool,
    task_id=OUTPUT_SLUG,
    output_dir=SAMPLING_DIR,
    output_sort_cols=["survey_date", "respondent_id"],
)

csv_sample = sample.copy()
csv_sample["subject_id"] = csv_sample["respondent_id"].astype(str)
for date_col in ["survey_date", "release_date"]:
    csv_sample[date_col] = pd.to_datetime(csv_sample[date_col], errors="coerce")
    if csv_sample[date_col].isna().any():
        raise SystemExit(f"Selected rows contain missing or invalid {date_col} values.")
    csv_sample[date_col] = csv_sample[date_col].dt.date.astype(str)


for output_column, label_column in CORE_LABEL_OUTPUT_COLUMNS.items():
    csv_sample[output_column] = csv_sample[label_column].astype("string")

optional_bounds = {
    "Q11": (0, 20, False),
    "Q16": (0, 240, True),
    "Q19": (0, 240, True),
}
for column, (lower, upper, include_upper) in optional_bounds.items():
    values = pd.to_numeric(csv_sample[column], errors="coerce")
    if include_upper:
        valid = values.between(lower, upper, inclusive="both")
    else:
        valid = values.ge(lower) & values.lt(upper)
    csv_sample[column] = values.where(valid, pd.NA)

required_output_columns = ["subject_id", "respondent_id", "survey_date", "release_date", "release_date_source"] + TARGET_COLUMNS + REQUIRED_CONTEXT_COLUMNS
required_missing = csv_sample[required_output_columns].isna().sum()
if required_missing.any():
    raise SystemExit(f"Selected rows contain missing required values: {required_missing[required_missing > 0].to_dict()}")

TABULAR_PATH.parent.mkdir(parents=True, exist_ok=True)
canonicalize_public_table(csv_sample, task_id=OUTPUT_SLUG).to_csv(TABULAR_PATH, index=False)
step_rows.append(
    construction_step(
        OUTPUT_SLUG, 9, "Export sample",
        int(rows_before_export), int(len(csv_sample)), int(rows_before_export) - int(len(csv_sample)),
    )
)

DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(step_rows, columns=STEP_COLUMNS).to_csv(STEP_DIAGNOSTICS_PATH, index=False)
pd.DataFrame(drop_detail_rows, columns=DROP_DETAIL_COLUMNS).to_csv(DROP_DETAIL_PATH, index=False)
cutoff_audit.loc[:, OUTLIER_CUTOFF_COLUMNS].to_csv(OUTLIER_CUTOFFS_PATH, index=False)
print(f"Wrote {TABULAR_PATH.relative_to(PROJECT_ROOT)} with {len(csv_sample):,} rows.")
