#!/usr/bin/env python
"""Build the HouseholdBench tabular file for SCE macro-risk uncertainty beliefs."""

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


OUTPUT_SLUG = 'macro_sce_uncertainty'
INPUT_PATH = PROJECT_ROOT / "data" / "intermediate" / "sce_core.parquet"
TABULAR_PATH = PROJECT_ROOT / "data" / "householdbench" / "tabular" / f"{OUTPUT_SLUG}.csv"
DIAGNOSTICS_DIR = PROJECT_ROOT / "output" / "householdbench" / "tasks" / OUTPUT_SLUG
SAMPLING_DIR = PROJECT_ROOT / "data" / "intermediate" / "householdbench" / OUTPUT_SLUG
STEP_DIAGNOSTICS_PATH = DIAGNOSTICS_DIR / "01_sample_construction_steps.csv"
DROP_DETAIL_PATH = DIAGNOSTICS_DIR / "01_sample_construction_drop_details.csv"
OUTLIER_CUTOFFS_PATH = DIAGNOSTICS_DIR / "01_outlier_cutoffs.csv"
AUXILIARY_DIAGNOSTICS_PATH = DIAGNOSTICS_DIR / "01_auxiliary_diagnostics.csv"
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
 'Q9_bin1__clean',
 'Q9_bin2__clean',
 'Q9_bin3__clean',
 'Q9_bin4__clean',
 'Q9_bin5__clean',
 'Q9_bin6__clean',
 'Q9_bin7__clean',
 'Q9_bin8__clean',
 'Q9_bin9__clean',
 'Q9_bin10__clean',
 'Q9__task_valid',
 'Q9c_mean',
 'Q9c_iqr',
 'Q9c_bin1__clean',
 'Q9c_bin2__clean',
 'Q9c_bin3__clean',
 'Q9c_bin4__clean',
 'Q9c_bin5__clean',
 'Q9c_bin6__clean',
 'Q9c_bin7__clean',
 'Q9c_bin8__clean',
 'Q9c_bin9__clean',
 'Q9c_bin10__clean',
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
 'Q41',
 'Q42',
 'Q43',
 'Q44',
 'Q45b',
 'D1',
 'D3',
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
    "Q43": "Q43__label",
    "Q44": "Q44__label",
    "D3": "D3__label",
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

Q9_BIN_COLUMNS = [f"Q9_bin{i}__clean" for i in range(1, 11)]
Q9C_BIN_COLUMNS = [f"Q9c_bin{i}__clean" for i in range(1, 11)]
TARGET_COLUMNS = Q9_BIN_COLUMNS + Q9C_BIN_COLUMNS + ['Q4new__clean', 'Q5new__clean', 'Q6new__clean']

LAG1_COLUMNS = (
    [f"lag1_{column}" for column in Q9_BIN_COLUMNS]
    + [f"lag1_{column}" for column in Q9C_BIN_COLUMNS]
    + ["lag1_unemployment_higher_probability", "lag1_savings_rate_higher_probability", "lag1_stock_prices_higher_probability"]
)
LAG1_DATE_COLUMN = 'lag1_survey_date'
AUXILIARY_DIAGNOSTIC_COLUMNS = ['Q9_iqr', 'Q9c_iqr']
OBSERVED_TARGET_COLUMNS = TARGET_COLUMNS + AUXILIARY_DIAGNOSTIC_COLUMNS

PROBABILITY_COLUMNS = TARGET_COLUMNS
EVENT_PROBABILITY_COLUMNS = [
    "Q4new__clean",
    "Q5new__clean",
    "Q6new__clean",
    "lag1_unemployment_higher_probability",
    "lag1_savings_rate_higher_probability",
    "lag1_stock_prices_higher_probability",
]
PROBABILITY_VECTOR_COLUMNS = [
    Q9_BIN_COLUMNS,
    Q9C_BIN_COLUMNS,
    [f"lag1_{column}" for column in Q9_BIN_COLUMNS],
    [f"lag1_{column}" for column in Q9C_BIN_COLUMNS],
]

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
 'Q9_iqr',
 'Q9c_iqr',
 *Q9_BIN_COLUMNS,
 *Q9C_BIN_COLUMNS,
 LAG1_DATE_COLUMN,
 *LAG1_COLUMNS,
 'Q4new__clean',
 'Q5new__clean',
 'Q6new__clean']


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
reference = sce.loc[sce[reference_keys].notna().all(axis=1), reference_keys].copy()
if reference.duplicated(reference_keys).any():
    raise SystemExit(f"{OUTPUT_SLUG}: source-wide reference is not unique by respondent and survey month.")
cutoff_audit_frames = []
for variable in AUXILIARY_DIAGNOSTIC_COLUMNS:
    cutoff_column = f"{variable}_trim"
    sce[cutoff_column] = pd.to_numeric(sce[variable], errors="coerce")
    variable_reference = sce.loc[
        sce[reference_keys].notna().all(axis=1) & sce[cutoff_column].notna(),
        reference_keys + [cutoff_column],
    ].copy()
    variable_summary = (
        variable_reference.groupby(["survey_year", "survey_month"])[cutoff_column]
        .agg(n_reference="size", p01=lambda values: values.quantile(0.01), p99=lambda values: values.quantile(0.99))
        .reset_index()
    )
    if variable_summary.empty or variable_summary[["p01", "p99"]].isna().any().any():
        raise SystemExit(f"{OUTPUT_SLUG}: missing source-wide monthly cutoffs for {variable}.")
    degenerate = variable_summary["p01"].eq(variable_summary["p99"])
    if degenerate.any():
        bad = variable_summary.loc[degenerate, ["survey_year", "survey_month", "n_reference", "p01"]]
        raise SystemExit(f"{OUTPUT_SLUG}: degenerate monthly {variable} cutoffs: {bad.to_dict('records')}")
    variable_summary["period"] = (
        variable_summary["survey_year"].astype(int).astype(str)
        + "-" + variable_summary["survey_month"].astype(int).astype(str).str.zfill(2)
    )
    variable_reference = variable_reference.merge(
        variable_summary, on=["survey_year", "survey_month"], how="left", validate="many_to_one"
    )
    lower_counts = (
        variable_reference[cutoff_column].le(variable_reference["p01"])
        .groupby([variable_reference["survey_year"], variable_reference["survey_month"]])
        .sum().rename("n_at_or_below_p01").reset_index()
    )
    upper_counts = (
        variable_reference[cutoff_column].ge(variable_reference["p99"])
        .groupby([variable_reference["survey_year"], variable_reference["survey_month"]])
        .sum().rename("n_at_or_above_p99").reset_index()
    )
    variable_audit = variable_summary.merge(
        lower_counts, on=["survey_year", "survey_month"], validate="one_to_one"
    ).merge(upper_counts, on=["survey_year", "survey_month"], validate="one_to_one")
    variable_audit["task_id"] = OUTPUT_SLUG
    variable_audit["variable"] = variable
    variable_audit["role"] = "target"
    cutoff_audit_frames.append(variable_audit)
    sce = sce.merge(
        variable_summary[["survey_year", "survey_month", "p01", "p99"]].rename(
            columns={"p01": f"{variable}_p01", "p99": f"{variable}_p99"}
        ),
        on=["survey_year", "survey_month"], how="left", validate="many_to_one",
    )
cutoff_audit = pd.concat(cutoff_audit_frames, ignore_index=True)
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
sce = sce.sort_values(["respondent_id", "survey_date"], kind="mergesort").copy()
sce["survey_month_index"] = sce["survey_date"].dt.year * 12 + sce["survey_date"].dt.month
sce["lag1_survey_date"] = sce.groupby("respondent_id", sort=False)["survey_date"].shift(1)
sce["lag1_panel_month"] = sce.groupby("respondent_id", sort=False)["panel_month"].shift(1)
sce["lag1_survey_month_index"] = sce.groupby("respondent_id", sort=False)["survey_month_index"].shift(1)
sce["lag1_Q9__task_valid"] = sce.groupby("respondent_id", sort=False)["Q9__task_valid"].shift(1)
sce["lag1_Q9c__task_valid"] = sce.groupby("respondent_id", sort=False)["Q9c__task_valid"].shift(1)
for source_column, lag_column in zip(
    TARGET_COLUMNS,
    LAG1_COLUMNS,
    strict=True,
):
    sce[lag_column] = sce.groupby("respondent_id", sort=False)[source_column].shift(1)
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
target_observed = pd.Series(True, index=sce.index)
for column in OBSERVED_TARGET_COLUMNS:
    pass_condition = f'pd.to_numeric(sce["{column}"], errors="coerce").notna()'
    pass_mask = pd.to_numeric(sce[column], errors="coerce").notna()
    condition_masks.append(pass_mask)
    drop_detail_rows.append(
        filter_count(
            OUTPUT_SLUG, 5, detail_order,
            f"target_{column}_numeric", pass_condition, int((~pass_mask).sum()),
        )
    )
    detail_order += 1
for pass_mask in condition_masks:
    target_observed &= pass_mask
sce = sce.loc[target_observed].copy()
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
for column in ["_HH_INC_DETAILED__label", *REQUIRED_CONTEXT_COLUMNS, *LAG1_COLUMNS]:
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
lag1_contiguous = (
    sce["lag1_survey_date"].notna()
    & (pd.to_numeric(sce["panel_month"], errors="coerce") - pd.to_numeric(sce["lag1_panel_month"], errors="coerce")).eq(1)
    & (sce["survey_month_index"] - sce["lag1_survey_month_index"]).eq(1)
)
for column in LAG1_COLUMNS:
    lag1_contiguous &= pd.to_numeric(sce[column], errors="coerce").between(0, 100)
lag1_contiguous &= sce["lag1_Q9__task_valid"].eq(True) & sce["lag1_Q9c__task_valid"].eq(True)
condition_masks.append(lag1_contiguous)
drop_detail_rows.append(filter_count(
                            OUTPUT_SLUG, 6, detail_order,
                            "predictor_complete_contiguous_lag1_uncertainty_distributions", "complete contiguous lag-1 uncertainty distribution vector", int((~lag1_contiguous).sum()),
                        ))
predictors_observed = pd.Series(True, index=sce.index)
for pass_mask in condition_masks:
    predictors_observed &= pass_mask
sce = sce.loc[predictors_observed].copy()
step_rows.append(
    construction_step(
        OUTPUT_SLUG, 6, "Require observed predictors",
        int(rows_before), int(len(sce)), int(rows_before) - int(len(sce)),
    )
)


# 7. Apply logical and validity filters.
rows_before = len(sce)
condition_masks = []
detail_order = 1
for column in ["Q9__task_valid", "Q9c__task_valid"]:
    pass_condition = f'sce["{column}"].eq(True)'
    pass_mask = sce[column].eq(True)
    condition_masks.append(pass_mask)
    drop_detail_rows.append(
        filter_count(
            OUTPUT_SLUG, 7, detail_order,
            f"{column}_true", pass_condition, int((~pass_mask).sum()),
        )
    )
    detail_order += 1
for column in PROBABILITY_COLUMNS:
    values = pd.to_numeric(sce[column], errors="coerce")
    pass_condition = f'pd.to_numeric(sce["{column}"], errors="coerce").between(0, 100)'
    pass_mask = values.between(0, 100)
    condition_masks.append(pass_mask)
    drop_detail_rows.append(
        filter_count(
            OUTPUT_SLUG, 7, detail_order,
            f"{column}_between_0_and_100", pass_condition, int((~pass_mask).sum()),
        )
    )
    detail_order += 1
quality_mask = pd.Series(True, index=sce.index)
for pass_mask in condition_masks:
    quality_mask &= pass_mask

pool = sce.loc[quality_mask].copy()
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
if pool.empty:
    raise SystemExit(f"{OUTPUT_SLUG} has no eligible rows.")


# 8. Apply outlier trimming.
rows_before = len(pool)
trim_conditions = []
detail_order = 1
for variable in AUXILIARY_DIAGNOSTIC_COLUMNS:
    if pool[[f"{variable}_p01", f"{variable}_p99"]].isna().any().any():
        missing_periods = pool.loc[
            pool[[f"{variable}_p01", f"{variable}_p99"]].isna().any(axis=1),
            ["survey_year", "survey_month"],
        ].drop_duplicates()
        raise SystemExit(f"{OUTPUT_SLUG}: eligible rows lack monthly {variable} cutoffs: {missing_periods.to_dict('records')}")
    pass_mask = pool[variable].gt(pool[f"{variable}_p01"]) & pool[variable].lt(pool[f"{variable}_p99"])
    pass_condition = (
        f'pool["{variable}"].gt(pool["{variable}_p01"]) '
        f'& pool["{variable}"].lt(pool["{variable}_p99"])'
    )
    trim_conditions.append(pass_mask)
    drop_detail_rows.append(
        filter_count(
            OUTPUT_SLUG, 8, detail_order,
            f"trim_{variable}_target_strict_inside_month_p01_p99", pass_condition, int((~pass_mask).sum()),
        )
    )
    detail_order += 1
trim_mask = pd.Series(True, index=pool.index)
for pass_mask in trim_conditions:
    trim_mask &= pass_mask
pool = pool.loc[trim_mask].copy()
step_rows.append(
    construction_step(
        OUTPUT_SLUG, 8, "Apply outlier trimming",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)
if pool.empty:
    raise SystemExit(f"{OUTPUT_SLUG} has no eligible rows after outlier trimming.")

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

required_output_columns = ["subject_id", "respondent_id", "survey_date", "release_date", "release_date_source", LAG1_DATE_COLUMN] + TARGET_COLUMNS + REQUIRED_CONTEXT_COLUMNS + LAG1_COLUMNS
required_missing = csv_sample[required_output_columns].isna().sum()
if required_missing.any():
    raise SystemExit(f"Selected rows contain missing required values: {required_missing[required_missing > 0].to_dict()}")

for columns in PROBABILITY_VECTOR_COLUMNS:
    values = csv_sample[columns].apply(pd.to_numeric, errors="coerce")
    totals = values.sum(axis=1)
    if totals.le(0).any():
        raise SystemExit("Selected probability vectors must have positive totals before normalization.")
    scaled = values.div(totals, axis=0).round(3)
    residual = (1 - scaled.sum(axis=1)).round(3)
    maximum_columns = scaled.idxmax(axis=1)
    for column in columns:
        rows = maximum_columns.eq(column)
        scaled.loc[rows, column] = scaled.loc[rows, column] + residual.loc[rows]
    csv_sample[columns] = scaled
    if not csv_sample[columns].apply(lambda column: column.between(0, 1)).all().all():
        raise SystemExit("Selected probability-vector values must lie between 0 and 1 after normalization.")
    if not csv_sample[columns].sum(axis=1).sub(1).abs().le(1e-9).all():
        raise SystemExit("Selected probability vectors must sum to 1 after normalization.")

csv_sample[EVENT_PROBABILITY_COLUMNS] = (
    csv_sample[EVENT_PROBABILITY_COLUMNS].apply(pd.to_numeric, errors="coerce") / 100
)
if not csv_sample[EVENT_PROBABILITY_COLUMNS].apply(lambda column: column.between(0, 1)).all().all():
    raise SystemExit("Selected event probabilities must lie between 0 and 1 after normalization.")

TABULAR_PATH.parent.mkdir(parents=True, exist_ok=True)
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
public_table = canonicalize_public_table(csv_sample, task_id=OUTPUT_SLUG)
auxiliary_diagnostics = public_table.loc[:, ["id", "time", "release_date"]].copy()
for column in AUXILIARY_DIAGNOSTIC_COLUMNS:
    auxiliary_diagnostics[column] = pd.to_numeric(csv_sample[column], errors="coerce").to_numpy()
if auxiliary_diagnostics[AUXILIARY_DIAGNOSTIC_COLUMNS].isna().any().any():
    raise SystemExit("Selected auxiliary uncertainty diagnostics contain missing values.")
if auxiliary_diagnostics.duplicated(["id", "time", "release_date"]).any():
    raise SystemExit("Selected auxiliary uncertainty diagnostics contain duplicate keys.")
public_table.to_csv(TABULAR_PATH, index=False)
auxiliary_diagnostics.to_csv(AUXILIARY_DIAGNOSTICS_PATH, index=False)
step_rows.append(
    construction_step(
        OUTPUT_SLUG, 9, "Export sample",
        int(rows_before_export), int(len(csv_sample)), int(rows_before_export) - int(len(csv_sample)),
    )
)

pd.DataFrame(step_rows, columns=STEP_COLUMNS).to_csv(STEP_DIAGNOSTICS_PATH, index=False)
pd.DataFrame(drop_detail_rows, columns=DROP_DETAIL_COLUMNS).to_csv(DROP_DETAIL_PATH, index=False)
cutoff_audit.loc[:, OUTLIER_CUTOFF_COLUMNS].to_csv(OUTLIER_CUTOFFS_PATH, index=False)
print(f"Wrote {TABULAR_PATH.relative_to(PROJECT_ROOT)} with {len(csv_sample):,} rows.")
