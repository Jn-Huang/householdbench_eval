#!/usr/bin/env python
"""Build the HouseholdBench tabular file for SCE moving probability."""

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


OUTPUT_SLUG = 'house_sce_move'
INPUT_PATH = PROJECT_ROOT / "data" / "intermediate" / "sce_core.parquet"
TABULAR_PATH = PROJECT_ROOT / "data" / "householdbench" / "tabular" / f"{OUTPUT_SLUG}.csv"
DIAGNOSTICS_DIR = PROJECT_ROOT / "output" / "householdbench" / "tasks" / OUTPUT_SLUG
SAMPLING_DIR = PROJECT_ROOT / "data" / "intermediate" / "householdbench" / OUTPUT_SLUG
STEP_DIAGNOSTICS_PATH = DIAGNOSTICS_DIR / "01_sample_construction_steps.csv"
DROP_DETAIL_PATH = DIAGNOSTICS_DIR / "01_sample_construction_drop_details.csv"
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

TARGET_COLUMNS = ['Q3']

LAG1_COLUMNS = ['lag1_move_probability']
LAG1_DATE_COLUMN = 'lag1_survey_date'

PROBABILITY_COLUMNS = ['Q3']

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
 'Q41',
 'Q42',
 'Q43',
 'Q44',
 'D3',
 *CORE_MACRO_COLUMNS,
 LAG1_DATE_COLUMN,
 'lag1_move_probability',
 'Q3']


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
sce["lag1_move_probability"] = sce.groupby("respondent_id", sort=False)["Q3"].shift(1)
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
target_values = pd.to_numeric(sce["Q3"], errors="coerce")
target_observed = target_values.notna()
drop_detail_rows.append(
    filter_count(
        OUTPUT_SLUG, 5, 1,
        "target_Q3_numeric", 'pd.to_numeric(sce["Q3"], errors="coerce").notna()', int((~target_observed).sum()),
    )
)
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
    & pd.to_numeric(sce["lag1_move_probability"], errors="coerce").between(0, 100)
)
condition_masks.append(lag1_contiguous)
drop_detail_rows.append(filter_count(
                            OUTPUT_SLUG, 6, detail_order,
                            "predictor_complete_contiguous_lag1_move_distribution", "complete contiguous lag-1 move probability", int((~lag1_contiguous).sum()),
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
for optional_duration in ["Q16", "Q19"]:
    values = pd.to_numeric(sce[optional_duration], errors="coerce")
    sce.loc[values.lt(0) | values.gt(240), optional_duration] = pd.NA

target_nonmissing = int(pd.to_numeric(sce["Q3"], errors="coerce").notna().sum())
target_missing = int(len(sce) - target_nonmissing)
target_out_of_range = int((~pd.to_numeric(sce["Q3"], errors="coerce").between(0, 100) & pd.to_numeric(sce["Q3"], errors="coerce").notna()).sum())
condition_masks = []
detail_order = 1
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
step_rows.append(
    construction_step(
        OUTPUT_SLUG, 8, "Apply outlier trimming",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)

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

probability_columns = [*TARGET_COLUMNS, *LAG1_COLUMNS]
csv_sample[probability_columns] = csv_sample[probability_columns].apply(pd.to_numeric, errors="coerce") / 100
if not csv_sample[probability_columns].apply(lambda column: column.between(0, 1)).all().all():
    raise SystemExit("Selected probability values must lie between 0 and 1 after normalization.")

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
print(f"Wrote {TABULAR_PATH.relative_to(PROJECT_ROOT)} with {len(csv_sample):,} rows.")
print(
    f"{OUTPUT_SLUG}: Q3 target filter missing={target_missing:,} "
    f"out_of_range={target_out_of_range:,} nonmissing={target_nonmissing:,}"
)
