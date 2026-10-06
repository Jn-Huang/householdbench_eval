#!/usr/bin/env python
"""Build the tabular CSV for the SCE public-policy household impact task."""

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


TASK_ID = 'income_sce_policy'
INPUT_PATH = PROJECT_ROOT / "data" / "intermediate" / "sce_public_policy.parquet"
CORE_INPUT_PATH = PROJECT_ROOT / "data" / "intermediate" / "sce_core.parquet"
TABULAR_PATH = PROJECT_ROOT / "data" / "householdbench" / "tabular" / f"{TASK_ID}.csv"
DIAGNOSTICS_DIR = PROJECT_ROOT / "output" / "householdbench" / "tasks" / TASK_ID
SAMPLING_DIR = PROJECT_ROOT / "data" / "intermediate" / "householdbench" / TASK_ID
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

POLICY_ITEMS = {'welfare_benefits': {'suffix': 5},
 'unemployment_benefits': {'suffix': 6},
 'payroll_tax_rate': {'suffix': 9},
 'average_income_tax_rate': {'suffix': 16}}

POLICY_IMPACT_LABELS = {1: 'very_negative',
 2: 'somewhat_negative',
 3: 'no_impact',
 4: 'somewhat_positive',
 5: 'very_positive'}

POLICY_DIRECTION_LABELS = {1: 'increase_or_expansion', 2: 'decrease_or_reduction'}

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

TASK_INPUT_COLUMNS = ['hidqp2_5',
 'qp2_5',
 'qp1x1_5',
 'qp1x2_5',
 'qp1x3_5',
 'hidqp2_6',
 'qp2_6',
 'qp1x1_6',
 'qp1x2_6',
 'qp1x3_6',
 'hidqp2_9',
 'qp2_9',
 'qp1x1_9',
 'qp1x2_9',
 'qp1x3_9',
 'hidqp2_16',
 'qp2_16',
 'qp1x1_16',
 'qp1x2_16',
 'qp1x3_16']

TASK_OUTPUT_COLUMNS = ['hidqp2_5',
 'qp2_5',
 'qp1x1_5',
 'qp1x2_5',
 'qp1x3_5',
 'hidqp2_6',
 'qp2_6',
 'qp1x1_6',
 'qp1x2_6',
 'qp1x3_6',
 'hidqp2_9',
 'qp2_9',
 'qp1x1_9',
 'qp1x2_9',
 'qp1x3_9',
 'hidqp2_16',
 'qp2_16',
 'qp1x1_16',
 'qp1x2_16',
 'qp1x3_16']

TASK_LABEL_OUTPUT_COLUMNS = {
    "hidqp2_5": "hidqp2_5__label",
    "qp2_5": "qp2_5__label",
    "hidqp2_6": "hidqp2_6__label",
    "qp2_6": "qp2_6__label",
    "hidqp2_9": "hidqp2_9__label",
    "qp2_9": "qp2_9__label",
    "hidqp2_16": "hidqp2_16__label",
    "qp2_16": "qp2_16__label",
}


step_rows = []
drop_detail_rows = []


# 1. Start from pre-processed main source.
if not INPUT_PATH.is_file():
    raise RuntimeError(f"Missing cleaned SCE public-policy file: {INPUT_PATH}")
sce = pd.read_parquet(INPUT_PATH)
require_columns(sce, BASE_INPUT_COLUMNS + TASK_INPUT_COLUMNS + list(TASK_LABEL_OUTPUT_COLUMNS.values()), label=str(INPUT_PATH))
sce["survey_date"] = pd.to_datetime(sce["survey_date"], errors="coerce")
sce["release_date"] = pd.to_datetime(sce["release_date"], errors="coerce")
if sce["survey_date"].isna().any():
    raise RuntimeError(f"{INPUT_PATH} contains missing or invalid survey_date values.")
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
required_policy_columns = []
for item in POLICY_ITEMS.values():
    suffix = item["suffix"]
    q_column = f"qp2_{suffix}"
    direction_column = f"hidqp2_{suffix}"
    for column, valid_codes in [(q_column, set(POLICY_IMPACT_LABELS)), (direction_column, set(POLICY_DIRECTION_LABELS))]:
        values = pd.to_numeric(pool[column], errors="coerce").dropna()
        observed = set(values.round().astype(int).unique())
        unexpected = sorted(observed - valid_codes)
        if unexpected:
            raise RuntimeError(f"{TASK_ID}: unexpected codes in {column}: {unexpected}")
    required_policy_columns.extend([q_column, direction_column])
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
direction_columns = [f"hidqp2_{item['suffix']}" for item in POLICY_ITEMS.values()]
impact_columns = [f"qp2_{item['suffix']}" for item in POLICY_ITEMS.values()]
condition_masks = []
detail_order = 1
for column in required_policy_columns:
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
if pool.empty:
    raise RuntimeError(f"{TASK_ID} has zero eligible rows.")


# 8. Apply outlier trimming.
rows_before = len(pool)
step_rows.append(
    construction_step(
        TASK_ID, 8, "Apply outlier trimming",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)

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
required_selected_columns = [
    'subject_id',
    'respondent_id',
    'survey_year',
    'survey_month',
    'survey_date',
    'release_date',
    'release_date_source',
    'age_years',
    '_NUM_CAT',
    '_REGION_CAT',
    '_EDU_CAT',
    '_HH_INC_DETAILED',
    *CORE_MACRO_COLUMNS,
] + required_policy_columns
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

print(
    f"{TASK_ID}: eligible={len(pool):,} sampled={len(sample):,} "
    f"tabular={TABULAR_PATH}"
)
