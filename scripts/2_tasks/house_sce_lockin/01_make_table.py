#!/usr/bin/env python
"""Build the tabular CSV for the SCE mortgage lock-in moving probability task."""

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

from scripts.utils.macro_context import (
    CORE_MACRO_COLUMNS,
    MORTGAGE_CONTEXT_COLUMNS,
    attach_macro_context,
)
from scripts.utils.sampling import sample_task_records


TASK_ID = 'house_sce_lockin'
INPUT_PATH = PROJECT_ROOT / "data" / "intermediate" / "sce_housing.parquet"
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

HOUSING_YES_NO = {1: 'yes', 2: 'no'}

MORTGAGE_TYPE = {1: 'adjustable_or_floating_rate', 2: 'fixed_rate', 3: 'dont_know'}

LOAN_STATUS = {1: 'mortgage_only',
 2: 'home_equity_only',
 3: 'mortgage_and_home_equity',
 4: 'no_loans_against_home'}

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
 *CORE_MACRO_COLUMNS,
 *MORTGAGE_CONTEXT_COLUMNS]

BASE_INPUT_COLUMNS = ['respondent_id',
 'survey_year',
 'survey_month',
 'survey_date',
 'release_date',
 'release_date_source']

TASK_INPUT_COLUMNS = ['q6a_lock']

OPTIONAL_OUTPUT_COLUMNS = ['q1_1',
 'HQ1_1',
 'q38',
 'hq38',
 'HQ38',
 'qh0',
 'HQH0',
 'qh2_1',
 'HQH2_1',
 'qh3_1',
 'HQH3_1',
 'qh4a',
 'HQH4a',
 'qh5',
 'qh5m2_1',
 'qh5a_1',
 'HQH5a_1',
 'qh5m',
 'HQH5m',
 'HQH5',
 'HQH5m2_1',
 'q6a_1',
 'hq6a_1',
 'HQ6a_1']

TASK_LABEL_OUTPUT_COLUMNS = {
    "q38": "q38__label",
    "hq38": "hq38__label",
    "HQ38": "HQ38__label",
    "qh0": "qh0__label",
    "HQH0": "HQH0__label",
    "qh4a": "qh4a__label",
    "HQH4a": "HQH4a__label",
    "qh5": "qh5__label",
    "HQH5": "HQH5__label",
    "qh5m": "qh5m__label",
    "HQH5m": "HQH5m__label",
}
CANONICAL_OUTPUT_COLUMNS = {
    "local_home_value": ["q1_1", "HQ1_1"],
    "partnered": ["q38", "hq38", "HQ38"],
    "owns_other_home": ["qh0", "HQH0"],
    "purchase_price": ["qh2_1", "HQH2_1"],
    "current_home_value": ["qh3_1", "HQH3_1"],
    "loan_balance": ["qh5a_1", "HQH5a_1"],
    "loan_status": ["qh5", "HQH5"],
    "mortgage_rate_type": ["qh5m", "HQH5m"],
    "current_mortgage_rate": ["qh5m2_1", "HQH5m2_1"],
}


def coalesce_existing_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    existing = [column for column in columns if column in df.columns]
    if not existing:
        return pd.Series(pd.NA, index=df.index)
    return df[existing].bfill(axis=1).iloc[:, 0]


step_rows = []
drop_detail_rows = []


# 1. Start from pre-processed main source.
if not INPUT_PATH.is_file():
    raise RuntimeError(f"Missing cleaned SCE housing file: {INPUT_PATH}")
sce = pd.read_parquet(INPUT_PATH)
require_columns(sce, BASE_INPUT_COLUMNS + TASK_INPUT_COLUMNS, label=str(INPUT_PATH))
mortgage_status_column = None
for candidate_column in ["qh5", "HQH5"]:
    if candidate_column in sce.columns:
        mortgage_status_column = candidate_column
        break
if mortgage_status_column is None:
    raise RuntimeError(f"{INPUT_PATH} is missing all required alternatives: ['qh5', 'HQH5']")
mortgage_rate_column = None
for candidate_column in ["qh5m2_1", "HQH5m2_1"]:
    if candidate_column in sce.columns:
        mortgage_rate_column = candidate_column
        break
if mortgage_rate_column is None:
    raise RuntimeError(f"{INPUT_PATH} is missing all required alternatives: ['qh5m2_1', 'HQH5m2_1']")
sce["survey_date"] = pd.to_datetime(sce["survey_date"], errors="coerce")
sce["release_date"] = pd.to_datetime(sce["release_date"], errors="coerce")
if sce["survey_date"].isna().any():
    raise RuntimeError(f"{INPUT_PATH} contains missing or invalid survey_date values.")
reference_keys = ["respondent_id", "survey_year", "survey_month"]
reference = sce.loc[sce[reference_keys].notna().all(axis=1), reference_keys].copy()
if reference.duplicated(reference_keys).any():
    raise RuntimeError(f"{TASK_ID}: source-wide reference is not unique by respondent and survey month.")
trim_source_columns = {
    "current_mortgage_rate": ["qh5m2_1", "HQH5m2_1"],
    "local_home_value": ["q1_1", "HQ1_1"],
    "purchase_price": ["qh2_1", "HQH2_1"],
    "current_home_value": ["qh3_1", "HQH3_1"],
    "loan_balance": ["qh5a_1", "HQH5a_1"],
}
cutoff_audit_frames = []
for variable, candidate_columns in trim_source_columns.items():
    task_column = f"_trim_{variable}"
    sce[task_column] = pd.to_numeric(coalesce_existing_columns(sce, candidate_columns), errors="coerce")
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
        variable_summary,
        on=["survey_year", "survey_month"],
        how="left",
        validate="many_to_one",
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
    variable_audit["role"] = "predictor"
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
    required_columns=[*CORE_MACRO_COLUMNS, *MORTGAGE_CONTEXT_COLUMNS],
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
pool["lockin_move_probability"] = pd.to_numeric(pool["q6a_lock"], errors="coerce")
pool["ordinary_move_probability"] = pd.to_numeric(
    coalesce_existing_columns(pool, ["Q6a_1", "q6a_1", "hq6a_1", "HQ6a_1"]),
    errors="coerce",
)
mortgage_status = pd.to_numeric(pool[mortgage_status_column], errors="coerce")
mortgage_rate = pd.to_numeric(pool[mortgage_rate_column], errors="coerce")
valid_lockin_answer = pool["lockin_move_probability"].between(0, 100, inclusive="both")
has_current_mortgage_rate = valid_lockin_answer & mortgage_rate.notna()
active_mortgage_and_rate_observed = (
    has_current_mortgage_rate
    & mortgage_status.isin([1, 3])
    & pool["release_date"].notna()
)
eligible = active_mortgage_and_rate_observed & pool[COMMON_PROMPT_CONTEXT_COLS].notna().all(axis=1)
sample_path_counts = {
    "linked_housing_core": int(len(pool)),
    "valid_lockin_answer": int(valid_lockin_answer.sum()),
    "current_mortgage_rate": int(has_current_mortgage_rate.sum()),
    "active_mortgage_rate_observed": int(active_mortgage_and_rate_observed.sum()),
    "shared_prompt_context": int(eligible.sum()),
}
step_rows.append(
    construction_step(
        TASK_ID, 3, "Construct leads and lags",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)


# 4. Restrict universe.
rows_before = len(pool)
active_mortgage_universe = mortgage_status.isin([1, 3])
drop_detail_rows.append(
    filter_count(
        TASK_ID, 4, 1,
        "active_mortgage_or_mortgage_and_home_equity", "mortgage_status.isin([1, 3])", int((~active_mortgage_universe).sum()),
    )
)
pool = pool.loc[active_mortgage_universe].copy()
mortgage_rate = pd.to_numeric(pool[mortgage_rate_column], errors="coerce")
step_rows.append(
    construction_step(
        TASK_ID, 4, "Restrict universe",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)


# 5. Require observed target(s).
rows_before = len(pool)
target_observed = pool["lockin_move_probability"].notna()
drop_detail_rows.append(
    filter_count(
        TASK_ID, 5, 1,
        "target_lockin_move_probability_not_missing", 'pool["lockin_move_probability"].notna()', int((~target_observed).sum()),
    )
)
pool = pool.loc[target_observed].copy()
mortgage_rate = pd.to_numeric(pool[mortgage_rate_column], errors="coerce")
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
for filter_id, pass_condition, pass_mask in [
    ("current_mortgage_rate_not_missing", f'pd.to_numeric(pool["{mortgage_rate_column}"], errors="coerce").notna()', mortgage_rate.notna()),
    ("release_date_not_missing", 'pool["release_date"].notna()', pool["release_date"].notna()),
    ("ordinary_move_probability_not_missing", 'pool["ordinary_move_probability"].notna()', pool["ordinary_move_probability"].notna()),
]:
    condition_masks.append(pass_mask)
    drop_detail_rows.append(
        filter_count(
            TASK_ID, 6, detail_order,
            filter_id, pass_condition, int((~pass_mask).sum()),
        )
    )
    detail_order += 1
for column in ["_HH_INC_DETAILED__label", *COMMON_PROMPT_CONTEXT_COLS]:
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
mortgage_rate = pd.to_numeric(pool[mortgage_rate_column], errors="coerce")
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
        "lockin_move_probability_between_0_and_100",
        'pool["lockin_move_probability"].between(0, 100, inclusive="both")',
        pool["lockin_move_probability"].between(0, 100, inclusive="both"),
    ),
    (
        "ordinary_move_probability_between_0_and_100",
        'pool["ordinary_move_probability"].between(0, 100, inclusive="both")',
        pool["ordinary_move_probability"].between(0, 100, inclusive="both"),
    ),
    (
        "current_mortgage_rate_positive",
        f'pd.to_numeric(pool["{mortgage_rate_column}"], errors="coerce").gt(0)',
        mortgage_rate.gt(0),
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
trim_conditions = []
detail_order = 1
home_value_fields_rendered = pool[
    ["_trim_local_home_value", "_trim_purchase_price", "_trim_current_home_value"]
].notna().all(axis=1)
for variable in trim_source_columns:
    task_column = f"_trim_{variable}"
    observed = pool[task_column].notna()
    missing_cutoff = observed & pool[[f"{task_column}_p01", f"{task_column}_p99"]].isna().any(axis=1)
    if missing_cutoff.any():
        missing_periods = pool.loc[missing_cutoff, ["survey_year", "survey_month"]].drop_duplicates()
        raise RuntimeError(f"{TASK_ID}: observed {variable} values lack monthly cutoffs: {missing_periods.to_dict('records')}")
    strict_inside = pool[task_column].gt(pool[f"{task_column}_p01"]) & pool[task_column].lt(
        pool[f"{task_column}_p99"]
    )
    if variable == "current_mortgage_rate":
        pass_mask = strict_inside
        pass_condition = (
            f'pool["{task_column}"].gt(pool["{task_column}_p01"]) '
            f'& pool["{task_column}"].lt(pool["{task_column}_p99"])'
        )
    elif variable in {"local_home_value", "purchase_price", "current_home_value"}:
        pass_mask = ~home_value_fields_rendered | strict_inside
        pass_condition = (
            '~home_value_fields_rendered | '
            f'(pool["{task_column}"].gt(pool["{task_column}_p01"]) '
            f'& pool["{task_column}"].lt(pool["{task_column}_p99"]))'
        )
    else:
        pass_mask = pool[task_column].isna() | strict_inside
        pass_condition = (
            f'pool["{task_column}"].isna() | '
            f'(pool["{task_column}"].gt(pool["{task_column}_p01"]) '
            f'& pool["{task_column}"].lt(pool["{task_column}_p99"]))'
        )
    trim_conditions.append(pass_mask)
    drop_detail_rows.append(
        filter_count(
            TASK_ID, 8, detail_order,
            f"trim_{variable}_predictor_strict_inside_month_p01_p99", pass_condition, int((~pass_mask).sum()),
        )
    )
    detail_order += 1
trim_mask = pd.Series(True, index=pool.index)
for pass_mask in trim_conditions:
    trim_mask &= pass_mask
pool = pool.loc[trim_mask].copy()
step_rows.append(
    construction_step(
        TASK_ID, 8, "Apply outlier trimming",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)
if pool.empty:
    raise RuntimeError(f"{TASK_ID} has zero eligible rows after outlier trimming.")

for columns, valid_codes in [
    (["q38", "hq38", "HQ38", "qh0", "HQH0", "qh4a", "HQH4a"], set(HOUSING_YES_NO)),
    (["qh5", "HQH5"], set(LOAN_STATUS)),
    (["qh5m", "HQH5m"], set(MORTGAGE_TYPE)),
]:
    for column in columns:
        if column in pool.columns:
            values = pd.to_numeric(pool[column], errors="coerce").dropna()
            observed = set(values.round().astype(int).unique())
            unexpected = sorted(observed - valid_codes)
    if unexpected:
        raise RuntimeError(f"{TASK_ID}: unexpected codes in {column}: {unexpected}")

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
for output_column, label_column in CORE_CONTEXT_LABEL_OUTPUT_COLUMNS.items():
    sample[output_column] = sample[label_column].astype("string")
for output_column, label_column in TASK_LABEL_OUTPUT_COLUMNS.items():
    if output_column in sample.columns:
        if label_column not in sample.columns:
            raise RuntimeError(f"{TASK_ID} output needs missing label column {label_column} for {output_column}.")
        sample[output_column] = sample[label_column].astype("string")
for output_column, candidate_columns in CANONICAL_OUTPUT_COLUMNS.items():
    sample[output_column] = coalesce_existing_columns(sample, candidate_columns)
sample["current_mortgage_rate"] = pd.to_numeric(sample["current_mortgage_rate"], errors="coerce")
sample["mortgage30us_rate_lag1"] = pd.to_numeric(sample["mortgage30us_rate_lag1"], errors="coerce")
sample["mortgage_rate_gap"] = sample["mortgage30us_rate_lag1"] - sample["current_mortgage_rate"]

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
        + ["lockin_move_probability", "ordinary_move_probability"]
        + list(CANONICAL_OUTPUT_COLUMNS)
        + ["mortgage_rate_gap", "q6a_lock"]
    )
)

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
    *MORTGAGE_CONTEXT_COLUMNS,
    'lockin_move_probability',
    'ordinary_move_probability',
] + ['loan_status', 'current_mortgage_rate', 'mortgage_rate_gap']
missing_selected = sample[required_selected_columns].isna().sum()
missing_selected = missing_selected[missing_selected.gt(0)]
if not missing_selected.empty:
    raise RuntimeError(f"{TASK_ID} selected sample has missing required values: {missing_selected.to_dict()}")
probability_columns = ["lockin_move_probability", "ordinary_move_probability"]
sample[probability_columns] = sample[probability_columns].apply(pd.to_numeric, errors="coerce") / 100
if not sample[probability_columns].apply(lambda column: column.between(0, 1)).all().all():
    raise RuntimeError(f"{TASK_ID}: selected probability values must lie between 0 and 1 after normalization.")
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
print(f"{TASK_ID}: sample_path={sample_path_counts}")
