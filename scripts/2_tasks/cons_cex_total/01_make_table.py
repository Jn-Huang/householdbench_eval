#!/usr/bin/env python
"""Build the HouseholdBench CEX total-consumption task table."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.sample_accounting import construction_step, filter_count

from scripts.utils.table_schema import canonicalize_public_table

from scripts.utils.sampling import sample_task_records
from scripts.utils.macro_context import CORE_MACRO_COLUMNS, attach_macro_context


TASK_ID = "cons_cex_total"
INPUT_PATH = PROJECT_ROOT / "data/intermediate/cex_interview.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data/householdbench/tabular/cons_cex_total.csv"
DIAGNOSTICS_DIR = PROJECT_ROOT / "output" / "householdbench" / "tasks" / TASK_ID
SAMPLING_DIR = PROJECT_ROOT / "data" / "intermediate" / "householdbench" / TASK_ID
STEP_DIAGNOSTICS_PATH = DIAGNOSTICS_DIR / "01_sample_construction_steps.csv"
DROP_DETAIL_PATH = DIAGNOSTICS_DIR / "01_sample_construction_drop_details.csv"
OUTLIER_CUTOFF_PATH = DIAGNOSTICS_DIR / "01_outlier_cutoffs.csv"
STEP_COLUMNS = [
    "task_id",
    "step",
    "step_label",
    "rows_before",
    "rows_after",
    "rows_dropped",
]
DETAIL_COLUMNS = [
    "task_id",
    "step",
    "detail_order",
    "filter_id",
    "pass_condition",
    "fail_count",
]

DEMOGRAPHIC_COLUMNS = [
    "age_ref",
    "sex_ref",
    "race_ref_harmonized",
    "educ_ref_harmonized",
    "marital1",
    "n_kids",
    "fam_size",
    "region",
    "bls_urbn",
]
DEMOGRAPHIC_INPUT_COLUMNS = [
    "age_ref",
    "sex_ref_label",
    "race_ref_harmonized",
    "educ_ref_harmonized",
    "marital1_label",
    "n_kids",
    "fam_size",
    "region_label",
    "bls_urbn_label",
]
CATEGORICAL_RENAMES = {
    "sex_ref_label": "sex_ref",
    "marital1_label": "marital1",
    "region_label": "region",
    "bls_urbn_label": "bls_urbn",
}
INPUT_COLUMNS = [
    "newid",
    "cu_id",
    "cex_panel_id",
    "interview_year",
    "interview_month",
    "quarter",
    "interview_position_harmonized",
    *DEMOGRAPHIC_INPUT_COLUMNS,
    "income_before_tax",
    "cons_parker_total",
    "cons_nondurables",
    "cons_durables",
    "cons_food",
    "release_date",
    "release_date_source",
]
LAGGED_CEX_COLUMNS = [
    "income_before_tax",
    "cons_parker_total",
    "cons_nondurables",
    "cons_durables",
    "cons_food",
    "income_before_tax_p01",
    "income_before_tax_p99",
    "cons_parker_total_p01",
    "cons_parker_total_p99",
    "interview_year",
    "interview_month",
    "quarter",
]
BASE_REQUIRED_COLUMNS = [
    *DEMOGRAPHIC_COLUMNS,
    "newid",
    "cu_id",
    "cex_panel_id",
    "interview_year",
    "interview_month",
    "quarter",
    "income_before_tax",
    "cons_parker_total",
    "cons_nondurables",
    "cons_durables",
]
CURRENT_PREDICTOR_COLUMNS = [
    *DEMOGRAPHIC_COLUMNS,
    "newid",
    "cu_id",
    "cex_panel_id",
    "interview_year",
    "interview_month",
    "quarter",
    "income_before_tax",
    "release_date",
    "release_date_source",
]
HISTORY_REQUIRED_PATTERNS = [
    "cons_parker_total_lag{lag}",
    "cons_nondurables_lag{lag}",
    "cons_durables_lag{lag}",
]
OUTPUT_COLUMNS = [
    "subject_id",
    "sample_id",
    "newid",
    "cu_id",
    "cex_panel_id",
    "selection_period",
    "prompt_time",
    "release_date",
    "release_date_source",
    "interview_year",
    "interview_month",
    "quarter",
    "q_index",
    "interview_month_index",
    "interview_month_index_lag1",
    "interview_month_index_lag2",
    "interview_month_index_lag3",
    *DEMOGRAPHIC_COLUMNS,
    "n_adults",
    "income_before_tax",
    "cons_parker_total_lag1",
    "cons_nondurables_lag1",
    "cons_durables_lag1",
    "cons_parker_total_lag2",
    "cons_nondurables_lag2",
    "cons_durables_lag2",
    "cons_parker_total_lag3",
    "cons_nondurables_lag3",
    "cons_durables_lag3",
    *CORE_MACRO_COLUMNS,
    "cons_parker_total",
    "cons_nondurables",
    "cons_durables",
]


step_rows = []
drop_detail_rows = []

source_columns = pq.read_schema(INPUT_PATH).names
missing_input_columns = [column for column in INPUT_COLUMNS if column not in source_columns]
if missing_input_columns:
    raise SystemExit(f"{INPUT_PATH} is missing required columns: {missing_input_columns}")

# 1. Start from pre-processed main source.
cex = pd.read_parquet(INPUT_PATH, columns=INPUT_COLUMNS)
cex = cex.rename(columns=CATEGORICAL_RENAMES)
text_columns = {
    "newid",
    "cu_id",
    "cex_panel_id",
    "cex_panel_id",
    "release_date",
    "release_date_source",
    "sex_ref",
    "race_ref_harmonized",
    "educ_ref_harmonized",
    "marital1",
    "region",
    "bls_urbn",
}
for column in INPUT_COLUMNS:
    renamed = CATEGORICAL_RENAMES.get(column, column)
    if renamed not in text_columns:
        cex[renamed] = pd.to_numeric(cex[renamed], errors="coerce")
for column in ["newid", "cu_id", "cex_panel_id"]:
    cex[column] = cex[column].astype("string").str.strip()
    if cex[column].isna().any() or cex[column].eq("").any():
        raise SystemExit(f"Cleaned CEX data contain missing {column} values.")
cex["newid"] = cex["newid"].str.zfill(8)
cex["release_date"] = pd.to_datetime(cex["release_date"], errors="coerce")
if cex["release_date"].isna().any():
    raise SystemExit("Cleaned CEX data contain missing release_date values.")
if cex["release_date_source"].isna().any() or cex["release_date_source"].str.strip().eq("").any():
    raise SystemExit("Cleaned CEX data contain missing release_date_source values.")
step_rows.append(
    construction_step(
        TASK_ID, 1, "Start from source observations",
        int(len(cex)), int(len(cex)), 0,
    )
)

cex["q_index"] = cex["interview_year"].astype(int) * 4 + cex["quarter"].astype(int)
cex["interview_month_index"] = (
    cex["interview_year"].astype(int) * 12 + cex["interview_month"].astype(int)
)
cex["n_adults"] = (cex["fam_size"] - cex["n_kids"]).clip(lower=0)
if cex.duplicated(["cex_panel_id", "interview_month_index"]).any():
    raise SystemExit(
        "Cleaned CEX data contain multiple interviews for a CU in one interview month."
    )
if cex.duplicated(["cex_panel_id", "interview_year", "quarter"]).any():
    raise SystemExit("CEX cutoff reference is not unique by CU and interview quarter.")

cutoff_rows = []
cutoff_frames = []
for variable, role in [
    ("income_before_tax", "predictor"),
    ("cons_parker_total", "predictor_and_target"),
]:
    reference = cex.loc[
        cex["cex_panel_id"].notna()
        & cex["interview_year"].notna()
        & cex["quarter"].notna()
        & cex[variable].notna(),
        ["interview_year", "quarter", variable],
    ].copy()
    cutoffs = (
        reference.groupby(["interview_year", "quarter"])[variable]
        .agg(
            n_reference="count",
            p01=lambda values: values.quantile(0.01),
            p99=lambda values: values.quantile(0.99),
        )
        .reset_index()
    )
    bad = cutoffs["p01"].ge(cutoffs["p99"])
    if bad.any():
        raise SystemExit(f"{TASK_ID}: degenerate {variable} quarterly cutoffs: {cutoffs.loc[bad].to_dict('records')}")
    reference = reference.merge(cutoffs, on=["interview_year", "quarter"], how="left", validate="many_to_one")
    boundary_counts = (
        reference.assign(
            n_at_or_below_p01=reference[variable].le(reference["p01"]),
            n_at_or_above_p99=reference[variable].ge(reference["p99"]),
        )
        .groupby(["interview_year", "quarter"], as_index=False)[["n_at_or_below_p01", "n_at_or_above_p99"]]
        .sum()
    )
    audit = cutoffs.merge(boundary_counts, on=["interview_year", "quarter"], validate="one_to_one")
    audit["task_id"] = TASK_ID
    audit["variable"] = variable
    audit["role"] = role
    audit["period"] = audit["interview_year"].astype(int).astype(str) + "-Q" + audit["quarter"].astype(int).astype(str)
    cutoff_rows.append(audit)
    cutoff_frames.append(
        cutoffs[["interview_year", "quarter", "p01", "p99"]].rename(
            columns={"p01": f"{variable}_p01", "p99": f"{variable}_p99"}
        )
    )
for cutoffs in cutoff_frames:
    cex = cex.merge(cutoffs, on=["interview_year", "quarter"], how="left", validate="many_to_one")

# 2. Attach additional sources.
rows_before = len(cex)
cex = attach_macro_context(
    cex,
    origin_q_index_col="q_index",
    required_columns=CORE_MACRO_COLUMNS,
)
if len(cex) != rows_before:
    raise SystemExit(f"{TASK_ID}: current macro/CPI merge changed the row count.")
step_rows.append(
    construction_step(
        TASK_ID, 2, "Attach additional sources",
        int(rows_before), int(len(cex)), int(rows_before) - int(len(cex)),
    )
)


# 3. Construct lagged household histories.
rows_before = len(cex)
cex = cex.sort_values(
    ["cex_panel_id", "interview_month_index", "interview_position_harmonized"],
    kind="mergesort",
).reset_index(drop=True)
grouped_cex = cex.groupby("cex_panel_id", sort=False)
for lag in [1, 2, 3]:
    for column in LAGGED_CEX_COLUMNS:
        cex[f"{column}_lag{lag}"] = grouped_cex[column].shift(lag)
    cex[f"interview_month_index_lag{lag}"] = grouped_cex["interview_month_index"].shift(lag)
usable_lag_masks = {}
usable_prefix = pd.Series(True, index=cex.index)
for lag in [1, 2, 3]:
    exact_gap = (
        cex["interview_month_index"]
        - cex[f"interview_month_index_lag{lag}"]
    ).eq(3 * lag)
    required_history_columns = [
        pattern.format(lag=lag)
        for pattern in HISTORY_REQUIRED_PATTERNS
    ]
    complete_lag = cex[required_history_columns].notna().all(axis=1)
    usable_prefix &= exact_gap & complete_lag
    usable_lag_masks[lag] = usable_prefix.copy()

for lag in [1, 2, 3]:
    lag_columns = [
        f"{column}_lag{lag}"
        for column in LAGGED_CEX_COLUMNS
    ]
    lag_columns.append(f"interview_month_index_lag{lag}")
    cex.loc[~usable_lag_masks[lag], lag_columns] = pd.NA

step_rows.append(
    construction_step(
        TASK_ID, 3, "Construct leads and lags",
        int(rows_before), int(len(cex)), int(rows_before) - int(len(cex)),
    )
)


# 4. Restrict universe.
rows_before = len(cex)
step_rows.append(
    construction_step(
        TASK_ID, 4, "Restrict universe",
        int(rows_before), int(len(cex)), int(rows_before) - int(len(cex)),
    )
)


# 5. Require observed target(s).
rows_before = len(cex)
target_columns = ["cons_parker_total", "cons_nondurables", "cons_durables"]
condition_masks = []
detail_order = 1
for column in target_columns:
    pass_condition = f'cex["{column}"].notna()'
    pass_mask = cex[column].notna()
    condition_masks.append(pass_mask)
    drop_detail_rows.append(
        filter_count(
            TASK_ID, 5, detail_order,
            f"target_{column}_not_missing", pass_condition, int((~pass_mask).sum()),
        )
    )
    detail_order += 1
target_observed = pd.Series(True, index=cex.index)
for pass_mask in condition_masks:
    target_observed &= pass_mask
cex = cex.loc[target_observed].copy()
step_rows.append(
    construction_step(
        TASK_ID, 5, "Require observed targets",
        int(rows_before), int(len(cex)), int(rows_before) - int(len(cex)),
    )
)


# 6. Require observed predictors.
rows_before = len(cex)
condition_masks = []
detail_order = 1
for column in CURRENT_PREDICTOR_COLUMNS:
    pass_condition = f'cex["{column}"].notna()'
    pass_mask = cex[column].notna()
    condition_masks.append(pass_mask)
    drop_detail_rows.append(
        filter_count(
            TASK_ID, 6, detail_order,
            f"current_predictor_{column}_not_missing", pass_condition, int((~pass_mask).sum()),
        )
    )
    detail_order += 1
for pattern in HISTORY_REQUIRED_PATTERNS:
    column = pattern.format(lag=1)
    pass_condition = f'cex["{column}"].notna()'
    pass_mask = cex[column].notna()
    condition_masks.append(pass_mask)
    drop_detail_rows.append(
        filter_count(
            TASK_ID, 6, detail_order,
            f"lag1_required_{column}_not_missing", pass_condition, int((~pass_mask).sum()),
        )
    )
    detail_order += 1
predictors_observed = pd.Series(True, index=cex.index)
for pass_mask in condition_masks:
    predictors_observed &= pass_mask
pool = cex.loc[predictors_observed].copy()
if pool.empty:
    raise SystemExit(f"{TASK_ID} produced no eligible rows.")
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
for column, minimum in [("income_before_tax", 1000.0), ("cons_parker_total", 250.0)]:
    pass_mask = pool[column].ge(minimum)
    condition_masks.append(pass_mask)
    drop_detail_rows.append(filter_count(
                                TASK_ID, 7, detail_order,
                                f"current_{column}_minimum", f'pool["{column}"].ge({minimum})', int((~pass_mask).sum()),
                            ))
    detail_order += 1
for column in target_columns:
    pass_condition = f'pool["{column}"].ge(0)'
    pass_mask = pool[column].ge(0)
    condition_masks.append(pass_mask)
    drop_detail_rows.append(
        filter_count(
            TASK_ID, 7, detail_order,
            f"current_{column}_nonnegative", pass_condition, int((~pass_mask).sum()),
        )
    )
    detail_order += 1
for lag in [1, 2, 3]:
    for column, minimum in [("cons_parker_total", 250.0)]:
        lag_column = f"{column}_lag{lag}"
        pass_mask = pool[lag_column].isna() | pool[lag_column].ge(minimum)
        condition_masks.append(pass_mask)
        drop_detail_rows.append(filter_count(
                                    TASK_ID, 7, detail_order,
                                    f"lag{lag}_{column}_minimum", f'pool["{lag_column}"].isna() | pool["{lag_column}"].ge({minimum})', int((~pass_mask).sum()),
                                ))
        detail_order += 1
    for column in target_columns:
        lag_column = f"{column}_lag{lag}"
        pass_condition = f'pool["{lag_column}"].isna() | pool["{lag_column}"].ge(0)'
        pass_mask = pool[lag_column].isna() | pool[lag_column].ge(0)
        condition_masks.append(pass_mask)
        drop_detail_rows.append(
            filter_count(
                TASK_ID, 7, detail_order,
                f"lag{lag}_{column}_nonnegative", pass_condition, int((~pass_mask).sum()),
            )
        )
        detail_order += 1
quality_mask = pd.Series(True, index=pool.index)
for pass_mask in condition_masks:
    quality_mask &= pass_mask
quality_drop = int((~quality_mask).sum())
pool = pool.loc[quality_mask].copy()
if pool.empty:
    raise SystemExit(f"{TASK_ID} produced no eligible rows after logical and validity filters.")
step_rows.append(
    construction_step(
        TASK_ID, 7, "Apply logical and validity filters",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)
print(f"{TASK_ID}: logical and validity filters dropped {quality_drop:,} rows.")

# 8. Apply outlier trimming.
rows_before = len(pool)
trim_masks = []
detail_order = 1
for timing in ["current", "lag1", "lag2", "lag3"]:
    suffix = "" if timing == "current" else f"_{timing}"
    variables = ["cons_parker_total"]
    if timing == "current":
        variables.insert(0, "income_before_tax")
    for variable in variables:
        value = f"{variable}{suffix}"
        p01 = f"{variable}_p01{suffix}"
        p99 = f"{variable}_p99{suffix}"
        observed = pool[value].notna()
        if pool.loc[observed, [p01, p99]].isna().any().any():
            raise SystemExit(f"{TASK_ID}: missing {variable} cutoffs for observed {timing} records entering step 8.")
        pass_mask = (~observed) | (pool[value].gt(pool[p01]) & pool[value].lt(pool[p99]))
        trim_masks.append(pass_mask)
        drop_detail_rows.append(filter_count(
                                    TASK_ID, 8, detail_order,
                                    f"trim_{variable}_{timing}_strict_inside_period_p01_p99", f'pool["{value}"].isna() | (pool["{value}"].gt(pool["{p01}"]) & pool["{value}"].lt(pool["{p99}"]))', int((~pass_mask).sum()),
                                ))
        detail_order += 1
trim_mask = pd.Series(True, index=pool.index)
for pass_mask in trim_masks:
    trim_mask &= pass_mask
pool = pool.loc[trim_mask].copy()
if pool.empty:
    raise SystemExit(f"{TASK_ID} produced no eligible rows after outlier trimming.")
step_rows.append(construction_step(
                     TASK_ID, 8, "Apply outlier trimming",
                     int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
                 ))

pool["selection_period"] = [
    f"{int(year):04d}-{int(month):02d}"
    for year, month in zip(
        pool["interview_year"], pool["interview_month"], strict=True
    )
]

# Public keys are fixed before sampling; source keys retain their sort types.
pool["id"] = pool["cex_panel_id"].astype("string")
pool["time"] = (
    pool["interview_year"].astype("Int64").astype(str)
    + "-"
    + pool["interview_month"].astype("Int64").astype(str).str.zfill(2)
    + "-01"
)
pool["release_date"] = pd.to_datetime(pool["release_date"], errors="raise").dt.strftime("%Y-%m-%d")
sample, sampling_registry, sampling_summary = sample_task_records(
    pool,
    task_id=TASK_ID,
    output_dir=SAMPLING_DIR,
    output_sort_cols=["cex_panel_id", "interview_month_index", "newid"],
)
rows_before_export = len(pool)

table = sample.reset_index(drop=True).copy()
table["subject_id"] = table["cex_panel_id"].astype(str)
table["sample_id"] = table["newid"].astype(str)
table["prompt_time"] = [
    f"{int(year):04d}-{int(month):02d}-01"
    for year, month in zip(
        table["interview_year"], table["interview_month"], strict=True
    )
]
if table["release_date"].isna().any():
    raise SystemExit("Selected CEX rows contain missing release_date values.")
if table["release_date_source"].isna().any() or table["release_date_source"].str.strip().eq("").any():
    raise SystemExit("Selected CEX rows contain missing release_date_source values.")

expected_prompt_time = pd.Series(
    [
        f"{int(year):04d}-{int(month):02d}-01"
        for year, month in zip(
            table["interview_year"],
            table["interview_month"],
            strict=True,
        )
    ],
    index=table.index,
)
if not table["prompt_time"].eq(expected_prompt_time).all():
    raise SystemExit("Selected rows contain prompt times that do not match interview month.")

for lag in [1, 2, 3]:
    available = table[f"interview_month_index_lag{lag}"].notna()
    prompt_history_columns = [
        pattern.format(lag=lag) for pattern in HISTORY_REQUIRED_PATTERNS
    ]
    if table.loc[available, prompt_history_columns].isna().any().any():
        raise SystemExit(f"{TASK_ID}: available lag-{lag} history is incomplete.")
    if table.loc[~available, prompt_history_columns].notna().any().any():
        raise SystemExit(
            f"{TASK_ID}: unavailable lag-{lag} history contains populated values."
        )
    gap = (
        table.loc[available, "interview_month_index"]
        - table.loc[available, f"interview_month_index_lag{lag}"]
    )
    if not gap.eq(3 * lag).all():
        raise SystemExit(
            f"Selected rows contain invalid available lag-{lag} interview-month gaps."
        )
    if lag > 1:
        previous_available = table[
            f"interview_month_index_lag{lag - 1}"
        ].notna()
        if (available & ~previous_available).any():
            raise SystemExit(
                f"{TASK_ID}: lag {lag} is populated without lag {lag - 1}."
            )

missing_output_columns = [column for column in OUTPUT_COLUMNS if column not in table.columns]
if missing_output_columns:
    raise SystemExit(f"{TASK_ID} table is missing output columns: {missing_output_columns}")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
canonicalize_public_table(table, task_id=TASK_ID).to_csv(OUTPUT_PATH, index=False)
step_rows.append(
    construction_step(
        TASK_ID, 9, "Export sample",
        int(rows_before_export), int(len(table)), int(rows_before_export) - int(len(table)),
    )
)

DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(step_rows, columns=STEP_COLUMNS).to_csv(STEP_DIAGNOSTICS_PATH, index=False)
pd.DataFrame(drop_detail_rows, columns=DETAIL_COLUMNS).to_csv(DROP_DETAIL_PATH, index=False)
pd.concat(cutoff_rows, ignore_index=True).loc[:, ["task_id", "variable", "role", "period", "n_reference", "p01", "p99", "n_at_or_below_p01", "n_at_or_above_p99"]].to_csv(OUTLIER_CUTOFF_PATH, index=False)

print(
    f"{TASK_ID}: wrote {len(table):,} rows to {OUTPUT_PATH} "
    f"(eligible={len(pool):,})"
)
