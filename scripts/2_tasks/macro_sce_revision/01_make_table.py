#!/usr/bin/env python
"""Build the HouseholdBench table for the SCE inflation-expectations task."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.sample_accounting import construction_step, filter_count

from scripts.utils.table_schema import canonicalize_public_table

from scripts.utils.sampling import sample_task_records
from scripts.utils.macro_context import CORE_MACRO_COLUMNS, attach_macro_context


OUTPUT_SLUG = "macro_sce_revision"
INPUT_PATH = PROJECT_ROOT / "data" / "intermediate" / "sce_core.parquet"
CPI_PATH = PROJECT_ROOT / "data" / "intermediate" / "cpi_major_groups.parquet"
TABULAR_PATH = PROJECT_ROOT / "data" / "householdbench" / "tabular" / f"{OUTPUT_SLUG}.csv"
DIAGNOSTICS_DIR = PROJECT_ROOT / "output" / "householdbench" / "tasks" / OUTPUT_SLUG
SAMPLING_DIR = PROJECT_ROOT / "data" / "intermediate" / "householdbench" / OUTPUT_SLUG
STEP_DIAGNOSTICS_PATH = DIAGNOSTICS_DIR / "01_sample_construction_steps.csv"
DROP_DETAIL_PATH = DIAGNOSTICS_DIR / "01_sample_construction_drop_details.csv"
OUTLIER_CUTOFFS_PATH = DIAGNOSTICS_DIR / "01_outlier_cutoffs.csv"

STEP_COLUMNS = ["task_id", "step", "step_label", "rows_before", "rows_after", "rows_dropped"]
DROP_DETAIL_COLUMNS = ["task_id", "step", "detail_order", "filter_id", "pass_condition", "fail_count"]
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

SCE_COLUMNS = [
    "source",
    "module",
    "release_id",
    "release_date",
    "release_date_source",
    "respondent_id",
    "userid",
    "survey_year",
    "survey_month",
    "survey_date",
    "panel_month",
    "weight_core",
    "age_years",
    "_NUM_CAT",
    "_REGION_CAT",
    "_EDU_CAT",
    "_HH_INC_DETAILED",
    "Q1",
    "Q2",
    "Q10_1",
    "Q10_2",
    "Q10_3",
    "Q10_4",
    "Q10_5",
    "Q11",
    "Q12new",
    "Q15",
    "Q16",
    "Q19",
    "Q9_mean",
    "Q9__task_valid",
    "Q9c_mean",
    "Q9c__task_valid",
]
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
}
SCE_READ_COLUMNS = list(dict.fromkeys(SCE_COLUMNS + list(CORE_LABEL_OUTPUT_COLUMNS.values())))
MACRO_COLUMNS = CORE_MACRO_COLUMNS

TARGET_COLUMNS = ["expected_inflation_next_12m_m12", "expected_inflation_2y3y_m12"]
PREDICTOR_COLUMNS = [
    "age_years_m12",
    "_NUM_CAT_m12",
    "_REGION_CAT_m12",
    "_EDU_CAT_m12",
    "_HH_INC_DETAILED_m12",
    "_NUM_CAT__label_m12",
    "_REGION_CAT__label_m12",
    "_EDU_CAT__label_m12",
    "_HH_INC_DETAILED__label_m12",
    "expected_inflation_next_12m_m1",
    "expected_inflation_2y3y_m1",
    *[f"{column}_m12" for column in MACRO_COLUMNS],
    "cpi_index_start",
    "cpi_index_end",
    "implied_expected_inflation_11m_m1",
    "realized_inflation_11m",
    "inflation_surprise_11m",
]
OUTPUT_COLUMNS = [
    "subject_id",
    "selection_period",
    "respondent_id",
    "survey_date_m1",
    "survey_date_m12",
    "release_date",
    "release_date_source",
    "release_date_m1",
    "release_date_source_m1",
    "release_date_m12",
    "release_date_source_m12",
    "panel_month_m1",
    "panel_month_m12",
    "elapsed_calendar_months",
    "elapsed_days",
    "age_years_m12",
    "_NUM_CAT_m12",
    "_REGION_CAT_m12",
    "_EDU_CAT_m12",
    "_HH_INC_DETAILED_m12",
    "Q1_m12",
    "Q2_m12",
    "Q10_1_m12",
    "Q10_2_m12",
    "Q10_3_m12",
    "Q10_4_m12",
    "Q10_5_m12",
    "Q11_m12",
    "Q12new_m12",
    "Q15_m12",
    "Q16_m12",
    "Q19_m12",
    "expected_inflation_next_12m_m1",
    "expected_inflation_2y3y_m1",
    "expected_inflation_next_12m_m12",
    "expected_inflation_2y3y_m12",
    "revision_expected_inflation_next_12m",
    "revision_expected_inflation_2y3y",
    "cpi_start_month",
    "cpi_end_month",
    "cpi_index_start",
    "cpi_index_end",
    "implied_expected_inflation_11m_m1",
    "realized_inflation_11m",
    "inflation_surprise_11m",
    *[f"{column}_m12" for column in MACRO_COLUMNS],
]
OPTIONAL_OUTPUT_COLUMNS = [
    "Q1_m12",
    "Q2_m12",
    "Q10_1_m12",
    "Q10_2_m12",
    "Q10_3_m12",
    "Q10_4_m12",
    "Q10_5_m12",
    "Q11_m12",
    "Q12new_m12",
    "Q15_m12",
    "Q16_m12",
    "Q19_m12",
]


step_rows: list[dict[str, object]] = []
drop_detail_rows: list[dict[str, object]] = []


def record_step(
    records: list[dict[str, object]], step: int, label: str,
    rows_before: int, rows_after: int,
) -> None:
    records.append(
        construction_step(
            OUTPUT_SLUG, step, label,
            int(rows_before), int(rows_after), int(rows_before) - int(rows_after),
        )
    )


def record_filter(
    records: list[dict[str, object]], step: int, order: int,
    filter_id: str, condition: str, pass_mask: pd.Series,
) -> None:
    records.append(
        filter_count(
            OUTPUT_SLUG, step, order,
            filter_id, condition, int((~pass_mask).sum()),
        )
    )


# 1. Start from the preprocessed SCE Core source.
if not INPUT_PATH.is_file():
    raise SystemExit(f"SCE Core parquet not found: {INPUT_PATH}")
sce = pd.read_parquet(INPUT_PATH, columns=SCE_READ_COLUMNS)
if not sce["module"].eq("core").all():
    raise SystemExit("Input contains non-Core SCE rows.")
sce["survey_date"] = pd.to_datetime(sce["survey_date"], errors="coerce")
sce["release_date"] = pd.to_datetime(sce["release_date"], errors="coerce")
if sce[["survey_date", "release_date"]].isna().any().any():
    raise SystemExit("SCE Core input contains missing or invalid survey/release dates.")
source_keys = ["respondent_id", "survey_year", "survey_month"]
if sce.loc[sce[source_keys].notna().all(axis=1)].duplicated(source_keys).any():
    raise SystemExit(f"{OUTPUT_SLUG}: source is not unique by respondent and survey month.")
sce = sce.sort_values(["survey_date", "respondent_id"], kind="mergesort").reset_index(drop=True)
record_step(step_rows, 1, "Start from source observations", len(sce), len(sce))


# 2. Attach the common lagged quarterly macro context to every SCE row.
rows_before = len(sce)
sce["survey_q_index"] = sce["survey_date"].dt.year * 4 + sce["survey_date"].dt.quarter
sce = attach_macro_context(
    sce,
    origin_q_index_col="survey_q_index",
    required_columns=MACRO_COLUMNS,
)
if len(sce) != rows_before:
    raise SystemExit(f"{OUTPUT_SLUG}: macro attachment changed the row count.")
record_step(step_rows, 2, "Attach additional sources", rows_before, len(sce))


# 3. Left-attach panel-month-1 fields to every Core observation.
base = sce.copy()
base["survey_month_period"] = base["survey_date"].dt.to_period("M")
earlier = base.loc[base["panel_month"].eq(1)].copy()
if earlier["respondent_id"].duplicated().any():
    raise SystemExit(f"{OUTPUT_SLUG}: duplicate panel-month-1 respondent rows.")

pairs = base.add_suffix("_m12").merge(
    earlier.add_suffix("_m1"),
    left_on="respondent_id_m12",
    right_on="respondent_id_m1",
    how="left",
    validate="many_to_one",
)
if len(pairs) != len(base):
    raise SystemExit(f"{OUTPUT_SLUG}: panel-month-1 attachment changed the source row count.")
linked_identifier_mask = pairs["respondent_id_m1"].notna()
if not pairs.loc[linked_identifier_mask, "respondent_id_m1"].eq(
    pairs.loc[linked_identifier_mask, "respondent_id_m12"]
).all():
    raise SystemExit(f"{OUTPUT_SLUG}: respondent identifiers differ across pair endpoints.")
pairs["elapsed_calendar_months"] = (
    pairs["survey_date_m12"].dt.year * 12
    + pairs["survey_date_m12"].dt.month
    - pairs["survey_date_m1"].dt.year * 12
    - pairs["survey_date_m1"].dt.month
)
pairs["elapsed_days"] = (pairs["survey_date_m12"] - pairs["survey_date_m1"]).dt.days
record_step(step_rows, 3, "Construct leads and lags", len(base), len(pairs))


# 4. Restrict to the panel-month-12, exact-gap panel endpoint and construct CPI information.
rows_before = len(pairs)
later_panel_mask = pairs["panel_month_m12"].eq(12)
record_filter(drop_detail_rows,
    4,
    1,
    "current_observation_panel_month_12",
    'pairs["panel_month_m12"].eq(12)',
    later_panel_mask,
)
linked_m1_mask = pairs["respondent_id_m1"].notna()
linked_m1_given_later_mask = ~later_panel_mask | linked_m1_mask
record_filter(drop_detail_rows,
    4,
    2,
    "linked_panel_month_1_observation_available",
    'pairs["respondent_id_m1"].notna()',
    linked_m1_given_later_mask,
)
gap_mask = pairs["elapsed_calendar_months"].eq(11) & pairs["survey_date_m12"].gt(pairs["survey_date_m1"])
gap_given_endpoint_mask = ~(later_panel_mask & linked_m1_mask) | gap_mask
record_filter(drop_detail_rows,
    4,
    3,
    "panel_month_1_to_12_exact_11_calendar_months",
    'pairs["elapsed_calendar_months"].eq(11) & pairs["survey_date_m12"].gt(pairs["survey_date_m1"])',
    gap_given_endpoint_mask,
)
pairs = pairs.loc[later_panel_mask & linked_m1_mask & gap_mask].copy()

pairs["expected_inflation_next_12m_m1"] = pd.to_numeric(pairs["Q9_mean_m1"], errors="coerce")
pairs["expected_inflation_2y3y_m1"] = pd.to_numeric(pairs["Q9c_mean_m1"], errors="coerce")
pairs["expected_inflation_next_12m_m12"] = pd.to_numeric(pairs["Q9_mean_m12"], errors="coerce")
pairs["expected_inflation_2y3y_m12"] = pd.to_numeric(pairs["Q9c_mean_m12"], errors="coerce")
pairs["revision_expected_inflation_next_12m"] = (
    pairs["expected_inflation_next_12m_m12"] - pairs["expected_inflation_next_12m_m1"]
)
pairs["revision_expected_inflation_2y3y"] = (
    pairs["expected_inflation_2y3y_m12"] - pairs["expected_inflation_2y3y_m1"]
)

if not CPI_PATH.is_file():
    raise SystemExit(f"CPI file not found: {CPI_PATH}")
cpi = pd.read_parquet(CPI_PATH, columns=["date", "cpi_all_items"])
cpi["date"] = pd.to_datetime(cpi["date"], errors="coerce")
cpi["cpi_all_items"] = pd.to_numeric(cpi["cpi_all_items"], errors="coerce")
cpi = cpi.dropna(subset=["date", "cpi_all_items"]).copy()
cpi["cpi_month"] = cpi["date"].dt.to_period("M")
if cpi["cpi_month"].duplicated().any():
    raise SystemExit(f"{OUTPUT_SLUG}: CPI is not unique by month.")

pairs["cpi_start_month"] = pairs["survey_month_period_m1"] - 1
pairs["cpi_end_month"] = pairs["survey_month_period_m12"] - 1
cpi_start = cpi[["cpi_month", "cpi_all_items"]].rename(
    columns={"cpi_month": "cpi_start_month", "cpi_all_items": "cpi_index_start"}
)
cpi_end = cpi[["cpi_month", "cpi_all_items"]].rename(
    columns={"cpi_month": "cpi_end_month", "cpi_all_items": "cpi_index_end"}
)
pairs = pairs.merge(cpi_start, on="cpi_start_month", how="left", validate="many_to_one")
pairs = pairs.merge(cpi_end, on="cpi_end_month", how="left", validate="many_to_one")
pairs["realized_inflation_11m"] = 100.0 * (
    pairs["cpi_index_end"] / pairs["cpi_index_start"] - 1.0
)
valid_compounding = pairs["expected_inflation_next_12m_m1"].gt(-100.0)
pairs["implied_expected_inflation_11m_m1"] = np.nan
pairs.loc[valid_compounding, "implied_expected_inflation_11m_m1"] = 100.0 * (
    (1.0 + pairs.loc[valid_compounding, "expected_inflation_next_12m_m1"] / 100.0) ** (11.0 / 12.0)
    - 1.0
)
pairs["inflation_surprise_11m"] = (
    pairs["realized_inflation_11m"] - pairs["implied_expected_inflation_11m_m1"]
)

if not (pairs["cpi_start_month"] == pairs["survey_month_period_m1"] - 1).all():
    raise SystemExit(f"{OUTPUT_SLUG}: incorrect CPI start month.")
if not (pairs["cpi_end_month"] == pairs["survey_month_period_m12"] - 1).all():
    raise SystemExit(f"{OUTPUT_SLUG}: incorrect CPI end month.")
record_step(step_rows, 4, "Restrict universe", rows_before, len(pairs))


# Source-wide later-month target-level cutoffs for the exact-gap pair population.
cutoff_summaries: dict[str, pd.DataFrame] = {}
cutoff_audit_frames: list[pd.DataFrame] = []
for variable in TARGET_COLUMNS:
    variable_reference = pairs.dropna(subset=[variable]).copy()
    variable_summary = (
        variable_reference.groupby(["survey_year_m12", "survey_month_m12"])[variable]
        .agg(n_reference="size", p01=lambda values: values.quantile(0.01), p99=lambda values: values.quantile(0.99))
        .reset_index()
    )
    if variable_summary.empty or variable_summary[["p01", "p99"]].isna().any().any():
        raise SystemExit(f"{OUTPUT_SLUG}: missing later-month cutoffs for {variable}.")
    if variable_summary["p01"].eq(variable_summary["p99"]).any():
        bad = variable_summary.loc[
            variable_summary["p01"].eq(variable_summary["p99"]),
            ["survey_year_m12", "survey_month_m12", "n_reference", "p01"],
        ]
        raise SystemExit(f"{OUTPUT_SLUG}: degenerate later-month {variable} cutoffs: {bad.to_dict('records')}")
    variable_summary["period"] = (
        variable_summary["survey_year_m12"].astype(int).astype(str)
        + "-"
        + variable_summary["survey_month_m12"].astype(int).astype(str).str.zfill(2)
    )
    reference_with_cutoffs = variable_reference.merge(
        variable_summary,
        on=["survey_year_m12", "survey_month_m12"],
        how="left",
        validate="many_to_one",
    )
    audit_counts = reference_with_cutoffs.assign(
        n_at_or_below_p01=reference_with_cutoffs[variable].le(reference_with_cutoffs["p01"]),
        n_at_or_above_p99=reference_with_cutoffs[variable].ge(reference_with_cutoffs["p99"]),
    )
    audit_counts = (
        audit_counts.groupby(["survey_year_m12", "survey_month_m12"], as_index=False)[
            ["n_at_or_below_p01", "n_at_or_above_p99"]
        ]
        .sum()
    )
    audit = variable_summary.merge(
        audit_counts,
        on=["survey_year_m12", "survey_month_m12"],
        how="left",
        validate="one_to_one",
    )
    audit["task_id"] = OUTPUT_SLUG
    audit["variable"] = variable
    audit["role"] = "target"
    cutoff_audit_frames.append(audit)
    cutoff_summaries[variable] = variable_summary
    pairs = pairs.merge(
        variable_summary[["survey_year_m12", "survey_month_m12", "p01", "p99"]].rename(
            columns={"p01": f"{variable}_p01", "p99": f"{variable}_p99"}
        ),
        on=["survey_year_m12", "survey_month_m12"],
        how="left",
        validate="many_to_one",
    )
cutoff_audit = pd.concat(cutoff_audit_frames, ignore_index=True)


# 5. Require observed targets.
rows_before = len(pairs)
target_mask = pd.Series(True, index=pairs.index)
for order, column in enumerate(TARGET_COLUMNS, start=1):
    pass_mask = pairs[column].notna() & np.isfinite(pairs[column])
    record_filter(drop_detail_rows, 5, order, f"target_{column}_finite", f'np.isfinite(pairs["{column}"])', pass_mask)
    target_mask &= pass_mask
pairs = pairs.loc[target_mask].copy()
record_step(step_rows, 5, "Require observed targets", rows_before, len(pairs))


# 6. Require observed predictors.
rows_before = len(pairs)
predictor_mask = pd.Series(True, index=pairs.index)
for order, column in enumerate(PREDICTOR_COLUMNS, start=1):
    if pd.api.types.is_numeric_dtype(pairs[column]):
        pass_mask = pairs[column].notna() & np.isfinite(pd.to_numeric(pairs[column], errors="coerce"))
        condition = f'np.isfinite(pairs["{column}"])'
    else:
        pass_mask = pairs[column].notna() & pairs[column].astype(str).str.strip().ne("")
        condition = f'pairs["{column}"].notna() & pairs["{column}"].astype(str).str.strip().ne("")'
    record_filter(drop_detail_rows, 6, order, f"predictor_{column}_observed", condition, pass_mask)
    predictor_mask &= pass_mask
compounding_mask = pairs["expected_inflation_next_12m_m1"].gt(-100.0)
record_filter(drop_detail_rows,
    6,
    len(PREDICTOR_COLUMNS) + 1,
    "earlier_expectation_above_negative_100",
    'pairs["expected_inflation_next_12m_m1"].gt(-100.0)',
    compounding_mask,
)
predictor_mask &= compounding_mask
pairs = pairs.loc[predictor_mask].copy()
record_step(step_rows, 6, "Require observed predictors", rows_before, len(pairs))


# 7. Apply response-validity and age filters.
rows_before = len(pairs)
validity_mask = pd.Series(True, index=pairs.index)
validity_columns = ["Q9__task_valid_m1", "Q9c__task_valid_m1", "Q9__task_valid_m12", "Q9c__task_valid_m12"]
for order, column in enumerate(validity_columns, start=1):
    pass_mask = pairs[column].eq(True)
    record_filter(drop_detail_rows, 7, order, f"{column}_true", f'pairs["{column}"].eq(True)', pass_mask)
    validity_mask &= pass_mask
age_mask = pairs["age_years_m12"].between(18, 100, inclusive="both")
record_filter(drop_detail_rows,
    7,
    len(validity_columns) + 1,
    "age_years_m12_between_18_and_100",
    'pairs["age_years_m12"].between(18, 100, inclusive="both")',
    age_mask,
)
validity_mask &= age_mask
pool = pairs.loc[validity_mask].copy()
record_step(step_rows, 7, "Apply logical and validity filters", rows_before, len(pool))
if pool.empty:
    raise SystemExit(f"{OUTPUT_SLUG} has no eligible rows.")


# 8. Trim later expectation levels strictly inside their later-month p1/p99 intervals.
rows_before = len(pool)
trim_mask = pd.Series(True, index=pool.index)
for order, variable in enumerate(TARGET_COLUMNS, start=1):
    cutoff_columns = [f"{variable}_p01", f"{variable}_p99"]
    if pool[cutoff_columns].isna().any().any():
        raise SystemExit(f"{OUTPUT_SLUG}: eligible rows lack later-month cutoffs for {variable}.")
    pass_mask = pool[variable].gt(pool[f"{variable}_p01"]) & pool[variable].lt(pool[f"{variable}_p99"])
    record_filter(drop_detail_rows,
        8,
        order,
        f"trim_{variable}_strict_inside_later_month_p01_p99",
        f'pool["{variable}"].gt(pool["{variable}_p01"]) & pool["{variable}"].lt(pool["{variable}_p99"])',
        pass_mask,
    )
    trim_mask &= pass_mask
pool = pool.loc[trim_mask].copy()
record_step(step_rows, 8, "Apply outlier trimming", rows_before, len(pool))
if pool.empty:
    raise SystemExit(f"{OUTPUT_SLUG} has no eligible rows after outlier trimming.")


# Reconcile the construction before selection and export.
expected_realized = 100.0 * (pool["cpi_index_end"] / pool["cpi_index_start"] - 1.0)
if not np.allclose(pool["realized_inflation_11m"], expected_realized, rtol=0, atol=1e-12):
    raise SystemExit(f"{OUTPUT_SLUG}: realized inflation does not reconcile.")
expected_implied = 100.0 * (
    (1.0 + pool["expected_inflation_next_12m_m1"] / 100.0) ** (11.0 / 12.0) - 1.0
)
if not np.allclose(pool["implied_expected_inflation_11m_m1"], expected_implied, rtol=0, atol=1e-12):
    raise SystemExit(f"{OUTPUT_SLUG}: implied 11-month expectation does not reconcile.")
if not np.allclose(
    pool["inflation_surprise_11m"],
    pool["realized_inflation_11m"] - pool["implied_expected_inflation_11m_m1"],
    rtol=0,
    atol=1e-12,
):
    raise SystemExit(f"{OUTPUT_SLUG}: inflation surprise does not reconcile.")


# 9. Select and export the sample.
pool["selection_period"] = pd.to_datetime(pool["survey_date_m12"]).dt.strftime("%Y-%m")
pool["row_id"] = (
    pool["respondent_id_m1"].astype(str)
    + "__m01_" + pool["survey_date_m1"].dt.strftime("%Y%m")
    + "__m12_" + pool["survey_date_m12"].dt.strftime("%Y%m")
)
rows_before_export = len(pool)
# Public keys are fixed before sampling; source keys retain their sort types.
pool["id"] = pool["respondent_id_m1"].astype("string")
pool["time"] = pd.to_datetime(pool["survey_date_m12"], errors="raise").dt.strftime("%Y-%m-%d")
pool["release_date"] = pool["release_date_m12"]
pool["release_date"] = pd.to_datetime(pool["release_date"], errors="raise").dt.strftime("%Y-%m-%d")
sample, sampling_registry, sampling_summary = sample_task_records(
    pool,
    task_id=OUTPUT_SLUG,
    output_dir=SAMPLING_DIR,
    output_sort_cols=["survey_date_m12", "respondent_id_m1", "survey_date_m1"],
)

csv_sample = sample.copy()
csv_sample["respondent_id"] = csv_sample["respondent_id_m1"]
csv_sample["subject_id"] = csv_sample["respondent_id"].astype(str)
if csv_sample.duplicated(["subject_id", "survey_date_m12"]).any():
    raise SystemExit(f"{OUTPUT_SLUG}: duplicate respondent/date keys.")
csv_sample["release_date"] = csv_sample["release_date_m12"]
csv_sample["release_date_source"] = csv_sample["release_date_source_m12"]

for output_column, label_column in CORE_LABEL_OUTPUT_COLUMNS.items():
    csv_sample[f"{output_column}_m12"] = csv_sample[f"{label_column}_m12"].astype("string")

for date_column in ["survey_date_m1", "survey_date_m12", "release_date", "release_date_m1", "release_date_m12"]:
    csv_sample[date_column] = pd.to_datetime(csv_sample[date_column], errors="coerce")
    if csv_sample[date_column].isna().any():
        raise SystemExit(f"Selected rows contain missing or invalid {date_column} values.")
    csv_sample[date_column] = csv_sample[date_column].dt.date.astype(str)
for period_column in ["cpi_start_month", "cpi_end_month"]:
    csv_sample[period_column] = csv_sample[period_column].astype(str)

required_output_columns = [column for column in OUTPUT_COLUMNS if column not in OPTIONAL_OUTPUT_COLUMNS]
required_missing = csv_sample[required_output_columns].isna().sum()
if required_missing.any():
    raise SystemExit(f"Selected rows contain missing required values: {required_missing[required_missing > 0].to_dict()}")

TABULAR_PATH.parent.mkdir(parents=True, exist_ok=True)
canonicalize_public_table(csv_sample, task_id=OUTPUT_SLUG).to_csv(TABULAR_PATH, index=False)
record_step(step_rows, 9, "Export sample", rows_before_export, len(csv_sample))

DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(step_rows, columns=STEP_COLUMNS).to_csv(STEP_DIAGNOSTICS_PATH, index=False)
pd.DataFrame(drop_detail_rows, columns=DROP_DETAIL_COLUMNS).to_csv(DROP_DETAIL_PATH, index=False)
cutoff_audit.loc[:, OUTLIER_CUTOFF_COLUMNS].to_csv(OUTLIER_CUTOFFS_PATH, index=False)
print(f"Wrote {TABULAR_PATH.relative_to(PROJECT_ROOT)} with {len(csv_sample):,} rows.")
