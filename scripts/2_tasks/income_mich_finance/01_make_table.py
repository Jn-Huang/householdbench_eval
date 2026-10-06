#!/usr/bin/env python
"""Build the tabular CSV for the Michigan personal-finance task."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.sample_accounting import construction_step, filter_count

from scripts.utils.macro_context import CORE_MACRO_COLUMNS
from scripts.utils.table_schema import canonicalize_public_table

from scripts.utils.michigan import (
    attach_macro_history,
    iso_date_or_none,
    load_microdata,
    validate_code_domains,
)
from scripts.utils.sampling import sample_task_records


TASK_SLUG = "income_mich_finance"
INPUT_PATH = PROJECT_ROOT / "data/intermediate/mich.parquet"
TABULAR_PATH = PROJECT_ROOT / "data" / "householdbench" / "tabular" / f"{TASK_SLUG}.csv"
DIAGNOSTICS_DIR = PROJECT_ROOT / "output" / "householdbench" / "tasks" / TASK_SLUG
SAMPLING_DIR = PROJECT_ROOT / "data" / "intermediate" / "householdbench" / TASK_SLUG
STEP_DIAGNOSTICS_PATH = DIAGNOSTICS_DIR / "01_sample_construction_steps.csv"
DROP_DETAIL_PATH = DIAGNOSTICS_DIR / "01_sample_construction_drop_details.csv"
RELEASE_SOURCE_COUNTS_PATH = (
    PROJECT_ROOT
    / "output"
    / "householdbench"
    / "tasks"
    / TASK_SLUG
    / "01_release_date_source_counts.csv"
)
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

CONTEXT_CORE_COLS = [
    "CASEID",
    "survey_date",
    "release_date",
    "release_date_source",
    "YYYYMM",
    "AGE",
    "SEX",
    "REGION",
    "MARRY",
    "NUMADT",
    "NUMKID",
    "EDUC",
    "INCOME",
]
MACRO_HISTORY_COLS = CORE_MACRO_COLUMNS.copy()
REQUIRED_PREDICTORS = ["PAGO", "BAGO"]
OPTIONAL_PREDICTORS = []
TARGET_COLS = ["INEXQ1", "INEXQ2"]
LABELLED_PREDICTORS = ["SEX", "REGION", "MARRY", "EDUC", "PAGO", "BAGO"]
LABEL_COLUMNS = [f"{column}_label" for column in LABELLED_PREDICTORS]
REQUIRED_INPUT_COLS = list(
    dict.fromkeys(
        CONTEXT_CORE_COLS
        + [
            "WT",
            "SAMPLE",
            "methodology_regime",
            "panel_id",
            "panel_wave",
            "panel_link_expected",
            "panel_link_ok",
        ]
        + REQUIRED_PREDICTORS
        + OPTIONAL_PREDICTORS
        + TARGET_COLS
        + LABEL_COLUMNS
    )
)
CODE_DOMAINS = {
    "SEX": {1, 2},
    "REGION": {1, 2, 3, 4},
    "MARRY": {1, 3, 4, 5},
    "EDUC": {1, 2, 3, 4, 5, 6},
    "PAGO": {1, 3, 5},
    "BAGO": {1, 3, 5},
    "INEXQ1": {1, 2, 3, 5},
}
SAMPLE_COLUMNS = list(
    dict.fromkeys(
        [
            "methodology_regime",
            "panel_link_expected",
            "panel_link_ok",
            "SAMPLE",
            "release_date",
            "release_date_source",
        ]
        + CONTEXT_CORE_COLS
        + MACRO_HISTORY_COLS
        + REQUIRED_PREDICTORS
        + OPTIONAL_PREDICTORS
        + ["INEXQ1", "INEXQ2", "expected_percent_change"]
    )
)


step_rows = []
drop_detail_rows = []


# 1. Start from pre-processed main source.
mich = load_microdata(INPUT_PATH, REQUIRED_INPUT_COLS)
if mich.duplicated(["CASEID", "YYYYMM"]).any():
    raise RuntimeError("Michigan cutoff reference is not unique by CASEID and survey month.")

reference_direction = pd.to_numeric(mich["INEXQ1"], errors="coerce")
reference_magnitude = pd.to_numeric(mich["INEXQ2"], errors="coerce").abs()
mich["expected_percent_change_reference"] = pd.NA
mich.loc[reference_direction.isin([1, 2]), "expected_percent_change_reference"] = reference_magnitude
mich.loc[reference_direction.eq(3), "expected_percent_change_reference"] = 0
mich.loc[reference_direction.eq(5), "expected_percent_change_reference"] = -reference_magnitude
mich["expected_percent_change_reference"] = pd.to_numeric(
    mich["expected_percent_change_reference"], errors="coerce"
)

cutoff_rows = []
for source_variable, cutoff_variable, role in [
    ("expected_percent_change_reference", "expected_percent_change", "target"),
    ("INCOME", "INCOME", "predictor"),
]:
    reference = mich.loc[
        mich["CASEID"].notna() & mich["YYYYMM"].notna() & mich[source_variable].notna(),
        ["YYYYMM", source_variable],
    ].copy()
    cutoffs = (
        reference.groupby("YYYYMM")[source_variable]
        .agg(
            n_reference="count",
            p01=lambda values: values.quantile(0.01),
            p99=lambda values: values.quantile(0.99),
        )
        .reset_index()
    )
    bad = cutoffs["p01"].ge(cutoffs["p99"])
    if bad.any():
        raise RuntimeError(
            f"{TASK_SLUG}: degenerate {cutoff_variable} monthly cutoffs: "
            f"{cutoffs.loc[bad].to_dict('records')}"
        )
    reference = reference.merge(cutoffs, on="YYYYMM", how="left", validate="many_to_one")
    boundary_counts = (
        reference.assign(
            n_at_or_below_p01=reference[source_variable].le(reference["p01"]),
            n_at_or_above_p99=reference[source_variable].ge(reference["p99"]),
        )
        .groupby("YYYYMM", as_index=False)[["n_at_or_below_p01", "n_at_or_above_p99"]]
        .sum()
    )
    audit = cutoffs.merge(boundary_counts, on="YYYYMM", validate="one_to_one")
    audit["task_id"] = TASK_SLUG
    audit["variable"] = cutoff_variable
    audit["role"] = role
    audit["period"] = audit["YYYYMM"].astype(int).astype(str).str.slice(0, 4) + "-" + audit["YYYYMM"].astype(int).astype(str).str.slice(4, 6)
    cutoff_rows.append(audit)
    cutoff_columns = cutoffs[["YYYYMM", "p01", "p99"]].rename(
        columns={"p01": f"{cutoff_variable}_p01", "p99": f"{cutoff_variable}_p99"}
    )
    mich = mich.merge(cutoff_columns, on="YYYYMM", how="left", validate="many_to_one")
step_rows.append(
    construction_step(
        TASK_SLUG, 1, "Start from source observations",
        0, int(len(mich)), 0,
    )
)


# 2. Attach additional sources.
rows_before = len(mich)
step_rows.append(
    construction_step(
        TASK_SLUG, 2, "Attach additional sources",
        int(rows_before), int(len(mich)), int(rows_before) - int(len(mich)),
    )
)


# 3. Construct leads and lags.
rows_before = len(mich)
mich = attach_macro_history(mich)
if len(mich) != rows_before:
    raise RuntimeError(f"{TASK_SLUG}: macro-history attachment changed the row count.")

pool = mich.copy()
income_change_direction = pd.to_numeric(pool["INEXQ1"], errors="coerce")
income_change_magnitude = pd.to_numeric(pool["INEXQ2"], errors="coerce").abs()
pool["expected_percent_change"] = pd.NA
pool.loc[income_change_direction.isin([1, 2]), "expected_percent_change"] = income_change_magnitude
pool.loc[income_change_direction.eq(3), "expected_percent_change"] = 0
pool.loc[income_change_direction.eq(5), "expected_percent_change"] = -income_change_magnitude
pool["expected_percent_change"] = pd.to_numeric(pool["expected_percent_change"], errors="coerce")
step_rows.append(
    construction_step(
        TASK_SLUG, 3, "Construct leads and lags",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)


# 4. Restrict universe.
rows_before = len(pool)
step_rows.append(
    construction_step(
        TASK_SLUG, 4, "Restrict universe",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)


# 5. Require observed target(s).
rows_before = len(pool)
condition_masks = []
detail_order = 1
for filter_id, pass_condition, pass_mask in [
    ("target_inexq1_not_missing", 'pool["INEXQ1"].notna()', pool["INEXQ1"].notna()),
    (
        "target_inexq2_not_missing_if_income_changes",
        '(~pool["INEXQ1"].isin([1, 2, 5])) | pool["INEXQ2"].notna()',
        (~pool["INEXQ1"].isin([1, 2, 5])) | pool["INEXQ2"].notna(),
    ),
]:
    condition_masks.append(pass_mask)
    drop_detail_rows.append(
        filter_count(
            TASK_SLUG, 5, detail_order,
            filter_id, pass_condition, int((~pass_mask).sum()),
        )
    )
    detail_order += 1
target_observed = pd.Series(True, index=pool.index)
for pass_mask in condition_masks:
    target_observed &= pass_mask
pool = pool.loc[target_observed].copy()
step_rows.append(
    construction_step(
        TASK_SLUG, 5, "Require observed targets",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)


# 6. Require observed predictors.
rows_before = len(pool)
required_nonmissing = CONTEXT_CORE_COLS + MACRO_HISTORY_COLS + REQUIRED_PREDICTORS
condition_masks = []
detail_order = 1
for column in required_nonmissing:
    pass_condition = f'pool["{column}"].notna()'
    pass_mask = pool[column].notna()
    condition_masks.append(pass_mask)
    drop_detail_rows.append(
        filter_count(
            TASK_SLUG, 6, detail_order,
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
        TASK_SLUG, 6, "Require observed predictors",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)


# 7. Apply logical and validity filters.
rows_before = len(pool)
condition_masks = []
detail_order = 1
for filter_id, pass_condition, pass_mask in [
    ("quality_inexq1_mapped", 'pool["INEXQ1"].isin([1, 2, 3, 5])', pool["INEXQ1"].isin([1, 2, 3, 5])),
    (
        "quality_expected_percent_change_not_missing",
        'pool["expected_percent_change"].notna()',
        pool["expected_percent_change"].notna(),
    ),
    ("quality_pago_mapped", 'pool["PAGO"].isin([1, 3, 5])', pool["PAGO"].isin([1, 3, 5])),
    ("quality_bago_mapped", 'pool["BAGO"].isin([1, 3, 5])', pool["BAGO"].isin([1, 3, 5])),
    ("quality_sex_mapped", 'pool["SEX"].isin([1, 2])', pool["SEX"].isin([1, 2])),
    ("quality_region_mapped", 'pool["REGION"].isin([1, 2, 3, 4])', pool["REGION"].isin([1, 2, 3, 4])),
    ("quality_marry_mapped", 'pool["MARRY"].isin([1, 3, 4, 5])', pool["MARRY"].isin([1, 3, 4, 5])),
    ("quality_educ_mapped", 'pool["EDUC"].isin([1, 2, 3, 4, 5, 6])', pool["EDUC"].isin([1, 2, 3, 4, 5, 6])),
]:
    condition_masks.append(pass_mask)
    drop_detail_rows.append(
        filter_count(
            TASK_SLUG, 7, detail_order,
            filter_id, pass_condition, int((~pass_mask).sum()),
        )
    )
    detail_order += 1
quality_mask = pd.Series(True, index=pool.index)
for pass_mask in condition_masks:
    quality_mask &= pass_mask
pool = pool.loc[quality_mask].copy()
if pool.empty:
    raise RuntimeError(f"{TASK_SLUG}: no rows remain after logical and validity filters.")

if pool.loc[pool["INEXQ1"].isin([1, 2]), "expected_percent_change"].lt(0).any():
    raise RuntimeError(f"{TASK_SLUG}: higher-income expectations have negative signed changes.")
if pool.loc[pool["INEXQ1"].eq(3), "expected_percent_change"].ne(0).any():
    raise RuntimeError(f"{TASK_SLUG}: same-income expectations have nonzero signed changes.")
if pool.loc[pool["INEXQ1"].eq(5), "expected_percent_change"].gt(0).any():
    raise RuntimeError(f"{TASK_SLUG}: lower-income expectations have positive signed changes.")
lower_nonzero = pool["INEXQ1"].eq(5) & pd.to_numeric(pool["INEXQ2"], errors="coerce").abs().gt(0)
if pool.loc[lower_nonzero, "expected_percent_change"].ge(0).any():
    raise RuntimeError(f"{TASK_SLUG}: lower-income nonzero magnitudes are not negative.")
validate_code_domains(
    pool,
    code_domains=CODE_DOMAINS,
    columns=["SEX", "REGION", "MARRY", "EDUC", "PAGO", "BAGO", "INEXQ1"],
    task_slug=TASK_SLUG,
)
step_rows.append(
    construction_step(
        TASK_SLUG, 7, "Apply logical and validity filters",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)

# 8. Apply outlier trimming.
rows_before = len(pool)
trim_masks = []
for detail_order, variable in enumerate(["expected_percent_change", "INCOME"], start=1):
    p01 = f"{variable}_p01"
    p99 = f"{variable}_p99"
    if pool[[p01, p99]].isna().any().any():
        raise RuntimeError(f"{TASK_SLUG}: missing monthly cutoffs for {variable} entering step 8.")
    pass_mask = pool[variable].gt(pool[p01]) & pool[variable].lt(pool[p99])
    trim_masks.append(pass_mask)
    drop_detail_rows.append(
        filter_count(
            TASK_SLUG, 8, detail_order,
            f"trim_{variable}_strict_inside_month_p01_p99", f'pool["{variable}"].gt(pool["{p01}"]) & pool["{variable}"].lt(pool["{p99}"])', int((~pass_mask).sum()),
        )
    )
trim_mask = pd.Series(True, index=pool.index)
for pass_mask in trim_masks:
    trim_mask &= pass_mask
pool = pool.loc[trim_mask].copy()
if pool.empty:
    raise RuntimeError(f"{TASK_SLUG}: no rows remain after outlier trimming.")
step_rows.append(
    construction_step(
        TASK_SLUG, 8, "Apply outlier trimming",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)

release_source_counts = (
    pool["release_date_source"]
    .value_counts(dropna=False)
    .rename_axis("release_date_source")
    .reset_index(name="rows")
)
RELEASE_SOURCE_COUNTS_PATH.parent.mkdir(parents=True, exist_ok=True)
release_source_counts.to_csv(RELEASE_SOURCE_COUNTS_PATH, index=False)

pool["selection_period"] = pd.to_datetime(pool["survey_date"]).dt.strftime("%Y-%m")
pool["row_id"] = pool["CASEID"].astype("int64").astype("string")
# Public keys are fixed before sampling; source keys retain their sort types.
pool["id"] = pool["row_id"]
pool["time"] = pd.to_datetime(pool["survey_date"], errors="raise").dt.strftime("%Y-%m-%d")
pool["release_date"] = pd.to_datetime(pool["release_date"], errors="raise").dt.strftime("%Y-%m-%d")
sample, sampling_registry, sampling_summary = sample_task_records(
    pool,
    task_id=TASK_SLUG,
    output_dir=SAMPLING_DIR,
    output_sort_cols=["survey_date", "CASEID"],
)
rows_before_export = len(pool)

out = sample.copy().reset_index(drop=True)
out["subject_id"] = out["panel_id"].astype("int64").astype(str)
out["release_date"] = out["release_date"].map(iso_date_or_none)
for column in LABELLED_PREDICTORS:
    label_column = f"{column}_label"
    if out[label_column].isna().any():
        raise RuntimeError(f"{TASK_SLUG}: labelled predictor {label_column} has missing values.")
    out[column] = out[label_column].astype("string")
keep = ["subject_id", "selection_period"] + [
    column for column in SAMPLE_COLUMNS if column in out.columns
]

TABULAR_PATH.parent.mkdir(parents=True, exist_ok=True)
canonicalize_public_table(out, task_id=TASK_SLUG).to_csv(TABULAR_PATH, index=False)
step_rows.append(
    construction_step(
        TASK_SLUG, 9, "Export sample",
        int(rows_before_export), int(len(sample)), int(rows_before_export) - int(len(sample)),
    )
)

DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(step_rows, columns=STEP_COLUMNS).to_csv(STEP_DIAGNOSTICS_PATH, index=False)
pd.DataFrame(drop_detail_rows, columns=DETAIL_COLUMNS).to_csv(DROP_DETAIL_PATH, index=False)
pd.concat(cutoff_rows, ignore_index=True).loc[:, [
    "task_id", "variable", "role", "period", "n_reference", "p01", "p99",
    "n_at_or_below_p01", "n_at_or_above_p99",
]].to_csv(OUTLIER_CUTOFF_PATH, index=False)
print(
    f"{TASK_SLUG}: eligible={len(pool):,} sampled={len(sample):,} "
    f"tabular={TABULAR_PATH}"
)
