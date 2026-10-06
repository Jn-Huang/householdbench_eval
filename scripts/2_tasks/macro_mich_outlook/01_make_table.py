#!/usr/bin/env python
"""Build the tabular CSV for the Michigan macro-outlook task."""

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


TASK_SLUG = "macro_mich_outlook"
INPUT_PATH = PROJECT_ROOT / "data/intermediate/mich.parquet"
TABULAR_PATH = PROJECT_ROOT / "data" / "householdbench" / "tabular" / f"{TASK_SLUG}.csv"
DIAGNOSTICS_DIR = PROJECT_ROOT / "output" / "householdbench" / "tasks" / TASK_SLUG
SAMPLING_DIR = PROJECT_ROOT / "data" / "intermediate" / "householdbench" / TASK_SLUG
STEP_DIAGNOSTICS_PATH = DIAGNOSTICS_DIR / "01_sample_construction_steps.csv"
DROP_DETAIL_PATH = DIAGNOSTICS_DIR / "01_sample_construction_drop_details.csv"
OUTLIER_CUTOFF_PATH = DIAGNOSTICS_DIR / "01_outlier_cutoffs.csv"
PANEL_GAP_PATH = DIAGNOSTICS_DIR / "01_panel_gap_distribution.csv"
PANEL_WAVE_PATH = DIAGNOSTICS_DIR / "01_panel_wave_distribution.csv"
PANEL_RECONCILIATION_PATH = DIAGNOSTICS_DIR / "01_panel_link_reconciliation.csv"
RELEASE_SOURCE_COUNTS_PATH = (
    PROJECT_ROOT
    / "output"
    / "householdbench"
    / "tasks"
    / TASK_SLUG
    / "01_release_date_source_counts.csv"
)
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
REQUIRED_PREDICTORS = ["PAGO", "BAGO", "GOVT", "DUR"]
OPTIONAL_PREDICTORS = ["HOM", "CAR", "SHOM"]
TARGET_COLS = ["PX1", "PX5"]
TARGET_AUDIT_COLS = ["PX1_outlier_abs_gt25", "PX5_outlier_abs_gt25"]
LABELLED_PREDICTORS = ["SEX", "REGION", "MARRY", "EDUC", "PAGO", "BAGO", "GOVT", "DUR", "HOM", "CAR", "SHOM"]
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
            "prev_caseid",
            "months_since_prev",
            "panel_chain_cycle_flag",
            "DATEPR",
        ]
        + REQUIRED_PREDICTORS
        + OPTIONAL_PREDICTORS
        + TARGET_COLS
        + TARGET_AUDIT_COLS
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
    "GOVT": {1, 3, 5},
    "DUR": {1, 3, 5},
}
SAMPLE_COLUMNS = list(
    dict.fromkeys(
        [
            "sample_id",
            "selection_period",
            "methodology_regime",
            "panel_link_expected",
            "panel_link_ok",
            "panel_chain_cycle_flag",
            "panel_id",
            "panel_wave",
            "prev_caseid",
            "prior_survey_date",
            "prior_release_date",
            "prior_YYYYMM",
            "months_since_prev",
            "prior_PX1",
            "prior_PX5",
            "SAMPLE",
            "release_date",
            "release_date_source",
        ]
        + CONTEXT_CORE_COLS
        + MACRO_HISTORY_COLS
        + REQUIRED_PREDICTORS
        + OPTIONAL_PREDICTORS
        + TARGET_COLS
        + TARGET_AUDIT_COLS
    )
)


step_rows = []
drop_detail_rows = []


# 1. Start from pre-processed main source.
mich = load_microdata(INPUT_PATH, REQUIRED_INPUT_COLS)
if mich["CASEID"].isna().any() or mich["CASEID"].duplicated().any():
    raise RuntimeError("Michigan source CASEID must be unique and nonmissing.")
source_rows = len(mich)
panel_links_expected = int(mich["panel_link_expected"].eq(1).sum())
panel_links_successful = int(mich["panel_link_ok"].eq(1).sum())

cutoff_rows = []
cutoffs_by_variable: dict[str, pd.DataFrame] = {}
for variable, role in [("PX1", "target"), ("PX5", "target"), ("INCOME", "predictor")]:
    reference = mich.loc[
        mich["CASEID"].notna() & mich["YYYYMM"].notna() & mich[variable].notna(),
        ["YYYYMM", variable],
    ].copy()
    cutoffs = (
        reference.groupby("YYYYMM")[variable]
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
            f"{TASK_SLUG}: degenerate {variable} monthly cutoffs: "
            f"{cutoffs.loc[bad].to_dict('records')}"
        )
    reference = reference.merge(cutoffs, on="YYYYMM", how="left", validate="many_to_one")
    boundary_counts = (
        reference.assign(
            n_at_or_below_p01=reference[variable].le(reference["p01"]),
            n_at_or_above_p99=reference[variable].ge(reference["p99"]),
        )
        .groupby("YYYYMM", as_index=False)[["n_at_or_below_p01", "n_at_or_above_p99"]]
        .sum()
    )
    audit = cutoffs.merge(boundary_counts, on="YYYYMM", validate="one_to_one")
    audit["task_id"] = TASK_SLUG
    audit["variable"] = variable
    audit["role"] = role
    audit["period"] = audit["YYYYMM"].astype(int).astype(str).str.slice(0, 4) + "-" + audit["YYYYMM"].astype(int).astype(str).str.slice(4, 6)
    cutoff_rows.append(audit)
    cutoff_columns = cutoffs[["YYYYMM", "p01", "p99"]].rename(
        columns={"p01": f"{variable}_p01", "p99": f"{variable}_p99"}
    )
    cutoffs_by_variable[variable] = cutoff_columns.copy()
    mich = mich.merge(cutoff_columns, on="YYYYMM", how="left", validate="many_to_one")
step_rows.append(
    construction_step(
        TASK_SLUG, 1, "Start from source observations",
        0, int(len(mich)), 0,
    )
)


# 2. Attach additional sources.
rows_before = len(mich)
predecessor_lookup = mich[
    ["CASEID", "survey_date", "release_date", "YYYYMM", "PX1", "PX5"]
].rename(
    columns={
        "CASEID": "prev_caseid_lookup",
        "survey_date": "prior_survey_date",
        "release_date": "prior_release_date",
        "YYYYMM": "prior_YYYYMM",
        "PX1": "prior_PX1",
        "PX5": "prior_PX5",
    }
)
mich = mich.merge(
    predecessor_lookup,
    left_on="prev_caseid",
    right_on="prev_caseid_lookup",
    how="left",
    validate="many_to_one",
)
for variable in ["PX1", "PX5"]:
    prior_cutoffs = cutoffs_by_variable[variable].rename(
        columns={
            "YYYYMM": "prior_YYYYMM",
            f"{variable}_p01": f"prior_{variable}_p01",
            f"{variable}_p99": f"prior_{variable}_p99",
        }
    )
    mich = mich.merge(prior_cutoffs, on="prior_YYYYMM", how="left", validate="many_to_one")
if len(mich) != rows_before:
    raise RuntimeError(f"{TASK_SLUG}: predecessor attachment changed the current row count.")
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
for column in ["survey_date", "release_date", "prior_survey_date", "prior_release_date"]:
    mich[column] = pd.to_datetime(mich[column], errors="coerce")
linked = mich["panel_link_ok"].eq(1)
if mich.loc[linked, [
    "prev_caseid_lookup",
    "prior_survey_date",
    "prior_release_date",
    "prior_YYYYMM",
    "months_since_prev",
    "panel_id",
    "panel_wave",
]].isna().any().any():
    raise RuntimeError(f"{TASK_SLUG}: linked rows contain missing predecessor fields.")
if not mich.loc[linked, "prev_caseid"].eq(mich.loc[linked, "prev_caseid_lookup"]).all():
    raise RuntimeError(f"{TASK_SLUG}: predecessor lookup does not match prev_caseid.")
computed_gap = (
    (mich["survey_date"].dt.year - mich["prior_survey_date"].dt.year) * 12
    + mich["survey_date"].dt.month
    - mich["prior_survey_date"].dt.month
)
if not computed_gap.loc[linked].eq(mich.loc[linked, "months_since_prev"]).all():
    raise RuntimeError(f"{TASK_SLUG}: computed predecessor gap disagrees with source months_since_prev.")
linked_rows_with_both_prior_expectations = int(
    (linked & mich[["prior_PX1", "prior_PX5"]].notna().all(axis=1)).sum()
)
step_rows.append(
    construction_step(
        TASK_SLUG, 3, "Construct leads and lags",
        int(rows_before), int(len(mich)), int(rows_before) - int(len(mich)),
    )
)

pool = mich.copy()


# 4. Restrict universe.
rows_before = len(pool)
universe_conditions = [
    ("universe_panel_link_expected", 'pool["panel_link_expected"].eq(1)', pool["panel_link_expected"].eq(1)),
    ("universe_panel_link_ok", 'pool["panel_link_ok"].eq(1)', pool["panel_link_ok"].eq(1)),
    ("universe_panel_chain_acyclic", 'pool["panel_chain_cycle_flag"].eq(0)', pool["panel_chain_cycle_flag"].eq(0)),
    ("universe_prev_caseid_observed", 'pool["prev_caseid"].notna()', pool["prev_caseid"].notna()),
    ("universe_panel_id_observed", 'pool["panel_id"].notna()', pool["panel_id"].notna()),
    ("universe_panel_wave_reinterview", 'pool["panel_wave"].isin([2, 3])', pool["panel_wave"].isin([2, 3])),
    ("universe_gap_6_7_8_months", 'pool["months_since_prev"].isin([6, 7, 8])', pool["months_since_prev"].isin([6, 7, 8])),
    ("universe_predecessor_earlier", 'pool["prior_survey_date"].lt(pool["survey_date"])', pool["prior_survey_date"].lt(pool["survey_date"])),
    ("universe_predecessor_month_matches_DATEPR", 'pool["prior_YYYYMM"].eq(pool["DATEPR"])', pd.to_numeric(pool["prior_YYYYMM"], errors="coerce").eq(pd.to_numeric(pool["DATEPR"], errors="coerce"))),
    ("universe_predecessor_public_by_current_survey", 'pool["prior_release_date"].le(pool["survey_date"])', pool["prior_release_date"].le(pool["survey_date"])),
]
universe_mask = pd.Series(True, index=pool.index)
for detail_order, (filter_id, pass_condition, pass_mask) in enumerate(universe_conditions, start=1):
    drop_detail_rows.append(
        filter_count(
            TASK_SLUG, 4, detail_order,
            filter_id, pass_condition, int((universe_mask & ~pass_mask).sum()),
        )
    )
    universe_mask &= pass_mask
pool = pool.loc[universe_mask].copy()
rows_passing_current_task_restrictions = len(pool)
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
for column in TARGET_COLS:
    pass_condition = f'pool["{column}"].notna()'
    pass_mask = pool[column].notna()
    condition_masks.append(pass_mask)
    drop_detail_rows.append(
        filter_count(
            TASK_SLUG, 5, detail_order,
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
        TASK_SLUG, 5, "Require observed targets",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)


# 6. Require observed predictors.
rows_before = len(pool)
required_nonmissing = (
    CONTEXT_CORE_COLS
    + MACRO_HISTORY_COLS
    + REQUIRED_PREDICTORS
    + ["prior_PX1", "prior_PX5", "months_since_prev"]
)
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
    ("quality_pago_mapped", 'pool["PAGO"].isin([1, 3, 5])', pool["PAGO"].isin([1, 3, 5])),
    ("quality_bago_mapped", 'pool["BAGO"].isin([1, 3, 5])', pool["BAGO"].isin([1, 3, 5])),
    ("quality_govt_mapped", 'pool["GOVT"].isin([1, 3, 5])', pool["GOVT"].isin([1, 3, 5])),
    ("quality_dur_mapped", 'pool["DUR"].isin([1, 3, 5])', pool["DUR"].isin([1, 3, 5])),
    ("quality_sex_mapped", 'pool["SEX"].isin([1, 2])', pool["SEX"].isin([1, 2])),
    ("quality_region_mapped", 'pool["REGION"].isin([1, 2, 3, 4])', pool["REGION"].isin([1, 2, 3, 4])),
    ("quality_marry_mapped", 'pool["MARRY"].isin([1, 3, 4, 5])', pool["MARRY"].isin([1, 3, 4, 5])),
    ("quality_educ_mapped", 'pool["EDUC"].isin([1, 2, 3, 4, 5, 6])', pool["EDUC"].isin([1, 2, 3, 4, 5, 6])),
    ("quality_current_predecessor_caseids_differ", 'pool["CASEID"].ne(pool["prev_caseid"])', pool["CASEID"].ne(pool["prev_caseid"])),
    ("quality_panel_wave_unique", 'no duplicate (panel_id, panel_wave)', ~pool.duplicated(["panel_id", "panel_wave"], keep=False)),
    ("quality_elapsed_months_exact", 'computed_gap.eq(pool["months_since_prev"])', computed_gap.loc[pool.index].eq(pool["months_since_prev"])),
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
validate_code_domains(
    pool,
    code_domains=CODE_DOMAINS,
    columns=["SEX", "REGION", "MARRY", "EDUC", "PAGO", "BAGO", "GOVT", "DUR"],
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
for detail_order, variable in enumerate(["PX1", "PX5", "INCOME"], start=1):
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
rows_passing_current_trims = len(pool)
prior_trim_masks = []
for detail_order, variable in enumerate(["PX1", "PX5"], start=4):
    value = f"prior_{variable}"
    p01 = f"prior_{variable}_p01"
    p99 = f"prior_{variable}_p99"
    if pool[[p01, p99]].isna().any().any():
        raise RuntimeError(
            f"{TASK_SLUG}: missing predecessor-month cutoffs for {variable}."
        )
    pass_mask = pool[value].gt(pool[p01]) & pool[value].lt(pool[p99])
    prior_trim_masks.append(pass_mask)
    drop_detail_rows.append(
        filter_count(
            TASK_SLUG, 8, detail_order,
            f"trim_{value}_strict_inside_predecessor_month_p01_p99", f'pool["{value}"].gt(pool["{p01}"]) & pool["{value}"].lt(pool["{p99}"])', int((~pass_mask).sum()),
        )
    )
prior_trim_mask = pd.Series(True, index=pool.index)
for pass_mask in prior_trim_masks:
    prior_trim_mask &= pass_mask
pool = pool.loc[prior_trim_mask].copy()
rows_passing_predecessor_trims = len(pool)
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

pool["subject_id"] = pool["panel_id"].round().astype(int).astype(str)
pool["sample_id"] = pool["CASEID"].round().astype(int).astype(str)
pool["selection_period"] = pd.to_datetime(pool["survey_date"]).dt.strftime("%Y-%m")
eligible_for_selection = pool.copy()
# Public keys are fixed before sampling; source keys retain their sort types.
pool["id"] = pool["subject_id"].astype("string")
pool["time"] = pd.to_datetime(pool["survey_date"], errors="raise").dt.strftime("%Y-%m-%d")
pool["release_date"] = pd.to_datetime(pool["release_date"], errors="raise").dt.strftime("%Y-%m-%d")
sample, sampling_registry, sampling_summary = sample_task_records(
    pool,
    task_id=TASK_SLUG,
    output_dir=SAMPLING_DIR,
    output_sort_cols=["survey_date", "panel_id", "CASEID"],
)
rows_before_export = len(pool)

out = sample.copy().reset_index(drop=True)
out["release_date"] = out["release_date"].map(iso_date_or_none)
out["prior_release_date"] = out["prior_release_date"].map(iso_date_or_none)
out["prior_survey_date"] = out["prior_survey_date"].map(iso_date_or_none)
for column in LABELLED_PREDICTORS:
    label_column = f"{column}_label"
    if column in REQUIRED_PREDICTORS or column in {"SEX", "REGION", "MARRY", "EDUC"}:
        if out[label_column].isna().any():
            raise RuntimeError(f"{TASK_SLUG}: labelled predictor {label_column} has missing values.")
    out[column] = out[label_column].astype("string")
keep = ["subject_id"] + [column for column in SAMPLE_COLUMNS if column in out.columns]

eligible_gap_counts = (
    eligible_for_selection["months_since_prev"].astype(int).value_counts().sort_index()
)
final_gap_counts = out["months_since_prev"].astype(int).value_counts().sort_index()
gap_distribution = pd.DataFrame({"months_since_prev": [6, 7, 8]})
gap_distribution["eligible_rows"] = gap_distribution["months_since_prev"].map(eligible_gap_counts).fillna(0).astype(int)
gap_distribution["final_rows"] = gap_distribution["months_since_prev"].map(final_gap_counts).fillna(0).astype(int)
eligible_wave_counts = eligible_for_selection["panel_wave"].astype(int).value_counts().sort_index()
final_wave_counts = out["panel_wave"].astype(int).value_counts().sort_index()
wave_distribution = pd.DataFrame({"panel_wave": [2, 3]})
wave_distribution["eligible_rows"] = wave_distribution["panel_wave"].map(eligible_wave_counts).fillna(0).astype(int)
wave_distribution["final_rows"] = wave_distribution["panel_wave"].map(final_wave_counts).fillna(0).astype(int)
panel_reconciliation = pd.DataFrame(
    [
        ("source rows", source_rows),
        ("panel links expected", panel_links_expected),
        ("panel links successful", panel_links_successful),
        ("linked rows with both prior expectations", linked_rows_with_both_prior_expectations),
        ("rows passing current task restrictions", rows_passing_current_task_restrictions),
        ("rows passing current trims", rows_passing_current_trims),
        ("rows passing predecessor trims", rows_passing_predecessor_trims),
        ("final selected rows", len(out)),
    ],
    columns=["metric", "rows"],
)

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
gap_distribution.to_csv(PANEL_GAP_PATH, index=False)
wave_distribution.to_csv(PANEL_WAVE_PATH, index=False)
panel_reconciliation.to_csv(PANEL_RECONCILIATION_PATH, index=False)
print(
    f"{TASK_SLUG}: eligible={len(pool):,} sampled={len(sample):,} "
    f"tabular={TABULAR_PATH}"
)
