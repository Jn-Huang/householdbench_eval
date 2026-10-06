#!/usr/bin/env python
"""Build the HouseholdBench CEX 2008 stimulus-payment task table."""

from __future__ import annotations

import sys
import zipfile
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


TASK_ID = "cons_cex_stimulus08"
INPUT_PATH = PROJECT_ROOT / "data/intermediate/cex_interview.parquet"
CEX_RAW_DIR = PROJECT_ROOT / "data/raw/micro/cex/interview"
REBATE_ZIP_PATH = CEX_RAW_DIR / "intrvw08.zip"
REBATE_MEMBER = "expn08/rbt08.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/householdbench/tabular/cons_cex_stimulus08.csv"
DIAGNOSTICS_DIR = PROJECT_ROOT / "output" / "householdbench" / "tasks" / TASK_ID
SAMPLING_DIR = PROJECT_ROOT / "data" / "intermediate" / "householdbench" / TASK_ID
STEP_DIAGNOSTICS_PATH = DIAGNOSTICS_DIR / "01_sample_construction_steps.csv"
DROP_DETAIL_PATH = DIAGNOSTICS_DIR / "01_sample_construction_drop_details.csv"
OUTLIER_CUTOFF_PATH = DIAGNOSTICS_DIR / "01_outlier_cutoffs.csv"
ASSIGNMENT_DETAIL_PATH = DIAGNOSTICS_DIR / "01_stimulus_payment_assignment_details.csv"
TIMING_SUMMARY_PATH = DIAGNOSTICS_DIR / "01_stimulus_timing_summary.csv"
DESCRIPTIVES_AUDIT_PATH = DIAGNOSTICS_DIR / "01_descriptives_audit_input.csv"

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
EXPENDITURE_COLUMNS = ["cons_parker_total", "cons_nondurables", "cons_durables"]
INPUT_COLUMNS = [
    "newid",
    "cu_id",
    "cu_id_raw",
    "cex_panel_id",
    "interview_year",
    "interview_month",
    "quarter",
    "interview_position_harmonized",
    *DEMOGRAPHIC_INPUT_COLUMNS,
    "income_before_tax",
    *EXPENDITURE_COLUMNS,
    "cons_food",
    "release_date",
    "release_date_source",
]
LAGGED_HOUSEHOLD_COLUMNS = [
    "income_before_tax",
    *EXPENDITURE_COLUMNS,
    "cons_food",
    "income_before_tax_p01",
    "income_before_tax_p99",
    "cons_parker_total_p01",
    "cons_parker_total_p99",
    "rebate_amount_total",
    "interview_month_index",
]
HISTORY_REQUIRED_PATTERNS = [
    "cons_parker_total_lag{lag}",
    "cons_nondurables_lag{lag}",
    "cons_durables_lag{lag}",
]
POLICY_COLUMNS = [
    "rebate_amount_total",
    "rebate_record_count",
    "rebate_month_unique_count",
    "rebate_month_first",
    "rebate_month_last",
    "stimulus_payment_received",
    "stimulus_payment_late_in_period",
    "stimulus_payment_reported_any",
    "stimulus_late_reported_any",
    "stimulus_timing_group",
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
    "rebate_amount_total_lag1",
    "rebate_amount_total_lag2",
    "rebate_amount_total_lag3",
    *POLICY_COLUMNS,
    *EXPENDITURE_COLUMNS,
    "cons_parker_total_change",
    "cons_nondurables_change",
    "cons_durables_change",
]


step_rows = []
drop_detail_rows = []
timing_summary_rows = []


# 1. Start from pre-processed main source.
source_columns = pq.read_schema(INPUT_PATH).names
missing_columns = [column for column in INPUT_COLUMNS if column not in source_columns]
if missing_columns:
    raise SystemExit(f"{INPUT_PATH} is missing required columns: {missing_columns}")

cex = pd.read_parquet(INPUT_PATH, columns=INPUT_COLUMNS)
cex = cex.rename(columns=CATEGORICAL_RENAMES)
text_columns = {
    "newid",
    "cu_id",
    "cu_id_raw",
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
for column in ["newid", "cu_id", "cu_id_raw", "cex_panel_id"]:
    cex[column] = cex[column].astype("string").str.strip()
    if cex[column].isna().any() or cex[column].eq("").any():
        raise SystemExit(f"Cleaned CEX data contain missing {column} values.")
cex["newid"] = cex["newid"].str.zfill(8)
if cex[["interview_year", "interview_month", "newid"]].duplicated().any():
    raise SystemExit(
        "Cleaned CEX data must contain unique, nonmissing newid values within year."
    )
cex["cex_row_id"] = (
    cex["interview_year"].astype(int).astype("string")
    + ":"
    + cex["interview_month"].astype(int).astype("string").str.zfill(2)
    + ":"
    + cex["newid"]
)

cex["release_date"] = pd.to_datetime(cex["release_date"], errors="coerce")
if cex["release_date"].isna().any():
    raise SystemExit("Cleaned CEX data contain missing release_date values.")
if (
    cex["release_date_source"].isna().any()
    or cex["release_date_source"].str.strip().eq("").any()
):
    raise SystemExit("Cleaned CEX data contain missing release_date_source values.")
cex["q_index"] = cex["interview_year"].astype(int) * 4 + cex["quarter"].astype(int)
cex["interview_month_index"] = cex["interview_year"].astype(int) * 12 + cex[
    "interview_month"
].astype(int)
cex["n_adults"] = (cex["fam_size"] - cex["n_kids"]).clip(lower=0)
if cex.duplicated(["cex_panel_id", "interview_month_index"]).any():
    raise SystemExit(
        "Cleaned CEX data contain multiple interviews for a CU in one interview month."
    )
if cex.duplicated(["cex_panel_id", "interview_year", "quarter"]).any():
    raise SystemExit("CEX cutoff reference is not unique by CU and interview quarter.")
cutoff_rows = []
cutoff_frames = []
for variable, role in [("income_before_tax", "predictor"), ("cons_parker_total", "predictor_and_target")]:
    reference = cex.loc[cex["cex_panel_id"].notna() & cex["interview_year"].notna() & cex["quarter"].notna() & cex[variable].notna(), ["interview_year", "quarter", variable]].copy()
    cutoffs = reference.groupby(["interview_year", "quarter"])[variable].agg(n_reference="count", p01=lambda values: values.quantile(0.01), p99=lambda values: values.quantile(0.99)).reset_index()
    bad = cutoffs["p01"].ge(cutoffs["p99"])
    if bad.any():
        raise SystemExit(f"{TASK_ID}: degenerate {variable} quarterly cutoffs: {cutoffs.loc[bad].to_dict('records')}")
    reference = reference.merge(cutoffs, on=["interview_year", "quarter"], how="left", validate="many_to_one")
    boundary = reference.assign(n_at_or_below_p01=reference[variable].le(reference["p01"]), n_at_or_above_p99=reference[variable].ge(reference["p99"])).groupby(["interview_year", "quarter"], as_index=False)[["n_at_or_below_p01", "n_at_or_above_p99"]].sum()
    audit = cutoffs.merge(boundary, on=["interview_year", "quarter"], validate="one_to_one")
    audit["task_id"] = TASK_ID
    audit["variable"] = variable
    audit["role"] = role
    audit["period"] = audit["interview_year"].astype(int).astype(str) + "-Q" + audit["quarter"].astype(int).astype(str)
    cutoff_rows.append(audit)
    cutoff_frames.append(cutoffs[["interview_year", "quarter", "p01", "p99"]].rename(columns={"p01": f"{variable}_p01", "p99": f"{variable}_p99"}))
for cutoffs in cutoff_frames:
    cex = cex.merge(cutoffs, on=["interview_year", "quarter"], how="left", validate="many_to_one")
step_rows.append(
    construction_step(
        TASK_ID, 1, "Start from source observations",
        int(len(cex)), int(len(cex)), 0,
    )
)


# 2. Attach macroeconomic context and assign raw payment reports.
rows_before = len(cex)
cex = attach_macro_context(
    cex,
    origin_q_index_col="q_index",
    required_columns=CORE_MACRO_COLUMNS,
)
if len(cex) != rows_before:
    raise SystemExit(f"{TASK_ID}: macroeconomic merge changed the CEX row count.")

if not REBATE_ZIP_PATH.exists():
    raise SystemExit(f"Required CEX raw archive not found: {REBATE_ZIP_PATH}")
with zipfile.ZipFile(REBATE_ZIP_PATH) as archive:
    with archive.open(REBATE_MEMBER) as handle:
        raw_rebates = pd.read_csv(handle)
raw_rebates.columns = [
    str(column).strip().strip('"').upper() for column in raw_rebates.columns
]
raw_rebates["raw_rebate_source_row"] = range(2, len(raw_rebates) + 2)
rebate_input_columns = [
    "NEWID",
    "CUID",
    "RBTMO",
    "RBTAMT",
    "CHCKEFT",
    "HOWUSED",
    "USDINTMO",
    "USDINTYR",
]
missing_rebate_columns = [
    column for column in rebate_input_columns if column not in raw_rebates.columns
]
if missing_rebate_columns:
    raise SystemExit(
        f"{REBATE_ZIP_PATH} is missing required columns: {missing_rebate_columns}"
    )
for column in rebate_input_columns:
    raw_rebates[column] = pd.to_numeric(raw_rebates[column], errors="coerce")
raw_rebates = raw_rebates.loc[
    raw_rebates["NEWID"].notna()
    & raw_rebates["RBTMO"].between(1, 12)
    & raw_rebates["RBTAMT"].gt(0)
].copy()
if raw_rebates.empty:
    raise SystemExit(
        "The stimulus-payment addendum contains no valid positive payment rows."
    )
if raw_rebates["raw_rebate_source_row"].duplicated().any():
    raise SystemExit("Raw stimulus-payment source-row identifiers are not unique.")
for column in ["NEWID", "CUID", "RBTMO"]:
    if raw_rebates[column].isna().any() or raw_rebates[column].mod(1).ne(0).any():
        raise SystemExit(
            f"Valid stimulus-payment rows contain noninteger {column} values."
        )
if raw_rebates["CUID"].ne(raw_rebates["NEWID"].floordiv(10)).any():
    raise SystemExit("Raw stimulus-payment rows violate CUID == NEWID // 10.")
invalid_methods = sorted(
    raw_rebates.loc[
        raw_rebates["CHCKEFT"].notna() & ~raw_rebates["CHCKEFT"].isin([1, 2]), "CHCKEFT"
    ]
    .unique()
    .tolist()
)
if invalid_methods:
    raise SystemExit(
        f"Stimulus-payment rows contain unsupported CHCKEFT codes: {invalid_methods}"
    )

raw_rebates["reported_newid"] = (
    raw_rebates["NEWID"].astype("Int64").astype("string").str.zfill(8)
)
raw_rebates["reported_cuid"] = (
    raw_rebates["CUID"].astype("Int64").astype("string").str.zfill(7)
)
reported_context = cex.loc[
    cex["newid"].isin(raw_rebates["reported_newid"]),
    [
        "newid",
        "cu_id_raw",
        "cex_panel_id",
        "interview_year",
        "interview_month",
        "interview_month_index",
    ],
].rename(
    columns={
        "newid": "reported_newid",
        "cu_id_raw": "cleaned_cu_id_raw",
        "interview_year": "reported_interview_year",
        "interview_month": "reported_interview_month",
        "interview_month_index": "reported_interview_month_index",
    }
)
if reported_context["reported_newid"].duplicated().any():
    raise SystemExit(
        "Raw stimulus-payment NEWIDs match multiple cleaned CEX interviews."
    )
raw_rebates = raw_rebates.merge(
    reported_context,
    on="reported_newid",
    how="left",
    validate="many_to_one",
)
if raw_rebates["cex_panel_id"].isna().any():
    missing_ids = raw_rebates.loc[
        raw_rebates["cex_panel_id"].isna(), "reported_newid"
    ].unique()
    raise SystemExit(
        f"Raw stimulus-payment NEWIDs do not match cleaned CEX rows: {missing_ids[:10].tolist()}"
    )
if raw_rebates["reported_cuid"].ne(raw_rebates["cleaned_cu_id_raw"].astype("string")).any():
    raise SystemExit(
        "Raw stimulus-payment CUID values disagree with cleaned CEX cu_id values."
    )
raw_rebates["cex_panel_id"] = raw_rebates["cex_panel_id"].astype("string")
raw_rebates["payment_year"] = raw_rebates["reported_interview_year"].astype(int)
previous_year = raw_rebates["RBTMO"].gt(raw_rebates["reported_interview_month"])
raw_rebates.loc[previous_year, "payment_year"] -= 1
raw_rebates["payment_month"] = raw_rebates["RBTMO"].astype(int)
raw_rebates["payment_month_index"] = (
    raw_rebates["payment_year"] * 12 + raw_rebates["payment_month"]
)
raw_rebates["payment_is_late_parker"] = (
    raw_rebates["CHCKEFT"].eq(1) & raw_rebates["payment_month_index"].gt(2008 * 12 + 8)
) | (
    (raw_rebates["CHCKEFT"].isna() | raw_rebates["CHCKEFT"].eq(2))
    & raw_rebates["payment_month_index"].gt(2008 * 12 + 6)
)

household_report_status = (
    raw_rebates.groupby("cex_panel_id", sort=False)
    .agg(stimulus_late_reported_any=("payment_is_late_parker", "max"))
    .reset_index()
)
household_report_status["stimulus_payment_reported_any"] = 1
cex = cex.merge(household_report_status, on="cex_panel_id", how="left", validate="many_to_one")
cex["stimulus_payment_reported_any"] = (
    cex["stimulus_payment_reported_any"].fillna(0).astype(int)
)
cex["stimulus_late_reported_any"] = (
    cex["stimulus_late_reported_any"].eq(True).astype(int)
)
cex["stimulus_timing_group"] = "no_positive_payment_reported"
cex.loc[
    cex["stimulus_payment_reported_any"].eq(1)
    & cex["stimulus_late_reported_any"].eq(0),
    "stimulus_timing_group",
] = "on_time_only_payment_reporter"
cex.loc[cex["stimulus_late_reported_any"].eq(1), "stimulus_timing_group"] = (
    "late_payment_reporter"
)

candidate_parts = []
for offset in [1, 2, 3]:
    candidates_part = cex[
        ["cex_row_id", "newid", "cex_panel_id", "interview_month_index"]
    ].copy()
    candidates_part["payment_month_index"] = (
        candidates_part["interview_month_index"] - offset
    )
    candidates_part["assignment_offset_months"] = offset
    candidate_parts.append(candidates_part)
candidates = pd.concat(candidate_parts, ignore_index=True).rename(
    columns={"cex_row_id": "assigned_cex_row_id", "newid": "assigned_newid"}
)
candidates = (
    candidates.sort_values(
        ["cex_panel_id", "payment_month_index", "assignment_offset_months", "assigned_cex_row_id"],
        kind="mergesort",
    )
    .drop_duplicates(["cex_panel_id", "payment_month_index"], keep="first")
)
assignment = raw_rebates.merge(
    candidates[
        ["cex_panel_id", "payment_month_index", "assigned_cex_row_id", "assigned_newid"]
    ],
    on=["cex_panel_id", "payment_month_index"],
    how="left",
    validate="many_to_one",
)
if (
    len(assignment) != len(raw_rebates)
    or assignment["raw_rebate_source_row"].duplicated().any()
):
    raise SystemExit("Payment assignment duplicated or dropped valid raw payment rows.")
assignment["assignment_status"] = "unassigned_no_observed_period"
assigned = assignment["assigned_cex_row_id"].notna()
assignment.loc[assigned, "assignment_status"] = "reassigned_same_cu"
assignment.loc[
    assigned & assignment["assigned_newid"].eq(assignment["reported_newid"]),
    "assignment_status",
] = "assigned_reported_newid"
assignment["boundary_drop_reason"] = ""
assignment.loc[~assigned, "boundary_drop_reason"] = (
    "payment_month_outside_observed_expenditure_windows"
)
assigned_panel = assignment.loc[assigned, "assigned_cex_row_id"].map(
    cex.set_index("cex_row_id")["cex_panel_id"]
)
if assigned_panel.isna().any() or assigned_panel.ne(assignment.loc[assigned, "cex_panel_id"]).any():
    raise SystemExit("Assigned payment rows do not belong to the reporting CEX panel.")

raw_amount = float(assignment["RBTAMT"].sum())
assigned_amount = float(assignment.loc[assigned, "RBTAMT"].sum())
unassigned_amount = float(assignment.loc[~assigned, "RBTAMT"].sum())
if abs(raw_amount - assigned_amount - unassigned_amount) > 1e-6:
    raise SystemExit(
        "Assigned and unassigned payment amounts do not conserve the valid raw total."
    )

assigned_payments = assignment.loc[assigned].copy()
payment_aggregate = (
    assigned_payments.groupby("assigned_cex_row_id", sort=False)
    .agg(
        rebate_amount_total=("RBTAMT", "sum"),
        rebate_record_count=("RBTAMT", "size"),
        rebate_month_unique_count=("payment_month", "nunique"),
        rebate_month_first=("payment_month", "min"),
        rebate_month_last=("payment_month", "max"),
        stimulus_payment_late_in_period=("payment_is_late_parker", "max"),
    )
    .reset_index()
    .rename(columns={"assigned_cex_row_id": "cex_row_id"})
)
cex = cex.merge(payment_aggregate, on="cex_row_id", how="left", validate="one_to_one")
cex["rebate_amount_total"] = cex["rebate_amount_total"].fillna(0.0)
cex["rebate_record_count"] = cex["rebate_record_count"].fillna(0).astype(int)
cex["rebate_month_unique_count"] = (
    cex["rebate_month_unique_count"].fillna(0).astype(int)
)
cex["stimulus_payment_received"] = cex["rebate_amount_total"].gt(0).astype(int)
cex["stimulus_payment_late_in_period"] = (
    cex["stimulus_payment_late_in_period"].eq(True).astype(int)
)
if len(cex) != rows_before:
    raise SystemExit(
        f"{TASK_ID}: additional-source attachment changed the CEX row count."
    )

assignment_detail_columns = [
    "raw_rebate_source_row",
    "reported_newid",
    "reported_cuid",
    "cex_panel_id",
    "reported_interview_year",
    "reported_interview_month",
    "payment_year",
    "payment_month",
    "payment_month_index",
    "RBTAMT",
    "CHCKEFT",
    "HOWUSED",
    "payment_is_late_parker",
    "assigned_newid",
    "assignment_status",
    "boundary_drop_reason",
]
timing_summary_rows.extend(
    [
        {"metric": "valid_payment_rows", "value": int(len(assignment))},
        {"metric": "valid_payment_amount", "value": raw_amount},
        {
            "metric": "late_payment_rows",
            "value": int(assignment["payment_is_late_parker"].sum()),
        },
        {
            "metric": "payment_reporting_panels",
            "value": int(assignment["cex_panel_id"].nunique()),
        },
        {"metric": "assigned_payment_rows", "value": int(assigned.sum())},
        {"metric": "assigned_payment_amount", "value": assigned_amount},
        {"metric": "unassigned_payment_rows", "value": int((~assigned).sum())},
        {"metric": "unassigned_payment_amount", "value": unassigned_amount},
    ]
)
for status, count in (
    assignment["assignment_status"].value_counts().sort_index().items()
):
    timing_summary_rows.append(
        {"metric": f"assignment_status_{status}", "value": int(count)}
    )
step_rows.append(
    construction_step(
        TASK_ID, 2, "Attach additional sources",
        int(rows_before), int(len(cex)), int(rows_before) - int(len(cex)),
    )
)


# 3. Construct lagged household and macroeconomic histories.
rows_before = len(cex)
cex = cex.sort_values(
    ["cex_panel_id", "interview_month_index", "interview_position_harmonized"],
    kind="mergesort",
).reset_index(drop=True)
grouped_cex = cex.groupby("cex_panel_id", sort=False)
for lag in [1, 2, 3]:
    for column in LAGGED_HOUSEHOLD_COLUMNS:
        cex[f"{column}_lag{lag}"] = grouped_cex[column].shift(lag)

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
        f"{column}_lag{lag}" for column in LAGGED_HOUSEHOLD_COLUMNS
    ]
    cex.loc[~usable_lag_masks[lag], lag_columns] = pd.NA
step_rows.append(
    construction_step(
        TASK_ID, 3, "Construct leads and lags",
        int(rows_before), int(len(cex)), int(rows_before) - int(len(cex)),
    )
)


# 4. Restrict the universe without conditioning on payment status.
rows_before = len(cex)
fielding_start = 2008 * 12 + 6
fielding_end = 2009 * 12 + 3
history_start = 2007 * 12 + 9
history_end = 2009 * 12 + 3
fielding_panels = set(
    cex.loc[cex["interview_month_index"].between(fielding_start, fielding_end), "cex_panel_id"]
)
date_mask = cex["interview_month_index"].between(history_start, history_end)
fielding_cu_mask = cex["cex_panel_id"].isin(fielding_panels)
for detail_order, filter_id, pass_condition, pass_mask in [
    (
        1,
        "interview_in_history_window",
        'cex["interview_month_index"].between(2007 * 12 + 9, 2009 * 12 + 3)',
        date_mask,
    ),
    (
        2,
        "cu_observed_in_stimulus_fielding_window",
        'cex["cex_panel_id"].isin(fielding_panels)',
        fielding_cu_mask,
    ),
]:
    drop_detail_rows.append(
        filter_count(
            TASK_ID, 4, detail_order,
            filter_id, pass_condition, int((~pass_mask).sum()),
        )
    )
event = cex.loc[date_mask & fielding_cu_mask].copy()
step_rows.append(
    construction_step(
        TASK_ID, 4, "Restrict universe",
        int(rows_before), int(len(event)), int(rows_before) - int(len(event)),
    )
)


# 5. Require observed current expenditure targets.
rows_before = len(event)
target_observed = pd.Series(True, index=event.index)
for detail_order, column in enumerate(EXPENDITURE_COLUMNS, start=1):
    pass_mask = event[column].notna()
    target_observed &= pass_mask
    drop_detail_rows.append(
        filter_count(
            TASK_ID, 5, detail_order,
            f"target_{column}_not_missing", f'event["{column}"].notna()', int((~pass_mask).sum()),
        )
    )
event = event.loc[target_observed].copy()
step_rows.append(
    construction_step(
        TASK_ID, 5, "Require observed targets",
        int(rows_before), int(len(event)), int(rows_before) - int(len(event)),
    )
)


# 6. Require current fields and one complete, adjacent household history.
rows_before = len(event)
current_predictor_columns = [
    *DEMOGRAPHIC_COLUMNS,
    "n_adults",
    "income_before_tax",
    "release_date",
    "release_date_source",
]
lag1_predictor_columns = [
    pattern.format(lag=1) for pattern in HISTORY_REQUIRED_PATTERNS
]
condition_masks = []
detail_order = 1
for column in [*current_predictor_columns, *lag1_predictor_columns]:
    pass_mask = event[column].notna()
    condition_masks.append(pass_mask)
    drop_detail_rows.append(
        filter_count(
            TASK_ID, 6, detail_order,
            f"predictor_{column}_not_missing", f'event["{column}"].notna()', int((~pass_mask).sum()),
        )
    )
    detail_order += 1
predictors_observed = pd.Series(True, index=event.index)
for pass_mask in condition_masks:
    predictors_observed &= pass_mask
pool = event.loc[predictors_observed].copy()
step_rows.append(
    construction_step(
        TASK_ID, 6, "Require observed predictors",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)


# 7. Apply logical and validity filters to current and observed histories.
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
for column in EXPENDITURE_COLUMNS:
    pass_mask = pool[column].ge(0)
    condition_masks.append(pass_mask)
    drop_detail_rows.append(filter_count(
                                TASK_ID, 7, detail_order,
                                f"current_{column}_nonnegative", f'pool["{column}"].ge(0)', int((~pass_mask).sum()),
                            ))
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
    for column in EXPENDITURE_COLUMNS:
        lag_column = f"{column}_lag{lag}"
        pass_mask = pool[lag_column].isna() | pool[lag_column].ge(0)
        condition_masks.append(pass_mask)
        drop_detail_rows.append(filter_count(
                                    TASK_ID, 7, detail_order,
                                    f"lag{lag}_{column}_nonnegative", f'pool["{lag_column}"].isna() | pool["{lag_column}"].ge(0)', int((~pass_mask).sum()),
                                ))
        detail_order += 1
quality_mask = pd.Series(True, index=pool.index)
for pass_mask in condition_masks:
    quality_mask &= pass_mask
pool = pool.loc[quality_mask].copy()
if pool.empty:
    raise SystemExit(f"{TASK_ID} produced no eligible rows.")
step_rows.append(
    construction_step(
        TASK_ID, 7, "Apply logical and validity filters",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)

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
        value, p01, p99 = f"{variable}{suffix}", f"{variable}_p01{suffix}", f"{variable}_p99{suffix}"
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

for column in EXPENDITURE_COLUMNS:
    pool[f"{column}_change"] = pool[column] - pool[f"{column}_lag1"]


# 9. Select and export task rows.
rows_before_export = len(pool)
pool["selection_period"] = [
    f"{int(year):04d}-{int(month):02d}"
    for year, month in zip(pool["interview_year"], pool["interview_month"], strict=True)
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

table = sample.reset_index(drop=True).copy()
table["subject_id"] = table["cex_panel_id"].astype(str)
table["sample_id"] = table["newid"].astype(str)
table["prompt_time"] = [
    f"{int(year):04d}-{int(month):02d}-01"
    for year, month in zip(
        table["interview_year"], table["interview_month"], strict=True
    )
]
if table["sample_id"].isna().any() or table["sample_id"].duplicated().any():
    raise SystemExit("Selected rows must contain unique, nonmissing sample_id values.")
if table["release_date"].isna().any():
    raise SystemExit(
        "Selected rows contain missing current-target release_date values."
    )
if (
    table["release_date_source"].isna().any()
    or table["release_date_source"].str.strip().eq("").any()
):
    raise SystemExit(
        "Selected rows contain missing current-target release_date_source values."
    )

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
            f"{TASK_ID}: available lag-{lag} interview-month gaps are invalid."
        )
    if lag > 1:
        previous_available = table[
            f"interview_month_index_lag{lag - 1}"
        ].notna()
        if (available & ~previous_available).any():
            raise SystemExit(
                f"{TASK_ID}: lag {lag} is populated without lag {lag - 1}."
            )

missing_output = [column for column in OUTPUT_COLUMNS if column not in table.columns]
if missing_output:
    raise SystemExit(f"{TASK_ID} table is missing output columns: {missing_output}")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
public_table = canonicalize_public_table(table, task_id=TASK_ID)
descriptives_audit = table.copy()
descriptives_audit[["id", "time", "release_date"]] = public_table[["id", "time", "release_date"]]
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
descriptives_audit.to_csv(DESCRIPTIVES_AUDIT_PATH, index=False)
public_table.to_csv(OUTPUT_PATH, index=False)
step_rows.append(
    construction_step(
        TASK_ID, 9, "Export sample",
        int(rows_before_export), int(len(table)), int(rows_before_export) - int(len(table)),
    )
)

timing_summary_rows.extend(
    [
        {"metric": "final_rows", "value": int(len(table))},
        {"metric": "final_panels", "value": int(table["cex_panel_id"].nunique())},
        {
            "metric": "final_positive_payment_rows",
            "value": int(table["stimulus_payment_received"].sum()),
        },
        {
            "metric": "final_zero_payment_rows",
            "value": int(table["stimulus_payment_received"].eq(0).sum()),
        },
    ]
)
for lag in [1, 2, 3]:
    lag_label = "lag" if lag == 1 else "lags"
    timing_summary_rows.append(
        {
            "metric": f"final_rows_with_{lag}_history_{lag_label}",
            "value": int(table[f"interview_month_index_lag{lag}"].notna().sum()),
        }
    )
for group, count in (
    table.groupby("stimulus_timing_group")["cex_panel_id"].nunique().sort_index().items()
):
    timing_summary_rows.append({"metric": f"final_cus_{group}", "value": int(count)})

DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(step_rows, columns=STEP_COLUMNS).to_csv(STEP_DIAGNOSTICS_PATH, index=False)
pd.DataFrame(drop_detail_rows, columns=DROP_DETAIL_COLUMNS).to_csv(
    DROP_DETAIL_PATH, index=False
)
pd.concat(cutoff_rows, ignore_index=True).loc[:, ["task_id", "variable", "role", "period", "n_reference", "p01", "p99", "n_at_or_below_p01", "n_at_or_above_p99"]].to_csv(OUTLIER_CUTOFF_PATH, index=False)
assignment.loc[:, assignment_detail_columns].to_csv(ASSIGNMENT_DETAIL_PATH, index=False)
pd.DataFrame(timing_summary_rows, columns=["metric", "value"]).to_csv(
    TIMING_SUMMARY_PATH, index=False
)

print(
    f"{TASK_ID}: wrote {len(table):,} rows to {OUTPUT_PATH} "
    f"(eligible={len(pool):,}, "
    f"diagnostics={DIAGNOSTICS_DIR})"
)
