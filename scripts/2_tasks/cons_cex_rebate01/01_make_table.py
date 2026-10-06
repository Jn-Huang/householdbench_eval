#!/usr/bin/env python
"""Build the HouseholdBench CEX 2001 tax-rebate task table."""

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


TASK_ID = "cons_cex_rebate01"
INPUT_PATH = PROJECT_ROOT / "data/intermediate/cex_interview.parquet"
CEX_RAW_DIR = PROJECT_ROOT / "data/raw/micro/cex/interview"
REBATE_ZIP_PATH = CEX_RAW_DIR / "intrvw01.zip"
REBATE_MEMBERS = [
    "intrvw01/intrvw01/Addendum - Tax Rebate/TAX013.csv",
    "intrvw01/intrvw01/Addendum - Tax Rebate/TAX014.csv",
]
OUTPUT_PATH = PROJECT_ROOT / "data/householdbench/tabular/cons_cex_rebate01.csv"
DIAGNOSTICS_DIR = PROJECT_ROOT / "output" / "householdbench" / "tasks" / TASK_ID
SAMPLING_DIR = PROJECT_ROOT / "data" / "intermediate" / "householdbench" / TASK_ID
STEP_DIAGNOSTICS_PATH = DIAGNOSTICS_DIR / "01_sample_construction_steps.csv"
DROP_DETAIL_PATH = DIAGNOSTICS_DIR / "01_sample_construction_drop_details.csv"
OUTLIER_CUTOFF_PATH = DIAGNOSTICS_DIR / "01_outlier_cutoffs.csv"
ASSIGNMENT_DETAIL_PATH = DIAGNOSTICS_DIR / "01_rebate_assignment_details.csv"
STATUS_SUMMARY_PATH = DIAGNOSTICS_DIR / "01_rebate_status_summary.csv"
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
    "rebate_amount_total_lag{lag}",
]
POLICY_COLUMNS = [
    "rebate_amount_total",
    "rebate_record_count",
    "rebate_month_unique_count",
    "rebate_month_first",
    "rebate_month_last",
    "rebate_payment_received",
    "rebate_status",
    "jps_repair_status",
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
status_summary_rows = []


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

# Official BLS Interview PUMD flags: A is a valid blank/not anticipated;
# B/C are invalid nonresponse; D/E/F/G and T/U/V/W are valid values.
VALID_VALUE_FLAGS = {"D", "E", "F", "G", "T", "U", "V", "W"}
raw_parts = []
with zipfile.ZipFile(REBATE_ZIP_PATH) as archive:
    for member in REBATE_MEMBERS:
        with archive.open(member) as handle:
            part = pd.read_csv(handle, dtype="string")
        part.columns = [
            str(column).strip().strip('"').upper() for column in part.columns
        ]
        part["source_file"] = Path(member).name
        part["raw_addendum_row"] = range(2, len(part) + 2)
        raw_parts.append(part)
raw_reports = pd.concat(raw_parts, ignore_index=True)
raw_input_columns = [
    "NEWID",
    "TAXQ1",
    "TAXQ1_",
    "TAXQ3",
    "TAXQ3_",
    *[
        column
        for check_index in range(1, 6)
        for column in [
            f"TAXMO{check_index}",
            f"TAXMO{check_index}_",
            f"TAXAMT{check_index}",
            f"TAXAMT{check_index}_",
        ]
    ],
]
missing_raw_columns = [
    column for column in raw_input_columns if column not in raw_reports
]
if missing_raw_columns:
    raise SystemExit(
        f"{REBATE_ZIP_PATH} is missing required columns: {missing_raw_columns}"
    )
raw_reports["reported_newid"] = (
    pd.to_numeric(raw_reports["NEWID"], errors="coerce")
    .astype("Int64")
    .astype("string")
    .str.zfill(8)
)
if (
    raw_reports["reported_newid"].isna().any()
    or raw_reports["reported_newid"].duplicated().any()
):
    raise SystemExit("Tax-rebate module NEWIDs must be unique and nonmissing.")

reported_context = cex.loc[
    cex["newid"].isin(raw_reports["reported_newid"]),
    [
        "cex_row_id",
        "newid",
        "cu_id",
        "cex_panel_id",
        "interview_year",
        "interview_month",
        "interview_month_index",
    ],
].rename(
    columns={
        "cex_row_id": "reported_cex_row_id",
        "newid": "reported_newid",
        "interview_year": "reported_interview_year",
        "interview_month": "reported_interview_month",
        "interview_month_index": "reported_interview_month_index",
    }
)
if reported_context["reported_newid"].duplicated().any():
    raise SystemExit("Raw tax-rebate NEWIDs match multiple cleaned CEX interviews.")
raw_reports = raw_reports.merge(
    reported_context, on="reported_newid", how="left", validate="one_to_one"
)
if raw_reports["cex_panel_id"].isna().any():
    missing = (
        raw_reports.loc[raw_reports["cex_panel_id"].isna(), "reported_newid"].head(10).tolist()
    )
    raise SystemExit(f"Raw tax-rebate NEWIDs do not match cleaned CEX rows: {missing}")
module_panels = set(raw_reports["cex_panel_id"].astype("string"))

for column in ["TAXQ1", "TAXQ3"]:
    raw_reports[column] = pd.to_numeric(raw_reports[column], errors="coerce")
for check_index in range(1, 6):
    for stem in ["TAXMO", "TAXAMT"]:
        raw_reports[f"{stem}{check_index}"] = pd.to_numeric(
            raw_reports[f"{stem}{check_index}"], errors="coerce"
        )
q1_flag = raw_reports["TAXQ1_"].fillna("").str.strip().str.upper()
raw_reports["lead_in_status"] = "unresolved_lead_in"
raw_reports.loc[
    q1_flag.isin(VALID_VALUE_FLAGS) & raw_reports["TAXQ1"].eq(1), "lead_in_status"
] = "valid_yes"
raw_reports.loc[
    q1_flag.isin(VALID_VALUE_FLAGS) & raw_reports["TAXQ1"].eq(2), "lead_in_status"
] = "valid_no"
q3_flag = raw_reports["TAXQ3_"].fillna("").str.strip().str.upper()
raw_reports["duplicate_report_status"] = "not_reported_or_unresolved"
raw_reports.loc[
    q3_flag.isin(VALID_VALUE_FLAGS) & raw_reports["TAXQ3"].eq(1),
    "duplicate_report_status",
] = "reported_in_section_22"
raw_reports.loc[
    q3_flag.isin(VALID_VALUE_FLAGS) & raw_reports["TAXQ3"].eq(2),
    "duplicate_report_status",
] = "not_reported_in_section_22"
# TAXQ3 records whether the rebate was also reported in the tax-refund section.
# Appendix B does not use it to invalidate rebate receipt, so retain it for
# diagnostics without changing the exposure or sample.

check_parts = []
for check_index in range(1, 6):
    check = raw_reports[
        [
            "source_file",
            "raw_addendum_row",
            "reported_cex_row_id",
            "reported_newid",
            "cex_panel_id",
            "reported_interview_year",
            "reported_interview_month",
            "reported_interview_month_index",
            "lead_in_status",
            "duplicate_report_status",
            "TAXQ1",
            "TAXQ1_",
            "TAXQ3",
            "TAXQ3_",
            f"TAXMO{check_index}",
            f"TAXMO{check_index}_",
            f"TAXAMT{check_index}",
            f"TAXAMT{check_index}_",
        ]
    ].copy()
    check = check.rename(
        columns={
            f"TAXMO{check_index}": "rebate_month",
            f"TAXMO{check_index}_": "rebate_month_flag",
            f"TAXAMT{check_index}": "rebate_amount",
            f"TAXAMT{check_index}_": "rebate_amount_flag",
        }
    )
    check["check_index"] = check_index
    check["raw_rebate_source_id"] = (
        check["source_file"]
        + ":"
        + check["raw_addendum_row"].astype("string")
        + ":"
        + check["check_index"].astype("string")
    )
    check_parts.append(check)
rebate_long = pd.concat(check_parts, ignore_index=True)
if rebate_long["raw_rebate_source_id"].duplicated().any():
    raise SystemExit("Raw tax-rebate source identifiers are not unique.")

month_flag = rebate_long["rebate_month_flag"].fillna("").str.strip().str.upper()
amount_flag = rebate_long["rebate_amount_flag"].fillna("").str.strip().str.upper()
rebate_long["month_valid"] = month_flag.isin(VALID_VALUE_FLAGS) & rebate_long[
    "rebate_month"
].between(1, 12)
rebate_long["amount_valid_positive"] = amount_flag.isin(
    VALID_VALUE_FLAGS
) & rebate_long["rebate_amount"].gt(0)
rebate_long["month_reported"] = rebate_long["rebate_month"].notna()
rebate_long["positive_amount_invalid_month"] = (
    rebate_long["amount_valid_positive"] & ~rebate_long["month_valid"]
)
rebate_long["month_invalid_amount"] = (
    rebate_long["month_valid"] & ~rebate_long["amount_valid_positive"]
)

report_checks = (
    rebate_long.groupby("reported_cex_row_id", sort=False)
    .agg(
        any_month_reported=("month_reported", "max"),
        positive_amount_invalid_month=("positive_amount_invalid_month", "max"),
        month_invalid_amount=("month_invalid_amount", "max"),
    )
    .reset_index()
)
raw_reports = raw_reports.merge(
    report_checks, on="reported_cex_row_id", how="left", validate="one_to_one"
)
raw_reports["report_status"] = "resolved_report"
raw_reports.loc[
    raw_reports["lead_in_status"].eq("unresolved_lead_in"), "report_status"
] = "unresolved_lead_in"
raw_reports.loc[
    raw_reports["lead_in_status"].eq("valid_no") & raw_reports["any_month_reported"],
    "report_status",
] = "contradictory_no_with_check"
raw_reports.loc[
    raw_reports["lead_in_status"].eq("valid_yes") & ~raw_reports["any_month_reported"],
    "report_status",
] = "contradictory_yes_without_check"
raw_reports.loc[raw_reports["month_invalid_amount"], "report_status"] = (
    "invalid_or_zero_amount"
)
raw_reports.loc[raw_reports["positive_amount_invalid_month"], "report_status"] = (
    "invalid_positive_month"
)
invalid_month_panels = set(
    raw_reports.loc[raw_reports["positive_amount_invalid_month"], "cex_panel_id"].astype(
        "string"
    )
)
resolved_report_rows = set(
    raw_reports.loc[
        raw_reports["report_status"].eq("resolved_report"),
        "reported_cex_row_id",
    ]
)

valid_checks = rebate_long.loc[
    rebate_long["month_valid"]
    & rebate_long["amount_valid_positive"]
    & rebate_long["reported_cex_row_id"].isin(resolved_report_rows)
].copy()
valid_checks["payment_year"] = valid_checks["reported_interview_year"].astype(int)
valid_checks.loc[
    valid_checks["rebate_month"].gt(valid_checks["reported_interview_month"]),
    "payment_year",
] -= 1
valid_checks["payment_month"] = valid_checks["rebate_month"].astype(int)
valid_checks["payment_month_index"] = (
    valid_checks["payment_year"] * 12 + valid_checks["payment_month"]
)

candidate_parts = []
for offset in [1, 2, 3]:
    candidate = cex[["cex_row_id", "newid", "cex_panel_id", "interview_month_index"]].copy()
    candidate["payment_month_index"] = candidate["interview_month_index"] - offset
    candidate["assignment_offset_months"] = offset
    candidate_parts.append(candidate)
candidates = pd.concat(candidate_parts, ignore_index=True).rename(
    columns={
        "cex_row_id": "assigned_cex_row_id",
        "newid": "assigned_newid",
        "interview_month_index": "assigned_interview_month_index",
    }
)
candidates = (
    candidates.sort_values(
        ["cex_panel_id", "payment_month_index", "assignment_offset_months", "assigned_cex_row_id"],
        kind="mergesort",
    )
    .drop_duplicates(["cex_panel_id", "payment_month_index"], keep="first")
)
assignment = valid_checks.merge(
    candidates[
        [
            "cex_panel_id",
            "payment_month_index",
            "assigned_cex_row_id",
            "assigned_newid",
            "assigned_interview_month_index",
        ]
    ],
    on=["cex_panel_id", "payment_month_index"],
    how="left",
    validate="many_to_one",
)
if len(assignment) != len(valid_checks):
    raise SystemExit("Rebate assignment dropped or duplicated valid checks.")
assigned = assignment["assigned_cex_row_id"].notna()
assignment["assignment_status"] = "unassigned_no_observed_period"
assignment.loc[
    assigned
    & assignment["assigned_interview_month_index"].lt(
        assignment["reported_interview_month_index"]
    ),
    "assignment_status",
] = "later_interview_fills_prior_period"
assignment.loc[
    assigned
    & assignment["assigned_interview_month_index"].gt(
        assignment["reported_interview_month_index"]
    ),
    "assignment_status",
] = "prior_interview_report_moved_forward"
assignment.loc[
    assigned & assignment["assigned_newid"].eq(assignment["reported_newid"]),
    "assignment_status",
] = "assigned_reported_newid"

# Appendix B: if a later interview reports a rebate in an earlier period whose
# interview explicitly said no, treat the later month as recall error and move
# exposure to the later interview's period if it otherwise has no rebate.
target_lead_in = assignment["assigned_cex_row_id"].map(
    raw_reports.set_index("reported_cex_row_id")["lead_in_status"]
)
reported_period_with_check = set(
    assignment.loc[
        assignment["assigned_cex_row_id"].eq(assignment["reported_cex_row_id"]),
        "reported_cex_row_id",
    ]
)
recall_shift = (
    assigned
    & assignment["assigned_newid"].ne(assignment["reported_newid"])
    & target_lead_in.eq("valid_no")
    & ~assignment["reported_cex_row_id"].isin(reported_period_with_check)
)
assignment.loc[recall_shift, "assigned_cex_row_id"] = assignment.loc[
    recall_shift, "reported_cex_row_id"
]
assignment.loc[recall_shift, "assigned_newid"] = assignment.loc[
    recall_shift, "reported_newid"
]
assignment.loc[recall_shift, "assignment_status"] = (
    "later_interview_recall_shifted_forward"
)
assigned = assignment["assigned_cex_row_id"].notna()

raw_amount = float(assignment["rebate_amount"].sum())
assigned_amount = float(assignment.loc[assigned, "rebate_amount"].sum())
unassigned_amount = float(assignment.loc[~assigned, "rebate_amount"].sum())
if abs(raw_amount - assigned_amount - unassigned_amount) > 1e-6:
    raise SystemExit("Assigned and unassigned rebate amounts do not conserve.")

payment_aggregate = (
    assignment.loc[assigned]
    .groupby("assigned_cex_row_id", sort=False)
    .agg(
        rebate_amount_total=("rebate_amount", "sum"),
        rebate_record_count=("rebate_amount", "size"),
        rebate_month_unique_count=("payment_month", "nunique"),
        rebate_month_first=("payment_month", "min"),
        rebate_month_last=("payment_month", "max"),
    )
    .reset_index()
    .rename(columns={"assigned_cex_row_id": "cex_row_id"})
)
cex = cex.merge(payment_aggregate, on="cex_row_id", how="left", validate="one_to_one")
cex = cex.merge(
    raw_reports[["reported_cex_row_id", "lead_in_status", "report_status"]].rename(
        columns={"reported_cex_row_id": "cex_row_id"}
    ),
    on="cex_row_id",
    how="left",
    validate="one_to_one",
)

cex["rebate_status"] = "unresolved_no_module_data"
cex["jps_repair_status"] = "unresolved"
positive = cex["rebate_amount_total"].gt(0).fillna(False)
cex.loc[positive, ["rebate_status", "jps_repair_status"]] = ["valid_positive", "direct"]
later_fill_rows = set(
    assignment.loc[
        assignment["assignment_status"].eq("later_interview_fills_prior_period"),
        "assigned_cex_row_id",
    ]
)
prior_forward_rows = set(
    assignment.loc[
        assignment["assignment_status"].eq("prior_interview_report_moved_forward"),
        "assigned_cex_row_id",
    ]
)
recall_rows = set(
    assignment.loc[
        assignment["assignment_status"].eq("later_interview_recall_shifted_forward"),
        "assigned_cex_row_id",
    ]
)
cex.loc[cex["cex_row_id"].isin(later_fill_rows), "jps_repair_status"] = (
    "later_interview_fills_prior_no_data"
)
cex.loc[cex["cex_row_id"].isin(prior_forward_rows), "jps_repair_status"] = (
    "prior_interview_report_moved_forward"
)
cex.loc[cex["cex_row_id"].isin(recall_rows), "jps_repair_status"] = (
    "later_interview_recall_shifted_forward"
)

raw_resolved_zero = (
    cex["lead_in_status"].isin(["valid_no", "valid_yes"])
    & cex["report_status"].eq("resolved_report")
    & ~positive
)
cex.loc[raw_resolved_zero, ["rebate_status", "jps_repair_status"]] = [
    "valid_zero",
    "direct",
]
reference_period_end = cex["interview_month_index"] - 1
reference_period_start = cex["interview_month_index"] - 3
known_zero = (
    cex["cex_panel_id"].isin(module_panels)
    & (
        reference_period_end.le(2001 * 12 + 6)
        | reference_period_start.ge(2001 * 12 + 10)
    )
    & ~positive
)
cex.loc[known_zero, ["rebate_status", "jps_repair_status"]] = [
    "valid_zero",
    "known_out_of_field_zero",
]
row_invalid = cex["report_status"].notna() & cex["report_status"].ne("resolved_report")
cex.loc[row_invalid, "rebate_status"] = cex.loc[row_invalid, "report_status"]
cex.loc[row_invalid, "jps_repair_status"] = "unresolved"
invalid_household = cex["cex_panel_id"].isin(invalid_month_panels)
cex.loc[invalid_household, ["rebate_status", "jps_repair_status"]] = [
    "invalid_positive_month_household",
    "unresolved",
]
resolved = cex["rebate_status"].isin(["valid_positive", "valid_zero"])
cex.loc[~resolved, "rebate_amount_total"] = pd.NA
cex.loc[cex["rebate_status"].eq("valid_zero"), "rebate_amount_total"] = 0.0
cex["rebate_record_count"] = cex["rebate_record_count"].fillna(0).astype(int)
cex["rebate_month_unique_count"] = (
    cex["rebate_month_unique_count"].fillna(0).astype(int)
)
cex["rebate_payment_received"] = (
    cex["rebate_amount_total"].gt(0).fillna(False).astype(int)
)

assignment["boundary_drop_reason"] = ""
assignment.loc[~assigned, "boundary_drop_reason"] = (
    "payment_month_outside_observed_expenditure_windows"
)
assignment_detail_columns = [
    "raw_rebate_source_id",
    "source_file",
    "raw_addendum_row",
    "check_index",
    "reported_newid",
    "cex_panel_id",
    "reported_interview_year",
    "reported_interview_month",
    "lead_in_status",
    "duplicate_report_status",
    "TAXQ1",
    "TAXQ1_",
    "TAXQ3",
    "TAXQ3_",
    "payment_year",
    "payment_month",
    "payment_month_index",
    "rebate_amount",
    "rebate_month_flag",
    "rebate_amount_flag",
    "assigned_newid",
    "assignment_status",
    "boundary_drop_reason",
]
status_summary_rows.extend(
    [
        {"metric": "raw_module_rows", "value": int(len(raw_reports))},
        {"metric": "raw_module_panels", "value": int(len(module_panels))},
        {"metric": "valid_check_rows", "value": int(len(assignment))},
        {"metric": "valid_check_amount", "value": raw_amount},
        {"metric": "assigned_check_rows", "value": int(assigned.sum())},
        {"metric": "assigned_check_amount", "value": assigned_amount},
        {"metric": "unassigned_check_rows", "value": int((~assigned).sum())},
        {"metric": "unassigned_check_amount", "value": unassigned_amount},
        {"metric": "invalid_positive_month_panels", "value": int(len(invalid_month_panels))},
    ]
)
for status, count in raw_reports["report_status"].value_counts().sort_index().items():
    status_summary_rows.append(
        {"metric": f"raw_report_status_{status}", "value": int(count)}
    )
for status, count in (
    raw_reports["duplicate_report_status"].value_counts().sort_index().items()
):
    status_summary_rows.append(
        {"metric": f"raw_duplicate_report_status_{status}", "value": int(count)}
    )
for status, count in (
    assignment["assignment_status"].value_counts().sort_index().items()
):
    status_summary_rows.append(
        {"metric": f"assignment_status_{status}", "value": int(count)}
    )
if len(cex) != rows_before:
    raise SystemExit(
        f"{TASK_ID}: additional-source attachment changed the CEX row count."
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
        f"{column}_lag{lag}"
        for column in LAGGED_HOUSEHOLD_COLUMNS
    ]
    cex.loc[~usable_lag_masks[lag], lag_columns] = pd.NA
step_rows.append(
    construction_step(
        TASK_ID, 3, "Construct leads and lags",
        int(rows_before), int(len(cex)), int(rows_before) - int(len(cex)),
    )
)


# 4. Restrict the JPS episode universe without conditioning on receipt.
rows_before = len(cex)
history_start = 2001 * 12 + 1
history_end = 2002 * 12 + 3
date_mask = cex["interview_month_index"].between(history_start, history_end)
fielding_cu_mask = cex["cex_panel_id"].isin(module_panels)
resolved_status_mask = cex["rebate_status"].isin(["valid_positive", "valid_zero"])
for detail_order, filter_id, pass_condition, pass_mask in [
    (
        1,
        "interview_in_paper_window",
        'cex["interview_month_index"].between(2001 * 12 + 1, 2002 * 12 + 3)',
        date_mask,
    ),
    (
        2,
        "cu_observed_in_rebate_module",
        'cex["cex_panel_id"].isin(module_panels)',
        fielding_cu_mask,
    ),
    (
        3,
        "rebate_status_resolved",
        'cex["rebate_status"].isin(["valid_positive", "valid_zero"])',
        resolved_status_mask,
    ),
]:
    drop_detail_rows.append(
        filter_count(
            TASK_ID, 4, detail_order,
            filter_id, pass_condition, int((~pass_mask).sum()),
        )
    )
event = cex.loc[date_mask & fielding_cu_mask & resolved_status_mask].copy()
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
for variable in ["income_before_tax", "cons_parker_total"]:
    value, p01, p99 = variable, f"{variable}_p01", f"{variable}_p99"
    if pool[[p01, p99]].isna().any().any():
        raise SystemExit(f"{TASK_ID}: missing current {variable} cutoffs entering step 8.")
    pass_mask = pool[value].gt(pool[p01]) & pool[value].lt(pool[p99])
    trim_masks.append(pass_mask)
    drop_detail_rows.append(filter_count(
                                TASK_ID, 8, detail_order,
                                f"trim_{variable}_current_strict_inside_period_p01_p99", f'pool["{value}"].gt(pool["{p01}"]) & pool["{value}"].lt(pool["{p99}"])', int((~pass_mask).sum()),
                            ))
    detail_order += 1
for lag in [1, 2, 3]:
    for variable in ["cons_parker_total"]:
        value, p01, p99 = f"{variable}_lag{lag}", f"{variable}_p01_lag{lag}", f"{variable}_p99_lag{lag}"
        observed = pool[value].notna()
        if pool.loc[observed, [p01, p99]].isna().any().any():
            raise SystemExit(f"{TASK_ID}: missing {variable} cutoffs for observed lag {lag} entering step 8.")
        pass_mask = (~observed) | (pool[value].gt(pool[p01]) & pool[value].lt(pool[p99]))
        trim_masks.append(pass_mask)
        drop_detail_rows.append(filter_count(
                                    TASK_ID, 8, detail_order,
                                    f"trim_{variable}_lag{lag}_strict_inside_period_p01_p99", f'pool["{value}"].isna() | (pool["{value}"].gt(pool["{p01}"]) & pool["{value}"].lt(pool["{p99}"]))', int((~pass_mask).sum()),
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

status_summary_rows.extend(
    [
        {"metric": "final_rows", "value": int(len(table))},
        {"metric": "final_panels", "value": int(table["cex_panel_id"].nunique())},
        {
            "metric": "final_positive_rebate_rows",
            "value": int(table["rebate_payment_received"].sum()),
        },
        {
            "metric": "final_zero_rebate_rows",
            "value": int(table["rebate_payment_received"].eq(0).sum()),
        },
    ]
)
for lag in [1, 2, 3]:
    lag_label = "lag" if lag == 1 else "lags"
    status_summary_rows.append(
        {
            "metric": f"final_rows_with_{lag}_history_{lag_label}",
            "value": int(table[f"interview_month_index_lag{lag}"].notna().sum()),
        }
    )
for status, count in cex["rebate_status"].value_counts().sort_index().items():
    status_summary_rows.append(
        {"metric": f"cex_rebate_status_{status}", "value": int(count)}
    )

DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(step_rows, columns=STEP_COLUMNS).to_csv(STEP_DIAGNOSTICS_PATH, index=False)
pd.DataFrame(drop_detail_rows, columns=DROP_DETAIL_COLUMNS).to_csv(
    DROP_DETAIL_PATH, index=False
)
pd.concat(cutoff_rows, ignore_index=True).loc[:, ["task_id", "variable", "role", "period", "n_reference", "p01", "p99", "n_at_or_below_p01", "n_at_or_above_p99"]].to_csv(OUTLIER_CUTOFF_PATH, index=False)
assignment.loc[:, assignment_detail_columns].to_csv(ASSIGNMENT_DETAIL_PATH, index=False)
pd.DataFrame(status_summary_rows, columns=["metric", "value"]).to_csv(
    STATUS_SUMMARY_PATH, index=False
)

print(
    f"{TASK_ID}: wrote {len(table):,} rows to {OUTPUT_PATH} "
    f"(eligible={len(pool):,}, "
    f"diagnostics={DIAGNOSTICS_DIR})"
)
