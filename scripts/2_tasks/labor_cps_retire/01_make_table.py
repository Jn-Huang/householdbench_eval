#!/usr/bin/env python
"""Build the tabular HouseholdBench CPS retirement task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd


TASK_ID = "cps_older_employed_12m_lf_status"
TASK_SLUG = "labor_cps_retire"
ROW_ID_PREFIX = "3c_cps_older_employed_12m_lf_status"
SOURCE_STATUSES = {"employed", "unemployed", "nilf_retired", "nilf_disabled", "nilf_other"}
TARGETS = ["retired", "not_retired"]
TWELVE_MONTH_BASELINE_MISH = {1, 2, 3, 4}
NOT_OBSERVED = "not observed"

REPO_ROOT = Path(__file__).resolve().parents[3]
BASIC_INPUT_PATH = REPO_ROOT / "data/intermediate/cps_basic.parquet"
LABELS_PATH = REPO_ROOT / "scripts/1_preprocessing/mappings/cps_retirement_labels.csv"
HOUSEHOLDBENCH_ROOT = REPO_ROOT / "data/householdbench"
DIAGNOSTICS_DIR = REPO_ROOT / "output" / "householdbench" / "tasks" / TASK_SLUG
SAMPLING_DIR = REPO_ROOT / "data" / "intermediate" / "householdbench" / TASK_SLUG
STEP_DIAGNOSTICS_PATH = DIAGNOSTICS_DIR / "01_sample_construction_steps.csv"
DROP_DETAIL_PATH = DIAGNOSTICS_DIR / "01_sample_construction_drop_details.csv"
OUTLIER_CUTOFFS_PATH = DIAGNOSTICS_DIR / "01_outlier_cutoffs.csv"
TARGET_AVAILABILITY_PATH = DIAGNOSTICS_DIR / "01_target_availability_diagnostics.csv"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.sample_accounting import construction_step, filter_count

from scripts.utils.table_schema import canonicalize_public_table

from scripts.utils import cps
from scripts.utils.parallel import default_worker_count, map_in_order
from scripts.utils.sampling import sample_task_records


FIELDNAMES = [
    "row_id",
    "subject_id",
    "selection_period",
    "baseline_year",
    "baseline_month",
    "outcome_year",
    "outcome_month",
    "release_date",
    "release_date_source",
    "mish_baseline",
    "mish_target",
    "baseline_status",
    "target",
    "age",
    "sex",
    "race",
    "educ",
    "marst",
    "state",
    "region",
    "metro",
    "relate",
    "famsize",
    "nchild",
    "hispan",
    "nativity",
    "citizen",
    "vetstat",
    "classwkr",
    "difficulty",
    "family_income",
    "occupation",
    "industry",
    "difficulty_code",
    "family_income_code",
    "occupation_code",
    "industry_code",
    "partner_linked_baseline",
    "partner_age_baseline",
    "partner_status_baseline",
    *cps.CORE_MACRO_COLUMNS,
    "ahrsworkt_baseline",
    "uhrsworkt_baseline",
    "multjob_baseline",
    "union_baseline",
    "earnweek_baseline",
    "link_validation_status",
    "outcome_empstat",
    "outcome_nilfact",
]
REQUIRED_OUTPUT_COLUMNS = [
    "row_id",
    "subject_id",
    "selection_period",
    "baseline_year",
    "baseline_month",
    "outcome_year",
    "outcome_month",
    "release_date",
    "release_date_source",
    "mish_baseline",
    "mish_target",
    "baseline_status",
    "target",
    "age",
    "sex",
    "race",
    "educ",
    "marst",
    "state",
    "region",
    "metro",
    "relate",
    "famsize",
    "nchild",
    "hispan",
    "nativity",
    "citizen",
    "vetstat",
    "difficulty",
    "family_income",
    "occupation",
    "industry",
    "partner_linked_baseline",
    "partner_status_baseline",
    *cps.CORE_MACRO_COLUMNS,
    "ahrsworkt_baseline",
    "uhrsworkt_baseline",
    "link_validation_status",
    "outcome_empstat",
]
STEP_COLUMNS = ["task_id", "step", "step_label", "rows_before", "rows_after", "rows_dropped"]
DETAIL_COLUMNS = ["task_id", "step", "detail_order", "filter_id", "pass_condition", "fail_count"]
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
TARGET_AVAILABILITY_COLUMNS = [
    "task_id",
    "requirement_order",
    "requirement_id",
    "pass_condition",
    "rows_before",
    "rows_after",
    "rows_removed",
]
COMMON_BASELINE_PREDICTOR_FIELDS = [
    "age",
    "sex",
    "race",
    "educ",
    "marst",
    "state",
    "region",
    "metro",
    "relate",
    "famsize",
    "nchild",
    "hispan",
    "nativity",
    "citizen",
    "vetstat",
]
COMMON_MACRO_PREDICTOR_FIELDS = cps.CORE_MACRO_COLUMNS
TASK_BASELINE_PREDICTOR_FIELDS = ["ahrsworkt", "uhrsworkt"]
STEP4_DETAILS = [
    ("universe_baseline_rotation_eligible", 'baseline["MISH"].isin([1, 2, 3, 4])'),
    ("universe_baseline_year_1995_or_later", 'baseline["YEAR"].ge(1995)'),
    ("universe_age_at_least_50", 'baseline["AGE"].ge(50)'),
    ("universe_age_at_most_75", 'baseline["AGE"].le(75)'),
    ("universe_baseline_status_employed", 'baseline["EMPSTAT"].isin([10, 12])'),
    (
        "universe_target_mish_equals_baseline_mish_plus_4_when_linked",
        'target is unlinked or target["MISH"] == baseline["MISH"] + 4',
    ),
]
STEP5_DETAILS = [
    ("target_unique_same_cpsidp_record_available", "one unique same-CPSIDP record exists twelve months later"),
    ("target_status_supported", "linked target labor-force status is supported"),
]
STEP7_DETAILS = [("link_valid_same_person_demographics", "cps.validate_12m_link(baseline, target_obs) is not None")]
STEP6_DETAILS = (
    [(f"predictor_{field}_not_missing", f'baseline.get("{field}") is not None') for field in COMMON_BASELINE_PREDICTOR_FIELDS]
    + [(f"predictor_macro_{field}_not_missing", f'macro.get("{field}") is not None') for field in COMMON_MACRO_PREDICTOR_FIELDS]
    + [(f"predictor_{field}_not_missing", f'baseline.get("{field}") is not None') for field in TASK_BASELINE_PREDICTOR_FIELDS]
)
BASIC_COLUMNS = [
    "YEAR",
    "SERIAL",
    "MONTH",
    "PERNUM",
    "CPSIDP",
    "CPSIDV",
    "MISH",
    "SPLOC",
    "AGE",
    "EMPSTAT",
    "NILFACT",
    "SEX",
    "RACE",
    "EDUC",
    "MARST",
    "STATEFIP",
    "REGION",
    "METRO",
    "RELATE",
    "FAMSIZE",
    "NCHILD",
    "CITIZEN",
    "NATIVITY",
    "HISPAN",
    "VETSTAT",
    "CLASSWKR",
    "AHRSWORKT",
    "UHRSWORKT",
    "MULTJOB",
    "UNION",
    "EARNWEEK",
    "DIFFANY",
    "FAMINC",
    "OCC2010",
    "IND1990",
    "sex",
    "race",
    "educ",
    "marst",
    "state",
    "region",
    "metro",
    "relate",
    "hispan",
    "nativity",
    "citizen",
    "vetstat",
    "classwkr",
    "empstat",
    "nilfact",
    "multjob",
    "union",
    "release_date",
    "release_date_source",
]






# Official IPUMS labels and occupation groups, frozen from the extract codebook.
# The supplied map includes code 0400 in the documented 0010--0430 major group.
label_rows = pd.read_csv(LABELS_PATH, keep_default_na=False)
IPUMS_LABELS = {
    variable: dict(zip(rows["code"], rows["label"], strict=True))
    for variable, rows in label_rows.groupby("variable", sort=False)
}
OCC2010_GROUP_LABELS = IPUMS_LABELS.pop("OCC2010")
IPUMS_UNAVAILABLE_CODES = {
    "DIFFANY": {0},
    "FAMINC": {995, 996, 997, 999},
    "IND1990": {0, 998},
}


def ipums_label(variable: str, value: Any) -> str:
    code = cps.safe_int(value)
    if code is None or code in IPUMS_UNAVAILABLE_CODES[variable]:
        return NOT_OBSERVED
    label = IPUMS_LABELS[variable].get(code)
    if label is None:
        raise RuntimeError(f"{variable} code {code} has no official IPUMS label in {LABELS_PATH}")
    return label


def occupation_group(value: Any) -> str:
    code = cps.safe_int(value)
    if code is None or code == 9999:
        return NOT_OBSERVED
    label = OCC2010_GROUP_LABELS.get(code)
    if label is None:
        raise RuntimeError(f"OCC2010 code {code} has no official group in {LABELS_PATH}")
    if label.lower() == "not in universe":
        return NOT_OBSERVED
    return label


def attach_partner_context(raw_month: pd.DataFrame) -> pd.DataFrame:
    keys = ["YEAR", "MONTH", "SERIAL", "PERNUM"]
    duplicated = raw_month.duplicated(keys, keep=False)
    if bool(duplicated.any()):
        examples = raw_month.loc[duplicated, keys].head(10).to_dict("records")
        raise RuntimeError(f"CPS month has duplicate YEAR/MONTH/SERIAL/PERNUM keys: {examples}")

    partner_lookup = raw_month[
        ["YEAR", "MONTH", "SERIAL", "PERNUM", "CPSIDP", "AGE", "EMPSTAT", "NILFACT"]
    ].rename(
        columns={
            "PERNUM": "SPLOC",
            "CPSIDP": "partner_cpsidp_source",
            "AGE": "partner_age_source",
            "EMPSTAT": "partner_empstat_source",
            "NILFACT": "partner_nilfact_source",
        }
    )
    out = raw_month.merge(
        partner_lookup,
        on=["YEAR", "MONTH", "SERIAL", "SPLOC"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if len(out) != len(raw_month):
        raise RuntimeError("Partner attachment changed the CPS monthly row count.")

    sploc = pd.to_numeric(out["SPLOC"], errors="coerce")
    positive = sploc.gt(0)
    resolved = out["partner_cpsidp_source"].notna()
    unresolved = positive & ~resolved
    if bool(unresolved.any()):
        examples = out.loc[unresolved, ["YEAR", "MONTH", "SERIAL", "PERNUM", "SPLOC"]].head(10)
        raise RuntimeError(f"Positive SPLOC values did not resolve uniquely:\n{examples.to_string(index=False)}")

    out["partner_linked_baseline"] = "no linked spouse or partner"
    out.loc[positive, "partner_linked_baseline"] = "linked"
    out["partner_age_baseline"] = pd.to_numeric(out["partner_age_source"], errors="coerce")
    out.loc[~positive, "partner_age_baseline"] = pd.NA

    partner_status = []
    for is_positive, empstat, nilfact in zip(
        positive,
        out["partner_empstat_source"],
        out["partner_nilfact_source"],
        strict=True,
    ):
        if not is_positive:
            partner_status.append("no linked spouse or partner")
            continue
        status = cps.older_status_from_empstat_nilfact(empstat, nilfact)
        partner_status.append(
            {
                "employed": "employed",
                "unemployed": "unemployed",
                "nilf_retired": "retired",
                "nilf_disabled": "disabled",
                "nilf_other": "other not in the labor force",
            }.get(status, NOT_OBSERVED)
        )
    out["partner_status_baseline"] = partner_status
    return out


def prepare_link_month(raw_month: pd.DataFrame) -> pd.DataFrame:
    out = raw_month.copy()
    cpsidp = pd.to_numeric(out["CPSIDP"], errors="coerce")
    cpsidp_round = cpsidp.round()
    # IPUMS uses zero when CPSIDP is unavailable in early CPS files.  Zero is
    # therefore not a person identifier and cannot support a longitudinal link.
    out["_valid_cpsidp"] = (
        cpsidp.notna()
        & cpsidp_round.gt(0)
        & ((cpsidp - cpsidp_round).abs() < 1e-6)
    )
    out["_cpsidp"] = cpsidp_round.astype("Int64")
    for column in ["YEAR", "MONTH", "MISH", "AGE", "EMPSTAT", "NILFACT"]:
        out[f"_{column.lower()}"] = pd.to_numeric(out[column], errors="coerce")
    out["_status"] = None
    out.loc[out["_empstat"].isin([10, 12]), "_status"] = "employed"
    out.loc[out["_empstat"].isin([20, 21, 22]), "_status"] = "unemployed"
    out.loc[out["_empstat"].isin([30, 31, 32, 33, 34, 35, 36]), "_status"] = "not_in_labor_force"
    out["_older_status"] = None
    out.loc[out["_empstat"].isin([10, 12]), "_older_status"] = "employed"
    out.loc[out["_empstat"].isin([20, 21, 22]), "_older_status"] = "unemployed"
    out.loc[out["_empstat"].eq(36), "_older_status"] = "nilf_retired"
    out.loc[out["_empstat"].eq(32) | out["_nilfact"].eq(1), "_older_status"] = "nilf_disabled"
    out.loc[out["_empstat"].isin([30, 31, 33, 34, 35]), "_older_status"] = "nilf_other"
    return out


def build_task_row(payload: dict[str, Any], macro: dict[str, float | None]) -> dict[str, Any]:
    baseline = cps.normalize_basic_record(payload, prefix="b_")
    target_obs = cps.normalize_basic_record(payload, prefix="t_")
    if baseline is None or target_obs is None:
        raise RuntimeError("A linked CPS record failed normalization after target filtering.")
    source_status = target_obs.get("older_status")
    if source_status not in SOURCE_STATUSES:
        raise RuntimeError(f"Unexpected supported target status: {source_status!r}")
    answer = "retired" if source_status == "nilf_retired" else "not_retired"
    if answer not in {"retired", "not_retired"}:
        raise RuntimeError(f"Unexpected retirement target label: {answer!r}")
    link_status = cps.validate_12m_link(baseline, target_obs)
    if link_status is None:
        raise RuntimeError("build_task_row called before the link-quality filter.")
    row_id = cps.stable_row_id(
        TASK_ID,
        ROW_ID_PREFIX,
        baseline["cpsidp"],
        baseline["year"],
        baseline["month"],
        target_obs["year"],
        target_obs["month"],
    )
    row = cps.base_table_row(FIELDNAMES, baseline, target_obs, macro, row_id, answer)
    difficulty_code = cps.safe_int(payload.get("b_DIFFANY"))
    family_income_code = cps.safe_int(payload.get("b_FAMINC"))
    occupation_code = cps.safe_int(payload.get("b_OCC2010"))
    industry_code = cps.safe_int(payload.get("b_IND1990"))
    row.update(
        {
            "selection_period": f"{int(target_obs['year']):04d}-{int(target_obs['month']):02d}",
            "difficulty": ipums_label("DIFFANY", difficulty_code),
            "family_income": ipums_label("FAMINC", family_income_code),
            "occupation": occupation_group(occupation_code),
            "industry": ipums_label("IND1990", industry_code),
            "difficulty_code": difficulty_code,
            "family_income_code": family_income_code,
            "occupation_code": occupation_code,
            "industry_code": industry_code,
            "partner_linked_baseline": payload.get("b_partner_linked_baseline"),
            "partner_age_baseline": cps.safe_int(payload.get("b_partner_age_baseline")),
            "partner_status_baseline": payload.get("b_partner_status_baseline"),
            "ahrsworkt_baseline": baseline.get("ahrsworkt"),
            "uhrsworkt_baseline": baseline.get("uhrsworkt"),
            "multjob_baseline": baseline.get("multjob"),
            "union_baseline": baseline.get("union"),
            "earnweek_baseline": baseline.get("earnweek"),
            "link_validation_status": link_status,
            "outcome_empstat": target_obs.get("empstat"),
            "outcome_nilfact": target_obs.get("nilfact"),
        }
    )
    cps.validate_table_row(row, REQUIRED_OUTPUT_COLUMNS, TARGETS)
    return row


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--jobs", type=int, default=default_worker_count(),
    help="Worker processes that link the months (default: available CPUs, at most 8).",
)
args = parser.parse_args()

destination = HOUSEHOLDBENCH_ROOT / "tabular" / f"{TASK_SLUG}.csv"

cps.check_columns(BASIC_INPUT_PATH, BASIC_COLUMNS)
cps.guard_parquet_chronology(BASIC_INPUT_PATH)
macro_map = cps.load_macro_context_map()

source_rows = 0
status_universe = 0
target_observed_rows = 0
predictor_observed_rows = 0
link_valid_rows = 0
eligible_rows = 0
trim_fail_count = 0
linked_rows = 0
selected_rows: list[dict[str, object]] = []
outlier_cutoff_rows = []
step4_fail_counts = {filter_id: 0 for filter_id, _ in STEP4_DETAILS}
step6_fail_counts = {filter_id: 0 for filter_id, _ in STEP6_DETAILS}
step7_fail_counts = {filter_id: 0 for filter_id, _ in STEP7_DETAILS}
step5_counts = {
    filter_id: {"rows_before": 0, "rows_after": 0}
    for filter_id, _pass_condition in STEP5_DETAILS
}

def build_month(raw_month_df: pd.DataFrame, target_month: pd.DataFrame | None) -> dict[str, Any]:
    """Link one baseline month to the month a year later and build its eligible rows.

    The counters are this month's; the main loop adds them to the task totals.
    """
    source_rows = int(len(raw_month_df))
    status_universe = 0
    target_observed_rows = 0
    predictor_observed_rows = 0
    link_valid_rows = 0
    trim_fail_count = 0
    linked_rows = 0
    outlier_cutoff_rows = []
    step4_fail_counts = {filter_id: 0 for filter_id, _ in STEP4_DETAILS}
    step6_fail_counts = {filter_id: 0 for filter_id, _ in STEP6_DETAILS}
    step7_fail_counts = {filter_id: 0 for filter_id, _ in STEP7_DETAILS}
    step5_counts = {
        filter_id: {"rows_before": 0, "rows_after": 0}
        for filter_id, _pass_condition in STEP5_DETAILS
    }

    reference_cpsidp = pd.to_numeric(raw_month_df["CPSIDP"], errors="coerce")
    reference_cpsidp_rounded = reference_cpsidp.round()
    reference_year = pd.to_numeric(raw_month_df["YEAR"], errors="coerce")
    reference_month = pd.to_numeric(raw_month_df["MONTH"], errors="coerce")
    valid_reference_key = (
        reference_cpsidp.notna()
        & reference_cpsidp_rounded.gt(0)
        & reference_cpsidp.sub(reference_cpsidp_rounded).abs().lt(1e-6)
        & reference_year.notna()
        & reference_month.notna()
    )
    reference_keys = pd.DataFrame(
        {
            "cpsidp": reference_cpsidp_rounded.loc[valid_reference_key].astype("int64"),
            "year": reference_year.loc[valid_reference_key].round().astype("int64"),
            "month": reference_month.loc[valid_reference_key].round().astype("int64"),
        }
    )
    duplicate_reference = reference_keys.duplicated(["cpsidp", "year", "month"], keep=False)
    if duplicate_reference.any():
        examples = reference_keys.loc[duplicate_reference].head(10).to_dict("records")
        raise RuntimeError(f"{TASK_SLUG}: CPS earnings reference is not unique by person-month: {examples}")
    periods = reference_keys[["year", "month"]].drop_duplicates()
    if len(periods) != 1:
        raise RuntimeError(f"{TASK_SLUG}: streamed source chunk does not contain exactly one CPS month.")
    reference_earnweek = pd.to_numeric(raw_month_df["EARNWEEK"], errors="coerce")
    valid_earnweek = valid_reference_key & reference_earnweek.between(0, 9999, inclusive="both")
    earnings_values = reference_earnweek.loc[valid_earnweek]
    period = periods.iloc[0]
    period_text = f"{int(period['year']):04d}-{int(period['month']):02d}"
    if earnings_values.empty:
        earnweek_p01 = None
        earnweek_p99 = None
    else:
        earnweek_p01 = float(earnings_values.quantile(0.01))
        earnweek_p99 = float(earnings_values.quantile(0.99))
        if earnweek_p01 > earnweek_p99:
            raise RuntimeError(f"{TASK_SLUG}: invalid weekly-earnings cutoff order in {period_text}.")
        if earnweek_p01 == earnweek_p99:
            raise RuntimeError(
                f"{TASK_SLUG}: degenerate weekly-earnings cutoffs in {period_text}; "
                f"n={len(earnings_values):,}, common_value={earnweek_p01}."
            )
        outlier_cutoff_rows.append(
            {
                "task_id": TASK_SLUG,
                "variable": "earnweek_baseline",
                "role": "predictor",
                "period": period_text,
                "n_reference": int(len(earnings_values)),
                "p01": earnweek_p01,
                "p99": earnweek_p99,
                "n_at_or_below_p01": int(earnings_values.le(earnweek_p01).sum()),
                "n_at_or_above_p99": int(earnings_values.ge(earnweek_p99).sum()),
            }
        )

    baseline_month = prepare_link_month(attach_partner_context(raw_month_df))

    # Step 3 is deliberately row-preserving: every baseline row remains, with
    # twelve-month fields attached where one unique valid person identifier permits it.
    if target_month is None:
        target_lookup = baseline_month.iloc[0:0].copy()
    else:
        target_lookup = target_month.loc[target_month["_valid_cpsidp"]].copy()
        target_lookup = target_lookup.loc[~target_lookup.duplicated("_cpsidp", keep=False)]
    joined = baseline_month.add_prefix("b_").merge(
        target_lookup.add_prefix("t_"),
        left_on="b__cpsidp",
        right_on="t__cpsidp",
        how="left",
        validate="many_to_one",
    )
    if len(joined) != len(baseline_month):
        raise RuntimeError(f"{TASK_SLUG}: step-3 link attachment changed the baseline row count.")
    joined["_baseline_unique_cpsidp"] = (
        joined["b__valid_cpsidp"] & ~joined.duplicated("b__cpsidp", keep=False)
    )

    rotation_pass = joined["b__mish"].isin(TWELVE_MONTH_BASELINE_MISH)
    year_pass = joined["b__year"].ge(1995)
    age_min_pass = joined["b__age"].ge(50)
    age_max_pass = joined["b__age"].le(75)
    status_pass = joined["b__status"].eq("employed")
    mish_pass = joined["t__cpsidp"].isna() | joined["t__mish"].eq(joined["b__mish"] + 4)
    step4_masks = [rotation_pass, year_pass, age_min_pass, age_max_pass, status_pass, mish_pass]
    step4_prior = pd.Series(True, index=joined.index)
    for (filter_id, _condition), pass_mask in zip(STEP4_DETAILS, step4_masks, strict=True):
        step4_fail_counts[filter_id] += int((step4_prior & ~pass_mask).sum())
        step4_prior &= pass_mask
    universe = joined.loc[step4_prior].copy()
    status_universe += int(len(universe))

    target_available = universe["_baseline_unique_cpsidp"] & universe["t__cpsidp"].notna()
    step5_counts["target_unique_same_cpsidp_record_available"]["rows_before"] += int(len(universe))
    step5_counts["target_unique_same_cpsidp_record_available"]["rows_after"] += int(target_available.sum())
    linked = universe.loc[target_available].copy()
    linked_rows += int(len(linked))

    target_supported_mask = linked["t__older_status"].isin(SOURCE_STATUSES)
    step5_counts["target_status_supported"]["rows_before"] += int(len(linked))
    step5_counts["target_status_supported"]["rows_after"] += int(target_supported_mask.sum())
    target_supported = linked.loc[target_supported_mask].copy()
    target_observed_rows += int(len(target_supported))

    rows: list[dict[str, Any]] = []
    for payload in target_supported.to_dict("records"):
        baseline = cps.normalize_basic_record(payload, prefix="b_")
        target_obs = cps.normalize_basic_record(payload, prefix="t_")
        if baseline is None or target_obs is None:
            raise RuntimeError("Supported target row failed CPS normalization.")
        macro = macro_map.get(cps.baseline_macro_key(baseline), {})
        predictor_complete = True
        for field in COMMON_BASELINE_PREDICTOR_FIELDS:
            if baseline.get(field) is None:
                step6_fail_counts[f"predictor_{field}_not_missing"] += 1
                predictor_complete = False
        for field in COMMON_MACRO_PREDICTOR_FIELDS:
            if macro.get(field) is None:
                step6_fail_counts[f"predictor_macro_{field}_not_missing"] += 1
                predictor_complete = False
        for field in TASK_BASELINE_PREDICTOR_FIELDS:
            if baseline.get(field) is None:
                step6_fail_counts[f"predictor_{field}_not_missing"] += 1
                predictor_complete = False
        if not predictor_complete:
            continue
        predictor_observed_rows += 1
        link_valid = cps.validate_12m_link(baseline, target_obs) is not None
        step7_fail_counts["link_valid_same_person_demographics"] += int(not link_valid)
        if not link_valid:
            continue
        link_valid_rows += 1
        row = build_task_row(payload, macro)
        earnings = row["earnweek_baseline"]
        if earnings is not None:
            if earnweek_p01 is None or earnweek_p99 is None:
                raise RuntimeError(f"{TASK_SLUG}: observed weekly earnings lack cutoffs in {period_text}.")
            if not (earnings > earnweek_p01 and earnings < earnweek_p99):
                trim_fail_count += 1
                continue
        rows.append(row)

    return {
        "source_rows": source_rows,
        "status_universe": status_universe,
        "target_observed_rows": target_observed_rows,
        "predictor_observed_rows": predictor_observed_rows,
        "link_valid_rows": link_valid_rows,
        "trim_fail_count": trim_fail_count,
        "linked_rows": linked_rows,
        "outlier_cutoff_rows": outlier_cutoff_rows,
        "step4_fail_counts": step4_fail_counts,
        "step5_counts": step5_counts,
        "step6_fail_counts": step6_fail_counts,
        "step7_fail_counts": step7_fail_counts,
        "rows": rows,
        "baseline_month": baseline_month,
    }


def build_month_block(baseline_months: list[int]) -> list[dict[str, Any]]:
    """Build consecutive baseline months, newest first, as the reverse month scan does."""
    targets = [month_idx + 12 for month_idx in baseline_months if month_idx + 12 in month_row_groups]
    months = sorted(set(baseline_months) | set(targets), reverse=True)
    prepared: dict[int, pd.DataFrame] = {}
    results = []
    for month_idx, raw_month_df in cps.iter_month_block(
        BASIC_INPUT_PATH, BASIC_COLUMNS, months, month_row_groups, reverse=True
    ):
        if month_idx in baseline_months:
            result = build_month(raw_month_df, prepared.get(month_idx + 12))
            prepared[month_idx] = result.pop("baseline_month")
            results.append(result)
        else:
            prepared[month_idx] = prepare_link_month(attach_partner_context(raw_month_df))
        for cached_month_idx in [cached for cached in prepared if cached > month_idx + 12]:
            del prepared[cached_month_idx]
    return results


# Worker processes build blocks of months; the results come back in the reverse
# month order of a single scan, so rows and diagnostics keep their order.
month_row_groups = cps.month_row_groups(BASIC_INPUT_PATH)
for block_results in map_in_order(
    build_month_block, cps.month_blocks(sorted(month_row_groups, reverse=True)), jobs=args.jobs
):
    for month in block_results:
        source_rows += month["source_rows"]
        status_universe += month["status_universe"]
        target_observed_rows += month["target_observed_rows"]
        predictor_observed_rows += month["predictor_observed_rows"]
        link_valid_rows += month["link_valid_rows"]
        trim_fail_count += month["trim_fail_count"]
        linked_rows += month["linked_rows"]
        outlier_cutoff_rows.extend(month["outlier_cutoff_rows"])
        for totals, monthly in [
            (step4_fail_counts, month["step4_fail_counts"]),
            (step6_fail_counts, month["step6_fail_counts"]),
            (step7_fail_counts, month["step7_fail_counts"]),
        ]:
            for filter_id, count in monthly.items():
                totals[filter_id] += count
        for filter_id, counts in month["step5_counts"].items():
            for key, count in counts.items():
                step5_counts[filter_id][key] += count
        eligible_rows += len(month["rows"])
        selected_rows.extend(month["rows"])

eligible_pool = pd.DataFrame(selected_rows, columns=FIELDNAMES)
# Public keys are fixed before sampling; source keys retain their sort types.
eligible_pool["id"] = eligible_pool["subject_id"].astype("string")
eligible_pool["time"] = (
    eligible_pool["baseline_year"].astype(int).astype(str)
    + "-"
    + eligible_pool["baseline_month"].astype(int).astype(str).str.zfill(2)
    + "-01"
)
eligible_pool["release_date"] = pd.to_datetime(
    eligible_pool["release_date"], errors="raise"
).dt.strftime("%Y-%m-%d")
selected, sampling_registry, sampling_summary = sample_task_records(
    eligible_pool,
    task_id=TASK_SLUG,
    output_dir=SAMPLING_DIR,
    output_sort_cols=["outcome_year", "outcome_month", "subject_id", "mish_target"],
)
selected_rows = selected.to_dict("records")

if not selected_rows:
    raise RuntimeError(f"{TASK_SLUG} produced zero rows.")
if {row["target"] for row in selected_rows} != {"retired", "not_retired"}:
    raise RuntimeError(f"{TASK_SLUG}: exported target domain is not exactly {{'retired', 'not_retired'}}.")

step5_rows = []
previous_after = status_universe
for requirement_order, (filter_id, pass_condition) in enumerate(STEP5_DETAILS, start=1):
    counts = step5_counts[filter_id]
    if counts["rows_before"] != previous_after:
        raise RuntimeError(
            f"Step-5 diagnostic chain breaks at {filter_id}: "
            f"before={counts['rows_before']}, previous_after={previous_after}"
        )
    rows_removed = counts["rows_before"] - counts["rows_after"]
    if rows_removed < 0:
        raise RuntimeError(f"Step-5 diagnostic {filter_id} increases the row count.")
    step5_rows.append(
        {
            "task_id": TASK_SLUG,
            "requirement_order": requirement_order,
            "requirement_id": filter_id,
            "pass_condition": pass_condition,
            "rows_before": counts["rows_before"],
            "rows_after": counts["rows_after"],
            "rows_removed": rows_removed,
        }
    )
    previous_after = counts["rows_after"]
if previous_after != target_observed_rows:
    raise RuntimeError("Step-5 diagnostics do not reconcile to target_observed_rows.")
if sum(row["rows_removed"] for row in step5_rows) != status_universe - target_observed_rows:
    raise RuntimeError("Step-5 diagnostic removals do not sum to the step-5 reduction.")

canonicalize_public_table(selected, task_id=TASK_SLUG).to_csv(destination, index=False)
step_rows = [
    construction_step(
        TASK_SLUG, 1, "Start from source observations",
        0, int(source_rows), 0,
    ),
    construction_step(
        TASK_SLUG, 2, "Attach additional sources",
        int(source_rows), int(source_rows), 0,
    ),
    construction_step(
        TASK_SLUG, 3, "Construct leads and lags",
        int(source_rows), int(source_rows), 0,
    ),
    construction_step(
        TASK_SLUG, 4, "Restrict universe",
        int(source_rows), int(status_universe), int(source_rows) - int(status_universe),
    ),
    construction_step(
        TASK_SLUG, 5, "Require observed targets",
        int(status_universe), int(target_observed_rows), int(status_universe) - int(target_observed_rows),
    ),
    construction_step(
        TASK_SLUG, 6, "Require observed predictors",
        int(target_observed_rows), int(predictor_observed_rows), int(target_observed_rows) - int(predictor_observed_rows),
    ),
    construction_step(
        TASK_SLUG, 7, "Apply logical and validity filters",
        int(predictor_observed_rows), int(link_valid_rows), int(predictor_observed_rows) - int(link_valid_rows),
    ),
    construction_step(
        TASK_SLUG, 8, "Apply outlier trimming",
        int(link_valid_rows), int(eligible_rows), int(link_valid_rows) - int(eligible_rows),
    ),
    construction_step(
        TASK_SLUG, 9, "Export sample",
        int(eligible_rows), int(len(selected_rows)), int(eligible_rows) - int(len(selected_rows)),
    ),
]

drop_detail_rows = []
for detail_order, (filter_id, pass_condition) in enumerate(STEP4_DETAILS, start=1):
    drop_detail_rows.append(
        filter_count(
            TASK_SLUG, 4, detail_order,
            filter_id, pass_condition, int(step4_fail_counts[filter_id]),
        )
    )
for row in step5_rows:
    drop_detail_rows.append(
        filter_count(
            TASK_SLUG, 5, row["requirement_order"],
            row["requirement_id"], row["pass_condition"], row["rows_removed"],
        )
    )
for detail_order, (filter_id, pass_condition) in enumerate(STEP6_DETAILS, start=1):
    drop_detail_rows.append(
        filter_count(
            TASK_SLUG, 6, detail_order,
            filter_id, pass_condition, int(step6_fail_counts[filter_id]),
        )
    )
for detail_order, (filter_id, pass_condition) in enumerate(STEP7_DETAILS, start=1):
    drop_detail_rows.append(
        filter_count(
            TASK_SLUG, 7, detail_order,
            filter_id, pass_condition, int(step7_fail_counts[filter_id]),
        )
    )
drop_detail_rows.append(
    filter_count(
        TASK_SLUG, 8, 1,
        "trim_optional_earnweek_baseline_strict_inside_baseline_month_p01_p99", "earnweek_baseline is missing or "
            "(earnweek_baseline > baseline-month p01 and earnweek_baseline < baseline-month p99)", int(trim_fail_count),
    )
)

DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(step_rows, columns=STEP_COLUMNS).to_csv(STEP_DIAGNOSTICS_PATH, index=False)
pd.DataFrame(drop_detail_rows, columns=DETAIL_COLUMNS).to_csv(DROP_DETAIL_PATH, index=False)
pd.DataFrame(outlier_cutoff_rows, columns=OUTLIER_CUTOFF_COLUMNS).to_csv(OUTLIER_CUTOFFS_PATH, index=False)
pd.DataFrame(step5_rows, columns=TARGET_AVAILABILITY_COLUMNS).to_csv(TARGET_AVAILABILITY_PATH, index=False)
print(
    f"Wrote {destination} with {len(selected_rows)} sampled rows "
    f"from {eligible_rows} prompt-complete rows scanned; "
    f"linked_rows={linked_rows}, status_universe={status_universe}."
)
