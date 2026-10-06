#!/usr/bin/env python
"""Build the tabular HouseholdBench CPS separation task."""

from __future__ import annotations

import argparse
import atexit
import sys
import shutil
import tempfile
from pathlib import Path

import pandas as pd


TASK_ID = "cps_employed_next_month_lf_status"
TASK_SLUG = "labor_cps_separation"
ROW_ID_PREFIX = "3c_cps_employed_next_month_lf_status"
TARGETS = ["employed", "unemployed", "not_in_labor_force"]
BASELINE_STATUS = "employed"
ADJACENT_BASELINE_MISH = {1, 2, 3, 5, 6, 7}

REPO_ROOT = Path(__file__).resolve().parents[3]
BASIC_INPUT_PATH = REPO_ROOT / "data/intermediate/cps_basic.parquet"
HOUSEHOLDBENCH_ROOT = REPO_ROOT / "data/householdbench"
DIAGNOSTICS_DIR = REPO_ROOT / "output" / "householdbench" / "tasks" / TASK_SLUG
SAMPLING_DIR = REPO_ROOT / "data" / "intermediate" / "householdbench" / TASK_SLUG
STEP_DIAGNOSTICS_PATH = DIAGNOSTICS_DIR / "01_sample_construction_steps.csv"
DROP_DETAIL_PATH = DIAGNOSTICS_DIR / "01_sample_construction_drop_details.csv"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.sample_accounting import construction_step, filter_count

from scripts.utils.table_schema import canonical_public_metadata, canonicalize_public_table

from scripts.utils import cps
from scripts.utils.parallel import default_worker_count, map_in_order
from scripts.utils.sampling import (
    build_partitioned_sampling_registry,
    write_sampling_candidate_partition,
)


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
    "age_group",
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
    *cps.CORE_MACRO_COLUMNS,
    "ahrsworkt_baseline",
    "uhrsworkt_baseline",
    "multjob_baseline",
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
    "age_group",
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
    *cps.CORE_MACRO_COLUMNS,
    "ahrsworkt_baseline",
    "uhrsworkt_baseline",
]
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
    ("universe_baseline_rotation_eligible", 'baseline["MISH"].isin(ADJACENT_BASELINE_MISH)'),
    ("universe_age_at_least_16", 'baseline["AGE"].ge(16)'),
    ("universe_baseline_status", 'baseline labor-force status is employed'),
    (
        "universe_target_mish_equals_baseline_mish_plus_1_when_linked",
        'target is unlinked or target["MISH"] == baseline["MISH"] + 1',
    ),
]
STEP5_DETAILS = [
    ("target_unique_same_cpsidp_record_available", "one unique same-CPSIDP record exists next month"),
    ("target_status_supported", "linked target labor-force status is supported"),
]
STEP7_DETAILS = [
    ("link_valid_unique_baseline_cpsidp", "baseline CPSIDP is valid and unique within its source month"),
]
STEP6_DETAILS = (
    [(f"predictor_{field}_not_missing", f'baseline.get("{field}") is not None') for field in COMMON_BASELINE_PREDICTOR_FIELDS]
    + [(f"predictor_macro_{field}_not_missing", f'macro.get("{field}") is not None') for field in COMMON_MACRO_PREDICTOR_FIELDS]
    + [(f"predictor_{field}_not_missing", f'baseline.get("{field}") is not None') for field in TASK_BASELINE_PREDICTOR_FIELDS]
)
BASIC_COLUMNS = [
    "YEAR",
    "MONTH",
    "CPSIDP",
    "MISH",
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
    "multjob",
    "release_date",
    "release_date_source",
]


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

step3_rows = 0
status_universe = 0
eligible_rows = 0
source_rows = 0
target_observed_rows = 0
predictor_observed_rows = 0
selected_rows: list[dict[str, object]] = []
step4_fail_counts = {filter_id: 0 for filter_id, _ in STEP4_DETAILS}
step5_fail_counts = {filter_id: 0 for filter_id, _ in STEP5_DETAILS}
step6_fail_counts = {filter_id: 0 for filter_id, _ in STEP6_DETAILS}
step7_fail_counts = {filter_id: 0 for filter_id, _ in STEP7_DETAILS}


def construct_rows(
    baseline_df: pd.DataFrame,
    target_df: pd.DataFrame | None,
    macro_map: dict,
) -> tuple[list[dict[str, object]], dict[str, int], dict[int, dict[str, int]]]:
    """Attach next-month fields without changing the baseline-row count."""
    row_counts = {
        "step3_rows": 0,
        "status_universe": 0,
        "target_observed_rows": 0,
        "predictor_observed_rows": 0,
        "eligible_rows": 0,
    }
    filter_counts = {
        4: {filter_id: 0 for filter_id, _ in STEP4_DETAILS},
        5: {filter_id: 0 for filter_id, _ in STEP5_DETAILS},
        6: {filter_id: 0 for filter_id, _ in STEP6_DETAILS},
        7: {filter_id: 0 for filter_id, _ in STEP7_DETAILS},
    }

    if target_df is None:
        target_lookup = baseline_df.iloc[0:0].copy()
    else:
        target_lookup = target_df.loc[
            target_df["_valid_link_key"] & target_df["_unique_cpsidp"]
        ].copy()
    joined = baseline_df.add_prefix("b_").merge(
        target_lookup.add_prefix("t_"),
        left_on="b__cpsidp",
        right_on="t__cpsidp",
        how="left",
        validate="many_to_one",
    )
    if len(joined) != len(baseline_df):
        raise RuntimeError(f"{TASK_SLUG}: step-3 link attachment changed the baseline row count.")
    row_counts["step3_rows"] += int(len(joined))
    joined["_baseline_link_key_valid"] = (
        joined["b__valid_link_key"] & joined["b__unique_cpsidp"]
    )

    rotation_pass = joined["b__mish"].isin(ADJACENT_BASELINE_MISH)
    age_pass = joined["b__age"].ge(16)
    status_pass = joined["b__status"].eq(BASELINE_STATUS)
    mish_pass = joined["t__cpsidp"].isna() | joined["t__mish"].eq(joined["b__mish"] + 1)
    step4_prior = pd.Series(True, index=joined.index)
    for (filter_id, _condition), pass_mask in zip(
        STEP4_DETAILS,
        [rotation_pass, age_pass, status_pass, mish_pass],
        strict=True,
    ):
        filter_counts[4][filter_id] += int((step4_prior & ~pass_mask).sum())
        step4_prior &= pass_mask
    universe = joined.loc[step4_prior].copy()
    row_counts["status_universe"] += int(len(universe))

    # Invalid or duplicate baseline link keys pass step 5 so that step 7,
    # rather than an implicit join filter, accounts for their removal.
    target_available = (~universe["_baseline_link_key_valid"]) | universe["t__cpsidp"].notna()
    filter_counts[5]["target_unique_same_cpsidp_record_available"] += int((~target_available).sum())
    linked = universe.loc[target_available].copy()
    target_supported = (~linked["_baseline_link_key_valid"]) | linked["t__status"].isin(TARGETS)
    filter_counts[5]["target_status_supported"] += int((~target_supported).sum())
    target_complete = linked.loc[target_supported].copy()
    row_counts["target_observed_rows"] += int(len(target_complete))

    rows: list[dict[str, object]] = []
    for payload in target_complete.to_dict("records"):
        link_valid = bool(payload["_baseline_link_key_valid"])
        if not link_valid:
            row_counts["predictor_observed_rows"] += 1
            filter_counts[7]["link_valid_unique_baseline_cpsidp"] += 1
            continue
        baseline = cps.normalize_basic_record(payload, prefix="b_")
        if baseline is None:
            raise RuntimeError(f"{TASK_SLUG}: baseline record failed normalization after target filtering.")
        macro = macro_map.get(cps.baseline_macro_key(baseline), {})
        predictor_complete = True
        for field in COMMON_BASELINE_PREDICTOR_FIELDS:
            present = baseline.get(field) is not None
            if predictor_complete and not present:
                filter_counts[6][f"predictor_{field}_not_missing"] += 1
            predictor_complete &= present
        for field in COMMON_MACRO_PREDICTOR_FIELDS:
            present = macro.get(field) is not None
            if predictor_complete and not present:
                filter_counts[6][f"predictor_macro_{field}_not_missing"] += 1
            predictor_complete &= present
        for field in TASK_BASELINE_PREDICTOR_FIELDS:
            present = baseline.get(field) is not None
            if predictor_complete and not present:
                filter_counts[6][f"predictor_{field}_not_missing"] += 1
            predictor_complete &= present
        if not predictor_complete:
            continue
        row_counts["predictor_observed_rows"] += 1
        target_obs = cps.normalize_basic_record(payload, prefix="t_")
        if target_obs is None:
            raise RuntimeError(f"{TASK_SLUG}: linked target failed normalization after target filtering.")
        answer = target_obs.get("status")
        if answer not in TARGETS:
            raise RuntimeError(f"{TASK_SLUG}: unsupported target survived the target-completeness step.")
        row_id = cps.stable_row_id(
            TASK_ID, ROW_ID_PREFIX, baseline["cpsidp"], baseline["year"], baseline["month"],
            target_obs["year"], target_obs["month"],
        )
        row = cps.base_table_row(FIELDNAMES, baseline, target_obs, macro, row_id, answer)
        row["selection_period"] = (
            f"{int(target_obs['year']):04d}-{int(target_obs['month']):02d}"
        )
        row.update({
            "ahrsworkt_baseline": baseline.get("ahrsworkt"),
            "uhrsworkt_baseline": baseline.get("uhrsworkt"),
            "multjob_baseline": baseline.get("multjob"),
        })
        cps.validate_table_row(row, REQUIRED_OUTPUT_COLUMNS, TARGETS)
        rows.append(row)
    row_counts["eligible_rows"] += len(rows)
    return rows, row_counts, filter_counts

def build_month(baseline_df: pd.DataFrame, target_df: pd.DataFrame) -> dict[str, object]:
    """Pass 1 for one outcome month: write its sampling candidates and keep its rows for pass 2."""
    rows, month_rows, month_filters = construct_rows(baseline_df, target_df, macro_map)
    result: dict[str, object] = {
        "source_rows": int(len(baseline_df)),
        "row_counts": month_rows,
        "filter_counts": month_filters,
        "period": None,
        "eligible_rows": len(rows),
    }
    if rows:
        period = str(rows[0]["selection_period"])
        if any(str(row["selection_period"]) != period for row in rows):
            raise RuntimeError(f"{TASK_SLUG}: pass-1 period grouping is invalid.")
        frame = pd.DataFrame(rows, columns=FIELDNAMES)
        if frame.duplicated(["subject_id", "outcome_year", "outcome_month", "mish_target"]).any():
            raise RuntimeError(f"{TASK_SLUG}: duplicate selection identities in {period}.")
        # Public keys are fixed before sampling; source keys retain their sort types.
        frame["id"] = frame["subject_id"].astype("string")
        frame["time"] = (
            frame["baseline_year"].astype(int).astype(str)
            + "-"
            + frame["baseline_month"].astype(int).astype(str).str.zfill(2)
            + "-01"
        )
        frame["release_date"] = pd.to_datetime(
            frame["release_date"], errors="raise"
        ).dt.strftime("%Y-%m-%d")
        write_sampling_candidate_partition(
            frame,
            task_id=TASK_SLUG,
            destination=candidate_dir / f"{period}.parquet",
        )
        frame.to_pickle(row_cache_dir / f"{period}.pkl")
        result["period"] = period
    return result


def build_month_block(outcome_months: list[int]) -> list[dict[str, object]]:
    """Pass 1 for consecutive outcome months, each linked to the month before it."""
    months = [
        month_idx for month_idx in [outcome_months[0] - 1, *outcome_months]
        if month_idx in month_row_groups
    ]
    results = []
    previous: tuple[int, pd.DataFrame] | None = None
    for month_idx, raw_month_df in cps.iter_month_block(
        BASIC_INPUT_PATH, BASIC_COLUMNS, months, month_row_groups
    ):
        target_df = cps.prepare_month_dataframe(raw_month_df, preserve_rows=True)
        if previous is not None and previous[0] == month_idx - 1 and month_idx in outcome_months:
            results.append(build_month(previous[1], target_df))
        previous = (month_idx, target_df)
    return results


def select_period_rows(period: str) -> tuple[str, int, pd.DataFrame]:
    """Pass 2 for one period: the pass-1 rows that the sampling registry exports."""
    frame = pd.read_pickle(row_cache_dir / f"{period}.pkl")
    metadata = canonical_public_metadata(frame, task_id=TASK_SLUG)
    selected_mask = pd.MultiIndex.from_frame(metadata[["id", "time"]]).isin(
        exported_key_index
    )
    return period, len(frame), frame.loc[selected_mask].copy()


# Pass 1: construct eligible rows and accumulate sample-accounting counts. Worker
# processes link blocks of months; the results come back in month order.
candidate_dir = Path(tempfile.mkdtemp(prefix=f"{TASK_SLUG}-sampling-"))
row_cache_dir = Path(tempfile.mkdtemp(prefix=f"{TASK_SLUG}-rows-"))
# Remove both folders also when the build fails; workers leave through os._exit.
for temporary_dir in (candidate_dir, row_cache_dir):
    atexit.register(shutil.rmtree, temporary_dir, ignore_errors=True)
month_row_groups = cps.month_row_groups(BASIC_INPUT_PATH)
eligible_by_period: dict[str, int] = {}
for block_results in map_in_order(
    build_month_block, cps.month_blocks(sorted(month_row_groups)), jobs=args.jobs
):
    for month in block_results:
        month_rows, month_filters = month["row_counts"], month["filter_counts"]
        source_rows += month["source_rows"]
        step3_rows += month_rows["step3_rows"]
        status_universe += month_rows["status_universe"]
        target_observed_rows += month_rows["target_observed_rows"]
        predictor_observed_rows += month_rows["predictor_observed_rows"]
        eligible_rows += month_rows["eligible_rows"]
        for totals, monthly in [
            (step4_fail_counts, month_filters[4]),
            (step5_fail_counts, month_filters[5]),
            (step6_fail_counts, month_filters[6]),
            (step7_fail_counts, month_filters[7]),
        ]:
            for filter_id, count in monthly.items():
                totals[filter_id] += count
        period = month["period"]
        if period is not None:
            if period in eligible_by_period:
                raise RuntimeError(f"{TASK_SLUG}: pass-1 period grouping is invalid.")
            eligible_by_period[period] = month["eligible_rows"]

sampling_sidecar, sampling_summary = build_partitioned_sampling_registry(
    candidate_dir,
    task_id=TASK_SLUG,
    output_dir=SAMPLING_DIR,
)
shutil.rmtree(candidate_dir)
exported_key_index = pd.MultiIndex.from_frame(sampling_sidecar[["id", "time"]])
print(
    f"{TASK_SLUG}: pass 1 complete; eligible={sum(eligible_by_period.values()):,} "
    f"periods={len(eligible_by_period):,}."
)
# Pass 2: retain the registered sample from the rows pass 1 kept.
selected_frames: list[pd.DataFrame] = []
pass2_counts: dict[str, int] = {}
for period, period_rows, selected_frame in map_in_order(
    select_period_rows, list(eligible_by_period), jobs=args.jobs
):
    pass2_counts[period] = period_rows
    selected_frames.append(selected_frame)
shutil.rmtree(row_cache_dir)
if pass2_counts != eligible_by_period:
    raise RuntimeError(f"{TASK_SLUG}: pass-2 eligible counts differ from pass 1.")
print(f"{TASK_SLUG}: pass 2 complete; selected={sampling_summary['exported_rows']:,}.")
selected = pd.concat(selected_frames, ignore_index=True)
selected = selected.sort_values(
    ["outcome_year", "outcome_month", "subject_id", "mish_target"], kind="mergesort"
).reset_index(drop=True)
if len(selected) != sampling_summary["exported_rows"]:
    raise RuntimeError(f"{TASK_SLUG}: pass-2 export count differs from the sampling registry.")
selected_rows = selected.to_dict("records")

if not selected_rows:
    raise RuntimeError(f"{TASK_SLUG} produced zero rows.")
if source_rows != step3_rows:
    raise RuntimeError(
        f"{TASK_SLUG}: step 3 is not row-preserving: "
        f"source_rows={source_rows}, attached_rows={step3_rows}."
    )
if sum(step4_fail_counts.values()) != step3_rows - status_universe:
    raise RuntimeError(f"{TASK_SLUG}: step-4 drop details do not reconcile.")
if sum(step5_fail_counts.values()) != status_universe - target_observed_rows:
    raise RuntimeError(f"{TASK_SLUG}: step-5 drop details do not reconcile.")
if sum(step6_fail_counts.values()) != target_observed_rows - predictor_observed_rows:
    raise RuntimeError(f"{TASK_SLUG}: step-6 drop details do not reconcile.")
if sum(step7_fail_counts.values()) != predictor_observed_rows - eligible_rows:
    raise RuntimeError(f"{TASK_SLUG}: step-7 drop details do not reconcile.")

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
        int(source_rows), int(step3_rows), 0,
    ),
    construction_step(
        TASK_SLUG, 4, "Restrict universe",
        int(step3_rows), int(status_universe), int(step3_rows) - int(status_universe),
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
        int(predictor_observed_rows), int(eligible_rows), int(predictor_observed_rows) - int(eligible_rows),
    ),
    construction_step(
        TASK_SLUG, 8, "Apply outlier trimming",
        int(eligible_rows), int(eligible_rows), 0,
    ),
    construction_step(
        TASK_SLUG, 9, "Export sample",
        int(eligible_rows), int(len(selected_rows)), int(eligible_rows) - int(len(selected_rows)),
    ),
]
drop_detail_rows = []
detail_order = 1
for filter_id, pass_condition in STEP4_DETAILS:
    drop_detail_rows.append(
        filter_count(
            TASK_SLUG, 4, detail_order,
            filter_id, pass_condition, int(step4_fail_counts[filter_id]),
        )
    )
    detail_order += 1
detail_order = 1
for filter_id, pass_condition in STEP5_DETAILS:
    drop_detail_rows.append(
        filter_count(
            TASK_SLUG, 5, detail_order,
            filter_id, pass_condition, int(step5_fail_counts[filter_id]),
        )
    )
    detail_order += 1
detail_order = 1
for filter_id, pass_condition in STEP6_DETAILS:
    drop_detail_rows.append(
        filter_count(
            TASK_SLUG, 6, detail_order,
            filter_id, pass_condition, int(step6_fail_counts[filter_id]),
        )
    )
    detail_order += 1
detail_order = 1
for filter_id, pass_condition in STEP7_DETAILS:
    drop_detail_rows.append(
        filter_count(
            TASK_SLUG, 7, detail_order,
            filter_id, pass_condition, int(step7_fail_counts[filter_id]),
        )
    )
    detail_order += 1
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(step_rows, columns=STEP_COLUMNS).to_csv(STEP_DIAGNOSTICS_PATH, index=False)
pd.DataFrame(drop_detail_rows, columns=DETAIL_COLUMNS).to_csv(DROP_DETAIL_PATH, index=False)
print(
    f"Wrote {destination} with {len(selected_rows)} sampled rows "
    f"from {eligible_rows} prompt-complete rows scanned; "
    f"step3_rows={step3_rows}, status_universe={status_universe}."
)
