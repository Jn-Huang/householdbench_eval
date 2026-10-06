#!/usr/bin/env python
"""Build the tabular HouseholdBench CPS displaced-worker employment task."""

from __future__ import annotations

import sys
import shutil
import tempfile
from pathlib import Path

import pandas as pd


TASK_SLUG = "labor_cps_displace"
ROW_ID_PREFIX = "3c_labor_cps_displace"
TARGETS = ["employed", "unemployed", "not_in_labor_force"]

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
    "dwresp",
    "dwstat",
    "dwreas",
    "dwnotice",
    "dwlastwrk",
    "dwfulltime",
    "dwunion",
    "dwclass",
    "dwhi",
    "dw_lookback_regime",
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
    "dwresp",
    "dwstat",
]
STEP_COLUMNS = ["task_id", "step", "step_label", "rows_before", "rows_after", "rows_dropped"]
DETAIL_COLUMNS = ["task_id", "step", "detail_order", "filter_id", "pass_condition", "fail_count"]
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
STEP4_DETAILS = [
    ("universe_dws_interview", "DWRESP == 2"),
    ("universe_displaced_worker_status", "DWSTAT == 1 among DWS interviews"),
]
STEP5_DETAILS = [("target_employment_status_mapped", 'obs.get("status") in TARGETS')]
STEP6_DETAILS = (
    [(f"predictor_{field}_not_missing", f'obs.get("{field}") is not None') for field in COMMON_BASELINE_PREDICTOR_FIELDS]
    + [(f"predictor_macro_{field}_not_missing", f'macro.get("{field}") is not None') for field in COMMON_MACRO_PREDICTOR_FIELDS]
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
    "DWRESP",
    "DWSTAT",
    "DWREAS",
    "DWNOTICE",
    "DWLASTWRK",
    "DWFULLTIME",
    "DWUNION",
    "DWHI",
    "DWCLASS",
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
    "dwresp",
    "dwstat",
    "dwreas",
    "dwnotice",
    "dwlastwrk",
    "dwfulltime",
    "dwunion",
    "dwhi",
    "dwclass",
    "release_date",
    "release_date_source",
]


destination = HOUSEHOLDBENCH_ROOT / "tabular" / f"{TASK_SLUG}.csv"

cps.check_columns(BASIC_INPUT_PATH, BASIC_COLUMNS)
cps.guard_parquet_chronology(BASIC_INPUT_PATH)
macro_map = cps.load_macro_context_map()

source_rows = 0
dws_interview_rows = 0
status_universe_rows = 0
target_observed_rows = 0
eligible_rows = 0
step4_fail_counts = {filter_id: 0 for filter_id, _ in STEP4_DETAILS}
step5_fail_counts = {filter_id: 0 for filter_id, _ in STEP5_DETAILS}
step6_fail_counts = {filter_id: 0 for filter_id, _ in STEP6_DETAILS}


def build_eligible_month(
    raw_month_df: pd.DataFrame,
    macro_map: dict,
) -> tuple[pd.DataFrame, dict[str, int], dict[int, dict[str, int]]]:
    """Return this month's rows and counts; the caller accumulates counts once."""
    row_counts = {
        "source_rows": 0,
        "dws_interview_rows": 0,
        "status_universe_rows": 0,
        "target_observed_rows": 0,
    }
    filter_counts = {
        4: {filter_id: 0 for filter_id, _ in STEP4_DETAILS},
        5: {filter_id: 0 for filter_id, _ in STEP5_DETAILS},
        6: {filter_id: 0 for filter_id, _ in STEP6_DETAILS},
    }

    row_counts["source_rows"] += int(len(raw_month_df))
    if raw_month_df.empty:
        return pd.DataFrame(columns=FIELDNAMES), row_counts, filter_counts

    dwresp = pd.to_numeric(raw_month_df["DWRESP"], errors="coerce").round()
    dwstat = pd.to_numeric(raw_month_df["DWSTAT"], errors="coerce").round()

    interview_pass = dwresp.eq(2)
    filter_counts[4]["universe_dws_interview"] += int((~interview_pass).sum())
    interviewed = raw_month_df.loc[interview_pass]
    row_counts["dws_interview_rows"] += int(len(interviewed))

    status_pass = dwstat.loc[interviewed.index].eq(1)
    filter_counts[4]["universe_displaced_worker_status"] += int((~status_pass).sum())
    universe = interviewed.loc[status_pass]
    row_counts["status_universe_rows"] += int(len(universe))

    rows: list[dict[str, object]] = []
    for payload in universe.itertuples(index=False):
        obs = cps.normalize_basic_record(payload._asdict())
        if obs is None:
            raise RuntimeError("A displaced-worker record has invalid required timing or identifier fields.")

        answer = obs.get("status")
        if answer not in TARGETS:
            filter_counts[5]["target_employment_status_mapped"] += 1
            continue
        row_counts["target_observed_rows"] += 1
        macro = macro_map.get(cps.baseline_macro_key(obs), {})
        predictor_complete = True
        for field in COMMON_BASELINE_PREDICTOR_FIELDS:
            if obs.get(field) is None:
                filter_counts[6][f"predictor_{field}_not_missing"] += 1
                predictor_complete = False
        for field in COMMON_MACRO_PREDICTOR_FIELDS:
            if macro.get(field) is None:
                filter_counts[6][f"predictor_macro_{field}_not_missing"] += 1
                predictor_complete = False
        if not predictor_complete:
            continue

        row_id = cps.stable_row_id(TASK_SLUG, ROW_ID_PREFIX, obs["cpsidp"], obs["year"], obs["month"], answer)
        row = cps.base_table_row(FIELDNAMES, obs, obs, macro, row_id, answer)
        row["selection_period"] = f"{int(obs['year']):04d}-{int(obs['month']):02d}"
        row.update(
            {
                "dwresp": obs.get("dwresp"),
                "dwstat": obs.get("dwstat"),
                "dwreas": obs.get("dwreas"),
                "dwnotice": obs.get("dwnotice"),
                "dwlastwrk": obs.get("dwlastwrk"),
                "dwfulltime": obs.get("dwfulltime"),
                "dwunion": obs.get("dwunion"),
                "dwclass": obs.get("dwclass"),
                "dwhi": obs.get("dwhi"),
                "dw_lookback_regime": (
                    "five_year_lookback_1984_1992"
                    if int(obs["year"]) <= 1992
                    else "three_year_lookback_1994_plus"
                ),
            }
        )
        cps.validate_table_row(row, REQUIRED_OUTPUT_COLUMNS, TARGETS)
        rows.append(row)
    return pd.DataFrame(rows, columns=FIELDNAMES), row_counts, filter_counts


# Pass 1: construct eligible rows and accumulate sample-accounting counts.
candidate_dir = Path(tempfile.mkdtemp(prefix=f"{TASK_SLUG}-sampling-"))
eligible_by_period: dict[str, int] = {}
for _month_idx, raw_month_df in cps.iter_month_dataframes(BASIC_INPUT_PATH, BASIC_COLUMNS):
    eligible_month, month_rows, month_filters = build_eligible_month(raw_month_df, macro_map)
    source_rows += month_rows["source_rows"]
    dws_interview_rows += month_rows["dws_interview_rows"]
    status_universe_rows += month_rows["status_universe_rows"]
    target_observed_rows += month_rows["target_observed_rows"]
    for totals, monthly in [
        (step4_fail_counts, month_filters[4]),
        (step5_fail_counts, month_filters[5]),
        (step6_fail_counts, month_filters[6]),
    ]:
        for filter_id, count in monthly.items():
            totals[filter_id] += count
    if eligible_month.empty:
        continue
    periods = eligible_month["selection_period"].unique().tolist()
    if len(periods) != 1 or periods[0] in eligible_by_period:
        raise RuntimeError(f"{TASK_SLUG}: pass-1 outcome periods are not unique by source month.")
    if eligible_month.duplicated(["subject_id", "outcome_year", "outcome_month", "mish_target"]).any():
        raise RuntimeError(f"{TASK_SLUG}: duplicate selection identities in {periods[0]}.")
    eligible_by_period[periods[0]] = int(len(eligible_month))
    # Public keys are fixed before sampling; source keys retain their sort types.
    eligible_month["id"] = eligible_month["subject_id"].astype("string")
    eligible_month["time"] = (
        eligible_month["baseline_year"].astype(int).astype(str)
        + "-"
        + eligible_month["baseline_month"].astype(int).astype(str).str.zfill(2)
        + "-01"
    )
    eligible_month["release_date"] = pd.to_datetime(
        eligible_month["release_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    write_sampling_candidate_partition(
        eligible_month,
        task_id=TASK_SLUG,
        destination=candidate_dir / f"{periods[0]}.parquet",
    )

eligible_rows = sum(eligible_by_period.values())
sampling_sidecar, sampling_summary = build_partitioned_sampling_registry(
    candidate_dir,
    task_id=TASK_SLUG,
    output_dir=SAMPLING_DIR,
)
shutil.rmtree(candidate_dir)
exported_key_index = pd.MultiIndex.from_frame(sampling_sidecar[["id", "time"]])
# Pass 2: reconstruct the same rows and retain the registered sample.
selected_frames: list[pd.DataFrame] = []
pass2_counts: dict[str, int] = {}
for _month_idx, raw_month_df in cps.iter_month_dataframes(BASIC_INPUT_PATH, BASIC_COLUMNS):
    eligible_month, _, _ = build_eligible_month(raw_month_df, macro_map)
    if eligible_month.empty:
        continue
    period = str(eligible_month["selection_period"].iloc[0])
    pass2_counts[period] = int(len(eligible_month))
    # Public keys are fixed before sampling; source keys retain their sort types.
    eligible_month["id"] = eligible_month["subject_id"].astype("string")
    eligible_month["time"] = (
        eligible_month["baseline_year"].astype(int).astype(str)
        + "-"
        + eligible_month["baseline_month"].astype(int).astype(str).str.zfill(2)
        + "-01"
    )
    eligible_month["release_date"] = pd.to_datetime(
        eligible_month["release_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    metadata = canonical_public_metadata(eligible_month, task_id=TASK_SLUG)
    selected_mask = pd.MultiIndex.from_frame(metadata[["id", "time"]]).isin(
        exported_key_index
    )
    selected_frames.append(eligible_month.loc[selected_mask].copy())
if pass2_counts != eligible_by_period:
    raise RuntimeError(f"{TASK_SLUG}: pass-2 eligible counts differ from pass 1.")
selected = pd.concat(selected_frames, ignore_index=True)
selected = selected.sort_values(
    ["outcome_year", "outcome_month", "subject_id", "mish_target"], kind="mergesort"
).reset_index(drop=True)
if len(selected) != sampling_summary["exported_rows"]:
    raise RuntimeError(f"{TASK_SLUG}: pass-2 export count differs from the sampling registry.")
selected_rows = selected.to_dict("records")
if not selected_rows:
    raise RuntimeError(f"{TASK_SLUG} produced zero rows.")

canonicalize_public_table(selected, task_id=TASK_SLUG).to_csv(destination, index=False)
step_rows = [
    construction_step(
        TASK_SLUG, 1, "Start from source observations",
        0, source_rows, 0,
    ),
    construction_step(
        TASK_SLUG, 2, "Attach additional sources",
        source_rows, source_rows, 0,
    ),
    construction_step(
        TASK_SLUG, 3, "Construct leads and lags",
        source_rows, source_rows, 0,
    ),
    construction_step(
        TASK_SLUG, 4, "Restrict universe",
        source_rows, status_universe_rows, source_rows - status_universe_rows,
    ),
    construction_step(
        TASK_SLUG, 5, "Require observed targets",
        status_universe_rows, target_observed_rows, status_universe_rows - target_observed_rows,
    ),
    construction_step(
        TASK_SLUG, 6, "Require observed predictors",
        target_observed_rows, eligible_rows, target_observed_rows - eligible_rows,
    ),
    construction_step(
        TASK_SLUG, 7, "Apply logical and validity filters",
        eligible_rows, eligible_rows, 0,
    ),
    construction_step(
        TASK_SLUG, 8, "Apply outlier trimming",
        eligible_rows, eligible_rows, 0,
    ),
    construction_step(
        TASK_SLUG, 9, "Export sample",
        eligible_rows, len(selected_rows), eligible_rows - len(selected_rows),
    ),
]

drop_detail_rows = []
for step, details, counts in [
    (4, STEP4_DETAILS, step4_fail_counts),
    (5, STEP5_DETAILS, step5_fail_counts),
    (6, STEP6_DETAILS, step6_fail_counts),
]:
    for detail_order, (filter_id, pass_condition) in enumerate(details, start=1):
        drop_detail_rows.append(
            filter_count(
                TASK_SLUG, step, detail_order,
                filter_id, pass_condition, int(counts[filter_id]),
            )
        )

step_table = pd.DataFrame(step_rows, columns=STEP_COLUMNS)
filter_steps = step_table.loc[step_table["step"].ge(2)]
if not (filter_steps["rows_before"] - filter_steps["rows_after"] == filter_steps["rows_dropped"]).all():
    raise RuntimeError("Sample-construction step accounting does not reconcile.")
if int(step4_fail_counts["universe_dws_interview"] + step4_fail_counts["universe_displaced_worker_status"]) != source_rows - status_universe_rows:
    raise RuntimeError("Sequential Step 4 detail counts do not reconcile to the universe restriction.")

DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
step_table.to_csv(STEP_DIAGNOSTICS_PATH, index=False)
pd.DataFrame(drop_detail_rows, columns=DETAIL_COLUMNS).to_csv(DROP_DETAIL_PATH, index=False)
print(
    f"Wrote {destination} with {len(selected_rows)} sampled rows "
    f"from {eligible_rows} prompt-complete rows scanned; "
    f"dws_interviews={dws_interview_rows}, displaced_worker_universe={status_universe_rows}."
)
