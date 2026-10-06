#!/usr/bin/env python
"""Build the tabular CPS displaced-worker current-earnings task."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


TASK_SLUG = "income_cps_displace"
ROW_ID_PREFIX = "3c_income_cps_displace"

REPO_ROOT = Path(__file__).resolve().parents[3]
BASIC_INPUT_PATH = REPO_ROOT / "data/intermediate/cps_basic.parquet"
HOUSEHOLDBENCH_ROOT = REPO_ROOT / "data/householdbench"
DIAGNOSTICS_DIR = REPO_ROOT / "output" / "householdbench" / "tasks" / TASK_SLUG
SAMPLING_DIR = REPO_ROOT / "data" / "intermediate" / "householdbench" / TASK_SLUG
STEP_DIAGNOSTICS_PATH = DIAGNOSTICS_DIR / "01_sample_construction_steps.csv"
DROP_DETAIL_PATH = DIAGNOSTICS_DIR / "01_sample_construction_drop_details.csv"
OUTLIER_CUTOFFS_PATH = DIAGNOSTICS_DIR / "01_outlier_cutoffs.csv"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.sample_accounting import construction_step, filter_count

from scripts.utils.table_schema import canonicalize_public_table

from scripts.utils import cps
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
    "dwyears",
    "dwweekl",
    "dwjobsince",
    "dwwksun",
    "dw_lookback_regime",
]
COMMON_PREDICTOR_FIELDS = [
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
]
REQUIRED_DWS_PREDICTORS = ["dwweekl", "dwyears", "dwjobsince", "dwwksun"]
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
    *COMMON_PREDICTOR_FIELDS,
    "dwresp",
    "dwstat",
    *REQUIRED_DWS_PREDICTORS,
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
    "dwclass",
    "dwhi",
    "dwyears",
    "dwweekl",
    "dwweekc",
    "dwjobsince",
    "dwwksun",
    "release_date",
    "release_date_source",
]

destination = HOUSEHOLDBENCH_ROOT / "tabular" / f"{TASK_SLUG}.csv"

cps.check_columns(BASIC_INPUT_PATH, BASIC_COLUMNS)
cps.guard_parquet_chronology(BASIC_INPUT_PATH)
macro_map = cps.load_macro_context_map()

# 1-4. Count all source rows, attach macro data without filtering, construct no
# leads or lags, and then apply the DWS/current-employment universe sequentially.
source_rows = 0
dws_interview_rows = 0
displaced_rows = 0
employed_displaced_rows = 0
displaced_records: list[dict[str, object]] = []
employed_records: list[dict[str, object]] = []

for _month_index, raw_month_df in cps.iter_month_dataframes(BASIC_INPUT_PATH, BASIC_COLUMNS):
    source_rows += int(len(raw_month_df))
    if raw_month_df.empty:
        continue
    dwresp_code = pd.to_numeric(raw_month_df["DWRESP"], errors="coerce").round()
    interviewed = raw_month_df.loc[dwresp_code.eq(2)]
    dws_interview_rows += int(len(interviewed))
    dwstat_code = pd.to_numeric(interviewed["DWSTAT"], errors="coerce").round()
    displaced = interviewed.loc[dwstat_code.eq(1)]
    displaced_rows += int(len(displaced))
    empstat_code = pd.to_numeric(displaced["EMPSTAT"], errors="coerce").round()
    employed_index = set(displaced.index[empstat_code.isin([10, 12])])
    employed_displaced_rows += len(employed_index)

    for source_index, payload in zip(displaced.index, displaced.itertuples(index=False), strict=True):
        obs = cps.normalize_basic_record(payload._asdict())
        if obs is None:
            raise RuntimeError(
                f"{TASK_SLUG}: a DWS displaced-worker row has invalid year, month, CPSIDP, or MISH."
            )
        displaced_records.append(obs)
        if source_index in employed_index:
            if obs.get("status") != "employed":
                raise RuntimeError(f"{TASK_SLUG}: EMPSTAT and normalized employment status disagree.")
            employed_records.append(obs)

if not displaced_records:
    raise RuntimeError(f"{TASK_SLUG}: no displaced-worker records found.")
if len(employed_records) != employed_displaced_rows:
    raise RuntimeError(f"{TASK_SLUG}: normalized employed-universe row count changed.")

# Compute unweighted, source-wave wage cutoffs before target or predictor
# completeness filters. Each variable uses its own finite valid observations.
reference_rows = []
for obs in displaced_records:
    period = f"{int(obs['year']):04d}-{int(obs['month']):02d}"
    reference_rows.append(
        {
            "period": period,
            "dwweekc": obs.get("dwweekc"),
            "dwweekl": obs.get("dwweekl"),
        }
    )
reference = pd.DataFrame(reference_rows)
cutoff_tables: dict[str, pd.DataFrame] = {}
outlier_cutoff_rows = []
for variable, role in [("dwweekc", "target"), ("dwweekl", "predictor")]:
    reference[variable] = pd.to_numeric(reference[variable], errors="coerce")
    valid_reference = reference.loc[reference[variable].notna(), ["period", variable]].copy()
    cutoffs = (
        valid_reference.groupby("period")[variable]
        .quantile([0.01, 0.99])
        .unstack()
        .rename(columns={0.01: f"{variable}_p01", 0.99: f"{variable}_p99"})
        .reset_index()
    )
    if cutoffs.empty or cutoffs[[f"{variable}_p01", f"{variable}_p99"]].isna().any().any():
        raise RuntimeError(f"{TASK_SLUG}: missing source-wave cutoffs for {variable}.")
    degenerate = cutoffs[f"{variable}_p01"].ge(cutoffs[f"{variable}_p99"])
    if degenerate.any():
        bad_periods = cutoffs.loc[degenerate, "period"].astype(str).tolist()
        raise RuntimeError(f"{TASK_SLUG}: degenerate source-wave {variable} cutoffs: {bad_periods}")
    cutoff_tables[variable] = cutoffs
    for cutoff in cutoffs.itertuples(index=False):
        values = valid_reference.loc[valid_reference["period"].eq(cutoff.period), variable]
        p01 = getattr(cutoff, f"{variable}_p01")
        p99 = getattr(cutoff, f"{variable}_p99")
        outlier_cutoff_rows.append(
            {
                "task_id": TASK_SLUG,
                "variable": variable,
                "role": role,
                "period": cutoff.period,
                "n_reference": int(len(values)),
                "p01": float(p01),
                "p99": float(p99),
                "n_at_or_below_p01": int(values.le(p01).sum()),
                "n_at_or_above_p99": int(values.ge(p99).sum()),
            }
        )

# 5. Require a valid current weekly earnings target.
target_observed_records = [obs for obs in employed_records if obs.get("dwweekc") is not None]

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
        source_rows, employed_displaced_rows, source_rows - employed_displaced_rows,
    ),
    construction_step(
        TASK_SLUG, 5, "Require observed targets",
        employed_displaced_rows, len(target_observed_records), employed_displaced_rows - len(target_observed_records),
    ),
]
drop_detail_rows = [
    filter_count(
        TASK_SLUG, 4, 1,
        "universe_dws_interview", 'DWRESP == 2', source_rows - dws_interview_rows,
    ),
    filter_count(
        TASK_SLUG, 4, 2,
        "universe_displaced_worker", 'DWSTAT == 1 among DWS interviews', dws_interview_rows - displaced_rows,
    ),
    filter_count(
        TASK_SLUG, 4, 3,
        "universe_currently_employed", 'EMPSTAT in {10, 12} among displaced workers', displaced_rows - employed_displaced_rows,
    ),
    filter_count(
        TASK_SLUG, 5, 1,
        "target_dwweekc_not_missing", 'dwweekc is finite and valid', employed_displaced_rows - len(target_observed_records),
    ),
]

# Create task rows before applying the required-predictor and wage-tail filters.
pool_rows = []
for obs in target_observed_records:
    macro = macro_map.get(cps.baseline_macro_key(obs), {})
    answer = float(obs["dwweekc"])
    row_id = cps.stable_row_id(
        TASK_SLUG,
        ROW_ID_PREFIX,
        obs["cpsidp"],
        obs["year"],
        obs["month"],
        answer,
    )
    row = cps.base_table_row(FIELDNAMES, obs, obs, macro, row_id, answer)
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
            "dwyears": obs.get("dwyears"),
            "dwweekl": obs.get("dwweekl"),
            "dwjobsince": obs.get("dwjobsince"),
            "dwwksun": obs.get("dwwksun"),
            "dw_lookback_regime": (
                "five_year_lookback_1984_1992"
                if int(obs["year"]) <= 1992
                else "three_year_lookback_1994_plus"
            ),
        }
    )
    pool_rows.append(row)
pool = pd.DataFrame(pool_rows, columns=FIELDNAMES)

# 6. Require common profile/macro fields and all four DWS predictors. Detail
# counts are sequential so they reconcile exactly to the stage total.
rows_before_step6 = len(pool)
detail_order = 1
for column in [*COMMON_PREDICTOR_FIELDS, "dwresp", "dwstat", *REQUIRED_DWS_PREDICTORS]:
    pass_mask = pool[column].notna()
    drop_detail_rows.append(
        filter_count(
            TASK_SLUG, 6, detail_order,
            f"predictor_{column}_not_missing", f'pool["{column}"].notna()', int((~pass_mask).sum()),
        )
    )
    pool = pool.loc[pass_mask].copy()
    detail_order += 1
step_rows.append(
    construction_step(
        TASK_SLUG, 6, "Require observed predictors",
        rows_before_step6, len(pool), rows_before_step6 - len(pool),
    )
)

# 7. Timing, identifier, domain, and target-separation conditions are validated
# below. They are assertions, not sample filters.
step_rows.append(
    construction_step(
        TASK_SLUG, 7, "Apply logical and validity filters",
        len(pool), len(pool), 0,
    )
)

# 8. Attach each variable's own wave cutoffs and trim sequentially, strictly
# inside the p01/p99 interval. Retained wage values are not transformed.
pool["period"] = pool["baseline_year"].astype(int).astype(str).str.zfill(4) + "-" + pool["baseline_month"].astype(int).astype(str).str.zfill(2)
for variable in ["dwweekc", "dwweekl"]:
    pool = pool.merge(cutoff_tables[variable], on="period", how="left", validate="many_to_one")
    if pool[[f"{variable}_p01", f"{variable}_p99"]].isna().any().any():
        bad_periods = sorted(pool.loc[pool[f"{variable}_p01"].isna(), "period"].unique())
        raise RuntimeError(f"{TASK_SLUG}: missing {variable} cutoffs for task waves {bad_periods}.")

rows_before_step8 = len(pool)
trim_variables = [("dwweekc", "target", "target"), ("dwweekl", "predictor", "dwweekl")]
for detail_order, (variable, role, value_column) in enumerate(trim_variables, start=1):
    pass_mask = pool[value_column].gt(pool[f"{variable}_p01"]) & pool[value_column].lt(pool[f"{variable}_p99"])
    drop_detail_rows.append(
        filter_count(
            TASK_SLUG, 8, detail_order,
            f"trim_{variable}_{role}_strict_inside_wave_p01_p99", f'{value_column} > {variable}_p01 and {value_column} < {variable}_p99', int((~pass_mask).sum()),
        )
    )
    pool = pool.loc[pass_mask].copy()
step_rows.append(
    construction_step(
        TASK_SLUG, 8, "Apply outlier trimming",
        rows_before_step8, len(pool), rows_before_step8 - len(pool),
    )
)
if pool.empty:
    raise RuntimeError(f"{TASK_SLUG}: no rows remain after wage trimming.")
pool["selection_period"] = [
    f"{int(year):04d}-{int(month):02d}"
    for year, month in zip(pool["outcome_year"], pool["outcome_month"], strict=True)
]

# Validate construction without changing sample membership.
if not pool["baseline_status"].eq("employed").all():
    raise RuntimeError(f"{TASK_SLUG}: non-employed row survived the task universe.")
if not (
    pool["baseline_year"].eq(pool["outcome_year"])
    & pool["baseline_month"].eq(pool["outcome_month"])
    & pool["mish_baseline"].eq(pool["mish_target"])
).all():
    raise RuntimeError(f"{TASK_SLUG}: contemporaneous target timing is inconsistent.")
if pool["row_id"].duplicated().any():
    raise RuntimeError(f"{TASK_SLUG}: row_id is not unique.")
if pool["subject_id"].isna().any() or pool["release_date"].isna().any():
    raise RuntimeError(f"{TASK_SLUG}: identifier or release metadata is missing.")
if not pd.to_numeric(pool["target"], errors="coerce").ge(0).all():
    raise RuntimeError(f"{TASK_SLUG}: target contains invalid weekly earnings.")

# 9. Apply the common sampling ledger, then export in chronological order.
eligible_rows = len(pool)
# Public keys are fixed before sampling; source keys retain their sort types.
pool["id"] = pool["subject_id"].astype("string")
pool["time"] = (
    pool["baseline_year"].astype(int).astype(str)
    + "-"
    + pool["baseline_month"].astype(int).astype(str).str.zfill(2)
    + "-01"
)
pool["release_date"] = pd.to_datetime(pool["release_date"], errors="raise").dt.strftime("%Y-%m-%d")
selected, sampling_registry, sampling_summary = sample_task_records(
    pool,
    task_id=TASK_SLUG,
    output_dir=SAMPLING_DIR,
    output_sort_cols=["outcome_year", "outcome_month", "subject_id", "mish_target"],
)
canonicalize_public_table(selected, task_id=TASK_SLUG).to_csv(destination, index=False)
step_rows.append(
    construction_step(
        TASK_SLUG, 9, "Export sample",
        eligible_rows, len(selected), eligible_rows - len(selected),
    )
)

DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(step_rows, columns=STEP_COLUMNS).to_csv(STEP_DIAGNOSTICS_PATH, index=False)
pd.DataFrame(drop_detail_rows, columns=DETAIL_COLUMNS).to_csv(DROP_DETAIL_PATH, index=False)
pd.DataFrame(outlier_cutoff_rows, columns=OUTLIER_CUTOFF_COLUMNS).to_csv(OUTLIER_CUTOFFS_PATH, index=False)
print(
    f"{TASK_SLUG}: source={source_rows:,} displaced={displaced_rows:,} "
    f"employed={employed_displaced_rows:,} eligible={eligible_rows:,} sampled={len(selected):,} "
    f"tabular={destination}"
)
