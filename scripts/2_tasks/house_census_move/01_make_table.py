#!/usr/bin/env python
"""Build the tabular CSV for the Census five-year mobility task."""

from __future__ import annotations

import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TASK_ID = "census_5y_mobility_destination_type"
TASK_SLUG = "house_census_move"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.sample_accounting import construction_step, filter_count

from scripts.utils.table_schema import canonicalize_public_table


from scripts.utils.macro_context import CORE_MACRO_COLUMNS, attach_macro_context
from scripts.utils.sampling import (
    build_partitioned_sampling_registry,
    write_sampling_candidate_partition,
)


def clean_label_series(batch: pd.DataFrame, column: str) -> pd.Series:
    if column not in batch.columns:
        raise RuntimeError(f"Cleaned input parquet is missing required label column {column}.")
    label = batch[column].astype("string").str.strip()
    return label.mask(label.eq(""))


INPUT_PATH = PROJECT_ROOT / "data/intermediate/census_decennial.parquet"
STATE_MACRO_INPUT_PATH = PROJECT_ROOT / "data/intermediate/state_macro_panel.parquet"
TABULAR_PATH = PROJECT_ROOT / "data" / "householdbench" / "tabular" / f"{TASK_SLUG}.csv"
DIAGNOSTICS_DIR = PROJECT_ROOT / "output" / "householdbench" / "tasks" / TASK_SLUG
SAMPLING_DIR = PROJECT_ROOT / "data" / "intermediate" / "householdbench" / TASK_SLUG
STEP_DIAGNOSTICS_PATH = DIAGNOSTICS_DIR / "01_sample_construction_steps.csv"
DROP_DETAIL_PATH = DIAGNOSTICS_DIR / "01_sample_construction_drop_details.csv"
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

MIGRATION_YEARS = {1990, 2000}
MOBILITY_TARGETS = {"stay", "move_within_state", "move_to_different_state"}
VALID_US_STATE_CODES = {
    1,
    2,
    4,
    5,
    6,
    8,
    9,
    10,
    11,
    12,
    13,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    53,
    54,
    55,
    56,
}
USECOLS = [
    "YEAR",
    "SAMPLE",
    "SERIAL",
    "PERNUM",
    "NUMPREC",
    "FAMSIZE",
    "STATEFIP",
    "REGION",
    "METRO",
    "GQ",
    "OWNERSHP",
    "RELATE",
    "AGE",
    "SEX",
    "SEX_label",
    "RACE",
    "RACE_label",
    "HISPAN",
    "HISPAN_label",
    "EDUC",
    "EDUC_label",
    "EDUCD",
    "EMPSTAT",
    "LABFORCE",
    "CLASSWKR",
    "INCTOT",
    "INCWAGE",
    "MIGRATE5",
    "MIGPLAC5",
    "STATEFIP_label",
    "MIGPLAC5_label",
    "release_date",
    "release_date_source",
]
SAMPLE_COLUMNS = [
    "subject_id",
    "selection_period",
    "id",
    "time",
    "release_date",
    "release_date_source",
    "year",
    "sample",
    "serial",
    "pernum",
    "origin_year",
    "state_macro_reference_year",
    "age_origin",
    "sex",
    "race_ethnicity",
    "education",
    "origin_state",
    "origin_statefip",
    "state_unemployment_rate_origin",
    "state_pcpi_growth_origin",
    "state_house_price_growth_origin",
    "us_unemployment_rate_origin",
    "us_pcpi_growth_origin",
    "us_house_price_growth_origin",
    *CORE_MACRO_COLUMNS,
    "target",
]
REQUIRED_INTEGER_COLUMNS = ["YEAR", "SAMPLE", "SERIAL", "PERNUM"]
STATE_MACRO_COLUMNS = [
    "state_unemployment_rate_origin",
    "state_pcpi_growth_origin",
    "state_house_price_growth_origin",
    "us_unemployment_rate_origin",
    "us_pcpi_growth_origin",
    "us_house_price_growth_origin",
]
REQUIRED_PROMPT_COLUMNS = ["sex", "race_ethnicity", "education", "origin_state"]
STEP4_DETAILS = [
    ("universe_migration_year", 'norm["year"].isin(MIGRATION_YEARS)'),
    ("universe_age_origin_at_least_30", 'norm["age"].sub(5).ge(30)'),
    ("universe_migration_status_mapped", 'norm["migrate5_code"].isin([1, 2, 3])'),
    ("universe_origin_state_valid_us_state", "origin_state_code.isin(sorted(VALID_US_STATE_CODES))"),
]
STEP6_DETAILS = (
    [(f"predictor_{column}_not_missing", f'eligible["{column}"].notna()') for column in STATE_MACRO_COLUMNS]
    + [(f"predictor_{column}_not_missing", f'eligible["{column}"].notna()') for column in CORE_MACRO_COLUMNS]
    + [(f"predictor_{column}_not_missing", f'eligible["{column}"].notna()') for column in REQUIRED_PROMPT_COLUMNS]
)

if not INPUT_PATH.exists():
    raise SystemExit(f"Census input not found: {INPUT_PATH}")
if not STATE_MACRO_INPUT_PATH.exists():
    raise SystemExit(f"State macro panel not found: {STATE_MACRO_INPUT_PATH}")

raw_state_macro = pd.read_parquet(STATE_MACRO_INPUT_PATH)
needed_state_macro = {
    "statefip",
    "year",
    "unemployment_rate",
    "pcpi_growth_1y",
    "hpi_growth_1y",
    "us_unemployment_rate",
    "us_pcpi_growth_1y",
    "us_hpi_growth_1y",
}
if not needed_state_macro.issubset(raw_state_macro.columns):
    raise SystemExit(
        f"Unexpected state macro schema in {STATE_MACRO_INPUT_PATH}; expected columns {sorted(needed_state_macro)}"
    )

state_macro = raw_state_macro.rename(
    columns={
        "statefip": "origin_state_code",
        "year": "state_macro_reference_year",
        "unemployment_rate": "state_unemployment_rate_origin",
        "pcpi_growth_1y": "state_pcpi_growth_origin",
        "hpi_growth_1y": "state_house_price_growth_origin",
        "us_unemployment_rate": "us_unemployment_rate_origin",
        "us_pcpi_growth_1y": "us_pcpi_growth_origin",
        "us_hpi_growth_1y": "us_house_price_growth_origin",
    }
)[["origin_state_code", "state_macro_reference_year", *STATE_MACRO_COLUMNS]].copy()
state_macro["origin_state_code"] = pd.to_numeric(state_macro["origin_state_code"], errors="coerce").astype("Int64")
state_macro["state_macro_reference_year"] = pd.to_numeric(
    state_macro["state_macro_reference_year"], errors="coerce"
).astype("Int64")
for column in STATE_MACRO_COLUMNS:
    state_macro[column] = pd.to_numeric(state_macro[column], errors="coerce")
if state_macro[["origin_state_code", "state_macro_reference_year", *STATE_MACRO_COLUMNS]].isna().any().any():
    raise SystemExit(f"State macro panel has missing values in key columns: {STATE_MACRO_INPUT_PATH}")
if state_macro.duplicated(["origin_state_code", "state_macro_reference_year"]).any():
    raise SystemExit("State macro panel has duplicate state-year rows.")

eligible_rows = 0
eligible_rows_by_year = {year: 0 for year in sorted(MIGRATION_YEARS)}
source_rows = 0
universe_rows = 0
target_observed_rows = 0
predictor_observed_rows = 0
step4_fail_counts = {filter_id: 0 for filter_id, _ in STEP4_DETAILS}
step6_fail_counts = {filter_id: 0 for filter_id, _ in STEP6_DETAILS}
temporary_root = Path(tempfile.mkdtemp(prefix=f"{TASK_SLUG}-sampling-"))
candidate_dir = temporary_root / "candidates"
eligible_cache_dir = temporary_root / "eligible"
eligible_cache_dir.mkdir(parents=True)
parquet_file = pq.ParquetFile(INPUT_PATH)
missing_parquet_columns = sorted(set(USECOLS) - set(parquet_file.schema_arrow.names))
if missing_parquet_columns:
    raise RuntimeError(
        f"Cleaned Census parquet is missing columns required by {TASK_SLUG}: {missing_parquet_columns}. "
        "Rerun scripts/1_preprocessing/census.py before building this task table."
    )

for row_group in range(parquet_file.num_row_groups):
    batch = parquet_file.read_row_group(row_group, columns=USECOLS).to_pandas()
    source_rows += int(len(batch))

    integer_series = {}
    for column in REQUIRED_INTEGER_COLUMNS:
        series = pd.to_numeric(batch[column], errors="coerce")
        if series.isna().any():
            raise RuntimeError(f"Column {column} contains missing or non-numeric values in the cleaned Census parquet.")
        integer_series[column] = series.astype(np.int64)

    age = pd.to_numeric(batch["AGE"], errors="coerce").round().astype("Int64")
    state_code = pd.to_numeric(batch["STATEFIP"], errors="coerce").round().astype("Int64")
    migrate5_code = pd.to_numeric(batch["MIGRATE5"], errors="coerce").round().astype("Int64")
    migplac5_code = pd.to_numeric(batch["MIGPLAC5"], errors="coerce").round().astype("Int64")
    release_date = pd.to_datetime(batch["release_date"], errors="coerce").dt.date.astype("string")
    release_date_source = batch["release_date_source"].astype("string").str.strip()
    if release_date.isna().any():
        raise RuntimeError("Cleaned Census parquet contains missing or invalid release_date values.")
    if release_date_source.isna().any() or release_date_source.eq("").any():
        raise RuntimeError("Cleaned Census parquet contains missing release_date_source values.")

    sex = clean_label_series(batch, "SEX_label")
    hispan = pd.to_numeric(batch["HISPAN"], errors="coerce").astype("Int64")
    race_label = clean_label_series(batch, "RACE_label")
    hispan_label = clean_label_series(batch, "HISPAN_label")
    race_ethnicity = race_label + ", " + hispan_label.str.lower()
    race_ethnicity = race_ethnicity.mask(hispan.notna() & hispan.ne(0), hispan_label + ", Hispanic")

    education = clean_label_series(batch, "EDUC_label")
    state_label = clean_label_series(batch, "STATEFIP_label")
    migplac5_label = clean_label_series(batch, "MIGPLAC5_label")

    norm = pd.DataFrame(
        {
            "id": (
                integer_series["YEAR"].astype(str)
                + "-"
                + integer_series["SAMPLE"].astype(str)
                + "-"
                + integer_series["SERIAL"].astype(str)
                + "-"
                + integer_series["PERNUM"].astype(str)
            ),
            "time": integer_series["YEAR"].astype(str) + "-01-01",
            "year": integer_series["YEAR"].astype(np.int64),
            "sample": integer_series["SAMPLE"].astype(np.int64),
            "serial": integer_series["SERIAL"].astype(np.int64),
            "pernum": integer_series["PERNUM"].astype(np.int64),
            "age": age,
            "sex": sex,
            "race_ethnicity": race_ethnicity,
            "education": education,
            "state_code": state_code,
            "migrate5_code": migrate5_code,
            "migplac5_code": migplac5_code,
            "release_date": release_date,
            "release_date_source": release_date_source,
        }
    )

    origin_state_code = pd.Series(pd.NA, index=norm.index, dtype="Int64")
    origin_state = pd.Series(pd.NA, index=norm.index, dtype="string")
    same_state_or_stay = norm["migrate5_code"].isin([1, 2])
    interstate = norm["migrate5_code"].eq(3)
    origin_state_code.loc[same_state_or_stay] = norm.loc[same_state_or_stay, "state_code"]
    origin_state_code.loc[interstate] = norm.loc[interstate, "migplac5_code"]
    origin_state.loc[same_state_or_stay] = state_label.loc[same_state_or_stay]
    origin_state.loc[interstate] = migplac5_label.loc[interstate]
    origin_year = norm["year"] - 5

    universe_condition_masks = {
        "universe_migration_year": norm["year"].isin(MIGRATION_YEARS),
        "universe_age_origin_at_least_30": norm["age"].sub(5).ge(30).fillna(False),
        "universe_migration_status_mapped": norm["migrate5_code"].isin([1, 2, 3]),
        "universe_origin_state_valid_us_state": origin_state_code.isin(sorted(VALID_US_STATE_CODES)),
    }
    universe_mask = pd.Series(True, index=norm.index)
    for filter_id, pass_mask in universe_condition_masks.items():
        pass_mask = pass_mask.astype(bool)
        step4_fail_counts[filter_id] += int((~pass_mask).sum())
        universe_mask &= pass_mask

    eligible = norm.loc[universe_mask].copy()
    if eligible.empty:
        continue

    eligible["origin_state_code"] = origin_state_code.loc[eligible.index]
    eligible["origin_state"] = origin_state.loc[eligible.index]
    eligible["origin_year"] = origin_year.loc[eligible.index]
    eligible["state_macro_reference_year"] = eligible["origin_year"] - 1
    eligible["origin_statefip"] = pd.to_numeric(eligible["origin_state_code"], errors="coerce").round().astype("Int64")
    eligible["age_origin"] = pd.to_numeric(eligible["age"] - 5, errors="coerce").round().astype("Int64")
    universe_rows += int(len(eligible))
    target_observed_rows += int(len(eligible))
    eligible["target"] = np.select(
        [
            eligible["migrate5_code"].eq(1),
            eligible["migrate5_code"].eq(2),
            eligible["migrate5_code"].eq(3),
        ],
        ["stay", "move_within_state", "move_to_different_state"],
        default="",
    )
    eligible = eligible.merge(
        state_macro,
        on=["origin_state_code", "state_macro_reference_year"],
        how="left",
        validate="many_to_one",
    )
    eligible["macro_origin_q_index"] = eligible["origin_year"].astype(int) * 4 + 1
    eligible = attach_macro_context(
        eligible,
        origin_q_index_col="macro_origin_q_index",
        required_columns=CORE_MACRO_COLUMNS,
    )
    if not eligible["state_macro_reference_year"].eq(eligible["origin_year"] - 1).all():
        raise RuntimeError(f"{TASK_SLUG}: state macro reference year is not the preceding year.")
    if not eligible["macro_reference_q_index"].eq(eligible["origin_year"] * 4).all():
        raise RuntimeError(f"{TASK_SLUG}: core macro reference is not Q4 of the preceding year.")
    predictor_condition_masks = {}
    for column in STATE_MACRO_COLUMNS:
        predictor_condition_masks[f"predictor_{column}_not_missing"] = eligible[column].notna()
    for column in CORE_MACRO_COLUMNS:
        predictor_condition_masks[f"predictor_{column}_not_missing"] = eligible[column].notna()
    for column in REQUIRED_PROMPT_COLUMNS:
        predictor_condition_masks[f"predictor_{column}_not_missing"] = eligible[column].notna()

    predictor_mask = pd.Series(True, index=eligible.index)
    for filter_id, pass_mask in predictor_condition_masks.items():
        pass_mask = pass_mask.astype(bool)
        step6_fail_counts[filter_id] += int((~pass_mask).sum())
        predictor_mask &= pass_mask

    eligible = eligible.loc[predictor_mask].copy()
    if eligible.empty:
        continue
    predictor_observed_rows += int(len(eligible))

    missing_prompt_columns = [column for column in REQUIRED_PROMPT_COLUMNS if eligible[column].isna().any()]
    if missing_prompt_columns:
        raise RuntimeError(f"{TASK_SLUG} has unmapped required prompt columns: {missing_prompt_columns}")

    eligible["subject_id"] = eligible["id"]
    eligible["selection_period"] = eligible["year"].astype(int).astype(str)
    eligible = eligible.reindex(columns=SAMPLE_COLUMNS)
    if eligible.empty:
        continue

    eligible_rows += int(len(eligible))
    for year, count in eligible["year"].value_counts().items():
        eligible_rows_by_year[int(year)] += int(count)
    # Public keys are fixed before sampling; source keys retain their sort types.
    eligible["time"] = eligible["year"].astype(int).astype(str) + "-01-01"
    eligible["release_date"] = pd.to_datetime(eligible["release_date"], errors="raise").dt.strftime(
        "%Y-%m-%d"
    )
    write_sampling_candidate_partition(
        eligible,
        task_id=TASK_SLUG,
        destination=candidate_dir / f"row_group_{row_group:05d}.parquet",
    )
    eligible.to_parquet(
        eligible_cache_dir / f"row_group_{row_group:05d}.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

sampling_sidecar, sampling_summary = build_partitioned_sampling_registry(
    candidate_dir,
    task_id=TASK_SLUG,
    output_dir=SAMPLING_DIR,
)
exported_key_index = pd.MultiIndex.from_frame(sampling_sidecar[["id", "time"]])
selected_parts = []
for path in sorted(eligible_cache_dir.glob("*.parquet")):
    eligible_part = pd.read_parquet(path)
    selected_mask = pd.MultiIndex.from_frame(eligible_part[["id", "time"]]).isin(
        exported_key_index
    )
    if selected_mask.any():
        selected_parts.append(eligible_part.loc[selected_mask].copy())
if not selected_parts:
    raise RuntimeError(f"{TASK_SLUG}: sampling registry selected zero Census rows.")
selected = pd.concat(selected_parts, ignore_index=True)
shutil.rmtree(temporary_root)
selected = selected.sort_values(["year", "id"], kind="mergesort").reset_index(drop=True)
if len(selected) != sampling_summary["exported_rows"]:
    raise RuntimeError(f"{TASK_SLUG}: cached export count differs from the sampling registry.")

missing_required = [column for column in SAMPLE_COLUMNS if selected[column].isna().any()]
if missing_required:
    raise RuntimeError(f"{TASK_SLUG} selected rows contain missing required columns: {missing_required}")
if not selected["year"].isin(MIGRATION_YEARS).all():
    raise RuntimeError(f"{TASK_SLUG} selected rows include years outside {sorted(MIGRATION_YEARS)}.")
if not selected["age_origin"].ge(30).all():
    raise RuntimeError(f"{TASK_SLUG} selected rows violate the origin-age 30+ rule.")
if not selected["target"].isin(MOBILITY_TARGETS).all():
    bad = sorted(set(selected.loc[~selected["target"].isin(MOBILITY_TARGETS), "target"].astype(str)))
    raise RuntimeError(f"{TASK_SLUG} selected rows contain invalid targets: {bad}")
year_counts = selected["year"].value_counts().sort_index().to_dict()

TABULAR_PATH.parent.mkdir(parents=True, exist_ok=True)
canonicalize_public_table(selected, task_id=TASK_SLUG).to_csv(TABULAR_PATH, index=False)
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
        int(source_rows), int(universe_rows), int(source_rows) - int(universe_rows),
    ),
    construction_step(
        TASK_SLUG, 5, "Require observed targets",
        int(universe_rows), int(target_observed_rows), int(universe_rows) - int(target_observed_rows),
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
        int(eligible_rows), int(len(selected)), int(eligible_rows) - int(len(selected)),
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
for filter_id, pass_condition in STEP6_DETAILS:
    drop_detail_rows.append(
        filter_count(
            TASK_SLUG, 6, detail_order,
            filter_id, pass_condition, int(step6_fail_counts[filter_id]),
        )
    )
    detail_order += 1
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(step_rows, columns=STEP_COLUMNS).to_csv(STEP_DIAGNOSTICS_PATH, index=False)
pd.DataFrame(drop_detail_rows, columns=DETAIL_COLUMNS).to_csv(DROP_DETAIL_PATH, index=False)
print(f"{TASK_SLUG}: eligible={eligible_rows:,} sampled={len(selected):,} tabular={TABULAR_PATH}")
print(f"{TASK_SLUG}: sampled_by_year={year_counts}")
