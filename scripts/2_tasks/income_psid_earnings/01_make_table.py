#!/usr/bin/env python
"""Build the tabular CSV for the PSID earnings task."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.sample_accounting import construction_step, filter_count

from scripts.utils.table_schema import canonicalize_public_table

from scripts.utils.psid import (
    safe_float,
    contiguous_history,
    is_missing,
    iso_date,
    load_person_panel,
    safe_int,
    validate_code_domains,
)
from scripts.utils.sampling import sample_task_records
from scripts.utils.macro_context import CORE_MACRO_COLUMNS, attach_macro_context


TASK_SLUG = "income_psid_earnings"
INPUT_PATH = PROJECT_ROOT / "data/intermediate/psid_main.parquet"
TABULAR_PATH = PROJECT_ROOT / "data" / "householdbench" / "tabular" / f"{TASK_SLUG}.csv"
DIAGNOSTICS_DIR = PROJECT_ROOT / "output" / "householdbench" / "tasks" / TASK_SLUG
SAMPLING_DIR = PROJECT_ROOT / "data" / "intermediate" / "householdbench" / TASK_SLUG
STEP_DIAGNOSTICS_PATH = DIAGNOSTICS_DIR / "01_sample_construction_steps.csv"
DROP_DETAIL_PATH = DIAGNOSTICS_DIR / "01_sample_construction_drop_details.csv"
OUTLIER_CUTOFFS_PATH = DIAGNOSTICS_DIR / "01_outlier_cutoffs.csv"
MAX_HISTORY_WAVES = 3
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

EARNINGS_HORIZONS = [2, 4, 10]
REQUIRED_MACROS = CORE_MACRO_COLUMNS
PERSON_BASE_COLS = [
    "id",
    "year",
    "release_date",
    "release_date_source",
    "response",
    "sample_person",
    "demo_age_gen",
    "demo_sex",
    "race_eth_maj_col",
    "geo_region",
    "geo_metro",
    "edu_year_max",
    "edu_level_max",
    "emp_work",
    "earn_tot_nd",
    "finc_tot_nd",
    "fam_partnered",
    "fam_size",
    "fam_size_chi",
    "occ_major_harmonized",
    "occ_source",
    "demo_sex_label",
    "race_eth_maj_col_label",
    "geo_region_label",
    "geo_metro_label",
    "edu_level_max_label",
    "emp_work_label",
    "fam_partnered_label",
    "occ_major_harmonized_label",
]
HISTORY_FIELDS = [
    "year",
    "earn_tot_nd",
    "emp_work",
    "edu_year_max",
    "edu_level_max",
    "occ_major_harmonized",
    "finc_tot_nd",
]
HISTORY_LABEL_FIELDS = ["emp_work", "edu_level_max", "occ_major_harmonized"]
HISTORY_FIELDS_WITH_LABELS = HISTORY_FIELDS + [f"{column}_label" for column in HISTORY_LABEL_FIELDS]
PERSON_LABEL_FIELDS = [
    "demo_sex",
    "race_eth_maj_col",
    "geo_region",
    "geo_metro",
    "edu_level_max",
    "emp_work",
    "fam_partnered",
    "occ_major_harmonized",
]
REQUIRED_PERSON_LABEL_FIELDS = [column for column in PERSON_LABEL_FIELDS if column not in {"geo_metro", "occ_major_harmonized"}]
CODE_DOMAINS = {
    "demo_sex": {1, 2},
    "race_eth_maj_col": {1, 2, 3, 4},
    "geo_region": {1, 2, 3, 4, 5, 6},
    "edu_level_max": {0, 1, 2, 3, 4},
    "emp_work": {0, 1},
    "fam_partnered": {0, 1},
}


if MAX_HISTORY_WAVES <= 0:
    raise SystemExit("MAX_HISTORY_WAVES must be positive.")

step_rows = []
drop_detail_rows = []


# 1. Start from source observations.
person = load_person_panel(INPUT_PATH, PERSON_BASE_COLS)
person["macro_origin_q_index"] = person["year"].astype(int) * 4 + 1
person = attach_macro_context(
    person,
    origin_q_index_col="macro_origin_q_index",
    required_columns=CORE_MACRO_COLUMNS,
)
if person.duplicated(["id", "year"]).any():
    raise RuntimeError(f"{TASK_SLUG}: PSID reference source is not unique by person and survey year.")

outlier_cutoff_rows = []
for variable, role in [
    ("earn_tot_nd", "predictor_and_target"),
    ("finc_tot_nd", "predictor"),
]:
    person[variable] = pd.to_numeric(person[variable], errors="coerce")
    reference = person.loc[person[variable].notna(), ["year", variable]].copy()
    cutoffs = (
        reference.groupby("year")[variable]
        .quantile([0.01, 0.99])
        .unstack()
        .rename(columns={0.01: f"{variable}_p01", 0.99: f"{variable}_p99"})
        .reset_index()
    )
    if cutoffs[[f"{variable}_p01", f"{variable}_p99"]].isna().any().any():
        raise RuntimeError(f"{TASK_SLUG}: missing annual cutoffs for {variable}.")
    if cutoffs[f"{variable}_p01"].gt(cutoffs[f"{variable}_p99"]).any():
        raise RuntimeError(f"{TASK_SLUG}: invalid annual cutoff order for {variable}.")
    degenerate = cutoffs[f"{variable}_p01"].eq(cutoffs[f"{variable}_p99"])
    if degenerate.any():
        bad_years = cutoffs.loc[degenerate, "year"].astype(int).tolist()
        raise RuntimeError(f"{TASK_SLUG}: degenerate annual cutoffs for {variable} in years {bad_years}.")
    person = person.merge(cutoffs, on="year", how="left", validate="many_to_one")
    for cutoff in cutoffs.itertuples(index=False):
        values = reference.loc[reference["year"].eq(cutoff.year), variable]
        p01 = getattr(cutoff, f"{variable}_p01")
        p99 = getattr(cutoff, f"{variable}_p99")
        outlier_cutoff_rows.append(
            {
                "task_id": TASK_SLUG,
                "variable": variable,
                "role": role,
                "period": int(cutoff.year),
                "n_reference": int(len(values)),
                "p01": p01,
                "p99": p99,
                "n_at_or_below_p01": int(values.le(p01).sum()),
                "n_at_or_above_p99": int(values.ge(p99).sum()),
            }
        )
step_rows.append(
    construction_step(
        TASK_SLUG, 1, "Start from source observations",
        0, int(len(person)), 0,
    )
)
step_rows.append(
    construction_step(
        TASK_SLUG, 2, "Attach additional sources",
        int(len(person)), int(len(person)), 0,
    )
)


# 3. Construct leads and lags.
rows = []
for _, group in person.groupby("id", sort=False):
    records = group.to_dict("records")
    year_map = {safe_int(record["year"]): record for record in records}
    for idx, current in enumerate(records):
        current_year = safe_int(current.get("year"))
        if current_year is None:
            raise RuntimeError(f"{TASK_SLUG}: source row has missing year after PSID loading.")
        future_records = {}
        for horizon in EARNINGS_HORIZONS:
            future_records[horizon] = year_map.get(current_year + horizon)
        future_earnings = {}
        for horizon in EARNINGS_HORIZONS:
            future_record = future_records[horizon]
            future_earn = None if future_record is None else safe_float(future_record.get("earn_tot_nd"))
            future_earnings[horizon] = None if future_earn is None else int(round(future_earn))

        history = contiguous_history(records, idx, MAX_HISTORY_WAVES)
        rows.append(
            {
                "id": int(current["id"]),
                "year": current_year,
                "current_wave_release_date": iso_date(current["release_date"]),
                "current_wave_release_date_source": current["release_date_source"],
                "year_2y": current_year + 2,
                "year_4y": current_year + 4,
                "year_10y": current_year + 10,
                "release_date": (
                    None
                    if future_records[10] is None
                    else iso_date(future_records[10]["release_date"])
                ),
                "release_date_source": (
                    None
                    if future_records[10] is None
                    else future_records[10]["release_date_source"]
                ),
                "release_date_2y": (
                    None
                    if future_records[2] is None
                    else iso_date(future_records[2]["release_date"])
                ),
                "release_date_source_2y": (
                    None if future_records[2] is None else future_records[2]["release_date_source"]
                ),
                "release_date_4y": (
                    None
                    if future_records[4] is None
                    else iso_date(future_records[4]["release_date"])
                ),
                "release_date_source_4y": (
                    None if future_records[4] is None else future_records[4]["release_date_source"]
                ),
                "release_date_10y": (
                    None
                    if future_records[10] is None
                    else iso_date(future_records[10]["release_date"])
                ),
                "release_date_source_10y": (
                    None
                    if future_records[10] is None
                    else future_records[10]["release_date_source"]
                ),
                "history_depth": len(history),
                **{
                    f"history_lag{lag}_{field}": (
                        history[-lag].get(field) if len(history) >= lag else None
                    )
                    for lag in range(1, MAX_HISTORY_WAVES + 1)
                    for field in HISTORY_FIELDS_WITH_LABELS
                    + ["earn_tot_nd_p01", "earn_tot_nd_p99", "finc_tot_nd_p01", "finc_tot_nd_p99"]
                },
                "demo_age_gen": current.get("demo_age_gen"),
                "demo_sex": current.get("demo_sex"),
                "race_eth_maj_col": current.get("race_eth_maj_col"),
                "geo_region": current.get("geo_region"),
                "geo_metro": current.get("geo_metro"),
                "edu_year_max": current.get("edu_year_max"),
                "edu_level_max": current.get("edu_level_max"),
                "emp_work": current.get("emp_work"),
                "earn_tot_nd": current.get("earn_tot_nd"),
                "finc_tot_nd": current.get("finc_tot_nd"),
                "earn_tot_nd_p01": current.get("earn_tot_nd_p01"),
                "earn_tot_nd_p99": current.get("earn_tot_nd_p99"),
                "finc_tot_nd_p01": current.get("finc_tot_nd_p01"),
                "finc_tot_nd_p99": current.get("finc_tot_nd_p99"),
                "fam_partnered": current.get("fam_partnered"),
                "fam_size": current.get("fam_size"),
                "fam_size_chi": current.get("fam_size_chi"),
                "occ_major_harmonized": current.get("occ_major_harmonized"),
                "demo_sex_label": current.get("demo_sex_label"),
                "race_eth_maj_col_label": current.get("race_eth_maj_col_label"),
                "geo_region_label": current.get("geo_region_label"),
                "geo_metro_label": current.get("geo_metro_label"),
                "edu_level_max_label": current.get("edu_level_max_label"),
                "emp_work_label": current.get("emp_work_label"),
                "fam_partnered_label": current.get("fam_partnered_label"),
                "occ_major_harmonized_label": current.get("occ_major_harmonized_label"),
                "occ_source": current.get("occ_source"),
                "occupation_observed": int(not is_missing(current.get("occ_major_harmonized"))),
                **{column: current.get(column) for column in REQUIRED_MACROS},
                "earn_tot_nd_2y": future_earnings[2],
                "earn_tot_nd_4y": future_earnings[4],
                "earn_tot_nd_10y": future_earnings[10],
                "earn_tot_nd_p01_2y": (
                    None if future_records[2] is None else future_records[2].get("earn_tot_nd_p01")
                ),
                "earn_tot_nd_p99_2y": (
                    None if future_records[2] is None else future_records[2].get("earn_tot_nd_p99")
                ),
                "earn_tot_nd_p01_4y": (
                    None if future_records[4] is None else future_records[4].get("earn_tot_nd_p01")
                ),
                "earn_tot_nd_p99_4y": (
                    None if future_records[4] is None else future_records[4].get("earn_tot_nd_p99")
                ),
                "earn_tot_nd_p01_10y": (
                    None
                    if future_records[10] is None
                    else future_records[10].get("earn_tot_nd_p01")
                ),
                "earn_tot_nd_p99_10y": (
                    None
                    if future_records[10] is None
                    else future_records[10].get("earn_tot_nd_p99")
                ),
            }
        )

pool = pd.DataFrame(rows)
step_rows.append(
    construction_step(
        TASK_SLUG, 3, "Construct leads and lags",
        int(len(person)), int(len(pool)), int(len(person)) - int(len(pool)),
    )
)

# 4. Restrict universe.
rows_before = len(pool)
condition_masks = []
detail_order = 1
for filter_id, pass_condition, pass_mask in [
    ("universe_age_not_missing", 'pool["demo_age_gen"].notna()', pool["demo_age_gen"].notna()),
    (
        "universe_age_20_to_60",
        'pool["demo_age_gen"].between(20, 60, inclusive="both")',
        pool["demo_age_gen"].between(20, 60, inclusive="both"),
    ),
]:
    condition_masks.append(pass_mask)
    drop_detail_rows.append(
        filter_count(
            TASK_SLUG, 4, detail_order,
            filter_id, pass_condition, int((~pass_mask).sum()),
        )
    )
    detail_order += 1
universe_mask = pd.Series(True, index=pool.index)
for pass_mask in condition_masks:
    universe_mask &= pass_mask
pool = pool.loc[universe_mask].copy()
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
for horizon in EARNINGS_HORIZONS:
    pass_condition = f'pool["release_date_{horizon}y"].notna()'
    pass_mask = pool[f"release_date_{horizon}y"].notna()
    condition_masks.append(pass_mask)
    drop_detail_rows.append(
        filter_count(
            TASK_SLUG, 5, detail_order,
            f"target_exact_{horizon}y_record_observed", pass_condition, int((~pass_mask).sum()),
        )
    )
    detail_order += 1
    pass_condition = f'pool["earn_tot_nd_{horizon}y"].notna()'
    pass_mask = pool[f"earn_tot_nd_{horizon}y"].notna()
    condition_masks.append(pass_mask)
    drop_detail_rows.append(
        filter_count(
            TASK_SLUG, 5, detail_order,
            f"target_earn_tot_nd_{horizon}y_not_missing", pass_condition, int((~pass_mask).sum()),
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
condition_masks = []
detail_order = 1
predictor_columns = ["earn_tot_nd", "finc_tot_nd", "edu_year_max", "edu_level_max", *REQUIRED_MACROS]
predictor_columns += [f"{column}_label" for column in REQUIRED_PERSON_LABEL_FIELDS]
for column in predictor_columns:
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
step_rows.append(
    construction_step(
        TASK_SLUG, 7, "Apply logical and validity filters",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)

# 8. Apply outlier trimming.
rows_before = len(pool)
required_cutoffs = [
    "earn_tot_nd_p01",
    "earn_tot_nd_p99",
    "finc_tot_nd_p01",
    "finc_tot_nd_p99",
    *[f"earn_tot_nd_p01_{horizon}y" for horizon in EARNINGS_HORIZONS],
    *[f"earn_tot_nd_p99_{horizon}y" for horizon in EARNINGS_HORIZONS],
]
if pool[required_cutoffs].isna().any().any():
    raise RuntimeError(f"{TASK_SLUG}: missing annual cutoffs for required current or future values.")
trim_conditions = [
    (
        "trim_earnings_current_strict_inside_year_p01_p99",
        'pool["earn_tot_nd"].gt(pool["earn_tot_nd_p01"]) & pool["earn_tot_nd"].lt(pool["earn_tot_nd_p99"])',
        pool["earn_tot_nd"].gt(pool["earn_tot_nd_p01"])
        & pool["earn_tot_nd"].lt(pool["earn_tot_nd_p99"]),
    ),
    (
        "trim_family_income_current_strict_inside_year_p01_p99",
        'pool["finc_tot_nd"].gt(pool["finc_tot_nd_p01"]) & pool["finc_tot_nd"].lt(pool["finc_tot_nd_p99"])',
        pool["finc_tot_nd"].gt(pool["finc_tot_nd_p01"])
        & pool["finc_tot_nd"].lt(pool["finc_tot_nd_p99"]),
    ),
]
for horizon in EARNINGS_HORIZONS:
    trim_conditions.append(
        (
            f"trim_earnings_{horizon}y_target_strict_inside_target_year_p01_p99",
            f'pool["earn_tot_nd_{horizon}y"].gt(pool["earn_tot_nd_p01_{horizon}y"]) & pool["earn_tot_nd_{horizon}y"].lt(pool["earn_tot_nd_p99_{horizon}y"])',
            pool[f"earn_tot_nd_{horizon}y"].gt(pool[f"earn_tot_nd_p01_{horizon}y"])
            & pool[f"earn_tot_nd_{horizon}y"].lt(pool[f"earn_tot_nd_p99_{horizon}y"]),
        )
    )
for lag in range(1, MAX_HISTORY_WAVES + 1):
    for variable, label in [("earn_tot_nd", "earnings"), ("finc_tot_nd", "family_income")]:
        value = pool[f"history_lag{lag}_{variable}"]
        p01 = pool[f"history_lag{lag}_{variable}_p01"]
        p99 = pool[f"history_lag{lag}_{variable}_p99"]
        value = pd.to_numeric(value, errors="coerce")
        p01 = pd.to_numeric(p01, errors="coerce")
        p99 = pd.to_numeric(p99, errors="coerce")
        displayed = pool["history_depth"].ge(lag)
        missing_cutoff = displayed & value.notna() & (p01.isna() | p99.isna())
        if missing_cutoff.any():
            raise RuntimeError(f"{TASK_SLUG}: missing lag-{lag} annual cutoffs for {variable}.")
        pass_mask = ~displayed | value.isna() | (value.gt(p01) & value.lt(p99))
        trim_conditions.append(
            (
                f"trim_{label}_lag{lag}_strict_inside_lag_year_p01_p99",
                f"history_depth < {lag} or lag{lag}_{variable} is missing or strictly inside its lag-year p01/p99",
                pass_mask,
            )
        )

trim_mask = pd.Series(True, index=pool.index)
for detail_order, (filter_id, pass_condition, pass_mask) in enumerate(trim_conditions, start=1):
    trim_mask &= pass_mask
    drop_detail_rows.append(
        filter_count(
            TASK_SLUG, 8, detail_order,
            filter_id, pass_condition, int((~pass_mask).sum()),
        )
    )
pool = pool.loc[trim_mask].copy()
step_rows.append(
    construction_step(
        TASK_SLUG, 8, "Apply outlier trimming",
        int(rows_before), int(len(pool)), int(rows_before) - int(len(pool)),
    )
)
if pool.empty:
    raise RuntimeError(f"{TASK_SLUG} has zero eligible rows.")
if pool[REQUIRED_MACROS].isna().any().any():
    missing_cols = [column for column in REQUIRED_MACROS if pool[column].isna().any()]
    raise RuntimeError(f"{TASK_SLUG} contains missing current-year macro values in {missing_cols}.")
if not pool["demo_age_gen"].between(20, 60, inclusive="both").all():
    raise RuntimeError("Working-age earnings pool contains ages outside 20-60.")
exact_horizons = (
    ((pool["year_2y"] - pool["year"]) == 2).all()
    and ((pool["year_4y"] - pool["year"]) == 4).all()
    and ((pool["year_10y"] - pool["year"]) == 10).all()
)
if not exact_horizons:
    raise RuntimeError("Working-age earnings pool violates exact horizon requirements.")
if pool[["earn_tot_nd_2y", "earn_tot_nd_4y", "earn_tot_nd_10y"]].isna().any().any():
    raise RuntimeError("Working-age earnings pool contains missing headline labels.")
required_label_columns = [f"{column}_label" for column in REQUIRED_PERSON_LABEL_FIELDS]
missing_required_labels = pool[required_label_columns].isna().any(axis=1)
required_label_filter_dropped = int(missing_required_labels.sum())
if required_label_filter_dropped:
    raise RuntimeError(f"{TASK_SLUG}: required-label filter should already be handled in step 6.")
if pool.empty:
    raise RuntimeError(f"{TASK_SLUG}: no rows remain after required-label filtering.")
validate_code_domains(pool, CODE_DOMAINS, TASK_SLUG)

pool = pool.copy()
pool["selection_period"] = pool["year"].astype(int).astype(str)
# Public keys are fixed before sampling; source keys retain their sort types.
pool["sampling_unit_id"] = pool["id"]
pool["id"] = pool["id"].astype("string")
pool["time"] = pool["year"].astype(int).astype(str) + "-01-01"
pool["release_date"] = pd.to_datetime(pool["release_date"], errors="raise").dt.strftime("%Y-%m-%d")
sample, sampling_registry, sampling_summary = sample_task_records(
    pool,
    task_id=TASK_SLUG,
    output_dir=SAMPLING_DIR,
    output_sort_cols=["year", "sampling_unit_id"],
)

out = sample.copy().reset_index(drop=True)
out["subject_id"] = out["id"].astype(str)
for column in PERSON_LABEL_FIELDS:
    label_column = f"{column}_label"
    if column in REQUIRED_PERSON_LABEL_FIELDS and out[label_column].isna().any():
        raise RuntimeError(f"{TASK_SLUG}: labelled predictor {label_column} has missing values.")
    out[column] = out[label_column].astype("string")
for column in HISTORY_LABEL_FIELDS:
    for lag in range(1, 4):
        history_column = f"history_lag{lag}_{column}"
        history_label_column = f"{history_column}_label"
        out[history_column] = out[history_label_column].astype("string")

TABULAR_PATH.parent.mkdir(parents=True, exist_ok=True)
canonicalize_public_table(out, task_id=TASK_SLUG).to_csv(TABULAR_PATH, index=False)
# 9. Export sample.
step_rows.append(
    construction_step(
        TASK_SLUG, 9, "Export sample",
        int(len(pool)), int(len(sample)), int(len(pool)) - int(len(sample)),
    )
)

DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(step_rows, columns=STEP_COLUMNS).to_csv(STEP_DIAGNOSTICS_PATH, index=False)
pd.DataFrame(drop_detail_rows, columns=DETAIL_COLUMNS).to_csv(DROP_DETAIL_PATH, index=False)
pd.DataFrame(outlier_cutoff_rows, columns=OUTLIER_CUTOFF_COLUMNS).to_csv(OUTLIER_CUTOFFS_PATH, index=False)
print(
    f"{TASK_SLUG}: eligible={len(pool):,} sampled={len(sample):,} "
    f"tabular={TABULAR_PATH}"
)
