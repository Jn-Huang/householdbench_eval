#!/usr/bin/env python
"""Build the tabular CSV for the PSID wealth task."""

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
    PSID_WEALTH_HISTORY_VALUE_SPECS,
    contiguous_history,
    contiguous_next,
    iso_date,
    load_family_panel,
    safe_float,
    safe_int,
    validate_code_domains,
)
from scripts.utils.sampling import sample_task_records
from scripts.utils.macro_context import CORE_MACRO_COLUMNS, PSID_ASSET_COLUMNS, attach_macro_context


TASK_SLUG = "cons_psid_wealth"
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

REQUIRED_MACROS = [*CORE_MACRO_COLUMNS, *PSID_ASSET_COLUMNS]
FAMILY_BASE_COLS = [
    "id",
    "year",
    "release_date",
    "release_date_source",
    "response",
    "is_reference_person",
    "demo_age_gen",
    "demo_sex",
    "race_eth_maj_col",
    "geo_region",
    "geo_metro",
    "finc_tot_nd",
    "fam_partnered",
    "fam_married",
    "fam_size",
    "fam_size_chi",
    "home_stat",
    "wlth_tot_net_nd",
    "wlth_savi_net_nd",
    "wlth_inve_net_nd",
    "wlth_home_net_nd",
    "wlth_odeb_net_nd",
]
FAMILY_LABEL_FIELDS = [
    "demo_sex",
    "race_eth_maj_col",
    "geo_region",
    "geo_metro",
    "home_stat",
    "fam_partnered",
    "fam_married",
]
FAMILY_LABEL_COLUMNS = [f"{column}_label" for column in FAMILY_LABEL_FIELDS]
FAMILY_BASE_COLS = FAMILY_BASE_COLS + FAMILY_LABEL_COLUMNS
REQUIRED_FAMILY_LABEL_FIELDS = [column for column in FAMILY_LABEL_FIELDS if column != "geo_metro"]
HISTORY_FIELDS = [
    "year",
    "finc_tot_nd",
    "wlth_tot_net_nd",
    "wlth_savi_net_nd",
    "wlth_inve_net_nd",
    "wlth_home_net_nd",
    "wlth_odeb_net_nd",
    "home_stat",
    "fam_partnered",
    "fam_married",
    "fam_size",
    "fam_size_chi",
]
HISTORY_LABEL_FIELDS = ["home_stat", "fam_partnered", "fam_married"]
HISTORY_FIELDS_WITH_LABELS = HISTORY_FIELDS + [f"{column}_label" for column in HISTORY_LABEL_FIELDS]
WEALTH_HISTORY_EXPORT_COLUMNS = [
    f"history_lag{lag}_{field}"
    for lag in range(1, MAX_HISTORY_WAVES + 1)
    for field, _label, _kind in PSID_WEALTH_HISTORY_VALUE_SPECS
]
CODE_DOMAINS = {
    "demo_sex": {1, 2},
    "race_eth_maj_col": {1, 2, 3, 4},
    "geo_region": {1, 2, 3, 4, 5, 6},
    "home_stat": {1, 2, 3},
    "fam_partnered": {0, 1},
    "fam_married": {0, 1},
}


if MAX_HISTORY_WAVES <= 0:
    raise SystemExit("MAX_HISTORY_WAVES must be positive.")

step_rows = []
drop_detail_rows = []


# 1. Start from source observations.
family = load_family_panel(INPUT_PATH, FAMILY_BASE_COLS)
family["macro_origin_q_index"] = family["year"].astype(int) * 4 + 1
family = attach_macro_context(
    family,
    origin_q_index_col="macro_origin_q_index",
    required_columns=CORE_MACRO_COLUMNS,
    optional_columns=PSID_ASSET_COLUMNS,
)
if family.duplicated(["id", "year"]).any():
    raise RuntimeError(f"{TASK_SLUG}: PSID reference source is not unique by family and survey year.")

outlier_cutoff_rows = []
for variable, role in [
    ("finc_tot_nd", "predictor"),
    ("wlth_tot_net_nd", "predictor_and_target"),
]:
    family[variable] = pd.to_numeric(family[variable], errors="coerce")
    reference = family.loc[family[variable].notna(), ["year", variable]].copy()
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
    family = family.merge(cutoffs, on="year", how="left", validate="many_to_one")
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
        0, int(len(family)), 0,
    )
)
step_rows.append(
    construction_step(
        TASK_SLUG, 2, "Attach additional sources",
        int(len(family)), int(len(family)), 0,
    )
)


# 3. Construct leads and lags.
rows = []
for _, group in family.groupby("id", sort=False):
    records = group.to_dict("records")
    for idx, current in enumerate(records):
        next_rec = contiguous_next(records, idx)
        current_age = safe_int(current.get("demo_age_gen"))
        current_wealth = safe_float(current.get("wlth_tot_net_nd"))
        next_wealth = None if next_rec is None else safe_float(next_rec.get("wlth_tot_net_nd"))
        current_income = safe_float(current.get("finc_tot_nd"))

        history = contiguous_history(records, idx, MAX_HISTORY_WAVES)
        lag1 = history[-1] if history else None
        lag1_wealth_change = None
        lag1_savings_change = None
        if lag1 is not None:
            lag1_wealth = safe_float(lag1.get("wlth_tot_net_nd"))
            lag1_savings = safe_float(lag1.get("wlth_savi_net_nd"))
            current_savings = safe_float(current.get("wlth_savi_net_nd"))
            if current_wealth is not None and lag1_wealth is not None:
                lag1_wealth_change = current_wealth - lag1_wealth
            if lag1_savings is not None and current_savings is not None:
                lag1_savings_change = current_savings - lag1_savings

        rows.append(
            {
                "id": int(current["id"]),
                "year": int(current["year"]),
                "current_wave_release_date": iso_date(current["release_date"]),
                "current_wave_release_date_source": current["release_date_source"],
                "next_year": None if next_rec is None else int(next_rec["year"]),
                "release_date": None if next_rec is None else iso_date(next_rec["release_date"]),
                "release_date_source": (
                    None if next_rec is None else next_rec["release_date_source"]
                ),
                "next_wave_release_date": (
                    None if next_rec is None else iso_date(next_rec["release_date"])
                ),
                "next_wave_release_date_source": (
                    None if next_rec is None else next_rec["release_date_source"]
                ),
                "history_depth": len(history),
                **{
                    f"history_lag{lag}_{field}": (
                        history[-lag].get(field) if len(history) >= lag else None
                    )
                    for lag in range(1, MAX_HISTORY_WAVES + 1)
                    for field in HISTORY_FIELDS_WITH_LABELS
                    + [
                        "finc_tot_nd_p01",
                        "finc_tot_nd_p99",
                        "wlth_tot_net_nd_p01",
                        "wlth_tot_net_nd_p99",
                    ]
                },
                "demo_age_gen": current.get("demo_age_gen"),
                "demo_sex": current.get("demo_sex"),
                "race_eth_maj_col": current.get("race_eth_maj_col"),
                "geo_region": current.get("geo_region"),
                "geo_metro": current.get("geo_metro"),
                "finc_tot_nd": current.get("finc_tot_nd"),
                "wlth_tot_net_nd": current.get("wlth_tot_net_nd"),
                "wlth_savi_net_nd": current.get("wlth_savi_net_nd"),
                "wlth_inve_net_nd": current.get("wlth_inve_net_nd"),
                "wlth_home_net_nd": current.get("wlth_home_net_nd"),
                "wlth_odeb_net_nd": current.get("wlth_odeb_net_nd"),
                "home_stat": current.get("home_stat"),
                "fam_partnered": current.get("fam_partnered"),
                "fam_married": current.get("fam_married"),
                "fam_size": current.get("fam_size"),
                "fam_size_chi": current.get("fam_size_chi"),
                "demo_sex_label": current.get("demo_sex_label"),
                "race_eth_maj_col_label": current.get("race_eth_maj_col_label"),
                "geo_region_label": current.get("geo_region_label"),
                "geo_metro_label": current.get("geo_metro_label"),
                "home_stat_label": current.get("home_stat_label"),
                "fam_partnered_label": current.get("fam_partnered_label"),
                "fam_married_label": current.get("fam_married_label"),
                **{column: current.get(column) for column in REQUIRED_MACROS},
                "lag1_wealth_change": lag1_wealth_change,
                "lag1_savings_asset_change": lag1_savings_change,
                "finc_tot_nd_p01": current.get("finc_tot_nd_p01"),
                "finc_tot_nd_p99": current.get("finc_tot_nd_p99"),
                "wlth_tot_net_nd_p01": current.get("wlth_tot_net_nd_p01"),
                "wlth_tot_net_nd_p99": current.get("wlth_tot_net_nd_p99"),
                "wlth_tot_net_nd_p01_next_wave": (
                    None if next_rec is None else next_rec.get("wlth_tot_net_nd_p01")
                ),
                "wlth_tot_net_nd_p99_next_wave": (
                    None if next_rec is None else next_rec.get("wlth_tot_net_nd_p99")
                ),
                "wlth_tot_net_nd_next_wave": (
                    None if next_wealth is None else int(round(next_wealth))
                ),
            }
        )

pool = pd.DataFrame(rows)
step_rows.append(
    construction_step(
        TASK_SLUG, 3, "Construct leads and lags",
        int(len(family)), int(len(pool)), int(len(family)) - int(len(pool)),
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
    ("target_contiguous_next_wave", 'pool["next_year"].notna()', pool["next_year"].notna()),
    (
        "target_wlth_tot_net_nd_next_wave_not_missing",
        'pool["wlth_tot_net_nd_next_wave"].notna()',
        pool["wlth_tot_net_nd_next_wave"].notna(),
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
condition_masks = []
detail_order = 1
predictor_columns = ["demo_age_gen", "finc_tot_nd", "wlth_tot_net_nd", *REQUIRED_MACROS]
predictor_columns += [f"{column}_label" for column in REQUIRED_FAMILY_LABEL_FIELDS]
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
    "finc_tot_nd_p01",
    "finc_tot_nd_p99",
    "wlth_tot_net_nd_p01",
    "wlth_tot_net_nd_p99",
    "wlth_tot_net_nd_p01_next_wave",
    "wlth_tot_net_nd_p99_next_wave",
]
if pool[required_cutoffs].isna().any().any():
    raise RuntimeError(f"{TASK_SLUG}: missing annual cutoffs for required current or next-wave values.")
trim_conditions = [
    (
        "trim_income_current_strict_inside_year_p01_p99",
        'pool["finc_tot_nd"].gt(pool["finc_tot_nd_p01"]) & pool["finc_tot_nd"].lt(pool["finc_tot_nd_p99"])',
        pool["finc_tot_nd"].gt(pool["finc_tot_nd_p01"])
        & pool["finc_tot_nd"].lt(pool["finc_tot_nd_p99"]),
    ),
    (
        "trim_wealth_current_strict_inside_year_p01_p99",
        'pool["wlth_tot_net_nd"].gt(pool["wlth_tot_net_nd_p01"]) & pool["wlth_tot_net_nd"].lt(pool["wlth_tot_net_nd_p99"])',
        pool["wlth_tot_net_nd"].gt(pool["wlth_tot_net_nd_p01"])
        & pool["wlth_tot_net_nd"].lt(pool["wlth_tot_net_nd_p99"]),
    ),
    (
        "trim_wealth_next_wave_strict_inside_next_year_p01_p99",
        'pool["wlth_tot_net_nd_next_wave"].gt(pool["wlth_tot_net_nd_p01_next_wave"]) & pool["wlth_tot_net_nd_next_wave"].lt(pool["wlth_tot_net_nd_p99_next_wave"])',
        pool["wlth_tot_net_nd_next_wave"].gt(pool["wlth_tot_net_nd_p01_next_wave"])
        & pool["wlth_tot_net_nd_next_wave"].lt(pool["wlth_tot_net_nd_p99_next_wave"]),
    ),
]
for lag in range(1, MAX_HISTORY_WAVES + 1):
    for variable, label in [("finc_tot_nd", "income"), ("wlth_tot_net_nd", "wealth")]:
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
required_label_columns = [f"{column}_label" for column in REQUIRED_FAMILY_LABEL_FIELDS]
missing_required_labels = pool[required_label_columns].isna().any(axis=1)
required_label_filter_dropped = int(missing_required_labels.sum())
if required_label_filter_dropped:
    raise RuntimeError(f"{TASK_SLUG}: required-label filter should already be handled in step 6.")
if pool.empty:
    raise RuntimeError(f"{TASK_SLUG}: no rows remain after required-label filtering.")
validate_code_domains(pool, CODE_DOMAINS, TASK_SLUG)
wave_gaps = pool["next_year"].astype(int) - pool["year"].astype(int)
if not wave_gaps.isin([1, 2]).all():
    raise RuntimeError(f"{TASK_SLUG}: next-wave gap is not one or two years for every eligible row.")

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
wave_gaps = out["next_year"].astype(int) - out["year"].astype(int)
if not wave_gaps.isin([1, 2]).all():
    raise RuntimeError(f"{TASK_SLUG}: next-wave gap is not one or two years for every exported row.")
out["subject_id"] = out["id"].astype(str)
missing_history_columns = sorted(set(WEALTH_HISTORY_EXPORT_COLUMNS) - set(out.columns))
if missing_history_columns:
    raise RuntimeError(
        f"{TASK_SLUG}: required wealth-history columns are absent: {missing_history_columns}"
    )
for lag in range(1, 4):
    history_year = f"history_lag{lag}_year"
    years_ago = f"history_lag{lag}_years_ago"
    out[years_ago] = out["year"] - out[history_year]
    observed = out[years_ago].dropna()
    if (observed <= 0).any() or (observed % 1 != 0).any():
        raise RuntimeError(f"{TASK_SLUG}: {years_ago} must contain positive integers.")
    out[years_ago] = out[years_ago].astype("Int64")
for column in FAMILY_LABEL_FIELDS:
    label_column = f"{column}_label"
    if column in REQUIRED_FAMILY_LABEL_FIELDS and out[label_column].isna().any():
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
    f"dropped={required_label_filter_dropped:,} "
    f"tabular={TABULAR_PATH}"
)
