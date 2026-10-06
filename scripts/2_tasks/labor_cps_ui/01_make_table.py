#!/usr/bin/env python
"""Build the strict-forward CPS UI-duration task from documented inputs only.

The Farber--Rothstein--Valletta policy file records state-level maximum
availability as of the fifth day of each month.  This task attaches that
baseline-month value, rather than the paper-estimation code's forward-month
value, because a prompt at month t may not reveal a realised policy state in
month t+1.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.sample_accounting import construction_step, filter_count

from scripts.utils.io import sha256_file

from scripts.utils import cps
from scripts.utils.sampling import sample_task_records
from scripts.utils.table_schema import write_public_table


TASK_ID = "labor_cps_ui"
SEED = 42011
MAX_PROMPTS = 500_000
EXPECTED_SELECTED_ROWS = 158_989
POLICY_FIRST = (2008, 1)
POLICY_LAST = (2014, 8)
SCAN_FIRST_TARGET = (2008, 1)
SCAN_LAST_TARGET = (2015, 1)
TARGETS = {"employed", "unemployed", "not_in_labor_force"}
ADJACENT_BASELINE_MISH = {1, 2, 3, 5, 6, 7}
JOB_LOSER_CODES = {1, 2, 3}
WHYUNEMP_TEXT = {
    1: "you were on temporary layoff from a job",
    2: "you lost a job for another reason",
    3: "a temporary job ended",
}
WKSTAT_TEXT = {
    "unemployed, seeking full-time work": "You are seeking full-time work.",
    "unemployed, seeking part-time work": "You are seeking part-time work.",
}
WNFTLOOK_TEXT = {
    "less than 5 years ago": "less than 5 years ago",
    "within the last 12 months": "within the last 12 months",
    "one to five years ago": "one to five years ago",
    "more than 12 months ago": "more than 12 months ago",
    "more than 5 years ago": "more than 5 years ago",
    "never worked": "never",
    "never worked full-time for 2 or more weeks": "never",
    "never worked at all": "never",
}
SEARCH_LAYOFF_STATUS = {
    1: "temporary_layoff",
    2: "active_search",
    3: "active_search",
}
OCCUPATION_LABELS = {
    1: "management occupations",
    2: "professional occupations",
    3: "technical occupations",
    4: "sales occupations",
    5: "administrative support occupations",
    7: "protective-service occupations",
    8: "other service occupations",
    9: "production, craft, and repair occupations",
    10: "machine-operating and production occupations",
    11: "transportation and material-moving occupations",
    12: "laboring and cleaning occupations",
    13: "farming occupations",
    14: "the armed forces",
}
INDUSTRY_LABELS = {
    1: "agriculture",
    2: "mining",
    3: "construction",
    4: "nondurable-goods manufacturing",
    5: "durable-goods manufacturing",
    6: "transportation, communications, and public utilities",
    7: "wholesale trade",
    8: "retail trade",
    9: "finance, insurance, and real estate",
    10: "business services",
    11: "personal services",
    12: "entertainment services",
    13: "professional services",
    14: "government",
    15: "the armed forces",
}

BASIC_PATH = PROJECT_ROOT / "data/intermediate/cps_basic.parquet"
POLICY_PATH = PROJECT_ROOT / "data/intermediate/cps_ui_policy.parquet"
POLICY_BUILD_SUMMARY = PROJECT_ROOT / "output/preprocessing/cps_ui_policy/1_build_summary.csv"
TABLE_PATH = PROJECT_ROOT / "data/householdbench/tabular/labor_cps_ui.csv"
OUTPUT_DIR = PROJECT_ROOT / "output/householdbench/tasks/labor_cps_ui"
SAMPLING_DIR = PROJECT_ROOT / "data" / "intermediate" / "householdbench" / TASK_ID

POLICY_JOIN_RULE = "baseline_CPS_month_to_panel_status_as_of_month_day_5"
POLICY_SOURCE = "Farber_Rothstein_Valletta_2015_eui_state_08_14"

STEP_COLUMNS = ["task_id", "step", "step_label", "rows_before", "rows_after", "rows_dropped"]
DETAIL_COLUMNS = ["task_id", "step", "detail_order", "filter_id", "pass_condition", "fail_count"]

BASE_COLUMNS = [
    "YEAR", "MONTH", "CPSIDP", "CPSIDV", "MISH", "AGE", "EMPSTAT", "NILFACT", "SEX", "RACE",
    "EDUC", "MARST", "STATEFIP", "REGION", "METRO", "RELATE", "FAMSIZE", "NCHILD", "HISPAN",
    "NATIVITY", "CITIZEN", "VETSTAT", "CLASSWKR", "WKSTAT", "DURUNEMP", "WHYUNEMP", "WNFTLOOK", "WTFINL",
    "OCC", "IND",
    "sex", "race", "educ", "marst", "state", "region", "metro", "relate", "hispan", "nativity",
    "citizen", "vetstat", "classwkr", "wkstat", "wnftlook", "release_date", "release_date_source",
]
PROMPT_COLUMNS = [
    "age", "sex", "race", "educ", "marst", "state", "region", "metro", "famsize", "nchild",
    "hispan", "nativity", "citizen", "vetstat", "whyunemp_text", "search_layoff_status", "wkstat_baseline",
    "durunemp_baseline", "last_job_occupation", "last_job_industry", "maximum_ui_weeks", "regular_ui_weeks",
    "extension_ui_weeks", *cps.CORE_MACRO_COLUMNS,
]
FIELDNAMES = [
    "row_id", "subject_id", "group_id", "person_month_id", "target_person_month_id", "spell_id",
    "spell_id_method", "state_month_policy_id", "selection_period", "baseline_year", "baseline_month",
    "outcome_year", "outcome_month", "release_date", "release_date_source", "mish_baseline", "mish_target",
    "policy_state_fips", "policy_effective_date", "policy_join_rule", "policy_source", "policy_archive_version",
    "policy_panel_sha256", "source_active_extension_weeks", "policy_merge", "macro_context_merge", "paper_universe",
    "direct_target_observed", "predictor_complete",
    "logical_valid", "outlier_retained", "baseline_status", "target", "analysis_weight", "age", "age_group",
    "sex", "race", "educ", "marst", "state", "region", "metro", "relate", "famsize", "nchild", "hispan",
    "nativity", "citizen", "citizenship_nativity_rendered", "vetstat", "veteran_status_rendered",
    "relationship_rendered", "whyunemp_code", "whyunemp_text", "search_layoff_status", "wkstat_baseline",
    "classwkr", "classwkr_rendered",
    "durunemp_baseline", "wnftlook_baseline", "last_full_time_work_rendered", "last_job_source",
    "last_job_occupation_code", "last_job_occupation", "last_job_occupation_rendered",
    "last_job_industry_code", "last_job_industry", "last_job_industry_rendered",
    "regular_ui_weeks", "extension_ui_weeks", "maximum_ui_weeks",
    *cps.CORE_MACRO_COLUMNS,
]


def numeric_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def month_key(year: int, month: int) -> int:
    return year * 12 + month - 1


def broad_occupation(code: object) -> tuple[int, str] | tuple[None, None]:
    value = cps.safe_int(code)
    category = None
    if value is None:
        return None, None
    if 10 <= value <= 950:
        category = 1
    elif 1000 <= value <= 2960:
        category = 2
    elif 4700 <= value <= 4965:
        category = 4
    elif 5000 <= value <= 5940:
        category = 5
    elif 3700 <= value <= 3955:
        category = 7
    elif 3600 <= value <= 3655 or 4000 <= value <= 4650:
        category = 8
    elif 6200 <= value <= 7630:
        category = 9
    elif 7700 <= value <= 8965:
        category = 10
    elif 9000 <= value <= 9750:
        category = 11
    elif 6000 <= value <= 6130:
        category = 13
    elif value == 9840:
        category = 14
    if value in {6260, 6600, 6930, 7610, 8950, 9610, 9620, 9640}:
        category = 12
    if value in {1550, 1560, 2140, 2900, 9030, 9040} or 3300 <= value <= 3540:
        category = 3
    if 3000 <= value <= 3260:
        category = 2
    if 3600 <= value <= 3650:
        category = 8
    return (category, OCCUPATION_LABELS[category]) if category is not None else (None, None)


def broad_industry(code: object) -> tuple[int, str] | tuple[None, None]:
    value = cps.safe_int(code)
    category = None
    if value is None:
        return None, None
    if 170 <= value <= 290:
        category = 1
    elif 370 <= value <= 490:
        category = 2
    elif value == 770:
        category = 3
    elif 1070 <= value <= 2390:
        category = 4
    elif 2470 <= value <= 3990:
        category = 5
    elif 6070 <= value <= 6780:
        category = 6
    elif 4070 <= value <= 4590:
        category = 7
    elif 4670 <= value <= 5790:
        category = 8
    elif 6870 <= value <= 7190:
        category = 9
    elif 8560 <= value <= 8590:
        category = 12
    elif 9370 <= value <= 9590:
        category = 14
    if 7270 <= value <= 7570 or 7860 <= value <= 8470 or 9160 <= value <= 9190:
        category = 13
    if 7580 <= value <= 7790 or 8770 <= value <= 8870 or value in {7380, 7470}:
        category = 10
    if 8660 <= value <= 8690 or 8880 <= value <= 9090 or value == 9290:
        category = 11
    if value in {570, 580, 590, 670, 680, 690}:
        category = 6
    if value == 9890:
        category = 15
    return (category, INDUSTRY_LABELS[category]) if category is not None else (None, None)


def iter_task_months(path: Path, columns: list[str]):
    """Read only row groups needed to evaluate the policy window and its lead."""
    parquet = pq.ParquetFile(path)
    names = parquet.schema.names
    year_index = names.index("YEAR")
    month_index = names.index("MONTH")
    minimum = month_key(*SCAN_FIRST_TARGET)
    maximum = month_key(*SCAN_LAST_TARGET)
    current_key: int | None = None
    pieces: list[pd.DataFrame] = []
    for group_number in range(parquet.num_row_groups):
        group = parquet.metadata.row_group(group_number)
        year_stats = group.column(year_index).statistics
        month_stats = group.column(month_index).statistics
        if year_stats is None or month_stats is None or not year_stats.has_min_max or not month_stats.has_min_max:
            raise RuntimeError(f"{TASK_ID}: CPS row group {group_number} lacks year/month statistics.")
        group_first = month_key(int(year_stats.min), int(month_stats.min))
        group_last = month_key(int(year_stats.max), int(month_stats.max))
        if group_first > group_last:
            raise RuntimeError(f"{TASK_ID}: CPS row group {group_number} has invalid chronological statistics.")
        if group_last < minimum or group_first > maximum:
            continue
        raw = parquet.read_row_group(group_number, columns=columns).to_pandas()
        raw["_month_key"] = numeric_int(raw["YEAR"]) * 12 + numeric_int(raw["MONTH"]) - 1
        raw = raw.loc[raw["_month_key"].between(minimum, maximum)].copy()
        for key, part in raw.groupby("_month_key", sort=True, observed=True):
            key_int = int(key)
            part = part.drop(columns=["_month_key"])
            if current_key is None:
                current_key, pieces = key_int, [part]
            elif key_int == current_key:
                pieces.append(part)
            else:
                if key_int < current_key:
                    raise RuntimeError(f"{TASK_ID}: CPS data are not chronological by actual year and month.")
                yield current_key, pd.concat(pieces, ignore_index=True)
                current_key, pieces = key_int, [part]
    if current_key is not None:
        yield current_key, pd.concat(pieces, ignore_index=True)


def clean_month(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    for column in ["YEAR", "MONTH", "MISH", "AGE", "EMPSTAT", "NILFACT", "SEX", "RACE", "STATEFIP", "WHYUNEMP"]:
        out[f"_{column.lower()}"] = numeric_int(out[column])
    cpsidp = pd.to_numeric(out["CPSIDP"], errors="coerce")
    cpsidp_round = cpsidp.round()
    out["_cpsidp"] = cpsidp_round.astype("Int64")
    out["_positive_cpsidp"] = cpsidp.notna() & cpsidp.gt(0) & (cpsidp - cpsidp_round).abs().lt(1e-6)
    out["_unique_cpsidp"] = out["_positive_cpsidp"] & ~out.duplicated("_cpsidp", keep=False)
    out["_status"] = [cps.broad_status_from_empstat(value) for value in out["EMPSTAT"]]
    return out


def one_month_identity_status(baseline: dict[str, object], target: dict[str, object]) -> str | None:
    if baseline.get("cpsidp") is None or target.get("cpsidp") is None or baseline["cpsidp"] != target["cpsidp"]:
        return None
    if baseline.get("mish") is None or target.get("mish") != baseline["mish"] + 1:
        return None
    if baseline.get("cpsidv") is not None and target.get("cpsidv") is not None and baseline["cpsidv"] != target["cpsidv"]:
        return None
    if baseline.get("sex_code") != target.get("sex_code") or baseline.get("race_code") != target.get("race_code"):
        return None
    base_age = baseline.get("age")
    target_age = target.get("age")
    if base_age is None or target_age is None or int(target_age) - int(base_age) not in {0, 1}:
        return None
    return "cpsidp_cpsidv_sex_race_age_mish_match"


def duration_band(values: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[-1, 4, 12, 26, 52, np.inf],
        labels=["0_4", "5_12", "13_26", "27_52", "53_plus"],
    ).astype("string")


def match_observations(universe: pd.DataFrame) -> pd.DataFrame:
    baseline_key = universe["b__positive_cpsidp"].eq(True) & universe["b__unique_cpsidp"].eq(True)
    target_key = (
        baseline_key
        & universe["t__positive_cpsidp"].eq(True)
        & universe["t__unique_cpsidp"].eq(True)
        & universe["t__cpsidp"].notna()
    )
    direct_target = target_key & universe["t__status"].isin(TARGETS)
    identity_match = []
    for is_direct, payload in zip(direct_target.tolist(), universe.to_dict("records"), strict=True):
        baseline = cps.normalize_basic_record(payload, prefix="b_")
        target = cps.normalize_basic_record(payload, prefix="t_")
        identity_match.append(
            bool(is_direct)
            and baseline is not None
            and target is not None
            and one_month_identity_status(baseline, target) is not None
        )
    weights = pd.to_numeric(universe["b_WTFINL"], errors="coerce")
    weight_supported = weights.notna() & weights.gt(0)
    return pd.DataFrame({
        "calendar_month": universe["b__year"].astype("Int64").astype(str) + "-" + universe["b__month"].astype("Int64").astype(str).str.zfill(2),
        "state": universe["b__statefip"].astype("Int64").astype(str).str.zfill(2),
        "mish": universe["b__mish"].astype("Int64").astype(str),
        "duration_band": duration_band(universe["b_DURUNEMP"]),
        "target_class": universe["t__status"].where(universe["t__status"].isin(TARGETS), "missing_or_unsupported"),
        "analysis_weight": weights.where(weight_supported, 0.0).astype(float),
        "weight_supported": weight_supported.astype(int),
        "baseline_unique_key": baseline_key.astype(int),
        "target_unique_key": target_key.astype(int),
        "direct_target_status": direct_target.astype(int),
        "clean_identity_match": pd.Series(identity_match, index=universe.index).astype(int),
    }).reset_index(drop=True)


def aggregate_match_rates(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dimensions = {
        "overall": None,
        "calendar_month": "calendar_month",
        "state": "state",
        "mish": "mish",
        "duration_band": "duration_band",
        "target_class": "target_class",
    }
    for dimension, column in dimensions.items():
        groups = [("all", frame)] if column is None else frame.groupby(column, observed=True, sort=True)
        for level, group in groups:
            universe_rows = len(group)
            universe_weight = float(group["analysis_weight"].sum())
            baseline_rows = int(group["baseline_unique_key"].sum())
            target_rows = int(group["target_unique_key"].sum())
            direct_rows = int(group["direct_target_status"].sum())
            identity_rows = int(group["clean_identity_match"].sum())
            identity_weight = float((group["analysis_weight"] * group["clean_identity_match"]).sum())
            hierarchy_ok = 0 <= identity_rows <= direct_rows <= target_rows <= baseline_rows <= universe_rows
            rows.append({
                "task_id": TASK_ID,
                "dimension": dimension,
                "level": str(level),
                "job_loser_universe_rows": universe_rows,
                "weight_supported_rows": int(group["weight_supported"].sum()),
                "baseline_unique_key_rows": baseline_rows,
                "target_unique_key_rows": target_rows,
                "direct_target_status_rows": direct_rows,
                "clean_identity_match_rows": identity_rows,
                "job_loser_universe_weight": universe_weight,
                "clean_identity_match_weight": identity_weight,
                "baseline_unique_key_rate": baseline_rows / universe_rows,
                "target_unique_key_rate": target_rows / universe_rows,
                "direct_target_status_rate": direct_rows / universe_rows,
                "clean_identity_match_rate": identity_rows / universe_rows,
                "weighted_clean_identity_match_rate": identity_weight / universe_weight,
                "matcher_definition": "positive unique CPSIDP; adjacent MISH; equal CPSIDV where observed; unchanged sex/race; age change zero or one; supported target status",
                "status": "pass" if hierarchy_ok and universe_weight > 0 else "fail",
            })
    audit = pd.DataFrame(rows)
    if not audit["status"].eq("pass").all():
        raise RuntimeError(f"{TASK_ID}: disaggregated matching audit failed.")
    return audit


for path in [
    BASIC_PATH,
    POLICY_PATH,
    POLICY_BUILD_SUMMARY,
]:
    if not path.is_file():
        raise SystemExit(f"{TASK_ID}: required input is missing: {path}")
cps.check_columns(BASIC_PATH, BASE_COLUMNS)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)

policy_build = pd.read_csv(POLICY_BUILD_SUMMARY)
policy_sha256 = sha256_file(POLICY_PATH)
if (
    len(policy_build) != 1
    or policy_build.iloc[0]["intermediate_sha256"] != policy_sha256
):
    raise RuntimeError(f"{TASK_ID}: the CPS UI-policy build metadata does not identify the intermediate.")
policy = pd.read_parquet(POLICY_PATH)
required_policy_columns = {
    "fips", "year", "month", "policy_effective_date", "regular_ui_weeks",
    "extension_ui_weeks", "maximum_ui_weeks", "source_active_extension_weeks",
}
if set(policy.columns) != required_policy_columns or policy.duplicated(["fips", "year", "month"]).any():
    raise RuntimeError(f"{TASK_ID}: CPS UI-policy intermediate schema or key uniqueness changed.")
policy_window = policy.loc[
    policy.apply(lambda row: POLICY_FIRST <= (int(row["year"]), int(row["month"])) <= POLICY_LAST, axis=1)
].copy()
expected_cells = 80
if len(policy_window) != expected_cells * 51:
    raise RuntimeError(f"{TASK_ID}: policy coverage is not complete in the intended baseline window.")

macro_map = cps.load_macro_context_map()
print(f"{TASK_ID}: validated policy panel and variable labels; beginning bounded CPS scan.", flush=True)
step_counts = {step: 0 for step in range(1, 10)}
detail_counts: dict[int, dict[str, list[object]]] = {step: {} for step in range(4, 9)}


def register_filters(
    detail_counts: dict[int, dict[str, list[object]]],
    frame: pd.DataFrame,
    step: int,
    filters: list[tuple[str, str, pd.Series]],
) -> pd.DataFrame:
    prior = pd.Series(True, index=frame.index)
    for filter_id, condition, mask in filters:
        mask = mask.fillna(False)
        failed = int((prior & ~mask).sum())
        if filter_id not in detail_counts[step]:
            detail_counts[step][filter_id] = [condition, 0]
        detail_counts[step][filter_id][1] += failed
        prior &= mask
    return frame.loc[prior].copy()


candidate_rows: list[dict[str, object]] = []
match_observation_frames: list[pd.DataFrame] = []
predictor_audit_frames: list[pd.DataFrame] = []
month_cache: dict[int, pd.DataFrame] = {}
for current_month_key, raw_target_month in iter_task_months(BASIC_PATH, BASE_COLUMNS):
    if current_month_key % 12 == 0:
        print(f"{TASK_ID}: reading {current_month_key // 12:04d}-{current_month_key % 12 + 1:02d}.", flush=True)
    target_month = clean_month(raw_target_month)
    month_cache[current_month_key] = target_month
    baseline_month_key = current_month_key - 1
    baseline_month = month_cache.get(baseline_month_key)
    if baseline_month is None:
        continue
    baseline_year = baseline_month_key // 12
    baseline_calendar_month = baseline_month_key % 12 + 1
    step_counts[1] += len(baseline_month)
    policy_base = baseline_month.merge(
        policy,
        left_on=["_statefip", "_year", "_month"], right_on=["fips", "year", "month"], how="left", indicator="_policy_merge", validate="many_to_one",
    )
    policy_base["_macro_complete"] = [
        all(
            macro_map.get(int(year) * 4 + ((int(month) - 1) // 3 + 1), {}).get(column) is not None
            for column in cps.CORE_MACRO_COLUMNS
        )
        for year, month in zip(policy_base["_year"], policy_base["_month"], strict=True)
    ]
    step_counts[2] += len(policy_base)
    target_lookup = target_month.loc[target_month["_unique_cpsidp"]].copy()
    joined = policy_base.add_prefix("b_").merge(
        target_lookup.add_prefix("t_"), left_on="b__cpsidp", right_on="t__cpsidp", how="left", validate="many_to_one",
    )
    if len(joined) != len(baseline_month):
        raise RuntimeError(f"{TASK_ID}: link construction changed baseline-row count.")
    step_counts[3] += len(joined)

    duration_value = pd.to_numeric(joined["b_DURUNEMP"], errors="coerce")
    universe = register_filters(detail_counts, joined, 4, [
        (
            "universe_policy_calendar_supported",
            "baseline month is within the frozen state UI policy panel from January 2008 through August 2014",
            pd.Series(
                [
                    POLICY_FIRST <= (int(year), int(month)) <= POLICY_LAST
                    for year, month in zip(joined["b__year"], joined["b__month"], strict=True)
                ],
                index=joined.index,
            ),
        ),
        ("universe_baseline_rotation_eligible", "baseline MISH is one of {1,2,3,5,6,7}, so an adjacent-month interview is scheduled", joined["b__mish"].isin(ADJACENT_BASELINE_MISH)),
        ("universe_baseline_status_unemployed", "baseline EMPSTAT maps to unemployed", joined["b__status"].eq("unemployed")),
        ("universe_baseline_job_loser_reason", "baseline WHYUNEMP is one of the released job-loser codes {1,2,3}", joined["b__whyunemp"].isin(JOB_LOSER_CODES)),
        ("universe_age_18_to_69", "baseline age is between 18 and 69 inclusive", joined["b__age"].between(18, 69)),
        ("universe_duration_source_valid_nonnegative", "baseline DURUNEMP is an integer from 0 through 998; no minimum is imposed", duration_value.notna() & duration_value.between(0, 998) & duration_value.eq(duration_value.round())),
    ])
    step_counts[4] += len(universe)
    match_observation_frames.append(match_observations(universe))

    observed_target = register_filters(detail_counts, universe, 5, [
        ("target_unique_same_cpsidp_record_available", "one unique same-CPSIDP target record exists in the next calendar month", universe["t__cpsidp"].notna()),
        ("target_status_supported", "linked target status is employed, unemployed, or not in the labor force", universe["t__status"].isin(TARGETS)),
    ])
    step_counts[5] += len(observed_target)
    occupation_observed = observed_target["b_OCC"].map(lambda value: broad_occupation(value)[1]).notna()
    industry_observed = observed_target["b_IND"].map(lambda value: broad_industry(value)[1]).notna()
    predictor_audit = pd.DataFrame({
        "baseline_year": observed_target["b__year"].astype("Int64"),
        "baseline_month": observed_target["b__month"].astype("Int64"),
        "target": observed_target["t__status"].astype("string"),
        "duration_band": duration_band(observed_target["b_DURUNEMP"]),
        "maximum_ui_weeks": pd.to_numeric(observed_target["b_maximum_ui_weeks"], errors="coerce"),
    })
    predictor_sources = {
        "sex": observed_target["b_sex"].notna(),
        "race": observed_target["b_race"].notna(),
        "hispanic_origin": observed_target["b_hispan"].notna(),
        "education": observed_target["b_educ"].notna(),
        "marital_status": observed_target["b_marst"].notna(),
        "state": observed_target["b_state"].notna(),
        "region": observed_target["b_region"].notna(),
        "metropolitan_status": observed_target["b_metro"].notna(),
        "household_relationship_audit_only": observed_target["b_relate"].notna(),
        "household_counts_observed": observed_target[["b_FAMSIZE", "b_NCHILD"]].notna().all(axis=1),
        "veteran_status": observed_target["b_vetstat"].notna(),
        "citizenship_nativity_joint": observed_target[["b_citizen", "b_nativity"]].notna().all(axis=1),
        "full_time_part_time_search": observed_target["b_wkstat"].isin(WKSTAT_TEXT),
        "time_since_last_full_time_work": observed_target["b_wnftlook"].isin(WNFTLOOK_TEXT),
        "most_recent_worker_class": observed_target["b_classwkr"].notna(),
        "last_job_occupation": occupation_observed,
        "last_job_industry": industry_observed,
        "policy_components": observed_target[["b_regular_ui_weeks", "b_extension_ui_weeks", "b_maximum_ui_weeks"]].notna().all(axis=1),
        "macro_block": observed_target["b__macro_complete"].eq(True),
    }
    for field, mask in predictor_sources.items():
        predictor_audit[field] = mask.to_numpy(dtype=bool)
    predictor_audit_frames.append(predictor_audit)
    predictor = register_filters(detail_counts, observed_target, 6, [
        ("predictor_state_month_policy_available", "the baseline state-month policy cell is present in the frozen panel", observed_target["b__policy_merge"].eq("both")),
        ("predictor_policy_components_observed", "regular, temporary additional, and total maximum UI weeks are observed and nonnegative", observed_target[["b_regular_ui_weeks", "b_extension_ui_weeks", "b_maximum_ui_weeks"]].notna().all(axis=1) & observed_target["b_regular_ui_weeks"].ge(0) & observed_target["b_extension_ui_weeks"].ge(0) & observed_target["b_maximum_ui_weeks"].gt(0)),
        ("predictor_demographic_labels_complete", "sex, race, Hispanic origin, education, marital status, state, region, metropolitan status, nativity, citizenship, and veteran labels are observed", observed_target[["b_sex", "b_race", "b_educ", "b_marst", "b_state", "b_region", "b_metro", "b_hispan", "b_nativity", "b_citizen", "b_vetstat"]].notna().all(axis=1)),
        ("predictor_household_counts_observed", "family size and own-child count are observed", observed_target[["b_FAMSIZE", "b_NCHILD"]].notna().all(axis=1)),
        ("predictor_search_context_complete", "current full-time or part-time search status is observed; time since last full-time work and worker class remain conditional", observed_target["b_wkstat"].isin(WKSTAT_TEXT)),
        ("predictor_last_job_occupation_complete", "the current unemployed record supplies a supported most-recent occupation", occupation_observed),
        ("predictor_last_job_industry_complete", "the current unemployed record supplies a supported most-recent industry", industry_observed),
        ("predictor_public_macro_context_complete", "all standard macro fields are available at the baseline anchor", observed_target["b__macro_complete"].eq(True)),
    ])
    step_counts[6] += len(predictor)

    logical_passes = []
    for payload in predictor.to_dict("records"):
        baseline = cps.normalize_basic_record(payload, prefix="b_")
        target = cps.normalize_basic_record(payload, prefix="t_")
        weight = cps.safe_float(payload["b_WTFINL"])
        valid = (
            baseline is not None and target is not None and bool(payload["b__positive_cpsidp"]) and
            bool(payload["b__unique_cpsidp"]) and one_month_identity_status(baseline, target) is not None and
            pd.notna(payload["t_release_date"]) and pd.notna(payload["t_release_date_source"]) and
            weight is not None and weight > 0
        )
        logical_passes.append(valid)
    household_size = pd.to_numeric(predictor["b_FAMSIZE"], errors="coerce")
    own_children = pd.to_numeric(predictor["b_NCHILD"], errors="coerce")
    policy_gap = (
        pd.to_numeric(predictor["b_regular_ui_weeks"], errors="coerce")
        + pd.to_numeric(predictor["b_extension_ui_weeks"], errors="coerce")
        - pd.to_numeric(predictor["b_maximum_ui_weeks"], errors="coerce")
    ).abs()
    logical = register_filters(detail_counts, predictor, 7, [
        (
            "clean_one_month_person_match_and_release_metadata",
            "positive unique baseline CPSIDP, adjacent MISH, CPSIDV agreement where observed, unchanged sex/race, age increment zero or one, target release metadata, and positive weight",
            pd.Series(logical_passes, index=predictor.index),
        ),
        ("logical_household_counts", "family size is at least one and own-child count is between zero and family size", household_size.ge(1) & own_children.ge(0) & own_children.le(household_size)),
        ("logical_policy_duration_decomposition", "regular plus temporary additional weeks equals total maximum weeks exactly", policy_gap.lt(1e-6)),
        ("logical_policy_effective_date", "the frozen policy value is measured on the fifth day of the baseline month", predictor["b_policy_effective_date"].astype("string").str.endswith("-05")),
    ])
    step_counts[7] += len(logical)
    untrimmed = register_filters(detail_counts, logical, 8, [
        ("no_unbounded_monetary_predictor", "no monetary predictor is retained, so p1/p99 treatment removes no source-valid record", pd.Series(True, index=logical.index)),
    ])
    step_counts[8] += len(untrimmed)

    logical_rows = []
    for payload in untrimmed.to_dict("records"):
        baseline = cps.normalize_basic_record(payload, prefix="b_")
        target = cps.normalize_basic_record(payload, prefix="t_")
        weight = cps.safe_float(payload["b_WTFINL"])
        if baseline is None or target is None or weight is None:
            raise RuntimeError(f"{TASK_ID}: normalized row became invalid after the logical gate.")
        macro = macro_map[cps.baseline_macro_key(baseline)]
        duration = int(round(float(payload["b_DURUNEMP"])))
        why_code = int(payload["b__whyunemp"])
        occupation_code, occupation_label = broad_occupation(payload["b_OCC"])
        industry_code, industry_label = broad_industry(payload["b_IND"])
        subject_digest = hashlib.blake2b(
            f"{TASK_ID}|{baseline['cpsidp']}".encode("utf-8"), digest_size=12
        ).hexdigest()
        subject_id = f"lcui_{subject_digest}"
        row_id = cps.stable_row_id(
            TASK_ID, "lcui", baseline["cpsidp"], baseline["year"], baseline["month"], target["year"], target["month"], baseline["mish"],
        )
        person_month_id = cps.stable_row_id(TASK_ID, "lcuipm", baseline["cpsidp"], baseline["year"], baseline["month"], baseline["mish"])
        target_person_month_id = cps.stable_row_id(TASK_ID, "lcuipm", target["cpsidp"], target["year"], target["month"], target["mish"])
        state_month_policy_id = cps.stable_row_id(TASK_ID, "lcuipol", int(payload["b_fips"]), baseline["year"], baseline["month"])
        row = {field: None for field in FIELDNAMES}
        row.update({
            "row_id": row_id, "subject_id": subject_id, "group_id": subject_id,
            "person_month_id": person_month_id, "target_person_month_id": target_person_month_id,
            "state_month_policy_id": state_month_policy_id,
            "selection_period": f"{baseline['year']:04d}-{baseline['month']:02d}",
            "baseline_year": baseline["year"], "baseline_month": baseline["month"],
            "outcome_year": target["year"], "outcome_month": target["month"],
            "release_date": cps.clean_release_date_value(payload["t_release_date"]),
            "release_date_source": cps.clean_release_date_source(payload["t_release_date_source"]),
            "mish_baseline": baseline["mish"], "mish_target": target["mish"],
            "policy_state_fips": int(payload["b_fips"]), "policy_effective_date": str(payload["b_policy_effective_date"]),
            "policy_join_rule": POLICY_JOIN_RULE, "policy_source": POLICY_SOURCE, "policy_archive_version": "P2015_1088_data",
            "policy_panel_sha256": policy_sha256, "source_active_extension_weeks": float(payload["b_source_active_extension_weeks"]),
            "policy_merge": 1, "macro_context_merge": 1, "paper_universe": int(duration >= 13), "direct_target_observed": 1,
            "predictor_complete": 1, "logical_valid": 1, "outlier_retained": 1,
            "baseline_status": "unemployed", "target": target["status"], "analysis_weight": weight,
            "age": baseline["age"], "age_group": cps.age_group(baseline["age"]), "sex": baseline["sex"],
            "race": baseline["race"], "educ": baseline["educ"], "marst": baseline["marst"], "state": baseline["state"],
            "region": baseline["region"], "metro": baseline["metro"], "relate": baseline["relate"],
            "famsize": baseline["famsize"], "nchild": baseline["nchild"], "hispan": baseline["hispan"],
            "nativity": baseline["nativity"], "citizen": baseline["citizen"], "citizenship_nativity_rendered": 1,
            "vetstat": baseline["vetstat"], "veteran_status_rendered": 1, "relationship_rendered": 0,
            "whyunemp_code": why_code, "whyunemp_text": WHYUNEMP_TEXT[why_code],
            "search_layoff_status": SEARCH_LAYOFF_STATUS[why_code], "wkstat_baseline": baseline["wkstat"],
            "classwkr": baseline["classwkr"], "classwkr_rendered": int(baseline["classwkr"] is not None),
            "durunemp_baseline": duration, "wnftlook_baseline": baseline["wnftlook"],
            "last_full_time_work_rendered": int(baseline["wnftlook"] in WNFTLOOK_TEXT),
            "last_job_source": "current_unemployed_CPS_record_OCC_IND_only",
            "last_job_occupation_code": occupation_code, "last_job_occupation": occupation_label,
            "last_job_occupation_rendered": int(occupation_label is not None),
            "last_job_industry_code": industry_code, "last_job_industry": industry_label,
            "last_job_industry_rendered": int(industry_label is not None),
            "regular_ui_weeks": float(payload["b_regular_ui_weeks"]),
            "extension_ui_weeks": float(payload["b_extension_ui_weeks"]),
            "maximum_ui_weeks": float(payload["b_maximum_ui_weeks"]), **macro,
        })
        if row["target"] not in TARGETS or any(row[column] is None for column in PROMPT_COLUMNS):
            raise RuntimeError(f"{TASK_ID}: incomplete final row {row_id}.")
        logical_rows.append(row)
    candidate_rows.extend(logical_rows)
    for old_month_key in list(month_cache):
        if old_month_key < current_month_key - 1:
            del month_cache[old_month_key]

if not candidate_rows:
    raise RuntimeError(f"{TASK_ID}: no prompt-complete rows survived construction.")
if not (step_counts[1] == step_counts[2] == step_counts[3]):
    raise RuntimeError(f"{TASK_ID}: one-month link construction was not row preserving.")
pool = pd.DataFrame(candidate_rows, columns=FIELDNAMES)
if pool.duplicated(["subject_id", "baseline_year", "baseline_month", "mish_baseline"]).any():
    raise RuntimeError(f"{TASK_ID}: duplicate person-month selection identities.")

pool = pool.sort_values(["subject_id", "baseline_year", "baseline_month", "mish_baseline"], kind="mergesort").reset_index(drop=True)
spell_sequence = []
previous_by_person: dict[str, tuple[int, int, int]] = {}
sequence_by_person: dict[str, int] = {}
for row in pool.itertuples(index=False):
    subject = str(row.subject_id)
    current_month = month_key(int(row.baseline_year), int(row.baseline_month))
    current_duration = int(row.durunemp_baseline)
    previous = previous_by_person.get(subject)
    new_spell = (
        previous is None
        or current_month != previous[0] + 1
        or current_duration < previous[1] - 2
        or current_duration > previous[1] + 9
    )
    if new_spell:
        sequence_by_person[subject] = sequence_by_person.get(subject, 0) + 1
    spell_sequence.append(sequence_by_person[subject])
    previous_by_person[subject] = (current_month, current_duration, int(row.whyunemp_code))
pool["spell_id"] = [
    cps.stable_row_id(TASK_ID, "lcuispell", subject, sequence)
    for subject, sequence in zip(pool["subject_id"], spell_sequence, strict=True)
]
pool["spell_id_method"] = "contiguous_eligible_person_months_duration_change_minus2_to_plus9_weeks"
if pool["spell_id"].isna().any():
    raise RuntimeError(f"{TASK_ID}: unemployment-spell reconstruction failed.")

pool["id"] = pool["subject_id"].astype("string")
pool["time"] = (
    pool["baseline_year"].astype(int).astype(str).str.zfill(4)
    + "-" + pool["baseline_month"].astype(int).astype(str).str.zfill(2) + "-01"
).astype("string")
# Public keys are fixed before sampling; source keys retain their sort types.
pool["release_date"] = pd.to_datetime(pool["release_date"], errors="raise").dt.strftime("%Y-%m-%d")
selected, _, _ = sample_task_records(
    pool,
    task_id=TASK_ID,
    output_dir=SAMPLING_DIR,
    output_sort_cols=["baseline_year", "baseline_month", "subject_id", "mish_baseline"],
)
step_counts[9] = len(selected)
if len(selected) != len(pool):
    raise RuntimeError(f"{TASK_ID}: the all-duration pool is below the cap but selection dropped records.")
if len(selected) != EXPECTED_SELECTED_ROWS:
    raise RuntimeError(
        f"{TASK_ID}: expected {EXPECTED_SELECTED_ROWS:,} rows from the frozen inputs, got {len(selected):,}."
    )
write_public_table(selected, task_id=TASK_ID, public_path=TABLE_PATH)

steps = [
    (1, "Start from source observations", step_counts[1], step_counts[1]),
    (2, "Attach additional sources", step_counts[1], step_counts[2]),
    (3, "Construct leads and lags", step_counts[2], step_counts[3]),
    (4, "Restrict universe", step_counts[3], step_counts[4]),
    (5, "Require observed targets", step_counts[4], step_counts[5]),
    (6, "Require observed predictors", step_counts[5], step_counts[6]),
    (7, "Apply logical and validity filters", step_counts[6], step_counts[7]),
    (8, "Apply outlier trimming", step_counts[7], step_counts[8]),
    (9, "Export sample", step_counts[8], step_counts[9]),
]
pd.DataFrame([
    construction_step(
        TASK_ID, step, label,
        before, after, 0 if step == 1 else before - after,
    )
    for step, label, before, after in steps
], columns=STEP_COLUMNS).to_csv(OUTPUT_DIR / "01_sample_construction_steps.csv", index=False)
detail_rows = []
for step, filter_id, condition in [
    (1, "common_cps_person_months", "start from every person-month in the validated common CPS intermediate for each potential baseline month"),
    (2, "policy_and_macro_context_attached", "attach the validated state-month unemployment-insurance policy cell and common pre-baseline macro block to every person-month"),
    (3, "next_month_person_record_linked", "construct the adjacent-month person link for every baseline person-month without yet requiring that a valid target record exists"),
    (9, "deterministic_export_cap", "apply the registered deterministic sampling order and export all eligible person-months because the sample is below the task cap"),
]:
    detail_rows.append(filter_count(
                           TASK_ID, step, 1,
                           filter_id, condition, 0,
                       ))
for step in range(4, 9):
    for order, (filter_id, (condition, fail_count)) in enumerate(detail_counts[step].items(), start=1):
        detail_rows.append(filter_count(
                               TASK_ID, step, order,
                               filter_id, condition, fail_count,
                           ))
pd.DataFrame(detail_rows, columns=DETAIL_COLUMNS).to_csv(OUTPUT_DIR / "01_sample_construction_drop_details.csv", index=False)

policy_pairs = policy_window[["year", "month"]].drop_duplicates().sort_values(["year", "month"], kind="mergesort")
minimum_policy_pair = tuple(policy_pairs.iloc[0].astype(int))
maximum_policy_pair = tuple(policy_pairs.iloc[-1].astype(int))
if minimum_policy_pair != POLICY_FIRST or maximum_policy_pair != POLICY_LAST:
    raise RuntimeError(
        f"{TASK_ID}: policy-window extrema differ from the registered baseline window: "
        f"{minimum_policy_pair} to {maximum_policy_pair}."
    )
panel_audit = pd.DataFrame([{
    "task_id": TASK_ID, "policy_file": str(POLICY_PATH.relative_to(PROJECT_ROOT)), "policy_panel_sha256": policy_sha256,
    "policy_join_rule": POLICY_JOIN_RULE, "states": int(policy_window["fips"].nunique()), "state_month_cells": int(len(policy_window)),
    "minimum_policy_year_month": f"{minimum_policy_pair[0]:04d}-{minimum_policy_pair[1]:02d}",
    "maximum_policy_year_month": f"{maximum_policy_pair[0]:04d}-{maximum_policy_pair[1]:02d}",
    "minimum_maximum_ui_weeks": float(policy_window["maximum_ui_weeks"].min()),
    "maximum_maximum_ui_weeks": float(policy_window["maximum_ui_weeks"].max()),
    "duration_decomposition_max_abs_gap": float((policy_window["regular_ui_weeks"] + policy_window["extension_ui_weeks"] - policy_window["maximum_ui_weeks"]).abs().max()),
    "source_active_extension_differs_from_total_minus_regular_cells": int((policy_window["source_active_extension_weeks"] - policy_window["extension_ui_weeks"]).abs().gt(1e-6).sum()),
    "temporary_additional_component_rule": "ui_weeks_minus_reg_UI_to_preserve_the_source_total_including_suspension_months",
    "source_reg_UI_label": "regular UI weeks (<26 for some)",
    "source_ext_wks_label": "total weeks of extended UI (13-73)",
    "source_ui_weeks_label": "Total UI weeks (no suspensions)",
    "coverage_status": "pass_exact_registered_window",
}])
panel_audit.to_csv(OUTPUT_DIR / "01_policy_panel_audit.csv", index=False)

match_frame = pd.concat(match_observation_frames, ignore_index=True)
match_rates = aggregate_match_rates(match_frame)
overall_match_rows = int(match_rates.loc[match_rates["dimension"].eq("overall"), "job_loser_universe_rows"].item())
if overall_match_rows != step_counts[4]:
    raise RuntimeError(f"{TASK_ID}: match audit universe does not reconcile with the construction waterfall.")
match_rates.to_csv(OUTPUT_DIR / "01_match_rates.csv", index=False)

selection_source = pool.copy()
selection_source["duration_band"] = duration_band(selection_source["durunemp_baseline"])
selection_source["calendar_month"] = selection_source["selection_period"]
selection_source["state_level"] = selection_source["policy_state_fips"].astype(int).astype(str).str.zfill(2)
selection_source["mish_level"] = selection_source["mish_baseline"].astype(int).astype(str)
selected_ids = set(selected["row_id"])
selection_source["selected"] = selection_source["row_id"].isin(selected_ids)
selection_rows = []
for dimension, column in {
    "overall": None,
    "calendar_month": "calendar_month",
    "state": "state_level",
    "mish": "mish_level",
    "duration_band": "duration_band",
    "target_class": "target",
}.items():
    groups = [("all", selection_source)] if column is None else selection_source.groupby(column, observed=True, sort=True)
    for level, group in groups:
        selected_rows = int(group["selected"].sum())
        selection_rows.append({
            "task_id": TASK_ID,
            "dimension": dimension,
            "level": str(level),
            "eligible_rows": len(group),
            "selected_rows": selected_rows,
            "selection_rate": selected_rows / len(group),
            "status": "pass_all_eligible_exported" if selected_rows == len(group) else "fail",
        })
pd.DataFrame(selection_rows).to_csv(OUTPUT_DIR / "01_selection_rates_by_dimension.csv", index=False)

predictor_audit = pd.concat(predictor_audit_frames, ignore_index=True)
conditional_rows = []
conditional_contract = {
    "household_relationship_audit_only": ("audit_only_not_model_facing", None, "exclude"),
    "veteran_status": ("required_model_facing_after_step6", 1.0, "include"),
    "citizenship_nativity_joint": ("required_model_facing_after_step6", None, "include"),
    "time_since_last_full_time_work": ("conditional_model_facing_exact_jobfind_rule", None, "include_conditionally"),
    "most_recent_worker_class": ("conditional_model_facing_exact_jobfind_rule", None, "include_conditionally"),
    "last_job_occupation": ("required_model_facing_after_step6", None, "include"),
    "last_job_industry": ("required_model_facing_after_step6", None, "include"),
}
for field, (rule, required_rate, decision) in conditional_contract.items():
    overall = float(predictor_audit[field].mean())
    strata = predictor_audit.assign(policy_regime=np.where(predictor_audit["maximum_ui_weeks"].gt(26), "temporary_additional_positive", "regular_only"))
    minimum_stratum = min(
        float(group[field].mean())
        for _, group in strata.groupby(["baseline_year", "duration_band", "target", "policy_regime"], observed=True)
        if len(group) > 0
    )
    if required_rate is not None and overall < required_rate:
        raise RuntimeError(f"{TASK_ID}: conditionally retained field {field} is not complete.")
    conditional_rows.append({
        "task_id": TASK_ID,
        "field": field,
        "otherwise_eligible_rows": len(predictor_audit),
        "source_valid_rows": int(predictor_audit[field].sum()),
        "source_valid_rate": overall,
        "minimum_major_stratum_valid_rate": minimum_stratum,
        "decision": decision,
        "operational_rule": rule,
    })
pd.DataFrame(conditional_rows).to_csv(OUTPUT_DIR / "01_conditional_field_coverage.csv", index=False)

source_inventory = pd.DataFrame([
    {"source_role": "CPS person-months", "path": str(BASIC_PATH.relative_to(PROJECT_ROOT)), "sha256": sha256_file(BASIC_PATH), "version_or_definition": "local IPUMS Basic Monthly CPS intermediate; source extract codebook cps_basic.txt"},
    {"source_role": "UI state-month policy intermediate", "path": str(POLICY_PATH.relative_to(PROJECT_ROOT)), "sha256": policy_sha256, "version_or_definition": "constructed policy panel; source policy measured on day 5 of each month"},
])
source_inventory.to_csv(OUTPUT_DIR / "01_source_inventory.csv", index=False)

crosswalk_rows = [
    ("CPSIDP/CPSIDV/MISH", "subject_id; person_month_id; target_person_month_id", "same-person adjacent-month link with unique positive keys and validation checks", "month t and t+1 identifiers", "not model-facing"),
    ("EMPSTAT at t+1", "target", "direct mapping to employed, unemployed, or not_in_labor_force", "one month after anchor", "categorical answer"),
    ("WHYUNEMP", "whyunemp_text; search_layoff_status", "released codes 1-3; natural job-loss and layoff/search labels", "month t", "category"),
    ("DURUNEMP", "durunemp_baseline", "source-valid integer 0-998; no minimum-duration filter", "month t", "weeks"),
    ("WKSTAT", "wkstat_baseline", "natural full-time or part-time search label", "month t", "category"),
    ("WNFTLOOK", "wnftlook_baseline", "render only when observed", "month t current record", "elapsed-time category"),
    ("OCC", "last_job_occupation", "broad recode from archived replication code; official IPUMS definition is most recent occupation for unemployed people", "month t current unemployed record", "broad category"),
    ("IND", "last_job_industry", "broad recode from archived replication code; official IPUMS definition is most recent industry for unemployed people", "month t current unemployed record", "broad category"),
    ("AGE/SEX/RACE/HISPAN/EDUC/MARST", "personal demographic fields", "natural-language labels", "month t", "years/categories"),
    ("STATEFIP/REGION/METRO", "state/region/metro", "harmonised geographic labels", "month t", "categories"),
    ("RELATE", "relate", "retained only for audit; not model-facing, matching the actual labor_cps_jobfind renderer", "month t", "not model-facing"),
    ("FAMSIZE/NCHILD", "famsize/nchild", "inclusive family size and own-child subset", "month t", "counts"),
    ("VETSTAT", "vetstat", "natural veteran sentence; retained after complete-coverage gate", "month t", "category"),
    ("NATIVITY/CITIZEN", "nativity/citizen", "natural nativity and citizenship sentence; required jointly at step 6", "month t", "categories"),
    ("CLASSWKR", "classwkr", "natural worker-class sentence; render only when the existing labor_cps_jobfind mapping supports the value", "month t current record", "category"),
    ("reg_UI", "regular_ui_weeks", "direct source field", "state at month-t day 5", "weeks"),
    ("ui_weeks - reg_UI", "extension_ui_weeks", "temporary additional component derived to preserve the source total in suspension months", "state at month-t day 5", "weeks"),
    ("ui_weeks", "maximum_ui_weeks", "direct source total, labelled Total UI weeks (no suspensions)", "state at month-t day 5", "weeks"),
]
pd.DataFrame(crosswalk_rows, columns=["source_field", "analysis_field", "transformation", "timing", "unit_or_domain"]).to_csv(OUTPUT_DIR / "01_source_variable_crosswalk.csv", index=False)

manifest_rows = [
    ("Age", "AGE", "integer", "month t", "years", "required"),
    ("Sex", "SEX/sex", "natural label", "month t", "category", "required"),
    ("Race and ethnicity", "RACE/race; HISPAN/hispan", "natural combined phrase", "month t", "category", "required"),
    ("Highest completed education", "EDUC/educ", "natural label", "month t", "category", "required"),
    ("Marital status", "MARST/marst", "natural label", "month t", "category", "required"),
    ("State; region; area type", "STATEFIP/state; REGION/region; METRO/metro", "natural labels", "month t", "categories", "required"),
    ("Family members in household; own children", "FAMSIZE; NCHILD", "inclusive counts", "month t", "counts", "required"),
    ("Nativity and citizenship", "NATIVITY/nativity; CITIZEN/citizen", "exact labor_cps_jobfind natural-language mapping", "month t", "categories", "required"),
    ("Veteran status", "VETSTAT/vetstat", "natural sentence", "month t", "category", "required after complete-coverage gate"),
    ("Reason unemployment began", "WHYUNEMP", "released job-loser mapping", "month t", "category", "required"),
    ("Reported unemployment duration", "DURUNEMP", "integer; no minimum", "month t", "weeks", "required"),
    ("Search or temporary-layoff status", "WHYUNEMP", "code 1 temporary layoff; codes 2-3 active search", "month t", "category", "required"),
    ("Full-time or part-time search", "WKSTAT", "natural label", "month t", "category", "required"),
    ("Time since last full-time work", "WNFTLOOK", "natural elapsed-time label", "month t current record", "category", "omit sentence when missing"),
    ("Most recent worker class", "CLASSWKR", "exact labor_cps_jobfind natural-language mapping", "month t current record", "category", "omit sentence when unsupported"),
    ("Last-job occupation", "OCC", "archived broad occupation recode", "month t current unemployed record", "category", "required"),
    ("Last-job industry", "IND", "archived broad industry recode", "month t current unemployed record", "category", "required"),
    ("Common macro block", "householdbench_macro_context", "four completed quarters and five-year reference", "last completed quarter before month t", "rates", "required"),
    ("Maximum potential duration", "ui_weeks", "direct state-month source total", "month-t day 5", "weeks", "required"),
    ("Regular state benefits", "reg_UI", "direct state-month source field", "month-t day 5", "weeks", "required"),
    ("Temporary additional benefits", "ui_weeks-reg_UI", "derived component", "month-t day 5", "weeks", "required"),
]
pd.DataFrame(manifest_rows, columns=["prompt_label", "source_field", "transformation", "timing", "unit", "missingness_rule"]).to_csv(OUTPUT_DIR / "01_predictor_manifest.csv", index=False)

print(f"{TASK_ID}: wrote {len(selected):,} selected rows from {len(pool):,} eligible rows to {TABLE_PATH}.")
