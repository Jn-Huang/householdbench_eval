#!/usr/bin/env python
"""Construct the CEX Social-Security-payment task from the daily Diary intermediate.

The authoritative source input is the validated general CEX Diary CU-day
intermediate.  This task owns the Stephens-specific universe, calendar,
histories, tail rule, and deterministic selection; it does not parse Diary
archives or reconstruct expenditure transactions.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.sample_accounting import filter_count

from scripts.utils.io import sha256_file

from scripts.utils.macro_context import CORE_MACRO_COLUMNS, attach_macro_context
from scripts.utils.sampling import sample_task_records
from scripts.utils.table_schema import write_public_table


TASK_ID = "cons_cex_sspay"
SEED = 42014
MAX_PROMPTS = 500_000
EVENT_MIN = -7
EVENT_MAX = 7
YEARS = list(range(1986, 1997))
STRICT_TARGET_VERSION = "stephens_1986_1996_strict_v1"
CALENDAR_VERSION = "ssa_oasdi_pre1997_local_rules_v1"
SPOT_CHECKS = {
    "1986-05": "1986-05-02",
    "1990-09": "1990-08-31",
    "1993-01": "1992-12-31",
    "1994-09": "1994-09-02",
    "1996-02": "1996-02-02",
}

INTERMEDIATE_PATH = PROJECT_ROOT / "data/intermediate/cex_diary_daily.parquet"
BUILD_MANIFEST_PATH = PROJECT_ROOT / "output/preprocessing/cex_diary/build/02_build_manifest.json"
TABLE_PATH = PROJECT_ROOT / "data/householdbench/tabular/cons_cex_sspay.csv"
OUTPUT_DIR = PROJECT_ROOT / "output/householdbench/tasks/cons_cex_sspay"
SAMPLING_DIR = PROJECT_ROOT / "data" / "intermediate" / "householdbench" / TASK_ID

STEP_COLUMNS = [
    "task_id", "step", "step_label", "rows_before", "rows_after", "rows_dropped",
    "consumer_units_after", "consumer_unit_weeks_after", "consumer_unit_days_after",
]
DETAIL_COLUMNS = ["task_id", "step", "detail_order", "filter_id", "pass_condition", "fail_count"]
TARGET_COLUMNS = ["daily_total_expenditure", "daily_food_at_home", "daily_food_away_from_home"]
RESOURCE_DOLLAR_COLUMNS = ["annual_income_before_tax", "annual_social_security_railroad_income"]
TRIM_DAILY_COLUMN = "daily_total_expenditure"
TRIM_RESOURCE_COLUMN = "annual_income_before_tax"
RETIRED_RAW_TASK_AUDITS = [
    "01_source_layout_audit.csv",
    "01_day_and_calendar_audit.csv",
    "01_ucc_crosswalk_audit.csv",
    "01_target_reconstruction_audit.csv",
    "01_stale_definition_comparison.csv",
]


def nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (occurrence - 1))


def last_weekday(year: int, month: int, weekday: int) -> date:
    following = date(year + (month == 12), month % 12 + 1, 1)
    last = following - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def observed_holiday(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def federal_holidays(year: int) -> dict[date, str]:
    holidays = {
        observed_holiday(date(year, 1, 1)): "New Year's Day",
        nth_weekday(year, 2, 0, 3): "Washington's Birthday",
        last_weekday(year, 5, 0): "Memorial Day",
        observed_holiday(date(year, 7, 4)): "Independence Day",
        nth_weekday(year, 9, 0, 1): "Labor Day",
        nth_weekday(year, 10, 0, 2): "Columbus Day",
        observed_holiday(date(year, 11, 11)): "Veterans Day",
        nth_weekday(year, 11, 3, 4): "Thanksgiving Day",
        observed_holiday(date(year, 12, 25)): "Christmas Day",
    }
    if year >= 1986:
        holidays[nth_weekday(year, 1, 0, 3)] = "Martin Luther King Jr. Day"
    return holidays


def oasdi_delivery_holidays(year: int) -> dict[date, str]:
    return {
        observed_holiday(date(year, 1, 1)): "New Year's Day",
        observed_holiday(date(year, 7, 4)): "Independence Day",
        nth_weekday(year, 9, 0, 1): "Labor Day",
    }


def build_oasdi_calendar(start_year: int, end_year: int) -> pd.DataFrame:
    holidays: dict[date, str] = {}
    for year in range(start_year - 1, end_year + 2):
        holidays.update(oasdi_delivery_holidays(year))
    rows: list[dict[str, object]] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            normal = date(year, month, 3)
            scheduled = normal
            reasons: list[str] = []
            while scheduled.weekday() >= 5 or scheduled in holidays:
                if scheduled.weekday() == 5:
                    reasons.append("Saturday")
                elif scheduled.weekday() == 6:
                    reasons.append("Sunday")
                else:
                    reasons.append(holidays[scheduled])
                scheduled -= timedelta(days=1)
            reason = (
                "regular_third"
                if not reasons
                else "preceding_business_day_due_to_" + "_and_".join(dict.fromkeys(reasons))
            )
            rows.append(
                {
                    "payment_month": f"{year:04d}-{month:02d}",
                    "normal_payment_date": pd.Timestamp(normal),
                    "scheduled_payment_date": pd.Timestamp(scheduled),
                    "adjustment_reason": reason,
                    "normal_date_weekday": normal.strftime("%A"),
                    "normal_date_holiday": holidays.get(normal, ""),
                    "programme": "OASDI",
                    "calendar_version": CALENDAR_VERSION,
                    "calendar_source": "local_SSA_Handbook_121_1_Act_708_and_SSA_History",
                }
            )
    return pd.DataFrame(rows)

# The task deliberately reads harmonised fields only.  Raw-code columns remain
# in the intermediate for preprocessing provenance but are not task inputs.
INTERMEDIATE_COLUMNS = [
    "source_day_id", "source_week_id", "source_year", "source_family", "source_archive",
    "week_key", "cu_id", "diary_week", "diary_day_sequence", "diary_date",
    "canonical_source_selected", "analysis_weight", "release_date", "release_date_source",
    "age_ref", "sex", "race", "education", "marital_status", "household_size",
    "children_under_18", "number_adults", "number_adults_valid", "region", "urban_status",
    "home_tenure", "reference_person_work_status", "annual_income_before_tax",
    "annual_social_security_railroad_income", "likely_social_security_recipients",
    "reference_or_spouse_oasdi", "reference_or_spouse_ssi", "reference_or_spouse_railroad",
    "reference_or_spouse_dual_oasdi_ssi", "reference_or_spouse_paper_recipient",
    "reference_person_oasdi_affirmative", "reference_person_ssi_affirmative",
    "reference_person_railroad_affirmative", "spouse_oasdi_affirmative",
    "spouse_ssi_affirmative", "spouse_railroad_affirmative",
    "response_income_complete_records", "positive_combined_ss_railroad_income_records",
    "stephens_1986_1996_quality_flag_v1", "stephens_food_available",
    "stephens_food_version", "legacy_strict_rule_version",
    "daily_total_expenditure_legacy_strict_v1", "daily_food_at_home_legacy_strict_v1",
    "daily_food_away_from_home_legacy_strict_v1", "start_date_valid",
    "both_diary_week_numbers_available", "both_diary_week_records_available",
    "legacy_strict_valid_start_week_records", "legacy_strict_valid_dated_week_records",
    "pickup_complete_records", "weekn_equals_two_records", "nonoverlapping_diary_week_dates",
    "pickup_complete_week", "weekn_equals_two",
    "legacy_strict_dated_week_valid", "general_aggregated_transaction_rows",
    "legacy_strict_aggregated_transaction_rows",
]


def require_intermediate() -> tuple[pd.DataFrame, dict[str, object]]:
    required = [INTERMEDIATE_PATH, BUILD_MANIFEST_PATH]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"{TASK_ID}: required intermediate inputs are absent: {missing}")
    build_manifest = json.loads(BUILD_MANIFEST_PATH.read_text(encoding="utf-8"))
    observed_hash = sha256_file(INTERMEDIATE_PATH)
    if build_manifest.get("intermediate_sha256") != observed_hash:
        raise RuntimeError(f"{TASK_ID}: CEX Diary build manifest does not match the intermediate hash.")
    daily = pd.read_parquet(INTERMEDIATE_PATH, columns=INTERMEDIATE_COLUMNS)
    if daily.empty or daily["source_day_id"].duplicated().any():
        raise RuntimeError(f"{TASK_ID}: intermediate is empty or source_day_id is not unique.")
    years = set(pd.to_numeric(daily["source_year"], errors="raise").astype(int))
    if years != set(range(1982, 2012)):
        raise RuntimeError(f"{TASK_ID}: intermediate years do not equal canonical 1982--2011 coverage: {sorted(years)}")
    if not daily["canonical_source_selected"].fillna(False).astype(bool).all():
        raise RuntimeError(f"{TASK_ID}: intermediate admits a noncanonical source row.")
    expected_source_day_id = daily["source_week_id"].astype("string") + ":D" + daily["diary_day_sequence"].astype(int).astype(str)
    if not daily["source_day_id"].astype("string").eq(expected_source_day_id).all():
        raise RuntimeError(f"{TASK_ID}: intermediate source-day key does not equal source-week plus day sequence.")
    return daily, build_manifest


def attach_nearest_event(days: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    event_columns = [
        "scheduled_payment_date", "normal_payment_date", "adjustment_reason", "payment_month",
        "programme", "calendar_version", "calendar_source",
    ]
    events = calendar[event_columns].sort_values("scheduled_payment_date")
    work = days.sort_values("diary_date", kind="mergesort").copy()
    previous = pd.merge_asof(work, events, left_on="diary_date", right_on="scheduled_payment_date", direction="backward")
    following = pd.merge_asof(work, events, left_on="diary_date", right_on="scheduled_payment_date", direction="forward")
    previous_gap = (previous["diary_date"] - previous["scheduled_payment_date"]).dt.days.abs()
    following_gap = (following["scheduled_payment_date"] - following["diary_date"]).dt.days.abs()
    choose_previous = following_gap.isna() | (previous_gap.notna() & previous_gap.le(following_gap))
    selected = following.copy()
    for column in event_columns:
        selected.loc[choose_previous, column] = previous.loc[choose_previous, column].to_numpy()
    selected["event_time"] = (selected["diary_date"] - selected["scheduled_payment_date"]).dt.days.astype(int)
    if selected["scheduled_payment_date"].isna().any() or selected["event_time"].abs().gt(17).any():
        examples = selected.loc[
            selected["scheduled_payment_date"].isna() | selected["event_time"].abs().gt(17),
            ["source_year", "week_key", "diary_date", "scheduled_payment_date", "event_time"],
        ].head(5).to_dict("records")
        raise RuntimeError(f"{TASK_ID}: could not attach a nearby scheduled payment date: {examples}")
    return selected


def attach_additional_sources(days: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Attach calendar and macro fields without dropping any source-day row."""
    out = days
    task_date_mask = out["source_year"].isin(YEARS) & out["diary_date"].notna()
    task_dated = out.loc[task_date_mask, ["source_day_id", "source_year", "week_key", "diary_date"]].copy()
    task_dated["_source_row_index"] = task_dated.index
    task_dated = attach_nearest_event(task_dated, calendar)
    calendar_columns = [
        "scheduled_payment_date", "normal_payment_date", "adjustment_reason", "payment_month",
        "programme", "calendar_version", "calendar_source", "event_time",
    ]
    for column in ["scheduled_payment_date", "normal_payment_date"]:
        out[column] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    for column in ["adjustment_reason", "payment_month", "programme", "calendar_version", "calendar_source"]:
        out[column] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["event_time"] = np.nan
    for column in calendar_columns:
        out.loc[task_dated["_source_row_index"].to_numpy(), column] = task_dated[column].to_numpy()
    out["calendar_attachment_success"] = out["scheduled_payment_date"].notna()

    dated = out.loc[out["diary_date"].notna(), ["source_day_id", "diary_date"]].copy()
    dated["origin_q_index"] = dated["diary_date"].dt.year * 4 + dated["diary_date"].dt.quarter
    dated = attach_macro_context(dated, origin_q_index_col="origin_q_index", required_columns=CORE_MACRO_COLUMNS)
    for column in CORE_MACRO_COLUMNS:
        out[column] = np.nan
        out.loc[dated.index, column] = dated[column].to_numpy()
    out["macro_attachment_success"] = out[CORE_MACRO_COLUMNS].notna().all(axis=1)
    out["macro_context_merge"] = out["macro_attachment_success"].astype(int)

    holidays: dict[object, str] = {}
    for calendar_year in range(1982, 2012):
        holidays.update(federal_holidays(calendar_year))
    out["day_of_week"] = out["diary_date"].dt.day_name().fillna("")
    out["is_weekend"] = out["diary_date"].dt.dayofweek.ge(5).fillna(False).astype(int)
    out["is_federal_holiday"] = out["diary_date"].dt.date.isin(holidays).fillna(False).astype(int)
    out["federal_holiday_name"] = out["diary_date"].dt.date.map(holidays).fillna("")
    out["adjacent_to_federal_holiday"] = (
        out["diary_date"].add(pd.Timedelta(days=1)).dt.date.isin(holidays)
        | out["diary_date"].sub(pd.Timedelta(days=1)).dt.date.isin(holidays)
    ).fillna(False).astype(int)
    return out


def add_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach strict prior-day histories while retaining every input row."""
    out = frame
    history_observed = (
        out["source_year"].isin(YEARS)
        & out["diary_date"].notna()
        & out[TARGET_COLUMNS].notna().all(axis=1)
    )
    history = out.loc[history_observed].copy().sort_values(
        ["cu_id", "diary_date", "week_key", "source_day_id"], kind="mergesort"
    )
    duplicate_cus = set(history.loc[history.duplicated(["cu_id", "diary_date"], keep=False), "cu_id"])
    history["unique_daily_history_unit"] = ~history["cu_id"].isin(duplicate_cus)
    usable = history.loc[history["unique_daily_history_unit"]].copy()
    usable_index = usable.index
    out["unique_daily_history_unit"] = False
    out.loc[usable_index, "unique_daily_history_unit"] = True
    grouped = usable.groupby("cu_id", sort=False, observed=True)
    for lag in range(1, 8):
        out[f"prior_observed_date_lag{lag}"] = pd.NaT
        out.loc[usable_index, f"prior_observed_date_lag{lag}"] = grouped["diary_date"].shift(lag).to_numpy()
        for target in TARGET_COLUMNS:
            out[f"{target}_lag{lag}"] = np.nan
            out.loc[usable_index, f"{target}_lag{lag}"] = grouped[target].shift(lag).to_numpy()
        out[f"calendar_day_gap_lag{lag}"] = pd.NA
        gap = (usable["diary_date"] - grouped["diary_date"].shift(lag)).dt.days.astype("Int64")
        out.loc[usable_index, f"calendar_day_gap_lag{lag}"] = gap.to_numpy()
    return out


def apply_upper_tails(frame: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply full-panel calendar-year p99 screens to total spending and income."""
    out = frame.copy()
    out["target_calendar_year"] = out["diary_date"].dt.year.astype(int)
    reference = reference.loc[
        reference["source_year"].isin(YEARS) & reference["diary_date"].notna()
    ].copy()
    reference["calendar_year"] = reference["diary_date"].dt.year.astype(int)
    cutoff_maps: dict[str, dict[int, float]] = {}
    audit_rows: list[dict[str, object]] = []
    for variable, scope in [
        (TRIM_DAILY_COLUMN, "current_and_seven_history_total_expenditure"),
        (TRIM_RESOURCE_COLUMN, "current_annual_household_income"),
    ]:
        variable_reference = reference.loc[
            reference[variable].notna(), ["calendar_year", variable]
        ].copy()
        cutoffs = variable_reference.groupby("calendar_year", observed=True)[variable].quantile(0.99)
        cutoff_maps[variable] = {int(year): float(value) for year, value in cutoffs.items()}
        for year, cutoff in cutoffs.items():
            values = variable_reference.loc[variable_reference["calendar_year"].eq(year), variable]
            audit_rows.append(
                {
                    "task_id": TASK_ID,
                    "field": variable,
                    "field_scope": scope,
                    "calendar_year": int(year),
                    "reference_rows": int(len(values)),
                    "zero_rows": int(values.eq(0).sum()),
                    "p99_cutoff": float(cutoff),
                    "rows_equal_p99": int(values.eq(cutoff).sum()),
                    "rows_above_p99": int(values.gt(cutoff).sum()),
                    "cutoff_universe": "full_1986_1996_source_year_household_day_panel_before_task_restrictions",
                    "pool_rule": "calendar_year_all_observed_values",
                    "zero_rule": "retained_and_included_in_cutoff_distribution",
                    "lower_tail_rule": "not_trimmed",
                    "boundary_rule": "drop_strictly_above_p99_retain_equality",
                }
            )

    total_cutoff_name = f"{TRIM_DAILY_COLUMN}_calendar_year_p99"
    income_cutoff_name = f"{TRIM_RESOURCE_COLUMN}_calendar_year_p99"
    out[total_cutoff_name] = out["target_calendar_year"].map(cutoff_maps[TRIM_DAILY_COLUMN])
    out[income_cutoff_name] = out["target_calendar_year"].map(cutoff_maps[TRIM_RESOURCE_COLUMN])
    if out[[total_cutoff_name, income_cutoff_name]].isna().any().any():
        raise RuntimeError(f"{TASK_ID}: current-year upper-tail cutoff mapping is incomplete.")

    out["daily_total_expenditure_upper_tail_trigger"] = (
        out[TRIM_DAILY_COLUMN].gt(out[total_cutoff_name])
    ).astype(int)
    history_trigger = pd.Series(False, index=out.index)
    for lag in range(1, 8):
        history_value = out[f"{TRIM_DAILY_COLUMN}_lag{lag}"]
        history_year = out[f"prior_observed_date_lag{lag}"].dt.year
        history_cutoff = history_year.map(cutoff_maps[TRIM_DAILY_COLUMN])
        missing_cutoff = history_value.notna() & history_cutoff.isna()
        if missing_cutoff.any():
            missing_years = sorted(history_year.loc[missing_cutoff].dropna().astype(int).unique())
            raise RuntimeError(f"{TASK_ID}: no total-expenditure cutoff for lag-{lag} years {missing_years}.")
        history_trigger |= history_value.gt(history_cutoff)

    out["annual_income_before_tax_upper_tail_trigger"] = (
        out[TRIM_RESOURCE_COLUMN].gt(out[income_cutoff_name])
    ).astype(int)
    out["direct_target_upper_tail_trigger"] = out["daily_total_expenditure_upper_tail_trigger"]
    out["history_target_upper_tail_trigger"] = history_trigger.astype(int)
    out["annual_resource_upper_tail_trigger"] = out["annual_income_before_tax_upper_tail_trigger"]
    out["outlier_retained"] = (
        out[[
            "direct_target_upper_tail_trigger",
            "history_target_upper_tail_trigger",
            "annual_resource_upper_tail_trigger",
        ]].eq(0).all(axis=1)
    ).astype(int)
    return out, pd.DataFrame(audit_rows)


def entity_counts(frame: pd.DataFrame) -> tuple[int, int, int]:
    return int(frame["cu_id"].nunique()), int(frame["week_key"].nunique()), int(len(frame))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in RETIRED_RAW_TASK_AUDITS:
        (OUTPUT_DIR / name).unlink(missing_ok=True)
    daily, build_manifest = require_intermediate()
    calendar = build_oasdi_calendar(1985, 1997)
    if len(calendar) != 13 * 12 or calendar["payment_month"].duplicated().any():
        raise RuntimeError(f"{TASK_ID}: OASDI calendar does not uniquely cover 1985--1997.")
    if not calendar["programme"].eq("OASDI").all() or not calendar["calendar_version"].eq(CALENDAR_VERSION).all():
        raise RuntimeError(f"{TASK_ID}: OASDI calendar programme or version is invalid.")
    delivery_holidays: dict[object, str] = {}
    for calendar_year in range(1984, 1999):
        delivery_holidays.update(oasdi_delivery_holidays(calendar_year))
    scheduled_business_day = (
        calendar["scheduled_payment_date"].dt.dayofweek.lt(5)
        & ~calendar["scheduled_payment_date"].dt.date.isin(delivery_holidays)
    )
    if not scheduled_business_day.all() or calendar["scheduled_payment_date"].gt(calendar["normal_payment_date"]).any():
        raise RuntimeError(f"{TASK_ID}: OASDI calendar violates the preceding-business-day rule.")
    for month, expected in SPOT_CHECKS.items():
        actual = calendar.loc[
            calendar["payment_month"].eq(month), "scheduled_payment_date"
        ].dt.strftime("%Y-%m-%d")
        if len(actual) != 1 or actual.iloc[0] != expected:
            raise RuntimeError(f"{TASK_ID}: OASDI calendar spot check failed for {month}.")

    input_audit = {
        "task_id": TASK_ID,
        "intermediate": str(INTERMEDIATE_PATH.relative_to(PROJECT_ROOT)),
        "intermediate_sha256": sha256_file(INTERMEDIATE_PATH),
        "build_manifest_sha256": build_manifest.get("intermediate_sha256"),
        "source_day_rows": len(daily),
        "source_day_id_unique": bool(~daily["source_day_id"].duplicated().any()),
        "canonical_years": sorted(pd.to_numeric(daily["source_year"], errors="raise").astype(int).unique().tolist()),
        "strict_target_version": STRICT_TARGET_VERSION,
    }
    (OUTPUT_DIR / "01_intermediate_input_audit.json").write_text(
        json.dumps(input_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # These aliases are the only daily target inputs to this task.  The broad
    # all-years total remains available to other users of the intermediate.
    daily["daily_total_expenditure"] = daily["daily_total_expenditure_legacy_strict_v1"]
    daily["daily_food_at_home"] = daily["daily_food_at_home_legacy_strict_v1"]
    daily["daily_food_away_from_home"] = daily["daily_food_away_from_home_legacy_strict_v1"]

    # Preserve the old member-specific Stephens definitions.  The intermediate
    # also carries CU-level summaries, but two 'only' concepts are person-level:
    # a spouse may be Railroad-only while the reference person reports OASDI.
    ref_oasdi = daily["reference_person_oasdi_affirmative"].fillna(False).astype(bool)
    ref_ssi = daily["reference_person_ssi_affirmative"].fillna(False).astype(bool)
    ref_railroad = daily["reference_person_railroad_affirmative"].fillna(False).astype(bool)
    spouse_oasdi = daily["spouse_oasdi_affirmative"].fillna(False).astype(bool)
    spouse_ssi = daily["spouse_ssi_affirmative"].fillna(False).astype(bool)
    spouse_railroad = daily["spouse_railroad_affirmative"].fillna(False).astype(bool)
    daily["reference_or_spouse_oasdi"] = (ref_oasdi | spouse_oasdi).astype(int)
    daily["reference_or_spouse_ssi"] = (ref_ssi | spouse_ssi).astype(int)
    daily["reference_or_spouse_railroad"] = (ref_railroad | spouse_railroad).astype(int)
    daily["reference_or_spouse_dual_oasdi_ssi"] = ((ref_oasdi & ref_ssi) | (spouse_oasdi & spouse_ssi)).astype(int)
    daily["reference_or_spouse_ssi_only"] = ((ref_ssi & ~ref_oasdi) | (spouse_ssi & ~spouse_oasdi)).astype(int)
    daily["reference_or_spouse_railroad_only"] = (
        (ref_railroad & ~ref_oasdi & ~ref_ssi) | (spouse_railroad & ~spouse_oasdi & ~spouse_ssi)
    ).astype(int)
    daily["reference_or_spouse_paper_recipient"] = (
        daily["reference_or_spouse_oasdi"].eq(1) | daily["reference_or_spouse_railroad"].eq(1)
    ).astype(int)
    daily["recipient_class"] = np.select(
        [
            daily["reference_or_spouse_oasdi"].eq(1) & daily["reference_or_spouse_railroad"].eq(1),
            daily["reference_or_spouse_oasdi"].eq(1) & daily["reference_or_spouse_ssi"].eq(1),
            daily["reference_or_spouse_oasdi"].eq(1),
            daily["reference_or_spouse_railroad"].eq(1),
            daily["reference_or_spouse_ssi"].eq(1),
        ],
        ["OASDI_and_Railroad", "OASDI_and_SSI", "pure_OASDI", "Railroad_only", "SSI_only"],
        default="excluded_ambiguous_or_no_affirmative_recipient",
    )
    daily["social_security_share"] = daily["annual_social_security_railroad_income"].div(daily["annual_income_before_tax"])

    step_rows: list[dict[str, object]] = []
    drop_rows: list[dict[str, object]] = []

    def add_stage(step: int, label: str, before: int, frame: pd.DataFrame, filter_id: str, condition: str) -> None:
        units, weeks, days = entity_counts(frame)
        step_rows.append({
            "task_id": TASK_ID, "step": step, "step_label": label, "rows_before": before,
            "rows_after": days, "rows_dropped": before - days, "consumer_units_after": units,
            "consumer_unit_weeks_after": weeks, "consumer_unit_days_after": days,
        })
        drop_rows.append(filter_count(
                             TASK_ID, step, 1,
                             filter_id, condition, before - days,
                         ))

    # Row 1: every canonical daily source observation, unfiltered.
    source_daily = daily
    add_stage(
        1,
        "Start from source observations",
        len(source_daily),
        source_daily,
        "source_observations",
        "start from validated CEX Diary consumer-unit days, with one unique record for each source day",
    )

    # Row 2: every source row remains after calendar/macro attachment.  Calendar
    # fields are intentionally unavailable outside task years, not filtered out.
    daily = attach_additional_sources(source_daily, calendar)
    add_stage(2, "Attach additional sources", len(source_daily), daily, "additional_source_attachment", "calendar and macro attachment flags retained without filtering")

    # Row 3: histories exist as fields; records lacking a usable history remain.
    daily = add_history(daily)
    add_stage(3, "Construct leads and lags", len(daily), daily, "strict_observed_histories", "strict prior observed dates and target histories are attached without filtering")

    explicit_diary_quality = (
        daily["both_diary_week_records_available"].fillna(False).astype(bool)
        & daily["both_diary_week_numbers_available"].fillna(False).astype(bool)
        & daily["legacy_strict_valid_start_week_records"].eq(2)
        & daily["legacy_strict_valid_dated_week_records"].eq(2)
        & daily["pickup_complete_records"].eq(2)
        & daily["weekn_equals_two_records"].eq(2)
        & daily["nonoverlapping_diary_week_dates"].fillna(False).astype(bool)
    )
    upstream_diary_quality = daily["stephens_1986_1996_quality_flag_v1"].fillna(False).astype(bool)
    in_scope = daily["source_year"].isin(YEARS)
    if not explicit_diary_quality.loc[in_scope].eq(upstream_diary_quality.loc[in_scope]).all():
        raise RuntimeError(f"{TASK_ID}: explicit two-week Diary conditions differ from the upstream audit flag.")
    paper_universe_mask = (
        in_scope
        & explicit_diary_quality
        & daily["reference_or_spouse_paper_recipient"].eq(1)
        & daily["response_income_complete_records"].eq(2)
        & daily["positive_combined_ss_railroad_income_records"].eq(2)
        & daily["diary_date"].notna()
    )
    daily["paper_universe_task"] = paper_universe_mask.astype(int)
    paper_daily = daily.loc[paper_universe_mask].copy()
    if paper_daily.empty:
        raise RuntimeError(f"{TASK_ID}: the paper-comparison Diary universe is empty.")

    # Row 4: economic universe only.
    universe_mask = (
        in_scope & daily["diary_date"].notna() & daily["reference_or_spouse_oasdi"].eq(1)
        & daily["calendar_attachment_success"]
        & ~daily["scheduled_payment_date"].dt.month.eq(1)
        & daily["event_time"].between(EVENT_MIN, EVENT_MAX)
    )
    universe_pool = daily.loc[universe_mask].copy()
    add_stage(
        4,
        "Restrict universe",
        len(daily),
        universe_pool,
        "oasdi_task_universe",
        "keep dated 1986 through 1996 observations within seven days of a scheduled non-January Social Security retirement or disability payment received by the reference person or spouse; exclude Railroad-Retirement-only households",
    )

    target_mask = (
        universe_pool["stephens_food_available"].fillna(False).astype(bool)
        & universe_pool["stephens_food_version"].eq(STRICT_TARGET_VERSION)
        & universe_pool[TARGET_COLUMNS].notna().all(axis=1)
        & np.isfinite(universe_pool[TARGET_COLUMNS]).all(axis=1)
    )
    target_pool = universe_pool.loc[target_mask].copy()
    target_pool["direct_target_observed"] = 1
    add_stage(5, "Require observed targets", len(universe_pool), target_pool, "observed_strict_targets", "the three strict-version current targets are observed and finite")

    history_columns = [
        *[f"prior_observed_date_lag{lag}" for lag in range(1, 8)],
        *[f"calendar_day_gap_lag{lag}" for lag in range(1, 8)],
        *[f"{target}_lag{lag}" for lag in range(1, 8) for target in TARGET_COLUMNS],
    ]
    numeric_predictors = [
        "age_ref", "household_size", "children_under_18", "number_adults",
        "annual_income_before_tax", "annual_social_security_railroad_income",
        "likely_social_security_recipients", *CORE_MACRO_COLUMNS,
    ]
    categorical_predictors = [
        "sex", "race", "education", "marital_status", "region", "urban_status", "home_tenure",
        "reference_person_work_status", "recipient_class",
    ]
    predictor_mask = (
        target_pool[history_columns].notna().all(axis=1)
        & np.isfinite(target_pool[[f"{target}_lag{lag}" for lag in range(1, 8) for target in TARGET_COLUMNS]]).all(axis=1)
        & target_pool[numeric_predictors].notna().all(axis=1)
        & np.isfinite(target_pool[numeric_predictors]).all(axis=1)
        & target_pool["macro_attachment_success"]
    )
    for lag in range(1, 8):
        predictor_mask &= target_pool[f"calendar_day_gap_lag{lag}"].eq(lag)
    for column in categorical_predictors:
        predictor_mask &= ~target_pool[column].astype("string").str.casefold().isin(["", "not reported", "nan", "<na>"])
    predictor_pool = target_pool.loc[predictor_mask].copy()
    predictor_pool["predictor_complete"] = 1
    add_stage(6, "Require observed predictors", len(target_pool), predictor_pool, "observed_predictors", "seven immediately preceding calendar days and observed finite household, recipient, area, resource, and macro predictors")

    benefit_share_valid = (
        np.isfinite(predictor_pool["social_security_share"])
        & predictor_pool["social_security_share"].gt(0)
        & predictor_pool["social_security_share"].lt(1)
    )
    validity_masks: list[tuple[str, str, pd.Series]] = [
        (
            "two_source_week_records",
            "require exactly two source weekly diary records for the consumer unit",
            predictor_pool["both_diary_week_records_available"].fillna(False).astype(bool),
        ),
        ("diary_week_numbers_one_and_two", "observed Diary week numbers are exactly 1 and 2", predictor_pool["both_diary_week_numbers_available"].fillna(False).astype(bool)),
        ("two_valid_start_weeks", "both weeks have a valid start date in the source year", predictor_pool["legacy_strict_valid_start_week_records"].eq(2)),
        ("two_valid_dated_weeks", "both weeks contain valid exact-dated transactions not all placed on the week-start date", predictor_pool["legacy_strict_valid_dated_week_records"].eq(2)),
        (
            "two_complete_pickups",
            "require both weekly diary records to have a completed pickup indicator",
            predictor_pool["pickup_complete_records"].eq(2),
        ),
        ("two_week_design", "both weekly records have WEEKN == 2", predictor_pool["weekn_equals_two_records"].eq(2)),
        ("nonoverlapping_week_dates", "both week-start dates are nonmissing and at least seven days apart", predictor_pool["nonoverlapping_diary_week_dates"].fillna(False).astype(bool)),
        ("two_complete_income_records", "both source-week records contain a complete annual-income response", predictor_pool["response_income_complete_records"].eq(2)),
        ("current_targets_nonnegative", "all current spending targets are nonnegative", predictor_pool[TARGET_COLUMNS].ge(0).all(axis=1)),
        ("current_food_within_total", "current food components do not exceed current total expenditure", predictor_pool["daily_total_expenditure"].add(1e-8).ge(predictor_pool["daily_food_at_home"] + predictor_pool["daily_food_away_from_home"])),
        ("annual_income_positive", "annual household income is strictly positive", predictor_pool["annual_income_before_tax"].gt(0)),
        ("annual_benefit_income_positive", "annual Social Security/Railroad income is strictly positive", predictor_pool["annual_social_security_railroad_income"].gt(0)),
        ("two_positive_benefit_records", "both source-week records report positive combined benefit income", predictor_pool["positive_combined_ss_railroad_income_records"].eq(2)),
        ("benefit_share_open_unit_interval", "benefit-income share is finite and strictly between zero and one", benefit_share_valid),
        ("adult_count_valid", "adult count is a nonnegative integer", predictor_pool["number_adults_valid"].fillna(False).astype(bool) & predictor_pool["number_adults"].ge(0) & np.isclose(predictor_pool["number_adults"], np.round(predictor_pool["number_adults"]))),
    ]
    for lag in range(1, 8):
        lag_columns = [f"{target}_lag{lag}" for target in TARGET_COLUMNS]
        validity_masks.extend([
            (f"lag{lag}_targets_nonnegative", f"all lag-{lag} spending targets are nonnegative", predictor_pool[lag_columns].ge(0).all(axis=1)),
            (f"lag{lag}_food_within_total", f"lag-{lag} food components do not exceed total expenditure", predictor_pool[f"daily_total_expenditure_lag{lag}"].add(1e-8).ge(predictor_pool[f"daily_food_at_home_lag{lag}"] + predictor_pool[f"daily_food_away_from_home_lag{lag}"])),
        ])
    logical_mask = pd.Series(True, index=predictor_pool.index)
    for _, _, mask in validity_masks:
        logical_mask &= mask
    logical_pool = predictor_pool.loc[logical_mask].copy()
    logical_pool["logical_valid"] = 1
    logical_pool["calendar_merge"] = 1
    add_stage(7, "Apply logical and validity filters", len(predictor_pool), logical_pool, "logical_validity", "explicit two-week Diary, spending-domain, component-total, resource, benefit-share, and adult-count restrictions")
    drop_rows.pop()
    for detail_order, (filter_id, condition, mask) in enumerate(validity_masks, start=1):
        drop_rows.append(filter_count(
                             TASK_ID, 7, detail_order,
                             filter_id, condition, int((~mask).sum()),
                         ))

    candidate_row_id = logical_pool["cu_id"] + ":" + logical_pool["diary_date"].dt.strftime("%Y%m%d")
    assertions = {
        "release date later than diary date": logical_pool["release_date"].gt(logical_pool["diary_date"]),
        "canonical calendar version": logical_pool["calendar_version"].eq(CALENDAR_VERSION),
        "event-time arithmetic": (logical_pool["diary_date"] - logical_pool["scheduled_payment_date"]).dt.days.eq(logical_pool["event_time"]),
        "nonmissing unique CU-day identifier": candidate_row_id.notna() & ~candidate_row_id.duplicated(keep=False),
    }
    for lag in range(1, 8):
        assertions[f"lag-{lag} date arithmetic"] = logical_pool[f"calendar_day_gap_lag{lag}"].eq(
            (logical_pool["diary_date"] - logical_pool[f"prior_observed_date_lag{lag}"]).dt.days
        )
    failed_assertions = {name: int((~mask).sum()) for name, mask in assertions.items() if not mask.all()}
    if failed_assertions:
        raise RuntimeError(f"{TASK_ID}: post-construction assertions failed: {failed_assertions}")

    logical_pool, cutoff_audit = apply_upper_tails(logical_pool, daily)
    for target in TARGET_COLUMNS:
        logical_pool[f"naive_{target}"] = logical_pool[f"{target}_lag1"]

    selection_pool = logical_pool.loc[logical_pool["outlier_retained"].eq(1)].copy()
    add_stage(8, "Apply outlier trimming", len(logical_pool), selection_pool, "calendar_year_p99", "current and seven-day-history total expenditure and annual household income do not exceed their full-panel calendar-year p99; equality, zeros, food components, benefit income, and the lower tail remain")
    if selection_pool.empty:
        raise RuntimeError(f"{TASK_ID}: no rows remain after the task filters.")
    selection_pool["paper_universe"] = 1
    selection_pool["calendar_schedule_type"] = "OASDI_third_of_month_pre_1997_cycle"
    selection_pool["selection_period"] = selection_pool["diary_date"].dt.strftime("%Y-%m")
    selection_pool["row_id"] = selection_pool["cu_id"] + ":" + selection_pool["diary_date"].dt.strftime("%Y%m%d")
    if selection_pool["row_id"].duplicated().any():
        raise RuntimeError(f"{TASK_ID}: final event-window row ID is not unique.")

    selection_pool["id"] = selection_pool["row_id"].astype("string")
    selection_pool["time"] = selection_pool["diary_date"].dt.strftime("%Y-%m-%d").astype("string")
    # Public keys are fixed before sampling; source keys retain their sort types.
    selection_pool["release_date"] = pd.to_datetime(
        selection_pool["release_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    sample, _, sampling_summary = sample_task_records(
        selection_pool,
        task_id=TASK_ID,
        output_dir=SAMPLING_DIR,
        output_sort_cols=["source_year", "diary_date", "cu_id", "week_key"],
    )
    sample["subject_id"] = sample["cu_id"]
    sample["group_id"] = sample["cu_id"]
    sample["prompt_time"] = sample["diary_date"].dt.strftime("%Y-%m-%d")
    for column in ["scheduled_payment_date", "normal_payment_date"]:
        sample[column] = sample[column].dt.strftime("%Y-%m-%d")
    for lag in range(1, 8):
        sample[f"prior_observed_date_lag{lag}"] = sample[f"prior_observed_date_lag{lag}"].dt.strftime("%Y-%m-%d")
    sample["selection_seed"] = 42
    sample["selection_hash_algorithm"] = "python_random_mt19937_shuffle_v1"

    output_columns = [
        "row_id", "subject_id", "group_id", "cu_id", "selection_period", "prompt_time", "release_date", "release_date_source",
        "source_year", "source_family", "source_archive", "week_key", "diary_week", "diary_day_sequence", "diary_date",
        "scheduled_payment_date", "normal_payment_date", "adjustment_reason", "calendar_schedule_type", "programme", "calendar_version", "calendar_source", "calendar_merge",
        "event_time", "day_of_week", "is_weekend", "is_federal_holiday", "federal_holiday_name", "adjacent_to_federal_holiday",
        "paper_universe", "direct_target_observed", "predictor_complete", "logical_valid", "outlier_retained",
        "age_ref", "sex", "race", "education", "marital_status", "household_size", "children_under_18", "region", "urban_status", "home_tenure", "reference_person_work_status",
        "annual_income_before_tax", "annual_social_security_railroad_income", "social_security_share", "likely_social_security_recipients",
        "reference_or_spouse_oasdi", "reference_or_spouse_ssi", "reference_or_spouse_railroad",
        "reference_or_spouse_dual_oasdi_ssi", "reference_or_spouse_ssi_only", "reference_or_spouse_railroad_only",
        "reference_or_spouse_paper_recipient", "recipient_class",
        *TARGET_COLUMNS,
        *[f"prior_observed_date_lag{lag}" for lag in range(1, 8)],
        *[f"calendar_day_gap_lag{lag}" for lag in range(1, 8)],
        *[f"{target}_lag{lag}" for lag in range(1, 8) for target in TARGET_COLUMNS],
        "target_calendar_year",
        "daily_total_expenditure_calendar_year_p99", "annual_income_before_tax_calendar_year_p99",
        "daily_total_expenditure_upper_tail_trigger", "annual_income_before_tax_upper_tail_trigger",
        "direct_target_upper_tail_trigger", "history_target_upper_tail_trigger", "annual_resource_upper_tail_trigger",
        "naive_daily_total_expenditure", "naive_daily_food_at_home", "naive_daily_food_away_from_home",
        "analysis_weight", "macro_context_merge", *CORE_MACRO_COLUMNS,
        "selection_seed", "selection_hash_algorithm",
    ]
    missing = sorted(set(output_columns) - set(sample.columns))
    if missing:
        raise RuntimeError(f"{TASK_ID}: final table misses columns {missing}.")
    table = sample[output_columns].copy()
    if len(table.columns) != 131:
        raise RuntimeError(f"{TASK_ID}: expected 131 internal construction columns, found {len(table.columns)}.")
    if table.empty or table["row_id"].duplicated().any() or table["subject_id"].ne(table["group_id"]).any():
        raise RuntimeError(f"{TASK_ID}: final row IDs or consumer-unit grouping are invalid.")
    if table[TARGET_COLUMNS + ["naive_daily_total_expenditure", "naive_daily_food_at_home", "naive_daily_food_away_from_home"]].isna().any().any():
        raise RuntimeError(f"{TASK_ID}: final direct target or naive baseline is missing.")
    if set(table["source_year"].astype(int)) != set(YEARS):
        raise RuntimeError(f"{TASK_ID}: final task table does not cover all 1986--1996 source years.")
    if (
        table["analysis_weight"].isna().any()
        or (~np.isfinite(table["analysis_weight"])).any()
        or table["analysis_weight"].le(0).any()
        or table["analysis_weight"].nunique() <= 1
    ):
        raise RuntimeError(f"{TASK_ID}: final Diary analysis weights are missing, nonpositive, or constant.")
    diary_date = pd.to_datetime(table["diary_date"], errors="raise")
    table["id"] = table["row_id"]
    table["time"] = table["prompt_time"]
    table["n_adults"] = table["household_size"].astype(int) - table["children_under_18"].astype(int)
    table["current_day_of_week"] = diary_date.dt.day_name()
    table["current_day_of_month"] = diary_date.dt.day.astype(int)
    table["current_month"] = diary_date.dt.month_name()
    write_public_table(table, task_id=TASK_ID, public_path=TABLE_PATH)
    # Event time and survey weight support the task-specific descriptive figures
    # but remain outside the model-facing public table and prompt contract.
    pd.DataFrame({
        "id": table["id"].astype("string"),
        "time": sample["prompt_time"].astype("string"),
        "event_time": table["event_time"].astype(int),
        "analysis_weight": table["analysis_weight"].astype(float),
    }).to_csv(OUTPUT_DIR / "01_event_time_plot_inputs.csv", index=False)
    add_stage(9, "Export sample", len(selection_pool), table, "deterministic_export", "canonical CU-day row IDs, CU grouping, and deterministic time-stratified selection")

    step_frame = pd.DataFrame(step_rows, columns=STEP_COLUMNS)
    step_frame[["task_id", "step", "step_label", "rows_before", "rows_after", "rows_dropped"]].to_csv(
        OUTPUT_DIR / "01_sample_construction_steps.csv", index=False
    )
    step_frame.to_csv(OUTPUT_DIR / "01_sample_construction_entity_counts.csv", index=False)
    pd.DataFrame(drop_rows, columns=DETAIL_COLUMNS).to_csv(OUTPUT_DIR / "01_sample_construction_drop_details.csv", index=False)
    cutoff_audit.to_csv(OUTPUT_DIR / "01_outlier_cutoffs.csv", index=False)

    source_entities = pd.DataFrame([
        {"task_id": TASK_ID, "stage": "general_intermediate", "consumer_units": int(daily["cu_id"].nunique()), "consumer_unit_weeks": int(daily["week_key"].nunique()), "consumer_unit_days": len(daily)},
        {"task_id": TASK_ID, "stage": "paper_universe", "consumer_units": int(paper_daily["cu_id"].nunique()), "consumer_unit_weeks": int(paper_daily["week_key"].nunique()), "consumer_unit_days": len(paper_daily)},
        {"task_id": TASK_ID, "stage": "scored_selected", "consumer_units": int(table["subject_id"].nunique()), "consumer_unit_weeks": int(sample["week_key"].nunique()), "consumer_unit_days": len(table)},
    ])
    source_entities.to_csv(OUTPUT_DIR / "01_source_entity_audit.csv", index=False)

    recipient_by_cu = daily.loc[daily["source_year"].isin(YEARS)].sort_values("source_day_id", kind="mergesort").drop_duplicates("cu_id")
    recipient_by_cu["paper_universe_consumer_unit"] = recipient_by_cu["cu_id"].isin(set(paper_daily["cu_id"])).astype(int)
    programme_audit = recipient_by_cu.groupby("recipient_class", observed=True).agg(
        all_classified_consumer_units=("cu_id", "nunique"),
        paper_universe_consumer_units=("paper_universe_consumer_unit", "sum"),
        reference_or_spouse_paper_recipient=("reference_or_spouse_paper_recipient", "max"),
    ).reset_index()
    selected_by_class = sample.groupby("recipient_class", observed=True).agg(
        final_selected_rows=("row_id", "size"),
        final_selected_consumer_units=("cu_id", "nunique"),
    ).reset_index()
    programme_audit = programme_audit.merge(selected_by_class, on="recipient_class", how="left", validate="one_to_one")
    programme_audit[["final_selected_rows", "final_selected_consumer_units"]] = programme_audit[
        ["final_selected_rows", "final_selected_consumer_units"]
    ].fillna(0).astype(int)
    programme_audit["task_id"] = TASK_ID
    programme_audit["inclusion_rule"] = "benchmark-eligible only if the reference person or spouse reports OASDI, with complete income and positive annual combined Social Security/Railroad income in both Diary weeks"
    programme_audit["schedule_scope_note"] = (
        "Railroad-only consumer units are excluded because the displayed OASDI payment rule does not establish Railroad Retirement timing."
    )
    programme_audit.to_csv(OUTPUT_DIR / "01_programme_recipient_audit.csv", index=False)

    pd.DataFrame([
        {"task_id": TASK_ID, "sample": "paper_compatible_days", "rows": len(paper_daily), "missing_weight_rows": int(paper_daily["analysis_weight"].isna().sum()), "nonfinite_weight_rows": int((paper_daily["analysis_weight"].notna() & ~np.isfinite(paper_daily["analysis_weight"])).sum()), "nonpositive_weight_rows": int(paper_daily["analysis_weight"].notna().sum() - paper_daily["analysis_weight"].gt(0).sum()), "positive_weight_rows": int(paper_daily["analysis_weight"].gt(0).sum()), "distinct_weights": int(paper_daily["analysis_weight"].nunique()), "minimum_weight": float(paper_daily["analysis_weight"].min()), "maximum_weight": float(paper_daily["analysis_weight"].max()), "weight_field": "analysis_weight", "weight_rule": "post-construction audit of the validated general Diary final weight; the weight does not select rows"},
        {"task_id": TASK_ID, "sample": "selected_public_days", "rows": len(table), "missing_weight_rows": int(table["analysis_weight"].isna().sum()), "nonfinite_weight_rows": int((table["analysis_weight"].notna() & ~np.isfinite(table["analysis_weight"])).sum()), "nonpositive_weight_rows": int(table["analysis_weight"].notna().sum() - table["analysis_weight"].gt(0).sum()), "positive_weight_rows": int(table["analysis_weight"].gt(0).sum()), "distinct_weights": int(table["analysis_weight"].nunique()), "minimum_weight": float(table["analysis_weight"].min()), "maximum_weight": float(table["analysis_weight"].max()), "weight_field": "analysis_weight", "weight_rule": "post-construction validation of the general Diary final weight; the weight does not select rows"},
    ]).to_csv(OUTPUT_DIR / "01_weight_audit.csv", index=False)

    source_year_audit = daily.loc[daily["source_year"].isin(YEARS)].groupby("source_year", observed=True).agg(
        general_consumer_unit_weeks=("week_key", "nunique"), general_consumer_units=("cu_id", "nunique"),
        strict_available_days=("daily_total_expenditure", lambda x: int(x.notna().sum())),
    ).reset_index()
    paper_by_year = paper_daily.groupby("source_year", observed=True).agg(
        paper_consumer_units=("cu_id", "nunique"), paper_recipient_days=("source_day_id", "size"),
    ).reset_index()
    selected_years = table.groupby("source_year", observed=True).agg(
        selected_rows=("row_id", "size"), selected_consumer_units=("subject_id", "nunique"),
    ).reset_index()
    source_year_audit = source_year_audit.merge(paper_by_year, on="source_year", how="left", validate="one_to_one").merge(selected_years, on="source_year", how="left", validate="one_to_one")
    source_year_audit["source_year_covered"] = source_year_audit["selected_rows"].gt(0).astype(int)
    source_year_audit.to_csv(OUTPUT_DIR / "01_source_year_coverage_audit.csv", index=False)

    calendar_audit = calendar.loc[calendar["payment_month"].between("1986-01", "1996-12")].copy()
    calendar_audit["scheduled_is_business_day"] = (
        calendar_audit["scheduled_payment_date"].dt.dayofweek.lt(5)
        & ~calendar_audit["scheduled_payment_date"].dt.date.isin(delivery_holidays)
    ).astype(int)
    calendar_audit["scheduled_after_normal"] = calendar_audit["scheduled_payment_date"].gt(calendar_audit["normal_payment_date"]).astype(int)
    calendar_audit.to_csv(OUTPUT_DIR / "01_payment_calendar_audit.csv", index=False)

    target_provenance = pd.DataFrame([
        {"task_id": TASK_ID, "target": "daily_total_expenditure", "intermediate_field": "daily_total_expenditure_legacy_strict_v1", "version": STRICT_TARGET_VERSION, "available_rows_1986_1996": int(daily.loc[daily["source_year"].isin(YEARS), "daily_total_expenditure"].notna().sum()), "construction_audit": "output/preprocessing/cex_diary/build/02_legacy_strict_1986_1996_compatibility.csv"},
        {"task_id": TASK_ID, "target": "daily_food_at_home", "intermediate_field": "daily_food_at_home_legacy_strict_v1", "version": STRICT_TARGET_VERSION, "available_rows_1986_1996": int(daily.loc[daily["source_year"].isin(YEARS), "daily_food_at_home"].notna().sum()), "construction_audit": "output/preprocessing/cex_diary/build/02_legacy_strict_1986_1996_compatibility.csv"},
        {"task_id": TASK_ID, "target": "daily_food_away_from_home", "intermediate_field": "daily_food_away_from_home_legacy_strict_v1", "version": STRICT_TARGET_VERSION, "available_rows_1986_1996": int(daily.loc[daily["source_year"].isin(YEARS), "daily_food_away_from_home"].notna().sum()), "construction_audit": "output/preprocessing/cex_diary/build/02_legacy_strict_1986_1996_compatibility.csv"},
    ])
    target_provenance.to_csv(OUTPUT_DIR / "01_target_provenance_audit.csv", index=False)

    baseline_rows: list[dict[str, object]] = []
    for target in TARGET_COLUMNS:
        exact = logical_pool[f"naive_{target}"].eq(logical_pool[f"{target}_lag1"])
        if not exact.all():
            raise RuntimeError(f"{TASK_ID}: previous-calendar-day baseline differs for {target}.")
        baseline_rows.append({
            "task_id": TASK_ID,
            "target": target,
            "evaluated_rows": int(len(logical_pool)),
            "calendar_day_gap_lag1_minimum": int(logical_pool["calendar_day_gap_lag1"].min()),
            "calendar_day_gap_lag1_maximum": int(logical_pool["calendar_day_gap_lag1"].max()),
            "maximum_absolute_lag1_difference": float(
                (logical_pool[f"naive_{target}"] - logical_pool[f"{target}_lag1"]).abs().max()
            ),
            "invariant": 1,
            "scope": "naive forecast equals expenditure on the immediately preceding calendar day",
        })
    pd.DataFrame(baseline_rows).to_csv(OUTPUT_DIR / "01_baseline_invariance_audit.csv", index=False)

    history_availability_rows = []
    for event_time, part in universe_pool.groupby("event_time", sort=True, observed=True):
        history_availability_rows.append({
            "task_id": TASK_ID, "event_time": int(event_time), "candidate_target_days": len(part),
            **{f"rows_with_at_least_{lag}_prior_days": int(part[f"prior_observed_date_lag{lag}"].notna().sum()) for lag in range(1, 8)},
            "rows_selected_after_all_rules": int(selection_pool["event_time"].eq(event_time).sum()),
        })
    pd.DataFrame(history_availability_rows).to_csv(OUTPUT_DIR / "01_history_availability_and_selection.csv", index=False)
    pd.DataFrame([
        {"task_id": TASK_ID, "screen_scope": "current_total_expenditure", "excluded_rows": int(logical_pool["direct_target_upper_tail_trigger"].eq(1).sum()), "rule": "flag current total expenditure strictly above the full-panel calendar-year p99"},
        {"task_id": TASK_ID, "screen_scope": "seven_day_total_expenditure_history", "excluded_rows": int(logical_pool["history_target_upper_tail_trigger"].eq(1).sum()), "rule": "flag any of seven total-expenditure histories strictly above its history-date full-panel calendar-year p99"},
        {"task_id": TASK_ID, "screen_scope": "annual_household_income", "excluded_rows": int(logical_pool["annual_resource_upper_tail_trigger"].eq(1).sum()), "rule": "flag annual household income strictly above the full-panel target-calendar-year p99"},
        {"task_id": TASK_ID, "screen_scope": "combined_union", "excluded_rows": int(logical_pool["outlier_retained"].eq(0).sum()), "rule": "drop the union of the three option-2 upper-tail flags; equality, zeros, food components, benefit income, and the lower tail remain"},
    ]).to_csv(OUTPUT_DIR / "01_upper_tail_exclusion_audit.csv", index=False)
    pd.DataFrame([
        {
            "task_id": TASK_ID,
            "evaluated_rows": len(predictor_pool),
            "valid_rows": int(benefit_share_valid.sum()),
            "invalid_rows": int((~benefit_share_valid).sum()),
            "invalid_at_or_above_one": int(predictor_pool["social_security_share"].ge(1).sum()),
            "invalid_nonpositive": int(predictor_pool["social_security_share"].le(0).sum()),
            "rule": "annual Social Security and Railroad Retirement income divided by annual household income must be finite, positive, and strictly below 1",
        }
    ]).to_csv(OUTPUT_DIR / "01_benefit_share_validity_audit.csv", index=False)
    pd.DataFrame([
        {"field": "annual_income_before_tax", "source": "validated intermediate annual_income_before_tax", "model_label": "reported annual household income", "unit": "current U.S. dollars per year"},
        {"field": "annual_social_security_railroad_income", "source": "validated intermediate annual_social_security_railroad_income", "model_label": "reported annual combined Social Security and Railroad Retirement income", "unit": "current U.S. dollars per year"},
        {"field": "social_security_share", "source": "annual_social_security_railroad_income / annual_income_before_tax", "model_label": "benefit income as share of reported household income", "unit": "ratio"},
        {"field": "daily_total_expenditure", "source": "daily_total_expenditure_legacy_strict_v1", "model_label": "total_expenditure", "unit": "current U.S. dollars on target day"},
        {"field": "daily_food_at_home", "source": "daily_food_at_home_legacy_strict_v1", "model_label": "food_at_home", "unit": "current U.S. dollars on target day"},
        {"field": "daily_food_away_from_home", "source": "daily_food_away_from_home_legacy_strict_v1", "model_label": "food_away_from_home", "unit": "current U.S. dollars on target day"},
    ]).to_csv(OUTPUT_DIR / "01_variable_manifest.csv", index=False)
    pd.DataFrame([
        {"task_id": TASK_ID, "group_contract": "consumer_unit", "groups": int(table["group_id"].nunique()), "rows": len(table), "max_rows_per_group": int(table.groupby("group_id", observed=True).size().max()), "group_id_equals_subject_id": int(table["group_id"].eq(table["subject_id"]).all()), "split_rule": "temporal quarter splits; consumer-unit grouping does not enforce split separation"},
    ]).to_csv(OUTPUT_DIR / "01_grouping_audit.csv", index=False)
    print(f"{TASK_ID}: wrote {len(table):,} selected event-window rows (eligible={sampling_summary['full_eligible_rows']:,}).")


if __name__ == "__main__":
    main()
