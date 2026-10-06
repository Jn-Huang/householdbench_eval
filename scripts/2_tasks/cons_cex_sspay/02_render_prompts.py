#!/usr/bin/env python
"""Render the approved SSPAY user prompts."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.macro_context import PUBLIC_CORE_MACRO_COLUMNS, render_core_macro_paragraph
from scripts.utils.table_schema import read_public_table
from scripts.utils.prompt_rendering import (
    age_article,
    format_dollars,
    validate_prompt_record,
)
from scripts.utils.responses import validate_prompt_response_structure


TASK_ID = "cons_cex_sspay"
TABLE_PATH = PROJECT_ROOT / "data/householdbench/tabular/cons_cex_sspay.csv"
PROMPTS_PATH = PROJECT_ROOT / "data/householdbench/prompts/cons_cex_sspay.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "output/householdbench/tasks/cons_cex_sspay"
REGISTRY_PATH = OUTPUT_DIR / "02_model_facing_predictor_registry.csv"
SYSTEM_PROMPT = (
    "You are the head of an American household making decisions and forming "
    "expectations about income, spending, and saving."
)
FINAL_QUESTION = (
    "Given this information, how much will your household spend today in total, "
    "on food at home, and on food away from home?"
)
RESPONSE_INSTRUCTION = (
    "Return only a valid JSON array with exactly 3 numeric elements, in this order: "
    "total_expenditure, food_at_home, food_away_from_home. Use 0 when there is no spending in"
    " a category. Your output must follow this exact array structure: [v_1, v_2, v_3]. "
    "Replace every v placeholder with a single number. Do not include keys or explanatory "
    "text."
)
ANNUAL_CAVEAT = (
    "These are annual figures and do not tell you the size of any particular monthly payment."
)
PAYMENT_POLICY = (
    "Social Security benefits are normally paid on the third day of each month. "
    "If the third falls on a weekend or federal holiday, payment is made on the preceding business day."
)
TARGET_COLUMNS = ["daily_total_expenditure", "daily_food_at_home", "daily_food_away_from_home"]
BASELINE_COLUMNS = [
    "naive_daily_total_expenditure",
    "naive_daily_food_at_home",
    "naive_daily_food_away_from_home",
]
HISTORY_GAP_COLUMNS = [f"calendar_day_gap_lag{lag}" for lag in range(1, 8)]
HISTORY_VALUE_COLUMNS = [
    f"{target}_lag{lag}" for lag in range(1, 8) for target in TARGET_COLUMNS
]


def registry_entry(group: str, predictor: str, source_columns: str, sections: str) -> dict[str, str]:
    return {
        "predictor_group": group,
        "predictor": predictor,
        "source_columns": source_columns,
        "rendered_sections": sections,
    }


PREDICTOR_REGISTRY = [
    registry_entry("household_composition", "household_size", "household_size", "compact_profile"),
    registry_entry("household_composition", "n_adults", "household_size;children_under_18", "compact_profile"),
    registry_entry("household_composition", "children_under_18", "children_under_18", "compact_profile"),
    registry_entry("demographics_and_location", "age_ref", "age_ref", "compact_profile"),
    registry_entry("demographics_and_location", "sex", "sex", "compact_profile"),
    registry_entry("demographics_and_location", "race", "race", "compact_profile"),
    registry_entry("demographics_and_location", "education", "education", "compact_profile"),
    registry_entry("demographics_and_location", "marital_status", "marital_status", "compact_profile"),
    registry_entry("demographics_and_location", "reference_person_work_status", "reference_person_work_status", "compact_profile"),
    registry_entry("demographics_and_location", "region", "region", "compact_profile"),
    registry_entry("demographics_and_location", "urban_status", "urban_status", "compact_profile"),
    registry_entry("demographics_and_location", "home_tenure", "home_tenure", "compact_profile"),
    *[
        registry_entry("daily_spending_history", f"{target}_lag{lag}", f"{target}_lag{lag}", "compact_history")
        for lag in range(1, 8)
        for target in TARGET_COLUMNS
    ],
    registry_entry("annual_household_resources", "annual_income_before_tax", "annual_income_before_tax", "compact_resources"),
    registry_entry("annual_household_resources", "annual_social_security_railroad_income", "annual_social_security_railroad_income", "compact_resources"),
    registry_entry("annual_household_resources", "social_security_share", "social_security_share", "compact_resources"),
    registry_entry("calendar_context", "current_day_of_week", "current_day_of_week;time", "compact_history_intro"),
    registry_entry("calendar_context", "current_day_of_month", "current_day_of_month;time", "compact_history_intro"),
    registry_entry("calendar_context", "current_month", "current_month;time", "compact_history_intro"),
    *[
        registry_entry("core_macro_context", column, column, "compact_macro")
        for column in PUBLIC_CORE_MACRO_COLUMNS
    ],
]
REGISTRY_PREDICTORS = [entry["predictor"] for entry in PREDICTOR_REGISTRY]
if len(PREDICTOR_REGISTRY) != 59 or len(set(REGISTRY_PREDICTORS)) != 59:
    raise RuntimeError(f"{TASK_ID}: model-facing predictor registry must contain exactly 59 unique fields.")
if "macro_reference_q_index" in REGISTRY_PREDICTORS:
    raise RuntimeError(f"{TASK_ID}: macro_reference_q_index is an internal alignment key, not a predictor.")
GROUP_ORDER = {
    "household_composition": 1,
    "demographics_and_location": 2,
    "daily_spending_history": 3,
    "annual_household_resources": 4,
    "calendar_context": 5,
    "core_macro_context": 6,
}
DISPLAY_LABELS = {
    "household_size": "Household size",
    "n_adults": "Adults in household",
    "children_under_18": "Children under 18",
    "age_ref": "Reference-person age",
    "sex": "Sex",
    "race": "Race",
    "education": "Highest completed education",
    "marital_status": "Marital status",
    "reference_person_work_status": "Employment or retirement status",
    "region": "Region",
    "urban_status": "Area type",
    "home_tenure": "Housing tenure",
    "annual_income_before_tax": "Before-tax household income",
    "annual_social_security_railroad_income": "Annual Social Security and Railroad Retirement income",
    "social_security_share": "Benefit-income share of household income",
    "current_day_of_week": "Current day of week",
    "current_day_of_month": "Current day of month",
    "current_month": "Current month",
    "gdp_growth_qoq_lag1": "Real GDP growth, most recent quarter",
    "gdp_growth_qoq_lag2": "Real GDP growth, two quarters ago",
    "gdp_growth_qoq_lag3": "Real GDP growth, three quarters ago",
    "gdp_growth_qoq_lag4": "Real GDP growth, four quarters ago",
    "gdp_growth_qoq_5y_compound": "Five-year compound quarterly real GDP growth",
    "unemployment_rate_lag1": "Unemployment rate, most recent quarter",
    "unemployment_rate_lag2": "Unemployment rate, two quarters ago",
    "unemployment_rate_lag3": "Unemployment rate, three quarters ago",
    "unemployment_rate_lag4": "Unemployment rate, four quarters ago",
    "unemployment_rate_5y_mean": "Five-year mean unemployment rate",
    "headline_cpi_qoq_lag1": "CPI inflation, most recent quarter",
    "headline_cpi_qoq_lag2": "CPI inflation, two quarters ago",
    "headline_cpi_qoq_lag3": "CPI inflation, three quarters ago",
    "headline_cpi_qoq_lag4": "CPI inflation, four quarters ago",
    "headline_cpi_qoq_5y_compound": "Five-year compound quarterly CPI inflation",
    "fedfunds_rate_lag1": "Federal funds rate, most recent quarter",
    "fedfunds_rate_lag2": "Federal funds rate, two quarters ago",
    "fedfunds_rate_lag3": "Federal funds rate, three quarters ago",
    "fedfunds_rate_lag4": "Federal funds rate, four quarters ago",
    "fedfunds_rate_5y_mean": "Five-year mean federal funds rate",
}
for lag in range(1, 8):
    DISPLAY_LABELS[f"daily_total_expenditure_lag{lag}"] = f"Total expenditure, history lag {lag}"
    DISPLAY_LABELS[f"daily_food_at_home_lag{lag}"] = f"Food at home, history lag {lag}"
    DISPLAY_LABELS[f"daily_food_away_from_home_lag{lag}"] = f"Food away from home, history lag {lag}"
CATEGORICAL_PREDICTORS = {
    "sex", "race", "education", "marital_status", "reference_person_work_status", "region",
    "urban_status", "home_tenure",
    "current_day_of_week", "current_month",
}
for predictor_order, entry in enumerate(PREDICTOR_REGISTRY, start=1):
    predictor = entry["predictor"]
    if predictor not in DISPLAY_LABELS:
        raise RuntimeError(f"{TASK_ID}: model-facing registry lacks display label for {predictor}.")
    entry["group_order"] = GROUP_ORDER[entry["predictor_group"]]
    entry["predictor_order"] = predictor_order
    entry["display_label"] = DISPLAY_LABELS[predictor]
    entry["descriptive_type"] = "categorical" if predictor in CATEGORICAL_PREDICTORS else "numeric"
    entry["predictor_scope"] = "model_facing"

REQUIRED_COLUMNS = [
    "row_id", "prompt_time", "release_date", "age_ref", "sex", "race", "education", "marital_status",
    "household_size", "children_under_18", "region", "urban_status", "home_tenure",
    "reference_person_work_status", "annual_income_before_tax", "annual_social_security_railroad_income",
    "social_security_share",
    "diary_date", "day_of_week",
    *TARGET_COLUMNS, *BASELINE_COLUMNS, *HISTORY_GAP_COLUMNS, *HISTORY_VALUE_COLUMNS,
    *PUBLIC_CORE_MACRO_COLUMNS,
]

EDUCATION_PHRASES = {
    "associate degree or some college": "an associate degree or some college education",
    "college graduate": "a college degree",
    "doctoral or professional degree": "a doctoral or professional degree",
    "elementary school": "an elementary school education",
    "high school graduate": "a high school diploma",
    "master's degree": "a master's degree",
    "more than four years of college": "more than four years of college education",
    "no reported schooling": "no reported schooling",
    "some college": "some college education",
    "some high school": "some high school education",
}
TENURE_SENTENCES = {
    "lives in college housing": "Your household lives in college housing.",
    "occupies housing without cash rent": "Your household occupies its home without cash rent.",
    "owns a home": "Your household owns its home.",
    "owns a home with a mortgage": "Your household owns its home with a mortgage.",
    "owns a home without a mortgage": "Your household owns its home without a mortgage.",
    "rents a home": "Your household rents its home.",
}
WORK_STATUS_PHRASES = {"currently working", "not currently working", "retired"}


def whole_number(value: object, *, field: str) -> int:
    number = float(value)
    if not math.isfinite(number) or number != round(number):
        raise RuntimeError(f"{TASK_ID}: {field} must be a finite integer.")
    return int(number)


def required_text(value: object, *, field: str) -> str:
    if pd.isna(value):
        raise RuntimeError(f"{TASK_ID}: {field} is missing.")
    text = str(value).strip()
    if not text:
        raise RuntimeError(f"{TASK_ID}: {field} is blank.")
    return text


def education_phrase(value: object) -> str:
    education = required_text(value, field="education").lower()
    if education not in EDUCATION_PHRASES:
        raise RuntimeError(f"{TASK_ID}: unrecognised education label {education!r}.")
    return EDUCATION_PHRASES[education]


def work_status_phrase(value: object) -> str:
    status = required_text(value, field="reference_person_work_status").lower()
    if status not in WORK_STATUS_PHRASES:
        raise RuntimeError(f"{TASK_ID}: unrecognised work-status label {status!r}.")
    return status


def household_counts(row: dict[str, object]) -> tuple[int, int, int, str, str, str]:
    household_size = whole_number(row["household_size"], field="household_size")
    children = whole_number(row["children_under_18"], field="children_under_18")
    adults = household_size - children
    if household_size < 1 or children < 0 or adults < 0:
        raise RuntimeError(f"{TASK_ID}: invalid household-size/children relationship.")
    adult_label = "adult" if adults == 1 else "adults"
    children_phrase = "no children" if children == 0 else f"{children} {'child' if children == 1 else 'children'}"
    total_phrase = "" if household_size == 1 else f", for {household_size} people in total"
    return household_size, children, adults, adult_label, children_phrase, total_phrase


def housing_tenure_sentence(value: object) -> str:
    tenure = required_text(value, field="home_tenure").lower()
    if tenure not in TENURE_SENTENCES:
        raise RuntimeError(f"{TASK_ID}: unrecognised home-tenure label {tenure!r}.")
    return TENURE_SENTENCES[tenure]


def ordinal_day(day: int) -> str:
    if 10 <= day % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def calendar_context(row: dict[str, object]) -> tuple[str, int, str]:
    diary_date = pd.Timestamp(row["diary_date"])
    if pd.isna(diary_date):
        raise RuntimeError(f"{TASK_ID}: diary_date is missing.")
    day_of_week = diary_date.day_name()
    if day_of_week != required_text(row["day_of_week"], field="day_of_week"):
        raise RuntimeError(f"{TASK_ID}: day_of_week does not agree with diary_date.")
    return day_of_week, int(diary_date.day), diary_date.month_name()


def history_text_for_prompt(row: dict[str, object]) -> str:
    sentences: list[str] = []
    for lag in range(7, 0, -1):
        gap = whole_number(row[f"calendar_day_gap_lag{lag}"], field=f"calendar_day_gap_lag{lag}")
        if gap != lag:
            raise RuntimeError(f"{TASK_ID}: the approved prompt requires seven consecutive prior days.")
        total = format_dollars(float(row[f"daily_total_expenditure_lag{lag}"]), decimals=2)
        food_home = format_dollars(float(row[f"daily_food_at_home_lag{lag}"]), decimals=2)
        food_away = format_dollars(float(row[f"daily_food_away_from_home_lag{lag}"]), decimals=2)
        elapsed = "1 day ago" if gap == 1 else f"{gap} days ago"
        sentences.append(
            f"- {elapsed}: {total} in total, including {food_home} on food at home and "
            f"{food_away} on food away from home."
        )
    return "\n".join(sentences)


def render_prompt(row: dict[str, object], row_number: int) -> dict[str, object]:
    age = whole_number(row["age_ref"], field="age_ref")
    household_size, children, adults, adult_label, children_text, total_phrase = household_counts(row)
    sex_source = required_text(row["sex"], field="sex").lower()
    if sex_source == "male":
        sex = "man"
    elif sex_source == "female":
        sex = "woman"
    else:
        raise RuntimeError(f"{TASK_ID}: unsupported sex label {sex_source!r}.")
    race = required_text(row["race"], field="race").lower()
    marital_status = required_text(row["marital_status"], field="marital_status").lower()
    urban_status = required_text(row["urban_status"], field="urban_status").lower()
    area_article = "an" if urban_status[:1] in {"a", "e", "i", "o", "u"} else "a"
    region = required_text(row["region"], field="region").rstrip(".")
    annual_income = format_dollars(float(row["annual_income_before_tax"]))
    annual_benefit_income = format_dollars(float(row["annual_social_security_railroad_income"]))
    benefit_share = f"{float(row['social_security_share']) * 100:.1f}"
    history_text = history_text_for_prompt(row)
    day_of_week, day_of_month, month = calendar_context(row)
    current_date = f"{day_of_week}, {ordinal_day(day_of_month)} {month}"
    compact_profile = (
        "Here is some background information about yourself and your household. "
        f"You are {age_article(age)} {age}-year-old {race} {sex}, are {marital_status}, "
        f"have {education_phrase(row['education'])}, and are {work_status_phrase(row['reference_person_work_status'])}. "
        f"You live in {area_article} {urban_status} in the {region}. "
        f"Your household consists of {adults} {adult_label} and {children_text}. "
        f"{housing_tenure_sentence(row['home_tenure'])}"
    )
    compact_resources = (
        f"Your household reported about {annual_income} in income before tax over the past 12 months. "
        f"Of this, about {annual_benefit_income}, or {benefit_share}%, came from Social Security and Railroad Retirement. "
        f"{ANNUAL_CAVEAT}"
    )
    compact_history = (
        "Here is your household's spending over the last seven days, listed from oldest to most recent. "
        f"Note that today is {current_date}, and that $ denotes amounts in U.S. dollars.\n\n{history_text}"
    )
    compact = "\n\n".join([
        compact_profile,
        compact_resources,
        compact_history,
        render_core_macro_paragraph(row),
        PAYMENT_POLICY,
        FINAL_QUESTION,
        RESPONSE_INSTRUCTION,
    ])

    record = {
        "id": str(row["id"]),
        "time": str(row["prompt_time"]),
        "release_date": str(row["release_date"]),
        "system": str(SYSTEM_PROMPT),
        "user": str(compact),
        "assistant": json.dumps(
            [round(float(row[column]), 2) for column in TARGET_COLUMNS], allow_nan=False
        ),
        "naive_baseline": json.dumps(
            [round(float(row[column]), 2) for column in BASELINE_COLUMNS], allow_nan=False
        ),
    }
    validate_prompt_record(record)
    validate_prompt_response_structure(TASK_ID, compact)
    if not compact.endswith(RESPONSE_INSTRUCTION):
        raise RuntimeError(f"{TASK_ID}: response instruction is missing or misplaced for row {row_number}.")
    return record


def representative_examples(table: pd.DataFrame) -> list[tuple[str, int]]:
    candidates = table.index[table["id"].eq("1986:0000281:19860127") & table["time"].eq("1986-01-27")]
    if len(candidates) != 1:
        raise RuntimeError(f"{TASK_ID}: approved calendar-policy example is unavailable.")
    return [("approved_calendar_policy_example", int(candidates[0]))]


def write_registry(render_scope: str) -> None:
    registry = pd.DataFrame(PREDICTOR_REGISTRY)
    registry.insert(0, "task_id", TASK_ID)
    registry.insert(1, "registry_version", "cons_cex_sspay_model_facing_v2")
    registry.insert(2, "render_scope", render_scope)
    registry.insert(3, "predictor_count", len(PREDICTOR_REGISTRY))
    registry.to_csv(REGISTRY_PATH, index=False)


def main() -> None:
    if not TABLE_PATH.is_file():
        raise SystemExit(f"{TASK_ID}: strict public task table is missing.")
    table = read_public_table(TABLE_PATH, task_id="cons_cex_sspay")
    table["row_id"] = table["id"] + ":" + table["time"].str.replace("-", "", regex=False)
    table["prompt_time"] = table["time"]
    table["diary_date"] = table["time"]
    table["day_of_week"] = table["current_day_of_week"]
    for lag in range(1, 8):
        table[f"calendar_day_gap_lag{lag}"] = lag
    for target, baseline in zip(TARGET_COLUMNS, BASELINE_COLUMNS, strict=True):
        table[baseline] = table[f"{target}_lag1"]
    missing = sorted(set(REQUIRED_COLUMNS) - set(table.columns))
    if missing:
        raise RuntimeError(f"{TASK_ID}: table misses required prompt columns: {missing}")
    if table.empty or table[["id", "time"]].isna().any().any() or table.duplicated(["id", "time"]).any():
        raise RuntimeError(f"{TASK_ID}: table is empty or has invalid observation keys.")
    finite_amounts = table[TARGET_COLUMNS + BASELINE_COLUMNS + HISTORY_VALUE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if (
        finite_amounts.isna().any().any()
        or not bool(finite_amounts.apply(lambda column: column.map(math.isfinite)).all().all())
        or finite_amounts.lt(0).any().any()
    ):
        raise RuntimeError(f"{TASK_ID}: targets, baselines, or history amounts are not finite nonnegative values.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    render_rows = [("authoritative_corpus", row) for row in table.to_dict(orient="records")]
    render_scope = "authoritative_corpus"
    write_registry(render_scope)

    prompt_records: list[dict[str, object]] = []
    for row_number, (selection_reason, row) in enumerate(render_rows, start=1):
        record = render_prompt(row, row_number)
        prompt_records.append(record)

    PROMPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROMPTS_PATH.open("w", encoding="utf-8") as handle:
        for record in prompt_records:
            handle.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
    default_examples = representative_examples(table)
    example_by_key = {(str(record["id"]), str(record["time"])): record for record in prompt_records}
    with (OUTPUT_DIR / "02_rendered_prompt_examples.jsonl").open("w", encoding="utf-8") as handle:
        for selection_reason, index in default_examples:
            row = table.loc[index]
            example = {
                "selection_reason": selection_reason,
                "row_id": row["row_id"],
                "record": example_by_key[(str(row["id"]), str(row["time"]))],
            }
            handle.write(json.dumps(example, ensure_ascii=True, allow_nan=False) + "\n")
    print(f"{TASK_ID}: wrote {len(prompt_records):,} authoritative prompts and the 59-predictor registry.")


if __name__ == "__main__":
    main()
