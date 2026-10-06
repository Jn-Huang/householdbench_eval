#!/usr/bin/env python
"""Render HouseholdBench CEX consumption-category prompts from the task table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.prompt_rendering import (
    cex_education_phrase as education_phrase,
    current_month_note,
    resolve_prompt_output_path,
    required_text,
    age_article,
    apply_prompt_render_limit,
    format_dollars,
)
from scripts.utils.table_schema import read_public_table, public_table_columns

from scripts.utils.macro_context import (
    CATEGORY_CPI_COLUMNS,
    CORE_MACRO_COLUMNS,
    render_category_cpi_paragraph,
    render_core_macro_paragraph,
)

TASK_ID = "cons_cex_categories"
INPUT_PATH = PROJECT_ROOT / "data/householdbench/tabular/cons_cex_categories.csv"
OUTPUT_PATH = resolve_prompt_output_path(PROJECT_ROOT / "data/householdbench/prompts/cons_cex_categories.jsonl", task_id="cons_cex_categories")
PARAGRAPH_BREAK = "\n\n"
SYSTEM_PROMPT = (
    "You are the head of an American household making decisions and forming "
    "expectations about income, spending, and saving."
)

DETAIL_COLS = [
    ("cons_food", "food"),
    ("cons_alcohol", "alcoholic beverages"),
    ("cons_housing", "housing"),
    ("cons_apparel", "apparel and services"),
    ("cons_transport", "transportation"),
    ("cons_health", "health care"),
    ("cons_entertainment", "entertainment"),
    ("cons_personal_care", "personal care"),
    ("cons_read", "reading"),
    ("cons_education", "education"),
    ("cons_tobacco", "tobacco"),
    ("cons_misc", "miscellaneous"),
]
DETAIL_ORDER_TEXT = ", ".join(label for _, label in DETAIL_COLS[:-1]) + f", and {DETAIL_COLS[-1][1]}"
HOUSEHOLD_LAG_LABELS = {
    1: "between three and six months ago",
    2: "between six and nine months ago",
    3: "between nine and twelve months ago",
}
FINAL_QUESTION = (
    "Given this information, how much did your household spend in each of the "
    "categories during the past three months, in current U.S. dollars?"
)
RESPONSE_INSTRUCTION = (
    "Return only a valid JSON array with 12 numeric elements, in this order: food, alcoholic "
    "beverages, housing, apparel and services, transportation, health care, entertainment, "
    "personal care, reading, education, tobacco, and miscellaneous. Your output must follow "
    "this exact array structure: [v_1, v_2, ..., v_12]. Replace every v placeholder with a "
    "single number. Do not include keys or explanatory text."
)
REQUIRED_COLUMNS = [column for column in [
    "id",
    "time",
    "release_date",
    "age_ref",
    "sex_ref",
    "race_ref_harmonized",
    "educ_ref_harmonized",
    "marital1",
    "n_kids",
    "fam_size",
    "n_adults",
    "region",
    "bls_urbn",
    "income_before_tax",
    *[f"{column}_lag{lag}" for lag in [1, 2, 3] for column, _ in DETAIL_COLS],
    *CORE_MACRO_COLUMNS,
    *CATEGORY_CPI_COLUMNS,
    *[column for column, _ in DETAIL_COLS],
] if column in set(public_table_columns("cons_cex_categories"))]
CATEGORICAL_COLUMNS = [
    "sex_ref",
    "race_ref_harmonized",
    "educ_ref_harmonized",
    "marital1",
    "region",
    "bls_urbn",
]
BASE_VALUE_COLUMNS = [
    column
    for column in REQUIRED_COLUMNS
    if "_lag" not in column and column not in CATEGORY_CPI_COLUMNS
]


def joined(items: list[str], *, sep: str = ", ") -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return sep.join(items[:-1]) + f"{sep}and {items[-1]}"


def spending_sentence(row: dict[str, object], lag: int) -> str:
    parts = [
        f"{format_dollars(row[f'{column}_lag{lag}'])} on {label}"
        for column, label in DETAIL_COLS
    ]
    return (
        f"{HOUSEHOLD_LAG_LABELS[lag].capitalize()}, your household spent "
        f"{joined(parts, sep='; ')}."
    )


def area_phrase(label: str) -> str:
    article = "an" if label[:1] in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {label}"


table = read_public_table(INPUT_PATH, task_id="cons_cex_categories")

missing_columns = [column for column in REQUIRED_COLUMNS if column not in table.columns]
if missing_columns:
    raise SystemExit(f"{INPUT_PATH} is missing required columns: {missing_columns}")
if table.empty:
    raise SystemExit(f"{INPUT_PATH} contains no rows.")
if table[BASE_VALUE_COLUMNS].isna().any().any():
    bad_columns = table[BASE_VALUE_COLUMNS].columns[table[BASE_VALUE_COLUMNS].isna().any()].tolist()
    raise SystemExit(f"{INPUT_PATH} contains missing values in required columns: {bad_columns}")
for column in CATEGORICAL_COLUMNS:
    numeric_like = table[column].astype("string").str.strip().str.fullmatch(r"\d+(?:\.0+)?").fillna(False)
    if numeric_like.any():
        raise SystemExit(f"{INPUT_PATH} contains numeric categorical values in {column}.")

detail_order = DETAIL_ORDER_TEXT

table = apply_prompt_render_limit(table, task_id=TASK_ID)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
    for row_number, row in enumerate(table.to_dict("records"), start=1):
        available_lags = []
        missing_lag_seen = False
        for lag in [1, 2, 3]:
            lag_columns = [f"{column}_lag{lag}" for column, _ in DETAIL_COLS]
            values_present = [not pd.isna(row[column]) for column in lag_columns]
            if any(values_present) and not all(values_present):
                raise RuntimeError(
                    f"Row {row_number} has a partially populated lag-{lag} history."
                )
            if all(values_present):
                if missing_lag_seen:
                    raise RuntimeError(
                        f"Row {row_number} has lag {lag} after a missing earlier lag."
                    )
                available_lags.append(lag)
            else:
                missing_lag_seen = True
        if not available_lags or available_lags[0] != 1:
            raise RuntimeError(
                f"Row {row_number} has no complete lag-1 household history."
            )

        age = int(np.round(float(row["age_ref"])))
        n_kids = int(np.round(float(row["n_kids"])))
        fam_size = int(np.round(float(row["fam_size"])))
        n_adults = max(fam_size - n_kids, 0)

        sex = required_text(row, "sex_ref", row_number).lower()
        race = required_text(row, "race_ref_harmonized", row_number).lower()
        education = required_text(row, "educ_ref_harmonized", row_number).lower()
        marital = required_text(row, "marital1", row_number).lower()
        region = required_text(row, "region", row_number)
        urban = required_text(row, "bls_urbn", row_number).lower()
        adult_word = "adult" if n_adults == 1 else "adults"
        child_phrase = "no children" if n_kids == 0 else f"{n_kids} {'child' if n_kids == 1 else 'children'}"
        total_phrase = "" if fam_size == 1 else f", for {fam_size} people in total"
        demographic_sentence = (
            f"Here is some background information about yourself and your household. "
            f"You are {age_article(age)} {age}-year-old {race} {sex}, are currently {marital}, "
            f"and have {education_phrase(education)}. "
            f"You currently reside in {area_phrase(urban)} area in the {region.rstrip('.')}. "
            f"Your household has {n_adults} {adult_word} and {child_phrase}{total_phrase}."
        )

        lags_desc = list(reversed(available_lags))
        detail_history = [spending_sentence(row, lag) for lag in lags_desc]
        assistant_target = [
            int(np.round(float(row[column])))
            for column, _ in DETAIL_COLS
        ]
        naive_baseline = [
            int(np.round(float(row[f"{column}_lag1"])))
            for column, _ in DETAIL_COLS
        ]
        prompt_parts = [
            demographic_sentence,
            (
                f"Here are your household's recent spending and income. {current_month_note(row['time'], currency=True)} "
                + " ".join(detail_history)
                + " Your before-tax household income over the past 12 months was "
                + f"{format_dollars(row['income_before_tax'])}."
            ),
            render_core_macro_paragraph(row),
        ]
        category_paragraph = render_category_cpi_paragraph(row)
        if category_paragraph:
            prompt_parts.append(category_paragraph)
        prompt_parts.extend([FINAL_QUESTION, RESPONSE_INSTRUCTION])
        user_prompt = PARAGRAPH_BREAK.join(prompt_parts)
        record = {
            "id": str(row["id"]),
            "time": str(row["time"]),
            "release_date": str(row["release_date"])[:10],
            "system": SYSTEM_PROMPT,
            "user": user_prompt,
            "assistant": json.dumps(assistant_target, ensure_ascii=True),
            "naive_baseline": json.dumps(naive_baseline, ensure_ascii=True),
        }

        handle.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")

print(f"{TASK_ID}: wrote {len(table):,} prompt records to {OUTPUT_PATH}")
