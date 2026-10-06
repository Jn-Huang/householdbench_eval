#!/usr/bin/env python
"""Render Michigan personal-finance prompts from the task tabular CSV."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.prompt_rendering import (
    current_month_note,
    resolve_prompt_output_path,
    age_article,
    apply_prompt_render_limit,
    format_dollars,
    phrase_from_label,
    validate_prompt_record,
)
from scripts.utils.table_schema import read_public_table, public_table_columns

from scripts.utils.macro_context import CORE_MACRO_COLUMNS, render_core_macro_paragraph


TASK_NAME = "income_mich_finance"
SYSTEM_PROMPT = (
    "You are an American adult forming expectations about economic conditions and "
    "your own finances."
)
TABLE_PATH = PROJECT_ROOT / "data" / "householdbench" / "tabular" / f"{TASK_NAME}.csv"
PROMPTS_PATH = resolve_prompt_output_path(PROJECT_ROOT / "data" / "householdbench" / "prompts" / f"{TASK_NAME}.jsonl", task_id="income_mich_finance")

SEX_PHRASES = {"Male": "man", "Female": "woman"}
REGION_PHRASES = {
    "West": "the West",
    "North Central": "the North Central United States",
    "Northeast": "the Northeast",
    "South": "the South",
}
MARITAL_PHRASES = {
    "Married/partner": "are married or partnered",
    "Married/partner (includes spouse absent)": "are married or partnered",
    "Separated": "are separated",
    "Divorced": "are divorced",
    "Divorced (includes separated)": "are divorced or separated",
    "Widowed": "are widowed",
    "Never married": "have never been married",
}
EDUCATION_PHRASES = {
    "Grade 0-8 no hs diploma": "0 to 8 years of schooling and no high school diploma",
    "Grade 9-12 no hs diploma": "9 to 12 years of schooling and no high school diploma",
    "Grade 0-12 w/ hs diploma": "a high school diploma",
    "Grade 13-17 no col degree": "some college without a college degree",
    "Grade 13-16 w/ col degree": "a college degree after 13 to 16 years of schooling",
    "Grade 17 W/ col degree": "a graduate or professional degree",
}
ASSESSMENT_PHRASES = {
    "Better now": "better",
    "Same": "the same",
    "About the same": "about the same",
    "Worse now": "worse",
}
FINAL_QUESTION = (
    "By what percentage do you expect your income to change over the next 12 months? "
    "Use a negative number if you expect your income to be lower, a positive number if "
    "you expect it to be higher, and 0 if you expect it to be about the same."
)
RESPONSE_INSTRUCTION = (
    "Return only a valid JSON array containing exactly one number. Your output must follow "
    "this exact array structure: [v_1]. Replace every v placeholder with a single number. Do "
    "not include explanatory text."
)
REQUIRED_COLUMNS = [column for column in [
    "id",
    "time",
    "release_date",
    "AGE",
    "SEX",
    "REGION",
    "MARRY",
    "NUMADT",
    "NUMKID",
    "EDUC",
    "INCOME",
    "PAGO",
    "BAGO",
    "expected_percent_change",
    *CORE_MACRO_COLUMNS,
] if column in set(public_table_columns("income_mich_finance"))]


def indefinite_article(word: str) -> str:
    return "an" if word[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def child_count_phrase(children: int) -> str:
    if children == 0:
        return "no children"
    if children == 1:
        return "1 child"
    return f"{children} children"


def comma_join_with_and(values: list[str]) -> str:
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return " and ".join(values)
    return ", ".join(values[:-1]) + f", and {values[-1]}"


if not TABLE_PATH.exists():
    raise SystemExit(f"Task table not found: {TABLE_PATH}")

table = read_public_table(TABLE_PATH, task_id="income_mich_finance")

missing_columns = [column for column in REQUIRED_COLUMNS if column not in table.columns]
if missing_columns:
    raise SystemExit(f"{TASK_NAME}: table is missing required columns: {missing_columns}")

table = apply_prompt_render_limit(table, task_id=TASK_NAME)
PROMPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with PROMPTS_PATH.open("w", encoding="utf-8") as handle:
    for row_number, row in enumerate(table.to_dict(orient="records"), start=1):
        row_id = str(row["id"])
        prompt_time = pd.Timestamp(row["time"]).date().isoformat()
        target = int(round(float(row["expected_percent_change"])))

        age = int(round(float(row["AGE"])))
        sex = phrase_from_label(row, "SEX", SEX_PHRASES, row_id)
        region = phrase_from_label(row, "REGION", REGION_PHRASES, row_id)
        marital_status = phrase_from_label(row, "MARRY", MARITAL_PHRASES, row_id)
        education = phrase_from_label(row, "EDUC", EDUCATION_PHRASES, row_id)

        n_adults = int(round(float(row["NUMADT"])))
        n_kids = int(round(float(row["NUMKID"])))
        article = age_article(age)
        demographic_paragraph = (
            f"Here is some background information about yourself and your household. "
            f"You are {article} {age}-year-old {sex} living in {region}. "
            f"You {marital_status}, and your highest completed education is "
            f"{education}. Your household has {n_adults} "
            f"{'adult' if n_adults == 1 else 'adults'} and {child_count_phrase(n_kids)}."
        )

        pago_phrase = phrase_from_label(row, "PAGO", ASSESSMENT_PHRASES, row_id)
        bago_phrase = phrase_from_label(row, "BAGO", ASSESSMENT_PHRASES, row_id)

        beliefs_paragraph = f"Here are your current views on your finances and the economy. {current_month_note(row['time'], currency=True)} Your current household income is {format_dollars(row['INCOME'])} per year. You report that your personal finances are {pago_phrase} than they were a year ago, while national business conditions are {bago_phrase} than they were a year ago."

        macro_history_paragraph = render_core_macro_paragraph(row)

        user_prompt = "\n\n".join(
            [
                demographic_paragraph,
                beliefs_paragraph,
                macro_history_paragraph,
                        FINAL_QUESTION,
                RESPONSE_INSTRUCTION,
            ]
        )

        record = {
            "id": row_id,
            "time": prompt_time,
            "release_date": str(row["release_date"]),
            "system": SYSTEM_PROMPT,
            "user": user_prompt,
            "assistant": json.dumps([target], ensure_ascii=True),
            "naive_baseline": json.dumps([0], ensure_ascii=True),
        }

        validate_prompt_record(record)
        handle.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")

print(f"Wrote {PROMPTS_PATH} from {TABLE_PATH}")
