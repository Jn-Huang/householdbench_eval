#!/usr/bin/env python
"""Render Census mobility prompts from the task tabular CSV."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.prompt_rendering import (
    current_month_note,
    resolve_prompt_output_path,
    HOUSING_SYSTEM_PROMPT,
    age_article,
    apply_prompt_render_limit,
    required_text,
    validate_prompt_record,
)
from scripts.utils.table_schema import read_public_table, public_table_columns

from scripts.utils.macro_context import CORE_MACRO_COLUMNS


def lower_initial(text: str) -> str:
    if not text:
        return text
    return text[:1].lower() + text[1:]


TASK_NAME = "house_census_move"
SYSTEM_PROMPT = HOUSING_SYSTEM_PROMPT
TABLE_PATH = PROJECT_ROOT / "data" / "householdbench" / "tabular" / f"{TASK_NAME}.csv"
PROMPTS_PATH = resolve_prompt_output_path(PROJECT_ROOT / "data" / "householdbench" / "prompts" / f"{TASK_NAME}.jsonl", task_id="house_census_move")
TARGETS = {"stay", "move_within_state", "move_to_different_state"}
REQUIRED_COLUMNS = [column for column in [
    "id",
    "time",
    "release_date",
    "target",
    "age_origin",
    "race_ethnicity",
    "sex",
    "education",
    "origin_state",
    "state_unemployment_rate_origin",
    "us_unemployment_rate_origin",
    "state_pcpi_growth_origin",
    "us_pcpi_growth_origin",
    "state_house_price_growth_origin",
    "us_house_price_growth_origin",
    "state_macro_reference_year",
    *CORE_MACRO_COLUMNS,
] if column in set(public_table_columns("house_census_move"))]
FINAL_QUESTION = (
    "Compared with where you lived five years ago, do you now live at the same residence, "
    "at a different residence in the same state, or in a different state?"
)
RESPONSE_INSTRUCTION = (
    'Return only a valid JSON array containing exactly one string, using one of: "stay", '
    '"move_within_state", "move_to_different_state". Your output must follow this exact array'
    ' structure: ["s_1"]. Replace every s placeholder with one of the permitted strings. Do '
    "not include explanatory text."
)
SEX_PHRASES = {
    "female": "woman",
    "male": "man",
}


def census_macro_paragraph(row: dict[str, object]) -> str:
    def path(prefix: str) -> str:
        values = [float(row[f"{prefix}_lag{lag}"]) for lag in [4, 3, 2, 1]]
        return f"{values[0]:.2f}%, {values[1]:.2f}%, {values[2]:.2f}%, and {values[3]:.2f}%"

    return (
        "You are also aware of national economic conditions before that five-year period began. "
        "Over the four calendar quarters immediately before that period, listed from oldest to most recent, "
        f"U.S. real GDP changed by {path('gdp_growth_qoq')} quarter over quarter; the unemployment rate was "
        f"{path('unemployment_rate')}; overall consumer prices changed by {path('headline_cpi_qoq')} quarter "
        f"over quarter; and the effective federal funds rate was {path('fedfunds_rate')}. Over the five years "
        "ending in the last of those four quarters, compound average quarterly real GDP growth was "
        f"{float(row['gdp_growth_qoq_5y_compound']):.2f}%, the average unemployment rate was "
        f"{float(row['unemployment_rate_5y_mean']):.2f}%, compound average quarterly CPI inflation was "
        f"{float(row['headline_cpi_qoq_5y_compound']):.2f}%, and the average effective federal funds rate was "
        f"{float(row['fedfunds_rate_5y_mean']):.2f}%."
    )


if not TABLE_PATH.exists():
    raise SystemExit(f"Task table not found: {TABLE_PATH}")

table = read_public_table(TABLE_PATH, task_id="house_census_move")

missing_columns = [column for column in REQUIRED_COLUMNS if column not in table.columns]
if missing_columns:
    raise SystemExit(f"{TASK_NAME}: table is missing required columns: {missing_columns}")

table = apply_prompt_render_limit(table, task_id=TASK_NAME)
PROMPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with PROMPTS_PATH.open("w", encoding="utf-8") as handle:
    for row_number, row in enumerate(table.to_dict(orient="records"), start=1):
        target = str(row["target"])
        if target not in TARGETS:
            raise RuntimeError(f"Invalid Census mobility target: {target}")

        required_text_values = {
            column: required_text(row, column)
            for column in ["race_ethnicity", "sex", "education", "origin_state"]
        }
        sex = SEX_PHRASES[required_text_values["sex"].casefold()]
        race_ethnicity = lower_initial(required_text_values["race_ethnicity"])
        race_ethnicity_compact = race_ethnicity.replace("not hispanic", "not Hispanic")
        education = lower_initial(required_text_values["education"])
        origin_state_compact = required_text_values["origin_state"].title()

        origin_state_sentence = (
            f"Your state of residence five years ago was {origin_state_compact}."
        )

        user_prompt = "\n\n".join(
            part
            for part in [
                (
                    f"Here is some background information about yourself and where you lived five years ago. {current_month_note(row['time'], currency=False, fixed_month=4)} Five years ago, you were {age_article(int(round(float(row['age_origin']))))} {int(round(float(row['age_origin'])))}-year-old {race_ethnicity_compact} {sex} with {education}. {origin_state_sentence}"
                ),
                census_macro_paragraph(row),
                (
                    "You are also aware of annual state and national conditions in the completed calendar year "
                    "immediately before that five-year period began. "
                    "Unemployment in your state was "
                    f"{float(row['state_unemployment_rate_origin']):.2f}%, compared with "
                    f"{float(row['us_unemployment_rate_origin']):.2f}% nationally; per-capita "
                    f"personal income was growing at {float(row['state_pcpi_growth_origin']):.2f}% "
                    f"in your state and {float(row['us_pcpi_growth_origin']):.2f}% nationally; "
                    "and house prices were growing at "
                    f"{float(row['state_house_price_growth_origin']):.2f}% in your state and "
                    f"{float(row['us_house_price_growth_origin']):.2f}% nationally."
                ),
                (FINAL_QUESTION),
                RESPONSE_INSTRUCTION,
            ]
            if part
        )

        record = {
            "id": str(row["id"]),
            "time": str(row["time"]),
            "release_date": str(row["release_date"]),
            "system": SYSTEM_PROMPT,
            "user": user_prompt,
            "assistant": json.dumps([target], ensure_ascii=True),
            "naive_baseline": json.dumps(["stay"], ensure_ascii=True),
        }

        validate_prompt_record(record)
        handle.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")

print(f"Wrote {PROMPTS_PATH} from {TABLE_PATH}")
