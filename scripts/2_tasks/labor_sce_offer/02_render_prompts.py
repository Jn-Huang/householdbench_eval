#!/usr/bin/env python
"""Render JSONL prompts for the SCE first job-offer decision task from its table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.prompt_rendering import (
    validate_prompt_record,
    validate_cross_cutting_prompt_contract,
    current_month_note,
    resolve_prompt_output_path,
    age_article,
    apply_prompt_render_limit,
    checked_label,
)
from scripts.utils.table_schema import read_public_table

from scripts.utils.io import write_jsonl
from scripts.utils.macro_context import render_core_macro_paragraph


TASK_ID = 'labor_sce_offer'
SYSTEM_PROMPT = (
    "You are an American person answering questions about your work, job search, "
    "and labor-force situation."
)
TABULAR_PATH = PROJECT_ROOT / "data" / "householdbench" / "tabular" / f"{TASK_ID}.csv"
PROMPT_PATH = resolve_prompt_output_path(PROJECT_ROOT / "data" / "householdbench" / "prompts" / f"{TASK_ID}.jsonl", task_id="labor_sce_offer")

EDUCATION_TEXT = {
    "High School": "have a high-school education",
    "Some College": "have some college education",
    "College": "have a college education or more",
}
REGION_TEXT = {
    "Midwest": "the Midwest",
    "Northeast": "the Northeast",
    "South": "the South",
    "West": "the West",
}
NUMERACY_TEXT = {"High": "high", "Low": "low"}
FINANCE_CHANGE_LABELS = {"much worse off", "somewhat worse off", "about the same", "somewhat better off", "much better off"}
YES_NO_LABELS = {"yes", "no"}
WORK_STATUS_LABELS = [
    ("Q10_1", "working full-time"),
    ("Q10_2", "working part-time"),
    ("Q10_3", "not working but would like to work"),
    ("Q10_4", "temporarily laid off or on leave from work"),
    ("Q10_5", "another current work attachment"),
]


FINAL_QUESTION = (
    'Thinking about the first job offer you received in the last four months, did you accept or reject that offer? '
    'Use "accepted" if you accepted the offer, whether or not you still work at that job. '
    'Use "rejected" if you did not accept it.'
)
RESPONSE_INSTRUCTION = (
    'Return exactly one valid JSON array containing one string, using one of: "accepted", '
    '"rejected". Your output must follow this exact array structure: ["s_1"]. Replace every s'
    " placeholder with one of the permitted strings. Do not include explanatory text."
)
PROMPT_INTRO = "Here is some background information about yourself and your household."


if not TABULAR_PATH.is_file():
    raise RuntimeError(f"Missing task table: {TABULAR_PATH}")
table = read_public_table(TABULAR_PATH, task_id="labor_sce_offer")
rows = table.to_dict("records")

rows = apply_prompt_render_limit(rows, task_id=TASK_ID)

records = []
for row_number, row in enumerate(rows, start=1):
    observed = {column: value for column, value in row.items() if not pd.isna(value)}
    prompt_parts = []
    age_value = int(round(float(observed["age_years"])))
    education_value = observed["_EDU_CAT"]
    if education_value not in EDUCATION_TEXT:
        raise RuntimeError(f"{TASK_ID} row {row_number}: unmapped education category {education_value!r}.")
    region_value = observed["_REGION_CAT"]
    if region_value not in REGION_TEXT:
        raise RuntimeError(f"{TASK_ID} row {row_number}: unmapped region category {region_value!r}.")
    numeracy_value = observed["_NUM_CAT"]
    if numeracy_value not in NUMERACY_TEXT:
        raise RuntimeError(f"{TASK_ID} row {row_number}: unmapped numeracy category {numeracy_value!r}.")
    prompt_parts.append(
        f"{PROMPT_INTRO} You are {age_article(age_value)} {age_value}-year-old, {EDUCATION_TEXT[education_value]}, "
        f"live in {REGION_TEXT[region_value]}, and have a {NUMERACY_TEXT[numeracy_value]} numeracy score."
    )

    income_value = observed["_HH_INC_DETAILED"]
    if income_value == "Less than $10,000":
        income_clause = "less than $10,000"
    elif income_value == "$200,000 or more":
        income_clause = "$200,000 or more"
    elif " to " in income_value:
        income_lower, income_upper = income_value.split(" to ", maxsplit=1)
        income_clause = f"between {income_lower} and {income_upper}"
    else:
        raise RuntimeError(f"Unexpected detailed household-income label {income_value!r}.")
    finance_sentences = [f"Your household's total pre-tax income over the past 12 months was {income_clause}."]
    if "Q1" in observed:
        q1_label = checked_label(observed["Q1"], FINANCE_CHANGE_LABELS, "Q1", TASK_ID, row_number)
    else:
        q1_label = None
    if "Q2" in observed:
        q2_label = checked_label(observed["Q2"], FINANCE_CHANGE_LABELS, "Q2", TASK_ID, row_number)
    else:
        q2_label = None
    if q1_label and q2_label:
        finance_sentences.append(
            f"Your household is {q1_label} than it was 12 months ago, and you expect it to be "
            f"{q2_label} 12 months from now."
        )
    if "Q9_mean" in observed:
        inflation_sentence = (
            f"You expect inflation to be {float(observed['Q9_mean']):.1f}% over the next 12 months"
        )
        if "Q9c_mean" in observed:
            inflation_sentence += (
                f" and {float(observed['Q9c_mean']):.1f}% over the next three years."
            )
        else:
            inflation_sentence += "."
        finance_sentences.append(inflation_sentence)
    work_labels = []
    for column, label in WORK_STATUS_LABELS:
        if column in observed:
            work_status = checked_label(
                observed[column], YES_NO_LABELS, column, TASK_ID, row_number
            )
            if work_status == "yes":
                work_labels.append(label)
    if work_labels:
        if len(work_labels) == 1:
            finance_sentences.append(f"You work {work_labels[0].replace('working ', '')}.")
        else:
            finance_sentences.append(f"Your current work situation includes {', '.join(work_labels)}.")
    prompt_parts.append(
        f"Here are your household finances, work situation, and expectations. {current_month_note(row['time'], currency=True)} "
        + " ".join(finance_sentences)
    )
    prompt_parts.append(render_core_macro_paragraph(observed))
    offer_sentences = ["For this question, consider the following job offer."]
    if "nl1" in observed:
        offer_count = int(round(float(observed["nl1"])))
        if offer_count >= 1:
            offer_sentences.append("You received at least one job offer in the last four months.")
    if "nl2a_1" in observed:
        salary_text = f"had an annual salary of ${int(round(float(observed['nl2a_1']))):,}"
    else:
        salary_text = ""
    if "nl2b_1" in observed:
        offer_type_label = checked_label(
            observed["nl2b_1"], {"full-time", "part-time"}, "nl2b_1", TASK_ID, row_number
        )
        if salary_text:
            offer_sentences.append(f"The first offer was {offer_type_label} and {salary_text}.")
        else:
            offer_sentences.append(f"The first offer was {offer_type_label}.")
    if "reservation_wage_annual_full_time" in observed:
        reservation_wage = int(round(float(observed["reservation_wage_annual_full_time"])))
        offer_sentences.append(
            f"Your reservation wage for a full-time job was approximately ${reservation_wage:,} per year."
        )
    prompt_parts.append(" ".join(offer_sentences))
    prompt_parts.append(FINAL_QUESTION)
    prompt_parts.append(RESPONSE_INSTRUCTION)
    if observed["first_offer_decision"] not in {"accepted", "rejected"}:
        raise RuntimeError(
            f"{TASK_ID} row {row_number}: unmapped first-offer decision {observed['first_offer_decision']!r}."
        )
    user_prompt = "\n\n".join(prompt_parts)
    records.append(
        {
            "id": str(observed["id"]),
            "time": str(observed["time"]),
            "release_date": str(observed["release_date"][:10]),
            "system": str(SYSTEM_PROMPT),
            "user": str(user_prompt),
            "assistant": json.dumps([observed["first_offer_decision"]], ensure_ascii=True),
            "naive_baseline": json.dumps(["rejected"], ensure_ascii=True),
        }
    )

for record in records:
    validate_prompt_record(record)
    validate_cross_cutting_prompt_contract("labor_sce_offer", record)
write_jsonl(records, PROMPT_PATH)
print(f"{TASK_ID}: rendered {len(records):,} prompts to {PROMPT_PATH}")
