#!/usr/bin/env python
"""Render JSONL prompts for the SCE mortgage lock-in moving probability task from its table."""

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
    HOUSING_SYSTEM_PROMPT,
    age_article,
    apply_prompt_render_limit,
    format_dollars,
    checked_label,
)
from scripts.utils.table_schema import read_public_table

from scripts.utils.io import write_jsonl
from scripts.utils.macro_context import render_core_macro_paragraph


TASK_ID = 'house_sce_lockin'
SYSTEM_PROMPT = HOUSING_SYSTEM_PROMPT
TABULAR_PATH = PROJECT_ROOT / "data" / "householdbench" / "tabular" / f"{TASK_ID}.csv"
PROMPT_PATH = resolve_prompt_output_path(PROJECT_ROOT / "data" / "householdbench" / "prompts" / f"{TASK_ID}.jsonl", task_id="house_sce_lockin")

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


LOAN_STATUS_TEXT = {
    "yes, mortgage(s) only": "mortgage debt only",
    "yes, home equity loans/lines of credit only": "home equity loans/lines of credit only",
    "yes, both mortgage(s) and home equity loans/lines of credit": "both mortgage(s) and home equity loans/lines of credit",
    "no": "no loans against home",
}
MORTGAGE_TYPE_TEXT = {
    "adjustable/floating": "adjustable/floating",
    "fixed": "fixed",
    "don't know": "don't know",
}
SCENARIO = (
    "For this question, consider the following mortgage-rate scenario. Suppose that, if you moved and bought "
    "a different home, you could keep the same interest rate as your current mortgage."
)
FINAL_QUESTION = (
    "Under that scenario, what probability do you think there is that you would move to a different primary residence "
    "over the next three years?"
)
RESPONSE_INSTRUCTION = (
    "Return only a valid JSON array containing exactly one two-element probability "
    "distribution. Use this order: probability of not moving under the scenario, probability "
    "of moving under the scenario. Express each probability as a number from 0 to 1. The "
    "distribution must sum to 1. Your output must follow this exact array structure: [[p_1_1,"
    " p_1_2]]. Replace every p placeholder with a number from 0 to 1. Do not include keys or "
    "explanatory text."
)
PROMPT_INTRO = "Here is some background information about yourself and your household."


if not TABULAR_PATH.is_file():
    raise RuntimeError(f"Missing task table: {TABULAR_PATH}")
table = read_public_table(TABULAR_PATH, task_id="house_sce_lockin")
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
    background_sentence = (
        f"{PROMPT_INTRO} You are {age_article(age_value)} {age_value}-year-old, {EDUCATION_TEXT[education_value]}, "
        f"live in {REGION_TEXT[region_value]}, have a {NUMERACY_TEXT[numeracy_value]} numeracy score"
    )
    if "partnered" in observed:
        partnered_label = checked_label(
            observed["partnered"], YES_NO_LABELS, "partner status", TASK_ID, row_number
        )
        if partnered_label == "yes":
            background_sentence += ", and are married or living with a partner"
    prompt_parts.append(background_sentence + ".")

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
        if q1_label == "about the same" and q2_label == "about the same":
            finance_sentences.append(
                "Your household is about as well off as it was 12 months ago, and you expect your household finances to remain about the same 12 months from now."
            )
        else:
            finance_sentences.append(
                f"Your household is {q1_label} than it was 12 months ago, and you expect your household finances to be {q2_label} 12 months from now."
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
    if all(
        column in observed
        for column in ["local_home_value", "purchase_price", "current_home_value"]
    ):
        finance_sentences.append(
            f"A typical local home is worth about {format_dollars(observed['local_home_value'])}, "
            f"while your primary residence was purchased for about {format_dollars(observed['purchase_price'])} "
            f"and is now worth about {format_dollars(observed['current_home_value'])}."
        )
    loan_sentence = None
    if "loan_status" in observed:
        loan_status_label = checked_label(
            observed["loan_status"], set(LOAN_STATUS_TEXT), "home-loan status", TASK_ID, row_number
        )
        if loan_status_label == "no":
            loan_sentence = "You have no loans against your home."
        else:
            loan_sentence = f"Your home loan consists of {LOAN_STATUS_TEXT[loan_status_label]}"
            if "loan_balance" in observed:
                loan_sentence += (
                    f", with a balance of about {format_dollars(observed['loan_balance'])}"
                )
            loan_sentence += (
                f" and an interest rate of {float(observed['current_mortgage_rate']):.1f}%."
            )
    if loan_sentence:
        finance_sentences.append(loan_sentence)
    finance_sentences.append(
        f"The national average 30-year mortgage rate is {float(observed['mortgage30us_rate_lag1']):.1f}%, "
        f"which is {float(observed['mortgage_rate_gap']):.1f} percentage points above your current mortgage rate."
    )
    ordinary_event_probability = round(float(observed["ordinary_move_probability"]), 3)
    finance_sentences.append(f"Given your current situation, you think the probability that you would move to a different primary residence over the next three years is {ordinary_event_probability:.3f}.")
    prompt_parts.append(
        f"Here are your household finances, work situation, expectations, housing, and mortgage situation. {current_month_note(row['time'], currency=True)} "
        + " ".join(finance_sentences)
    )
    prompt_parts.append(render_core_macro_paragraph(observed))
    prompt_parts.append(SCENARIO)
    prompt_parts.append(FINAL_QUESTION)
    prompt_parts.append(RESPONSE_INSTRUCTION)
    user_prompt = "\n\n".join(prompt_parts)
    event_probability = round(float(observed["lockin_move_probability"]), 3)
    assistant_payload = [[round(1 - event_probability, 3), event_probability]]
    naive_baseline = [[round(1 - ordinary_event_probability, 3), ordinary_event_probability]]
    records.append(
        {
            "id": str(observed["id"]),
            "time": str(observed["time"]),
            "release_date": str(observed["release_date"][:10]),
            "system": str(SYSTEM_PROMPT),
            "user": str(user_prompt),
            "assistant": json.dumps(assistant_payload, ensure_ascii=True),
            "naive_baseline": json.dumps(naive_baseline, ensure_ascii=True),
        }
    )

for record in records:
    validate_prompt_record(record)
    validate_cross_cutting_prompt_contract("house_sce_lockin", record)
write_jsonl(records, PROMPT_PATH)
print(f"{TASK_ID}: rendered {len(records):,} prompts to {PROMPT_PATH}")
