#!/usr/bin/env python
"""Render JSONL prompts for the SCE income-shock spending response task from its table."""

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


TASK_ID = 'cons_sce_shock'
SYSTEM_PROMPT = (
    "You are the head of an American household making decisions and forming expectations about income, spending, and saving."
)
TABULAR_PATH = PROJECT_ROOT / "data" / "householdbench" / "tabular" / f"{TASK_ID}.csv"
PROMPT_PATH = resolve_prompt_output_path(PROJECT_ROOT / "data" / "householdbench" / "prompts" / f"{TASK_ID}.jsonl", task_id="cons_sce_shock")

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


SCENARIO_COMPACT = (
    "For this question, suppose your household income changes unexpectedly next year. In one case, "
    "your household income rises by 10 percent. In the other case, your household income falls by 10 percent."
)
FINAL_QUESTION = (
    "How much of the income gain would you spend or donate, and how much of the income loss would you "
    "cover by reducing spending? Give each answer as a number from 0 to 100, not as a fraction."
)
RESPONSE_INSTRUCTION = (
    "Return only a valid JSON array with exactly two numeric elements, in this order: percent"
    " of the income gain spent or donated; percent of the income loss covered by reducing "
    "spending. Your output must follow this exact array structure: [v_1, v_2]. Replace every "
    "v placeholder with a single number. Do not include keys or explanatory text."
)
PROMPT_INTRO = "Here is some background information about yourself and your household."


if not TABULAR_PATH.is_file():
    raise RuntimeError(f"Missing task table: {TABULAR_PATH}")
table = read_public_table(TABULAR_PATH, task_id="cons_sce_shock")
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
            f"Your household is {q1_label} than it was 12 months ago, and you expect your household "
            f"finances to be {q2_label} 12 months from now."
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
    if "qsp2__clean" in observed:
        if "qsp1" not in observed:
            raise RuntimeError(f"{TASK_ID} row {row_number}: qsp2__clean is present but qsp1 is missing.")
        spending_direction = checked_label(
            observed["qsp1"], {"higher", "lower"}, "qsp1", TASK_ID, row_number
        )
        spending_magnitude = abs(float(observed["qsp2__clean"]))
        spending_budget_work = (
            f"Your monthly household spending is {spending_magnitude:.1f} percent {spending_direction} "
            "than 12 months ago"
        )
    else:
        spending_budget_work = ""
    if "k2e" in observed:
        budget_label = checked_label(observed["k2e"], YES_NO_LABELS, "k2e", TASK_ID, row_number)
        if budget_label == "yes":
            spending_budget_work += ", your household currently uses a budget"
        else:
            spending_budget_work += ", your household does not currently use a budget"
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
            spending_budget_work += f", and you are {work_labels[0]}"
        else:
            spending_budget_work += f", and your current work situation includes {', '.join(work_labels)}"
    if spending_budget_work:
        finance_sentences.append(spending_budget_work + ".")
    prompt_parts.append(
        f"Here are your household finances, work situation, and expectations. {current_month_note(row['time'], currency=True)} "
        + " ".join(finance_sentences)
    )
    prompt_parts.append(render_core_macro_paragraph(observed))
    prompt_parts.append(SCENARIO_COMPACT)
    prompt_parts.append(FINAL_QUESTION)
    prompt_parts.append(RESPONSE_INSTRUCTION)
    user_prompt = "\n\n".join(prompt_parts)
    assistant_payload = [
        round(float(observed["gain_spend_or_donate_share"]), 2),
        round(float(observed["loss_reduce_spending_share"]), 2),
    ]
    records.append(
        {
            "id": str(observed["id"]),
            "time": str(observed["time"]),
            "release_date": str(observed["release_date"][:10]),
            "system": str(SYSTEM_PROMPT),
            "user": str(user_prompt),
            "assistant": json.dumps(assistant_payload, ensure_ascii=True),
            "naive_baseline": json.dumps([0, 0], ensure_ascii=True),
        }
    )

for record in records:
    validate_prompt_record(record)
    validate_cross_cutting_prompt_contract("cons_sce_shock", record)
write_jsonl(records, PROMPT_PATH)
print(f"{TASK_ID}: rendered {len(records):,} prompts to {PROMPT_PATH}")
