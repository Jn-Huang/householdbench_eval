#!/usr/bin/env python
"""Render prompts for the SCE household income growth task from its table."""

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
from scripts.utils.table_schema import (
    read_public_table,
    public_table_columns,
    require_columns,
)

from scripts.utils.io import write_jsonl
from scripts.utils.macro_context import CORE_MACRO_COLUMNS, render_core_macro_paragraph


OUTPUT_SLUG = 'income_sce_growth'
TABULAR_PATH = PROJECT_ROOT / "data" / "householdbench" / "tabular" / f"{OUTPUT_SLUG}.csv"
PROMPTS_PATH = resolve_prompt_output_path(PROJECT_ROOT / "data" / "householdbench" / "prompts" / f"{OUTPUT_SLUG}.jsonl", task_id="income_sce_growth")
SYSTEM_PROMPT = (
    "You are an American adult making household decisions and forming expectations "
    "about your finances, work, and the economy."
)

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
HEALTH_LABELS = {"excellent", "very good", "good", "fair", "poor"}
SELF_EMP_LABELS = {"work for someone else", "are self-employed"}
YES_NO_LABELS = {"yes", "no"}
WORK_STATUS_LABELS = [
    ("Q10_1", "working full-time"),
    ("Q10_2", "working part-time"),
    ("Q10_3", "not working but would like to work"),
    ("Q10_4", "temporarily laid off or on leave from work"),
    ("Q10_5", "another current work attachment"),
]


FINAL_QUESTION = (
    "By what percentage do you expect total household income to change over the next 12 months? "
    "Use a negative number for a decline, a positive number for an increase, and 0 for no change."
)
RESPONSE_INSTRUCTION = (
    "Return only a valid JSON array containing one number. Your output must follow this exact"
    " array structure: [v_1]. Replace every v placeholder with a single number. Do not "
    "include explanatory text."
)
PROMPT_INTRO = "Here is some background information about yourself and your household."

required_columns = [column for column in ['id',
 'id',
 'time',
 'release_date',
 'age_years',
 '_NUM_CAT',
 '_REGION_CAT',
 '_EDU_CAT',
 '_HH_INC_DETAILED',
 'Q1',
 'Q2',
 'Q45b',
 'Q10_1',
 'Q10_2',
 'Q10_3',
 'Q10_4',
 'Q10_5',
 'Q11',
 'Q12new',
 'Q15',
 'Q16',
 'Q19',
 *CORE_MACRO_COLUMNS,
 'Q25v2part2'] if column in set(public_table_columns("income_sce_growth"))]

if not TABULAR_PATH.exists():
    raise SystemExit(f"Missing tabular input: {TABULAR_PATH}")
table = read_public_table(TABULAR_PATH, task_id="income_sce_growth")

require_columns(table, required_columns, label=TABULAR_PATH.name)
if table.empty:
    raise SystemExit(f"{TABULAR_PATH.name} has no rows.")

table = apply_prompt_render_limit(table, task_id=OUTPUT_SLUG)

records = []
for row_number, row in enumerate(table.to_dict("records"), start=1):
    prompt_parts = []
    age_value = int(round(float(row["age_years"])))
    education_value = str(row["_EDU_CAT"])
    if education_value not in EDUCATION_TEXT:
        raise RuntimeError(f"{OUTPUT_SLUG} row {row_number}: unmapped education category {education_value!r}.")
    region_value = str(row["_REGION_CAT"])
    if region_value not in REGION_TEXT:
        raise RuntimeError(f"{OUTPUT_SLUG} row {row_number}: unmapped region category {region_value!r}.")
    numeracy_value = str(row["_NUM_CAT"])
    if numeracy_value not in NUMERACY_TEXT:
        raise RuntimeError(f"{OUTPUT_SLUG} row {row_number}: unmapped numeracy category {numeracy_value!r}.")
    prompt_parts.append(
        f"{PROMPT_INTRO} You are {age_article(age_value)} {age_value}-year-old, {EDUCATION_TEXT[education_value]}, "
        f"live in {REGION_TEXT[region_value]}, and have a {NUMERACY_TEXT[numeracy_value]} numeracy score."
    )

    income_value = str(row["_HH_INC_DETAILED"])
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
    if pd.notna(row.get("Q1")):
        q1_label = checked_label(row.get("Q1"), FINANCE_CHANGE_LABELS, "Q1", OUTPUT_SLUG, row_number)
    else:
        q1_label = None
    if pd.notna(row.get("Q2")):
        q2_label = checked_label(row.get("Q2"), FINANCE_CHANGE_LABELS, "Q2", OUTPUT_SLUG, row_number)
    else:
        q2_label = None
    if q1_label and q2_label:
        connector = "but" if "worse" in q1_label and "same" in q2_label else "and"
        finance_sentences.append(
            f"Your household is {q1_label} than it was 12 months ago, {connector} you expect your household "
            f"finances to be {q2_label} 12 months from now."
        )
    work_labels = []
    for column, label in WORK_STATUS_LABELS:
        if pd.notna(row.get(column)):
            work_status = checked_label(row[column], YES_NO_LABELS, column, OUTPUT_SLUG, row_number)
            if work_status == "yes":
                work_labels.append(label)
    work_sentence = ""
    if work_labels:
        if len(work_labels) == 1:
            work_sentence = f"You work {work_labels[0].replace('working ', '')}"
        else:
            work_sentence = f"Your current work situation includes {', '.join(work_labels)}"
    if pd.notna(row.get("Q11")) and 0 <= float(row["Q11"]) < 20:
        job_count = int(round(float(row["Q11"])))
        if work_sentence:
            work_sentence += f" and currently have {job_count} paid {'job' if job_count == 1 else 'jobs'}"
        else:
            work_sentence = f"You currently have {job_count} paid {'job' if job_count == 1 else 'jobs'}"
    if pd.notna(row.get("Q12new")):
        self_emp_label = checked_label(row["Q12new"], SELF_EMP_LABELS, "Q12new", OUTPUT_SLUG, row_number)
        if work_sentence:
            work_sentence += f", and you {self_emp_label}"
    if pd.notna(row.get("Q15")):
        looking_label = checked_label(row["Q15"], YES_NO_LABELS, "Q15", OUTPUT_SLUG, row_number)
        if looking_label == "yes":
            work_sentence += ", and you are currently looking for a job"
        else:
            work_sentence += ", and you are not currently looking for a job"
    if pd.notna(row.get("Q16")) and 0 <= float(row["Q16"]) <= 240:
        unemployed_months = int(round(float(row["Q16"])))
        month_word = "month" if unemployed_months == 1 else "months"
        work_sentence += f", and you have been unemployed for {unemployed_months} {month_word}"
    if pd.notna(row.get("Q19")) and 0 <= float(row["Q19"]) <= 240:
        out_months = int(round(float(row["Q19"])))
        month_word = "month" if out_months == 1 else "months"
        work_sentence += f", and you have been out of work for {out_months} {month_word}"
    if work_sentence:
        finance_sentences.append(work_sentence + ".")
    prompt_parts.append(
        f"Here are your household finances, work situation, and expectations. {current_month_note(row['time'], currency=True)} "
        + " ".join(finance_sentences)
    )
    prompt_parts.append(render_core_macro_paragraph(row))
    prompt_parts.append(FINAL_QUESTION)
    prompt_parts.append(RESPONSE_INSTRUCTION)
    user_prompt = "\n\n".join(prompt_parts)
    assistant_payload = round(float(row['Q25v2part2']), 3)
    records.append(
        {
            "id": str(row["id"]),
            "time": str(str(row["time"])[:10]),
            "release_date": str(str(row["release_date"])[:10]),
            "system": str(SYSTEM_PROMPT),
            "user": str(user_prompt),
            "assistant": json.dumps([assistant_payload], ensure_ascii=True),
            "naive_baseline": json.dumps([0], ensure_ascii=True),
        }
    )

for record in records:
    validate_prompt_record(record)
    validate_cross_cutting_prompt_contract("income_sce_growth", record)
write_jsonl(records, PROMPTS_PATH)
print(f"Wrote {PROMPTS_PATH} with {len(records):,} records.")
