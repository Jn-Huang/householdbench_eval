#!/usr/bin/env python
"""Render prompts for the SCE inflation-expectations task from its table."""

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
    checked_label,
    age_article,
    apply_prompt_render_limit,
)
from scripts.utils.table_schema import (
    read_public_table,
    public_table_columns,
    require_columns,
)

from scripts.utils.io import write_jsonl
from scripts.utils.macro_context import CORE_MACRO_COLUMNS, render_core_macro_paragraph


OUTPUT_SLUG = "macro_sce_revision"
TABULAR_PATH = PROJECT_ROOT / "data" / "householdbench" / "tabular" / f"{OUTPUT_SLUG}.csv"
PROMPTS_PATH = resolve_prompt_output_path(PROJECT_ROOT / "data" / "householdbench" / "prompts" / f"{OUTPUT_SLUG}.jsonl", task_id="macro_sce_revision")
SYSTEM_PROMPT = (
    "You are an American adult making household decisions and forming expectations "
    "about your finances, work, housing, and the economy."
)
BACKGROUND_INTRO = "Here is some background information about yourself and your household."
QUESTION = (
    "Given this information, by what percentage do you now expect consumer prices to change over "
    "the next 12 months? By what percentage do you now expect them to change over the 12-month "
    "period starting two years from now?"
)
RESPONSE_INSTRUCTION = (
    "Return only a valid JSON array with exactly two numbers, in this order: expected "
    "consumer-price change over the next 12 months; expected consumer-price change over the "
    "12-month period starting two years from now. Use a negative number if you expect "
    "consumer prices to fall. Your output must follow this exact array structure: [v_1, v_2]."
    " Replace every v placeholder with a single number. Do not include keys or explanatory "
    "text."
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
FINANCE_CHANGE_LABELS = {
    "much worse off",
    "somewhat worse off",
    "about the same",
    "somewhat better off",
    "much better off",
}
SELF_EMP_LABELS = {"work for someone else", "are self-employed"}
YES_NO_LABELS = {"yes", "no"}
WORK_STATUS_LABELS = [
    ("Q10_1_m12", "working full-time"),
    ("Q10_2_m12", "working part-time"),
    ("Q10_3_m12", "not working but would like to work"),
    ("Q10_4_m12", "temporarily laid off or on leave from work"),
    ("Q10_5_m12", "another current work attachment"),
]

REQUIRED_COLUMNS = [column for column in [
    "id",
    "time",
    "release_date",
    "age_years_m12",
    "_NUM_CAT_m12",
    "_REGION_CAT_m12",
    "_EDU_CAT_m12",
    "_HH_INC_DETAILED_m12",
    "Q1_m12",
    "Q2_m12",
    "Q10_1_m12",
    "Q10_2_m12",
    "Q10_3_m12",
    "Q10_4_m12",
    "Q10_5_m12",
    "Q11_m12",
    "Q12new_m12",
    "Q15_m12",
    "Q16_m12",
    "Q19_m12",
    "expected_inflation_next_12m_m1",
    "expected_inflation_2y3y_m1",
    "expected_inflation_next_12m_m12",
    "expected_inflation_2y3y_m12",
    "implied_expected_inflation_11m_m1",
    "realized_inflation_11m",
    "inflation_surprise_11m",
    *[f"{column}_m12" for column in CORE_MACRO_COLUMNS],
] if column in set(public_table_columns("macro_sce_revision"))]


def income_clause(income_label: str) -> str:
    if income_label == "Less than $10,000":
        return "less than $10,000"
    if income_label == "$200,000 or more":
        return "$200,000 or more"
    if " to " in income_label:
        lower, upper = income_label.split(" to ", maxsplit=1)
        return f"between {lower} and {upper}"
    raise RuntimeError(f"Unexpected detailed household-income label {income_label!r}.")


def change_phrase(value: float, *, verb: bool = False) -> str:
    if verb:
        return "increased" if value >= 0 else "fell"
    return "increase" if value >= 0 else "fall"


if not TABULAR_PATH.exists():
    raise SystemExit(f"Missing tabular input: {TABULAR_PATH}")
table = read_public_table(TABULAR_PATH, task_id="macro_sce_revision")

require_columns(table, REQUIRED_COLUMNS, label=TABULAR_PATH.name)
if table.empty:
    raise SystemExit(f"{TABULAR_PATH.name} has no rows.")
table = apply_prompt_render_limit(table, task_id=OUTPUT_SLUG)

records = []
for row_number, row in enumerate(table.to_dict("records"), start=1):
    age = int(round(float(row["age_years_m12"])))
    education = str(row["_EDU_CAT_m12"])
    region = str(row["_REGION_CAT_m12"])
    numeracy = str(row["_NUM_CAT_m12"])
    income = str(row["_HH_INC_DETAILED_m12"])
    if education not in EDUCATION_TEXT:
        raise RuntimeError(f"{OUTPUT_SLUG} row {row_number}: unmapped education category {education!r}.")
    if region not in REGION_TEXT:
        raise RuntimeError(f"{OUTPUT_SLUG} row {row_number}: unmapped region category {region!r}.")
    if numeracy not in NUMERACY_TEXT:
        raise RuntimeError(f"{OUTPUT_SLUG} row {row_number}: unmapped numeracy category {numeracy!r}.")

    earlier_12m = float(row["expected_inflation_next_12m_m1"])
    earlier_2y3y = float(row["expected_inflation_2y3y_m1"])
    implied_11m = float(row["implied_expected_inflation_11m_m1"])
    realized_11m = float(row["realized_inflation_11m"])
    implied_display = round(implied_11m, 1)
    realized_display = round(realized_11m, 1)
    surprise_display = round(realized_display - implied_display, 1)
    comparison = "above" if surprise_display >= 0 else "below"

    background = (
        f"{BACKGROUND_INTRO} You are {age_article(age)} {age}-year-old, {EDUCATION_TEXT[education]}, "
        f"live in {REGION_TEXT[region]}, and have a {NUMERACY_TEXT[numeracy]} numeracy score."
    )
    finance_sentences = [
        f"Your household's total pre-tax income over the past 12 months was {income_clause(income)}."
    ]
    q1_label = (
        checked_label(row.get("Q1_m12"), FINANCE_CHANGE_LABELS, "Q1_m12", OUTPUT_SLUG, row_number)
        if pd.notna(row.get("Q1_m12"))
        else None
    )
    q2_label = (
        checked_label(row.get("Q2_m12"), FINANCE_CHANGE_LABELS, "Q2_m12", OUTPUT_SLUG, row_number)
        if pd.notna(row.get("Q2_m12"))
        else None
    )
    if q1_label and q2_label:
        if q1_label == "about the same" and q2_label == "about the same":
            finance_sentences.append(
                "Your household is about as well off as it was 12 months ago, and you expect your "
                "household finances to remain about the same 12 months from now."
            )
        else:
            finance_sentences.append(
                f"Your household is {q1_label} than it was 12 months ago, but you expect your "
                f"household finances to be {q2_label} 12 months from now."
            )
    elif q1_label:
        finance_sentences.append(f"Your household is {q1_label} than it was 12 months ago.")
    elif q2_label:
        finance_sentences.append(f"You expect your household finances to be {q2_label} 12 months from now.")
    work_labels = []
    for column, label in WORK_STATUS_LABELS:
        if pd.notna(row.get(column)):
            work_status = checked_label(row[column], YES_NO_LABELS, column, OUTPUT_SLUG, row_number)
            if work_status == "yes":
                work_labels.append(label)
    if work_labels:
        if len(work_labels) == 1 and work_labels[0] == "working part-time":
            finance_sentences.append("You work part-time")
        elif len(work_labels) == 1 and work_labels[0] == "working full-time":
            finance_sentences.append("You work full-time")
        else:
            finance_sentences.append(f"Your current work situation includes {', '.join(work_labels)}")
    if pd.notna(row.get("Q11_m12")) and 0 <= float(row["Q11_m12"]) < 20:
        job_count = int(round(float(row["Q11_m12"])))
        if finance_sentences[-1].startswith("You work"):
            finance_sentences[-1] += f" and currently have {job_count} paid {'job' if job_count == 1 else 'jobs'}."
        else:
            finance_sentences.append(f"You currently have {job_count} paid {'job' if job_count == 1 else 'jobs'}.")
    if pd.notna(row.get("Q12new_m12")):
        self_emp_label = checked_label(
            row["Q12new_m12"], SELF_EMP_LABELS, "Q12new_m12", OUTPUT_SLUG, row_number
        )
        finance_sentences.append(f"For the relevant job, you {self_emp_label}.")
    if pd.notna(row.get("Q15_m12")):
        looking_label = checked_label(row["Q15_m12"], YES_NO_LABELS, "Q15_m12", OUTPUT_SLUG, row_number)
        finance_sentences.append(
            "You are currently looking for a job."
            if looking_label == "yes"
            else "You are not currently looking for a job."
        )
    if pd.notna(row.get("Q16_m12")) and 0 <= float(row["Q16_m12"]) <= 240:
        unemployed_months = int(round(float(row["Q16_m12"])))
        finance_sentences.append(
            f"You have been unemployed for {unemployed_months} "
            f"{'month' if unemployed_months == 1 else 'months'}."
        )
    if pd.notna(row.get("Q19_m12")) and 0 <= float(row["Q19_m12"]) <= 240:
        out_months = int(round(float(row["Q19_m12"])))
        finance_sentences.append(
            f"You have been out of work for {out_months} {'month' if out_months == 1 else 'months'}."
        )
    finance_paragraph = (
        "Here are your household finances, work situation, and expectations. "
        + " ".join(finance_sentences)
    )
    macro_row = {
        column: row[f"{column}_m12"]
        for column in CORE_MACRO_COLUMNS
        if column != "macro_reference_q_index"
    }
    macro_conditions = render_core_macro_paragraph(macro_row)
    policy_variation = f"Now consider how actual consumer prices have evolved compared with your expectations over recent months. {current_month_note(row['time'], currency=True)} Eleven months ago, you expected consumer prices to {change_phrase(earlier_12m)} by {abs(earlier_12m):.1f}% over the following 12 months, and to {change_phrase(earlier_2y3y)} by {abs(earlier_2y3y):.1f}% over the 12-month period two to three years ahead. Assuming a constant monthly expected inflation rate over the following 12 months, your first expectation corresponds to an expected {change_phrase(implied_display)} of {abs(implied_display):.1f}% over the first 11 months of that period. Actual consumer prices {change_phrase(realized_display, verb=True)} by {abs(realized_display):.1f}% over those 11 months. This was {abs(surprise_display):.1f} percentage points {comparison} the implied 11-month expectation."
    user_prompt = "\n\n".join(
        [background, finance_paragraph, macro_conditions, policy_variation, QUESTION, RESPONSE_INSTRUCTION]
    )

    assistant_payload = [
        round(float(row["expected_inflation_next_12m_m12"]), 3),
        round(float(row["expected_inflation_2y3y_m12"]), 3),
    ]
    records.append(
        {
            "id": str(row["id"]),
            "time": str(str(row["time"])[:10]),
            "release_date": str(str(row["release_date"])[:10]),
            "system": str(SYSTEM_PROMPT),
            "user": str(user_prompt),
            "assistant": json.dumps(assistant_payload, ensure_ascii=True),
            "naive_baseline": json.dumps(
                [
                    round(float(row["expected_inflation_next_12m_m1"]), 3),
                    round(float(row["expected_inflation_2y3y_m1"]), 3),
                ],
                ensure_ascii=True,
            ),
        }
    )

for record in records:
    validate_prompt_record(record)
    validate_cross_cutting_prompt_contract("macro_sce_revision", record)
write_jsonl(records, PROMPTS_PATH)
print(f"Wrote {PROMPTS_PATH} with {len(records):,} records.")
