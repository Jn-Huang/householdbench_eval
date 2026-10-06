#!/usr/bin/env python
"""Render prompts for the SCE macro-risk uncertainty task from its table."""

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


OUTPUT_SLUG = 'macro_sce_uncertainty'
TABULAR_PATH = PROJECT_ROOT / "data" / "householdbench" / "tabular" / f"{OUTPUT_SLUG}.csv"
PROMPTS_PATH = resolve_prompt_output_path(PROJECT_ROOT / "data" / "householdbench" / "prompts" / f"{OUTPUT_SLUG}.jsonl", task_id="macro_sce_uncertainty")
SYSTEM_PROMPT = (
    "You are an American adult making household decisions and forming expectations "
    "about your finances, work, housing, and the economy."
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
Q9_BIN_COLUMNS = [f"Q9_bin{i}__clean" for i in range(1, 11)]
Q9C_BIN_COLUMNS = [f"Q9c_bin{i}__clean" for i in range(1, 11)]
MACRO_PROBABILITY_COLUMNS = ["Q4new__clean", "Q5new__clean", "Q6new__clean"]
INFLATION_BIN_ORDER_TEXT = (
    "inflation at least 12 percent, inflation 8 to 12 percent, inflation 4 to 8 percent, "
    "inflation 2 to 4 percent, inflation 0 to 2 percent, deflation 0 to 2 percent, "
    "deflation 2 to 4 percent, deflation 4 to 8 percent, deflation 8 to 12 percent, "
    "deflation at least 12 percent"
)


FINAL_QUESTION = (
    "Now think about the different things that may happen to inflation. For each of the following ten "
    "ranges, what probability do you think there is that inflation over the next 12 months will fall "
    f"in that range? Use this order: {INFLATION_BIN_ORDER_TEXT}. Then, for the 12-month period two to "
    "three years from now, what probability do you think there is that inflation will fall in each "
    "of the same ranges? Finally, what probability "
    "do you think there is that 12 months from now the U.S. unemployment rate will be higher than it is "
    "now, that the average interest rate on savings accounts will be higher than it is now, and that "
    "U.S. stock prices will be higher than they are now?"
)
RESPONSE_INSTRUCTION = (
    "Return only a valid JSON array containing exactly five probability distributions, in "
    "this order: the ten-element one-year inflation distribution; the ten-element "
    "two-to-three-year inflation distribution; the two-element higher-unemployment "
    "distribution; the two-element higher-savings-rate distribution; the two-element "
    "higher-stock-price distribution. Preserve the inflation-bin order stated above. Within "
    "each two-element distribution, use this order: event does not occur, event occurs. "
    "Express each probability as a number from 0 to 1. Each distribution must sum to 1. Your "
    "output must follow this exact array structure: [[p_1_1, ..., p_1_10], [p_2_1, ..., "
    "p_2_10], [p_3_1, p_3_2], [p_4_1, p_4_2], [p_5_1, p_5_2]]. Replace every p placeholder "
    "with a number from 0 to 1. Do not include keys or explanatory text."
)

required_columns = [column for column in ['id',
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
 *Q9_BIN_COLUMNS,
 *Q9C_BIN_COLUMNS,
 'Q4new__clean',
 'Q5new__clean',
 'Q6new__clean', *[f'lag1_{column}' for column in Q9_BIN_COLUMNS], *[f'lag1_{column}' for column in Q9C_BIN_COLUMNS], 'lag1_unemployment_higher_probability', 'lag1_savings_rate_higher_probability', 'lag1_stock_prices_higher_probability'] if column in set(public_table_columns("macro_sce_uncertainty"))]

if not TABULAR_PATH.exists():
    raise SystemExit(f"Missing tabular input: {TABULAR_PATH}")
table = read_public_table(TABULAR_PATH, task_id="macro_sce_uncertainty")

require_columns(table, required_columns, label=TABULAR_PATH.name)
if table.empty:
    raise SystemExit(f"{TABULAR_PATH.name} has no rows.")

table = apply_prompt_render_limit(table, task_id=OUTPUT_SLUG)

records = []
for row_number, row in enumerate(table.to_dict("records"), start=1):
    age_value = int(round(float(row["age_years"])))
    education_value = str(row["_EDU_CAT"])
    if education_value not in EDUCATION_TEXT:
        raise RuntimeError(f"{OUTPUT_SLUG} row {row_number}: unmapped education category {education_value!r}.")
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
    region_value = str(row["_REGION_CAT"])
    if region_value not in REGION_TEXT:
        raise RuntimeError(f"{OUTPUT_SLUG} row {row_number}: unmapped region category {region_value!r}.")
    numeracy_value = str(row["_NUM_CAT"])
    if numeracy_value not in NUMERACY_TEXT:
        raise RuntimeError(f"{OUTPUT_SLUG} row {row_number}: unmapped numeracy category {numeracy_value!r}.")
    background_paragraph = (
        "Here is some background information about yourself and your household. "
        f"You are {age_article(age_value)} {age_value}-year-old, {EDUCATION_TEXT[education_value]}, live in "
        f"{REGION_TEXT[region_value]}, and have a {NUMERACY_TEXT[numeracy_value]} numeracy score."
    )
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
        if q1_label == "about the same" and q2_label == "about the same":
            finance_sentences.append(
                "Your household is about as well off as it was 12 months ago, and you expect your household finances to remain about the same 12 months from now."
            )
        else:
            finance_sentences.append(
                f"Your household is {q1_label} than it was 12 months ago, but you expect your household finances to be {q2_label} 12 months from now."
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
    if pd.notna(row.get("Q11")) and 0 <= float(row["Q11"]) < 20:
        job_count = int(round(float(row["Q11"])))
        job_word = "current paid job" if job_count == 1 else "current paid jobs"
        if finance_sentences[-1].startswith("You work"):
            finance_sentences[-1] += f" and currently have {job_count} paid {'job' if job_count == 1 else 'jobs'}."
        else:
            finance_sentences.append(f"You currently have {job_count} paid {'job' if job_count == 1 else 'jobs'}.")
    if pd.notna(row.get("Q12new")):
        self_emp_label = checked_label(row["Q12new"], SELF_EMP_LABELS, "Q12new", OUTPUT_SLUG, row_number)
        finance_sentences.append(f"For the relevant job, you {self_emp_label}.")
    if pd.notna(row.get("Q15")):
        looking_label = checked_label(row["Q15"], YES_NO_LABELS, "Q15", OUTPUT_SLUG, row_number)
        if looking_label == "yes":
            finance_sentences.append("You are currently looking for a job.")
        else:
            finance_sentences.append("You are not currently looking for a job.")
    if pd.notna(row.get("Q16")) and 0 <= float(row["Q16"]) <= 240:
        unemployed_months = int(round(float(row["Q16"])))
        month_word = "month" if unemployed_months == 1 else "months"
        finance_sentences.append(f"You have been unemployed for {unemployed_months} {month_word}.")
    if pd.notna(row.get("Q19")) and 0 <= float(row["Q19"]) <= 240:
        out_months = int(round(float(row["Q19"])))
        month_word = "month" if out_months == 1 else "months"
        finance_sentences.append(f"You have been out of work for {out_months} {month_word}.")
    lag1_one_year = [float(row[f"lag1_{column}"]) for column in Q9_BIN_COLUMNS]
    lag1_two_to_three_year = [float(row[f"lag1_{column}"]) for column in Q9C_BIN_COLUMNS]
    finance_sentences.append(f"One month ago, your probabilities for inflation over the following 12 months, in the same ten-bin order used below, were {lag1_one_year}. Your probabilities for the 12-month period two to three years ahead were {lag1_two_to_three_year}. You assigned probability {float(row['lag1_unemployment_higher_probability']):.3f} to the U.S. unemployment rate being higher 12 months later, probability {float(row['lag1_savings_rate_higher_probability']):.3f} to the average interest rate on savings accounts being higher, and probability {float(row['lag1_stock_prices_higher_probability']):.3f} to U.S. stock prices being higher.")
    finance_paragraph = (
        f"Here are your household finances, work situation, and expectations. {current_month_note(row['time'], currency=True)} "
        + " ".join(finance_sentences)
    )
    macro_paragraph = render_core_macro_paragraph(row)
    prompt_parts = [
        background_paragraph,
        finance_paragraph,
        macro_paragraph,
        FINAL_QUESTION,
        RESPONSE_INSTRUCTION,
    ]
    user_prompt = "\n\n".join(prompt_parts)
    assistant_payload = [
        [float(row[column]) for column in Q9_BIN_COLUMNS],
        [float(row[column]) for column in Q9C_BIN_COLUMNS],
    ]
    naive_baseline = [lag1_one_year, lag1_two_to_three_year]
    for target_column, lag_column in zip(MACRO_PROBABILITY_COLUMNS, ['lag1_unemployment_higher_probability', 'lag1_savings_rate_higher_probability', 'lag1_stock_prices_higher_probability'], strict=True):
        event_probability = round(float(row[target_column]), 3)
        lag_event_probability = round(float(row[lag_column]), 3)
        assistant_payload.append([round(1 - event_probability, 3), event_probability])
        naive_baseline.append([round(1 - lag_event_probability, 3), lag_event_probability])
    records.append(
        {
            "id": str(row["id"]),
            "time": str(str(row["time"])[:10]),
            "release_date": str(str(row["release_date"])[:10]),
            "system": str(SYSTEM_PROMPT),
            "user": str(user_prompt),
            "assistant": json.dumps(assistant_payload, ensure_ascii=True),
            "naive_baseline": json.dumps(naive_baseline, ensure_ascii=True),
        }
    )

for record in records:
    validate_prompt_record(record)
    validate_cross_cutting_prompt_contract("macro_sce_uncertainty", record)
write_jsonl(records, PROMPTS_PATH)
print(f"Wrote {PROMPTS_PATH} with {len(records):,} records.")
