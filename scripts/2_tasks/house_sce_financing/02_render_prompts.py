#!/usr/bin/env python
"""Render source-exact user prompts for SCE financing choices."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.table_schema import read_public_table
from scripts.utils.prompt_rendering import (
    validate_prompt_record,
    validate_cross_cutting_prompt_contract,
    age_article,
    apply_prompt_render_limit,
    resolve_prompt_output_path,
    format_dollars,
    validate_model_facing_text,
)
from scripts.utils.io import write_jsonl
from scripts.utils.macro_context import CORE_MACRO_COLUMNS, render_core_macro_paragraph


TASK_ID = "house_sce_financing"
SYSTEM_PROMPT = (
    "You are an American adult making decisions about housing, borrowing, and household finances."
)
TABULAR_PATH = PROJECT_ROOT / "data/householdbench/tabular/house_sce_financing.csv"
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "data/householdbench/prompts/house_sce_financing.jsonl"
PROMPT_PATH = resolve_prompt_output_path(DEFAULT_PROMPT_PATH, task_id=TASK_ID)
OUTPUT_DIR = PROJECT_ROOT / "output/householdbench/tasks/house_sce_financing"
EXAMPLES_PATH = OUTPUT_DIR / "02_representative_prompt_pairs.jsonl"
EXAMPLE_COVERAGE_PATH = OUTPUT_DIR / "02_representative_prompt_coverage.csv"

INTRO = "Here is some background information about yourself and your household."
CURRENCY_NOTE = "$ denotes amounts in U.S. dollars."
FINAL_QUESTION = (
    "For each situation, state the maximum home price you would be willing to pay and the dollar down "
    "payment you would choose."
)
RESPONSE_INSTRUCTION = (
    "Return only a valid JSON array with exactly 6 numbers in this order: "
    "situation_1_maximum_price, situation_1_down_payment, situation_2_maximum_price, "
    "situation_2_down_payment, situation_3_maximum_price, situation_3_down_payment. All six "
    "numbers must be nominal U.S. dollars. Your output must follow this exact array "
    "structure: [v_1, v_2, v_3, v_4, v_5, v_6]. Replace every v placeholder with a single "
    "number. Do not include keys or explanatory text."
)
TARGET_COLUMNS = [
    "wtp_flexible_original_rate",
    "down_payment_flexible_original_rate",
    "wtp_flexible_alternative_rate",
    "down_payment_flexible_alternative_rate",
    "wtp_after_cash_inheritance",
    "down_payment_after_cash_inheritance",
]
REQUIRED_COLUMNS = [
    "id",
    "time",
    "release_date",
    "age",
    "sex",
    "education",
    "married",
    "residence_children_under18",
    "residence_children_over18",
    "income_band",
    "region",
    "liquid_savings_band",
    "non_housing_debt_band",
    "credit_score_band",
    "numeracy_score",
    "willingness_to_take_risks",
    "probability_move_three_years",
    "property_good_investment",
    "tenure",
    "current_home_value",
    "housing_debt",
    "comparable_home_value",
    "initial_rate",
    "initial_wtp",
    "initial_down_payment",
    "initial_effective_monthly_payment",
    *[column for column in CORE_MACRO_COLUMNS if column != "macro_reference_q_index"],
    *TARGET_COLUMNS,
]


def text_row(row: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for column, value in row.items():
        value = str(value).strip()
        if value and value.lower() not in {"nan", "nat", "none", "null"}:
            out[column] = value
    return out


def dollar(value: str) -> str:
    return format_dollars(float(value))


def range_clause(value: str) -> str:
    if " to " in value:
        lower, upper = value.split(" to ", maxsplit=1)
        return f"between {lower} and {upper}"
    return value


def numeric_dollar(value: str) -> int | float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Expected a finite dollar value, got {value!r}.")
    return int(number) if number.is_integer() else round(number, 2)


def children_text(under_18: int, over_18: int) -> str:
    if under_18 + over_18 == 0:
        return "no children"
    parts = []
    if under_18:
        parts.append(f"{under_18} child" + ("" if under_18 == 1 else "ren") + " under 18")
    if over_18:
        parts.append(f"{over_18} adult child" + ("" if over_18 == 1 else "ren"))
    return " and ".join(parts)


def ownership_instruction(row: dict[str, str]) -> str:
    if row["tenure"] == "owner":
        return "You would sell your current home and use the proceeds to pay off any remaining mortgage."
    if row["tenure"] == "renter":
        return "You do not need to sell a current home."
    raise RuntimeError(f"Unexpected tenure value: {row['tenure']!r}.")


def housing_sentence(row: dict[str, str]) -> str:
    if row["tenure"] == "owner":
        return (
            f"You own your current home, which is worth about {dollar(row['current_home_value'])}, and "
            f"you have about {dollar(row['housing_debt'])} in outstanding housing-related debt."
        )
    if row["tenure"] == "renter":
        return "You rent your current home."
    raise RuntimeError(f"Unexpected tenure value: {row['tenure']!r}.")


def scenario_text(row: dict[str, str]) -> str:
    alternative_rate = 6.5 if float(row["initial_rate"]) == 4.5 else 4.5
    return (
        "Now consider three further situations, in this order.\n\n"
        f"Situation 1: The mortgage rate remains {float(row['initial_rate']):.1f}%, but you may choose any "
        "down payment of at least 5% of the home price. You receive no extra cash before purchase.\n\n"
        f"Situation 2: The mortgage rate is {alternative_rate:.1f}%, and you may again choose "
        "any down payment of at least 5% of the home price. You receive no extra cash before purchase.\n\n"
        f"Situation 3: The mortgage rate remains {alternative_rate:.1f}%, you may choose any "
        "down payment of at least 5%, and you unexpectedly receive $100,000 in "
        "cash before buying the home. You may use all, some, or none of this cash for the down payment."
    )


if not TABULAR_PATH.is_file():
    raise RuntimeError(f"{TASK_ID}: task table is missing")
table = read_public_table(TABULAR_PATH, task_id="house_sce_financing")
rows = table.to_dict(orient="records")
fieldnames = list(table.columns)
if not rows:
    raise RuntimeError(f"Task table is empty: {TABULAR_PATH}")
missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
if missing_columns:
    raise RuntimeError(f"Task table is missing required columns: {missing_columns}")

rows = apply_prompt_render_limit(rows, task_id=TASK_ID)
records = []
record_metadata = []
for row_number, raw_row in enumerate(rows, start=1):
    row = text_row(raw_row)
    missing_values = [column for column in REQUIRED_COLUMNS if column not in row]
    if missing_values:
        raise RuntimeError(f"{TASK_ID} row {row_number}: missing required values {missing_values}")
    if row["tenure"] == "renter":
        row["current_home_value"] = ""
        row["housing_debt"] = ""
    elif row["tenure"] != "owner":
        raise RuntimeError(f"{TASK_ID} row {row_number}: invalid tenure {row['tenure']!r}")

    age = int(float(row["age"]))
    under_18 = int(float(row["residence_children_under18"]))
    over_18 = int(float(row["residence_children_over18"]))
    marriage_text = "are married" if row["married"].lower() == "true" else "are not married"
    child_text = children_text(under_18, over_18)
    owner_renter_instruction = ownership_instruction(row)
    personal_sentence = (
        f"You are {age_article(age)} {age}-year-old {row['sex']}, {marriage_text}, and your highest completed "
        f"education is {row['education']}. You live in the {row['region']} and have {child_text} living with you. "
        f"You answered {int(float(row['numeracy_score']))} of 5 numeracy questions correctly."
    )
    financial_sentence = (
        f"Your household income is {range_clause(row['income_band'])}. {housing_sentence(row)} Your liquid "
        f"savings are {row['liquid_savings_band']}, your non-housing debt is "
        f"{range_clause(row['non_housing_debt_band'])}, and your credit-score band is {row['credit_score_band']}. "
        f"A typical comparable home costs about {dollar(row['comparable_home_value'])}. On a scale "
        f"from 1 to 10, your willingness to take risks is {int(float(row['willingness_to_take_risks']))}. "
        f"You give a {float(row['probability_move_three_years']):.1f}% chance of moving within the next three "
        f"years. You {'do' if row['property_good_investment'] == 'yes' else 'do not'} regard property in your "
        "ZIP code as a good investment."
    )
    initial_sentence = (
        "Imagine that you move to a comparable city and are considering a comparable home where you plan to "
        f"remain for a long time. {owner_renter_instruction} With a 20% down payment and a 30-year mortgage "
        f"rate of {float(row['initial_rate']):.1f}%, the most you would pay for the home is "
        f"{dollar(row['initial_wtp'])}, implying a down payment of {dollar(row['initial_down_payment'])}. "
        f"This is equivalent to an effective total monthly payment of about "
        f"{dollar(row['initial_effective_monthly_payment'])} for a 30-year fixed-rate loan, including principal "
        "and interest, estimated mortgage-interest tax savings assuming that 95% of interest is deductible at "
        "a 34% income tax rate, and maintenance, property taxes, and homeowners insurance equal in total to "
        "3.35% of the comparable-home value per year. For reference, the principal-and-interest payment alone "
        "on each $100,000 borrowed is about $507 per month at 4.5% and $632 per month at 6.5%."
    )
    user_prompt = "\n\n".join(
        [
            f"{INTRO} {personal_sentence}",
            f"Here are your household finances, housing situation, preferences, and expectations. "
            f"Note that {CURRENCY_NOTE} {financial_sentence}",
            render_core_macro_paragraph(row),
            initial_sentence,
            scenario_text(row),
            FINAL_QUESTION,
            RESPONSE_INSTRUCTION,
        ]
    )

    validate_model_facing_text(user_prompt, field="user")
    target = [numeric_dollar(row[column]) for column in TARGET_COLUMNS]
    baseline = [
        numeric_dollar(row["initial_wtp"]),
        numeric_dollar(row["initial_down_payment"]),
        numeric_dollar(row["initial_wtp"]),
        numeric_dollar(row["initial_down_payment"]),
        numeric_dollar(row["initial_wtp"]),
        numeric_dollar(row["initial_down_payment"]),
    ]
    records.append(
        {
            "id": str(row["id"]),
            "time": str(row["time"]),
            "release_date": str(row["release_date"]),
            "system": str(SYSTEM_PROMPT),
            "user": str(user_prompt),
            "assistant": json.dumps(target, allow_nan=False),
            "naive_baseline": json.dumps(baseline, allow_nan=False),
        }
    )
    record_metadata.append(
        {
            "respondent_id": row["id"],
            "tenure": row["tenure"],
            "initial_rate": float(row["initial_rate"]),
        }
    )

PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
for record in records:
    validate_prompt_record(record)
    validate_cross_cutting_prompt_contract("house_sce_financing", record)
write_jsonl(records, PROMPT_PATH)

# A diagnostic subset need not cover every representative example category.
if PROMPT_PATH != DEFAULT_PROMPT_PATH:
    print(f"{TASK_ID}: rendered {len(records):,} diagnostic prompts to {PROMPT_PATH}")
    raise SystemExit(0)

representative_specs = [("owner", 4.5), ("renter", 4.5), ("owner", 6.5)]
representative_records = []
coverage_rows = []
for example_number, (tenure, initial_rate) in enumerate(representative_specs, start=1):
    matches = [
        index
        for index, metadata in enumerate(record_metadata)
        if metadata["tenure"] == tenure and metadata["initial_rate"] == initial_rate
    ]
    if not matches:
        raise RuntimeError(
            f"No representative prompt is available for tenure={tenure}, initial_rate={initial_rate}."
        )
    selected_index = matches[0]
    representative_records.append(records[selected_index])
    coverage_rows.append(
        {
            "task_id": TASK_ID,
            "example_number": example_number,
            "respondent_id": record_metadata[selected_index]["respondent_id"],
            "tenure": tenure,
            "initial_rate": initial_rate,
            "compact_present": 1,
        }
    )
write_jsonl(representative_records, EXAMPLES_PATH)
with EXAMPLE_COVERAGE_PATH.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(coverage_rows[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(coverage_rows)

print(
    f"{TASK_ID}: rendered {len(records):,} complete user prompts and "
    f"{len(representative_records)} representative prompts."
)
