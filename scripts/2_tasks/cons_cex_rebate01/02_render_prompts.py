#!/usr/bin/env python
"""Render HouseholdBench prompts for the 2001 tax-rebate CEX policy task."""

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
    validate_prompt_record,
)
from scripts.utils.table_schema import read_public_table, public_table_columns

from scripts.utils.macro_context import CORE_MACRO_COLUMNS, render_core_macro_paragraph


TASK_ID = "cons_cex_rebate01"
TABLE_PATH = PROJECT_ROOT / "data/householdbench/tabular/cons_cex_rebate01.csv"
PROMPTS_PATH = resolve_prompt_output_path(PROJECT_ROOT / "data/householdbench/prompts/cons_cex_rebate01.jsonl", task_id="cons_cex_rebate01")

SYSTEM_PROMPT = (
    "You are the head of an American household making decisions and forming "
    "expectations about income, spending, and saving."
)
POLICY_INTRODUCTION = (
    "The federal government has recently implemented a policy of sending tax "
    "rebates to households who satisfy certain criteria."
)
FINAL_QUESTION = (
    "Given this information, what were your total household expenditure, "
    "non-durable expenditure, and durable expenditure during the past three months, "
    "in current U.S. dollars?"
)
RESPONSE_INSTRUCTION = (
    "Return only a valid JSON array with exactly 3 numeric elements, in this order: "
    "total_expenditure, non_durable_expenditure, durable_expenditure. Your output must follow"
    " this exact array structure: [v_1, v_2, v_3]. Replace every v placeholder with a single "
    "number. Do not include keys or explanatory text."
)
PARAGRAPH_BREAK = "\n\n"

HOUSEHOLD_LAG_LABELS = {
    1: "between three and six months ago",
    2: "between six and nine months ago",
    3: "between nine and twelve months ago",
}

TARGET_COLUMNS = ["cons_parker_total", "cons_nondurables", "cons_durables"]
HISTORY_COLUMNS = [
    "cons_parker_total",
    "cons_nondurables",
    "cons_durables",
    "rebate_amount_total",
]
REQUIRED_COLUMNS = [column for column in [
    "id",
    "sample_id",
    "time",
    "release_date",
    "release_date_source",
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
    "rebate_amount_total",
    "rebate_payment_received",
    *[f"{column}_lag{lag}" for column in HISTORY_COLUMNS for lag in [1, 2, 3]],
    *CORE_MACRO_COLUMNS,
    *TARGET_COLUMNS,
] if column in set(public_table_columns("cons_cex_rebate01"))]
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
    if "_lag" not in column and column != "release_date_source"
]


def joined(items: list[str], *, sep: str = ", ") -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return sep.join(items[:-1]) + f"{sep}and {items[-1]}"


def render_policy_paragraph(
    row: dict[str, object], available_lags: list[int]
) -> str:
    if float(row["rebate_amount_total"]) > 0:
        current_payment = (
            "Over the past three months, your household received a total of "
            f"{format_dollars(row['rebate_amount_total'])} as part of the program."
        )
    else:
        current_payment = (
            "Over the past three months, no tax rebate is recorded for your household."
        )

    payment_parts = [
        f"{format_dollars(row[f'rebate_amount_total_lag{lag}'])} "
        f"{HOUSEHOLD_LAG_LABELS[lag]}"
        for lag in available_lags
    ]
    if len(available_lags) == 1:
        payment_history = (
            "Before this, the tax-rebate amount recorded for your household was "
            f"{payment_parts[0]}."
        )
    else:
        payment_history = (
            "Before this, the tax-rebate amounts recorded for your household were "
            f"{joined(payment_parts)}."
        )
    return " ".join([POLICY_INTRODUCTION, current_payment, payment_history])


table = read_public_table(TABLE_PATH, task_id="cons_cex_rebate01")

missing_columns = [column for column in REQUIRED_COLUMNS if column not in table.columns]
if missing_columns:
    raise SystemExit(f"{TABLE_PATH} is missing required columns: {missing_columns}")
if table.empty:
    raise SystemExit(f"{TABLE_PATH} contains no rows.")
if table[BASE_VALUE_COLUMNS].isna().any().any():
    bad_columns = (
        table[BASE_VALUE_COLUMNS]
        .columns[table[BASE_VALUE_COLUMNS].isna().any()]
        .tolist()
    )
    raise SystemExit(
        f"{TABLE_PATH} contains missing values in required columns: {bad_columns}"
    )
for column in CATEGORICAL_COLUMNS:
    numeric_like = (
        table[column]
        .astype("string")
        .str.strip()
        .str.fullmatch(r"\d+(?:\.0+)?")
        .fillna(False)
    )
    if numeric_like.any():
        raise SystemExit(
            f"{TABLE_PATH} contains numeric categorical values in {column}."
        )

table = apply_prompt_render_limit(table, task_id=TASK_ID)
PROMPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
records_written = 0

with PROMPTS_PATH.open("w", encoding="utf-8") as handle:
    for row_index, row_series in table.iterrows():
        row_number = row_index + 1
        row = row_series.to_dict()
        available_lags = []
        missing_lag_seen = False
        for lag in [1, 2, 3]:
            lag_columns = [
                f"{column}_lag{lag}" for column in HISTORY_COLUMNS
            ]
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
        negative_payment_lags = [
            f"rebate_amount_total_lag{lag}"
            for lag in available_lags
            if float(row[f"rebate_amount_total_lag{lag}"]) < 0
        ]
        if negative_payment_lags:
            raise RuntimeError(
                f"Row {row_number} has negative available payment lags: "
                f"{negative_payment_lags}."
            )

        if float(row["rebate_amount_total"]) < 0:
            raise RuntimeError(
                f"Row {row_number} has a negative current payment amount."
            )

        age = int(np.round(float(row["age_ref"])))
        sex = required_text(row, "sex_ref", row_number).lower()
        race = required_text(row, "race_ref_harmonized", row_number).lower()
        education = required_text(row, "educ_ref_harmonized", row_number).lower()
        marital = required_text(row, "marital1", row_number).lower()
        region = required_text(row, "region", row_number).rstrip(".")
        urban = required_text(row, "bls_urbn", row_number).lower()
        area_article = "an" if urban[:1] in {"a", "e", "i", "o", "u"} else "a"

        n_kids = int(np.round(float(row["n_kids"])))
        fam_size = int(np.round(float(row["fam_size"])))
        n_adults = max(fam_size - n_kids, 0)
        adult_label = "adult" if n_adults == 1 else "adults"
        child_phrase = (
            "no children"
            if n_kids == 0
            else f"{n_kids} {'child' if n_kids == 1 else 'children'}"
        )
        total_phrase = "" if fam_size == 1 else f", for {fam_size} people in total"

        expenditure_history_parts = []
        for lag in reversed(available_lags):
            lag_label = HOUSEHOLD_LAG_LABELS[lag]
            expenditure_history_parts.append(
                f"{lag_label.capitalize()}, your total household expenditure was "
                f"{format_dollars(row[f'cons_parker_total_lag{lag}'])}, non-durable expenditure was "
                f"{format_dollars(row[f'cons_nondurables_lag{lag}'])}, and durable expenditure was "
                f"{format_dollars(row[f'cons_durables_lag{lag}'])}."
            )

        macro_paragraph = render_core_macro_paragraph(row)

        policy_paragraph = render_policy_paragraph(row, available_lags)
        user_prompt = PARAGRAPH_BREAK.join(
            [
                (
                    "Here is some background information about yourself and your household. "
                    f"You are {age_article(age)} {age}-year-old {race} {sex}, are currently {marital}, "
                    f"and have {education_phrase(education)}. "
                    f"You currently reside in {area_article} {urban} area in the {region}. "
                    f"Your household has {n_adults} {adult_label} and {child_phrase}{total_phrase}."
                ),
                (
                    f"Here are your household's recent spending and income. {current_month_note(row['time'], currency=True)} "
                    + " ".join(expenditure_history_parts)
                    + " Your before-tax household income over the past 12 months was "
                    + f"{format_dollars(row['income_before_tax'])}."
                ),
                macro_paragraph,
                policy_paragraph,
                FINAL_QUESTION,
                RESPONSE_INSTRUCTION,
            ]
        )

        record = {
            "id": str(row["id"]),
            "time": str(row["time"]),
            "release_date": str(row["release_date"])[:10],
            "system": SYSTEM_PROMPT,
            "user": user_prompt,
            "assistant": json.dumps(
                [int(np.round(float(row[column]))) for column in TARGET_COLUMNS], ensure_ascii=True
            ),
            "naive_baseline": json.dumps(
                [int(np.round(float(row[f"{column}_lag1"]))) for column in TARGET_COLUMNS],
                ensure_ascii=True,
            ),
        }

        validate_prompt_record(record)
        handle.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
        records_written += 1

print(f"{TASK_ID}: wrote {records_written:,} prompt records to {PROMPTS_PATH}")
