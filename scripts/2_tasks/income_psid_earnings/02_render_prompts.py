#!/usr/bin/env python
"""Render JSONL prompts for the PSID earnings task from its table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.table_schema import read_public_table, public_table_columns
from scripts.utils.prompt_rendering import (
    resolve_prompt_output_path,
    age_article,
    apply_prompt_render_limit,
    format_dollars,
    optional_phrase_from_label,
    phrase_from_label,
    validate_prompt_record,
)

from scripts.utils.macro_context import CORE_MACRO_COLUMNS, render_core_macro_paragraph


TASK_NAME = "income_psid_earnings"
SYSTEM_PROMPT = (
    "You are an American individual or family making education, work, housing, "
    "and long-run financial decisions."
)
TABLE_PATH = PROJECT_ROOT / "data" / "householdbench" / "tabular" / f"{TASK_NAME}.csv"
PROMPTS_PATH = resolve_prompt_output_path(PROJECT_ROOT / "data" / "householdbench" / "prompts" / f"{TASK_NAME}.jsonl", task_id="income_psid_earnings")

FINAL_QUESTION = (
    "What will your labor earnings be in current dollars two years from now, four years from now, "
    "and ten years from now?"
)
RESPONSE_INSTRUCTION = (
    "Return only a valid JSON array with exactly 3 numeric elements, in this order: earn_2y, "
    "earn_4y, earn_10y. Your output must follow this exact array structure: [v_1, v_2, v_3]. "
    "Replace every v placeholder with a single number. Do not include keys or explanatory "
    "text."
)
DOLLAR_NOTE = "Note that $ denotes amounts in U.S. dollars."
REQUIRED_COLUMNS = [column for column in [
    "id",
    "year",
    "release_date",
    "demo_age_gen",
    "demo_sex",
    "geo_region",
    "earn_tot_nd",
    "earn_tot_nd_2y",
    "earn_tot_nd_4y",
    "earn_tot_nd_10y",
    *CORE_MACRO_COLUMNS,
] if column in set(public_table_columns("income_psid_earnings"))]

SEX_PHRASES = {"Male": "man", "Female": "woman"}
RACE_PHRASES = {
    "White": "White",
    "Black": "Black",
    "Hispanic": "Hispanic",
    "Other race": "other race or ethnicity",
}
REGION_PHRASES = {
    "Northeast": "the Northeast",
    "Midwest": "the Midwest",
    "South": "the South",
    "West": "the West",
    "Alaska or Hawaii": "Alaska or Hawaii",
    "Country outside the United States": "a country outside the United States",
}
METRO_PHRASES = {
    "Yes, metropolitan area": "metropolitan area",
    "No, non-metropolitan area": "non-metropolitan area",
}
EDU_PHRASES = {
    "Did not complete high school": "did not complete high school",
    "Completed high school, did not attend college": "completed high school and did not attend college",
    "Attended college, no bachelor's degree": "attended college without a bachelor's degree",
    "Bachelor's degree, no postgraduate degree": "has a bachelor's degree and no postgraduate degree",
    "Postgraduate degree": "has a postgraduate degree",
}
EMP_PHRASES = {
    "Not currently working": "not currently working",
    "Currently working": "currently working",
}
PARTNER_PHRASES = {
    "No, reference person has no spouse or partner in the family unit": "not partnered",
    "Yes, reference person has a spouse or partner in the family unit": "partnered",
}


def child_phrase(children: int) -> str:
    if children == 0:
        return "no children"
    if children == 1:
        return "1 child"
    return f"{children} children"


def signed_growth_phrase(label: str, value: float) -> str:
    if value < 0:
        return f"{label} fell by {abs(value):.2f}%"
    return f"{label} grew by {value:.2f}%"


def compact_occupation(text: str) -> str:
    occupation = text.strip()
    if occupation.lower().endswith(" occupation"):
        occupation = occupation[:-11]
    return occupation[:1].lower() + occupation[1:]


if not TABLE_PATH.exists():
    raise SystemExit(f"Task table not found: {TABLE_PATH}")

table = read_public_table(TABLE_PATH, task_id="income_psid_earnings")

missing_columns = [column for column in REQUIRED_COLUMNS if column not in table.columns]
if missing_columns:
    raise SystemExit(f"{TASK_NAME}: table is missing required columns: {missing_columns}")

table = apply_prompt_render_limit(table, task_id=TASK_NAME)
PROMPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with PROMPTS_PATH.open("w", encoding="utf-8") as handle:
    for row_number, row in enumerate(table.to_dict(orient="records"), start=1):
        row_id = str(row["id"])
        prompt_time = str(row["time"])
        target = [
            int(round(float(row["earn_tot_nd_2y"]))),
            int(round(float(row["earn_tot_nd_4y"]))),
            int(round(float(row["earn_tot_nd_10y"]))),
        ]

        age = int(round(float(row["demo_age_gen"])))
        sex = phrase_from_label(row, "demo_sex", SEX_PHRASES, row_id)

        race_part = ""
        race_phrase = optional_phrase_from_label(row, "race_eth_maj_col", RACE_PHRASES, row_id)
        if race_phrase:
            race_part = f" {race_phrase}"
        demographic_sentence = f"You are {age_article(age)} {age}-year-old{race_part} {sex}."

        geography_sentence = None
        region = optional_phrase_from_label(row, "geo_region", REGION_PHRASES, row_id)
        if region:
            geography_sentence = f"You live in {region}."
        if geography_sentence and row.get("geo_metro") is not None and not pd.isna(row.get("geo_metro")):
            metro = optional_phrase_from_label(row, "geo_metro", METRO_PHRASES, row_id)
            if metro:
                geography_sentence = f"You live in {region}, in a {metro}."

        education_sentences = []
        if row.get("edu_year_max") is not None and not pd.isna(row.get("edu_year_max")):
            education_years = int(round(float(row["edu_year_max"])))
        else:
            education_years = None
        if row.get("edu_level_max") is not None and not pd.isna(row.get("edu_level_max")):
            education_level = phrase_from_label(row, "edu_level_max", EDU_PHRASES, row_id)
        else:
            education_level = None
        if education_years is not None and education_level:
            education_sentences.append(
                f"You completed {education_years} years of schooling and {education_level}."
            )
        elif education_years is not None:
            education_sentences.append(f"You completed {education_years} years of schooling.")
        elif education_level:
            education_sentences.append(f"You {education_level}.")

        family_sentences = []
        partnered_phrase = optional_phrase_from_label(row, "fam_partnered", PARTNER_PHRASES, row_id)
        if partnered_phrase == "partnered":
            family_sentences.append("You have a spouse or partner in your household.")
        elif partnered_phrase == "not partnered":
            family_sentences.append("You do not have a spouse or partner in your household.")
        if row.get("fam_size") is not None and not pd.isna(row.get("fam_size")):
            family_size = int(round(float(row["fam_size"])))
            if row.get("fam_size_chi") is not None and not pd.isna(row.get("fam_size_chi")):
                children = int(round(float(row["fam_size_chi"])))
                family_sentences.append(
                    f"Your household has {family_size} {'person' if family_size == 1 else 'people'}, "
                    f"including {child_phrase(children)}."
                )
            else:
                family_sentences.append(
                    f"Your household has {family_size} {'person' if family_size == 1 else 'people'}."
                )

        work_sentences = []
        if row.get("emp_work") is not None and not pd.isna(row.get("emp_work")):
            emp_phrase = phrase_from_label(row, "emp_work", EMP_PHRASES, row_id)
            if emp_phrase == "currently working":
                work_sentences.append("You are currently working.")
            else:
                work_sentences.append("You are not currently working.")
        earnings_income_parts = []
        if row.get("earn_tot_nd") is not None and not pd.isna(row.get("earn_tot_nd")):
            earnings_income_parts.append(f"Your labor earnings are {format_dollars(row['earn_tot_nd'])}")
        if row.get("finc_tot_nd") is not None and not pd.isna(row.get("finc_tot_nd")):
            earnings_income_parts.append(f"your household income is {format_dollars(row['finc_tot_nd'])}")
        if row.get("occ_major_harmonized") is not None and not pd.isna(row.get("occ_major_harmonized")):
            occupation = str(row["occ_major_harmonized"]).strip()
            if occupation and occupation.lower() != "unknown":
                earnings_income_parts.append(f"your broad occupation is {compact_occupation(occupation)}")
        if earnings_income_parts:
            if len(earnings_income_parts) == 1:
                work_sentences.append(earnings_income_parts[0] + ".")
            elif len(earnings_income_parts) == 2:
                work_sentences.append(" and ".join(earnings_income_parts) + ".")
            else:
                work_sentences.append(", ".join(earnings_income_parts[:-1]) + f", and {earnings_income_parts[-1]}.")

        macro_sentence = render_core_macro_paragraph(row)

        background_paragraph = (
            "Here is some background information about yourself and your household. "
            + " ".join(part for part in [demographic_sentence, geography_sentence, *education_sentences, *family_sentences] if part)
        )
        work_paragraph = (
            f"Here is your current work situation. {DOLLAR_NOTE} "
            + " ".join(work_sentences)
        )
        user_prompt_parts = [background_paragraph, work_paragraph]
        if macro_sentence:
            user_prompt_parts.append(macro_sentence)
        user_prompt_parts.extend([FINAL_QUESTION, RESPONSE_INSTRUCTION])
        user_prompt = "\n\n".join(part for part in user_prompt_parts if part)

        record = {
            "id": row_id,
            "time": prompt_time,
            "release_date": str(row["release_date"]),
            "system": SYSTEM_PROMPT,
            "user": user_prompt,
            "assistant": json.dumps(target, ensure_ascii=True),
            "naive_baseline": json.dumps(
                [int(round(float(row["earn_tot_nd"])))] * 3, ensure_ascii=True
            ),
        }
        validate_prompt_record(record)
        handle.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")

print(f"Wrote {PROMPTS_PATH} from {TABLE_PATH}")
