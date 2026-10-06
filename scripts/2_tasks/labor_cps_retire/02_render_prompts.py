#!/usr/bin/env python
"""Render CPS retirement prompts from the task table."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


TASK_SLUG = "labor_cps_retire"
TARGETS = {"retired", "not_retired"}
NOT_OBSERVED = "not observed"
SYSTEM_PROMPT = (
    "You are an American person answering questions about your work, job search, "
    "and labor-force situation."
)
QUESTION = "Will you be retired one year from now?"
TARGET_INSTRUCTION = 'Use "retired" if you will be retired and "not_retired" if you will not be retired.'

REPO_ROOT = Path(__file__).resolve().parents[3]
TABLE_PATH = REPO_ROOT / f"data/householdbench/tabular/{TASK_SLUG}.csv"
PROMPT_PATH = REPO_ROOT / f"data/householdbench/prompts/{TASK_SLUG}.jsonl"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.prompt_rendering import (
    cps_household_sentence as household_sentence,
    current_month_note,
    resolve_prompt_output_path,
    age_article,
    apply_prompt_render_limit,
    format_dollars,
    validate_prompt_record,
    write_prompt_lines,
)
from scripts.utils.table_schema import read_public_table, public_table_columns

PROMPT_PATH = resolve_prompt_output_path(PROMPT_PATH, task_id="labor_cps_retire")

from scripts.utils.macro_context import CORE_MACRO_COLUMNS, render_core_macro_paragraph


SCHEMA_INSTRUCTION = (
    'Return exactly one valid JSON array containing one string: "retired" or "not_retired". '
    'Your output must follow this exact array structure: ["s_1"]. Replace every s placeholder'
    " with one of the permitted strings. Do not include explanatory text."
)


IDENTIFIER_COLUMNS = [
    "id",
    "time",
    "release_date",
    "target",
]
PROMPT_PREDICTOR_COLUMNS = [
    "age",
    "sex",
    "race",
    "educ",
    "marst",
    "state",
    "region",
    "metro",
    "famsize",
    "nchild",
    "hispan",
    "nativity",
    "citizen",
    "vetstat",
    "baseline_status",
    "ahrsworkt_baseline",
    "uhrsworkt_baseline",
    "classwkr",
    "multjob_baseline",
    "union_baseline",
    "earnweek_baseline",
    "difficulty",
    "family_income",
    "occupation",
    "industry",
    "partner_linked_baseline",
    "partner_age_baseline",
    "partner_status_baseline",
    *[column for column in CORE_MACRO_COLUMNS if column != "macro_reference_q_index"],
]
REQUIRED_COLUMNS = [column for column in IDENTIFIER_COLUMNS + PROMPT_PREDICTOR_COLUMNS if column in set(public_table_columns("labor_cps_retire"))]

MISSING_STRINGS = {"", "<na>", "nan", "nat", "none", "null", "missing", "unknown"}

EDUC_PHRASE = {
    "high school or less": "high school education or less",
    "high school diploma or GED": "a high school diploma or GED",
    "some college or associate degree": "some college or an associate degree",
    "bachelor's degree": "a bachelor's degree",
    "graduate or professional education": "graduate or professional education",
}
MARITAL_BACKGROUND = {
    "married, spouse present": "are married, your spouse is present",
    "married, spouse absent": "are married, your spouse is absent",
    "separated": "are separated",
    "divorced": "are divorced",
    "widowed": "are widowed",
    "never married": "have never been married",
    "widowed or divorced": "are widowed or divorced",
}
HISPAN_SENTENCE = {
    "not Hispanic": "You are not Hispanic.",
    "Mexican origin": "You are of Mexican origin.",
    "Puerto Rican": "You are Puerto Rican.",
    "Cuban": "You are Cuban.",
    "Dominican": "You are Dominican.",
    "other Hispanic": "You are Hispanic, with another Hispanic origin.",
}
NATIVITY_CITIZENSHIP_SENTENCE = {
    ("native-born", "U.S.-born citizen"): "You were born in the United States and are a U.S. citizen.",
    ("native-born", "not a U.S. citizen"): "You are native-born and are not a U.S. citizen.",
    ("foreign-born", "U.S.-born citizen"): "You were born outside the United States and are a U.S. citizen.",
    ("foreign-born", "naturalized citizen"): "You were born outside the United States and are a naturalized U.S. citizen.",
    ("foreign-born", "not a U.S. citizen"): "You were born outside the United States and are not a U.S. citizen.",
}
VETERAN_SENTENCE = {
    "not a veteran": "You are not a veteran.",
    "veteran": "You are a veteran.",
}
CURRENT_EMPLOYER_PHRASE = {
    "private for-profit wage and salary worker": "by a private for-profit employer",
    "private wage and salary worker": "in the private sector",
    "private nonprofit wage and salary worker": "by a private nonprofit employer",
    "state government employee": "by state government",
    "federal government employee": "by the federal government",
    "local government employee": "by local government",
    "government wage and salary worker": "by the government",
    "self-employed": "as self-employed",
    "self-employed, incorporated": "as self-employed in an incorporated business",
    "self-employed, not incorporated": "as self-employed in an unincorporated business",
    "wage or salary worker": "as a wage and salary worker",
    "unpaid family worker": "without pay in a family business or farm",
    "armed forces": "in the armed forces",
}
UNION_SENTENCE = {
    "no union coverage": "You are not covered by a labor union.",
    "member of a labor union": "You are a labor-union member.",
    "covered by a union but not a member": "You are covered by a labor union but are not a member.",
}


CATEGORICAL_DOMAINS = {
    "sex": {"man", "woman"},
    "race": {
        "White",
        "Black",
        "American Indian",
        "Asian or Pacific Islander",
        "other race",
        "multiracial",
    },
    "educ": {
        "high school or less",
        "high school diploma or GED",
        "some college or associate degree",
        "bachelor's degree",
        "graduate or professional education",
    },
    "marst": {
        "married, spouse present",
        "married, spouse absent",
        "separated",
        "divorced",
        "widowed",
        "never married",
        "widowed or divorced",
    },
    "region": {"Northeast", "Midwest", "South", "West"},
    "metro": {"metropolitan area", "non-metropolitan area"},
    "hispan": {"not Hispanic", "Mexican origin", "Puerto Rican", "Cuban", "Dominican", "other Hispanic"},
    "nativity": {"native-born", "foreign-born"},
    "citizen": {"U.S.-born citizen", "naturalized citizen", "not a U.S. citizen"},
    "vetstat": {"not a veteran", "veteran"},
    "baseline_status": {"employed"},
    "classwkr": {
        "private for-profit wage and salary worker",
        "private wage and salary worker",
        "private nonprofit wage and salary worker",
        "state government employee",
        "federal government employee",
        "local government employee",
        "government wage and salary worker",
        "self-employed",
        "self-employed, incorporated",
        "self-employed, not incorporated",
        "wage or salary worker",
        "unpaid family worker",
        "armed forces",
    },
    "multjob_baseline": {"yes", "no"},
    "union_baseline": {"no union coverage", "member of a labor union", "covered by a union but not a member"},
    "difficulty": {"No difficulty", "Has difficulty", NOT_OBSERVED},
    "partner_linked_baseline": {"linked", "no linked spouse or partner"},
    "partner_status_baseline": {
        "employed",
        "unemployed",
        "retired",
        "disabled",
        "other not in the labor force",
        "not observed",
        "no linked spouse or partner",
    },
}


def value_is_missing(value: Any) -> bool:
    if value is None or pd.isna(value):
        return True
    return str(value).strip().lower() in MISSING_STRINGS


def format_number(value: Any) -> str:
    number = float(value)
    if not math.isfinite(number):
        return NOT_OBSERVED
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def display_value(row: dict[str, Any], column: str, value_format: str) -> str:
    value = row[column]
    if column == "partner_age_baseline" and row["partner_linked_baseline"] == "no linked spouse or partner":
        return "not applicable (no linked spouse or partner)"
    if value_is_missing(value):
        return NOT_OBSERVED
    if value_format == "integer":
        return str(int(round(float(value))))
    if value_format in {"number", "partner_age"}:
        return format_number(value)
    if value_format == "dollar":
        return format_dollars(value, decimals=2)
    if value_format == "percent":
        return f"{float(value):.2f}%"
    return str(value).strip()


def background_paragraph(row: dict[str, Any]) -> str:
    age = int(round(float(row["age"])))
    famsize = int(round(float(row["famsize"])))
    nchild = int(round(float(row["nchild"])))
    marst = str(row["marst"]).strip()
    state = str(row["state"]).strip()
    region = str(row["region"]).strip()
    metro = str(row["metro"]).strip()
    marital_clause = MARITAL_BACKGROUND[marst]
    if "," in marital_clause:
        location = f"You {marital_clause}, and you live in {state}, in a {metro} in the {region}."
    else:
        location = f"You {marital_clause} and live in {state}, in a {metro} in the {region}."
    if (
        str(row["hispan"]).strip() == "not Hispanic"
        and str(row["nativity"]).strip() == "native-born"
        and str(row["citizen"]).strip() == "U.S.-born citizen"
        and str(row["vetstat"]).strip() == "not a veteran"
    ):
        origin = "You are not Hispanic, were born in the United States, are a U.S. citizen, and are not a veteran."
    else:
        origin = " ".join(
            [
                HISPAN_SENTENCE[str(row["hispan"]).strip()],
                NATIVITY_CITIZENSHIP_SENTENCE[(str(row["nativity"]).strip(), str(row["citizen"]).strip())],
                VETERAN_SENTENCE[str(row["vetstat"]).strip()],
            ]
        )
    base = (
        "Here is some background information about yourself and your household. "
        f"You are {age_article(age)} {age}-year-old {str(row['race']).strip()} {str(row['sex']).strip()} "
        f"with {EDUC_PHRASE[str(row['educ']).strip()]}. "
        f"{location} {household_sentence(famsize, nchild)} {origin}"
    )
    return base + " " + partner_sentence(row)


def work_paragraph(row: dict[str, Any]) -> str:
    hours_last_week = format_number(row["ahrsworkt_baseline"])
    usual_hours = format_number(row["uhrsworkt_baseline"])
    classwkr = str(row["classwkr"]).strip()
    multjob = str(row["multjob_baseline"]).strip()
    sentences = [
        f"Here are your current work, resources, and health. {current_month_note(row['time'], currency=True)} You are currently employed {CURRENT_EMPLOYER_PHRASE[classwkr]}."
    ]
    multiple_jobs_clause = "do not hold multiple jobs" if multjob == "no" else "hold multiple jobs"
    sentences.append(
        f"You worked {hours_last_week} hours last week, usually work {usual_hours} hours per week, "
        f"and {multiple_jobs_clause}."
    )

    union = display_value(row, "union_baseline", "text")
    earnings = display_value(row, "earnweek_baseline", "dollar")
    if union == NOT_OBSERVED and earnings == NOT_OBSERVED:
        sentences.append("Your union coverage and weekly earnings are not observed.")
    else:
        sentences.append(
            "Your union coverage is not observed."
            if union == NOT_OBSERVED
            else UNION_SENTENCE[union]
        )
        sentences.append(
            "Your weekly earnings are not observed."
            if earnings == NOT_OBSERVED
            else f"Your weekly earnings are {earnings}."
        )

    difficulty = display_value(row, "difficulty", "text")
    difficulty_sentence = {
        "No difficulty": "You do not have a functional difficulty.",
        "Has difficulty": "You have a functional difficulty.",
        NOT_OBSERVED: "Whether you have a functional difficulty is not observed.",
    }[difficulty]
    sentences.append(difficulty_sentence)

    family_income = display_value(row, "family_income", "text")
    sentences.append(
        "Your family-income category is not observed."
        if family_income == NOT_OBSERVED
        else f"Your family-income category is {family_income}."
    )

    occupation = display_value(row, "occupation", "text")
    industry = display_value(row, "industry", "text")
    occupation_text = "not observed" if occupation == NOT_OBSERVED else occupation
    industry_text = "not observed" if industry == NOT_OBSERVED else industry
    sentences.append(f"Your occupation is {occupation_text}, and your industry is {industry_text}.")
    return " ".join(sentences)


def partner_sentence(row: dict[str, Any]) -> str:
    if row["partner_linked_baseline"] == "no linked spouse or partner":
        return "You do not live with a spouse or partner."
    age = int(round(float(row["partner_age_baseline"])))
    status = str(row["partner_status_baseline"]).strip()
    if status == NOT_OBSERVED:
        detail = f"They are {age_article(age)} {age}-year-old whose labor-force status is not observed."
    else:
        status_phrase = {
            "employed": "employed",
            "unemployed": "unemployed",
            "retired": "retired",
            "disabled": "disabled",
            "other not in the labor force": "otherwise not in the labor force",
        }[status]
        detail = f"They are {age_article(age)} {age}-year-old who is {status_phrase}."
    return "You live with a spouse or partner. " + detail


def render_prompt_line(row: dict[str, object]) -> str:
    """Render and check one table row's prompt record as a JSON line."""
    row_id = str(row["id"]).strip()
    target = str(row["target"])
    if target not in TARGETS:
        raise RuntimeError(f"Invalid target {target!r} for row_id={row_id}")

    user_prompt = "\n\n".join(
        [
            background_paragraph(row),
            work_paragraph(row),
            render_core_macro_paragraph(row),
                        QUESTION + " " + TARGET_INSTRUCTION,
            SCHEMA_INSTRUCTION,
        ]
    )

    record = {
        "id": str(row["id"]).strip(),
        "time": str(row["time"]),
        "release_date": str(row["release_date"]).strip()[:10],
        "system": SYSTEM_PROMPT,
        "user": user_prompt,
        "assistant": json.dumps([target], ensure_ascii=True),
        "naive_baseline": json.dumps(["not_retired"], ensure_ascii=True),
    }

    validate_prompt_record(record)
    return json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n"


def main() -> None:
    table = read_public_table(TABLE_PATH, task_id="labor_cps_retire")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing_columns:
        raise RuntimeError(f"{TABLE_PATH} is missing required columns: {missing_columns}")

    for column in IDENTIFIER_COLUMNS:
        missing = table[column].isna() | table[column].astype("string").str.strip().str.lower().isin(MISSING_STRINGS)
        if bool(missing.any()):
            examples = table.loc[missing, "id"].head(5).astype(str).tolist()
            raise RuntimeError(f"{column} has missing values in {TASK_SLUG}; example row_id values: {examples}")

    if set(table["target"].unique()) != TARGETS:
        raise RuntimeError(
            f"{TABLE_PATH} target must contain exactly 'retired' and 'not_retired'."
        )

    for column, allowed in CATEGORICAL_DOMAINS.items():
        values = table[column].dropna().astype(str).str.strip()
        values = set(values.loc[~values.str.lower().isin(MISSING_STRINGS)])
        unknown = sorted(values - allowed)
        if unknown:
            raise RuntimeError(f"{column} has unmapped prompt values in {TASK_SLUG}: {unknown}")

    nativity_citizenship = set(
        zip(table["nativity"].astype(str).str.strip(), table["citizen"].astype(str).str.strip(), strict=True)
    )
    unknown_nativity_citizenship = sorted(nativity_citizenship - set(NATIVITY_CITIZENSHIP_SENTENCE))
    if unknown_nativity_citizenship:
        raise RuntimeError(
            f"nativity/citizen has unmapped prompt values in {TASK_SLUG}: {unknown_nativity_citizenship}"
        )

    linked = table["partner_linked_baseline"].eq("linked")
    no_partner = table["partner_linked_baseline"].eq("no linked spouse or partner")
    if bool((linked & table["partner_age_baseline"].isna()).any()):
        raise RuntimeError("A linked spouse or partner has no baseline age.")
    if bool((no_partner & table["partner_age_baseline"].notna()).any()):
        raise RuntimeError("A row without a linked spouse or partner has a partner age.")
    if bool((no_partner & ~table["partner_status_baseline"].eq("no linked spouse or partner")).any()):
        raise RuntimeError("Partner status is inconsistent with no linked spouse or partner.")
    if bool((linked & table["partner_status_baseline"].eq("no linked spouse or partner")).any()):
        raise RuntimeError("A linked spouse or partner has the no-partner status.")

    render_table = apply_prompt_render_limit(table, task_id=TASK_SLUG)
    PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROMPT_PATH.open("w", encoding="utf-8") as handle:
        write_prompt_lines(handle, render_table.to_dict("records"), render_prompt_line)

    print(f"Wrote {PROMPT_PATH} with {len(render_table)} prompt records.")


if __name__ == "__main__":
    main()
