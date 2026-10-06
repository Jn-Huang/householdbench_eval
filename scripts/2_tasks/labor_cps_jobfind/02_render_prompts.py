#!/usr/bin/env python
"""Render CPS job-finding prompts from the task table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


TASK_SLUG = "labor_cps_jobfind"
TARGETS = ["employed", "unemployed", "not_in_labor_force"]
SYSTEM_PROMPT = (
    "You are an American person answering questions about your work, job search, "
    "and labor-force situation."
)

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
    validate_prompt_record,
    write_prompt_lines,
)
from scripts.utils.table_schema import read_public_table, public_table_columns

PROMPT_PATH = resolve_prompt_output_path(PROMPT_PATH, task_id="labor_cps_jobfind")

from scripts.utils.macro_context import CORE_MACRO_COLUMNS, render_core_macro_paragraph


REQUIRED_COLUMNS = [column for column in [
    "id",
    "baseline_year",
    "baseline_month",
    "release_date",
    "target",
    "age",
    "sex",
    "race",
    "educ",
    "marst",
    "state",
    "region",
    "metro",
    "relate",
    "famsize",
    "nchild",
    "hispan",
    "nativity",
    "citizen",
    "vetstat",
    *CORE_MACRO_COLUMNS,
    "wkstat_baseline",
    "durunemp_baseline",
    "wnftlook_baseline",
] if column in set(public_table_columns("labor_cps_jobfind"))]
MISSING_STRINGS = {"", "nan", "none", "nat", "<na>"}

EDUC_PHRASE = {
    "high school or less": "high school education or less",
    "high school diploma or GED": "a high school diploma or GED",
    "some college or associate degree": "some college or an associate degree",
    "bachelor's degree": "a bachelor's degree",
    "graduate or professional education": "graduate or professional education",
}
RACE_SENTENCE = {
    "White": "You are White.",
    "Black": "You are Black.",
    "American Indian": "You are American Indian.",
    "Asian or Pacific Islander": "You are Asian or Pacific Islander.",
    "other race": "You identify with another race.",
    "multiracial": "You are multiracial.",
}
MARITAL_SENTENCE = {
    "married, spouse present": "You are married and your spouse is present.",
    "married, spouse absent": "You are married and your spouse is absent.",
    "separated": "You are separated.",
    "divorced": "You are divorced.",
    "widowed": "You are widowed.",
    "never married": "You have never been married.",
    "widowed or divorced": "You are widowed or divorced.",
}
RELATE_PHRASE = {
    "reference person": "household reference person",
    "spouse": "spouse",
    "child": "child",
    "parent": "parent",
    "sibling": "sibling",
    "grandchild": "grandchild",
    "other relative": "other relative",
    "partner": "partner",
    "housemate or roommate": "housemate or roommate",
    "foster child": "foster child",
    "other nonrelative": "other nonrelative",
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
CLASSWKR_SENTENCE = {
    "private for-profit wage and salary worker": "In your most recent job, you worked in the private sector.",
    "private wage and salary worker": "In your most recent job, you worked in the private sector.",
    "private nonprofit wage and salary worker": "In your most recent job, you worked for a private nonprofit employer.",
    "state government employee": "In your most recent job, you worked for state government.",
    "federal government employee": "In your most recent job, you worked for the federal government.",
    "local government employee": "In your most recent job, you worked for local government.",
    "government wage and salary worker": "In your most recent job, you worked for the government.",
    "self-employed": "In your most recent job, you were self-employed.",
    "self-employed, incorporated": "In your most recent job, you were self-employed in an incorporated business.",
    "self-employed, not incorporated": "In your most recent job, you were self-employed in an unincorporated business.",
    "wage or salary worker": "In your most recent job, you worked as a wage and salary worker.",
    "unpaid family worker": "In your most recent job, you worked without pay in a family business or farm.",
    "armed forces": "In your most recent job, you served in the armed forces.",
}
WKSTAT_SENTENCE = {
    "unemployed, seeking full-time work": "You are currently unemployed and are seeking full-time work.",
    "unemployed, seeking part-time work": "You are currently unemployed and are seeking part-time work.",
}
WNFTLOOK_PHRASE = {
    "less than 5 years ago": "less than 5 years ago",
    "within the last 12 months": "within the last 12 months",
    "one to five years ago": "one to five years ago",
    "more than 12 months ago": "more than 12 months ago",
    "more than 5 years ago": "more than 5 years ago",
    "never worked": "never",
    "never worked full-time for 2 or more weeks": "never",
    "never worked at all": "never",
}
TARGET_INSTRUCTION = (
    'For this question, "employed" means working for pay or profit or being temporarily '
    'absent from a job, "unemployed" means not employed but available for work and actively '
    'searching or on temporary layoff, and "not_in_labor_force" means neither employed nor '
    "unemployed."
)
SCHEMA_INSTRUCTION = (
    'Return exactly one valid JSON array containing one string, using one of: "employed", '
    '"unemployed", "not_in_labor_force". Your output must follow this exact array structure: '
    '["s_1"]. Replace every s placeholder with one of the permitted strings. Do not include '
    "explanatory text."
)


MARITAL_BACKGROUND = {
    "married, spouse present": "are married, your spouse is present",
    "married, spouse absent": "are married, your spouse is absent",
    "separated": "are separated",
    "divorced": "are divorced",
    "widowed": "are widowed",
    "never married": "have never been married",
    "widowed or divorced": "are widowed or divorced",
}


def background_paragraph(row: dict[str, object], age: int, famsize: int, nchild: int) -> str:
    sex = str(row["sex"]).strip()
    race = str(row["race"]).strip()
    educ = str(row["educ"]).strip()
    marst = str(row["marst"]).strip()
    state = str(row["state"]).strip()
    region = str(row["region"]).strip()
    metro = str(row["metro"]).strip()
    marital_clause = MARITAL_BACKGROUND[marst]
    if "," in marital_clause:
        location_sentence = f"You {marital_clause}, and live in {state}, in a {metro} in the {region}."
    else:
        location_sentence = f"You {marital_clause} and live in {state}, in a {metro} in the {region}."
    if (
        str(row["hispan"]).strip() == "not Hispanic"
        and str(row["nativity"]).strip() == "native-born"
        and str(row["citizen"]).strip() == "U.S.-born citizen"
        and str(row["vetstat"]).strip() == "not a veteran"
    ):
        origin_sentence = "You are not Hispanic, were born in the United States, are a U.S. citizen, and are not a veteran."
    else:
        origin_sentence = " ".join(
            [
                HISPAN_SENTENCE[str(row["hispan"]).strip()],
                NATIVITY_CITIZENSHIP_SENTENCE[(str(row["nativity"]).strip(), str(row["citizen"]).strip())],
                VETERAN_SENTENCE[str(row["vetstat"]).strip()],
            ]
        )
    return (
        "Here is some background information about yourself and your household. "
        f"You are {age_article(age)} {age}-year-old {race} {sex} with {EDUC_PHRASE[educ]}. "
        f"{location_sentence} {household_sentence(famsize, nchild)} {origin_sentence}"
    )

def render_prompt_line(row: dict[str, object]) -> str:
    """Render and check one table row's prompt record as a JSON line."""
    row_id = str(row["id"]).strip()
    target = str(row["target"]).strip()
    age = int(round(float(row["age"])))
    famsize = int(round(float(row["famsize"])))
    nchild = int(round(float(row["nchild"])))
    unemployment_duration = int(round(float(row["durunemp_baseline"])))

    sex = str(row["sex"]).strip()
    if sex not in {"man", "woman"}:
        raise RuntimeError(f"sex has unmapped prompt value {sex!r} for row_id={row_id}")
    wkstat = str(row["wkstat_baseline"]).strip()
    wnftlook = str(row["wnftlook_baseline"]).strip()

    classwkr_sentence = ""
    classwkr = row.get("classwkr")
    if classwkr is not None and not pd.isna(classwkr):
        classwkr = str(classwkr).strip()
        if classwkr and classwkr.lower() not in MISSING_STRINGS:
            classwkr_sentence = CLASSWKR_SENTENCE[classwkr]

    duration_phrase = "1 week" if unemployment_duration == 1 else f"{unemployment_duration} weeks"
    labor_parts = [
        WKSTAT_SENTENCE[wkstat],
        f"You have been unemployed for {duration_phrase}.",
        f"The last time you worked full-time for at least 2 consecutive weeks was {WNFTLOOK_PHRASE[wnftlook]}.",
        classwkr_sentence,
    ]
    user_parts = [
        background_paragraph(row, age, famsize, nchild),
        f"Here is your current and recent labor-market situation. {current_month_note(row['time'], currency=False)} "
        + " ".join(part for part in labor_parts if part),
        render_core_macro_paragraph(row),
        "Next month, what will be your work status? " + TARGET_INSTRUCTION,
        SCHEMA_INSTRUCTION,
    ]
    user_prompt = "\n\n".join(part.strip() for part in user_parts if part and part.strip())

    record = {
        "id": str(row["id"]).strip(),
        "time": str(row["time"]),
        "release_date": str(row["release_date"]).strip()[:10],
        "system": SYSTEM_PROMPT,
        "user": user_prompt,
        "assistant": json.dumps([target], ensure_ascii=True),
        "naive_baseline": json.dumps(["unemployed"], ensure_ascii=True),
    }

    validate_prompt_record(record)
    return json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n"


table = read_public_table(TABLE_PATH, task_id="labor_cps_jobfind")

missing_columns = [column for column in REQUIRED_COLUMNS if column not in table.columns]
if missing_columns:
    raise RuntimeError(f"{TABLE_PATH} is missing required columns: {missing_columns}")

for column in REQUIRED_COLUMNS:
    text = table[column].astype("string").str.strip().str.lower()
    missing = table[column].isna() | text.isin(MISSING_STRINGS)
    if bool(missing.any()):
        examples = table.loc[missing, "id"].head(5).astype(str).tolist()
        raise RuntimeError(f"{column} has missing values in {TASK_SLUG}; example row_id values: {examples}")

if not set(table["target"].astype(str)).issubset(TARGETS):
    bad_targets = sorted(set(table["target"].astype(str)) - set(TARGETS))
    raise RuntimeError(f"Unexpected target values in {TASK_SLUG}: {bad_targets}")

for column, allowed in {
    "educ": EDUC_PHRASE,
    "race": RACE_SENTENCE,
    "marst": MARITAL_SENTENCE,
    "relate": RELATE_PHRASE,
    "hispan": HISPAN_SENTENCE,
    "vetstat": VETERAN_SENTENCE,
    "wkstat_baseline": WKSTAT_SENTENCE,
    "wnftlook_baseline": WNFTLOOK_PHRASE,
}.items():
    values = set(table[column].astype(str).str.strip())
    unknown = sorted(values - set(allowed))
    if unknown:
        raise RuntimeError(f"{column} has unmapped prompt values in {TASK_SLUG}: {unknown}")

nativity_citizenship_values = set(
    zip(table["nativity"].astype(str).str.strip(), table["citizen"].astype(str).str.strip(), strict=True)
)
unknown_nativity_citizenship = sorted(nativity_citizenship_values - set(NATIVITY_CITIZENSHIP_SENTENCE))
if unknown_nativity_citizenship:
    raise RuntimeError(
        f"nativity/citizen has unmapped prompt values in {TASK_SLUG}: {unknown_nativity_citizenship}"
    )

if "classwkr" in table.columns:
    classwkr_values = table["classwkr"].dropna().astype(str).str.strip()
    classwkr_values = set(classwkr_values.loc[~classwkr_values.str.lower().isin(MISSING_STRINGS)])
    unknown_classwkr = sorted(classwkr_values - set(CLASSWKR_SENTENCE))
    if unknown_classwkr:
        raise RuntimeError(f"classwkr has unmapped prompt values in {TASK_SLUG}: {unknown_classwkr}")

table = apply_prompt_render_limit(table, task_id=TASK_SLUG)
PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
with PROMPT_PATH.open("w", encoding="utf-8") as handle:
    write_prompt_lines(handle, table.to_dict("records"), render_prompt_line)

print(f"Wrote {PROMPT_PATH} with {len(table)} prompt records.")
