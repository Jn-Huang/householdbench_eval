#!/usr/bin/env python
"""Render CPS displaced-worker employment prompts from the task table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


TASK_SLUG = "labor_cps_displace"
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
)
from scripts.utils.table_schema import read_public_table, public_table_columns

PROMPT_PATH = resolve_prompt_output_path(PROMPT_PATH, task_id="labor_cps_displace")

from scripts.utils.macro_context import CORE_MACRO_COLUMNS, render_core_macro_paragraph


REQUIRED_COLUMNS = [column for column in [
    "id",
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
    "dwstat",
] if column in set(public_table_columns("labor_cps_displace"))]
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
LOOKBACK_PHRASE = {
    "five_year_lookback_1984_1992": "the past five years",
    "three_year_lookback_1994_plus": "the past three years",
}
DWREAS_SENTENCE = {
    "plant or company closed down or moved": "The lost job ended because the plant or company closed down or moved.",
    "insufficient work": "The lost job ended because there was insufficient work.",
    "position or shift abolished": "The lost job ended because the position or shift was abolished.",
    "seasonal job completed": "The lost job ended because a seasonal job was completed.",
    "self-operated business failed": "The lost job ended because a self-operated business failed.",
    "other reason": "The lost job ended for another reason.",
}
DWLASTWRK_SENTENCE = {
    "this year": "The last work at the lost job was this year.",
    "last year": "The last work at the lost job was last year.",
    "two years ago": "The last work at the lost job was two years ago.",
    "three years ago": "The last work at the lost job was three years ago.",
    "four years ago": "The last work at the lost job was four years ago.",
    "five years ago": "The last work at the lost job was five years ago.",
    "other timing": "The last work at the lost job was at another time.",
}
DWNOTICE_SENTENCE = {
    "no notice": "You received no notice before the job loss.",
    "less than 1 month notice": "You received less than 1 month of notice before the job loss.",
    "1 to 2 months notice": "You received 1 to 2 months of notice before the job loss.",
    "more than 2 months notice": "You received more than 2 months of notice before the job loss.",
    "notice given, time period not reported": "You received notice before the job loss, with the notice period unspecified.",
}
DWFULLTIME_SENTENCE = {
    "not full time": "The lost job was not full time.",
    "full time": "The lost job was full time.",
    "hours varied": "The lost job had variable hours.",
}
DWCLASS_SENTENCE = {
    "government": "The lost job was in government.",
    "private for-profit": "The lost job was in a private for-profit employer.",
    "private nonprofit": "The lost job was in a private nonprofit employer.",
    "self-employed": "The lost job was self-employment.",
    "without pay or family business": "The lost job was unpaid work in a family business or farm.",
}
YES_NO_SENTENCE = {
    "dwunion": {
        "yes": "You were a labor-union member at the lost job.",
        "no": "You were not a labor-union member at the lost job.",
    },
    "dwhi": {
        "yes": "The lost job provided health insurance.",
        "no": "The lost job did not provide health insurance.",
    },
}
FINAL_QUESTION = "What is your current employment status?"
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
LOST_JOB_CLASS_PHRASE = {
    "government": "at a government employer",
    "private for-profit": "at a private for-profit employer",
    "private nonprofit": "at a private nonprofit employer",
    "self-employed": "in self-employment",
    "without pay or family business": "in unpaid work in a family business or farm",
}
LOST_JOB_REASON_PHRASE = {
    "plant or company closed down or moved": "because the plant or company closed down or moved",
    "insufficient work": "because there was insufficient work",
    "position or shift abolished": "because the position or shift was abolished",
    "seasonal job completed": "because a seasonal job was completed",
    "self-operated business failed": "because a self-operated business failed",
    "other reason": "for another reason",
}
LOST_JOB_TIME_PHRASE = {
    "this year": "this year",
    "last year": "last year",
    "two years ago": "two years ago",
    "three years ago": "three years ago",
    "four years ago": "four years ago",
    "five years ago": "five years ago",
    "other timing": "at another time",
}
NOTICE_PHRASE = {
    "no notice": "no notice",
    "less than 1 month notice": "less than 1 month of notice",
    "1 to 2 months notice": "1 to 2 months of notice",
    "more than 2 months notice": "more than 2 months of notice",
    "notice given, time period not reported": "notice, with the notice period unspecified",
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


def job_loss_paragraph(row: dict[str, object], lookback_phrase: str) -> str:
    fulltime = str(row.get("dwfulltime", "")).strip()
    fulltime_phrase = {
        "full time": "full-time",
        "not full time": "not full-time",
        "hours varied": "variable-hours",
    }.get(fulltime)
    job_phrase = "job" if fulltime_phrase is None else f"{fulltime_phrase} job"
    dwclass = str(row.get("dwclass", "")).strip()
    class_phrase = f" {LOST_JOB_CLASS_PHRASE[dwclass]}" if dwclass in LOST_JOB_CLASS_PHRASE else ""
    reason = str(row.get("dwreas", "")).strip()
    reason_phrase = f" {LOST_JOB_REASON_PHRASE[reason]}" if reason in LOST_JOB_REASON_PHRASE else ""
    first_sentence = f"{current_month_note(row['time'], currency=False)} During {lookback_phrase}, you lost or left a {job_phrase}{class_phrase}{reason_phrase}."

    second_parts = []
    last_work = str(row.get("dwlastwrk", "")).strip()
    if last_work in LOST_JOB_TIME_PHRASE:
        second_parts.append(f"You last worked at that job {LOST_JOB_TIME_PHRASE[last_work]}")
    notice = str(row.get("dwnotice", "")).strip()
    if notice in NOTICE_PHRASE:
        if second_parts:
            second_parts[-1] += f" and received {NOTICE_PHRASE[notice]} before the job loss"
        else:
            second_parts.append(f"You received {NOTICE_PHRASE[notice]} before the job loss")
    second_sentence = second_parts[0] + "." if second_parts else None

    union = str(row.get("dwunion", "")).strip()
    health = str(row.get("dwhi", "")).strip()
    final_clauses = []
    if union == "yes":
        final_clauses.append("You were a labor-union member at that job")
    elif union == "no":
        final_clauses.append("You were not a labor-union member at that job")
    if health == "yes":
        final_clauses.append("the job provided health insurance")
    elif health == "no":
        final_clauses.append("the job did not provide health insurance")
    final_sentence = None
    if len(final_clauses) == 1:
        final_sentence = final_clauses[0] + "."
    elif len(final_clauses) == 2:
        final_sentence = final_clauses[0] + ", and " + final_clauses[1] + "."

    return " ".join(part for part in [first_sentence, second_sentence, final_sentence] if part)

table = read_public_table(TABLE_PATH, task_id="labor_cps_displace")

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

for column, allowed in {
    "dw_lookback_regime": LOOKBACK_PHRASE,
    "dwreas": DWREAS_SENTENCE,
    "dwlastwrk": DWLASTWRK_SENTENCE,
    "dwnotice": DWNOTICE_SENTENCE,
    "dwfulltime": DWFULLTIME_SENTENCE,
    "dwclass": DWCLASS_SENTENCE,
}.items():
    if column in table.columns:
        values = table[column].dropna().astype(str).str.strip()
        values = set(values.loc[~values.str.lower().isin(MISSING_STRINGS)])
        unknown = sorted(values - set(allowed))
        if unknown:
            raise RuntimeError(f"{column} has unmapped prompt values in {TASK_SLUG}: {unknown}")

for column, allowed in YES_NO_SENTENCE.items():
    if column in table.columns:
        values = table[column].dropna().astype(str).str.strip()
        values = set(values.loc[~values.str.lower().isin(MISSING_STRINGS)])
        unknown = sorted(values - set(allowed))
        if unknown:
            raise RuntimeError(f"{column} has unmapped prompt values in {TASK_SLUG}: {unknown}")

table = apply_prompt_render_limit(table, task_id=TASK_SLUG)
PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
with PROMPT_PATH.open("w", encoding="utf-8") as handle:
    for row_number, row in enumerate(table.to_dict("records"), start=1):
        row_id = str(row["id"]).strip()
        target = str(row["target"]).strip()
        age = int(round(float(row["age"])))
        baseline_year = int(str(row["time"])[:4])
        famsize = int(round(float(row["famsize"])))
        nchild = int(round(float(row["nchild"])))

        sex = str(row["sex"]).strip()
        if sex not in {"man", "woman"}:
            raise RuntimeError(f"sex has unmapped prompt value {sex!r} for row_id={row_id}")
        race = str(row["race"]).strip()
        educ = str(row["educ"]).strip()
        marst = str(row["marst"]).strip()
        relate = str(row["relate"]).strip()
        hispan = str(row["hispan"]).strip()
        nativity = str(row["nativity"]).strip()
        citizen = str(row["citizen"]).strip()
        vetstat = str(row["vetstat"]).strip()

        lookback = row.get("dw_lookback_regime")
        if lookback is not None and not pd.isna(lookback):
            lookback = str(lookback).strip()
            lookback_phrase = LOOKBACK_PHRASE[lookback] if lookback.lower() not in MISSING_STRINGS else None
        else:
            lookback_phrase = None
        if lookback_phrase is None:
            lookback_phrase = "the past five years" if baseline_year <= 1992 else "the past three years"
        job_loss_text = job_loss_paragraph(row, lookback_phrase)

        user_parts = [
            background_paragraph(row, age, famsize, nchild),
            render_core_macro_paragraph(row),
            job_loss_text,
            FINAL_QUESTION,
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
            "naive_baseline": json.dumps(["employed"], ensure_ascii=True),
        }

        validate_prompt_record(record)
        handle.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")

print(f"Wrote {PROMPT_PATH} with {len(table)} prompt records.")
