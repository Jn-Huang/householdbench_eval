#!/usr/bin/env python
"""Render CPS displaced-worker current-weekly-earnings prompts."""

from __future__ import annotations

import json
import sys
from pathlib import Path


TASK_SLUG = "income_cps_displace"
SYSTEM_PROMPT = (
    "You are an American person answering questions about your income, work, "
    "and labor-force situation."
)
FINAL_QUESTION = "Approximately how much do you earn per week at your current job?"

REPO_ROOT = Path(__file__).resolve().parents[3]
TABLE_PATH = REPO_ROOT / f"data/householdbench/tabular/{TASK_SLUG}.csv"
PROMPT_PATH = REPO_ROOT / f"data/householdbench/prompts/{TASK_SLUG}.jsonl"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.prompt_rendering import (
    current_month_note,
    resolve_prompt_output_path,
    age_article,
    apply_prompt_render_limit,
    format_dollars,
    validate_prompt_record,
)
from scripts.utils.table_schema import read_public_table, public_table_columns

PROMPT_PATH = resolve_prompt_output_path(PROMPT_PATH, task_id="income_cps_displace")

from scripts.utils.macro_context import CORE_MACRO_COLUMNS, render_core_macro_paragraph


SCHEMA_INSTRUCTION = (
    "Return exactly one valid JSON array containing one number in current U.S. dollars per "
    "week. Your output must follow this exact array structure: [v_1]. Replace every v "
    "placeholder with a single number. Do not include a dollar sign, commas, keys, or "
    "explanatory text."
)


REQUIRED_COLUMNS = [column for column in [
    "id",
    "id",
    "baseline_year",
    "baseline_month",
    "release_date",
    "baseline_status",
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
    "dwyears",
    "dwweekl",
    "dwjobsince",
    "dwwksun",
] if column in set(public_table_columns("income_cps_displace"))]
MISSING_STRINGS = {"", "nan", "none", "nat", "<na>"}

EDUC_PHRASE = {
    "high school or less": "high school education or less",
    "high school diploma or GED": "a high school diploma or GED",
    "some college or associate degree": "some college or an associate degree",
    "bachelor's degree": "a bachelor's degree",
    "graduate or professional education": "graduate or professional education",
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


def household_sentence(famsize: int, nchild: int) -> str:
    if famsize == 1 and nchild == 0:
        return "You live alone and have no own children in your household."
    household = f"You live in a household of {famsize} {'person' if famsize == 1 else 'people'}"
    if nchild == 0:
        return household + " and have no own children in your household."
    if nchild == 1:
        return household + ", including 1 of your own children."
    return household + f", including {nchild} of your own children."


def background_paragraph(row: dict[str, object], age: int, famsize: int, nchild: int) -> str:
    marital_clause = MARITAL_BACKGROUND[str(row["marst"]).strip()]
    location = (
        f"You {marital_clause}, and live in {row['state']}, in a {row['metro']} in the {row['region']}."
        if "," in marital_clause
        else f"You {marital_clause} and live in {row['state']}, in a {row['metro']} in the {row['region']}."
    )
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
    return (
        "Here is some background information about yourself and your household. "
        f"You are {age_article(age)} {age}-year-old {str(row['race']).strip()} {str(row['sex']).strip()} "
        f"with {EDUC_PHRASE[str(row['educ']).strip()]}. {location} "
        f"{household_sentence(famsize, nchild)} {origin}"
    )


def job_loss_paragraph(row: dict[str, object], lookback_phrase: str) -> str:
    fulltime = str(row.get("dwfulltime", "")).strip()
    fulltime_phrase = {
        "full time": "full-time",
        "not full time": "not full-time",
        "hours varied": "variable-hours",
    }.get(fulltime)
    job_phrase = "job" if fulltime_phrase is None else f"{fulltime_phrase} job"
    job_class = str(row.get("dwclass", "")).strip()
    class_phrase = f" {LOST_JOB_CLASS_PHRASE[job_class]}" if job_class in LOST_JOB_CLASS_PHRASE else ""
    reason = str(row.get("dwreas", "")).strip()
    reason_phrase = f" {LOST_JOB_REASON_PHRASE[reason]}" if reason in LOST_JOB_REASON_PHRASE else ""
    sentences = [
        f"{current_month_note(row['time'], currency=True)} During {lookback_phrase}, you lost or left a {job_phrase}{class_phrase}{reason_phrase}."
    ]

    last_work = str(row.get("dwlastwrk", "")).strip()
    notice = str(row.get("dwnotice", "")).strip()
    if last_work in LOST_JOB_TIME_PHRASE and notice in NOTICE_PHRASE:
        sentences.append(
            f"You last worked at that job {LOST_JOB_TIME_PHRASE[last_work]} and received {NOTICE_PHRASE[notice]} before the job loss."
        )
    elif last_work in LOST_JOB_TIME_PHRASE:
        sentences.append(f"You last worked at that job {LOST_JOB_TIME_PHRASE[last_work]}.")
    elif notice in NOTICE_PHRASE:
        sentences.append(f"You received {NOTICE_PHRASE[notice]} before the job loss.")

    union = str(row.get("dwunion", "")).strip()
    health = str(row.get("dwhi", "")).strip()
    clauses = []
    if union in {"yes", "no"}:
        clauses.append(f"You were {'a' if union == 'yes' else 'not a'} labor-union member at that job")
    if health in {"yes", "no"}:
        clauses.append(f"the job {'provided' if health == 'yes' else 'did not provide'} health insurance")
    if len(clauses) == 1:
        sentences.append(clauses[0] + ".")
    elif len(clauses) == 2:
        sentences.append(clauses[0] + ", and " + clauses[1] + ".")

    tenure = float(row["dwyears"])
    tenure_text = f"{tenure:.2f}".rstrip("0").rstrip(".")
    tenure_unit = "year" if abs(tenure - 1.0) < 1e-9 else "years"
    sentences.append(
        f"You worked at the lost job for about {tenure_text} {tenure_unit} "
        f"and earned approximately {format_dollars(row['dwweekl'], decimals=2)} per week."
    )
    return " ".join(sentences)


table = read_public_table(TABLE_PATH, task_id="income_cps_displace")

missing_columns = [column for column in REQUIRED_COLUMNS if column not in table.columns]
if missing_columns:
    raise RuntimeError(f"{TABLE_PATH} is missing required columns: {missing_columns}")
for column in REQUIRED_COLUMNS:
    text = table[column].astype("string").str.strip().str.lower()
    missing = table[column].isna() | text.isin(MISSING_STRINGS)
    if missing.any():
        examples = table.loc[missing, "id"].head(5).astype(str).tolist()
        raise RuntimeError(f"{column} has missing values in {TASK_SLUG}; example row_id values: {examples}")
if not table["baseline_status"].astype(str).eq("employed").all():
    raise RuntimeError(f"{TASK_SLUG} contains a respondent who is not currently employed.")
table["dwweekl_display"] = table["dwweekl"].map(
    lambda value: format_dollars(value, decimals=2)
)
for column, allowed in {
    "educ": EDUC_PHRASE,
    "marst": MARITAL_BACKGROUND,
    "hispan": HISPAN_SENTENCE,
    "vetstat": VETERAN_SENTENCE,
    "dw_lookback_regime": LOOKBACK_PHRASE,
}.items():
    unknown = sorted(set(table[column].astype(str).str.strip()) - set(allowed))
    if unknown:
        raise RuntimeError(f"{column} has unmapped prompt values in {TASK_SLUG}: {unknown}")
unknown_origin = sorted(
    set(zip(table["nativity"].astype(str).str.strip(), table["citizen"].astype(str).str.strip(), strict=True))
    - set(NATIVITY_CITIZENSHIP_SENTENCE)
)
if unknown_origin:
    raise RuntimeError(f"nativity/citizen has unmapped values in {TASK_SLUG}: {unknown_origin}")

table = apply_prompt_render_limit(table, task_id=TASK_SLUG)
PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
with PROMPT_PATH.open("w", encoding="utf-8") as handle:
    for row_number, row in enumerate(table.to_dict("records"), start=1):
        row_id = str(row["id"]).strip()
        age = int(round(float(row["age"])))
        famsize = int(round(float(row["famsize"])))
        nchild = int(round(float(row["nchild"])))
        target = float(row["target"])
        lookback = str(row["dw_lookback_regime"]).strip()
        job_loss_text = job_loss_paragraph(row, LOOKBACK_PHRASE[lookback])
        jobs = int(round(float(row["dwjobsince"])))
        weeks = int(round(float(row["dwwksun"])))
        after_loss_text = (
            f"Since losing or leaving that job, you have held {jobs} {'job' if jobs == 1 else 'jobs'}. "
            f"You went {weeks} {'week' if weeks == 1 else 'weeks'} without work between the end of the lost job "
            "and the start of your next job. You are currently employed."
        )
        user_prompt = "\n\n".join(
            [
                background_paragraph(row, age, famsize, nchild),
                render_core_macro_paragraph(row),
                job_loss_text,
                after_loss_text,
                        FINAL_QUESTION,
                SCHEMA_INSTRUCTION,
            ]
        )
        record = {
            "id": str(row["id"]).strip(),
            "time": str(row["time"]),
            "release_date": str(row["release_date"]).strip()[:10],
            "system": SYSTEM_PROMPT,
            "user": user_prompt,
            "assistant": json.dumps([target], ensure_ascii=True, allow_nan=False),
            "naive_baseline": json.dumps(
                [float(row["dwweekl"])], ensure_ascii=True, allow_nan=False
            ),
        }

        validate_prompt_record(record)
        handle.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")

print(f"Wrote {PROMPT_PATH} with {len(table)} prompt records.")
