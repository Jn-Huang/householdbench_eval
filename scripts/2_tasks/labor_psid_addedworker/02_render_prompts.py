#!/usr/bin/env python
"""Render main-suite-style prompts for ``labor_psid_addedworker``."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.macro_context import PUBLIC_CORE_MACRO_COLUMNS, render_core_macro_paragraph
from scripts.utils.table_schema import read_public_table
from scripts.utils.prompt_rendering import (
    format_dollars,
    validate_model_facing_text,
    validate_prompt_record,
)
from scripts.utils.responses import (
    validate_numeric_array,
    validate_prompt_response_structure,
)

TASK_ID = "labor_psid_addedworker"
TABLE_PATH = PROJECT_ROOT / "data/householdbench/tabular/labor_psid_addedworker.csv"
PROMPTS_PATH = PROJECT_ROOT / "data/householdbench/prompts/labor_psid_addedworker.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "output/householdbench/tasks/labor_psid_addedworker"
REGISTRY_PATH = OUTPUT_DIR / "02_model_facing_predictor_registry.csv"
GOLDEN_KEYS = {
    "male_displaced": ("psid_household_shock_psid_pair_10021_10171_1976", "1975-12-31"),
    "reference_displaced": ("psid_household_shock_psid_pair_10021_10171_1976", "1975-12-31"),
    "you_worked": ("psid_household_shock_psid_pair_10021_10171_1976", "1975-12-31"),
    "you_did_not_work": ("psid_household_shock_psid_pair_1014001_1014002_1976", "1975-12-31"),
    "prior_displacement": ("psid_household_shock_psid_pair_1148001_1148002_1976", "1975-12-31"),
    "female_displaced": ("psid_household_shock_psid_pair_1154001_1154170_1976", "1975-12-31"),
    "partner_displaced": ("psid_household_shock_psid_pair_1154001_1154170_1976", "1975-12-31"),
}
SYSTEM_PROMPT = "You are an American adult making decisions about work, income, and household finances."
MIXED_YEAR_NOTE = (
    "The first two outcomes below cover the entire current calendar year. They can therefore include "
    "work before as well as after your spouse or partner lost their job. The last two outcomes cover "
    "one year from now."
)
QUESTION = (
    "Given this information, what are your annual work hours and labor earnings in the current "
    "calendar year, and what will they be in one year?"
)
RESPONSE_INSTRUCTION = (
    "Return only a valid JSON array with exactly four numbers, in this order: your annual "
    "work hours in the current calendar year; your annual labor earnings in the current "
    "calendar year; your annual work hours in one year; your annual labor earnings in one "
    "year. Earnings must be in current U.S. dollars. Your output must follow this exact array"
    " structure: [v_1, v_2, v_3, v_4]. Replace every v placeholder with a single number. Do "
    "not include dollar signs, commas, keys, or explanatory text."
)
DOLLAR_NOTE = "Note that $ denotes amounts in U.S. dollars."
NUMBER_WORDS = {
    1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
    6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten",
}


def has_value(value: object) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip() not in {"", "nan"}


def integer(value: object) -> int:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{TASK_ID}: visible integer is nonfinite")
    return int(round(number))


def years_ago(value: object, *, lower: bool = False) -> str:
    number = integer(value)
    word = NUMBER_WORDS.get(number, str(number))
    phrase = f"{word} {'year' if number == 1 else 'years'} ago"
    return phrase.lower() if lower else phrase


def age_article(age: int) -> str:
    return "an" if str(age).startswith(("8", "11", "18")) else "a"


def person_phrase(age: object, sex: object) -> str:
    age_value = integer(age)
    sex_value = str(sex).strip().lower()
    if sex_value == "male":
        sex_value = "man"
    elif sex_value == "female":
        sex_value = "woman"
    return f"{age_article(age_value)} {age_value}-year-old {sex_value}"


def children_phrase(value: object) -> str:
    children = integer(value)
    if children == 0:
        return "no children"
    if children == 1:
        return "including 1 child"
    return f"including {children} children"


def education_sentence(subject: str, value: object) -> str | None:
    if not has_value(value):
        return None
    return f"{subject} completed {integer(value)} years of schooling."


def tenure_sentence(value: object) -> str | None:
    if not has_value(value):
        return None
    label = str(value).strip()
    if label == "Owns home":
        return "You currently own your home."
    if label == "Pays rent":
        return "You currently rent your home."
    return f"Your current housing arrangement is {label.lower()}."


def work_sentence(subject: str, hours: object, earnings: object) -> str:
    hours_value = integer(hours)
    possessive = "your" if subject == "you" else "your spouse or partner's"
    if hours_value == 0:
        return f"{subject.capitalize()} did not work and {possessive} annual labor earnings were {format_dollars(earnings)}."
    return (
        f"{subject.capitalize()} worked {hours_value:,} hours in total and had "
        f"{format_dollars(earnings)} in annual labor earnings."
    )


def reason_clause(reason: str) -> str:
    if reason == "plant_closure_or_employer_move":
        return "because their employer closed, changed hands, moved, or ceased operating"
    if reason == "layoff_or_firing":
        return "because of a layoff or firing"
    raise RuntimeError(f"{TASK_ID}: unapproved displacement reason {reason!r}")


def split_values(value: object) -> list[str]:
    if not has_value(value):
        return []
    return [item for item in str(value).split("|") if item]


def background_paragraph(row: dict[str, object]) -> str:
    parts = [
        f"You are {person_phrase(row['current_you_demo_age_gen'], row['current_you_demo_sex_label'])}.",
        f"Your spouse or partner is {person_phrase(row['current_partner_demo_age_gen'], row['current_partner_demo_sex_label'])}.",
    ]
    family_size = integer(row["current_fam_size"])
    parts.append(
        f"Your household has {family_size} {'person' if family_size == 1 else 'people'}, "
        f"{children_phrase(row['current_fam_size_chi'])}."
    )
    for sentence in [
        education_sentence("You", row.get("current_you_edu_year")),
        education_sentence("Your spouse or partner", row.get("current_partner_edu_year")),
        tenure_sentence(row.get("current_home_stat_label")),
    ]:
        if sentence:
            parts.append(sentence)
    if has_value(row.get("current_geo_region_label")):
        parts.append(f"You live in the {str(row['current_geo_region_label']).strip()}.")
    return "Here is some background information about yourself and your household. " + " ".join(parts)


def history_sentence(row: dict[str, object], lag: int) -> str:
    prefix = f"history_lag{lag}_"
    parts = [
        work_sentence(
            "you",
            row[f"{prefix}nondisplaced_partner_annual_hours"],
            row[f"{prefix}nondisplaced_partner_earn_tot_nd"],
        ),
        work_sentence(
            "your spouse or partner",
            row[f"{prefix}displaced_annual_hours"],
            row[f"{prefix}displaced_earn_tot_nd"],
        ),
        f"Your total family income was {format_dollars(row[f'{prefix}finc_tot_nd'])}.",
    ]
    body = " ".join(parts)
    return f"{years_ago(row[f'{prefix}elapsed_years_to_event'])}, {body[0].lower()}{body[1:]}"


def history_paragraph(row: dict[str, object]) -> str:
    sentences = [history_sentence(row, lag) for lag in range(integer(row["history_depth"]), 0, -1)]
    return f"Here is your household's work and income history. {DOLLAR_NOTE} " + " ".join(sentences)


def event_paragraph(row: dict[str, object]) -> str:
    sentences = [
        f"During the current calendar year, your spouse or partner lost their job {reason_clause(str(row['displacement_reason']))}."
    ]
    years = split_values(row.get("prior_displacement_years_ago"))
    reasons = split_values(row.get("prior_displacement_reasons"))
    if len(years) != len(reasons):
        raise RuntimeError(f"{TASK_ID}: prior-displacement timing and reason counts differ")
    for elapsed, reason in zip(years, reasons, strict=True):
        sentences.append(
            f"Your spouse or partner also lost a job {years_ago(elapsed, lower=True)} {reason_clause(reason)}."
        )
    return " ".join(sentences)


def compact_prompt(row: dict[str, object]) -> str:
    return "\n\n".join([
        background_paragraph(row),
        history_paragraph(row),
        render_core_macro_paragraph(row),
        event_paragraph(row),
        MIXED_YEAR_NOTE,
        QUESTION,
        RESPONSE_INSTRUCTION,
    ])


def registry_rows() -> list[dict[str, object]]:
    specs: list[tuple[str, str, str, str]] = []
    current = [
        ("current_you_demo_age_gen", "Age", "numeric"),
        ("current_you_demo_sex_label", "Sex", "categorical"),
        ("current_you_edu_year", "Years of schooling", "numeric"),
        ("current_partner_demo_age_gen", "Spouse or partner age", "numeric"),
        ("current_partner_demo_sex_label", "Spouse or partner sex", "categorical"),
        ("current_partner_edu_year", "Spouse or partner years of schooling", "numeric"),
        ("current_fam_size", "Household size", "numeric"),
        ("current_fam_size_chi", "Children in household", "numeric"),
        ("current_home_stat_label", "Housing status", "categorical"),
        ("current_geo_region_label", "Region", "categorical"),
    ]
    specs.extend(("demographics_and_household", column, label, kind) for column, label, kind in current)
    for lag in range(1, 4):
        for suffix, label, kind in [
            ("elapsed_years_to_event", f"History {lag} timing", "numeric"),
            ("nondisplaced_partner_annual_hours", f"Your annual work hours, history {lag}", "numeric"),
            ("nondisplaced_partner_earn_tot_nd", f"Your annual labor earnings, history {lag}", "numeric"),
            ("displaced_annual_hours", f"Spouse or partner annual work hours, history {lag}", "numeric"),
            ("displaced_earn_tot_nd", f"Spouse or partner annual labor earnings, history {lag}", "numeric"),
            ("finc_tot_nd", f"Total family income, history {lag}", "numeric"),
        ]:
            specs.append(("work_and_income_history", f"history_lag{lag}_{suffix}", label, kind))
    for column, label in [
        ("displacement_reason", "Current job-loss reason"),
        ("prior_displacement_years_ago", "Prior job-loss timing"),
        ("prior_displacement_reasons", "Prior job-loss reasons"),
    ]:
        specs.append(("job_loss_context", column, label, "categorical"))
    for column in PUBLIC_CORE_MACRO_COLUMNS:
        specs.append(("core_macro_context", column, column.replace("_", " ").title(), "numeric"))
    groups = ["demographics_and_household", "work_and_income_history", "job_loss_context", "core_macro_context"]
    count = len(specs)
    return [
        {
            "task_id": TASK_ID,
            "registry_version": "labor_psid_addedworker_model_facing_v1",
            "render_scope": "authoritative_corpus",
            "predictor_count": count,
            "predictor_group": group,
            "predictor": column,
            "source_columns": column,
            "rendered_sections": "background" if group == groups[0] else "history" if group == groups[1] else "event" if group == groups[2] else "macro",
            "group_order": groups.index(group) + 1,
            "predictor_order": index,
            "display_label": label,
            "descriptive_type": kind,
            "predictor_scope": "model_facing",
        }
        for index, (group, column, label, kind) in enumerate(specs, start=1)
    ]


def main() -> None:
    if not TABLE_PATH.is_file():
        raise RuntimeError(f"{TASK_ID}: public task table is missing")
    table = read_public_table(TABLE_PATH, task_id="labor_psid_addedworker")
    if table.empty:
        raise RuntimeError(f"{TASK_ID}: tabular input is empty")
    if table[["id", "time"]].isna().any().any() or table.duplicated(["id", "time"]).any():
        raise RuntimeError(f"{TASK_ID}: invalid public observation keys")
    elapsed_columns = [f"history_lag{lag}_elapsed_years_to_event" for lag in range(1, 4)]
    table["history_depth"] = table[elapsed_columns].notna().sum(axis=1)
    if not table["history_depth"].between(1, 3).all():
        raise RuntimeError(f"{TASK_ID}: every public row must expose one to three history observations")
    table["naive_partner_annual_hours_event_year"] = table["history_lag1_nondisplaced_partner_annual_hours"]
    table["naive_partner_annual_earnings_event_year"] = table["history_lag1_nondisplaced_partner_earn_tot_nd"]
    table["naive_partner_annual_hours_following_year"] = table["history_lag1_nondisplaced_partner_annual_hours"]
    table["naive_partner_annual_earnings_following_year"] = table["history_lag1_nondisplaced_partner_earn_tot_nd"]
    PROMPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = registry_rows()
    pd.DataFrame(registry).to_csv(REGISTRY_PATH, index=False)
    records = []
    examples = []
    example_indices = {}
    for branch, (unit_id, time) in GOLDEN_KEYS.items():
        matches = table.index[table["id"].eq(unit_id) & table["time"].eq(time)]
        if len(matches) != 1:
            raise RuntimeError(f"{TASK_ID}: golden observation for {branch} is absent or duplicated")
        example_indices[branch] = int(matches[0])
    for index, series in table.iterrows():
        row = series.to_dict()
        compact = compact_prompt(row)
        validate_model_facing_text(compact, field="user")
        validate_prompt_response_structure(TASK_ID, compact)
        target = [
            float(row["partner_annual_hours_event_year"]),
            float(row["partner_annual_earnings_event_year"]),
            float(row["partner_annual_hours_following_year"]),
            float(row["partner_annual_earnings_following_year"]),
        ]
        baseline = [
            float(row["naive_partner_annual_hours_event_year"]),
            float(row["naive_partner_annual_earnings_event_year"]),
            float(row["naive_partner_annual_hours_following_year"]),
            float(row["naive_partner_annual_earnings_following_year"]),
        ]
        validate_numeric_array(target, length=4, field="assistant")
        validate_numeric_array(baseline, length=4, field="naive_baseline")
        record = {
            "id": str(row["id"]),
            "time": str(row["time"]),
            "release_date": str(row["release_date"]),
            "system": str(SYSTEM_PROMPT),
            "user": str(compact),
            "assistant": json.dumps(target, separators=(",", ":")),
            "naive_baseline": json.dumps(baseline, separators=(",", ":")),
        }
        validate_prompt_record(record)
        records.append(record)
        for branch, example_index in example_indices.items():
            if index == example_index:
                examples.append({"branch": branch, "id": row["id"], "time": row["time"], "compact_prompt": compact})
    with PROMPTS_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (OUTPUT_DIR / "02_representative_prompt_pairs.jsonl").open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    pd.DataFrame([{
        "artifact": str(PROMPTS_PATH.relative_to(PROJECT_ROOT)),
        "sha256": hashlib.sha256(PROMPTS_PATH.read_bytes()).hexdigest(),
        "records": len(records),
    }]).to_csv(OUTPUT_DIR / "02_render_hashes.csv", index=False)
    print(f"{TASK_ID}: rendered {len(records):,} prompts across {len(example_indices)} branches")


if __name__ == "__main__":
    main()
