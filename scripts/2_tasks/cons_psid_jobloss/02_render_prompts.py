#!/usr/bin/env python
"""Render main-suite-style prompts for ``cons_psid_jobloss``."""

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

TASK_ID = "cons_psid_jobloss"
TABLE_PATH = PROJECT_ROOT / "data/householdbench/tabular/cons_psid_jobloss.csv"
PROMPTS_PATH = PROJECT_ROOT / "data/householdbench/prompts/cons_psid_jobloss.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "output/householdbench/tasks/cons_psid_jobloss"
REGISTRY_PATH = OUTPUT_DIR / "02_model_facing_predictor_registry.csv"
GOLDEN_KEYS = {
    "unpartnered_control": ("psid_control_psid_single_1070001_1971", "1970-12-31"),
    "partnered_one": ("psid_household_shock_psid_pair_1055001_1055002_1971", "1970-12-31"),
    "prior_displacement": ("psid_household_shock_psid_pair_2729001_2729002_1971", "1970-12-31"),
    "unpartnered": ("psid_household_shock_psid_single_2066001_1971", "1970-12-31"),
    "partnered_control": ("psid_control_psid_pair_10001_10002_1976", "1975-12-31"),
    "partnered_two": ("psid_household_shock_psid_pair_2410001_2410002_1976", "1975-12-31"),
}
SYSTEM_PROMPT = (
    "You are the head of an American household making decisions about food spending, "
    "work, income, and household finances."
)
QUESTION = (
    "Given this information, in one year, what will your household's annual food spending be, "
    "including food at home and food away from home?"
)
RESPONSE_INSTRUCTION = (
    "Return only a valid JSON array with exactly one number in current U.S. dollars. Your "
    "output must follow this exact array structure: [v_1]. Replace v_1 with one number. Do "
    "not include a dollar sign, commas, keys, or explanatory text."
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


def person_phrase(age: object, sex: object, race: object | None = None) -> str:
    age_value = integer(age)
    sex_value = str(sex).strip().lower()
    if sex_value == "male":
        sex_value = "man"
    elif sex_value == "female":
        sex_value = "woman"
    race_part = f" {str(race).strip()}" if has_value(race) else ""
    return f"{age_article(age_value)} {age_value}-year-old{race_part} {sex_value}"


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
    years = integer(value)
    return f"{subject} completed {years} years of schooling."


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
    possessive = "Your" if subject == "You" else f"{subject}'s"
    if hours_value == 0:
        return f"{subject} did not work and {possessive.lower()} annual labor earnings were {format_dollars(earnings)}."
    return (
        f"{subject} worked {hours_value:,} hours in total and "
        f"had {format_dollars(earnings)} in annual labor earnings."
    )


def reason_clause(reason: str, *, partner: bool) -> str:
    if reason == "plant_closure_or_employer_move":
        return "because their employer closed, changed hands, moved, or ceased operating" if partner else "because your employer closed, changed hands, moved, or ceased operating"
    if reason == "layoff_or_firing":
        return "because of a layoff or firing"
    raise RuntimeError(f"{TASK_ID}: unapproved displacement reason {reason!r}")


def split_values(value: object) -> list[str]:
    if not has_value(value):
        return []
    return [item for item in str(value).split("|") if item]


def prior_displacement_sentences(row: dict[str, object], role: str) -> list[str]:
    years = split_values(row.get(f"{role}_prior_displacement_years_ago"))
    reasons = split_values(row.get(f"{role}_prior_displacement_reasons"))
    if len(years) != len(reasons):
        raise RuntimeError(f"{TASK_ID}: prior-displacement timing and reason counts differ")
    partner = role == "partner"
    subject = "Your spouse or partner" if partner else "You"
    sentences = []
    for elapsed, reason in zip(years, reasons, strict=True):
        sentences.append(
            f"{subject} also lost a job {years_ago(elapsed, lower=True)} {reason_clause(reason, partner=partner)}."
        )
    return sentences


def background_paragraph(row: dict[str, object]) -> str:
    you = person_phrase(
        row["current_reference_demo_age_gen"],
        row["current_reference_demo_sex_label"],
        row.get("current_reference_race_eth_maj_col_label"),
    )
    parts = [f"You are {you}."]
    if has_value(row.get("current_reference_geo_region_label")):
        parts.append(f"You live in the {str(row['current_reference_geo_region_label']).strip()}.")
    if row["household_structure"] == "partnered":
        parts.append("You have a spouse or partner in your household.")
        partner = person_phrase(
            row["current_partner_demo_age_gen"],
            row["current_partner_demo_sex_label"],
        )
        parts.append(f"Your spouse or partner is {partner}.")
    else:
        parts.append("You do not have a spouse or partner in your household.")
    family_size = integer(row["current_reference_fam_size"])
    parts.append(
        f"Your household has {family_size} {'person' if family_size == 1 else 'people'}, "
        f"{children_phrase(row['current_reference_fam_size_chi'])}."
    )
    for sentence in [
        education_sentence("You", row.get("current_reference_edu_year")),
        education_sentence("Your spouse or partner", row.get("current_partner_edu_year"))
        if row["household_structure"] == "partnered" else None,
        tenure_sentence(row.get("current_reference_home_stat_label")),
    ]:
        if sentence:
            parts.append(sentence)
    return "Here is some background information about yourself and your household. " + " ".join(parts)


def history_sentence(row: dict[str, object], lag: int) -> str:
    prefix = f"history_lag{lag}_"
    parts = [
        work_sentence(
            "You",
            row[f"{prefix}reference_annual_hours"],
            row[f"{prefix}reference_earn_tot_nd"],
        )
    ]
    if row["household_structure"] == "partnered":
        parts.append(
            work_sentence(
                "Your spouse or partner",
                row[f"{prefix}partner_annual_hours"],
                row[f"{prefix}partner_earn_tot_nd"],
            )
        )
    parts.append(f"Your total family income was {format_dollars(row[f'{prefix}reference_finc_tot_nd'])}.")
    if has_value(row.get(f"{prefix}annual_food_spending")):
        parts.append(
            f"Your household's annual food spending was {format_dollars(row[f'{prefix}annual_food_spending'])}, "
            "including food at home and food away from home."
        )
    body = " ".join(parts)
    return f"{years_ago(row[f'{prefix}elapsed_years'])}, {body[0].lower()}{body[1:]}"


def history_paragraph(row: dict[str, object]) -> str:
    sentences = [history_sentence(row, lag) for lag in range(integer(row["history_depth"]), 0, -1)]
    return f"Here is your household's work, income, and food-spending history. {DOLLAR_NOTE} " + " ".join(sentences)


def event_paragraph(row: dict[str, object]) -> str:
    if row["observation_type"] == "control":
        if row["household_structure"] == "partnered":
            return "Neither you nor your spouse or partner lost your job during the current calendar year."
        return "You did not lose your job during the current calendar year."
    sentences = []
    reference_reasons = split_values(row.get("reference_current_displacement_reasons"))
    partner_reasons = split_values(row.get("partner_current_displacement_reasons"))
    for reason in reference_reasons:
        sentences.append(f"During the current calendar year, you lost your job {reason_clause(reason, partner=False)}.")
    for reason in partner_reasons:
        sentences.append(f"During the current calendar year, your spouse or partner lost their job {reason_clause(reason, partner=True)}.")
    sentences.extend(prior_displacement_sentences(row, "reference"))
    sentences.extend(prior_displacement_sentences(row, "partner"))
    if not sentences:
        raise RuntimeError(f"{TASK_ID}: event row produced no event sentence")
    return " ".join(sentences)


def compact_prompt(row: dict[str, object]) -> str:
    parts = [
        background_paragraph(row),
        history_paragraph(row),
        render_core_macro_paragraph(row),
        event_paragraph(row),
        QUESTION,
        RESPONSE_INSTRUCTION,
    ]
    return "\n\n".join(part for part in parts if part)


def registry_rows() -> list[dict[str, object]]:
    specs: list[tuple[str, str, str, str]] = []
    current = [
        ("current_reference_demo_age_gen", "Age", "numeric"),
        ("current_reference_demo_sex_label", "Sex", "categorical"),
        ("current_reference_race_eth_maj_col_label", "Race and ethnicity", "categorical"),
        ("current_reference_geo_region_label", "Region", "categorical"),
        ("current_reference_edu_year", "Years of schooling", "numeric"),
        ("household_structure", "Spouse or partner in household", "categorical"),
        ("current_partner_demo_age_gen", "Spouse or partner age", "numeric"),
        ("current_partner_demo_sex_label", "Spouse or partner sex", "categorical"),
        ("current_partner_edu_year", "Spouse or partner years of schooling", "numeric"),
        ("current_reference_fam_size", "Household size", "numeric"),
        ("current_reference_fam_size_chi", "Children in household", "numeric"),
        ("current_reference_home_stat_label", "Housing status", "categorical"),
    ]
    specs.extend(("demographics_and_household", column, label, kind) for column, label, kind in current)
    for lag in range(1, 4):
        for suffix, label, kind in [
            ("elapsed_years", f"History {lag} timing", "numeric"),
            ("reference_annual_hours", f"Your annual work hours, history {lag}", "numeric"),
            ("reference_earn_tot_nd", f"Your annual labor earnings, history {lag}", "numeric"),
            ("partner_annual_hours", f"Spouse or partner annual work hours, history {lag}", "numeric"),
            ("partner_earn_tot_nd", f"Spouse or partner annual labor earnings, history {lag}", "numeric"),
            ("reference_finc_tot_nd", f"Total family income, history {lag}", "numeric"),
            ("annual_food_spending", f"Annual food spending, history {lag}", "numeric"),
        ]:
            specs.append(("household_history", f"history_lag{lag}_{suffix}", label, kind))
    for column, label in [
        ("observation_type", "Job-loss observation type"),
        ("reference_current_displacement_reasons", "Your current job-loss reason"),
        ("partner_current_displacement_reasons", "Spouse or partner current job-loss reason"),
        ("reference_prior_displacement_years_ago", "Your prior job-loss timing"),
        ("reference_prior_displacement_reasons", "Your prior job-loss reasons"),
        ("partner_prior_displacement_years_ago", "Spouse or partner prior job-loss timing"),
        ("partner_prior_displacement_reasons", "Spouse or partner prior job-loss reasons"),
    ]:
        specs.append(("job_loss_context", column, label, "categorical"))
    for column in PUBLIC_CORE_MACRO_COLUMNS:
        specs.append(("core_macro_context", column, column.replace("_", " ").title(), "numeric"))
    groups = ["demographics_and_household", "household_history", "job_loss_context", "core_macro_context"]
    count = len(specs)
    return [
        {
            "task_id": TASK_ID,
            "registry_version": "cons_psid_jobloss_model_facing_v1",
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
    table = read_public_table(TABLE_PATH, task_id="cons_psid_jobloss")
    if table.empty:
        raise RuntimeError(f"{TASK_ID}: tabular input is empty")
    if table[["id", "time"]].isna().any().any() or table.duplicated(["id", "time"]).any():
        raise RuntimeError(f"{TASK_ID}: invalid public observation keys")
    elapsed_columns = [f"history_lag{lag}_elapsed_years" for lag in range(1, 4)]
    table["history_depth"] = table[elapsed_columns].notna().sum(axis=1)
    if not table["history_depth"].between(1, 3).all():
        raise RuntimeError(f"{TASK_ID}: every public row must expose one to three history observations")
    food_history = table[[f"history_lag{lag}_annual_food_spending" for lag in range(1, 4)]]
    table["naive_annual_food_spending"] = food_history.where(food_history.gt(0)).bfill(axis=1).iloc[:, 0]
    if table["naive_annual_food_spending"].isna().any():
        raise RuntimeError(f"{TASK_ID}: public history cannot construct the naive food-spending baseline")
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
        target = [float(row["annual_food_spending"])]
        baseline = [float(row["naive_annual_food_spending"])]
        validate_numeric_array(target, length=1, field="assistant")
        validate_numeric_array(baseline, length=1, field="naive_baseline")
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
