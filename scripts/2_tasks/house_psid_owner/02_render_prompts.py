#!/usr/bin/env python
"""Render JSONL prompts for the PSID renter-to-owner task from its table."""

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
    HOUSING_SYSTEM_PROMPT,
    age_article,
    apply_prompt_render_limit,
    format_dollars,
    optional_phrase_from_label,
    phrase_from_label,
    validate_model_facing_text,
    validate_prompt_record,
)

from scripts.utils.macro_context import (
    CORE_MACRO_COLUMNS,
    PSID_ASSET_COLUMNS,
    render_core_macro_paragraph,
    render_psid_asset_paragraph,
)


TASK_NAME = "house_psid_owner"
SYSTEM_PROMPT = HOUSING_SYSTEM_PROMPT
TABLE_PATH = PROJECT_ROOT / "data" / "householdbench" / "tabular" / f"{TASK_NAME}.csv"
PROMPTS_PATH = resolve_prompt_output_path(PROJECT_ROOT / "data" / "householdbench" / "prompts" / f"{TASK_NAME}.jsonl", task_id="house_psid_owner")

RESPONSE_INSTRUCTION = (
    'Return only a valid JSON array containing exactly one string, using one of: "yes", "no".'
    ' Your output must follow this exact array structure: ["s_1"]. Replace every s '
    "placeholder with one of the permitted strings. Do not include explanatory text."
)
DOLLAR_NOTE = "Note that $ denotes amounts in U.S. dollars."
VALID_TARGETS = {"yes", "no"}
REQUIRED_COLUMNS = [column for column in [
    "id",
    "year",
    "next_year",
    "release_date",
    "demo_age_gen",
    "demo_sex",
    "race_eth_maj_col",
    "geo_region",
    "finc_tot_rd",
    "home_stat",
    *CORE_MACRO_COLUMNS,
    *PSID_ASSET_COLUMNS,
    "becomes_owner_next_wave",
] if column in set(public_table_columns("house_psid_owner"))]
OPTIONAL_PROMPT_COLUMNS = [
    "geo_metro",
    "wlth_tot_net_rd",
    "wlth_savi_net_rd",
    "wlth_inve_net_rd",
    "wlth_home_net_rd",
    "fam_partnered",
    "fam_married",
    "fam_size",
    "fam_size_chi",
    "lag1_family_size_change",
    "lag1_children_change",
    "lag1_partnered_change",
    "lag1_married_change",
] + [
    f"history_lag{lag}_{column}"
    for lag in range(1, 4)
    for column in [
        "years_ago",
        "home_stat",
        "finc_tot_rd",
        "wlth_tot_net_rd",
        "wlth_savi_net_rd",
        "wlth_inve_net_rd",
    ]
]

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
CURRENT_HOME_SENTENCES = {
    "Owns home": "You currently own your home",
    "Pays rent": "You currently rent your home",
    "Neither owns home nor pays rent": "You currently neither own nor rent your home",
}
HISTORY_HOME_PHRASES = {
    "Owns home": "owned its home",
    "Pays rent": "rented its home",
    "Neither owns home nor pays rent": "neither owned nor rented its home",
}
NUMBER_WORDS = {
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
    10: "Ten",
}


def has_value(value: object) -> bool:
    return value is not None and not pd.isna(value)


def dollar(value: object) -> str:
    return format_dollars(value)


def format_years_ago(years_ago: object) -> str:
    years_ago = int(round(float(years_ago)))
    if years_ago <= 0:
        raise RuntimeError(f"Historical elapsed years must be positive: {years_ago=}")
    word = NUMBER_WORDS.get(years_ago, str(years_ago))
    return f"{word} {'year' if years_ago == 1 else 'years'} ago"


def join_phrases(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return " and ".join(parts)
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def child_phrase(children: int) -> str:
    if children == 0:
        return "no children"
    if children == 1:
        return "1 child"
    return f"{children} children"


def change_clause(value: int, subject: str) -> str:
    if value == 0:
        return f"{subject} is unchanged"
    if value > 0:
        return f"{subject} has increased by {value}"
    return f"{subject} has fallen by {abs(value)}"


def signed_growth_phrase(label: str, value: float) -> str:
    if value < 0:
        return f"{label} fell by {abs(value):.2f}%"
    return f"{label} grew by {value:.2f}%"


if not TABLE_PATH.exists():
    raise SystemExit(f"Task table not found: {TABLE_PATH}")

table = read_public_table(TABLE_PATH, task_id="house_psid_owner")

missing_columns = [column for column in REQUIRED_COLUMNS + OPTIONAL_PROMPT_COLUMNS if column not in table.columns]
if missing_columns:
    raise SystemExit(f"{TASK_NAME}: table is missing required columns: {missing_columns}")

table = apply_prompt_render_limit(table, task_id=TASK_NAME)
PROMPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with PROMPTS_PATH.open("w", encoding="utf-8") as handle:
    for row in table.to_dict(orient="records"):
        row_id = str(row["id"])
        current_year = int(str(row["time"])[:4])
        prompt_time = str(row["time"])
        target = str(row["becomes_owner_next_wave"])
        if target not in VALID_TARGETS:
            raise RuntimeError(f"{TASK_NAME}: invalid owner target {target!r} for id={row_id}")
        wave_gap = int(round(float(row["forecast_horizon_years"])))
        if wave_gap == 1:
            horizon_phrase = "In one year"
        elif wave_gap == 2:
            horizon_phrase = "In two years"
        else:
            raise RuntimeError(f"{TASK_NAME}: unexpected wave gap {wave_gap} for id={row_id}")

        age = int(round(float(row["demo_age_gen"])))
        sex = phrase_from_label(row, "demo_sex", SEX_PHRASES, row_id)

        race_part = ""
        race_phrase = optional_phrase_from_label(row, "race_eth_maj_col", RACE_PHRASES, row_id)
        if race_phrase:
            race_part = f" {race_phrase}"
        demographic_sentence = f"You are {age_article(age)} {age}-year-old{race_part} {sex}"

        region = optional_phrase_from_label(row, "geo_region", REGION_PHRASES, row_id)
        geography_phrase = ""
        if region:
            geography_phrase = f" living in {region}"
        if region and row.get("geo_metro") is not None and not pd.isna(row.get("geo_metro")):
            metro = optional_phrase_from_label(row, "geo_metro", METRO_PHRASES, row_id)
            if metro:
                geography_phrase = f" living in {region}, in a {metro}"
        demographic_sentence = demographic_sentence + geography_phrase + "."

        partnered_raw = None
        if row.get("fam_partnered") is not None and not pd.isna(row.get("fam_partnered")):
            partnered_raw = str(row["fam_partnered"]).strip()
        married_raw = None
        if row.get("fam_married") is not None and not pd.isna(row.get("fam_married")):
            married_raw = str(row["fam_married"]).strip()
        family_status_sentence = None
        if partnered_raw == "Yes, reference person has a spouse or partner in the family unit" and married_raw == "Yes, reference person is legally married to a spouse in the family unit":
            family_status_sentence = "You have a spouse or partner in your household and are legally married."
        elif partnered_raw == "Yes, reference person has a spouse or partner in the family unit" and married_raw == "No, reference person is not legally married to a spouse in the family unit":
            family_status_sentence = "You have a spouse or partner in your household and are not legally married."
        elif partnered_raw == "Yes, reference person has a spouse or partner in the family unit":
            family_status_sentence = "You have a spouse or partner in your household."
        elif partnered_raw == "No, reference person has no spouse or partner in the family unit" and married_raw == "Yes, reference person is legally married to a spouse in the family unit":
            family_status_sentence = "You do not have a spouse or partner in your household and are legally married."
        elif partnered_raw == "No, reference person has no spouse or partner in the family unit" and married_raw == "No, reference person is not legally married to a spouse in the family unit":
            family_status_sentence = "You do not have a spouse or partner in your household and are not legally married."
        elif partnered_raw == "No, reference person has no spouse or partner in the family unit":
            family_status_sentence = "You do not have a spouse or partner in your household."

        family_size_sentence = None
        if row.get("fam_size") is not None and not pd.isna(row.get("fam_size")):
            family_size = int(round(float(row["fam_size"])))
            household_phrase = f"{family_size} {'person' if family_size == 1 else 'people'}"
            if row.get("fam_size_chi") is not None and not pd.isna(row.get("fam_size_chi")):
                children = int(round(float(row["fam_size_chi"])))
                family_size_sentence = f"Your household has {household_phrase} and {child_phrase(children)}."
            else:
                family_size_sentence = f"Your household has {household_phrase}."

        change_sentence = None
        change_clauses = []
        if row.get("lag1_family_size_change") is not None and not pd.isna(row.get("lag1_family_size_change")):
            change_clauses.append(change_clause(int(round(float(row["lag1_family_size_change"]))), "household size"))
        if row.get("lag1_children_change") is not None and not pd.isna(row.get("lag1_children_change")):
            change_clauses.append(change_clause(int(round(float(row["lag1_children_change"]))), "the number of children"))
        if row.get("lag1_partnered_change") is not None and not pd.isna(row.get("lag1_partnered_change")):
            change_clauses.append(change_clause(int(round(float(row["lag1_partnered_change"]))), "partner status"))
        if row.get("lag1_married_change") is not None and not pd.isna(row.get("lag1_married_change")):
            change_clauses.append(change_clause(int(round(float(row["lag1_married_change"]))), "marital status"))
        if change_clauses:
            if not has_value(row.get("history_lag1_years_ago")):
                raise RuntimeError(f"{TASK_NAME}: household changes lack a prior observation interval for id={row_id}")
            comparison_time = format_years_ago(row["history_lag1_years_ago"]).lower()
            change_sentence = f"Compared with {comparison_time}, {join_phrases(change_clauses)}."

        current_home_sentence = None
        if row.get("home_stat") is not None and not pd.isna(row.get("home_stat")):
            home_value = str(row["home_stat"]).strip()
            if home_value not in CURRENT_HOME_SENTENCES:
                raise RuntimeError(f"{TASK_NAME}: unmapped home status {home_value!r} for id={row_id}")
            current_home_sentence = CURRENT_HOME_SENTENCES[home_value]
        current_financial_sentences = []
        if has_value(row.get("finc_tot_rd")) and has_value(row.get("wlth_tot_net_rd")):
            current_financial_sentences.append(
                f"Your current household income is {dollar(row['finc_tot_rd'])}, "
                f"and your total net worth is {dollar(row['wlth_tot_net_rd'])}."
            )
        elif has_value(row.get("finc_tot_rd")):
            current_financial_sentences.append(f"Your current household income is {dollar(row['finc_tot_rd'])}.")
        elif has_value(row.get("wlth_tot_net_rd")):
            current_financial_sentences.append(f"Your total net worth is {dollar(row['wlth_tot_net_rd'])}.")

        component_values = []
        for column, label in [
            ("wlth_savi_net_rd", "savings"),
            ("wlth_inve_net_rd", "financial investments"),
            ("wlth_home_net_rd", "home equity"),
        ]:
            if has_value(row.get(column)):
                component_values.append(f"{dollar(row[column])} in {label}")
        if component_values:
            current_financial_sentences.append(f"You have {join_phrases(component_values)}.")

        current_finance_parts = []
        if current_home_sentence:
            current_finance_parts.append(current_home_sentence + ".")
        current_finance_parts.extend(current_financial_sentences)

        history_sentences = []
        for lag in [3, 2, 1]:
            years_ago_column = f"history_lag{lag}_years_ago"
            if not has_value(row.get(years_ago_column)):
                continue
            history_values = []
            home_column = f"history_lag{lag}_home_stat"
            if has_value(row.get(home_column)):
                home_value = str(row[home_column]).strip()
                if home_value not in HISTORY_HOME_PHRASES:
                    raise RuntimeError(f"{TASK_NAME}: unmapped lagged home status {home_value!r} for id={row_id}")
                history_values.append(f"your household {HISTORY_HOME_PHRASES[home_value]}")
            for column, phrase in [
                ("finc_tot_rd", "household income was"),
                ("wlth_tot_net_rd", "total net worth was"),
                ("wlth_savi_net_rd", "savings were"),
                ("wlth_inve_net_rd", "financial investments were"),
            ]:
                history_column = f"history_lag{lag}_{column}"
                if has_value(row.get(history_column)):
                    history_values.append(f"{phrase} {dollar(row[history_column])}")
            if history_values:
                history_sentences.append(
                    f"{format_years_ago(row[years_ago_column])}, {join_phrases(history_values)}."
                )

        macro_sentence = render_core_macro_paragraph(row) + " " + render_psid_asset_paragraph(row)

        parts = [
            demographic_sentence,
            family_status_sentence,
            family_size_sentence,
            change_sentence,
        ]
        background_paragraph = (
            "Here is some background information about yourself and your household. "
            + " ".join(part for part in parts if part)
        )

        housing_finance_paragraph_parts = [*current_finance_parts, *history_sentences]
        housing_finance_paragraph = (
            f"Here are your household's housing, finances, and past household records. {DOLLAR_NOTE} "
            + " ".join(part for part in housing_finance_paragraph_parts if part)
        )
        user_prompt_parts = [background_paragraph, housing_finance_paragraph]
        if macro_sentence:
            user_prompt_parts.append(macro_sentence)
        user_prompt_parts.append(f"{horizon_phrase}, will your household own its home?")
        user_prompt_parts.append(RESPONSE_INSTRUCTION)
        user_prompt = "\n\n".join(part for part in user_prompt_parts if part)
        validate_model_facing_text(user_prompt, field="user")

        record = {
            "id": row_id,
            "time": prompt_time,
            "release_date": str(row["release_date"]),
            "system": SYSTEM_PROMPT,
            "user": user_prompt,
            "assistant": json.dumps([target], ensure_ascii=True),
            "naive_baseline": json.dumps(["no"], ensure_ascii=True),
        }
        validate_prompt_record(record)
        handle.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")

print(f"Wrote {PROMPTS_PATH} from {TABLE_PATH}")
