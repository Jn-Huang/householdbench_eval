#!/usr/bin/env python
"""Render JSONL prompts for the PSID wealth task from its table."""

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

from scripts.utils.macro_context import (
    CORE_MACRO_COLUMNS,
    PSID_ASSET_COLUMNS,
    render_core_macro_paragraph,
    render_psid_asset_paragraph,
)


TASK_NAME = "cons_psid_wealth"
SYSTEM_PROMPT = (
    "You are the head of an American household making decisions about income, "
    "saving, housing, and household wealth."
)
TABLE_PATH = PROJECT_ROOT / "data" / "householdbench" / "tabular" / f"{TASK_NAME}.csv"
PROMPTS_PATH = resolve_prompt_output_path(PROJECT_ROOT / "data" / "householdbench" / "prompts" / f"{TASK_NAME}.jsonl", task_id="cons_psid_wealth")

MACRO_LABELS = {
    "house_price_growth": "house price growth",
    "sp500_growth": "S&P 500 growth",
    "mortgage30us": "30-year mortgage rate",
    "fedfunds": "federal funds rate",
    "unrate": "unemployment rate",
    "inflation": "inflation",
}
FINAL_QUESTION = (
    "Given this information, in two years, what will your household's total net worth be, "
    "in current U.S. dollars?"
)
RESPONSE_INSTRUCTION = (
    "Return only a valid JSON array containing exactly one integer, with no explanatory text."
    " Your output must follow this exact array structure: [v_1]. Replace every v placeholder "
    "with a single number."
)
DOLLAR_NOTE = "Note that $ denotes amounts in U.S. dollars."
PROMPT_INTRO = "Here is some background information about yourself and your household."
REQUIRED_COLUMNS = [column for column in [
    "id",
    "year",
    "release_date",
    "demo_age_gen",
    "demo_sex",
    "geo_region",
    "wlth_tot_net_nd_next_wave",
    *[f"history_lag{lag}_years_ago" for lag in range(1, 4)],
    *[
        f"history_lag{lag}_{column}"
        for lag in range(1, 4)
        for column in [
            "finc_tot_nd",
            "wlth_tot_net_nd",
            "wlth_savi_net_nd",
            "wlth_inve_net_nd",
            "wlth_home_net_nd",
            "wlth_odeb_net_nd",
            "home_stat",
        ]
    ],
    *CORE_MACRO_COLUMNS,
    *PSID_ASSET_COLUMNS,
] if column in set(public_table_columns("cons_psid_wealth"))]

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
HOME_PHRASES = {
    "Owns home": "own your home",
    "Pays rent": "rent your home",
    "Neither owns home nor pays rent": "neither own nor rent your home",
}
HOME_HISTORY_PHRASES = {
    "Owns home": "owned its home",
    "Pays rent": "rented its home",
    "Neither owns home nor pays rent": "neither owned nor rented its home",
}
PARTNER_PHRASES = {
    "No, reference person has no spouse or partner in the family unit": "not partnered",
    "Yes, reference person has a spouse or partner in the family unit": "partnered",
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
    word = NUMBER_WORDS.get(years_ago, str(years_ago))
    return f"{word} {'year' if years_ago == 1 else 'years'} ago"


MARRIED_PHRASES = {
    "No, reference person is not legally married to a spouse in the family unit": "not married",
    "Yes, reference person is legally married to a spouse in the family unit": "married",
}


if not TABLE_PATH.exists():
    raise SystemExit(f"Task table not found: {TABLE_PATH}")

table = read_public_table(TABLE_PATH, task_id="cons_psid_wealth")

missing_columns = [column for column in REQUIRED_COLUMNS if column not in table.columns]
if missing_columns:
    raise SystemExit(f"{TASK_NAME}: table is missing required columns: {missing_columns}")

table = apply_prompt_render_limit(table, task_id=TASK_NAME)
PROMPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with PROMPTS_PATH.open("w", encoding="utf-8") as handle:
    for row_number, row in enumerate(table.to_dict(orient="records"), start=1):
        row_id = str(row["id"])
        current_year = int(str(row["time"])[:4])
        prompt_time = str(row["time"])
        target = int(round(float(row["wlth_tot_net_nd_next_wave"])))
        naive_baseline = int(round(float(row["wlth_tot_net_nd"])))

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

        current_income_sentence = None
        if has_value(row.get("finc_tot_nd")) and has_value(row.get("wlth_tot_net_nd")):
            current_income_sentence = (
                f"Your current household income is {dollar(row['finc_tot_nd'])}, "
                f"and your total net worth is {dollar(row['wlth_tot_net_nd'])}."
            )

        component_values = []
        for column, label in [
            ("wlth_savi_net_nd", "savings"),
            ("wlth_inve_net_nd", "financial investments"),
            ("wlth_home_net_nd", "home equity"),
            ("wlth_odeb_net_nd", "other debt"),
        ]:
            if has_value(row.get(column)):
                component_values.append(f"{dollar(row[column])} in {label}")
        components_sentence = None
        if component_values:
            if len(component_values) == 1:
                components_sentence = f"You have {component_values[0]}."
            else:
                components_sentence = "You have " + ", ".join(component_values[:-1]) + f", and {component_values[-1]}."

        housing_sentence = None
        home_phrase = optional_phrase_from_label(row, "home_stat", HOME_PHRASES, row_id)
        if home_phrase:
            housing_sentence = f"You currently {home_phrase}."

        family_values = []
        partnered_phrase = optional_phrase_from_label(row, "fam_partnered", PARTNER_PHRASES, row_id)
        if partnered_phrase:
            if partnered_phrase == "partnered":
                family_values.append("have a spouse or partner in your household")
            elif partnered_phrase == "not partnered":
                family_values.append("do not have a spouse or partner in your household")
        married_phrase = optional_phrase_from_label(row, "fam_married", MARRIED_PHRASES, row_id)
        if married_phrase:
            if married_phrase == "married":
                family_values.append("are legally married")
            elif married_phrase == "not married":
                family_values.append("are not legally married")
        family_sentence = None
        if family_values:
            family_sentence = "You " + " and ".join(family_values) + "."

        household_sentence = None
        if has_value(row.get("fam_size")):
            family_size = int(round(float(row["fam_size"])))
            household_sentence = f"Your household has {family_size} {'person' if family_size == 1 else 'people'}"
        if has_value(row.get("fam_size_chi")):
            children = int(round(float(row["fam_size_chi"])))
            child_phrase = "no children" if children == 0 else f"including {children} {'child' if children == 1 else 'children'}"
            if household_sentence:
                household_sentence += f", {child_phrase}."

        macro_sentence = render_core_macro_paragraph(row) + " " + render_psid_asset_paragraph(row)

        parts = [
            (
                f"{PROMPT_INTRO} "
                + " ".join(part for part in [demographic_sentence, geography_sentence, family_sentence, household_sentence] if part)
            ),
            (
                f"Here is your household's financial situation now and in past years. {DOLLAR_NOTE} "
                + " ".join(part for part in [current_income_sentence, components_sentence, housing_sentence] if part)
            ),
        ]

        history_sentences = []
        for lag in [3, 2, 1]:
            if not has_value(row.get(f"history_lag{lag}_years_ago")):
                continue
            history_values = []
            if has_value(row.get(f"history_lag{lag}_finc_tot_nd")):
                history_values.append(f"your household income was {dollar(row[f'history_lag{lag}_finc_tot_nd'])}")
            for column, label in [
                ("wlth_tot_net_nd", "your total net worth was"),
                ("wlth_savi_net_nd", "your savings were"),
                ("wlth_inve_net_nd", "your financial investments were"),
                ("wlth_home_net_nd", "your home equity was"),
                ("wlth_odeb_net_nd", "your other debt was"),
            ]:
                history_column = f"history_lag{lag}_{column}"
                if has_value(row.get(history_column)):
                    history_values.append(f"{label} {dollar(row[history_column])}")
            if has_value(row.get(f"history_lag{lag}_home_stat")):
                home_history = phrase_from_label(
                    row,
                    f"history_lag{lag}_home_stat",
                    HOME_HISTORY_PHRASES,
                    row_id,
                )
                history_values.append(f"your household {home_history}")
            if history_values:
                history_sentences.append(
                    f"{format_years_ago(row[f'history_lag{lag}_years_ago'])}, "
                    + (
                        history_values[0]
                        if len(history_values) == 1
                        else ", ".join(history_values[:-1]) + f", and {history_values[-1]}"
                    )
                    + "."
                )
        if history_sentences:
            parts[-1] += " " + " ".join(history_sentences)

        if macro_sentence:
            parts.append(macro_sentence)

        parts.append(FINAL_QUESTION)
        parts.append(RESPONSE_INSTRUCTION)
        user_prompt = "\n\n".join(part for part in parts if part)

        record = {
            "id": row_id,
            "time": prompt_time,
            "release_date": str(row["release_date"]),
            "system": SYSTEM_PROMPT,
            "user": user_prompt,
            "assistant": json.dumps([target], ensure_ascii=True),
            "naive_baseline": json.dumps([naive_baseline], ensure_ascii=True),
        }
        validate_prompt_record(record)
        handle.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")

print(f"Wrote {PROMPTS_PATH} from {TABLE_PATH}")
