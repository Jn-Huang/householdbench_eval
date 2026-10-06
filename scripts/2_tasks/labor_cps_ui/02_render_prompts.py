#!/usr/bin/env python
"""Render CPS UI prompts as the jobfind task plus approved UI information."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.table_schema import read_public_table
from scripts.utils.prompt_rendering import (
    cps_household_sentence as household_sentence,
    current_month_note,
    age_article,
    validate_prompt_record,
)
from scripts.utils.macro_context import PUBLIC_CORE_MACRO_COLUMNS, render_core_macro_paragraph
from scripts.utils.responses import validate_prompt_response_structure

TASK_ID = "labor_cps_ui"
JOBFIND_CONTRACT_ID = "labor_cps_jobfind"
TABLE_PATH = PROJECT_ROOT / "data/householdbench/tabular/labor_cps_ui.csv"
PROMPT_PATH = PROJECT_ROOT / "data/householdbench/prompts/labor_cps_ui.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "output/householdbench/tasks/labor_cps_ui"
REPRESENTATIVE_PATH = OUTPUT_DIR / "02_representative_prompt_pairs.jsonl"
REGISTRY_PATH = OUTPUT_DIR / "02_model_facing_predictor_registry.csv"
SYSTEM_PROMPT = (
    "You are an American person answering questions about your work, job search, "
    "and labor-force situation."
)
TARGETS = {"employed", "unemployed", "not_in_labor_force"}
QUESTION = (
    "Next month, what will be your work status? "
    'For this question, "employed" means working for pay or profit or being temporarily absent from a job, '
    '"unemployed" means not employed but available for work and actively searching or on temporary layoff, '
    'and "not_in_labor_force" means neither employed nor unemployed.'
)
RESPONSE_INSTRUCTION = (
    'Return exactly one valid JSON array containing one string, using one of: "employed", '
    '"unemployed", "not_in_labor_force". Your output must follow this exact array structure: '
    '["s_1"]. Replace every s placeholder with one of the permitted strings. Do not include '
    "explanatory text."
)
POLICY_CAVEAT = (
    "This maximum does not establish that you personally qualified, received benefits continuously, "
    "or had that many weeks remaining."
)
EDUC_PHRASE = {
    "high school or less": "high school education or less",
    "high school diploma or GED": "a high school diploma or GED",
    "some college or associate degree": "some college or an associate degree",
    "bachelor's degree": "a bachelor's degree",
    "graduate or professional education": "graduate or professional education",
}
RACE_SENTENCE = {
    "White": "You are White.", "Black": "You are Black.",
    "American Indian": "You are American Indian.",
    "Asian or Pacific Islander": "You are Asian or Pacific Islander.",
    "other race": "You identify with another race.", "multiracial": "You are multiracial.",
}
HISPAN_SENTENCE = {
    "not Hispanic": "You are not Hispanic.", "Mexican origin": "You are of Mexican origin.",
    "Puerto Rican": "You are Puerto Rican.", "Cuban": "You are Cuban.",
    "Dominican": "You are Dominican.",
    "other Hispanic": "You are Hispanic, with another Hispanic origin.",
}
NATIVITY_CITIZENSHIP_SENTENCE = {
    ("native-born", "U.S.-born citizen"): "You were born in the United States and are a U.S. citizen.",
    ("foreign-born", "U.S.-born citizen"): "You were born outside the United States and are a U.S. citizen.",
    ("foreign-born", "naturalized citizen"): "You were born outside the United States and are a naturalized U.S. citizen.",
    ("foreign-born", "not a U.S. citizen"): "You were born outside the United States and are not a U.S. citizen.",
}
VETERAN_SENTENCE = {"not a veteran": "You are not a veteran.", "veteran": "You are a veteran."}
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
MARITAL_BACKGROUND = {
    "married, spouse present": "are married, your spouse is present",
    "married, spouse absent": "are married, your spouse is absent",
    "separated": "are separated", "divorced": "are divorced", "widowed": "are widowed",
    "never married": "have never been married", "widowed or divorced": "are widowed or divorced",
}
JOB_LOSS_LABELS = {
    "you were on temporary layoff from a job": "On temporary layoff from a job",
    "you lost a job for another reason": "Lost a job for another reason",
    "a temporary job ended": "A temporary job ended",
}
SEARCH_LABELS = {
    "temporary_layoff": "On temporary layoff", "active_search": "Actively looking for work",
}


def background_paragraph(row: dict[str, object], age: int, famsize: int, nchild: int) -> str:
    sex = str(row["sex"]).strip()
    race = str(row["race"]).strip()
    educ = str(row["educ"]).strip()
    marital_clause = MARITAL_BACKGROUND[str(row["marst"]).strip()]
    comma = "," if "," in marital_clause else ""
    location_sentence = (
        f"You {marital_clause}{comma} and live in {str(row['state']).strip()}, in a "
        f"{str(row['metro']).strip()} in the {str(row['region']).strip()}."
    )
    if (
        str(row["hispan"]).strip() == "not Hispanic"
        and str(row["nativity"]).strip() == "native-born"
        and str(row["citizen"]).strip() == "U.S.-born citizen"
        and str(row["vetstat"]).strip() == "not a veteran"
    ):
        origin_sentence = "You are not Hispanic, were born in the United States, are a U.S. citizen, and are not a veteran."
    else:
        origin_sentence = " ".join([
            HISPAN_SENTENCE[str(row["hispan"]).strip()],
            NATIVITY_CITIZENSHIP_SENTENCE[(str(row["nativity"]).strip(), str(row["citizen"]).strip())],
            VETERAN_SENTENCE[str(row["vetstat"]).strip()],
        ])
    return (
        "Here is some background information about yourself and your household. "
        f"You are {age_article(age)} {age}-year-old {race} {sex} with {EDUC_PHRASE[educ]}. "
        f"{location_sentence} {household_sentence(famsize, nchild)} {origin_sentence}"
    )


def integer(value: Any) -> int:
    return int(round(float(value)))


def weeks(value: Any) -> str:
    number = float(value)
    rounded = round(number)
    amount = str(int(rounded)) if abs(number - rounded) < 1e-6 else f"{number:.1f}"
    return f"{amount} {'week' if amount == '1' else 'weeks'}"


def assert_supported_rows(rows: list[dict[str, object]]) -> None:
    contracts = {
        "educ": EDUC_PHRASE, "race": RACE_SENTENCE, "marst": MARITAL_BACKGROUND,
        "hispan": HISPAN_SENTENCE, "vetstat": VETERAN_SENTENCE, "wkstat_baseline": WKSTAT_SENTENCE,
        "search_layoff_status": SEARCH_LABELS, "whyunemp_text": JOB_LOSS_LABELS,
    }
    for column, mapping in contracts.items():
        unsupported = sorted({str(row[column]).strip() for row in rows} - set(mapping))
        if unsupported:
            raise RuntimeError(f"{TASK_ID}: unsupported {column} labels {unsupported}")
    pairs = {(str(row["nativity"]).strip(), str(row["citizen"]).strip()) for row in rows}
    if pairs - set(NATIVITY_CITIZENSHIP_SENTENCE):
        raise RuntimeError(f"{TASK_ID}: unsupported nativity/citizenship labels {sorted(pairs - set(NATIVITY_CITIZENSHIP_SENTENCE))}")
    for row in rows:
        if integer(row["last_full_time_work_rendered"]) == 1 and str(row["wnftlook_baseline"]).strip() not in WNFTLOOK_PHRASE:
            raise RuntimeError(f"{TASK_ID}: unsupported last-full-time-work label {row['wnftlook_baseline']!r}")
        if integer(row["classwkr_rendered"]) == 1 and str(row["classwkr"]).strip() not in CLASSWKR_SENTENCE:
            raise RuntimeError(f"{TASK_ID}: unsupported worker-class label {row['classwkr']!r}")
REQUIRED_COLUMNS = [
    "row_id", "subject_id", "baseline_year", "baseline_month", "release_date", "target", "age", "sex",
    "race", "educ", "marst", "state", "region", "metro", "famsize", "nchild", "hispan", "nativity",
    "citizen", "vetstat", "whyunemp_text", "search_layoff_status", "wkstat_baseline",
    "classwkr", "classwkr_rendered", "durunemp_baseline", "wnftlook_baseline",
    "last_full_time_work_rendered", "last_job_occupation", "last_job_industry",
    "regular_ui_weeks", "extension_ui_weeks", "maximum_ui_weeks", *PUBLIC_CORE_MACRO_COLUMNS,
]


def labor_paragraph(row: dict[str, object]) -> str:
    duration = integer(row["durunemp_baseline"])
    duration_phrase = "1 week" if duration == 1 else f"{duration} weeks"
    parts = [
        WKSTAT_SENTENCE[str(row["wkstat_baseline"]).strip()],
        f"You have been unemployed for {duration_phrase}.",
    ]
    if integer(row["last_full_time_work_rendered"]) == 1:
        value = WNFTLOOK_PHRASE[str(row["wnftlook_baseline"]).strip()]
        parts.append(f"The last time you worked full-time for at least 2 consecutive weeks was {value}.")
    if integer(row["classwkr_rendered"]) == 1:
        parts.append(CLASSWKR_SENTENCE[str(row["classwkr"]).strip()])
    return (
        f"Here is your current and recent labor-market situation. {current_month_note(row['time'], currency=False)} "
        + " ".join(parts)
    )


def additional_information_paragraph(row: dict[str, object]) -> str:
    search_sentence = (
        "You are currently on temporary layoff."
        if str(row["search_layoff_status"]) == "temporary_layoff"
        else "You are currently actively looking for work."
    )
    return (
        "Here is some additional information about your unemployment spell and most recent job. "
        f"Your unemployment began because {str(row['whyunemp_text']).strip()}. {search_sentence} "
        f"In your most recent job, your industry was {str(row['last_job_industry']).strip()}, and your "
        f"occupation was {str(row['last_job_occupation']).strip()}."
    )


def policy_paragraph(row: dict[str, object]) -> str:
    return (
        "Under the rules in your state in the current month, a job loser who met the program requirements "
        f"could potentially receive unemployment benefits for up to {weeks(row['maximum_ui_weeks'])}. "
        f"This total consists of {weeks(row['regular_ui_weeks'])} of regular state benefits and "
        f"{weeks(row['extension_ui_weeks'])} of temporary additional benefits. {POLICY_CAVEAT}"
    )


def render_user_prompt(row: dict[str, object]) -> str:
    compact = "\n\n".join([
        background_paragraph(row, integer(row["age"]), integer(row["famsize"]), integer(row["nchild"])),
        labor_paragraph(row),
        additional_information_paragraph(row),
        render_core_macro_paragraph(row),
        policy_paragraph(row),
        QUESTION,
        RESPONSE_INSTRUCTION,
    ])
    return compact


def write_predictor_registry() -> None:
    records: list[dict[str, object]] = []
    groups = [
        (1, "Household composition", [("famsize", "FAMSIZE", "Household size", "numeric"), ("nchild", "NCHILD", "Own children in household", "numeric")]),
        (2, "Demographics and location", [
            ("age", "AGE", "Age", "numeric"), ("sex", "SEX;sex", "Sex", "categorical"),
            ("race", "RACE;race", "Race", "categorical"), ("educ", "EDUC;educ", "Highest completed education", "categorical"),
            ("marst", "MARST;marst", "Marital status", "categorical"), ("state", "STATEFIP;state", "State", "categorical"),
            ("region", "REGION;region", "Region", "categorical"), ("metro", "METRO;metro", "Area type", "categorical"),
            ("hispan", "HISPAN;hispan", "Hispanic origin", "categorical"), ("nativity", "NATIVITY;nativity", "Nativity", "categorical"),
            ("citizen", "CITIZEN;citizen", "Citizenship", "categorical"), ("vetstat", "VETSTAT;vetstat", "Veteran status", "categorical"),
        ]),
        (3, "Employment and job search", [
            ("wkstat_baseline", "WKSTAT;wkstat", "Current unemployment status", "categorical"),
            ("durunemp_baseline", "DURUNEMP", "Unemployment duration", "numeric"),
            ("wnftlook_baseline", "WNFTLOOK;wnftlook", "Last full-time work", "categorical"),
            ("classwkr", "CLASSWKR;classwkr", "Most recent worker class", "categorical"),
            ("whyunemp_text", "WHYUNEMP", "Reason unemployment began", "categorical"),
            ("search_layoff_status", "WHYUNEMP", "Search or temporary-layoff status", "categorical"),
            ("last_job_occupation", "OCC", "Most recent occupation", "categorical"),
            ("last_job_industry", "IND", "Most recent industry", "categorical"),
        ]),
        (4, "Public macro context", [(name, name, name.replace("_", " ").capitalize(), "numeric") for name in PUBLIC_CORE_MACRO_COLUMNS]),
        (5, "UI policy", [
            ("regular_ui_weeks", "reg_UI", "Regular state benefits", "numeric"),
            ("extension_ui_weeks", "ui_weeks-reg_UI", "Temporary additional benefits", "numeric"),
            ("maximum_ui_weeks", "ui_weeks", "Maximum potential duration", "numeric"),
        ]),
    ]
    predictor_order = 0
    for group_order, group_name, predictors in groups:
        for predictor, source_columns, label, descriptive_type in predictors:
            predictor_order += 1
            records.append({
                "task_id": TASK_ID, "registry_version": "labor_cps_ui_model_facing_v1",
                "render_scope": "authoritative_corpus", "predictor_count": 45,
                "predictor_group": group_name, "predictor": predictor, "source_columns": source_columns,
                "rendered_sections": "user", "group_order": group_order,
                "predictor_order": predictor_order, "display_label": label,
                "descriptive_type": descriptive_type, "predictor_scope": "model_facing",
            })
    if len(records) != 45:
        raise RuntimeError(f"{TASK_ID}: predictor registry contains {len(records)} rows, expected 45.")
    pd.DataFrame(records).to_csv(REGISTRY_PATH, index=False)


def representative_row_ids(table: pd.DataFrame) -> list[str]:
    ordered = table.sort_values(["baseline_year", "baseline_month", "row_id"], kind="mergesort")
    cases = [
        ordered.loc[ordered["search_layoff_status"].eq("temporary_layoff") & ordered["extension_ui_weeks"].eq(0)],
        ordered.loc[ordered["search_layoff_status"].eq("active_search") & ordered["extension_ui_weeks"].gt(0)],
        ordered.loc[ordered["durunemp_baseline"].ge(52) & ordered["extension_ui_weeks"].gt(0)],
    ]
    chosen = [str(case.iloc[0]["row_id"]) for case in cases if not case.empty]
    chosen = list(dict.fromkeys(chosen))
    if len(chosen) != 3:
        raise RuntimeError(f"{TASK_ID}: representative cases do not cover the approved branches.")
    return chosen


if not TABLE_PATH.is_file():
    raise SystemExit(f"{TASK_ID}: public task table is missing")
table = read_public_table(TABLE_PATH, task_id="labor_cps_ui")
time_parts = table["time"].str.extract(r"^(\d{4})-(\d{2})-")
if time_parts.isna().any().any():
    raise RuntimeError(f"{TASK_ID}: public time values must use YYYY-MM-DD format")
table["row_id"] = table["id"]
table["subject_id"] = table["id"]
table["baseline_year"] = time_parts[0].astype(int)
table["baseline_month"] = time_parts[1].astype(int)
table["last_full_time_work_rendered"] = table["search_layoff_status"].eq("active_search").astype(int)
table["classwkr_rendered"] = table["classwkr"].notna().astype(int)
table["release_date"] = table["release_date"].astype("string")
missing = sorted(set(REQUIRED_COLUMNS) - set(table.columns))
if missing:
    raise RuntimeError(f"{TASK_ID}: missing table columns {missing}")
conditionally_missing = {"wnftlook_baseline", "classwkr"}
required_nonmissing = [column for column in REQUIRED_COLUMNS if column not in conditionally_missing]
if table[required_nonmissing].isna().any().any():
    bad = table.loc[table[required_nonmissing].isna().any(axis=1), "row_id"].head(5).tolist()
    raise RuntimeError(f"{TASK_ID}: prompt-required values are missing; row IDs {bad}")
if not set(table["target"]).issubset(TARGETS):
    raise RuntimeError(f"{TASK_ID}: target labels are outside the contract")
expected_wnftlook = table["search_layoff_status"].eq("active_search").astype(int)
if not table["last_full_time_work_rendered"].astype(int).eq(expected_wnftlook).all():
    raise RuntimeError(f"{TASK_ID}: last-full-time-work rendering does not follow the source-observation branch")
if not table["classwkr_rendered"].astype(int).eq(table["classwkr"].notna().astype(int)).all():
    raise RuntimeError(f"{TASK_ID}: worker-class rendering flag disagrees with source availability")
rows = table.to_dict("records")
assert_supported_rows(rows)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
write_predictor_registry()
representative_ids = set(representative_row_ids(table))
representatives = []
with PROMPT_PATH.open("w", encoding="utf-8") as handle:
    for row_number, row in enumerate(rows, start=1):
        compact = render_user_prompt(row)
        record = {
            "id": str(row["subject_id"]),
            "time": f"{int(row['baseline_year']):04d}-{int(row['baseline_month']):02d}-01",
            "release_date": str(row["release_date"]),
            "system": str(SYSTEM_PROMPT),
            "user": str(compact),
            "assistant": json.dumps([str(row["target"])], ensure_ascii=True),
            "naive_baseline": json.dumps(["unemployed"], ensure_ascii=True),
        }

        validate_prompt_record(record)
        validate_prompt_response_structure(TASK_ID, record["user"])
        handle.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
        if str(row["row_id"]) in representative_ids:
            representatives.append({"row_id": str(row["row_id"]), **record})
with REPRESENTATIVE_PATH.open("w", encoding="utf-8") as handle:
    for record in representatives:
        handle.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
if {record["row_id"] for record in representatives} != representative_ids:
    raise RuntimeError(f"{TASK_ID}: representative prompt export is incomplete")
print(f"{TASK_ID}: wrote {len(table):,} prompts, {len(representatives)} representative prompts, and the 45-predictor registry.")
