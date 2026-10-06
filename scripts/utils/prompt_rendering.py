"""Shared helpers for prompt rendering scripts."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.utils.responses import validate_prompt_response_structure
from scripts.utils.table_schema import load_table_schema


PROMPT_RENDER_LIMIT_ENV = "HOUSEHOLDBENCH_RENDER_PROMPT_LIMIT"
HOUSING_SYSTEM_PROMPT = (
    "You are an American adult making decisions about housing and residential location."
)


def age_article(age: int) -> str:
    if age in {8, 11, 18} or 80 <= age <= 89:
        return "an"
    return "a"


def format_dollars(value: float | int, *, decimals: int = 0) -> str:
    if decimals not in {0, 2}:
        raise ValueError("Dollar decimals must be 0 or 2.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Dollar value must be finite.")
    rounded = round(number, decimals)
    magnitude = f"{abs(rounded):,.{decimals}f}"
    return f"-${magnitude}" if rounded < 0 else f"${magnitude}"


def prompt_render_limit() -> int | None:
    raw = os.environ.get(PROMPT_RENDER_LIMIT_ENV)
    if raw is None or raw.strip() == "":
        return None
    try:
        limit = int(raw)
    except ValueError as exc:
        raise SystemExit(f"{PROMPT_RENDER_LIMIT_ENV} must be a positive integer if set.") from exc
    if limit <= 0:
        raise SystemExit(f"{PROMPT_RENDER_LIMIT_ENV} must be a positive integer if set.")
    return limit


def apply_prompt_render_limit(table: Any, *, task_id: str) -> Any:
    limit = prompt_render_limit()
    if limit is None:
        return table
    original_rows = len(table)
    if hasattr(table, "head"):
        limited = table.head(limit).copy()
    else:
        limited = table[:limit]
    print(
        f"{task_id}: {PROMPT_RENDER_LIMIT_ENV}={limit}; "
        f"rendering {len(limited):,} of {original_rows:,} rows."
    )
    return limited


PROMPT_REQUIRED_KEYS = (
    "id",
    "time",
    "release_date",
    "system",
    "user",
    "assistant",
    "naive_baseline",
)
PROMPT_ALLOWED_KEYS = PROMPT_REQUIRED_KEYS
MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
CURRENT_MONTH_NOTE_PATTERN = re.compile(
    r"(?:Note that the current month is (?:"
    + "|".join(MONTH_NAMES)
    + r")(?:, and that \$ denotes amounts in U\.S\. dollars)?\."
    + r"|Note that today is (?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday), "
    + r"(?:[1-9]|[12]\d|3[01])(?:st|nd|rd|th) (?:"
    + "|".join(MONTH_NAMES)
    + r"), and that \$ denotes amounts in U\.S\. dollars\.)"
)

FORBIDDEN_MODEL_TEXT_PATTERNS = [
    (
        re.compile(r"\b(?:survey|surveyed|interviewed respondent|respondent)\b", re.I),
        "survey/respondent reference",
    ),
    (re.compile(r"\b(?:panel|panelist|panel month)\b", re.I), "panel reference"),
    (re.compile(r"\bbenchmark\b", re.I), "benchmark reference"),
    (re.compile(r"\bParker-style\b", re.I), "task-construction jargon"),
    (re.compile(r"\b(?:CPS|SCE|CEX|PSID|Census)\b"), "dataset name"),
    (re.compile(r"\bMichigan Surveys? of Consumers\b", re.I), "survey name"),
    (re.compile(r"\bSurvey of Consumer Expectations\b", re.I), "survey name"),
    (re.compile(r"\bCurrent Population Survey\b", re.I), "survey name"),
    (re.compile(r"\bConsumer Expenditure(?:s)?(?: Survey)?\b", re.I), "survey name"),
    (re.compile(r"\bPanel Study of Income Dynamics\b", re.I), "survey name"),
    (
        re.compile(r"\b(?:Economic Stimulus Payment|federal income-tax rebate)\b", re.I),
        "named policy event",
    ),
    (re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b"), "absolute date"),
    (re.compile(r"\b(?:19|20)\d{2}\s*(?:through|to|-)\s*(?:19|20)?\d{2}\b"), "absolute year range"),
    (
        re.compile(
            r"\b(?:in|during|from|through|since|before|after|year|month)\s+(?:19|20)\d{2}\b", re.I
        ),
        "calendar year reference",
    ),
    (re.compile(r"\b(?:" + "|".join(MONTH_NAMES) + r")\b"), "month-name reference"),
    (re.compile(r"\bqoq\b", re.I), "abbreviated quarter-over-quarter wording"),
    (re.compile(r"\bQ-\d+\b"), "mechanical lag label"),
    (
        re.compile(r"\b(?:unknown|not available|not reported|missing)\b", re.I),
        "missing-value placeholder",
    ),
    (re.compile(r"\b(?:day|month|year|job)\(s\)\b", re.I), "parenthetical plural placeholder"),
    (re.compile(r"\b1 people\b", re.I), "incorrect singular person phrasing"),
    (re.compile(r"\b1 children\b", re.I), "incorrect singular child phrasing"),
    (re.compile(r"\byou say\b", re.I), "survey-like reporting language"),
    (re.compile(r"\breported reason\b", re.I), "survey-like reporting language"),
    (re.compile(r"\bretrospective report\b", re.I), "survey-like reporting language"),
    (re.compile(r"\bnumeracy classification\b", re.I), "mechanical predictor label"),
    (re.compile(r"\bwork-status indicators\b", re.I), "mechanical predictor label"),
]


def _lowercase_prefilter(pattern: re.Pattern[str]) -> re.Pattern[str] | None:
    """A quick test that matches lowercased ASCII text wherever `pattern` matches the text.

    The leading \\b and the case-insensitive flag stop the regex engine from scanning for a
    literal prefix, which makes each search slow. Without them, and with the literals
    lowercased, the pattern matches a superset of positions in the lowercased text, so a
    miss proves the original cannot match. Patterns with character classes or uppercase
    escapes, whose meaning lowercasing would change, get no prefilter.
    """
    source = pattern.pattern.removeprefix(r"\b")
    if not source.isascii() or "[" in source or re.search(r"\\[A-Z]", source):
        return None
    return re.compile(source.lower())


FORBIDDEN_TEXT_PREFILTERS = [
    _lowercase_prefilter(pattern) for pattern, _reason in FORBIDDEN_MODEL_TEXT_PATTERNS
]


def validate_model_facing_text(text: str, *, field: str) -> None:
    text_without_current_month_note = CURRENT_MONTH_NOTE_PATTERN.sub("", text)
    ascii_text = text.isascii()
    if ascii_text:
        lowered = text.lower()
        lowered_without_current_month_note = text_without_current_month_note.lower()
    for (pattern, reason), prefilter in zip(
        FORBIDDEN_MODEL_TEXT_PATTERNS, FORBIDDEN_TEXT_PREFILTERS, strict=True
    ):
        month_names = reason == "month-name reference"
        search_text = text_without_current_month_note if month_names else text
        if ascii_text and prefilter is not None:
            lowered_text = lowered_without_current_month_note if month_names else lowered
            if prefilter.search(lowered_text) is None:
                continue
        match = pattern.search(search_text)
        if match:
            snippet = search_text[max(0, match.start() - 50) : match.end() + 50]
            raise ValueError(
                f"Forbidden {reason} in {field} prompt: {match.group(0)!r}. "
                f"Context: {snippet!r}"
            )


_RENDER_JOB: dict[str, Any] = {}


def _render_block(bounds: tuple[int, int]) -> list[str]:
    start, stop = bounds
    render_line = _RENDER_JOB["render_line"]
    return [render_line(record) for record in _RENDER_JOB["records"][start:stop]]


def write_prompt_lines(
    handle,
    records: list[dict[str, Any]],
    render_line: Callable[[dict[str, Any]], str],
    *,
    jobs: int | None = None,
    block_size: int = 10_000,
) -> None:
    """Write render_line(record) for each record, in order, rendering blocks in forked workers.

    The records and the renderer reach the workers through fork rather than pickling, so
    render_line may be any function. A rendering error stops the file at the last
    complete block, as a sequential loop would stop at the failing row.
    """
    from scripts.utils.parallel import default_worker_count, map_in_order

    blocks = [(start, min(start + block_size, len(records))) for start in range(0, len(records), block_size)]
    _RENDER_JOB.update(records=records, render_line=render_line)
    try:
        for lines in map_in_order(_render_block, blocks, jobs=default_worker_count() if jobs is None else jobs):
            handle.writelines(lines)
    finally:
        _RENDER_JOB.clear()


def validate_prompt_record(record: dict[str, Any]) -> None:
    missing = [key for key in PROMPT_REQUIRED_KEYS if key not in record]
    if missing:
        raise ValueError(f"Prompt record missing required keys: {missing}")
    extra = [key for key in record if key not in PROMPT_ALLOWED_KEYS]
    if extra:
        raise ValueError(f"Prompt record has non-schema keys: {extra}")
    if not record["id"]:
        raise ValueError("Prompt record has empty id.")
    if not record["time"]:
        raise ValueError("Prompt record has empty time.")
    if record["release_date"] is None:
        raise ValueError("Prompt record has empty release_date.")
    if len(str(record["release_date"])) != 10:
        raise ValueError(f"Invalid release_date: {record['release_date']!r}")
    if not record["system"]:
        raise ValueError("Prompt record has empty system prompt.")
    if not record["user"]:
        raise ValueError("Prompt record has empty user prompt.")
    validate_model_facing_text(record["system"], field="system")
    validate_model_facing_text(record["user"], field="user")
    assistant = json.loads(record["assistant"])
    if not isinstance(assistant, list):
        raise ValueError("Prompt record assistant must decode to a JSON array.")
    naive_baseline = json.loads(record["naive_baseline"])
    if not isinstance(naive_baseline, list):
        raise ValueError("Prompt record naive_baseline must decode to a JSON array.")
    json.dumps(record, allow_nan=False)


def required_text(row: dict[str, object], column: str, row_id: object | None = None) -> str:
    value = row.get(column) if hasattr(row, "get") else row[column]
    if value is None or pd.isna(value):
        suffix = "" if row_id is None else f" in row {row_id}"
        raise RuntimeError(f"Missing labelled value for {column}{suffix}.")
    text = str(value).strip()
    if not text:
        suffix = "" if row_id is None else f" in row {row_id}"
        raise RuntimeError(f"Blank labelled value for {column}{suffix}.")
    return text


def phrase_from_label(
    row: dict[str, object],
    column: str,
    phrases: dict[str, str],
    row_id: object,
) -> str:
    label = required_text(row, column, row_id)
    if label not in phrases:
        raise RuntimeError(f"Unexpected label {label!r} for {column} in row {row_id}.")
    return phrases[label]


def optional_phrase_from_label(
    row: dict[str, object],
    column: str,
    phrases: dict[str, str],
    row_id: object,
) -> str | None:
    value = row.get(column)
    if value is None or pd.isna(value):
        return None
    label = str(value).strip()
    if not label or label.startswith("N/A"):
        return None
    if label not in phrases:
        raise RuntimeError(f"Unexpected label {label!r} for {column} in row {row_id}.")
    return phrases[label]


def checked_label(
    value: object,
    allowed_labels: set[str],
    column: str,
    task_id: str,
    row_number: int,
) -> str:
    label = " ".join(str(value).strip().split()).lower()
    if label not in allowed_labels:
        raise RuntimeError(f"{task_id} row {row_number}: unmapped {column} label {value!r}.")
    return label


DOLLAR_NOTE = "Note that $ denotes amounts in U.S. dollars."
HOUSING_TASKS = {
    "house_census_move",
    "house_psid_owner",
    "house_sce_lockin",
    "house_sce_move",
}


def resolve_prompt_output_path(default_path: Path, *, task_id: str) -> Path:
    """Redirect bounded diagnostic renders without touching canonical JSONLs."""
    output_root = os.environ.get("HB_PROMPT_OUTPUT_ROOT", "").strip()
    limit = prompt_render_limit()
    if not output_root:
        if limit is not None:
            raise SystemExit(
                f"{PROMPT_RENDER_LIMIT_ENV} requires HB_PROMPT_OUTPUT_ROOT outside "
                "the canonical prompts directory."
            )
        return default_path
    output_path = Path(output_root) / f"{task_id}.jsonl"
    if output_path.resolve().is_relative_to(default_path.parent.resolve()) or (
        output_path.exists() and default_path.exists() and output_path.samefile(default_path)
    ):
        raise SystemExit("HB_PROMPT_OUTPUT_ROOT must not write into the canonical prompts directory or alias its output file.")
    return output_path


PROMPT_MONTH_RULES = {
    task_id: spec["prompt_month"]
    for task_id, spec in load_table_schema().items()
    if spec["prompt_month"] is not None
}
TASKS_WITHOUT_CURRENT_MONTH = {"cons_psid_wealth", "house_psid_owner", "income_psid_earnings"}
CURRENT_MONTH_CONTRACT_TASKS = set(PROMPT_MONTH_RULES) | TASKS_WITHOUT_CURRENT_MONTH


def current_month_note(time: str, *, currency: bool, fixed_month: int | None = None) -> str:
    """Return the note; each renderer places it directly in its own template."""
    month = int(time[5:7]) if fixed_month is None else fixed_month
    if not 1 <= month <= 12:
        raise ValueError(f"Invalid prompt month: {month}")
    currency_clause = ", and that $ denotes amounts in U.S. dollars" if currency else ""
    return f"Note that the current month is {MONTH_NAMES[month - 1]}{currency_clause}."


def validate_prompt_note_contract(task_id: str, record: dict[str, Any]) -> None:
    prompt = record["user"]
    notes = CURRENT_MONTH_NOTE_PATTERN.findall(prompt)
    if task_id in TASKS_WITHOUT_CURRENT_MONTH:
        if notes or prompt.count(DOLLAR_NOTE) != 1 or f"\n\n{DOLLAR_NOTE}\n\n" in prompt:
            raise ValueError(f"{task_id}: expected one embedded dollar note and no month note.")
    else:
        rule = PROMPT_MONTH_RULES[task_id]
        expected = current_month_note(record["time"], **rule)
        if notes != [expected]:
            raise ValueError(f"{task_id}: expected exactly one note matching the prompt month.")
        if DOLLAR_NOTE in prompt:
            raise ValueError(f"{task_id}: unexpected standalone dollar note.")
    if "$-" in prompt:
        raise ValueError(f"{task_id}: invalid dollar notation.")


def validate_cross_cutting_prompt_contract(task_id: str, record: dict[str, Any]) -> None:
    if task_id in CURRENT_MONTH_CONTRACT_TASKS:
        validate_prompt_note_contract(task_id, record)
    validate_prompt_response_structure(task_id, record["user"])
    if task_id in HOUSING_TASKS and record["system"] != HOUSING_SYSTEM_PROMPT:
        raise ValueError(f"{task_id}: housing system prompt differs from the common contract.")


def cps_household_sentence(famsize: int, nchild: int) -> str:
    household = f"There {'is' if famsize == 1 else 'are'} {famsize} {'person' if famsize == 1 else 'people'} in your household"
    if nchild == 0:
        return household + ", and you do not live with any of your own children."
    if nchild == 1:
        return household + ", including 1 of your own children."
    return household + f", including {nchild} of your own children."


def cex_education_phrase(label: str) -> str:
    if label.startswith(("a ", "an ", "some ", "less ")):
        return label
    article = "an" if label[:1] in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {label}"
