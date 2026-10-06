"""Response families, target domains, and array validation for public tasks."""

from __future__ import annotations

import math
import re
from collections.abc import Collection
from typing import Any


from scripts.utils.table_schema import load_table_schema


RESPONSE_STRUCTURE_SPECS = {
    task_id: spec["response"] for task_id, spec in load_table_schema().items()
}
NUMERIC_TASKS = {
    task_id for task_id, spec in RESPONSE_STRUCTURE_SPECS.items() if spec["family"] == "numeric"
}
CATEGORICAL_TARGET_DOMAINS = {
    task_id: spec["categories"]
    for task_id, spec in RESPONSE_STRUCTURE_SPECS.items()
    if spec["family"] == "categorical"
}
DISTRIBUTION_GROUPS = {
    task_id: {name: tuple(columns) for name, columns in spec["groups"].items()}
    for task_id, spec in RESPONSE_STRUCTURE_SPECS.items()
    if spec["family"] == "distribution"
}


def response_structure_insertion(task_id: str) -> str:
    try:
        spec = RESPONSE_STRUCTURE_SPECS[task_id]
    except KeyError as exc:
        raise KeyError(f"Unknown HouseholdBench task: {task_id}") from exc
    return (
        f"Your output must follow this exact array structure: {spec['skeleton']}. "
        f"{spec['replacement_instruction']}"
    )


def validate_response_structure_registry(task_ids: Collection[str]) -> None:
    expected = sorted(str(task_id) for task_id in task_ids)
    actual = sorted(RESPONSE_STRUCTURE_SPECS)
    if expected != actual:
        raise ValueError(
            f"Response-structure task set differs; missing={sorted(set(expected) - set(actual))}, "
            f"unexpected={sorted(set(actual) - set(expected))}."
        )
    family_counts = {"v": 0, "p": 0, "s": 0}
    for task_id, spec in RESPONSE_STRUCTURE_SPECS.items():
        skeleton = spec["skeleton"]
        closing = spec["closing_sentence"]
        if not skeleton.startswith("[") or not skeleton.endswith("]"):
            raise ValueError(f"{task_id}: response skeleton is not an array structure.")
        if not closing or closing != closing.strip() or not closing.endswith("."):
            raise ValueError(f"{task_id}: closing sentence is invalid.")
        family = {"numeric": "v", "distribution": "p", "categorical": "s"}[spec["family"]]
        family_counts[family] += 1
        tokens = re.findall(r"[vps]_[0-9]+(?:_[0-9]+)?", skeleton)
        if not tokens or any(not token.startswith(family + "_") for token in tokens):
            raise ValueError(f"{task_id}: skeleton placeholder family is invalid.")
        without_tokens = re.sub(r"[vps]_[0-9]+(?:_[0-9]+)?", "", skeleton)
        without_tokens = without_tokens.replace("...", "")
        if re.sub(r"[\[\], \" ]", "", without_tokens):
            raise ValueError(f"{task_id}: skeleton contains invalid text.")
    if sum(family_counts.values()) != len(RESPONSE_STRUCTURE_SPECS):
        raise ValueError(f"Unexpected response-structure family counts: {family_counts}")


def validate_prompt_response_structure(task_id: str, prompt: str) -> None:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Prompt must be a nonempty string.")
    if task_id not in RESPONSE_STRUCTURE_SPECS:
        raise KeyError(f"Unknown HouseholdBench task: {task_id}")
    insertion = response_structure_insertion(task_id)
    closing = RESPONSE_STRUCTURE_SPECS[task_id]["closing_sentence"]
    if task_id == "cons_psid_wealth":
        expected_ending = closing + " " + insertion
    else:
        expected_ending = insertion + " " + closing
    if prompt.count(insertion) != 1:
        raise ValueError(f"{task_id}: prompt must contain its response structure exactly once.")
    if prompt.count(closing) != 1:
        raise ValueError(f"{task_id}: prompt must contain its closing sentence exactly once.")
    if not prompt.endswith(expected_ending):
        raise ValueError(f"{task_id}: response instruction has the wrong structure placement.")


def validate_numeric_array(value: Any, *, length: int, field: str) -> None:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field} must be a {length}-element JSON array.")
    for index, element in enumerate(value):
        if isinstance(element, bool) or not isinstance(element, (int, float)):
            raise ValueError(f"{field}[{index}] must be numeric.")
        if not math.isfinite(float(element)):
            raise ValueError(f"{field}[{index}] must be finite.")


def validate_categorical_array(
    value: Any,
    *,
    length: int,
    categories: set[str] | frozenset[str],
    field: str,
) -> None:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field} must be a {length}-element JSON array.")
    for index, element in enumerate(value):
        if not isinstance(element, str) or element not in categories:
            raise ValueError(f"{field}[{index}] is outside the categorical target domain.")


def validate_distribution_array(
    value: Any,
    *,
    category_counts: tuple[int, ...] | list[int],
    field: str,
    sum_tolerance: float = 1e-9,
) -> None:
    if not isinstance(value, list) or len(value) != len(category_counts):
        raise ValueError(f"{field} must contain {len(category_counts)} atomic distributions.")
    for distribution_index, (distribution, category_count) in enumerate(
        zip(value, category_counts, strict=True)
    ):
        if not isinstance(distribution, list) or len(distribution) != category_count:
            raise ValueError(
                f"{field}[{distribution_index}] must contain {category_count} probabilities."
            )
        for category_index, probability in enumerate(distribution):
            if isinstance(probability, bool) or not isinstance(probability, (int, float)):
                raise ValueError(
                    f"{field}[{distribution_index}][{category_index}] must be numeric."
                )
            if not math.isfinite(float(probability)) or not 0 <= float(probability) <= 1:
                raise ValueError(
                    f"{field}[{distribution_index}][{category_index}] must be in [0, 1]."
                )
        if not math.isclose(
            sum(float(probability) for probability in distribution),
            1.0,
            rel_tol=0.0,
            abs_tol=sum_tolerance,
        ):
            raise ValueError(f"{field}[{distribution_index}] probabilities must sum to one.")
