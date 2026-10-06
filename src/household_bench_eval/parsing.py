"""Strict JSON parsing of model responses against each task's answer format."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from household_bench_eval.tasks import FieldSpec, TaskSpec

ScalarValue = float | str | int
DistributionValue = tuple[float, ...]
ParsedRecord = dict[str, ScalarValue | DistributionValue]
PROBABILITY_SUM_TOLERANCE = 1e-6

# Human number formatting that is not valid inside a JSON number. Applied only as a
# retry on a numeric response that already failed strict parsing, so a response that
# parses strictly is never rewritten.
_NUMBER_REPAIRS = (
    re.compile(r"\$"),  # "[-$5,000]" -> "[-5,000]"
    re.compile(r"(?<=\d),(?=\d{3}(?!\d))"),  # "[6,620, 3,800]" -> "[6620, 3800]"
)


@dataclass(frozen=True)
class ParseResult:
    value: ParsedRecord | None
    error: str | None = None
    repaired: bool = False

    @property
    def ok(self) -> bool:
        return self.value is not None


def extract_json_value(raw: Any) -> Any:
    """Extract one JSON value while tolerating common model wrappers."""
    if not isinstance(raw, str):
        return raw

    text = _strip_known_wrappers(raw.strip())
    if not text:
        raise ValueError("empty_response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    candidates: list[Any] = []
    starts = [idx for idx, char in enumerate(text) if char in "[{\""]

    for start in starts:
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        candidates.append(value)

    if not candidates:
        return _strip_known_wrappers(text)

    structured = [value for value in candidates if isinstance(value, (dict, list))]
    if structured:
        return structured[0]
    return candidates[0]


def _parse_strict(raw: Any, spec: TaskSpec) -> ParseResult:
    try:
        decoded = extract_json_value(raw)
        return ParseResult(_normalize_decoded_value(decoded, spec))
    except (TypeError, ValueError) as exc:
        return ParseResult(None, str(exc))


def _repair_number_formatting(text: str) -> str:
    for pattern in _NUMBER_REPAIRS:
        text = pattern.sub("", text)
    return text


def _top_level_json_values(text: str) -> list[Any]:
    """Decode every top-level JSON value in order, not just the first one."""
    decoder = json.JSONDecoder()
    values: list[Any] = []
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index] in " \t\r\n,":
            index += 1
        if index >= length:
            break
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            break
        values.append(value)
        index = end
    return values


def _repair_distribution_nesting(text: str, spec: TaskSpec) -> list[Any] | None:
    """Rebuild the outer list of a distribution response that lost its brackets.

    Two shapes, both unambiguous given the task's answer format:

    * one bracketed vector per distribution but no enclosing list, e.g.
      ``[0.96, 0.04], [0.97, 0.03]`` for two 2-length distributions, and the
      single-distribution case ``[0.985, 0.015]``;
    * one flat vector whose length is the concatenation of the expected lengths.

    Returns ``None`` when neither shape matches exactly. The rebuilt value is fed
    back through the normal checks, so range and sum-to-one still apply.
    """
    expected = [len(distribution.labels) for distribution in spec.distributions]
    vectors = [value for value in _top_level_json_values(text) if isinstance(value, list)]

    if len(vectors) == len(expected) and all(
        len(vector) == size for vector, size in zip(vectors, expected, strict=True)
    ):
        return vectors

    if len(vectors) == 1 and len(vectors[0]) == sum(expected):
        flat = list(vectors[0])
        rebuilt: list[Any] = []
        for size in expected:
            rebuilt.append(flat[:size])
            flat = flat[size:]
        return rebuilt

    return None


def parse_response(
    raw: Any,
    spec: TaskSpec,
    *,
    repair_number_formatting: bool = False,
    repair_distribution_nesting: bool = False,
) -> ParseResult:
    """Parse one response against a task's answer format.

    Both repairs run only on a response that already FAILED strict parsing, so a
    response that parses strictly is returned untouched and enabling either one can
    only move rows from invalid to valid.

    ``repair_number_formatting`` retries a numeric response after removing currency
    symbols and thousands separators, which JSON reads as element separators.

    ``repair_distribution_nesting`` retries a distribution response whose outer list
    is missing, which is the dominant invalidity mode for weaker open-weight models.
    """
    result = _parse_strict(raw, spec)
    if result.ok or not isinstance(raw, str):
        return result

    if repair_distribution_nesting and spec.output_shape == "distribution_array":
        rebuilt = _repair_distribution_nesting(_strip_known_wrappers(raw.strip()), spec)
        if rebuilt is not None:
            try:
                return ParseResult(_normalize_decoded_value(rebuilt, spec), repaired=True)
            except (TypeError, ValueError):
                return result

    if not repair_number_formatting:
        return result
    if spec.target_type != "numeric":
        return result
    repaired_text = _repair_number_formatting(raw)
    if repaired_text == raw:
        return result
    retry = _parse_strict(repaired_text, spec)
    if not retry.ok:
        return result
    return ParseResult(retry.value, retry.error, repaired=True)


def parse_gold(raw: Any, spec: TaskSpec) -> ParsedRecord:
    result = parse_response(raw, spec)
    if not result.ok or result.value is None:
        raise ValueError(f"Invalid gold answer for {spec.name}: {result.error}")
    return result.value


def parse_naive_baseline(raw: Any, spec: TaskSpec) -> ParsedRecord:
    if spec.target_type != "numeric":
        raise ValueError(f"Naive baseline parsing is only defined for numeric tasks: {spec.name}")
    result = parse_response(raw, spec)
    if not result.ok or result.value is None:
        raise ValueError(f"Invalid naive baseline for {spec.name}: {result.error}")
    return result.value


def _normalize_decoded_value(decoded: Any, spec: TaskSpec) -> ParsedRecord:
    if spec.output_shape == "array":
        return _normalize_array(decoded, spec)
    if spec.output_shape == "distribution_array":
        return _normalize_distribution_array(decoded, spec)
    raise ValueError(f"unsupported_output_shape:{spec.output_shape}")


def _normalize_array(decoded: Any, spec: TaskSpec) -> ParsedRecord:
    if not isinstance(decoded, list):
        raise ValueError("expected_json_array")
    if len(decoded) != len(spec.fields):
        raise ValueError(f"expected_{len(spec.fields)}_array_elements")

    output: ParsedRecord = {}
    for field, value in zip(spec.fields, decoded, strict=True):
        output[field.name] = _normalize_field(value, field)
    return output


def _normalize_distribution_array(decoded: Any, spec: TaskSpec) -> ParsedRecord:
    if not isinstance(decoded, list):
        raise ValueError("expected_distribution_array")
    if len(decoded) != len(spec.distributions):
        raise ValueError(f"expected_{len(spec.distributions)}_distributions")

    output: ParsedRecord = {}
    for distribution, raw_vector in zip(spec.distributions, decoded, strict=True):
        if not isinstance(raw_vector, list):
            raise ValueError(f"{distribution.name}:expected_probability_vector")
        if len(raw_vector) != len(distribution.labels):
            raise ValueError(
                f"{distribution.name}:expected_{len(distribution.labels)}_probabilities"
            )
        vector = tuple(
            _normalize_probability(value, distribution.name) for value in raw_vector
        )
        if not math.isclose(
            sum(vector),
            1.0,
            rel_tol=0.0,
            abs_tol=PROBABILITY_SUM_TOLERANCE,
        ):
            raise ValueError(f"{distribution.name}:probabilities_must_sum_to_one")
        output[distribution.name] = vector
    return output


def _normalize_field(value: Any, field: FieldSpec) -> ScalarValue:
    if field.kind == "numeric":
        return _normalize_number(value, field.name)

    if field.kind == "categorical":
        if isinstance(value, str):
            normalized: str | int = value
        elif isinstance(value, int) and not isinstance(value, bool):
            normalized = value
        elif isinstance(value, float) and value.is_integer():
            normalized = int(value)
        else:
            raise ValueError(f"{field.name}:expected_categorical_label")

        if normalized not in field.labels:
            allowed = ",".join(str(label) for label in field.labels)
            raise ValueError(f"{field.name}:invalid_label:{normalized!r};allowed={allowed}")
        return normalized

    raise ValueError(f"{field.name}:unsupported_field_kind:{field.kind}")


def _strip_known_wrappers(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def _normalize_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name}:expected_number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name}:non_finite_number")
    return number


def _normalize_probability(value: Any, distribution_name: str) -> float:
    number = _normalize_number(value, distribution_name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{distribution_name}:probability_out_of_range")
    return number
