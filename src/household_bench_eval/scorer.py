"""Join gold and prediction files, parse responses, and compute task metrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

from household_bench_eval.io import load_jsonl
from household_bench_eval.metrics import compute_task_metrics
from household_bench_eval.parsing import (
    ParsedRecord,
    parse_gold,
    parse_naive_baseline,
    parse_response,
)
from household_bench_eval.tasks import get_task_spec

PREDICTION_FIELDS = ("prediction", "raw_output", "response", "model_output", "assistant")

# Maps (row, row_index, require_key=False) to the string key that joins gold and predictions.
RowKey = Callable[..., str]


def score_task(
    task_name: str,
    gold_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    prediction_field: str | None = None,
    repair_number_formatting: bool = False,
    repair_distribution_nesting: bool = False,
) -> dict[str, Any]:
    spec = get_task_spec(task_name)
    row_key, keyed_on_time = _select_row_key(gold_rows)
    predictions_by_id = _index_prediction_rows(prediction_rows, row_key)
    predicted_ids = {str(row["id"]) for row in prediction_rows if row.get("id") is not None}

    valid_predictions: list[ParsedRecord] = []
    valid_references: list[ParsedRecord] = []
    valid_naive_baselines: list[ParsedRecord] | None = (
        [] if spec.target_type == "numeric" else None
    )
    valid_example_ids: list[str] = []
    invalid_examples: list[dict[str, str]] = []
    details: list[dict[str, Any]] = []
    error_counts: Counter[str] = Counter()
    matched_prediction_ids: set[str] = set()
    num_repaired = 0

    for row_index, gold_row in enumerate(gold_rows):
        example_id = row_key(gold_row, row_index)
        reference = parse_gold(gold_row.get("assistant"), spec)
        naive_baseline = (
            parse_naive_baseline(gold_row.get("naive_baseline"), spec)
            if spec.target_type == "numeric"
            else None
        )
        prediction_row = predictions_by_id.get(example_id)

        if prediction_row is None:
            # An id the predictions do carry, under another time, is a time mismatch.
            id_predicted = str(gold_row.get("id")) in predicted_ids
            error = "time_mismatch" if keyed_on_time and id_predicted else "missing_prediction"
            error_counts[error] += 1
            invalid_examples.append({"id": example_id, "error": error})
            details.append(
                _detail(
                    example_id,
                    None,
                    reference,
                    naive_baseline,
                    None,
                    error,
                )
            )
            continue

        matched_prediction_ids.add(example_id)
        try:
            raw_prediction = _get_raw_prediction(prediction_row, prediction_field=prediction_field)
        except ValueError as exc:
            error = str(exc)
            error_counts[error] += 1
            invalid_examples.append({"id": example_id, "error": error})
            details.append(
                _detail(
                    example_id,
                    None,
                    reference,
                    naive_baseline,
                    None,
                    error,
                )
            )
            continue

        parsed = parse_response(
            raw_prediction,
            spec,
            repair_number_formatting=repair_number_formatting,
            repair_distribution_nesting=repair_distribution_nesting,
        )
        if not parsed.ok or parsed.value is None:
            error = parsed.error or "parse_failed"
            error_counts[error] += 1
            invalid_examples.append({"id": example_id, "error": error})
            details.append(
                _detail(
                    example_id,
                    raw_prediction,
                    reference,
                    naive_baseline,
                    None,
                    error,
                )
            )
            continue

        valid_predictions.append(parsed.value)
        valid_references.append(reference)
        if valid_naive_baselines is not None:
            if naive_baseline is None:
                raise ValueError(f"Missing normalized naive baseline for {task_name}:{example_id}")
            valid_naive_baselines.append(naive_baseline)
        valid_example_ids.append(example_id)
        num_repaired += 1 if parsed.repaired else 0
        details.append(
            _detail(
                example_id,
                raw_prediction,
                reference,
                naive_baseline,
                parsed.value,
                None,
                repaired=parsed.repaired,
            )
        )

    metrics = compute_task_metrics(
        valid_predictions,
        valid_references,
        spec,
        naive_baselines=valid_naive_baselines,
    )
    total = len(gold_rows)
    num_valid = len(valid_predictions)
    extra_prediction_ids = sorted(set(predictions_by_id) - matched_prediction_ids)

    return {
        "task": task_name,
        "mode": spec.mode,
        "num_examples": total,
        "num_predictions": len(prediction_rows),
        "num_valid": num_valid,
        "num_invalid": total - num_valid,
        "num_repaired": num_repaired,
        "valid_response_rate": (num_valid / total) if total else 0.0,
        "coverage_rate": (len(matched_prediction_ids) / total) if total else 0.0,
        "num_extra_predictions": len(extra_prediction_ids),
        "extra_prediction_ids": extra_prediction_ids,
        "error_counts": dict(error_counts),
        "valid_example_ids": valid_example_ids,
        "invalid_examples": invalid_examples,
        "metrics": metrics,
        "details": details,
    }


def score_task_files(
    task_name: str,
    gold_jsonl: str,
    pred_jsonl: str,
    prediction_field: str | None = None,
    repair_number_formatting: bool = False,
    repair_distribution_nesting: bool = False,
) -> dict[str, Any]:
    return score_task(
        task_name,
        load_jsonl(gold_jsonl),
        load_jsonl(pred_jsonl),
        prediction_field=prediction_field,
        repair_number_formatting=repair_number_formatting,
        repair_distribution_nesting=repair_distribution_nesting,
    )



def _index_prediction_rows(
    rows: list[dict[str, Any]], row_key: RowKey
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for row_index, row in enumerate(rows):
        example_id = row_key(row, row_index, require_key=True)
        if example_id in indexed:
            duplicate_ids.add(example_id)
        indexed[example_id] = row
    if duplicate_ids:
        duplicates = ", ".join(sorted(duplicate_ids)[:10])
        raise ValueError(f"Duplicate prediction ids: {duplicates}")
    return indexed


def _select_row_key(gold_rows: list[dict[str, Any]]) -> tuple[RowKey, bool]:
    """Choose how gold and prediction rows are matched, and require unique gold keys.

    Rows match on `example_id` when the gold file carries one. Otherwise, when every
    gold row has `id` and `time`, they match on both, written as "<id>@<time>":
    released prompt files repeat an `id` when a household appears in several periods.
    Gold files without `time` match on `id`. Returns the key function and whether it
    keys on `time`.
    """
    if any(row.get("example_id") is not None for row in gold_rows):
        row_key, keyed_on_time = _row_key, False
    elif gold_rows and all(
        row.get("id") is not None and row.get("time") is not None for row in gold_rows
    ):
        row_key, keyed_on_time = _id_time_row_key(gold_rows), True
    else:
        row_key, keyed_on_time = _row_key_prefer_id, False

    seen: set[str] = set()
    for row_index, row in enumerate(gold_rows):
        key = row_key(row, row_index)
        if key in seen:
            raise ValueError(
                f"Gold rows repeat the key {key!r} (row {row_index}); every gold row needs a "
                "unique example_id, id, or (id, time)"
            )
        seen.add(key)
    return row_key, keyed_on_time


def _row_key(row: dict[str, Any], row_index: int, require_key: bool = False) -> str:
    for field in ("example_id", "id"):
        value = row.get(field)
        if value is not None:
            return str(value)
    if require_key:
        raise ValueError(f"Prediction row {row_index} is missing required field 'example_id' or 'id'")
    return str(row_index)


def _row_key_prefer_id(row: dict[str, Any], row_index: int, require_key: bool = False) -> str:
    """Key on `id` first, for gold files that have ids but no example_id or time."""
    if row.get("id") is not None:
        return str(row["id"])
    return _row_key(row, row_index, require_key=require_key)


def _join_id_time(row: dict[str, Any]) -> str:
    return f"{row['id']}@{row['time']}"


def _id_time_row_key(gold_rows: list[dict[str, Any]]) -> RowKey:
    """Key on (id, time). A prediction without `time` matches by `id` only when ids are unique."""
    gold_ids = [str(row["id"]) for row in gold_rows]
    ids_unique = len(set(gold_ids)) == len(gold_ids)
    key_by_id = {str(row["id"]): _join_id_time(row) for row in gold_rows} if ids_unique else {}

    def row_key(row: dict[str, Any], row_index: int, require_key: bool = False) -> str:
        if row.get("id") is None:
            return _row_key(row, row_index, require_key=require_key)
        if row.get("time") is not None:
            return _join_id_time(row)
        if not ids_unique:
            raise ValueError(
                f"Prediction row {row_index} (id {row['id']!r}) has no 'time': this task's gold "
                "rows repeat ids across periods, so predictions need both 'id' and 'time'"
            )
        return key_by_id.get(str(row["id"]), str(row["id"]))

    return row_key


def _get_raw_prediction(row: dict[str, Any], prediction_field: str | None = None) -> Any:
    if prediction_field is not None:
        if prediction_field not in row:
            raise ValueError(f"Prediction row has no configured response field {prediction_field!r}")
        return row[prediction_field]

    for field in PREDICTION_FIELDS:
        if field in row:
            return row[field]
    accepted = ", ".join(PREDICTION_FIELDS)
    raise ValueError(f"Prediction row has no response field. Accepted fields: {accepted}")


def _detail(
    example_id: str,
    raw_prediction: Any,
    reference: ParsedRecord,
    naive_baseline: ParsedRecord | None,
    parsed_prediction: ParsedRecord | None,
    error: str | None,
    repaired: bool = False,
) -> dict[str, Any]:
    return {
        "id": example_id,
        "valid": error is None,
        "raw_prediction": raw_prediction,
        "parsed_prediction": parsed_prediction,
        "reference": reference,
        "naive_baseline": naive_baseline,
        "error": error,
        "repaired": repaired,
    }
