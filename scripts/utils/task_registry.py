"""Read the single registry of task metadata, sampling choices and table definitions."""

import json
from pathlib import Path
import re
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_REGISTRY_PATH = PROJECT_ROOT / "scripts/2_tasks/task_registry.json"
TASK_METADATA_FIELDS = (
    "dataset", "topic", "mode", "split_unit", "sort_order", "sort_id_column", "sort_id_type",
)
TABLE_SCHEMA_FIELDS = (
    "targets", "prompt_predictors", "categorical_predictors", "required_predictors",
    "response", "prompt_month",
)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject repeated task IDs or nested fields instead of silently overwriting them."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key in task registry: {key}")
        result[key] = value
    return result


def load_task_registry(path: Path = TASK_REGISTRY_PATH) -> dict[str, dict[str, Any]]:
    """Read and validate complete definitions without importing task execution code."""
    registry = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object)
    if not isinstance(registry, dict) or not registry:
        raise ValueError(f"Task registry must be a nonempty object: {path}")
    for task_id, task_schema in registry.items():
        if not task_id or task_id != task_id.strip():
            raise ValueError(f"Invalid task ID in {path}: {task_id!r}")
        if not isinstance(task_schema, dict) or set(task_schema) != set(TASK_METADATA_FIELDS + TABLE_SCHEMA_FIELDS):
            raise ValueError(f"{task_id}: missing or extra task-registry fields.")
        for field in TASK_METADATA_FIELDS:
            value = task_schema[field]
            if (not isinstance(value, str) or value != value.strip()
                    or (not value and field not in {"sort_id_column", "sort_id_type"})):
                raise ValueError(f"{task_id}: {field} must be a trimmed string; only typed sort ID fields may be empty.")
        if task_schema["split_unit"] not in {"quarter", "row"}:
            raise ValueError(f"{task_id}: split_unit must be quarter or row.")
        if task_schema["sort_order"] not in {"time_typed_id_id", "id_time"}:
            raise ValueError(f"{task_id}: invalid sort_order.")
        column, kind = task_schema["sort_id_column"], task_schema["sort_id_type"]
        if column or kind:
            if (not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", column)
                    or kind not in {"integer", "string"}
                    or task_schema["sort_order"] != "time_typed_id_id"):
                raise ValueError(f"{task_id}: invalid typed sort ID.")
        targets = task_schema["targets"]
        predictors = task_schema["prompt_predictors"]
        if not isinstance(targets, list) or not targets:
            raise ValueError(f"{task_id}: targets must be a nonempty list.")
        if not isinstance(predictors, list):
            raise ValueError(f"{task_id}: prompt_predictors must be a list.")
        columns = ["id", "time", "release_date", *targets, *predictors]
        if any(not isinstance(column, str) or not column for column in columns):
            raise ValueError(f"{task_id}: schema columns must be nonempty strings.")
        duplicates = sorted({column for column in columns if columns.count(column) > 1})
        if duplicates:
            raise ValueError(f"{task_id}: duplicate public columns: {duplicates}")
        for subset in ("categorical_predictors", "required_predictors"):
            values = task_schema[subset]
            if (
                not isinstance(values, list)
                or len(values) != len(set(values))
                or not set(values) <= set(predictors)
            ):
                raise ValueError(f"{task_id}: invalid {subset}.")
        response = task_schema["response"]
        family = response["family"]
        if family not in {"numeric", "categorical", "distribution"}:
            raise ValueError(f"{task_id}: invalid response family.")
        if family == "categorical" and list(response["categories"]) != targets:
            raise ValueError(f"{task_id}: category domains differ from the ordered targets.")
        if (
            family == "distribution"
            and [column for group in response["groups"].values() for column in group] != targets
        ):
            raise ValueError(f"{task_id}: distribution groups differ from the ordered targets.")
        month = task_schema["prompt_month"]
        if month is not None and (
            set(month) != {"fixed_month", "currency"}
            or month["fixed_month"] not in {None, *range(1, 13)}
            or not isinstance(month["currency"], bool)
        ):
            raise ValueError(f"{task_id}: invalid prompt month definition.")
    return registry
