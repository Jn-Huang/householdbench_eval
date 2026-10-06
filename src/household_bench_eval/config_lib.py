"""Typed experiment configs for HouseholdBench evaluation.

Experiments are driven by file-backed configs: each run writes the resolved
config next to the results, and model/inference settings are recorded even though
this package only scores already-generated predictions.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from household_bench_eval.tasks import ALL_TASKS


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    model_type: str = "external_predictions"
    model_name: str | None = None
    api_base: str | None = None
    max_tokens: int | None = None
    timeout: float | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    reasoning_effort: str | None = None
    concurrency: int | None = None
    request_seed: int | None = None
    served_name: str | None = None


@dataclasses.dataclass(frozen=True)
class TaskGroupConfig:
    name: str
    tasks: tuple[str, ...]
    output_subdir: str


@dataclasses.dataclass(frozen=True)
class RunConfig:
    seed: int = 42
    split: str = "test"
    num_examples_per_task: int | None = None
    prediction_field: str | None = None
    repair_number_formatting: bool = False
    repair_distribution_nesting: bool = False
    include_details: bool = True


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str
    data_root: str
    prediction_root: str
    output_dir: str = "results"
    gold_path_template: str = "{data_root}/prompts/{task}.jsonl"
    prediction_path_template: str = "{prediction_root}/{task}.jsonl"
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    run: RunConfig = dataclasses.field(default_factory=RunConfig)
    task_groups: tuple[TaskGroupConfig, ...] = ()
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def replace(self, **updates: Any) -> ExperimentConfig:
        return dataclasses.replace(self, **updates)


def resolve_gold_path(config: ExperimentConfig, task: str) -> str:
    return config.gold_path_template.format(
        data_root=config.data_root,
        prediction_root=config.prediction_root,
        task=task,
        split=config.run.split,
    )


def resolve_prediction_path(config: ExperimentConfig, task: str) -> str:
    return config.prediction_path_template.format(
        data_root=config.data_root,
        prediction_root=config.prediction_root,
        task=task,
        split=config.run.split,
    )


def _construct_dataclass(cls: type, value: Any):
    if dataclasses.is_dataclass(cls) and isinstance(value, dict):
        known_fields = {field.name for field in dataclasses.fields(cls)}
        unknown_fields = sorted(set(value) - known_fields)
        if unknown_fields:
            unknown = ", ".join(unknown_fields)
            raise ValueError(f"Unknown {cls.__name__} field(s): {unknown}")
        kwargs = {}
        for field in dataclasses.fields(cls):
            if field.name not in value:
                continue
            raw = value[field.name]
            if field.name == "model":
                raw = _construct_dataclass(ModelConfig, raw)
            elif field.name == "run":
                raw = _construct_dataclass(RunConfig, raw)
            elif field.name == "task_groups":
                raw = tuple(_construct_dataclass(TaskGroupConfig, item) for item in raw)
            elif field.name == "tasks":
                raw = tuple(raw)
            kwargs[field.name] = raw
        return cls(**kwargs)
    return value


def load_config_path(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open() as handle:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            raw = yaml.safe_load(handle)
        else:
            raw = json.load(handle)
    config = _construct_dataclass(ExperimentConfig, raw)
    if not isinstance(config, ExperimentConfig):
        raise ValueError("Experiment config must be a mapping")

    invalid_tasks = sorted(
        {
            task
            for group in config.task_groups
            for task in group.tasks
            if task not in ALL_TASKS
        }
    )
    if invalid_tasks:
        invalid = ", ".join(invalid_tasks)
        raise ValueError(f"Unsupported HouseholdBench task(s): {invalid}")
    return config


def dump_config(config: ExperimentConfig) -> dict[str, Any]:
    return dataclasses.asdict(config)


def write_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_yaml(path: str | Path, data: Any) -> None:
    import yaml

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
