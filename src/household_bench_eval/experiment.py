"""Config-driven HouseholdBench scoring experiments."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from household_bench_eval.config_lib import (
    ExperimentConfig,
    dump_config,
    resolve_gold_path,
    resolve_prediction_path,
    write_json,
    write_yaml,
)
from household_bench_eval.io import load_jsonl
from household_bench_eval.scorer import score_task


def run_experiment(config: ExperimentConfig, experiment_dir: str | Path) -> dict[str, Any]:
    exp_dir = Path(experiment_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = dump_config(config)
    write_json(exp_dir / "experiment_config.json", resolved_config)
    write_yaml(exp_dir / "resolved_config.yaml", resolved_config)

    results_root = exp_dir / config.output_dir
    group_summaries: list[dict[str, Any]] = []
    written_files: list[str] = []

    for group in config.task_groups:
        group_dir = results_root / group.output_subdir / (config.model.model_name or "model")
        group_dir.mkdir(parents=True, exist_ok=True)
        task_summaries: list[dict[str, Any]] = []

        for task in group.tasks:
            gold_path = resolve_gold_path(config, task)
            prediction_path = resolve_prediction_path(config, task)
            result = score_task(
                task,
                load_jsonl(gold_path),
                load_jsonl(prediction_path),
                prediction_field=config.run.prediction_field,
                repair_number_formatting=config.run.repair_number_formatting,
                repair_distribution_nesting=config.run.repair_distribution_nesting,
            )
            result["experiment"] = _experiment_metadata(config, gold_path, prediction_path)
            if not config.run.include_details:
                result.pop("details", None)

            result_path = group_dir / f"{task}.json"
            write_json(result_path, result)
            written_files.append(str(result_path))
            task_summaries.append(
                {
                    "task": task,
                    "gold_jsonl": gold_path,
                    "prediction_jsonl": prediction_path,
                    "result_file": str(result_path),
                    "num_examples": result["num_examples"],
                    "num_valid": result["num_valid"],
                    "valid_response_rate": result["valid_response_rate"],
                    "metrics": result["metrics"],
                }
            )

        group_summary = {
            "name": group.name,
            "tasks": task_summaries,
            "output_dir": str(group_dir),
        }
        group_summaries.append(group_summary)
        write_json(group_dir / "group_summary.json", group_summary)
        written_files.append(str(group_dir / "group_summary.json"))

    final_result: dict[str, Any] = {
        "experiment": config.name,
        "finished_at": datetime.now().isoformat(),
        "result_root": str(results_root),
        "groups": group_summaries,
        "files": written_files,
        "config_files": {
            "experiment_config_json": str(exp_dir / "experiment_config.json"),
            "resolved_config_yaml": str(exp_dir / "resolved_config.yaml"),
        },
    }
    write_json(exp_dir / "final_result.json", final_result)
    return final_result


def _experiment_metadata(
    config: ExperimentConfig,
    gold_path: str,
    prediction_path: str,
) -> dict[str, Any]:
    return {
        "name": config.name,
        "description": config.description,
        "split": config.run.split,
        "seed": config.run.seed,
        "model": dump_config(config)["model"],
        "metadata": config.metadata,
        "gold_jsonl": gold_path,
        "prediction_jsonl": prediction_path,
    }
