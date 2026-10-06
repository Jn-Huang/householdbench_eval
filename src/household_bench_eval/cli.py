"""Command line entrypoint for household_bench_eval."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from household_bench_eval.config_lib import load_config_path
from household_bench_eval.experiment import run_experiment
from household_bench_eval.io import dump_json
from household_bench_eval.scorer import score_task_files
from household_bench_eval.tasks import ALL_TASKS, get_task_spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="household-bench-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe = subparsers.add_parser("describe", help="Print a task's answer format and metrics.")
    describe.add_argument("--task", choices=ALL_TASKS, required=True)

    score = subparsers.add_parser("score", help="Score model predictions for one task.")
    score.add_argument("--task", choices=ALL_TASKS, required=True)
    score.add_argument("--gold-jsonl", required=True)
    score.add_argument("--pred-jsonl", required=True)
    score.add_argument(
        "--prediction-field",
        help="Require this field in each prediction row. Defaults to accepted fallback fields.",
    )
    score.add_argument(
        "--repair-number-formatting",
        action="store_true",
        help=(
            "Retry a failed numeric response after stripping currency symbols and "
            "thousands separators. Never rewrites a response that already parses."
        ),
    )
    score.add_argument(
        "--repair-distribution-nesting",
        action="store_true",
        help=(
            "Retry a failed distribution response whose outer list is missing. "
            "Never rewrites a response that already parses."
        ),
    )
    score.add_argument("--output-json")
    score.add_argument(
        "--no-details",
        action="store_true",
        help="Omit per-example parse details from stdout/output JSON.",
    )

    run = subparsers.add_parser(
        "run-experiment",
        help="Run a file-backed scoring experiment and write config snapshots/results.",
    )
    run.add_argument("--experiment-config-path", required=True)
    run.add_argument(
        "--experiment-dir",
        required=True,
        help="Directory where experiment_config.json, resolved_config.yaml, results, and final_result.json are written.",
    )

    args = parser.parse_args(argv)

    if args.command == "describe":
        spec = get_task_spec(args.task)
        description = asdict(spec)
        description["official_metrics"] = spec.official_metrics
        description["diagnostic_metrics"] = spec.diagnostic_metrics
        print(json.dumps(description, indent=2, sort_keys=True))
        return 0

    if args.command == "score":
        result = score_task_files(
            args.task,
            args.gold_jsonl,
            args.pred_jsonl,
            prediction_field=args.prediction_field,
            repair_number_formatting=args.repair_number_formatting,
            repair_distribution_nesting=args.repair_distribution_nesting,
        )
        if args.no_details:
            result = {key: value for key, value in result.items() if key != "details"}
        if args.output_json:
            dump_json(result, args.output_json)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "run-experiment":
        config = load_config_path(args.experiment_config_path)
        result = run_experiment(config, args.experiment_dir)
        print(json.dumps({"final_result": result, "ok": True}, indent=2, sort_keys=True))
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2
