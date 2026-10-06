"""End-to-end CLI tests for the describe, score, and run-experiment commands."""

import json

from household_bench_eval.cli import main


def _write_gold_and_predictions(tmp_path):
    gold = tmp_path / "gold.jsonl"
    pred = tmp_path / "pred.jsonl"
    gold.write_text('{"id": "a", "assistant": "[1, 2, 3]", "naive_baseline": "[0, 1, 2]"}\n')
    pred.write_text('{"id": "a", "prediction": "[2, 2, 4]"}\n')
    return gold, pred


def test_describe_prints_task_spec(capsys):
    assert main(["describe", "--task", "house_sce_lockin"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "house_sce_lockin"
    assert payload["official_metrics"] == ["total_variation"]
    assert payload["distributions"][0]["labels"] == ["no_move", "move"]


def test_score_writes_output_json_without_details(tmp_path, capsys):
    gold, pred = _write_gold_and_predictions(tmp_path)
    output = tmp_path / "metrics.json"

    code = main(
        [
            "score",
            "--task",
            "cons_cex_total",
            "--gold-jsonl",
            str(gold),
            "--pred-jsonl",
            str(pred),
            "--output-json",
            str(output),
            "--no-details",
        ]
    )

    assert code == 0
    result = json.loads(output.read_text())
    assert result["num_valid"] == 1
    assert "details" not in result
    assert result["metrics"]["primary_metric"] == "mean_relmae"
    stdout_result = json.loads(capsys.readouterr().out)
    assert stdout_result["num_valid"] == 1


def test_run_experiment_command_writes_results(tmp_path, capsys):
    data_root = tmp_path / "data"
    pred_root = tmp_path / "pred"
    (data_root / "prompts").mkdir(parents=True)
    pred_root.mkdir()
    (data_root / "prompts" / "cons_cex_total.jsonl").write_text(
        '{"id": "a", "assistant": "[1, 2, 3]", "naive_baseline": "[0, 1, 2]"}\n'
    )
    (pred_root / "cons_cex_total.jsonl").write_text('{"id": "a", "prediction": "[2, 2, 4]"}\n')

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: cli_run",
                "description: CLI run-experiment test",
                f"data_root: {data_root}",
                f"prediction_root: {pred_root}",
                "task_groups:",
                "  - name: tiny",
                "    output_subdir: tiny",
                "    tasks: [cons_cex_total]",
            ]
        )
        + "\n"
    )

    code = main(
        [
            "run-experiment",
            "--experiment-config-path",
            str(config_path),
            "--experiment-dir",
            str(tmp_path / "run"),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    final_result = tmp_path / "run" / "final_result.json"
    assert final_result.exists()
    assert (tmp_path / "run" / "experiment_config.json").exists()
    assert (tmp_path / "run" / "resolved_config.yaml").exists()
