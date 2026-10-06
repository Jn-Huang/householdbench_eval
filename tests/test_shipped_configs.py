"""The batch-scoring configs shipped in configs/experiments load and name real tasks."""

from pathlib import Path

import pytest

from household_bench_eval.config_lib import (
    load_config_path,
    resolve_gold_path,
    resolve_prediction_path,
)
from household_bench_eval.tasks import ALL_TASKS

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "experiments"
CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))


def test_configs_are_shipped():
    assert [path.name for path in CONFIGS] == [
        "householdbench_evalsample_post_cutoff.yaml",
        "householdbench_evalsample_pre_cutoff.yaml",
    ]


# The evaluation sample's non-empty cells (data/householdbench_evalsample/manifest.csv).
POST_CUTOFF_TASKS = [
    "cons_sce_growth", "cons_sce_shock", "house_sce_move", "income_sce_growth",
    "labor_sce_risk", "labor_sce_search", "macro_sce_revision", "macro_sce_uncertainty",
]
PRE_CUTOFF_TASKS = sorted(POST_CUTOFF_TASKS + [
    "cons_cex_categories", "cons_cex_rebate01", "cons_cex_stimulus08", "cons_cex_total",
    "house_sce_financing", "house_sce_lockin", "income_sce_policy", "labor_sce_offer",
    "labor_sce_reswage",
])


@pytest.mark.parametrize(
    ("split", "expected"),
    [("pre_cutoff", PRE_CUTOFF_TASKS), ("post_cutoff", POST_CUTOFF_TASKS)],
)
def test_configs_list_every_evaluation_sample_task(split, expected):
    config = load_config_path(CONFIG_DIR / f"householdbench_evalsample_{split}.yaml")
    assert config.run.split == split
    assert [task for group in config.task_groups for task in group.tasks] == expected


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.stem)
def test_config_loads_and_resolves_split_paths(path):
    config = load_config_path(path)
    split = config.run.split
    tasks = [task for group in config.task_groups for task in group.tasks]
    assert tasks and set(tasks) <= set(ALL_TASKS)
    assert resolve_gold_path(config, tasks[0]) == (
        f"data/householdbench_evalsample/{split}/{tasks[0]}.jsonl"
    )
    assert resolve_prediction_path(config, tasks[0]).endswith(f"/{split}/{tasks[0]}.jsonl")
