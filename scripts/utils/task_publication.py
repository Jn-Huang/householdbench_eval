"""Check task products and publish the benchmark manifest and fixed sample splits.

Execution order lives in scripts/run_pipeline.py. These functions retain the
construction checks needed before publishing task metadata and split files.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


from scripts.utils.prompt_rendering import validate_cross_cutting_prompt_contract
from scripts.utils.responses import validate_prompt_response_structure
from scripts.utils.sampling import (
    REGISTRY_COLUMNS,
    SIDECAR_COLUMNS,
    SUMMARY_COLUMNS,
    load_sampling_config,
    load_sampling_order,
    validate_task_sampling_sidecar,
)
from scripts.utils.io import publish_staged_directory, sha256_file, sha256_json
from scripts.utils.table_schema import load_table_schema, public_table_columns
from scripts.utils.task_registry import TASK_REGISTRY_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "data" / "householdbench"
PROMPTS_DIR = PACKAGE_ROOT / "prompts"
TABULAR_DIR = PACKAGE_ROOT / "tabular"
MANIFEST_PATH = PACKAGE_ROOT / "manifest.csv"
MANIFEST_TMP_PATH = PACKAGE_ROOT / "manifest.tmp.csv"
SPLITS_DIR = PACKAGE_ROOT / "splits"
SAMPLING_ROOT = PROJECT_ROOT / "data" / "intermediate" / "householdbench"

MANIFEST_COLUMNS = [
    "task_id",
    "dataset",
    "topic",
    "mode",
    "prompt_file",
    "tabular_file",
    "tabular_columns",
    "min_time",
    "max_time",
    "min_release_date",
    "max_release_date",
    "eligible_rows",
    "benchmark_rows",
    "post_cutoff_rows",
    "split_file",
]
REQUIRED_PROMPT_KEYS = [
    "id",
    "time",
    "release_date",
    "system",
    "user",
    "assistant",
    "naive_baseline",
]


def read_prompt_keys(prompt_path: Path, *, task_id: str) -> pd.DataFrame:
    rows = []
    with prompt_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = [column for column in ("id", "time", "release_date") if column not in record]
            if missing:
                raise RuntimeError(f"{prompt_path} line {line_number} is missing keys: {missing}")
            rows.append(
                {
                    "id": str(record["id"]),
                    "time": str(record["time"]),
                    "release_date": str(record["release_date"]),
                }
            )
    if not rows:
        raise RuntimeError(f"Prompt file has no records: {prompt_path}")
    return pd.DataFrame(rows, columns=["id", "time", "release_date"], dtype="string")


def stage_01_artifact_paths(task_id: str) -> dict[str, Path]:
    sampling_dir = SAMPLING_ROOT / task_id
    return {
        "registry": sampling_dir / "01_sampling_registry.parquet",
        "summary": sampling_dir / "01_sampling_summary.csv",
    }


def validate_stage_01_artifacts(
    task_id: str, *, require_prompt: bool
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate one full registry in bounded batches and its exported table keys."""
    paths = stage_01_artifact_paths(task_id)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{task_id}: missing stage-01 artifacts: {missing}")
    table_path = TABULAR_DIR / f"{task_id}.csv"
    if not table_path.is_file():
        raise FileNotFoundError(f"{task_id}: missing stage-01 table: {table_path}")
    with table_path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), None)
    if header != public_table_columns(task_id):
        raise RuntimeError(f"{task_id}: table header does not equal the strict schema registry.")
    table_keys = pd.read_csv(
        table_path,
        dtype="string",
        usecols=["id", "time", "release_date"],
        keep_default_na=False,
    )
    summary_frame = pd.read_csv(paths["summary"], keep_default_na=False)
    if list(summary_frame.columns) != list(SUMMARY_COLUMNS) or len(summary_frame) != 1:
        raise RuntimeError(f"{task_id}: invalid stage-01 summary schema or row count.")
    summary = summary_frame.iloc[0].to_dict()
    if str(summary["task_id"]) != task_id:
        raise RuntimeError(f"{task_id}: invalid stage-01 summary identity.")

    parquet_file = pq.ParquetFile(paths["registry"])
    if parquet_file.schema_arrow.names != list(REGISTRY_COLUMNS):
        raise RuntimeError(f"{task_id}: invalid full sampling-registry schema.")
    if parquet_file.metadata.num_rows != int(summary["full_eligible_rows"]):
        raise RuntimeError(f"{task_id}: registry and summary full-row counts differ.")
    cell_stats: dict[tuple[str, str], dict[str, int]] = {}
    split_stats: dict[str, dict[str, int]] = {}
    cell_expected: dict[tuple[str, str], int] = {}
    split_expected: dict[str, int] = {}
    exported_parts = []
    pre_cutoff_rows = 0
    post_cutoff_rows = 0
    pre_cutoff_exported = 0
    post_cutoff_exported = 0
    cutoff = load_sampling_config()["cutoff_date"]
    scan_columns = [
        "task_id",
        *SIDECAR_COLUMNS,
        "split_rows",
        "period_rows",
        "exported",
    ]
    for batch in parquet_file.iter_batches(batch_size=250_000, columns=scan_columns):
        frame = batch.to_pandas()
        if frame.drop(columns=["split_position"]).isna().any().any():
            raise RuntimeError(f"{task_id}: registry contains missing required values.")
        if set(frame["task_id"].astype(str)) != {task_id}:
            raise RuntimeError(f"{task_id}: registry contains a different task identity.")
        if frame.duplicated(["id", "time"]).any():
            raise RuntimeError(f"{task_id}: duplicate registry keys within a scan batch.")
        expected_quarters = (
            pd.to_datetime(frame["time"], format="%Y-%m-%d", errors="raise").dt.year.astype(str)
            + "Q"
            + (((pd.to_datetime(frame["time"], format="%Y-%m-%d").dt.month - 1) // 3) + 1).astype(str)
        )
        if not frame["split_quarter"].astype(str).eq(expected_quarters).all():
            raise RuntimeError(f"{task_id}: registry split quarters do not match public time.")
        post = frame["release_date"].astype(str).gt(cutoff)
        if not frame.loc[post, "split"].astype(str).eq("post_cutoff_test").all():
            raise RuntimeError(f"{task_id}: a post-cutoff registry row has the wrong split.")
        if frame.loc[~post, "split"].astype(str).eq("post_cutoff_test").any():
            raise RuntimeError(f"{task_id}: a pre-cutoff registry row is in post_cutoff_test.")
        if not frame.loc[post, "exported"].astype(bool).all():
            raise RuntimeError(f"{task_id}: a post-cutoff registry row is not exported.")
        pre_cutoff_rows += int((~post).sum())
        post_cutoff_rows += int(post.sum())
        pre_cutoff_exported += int((~post & frame["exported"].astype(bool)).sum())
        post_cutoff_exported += int((post & frame["exported"].astype(bool)).sum())
        for (split, quarter), group in frame.groupby(
            ["split", "split_quarter"], observed=True, sort=False
        ):
            key = (str(split), str(quarter))
            values = pd.to_numeric(group["within_period_order"], errors="raise").astype("int64")
            stats = cell_stats.setdefault(
                key, {"count": 0, "minimum": 2**63 - 1, "maximum": 0, "sum": 0, "sum_sq": 0}
            )
            stats["count"] += len(values)
            stats["minimum"] = min(stats["minimum"], int(values.min()))
            stats["maximum"] = max(stats["maximum"], int(values.max()))
            stats["sum"] += int(values.sum())
            stats["sum_sq"] += sum(int(value) * int(value) for value in values)
            expected_values = set(pd.to_numeric(group["period_rows"], errors="raise").astype(int))
            if len(expected_values) != 1:
                raise RuntimeError(f"{task_id}: inconsistent period_rows for {key}.")
            expected = expected_values.pop()
            if key in cell_expected and cell_expected[key] != expected:
                raise RuntimeError(f"{task_id}: period_rows changes across batches for {key}.")
            cell_expected[key] = expected
        for split, group in frame.groupby("split", observed=True, sort=False):
            split = str(split)
            expected_values = set(pd.to_numeric(group["split_rows"], errors="raise").astype(int))
            if len(expected_values) != 1:
                raise RuntimeError(f"{task_id}: inconsistent split_rows for {split}.")
            expected = expected_values.pop()
            if split in split_expected and split_expected[split] != expected:
                raise RuntimeError(f"{task_id}: split_rows changes across batches for {split}.")
            split_expected[split] = expected
            positions = pd.to_numeric(group["split_position"], errors="coerce").dropna().astype("int64")
            stats = split_stats.setdefault(
                split, {"count": 0, "minimum": 2**63 - 1, "maximum": 0, "sum": 0, "sum_sq": 0}
            )
            if not positions.empty:
                stats["count"] += len(positions)
                stats["minimum"] = min(stats["minimum"], int(positions.min()))
                stats["maximum"] = max(stats["maximum"], int(positions.max()))
                stats["sum"] += int(positions.sum())
                stats["sum_sq"] += sum(int(value) * int(value) for value in positions)
        exported = frame.loc[frame["exported"].astype(bool), SIDECAR_COLUMNS].copy()
        if not exported.empty:
            exported_parts.append(exported)
    for key, stats in cell_stats.items():
        count = stats["count"]
        expected_sum = count * (count + 1) // 2
        expected_sum_sq = count * (count + 1) * (2 * count + 1) // 6
        if (
            cell_expected.get(key) != count
            or stats["minimum"] != 1
            or stats["maximum"] != count
            or stats["sum"] != expected_sum
            or stats["sum_sq"] != expected_sum_sq
        ):
            raise RuntimeError(f"{task_id}: noncontiguous within-period order for {key}.")
    for split, stats in split_stats.items():
        if sum(value["count"] for key, value in cell_stats.items() if key[0] == split) != split_expected[split]:
            raise RuntimeError(f"{task_id}: split_rows does not match registry rows for {split}.")
        count = stats["count"]
        expected_sum = count * (count + 1) // 2
        expected_sum_sq = count * (count + 1) * (2 * count + 1) // 6
        if count and (
            stats["minimum"] != 1
            or stats["maximum"] != count
            or stats["sum"] != expected_sum
            or stats["sum_sq"] != expected_sum_sq
        ):
            raise RuntimeError(f"{task_id}: noncontiguous materialized positions for {split}.")
    expected_summary_counts = {
        "pre_cutoff_eligible_rows": pre_cutoff_rows,
        "post_cutoff_eligible_rows": post_cutoff_rows,
        "pre_cutoff_exported_rows": pre_cutoff_exported,
        "post_cutoff_exported_rows": post_cutoff_exported,
        "exported_rows": pre_cutoff_exported + post_cutoff_exported,
    }
    for field, expected in expected_summary_counts.items():
        if int(summary[field]) != expected:
            raise RuntimeError(f"{task_id}: summary {field} differs from the registry scan.")
    exported_registry = pd.concat(exported_parts, ignore_index=True)
    for column in ("within_period_order", "split_position"):
        exported_registry[column] = pd.to_numeric(
            exported_registry[column], errors="raise"
        ).astype("int64")
    sidecar = exported_registry.sort_values(["id", "time"]).reset_index(drop=True)
    if len(table_keys) != int(summary["exported_rows"]):
        raise RuntimeError(f"{task_id}: public table and exported-row counts differ.")
    if require_prompt:
        prompt_path = PROMPTS_DIR / f"{task_id}.jsonl"
        if not prompt_path.is_file():
            raise FileNotFoundError(f"{task_id}: missing prompt input: {prompt_path}")
        prompt_keys = read_prompt_keys(prompt_path, task_id=task_id)
        validate_task_sampling_sidecar(
            table_keys=table_keys,
            prompt_keys=prompt_keys,
            sidecar=sidecar,
            task_id=task_id,
            config=load_sampling_config(),
        )
    else:
        table = table_keys.astype("string").sort_values(["id", "time"]).reset_index(drop=True)
        split = (
            sidecar.loc[:, ["id", "time", "release_date"]]
            .astype("string")
            .sort_values(["id", "time"])
            .reset_index(drop=True)
        )
        if not table.equals(split):
            raise RuntimeError(f"{task_id}: stage-01 table and exported registry keys differ.")
    return sidecar, summary


def active_split_manifest_fields(task_id: str) -> dict[str, object]:
    """Reference a split only when its completed publication is present."""
    if not (SPLITS_DIR / "READY.json").is_file() or not (SPLITS_DIR / f"{task_id}.csv").is_file():
        return {"split_file": ""}
    return {"split_file": f"data/householdbench/splits/{task_id}.csv"}


def add_split_fields_to_main_manifest(*, task_ids: set[str]) -> None:
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"task_id": "string"})
    if set(manifest["task_id"].astype(str)) != task_ids:
        raise RuntimeError("The main manifest task set does not match the split task set.")
    manifest["split_file"] = manifest["task_id"].map(
        lambda task_id: f"data/householdbench/splits/{task_id}.csv"
    )
    manifest.to_csv(MANIFEST_TMP_PATH, index=False)
    MANIFEST_TMP_PATH.replace(MANIFEST_PATH)


def build_authoritative_splits(tasks: list[tuple[Path, dict]]) -> None:
    """Publish public splits from the full registries fixed by stage 01."""
    seen_task_ids = {str(metadata["task_id"]) for _, metadata in tasks}
    config = load_sampling_config()
    if set(config["row_fallback_tasks"]) - seen_task_ids:
        unknown = sorted(set(config["row_fallback_tasks"]) - seen_task_ids)
        raise RuntimeError(f"Sampling configuration names unknown fallback tasks: {unknown}")
    staging_dir = SPLITS_DIR.with_name(f"{SPLITS_DIR.name}.staging.{os.getpid()}")
    if staging_dir.exists():
        raise FileExistsError(f"Refusing to overwrite split staging directory: {staging_dir}")
    staging_dir.mkdir(parents=True)
    schema_sha256 = sha256_json(load_table_schema())
    manifest_rows = []
    total_records = 0
    try:
        for task_index, (_, metadata) in enumerate(tasks, start=1):
            task_id = str(metadata["task_id"])
            print(f"[{task_index}/{len(tasks)}] Validating stage-01 sidecar for {task_id}", flush=True)
            table_path = TABULAR_DIR / f"{task_id}.csv"
            prompt_path = PROMPTS_DIR / f"{task_id}.jsonl"
            sidecar, summary = validate_stage_01_artifacts(task_id, require_prompt=True)
            table_keys = pd.read_csv(
                table_path,
                usecols=["id", "time", "release_date"],
                dtype="string",
                keep_default_na=False,
            )
            sidecar = table_keys.merge(
                sidecar,
                on=["id", "time", "release_date"],
                how="left",
                sort=False,
                validate="one_to_one",
            ).loc[:, SIDECAR_COLUMNS]
            if sidecar.isna().any().any() or not table_keys.equals(
                sidecar.loc[:, ["id", "time", "release_date"]].astype("string")
            ):
                raise RuntimeError(f"{task_id}: failed to align public sidecar to table order.")
            sidecar_path = staging_dir / f"{task_id}.csv"
            sidecar.to_csv(sidecar_path, index=False)
            total_records += len(sidecar)
            manifest_rows.append(
                {
                    "task_id": task_id,
                    "sidecar_file": f"data/householdbench/splits/{task_id}.csv",
                    "sidecar_sha256": sha256_file(sidecar_path),
                    **summary,
                    "table_sha256": sha256_file(table_path),
                    "prompt_sha256": sha256_file(prompt_path),
                    "prompt_key_check": "exact_full_jsonl_key_reconciliation",
                }
            )
        split_manifest = pd.DataFrame(manifest_rows)
        if len(split_manifest) != len(tasks) or split_manifest["task_id"].duplicated().any():
            raise RuntimeError("Split manifest does not contain exactly one row per task.")
        split_manifest.to_csv(staging_dir / "manifest.csv", index=False)
        # Record the effective settings, including each task's canonical ordering.
        # This is generated provenance, not an input configuration file.
        settings_lines = ["# Generated from scripts/utils/sampling.py and scripts/2_tasks/task_registry.json."]
        settings_lines.extend(f"{key} = {json.dumps(value)}" for key, value in config.items())
        ordering = load_sampling_order()["tasks"]
        for task_id in sorted(seen_task_ids):
            settings_lines.append(f"\n[tasks.{task_id}]")
            settings_lines.extend(f"{key} = {json.dumps(value)}" for key, value in ordering[task_id].items())
        settings_path = staging_dir / "split_config.toml"
        settings_path.write_text("\n".join(settings_lines) + "\n", encoding="utf-8")
        ready = {
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "task_count": len(tasks),
            "total_records": total_records,
            "cutoff_date": config["cutoff_date"],
            "seed": config["seed"],
            "python_version": sys.version.split()[0],
            "split_config_sha256": sha256_file(settings_path),
            "task_registry_sha256": sha256_file(TASK_REGISTRY_PATH),
            "sampling_rules_sha256": sha256_file(Path(__file__).with_name("sampling.py")),
            "table_schema_sha256": schema_sha256,
            "fallback_tasks": config["row_fallback_tasks"],
            "build_status": "complete",
            "prompt_key_check": "exact_full_jsonl_key_reconciliation",
        }
        (staging_dir / "READY.json").write_text(
            json.dumps(ready, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        expected_files = {
            *(f"{task_id}.csv" for task_id in seen_task_ids),
            "manifest.csv",
            "split_config.toml",
            "READY.json",
        }
        actual_files = {path.name for path in staging_dir.iterdir() if path.is_file()}
        if actual_files != expected_files:
            raise RuntimeError(f"Unexpected split staging contents: {sorted(actual_files ^ expected_files)}")
        publish_staged_directory(staging_dir, SPLITS_DIR)
        add_split_fields_to_main_manifest(
            task_ids=seen_task_ids
        )
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise
    print(
        f"Published {SPLITS_DIR} with {len(tasks)} validated stage-01 sidecars "
        f"and {total_records:,} rows."
    )


def task_manifest_row(metadata: dict) -> dict[str, object]:
    """Reconcile a completed task table, prompts and sampling counts."""
    task_id = str(metadata["task_id"])
    prompt_path = PROMPTS_DIR / f"{task_id}.jsonl"
    tabular_path = TABULAR_DIR / f"{task_id}.csv"
    if not prompt_path.is_file():
        raise RuntimeError(f"Missing prompt file for {task_id}: {prompt_path}")
    if not tabular_path.is_file():
        raise RuntimeError(f"Missing tabular file for {task_id}: {tabular_path}")

    with tabular_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise RuntimeError(f"Tabular file is empty: {tabular_path}") from error
        tabular_rows = sum(1 for _ in reader)
        tabular_columns = len(header)

    records = 0
    min_time = None
    max_time = None
    min_release_date = None
    max_release_date = None

    with prompt_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing_keys = [key for key in REQUIRED_PROMPT_KEYS if key not in record]
            if missing_keys:
                raise RuntimeError(f"{prompt_path} line {line_number} is missing keys: {missing_keys}")
            assistant = json.loads(record["assistant"])
            naive_baseline = json.loads(record["naive_baseline"])
            if not isinstance(assistant, list) or not isinstance(naive_baseline, list):
                raise RuntimeError(
                    f"{prompt_path} line {line_number}: assistant and naive_baseline "
                    "must decode to JSON arrays."
                )
            validate_prompt_response_structure(task_id, record["user"])
            validate_cross_cutting_prompt_contract(task_id, record)

            records += 1
            observation_time = str(record["time"])[:10]
            release_date = str(record["release_date"])[:10]
            try:
                pd.Timestamp(release_date)
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"{prompt_path} line {line_number}: invalid release_date {release_date!r}."
                ) from error

            min_time = observation_time if min_time is None else min(min_time, observation_time)
            max_time = observation_time if max_time is None else max(max_time, observation_time)
            min_release_date = release_date if min_release_date is None else min(min_release_date, release_date)
            max_release_date = release_date if max_release_date is None else max(max_release_date, release_date)

    if records == 0:
        raise RuntimeError(f"Prompt file has no records: {prompt_path}")
    if tabular_rows != records:
        raise RuntimeError(f"{task_id}: tabular rows={tabular_rows:,} but prompt records={records:,}.")

    sampling_paths = stage_01_artifact_paths(task_id)
    sampling_summary = pd.read_csv(sampling_paths["summary"], keep_default_na=False)
    if list(sampling_summary.columns) != list(SUMMARY_COLUMNS) or len(sampling_summary) != 1:
        raise RuntimeError(f"{task_id}: invalid stage-01 sampling summary.")
    sampling_summary = sampling_summary.iloc[0]
    if int(sampling_summary["exported_rows"]) != records:
        raise RuntimeError(
            f"{task_id}: sampling export rows={int(sampling_summary['exported_rows']):,} "
            f"but prompt records={records:,}."
        )

    if (
        int(sampling_summary["full_eligible_rows"])
        != int(sampling_summary["pre_cutoff_eligible_rows"]) + int(sampling_summary["post_cutoff_eligible_rows"])
        or int(sampling_summary["exported_rows"])
        != int(sampling_summary["pre_cutoff_exported_rows"]) + int(sampling_summary["post_cutoff_exported_rows"])
        or int(sampling_summary["post_cutoff_eligible_rows"])
        != int(sampling_summary["post_cutoff_exported_rows"])
    ):
        raise RuntimeError(f"{task_id}: sampling counts do not reconcile or omit post-cutoff rows.")

    return {
        **metadata,
        "prompt_file": str(prompt_path.relative_to(PROJECT_ROOT)),
        "tabular_file": str(tabular_path.relative_to(PROJECT_ROOT)),
        "tabular_columns": tabular_columns,
        "min_time": min_time,
        "max_time": max_time,
        "min_release_date": min_release_date,
        "max_release_date": max_release_date,
        "eligible_rows": int(sampling_summary["full_eligible_rows"]),
        "benchmark_rows": int(sampling_summary["exported_rows"]),
        "post_cutoff_rows": int(sampling_summary["post_cutoff_exported_rows"]),
        **active_split_manifest_fields(task_id),
    }
