#!/usr/bin/env python
"""Copy the paper's evaluation subsamples from completed HouseholdBench files.

Run from the repository root: python scripts/2_tasks/select_evalsample.py
Only tasks listed in data/householdbench/manifest.csv are processed.
"""

import hashlib
import json
from pathlib import Path
import sys
import tomllib

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.io import sha256_file
from scripts.utils.sampling import interleave_prefix, split_prefix, validate_sampling_sidecar

# These settings reproduce the paper's evaluation sample. Changing them defines a new sample.
ROWS_PER_TASK = 200
POST_CUTOFF_DATE = "2026-02-16"
SOURCE_ROOT = PROJECT_ROOT / "data/householdbench"
OUTPUT_ROOT = PROJECT_ROOT / "data/householdbench_evalsample"

tasks = pd.read_csv(SOURCE_ROOT / "manifest.csv", dtype=str, keep_default_na=False)
split_manifest = pd.read_csv(SOURCE_ROOT / "splits/manifest.csv", dtype=str, keep_default_na=False)
if (tasks.empty or tasks.task_id.duplicated().any() or split_manifest.task_id.duplicated().any()
        or set(tasks.task_id) != set(split_manifest.task_id)):
    raise ValueError("Task and split manifests must list the same distinct tasks.")
split_manifest = split_manifest.set_index("task_id")
with (SOURCE_ROOT / "splits/split_config.toml").open("rb") as handle:
    split_config = tomllib.load(handle)
if POST_CUTOFF_DATE < split_config["cutoff_date"]:
    raise ValueError("The evaluation cutoff cannot precede the stored split cutoff.")

# Keep the small selected files in memory until every input has been checked.
outputs = {}
manifest_rows = []
for task in tasks.itertuples(index=False):
    source = PROJECT_ROOT / task.prompt_file
    split_source = PROJECT_ROOT / task.split_file
    receipt = split_manifest.loc[task.task_id]
    if task.split_file != receipt.sidecar_file or sha256_file(split_source) != receipt.sidecar_sha256:
        raise ValueError(f"{task.task_id}: split file differs from its published fingerprint.")
    sidecar = pd.read_csv(split_source, dtype=str, keep_default_na=False)
    for column in ["within_period_order", "split_position"]:
        sidecar[column] = pd.to_numeric(sidecar[column], errors="raise")
    validate_sampling_sidecar(sidecar, task_id=task.task_id, config=split_config)
    if len(sidecar) != int(task.benchmark_rows):
        raise ValueError(f"{task.task_id}: split and task row counts differ.")

    # Retain the original pre-cutoff membership; do not reassign February rows.
    pre = split_prefix(sidecar, split="pre_cutoff_test", requested_rows=ROWS_PER_TASK)
    keys = pd.MultiIndex.from_frame(sidecar[["id", "time"]])
    pre_indices = keys.get_indexer(pd.MultiIndex.from_frame(pre[["id", "time"]])).tolist()

    # Removing early-February releases changes the quarter weights. Rebalance
    # the remaining pool while preserving the stored random order in each quarter.
    post = sidecar.loc[sidecar.split.eq("post_cutoff_test") & sidecar.release_date.gt(POST_CUTOFF_DATE)]
    queues = {
        quarter: group.sort_values("within_period_order").index.tolist()
        for quarter, group in post.groupby("split_quarter")
    }
    post_indices = sorted(interleave_prefix(queues, prefix_size=ROWS_PER_TASK))
    selections = {"pre_cutoff": pre_indices, "post_cutoff": post_indices}
    selected_rows = {index: split for split, indices in selections.items() for index in indices}
    payloads = {split: bytearray() for split in selections}
    written = {split: 0 for split in selections}
    digest = hashlib.sha256()
    source_rows = 0
    with source.open("rb") as handle:
        for index, line in enumerate(handle):
            source_rows += 1
            digest.update(line)
            if index not in selected_rows:
                continue
            record = json.loads(line)
            expected = sidecar.iloc[index]
            if any(str(record[key]) != expected[key] for key in ["id", "time", "release_date"]):
                raise ValueError(f"{task.task_id}: prompt and split keys differ at row {index}.")
            split = selected_rows[index]
            payloads[split].extend(line)
            written[split] += 1
    if source_rows != len(sidecar) or digest.hexdigest() != receipt.prompt_sha256:
        raise ValueError(f"{task.task_id}: prompt file differs from its published count or fingerprint.")

    for split, indices in selections.items():
        if written[split] != len(indices):
            raise ValueError(f"{task.task_id}/{split}: selected prompts are missing.")
        destination = OUTPUT_ROOT / split / f"{task.task_id}.jsonl"
        payload = bytes(payloads[split])
        if indices:
            outputs[destination] = payload
        manifest_rows.append({
            "task_id": task.task_id, "split": split, "rows": len(indices),
            "requested_rows": ROWS_PER_TASK, "post_cutoff_date": POST_CUTOFF_DATE,
            "source_split_cutoff_date": split_config["cutoff_date"],
            "source_prompt_file": task.prompt_file, "source_prompt_sha256": digest.hexdigest(),
            "source_split_file": task.split_file, "source_split_sha256": receipt.sidecar_sha256,
            "prompt_file": str(destination.relative_to(PROJECT_ROOT)) if indices else "",
            "prompt_sha256": hashlib.sha256(payload).hexdigest() if indices else "",
        })
    print(f"{task.task_id}: {written['pre_cutoff']} pre-cutoff, {written['post_cutoff']} post-cutoff", flush=True)

# Refuse to leave stale task files when reusing a directory for a smaller release.
unexpected = set(OUTPUT_ROOT.glob("*/*.jsonl")) - set(outputs)
if unexpected:
    raise ValueError(f"Use an empty output directory; stale prompt files exist: {sorted(unexpected)}")
for destination, payload in outputs.items():
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
pd.DataFrame(manifest_rows).to_csv(OUTPUT_ROOT / "manifest.csv", index=False)
print(f"Wrote {sum(row['rows'] for row in manifest_rows):,} prompts to {OUTPUT_ROOT}.", flush=True)
