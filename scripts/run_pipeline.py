#!/usr/bin/env python
"""Reconstruct HouseholdBench from the frozen raw inputs.

Run from the repository root: python scripts/run_pipeline.py
Use --check-inputs to verify all input fingerprints without rebuilding outputs.
Use --jobs N to set how many source builders and tasks run at once (default: all
available CPUs; --jobs 1 runs every step in sequence). The CPS, CEX Diary and CPS
linking steps also use their own worker processes, whatever N is.
"""

# %% 1. Paths, settings and required inputs

import argparse
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.parallel import available_cpus, python_step, run_steps
from scripts.utils.source_inputs import check_production_dependencies, validate_source_inputs

# Models are fitted sequentially; this controls the threads within each fit.
XGBOOST_THREADS = 4

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--check-inputs", action="store_true",
    help="Verify dependencies and all frozen input fingerprints, then exit without building.",
)
parser.add_argument(
    "--jobs", type=int, default=available_cpus(),
    help="Source builders and tasks to run at once (default: available CPUs). "
         "Each step holds its data in memory, so lower this on machines with little RAM; "
         "the CPS builder's own workers need about 32 GB even with --jobs 1.",
)
args = parser.parse_args()
if args.jobs < 1:
    parser.error("--jobs must be at least 1.")

# Diagnostic renders belong in separate files and cannot form a complete build.
if not args.check_inputs and any(
    os.environ.get(name, "").strip()
    for name in ("HOUSEHOLDBENCH_RENDER_PROMPT_LIMIT", "HB_PROMPT_OUTPUT_ROOT")
):
    raise SystemExit(
        "Unset HOUSEHOLDBENCH_RENDER_PROMPT_LIMIT and HB_PROMPT_OUTPUT_ROOT before "
        "running the production pipeline. Use an individual prompt script for diagnostic renders."
    )

# Fail before construction if a required library or input file is missing.
# Each source builder verifies its own checksums before reading its data.
check_production_dependencies()
input_count = validate_source_inputs(root=PROJECT_ROOT, verify_checksums=args.check_inputs)
if args.check_inputs:
    print(f"Verified {input_count} source/file entries; no products were rebuilt.")
    raise SystemExit(0)

import pandas as pd

from scripts.utils.task_publication import (
    MANIFEST_COLUMNS,
    MANIFEST_PATH,
    MANIFEST_TMP_PATH,
    build_authoritative_splits,
    task_manifest_row,
)
from scripts.utils.tasks import load_tasks

tasks = load_tasks()
STEP_LOG_DIR = PROJECT_ROOT / "output/pipeline_logs"
print(f"Running up to {args.jobs} steps at once (--jobs).", flush=True)


# %% 2. Build source intermediates

# CPI precedes macro_context. BEA, LAUS and FHFA precede state_macro_panel.
# The survey builders write only their own intermediates, so each wave runs in
# parallel and a wave starts only after the previous one has finished.
source_waves = [
    ["cpi", "bea_sainc", "bls_laus", "fhfa_hpi"],
    ["macro_context", "state_macro_panel"],
    ["cex", "cex_diary", "cps", "cps_ui_policy", "census", "psid", "mich", "sce", "sce_financing"],
]
for wave in source_waves:
    print(f"Building sources: {', '.join(wave)}", flush=True)
    run_steps(
        [python_step(f"source_{source}", PROJECT_ROOT / "scripts/1_preprocessing" / f"{source}.py")
         for source in wave],
        jobs=args.jobs, cwd=PROJECT_ROOT, log_dir=STEP_LOG_DIR,
    )


# %% 3. Build each task table and render its prompts

# load_tasks validates 2_tasks/task_registry.json and the task folders/scripts,
# then returns tasks in name order. Tasks write only their own files and draw their
# samples from seeds derived from their task IDs, so they run in parallel without
# changing any output. A failed subprocess stops the pipeline.
print(f"Building {len(tasks)} tasks", flush=True)
run_steps(
    [python_step(f"task_{metadata['task_id']}", task_dir / "01_make_table.py", task_dir / "02_render_prompts.py")
     for task_dir, metadata in tasks],
    jobs=args.jobs, cwd=PROJECT_ROOT, log_dir=STEP_LOG_DIR,
)
# Check prompt structure and reconcile table, prompt and sampling counts, in task order.
manifest_rows = [task_manifest_row(metadata) for _, metadata in tasks]


# %% 4. Publish the task manifest and sample splits

manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
if len(manifest) != len(tasks):
    raise RuntimeError(f"Expected {len(tasks)} manifest rows, got {len(manifest)}.")
manifest.to_csv(MANIFEST_TMP_PATH, index=False)
MANIFEST_TMP_PATH.replace(MANIFEST_PATH)
print(f"Wrote {MANIFEST_PATH} with {len(manifest)} tasks.", flush=True)

# Table construction has already fixed every eligible row's split and ordering.
# Check those registries against the tables and prompts, then publish their
# selected rows in table order. This step does not draw a new sample.
build_authoritative_splits(tasks)

# Copy the paper's evaluation subsamples from the completed prompts and splits.
subprocess.run(
    [sys.executable, str(PROJECT_ROOT / "scripts/2_tasks/select_evalsample.py")],
    cwd=PROJECT_ROOT, check=True,
)


# %% 5. Fit XGBoost baselines and publish predictions

# The fitting stage uses 3_xgboost/model_seeds.csv and 2_tasks/task_registry.json.
# Existing models are reused only when their input and specification hashes match.
print("Fitting XGBoost baselines and generating predictions", flush=True)
subprocess.run(
    [sys.executable, str(PROJECT_ROOT / "scripts/3_xgboost/build.py"),
     "--threads", str(XGBOOST_THREADS)],
    cwd=PROJECT_ROOT, check=True,
)
print("HouseholdBench production build completed.", flush=True)
