# HouseholdBench

**Evaluating Large Language Models as Predictors of Household Economic Behavior**

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Website](https://img.shields.io/badge/Website-leaderboard-1f6feb.svg)](https://jn-huang.github.io/householdbench)
[![Test set](https://img.shields.io/badge/%F0%9F%A4%97%20Test%20set-householdbench--eval-ffd21e.svg)](https://huggingface.co/datasets/householdbench/householdbench-eval)
[![Full data](https://img.shields.io/badge/%F0%9F%A4%97%20Full%20data-householdbench--full-ffd21e.svg)](https://huggingface.co/datasets/householdbench/householdbench-full)
[![Code license: MIT](https://img.shields.io/badge/Code%20license-MIT-green.svg)](LICENSE)
<!-- TODO: replace XXXX.XXXXX with the arXiv identifier once the preprint is announced. -->

HouseholdBench unites 6 U.S. household surveys and 32 prediction tasks spanning
numeric, categorical and probabilistic outcomes, related to consumption, income,
labor, expectations, and housing. Using past behavior, demographics and macroeconomic
conditions, the tasks test whether LLMs predict behavior, including how households
adjust to changes in various policies.

This repository holds everything needed to use the benchmark:

| Part | Location | What it does |
|---|---|---|
| Scoring package | [`src/household_bench_eval/`](src/household_bench_eval) | Parses model responses and scores them with the paper's metrics |
| Construction pipeline | [`scripts/`](scripts) | Rebuilds all 32 tasks from the raw survey files: tables, prompts, splits, the evaluation sample and the XGBoost reference |

## Setup

One environment covers the whole repository. Install [uv](https://docs.astral.sh/uv/),
then run:

```bash
git clone https://github.com/Jn-Huang/householdbench_eval.git
cd householdbench_eval
uv sync                    # scoring package only
uv sync --extra pipeline   # also the pinned libraries for rebuilding the data
```

Python 3.12 or later is required; uv installs 3.13.5, the version the pipeline was
built with, from `.python-version`. The `pipeline` extra pins the library versions used
to build the released data. On macOS, XGBoost also needs `brew install libomp`. Run all
commands from the repository root with `uv run`.

## Getting the data

Seventeen tasks (four from the BLS Consumer Expenditure Survey, thirteen from the New
York Fed Survey of Consumer Expectations) are released on Hugging Face. The other
fifteen use IPUMS, PSID, Michigan Surveys of Consumers or ICPSR data, which we may not
redistribute, so you need to rebuild them from your own downloads (option C).

| Option | What you get | Tasks | Size | Where |
|---|---|---:|---:|---|
| **A. Evaluation sample** | The exact test prompts scored in the paper: 3,200 pre-cutoff and 1,500 post-cutoff | 17 | 13 MB | [`householdbench/householdbench-eval`](https://huggingface.co/datasets/householdbench/householdbench-eval) |
| **B. Full benchmark** | All 1,670,349 rows: task tables, prompts, split assignments and XGBoost reference predictions | 17 | 6.5 GB | [`householdbench/householdbench-full`](https://huggingface.co/datasets/householdbench/householdbench-full) |
| **C. Rebuild from source** | Options A and B for all 32 tasks: 4,663,568 rows, 6,079 pre-cutoff and 2,500 post-cutoff test prompts | 32 | 2.9 GB supplied + your own downloads | [Step-by-step below](#rebuilding-all-32-tasks) |

Each prompt record holds `id`, `time`, `release_date`, `system`, `user`, the reference
answer `assistant`, and `naive_baseline` (the no-change prediction). Send only `system`
and `user` to a model. Rows are unique on `(id, time)`; the same household can appear
in several periods.

## Quickstart: evaluate a model on the evaluation sample

1. **Set up the environment** as in [Setup](#setup); `uv sync` is enough for scoring.

2. **Download the evaluation sample** (option A) into `data/householdbench_evalsample/`;
   `--exclude "data/*"` skips the Parquet copies that only the dataset viewer and
   `load_dataset` read:

   ```bash
   uv run --with huggingface_hub hf download householdbench/householdbench-eval \
     --repo-type dataset --exclude "data/*" --local-dir data/householdbench_evalsample
   ```

   This creates `data/householdbench_evalsample/{pre_cutoff,post_cutoff}/<task>.jsonl`
   and a `manifest.csv` with row counts and SHA-256 checksums.

3. **Query your model.** For every record, send `system` and `user` and write one
   prediction row per record, carrying `id` and `time`:

   ```json
   {"id": "70102645", "time": "2019-12-01", "prediction": "[-2.0]"}
   ```

   `uv run household-bench-eval describe --task <task>` prints the answer format each task
   expects.

4. **Score each task:**

   ```bash
   uv run household-bench-eval score \
     --task cons_sce_growth \
     --gold-jsonl data/householdbench_evalsample/pre_cutoff/cons_sce_growth.jsonl \
     --pred-jsonl predictions/my_model/pre_cutoff/cons_sce_growth.jsonl \
     --output-json results/my_model/pre_cutoff/cons_sce_growth.json
   ```

   To score every task at once, copy the two configs in `configs/experiments/`, set
   `prediction_root` and `model_name`, and see
   [Batch scoring](#batch-scoring-and-the-python-api).

5. **Compare with the leaderboard** on the [website](https://jn-huang.github.io/householdbench).
   The leaderboard covers all 32 tasks, so a summary over the 17 released tasks is not
   comparable with its overall numbers. Post-cutoff prompts contain data released after
   16 February 2026, so they are unseen only by models whose training data end before
   that date. Report the two splits separately.

## Rebuilding all 32 tasks

The pipeline builds every task table, prompt file, split, the evaluation sample and the
XGBoost reference predictions from the raw survey files. Details are in
[`scripts/README.md`](scripts/README.md).

1. **Set up the environment with the pipeline libraries** (see [Setup](#setup)):

   ```bash
   uv sync --extra pipeline
   ```

2. **Download the supplied raw inputs** (94 of the 107 files) from the `raw/` folder of
   the full-data dataset into `data/raw/` (git ignores `data/`):

   ```bash
   uv run --with huggingface_hub hf download householdbench/householdbench-full \
     --repo-type dataset --include "raw/*" --local-dir data
   ```

3. **Download the 13 remaining files from their providers.** Most need a free
   account and acceptance of the provider's terms. Request the exact version listed:
   the pipeline checks every input before it builds anything. Every input must match
   its stored checksum except three downloads, which are checked for presence only.
   The Census and CEX Diary files differ on every download: IPUMS stamps the Census
   extract with its creation time, and each ICPSR archive includes a terms-of-use page
   that names the person who downloaded it. IPUMS serves only its current CPS
   revision, which differs from the one the release used in a few values, so a new CPS
   extract cannot match the release's checksum either.

   | Source | Files | Provider | Save as, under `data/raw/` |
   |---|---:|---|---|
   | CEX Diary 1982–1989 | 7 | [ICPSR](https://www.icpsr.umich.edu/web/ICPSR/series/20) | `micro/cex/diary/icpsr/ICPSR_08599-V1.zip` and six more (study numbers in [`scripts/README.md`](scripts/README.md#cex-diary)) |
   | CPS Basic Monthly + Displaced Worker Supplement, 1976–2026 | 1 | [IPUMS CPS](https://cps.ipums.org/cps/) | `micro/cps/cps_basic.csv.gz` |
   | Census 1990 and 2000 1% samples | 1 | [IPUMS USA](https://usa.ipums.org/usa/) | `micro/census/census.gz` |
   | Michigan Surveys of Consumers, 1978–2026 | 1 | [SCA Cross-Section Archive](https://data.sca.isr.umich.edu/sda.php) | `micro/mich/mich.csv` |
   | PSID-SHELF V2 (two archives, 18.7 GB) | 2 | [openICPSR 194322](https://doi.org/10.3886/E194322V2) | `micro/psid-shelf/PSIDSHELF_1968_2021_LONG_7.9_GB.zip` and `micro/psid-shelf/Construction_Files/Data/PSID_COMPLETE_MAIN_STUDY_1968_2021_FULL_10.8_GB.zip` |
   | FRED-QD, July 2026 vintage | 1 | [St. Louis Fed](https://www.stlouisfed.org/research/economists/mccracken/fred-databases) | `macro/fred-qd/2026-07-QD.csv` |

   [`scripts/README.md`](scripts/README.md#manual-downloads) lists the exact variables,
   samples and formats for each extract. The Michigan archive revises its data, so a
   new extract can fail its checksum and stop the pipeline; open an issue rather than
   editing the stored checksum.

4. **Check the inputs:**

   ```bash
   uv run python scripts/run_pipeline.py --check-inputs
   ```

5. **Run the pipeline:**

   ```bash
   uv run python scripts/run_pipeline.py
   ```

   Source builders and tasks run in parallel on all available CPUs. Use `--jobs N` to
   cap the number of simultaneous steps (each holds its data in memory), or `--jobs 1`
   to run them one after another. The outputs are the same either way.

6. **Find the outputs** in `data/householdbench/` (tables, prompts, splits, XGBoost
   predictions), `data/householdbench_evalsample/` (the paper's test prompts) and
   `output/` (construction summaries and diagnostics). To confirm you reproduced the
   paper's test prompts for the 17 released tasks, compare the SHA-256 values in
   `data/householdbench_evalsample/manifest.csv` with the `manifest.csv` of the Hugging
   Face test set (the rebuild overwrites a copy kept in that folder).

## Tasks and dataset statistics

Rows released by 31 January 2026 (at most 500,000 per task) are assigned to training,
validation and pre-cutoff test in an 8:1:1 ratio by whole quarter (seed 42); five tasks
with too few quarters assign rows instead. *Post-cutoff rows* were released after
16 February 2026, the paper's evaluation cutoff. *Train (SFT)* is the paper's
fine-tuning set: the first 6,000 training rows of each baseline task. *Test* columns are
the evaluation sample. *Download* marks the 17 tasks released on Hugging Face; the
others need a rebuild.

| Task | Topic | Survey | Mode | Target type | Targets | Years | Rows | Post-cutoff rows | Train (SFT) | Test, pre | Test, post | Download |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|
| `cons_cex_categories` | Consumption and saving | CEX | Baseline | Numeric | 12 | 1980–2025 | 500,000 | 0 | 6,000 | 200 | – | yes |
| `cons_cex_rebate01` | Consumption and saving | CEX | Policy response | Numeric | 3 | 2001–2002 | 9,168 | 0 | – | 200 | – | yes |
| `cons_cex_sspay` | Consumption and saving | CEX | Policy response | Numeric | 3 | 1986–1996 | 20,769 | 0 | – | 200 | – | rebuild |
| `cons_cex_stimulus08` | Consumption and saving | CEX | Policy response | Numeric | 3 | 2007–2009 | 17,940 | 0 | – | 200 | – | yes |
| `cons_cex_total` | Consumption and saving | CEX | Baseline | Numeric | 3 | 1980–2025 | 500,000 | 0 | 6,000 | 200 | – | yes |
| `cons_psid_jobloss` | Consumption and saving | PSID | Policy response | Numeric | 1 | 1970–1991 | 3,676 | 0 | – | 200 | – | rebuild |
| `cons_psid_wealth` | Consumption and saving | PSID | Baseline | Numeric | 1 | 1999–2019 | 74,578 | 0 | 6,000 | 200 | – | rebuild |
| `cons_sce_growth` | Consumption and saving | SCE | Baseline | Numeric | 1 | 2015–2024 | 28,969 | 919 | 6,000 | 200 | 200 | yes |
| `cons_sce_shock` | Consumption and saving | SCE | Policy response | Numeric | 2 | 2015–2024 | 20,169 | 632 | – | 200 | 200 | yes |
| `house_census_move` | Housing and location | Census | Baseline | Categorical | 1 | 1990–2000 | 500,000 | 0 | 6,000 | 200 | – | rebuild |
| `house_psid_owner` | Housing and location | PSID | Baseline | Categorical | 1 | 1978–2019 | 70,170 | 0 | 6,000 | 200 | – | rebuild |
| `house_sce_financing` | Housing and location | SCE | Policy response | Numeric | 6 | 2014 | 995 | 0 | – | 100 | – | yes |
| `house_sce_lockin` | Housing and location | SCE | Policy response | Probabilistic | 1 | 2023–2024 | 734 | 0 | – | 100 | – | yes |
| `house_sce_move` | Housing and location | SCE | Baseline | Probabilistic | 1 | 2013–2025 | 148,995 | 5,247 | 6,000 | 200 | 200 | yes |
| `income_cps_displace` | Income dynamics and household resources | CPS | Policy response | Numeric | 1 | 1996–2022 | 22,553 | 0 | – | 200 | – | rebuild |
| `income_mich_finance` | Income dynamics and household resources | Michigan | Baseline | Numeric | 1 | 1978–2026 | 252,870 | 1,916 | 6,000 | 200 | 200 | rebuild |
| `income_psid_earnings` | Income dynamics and household resources | PSID | Baseline | Numeric | 3 | 1968–2011 | 76,938 | 0 | 6,000 | 200 | – | rebuild |
| `income_sce_growth` | Income dynamics and household resources | SCE | Baseline | Numeric | 1 | 2013–2025 | 179,318 | 6,126 | 6,000 | 200 | 200 | yes |
| `income_sce_policy` | Income dynamics and household resources | SCE | Policy response | Categorical | 4 | 2016–2025 | 19,254 | 0 | – | 200 | – | yes |
| `labor_cps_displace` | Labor supply, job search, and retirement | CPS | Policy response | Categorical | 1 | 1994–2024 | 53,787 | 0 | – | 200 | – | rebuild |
| `labor_cps_jobfind` | Labor supply, job search, and retirement | CPS | Baseline | Categorical | 1 | 1994–2026 | 505,774 | 4,872 | 6,000 | 200 | 200 | rebuild |
| `labor_cps_retire` | Labor supply, job search, and retirement | CPS | Baseline | Categorical | 1 | 1995–2025 | 528,680 | 23,776 | 6,000 | 200 | 200 | rebuild |
| `labor_cps_separation` | Labor supply, job search, and retirement | CPS | Baseline | Categorical | 1 | 1994–2026 | 656,473 | 130,787 | 6,000 | 200 | 200 | rebuild |
| `labor_cps_ui` | Labor supply, job search, and retirement | CPS | Policy response | Categorical | 1 | 2008–2014 | 158,989 | 0 | – | 200 | – | rebuild |
| `labor_psid_addedworker` | Labor supply, job search, and retirement | PSID | Policy response | Numeric | 4 | 1975–1990 | 1,117 | 0 | – | 79 | – | rebuild |
| `labor_sce_offer` | Labor supply, job search, and retirement | SCE | Baseline | Categorical | 1 | 2014–2024 | 3,623 | 0 | 2,686 | 200 | – | yes |
| `labor_sce_reswage` | Labor supply, job search, and retirement | SCE | Policy response | Numeric | 2 | 2013–2021 | 5,994 | 0 | – | 200 | – | yes |
| `labor_sce_risk` | Labor supply, job search, and retirement | SCE | Baseline | Probabilistic | 3 | 2013–2025 | 85,720 | 3,113 | 6,000 | 200 | 200 | yes |
| `labor_sce_search` | Labor supply, job search, and retirement | SCE | Baseline | Probabilistic | 2 | 2013–2025 | 2,636 | 100 | 2,016 | 200 | 100 | yes |
| `macro_mich_outlook` | Macroeconomic expectations | Michigan | Baseline | Numeric | 2 | 1981–2026 | 66,845 | 696 | 6,000 | 200 | 200 | rebuild |
| `macro_sce_revision` | Macroeconomic expectations | SCE | Policy response | Numeric | 2 | 2014–2025 | 7,100 | 264 | – | 200 | 200 | yes |
| `macro_sce_uncertainty` | Macroeconomic expectations | SCE | Baseline | Probabilistic | 5 | 2013–2025 | 139,734 | 4,974 | 6,000 | 200 | 200 | yes |
| **Total (32)** | | | | | | | **4,663,568** | **183,422** | **100,702** | **6,079** | **2,500** | 17 yes |

Policy-response tasks are evaluation-only and have no fine-tuning rows. Four cells hold
fewer than 200 test prompts because their test pool is smaller than 200 rows.

## Scoring

### Input format

**Gold rows** are the prompt records themselves. The scorer reads `assistant` (the
reference answer) and, for numeric tasks, `naive_baseline`; both are JSON-formatted
strings.

**Prediction rows** carry the record's `id` and `time` and one raw model-response field
(`prediction` by default; `raw_output`, `response`, `model_output` and `assistant` are
fallbacks, or pass `--prediction-field` to require one). Rows match on `id` and `time`,
reported as `<id>@<time>`, and a wrong `time` is reported as `time_mismatch`. `time` may
be left out only in tasks whose ids are unique. Gold files with an `example_id` field
match on that instead.

Parsing is strict by design:

- Numeric and categorical tasks require a flat JSON array of the documented length.
  Keyed-object predictions are rejected.
- Categorical labels are case-sensitive and must come from the task's label set.
- Probabilistic tasks require an outer array with one complete probability vector per
  question. Probabilities must lie in `[0, 1]` and each vector must sum to 1 within
  `1e-6`.
- `<think>…</think>` reasoning wrappers and surrounding prose are stripped before the
  JSON payload is extracted.

Unparseable or missing predictions are excluded from scoring and reported per example
with a reason; every result includes coverage and valid-response rates. A prediction
file with duplicate keys, a row without an `id`, or a row without `time` where ids
repeat stops scoring with an error. Two opt-in flags, `--repair-number-formatting` and
`--repair-distribution-nesting`, retry responses that fail only on number formatting or
a missing outer list.

### Metrics

Metrics are fixed by target type and cannot be overridden:

| Target type | Tasks | Primary metric | Notes |
|---|---:|---|---|
| Numeric | 18 | RelMAE: model MAE divided by the no-change baseline's MAE, averaged over targets | nMAE (MAE divided by the target's standard deviation) is reported as a diagnostic |
| Categorical | 9 | Macro-F1, averaged over targets | Accuracy is reported as a secondary metric |
| Probabilistic | 5 | Total variation distance, averaged over examples and then questions | |

### Batch scoring and the Python API

```bash
for split in pre_cutoff post_cutoff; do
  uv run household-bench-eval run-experiment \
    --experiment-config-path configs/experiments/householdbench_evalsample_${split}.yaml \
    --experiment-dir runs/my_model_${split}
done
```

The configs cover the evaluation sample's 17 pre-cutoff and 8 post-cutoff tasks and read
`{prediction_root}/{split}/<task>.jsonl`; after a full rebuild, add the other 15 tasks to
your copies. Each run writes one result JSON per task, group summaries, a
`final_result.json` and config snapshots. Unknown config fields, including metric
overrides, are rejected.

```python
from household_bench_eval import score_task

result = score_task("cons_cex_total", gold_rows, prediction_rows)
print(result["metrics"]["primary_score"])
```

[docs/tasks.md](docs/tasks.md) summarizes each task's targets, mode and metrics;
`uv run household-bench-eval describe --task <task>` prints its full answer format,
including target names and label sets.

## Repository layout

```text
src/household_bench_eval/   scoring package: task specs, parsing, metrics, CLI
tests/                      pytest suite for the scoring package
configs/experiments/        batch-scoring configs for the evaluation sample
docs/                       answer format of every task
scripts/
  run_pipeline.py           runs the whole construction pipeline
  1_preprocessing/          one builder per source survey, plus variable mappings
  2_tasks/<task>/           01_make_table.py and 02_render_prompts.py for each task
  2_tasks/select_evalsample.py   selects the paper's test prompts
  3_xgboost/                XGBoost reference models
  utils/                    shared helpers
```

## Known issues

- Two prompt templates contain wording errors that every evaluated model saw, so they
  are kept to preserve comparability: "about the same than" (SCE household-finance
  questions) and a doubled "Your household your household" (`cons_sce_growth`).
- `macro_sce_revision` record IDs in this release drop the `__m01_YYYYMM__m12_YYYYMM`
  suffix used during the paper's evaluation runs; the records are otherwise identical.

## Development

```bash
uv sync --extra pipeline
uv run pytest
uv run ruff check src tests
uv run python -m compileall -q scripts
```

CI runs these checks, plus an import of the pipeline libraries, on pull requests and
pushes to `main`.

## License

The code is released under the MIT License (see [LICENSE](LICENSE)). The data keep
their sources' terms, so no single license covers them. Each Hugging Face dataset's
`LICENSE.md` lists those terms and the attribution notices you must keep.

The prompt templates and question mappings in `scripts/` take or adapt questions from
the New York Fed Survey of Consumer Expectations (SCE), and the benchmark uses SCE data.
The SCE license requires these notices:

> Some survey questions were taken or adapted from the Survey of Consumer Expectations,
> © 2013-2019 Federal Reserve Bank of New York (FRBNY). The SCE questions are available
> without charge at <https://www.newyorkfed.org/microeconomics/sce> and may be used
> subject to license terms posted there. FRBNY did not participate in or endorse
> HouseholdBench, and FRBNY disclaims any responsibility or legal liability for the
> administration of the survey and the analysis and interpretation of data collected.
>
> Source: Survey of Consumer Expectations, © 2013-2019 Federal Reserve Bank of New York
> (FRBNY). The SCE data are available without charge at
> <https://www.newyorkfed.org/microeconomics/sce> and may be used subject to license
> terms posted there. FRBNY disclaims any responsibility or legal liability for this
> analysis and interpretation of Survey of Consumer Expectations data.

## Citation

If you find our work helpful, please cite:

```bibtex
@article{huang2026householdbench,
  title   = {{HouseholdBench}: Evaluating Large Language Models as Predictors of Household Economic Behavior},
  author  = {Huang, Jin and Ferreras Garrucho, Diego and Xie, Yutong and Yuan, Walter M. and Mei, Qiaozhu and Lian, Chen and Hazell, Jonathon},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

<!-- TODO: placeholder entry; fill in the arXiv identifier once the preprint is announced. -->
