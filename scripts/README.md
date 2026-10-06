# HouseholdBench code

These scripts construct 32 household prediction tasks, render their prompts,
assign training and test samples, and fit XGBoost baseline models. The 17 tasks we
may redistribute are on Hugging Face; see
[Getting the data](../README.md#getting-the-data).

## Setup and execution

Set up the environment once, as described in the
[repository README](../README.md#setup), with the pipeline libraries included:

```sh
uv sync --extra pipeline
```

Run every command in this guide from the repository root. Data go in a `data/`
folder there (it is ignored by git); paths below are relative to the repository
root:

```text
household_bench_eval/
  scripts/
  data/
    raw/                         supplied raw inputs, for a rebuild
    householdbench/              full benchmark (option B), if downloaded
    householdbench_evalsample/   evaluation sample (option A), if downloaded
```

1. Download the supplied raw inputs from the `raw/` folder of the full-data dataset:

   ```sh
   uv run --with huggingface_hub hf download householdbench/householdbench-full \
     --repo-type dataset --include "raw/*" --local-dir data
   ```

2. Obtain the 13 additional files listed under [Manual downloads](#manual-downloads)
   and place them at the specified paths. A full rebuild requires all 107 inputs.

3. Check the inputs and run the pipeline:

   ```sh
   uv run python scripts/run_pipeline.py --check-inputs
   uv run python scripts/run_pipeline.py
   ```

`--jobs N` sets how many source builders and tasks run at once (default: all
available CPUs; `--jobs 1` runs them in sequence). Builders that depend on others
wait for them: CPI before the macro context, and BEA, LAUS and FHFA before the state
panel. Each task writes only its own files and draws its sample from seeds derived
from its task ID, so the number of jobs does not change any output. With more than
one job, step logs are written to `output/pipeline_logs/`.

Inside the CPS builder, `scripts/1_preprocessing/cps.py --jobs N` parses and transforms
its 250,000-row chunks in N worker processes (default: available CPUs, at most 16;
about 2 GB each) and writes them in file order, so its output is the same for any N. The
CEX Diary builder takes the same option for its 30 survey years (default at most 8,
about 3 GB per worker), writing and summing the years in calendar order. The PSID and
Census builders decompress their Stata file once to a temporary file before reading it
in chunks; reading the archive directly would decompress it again for every chunk.

The three CPS tasks that link months (`labor_cps_jobfind`, `labor_cps_separation` and
`labor_cps_retire`) take `--jobs N` on their `01_make_table.py` (default: available CPUs,
at most 8; about 3 GB each). Each worker reads a block of months from the parquet file
and links them; the main process adds up the results in month order, so the tables and
diagnostics are the same for any N. `run_pipeline.py --jobs` does not change these
worker counts, so even `--jobs 1` needs about 32 GB of memory for the CPS builder.

`--check-inputs` checks the required libraries and input files without rebuilding
anything. It rejects missing files or files whose contents differ from the
reference versions. The CPS, Census and CEX Diary downloads have no stored checksum
(an empty `sha256` in the input manifest) and are checked for presence only: the
Census and ICPSR files carry the download time or the downloader's name, and IPUMS
serves only its current CPS revision. The pipeline then runs all stages and replaces derived
files. It does not download data. The pipeline was built on macOS ARM64 (where XGBoost
needs `brew install libomp`); on Linux the environment installs and the supplied inputs
pass their checksums, but a full rebuild has not yet been run there.

To use the 17 released tasks without rebuilding them, download them from Hugging
Face ([Getting the data](../README.md#getting-the-data)) into `data/householdbench/`
(option B) and `data/householdbench_evalsample/` (option A). For option B:

```sh
uv run --with huggingface_hub hf download householdbench/householdbench-full \
  --repo-type dataset --include "householdbench/*" --local-dir data
```

The full benchmark holds completed tables, prompts, splits and baseline predictions;
the evaluation sample holds 3,200 pre-cutoff prompts across 17 tasks and 1,500
post-cutoff prompts across eight tasks, and can be used on its own. Keep the source
notices in each dataset's `LICENSE.md`.

To regenerate the evaluation sample from `data/householdbench/`, run:

```sh
uv run python scripts/2_tasks/select_evalsample.py
```

The selector needs completed prompts, splits and their manifests, but no raw data
or fitted models. It processes the tasks in `data/householdbench/manifest.csv`:
17 in the Hugging Face release, or 32 after a complete reconstruction.
`run_pipeline.py` also calls it automatically after publishing the splits. A
complete reconstruction yields 6,079 pre-cutoff and 2,500 post-cutoff evaluation
prompts.

The default selection reproduces the paper's evaluation sample: up to 200
observations per task and split. Pre-cutoff selection uses the first 200 stored
split positions. For post-cutoff selection, the script first removes releases on
or before 16 February 2026 and then rebalances the remaining quarter pools using
the existing within-quarter random order. The underlying benchmark splits retain
their 31 January cutoff; early-February observations are not moved into the
pre-cutoff evaluation. Changing `ROWS_PER_TASK` or `POST_CUTOFF_DATE` at the top
of the selector produces a different sample.

Outputs are `data/householdbench_evalsample/pre_cutoff/<task_id>.jsonl` and
`post_cutoff/<task_id>.jsonl`, plus a manifest with counts, selection settings,
source fingerprints and output fingerprints. Empty cells have no prompt file.
Records retain their original file order and JSONL bytes, including reference
answers and naive baselines. Send only `system` and `user` to a model. To
reproduce the paper's results on rows released in or after July 2026, keep the
post-cutoff rows with `release_date >= "2026-07-01"`; do not draw a new sample.

## Code and outputs

| Location | Purpose |
|---|---|
| `scripts/run_pipeline.py` | Runs the complete workflow in order |
| `scripts/1_preprocessing/` | Reads raw sources and constructs cleaned data |
| `scripts/2_tasks/<task_id>/01_make_table.py` | Constructs a task's table |
| `scripts/2_tasks/<task_id>/02_render_prompts.py` | Renders that table as prompts |
| `scripts/2_tasks/select_evalsample.py` | Selects the paper's evaluation prompts from completed tasks |
| `scripts/3_xgboost/build.py` | Fits baseline models and writes predictions |
| `scripts/utils/` | Shared functions used by the stages |
| `data/raw/` | Source inputs |
| `data/intermediate/` | Cleaned sources and sampling records |
| `data/householdbench/` | Task tables, prompts, splits and baseline predictions |
| `data/householdbench_evalsample/` | Selected pre-cutoff and post-cutoff evaluation prompts |
| `output/` | Construction summaries, model files and diagnostics |

The main script runs preprocessing, task construction, split assignment,
evaluation-sample selection and baseline fitting. Preprocessing follows this order:

```text
CPI → macro context → BEA → LAUS → FHFA → state macro panel →
CEX Interview → CEX Diary → CPS Basic/DWS → UI policy → Census →
PSID → Michigan → SCE → SCE financing
```

To run an individual stage after its required inputs have been constructed:

```sh
uv run python scripts/1_preprocessing/<dataset>.py
uv run python scripts/2_tasks/<task_id>/01_make_table.py
uv run python scripts/2_tasks/<task_id>/02_render_prompts.py
uv run python scripts/3_xgboost/build.py --threads 4
```

Replace `<dataset>` and `<task_id>` with the corresponding script or task name.
Individual table and prompt scripts do not update the task manifest or sample
splits. Use `run_pipeline.py` to rebuild all outputs together. Baseline fitting
uses four threads by default, set by `XGBOOST_THREADS` in that script.

The following files define inputs and model settings:

| File (paths relative to `scripts/`) | Contents |
|---|---|
| [../pyproject.toml](../pyproject.toml) | Python version and pinned libraries (the `pipeline` extra) |
| [1_preprocessing/input_manifest.csv](1_preprocessing/input_manifest.csv) | Input paths, versions and checksums; `kind` identifies supplied or separately downloaded files |
| [2_tasks/task_registry.json](2_tasks/task_registry.json) | Task definitions, predictors, outcomes and sampling choices |
| [3_xgboost/model_seeds.csv](3_xgboost/model_seeds.csv) | Random seeds for baseline models |
| [output_manifest.csv](output_manifest.csv) | Output paths and the scripts that write them |

Source-variable mappings are in `scripts/1_preprocessing/mappings/`. The sampling
settings are in `scripts/utils/sampling.py`: seed 42, release-date cutoff
31 January 2026, and training/validation/test shares of 8:1:1 before the cutoff.
Up to 500,000 observations per task are retained before the cutoff; all eligible
observations after it are retained. XGBoost uses at most 400,000 training and
50,000 validation observations per task. Baseline CSVs contain `actual_answer`,
`naive_baseline` and `xgb_400k_tuned_direct`.

## Inputs

### Provided inputs

The `raw/` folder of the full-data dataset supplies 94 files: BLS CEX Interview and 1990–2011 Diary data;
SCE data and the Fuster–Zafar financing experiment; Farber–Rothstein–Valletta
UI-duration data; BLS, BEA and FHFA macroeconomic data; and six release-date
crosswalks. Download it into `data/` as shown above. Keep the supplied versions,
including files with `latest` in their names. Source descriptions and licenses
are in each Hugging Face dataset's `LICENSE.md`.

### Manual downloads

The remaining 13 files must be obtained from their providers:

| Source | Files | Instructions |
|---|---:|---|
| ICPSR CEX Diary, 1982–1989 | 7 | [CEX Diary](#cex-diary) |
| IPUMS CPS Basic/DWS | 1 | [CPS](#cps-basic-and-dws) |
| IPUMS Census | 1 | [Census](#census) |
| Michigan Surveys of Consumers | 1 | [Michigan](#michigan) |
| PSID-SHELF V2 | 2 | [PSID-SHELF](#psid-shelf) |
| FRED-QD, July 2026 | 1 | [FRED-QD](#fred-qd) |

Follow each provider's access and use terms. Download the versions below and
preserve file contents and ZIP member paths. New extracts can differ because of
source revisions or record ordering. If `--check-inputs` reports a mismatch,
check the requested version, variables and format; changing the stored checksum
does not make a different input equivalent.

### CEX Diary

From the [ICPSR Consumer Expenditure series](https://www.icpsr.umich.edu/web/ICPSR/series/20),
download the following **version 1, original fixed-width ASCII** packages.
Keep each complete ZIP under `data/raw/micro/cex/diary/icpsr/`:

| Survey years | Study | Filename |
|---|---|---|
| 1982–1983 | [8599](https://www.icpsr.umich.edu/web/ICPSR/studies/8599) | `ICPSR_08599-V1.zip` |
| 1984 | [8628](https://www.icpsr.umich.edu/web/ICPSR/studies/8628) | `ICPSR_08628-V1.zip` |
| 1985 | [8905](https://www.icpsr.umich.edu/web/ICPSR/studies/8905) | `ICPSR_08905-V1.zip` |
| 1986 | [9114](https://www.icpsr.umich.edu/web/ICPSR/studies/9114) | `ICPSR_09114-V1.zip` |
| 1987 | [9333](https://www.icpsr.umich.edu/web/ICPSR/studies/9333) | `ICPSR_09333-V1.zip` |
| 1988 | [9570](https://www.icpsr.umich.edu/web/ICPSR/studies/9570) | `ICPSR_09570-V1.zip` |
| 1989 | [9714](https://www.icpsr.umich.edu/web/ICPSR/studies/9714) | `ICPSR_09714-V1.zip` |

Retain all family, member, expenditure and diary-week files, the manifest,
`*-descriptioncitation.html` and `*-Documentation.txt`, with their original
member paths. Construction requires the full 1982–2011 panel; the task selects
1986–1996 downstream. The 1990–2011 BLS packages are supplied.

### CPS Basic and DWS

At [IPUMS CPS](https://cps.ipums.org/cps/), create a Basic Monthly extract for
**January 1976–June 2026**, excluding October 2025, with DWS variables included.
The reference is version 13.0, `20260814 Basic refresh`. Select rectangular person
records, numeric CSV output and no case selection or subsampling; exclude ASEC.
Retain these 149 columns in order, including provider-added fields:

```text
YEAR SERIAL MONTH HWTFINL CPSID ASECFLAG MISH NUMPREC HHTENURE GQTYPE HHINTYPE REGION
STATEFIP COUNTY METFIPS METRO CBSASZ FAMINC MARBASECIDH HRHHID HRHHID2 HUHHNUM HRSAMPLE
HHRESPLN INTTYPE PERNUM WTFINL CPSIDP CPSIDV UCAPLAST UCNOMAIN UCNORECLAST UCNORECWK
UCRECLAST UCRECWK UCUNION UCWHYNOELG UCSUPFLG EARNWEEK2 HOURWAGE2 SUBMINWAGE RELATE AGE
SEX RACE MARST POPSTAT ASIAN VETSTAT SPLOC SPRULE FAMSIZE NCHILD NCHLT5 FAMUNIT FTYPE
FAMREL BPL YRIMMIG CITIZEN NATIVITY HISPAN EMPSTAT LABFORCE OCC OCC2010 OCC1990 IND1990
OCC1950 IND IND1950 CLASSWKR UHRSWORKT UHRSWORK1 UHRSWORK2 AHRSWORKT AHRSWORK1 AHRSWORK2
ABSENT DURUNEM2 DURUNEMP WHYUNEMP WHYABSNT WHYPTLWK WNFTLOOK WNLOOK WKSTAT EMPSAME MULTJOB
NUMJOB PAIDEMP1 PAIDEMP1N PAIDEMP2 PAIDEMP2N PROFCERT STATECERT JOBCERT WRKOFFER NILFACT
ACTSAME EDUC EDUC99 EDCYC EDDIPGED EDHGCGED SCHLCOLL DIFFHEAR DIFFEYE DIFFREM DIFFPHYS
DIFFMOB DIFFCARE DIFFANY MARBASECIDP DWLOSTJOB DWSTAT DWREAS DWRECALL DWNOTICE DWLASTWRK
DWYEARS DWFULLTIME DWWEEKL DWWAGEL DWUNION DWBEN DWEXBEN DWHI DWCLASS DWIND DWIND1990
DWOCC DWOCC1990 DWMOVE DWHINOW DWJOBSINCE DWWEEKC DWWAGEC DWHRSWKC DWRESP DWWKSUN HOURWAGE
PAIDHOUR UNION EARNWEEK UHRSWORKORG WKSWORKORG ELIGORG OTPAY
```

Save the compressed CSV as `data/raw/micro/cps/cps_basic.csv.gz` without resaving
its contents. The supplied CPS mappings preserve types, missing codes and labels;
the text codebook is not required at runtime.

### Census

At [IPUMS USA](https://usa.ipums.org/usa/), select **1990 1% and 2000 1%**.
Choose rectangular person records, full sample density, no case selection,
and Stata output with embedded value labels. Save the compressed extract as
`data/raw/micro/census/census.gz` and its codebook as
`data/raw/micro/census/census_codebook.txt`. Include these 262 variables in order:

```text
YEAR SAMPLE SERIAL NUMPREC SUBSAMP HHWT HHTYPE CLUSTER CPI99 REGION STATEICP
STATEFIP COUNTYICP COUNTYFIP PUMA PUMASUPR METRO METAREA METAREAD CITY CITYERR
CITYPOP SIZEPL URBAN STRATA PUMATYPE PUMATY00 PUMALAND PUMAAREA CNTRY GQ GQTYPE
GQTYPED FARM OWNERSHP OWNERSHPD MORTGAGE MORTGAG2 COMMUSE FARMPROD ACREPROP
ACREHOUS MORTAMT1 MORTAMT2 TAXINCL INSINCL PROPINSR PROPTX99 OWNCOST RENT
RENTGRS RENTMEAL CONDO CONDOFEE MOBLHOME MOBLHOM2 MOBLOAN COSTELEC COSTGAS
COSTWATR COSTFUEL HHINCOME VALUEH LINGISOL VACANCY VACELSE VACBOARD VACDUR
KITCHEN ROOMS PLUMBING BUILTYR BUILTYR2 UNITSSTR WATERSRC SEWAGE BEDROOMS PHONE
FUELHEAT VEHICLES NFAMS NSUBFAM NCOUPLES NMOTHERS NFATHERS MULTGEN MULTGEND
CBNSUBFAM PERNUM PERWT SLWT FAMUNIT FAMSIZE SUBFAM SFTYPE SFRELATE CBSUBFAM
CBSFTYPE CBSFRELATE MOMLOC MOMRULE POPLOC POPRULE SPLOC SPRULE MOMLOC2 MOM2RULE
POPLOC2 POP2RULE NCHILD NCHLT5 NSIBS ELDCH YNGCH RELATE RELATED SEX AGE MARST
BIRTHYR CHBORN RACE RACED HISPAN HISPAND BPL BPLD ANCESTR1 ANCESTR1D ANCESTR2
ANCESTR2D CITIZEN YRIMMIG YRSUSA1 YRSUSA2 LANGUAGE LANGUAGED SPEAKENG TRIBE
TRIBED RACHSING PREDAI PREDAPI PREDBLK PREDWHT PREDHISP RACAMIND RACASIAN RACBLK
RACPACIS RACWHT RACOTHER RACNUM SCHOOL EDUC EDUCD GRADEATT GRADEATTD SCHLTYPE
EMPSTAT EMPSTATD LABFORCE CLASSWKR CLASSWKRD OCC OCC1950 OCC1990 OCC2010 OCCSOC
IND IND1950 IND1990 INDNAICS WKSWORK1 WKSWORK2 HRSWORK1 HRSWORK2 UHRSWORK
YRLASTWK WRKLSTWK ABSENT LOOKING AVAILBLE WRKRECAL WORKEDYR INCTOT FTOTINC
INCWAGE INCBUS INCBUS00 INCFARM INCSS INCWELFR INCINVST INCRETIR INCSUPP
INCOTHER INCEARN POVERTY OCCSCORE SEI HWSEI PRESGL PRENT ERSCOR50 ERSCOR90
EDSCOR50 EDSCOR90 NPBOSS50 NPBOSS90 MIGRATE5 MIGRATE5D MIGPLAC5 MIGPUMA5
MIGSPUMA5 MIGMETAREA5 MIGMETRO5 MIGMETROD5 MIGCITY5 MOVEDIN DISABWRK DIFFREM
DIFFPHYS DIFFMOB DIFFCARE DIFFSENS VETSTAT VETSTATD VET95X00 VET90X95 VET75X90
VET80X90 VET75X80 VETVIETN VET55X64 VETKOREA VETWWII VETOTHER VETOTHERD VETYRS
PWSTATE2 PWMETAREA PWCITY PWMETSTAT PWMETSTATD PWPUMA PWSPUMA TRANWORK CARPOOL
RIDERS TRANTIME DEPARTS GCHOUSE GCMONTHS GCRESPON RACESING RACESINGD PROBAI
PROBAPI PROBBLK PROBOTH PROBWHT
```

The samples have codes `199002` and `200007`. The builder uses the supplied
availability and missing-value mappings and the embedded Stata labels. The text
codebook is for reference and is not required at runtime.

### Michigan

Open the [Surveys of Consumers Cross-Section Archive](https://data.sca.isr.umich.edu/sda.php)
and follow **Cross-Section Archive** to SDA. Request **January 1978–February 2026**,
numeric codes, no demographic restrictions or subsampling, and these variables:

```text
CASEID YYYYMM YYYYQ YYYY ID IDPREV DATEPR IDPREV2 DATEPR2 SAMPLE METHOD ICS ICC ICE PAGO
PAGOR1 PAGOR2 PAGO5 PEXP PEXP5 INEXQ1 INEXQ2 INEX RINC BAGO BEXP BUS12 BUS5 NEWS1 NEWS2
UNEMP GOVT RATEX PX1Q1 PX1Q2 PX1 PX5Q1 PX5Q2 PX5 DUR DURRN1 DURRN2 HOM HOMRN1 HOMRN2 SHOM
SHOMRN1 SHOMRN2 CAR CARRN1 CARRN2 INCOME INCQFM YTL10 YTL90 YTL50 YTL5 YTL4 YTL3 HOMEOWN
HOMEAMT HOMEQFM HTL10 HTL90 HTL50 HTL5 HTL4 HTL3 HOMEVAL HOMPX1Q1 HOMPX1Q2 HOMPX1 HOMPX5Q1
HOMPX5Q2 HOMPX5 INVEST INVAMT INVQFM STL10 STL90 STL50 STL5 STL4 STL3 AGE BIRTHM BIRTHY
REGION SEX MARRY NUMKID NUMADT EDUC ECLGRD EHSGRD EGRADE EDUCATION POLAFF POLREP POLDEM
POLCRD VEHOWN VEHNUM GASPX1 GASPX2 GAS5 GAS1PX1 GAS1PX2 GAS1 PINC PINC2 PJOB PSSA PCRY
PSTK WT WT_HH WT_QUINTMED
```

Save the comma-delimited file as `data/raw/micro/mich/mich.csv`. Retain the
previous-interview identifiers. The supplied mappings preserve labels and missing
codes from the reference dictionary, dated 13 August 2026.

### PSID-SHELF

From [study 194322, version 2](https://doi.org/10.3886/E194322V2), published
24 February 2025, download these files under `data/raw/micro/psid-shelf/`:

```text
PSIDSHELF_1968_2021_LONG_7.9_GB.zip
Construction_Files/Data/PSID_COMPLETE_MAIN_STUDY_1968_2021_FULL_10.8_GB.zip
```

Keep both ZIPs compressed and preserve the folders. Their data members are
`PSIDSHELF_1968_2021_LONG.dta` and `PSID_COMPLETE_MAIN_STUDY_1968_2021_FULL.dta`.
Both archives are required; the builder reads them directly without a separate
extraction step.

### FRED-QD

On the [FRED-MD/FRED-QD page](https://www.stlouisfed.org/research/economists/mccracken/fred-databases),
select **FRED-QD: Quarterly Data → 2026-07.csv** and save it as
`data/raw/macro/fred-qd/2026-07-QD.csv`. Keep the metadata rows. Use the July
quarterly vintage, not FRED-MD, `current.csv` or a newer file. Construction reads
`GDPC1`, `UNRATE`, `CPIAUCSL`, `FEDFUNDS`, `USSTHPI`, `S&P 500` and `MORTGAGE30US`.
