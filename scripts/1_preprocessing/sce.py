#!/usr/bin/env python
"""Build first-pass cleaned NY Fed SCE respondent-level microdata tables."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.source_inputs import validate_source_inputs

from scripts.utils.io import exclusive_lock, sha256_file

RAW_DIR = Path("data/raw/micro/sce")
INTERMEDIATE_DIR = Path("data/intermediate")
OUT_DIR = Path("output/preprocessing/sce/build")
RELEASE_DATE_CROSSWALK_PATH = Path("data/raw/release_dates/release_sce.csv")
RELEASE_DATE_COVERAGE_PATH = Path("output/preprocessing/release_dates/metadata_coverage.csv")
PIPELINE_VERSION = "0.1.0"

ROW_LEVEL_RELEASE_STATUS = "inferred_from_collection_month_and_official_lag"
MICRODATA_RELEASE_LAG_MONTHS = {
    "core": 9,
    "household_spending": 18,
    "labor_market": 18,
    "public_policy": 9,
    "housing": 18,
    "job_search": 9,
}
SOURCE_URLS = {
    "core": "https://www.newyorkfed.org/microeconomics/sce",
    "household_spending": "https://www.newyorkfed.org/microeconomics/sce/household-spending",
    "labor_market": "https://www.newyorkfed.org/microeconomics/sce/labor-market",
    "public_policy": "https://www.newyorkfed.org/microeconomics/sce/public-policy",
    "housing": "https://www.newyorkfed.org/microeconomics/sce/housing",
    "job_search": "https://www.newyorkfed.org/microeconomics/databank.html",
}

MAPPING_DIR = Path(__file__).resolve().parent / "mappings"
ADDITIONAL_MICRODATA_FILES = [
    "housing/FRBNY-SCE-Housing-Survey-Public-Microdata-Complete.xlsx",
    "job_search/SCE-Public-LM-Quarterly-Microdata.xlsx",
]

MICRODATA_SPECS = [
    {
        "module": "core",
        "table": "core",
        "relative_path": "core/FRBNY-SCE-Public-Microdata-Complete-13-16.xlsx",
        "sheet": "Data",
        "header": 1,
        "notes": "Core public microdata split covering 2013-2016.",
    },
    {
        "module": "core",
        "table": "core",
        "relative_path": "core/FRBNY-SCE-Public-Microdata-Complete-17-19.xlsx",
        "sheet": "Data",
        "header": 1,
        "notes": "Core public microdata split covering 2017-2019.",
    },
    {
        "module": "core",
        "table": "core",
        "relative_path": "core/frbny-sce-public-microdata-20-24.xlsx",
        "sheet": "Data",
        "header": 1,
        "notes": "Core public microdata split covering 2020-2024.",
    },
    {
        "module": "core",
        "table": "core",
        "relative_path": "core/frbny-sce-public-microdata-latest.xlsx",
        "sheet": "Data",
        "header": 1,
        "notes": "Current core public microdata split covering 2025 onward in this download.",
    },
    {
        "module": "household_spending",
        "table": "household_spending",
        "relative_path": "household_spending/sce-household-spending-microdata.xlsx",
        "sheet": "Data",
        "header": 1,
        "notes": "Household Spending respondent-level public microdata; no weight column is present.",
    },
    {
        "module": "labor_market",
        "table": "labor_market",
        "relative_path": "labor_market/sce-labor-microdata-public.xlsx",
        "sheet": "Data",
        "header": 1,
        "notes": "Labor Market respondent-level public microdata; no weight column is present.",
    },
    {
        "module": "public_policy",
        "table": "public_policy",
        "relative_path": "public_policy/sce-policy-public-microdata.xlsx",
        "sheet": "Data",
        "header": 10,
        "notes": "Public Policy respondent-level public microdata; data header follows title notes.",
    },
]

TABLE_OUTPUTS = {
    "core": "sce_core.parquet",
    "household_spending": "sce_household_spending.parquet",
    "labor_market": "sce_labor_market.parquet",
    "public_policy": "sce_public_policy.parquet",
    "housing": "sce_housing.parquet",
    "job_search": "sce_job_search.parquet",
}

TASK_USE_BY_MODULE = {
    "core": "macro expectations; personal finances; income expectations",
    "household_spending": "consumption, saving, and large-purchase expectations",
    "labor_market": "labor-market dynamics",
    "public_policy": "policy expectations and hypothetical policy impacts",
    "housing": "housing expectations, mobility expectations, and housing-balance-sheet variables",
    "job_search": "job search, employment status, wage offers, benefits, and labor-market dynamics",
}

JOB_SEARCH_DOCUMENTED_SPECIAL_CODES = {
    "ec1a_cps_job_employer_type_rc": {999999: "dont_know_or_remember"},
    "ec4b_cps_job_temp_duration": {999992: "less_than_one_month"},
    "ec5_cps_job_commute_time": {999996: "mostly_works_from_home"},
    "el2a_last_job_employer_type_rc": {999999: "dont_know_or_remember"},
    "el6_last_job_commute_time": {999996: "mostly_works_from_home"},
    "es9_selfemp_commute_time": {999996: "mostly_works_from_home"},
    "hh4_spouse_earn_hrly": {999999: "dont_know_or_remember"},
    "jh10_prev_job_earn_hrly": {999999: "dont_know_or_remember"},
    "jh10s_prev_job_earn_hrly": {999999: "dont_know_or_remember"},
    "jh13_cpsj_wks_spent_search": {
        999993: "opportunity_without_search",
        999999: "dont_know_or_remember",
    },
    "jh13b_cpsj_wks_btwn_jobs": {
        999994: "started_immediately",
        999999: "dont_know_or_remember",
    },
    "jh14_cpsj_applications": {999999: "dont_know_or_remember"},
    "jh15_cpsj_contacts": {999999: "dont_know_or_remember"},
    "jh15b_cpsj_contacts_unsolicited": {999999: "dont_know_or_remember"},
    "jh15c_cpsj_contacts_referral": {999999: "dont_know_or_remember"},
    "jh16_cpsj_offers": {999999: "dont_know_or_remember"},
    "jh1c_cpsj_refer_occup": {999999: "dont_know_or_remember"},
    "jh2b_cpsj_wks_btwn_jobs": {999999: "dont_know_or_remember"},
    "jh2bs_semp_wks_btwn_jobs": {999999: "dont_know_or_remember"},
    "jh2d_cpsj_adv_notice_mos": {999999: "dont_know_or_remember"},
    "jh2ds_semp_adv_notice_mos": {999999: "dont_know_or_remember"},
    "jh5bs_semp_usual_hrs": {999999: "dont_know_or_remember"},
    "jh9_cps_job_start_earn_hrly": {999999: "dont_know_or_remember"},
    "js25c_referral_occup": {999999: "dont_know_or_remember"},
    "js28_accept_start_time": {999997: "already_started"},
    "l8_months_no_work": {999995: "never_had_paid_job"},
    "rw3b_pct_incr_relocation": {999998: "would_not_accept"},
    "rw4b_pct_incr_commute": {999998: "would_not_accept"},
    "rw6b_pct_incr_hours": {999998: "would_not_accept"},
    "rw7b_pct_incr_healthins": {999998: "would_not_accept"},
}
JOB_SEARCH_SPECIAL_CODE_SOURCE = (
    "NY Fed SCE Labor Market Survey Data Codebook, Job Search public microdata special-value definitions."
)

STRUCTURAL_COLUMNS = {
    "source",
    "release_id",
    "release_date",
    "aggregate_release_date_proxy",
    "microdata_release_lag_months",
    "release_date_status",
    "release_date_source",
    "module",
    "raw_file",
    "respondent_id",
    "survey_year",
    "survey_month",
    "survey_date",
    "panel_month",
    "weight_core",
    "weight_household_spending",
    "weight_labor_market",
    "weight_public_policy",
    "weight_housing",
    "weight_job_search",
    "missing_core_weight",
    "source_wave_year",
    "wave_number",
    "wave_label",
    "source_wave_date",
}

POLICY_ITEMS = {
    1: "gasoline tax",
    2: "federal student loan debt forgiveness",
    3: "federal student aid/Pell grants",
    4: "public college tuition in respondent state",
    5: "federal welfare benefits",
    6: "unemployment benefits",
    7: "federal minimum wage",
    8: "state minimum wage",
    9: "payroll tax rate",
    10: "mortgage interest tax deduction",
    11: "capital gains tax rate",
    12: "affordable housing/housing assistance",
    13: "cost of public transportation",
    14: "free/subsidized public preschool education",
    15: "mandatory paid parental leave",
    16: "average income tax rate",
    17: "income tax rate for highest income bracket",
    18: "retirement age to claim social security benefits",
    19: "social security retirement benefits",
    20: "Medicare benefits",
}

CORE_PROBABILITY_COLUMNS = {
    "Q4new",
    "Q5new",
    "Q6new",
    "ES3new",
    "Q13new",
    "Q14new",
    "Q17new",
    "Q18new",
    "Q20new",
    "Q21new",
    "Q22new",
    "Q30new",
}

DENSITY_CONCEPTS = {
    "Q9": ("one-year ahead inflation density", "macro expectations"),
    "Q9c": ("three-year ahead inflation density", "macro expectations"),
    "Q24": ("one-year ahead earnings-growth density", "income expectations"),
    "C1": ("one-year ahead nationwide home-price density", "housing expectations"),
    "qsp7dens": ("expected household-spending-growth density", "consumption and saving"),
}

TASK_KEY_VARS_BY_MODULE = {
    "core": {
        "Q9_mean": ("one-year ahead expected inflation density mean", "macro expectations"),
        "Q9c_mean": ("three-year ahead expected inflation density mean", "macro expectations"),
        "Q13new": ("probability of higher unemployment", "labor-market dynamics"),
        "Q14new": ("probability of losing job", "labor-market dynamics"),
        "Q17new": ("probability of finding job", "labor-market dynamics"),
        "Q18new": ("probability of leaving job voluntarily", "labor-market dynamics"),
        "Q22new": ("probability of missing debt payment", "financial fragility"),
        "Q30new": ("probability of higher taxes", "personal finances"),
        "Q24_mean": ("expected earnings-growth density mean", "income expectations"),
        "C1_mean": ("expected nationwide home-price density mean", "housing expectations"),
    },
    "household_spending": {
        "qsp2": ("realized household spending change", "consumption and saving"),
        "qsp12a_1": ("saving or financial behavior share", "consumption and saving"),
        "qsp12a_2": ("saving or financial behavior share", "consumption and saving"),
        "qsp12a_3": ("saving or financial behavior share", "consumption and saving"),
        "qsp13a_1": ("saving or financial behavior share", "consumption and saving"),
        "qsp13a_2": ("saving or financial behavior share", "consumption and saving"),
        "qsp13a_3": ("saving or financial behavior share", "consumption and saving"),
    },
    "labor_market": {
        "l3": ("annual earnings at current job", "labor-market dynamics"),
        "js5": ("job-search expectation or experience", "labor-market dynamics"),
        "js6": ("job-search expectation or experience", "labor-market dynamics"),
        "rw2a": ("reservation wage or wage offer", "labor-market dynamics"),
        "rw2b": ("reservation wage or wage offer", "labor-market dynamics"),
        "q113": ("labor-market probability variable", "labor-market dynamics"),
        "q115": ("labor-market probability variable", "labor-market dynamics"),
        "q121": ("labor-market probability variable", "labor-market dynamics"),
        "q122": ("labor-market probability variable", "labor-market dynamics"),
    },
    "public_policy": {
        "qp2_1": ("expected impact of gasoline tax change", "policy expectations"),
        "qp2_7": ("expected impact of federal minimum wage change", "policy expectations"),
        "qp2_16": ("expected impact of average income tax change", "policy expectations"),
    },
    "housing": {
        "HQ1_1": ("current typical local home value", "housing expectations"),
        "q1_1": ("current typical local home value", "housing expectations"),
        "HQ38": ("housing as investment assessment", "housing expectations"),
        "hq38": ("housing as investment assessment", "housing expectations"),
    },
    "job_search": {
        "l1a_lfs_rc": ("employment status", "labor-market dynamics"),
        "l1_lfs_rc": ("nonemployment status", "labor-market dynamics"),
        "l2_num_jobs": ("number of jobs", "labor-market dynamics"),
        "l7_days_spent_searching": ("job-search duration in days", "job search"),
        "l11_cps_job_earn_ann": ("annual earnings at current/main job", "labor-market dynamics"),
    },
}

CORE_FINANCE_CHANGE_LABELS = {
    1: "much worse off",
    2: "somewhat worse off",
    3: "about the same",
    4: "somewhat better off",
    5: "much better off",
}
CORE_HEALTH_LABELS = {
    1: "excellent",
    2: "very good",
    3: "good",
    4: "fair",
    5: "poor",
}
CORE_DETAILED_HOUSEHOLD_INCOME_LABELS = {
    1: "Less than $10,000",
    2: "$10,000 to $19,999",
    3: "$20,000 to $29,999",
    4: "$30,000 to $39,999",
    5: "$40,000 to $49,999",
    6: "$50,000 to $59,999",
    7: "$60,000 to $74,999",
    8: "$75,000 to $99,999",
    9: "$100,000 to $149,999",
    10: "$150,000 to $199,999",
    11: "$200,000 or more",
}
YES_NO_1_2_LABELS = {1: "yes", 2: "no"}
YES_NO_0_1_LABELS = {0: "no", 1: "yes"}

MANUAL_CATEGORICAL_LABELS_BY_MODULE = {
    "core": {
        "_AGE_CAT": {"Under 40": "Under 40", "40 to 60": "40 to 60", "Over 60": "Over 60"},
        "_NUM_CAT": {"High": "High", "Low": "Low"},
        "_REGION_CAT": {"Midwest": "Midwest", "Northeast": "Northeast", "South": "South", "West": "West"},
        "_EDU_CAT": {"High School": "High School", "Some College": "Some College", "College": "College"},
        "_HH_INC_CAT": {"Under 50k": "Under 50k", "50k to 100k": "50k to 100k", "Over 100k": "Over 100k"},
        "_HH_INC_DETAILED": CORE_DETAILED_HOUSEHOLD_INCOME_LABELS,
        "Q1": CORE_FINANCE_CHANGE_LABELS,
        "Q2": CORE_FINANCE_CHANGE_LABELS,
        "Q10_1": YES_NO_0_1_LABELS,
        "Q10_2": YES_NO_0_1_LABELS,
        "Q10_3": YES_NO_0_1_LABELS,
        "Q10_4": YES_NO_0_1_LABELS,
        "Q10_5": YES_NO_0_1_LABELS,
        "Q12new": {1: "work for someone else", 2: "are self-employed"},
        "Q15": YES_NO_1_2_LABELS,
        "Q43": {1: "owns the home", 2: "rents the home", 3: "has another arrangement"},
        "Q44": YES_NO_1_2_LABELS,
        "D3": YES_NO_1_2_LABELS,
        "Q45b": CORE_HEALTH_LABELS,
    },
    "household_spending": {
        "k2e": YES_NO_1_2_LABELS,
        "qsp1": {1: "higher", 2: "lower"},
        "qsp12n": {
            1: "save or invest all",
            2: "spend or donate all",
            3: "pay down debt all",
            4: "save or invest and spend or donate",
            5: "save or invest and pay down debt",
            6: "spend or donate and pay down debt",
            7: "save or invest, spend or donate, and pay down debt",
        },
        "qsp13new": {
            1: "reduce spending all",
            2: "reduce saving all",
            3: "increase borrowing all",
            4: "reduce spending and reduce saving",
            5: "reduce spending and increase borrowing",
            6: "reduce saving and increase borrowing",
            7: "reduce spending, reduce saving, and increase borrowing",
        },
    },
    "labor_market": {
        "nl2b_1": {1: "full-time", 2: "part-time"},
        "nl3_1": {
            1: "accepted",
            2: "accepted",
            3: "rejected",
            4: "still deciding",
        },
        "rw2b": {1: "hour", 2: "week", 3: "two weeks", 4: "month", 5: "year"},
        "rw2b__clean": {1: "hour", 2: "week", 3: "two weeks", 4: "month", 5: "year"},
    },
    "public_policy": {
        "hidqp2_5": {1: "increase_or_expansion", 2: "decrease_or_reduction"},
        "hidqp2_6": {1: "increase_or_expansion", 2: "decrease_or_reduction"},
        "hidqp2_9": {1: "increase_or_expansion", 2: "decrease_or_reduction"},
        "hidqp2_16": {1: "increase_or_expansion", 2: "decrease_or_reduction"},
        "qp2_5": {
            1: "very_negative",
            2: "somewhat_negative",
            3: "no_impact",
            4: "somewhat_positive",
            5: "very_positive",
        },
        "qp2_6": {
            1: "very_negative",
            2: "somewhat_negative",
            3: "no_impact",
            4: "somewhat_positive",
            5: "very_positive",
        },
        "qp2_9": {
            1: "very_negative",
            2: "somewhat_negative",
            3: "no_impact",
            4: "somewhat_positive",
            5: "very_positive",
        },
        "qp2_16": {
            1: "very_negative",
            2: "somewhat_negative",
            3: "no_impact",
            4: "somewhat_positive",
            5: "very_positive",
        },
    },
    "housing": {
        "hq38": YES_NO_1_2_LABELS,
        "qh0": YES_NO_1_2_LABELS,
        "qh4a": YES_NO_1_2_LABELS,
    },
    "job_search": {
        "rw3_accept_relocation": YES_NO_0_1_LABELS,
        "rw4_accept_commute": YES_NO_0_1_LABELS,
        "rw6_accept_hours": YES_NO_0_1_LABELS,
        "rw7_accept_healthins": YES_NO_0_1_LABELS,
    },
}

TASK_METADATA_CATEGORICAL_COLUMNS_BY_MODULE = {
    "housing": {
        "HQ38",
        "hq38",
        "q38",
        "HQH0",
        "qh0",
        "HQH4a",
        "qh4a",
        "HQH5",
        "qh5",
        "HQH5m",
        "qh5m",
    },
}


def relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def infer_module(path: Path, raw_dir: Path) -> str:
    rel = path.relative_to(raw_dir)
    if len(rel.parts) == 1:
        return "root"
    return rel.parts[0]


def excel_sheet_names(path: Path) -> tuple[str, str]:
    if path.suffix.lower() != ".xlsx":
        return "", "sheet inventory skipped for non-xlsx file"
    try:
        workbook = load_workbook(path, read_only=True, data_only=True)
        sheets = "; ".join(workbook.sheetnames)
        workbook.close()
        return sheets, ""
    except Exception as exc:
        return "", f"could not read workbook sheets: {type(exc).__name__}: {exc}"


def build_source_inventory(raw_dir: Path) -> pd.DataFrame:
    rows = []
    required = set(ADDITIONAL_MICRODATA_FILES) | {spec["relative_path"] for spec in MICRODATA_SPECS}
    for path in sorted(raw_dir / relative_path for relative_path in required):
        module = infer_module(path, raw_dir)
        sheet_names, sheet_note = excel_sheet_names(path)
        rows.append(
            {
                "source": "sce",
                "module": module,
                "relative_path": relpath(path, raw_dir),
                "file_name": path.name,
                "suffix": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "download_url": SOURCE_URLS.get(module, "https://www.newyorkfed.org/microeconomics/databank.html"),
                "excel_sheets": sheet_names,
                "inventory_note": sheet_note,
            }
        )
    return pd.DataFrame(rows)


def clean_column_names(columns: pd.Index) -> list[str]:
    out = []
    for col in columns:
        text = str(col).strip()
        if not text or text.lower() == "nan":
            raise ValueError("Encountered blank column name in SCE workbook")
        out.append(text)
    duplicates = pd.Series(out).value_counts()
    duplicates = duplicates[duplicates > 1]
    if not duplicates.empty:
        raise ValueError(f"Duplicate column names after stripping: {duplicates.index.tolist()}")
    return out


def parse_survey_date(series: pd.Series, table_name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    bad_numeric = series.notna() & numeric.isna()
    if bad_numeric.any():
        examples = series[bad_numeric].head(5).tolist()
        raise ValueError(f"{table_name}: nonnumeric date values: {examples}")

    if numeric.isna().any():
        raise ValueError(f"{table_name}: missing date values are not allowed in respondent rows")

    date_int = numeric.astype("int64")
    year = date_int // 100
    month = date_int % 100
    invalid = (year < 1900) | (month < 1) | (month > 12)
    if invalid.any():
        examples = date_int[invalid].head(5).tolist()
        raise ValueError(f"{table_name}: invalid YYYYMM date values: {examples}")

    return pd.to_datetime(date_int.astype(str) + "01", format="%Y%m%d")


def parse_calendar_date(series: pd.Series, table_name: str, column: str) -> pd.Series:
    out = pd.to_datetime(series, errors="coerce")
    bad = series.notna() & out.isna()
    if bad.any():
        examples = series[bad].head(5).tolist()
        raise ValueError(f"{table_name}: could not parse {column} as calendar dates: {examples}")
    if out.isna().any():
        raise ValueError(f"{table_name}: missing {column} values are not allowed when the column is present")
    return out


def reference_month_bounds(df: pd.DataFrame) -> tuple[str, str]:
    ref = pd.to_datetime(
        {
            "year": df["survey_year"].astype(int),
            "month": df["survey_month"].astype(int),
            "day": 1,
        }
    )
    return ref.min().date().isoformat(), ref.max().date().isoformat()


def release_summary_fields(df: pd.DataFrame) -> dict[str, object]:
    release_date = pd.to_datetime(df["release_date"], errors="raise")
    sources = sorted(df["release_date_source"].dropna().astype(str).unique())
    lag_months = pd.to_numeric(df["microdata_release_lag_months"], errors="raise")
    if not sources:
        raise ValueError("SCE input piece has no release_date_source values.")
    return {
        "max_release_date": release_date.max().date().isoformat(),
        "release_date_source": "; ".join(sources),
        "microdata_release_lag_months": int(lag_months.max()),
    }


def normalize_respondent_id(series: pd.Series, table_name: str) -> pd.Series:
    out = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    missing = out.isna() | out.eq("") | out.str.lower().eq("nan")
    if missing.any():
        raise ValueError(f"{table_name}: missing respondent identifiers are not allowed")
    return out


def numeric_without_new_missing(series: pd.Series, column: str, table_name: str) -> pd.Series:
    cleaned = series.copy()
    if pd.api.types.is_object_dtype(cleaned) or pd.api.types.is_string_dtype(cleaned):
        stripped = cleaned.astype("string").str.replace("\u00a0", " ", regex=False).str.strip()
        cleaned = cleaned.mask(stripped.eq(""))
    numeric = pd.to_numeric(cleaned, errors="coerce")
    bad = cleaned.notna() & numeric.isna()
    if bad.any():
        examples = cleaned[bad].head(5).tolist()
        raise ValueError(f"{table_name}: could not parse {column} as numeric: {examples}")
    return numeric


def release_id_for(module: str, raw_file: str, release_date: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", Path(raw_file).stem.lower()).strip("_")
    return f"sce_{module}_{slug}_{release_date.replace('-', '_')}"


def release_lag_months(module: str) -> int:
    if module not in MICRODATA_RELEASE_LAG_MONTHS:
        raise ValueError(f"No SCE microdata release-lag rule is defined for module {module}")
    return MICRODATA_RELEASE_LAG_MONTHS[module]


def shared_release_rule_id(module: str) -> str:
    return f"sce_{module}_{release_lag_months(module)}m_inferred"


def load_sce_release_date_crosswalk() -> pd.DataFrame:
    required_columns = [
        "module",
        "survey_year",
        "survey_month",
        "release_date",
        "source",
        "source_url",
        "comments",
        "aggregate_release_date_proxy",
        "microdata_release_lag_months",
    ]
    if not RELEASE_DATE_CROSSWALK_PATH.exists():
        raise SystemExit(f"Missing SCE release-date crosswalk: {RELEASE_DATE_CROSSWALK_PATH}")

    crosswalk = pd.read_csv(
        RELEASE_DATE_CROSSWALK_PATH,
        dtype={
            "module": "string",
            "source": "string",
            "source_url": "string",
            "comments": "string",
        },
    )
    missing = [column for column in required_columns if column not in crosswalk.columns]
    if missing:
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} is missing columns: {missing}")

    crosswalk = crosswalk.loc[:, required_columns].copy()
    crosswalk["module"] = crosswalk["module"].str.strip()
    crosswalk = crosswalk.loc[crosswalk["module"].isin(TABLE_OUTPUTS)].copy()
    missing_modules = sorted(set(TABLE_OUTPUTS) - set(crosswalk["module"]))
    if missing_modules:
        raise ValueError(f"SCE release crosswalk is missing active modules: {missing_modules}")
    crosswalk["survey_year"] = pd.to_numeric(crosswalk["survey_year"], errors="coerce").astype("Int64")
    crosswalk["survey_month"] = pd.to_numeric(crosswalk["survey_month"], errors="coerce").astype("Int64")
    crosswalk["release_date"] = pd.to_datetime(crosswalk["release_date"], errors="coerce")
    crosswalk["aggregate_release_date_proxy"] = pd.to_datetime(
        crosswalk["aggregate_release_date_proxy"],
        errors="coerce",
    )
    crosswalk["microdata_release_lag_months"] = pd.to_numeric(
        crosswalk["microdata_release_lag_months"],
        errors="coerce",
    ).astype("Int64")

    if crosswalk["module"].isna().any() or crosswalk["module"].eq("").any():
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains missing module values.")
    unexpected_modules = sorted(set(crosswalk["module"].dropna().astype(str)) - set(TABLE_OUTPUTS))
    if unexpected_modules:
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains unexpected modules: {unexpected_modules}")
    key_columns = ["survey_year", "survey_month", "microdata_release_lag_months"]
    if crosswalk[key_columns].isna().any().any():
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains missing or nonnumeric key/lag values.")
    if crosswalk.duplicated(["module", "survey_year", "survey_month"]).any():
        duplicates = crosswalk.loc[
            crosswalk.duplicated(["module", "survey_year", "survey_month"]),
            ["module", "survey_year", "survey_month"],
        ]
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains duplicate keys:\n{duplicates.to_string(index=False)}")
    if crosswalk["release_date"].isna().any() or crosswalk["aggregate_release_date_proxy"].isna().any():
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains missing release date values.")
    for column in ["source", "source_url", "comments"]:
        values = crosswalk[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains missing {column} values.")
        crosswalk[column] = values
    for module, lag in MICRODATA_RELEASE_LAG_MONTHS.items():
        observed = set(
            crosswalk.loc[crosswalk["module"].eq(module), "microdata_release_lag_months"]
            .dropna()
            .astype(int)
            .unique()
        )
        if observed and observed != {lag}:
            raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} has unexpected lag months for {module}: {sorted(observed)}")

    crosswalk["release_date"] = crosswalk["release_date"].dt.strftime("%Y-%m-%d").astype("string")
    crosswalk["aggregate_release_date_proxy"] = (
        crosswalk["aggregate_release_date_proxy"].dt.strftime("%Y-%m-%d").astype("string")
    )
    return crosswalk


def assign_sce_release_dates(
    module: str,
    survey_month_start: pd.Series,
    release_crosswalk: pd.DataFrame,
    table_name: str,
) -> pd.DataFrame:
    key = pd.DataFrame(
        {
            "module": module,
            "survey_year": survey_month_start.dt.year.astype(int),
            "survey_month": survey_month_start.dt.month.astype(int),
        },
        index=survey_month_start.index,
    )
    lookup = release_crosswalk.set_index(["module", "survey_year", "survey_month"])
    mapped = key.join(lookup, on=["module", "survey_year", "survey_month"], how="left")
    if mapped["release_date"].isna().any():
        examples = (
            mapped.loc[mapped["release_date"].isna(), ["module", "survey_year", "survey_month"]]
            .drop_duplicates()
            .head(10)
        )
        raise ValueError(f"{table_name}: SCE release-date crosswalk does not cover keys:\n{examples.to_string(index=False)}")

    return pd.DataFrame(
        {
            "release_date": mapped["release_date"].astype("string"),
            "aggregate_release_date_proxy": mapped["aggregate_release_date_proxy"].astype("string"),
            "microdata_release_lag_months": mapped["microdata_release_lag_months"].astype("Int64"),
            "release_date_source": mapped["source"].astype("string"),
        },
        index=survey_month_start.index,
    )


def sce_harmonization_specs(module: str, columns: list[str]) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    seen: set[str] = set()

    def add(raw_name: str, concept: str, task_family: str, lower: float | None, upper: float | None, rule: str) -> None:
        if raw_name in seen or raw_name not in columns:
            return
        seen.add(raw_name)
        specs.append(
            {
                "source": "sce",
                "module": module,
                "raw_name": raw_name,
                "clean_name": f"{raw_name}__clean",
                "status_name": f"{raw_name}__status",
                "concept": concept,
                "task_family": task_family,
                "lower_bound": lower,
                "upper_bound": upper,
                "rule": rule,
                "source_citation": "NY Fed public microdata/questionnaires; numeric range validated by repository diagnostics.",
                "notes": "Raw value preserved; companion clean/status columns added for task construction.",
            }
        )

    for prefix, (concept, task_family) in DENSITY_CONCEPTS.items():
        for col in columns:
            if re.fullmatch(fr"{re.escape(prefix)}_(bin|dens_?)?\d+", col) or re.fullmatch(fr"{re.escape(prefix)}_bin\d+", col):
                add(col, concept, task_family, 0.0, 100.0, "density_or_probability_bin_0_100")
        if prefix == "qsp7dens":
            for col in columns:
                if re.fullmatch(r"qsp7dens_\d+", col):
                    add(col, concept, task_family, 0.0, 100.0, "density_or_probability_bin_0_100")

    for raw_name, (concept, task_family) in TASK_KEY_VARS_BY_MODULE.get(module, {}).items():
        add(raw_name, concept, task_family, None, None, "numeric_preserve_with_status")

    for col in columns:
        if module == "core" and col in CORE_PROBABILITY_COLUMNS:
            add(col, "core subjective probability", "expectations", 0.0, 100.0, "probability_0_100")
        elif module == "public_policy" and re.fullmatch(r"qp1x[123]_\d+", col):
            add(col, "policy probability triad component", "policy expectations", 0.0, 100.0, "probability_0_100")
        elif module == "household_spending" and (
            re.fullmatch(r"qsp(4|5)_\d+", col) or re.fullmatch(r"qsp1[23]a_\d+", col)
        ):
            add(col, "spending probability/share variable", "consumption and saving", 0.0, 100.0, "probability_or_share_0_100")
        elif module == "labor_market" and (
            re.fullmatch(r"nl4_\d+", col)
            or re.fullmatch(r"oo1_\d+", col)
            or re.fullmatch(r"oo2b_\d+", col)
            or re.fullmatch(r"oo2c\d+", col)
            or col in {"oo2u", "oo2e", "oo2f", "q113", "q115", "q119", "q121", "q122"}
        ):
            add(col, "labor-market probability variable", "labor-market dynamics", 0.0, 100.0, "probability_0_100")

    return specs


def apply_harmonization(table: pd.DataFrame, module: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = sce_harmonization_specs(module, list(table.columns))
    out = table.copy()
    additions: dict[str, pd.Series] = {}
    for spec in specs:
        raw_name = str(spec["raw_name"])
        clean_name = str(spec["clean_name"])
        status_name = str(spec["status_name"])
        raw = out[raw_name]
        numeric = pd.to_numeric(raw, errors="coerce")
        raw_present = raw.notna()
        status = pd.Series(pd.NA, index=out.index, dtype="string")
        clean = pd.Series(np.nan, index=out.index, dtype="float64")
        invalid_numeric = raw_present & numeric.isna()
        undocumented_special = raw_present & numeric.notna() & numeric.le(-900)
        lower = spec["lower_bound"]
        upper = spec["upper_bound"]
        invalid_range = pd.Series(False, index=out.index)
        if lower is not None:
            invalid_range |= raw_present & numeric.notna() & numeric.lt(float(lower))
        if upper is not None:
            invalid_range |= raw_present & numeric.notna() & numeric.gt(float(upper))
        valid = raw_present & numeric.notna() & ~invalid_numeric & ~undocumented_special & ~invalid_range
        status.loc[valid] = "valid"
        status.loc[invalid_numeric | undocumented_special] = "undocumented_special"
        status.loc[invalid_range] = "invalid_range"
        clean.loc[valid] = numeric.loc[valid]
        additions[clean_name] = clean
        additions[status_name] = status
    if additions:
        out = pd.concat([out, pd.DataFrame(additions, index=out.index)], axis=1)
    return out, pd.DataFrame(specs)


def parse_value_label_string(value_labels: str) -> dict[object, str]:
    out: dict[object, str] = {}
    if value_labels is None or pd.isna(value_labels) or str(value_labels).strip() == "":
        return out
    for piece in str(value_labels).split(";"):
        text = piece.strip()
        if not text or ":" not in text:
            continue
        code_text, label_text = text.split(":", 1)
        code_text = code_text.strip()
        label_text = " ".join(label_text.strip().split())
        if not code_text or not label_text:
            continue
        numeric = pd.to_numeric(pd.Series([code_text]), errors="coerce").iloc[0]
        if pd.notna(numeric) and float(numeric).is_integer():
            code: object = int(numeric)
        elif pd.notna(numeric):
            code = float(numeric)
        else:
            code = code_text
        existing = out.get(code)
        if existing is not None and existing != label_text:
            raise ValueError(f"Conflicting value labels for code {code!r}: {existing!r} versus {label_text!r}")
        out[code] = label_text
    return out


def metadata_label_maps_for_module(
    module: str,
    module_metadata: dict[str, dict[str, dict[str, str]]],
) -> dict[str, dict[object, str]]:
    allowed_columns = TASK_METADATA_CATEGORICAL_COLUMNS_BY_MODULE.get(module, set())
    if not allowed_columns:
        return {}
    out: dict[str, dict[object, str]] = {}
    for raw_name, meta in module_metadata.get(module, {}).items():
        if raw_name not in allowed_columns:
            continue
        parsed = parse_value_label_string(meta.get("value_labels", ""))
        if parsed:
            out[raw_name] = parsed
    return out


def normalize_categorical_code(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if text == "":
        return pd.NA
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.notna(numeric) and float(numeric).is_integer():
        return int(numeric)
    if pd.notna(numeric):
        return float(numeric)
    return text


def categorical_label_series(
    table: pd.DataFrame,
    column: str,
    label_map: dict[object, str],
    table_name: str,
    allow_unmapped: bool = False,
) -> pd.Series:
    raw = table[column]
    normalized = raw.map(normalize_categorical_code)
    raw_present = raw.notna() & normalized.notna()
    labels = normalized.map(label_map)
    unmapped = raw_present & labels.isna()
    if unmapped.any() and not allow_unmapped:
        examples = sorted({repr(value) for value in raw.loc[unmapped].head(20).tolist()})
        raise ValueError(f"{table_name}: unmapped nonmissing categorical codes in {column}: {examples}")
    categories = list(dict.fromkeys(label_map.values()))
    return pd.Series(pd.Categorical(labels, categories=categories), index=table.index)


def add_categorical_label_columns(
    table: pd.DataFrame,
    table_name: str,
    module_metadata: dict[str, dict[str, dict[str, str]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_maps = metadata_label_maps_for_module(table_name, module_metadata)
    manual_maps = MANUAL_CATEGORICAL_LABELS_BY_MODULE.get(table_name, {})
    label_maps = dict(metadata_maps)
    label_maps.update(manual_maps)

    additions: dict[str, pd.Series] = {}
    rule_rows: list[dict[str, object]] = []
    for raw_name, label_map in label_maps.items():
        if raw_name not in table.columns:
            continue
        label_name = f"{raw_name}__label"
        additions[label_name] = categorical_label_series(
            table,
            raw_name,
            label_map,
            table_name,
            allow_unmapped=raw_name == "_HH_INC_DETAILED",
        )
        source = "manual_task_label_map"
        if raw_name in metadata_maps and raw_name not in manual_maps:
            source = "metadata_value_labels"
        elif raw_name in metadata_maps and raw_name in manual_maps:
            source = "manual_task_label_map_overrides_metadata"
        rule_rows.append(
            {
                "source": "sce",
                "module": table_name,
                "raw_name": raw_name,
                "label_name": label_name,
                "label_source": source,
                "n_categories": len(set(label_map.values())),
                "labels": "; ".join(dict.fromkeys(label_map.values())),
                "notes": "Raw source code is preserved; this companion column stores task-facing category labels.",
            }
        )
    if additions:
        table = pd.concat([table, pd.DataFrame(additions, index=table.index)], axis=1)
    return table, pd.DataFrame(rule_rows)


def add_job_search_documented_special_code_companions(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = sorted(set(JOB_SEARCH_DOCUMENTED_SPECIAL_CODES) - set(table.columns))
    if missing:
        raise ValueError(f"job_search: registered special-code columns are absent: {missing}")

    observed_pairs: set[tuple[str, int]] = set()
    for column in table.columns:
        numeric = pd.to_numeric(table[column], errors="coerce")
        observed = numeric[numeric.between(999990, 999999, inclusive="both") & numeric.mod(1).eq(0)]
        observed_pairs.update((column, int(code)) for code in observed.unique())
    registered_pairs = {
        (column, code)
        for column, code_map in JOB_SEARCH_DOCUMENTED_SPECIAL_CODES.items()
        for code in code_map
    }
    unregistered = sorted(observed_pairs - registered_pairs)
    if unregistered:
        raise ValueError(f"job_search: observed unregistered 99999x special codes: {unregistered}")

    out = table.copy()
    rule_rows: list[dict[str, object]] = []
    for raw_name, code_map in JOB_SEARCH_DOCUMENTED_SPECIAL_CODES.items():
        if raw_name == "l8_months_no_work":
            continue
        raw = out[raw_name]
        numeric = pd.to_numeric(raw, errors="coerce")
        raw_present = raw.notna()
        status = pd.Series(pd.NA, index=out.index, dtype="string")
        clean = pd.Series(np.nan, index=out.index, dtype="float64")
        is_special = pd.Series(False, index=out.index)
        for code, semantic_status in code_map.items():
            code_rows = numeric.eq(code)
            status.loc[code_rows] = semantic_status
            is_special |= code_rows
        invalid_numeric = raw_present & numeric.isna()
        invalid_negative = raw_present & numeric.notna() & numeric.lt(0) & ~is_special
        valid = raw_present & numeric.notna() & ~is_special & ~invalid_negative
        status.loc[invalid_numeric] = "invalid_numeric"
        status.loc[invalid_negative] = "invalid_negative"
        status.loc[valid] = "valid"
        clean.loc[valid] = numeric.loc[valid]
        out[f"{raw_name}__clean"] = clean
        out[f"{raw_name}__status"] = status
        rule_rows.append(
            {
                "source": "sce",
                "module": "job_search",
                "raw_name": raw_name,
                "clean_name": f"{raw_name}__clean",
                "status_name": f"{raw_name}__status",
                "concept": "Job Search numeric response with documented semantic special codes",
                "task_family": "job search and labor-market dynamics",
                "lower_bound": 0.0,
                "upper_bound": None,
                "rule": "documented_positive_special_codes_to_semantic_status",
                "documented_special_codes": "; ".join(
                    f"{code}:{status_name}" for code, status_name in sorted(code_map.items())
                ),
                "source_citation": JOB_SEARCH_SPECIAL_CODE_SOURCE,
                "notes": "Raw value preserved; documented special values are missing in the numeric companion.",
            }
        )
    return out, pd.DataFrame(rule_rows)


def add_job_search_months_no_work(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"age", "l8_months_no_work"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"job_search: missing months-not-working validation columns: {missing}")

    out = table.copy()
    raw = out["l8_months_no_work"]
    numeric = pd.to_numeric(raw, errors="coerce")
    age = pd.to_numeric(out["age"], errors="coerce")
    raw_present = raw.notna()
    special = numeric.eq(999995)
    invalid_numeric = raw_present & numeric.isna()
    invalid_negative = raw_present & numeric.notna() & numeric.lt(0) & ~special
    invalid_noninteger = raw_present & numeric.notna() & numeric.mod(1).ne(0) & ~special
    ordinary = raw_present & numeric.notna() & ~special & ~invalid_negative & ~invalid_noninteger
    valid_age = age.between(18, 100, inclusive="both") & age.mod(1).eq(0)
    age_unavailable = ordinary & ~valid_age
    exceeds_age = ordinary & valid_age & numeric.gt(12 * age)
    valid = ordinary & valid_age & ~exceeds_age

    clean = pd.Series(np.nan, index=out.index, dtype="float64")
    clean.loc[valid] = numeric.loc[valid]
    status = pd.Series(pd.NA, index=out.index, dtype="string")
    status.loc[special] = "never_had_paid_job"
    status.loc[invalid_numeric] = "invalid_numeric"
    status.loc[invalid_negative] = "invalid_negative"
    status.loc[invalid_noninteger] = "invalid_noninteger"
    status.loc[age_unavailable] = "age_unavailable_for_validation"
    status.loc[exceeds_age] = "exceeds_age_in_months"
    status.loc[valid] = "valid"
    unclassified = raw_present & status.isna()
    if unclassified.any():
        examples = out.loc[unclassified, ["respondent_id", "age", "l8_months_no_work"]].head(20)
        raise ValueError(f"job_search: unclassified months-not-working rows: {examples.to_dict('records')}")

    out["l8_months_no_work__clean"] = clean
    out["l8_months_no_work__status"] = status
    rule = pd.DataFrame(
        [
            {
                "source": "sce",
                "module": "job_search",
                "raw_name": "l8_months_no_work",
                "clean_name": "l8_months_no_work__clean",
                "status_name": "l8_months_no_work__status",
                "concept": "months since last paid work",
                "task_family": "job search and labor-market dynamics",
                "lower_bound": 0.0,
                "upper_bound": "12 * reported age",
                "rule": "documented_special_code_and_age_feasibility",
                "documented_special_codes": "999995:never_had_paid_job",
                "source_citation": JOB_SEARCH_SPECIAL_CODE_SOURCE,
                "notes": (
                    "Ordinary values must be nonnegative integers and no greater than reported age in months. "
                    "Rows without a valid reported age of 18--100 are not assigned a numeric clean value."
                ),
            }
        ]
    )
    return out, rule


def add_labor_market_reservation_wage_annual_full_time(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"survey_date", "rw2a", "rw2a__clean", "rw2a__status", "rw2b__clean__label"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"labor_market: missing reservation-wage conversion columns: {missing}")

    out = table.copy()
    survey_date = pd.to_datetime(out["survey_date"], errors="coerce")
    if survey_date.isna().any():
        raise ValueError("labor_market: invalid survey_date in reservation-wage conversion")

    amount = pd.to_numeric(out["rw2a__clean"], errors="coerce")
    source_status = out["rw2a__status"].astype("string")
    effective_unit = out["rw2b__clean__label"].astype("string")
    post_2017_regime = survey_date.ge(pd.Timestamp("2017-03-01"))
    effective_unit = effective_unit.mask(
        amount.notna() & post_2017_regime & effective_unit.isna(),
        "year",
    )

    annual_factors = {
        "hour": 40.0 * 52.0,
        "week": 52.0,
        "two weeks": 26.0,
        "month": 12.0,
        "year": 1.0,
    }
    unsupported_unit = effective_unit.notna() & ~effective_unit.isin(annual_factors)
    if unsupported_unit.any():
        examples = sorted(effective_unit.loc[unsupported_unit].dropna().unique().tolist())
        raise ValueError(f"labor_market: unsupported reservation-wage units: {examples}")
    contradictory_post_2017_unit = amount.notna() & post_2017_regime & effective_unit.notna() & effective_unit.ne("year")
    if contradictory_post_2017_unit.any():
        examples = out.loc[
            contradictory_post_2017_unit,
            ["survey_year", "survey_month", "rw2a__clean", "rw2b__clean__label"],
        ].head(20)
        raise ValueError(
            "labor_market: nonannual reservation-wage units appear from March 2017 onward: "
            f"{examples.to_dict('records')}"
        )

    raw_present = out["rw2a"].notna()
    valid_source = source_status.eq("valid") & amount.notna()
    positive_source = valid_source & amount.gt(0)
    valid_conversion = positive_source & effective_unit.isin(annual_factors)

    annual_full_time = pd.Series(np.nan, index=out.index, dtype="float64")
    for unit, factor in annual_factors.items():
        use_unit = valid_conversion & effective_unit.eq(unit)
        annual_full_time.loc[use_unit] = amount.loc[use_unit] * factor

    conversion_status = pd.Series(pd.NA, index=out.index, dtype="string")
    conversion_status.loc[~raw_present] = "missing_amount"
    conversion_status.loc[raw_present & ~source_status.eq("valid")] = "invalid_source_amount"
    conversion_status.loc[valid_source & amount.le(0)] = "nonpositive_amount"
    conversion_status.loc[positive_source & effective_unit.isna() & ~post_2017_regime] = "missing_pre2017_unit"
    for unit in annual_factors:
        conversion_status.loc[valid_conversion & effective_unit.eq(unit)] = f"valid_{unit.replace(' ', '_')}"
    if conversion_status.isna().any():
        examples = out.loc[
            conversion_status.isna(),
            ["survey_year", "survey_month", "rw2a", "rw2a__clean", "rw2a__status", "rw2b__clean__label"],
        ].head(20)
        raise ValueError(
            "labor_market: unclassified reservation-wage conversion rows: "
            f"{examples.to_dict('records')}"
        )

    out["reservation_wage_annual_full_time"] = annual_full_time
    out["reservation_wage_annual_full_time_status"] = conversion_status
    rule = pd.DataFrame(
        [
            {
                "source": "sce",
                "module": "labor_market",
                "raw_name": "rw2a__clean + rw2b__clean__label",
                "clean_name": "reservation_wage_annual_full_time",
                "status_name": "reservation_wage_annual_full_time_status",
                "concept": "annual full-time-equivalent reservation wage",
                "task_family": "labor-market dynamics",
                "lower_bound": 0.0,
                "upper_bound": None,
                "rule": "documented_unit_to_annual_full_time_40_hours_52_weeks",
                "source_citation": "NY Fed SCE Labor Market public microdata and questionnaire.",
                "notes": (
                    "Positive valid source amounts are converted using hour*40*52, week*52, "
                    "two weeks*26, month*12, and year*1. Missing units from March 2017 onward "
                    "are annual by the documented questionnaire regime; earlier missing units "
                    "remain unconverted. Raw amount and unit fields are preserved."
                ),
            }
        ]
    )
    return out, rule


def density_family_columns(module: str, columns: pd.Index) -> dict[str, list[str]]:
    families: dict[str, list[str]] = {}
    if module == "core":
        for family in ["Q9", "Q9c", "Q24", "C1"]:
            cols = [f"{family}_bin{i}" for i in range(1, 11)]
            if all(col in columns for col in cols):
                families[family] = cols
    if module == "household_spending":
        cols = [f"qsp7dens_{i}" for i in range(1, 11)]
        if all(col in columns for col in cols):
            families["qsp7dens"] = cols
    return families


def add_density_family_filters(table: pd.DataFrame, module: str) -> pd.DataFrame:
    families = density_family_columns(module, table.columns)
    if not families:
        return table
    out = table.copy()
    additions: dict[str, pd.Series] = {}
    tolerance = 1.0
    for family, cols in families.items():
        numeric = out[cols].apply(pd.to_numeric, errors="coerce")
        raw_present = out[cols].notna()
        numeric_present = numeric.notna()
        any_present = raw_present.any(axis=1)
        all_present = raw_present.all(axis=1)
        all_numeric = numeric_present.all(axis=1)
        range_invalid = (raw_present & (numeric.isna() | numeric.lt(0) | numeric.gt(100))).any(axis=1)
        row_sum = numeric.sum(axis=1, min_count=1)
        all_zero = all_numeric & numeric.eq(0).all(axis=1)
        all_same_nonzero = all_numeric & numeric.nunique(axis=1).eq(1) & numeric.iloc[:, 0].ne(0) & row_sum.sub(100).abs().gt(tolerance)
        sum_off = all_numeric & row_sum.sub(100).abs().gt(tolerance)

        status = pd.Series(pd.NA, index=out.index, dtype="string")
        status.loc[any_present & all_numeric & ~range_invalid & ~all_zero & ~all_same_nonzero & ~sum_off] = "valid"
        status.loc[any_present & range_invalid] = "range_invalid"
        status.loc[any_present & ~range_invalid & ~all_present] = "partial_missing"
        status.loc[any_present & ~range_invalid & all_zero] = "all_zero"
        status.loc[any_present & ~range_invalid & all_same_nonzero] = "all_same_nonzero"
        status.loc[any_present & ~range_invalid & ~all_zero & ~all_same_nonzero & sum_off] = "sum_off_100"
        task_valid = status.eq("valid")

        additions[f"{family}__sum"] = row_sum
        additions[f"{family}__task_valid"] = task_valid.astype("boolean")
        additions[f"{family}__task_status"] = status
    return pd.concat([out, pd.DataFrame(additions, index=out.index)], axis=1)


def build_structural_fields(
    df: pd.DataFrame,
    module: str,
    table_name: str,
    path: Path,
    raw_dir: Path,
    release_crosswalk: pd.DataFrame,
    id_column: str = "userid",
    date_column: str = "date",
    weight_column: str = "weight",
    extra_fields: dict[str, object] | None = None,
) -> pd.DataFrame:
    survey_month_start = parse_survey_date(df[date_column], table_name)
    if "survey_date" in df.columns:
        survey_date = parse_calendar_date(df["survey_date"], table_name, "survey_date")
        df = df.drop(columns=["survey_date"])
    else:
        survey_date = survey_month_start
    respondent_id = normalize_respondent_id(df[id_column], table_name)
    release_assignment = assign_sce_release_dates(module, survey_month_start, release_crosswalk, table_name)
    release_date = release_assignment["release_date"]
    release_id_slug = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")
    release_id = (
        "sce_"
        + module
        + "_"
        + release_id_slug
        + "_"
        + survey_month_start.dt.strftime("%Y_%m")
        + "_"
        + release_date.str.replace("-", "_", regex=False)
    )

    structural = pd.DataFrame(
        {
            "source": "sce",
            "release_id": release_id,
            "release_date": release_date,
            "aggregate_release_date_proxy": release_assignment["aggregate_release_date_proxy"],
            "microdata_release_lag_months": release_assignment["microdata_release_lag_months"],
            "release_date_status": ROW_LEVEL_RELEASE_STATUS,
            "release_date_source": release_assignment["release_date_source"],
            "module": module,
            "raw_file": relpath(path, raw_dir),
            "respondent_id": respondent_id,
            "survey_year": survey_month_start.dt.year.astype("Int64"),
            "survey_month": survey_month_start.dt.month.astype("Int64"),
            "survey_date": survey_date,
        }
    )

    if "tenure" in df.columns:
        structural["panel_month"] = numeric_without_new_missing(df["tenure"], "tenure", table_name).astype("Int64")

    if weight_column in df.columns:
        structural[f"weight_{module}"] = numeric_without_new_missing(df[weight_column], weight_column, table_name)

    if extra_fields:
        for key, value in extra_fields.items():
            structural[key] = value

    return pd.concat([structural, df], axis=1)


def read_microdata(raw_dir: Path, spec: dict[str, object], release_crosswalk: pd.DataFrame) -> pd.DataFrame:
    path = raw_dir / str(spec["relative_path"])
    table_name = str(spec["table"])
    module = str(spec["module"])
    id_column = str(spec.get("id_column", "userid"))
    date_column = str(spec.get("date_column", "date"))
    weight_column = str(spec.get("weight_column", "weight"))
    df = pd.read_excel(
        path,
        sheet_name=str(spec["sheet"]),
        header=int(spec["header"]),
        engine="openpyxl",
    )
    df.columns = clean_column_names(df.columns)
    df = df.dropna(how="all").reset_index(drop=True)

    required = {id_column, date_column}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{table_name}: missing required columns in {path}: {missing}")

    return build_structural_fields(
        df,
        module,
        table_name,
        path,
        raw_dir,
        release_crosswalk,
        id_column,
        date_column,
        weight_column,
    )


HOUSING_YEARS = list(range(2014, 2025))


def read_housing_microdata(raw_dir: Path, release_crosswalk: pd.DataFrame) -> list[pd.DataFrame]:
    path = raw_dir / "housing/FRBNY-SCE-Housing-Survey-Public-Microdata-Complete.xlsx"
    pieces = []
    for year in HOUSING_YEARS:
        sheet = f"Data {year}"
        table_name = f"housing_{year}"
        df = pd.read_excel(path, sheet_name=sheet, header=0, engine="openpyxl")
        df.columns = clean_column_names(df.columns)
        df = df.dropna(how="all").reset_index(drop=True)
        if "userid" not in df.columns:
            raise ValueError(f"{table_name}: missing userid in {path}")
        expected_date = year * 100 + 2
        if "date" in df.columns:
            observed_dates = pd.to_numeric(df["date"], errors="raise").dropna().astype(int).unique().tolist()
            if sorted(observed_dates) != [expected_date]:
                raise ValueError(f"{table_name}: observed date values {sorted(observed_dates)} do not match February fielding {expected_date}")
        else:
            insert_at = 1 if "userid" in df.columns else 0
            df.insert(insert_at, "date", expected_date)
        pieces.append(
            build_structural_fields(
                df,
                "housing",
                table_name,
                path,
                raw_dir,
                release_crosswalk,
                id_column="userid",
                date_column="date",
                weight_column="weights",
                extra_fields={"source_wave_year": year},
            )
        )
    return pieces


def read_job_search_microdata(raw_dir: Path, release_crosswalk: pd.DataFrame) -> pd.DataFrame:
    path = raw_dir / "job_search/SCE-Public-LM-Quarterly-Microdata.xlsx"
    table_name = "job_search"
    df = pd.read_excel(path, sheet_name="Data", header=0, engine="openpyxl")
    df.columns = clean_column_names(df.columns)
    df = df.dropna(how="all").reset_index(drop=True)
    required = {"userid", "year"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{table_name}: missing required columns in {path}: {missing}")
    years = pd.to_numeric(df["year"], errors="raise").astype(int)
    df.insert(list(df.columns).index("year") + 1, "date", years * 100 + 10)
    return build_structural_fields(
        df,
        "job_search",
        table_name,
        path,
        raw_dir,
        release_crosswalk,
        id_column="userid",
        date_column="date",
        weight_column="survey_weight",
        extra_fields={"source_wave_year": years},
    )


def convert_object_columns_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = out[col].astype("string")
    return out


def normalize_codebook_variable(value: object, valid_lookup: dict[str, str]) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    text = text.strip("[]")
    text = text.split(":", 1)[0].strip()
    if not text:
        return ""
    if text in valid_lookup.values():
        return text
    return valid_lookup.get(text.lower(), "")


def codebook_variable_matches(value: object, valid_variables: set[str], valid_lookup: dict[str, str]) -> list[str]:
    exact = normalize_codebook_variable(value, valid_lookup)
    if exact:
        return [exact]

    text = "" if pd.isna(value) else str(value).strip()
    text = text.strip("[]")
    text = text.split(":", 1)[0].strip()
    if not text:
        return []

    placeholder = text
    placeholder = re.sub(r"__+", "_", placeholder)
    placeholder = re.sub(r"(?i)_n\b", "_{person}", placeholder)
    placeholder = re.sub(r"(?i)_m\b", "_{activity}", placeholder)
    placeholder = re.sub(r"(?i)_t\b", "_{task}", placeholder)
    if "{" not in placeholder:
        return []

    pattern = re.escape(placeholder)
    pattern = pattern.replace(re.escape("{person}"), r"\d+")
    pattern = pattern.replace(re.escape("{activity}"), r"\d+")
    pattern = pattern.replace(re.escape("{task}"), r"(?:\d+|9[148]?|98)")
    pattern = "^" + pattern + "$"
    compiled = re.compile(pattern, flags=re.IGNORECASE)
    matches = [variable for variable in sorted(valid_variables) if compiled.match(variable)]
    return matches


def merge_metadata(
    out: dict[str, dict[str, str]],
    variable: str,
    question_text: str,
    value_label: str = "",
    notes: str = "",
) -> None:
    if not variable:
        return
    current = out.setdefault(variable, {"question_text": "", "value_labels": "", "notes": ""})
    if question_text and not current["question_text"]:
        current["question_text"] = " ".join(str(question_text).split())[:900]
    if value_label:
        existing = current["value_labels"]
        if value_label not in existing:
            current["value_labels"] = f"{existing}; {value_label}" if existing else value_label
    if notes and not current["notes"]:
        current["notes"] = notes


def parse_workbook_codebook_sheet(path: Path, sheet: str, valid_variables: set[str], notes: str) -> dict[str, dict[str, str]]:
    frame = pd.read_excel(path, sheet_name=sheet, header=None, engine="openpyxl")
    valid_lookup = {variable.lower(): variable for variable in valid_variables}
    out: dict[str, dict[str, str]] = {}
    current_variables: list[str] = []
    for row in frame.itertuples(index=False, name=None):
        cells = ["" if pd.isna(cell) else str(cell).strip() for cell in row]
        if not any(cells):
            current_variables = []
            continue

        bracket_hits: list[str] = []
        for cell in cells:
            match = re.match(r"^\[([A-Za-z0-9_]+)\]\s*:\s*(.+)$", cell)
            if match:
                bracket_hits = codebook_variable_matches(match.group(1), valid_variables, valid_lookup)
                if bracket_hits:
                    for variable in bracket_hits:
                        merge_metadata(out, variable, match.group(2), notes=notes)
                    current_variables = bracket_hits
                    break
        if bracket_hits:
            continue

        variable_hits: list[str] = []
        for pos, cell in enumerate(cells):
            candidates = codebook_variable_matches(cell, valid_variables, valid_lookup)
            if candidates:
                variable_hits = candidates
                description = ""
                for later in cells[pos + 1 :]:
                    if later and not later.lower().startswith(("values:", "open text response", "open numeric response")):
                        description = later
                        break
                for variable in candidates:
                    merge_metadata(out, variable, description, notes=notes)
                current_variables = candidates
                break
        if variable_hits or not current_variables:
            continue

        code = cells[1] if len(cells) > 1 else ""
        label = cells[2] if len(cells) > 2 else ""
        if code and label and code.lower() not in {"question", "code value", "values:"}:
            for variable in current_variables:
                merge_metadata(out, variable, "", f"{code}: {label}", notes=notes)
    return out


def read_sheet_columns(path: Path, sheet: str) -> set[str]:
    header = pd.read_excel(path, sheet_name=sheet, header=0, nrows=0, engine="openpyxl")
    return set(clean_column_names(header.columns))


def parse_housing_codebooks(raw_dir: Path) -> dict[str, dict[str, str]]:
    path = raw_dir / "housing/FRBNY-SCE-Housing-Survey-Public-Microdata-Complete.xlsx"
    out: dict[str, dict[str, str]] = {}
    for year in HOUSING_YEARS:
        valid_variables = read_sheet_columns(path, f"Data {year}")
        sheet_meta = parse_workbook_codebook_sheet(
            path,
            f"Codebook {year}",
            valid_variables,
            f"Extracted from Housing workbook Codebook {year}.",
        )
        for variable, meta in sheet_meta.items():
            merge_metadata(out, variable, meta["question_text"], meta["value_labels"], meta["notes"])
    return out


def normalize_question_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def base_question_key(module: str, raw_name: str) -> str:
    raw = raw_name.strip()
    if module == "core":
        for prefix in ["Q9new2", "Q9c", "Q9", "Q24", "C1", "C2"]:
            if re.match(fr"^{prefix}(_(cent25|cent50|cent75|var|mean|iqr|probdeflation|bin\d+)|$)", raw, re.IGNORECASE):
                return normalize_question_key(prefix)
        return normalize_question_key(raw)
    if module == "household_spending":
        upper = raw.upper()
        if upper.startswith("QSP7DENS"):
            return normalize_question_key("QSP7dens")
        match = re.match(r"^(QSP\d+[A-Z]*|Q\d+C\d+(?:PART\d+)?)", upper)
        if match:
            return normalize_question_key(match.group(1))
        return normalize_question_key(upper)
    if module == "labor_market":
        upper = raw.upper()
        if re.match(r"^L1[MY]$", upper):
            return normalize_question_key(upper)
        match = re.match(r"^(L\d+[A-Z]*|JS\d+|RW\d+[A-Z]*|Q\d+|OO\d+[A-Z]*|NL\d+[A-Z]*)", upper)
        if match:
            return normalize_question_key(match.group(1))
        return normalize_question_key(upper)
    return normalize_question_key(raw)


def public_policy_question(raw_name: str) -> str:
    parts = raw_name.rsplit("_", 1)
    item = ""
    if len(parts) == 2 and parts[1].isdigit():
        item = POLICY_ITEMS.get(int(parts[1]), f"policy item {parts[1]}")
    if raw_name.startswith("hidqp2_"):
        return f"Indicator for whether respondent had heard of {item}"
    if raw_name.startswith("qp1x1_"):
        return f"Percent chance of increase/expansion in {item}"
    if raw_name.startswith("qp1x2_"):
        return f"Percent chance of no change in {item}"
    if raw_name.startswith("qp1x3_"):
        return f"Percent chance of decrease/reduction in {item}"
    if raw_name.startswith("qp1x4_"):
        return f"Indicator that respondent has no idea what {item} is"
    if raw_name.startswith("qp2_"):
        return f"Expected impact of possible policy change for {item}"
    return ""


def spending_question(raw_name: str) -> str:
    if raw_name == "qsp1":
        return "Direction of current monthly household spending relative to 12 months ago"
    if raw_name == "qsp2":
        return "Percent change in current monthly household spending relative to 12 months ago"
    if raw_name.startswith("qsp3_"):
        return "Large-purchase category made by household during the last 4 months"
    if raw_name.startswith("qsp4_"):
        return "Percent chance household will make a large purchase in the next 4 months"
    if raw_name.startswith("qsp5_"):
        return "Expected percent chance of large purchase by category in the next 4 months"
    if raw_name.startswith("qsp6") or raw_name.startswith("qsp7dens"):
        return "Spending-growth expectation distribution variable"
    if raw_name.startswith("qsp10") or raw_name.startswith("qsp11"):
        return "Monthly spending category or expected category change variable"
    if raw_name.startswith("qsp12") or raw_name.startswith("qsp13") or raw_name.startswith("qsp14"):
        return "Household saving or financial behavior variable"
    return ""


def labor_question(raw_name: str) -> str:
    if raw_name in {"l1m", "l1y"}:
        return "Month/year respondent started current/main job"
    if raw_name == "l3":
        return "Annual earnings before taxes and deductions at current/main job"
    if raw_name.startswith("lm"):
        return "Current/main job characteristics or employment status detail"
    if raw_name.startswith("js"):
        return "Job-search expectation or experience variable"
    if raw_name.startswith("rw"):
        return "Reservation-wage or wage-offer variable"
    if raw_name.startswith("nl"):
        return "Nonemployment or nonlabor-force variable"
    if raw_name.startswith("oo"):
        return "Outside offer or job opportunity variable"
    return ""


def core_manual_question(raw_name: str) -> tuple[str, str]:
    derived_labels = {
        "_STATE": "Derived respondent state code.",
        "_AGE_CAT": "Derived respondent age category.",
        "_NUM_CAT": "Derived numeracy category.",
        "_REGION_CAT": "Derived Census-region category.",
        "_COMMUTING_ZONE": "Derived commuting-zone identifier.",
        "_EDU_CAT": "Derived respondent education category.",
        "_HH_INC_CAT": "Derived household-income category.",
        "_HH_INC_DETAILED": "Detailed household pre-tax income category, using Q47 on panel entry and D6 on follow-up.",
        "age_years": "Respondent age in years, using reported Q32 when observed and otherwise advancing the first reported age by elapsed panel time.",
        "DSAME": "Derived demographic or household-status flag; exact source label not recovered.",
        "D5b": "Derived demographic variable; exact source label not recovered.",
        "DHH2_11_other": "Derived household-composition free-text other category.",
        "Q12new": "Core labor-market/employment variable Q12new; exact questionnaire wording not recovered by parser.",
    }
    if raw_name in derived_labels:
        note = "Manual metadata mapping from SCE naming convention and derived-variable families; exact source wording still needs codebook review."
        if raw_name.startswith("_"):
            note = "Manual metadata mapping of derived SCE demographic/geographic category variable."
        return derived_labels[raw_name], note

    patterns = [
        (r"^Q10_(\d+)$", "Employment-status option indicator from core question Q10"),
        (r"^Q35_(\d+)$", "Core demographic option indicator from question Q35"),
        (r"^HH2_(\d+)$", "Household-composition option indicator from core question HH2"),
        (r"^Q45new_(\d+)$", "Core question Q45new option indicator"),
        (r"^D2new_(\d+)$", "Derived demographic option indicator from the D2new family"),
        (r"^DHH2_(\d+)$", "Derived household-composition indicator from the HH2 family"),
        (r"^ES1_(\d+)$", "Employment-status screener option indicator from the ES1 family"),
        (r"^C4_(\d+)$", "Credit or household-finance option indicator from the C4 family"),
    ]
    for pattern, label in patterns:
        match = re.match(pattern, raw_name)
        if match:
            return (
                f"{label}, option {match.group(1)}; exact option text not recovered by parser.",
                "Manual prefix-family metadata mapping; exact response-option label requires questionnaire/codebook follow-up.",
            )
    return "", ""


def question_metadata(
    module: str,
    raw_name: str,
    module_metadata: dict[str, dict[str, dict[str, str]]],
    questionnaire_markers: dict[str, dict[str, str]],
) -> dict[str, str]:
    if raw_name == "userid":
        return {"question_text": "Respondent ID", "value_labels": "", "notes": "Canonical copy written as respondent_id."}
    if raw_name == "responseid":
        return {"question_text": "Response ID", "value_labels": "", "notes": "Response-level identifier preserved from raw Job Search file."}
    if raw_name == "date":
        return {
            "question_text": "Survey administration month in YYYYMM format",
            "value_labels": "",
            "notes": "Canonical year/month/date fields derived from this column.",
        }
    if raw_name == "year":
        return {
            "question_text": "Survey administration year",
            "value_labels": "",
            "notes": "Job Search month is documented as October and the canonical date is derived as YYYY10.",
        }
    if raw_name == "tenure":
        return {"question_text": "Panel tenure/wave count", "value_labels": "", "notes": "Canonical copy written as panel_month."}
    if raw_name in {"weight", "weights", "quarterly_weights", "survey_weight", "ACS_weight", "ACS_weights", "acsweights"}:
        return {"question_text": "Public microdata survey weight", "value_labels": "", "notes": "Canonical module-specific copy written as weight_*."}
    module_book = module_metadata.get(module, {})
    if raw_name in module_book:
        meta = module_book[raw_name]
        return {"question_text": meta["question_text"], "value_labels": meta["value_labels"], "notes": meta.get("notes", "Extracted from local official metadata.")}
    if module_book:
        case_lookup = {name.lower(): name for name in module_book}
        case_hit = case_lookup.get(raw_name.lower())
        if case_hit:
            meta = module_book[case_hit]
            return {
                "question_text": meta["question_text"],
                "value_labels": meta["value_labels"],
                "notes": meta.get("notes", "Extracted from local official metadata.") + " Matched case-insensitively across wave naming conventions.",
            }
    module_markers = questionnaire_markers.get(module, {})
    marker_text = module_markers.get(base_question_key(module, raw_name), "")
    if marker_text:
        return {"question_text": marker_text, "value_labels": "", "notes": "Extracted from local official questionnaire PDF."}
    if module == "core":
        text, note = core_manual_question(raw_name)
        if text:
            return {"question_text": text, "value_labels": "", "notes": note}
    if module == "household_spending":
        return {"question_text": spending_question(raw_name), "value_labels": "", "notes": "Question text summarized from local questionnaire/glossary PDFs when matched by prefix."}
    if module == "labor_market":
        return {"question_text": labor_question(raw_name), "value_labels": "", "notes": "Question text summarized from local questionnaire/glossary PDFs when matched by prefix."}
    if module == "public_policy":
        return {"question_text": public_policy_question(raw_name), "value_labels": "", "notes": "Question text mapped from local questionnaire/chart-guide PDFs when matched by prefix."}
    return {"question_text": "", "value_labels": "", "notes": "Raw variable retained; detailed label requires questionnaire-level review."}


def canonical_name(module: str, raw_name: str) -> str:
    if raw_name == "userid":
        return "respondent_id"
    if raw_name == "date":
        return "survey_date"
    if raw_name == "year":
        return "survey_year"
    if raw_name == "tenure":
        return "panel_month"
    if raw_name in {"weight", "weights", "quarterly_weights", "survey_weight", "ACS_weight", "ACS_weights", "acsweights"}:
        return f"weight_{module}"
    return ""


def build_variable_inventory(
    tables: dict[str, pd.DataFrame],
    raw_columns: dict[str, list[str]],
    module_metadata: dict[str, dict[str, dict[str, str]]],
    questionnaire_markers: dict[str, dict[str, str]],
) -> pd.DataFrame:
    rows = []
    for table_name, df in tables.items():
        module = str(df["module"].iloc[0])
        for raw_name in raw_columns[table_name]:
            meta = question_metadata(module, raw_name, module_metadata, questionnaire_markers)
            if raw_name in df.columns:
                nonmissing = int(df[raw_name].notna().sum())
                dtype = str(df[raw_name].dtype)
                missing_share = float(df[raw_name].isna().mean())
            else:
                nonmissing = 0
                dtype = "absent"
                missing_share = 1.0
            rows.append(
                {
                    "source": "sce",
                    "module": module,
                    "table": table_name,
                    "raw_name": raw_name,
                    "canonical_name": canonical_name(module, raw_name),
                    "dtype": dtype,
                    "nonmissing_rows": nonmissing,
                    "missing_share": missing_share,
                    "question_text": meta["question_text"],
                    "value_labels": meta["value_labels"],
                    "target_use": TASK_USE_BY_MODULE[module],
                    "notes": meta["notes"],
                }
            )
    return pd.DataFrame(rows)


def metadata_audit_reason(row: pd.Series) -> str:
    text = "" if pd.isna(row.get("question_text", "")) else str(row.get("question_text", "")).strip()
    notes = "" if pd.isna(row.get("notes", "")) else str(row.get("notes", "")).strip().lower()
    raw_name = str(row.get("raw_name", ""))
    module = str(row.get("module", ""))
    unresolved_markers = [
        "exact option text not recovered",
        "exact source label not recovered",
        "exact questionnaire wording not recovered",
    ]
    if text and not any(marker in text.lower() for marker in unresolved_markers):
        return "resolved"
    if raw_name.lower().find("random") >= 0:
        return "randomizer_or_internal"
    if raw_name.startswith("_") or raw_name.startswith("DHH2") or raw_name.startswith("D2new") or raw_name in {"DSAME", "D5b"}:
        return "derived_variable"
    if any(marker in text.lower() for marker in unresolved_markers):
        return "needs_manual_review"
    if "summarized from local questionnaire" in notes and not text:
        return "parser_gap"
    if not text:
        return "needs_manual_review"
    return "needs_manual_review"


def write_metadata_coverage_audit(variable_inventory: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    audit = variable_inventory.copy()
    audit["metadata_status"] = np.where(
        audit["question_text"].fillna("").astype(str).str.strip().ne(""),
        "covered",
        "unresolved",
    )
    audit["unresolved_reason"] = audit.apply(metadata_audit_reason, axis=1)
    audit.loc[audit["unresolved_reason"].eq("resolved"), "metadata_status"] = "covered"
    audit.loc[audit["unresolved_reason"].ne("resolved"), "metadata_status"] = "needs_review"
    out = audit[
        [
            "source",
            "module",
            "raw_name",
            "metadata_status",
            "unresolved_reason",
            "nonmissing_rows",
            "missing_share",
            "question_text",
            "notes",
        ]
    ].copy()
    out.to_csv(output_dir / "metadata_coverage_audit.csv", index=False)
    return out


def check_duplicate_keys(table_name: str, df: pd.DataFrame) -> None:
    duplicates = df.duplicated(["respondent_id", "survey_year", "survey_month"])
    if duplicates.any():
        examples = df.loc[duplicates, ["respondent_id", "survey_year", "survey_month", "raw_file"]].head(10)
        raise ValueError(f"{table_name}: duplicate respondent-month keys detected:\n{examples}")


def write_manifest(
    intermediate_dir: Path,
    source_inventory: pd.DataFrame,
    input_summaries: pd.DataFrame,
) -> pd.DataFrame:
    inventory = source_inventory.set_index("relative_path")
    rows = []
    grouped = (
        input_summaries.groupby(["module", "raw_file"], as_index=False)
        .agg(
            rows=("rows", "sum"),
            min_survey_date=("min_survey_date", "min"),
            max_survey_date=("max_survey_date", "max"),
            max_release_date=("max_release_date", "max"),
            release_date_source=(
                "release_date_source",
                lambda values: "; ".join(dict.fromkeys(str(value) for value in values if str(value).strip())),
            ),
            microdata_release_lag_months=("microdata_release_lag_months", "max"),
            notes=("notes", lambda values: " ".join(dict.fromkeys(str(value) for value in values if str(value).strip()))),
        )
        .sort_values(["module", "raw_file"])
    )
    for summary in grouped.to_dict("records"):
        rel = str(summary["raw_file"])
        module = str(summary["module"])
        inv = inventory.loc[rel]
        note = str(summary["notes"])
        if module in {"household_spending", "labor_market", "public_policy"}:
            note += " Public microdata workbook has no standalone weight column; module weight is joined from core respondent-month weights when available."
        if module in {"job_search", "public_policy"}:
            note += " Release dates use a conservative benchmark proxy because module-specific public microdata cadence evidence is not cleanly recoverable from the cumulative workbook."
        max_release_date = str(summary["max_release_date"])
        rows.append(
            {
                "source": "sce",
                "release_id": release_id_for(module, rel, max_release_date),
                "release_date": max_release_date,
                "official_release_date": "",
                "release_date_source": summary["release_date_source"],
                "release_date_status": ROW_LEVEL_RELEASE_STATUS,
                "release_date_unit": "max_row_level_inferred_microdata_release_date",
                "microdata_release_lag_months": int(summary["microdata_release_lag_months"]),
                "reference_period_start": summary["min_survey_date"],
                "reference_period_end": summary["max_survey_date"],
                "module": module,
                "raw_file": rel,
                "download_url": inv["download_url"],
                "sha256": inv["sha256"],
                "task_ready": 0,
                "leakage_ready": 1,
                "notes": note + " Manifest release_date is the maximum inferred row-level release date for this cumulative file; row eligibility must use the row-level release_date in the cleaned parquet.",
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_parquet(intermediate_dir / "sce_release_manifest.parquet", index=False)
    return manifest


def build_shared_release_coverage(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for module, table in sorted(tables.items()):
        missing = sorted({"release_date", "module"} - set(table.columns))
        if missing:
            raise ValueError(f"SCE {module}: missing release coverage columns: {missing}")
        if not table["module"].eq(module).all():
            bad = sorted(table.loc[~table["module"].eq(module), "module"].dropna().astype(str).unique())
            raise ValueError(f"SCE {module}: unexpected module values in release coverage: {bad}")
        release_dates = pd.to_datetime(table["release_date"], errors="raise").dt.date.astype(str)
        frame = pd.DataFrame(
            {
                "dataset": "SCE",
                "release_rule_id": shared_release_rule_id(module),
                "microdata_release_date": release_dates,
            }
        )
        counts = (
            frame.groupby(["dataset", "release_rule_id", "microdata_release_date"], dropna=False)
            .size()
            .reset_index(name="rows")
        )
        counts["unmatched_rows"] = 0
        frames.append(counts)
    if not frames:
        raise ValueError("No SCE tables were available for release-date coverage.")
    out = pd.concat(frames, ignore_index=True)
    return out[["dataset", "release_rule_id", "microdata_release_date", "rows", "unmatched_rows"]]


def update_release_date_coverage(dataset: str, coverage: pd.DataFrame) -> None:
    RELEASE_DATE_COVERAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = ["dataset", "release_rule_id", "microdata_release_date", "rows", "unmatched_rows"]
    missing = [column for column in columns if column not in coverage.columns]
    if missing:
        raise SystemExit(f"SCE release-date coverage is missing columns: {missing}")

    if RELEASE_DATE_COVERAGE_PATH.exists():
        existing = pd.read_csv(RELEASE_DATE_COVERAGE_PATH)
        missing_existing = [column for column in columns if column not in existing.columns]
        if missing_existing:
            raise SystemExit(f"{RELEASE_DATE_COVERAGE_PATH} is missing columns: {missing_existing}")
        existing = existing.loc[existing["dataset"].ne(dataset), columns].copy()
        output = pd.concat([existing, coverage.loc[:, columns]], ignore_index=True)
    else:
        output = coverage.loc[:, columns].copy()

    output["rows"] = pd.to_numeric(output["rows"], errors="raise").astype(int)
    output["unmatched_rows"] = pd.to_numeric(output["unmatched_rows"], errors="raise").astype(int)
    output = output.sort_values(["dataset", "release_rule_id", "microdata_release_date"], kind="mergesort")
    output.to_csv(RELEASE_DATE_COVERAGE_PATH, index=False)


def main() -> None:
    # 1. Resolve paths and verify the documented source inventory.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--intermediate-dir", type=Path, default=INTERMEDIATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    validate_source_inputs("sce", overrides={RAW_DIR: args.raw_dir})
    args.intermediate_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    expected = set(ADDITIONAL_MICRODATA_FILES)
    expected.update(spec["relative_path"] for spec in MICRODATA_SPECS)
    missing = sorted(rel for rel in expected if not (args.raw_dir / rel).is_file())
    if missing:
        raise FileNotFoundError("Missing expected SCE files:\n" + "\n".join(missing))
    release_crosswalk = load_sce_release_date_crosswalk()

    source_inventory = build_source_inventory(args.raw_dir)
    source_inventory.to_csv(args.output_dir / "source_inventory.csv", index=False)

    # 2. Read the module releases and retain their source-column inventories.
    pieces_by_table: dict[str, list[pd.DataFrame]] = {table: [] for table in TABLE_OUTPUTS}
    raw_columns_by_table: dict[str, list[str]] = {table: [] for table in TABLE_OUTPUTS}
    input_rows = []

    for spec in MICRODATA_SPECS:
        table_name = str(spec["table"])
        module = str(spec["module"])
        rel = str(spec["relative_path"])
        df = read_microdata(args.raw_dir, spec, release_crosswalk)
        raw_cols = [col for col in df.columns if col not in STRUCTURAL_COLUMNS]
        for col in raw_cols:
            if col not in raw_columns_by_table[table_name]:
                raw_columns_by_table[table_name].append(col)
        pieces_by_table[table_name].append(df)
        min_ref, max_ref = reference_month_bounds(df)
        input_rows.append(
            {
                "module": spec["module"],
                "table": table_name,
                "raw_file": spec["relative_path"],
                "rows": len(df),
                "columns": df.shape[1],
                "min_survey_date": min_ref,
                "max_survey_date": max_ref,
                "notes": spec["notes"],
                **release_summary_fields(df),
            }
        )

    for df in read_housing_microdata(args.raw_dir, release_crosswalk):
        table_name = "housing"
        raw_cols = [col for col in df.columns if col not in STRUCTURAL_COLUMNS]
        for col in raw_cols:
            if col not in raw_columns_by_table[table_name]:
                raw_columns_by_table[table_name].append(col)
        pieces_by_table[table_name].append(df)
        min_ref, max_ref = reference_month_bounds(df)
        source_year = int(df["source_wave_year"].iloc[0])
        input_rows.append(
            {
                "module": "housing",
                "table": table_name,
                "raw_file": df["raw_file"].iloc[0],
                "rows": len(df),
                "columns": df.shape[1],
                "min_survey_date": min_ref,
                "max_survey_date": max_ref,
                "notes": f"Housing annual February survey, Data {source_year}; source_wave_year retained.",
                **release_summary_fields(df),
            }
        )

    job_search = read_job_search_microdata(args.raw_dir, release_crosswalk)
    raw_cols = [col for col in job_search.columns if col not in STRUCTURAL_COLUMNS]
    for col in raw_cols:
        if col not in raw_columns_by_table["job_search"]:
            raw_columns_by_table["job_search"].append(col)
    pieces_by_table["job_search"].append(job_search)
    min_ref, max_ref = reference_month_bounds(job_search)
    input_rows.append(
        {
            "module": "job_search",
            "table": "job_search",
            "raw_file": job_search["raw_file"].iloc[0],
            "rows": len(job_search),
            "columns": job_search.shape[1],
            "min_survey_date": min_ref,
            "max_survey_date": max_ref,
            "notes": "Job Search annual October supplement; date derived from year as YYYY10.",
            **release_summary_fields(job_search),
        }
    )

    input_summaries = pd.DataFrame(input_rows)

    descriptions = pd.read_csv(MAPPING_DIR / "sce_question_descriptions.csv", keep_default_na=False)
    questionnaire_markers = {
        module: dict(zip(rows["marker"], rows["question_text"], strict=True))
        for module, rows in descriptions.groupby("module", sort=False)
    }
    job_descriptions = pd.read_csv(MAPPING_DIR / "sce_job_search_descriptions.csv", keep_default_na=False)
    module_metadata = {
        "housing": parse_housing_codebooks(args.raw_dir),
        "job_search": job_descriptions.set_index("variable").to_dict("index"),
    }

    base_tables: dict[str, pd.DataFrame] = {}
    for table_name, pieces in pieces_by_table.items():
        if not pieces:
            raise ValueError(f"No input pieces were read for table {table_name}")
        base_tables[table_name] = pd.concat(pieces, ignore_index=True, sort=False)

    core = base_tables["core"]
    require_core_columns = ["respondent_id", "survey_year", "survey_month", "panel_month", "Q32", "Q47", "D6"]
    missing_core_columns = sorted(set(require_core_columns) - set(core.columns))
    if missing_core_columns:
        raise ValueError(f"core: missing columns required for harmonised age and income: {missing_core_columns}")
    core_month_index = pd.to_numeric(core["survey_year"], errors="raise") * 12 + pd.to_numeric(
        core["survey_month"], errors="raise"
    )
    reported_age = pd.to_numeric(core["Q32"], errors="coerce")
    first_reported_age = (
        pd.DataFrame(
            {
                "respondent_id": core["respondent_id"],
                "first_age_month_index": core_month_index,
                "first_reported_age": reported_age,
            }
        )
        .loc[reported_age.notna()]
        .sort_values(["respondent_id", "first_age_month_index"], kind="mergesort")
        .drop_duplicates("respondent_id", keep="first")
    )
    age_inputs = pd.DataFrame({"respondent_id": core["respondent_id"], "core_month_index": core_month_index}).merge(
        first_reported_age,
        on="respondent_id",
        how="left",
        validate="many_to_one",
    )
    advanced_age = np.floor(
        age_inputs["first_reported_age"]
        + (age_inputs["core_month_index"] - age_inputs["first_age_month_index"]) / 12.0
    )
    core["age_years"] = reported_age.where(reported_age.notna(), advanced_age)
    panel_month = pd.to_numeric(core["panel_month"], errors="coerce")
    core["_HH_INC_DETAILED"] = core["Q47"].where(panel_month.eq(1), core["D6"].where(panel_month.gt(1)))
    base_tables["core"] = core

    core_weights = base_tables["core"][["respondent_id", "survey_year", "survey_month", "weight_core"]].copy()
    for table_name in ["household_spending", "labor_market", "public_policy"]:
        table = base_tables[table_name]
        weight_col = f"weight_{table_name}"
        if weight_col in table.columns:
            table["missing_core_weight"] = table[weight_col].isna()
            continue
        table = table.merge(core_weights, on=["respondent_id", "survey_year", "survey_month"], how="left")
        table = table.rename(columns={"weight_core": weight_col})
        table["missing_core_weight"] = table[weight_col].isna()
        base_tables[table_name] = table

    # Harmonise each module and write its constructed table.
    tables: dict[str, pd.DataFrame] = {}
    harmonization_rule_frames = []
    categorical_label_rule_frames = []
    for table_name, table in base_tables.items():
        check_duplicate_keys(table_name, table)
        table, harmonization_rules = apply_harmonization(table, table_name)
        table = add_density_family_filters(table, table_name)
        table, categorical_label_rules = add_categorical_label_columns(table, table_name, module_metadata)
        harmonization_rule_frames.append(harmonization_rules)
        if table_name == "labor_market":
            table, reservation_wage_rule = add_labor_market_reservation_wage_annual_full_time(table)
            harmonization_rule_frames.append(reservation_wage_rule)
        if table_name == "job_search":
            table, job_search_special_rules = add_job_search_documented_special_code_companions(table)
            table, months_no_work_rule = add_job_search_months_no_work(table)
            harmonization_rule_frames.extend([job_search_special_rules, months_no_work_rule])
        categorical_label_rule_frames.append(categorical_label_rules)
        table = convert_object_columns_for_parquet(table)
        table.to_parquet(args.intermediate_dir / TABLE_OUTPUTS[table_name], index=False)
        tables[table_name] = table

    harmonization_rule_rows = []
    for frame in harmonization_rule_frames:
        harmonization_rule_rows.extend(frame.dropna(how="all").to_dict("records"))
    if not harmonization_rule_rows:
        raise RuntimeError("No SCE harmonization rules were generated")
    harmonization_rules = pd.DataFrame.from_records(harmonization_rule_rows)
    harmonization_rules.to_csv(args.output_dir / "harmonization_rules.csv", index=False)

    categorical_label_rule_rows = []
    for frame in categorical_label_rule_frames:
        categorical_label_rule_rows.extend(frame.dropna(how="all").to_dict("records"))
    if not categorical_label_rule_rows:
        raise RuntimeError("No SCE categorical label rules were generated")
    categorical_label_rules = pd.DataFrame.from_records(categorical_label_rule_rows)
    categorical_label_rules.to_csv(args.output_dir / "categorical_label_rules.csv", index=False)

    # Record variable metadata, source provenance and release-date coverage.
    variable_inventory = build_variable_inventory(
        tables, raw_columns_by_table, module_metadata, questionnaire_markers,
    )
    variable_inventory.to_csv(args.output_dir / "variable_inventory.csv", index=False)
    metadata_audit = write_metadata_coverage_audit(variable_inventory, args.output_dir)

    manifest = write_manifest(args.intermediate_dir, source_inventory, input_summaries)
    shared_release_coverage = build_shared_release_coverage(tables)
    with exclusive_lock(RELEASE_DATE_COVERAGE_PATH):
        update_release_date_coverage("SCE", shared_release_coverage)

    summary_rows = []
    for table_name, table in tables.items():
        min_ref, max_ref = reference_month_bounds(table)
        summary_rows.append(
            {
                "table": table_name,
                "output_file": f"data/intermediate/{TABLE_OUTPUTS[table_name]}",
                "rows": len(table),
                "columns": table.shape[1],
                "min_survey_date": min_ref,
                "max_survey_date": max_ref,
                "duplicate_respondent_month_keys": int(table.duplicated(["respondent_id", "survey_year", "survey_month"]).sum()),
                "pipeline_version": PIPELINE_VERSION,
            }
        )
    build_summary = pd.DataFrame(summary_rows)
    build_summary.to_csv(args.output_dir / "build_summary.csv", index=False)
    input_summaries.to_csv(args.output_dir / "input_file_summary.csv", index=False)

    print("Wrote SCE outputs:")
    for row in summary_rows:
        print(f"  {row['output_file']}: {row['rows']} rows x {row['columns']} columns")
    print(f"  data/intermediate/sce_release_manifest.parquet: {len(manifest)} rows")
    print(f"  output/preprocessing/release_dates/metadata_coverage.csv: {len(shared_release_coverage)} SCE release-date rows")
    print(f"  output/preprocessing/sce/build/source_inventory.csv: {len(source_inventory)} rows")
    print(f"  output/preprocessing/sce/build/variable_inventory.csv: {len(variable_inventory)} rows")
    print(f"  output/preprocessing/sce/build/categorical_label_rules.csv: {len(categorical_label_rules)} rows")
    print(f"  output/preprocessing/sce/build/metadata_coverage_audit.csv: {len(metadata_audit)} rows")


if __name__ == "__main__":
    main()
