#!/usr/bin/env python
"""Build the PSID person-by-wave panel from both PSID-SHELF source archives.

First clean the long-format release, then append harmonised Complete Main Study
measurements and publish the enriched panel as data/intermediate/psid_main.parquet.
Family measurements are repeated on current family members' rows. Task-specific
displacement episodes and prediction samples are constructed downstream.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.io.stata import StataReader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.source_inputs import validate_source_inputs

from scripts.utils.io import exclusive_lock, sha256_file
COMPLETE_MAIN_STUDY_ARCHIVE_PATH = (
    PROJECT_ROOT
    / "data/raw/micro/psid-shelf/Construction_Files/Data/PSID_COMPLETE_MAIN_STUDY_1968_2021_FULL_10.8_GB.zip"
)
COMPLETE_MAIN_STUDY_OUTPUT_DIR = PROJECT_ROOT / "output/preprocessing/psid/complete_main_study"

RAW_ZIP_PATH = Path("data/raw/micro/psid-shelf/PSIDSHELF_1968_2021_LONG_7.9_GB.zip")
OUT_DATA_PATH = Path("data/intermediate/psid_main.parquet")
OUT_METADATA_PATH = Path("data/intermediate/psid_main_metadata.json")
OUT_DIR = Path("output/preprocessing/psid/build")
RELEASE_DATE_CROSSWALK_PATH = Path("data/raw/release_dates/release_psid.csv")
RELEASE_DATE_COVERAGE_PATH = Path("output/preprocessing/release_dates/metadata_coverage.csv")

PSIDSHELF_PUBLIC_PROVENANCE = {
    "public_project": "openICPSR project 194322",
    "title": "PSID-SHELF, 1968-2021: The PSID's Social, Health, and Economic Longitudinal File (PSID-SHELF), Beta Release.",
    "version": "V2",
    "version_title": "2025-01",
    "published_distributor_date": "2025-02-24",
    "version_doi": "https://doi.org/10.3886/E194322V2",
    "permanent_data_doi": "10.3886/E194322",
    "documentation_doi": "10.7302/25205",
    "authors": ["Fabian T. Pfeffer", "Davis Daumler", "Esther Friedman"],
    "status": "beta release",
    "dependency_note": "Repository build starts from pre-harmonized PSID-SHELF beta derivative, not from a from-scratch PSID family-file and individual-file reconstruction.",
}

EXPECTED_YEARS = list(range(1968, 1998)) + list(range(1999, 2022, 2))
SUBSAMPLE_ORDER = [
    "src",
    "seo",
    "latino",
    "immigrant_1997_1999",
    "immigrant_2017_2019",
    "unknown",
]
SUBSAMPLE_LABELS = {
    "src": "Main sample (SRC)",
    "seo": "Survey of Economic Opportunity (SEO)",
    "latino": "Latino sample",
    "immigrant_1997_1999": "Immigrant refresher, 1997/1999",
    "immigrant_2017_2019": "Immigrant refresher, 2017/2019",
    "unknown": "Unknown / outside documented lineage ranges",
}
DERIVED_LABELS = {
    "subsample": "Derived PSID subsample from LINEAGE",
    "sample_person": "Derived indicator: SAMPSTAT != 0",
    "is_reference_person": "Derived indicator: REL == 1",
    "income_year": "Derived tax year for family income variables (YEAR - 1)",
    "occ_major_harmonized": "Derived broad occupation group from the most recent available PSID occupation coding.",
    "occ_source": "Derived source occupation variable used for occ_major_harmonized.",
    "occ_major_harmonized_label": "Derived label for occ_major_harmonized.",
}
PSID_CATEGORICAL_LABEL_MAPS = {
    "demo_sex": {
        1: "Male",
        2: "Female",
    },
    "race_eth_maj_col": {
        1: "White",
        2: "Black",
        3: "Other race",
        4: "Hispanic",
    },
    "geo_region": {
        1: "Northeast",
        2: "Midwest",
        3: "South",
        4: "West",
        5: "Alaska or Hawaii",
        6: "Country outside the United States",
    },
    "geo_metro": {
        0: "No, non-metropolitan area",
        1: "Yes, metropolitan area",
        9: "N/A, not classified",
    },
    "edu_level_max": {
        0: "Did not complete high school",
        1: "Completed high school, did not attend college",
        2: "Attended college, no bachelor's degree",
        3: "Bachelor's degree, no postgraduate degree",
        4: "Postgraduate degree",
    },
    "emp_work": {
        0: "Not currently working",
        1: "Currently working",
    },
    "home_stat": {
        1: "Owns home",
        2: "Pays rent",
        3: "Neither owns home nor pays rent",
    },
    "fam_partnered": {
        0: "No, reference person has no spouse or partner in the family unit",
        1: "Yes, reference person has a spouse or partner in the family unit",
    },
    "fam_married": {
        0: "No, reference person is not legally married to a spouse in the family unit",
        1: "Yes, reference person is legally married to a spouse in the family unit",
    },
}
PSID_CATEGORICAL_LABEL_DESCRIPTIONS = {
    f"{variable}_label": f"Label for PSID-SHELF harmonized categorical variable {variable}."
    for variable in PSID_CATEGORICAL_LABEL_MAPS
}
DERIVED_LABELS.update(PSID_CATEGORICAL_LABEL_DESCRIPTIONS)
OCC_MAJOR_LABELS = {
    "professional_managerial": "Professional or managerial occupation",
    "sales_office_service": "Sales, office, or service occupation",
    "farming_construction_repair": "Farming, construction, or repair occupation",
    "production_transport_labor": "Production, transport, or labor occupation",
    "military_other": "Military or other occupation",
}


# Wave-specific source fields for the Complete Main Study enrichment.
SOURCE_WAVES = list(range(1968, 1993))
QUALIFYING_REASON_CODES = {1, 3}
REASON_LABELS = {
    0: "inapplicable",
    1: "company folded/changed hands/moved; employer died or went out of business",
    2: "strike or lockout",
    3: "laid off or fired",
    4: "quit/resigned/retired/pregnant or other voluntary change",
    5: "first job or not previously working",
    6: "promotion or previously self-employed, depending on wave",
    7: "other, transfer, or armed services",
    8: "job completed, seasonal work, or temporary job",
    9: "not ascertained or do not know",
}
HARMONISED_REASONS = {
    1: "plant_closure_or_employer_move",
    3: "layoff_or_firing",
}
VALID_ACCURACY_CODES = {0, 1, 2}
HOURS_TOPCODE = 9_999
EARNINGS_TOPCODE = 9_999_999
CHUNK_ROWS = 512
OFFICIAL_CODEBOOK = (
    "https://psidonline.isr.umich.edu/documents/psid/codebook/FAM{year}_codebook.pdf"
)


# Each tuple is employed-route field, unemployed-route field, employed
# questionnaire reference, and unemployed questionnaire reference.
REFERENCE_REASON_VARIABLES = {
    1968: ("V201", "V201", "F7", "G5"),
    1969: ("V643", "V651", "D6", "E6"),
    1970: ("V1282", "V1332", "D6", "E6a"),
    1971: ("V1988", "V2038", "D6", "E6b"),
    1972: ("V2586", "V2638", "D6", "E6b"),
    1973: ("V3119", "V3155", "D6", "E6b"),
    1974: ("V3534", "V3571", "D7", "E8"),
    1975: ("V3986", "V4026", "D20", "E9"),
    1976: ("V4490", "V4556", "D34", "E15"),
    1977: ("V5399", "V5458", "D28", "E18"),
    1978: ("V5890", "V5986", "D20", "E29"),
    1979: ("V6501", "V6559", "C12", "D15"),
    1980: ("V7104", "V7161", "C12", "D14"),
    1981: ("V7727", "V7809", "C23", "D11"),
    1982: ("V8391", "V8470", "C20", "D10"),
    1983: ("V9022", "V9107", "C20", "D11"),
    1984: ("V10539", "V10609", "C82", "D13"),
    1985: ("V11679", "V11764", "B44", "C25"),
    1986: ("V13079", "V13160", "B35", "C25"),
    1987: ("V14177", "V14256", "B32", "C22"),
    1988: ("V15240", "V15328", "B56", "C15"),
    1989: ("V16741", "V16843", "B56", "C15"),
    1990: ("V18179", "V18267", "B56", "C15"),
    1991: ("V19479", "V19567", "B56", "C15"),
    1992: ("V20779", "V20867", "B56", "C15"),
}

SPOUSE_REASON_VARIABLES = {
    1976: ("V4873", "V4940", "D34", "E16"),
    1979: ("V6600", "V6631", "F13", "G8"),
    1980: ("V7202", "V7233", "F13", "G8"),
    1981: ("V7893", "V7922", "F18", "G8"),
    1982: ("V8551", "V8577", "F17", "G8"),
    1983: ("V9201", "V9236", "F17", "G9"),
    1984: ("V10753", "V10809", "F79", "G11"),
    1985: ("V12042", "V12127", "J44", "K25"),
    1986: ("V13256", "V13328", "D33", "E23"),
    1987: ("V14350", "V14420", "D30", "E20"),
    1988: ("V15542", "V15630", "D56", "E15"),
    1989: ("V17060", "V17162", "D56", "E15"),
    1990: ("V18481", "V18569", "D56", "E15"),
    1991: ("V19781", "V19869", "D56", "E15"),
    1992: ("V21081", "V21169", "D56", "E15"),
}

ROLE_REASON_VARIABLES = {
    "rp": REFERENCE_REASON_VARIABLES,
    "sp": SPOUSE_REASON_VARIABLES,
}


def hours_spec(
    year: int,
    rp_hours: str,
    rp_accuracy: str,
    sp_hours: str,
    sp_accuracy: str,
) -> dict[str, object]:
    return {
        "survey_year": year,
        "reference_year": year - 1,
        "rp_hours": rp_hours,
        "rp_accuracy": rp_accuracy,
        "sp_hours": sp_hours,
        "sp_accuracy": sp_accuracy,
    }


HOURS_VARIABLES = {
    1968: hours_spec(1968, "V47", "V48", "V53", "V54"),
    1969: hours_spec(1969, "V465", "V466", "V475", "V476"),
    1970: hours_spec(1970, "V1138", "V1139", "V1148", "V1149"),
    1971: hours_spec(1971, "V1839", "V1840", "V1849", "V1850"),
    1972: hours_spec(1972, "V2439", "V2440", "V2449", "V2450"),
    1973: hours_spec(1973, "V3027", "V3028", "V3035", "V3036"),
    1974: hours_spec(1974, "V3423", "V3424", "V3431", "V3432"),
    1975: hours_spec(1975, "V3823", "V3824", "V3831", "V3832"),
    1976: hours_spec(1976, "V4332", "V4333", "V4344", "V4345"),
    1977: hours_spec(1977, "V5232", "V5233", "V5244", "V5245"),
    1978: hours_spec(1978, "V5731", "V5732", "V5743", "V5744"),
    1979: hours_spec(1979, "V6336", "V6337", "V6348", "V6349"),
    1980: hours_spec(1980, "V6934", "V6935", "V6946", "V6947"),
    1981: hours_spec(1981, "V7530", "V7531", "V7540", "V7541"),
    1982: hours_spec(1982, "V8228", "V8229", "V8238", "V8239"),
    1983: hours_spec(1983, "V8830", "V8831", "V8840", "V8841"),
    1984: hours_spec(1984, "V10037", "V10038", "V10131", "V10132"),
    1985: hours_spec(1985, "V11146", "V11141", "V11258", "V11253"),
    1986: hours_spec(1986, "V12545", "V12540", "V12657", "V12652"),
    1987: hours_spec(1987, "V13745", "V13740", "V13809", "V13804"),
    1988: hours_spec(1988, "V14835", "V14830", "V14865", "V14860"),
    1989: hours_spec(1989, "V16335", "V16330", "V16365", "V16360"),
    1990: hours_spec(1990, "V17744", "V17739", "V17774", "V17769"),
    1991: hours_spec(1991, "V19044", "V19039", "V19074", "V19069"),
    1992: hours_spec(1992, "V20344", "V20339", "V20374", "V20369"),
}


def food_spec(
    year: int,
    home: str,
    away: str,
    stamps: str,
    *,
    home_topcode: int,
    stamp_topcode: int,
    stamp_period: str,
) -> dict[str, object]:
    return {
        "survey_year": year,
        "home": home,
        "home_accuracy": f"V{int(home[1:]) + 1}",
        "home_topcode": home_topcode,
        "away": away,
        "away_accuracy": f"V{int(away[1:]) + 1}",
        "away_topcode": 9_999,
        "stamps": stamps,
        "stamps_accuracy": f"V{int(stamps[1:]) + 1}",
        "stamps_topcode": stamp_topcode,
        "stamps_reference_period": stamp_period,
        "stamps_annualisation_factor": 12.0 if stamp_period == "last_month" else 1.0,
    }


FOOD_VARIABLES = {
    1969: food_spec(1969, "V500", "V506", "V510", home_topcode=9_999, stamp_topcode=9_999, stamp_period="previous_calendar_year"),
    1970: food_spec(1970, "V1175", "V1185", "V1183", home_topcode=9_999, stamp_topcode=9_999, stamp_period="previous_calendar_year"),
    1971: food_spec(1971, "V1876", "V1886", "V1884", home_topcode=9_999, stamp_topcode=9_999, stamp_period="previous_calendar_year"),
    1972: food_spec(1972, "V2476", "V2480", "V2478", home_topcode=9_999, stamp_topcode=9_999, stamp_period="previous_calendar_year"),
    1974: food_spec(1974, "V3441", "V3445", "V3443", home_topcode=9_999, stamp_topcode=9_999, stamp_period="previous_calendar_year"),
    1975: food_spec(1975, "V3841", "V3853", "V3851", home_topcode=9_999, stamp_topcode=9_999, stamp_period="previous_calendar_year"),
    1976: food_spec(1976, "V4354", "V4368", "V4364", home_topcode=9_999, stamp_topcode=9_999, stamp_period="previous_calendar_year"),
    1977: food_spec(1977, "V5271", "V5273", "V5269", home_topcode=9_999, stamp_topcode=999, stamp_period="last_month"),
    1978: food_spec(1978, "V5770", "V5772", "V5768", home_topcode=9_999, stamp_topcode=999, stamp_period="last_month"),
    1979: food_spec(1979, "V6376", "V6378", "V6374", home_topcode=9_999, stamp_topcode=999, stamp_period="last_month"),
    1980: food_spec(1980, "V6972", "V6974", "V6970", home_topcode=9_999, stamp_topcode=999, stamp_period="last_month"),
    1981: food_spec(1981, "V7564", "V7566", "V7562", home_topcode=9_999, stamp_topcode=999, stamp_period="last_month"),
    1982: food_spec(1982, "V8256", "V8258", "V8254", home_topcode=9_999, stamp_topcode=999, stamp_period="last_month"),
    1983: food_spec(1983, "V8864", "V8866", "V8862", home_topcode=99_999, stamp_topcode=999, stamp_period="last_month"),
    1984: food_spec(1984, "V10235", "V10237", "V10233", home_topcode=99_999, stamp_topcode=999, stamp_period="last_month"),
    1985: food_spec(1985, "V11375", "V11377", "V11373", home_topcode=99_999, stamp_topcode=999, stamp_period="last_month"),
    1986: food_spec(1986, "V12774", "V12776", "V12772", home_topcode=99_999, stamp_topcode=999, stamp_period="last_month"),
    1987: food_spec(1987, "V13876", "V13878", "V13874", home_topcode=99_999, stamp_topcode=999, stamp_period="last_month"),
    1990: food_spec(1990, "V17807", "V17809", "V17805", home_topcode=99_999, stamp_topcode=999, stamp_period="last_month"),
    1991: food_spec(1991, "V19107", "V19109", "V19105", home_topcode=99_999, stamp_topcode=999, stamp_period="last_month"),
    1992: food_spec(1992, "V20407", "V20409", "V20405", home_topcode=99_999, stamp_topcode=999, stamp_period="last_month"),
}

FOOD_UNAVAILABLE_WAVES = {1968, 1973, 1988, 1989}


def source_columns() -> list[str]:
    columns = ["ID"]
    for role_variables in ROLE_REASON_VARIABLES.values():
        for employed, unemployed, _, _ in role_variables.values():
            columns.extend([employed, unemployed])
    for spec in HOURS_VARIABLES.values():
        columns.extend([str(spec[key]) for key in ["rp_hours", "rp_accuracy", "sp_hours", "sp_accuracy"]])
    for spec in FOOD_VARIABLES.values():
        columns.extend(
            [
                str(spec[key])
                for key in ["home", "home_accuracy", "away", "away_accuracy", "stamps", "stamps_accuracy"]
            ]
        )
    return list(dict.fromkeys(columns))


COMPLETE_MAIN_STUDY_COLUMNS = [
    "reference_person_id",
    "spouse_person_id",
    "demo_age_gen_rp",
    "demo_age_gen_sp",
    "demo_sex_rp",
    "demo_sex_sp",
    "demo_sex_label_rp",
    "demo_sex_label_sp",
    "labor_reference_year",
    "annual_hours_rp",
    "annual_hours_sp",
    "annual_hours_valid_rp",
    "annual_hours_valid_sp",
    "annual_hours_accuracy_code_rp",
    "annual_hours_accuracy_code_sp",
    "annual_hours_assignment_status_rp",
    "annual_hours_assignment_status_sp",
    "annual_work_status_rp",
    "annual_work_status_sp",
    "job_end_reason_ascertained_rp",
    "job_end_reason_ascertained_sp",
    "job_end_reason_employed_code_rp",
    "job_end_reason_employed_code_sp",
    "job_end_reason_unemployed_code_rp",
    "job_end_reason_unemployed_code_sp",
    "food_at_home_nominal_annual",
    "food_away_nominal_annual",
    "food_stamps_net_nominal_annual",
    "food_at_home_assignment_status",
    "food_away_assignment_status",
    "food_stamps_assignment_status",
    "food_stamps_reference_period",
    "food_stamps_annualisation_factor",
    "food_cash_nominal_annual",
    "food_cash_valid",
    "food_cash_positive",
    "food_total_including_assistance_nominal_annual",
]

COMPLETE_MAIN_STUDY_LABELS = {
    "reference_person_id": "Permanent person ID of the family reference person in this survey wave.",
    "spouse_person_id": "Permanent person ID of the reference person's spouse or partner in this survey wave.",
    "demo_age_gen_rp": "Generated age of the family reference person in this survey wave.",
    "demo_age_gen_sp": "Generated age of the reference person's spouse or partner in this survey wave.",
    "demo_sex_rp": "Sex code of the family reference person in this survey wave.",
    "demo_sex_sp": "Sex code of the reference person's spouse or partner in this survey wave.",
    "demo_sex_label_rp": "Sex label of the family reference person in this survey wave.",
    "demo_sex_label_sp": "Sex label of the reference person's spouse or partner in this survey wave.",
    "labor_reference_year": "Calendar year to which Complete Main Study annual labor measurements refer; survey year minus one.",
    "annual_hours_rp": "Valid annual work hours of the family reference person in the preceding calendar year.",
    "annual_hours_sp": "Valid annual work hours of the reference person's spouse or partner in the preceding calendar year.",
    "annual_hours_valid_rp": "Indicator that reference-person annual hours satisfy the source value and accuracy-code contract.",
    "annual_hours_valid_sp": "Indicator that spouse or partner annual hours satisfy the source value and accuracy-code contract.",
    "annual_hours_accuracy_code_rp": "Source assignment or accuracy code for reference-person annual hours.",
    "annual_hours_accuracy_code_sp": "Source assignment or accuracy code for spouse or partner annual hours.",
    "annual_hours_assignment_status_rp": "Harmonised assignment status for reference-person annual hours.",
    "annual_hours_assignment_status_sp": "Harmonised assignment status for spouse or partner annual hours.",
    "annual_work_status_rp": "Indicator that valid reference-person annual hours are positive.",
    "annual_work_status_sp": "Indicator that valid spouse or partner annual hours are positive.",
    "job_end_reason_ascertained_rp": "Indicator that comparable reference-person job-ending reason routes were collected in this wave.",
    "job_end_reason_ascertained_sp": "Indicator that comparable spouse or partner job-ending reason routes were collected in this wave.",
    "job_end_reason_employed_code_rp": "Reference-person job-ending reason code on the employed or re-employed questionnaire route.",
    "job_end_reason_employed_code_sp": "Spouse or partner job-ending reason code on the employed or re-employed questionnaire route.",
    "job_end_reason_unemployed_code_rp": "Reference-person job-ending reason code on the unemployed questionnaire route.",
    "job_end_reason_unemployed_code_sp": "Spouse or partner job-ending reason code on the unemployed questionnaire route.",
    "food_at_home_nominal_annual": "Annual nominal family cash spending on food at home.",
    "food_away_nominal_annual": "Annual nominal family cash spending on food away from home.",
    "food_stamps_net_nominal_annual": "Annual nominal net value of family food-stamp assistance.",
    "food_at_home_assignment_status": "Harmonised assignment status for food-at-home spending.",
    "food_away_assignment_status": "Harmonised assignment status for food-away spending.",
    "food_stamps_assignment_status": "Harmonised assignment status for net food-stamp assistance.",
    "food_stamps_reference_period": "Original reference period of the food-stamp measure.",
    "food_stamps_annualisation_factor": "Factor used to annualise the source food-stamp measure.",
    "food_cash_nominal_annual": "Annual nominal family cash food spending: food at home plus food away from home.",
    "food_cash_valid": "Indicator that both cash food components satisfy their source contracts.",
    "food_cash_positive": "Indicator that valid annual nominal cash food spending is positive.",
    "food_total_including_assistance_nominal_annual": "Annual nominal family food resources including cash food spending and net food-stamp assistance.",
}

if set(COMPLETE_MAIN_STUDY_LABELS) != set(COMPLETE_MAIN_STUDY_COLUMNS):
    raise RuntimeError("Complete Main Study labels do not match the enriched-column contract.")

ENRICHED_INTEGER_DTYPES = {
    "reference_person_id": "Int64",
    "spouse_person_id": "Int64",
    "demo_sex_rp": "Int8",
    "demo_sex_sp": "Int8",
    "labor_reference_year": "Int64",
    "annual_hours_valid_rp": "Int8",
    "annual_hours_valid_sp": "Int8",
    "annual_hours_accuracy_code_rp": "Int8",
    "annual_hours_accuracy_code_sp": "Int8",
    "job_end_reason_ascertained_rp": "Int8",
    "job_end_reason_ascertained_sp": "Int8",
    "job_end_reason_employed_code_rp": "Int8",
    "job_end_reason_employed_code_sp": "Int8",
    "job_end_reason_unemployed_code_rp": "Int8",
    "job_end_reason_unemployed_code_sp": "Int8",
    "food_cash_valid": "Int8",
    "food_cash_positive": "Int8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-zip", type=Path, default=RAW_ZIP_PATH)
    parser.add_argument("--out-data", type=Path, default=OUT_DATA_PATH)
    parser.add_argument("--metadata-path", type=Path, default=OUT_METADATA_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--chunksize", type=int, default=20_000)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument(
        "--complete-main-study-archive",
        type=Path,
        default=COMPLETE_MAIN_STUDY_ARCHIVE_PATH,
        help="Complete Main Study archive distributed with the PSID-SHELF construction files.",
    )
    parser.add_argument(
        "--skip-complete-main-study-enrichment",
        action="store_true",
        help="Build only the legacy PSID-SHELF columns. Intended for migration diagnostics.",
    )
    parser.add_argument("--max-chunks", type=int, default=None, help="Optional smoke-test cap.")
    parser.add_argument("--sample-persons-only", action="store_true")
    parser.add_argument("--reference-person-only", action="store_true")
    parser.add_argument("--src-only", action="store_true")
    parser.add_argument(
        "--exclude-subsample",
        action="append",
        default=[],
        choices=[s for s in SUBSAMPLE_ORDER if s != "unknown"],
        help="Optional subsample exclusion. May be passed multiple times.",
    )
    parser.add_argument("--year-min", type=int, default=None)
    parser.add_argument("--year-max", type=int, default=None)
    return parser.parse_args()


def assert_inputs(
    raw_zip: Path,
    complete_main_study_archive: Path | None = None,
) -> None:
    if not raw_zip.exists():
        raise SystemExit(f"Missing raw PSID-SHELF zip: {raw_zip}")
    if complete_main_study_archive is not None and not complete_main_study_archive.exists():
        raise SystemExit(f"Missing PSID-SHELF Complete Main Study archive: {complete_main_study_archive}")


def resolve_member_name(raw_zip: Path) -> str:
    with zipfile.ZipFile(raw_zip) as zf:
        members = [info.filename for info in zf.infolist() if not info.is_dir()]
    if len(members) != 1:
        raise SystemExit(f"Expected exactly one file in {raw_zip}, found {len(members)}: {members}")
    return members[0]


def extract_member(raw_zip: Path, member_name: str, directory: Path) -> Path:
    """Decompress the Stata member once to a local file and return its path.

    pandas' Stata reader seeks back to the file's string table before every chunk it reads.
    On a zip stream each backward seek restarts decompression, so reading the archive
    directly decompresses the whole 7.9 GB member once per 20,000-row chunk. The extracted
    file holds the same bytes, so the chunks, and the output, do not change.
    """
    path = directory / Path(member_name).name
    with zipfile.ZipFile(raw_zip) as zf, zf.open(member_name) as source, path.open("wb") as target:
        shutil.copyfileobj(source, target, length=1 << 24)
    return path


def get_source_metadata(stata_path: Path) -> tuple[dict[str, str | int | None], dict[str, str]]:
    with pd.read_stata(stata_path, convert_categoricals=False, iterator=True) as reader:
        variable_labels = {k.lower(): v for k, v in reader.variable_labels().items()}
        data_label = reader.data_label
        first_row = reader.read(1)

    first_row.columns = [c.lower() for c in first_row.columns]
    meta = {
        "source_data_label": data_label,
        "psid_retrieve": _first_scalar(first_row, "psid_retrieve"),
        "psidshelf_compile": _first_scalar(first_row, "psidshelf_compile"),
        "psidshelf_release": _first_scalar(first_row, "psidshelf_release"),
    }
    return meta, variable_labels


def _first_scalar(df: pd.DataFrame, col: str) -> int | str | None:
    if col not in df.columns or df.empty:
        return None
    value = df.iloc[0][col]
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        as_int = int(value)
        return as_int if float(as_int) == float(value) else float(value)
    return str(value)


def derive_subsample(lineage: pd.Series) -> pd.Series:
    lin = pd.to_numeric(lineage, errors="coerce")
    out = pd.Series("unknown", index=lineage.index, dtype="string")
    out.loc[lin.between(1, 2930, inclusive="both")] = "src"
    out.loc[lin.between(3001, 3511, inclusive="both")] = "immigrant_1997_1999"
    out.loc[lin.between(4001, 4851, inclusive="both")] = "immigrant_2017_2019"
    out.loc[lin.between(5001, 6872, inclusive="both")] = "seo"
    out.loc[lin.between(7001, 9308, inclusive="both")] = "latino"
    return out


def integer_codes(series: pd.Series, variable: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    source_nonmissing = series.notna()
    nonnumeric = source_nonmissing & numeric.isna()
    rounded = numeric.round()
    noninteger = numeric.notna() & ((numeric - rounded).abs() > 1e-9)
    if nonnumeric.any() or noninteger.any():
        bad = nonnumeric | noninteger
        examples = sorted(series.loc[bad].dropna().astype(str).unique().tolist())[:20]
        raise RuntimeError(f"{variable} contains non-integer categorical codes: {examples}")
    return rounded.astype("Int64")


def categorical_labels_from_codes(series: pd.Series, label_map: dict[int, str], variable: str) -> pd.Categorical:
    codes = integer_codes(series, variable)
    unmapped = codes.notna() & ~codes.isin(set(label_map))
    if unmapped.any():
        examples = sorted(codes.loc[unmapped].dropna().astype(int).unique().tolist())[:20]
        raise RuntimeError(f"{variable} contains nonmissing codes without PSID-SHELF labels: {examples}")
    categories = list(dict.fromkeys(label_map[key] for key in sorted(label_map)))
    return pd.Categorical(codes.map(label_map), categories=categories)


def scalar_integer_code(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if np.isnan(number):
        return None
    rounded = int(round(number))
    if abs(number - rounded) > 1e-9:
        return None
    return rounded


def occ_major_from_2010(value: object) -> str | None:
    code = scalar_integer_code(value)
    if code is None:
        return None
    if 10 <= code <= 3540:
        return "professional_managerial"
    if 3600 <= code <= 5940:
        return "sales_office_service"
    if 6000 <= code <= 7630:
        return "farming_construction_repair"
    if 7700 <= code <= 9750:
        return "production_transport_labor"
    if 9800 <= code <= 9830:
        return "military_other"
    return None


def occ_major_from_2000(value: object) -> str | None:
    code = scalar_integer_code(value)
    if code is None:
        return None
    if 10 <= code <= 354:
        return "professional_managerial"
    if 360 <= code <= 593:
        return "sales_office_service"
    if 600 <= code <= 762:
        return "farming_construction_repair"
    if 770 <= code <= 975:
        return "production_transport_labor"
    if 980 <= code <= 983:
        return "military_other"
    return None


def occ_major_from_1970(value: object) -> str | None:
    code = scalar_integer_code(value)
    if code is None:
        return None
    if 1 <= code <= 245:
        return "professional_managerial"
    if 250 <= code <= 395:
        return "sales_office_service"
    if 401 <= code <= 444:
        return "farming_construction_repair"
    if 500 <= code <= 785:
        return "production_transport_labor"
    if 801 <= code <= 846:
        return "farming_construction_repair"
    if 901 <= code <= 965:
        return "sales_office_service"
    if 980 <= code <= 983:
        return "military_other"
    return None


def harmonize_occupation(record: pd.Series) -> tuple[str | None, str | None]:
    value_2010 = occ_major_from_2010(record.get("occ_2010c_1m"))
    if value_2010 is not None:
        return value_2010, "occ_2010c_1m"
    value_2000 = occ_major_from_2000(record.get("occ_2000c_1m"))
    if value_2000 is not None:
        return value_2000, "occ_2000c_1m"
    value_1970 = occ_major_from_1970(record.get("occ_1970c"))
    if value_1970 is not None:
        return value_1970, "occ_1970c"
    return None, None


def add_labelled_categorical_columns(chunk: pd.DataFrame) -> pd.DataFrame:
    out = chunk.copy()
    missing = [column for column in PSID_CATEGORICAL_LABEL_MAPS if column not in out.columns]
    if missing:
        raise RuntimeError(f"PSID-SHELF chunk is missing categorical variables required for labels: {missing}")

    for variable, label_map in PSID_CATEGORICAL_LABEL_MAPS.items():
        out[f"{variable}_label"] = categorical_labels_from_codes(out[variable], label_map, variable)

    occupation = out.apply(harmonize_occupation, axis=1, result_type="expand")
    out["occ_major_harmonized"] = pd.Categorical(occupation[0], categories=list(OCC_MAJOR_LABELS))
    out["occ_source"] = occupation[1].astype("string")
    out["occ_major_harmonized_label"] = pd.Categorical(
        occupation[0].map(OCC_MAJOR_LABELS),
        categories=list(OCC_MAJOR_LABELS.values()),
    )
    return out


def normalize_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk.columns = [c.lower() for c in chunk.columns]

    sampstat = pd.to_numeric(chunk["sampstat"], errors="coerce")
    rel = pd.to_numeric(chunk["rel"], errors="coerce")
    year = pd.to_numeric(chunk["year"], errors="coerce")

    chunk["subsample"] = derive_subsample(chunk["lineage"])
    chunk["sample_person"] = sampstat.ne(0).fillna(False)
    chunk["is_reference_person"] = rel.eq(1).fillna(False)
    chunk["income_year"] = (year - 1).round().astype("Int64")
    chunk = add_labelled_categorical_columns(chunk)
    return chunk


def allowed_subsamples(args: argparse.Namespace) -> set[str]:
    if args.src_only:
        allowed = {"src"}
    else:
        allowed = set(SUBSAMPLE_ORDER)
    allowed -= set(args.exclude_subsample)
    if not allowed:
        raise SystemExit("All subsamples were excluded; no rows would be written.")
    return allowed


def expected_years_in_scope(args: argparse.Namespace) -> list[int]:
    years = EXPECTED_YEARS
    if args.year_min is not None:
        years = [y for y in years if y >= args.year_min]
    if args.year_max is not None:
        years = [y for y in years if y <= args.year_max]
    return years


def apply_filters(chunk: pd.DataFrame, args: argparse.Namespace, allowed: set[str]) -> pd.DataFrame:
    mask = pd.Series(True, index=chunk.index)
    if args.year_min is not None:
        mask &= pd.to_numeric(chunk["year"], errors="coerce").ge(args.year_min)
    if args.year_max is not None:
        mask &= pd.to_numeric(chunk["year"], errors="coerce").le(args.year_max)
    mask &= chunk["subsample"].isin(sorted(allowed))
    if args.sample_persons_only:
        mask &= chunk["sample_person"]
    if args.reference_person_only:
        mask &= chunk["is_reference_person"]
    return chunk.loc[mask].copy()


def load_psid_release_date_crosswalk() -> pd.DataFrame:
    required_columns = ["year", "release_date", "source", "source_url", "comments"]
    if not RELEASE_DATE_CROSSWALK_PATH.exists():
        raise SystemExit(f"Missing PSID release-date crosswalk: {RELEASE_DATE_CROSSWALK_PATH}")

    crosswalk = pd.read_csv(
        RELEASE_DATE_CROSSWALK_PATH,
        dtype={"source": "string", "source_url": "string", "comments": "string"},
    )
    missing = [column for column in required_columns if column not in crosswalk.columns]
    if missing:
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} is missing columns: {missing}")

    crosswalk = crosswalk.loc[:, required_columns].copy()
    crosswalk["year"] = pd.to_numeric(crosswalk["year"], errors="coerce").astype("Int64")
    crosswalk["release_date"] = pd.to_datetime(crosswalk["release_date"], errors="coerce")
    if crosswalk["year"].isna().any():
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains missing or nonnumeric year values.")
    if crosswalk["year"].duplicated().any():
        duplicate_years = sorted(crosswalk.loc[crosswalk["year"].duplicated(), "year"].astype(int).unique())
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains duplicate years: {duplicate_years}")
    observed_years = set(crosswalk["year"].astype(int))
    missing_years = sorted(set(EXPECTED_YEARS) - observed_years)
    unexpected_years = sorted(observed_years - set(EXPECTED_YEARS))
    if missing_years or unexpected_years:
        raise SystemExit(
            f"{RELEASE_DATE_CROSSWALK_PATH} does not match the expected PSID wave schedule. "
            f"missing={missing_years}; unexpected={unexpected_years}"
        )
    if crosswalk["release_date"].isna().any():
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains missing release_date values.")
    if crosswalk["source"].isna().any() or crosswalk["source"].str.strip().eq("").any():
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains missing source values.")
    if crosswalk["source_url"].isna().any() or crosswalk["source_url"].str.strip().eq("").any():
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains missing source_url values.")
    if crosswalk["comments"].isna().any() or crosswalk["comments"].str.strip().eq("").any():
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains missing comments values.")
    crosswalk["release_date"] = crosswalk["release_date"].dt.strftime("%Y-%m-%d").astype("string")
    crosswalk["source"] = crosswalk["source"].str.strip()
    return crosswalk


def assign_psid_release_dates(df: pd.DataFrame, release_crosswalk: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    release_lookup = release_crosswalk.set_index("year")
    year = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["release_date"] = year.map(release_lookup["release_date"]).astype("string")
    out["release_date_source"] = year.map(release_lookup["source"]).astype("string")

    if out["release_date"].isna().any():
        missing_years = sorted(year.loc[out["release_date"].isna()].dropna().astype(int).unique())[:10]
        raise SystemExit(f"PSID release-date crosswalk does not cover years: {missing_years}")
    if out["release_date_source"].isna().any() or out["release_date_source"].str.strip().eq("").any():
        raise SystemExit("PSID release-date assignment left missing release_date_source values.")
    return out


def add_release_assignment_to_counter(counter: Counter[tuple[str, str]], df: pd.DataFrame) -> None:
    required = {"release_date_source", "release_date"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"PSID release-date coverage is missing columns: {sorted(missing)}")
    counts = df.groupby(["release_date_source", "release_date"], dropna=False).size()
    for key, count in counts.items():
        counter[(str(key[0]), str(key[1]))] += int(count)


def coverage_rows_from_counter(dataset: str, counter: Counter[tuple[str, str]]) -> pd.DataFrame:
    rows = [
        {
            "dataset": dataset,
            "release_rule_id": source,
            "microdata_release_date": date,
            "rows": count,
            "unmatched_rows": 0,
        }
        for (source, date), count in sorted(counter.items())
    ]
    return pd.DataFrame(
        rows,
        columns=["dataset", "release_rule_id", "microdata_release_date", "rows", "unmatched_rows"],
    )


def update_release_date_coverage(dataset: str, coverage: pd.DataFrame) -> None:
    RELEASE_DATE_COVERAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = ["dataset", "release_rule_id", "microdata_release_date", "rows", "unmatched_rows"]
    missing = [column for column in columns if column not in coverage.columns]
    if missing:
        raise SystemExit(f"PSID release-date coverage is missing columns: {missing}")

    if RELEASE_DATE_COVERAGE_PATH.exists():
        existing = pd.read_csv(RELEASE_DATE_COVERAGE_PATH)
        missing_existing = [column for column in columns if column not in existing.columns]
        if missing_existing:
            raise SystemExit(
                f"{RELEASE_DATE_COVERAGE_PATH} is missing required columns: {missing_existing}"
            )
        existing = existing.loc[existing["dataset"].ne(dataset), columns].copy()
        output = pd.concat([existing, coverage.loc[:, columns]], ignore_index=True)
    else:
        output = coverage.loc[:, columns].copy()

    output["rows"] = pd.to_numeric(output["rows"], errors="raise").astype(int)
    output["unmatched_rows"] = pd.to_numeric(output["unmatched_rows"], errors="raise").astype(int)
    output = output.sort_values(
        ["dataset", "release_rule_id", "microdata_release_date"],
        kind="mergesort",
    )
    output.to_csv(RELEASE_DATE_COVERAGE_PATH, index=False)


def write_parquet(
    stata_path: Path,
    out_data: Path,
    args: argparse.Namespace,
    audit_log: list[dict[str, object]],
) -> tuple[int, int, pd.DataFrame]:
    if out_data.exists():
        out_data.unlink()
    out_data.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    rows_read = 0
    rows_written = 0
    release_counter: Counter[tuple[str, str]] = Counter()
    allowed = allowed_subsamples(args)
    release_crosswalk = load_psid_release_date_crosswalk()

    with pd.read_stata(
        stata_path,
        convert_categoricals=False,
        convert_missing=False,
        chunksize=args.chunksize,
    ) as reader:
        for chunk_idx, raw_chunk in enumerate(reader, start=1):
            if args.max_chunks is not None and chunk_idx > args.max_chunks:
                break

            rows_read += len(raw_chunk)
            chunk = normalize_chunk(raw_chunk)
            chunk = apply_filters(chunk, args, allowed)

            if chunk.empty:
                if chunk_idx % 10 == 0:
                    print(f"[1_preprocessing/psid.py] chunk={chunk_idx} rows_read={rows_read:,} rows_written={rows_written:,}")
                continue

            chunk = assign_psid_release_dates(chunk, release_crosswalk)
            add_release_assignment_to_counter(release_counter, chunk)
            rows_written += len(chunk)

            table = pa.Table.from_pandas(chunk, preserve_index=False)
            table_schema = table.schema.remove_metadata()

            if writer is None:
                schema = table_schema
                writer = pq.ParquetWriter(
                    out_data,
                    schema=schema,
                    compression=args.compression,
                    use_dictionary=True,
                )
            if not schema.equals(table_schema, check_metadata=False):
                table = table.cast(schema)

            writer.write_table(table)

            if chunk_idx % 10 == 0:
                print(f"[1_preprocessing/psid.py] chunk={chunk_idx} rows_read={rows_read:,} rows_written={rows_written:,}")

    if writer is None:
        raise SystemExit("No rows were written. Check the selected filters.")
    writer.close()

    audit_log.append(
        {
            "step": "write_parquet",
            "rows_read": rows_read,
            "rows_written": rows_written,
            "compression": args.compression,
            "chunksize": args.chunksize,
        }
    )
    return rows_read, rows_written, coverage_rows_from_counter("PSID", release_counter)


def summarize_parquet(
    data_path: Path,
    variable_labels: dict[str, str],
    expected_years: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pf = pq.ParquetFile(data_path)
    columns = pf.schema_arrow.names
    missing_counts = Counter({col: 0 for col in columns})
    year_counts: Counter[int] = Counter()
    subsample_counts: Counter[str] = Counter()
    unique_ids: set[int] = set()
    total_rows = 0

    for batch in pf.iter_batches(batch_size=10_000):
        chunk = batch.to_pandas()
        total_rows += len(chunk)

        nulls = chunk.isna().sum()
        for col, count in nulls.items():
            missing_counts[col] += int(count)

        if "year" in chunk.columns:
            years = pd.to_numeric(chunk["year"], errors="coerce").dropna().astype(int)
            year_counts.update(years.value_counts().to_dict())
        if "subsample" in chunk.columns:
            subsample_counts.update(chunk["subsample"].astype("string").fillna("unknown").value_counts().to_dict())
        if "id" in chunk.columns:
            ids = pd.to_numeric(chunk["id"], errors="coerce").dropna().astype(int).unique().tolist()
            unique_ids.update(ids)

    year_counts_df = (
        pd.DataFrame({"year": list(year_counts.keys()), "rows": list(year_counts.values())})
        .sort_values("year")
        .reset_index(drop=True)
    )
    subsample_counts_df = (
        pd.DataFrame({"subsample": list(subsample_counts.keys()), "rows": list(subsample_counts.values())})
        .sort_values("subsample")
        .reset_index(drop=True)
    )
    if not subsample_counts_df.empty:
        subsample_counts_df["subsample_label"] = subsample_counts_df["subsample"].map(SUBSAMPLE_LABELS)

    variable_rows = []
    for field in pf.schema_arrow:
        name = field.name
        if name in COMPLETE_MAIN_STUDY_COLUMNS:
            source = "psid_shelf_complete_main_study"
            description = COMPLETE_MAIN_STUDY_LABELS[name]
        elif name in DERIVED_LABELS:
            source = "derived"
            description = DERIVED_LABELS[name]
        else:
            source = "psid_shelf_long"
            description = variable_labels.get(name, "")
        variable_rows.append(
            {
                "variable": name,
                "dtype": str(field.type),
                "source": source,
                "description": description,
                "missing_count": int(missing_counts[name]),
                "missing_rate": float(missing_counts[name] / total_rows) if total_rows else np.nan,
            }
        )
    variable_inventory = pd.DataFrame(variable_rows).sort_values("variable").reset_index(drop=True)

    summary = pd.DataFrame(
        [
            {
                "rows": int(total_rows),
                "columns": int(len(columns)),
                "unique_ids": int(len(unique_ids)),
                "row_groups": int(pf.num_row_groups),
                "min_year": int(year_counts_df["year"].min()) if not year_counts_df.empty else np.nan,
                "max_year": int(year_counts_df["year"].max()) if not year_counts_df.empty else np.nan,
                "n_years_present": int(year_counts_df["year"].nunique()),
                "all_expected_years_present": int(set(year_counts_df["year"]) == set(expected_years)),
                "missing_expected_years": ",".join(str(y) for y in expected_years if y not in set(year_counts_df["year"])),
                "unexpected_years": ",".join(str(y) for y in sorted(set(year_counts_df["year"]) - set(expected_years))),
            }
        ]
    )

    return summary, year_counts_df, subsample_counts_df, variable_inventory


def metadata_payload(
    args: argparse.Namespace,
    member_name: str,
    source_meta: dict[str, str | int | None],
    summary: pd.DataFrame,
    variable_inventory: pd.DataFrame,
    expected_years: list[int],
    complete_main_study_metadata: dict[str, object] | None,
) -> dict[str, object]:
    filters = {
        "sample_persons_only": bool(args.sample_persons_only),
        "reference_person_only": bool(args.reference_person_only),
        "src_only": bool(args.src_only),
        "exclude_subsample": list(args.exclude_subsample),
        "year_min": args.year_min,
        "year_max": args.year_max,
    }
    return {
        "dataset_name": "PSID-SHELF cleaned long panel",
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "public_provenance": PSIDSHELF_PUBLIC_PROVENANCE,
        "source_zip": os.path.relpath(args.raw_zip, PROJECT_ROOT),
        "source_zip_sha256": sha256_file(args.raw_zip),
        "source_member": member_name,
        "source_data_label": source_meta["source_data_label"],
        "psid_retrieve": source_meta["psid_retrieve"],
        "psidshelf_compile": source_meta["psidshelf_compile"],
        "psidshelf_release": source_meta["psidshelf_release"],
        "output_path": os.path.relpath(args.out_data, PROJECT_ROOT),
        "output_format": "parquet",
        "output_parquet_sha256": sha256_file(args.out_data),
        "expected_years": expected_years,
        "active_filters": filters,
        "summary": summary.iloc[0].to_dict(),
        "variables": variable_inventory.to_dict(orient="records"),
        "complete_main_study_enrichment": complete_main_study_metadata,
    }


def save_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# Append source measurements to the base person-by-wave panel.
def archive_metadata(archive_path: Path = COMPLETE_MAIN_STUDY_ARCHIVE_PATH) -> tuple[str, dict[str, str]]:
    if not archive_path.is_file():
        raise RuntimeError(f"Missing Complete Main Study archive: {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1 or not members[0].endswith(".dta"):
            raise RuntimeError("Complete Main Study archive must contain exactly one Stata file.")
        with archive.open(members[0]) as handle:
            labels = StataReader(handle, convert_categoricals=False).variable_labels()
    missing = sorted(set(source_columns()) - set(labels))
    if missing:
        raise RuntimeError(f"Complete Main Study archive lacks contracted fields: {missing}")
    return members[0], labels


def read_complete_main_study_fields(
    archive_path: Path = COMPLETE_MAIN_STUDY_ARCHIVE_PATH,
) -> pd.DataFrame:
    """Read the selected raw fields for this run; no persistent extract is reused."""
    contracted = source_columns()
    member_name, _ = archive_metadata(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member_name) as handle:
            reader = StataReader(handle, convert_categoricals=False, preserve_dtypes=True)
            reader.variable_labels()
            dtype = reader._setup_dtype()
            indices = [reader._varlist.index(column) for column in contracted]
            string_fields = [
                column
                for column, index in zip(contracted, indices, strict=True)
                if isinstance(reader._typlist[index], int)
            ]
            if string_fields:
                raise RuntimeError(f"Contract unexpectedly contains string fields: {string_fields}")

            handle.seek(reader._data_location)
            chunks: list[pd.DataFrame] = []
            rows_read = 0
            while rows_read < reader._nobs:
                requested_rows = min(CHUNK_ROWS, reader._nobs - rows_read)
                requested_bytes = requested_rows * dtype.itemsize
                block = handle.read(requested_bytes)
                if len(block) != requested_bytes:
                    raise RuntimeError(
                        f"Short Complete Main Study read at row {rows_read}: "
                        f"expected {requested_bytes:,} bytes, received {len(block):,}."
                    )
                view = np.frombuffer(block, dtype=dtype, count=requested_rows)
                if reader._byteorder != reader._native_byteorder:
                    view = view.byteswap().view(view.dtype.newbyteorder())
                chunk = pd.DataFrame(
                    {
                        column: view[f"s{index}"].copy()
                        for column, index in zip(contracted, indices, strict=True)
                    }
                )
                for column, index in zip(contracted, indices, strict=True):
                    storage_type = reader._typlist[index]
                    if storage_type not in reader.VALID_RANGE:
                        continue
                    lower, upper = reader.VALID_RANGE[storage_type]
                    values = chunk[column].to_numpy()
                    missing = (values < lower) | (values > upper)
                    if missing.any():
                        chunk[column] = chunk[column].astype("float64")
                        chunk.loc[missing, column] = np.nan
                chunks.append(chunk)
                rows_read += requested_rows
                if rows_read % (CHUNK_ROWS * 20) == 0 or rows_read == reader._nobs:
                    print(
                        f"[complete_main_study] streamed {rows_read:,} of {reader._nobs:,} rows",
                        flush=True,
                    )

    direct = pd.concat(chunks, ignore_index=True)
    direct["ID"] = pd.to_numeric(direct["ID"], errors="raise").astype("int64")
    if list(direct.columns) != contracted or direct["ID"].duplicated().any():
        raise RuntimeError("Complete Main Study extract violates its field or person-ID contract.")
    return direct


def assignment_component(
    frame: pd.DataFrame,
    value_column: str,
    accuracy_column: str,
    topcode: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    value = pd.to_numeric(frame[value_column], errors="coerce")
    accuracy = pd.to_numeric(frame[accuracy_column], errors="coerce")
    valid = value.ge(0) & value.lt(topcode) & accuracy.isin(VALID_ACCURACY_CODES)
    status = accuracy.map(
        {0: "direct_or_valid_zero", 1: "minor_assignment", 2: "major_assignment"}
    ).where(valid, "missing_or_topcoded")
    return value.where(valid), accuracy, status


def role_lookup(panel_path: Path) -> pd.DataFrame:
    panel = pd.read_parquet(
        panel_path,
        columns=["id", "year", "fuid", "rel", "panel_current", "demo_age_gen", "demo_sex", "demo_sex_label"],
        filters=[("panel_current", "=", 1)],
    )
    panel = panel.loc[panel["fuid"].notna()].copy()
    panel["id"] = pd.to_numeric(panel["id"], errors="raise").astype("int64")
    panel["year"] = pd.to_numeric(panel["year"], errors="raise").astype(int)
    panel["fuid"] = pd.to_numeric(panel["fuid"], errors="raise")
    roles = panel.loc[panel["rel"].isin([1, 2])].copy()
    duplicate = roles.duplicated(["year", "fuid", "rel"], keep=False)
    ambiguous = roles.loc[duplicate, ["year", "fuid"]].drop_duplicates()
    if not ambiguous.empty:
        roles = roles.merge(ambiguous.assign(_ambiguous=1), on=["year", "fuid"], how="left")
        roles = roles.loc[roles["_ambiguous"].isna()].drop(columns="_ambiguous")

    fields = ["id", "demo_age_gen", "demo_sex", "demo_sex_label"]
    reference = roles.loc[roles["rel"].eq(1), ["year", "fuid", *fields]].rename(
        columns={
            "id": "reference_person_id",
            "demo_age_gen": "demo_age_gen_rp",
            "demo_sex": "demo_sex_rp",
            "demo_sex_label": "demo_sex_label_rp",
        }
    )
    spouse = roles.loc[roles["rel"].eq(2), ["year", "fuid", *fields]].rename(
        columns={
            "id": "spouse_person_id",
            "demo_age_gen": "demo_age_gen_sp",
            "demo_sex": "demo_sex_sp",
            "demo_sex_label": "demo_sex_label_sp",
        }
    )
    family_roles = reference.merge(spouse, on=["year", "fuid"], how="left", validate="one_to_one")
    lookup = panel[["id", "year", "fuid"]].merge(
        family_roles,
        on=["year", "fuid"],
        how="left",
        validate="many_to_one",
    )
    if lookup.duplicated(["id", "year"]).any():
        raise RuntimeError("Current family membership is not unique by person and wave.")
    return lookup.drop(columns="fuid")


def wave_measurements(members: pd.DataFrame, direct: pd.DataFrame, year: int) -> pd.DataFrame:
    needed = {"ID"}
    if year in HOURS_VARIABLES:
        needed.update(str(HOURS_VARIABLES[year][key]) for key in ["rp_hours", "rp_accuracy", "sp_hours", "sp_accuracy"])
    for variables in ROLE_REASON_VARIABLES.values():
        if year in variables:
            needed.update(variables[year][:2])
    if year in FOOD_VARIABLES:
        needed.update(str(FOOD_VARIABLES[year][key]) for key in ["home", "home_accuracy", "away", "away_accuracy", "stamps", "stamps_accuracy"])

    source = direct.loc[:, [column for column in direct.columns if column in needed]]
    out = members.merge(source, left_on="id", right_on="ID", how="left", validate="many_to_one").drop(columns="ID")

    if year in HOURS_VARIABLES:
        spec = HOURS_VARIABLES[year]
        out["labor_reference_year"] = year - 1
        for role in ["rp", "sp"]:
            hours, accuracy, status = assignment_component(
                out,
                str(spec[f"{role}_hours"]),
                str(spec[f"{role}_accuracy"]),
                HOURS_TOPCODE,
            )
            out[f"annual_hours_{role}"] = hours
            out[f"annual_hours_valid_{role}"] = hours.notna().astype("int8")
            out[f"annual_hours_accuracy_code_{role}"] = accuracy
            out[f"annual_hours_assignment_status_{role}"] = status
            out[f"annual_work_status_{role}"] = np.where(hours.notna(), hours.gt(0).astype(float), np.nan)

    for role, variables in ROLE_REASON_VARIABLES.items():
        ascertained = year in variables
        out[f"job_end_reason_ascertained_{role}"] = np.int8(ascertained)
        if ascertained:
            employed, unemployed, _, _ = variables[year]
            out[f"job_end_reason_employed_code_{role}"] = pd.to_numeric(out[employed], errors="coerce")
            out[f"job_end_reason_unemployed_code_{role}"] = pd.to_numeric(out[unemployed], errors="coerce")

    if year in FOOD_VARIABLES:
        spec = FOOD_VARIABLES[year]
        home, _, home_status = assignment_component(
            out, str(spec["home"]), str(spec["home_accuracy"]), int(spec["home_topcode"])
        )
        away, _, away_status = assignment_component(
            out, str(spec["away"]), str(spec["away_accuracy"]), int(spec["away_topcode"])
        )
        stamps, _, stamps_status = assignment_component(
            out, str(spec["stamps"]), str(spec["stamps_accuracy"]), int(spec["stamps_topcode"])
        )
        factor = float(spec["stamps_annualisation_factor"])
        stamps = stamps * factor
        out["food_at_home_nominal_annual"] = home
        out["food_away_nominal_annual"] = away
        out["food_stamps_net_nominal_annual"] = stamps
        out["food_at_home_assignment_status"] = home_status
        out["food_away_assignment_status"] = away_status
        out["food_stamps_assignment_status"] = stamps_status
        out["food_stamps_reference_period"] = str(spec["stamps_reference_period"])
        out["food_stamps_annualisation_factor"] = factor
        out["food_cash_nominal_annual"] = home + away
        out["food_cash_valid"] = out["food_cash_nominal_annual"].notna().astype("int8")
        out["food_cash_positive"] = out["food_cash_nominal_annual"].gt(0).astype("int8")
        out["food_total_including_assistance_nominal_annual"] = home + away + stamps
    elif year in FOOD_UNAVAILABLE_WAVES:
        for column in [
            "food_at_home_assignment_status",
            "food_away_assignment_status",
            "food_stamps_assignment_status",
            "food_stamps_reference_period",
        ]:
            out[column] = "not_collected"
        out["food_cash_valid"] = np.int8(0)
        out["food_cash_positive"] = np.int8(0)

    raw_columns = sorted(needed - {"ID"})
    out = out.drop(columns=[column for column in raw_columns if column in out.columns])
    for column in COMPLETE_MAIN_STUDY_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
    return out[["id", "year", *COMPLETE_MAIN_STUDY_COLUMNS]]


def build_enrichment_lookup(
    panel_path: Path,
    direct: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    roles = role_lookup(panel_path)
    direct["ID"] = pd.to_numeric(direct["ID"], errors="raise").astype("int64")
    if direct["ID"].duplicated().any():
        raise RuntimeError("Complete Main Study extract is not unique by permanent ID.")

    waves = []
    for year, members in roles.groupby("year", sort=True):
        waves.append(wave_measurements(members.copy(), direct, int(year)))
    lookup = pd.concat(waves, ignore_index=True)
    for column, dtype in ENRICHED_INTEGER_DTYPES.items():
        lookup[column] = pd.to_numeric(lookup[column], errors="coerce").astype(dtype)
    if lookup.duplicated(["id", "year"]).any():
        raise RuntimeError("Enrichment lookup is not unique by person and wave.")

    conflicts = []
    panel_keys = pd.read_parquet(
        panel_path,
        columns=["id", "year", "fuid", "panel_current"],
        filters=[("panel_current", "=", 1)],
    )
    repeated = lookup.merge(panel_keys[["id", "year", "fuid"]], on=["id", "year"], how="left", validate="one_to_one")
    family_value_columns = [column for column in COMPLETE_MAIN_STUDY_COLUMNS if column not in {
        "reference_person_id", "spouse_person_id", "demo_age_gen_rp", "demo_age_gen_sp",
        "demo_sex_rp", "demo_sex_sp", "demo_sex_label_rp", "demo_sex_label_sp",
    }]
    for column in family_value_columns:
        inconsistent = repeated.groupby(["year", "fuid"], sort=False)[column].nunique(dropna=True).gt(1)
        if inconsistent.any():
            conflicts.append({"variable": column, "conflicting_family_waves": int(inconsistent.sum())})
    conflict_frame = pd.DataFrame(conflicts, columns=["variable", "conflicting_family_waves"])
    if not conflict_frame.empty:
        raise RuntimeError(f"Complete Main Study values conflict across current family members:\n{conflict_frame}")
    return lookup, conflict_frame


def enrich_complete_main_study(
    panel_path: Path,
    output_path: Path,
    *,
    archive_path: Path = COMPLETE_MAIN_STUDY_ARCHIVE_PATH,
    compression: str = "zstd",
) -> dict[str, object]:
    if panel_path.resolve() == output_path.resolve():
        raise ValueError("Source and enriched output paths must differ.")
    direct = read_complete_main_study_fields(archive_path)
    lookup, conflicts = build_enrichment_lookup(panel_path, direct)
    lookup = lookup.set_index(["id", "year"], verify_integrity=True)

    source_file = pq.ParquetFile(panel_path)
    legacy_columns = source_file.schema_arrow.names
    overlap = sorted(set(legacy_columns) & set(COMPLETE_MAIN_STUDY_COLUMNS))
    if overlap:
        raise RuntimeError(f"Enrichment columns already exist in source panel: {overlap}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    writer: pq.ParquetWriter | None = None
    output_schema: pa.Schema | None = None
    rows_written = 0
    for row_group in range(source_file.num_row_groups):
        table = source_file.read_row_group(row_group)
        keys = table.select(["id", "year"]).to_pandas()
        index = pd.MultiIndex.from_frame(keys)
        appended = lookup.reindex(index).reset_index(drop=True)
        for column in COMPLETE_MAIN_STUDY_COLUMNS:
            table = table.append_column(column, pa.array(appended[column], from_pandas=True))
        table = table.replace_schema_metadata(None)
        if writer is None:
            output_schema = table.schema
            writer = pq.ParquetWriter(
                output_path,
                schema=output_schema,
                compression=compression,
                use_dictionary=True,
            )
        elif not output_schema.equals(table.schema, check_metadata=False):
            table = table.cast(output_schema)
        writer.write_table(table)
        rows_written += table.num_rows
        print(
            f"[complete_main_study] enriched row group {row_group + 1:,}/{source_file.num_row_groups:,}; "
            f"rows={rows_written:,}",
            flush=True,
        )
    if writer is None:
        raise RuntimeError("Source panel has no Parquet row groups.")
    writer.close()

    member_name, labels = archive_metadata(archive_path)
    metadata = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_panel": os.path.relpath(panel_path, PROJECT_ROOT),
        "source_panel_sha256": sha256_file(panel_path),
        "complete_main_study_archive": os.path.relpath(archive_path, PROJECT_ROOT),
        "complete_main_study_archive_sha256": sha256_file(archive_path),
        "complete_main_study_member": member_name,
        "output_panel": os.path.relpath(output_path, PROJECT_ROOT),
        "output_panel_sha256": sha256_file(output_path),
        "legacy_columns": len(legacy_columns),
        "appended_columns": COMPLETE_MAIN_STUDY_COLUMNS,
        "rows": rows_written,
        "current_person_wave_enrichment_rows": len(lookup),
        "source_variable_labels": {column: labels[column] for column in source_columns()},
        "family_member_conflicts": conflicts.to_dict(orient="records"),
        "task_flags_stored": False,
    }
    return metadata


def contract_crosswalk(labels: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for role, variables_by_year in ROLE_REASON_VARIABLES.items():
        for year in SOURCE_WAVES:
            if year not in variables_by_year:
                rows.append(
                    {
                        "domain": "job_end_reason",
                        "survey_year": year,
                        "reference_year": year - 1,
                        "role": role,
                        "route_or_component": "not_ascertained",
                        "source_variable": "",
                        "accuracy_variable": "",
                        "source_label": "",
                        "canonical_variable": f"job_end_reason_ascertained_{role}",
                        "validity_rule": "structural non-ascertainment",
                        "annualisation_factor": "",
                        "codebook_url": OFFICIAL_CODEBOOK.format(year=year),
                    }
                )
                continue
            employed, unemployed, _, _ = variables_by_year[year]
            for route, variable in [("employed", employed), ("unemployed", unemployed)]:
                rows.append(
                    {
                        "domain": "job_end_reason",
                        "survey_year": year,
                        "reference_year": year - 1,
                        "role": role,
                        "route_or_component": route,
                        "source_variable": variable,
                        "accuracy_variable": "",
                        "source_label": labels[variable],
                        "canonical_variable": f"job_end_reason_{route}_code_{role}",
                        "validity_rule": "retain documented integer codes 0 through 9; convert Stata missing sentinels to null",
                        "annualisation_factor": "",
                        "codebook_url": OFFICIAL_CODEBOOK.format(year=year),
                    }
                )
    for year, spec in HOURS_VARIABLES.items():
        for role in ["rp", "sp"]:
            variable = str(spec[f"{role}_hours"])
            accuracy = str(spec[f"{role}_accuracy"])
            rows.append(
                {
                    "domain": "annual_hours",
                    "survey_year": year,
                    "reference_year": year - 1,
                    "role": role,
                    "route_or_component": "annual_total",
                    "source_variable": variable,
                    "accuracy_variable": accuracy,
                    "source_label": labels[variable],
                    "canonical_variable": f"annual_hours_{role}",
                    "validity_rule": "0 <= hours < 9999 and accuracy code in {0,1,2}",
                    "annualisation_factor": 1,
                    "codebook_url": OFFICIAL_CODEBOOK.format(year=year),
                }
            )
    for year in SOURCE_WAVES:
        if year not in FOOD_VARIABLES:
            rows.append(
                {
                    "domain": "food",
                    "survey_year": year,
                    "reference_year": year - 1,
                    "role": "family",
                    "route_or_component": "not_collected",
                    "source_variable": "",
                    "accuracy_variable": "",
                    "source_label": "",
                    "canonical_variable": "food_cash_nominal_annual",
                    "validity_rule": "no documented comparable food measure",
                    "annualisation_factor": "",
                    "codebook_url": OFFICIAL_CODEBOOK.format(year=year),
                }
            )
            continue
        spec = FOOD_VARIABLES[year]
        for component, value_key, accuracy_key, topcode_key, canonical in [
            ("food_at_home", "home", "home_accuracy", "home_topcode", "food_at_home_nominal_annual"),
            ("food_away", "away", "away_accuracy", "away_topcode", "food_away_nominal_annual"),
            ("food_stamps_net", "stamps", "stamps_accuracy", "stamps_topcode", "food_stamps_net_nominal_annual"),
        ]:
            variable = str(spec[value_key])
            factor = float(spec["stamps_annualisation_factor"]) if component == "food_stamps_net" else 1.0
            rows.append(
                {
                    "domain": "food",
                    "survey_year": year,
                    "reference_year": year - 1,
                    "role": "family",
                    "route_or_component": component,
                    "source_variable": variable,
                    "accuracy_variable": str(spec[accuracy_key]),
                    "source_label": labels[variable],
                    "canonical_variable": canonical,
                    "validity_rule": f"0 <= value < {int(spec[topcode_key])} and accuracy code in {{0,1,2}}",
                    "annualisation_factor": factor,
                    "codebook_url": OFFICIAL_CODEBOOK.format(year=year),
                }
            )
    return pd.DataFrame(rows)


def write_complete_main_study_contract_outputs(
    metadata: dict[str, object],
    *,
    output_dir: Path = COMPLETE_MAIN_STUDY_OUTPUT_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = metadata["source_variable_labels"]
    if not isinstance(labels, dict):
        raise RuntimeError("Complete Main Study metadata lacks source variable labels.")
    contract_crosswalk(labels).to_csv(output_dir / "00_complete_main_study_crosswalk.csv", index=False)
    pd.DataFrame(
        [
            {
                "source_code": code,
                "source_label": label,
                "qualifying_for_current_tasks": int(code in QUALIFYING_REASON_CODES),
                "harmonised_qualifying_reason": HARMONISED_REASONS.get(code, ""),
            }
            for code, label in REASON_LABELS.items()
        ]
    ).to_csv(output_dir / "00_job_end_reason_value_labels.csv", index=False)
    pd.DataFrame(
        [
            {
                "variable": variable,
                "description": COMPLETE_MAIN_STUDY_LABELS[variable],
                "unit": "person by survey wave",
                "source": (
                    "psid_shelf_long_family_membership"
                    if variable.startswith(("reference_person_id", "spouse_person_id", "demo_"))
                    else "psid_shelf_complete_main_study"
                ),
                "task_construct": 0,
            }
            for variable in COMPLETE_MAIN_STUDY_COLUMNS
        ]
    ).to_csv(output_dir / "00_enriched_variable_dictionary.csv", index=False)
    (output_dir / "00_complete_main_study_source_inventory.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    # 1. Resolve the source archives and record the construction choices.
    args = parse_args()
    validate_source_inputs("psid", overrides={RAW_ZIP_PATH: args.raw_zip,
        COMPLETE_MAIN_STUDY_ARCHIVE_PATH: args.complete_main_study_archive})
    assert_inputs(
        args.raw_zip,
        None if args.skip_complete_main_study_enrichment else args.complete_main_study_archive,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    member_name = resolve_member_name(args.raw_zip)
    scratch = tempfile.TemporaryDirectory(prefix="psidshelf_")
    stata_path = extract_member(args.raw_zip, member_name, Path(scratch.name))
    source_meta, variable_labels = get_source_metadata(stata_path)

    audit_log: list[dict[str, object]] = [
        {
            "step": "source_resolution",
            "raw_zip": os.path.relpath(args.raw_zip, PROJECT_ROOT),
            "source_member": member_name,
            "source_data_label": source_meta["source_data_label"],
            "psid_retrieve": source_meta["psid_retrieve"],
            "psidshelf_compile": source_meta["psidshelf_compile"],
            "psidshelf_release": source_meta["psidshelf_release"],
        },
        {
            "step": "transformations",
            "operations": [
                "lowercase all column names",
                "derive subsample from lineage",
                "derive sample_person from sampstat != 0",
                "derive is_reference_person from rel == 1",
                "derive income_year from year - 1",
                "assign release_date and release_date_source from data/raw/release_dates/release_psid.csv",
            ],
        },
        {
            "step": "filters",
            "sample_persons_only": bool(args.sample_persons_only),
            "reference_person_only": bool(args.reference_person_only),
            "src_only": bool(args.src_only),
            "exclude_subsample": list(args.exclude_subsample),
            "year_min": args.year_min,
            "year_max": args.year_max,
        },
    ]

    expected_years = expected_years_in_scope(args)
    complete_main_study_metadata: dict[str, object] | None = None
    if args.skip_complete_main_study_enrichment:
        build_path = args.out_data
    else:
        build_path = args.out_data.with_name(f".{args.out_data.name}.psidshelf_base.tmp")
        enriched_path = args.out_data.with_name(f".{args.out_data.name}.enriched.tmp")
        for temporary_path in [build_path, enriched_path]:
            if temporary_path.exists():
                temporary_path.unlink()

    # 2. Build the base panel, append Complete Main Study measurements, and publish.
    rows_read, rows_written, release_coverage = write_parquet(stata_path, build_path, args, audit_log)
    scratch.cleanup()
    if not args.skip_complete_main_study_enrichment:
        complete_main_study_metadata = enrich_complete_main_study(
            build_path,
            enriched_path,
            archive_path=args.complete_main_study_archive,
            compression=args.compression,
        )
        enriched_path.replace(args.out_data)
        build_path.unlink()
        complete_main_study_metadata["output_panel"] = os.path.relpath(args.out_data, PROJECT_ROOT)
        complete_main_study_metadata["output_panel_sha256"] = sha256_file(args.out_data)
        audit_log.append(
            {
                "step": "complete_main_study_enrichment",
                "source_archive": os.path.relpath(args.complete_main_study_archive, PROJECT_ROOT),
                "source_archive_sha256": complete_main_study_metadata["complete_main_study_archive_sha256"],
                "appended_columns": COMPLETE_MAIN_STUDY_COLUMNS,
                "task_flags_stored": False,
            }
        )
        write_complete_main_study_contract_outputs(complete_main_study_metadata)
    with exclusive_lock(RELEASE_DATE_COVERAGE_PATH):
        update_release_date_coverage("PSID", release_coverage)
    summary, year_counts, subsample_counts, variable_inventory = summarize_parquet(
        args.out_data,
        variable_labels,
        expected_years,
    )

    # 3. Record source provenance, coverage and the final variable inventory.
    metadata = metadata_payload(
        args,
        member_name,
        source_meta,
        summary,
        variable_inventory,
        expected_years,
        complete_main_study_metadata,
    )
    with args.metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    with (args.out_dir / "processing_audit.json").open("w", encoding="utf-8") as fh:
        json.dump(audit_log, fh, indent=2)

    save_csv(summary.assign(rows_read=rows_read, rows_written=rows_written), args.out_dir / "build_summary.csv")
    save_csv(year_counts, args.out_dir / "year_counts.csv")
    save_csv(subsample_counts, args.out_dir / "subsample_counts.csv")
    save_csv(variable_inventory, args.out_dir / "variable_inventory.csv")

    print(
        f"[1_preprocessing/psid.py] wrote {rows_written:,} rows to {args.out_data} "
        f"({summary.iloc[0]['columns']} columns; {summary.iloc[0]['unique_ids']:,} ids)."
    )


if __name__ == "__main__":
    main()
