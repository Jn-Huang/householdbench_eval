#!/usr/bin/env python
"""Build harmonized CEX Interview CU x interview-quarter panel with aggregated consumption (CQ + PQ)."""

import re
import zipfile
from collections import defaultdict
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.io import exclusive_lock
from scripts.utils.source_inputs import validate_source_inputs

RAW_DIR = Path("data/raw/micro/cex/interview")
RELEASE_YEARS = (1980, 1981, *range(1984, 2025))
REQUIRED_ARCHIVES = tuple(f"intrvw{year % 100:02d}.zip" for year in RELEASE_YEARS)
INTERMEDIATE_DIR = Path("data/intermediate")
OUT_PATH = INTERMEDIATE_DIR / "cex_interview.parquet"
TEMP_CSV_PATH = INTERMEDIATE_DIR / "cex_interview_build.tmp.csv"
# Frozen from the CEX PUMD Dictionary, Codes sheet, INTERVIEW/FMLI STATE.
# Pre-1982 state codes are mapped to the documented modern coding.
STATE_CROSSWALK = {
    "legacy_code_map": {
        "14": "25",
        "16": "16",
        "21": "21",
        "22": "22",
        "23": "23",
        "31": "31",
        "32": "32",
        "33": "33",
        "34": "34",
        "35": "55",
        "41": "41",
        "42": "42",
        "43": "29",
        "47": "47",
        "52": "24",
        "53": "53",
        "54": "51",
        "55": "55",
        "56": "37",
        "58": "13",
        "59": "12",
        "61": "21",
        "62": "47",
        "63": "01",
        "72": "22",
        "74": "48",
        "84": "08",
        "86": "04",
        "91": "53",
        "92": "41",
        "93": "06",
        "95": "15",
    },
    "canonical_state_codes": {
        "01", "02", "04", "05", "06", "08", "09", "10", "11", "12",
        "13", "15", "16", "17", "18", "19", "20", "21", "22", "23",
        "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
        "34", "35", "36", "37", "39", "40", "41", "42", "44", "45",
        "46", "47", "48", "49", "51", "53", "54", "55",
    },
}
RELEASE_DATE_CROSSWALK_PATH = Path("data/raw/release_dates/release_cex.csv")
RELEASE_DATE_COVERAGE_PATH = Path("output/preprocessing/release_dates/metadata_coverage.csv")
BUILD_DIR = Path("output/preprocessing/cex/build")
MEMBER_INVENTORY_PATH = BUILD_DIR / "fmli_member_inventory.csv"
RECORD_OVERLAP_AUDIT_PATH = BUILD_DIR / "fmli_record_overlap_audit.csv"
QOQ_OUTLIER_QUANTILE = 0.01
CEX_RELEASE_DATE_SOURCE_VALUES = {
    "Official release date",
    "Inferred from official rule",
    "Latest survey date in release",
}
SCREEN_COMPONENT_FLAG_COLS = [
    "flag_hh_income_lt_1000_ever",
    "flag_hh_cons_total_lt_250_ever",
    "flag_hh_food_annualized_gt_mean_income",
    "flag_hh_qoq_income_change_bottom_1pct_ever",
    "flag_hh_qoq_income_change_top_1pct_ever",
    "flag_hh_qoq_cons_total_change_bottom_1pct_ever",
    "flag_hh_qoq_cons_total_change_top_1pct_ever",
]
SCREEN_ALL_FLAG_COLS = SCREEN_COMPONENT_FLAG_COLS + ["flag_hh_any_screen"]
CEX_SEX_LABELS = {
    1: "Man",
    2: "Woman",
}
CEX_RACE_LABELS = {
    1: "White",
    2: "Black",
    3: "Native American",
    4: "Asian",
    5: "Pacific Islander",
    6: "Multiracial",
}
CEX_EDUCATION_LABELS = {
    1: "No formal schooling",
    2: "Elementary school",
    3: "Some high school",
    4: "High school diploma",
    5: "Some college",
    6: "Associate degree",
    7: "Bachelor's degree",
    8: "Graduate or professional degree",
}
CEX_MARITAL_LABELS = {
    1: "Married",
    2: "Widowed",
    3: "Divorced",
    4: "Separated",
    5: "Never married",
}
CEX_TENURE_LABELS = {
    1: "Owned with mortgage",
    2: "Owned without mortgage",
    3: "Owned mortgage not reported",
    4: "Rented",
    5: "Occupied without payment of cash rent",
    6: "Student housing",
}
CEX_REGION_LABELS = {
    1: "Northeast",
    2: "Midwest",
    3: "South",
    4: "West",
}
CEX_URBAN_LABELS = {
    1: "Urban",
    2: "Rural",
}

# Canonical source map from CE dictionary and novice guide.
# 1984-1989 Z* summary expenditures are kept in source units (no additional rescaling).
VAR_SPECS = {
    "newid": [("NEWID", 1.0)],
    "interview_year": [("QINTRVYR", 1.0)],
    "interview_month": [("QINTRVMO", 1.0)],
    "state": [("STATE", 1.0), ("STATE_", 1.0)],
    # Geography (easy, consistent fields available across most of the sample)
    "region": [("REGION", 1.0)],
    "bls_urbn": [("BLS_URBN", 1.0)],
    "weight": [("FINLWT21", 1.0), ("FINLWT", 1.0)],
    "fam_size": [("FAM_SIZE", 1.0)],
    "fam_type": [("FAM_TYPE", 1.0)],
    "age_ref": [("AGE_REF", 1.0)],
    "age2": [("AGE2", 1.0)],
    "sex_ref": [("SEX_REF", 1.0)],
    "sex2": [("SEX2", 1.0)],
    "marital1": [("MARITAL1", 1.0)],
    "tenure": [("CUTENURE", 1.0)],
    "race_ref": [("REF_RACE", 1.0)],
    "race2": [("RACE2", 1.0)],
    "educ_ref": [("EDUC_REF", 1.0), ("EDUC0REF", 1.0)],
    "educ2": [("EDUCA2", 1.0)],
    "n_kids": [("PERSLT18", 1.0)],
    # Dictionary transition: 2004+ income means are FINCBTXM / FSALARYM.
    "income_before_tax": [("FINCBTXM", 1.0), ("FINCBTAX", 1.0)],
    "salary": [("FSALARYM", 1.0), ("FSALARYX", 1.0)],
    # Expenditures, current quarter
    "exp_total_cq": [("TOTEXPCQ", 1.0), ("ZTOTAL", 1.0)],
    "exp_food_cq": [("FOODCQ", 1.0), ("ZFOODTOT", 1.0)],
    "exp_grocery_cq": [("GROCERCQ", 1.0)],
    "exp_foodaway_total_cq": [("FDAWAYCQ", 1.0), ("TFOODTOC", 1.0)],
    "exp_housing_cq": [("HOUSCQ", 1.0), ("ZHOUSING", 1.0)],
    "exp_transport_cq": [("TRANSCQ", 1.0), ("ZTRANPRT", 1.0)],
    "exp_health_cq": [("HEALTHCQ", 1.0), ("ZHEALTH", 1.0)],
    "exp_education_cq": [("EDUCACQ", 1.0), ("ZEDUCATN", 1.0)],
    "exp_apparel_cq": [("APPARCQ", 1.0), ("ZAPPAREL", 1.0)],
    "exp_entertainment_cq": [("ENTERTCQ", 1.0), ("ZENTRMNT", 1.0)],
    "exp_personal_care_cq": [("PERSCACQ", 1.0), ("ZPERCARE", 1.0)],
    "exp_cash_contrib_cq": [("CASHCOCQ", 1.0), ("ZCASHCTB", 1.0)],
    "exp_life_insurance_cq": [("LIFINSCQ", 1.0), ("ZLIFOTHR", 1.0)],
    "exp_retirement_pension_social_security_cq": [("RETPENCQ", 1.0), ("ZRETIRES", 1.0)],
    "exp_utilities_cq": [("UTILCQ", 1.0), ("ZUTILSPS", 1.0)],
    "exp_alcohol_cq": [("ALCBEVCQ", 1.0), ("ZALCBEVS", 1.0)],
    "exp_tobacco_cq": [("TOBACCCQ", 1.0), ("ZTOBACCO", 1.0)],
    "exp_housop_cq": [("HOUSOPCQ", 1.0), ("ZHOUSEOP", 1.0)],
    "exp_pubtra_cq": [("PUBTRACQ", 1.0), ("ZPUBTRAN", 1.0)],
    "exp_gasmo_cq": [("GASMOCQ", 1.0), ("ZGASMOTO", 1.0)],
    "exp_misc_cq": [("MISCCQ", 1.0), ("ZMISCELS", 1.0)],
    "exp_read_cq": [("READCQ", 1.0), ("ZREADING", 1.0)],
    # Expenditures, previous quarter
    "exp_total_pq": [("TOTEXPPQ", 1.0)],
    "exp_food_pq": [("FOODPQ", 1.0)],
    "exp_grocery_pq": [("GROCERPQ", 1.0)],
    "exp_foodaway_total_pq": [("FDAWAYPQ", 1.0), ("TFOODTOP", 1.0)],
    "exp_housing_pq": [("HOUSPQ", 1.0)],
    "exp_transport_pq": [("TRANSPQ", 1.0)],
    "exp_health_pq": [("HEALTHPQ", 1.0)],
    "exp_education_pq": [("EDUCAPQ", 1.0)],
    "exp_apparel_pq": [("APPARPQ", 1.0)],
    "exp_entertainment_pq": [("ENTERTPQ", 1.0)],
    "exp_personal_care_pq": [("PERSCAPQ", 1.0)],
    "exp_cash_contrib_pq": [("CASHCOPQ", 1.0)],
    "exp_life_insurance_pq": [("LIFINSPQ", 1.0)],
    "exp_retirement_pension_social_security_pq": [("RETPENPQ", 1.0)],
    "exp_utilities_pq": [("UTILPQ", 1.0)],
    "exp_alcohol_pq": [("ALCBEVPQ", 1.0)],
    "exp_tobacco_pq": [("TOBACCPQ", 1.0)],
    "exp_housop_pq": [("HOUSOPPQ", 1.0)],
    "exp_pubtra_pq": [("PUBTRAPQ", 1.0)],
    "exp_gasmo_pq": [("GASMOPQ", 1.0)],
    "exp_misc_pq": [("MISCPQ", 1.0)],
    "exp_read_pq": [("READPQ", 1.0)],
}

EXTRA_COLS = ["INTERI"]

CONSUMPTION_COMPONENTS = {
    "cons_total": ("exp_total_cq", "exp_total_pq"),
    "cons_food": ("exp_food_cq", "exp_food_pq"),
    "cons_housing": ("exp_housing_cq", "exp_housing_pq"),
    "cons_transport": ("exp_transport_cq", "exp_transport_pq"),
    "cons_health": ("exp_health_cq", "exp_health_pq"),
    "cons_education": ("exp_education_cq", "exp_education_pq"),
    "cons_apparel": ("exp_apparel_cq", "exp_apparel_pq"),
    "cons_entertainment": ("exp_entertainment_cq", "exp_entertainment_pq"),
    "cons_personal_care": ("exp_personal_care_cq", "exp_personal_care_pq"),
    "cons_cash_contrib": ("exp_cash_contrib_cq", "exp_cash_contrib_pq"),
    "cons_life_insurance": ("exp_life_insurance_cq", "exp_life_insurance_pq"),
    "cons_retirement_pension_social_security": (
        "exp_retirement_pension_social_security_cq",
        "exp_retirement_pension_social_security_pq",
    ),
    "cons_utilities": ("exp_utilities_cq", "exp_utilities_pq"),
    "cons_alcohol": ("exp_alcohol_cq", "exp_alcohol_pq"),
    "cons_tobacco": ("exp_tobacco_cq", "exp_tobacco_pq"),
    "cons_housop": ("exp_housop_cq", "exp_housop_pq"),
    "cons_pubtra": ("exp_pubtra_cq", "exp_pubtra_pq"),
    "cons_gasmo": ("exp_gasmo_cq", "exp_gasmo_pq"),
    "cons_misc": ("exp_misc_cq", "exp_misc_pq"),
    "cons_read": ("exp_read_cq", "exp_read_pq"),
}

FINAL_COLS = [
    "newid",
    "cu_id",
    "cu_id_raw",
    "cex_panel_id",
    "cex_sample_design",
    "interview_year",
    "interview_month",
    "quarter",
    "year_quarter",
    "release_date",
    "release_date_source",
    "cex_source_package",
    "cex_source_member",
    "cex_quarter_token",
    "cex_overlap_versions",
    "cex_first_release_year",
    "cex_retained_package_year",
    "interview_number",
    "interview_position_harmonized",
    "weight",
    "fam_size",
    "fam_type",
    "state",
    "region",
    "region_label",
    "bls_urbn",
    "bls_urbn_label",
    "n_kids",
    "age_ref",
    "age2",
    "sex_ref",
    "sex_ref_label",
    "sex2",
    "sex2_label",
    "marital1",
    "marital1_label",
    "tenure",
    "tenure_harmonized",
    "race_ref",
    "race_ref_harmonized",
    "race2",
    "race2_harmonized",
    "educ_ref",
    "educ_ref_harmonized",
    "educ2",
    "educ2_harmonized",
    "income_before_tax",
    "salary",
    "cons_total",
    "cons_parker_total",
    "cons_food",
    "food_definition_regime",
    "cons_housing",
    "cons_transport",
    "cons_health",
    "cons_education",
    "cons_apparel",
    "cons_entertainment",
    "cons_personal_care",
    "cons_cash_contrib",
    "cons_life_insurance",
    "cons_retirement_pension_social_security",
    "cons_utilities",
    "cons_alcohol",
    "cons_tobacco",
    "cons_misc",
    "cons_read",
    "cons_food_agg",
    "cons_strict_nondurables",
    "cons_nondurables",
    "cons_durables",
    "flag_hh_income_lt_1000_ever",
    "flag_hh_cons_total_lt_250_ever",
    "flag_hh_food_annualized_gt_mean_income",
    "flag_hh_qoq_income_change_bottom_1pct_ever",
    "flag_hh_qoq_income_change_top_1pct_ever",
    "flag_hh_qoq_cons_total_change_bottom_1pct_ever",
    "flag_hh_qoq_cons_total_change_top_1pct_ever",
    "flag_hh_any_screen",
]

TEXT_KEYS = {"newid", "state"}
NUMERIC_KEYS = [k for k in VAR_SPECS if k not in TEXT_KEYS]


def yy_to_year(yy):
    yy = int(yy)
    return 1900 + yy if yy >= 80 else 2000 + yy


def parse_zip_year(zip_name):
    m = re.search(r"intrvw(\d{2})\.zip", zip_name)
    if not m:
        return None
    return yy_to_year(m.group(1))


def parse_member_token(member_name):
    m = re.search(r"fmli(\d{3})x?\.csv$", member_name.lower())
    return m.group(1) if m else None


def canonicalize_newid(values):
    newid = values.astype("string").str.strip()
    bad = newid.notna() & ~newid.str.fullmatch(r"[0-9]{1,8}")
    if bad.any():
        examples = newid.loc[bad].drop_duplicates().head(10).tolist()
        raise SystemExit(f"Invalid NEWID values: {examples}")
    return newid.str.zfill(8)


def normalize_year(y):
    if pd.isna(y):
        return pd.NA
    try:
        y = int(y)
    except Exception:
        return pd.NA
    if y < 100:
        return 1900 + y if y >= 80 else 2000 + y
    return y


def _normalize_state_code(value):
    if pd.isna(value):
        return pd.NA
    s = str(value).strip().strip("'\"")
    if not s:
        return pd.NA
    m = re.match(r"^([0-9]+)(?:\.0+)?$", s)
    if not m:
        return pd.NA
    try:
        code_int = int(m.group(1))
    except Exception:
        return pd.NA
    if code_int < 0:
        return pd.NA
    return f"{code_int:02d}"




def harmonize_state_codes(state_series, state_crosswalk):
    raw_state = state_series.map(_normalize_state_code)
    out = raw_state.copy()
    if state_crosswalk:
        legacy_code_map = state_crosswalk.get("legacy_code_map", {})
        if legacy_code_map:
            # Use dictionary label-based remaps for known legacy/state-code variants whenever
            # the legacy raw code is present.
            out = out.map(legacy_code_map).fillna(out)

        canonical_state_codes = state_crosswalk.get("canonical_state_codes", set())
        if canonical_state_codes:
            out = out.where(out.isin(canonical_state_codes), pd.NA)
    return out.astype("string")


def compute_quarter(month):
    if pd.isna(month):
        return pd.NA
    try:
        month = int(month)
    except Exception:
        return pd.NA
    return ((month - 1) // 3) + 1 if 1 <= month <= 12 else pd.NA


def assign_food_definition_regime(interview_year, quarter):
    y = pd.to_numeric(interview_year, errors="coerce")
    q = pd.to_numeric(quarter, errors="coerce")
    q_index = y * 4 + q
    out = pd.Series(pd.NA, index=y.index, dtype="string")
    out.loc[q_index.lt(2023 * 4 + 2).fillna(False)] = "legacy"
    out.loc[q_index.between(2023 * 4 + 2, 2024 * 4 + 1).fillna(False)] = "transition"
    out.loc[q_index.ge(2024 * 4 + 2).fillna(False)] = "redesigned"
    return out


def assign_cex_sample_design(df):
    year = pd.to_numeric(df["interview_year"], errors="coerce")
    quarter = pd.to_numeric(df["quarter"], errors="coerce")
    package_year = pd.to_numeric(df["cex_retained_package_year"], errors="coerce")
    token = df["cex_quarter_token"].astype("string")
    q_index = year * 4 + quarter

    regimes = {
        "pre_1986": q_index.le(1986 * 4 + 1),
        "frame_1986": q_index.between(1986 * 4 + 2, 1995 * 4 + 4)
        | (token.eq("961") & package_year.eq(1995)),
        "frame_1996": q_index.between(1996 * 4 + 2, 2004 * 4 + 4)
        | (token.eq("961") & package_year.eq(1996))
        | (token.eq("051") & package_year.eq(2004)),
        "frame_2005": q_index.between(2005 * 4 + 2, 2014 * 4 + 4)
        | (token.eq("051") & package_year.eq(2005))
        | (token.eq("151") & package_year.eq(2014)),
        "frame_2015": q_index.ge(2015 * 4 + 2)
        | (token.eq("151") & package_year.eq(2015)),
    }
    match_count = pd.DataFrame(regimes).fillna(False).sum(axis=1)
    if not match_count.eq(1).all():
        examples = df.loc[
            ~match_count.eq(1),
            [
                "newid",
                "interview_year",
                "interview_month",
                "cex_quarter_token",
                "cex_retained_package_year",
            ],
        ].head(10)
        raise SystemExit(
            "CEX rows must match exactly one sample design; examples:\n"
            + examples.to_string(index=False)
        )
    design = pd.Series(pd.NA, index=df.index, dtype="string")
    for label, mask in regimes.items():
        design.loc[mask.fillna(False)] = label
    return design


def choose_sources(columns):
    chosen = {}
    for canon, specs in VAR_SPECS.items():
        source = None
        scale = 1.0
        for var, factor in specs:
            if var in columns:
                source = var
                scale = factor
                break
        if source is not None:
            chosen[canon] = (source, scale)
    return chosen


def read_header(zf, member):
    with zf.open(member) as f:
        header = f.readline().decode("utf-8", errors="replace").strip()
    cols = header.split(",")
    return [c.strip().strip('"').strip("'") for c in cols]


def recode_education(series, year_series, legacy_zero_is_missing):
    s = pd.to_numeric(series, errors="coerce")
    y = pd.to_numeric(year_series, errors="coerce")
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")

    mapping = {
        1: 2,
        2: 3,
        3: 4,
        4: 5,
        5: 7,
        6: 8,
        7: 1,
        10: 2,
        11: 3,
        12: 4,
        13: 5,
        14: 6,
        15: 7,
        16: 8,
        17: 8,
    }
    for src, dst in mapping.items():
        out = out.where(s != src, dst)

    # Code 0 is "don't know" in older spouse coding but "never attended" in modern coding.
    modern_zero = (s == 0) & (~(legacy_zero_is_missing & (y <= 1981)))
    out = out.where(~modern_zero, 1)

    return out


def recode_race(series):
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    for code in [1, 2, 3, 4, 5, 6]:
        out = out.where(s != code, code)
    return out


def recode_tenure(series):
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    for code in [1, 2, 3, 4, 5, 6]:
        out = out.where(s != code, code)
    return out


def labelled_categorical(code_series, labels, source_name, output_name, *, ordered=False):
    codes = pd.to_numeric(code_series, errors="coerce").astype("Int64")
    nonmissing = codes.notna()
    unmapped = nonmissing & ~codes.isin(set(labels))
    if unmapped.any():
        bad_values = sorted(codes.loc[unmapped].dropna().astype(int).unique().tolist())
        raise ValueError(
            f"{source_name} has nonmissing codes not mapped into {output_name}: {bad_values}"
        )
    values = codes.map(labels).astype("string")
    return pd.Series(
        pd.Categorical(values, categories=list(labels.values()), ordered=ordered),
        index=code_series.index,
    )


def validate_recode_complete(raw_series, recoded_series, source_name, output_name):
    raw_codes = pd.to_numeric(raw_series, errors="coerce")
    recoded_codes = pd.to_numeric(recoded_series, errors="coerce")
    unmapped = raw_codes.notna() & recoded_codes.isna()
    if unmapped.any():
        bad_values = sorted(raw_codes.loc[unmapped].dropna().astype(int).unique().tolist())
        raise ValueError(
            f"{source_name} has nonmissing codes not mapped into {output_name}: {bad_values}"
        )


def harmonize_interview_position(interview_number, interview_year):
    n = pd.to_numeric(interview_number, errors="coerce")
    y = pd.to_numeric(interview_year, errors="coerce")
    out = pd.Series(pd.NA, index=n.index, dtype="Int64")

    pre = y <= 2014
    post = y >= 2015

    # Pre-2015 Interview numbering is typically 2-5 (with bounding interview context).
    out = out.where(~(pre & n.isin([2, 3, 4, 5])), (n - 1))
    # Post-2015 numbering should be 1-4.
    out = out.where(~(post & n.isin([1, 2, 3, 4])), n)

    return out.astype("Int64")


def validate_cex_panel_designs(df):
    record_key = ["newid", "interview_year", "interview_month"]
    if df.duplicated(record_key).any():
        raise SystemExit("CEX panel contains duplicate canonical interview-record keys.")
    panel_time_key = ["cex_panel_id", "interview_year", "interview_month"]
    if df.duplicated(panel_time_key).any():
        raise SystemExit("CEX panel contains duplicate panel-year-month rows.")
    designs_per_panel = df.groupby("cex_panel_id")["cex_sample_design"].nunique(dropna=False)
    if designs_per_panel.ne(1).any():
        raise SystemExit("A CEX panel identifier spans multiple sample designs.")


def build_adjacent_qoq_pct_changes(df, value_col):
    panel = df[["cex_panel_id", "interview_year", "quarter", value_col]].copy()
    panel = panel.dropna(subset=["cex_panel_id"])
    panel["interview_year"] = pd.to_numeric(panel["interview_year"], errors="coerce")
    panel["quarter"] = pd.to_numeric(panel["quarter"], errors="coerce")
    panel[value_col] = pd.to_numeric(panel[value_col], errors="coerce")
    panel["q_index"] = panel["interview_year"] * 4 + panel["quarter"]
    panel = panel.dropna(subset=["q_index"]).sort_values(["cex_panel_id", "q_index"], kind="mergesort")

    g = panel.groupby("cex_panel_id", sort=False)
    panel["lag_q_index"] = g["q_index"].shift(1)
    panel["lag_value"] = g[value_col].shift(1)
    contiguous = (panel["q_index"] - panel["lag_q_index"]).eq(1)
    valid = contiguous & panel[value_col].notna() & panel["lag_value"].notna() & panel["lag_value"].gt(0)

    changes = panel.loc[valid, ["cex_panel_id"]].copy()
    changes["pct_change"] = (
        100.0 * (panel.loc[valid, value_col] - panel.loc[valid, "lag_value"]) / panel.loc[valid, "lag_value"]
    )
    return changes


def build_household_quality_flags(df):
    panel = df[
        ["cex_panel_id", "interview_year", "quarter", "income_before_tax", "cons_parker_total", "cons_food"]
    ].copy()
    panel = panel.dropna(subset=["cex_panel_id"])
    for col in ["income_before_tax", "cons_parker_total", "cons_food"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

    hh_flags = pd.DataFrame(
        index=pd.Index(panel["cex_panel_id"].drop_duplicates(), name="cex_panel_id")
    )
    hh_flags["flag_hh_income_lt_1000_ever"] = panel.groupby("cex_panel_id")["income_before_tax"].min().lt(1000)
    hh_flags["flag_hh_cons_total_lt_250_ever"] = panel.groupby("cex_panel_id")["cons_parker_total"].min().lt(250)

    income_mean = panel.groupby("cex_panel_id")["income_before_tax"].mean()
    food_obs = panel.dropna(subset=["cons_food"]).groupby("cex_panel_id").agg(
        food_quarters=("cons_food", "size"),
        food_sum=("cons_food", "sum"),
    )
    annualized_food = food_obs["food_sum"] * 4.0 / food_obs["food_quarters"]
    hh_flags["flag_hh_food_annualized_gt_mean_income"] = annualized_food.gt(
        income_mean.reindex(annualized_food.index)
    ).reindex(hh_flags.index).fillna(False)

    for value_col, bottom_flag, top_flag in [
        (
            "income_before_tax",
            "flag_hh_qoq_income_change_bottom_1pct_ever",
            "flag_hh_qoq_income_change_top_1pct_ever",
        ),
        (
            "cons_parker_total",
            "flag_hh_qoq_cons_total_change_bottom_1pct_ever",
            "flag_hh_qoq_cons_total_change_top_1pct_ever",
        ),
    ]:
        changes = build_adjacent_qoq_pct_changes(panel, value_col)
        if changes.empty:
            raise SystemExit(f"No valid adjacent quarter-on-quarter pct changes found for {value_col}.")
        lower = changes["pct_change"].quantile(QOQ_OUTLIER_QUANTILE)
        upper = changes["pct_change"].quantile(1.0 - QOQ_OUTLIER_QUANTILE)
        hh_flags[bottom_flag] = (
            changes["pct_change"].le(lower).groupby(changes["cex_panel_id"]).any().reindex(hh_flags.index).fillna(False)
        )
        hh_flags[top_flag] = (
            changes["pct_change"].ge(upper).groupby(changes["cex_panel_id"]).any().reindex(hh_flags.index).fillna(False)
        )
        print(
            f"screen_flag_threshold::{value_col}: "
            f"p{100 * QOQ_OUTLIER_QUANTILE:.1f}={lower:.4f}, "
            f"p{100 * (1.0 - QOQ_OUTLIER_QUANTILE):.1f}={upper:.4f}"
        )

    hh_flags = hh_flags.reindex(columns=SCREEN_COMPONENT_FLAG_COLS).fillna(False)
    hh_flags["flag_hh_any_screen"] = hh_flags[SCREEN_COMPONENT_FLAG_COLS].any(axis=1)
    hh_flags[SCREEN_ALL_FLAG_COLS] = hh_flags[SCREEN_ALL_FLAG_COLS].astype("int8")

    base = df.drop(columns=[col for col in SCREEN_ALL_FLAG_COLS if col in df.columns], errors="ignore")
    out = base.merge(hh_flags.reset_index(), on="cex_panel_id", how="left")
    for col in SCREEN_ALL_FLAG_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype("int8")
    out = out.reindex(columns=FINAL_COLS)

    n_households = len(hh_flags)
    n_rows = len(out)
    for col in SCREEN_ALL_FLAG_COLS:
        hh_count = int(hh_flags[col].sum())
        obs_count = int(out[col].sum())
        print(
            f"screen_flag::{col}: households={hh_count} ({100 * hh_count / n_households:.4f}%), "
            f"observations={obs_count} ({100 * obs_count / n_rows:.4f}%)"
        )

    return out


def validate_geography_blocks(df, state_crosswalk):
    geo_checks = [
        ("region", {1, 2, 3, 4}, "Northeastern U.S.; Midwestern U.S.; Southern U.S.; Western U.S."),
        ("bls_urbn", {1, 2}, "Urban; Rural"),
        ("state", None, "State FIPS label harmonized to 1984+ coding"),
    ]
    n_rows = len(df)
    canonical_state_codes = (
        state_crosswalk.get("canonical_state_codes", set()) if state_crosswalk else set()
    )
    for col, allowed, label in geo_checks:
        if col not in df.columns:
            continue
        if col == "state":
            observed = df[col].map(_normalize_state_code).astype("string")
            allowed_codes = sorted(canonical_state_codes) if canonical_state_codes else []
            if canonical_state_codes:
                invalid_mask = (
                    ~observed.isin(pd.Series(allowed_codes, dtype="string"))
                    & observed.notna()
                )
                invalid = int(invalid_mask.fillna(False).sum())
            else:
                invalid = 0
        else:
            observed = pd.to_numeric(df[col], errors="coerce")
            allowed_codes = sorted(allowed) if allowed else []
            invalid_mask = ~observed.isin(allowed_codes)
            invalid = int((invalid_mask & observed.notna()).fillna(False).sum())

        missing = int(observed.isna().sum())
        present = n_rows - missing
        if present:
            coverage = present / n_rows
            share_invalid = invalid / present if present else 0
            top_codes = (
                observed.value_counts(dropna=True)
                .head(10)
                .to_dict()
            )
        else:
            coverage = 0.0
            share_invalid = 0
            top_codes = {}
            invalid = 0
        print(
            f"geo_check::{col}: coverage={coverage:.4f}, missing={missing}, invalid={invalid} "
            f"({share_invalid:.4f} of non-missing), allowed={allowed_codes} | labels={label}"
        )
        if top_codes:
            print(f"  top_codes={top_codes}")


def build_member_inventory(zip_files):
    inventory = []
    for zip_path in zip_files:
        zip_year = parse_zip_year(zip_path.name)
        if zip_year is None:
            raise SystemExit(f"Cannot parse CEX package year from {zip_path.name!r}.")
        with zipfile.ZipFile(zip_path) as zf:
            members = sorted(
                m for m in zf.namelist() if m.lower().endswith(".csv") and "fmli" in m.lower()
            )
            for member in members:
                token = parse_member_token(member)
                if token is None:
                    raise SystemExit(f"Cannot parse FMLI quarter token from {member!r}.")
                inventory.append(
                    {
                        "zip_path": zip_path,
                        "zip_year": zip_year,
                        "member": member,
                        "token": token,
                    }
                )
    inventory.sort(key=lambda entry: (entry["zip_year"], entry["member"]))
    if not inventory:
        raise SystemExit("No parseable FMLI members found in CEX Interview packages.")
    print(f"inventoried {len(inventory)} FMLI members")
    return inventory


def process_member(
    zf,
    package_filename,
    package_year,
    member,
    token,
    write_header,
    state_crosswalk,
):
    cols = read_header(zf, member)
    chosen = choose_sources(cols)

    if "newid" not in chosen:
        raise SystemExit(f"FMLI member has no NEWID column: {package_filename}:{member}")

    usecols = sorted(set([src for src, _ in chosen.values()] + [c for c in EXTRA_COLS if c in cols]))
    newid_source = chosen["newid"][0]
    member_counts = {
        "zip": package_filename,
        "zip_year": package_year,
        "member": member,
        "token": token,
        "raw_rows": 0,
        "month_01_rows": 0,
        "month_02_rows": 0,
        "month_03_rows": 0,
    }

    with zf.open(member) as f:
        for chunk in pd.read_csv(
            f,
            usecols=usecols,
            chunksize=200000,
            dtype={newid_source: "string"},
        ):
            out = pd.DataFrame(index=chunk.index)
            member_counts["raw_rows"] += len(chunk)

            for canon in VAR_SPECS:
                src_scale = chosen.get(canon)
                if src_scale is None:
                    out[canon] = pd.NA
                    continue
                src, scale = src_scale
                s = chunk[src]
                if canon in NUMERIC_KEYS:
                    s = pd.to_numeric(s, errors="coerce")
                    if scale != 1.0:
                        s = s * scale
                out[canon] = s

            out["newid"] = canonicalize_newid(out["newid"])

            if "INTERI" in chunk.columns:
                interi = pd.to_numeric(chunk["INTERI"], errors="coerce")
            else:
                interi = pd.Series(pd.NA, index=out.index, dtype="Float64")

            inferred = pd.to_numeric(
                out["newid"].astype("string").str[-1].where(lambda s: s.str.match(r"\d")),
                errors="coerce",
            )
            out["interview_number"] = interi.fillna(inferred).astype("Int64")
            out["interview_year"] = out["interview_year"].apply(normalize_year).astype("Int64")
            out["interview_position_harmonized"] = harmonize_interview_position(
                out["interview_number"], out["interview_year"]
            )
            out["interview_month"] = pd.to_numeric(out["interview_month"], errors="coerce").astype("Int64")
            month_counts = out["interview_month"].value_counts(dropna=False)
            for month in [1, 2, 3]:
                member_counts[f"month_{month:02d}_rows"] += int(month_counts.get(month, 0))
            out["quarter"] = out["interview_month"].apply(compute_quarter).astype("Int64")
            out["cu_id"] = pd.Series(pd.NA, index=out.index, dtype="string")
            out["cu_id_raw"] = pd.Series(pd.NA, index=out.index, dtype="string")
            out["cex_panel_id"] = pd.Series(pd.NA, index=out.index, dtype="string")
            out["cex_sample_design"] = pd.Series(pd.NA, index=out.index, dtype="string")

            out["year_quarter"] = pd.Series(pd.NA, index=out.index, dtype="string")
            mask = out["interview_year"].notna() & out["quarter"].notna()
            out.loc[mask, "year_quarter"] = (
                out.loc[mask, "interview_year"].astype(int).astype(str)
                + "Q"
                + out.loc[mask, "quarter"].astype(int).astype(str)
            )
            out["release_date"] = pd.Series(pd.NA, index=out.index, dtype="string")
            out["release_date_source"] = pd.Series(pd.NA, index=out.index, dtype="string")
            out["cex_source_package"] = package_filename
            out["cex_source_member"] = member
            out["cex_quarter_token"] = token
            out["cex_overlap_versions"] = pd.NA
            out["cex_first_release_year"] = package_year
            out["cex_retained_package_year"] = package_year

            # Harmonize state codes (including legacy state-code variants).
            out["state"] = harmonize_state_codes(out["state"], state_crosswalk)
            out["state"] = out["state"].map(_normalize_state_code).astype("string")

            for col in NUMERIC_KEYS:
                if col in out.columns:
                    out[col] = pd.to_numeric(out[col], errors="coerce")

            food_definition_regime = assign_food_definition_regime(
                out["interview_year"], out["quarter"]
            )
            redesigned_food = food_definition_regime.eq("redesigned")
            out.loc[redesigned_food, "exp_food_cq"] = out.loc[
                redesigned_food,
                ["exp_grocery_cq", "exp_foodaway_total_cq"],
            ].sum(axis=1, min_count=2)
            out.loc[redesigned_food, "exp_food_pq"] = out.loc[
                redesigned_food,
                ["exp_grocery_pq", "exp_foodaway_total_pq"],
            ].sum(axis=1, min_count=2)

            out["sex_ref_label"] = labelled_categorical(
                out["sex_ref"], CEX_SEX_LABELS, "sex_ref", "sex_ref_label"
            )
            out["sex2_label"] = labelled_categorical(
                out["sex2"], CEX_SEX_LABELS, "sex2", "sex2_label"
            )
            out["marital1_label"] = labelled_categorical(
                out["marital1"], CEX_MARITAL_LABELS, "marital1", "marital1_label"
            )
            out["region_label"] = labelled_categorical(
                out["region"], CEX_REGION_LABELS, "region", "region_label"
            )
            out["bls_urbn_label"] = labelled_categorical(
                out["bls_urbn"], CEX_URBAN_LABELS, "bls_urbn", "bls_urbn_label"
            )

            # Harmonized categorical recodes from CE dictionary code transitions.
            tenure_harmonized = recode_tenure(out["tenure"])
            race_ref_harmonized = recode_race(out["race_ref"])
            race2_harmonized = recode_race(out["race2"])
            educ_ref_harmonized = recode_education(
                out["educ_ref"], out["interview_year"], legacy_zero_is_missing=False
            )
            educ2_harmonized = recode_education(
                out["educ2"], out["interview_year"], legacy_zero_is_missing=True
            )
            validate_recode_complete(out["tenure"], tenure_harmonized, "tenure", "tenure_harmonized")
            validate_recode_complete(out["race_ref"], race_ref_harmonized, "race_ref", "race_ref_harmonized")
            validate_recode_complete(out["race2"], race2_harmonized, "race2", "race2_harmonized")
            validate_recode_complete(out["educ_ref"], educ_ref_harmonized, "educ_ref", "educ_ref_harmonized")
            validate_recode_complete(out["educ2"], educ2_harmonized, "educ2", "educ2_harmonized")
            out["tenure_harmonized"] = labelled_categorical(
                tenure_harmonized, CEX_TENURE_LABELS, "tenure", "tenure_harmonized"
            )
            out["race_ref_harmonized"] = labelled_categorical(
                race_ref_harmonized, CEX_RACE_LABELS, "race_ref", "race_ref_harmonized"
            )
            out["race2_harmonized"] = labelled_categorical(
                race2_harmonized, CEX_RACE_LABELS, "race2", "race2_harmonized"
            )
            out["educ_ref_harmonized"] = labelled_categorical(
                educ_ref_harmonized,
                CEX_EDUCATION_LABELS,
                "educ_ref",
                "educ_ref_harmonized",
                ordered=True,
            )
            out["educ2_harmonized"] = labelled_categorical(
                educ2_harmonized,
                CEX_EDUCATION_LABELS,
                "educ2",
                "educ2_harmonized",
                ordered=True,
            )

            # Aggregate consumption measure requested by user: CQ + PQ.
            for out_col, (cq_col, pq_col) in CONSUMPTION_COMPONENTS.items():
                out[out_col] = out[[cq_col, pq_col]].sum(axis=1, min_count=1)
            out["food_definition_regime"] = food_definition_regime

            parker_subtractions = out[
                [
                    "cons_cash_contrib",
                    "cons_life_insurance",
                    "cons_retirement_pension_social_security",
                ]
            ].fillna(0).sum(axis=1)
            out["cons_parker_total"] = out["cons_total"] - parker_subtractions

            # Aggregates for task design:
            # - food: food + alcohol
            # - strictly non-durables: food + utilities + household operations + public transportation
            #   + gasoline/motor oil + personal care + tobacco + miscellaneous
            # - non-durables: strictly non-durables + apparel + health + reading
            # - durables: Parker-style total - non-durables
            out["cons_food_agg"] = out[["cons_food", "cons_alcohol"]].sum(axis=1, min_count=1)
            out["cons_strict_nondurables"] = out[
                [
                    "cons_food_agg",
                    "cons_utilities",
                    "cons_housop",
                    "cons_pubtra",
                    "cons_gasmo",
                    "cons_personal_care",
                    "cons_tobacco",
                    "cons_misc",
                ]
            ].sum(axis=1, min_count=1)
            out["cons_nondurables"] = out[
                ["cons_strict_nondurables", "cons_apparel", "cons_health", "cons_read"]
            ].sum(axis=1, min_count=1)
            out["cons_durables"] = out["cons_parker_total"] - out["cons_nondurables"]
            missing_food_input = out["cons_food"].isna()
            out.loc[
                missing_food_input,
                [
                    "cons_food_agg",
                    "cons_strict_nondurables",
                    "cons_nondurables",
                    "cons_durables",
                ],
            ] = pd.NA
            missing_durable_inputs = out[["cons_parker_total", "cons_nondurables"]].isna().any(axis=1)
            out.loc[missing_durable_inputs, "cons_durables"] = pd.NA

            out = out.reindex(columns=FINAL_COLS)
            out.to_csv(TEMP_CSV_PATH, mode="w" if write_header else "a", index=False, header=write_header)
            write_header = False

    return write_header, member_counts


def reconcile_cex_records(candidates):
    record_key = ["newid", "interview_year", "interview_month"]
    source_key = [
        *record_key,
        "cex_retained_package_year",
        "cex_source_member",
    ]
    required = [
        *source_key,
        "cex_source_package",
        "cex_quarter_token",
    ]
    missing_columns = [column for column in required if column not in candidates.columns]
    if missing_columns:
        raise SystemExit(f"CEX reconciliation is missing columns: {missing_columns}")
    missing_components = candidates[required].isna() | candidates[required].astype("string").apply(
        lambda column: column.str.strip().eq("")
    )
    if missing_components.any(axis=1).any():
        counts = missing_components.sum().loc[lambda values: values.gt(0)].to_dict()
        raise SystemExit(f"CEX reconciliation has missing key or provenance values: {counts}")

    within_source = candidates.duplicated(source_key, keep=False)
    if within_source.any():
        raise SystemExit(
            f"CEX reconciliation found {int(within_source.sum()):,} within-source duplicate rows."
        )
    package_key = [*record_key, "cex_retained_package_year"]
    member_counts = candidates.groupby(package_key, dropna=False)["cex_source_member"].nunique()
    competing = member_counts.gt(1)
    if competing.any():
        raise SystemExit(
            f"CEX reconciliation found {int(competing.sum()):,} record keys in competing members "
            "within one annual package."
        )

    grouped = candidates.groupby(record_key, sort=False, dropna=False)[
        "cex_retained_package_year"
    ]
    candidates["cex_first_release_year"] = grouped.transform("min").astype(int)
    candidates["cex_overlap_versions"] = grouped.transform("nunique").astype(int)
    candidates = candidates.sort_values(
        [*record_key, "cex_retained_package_year", "cex_source_member"],
        kind="mergesort",
    ).reset_index(drop=True)
    retained = ~candidates.duplicated(record_key, keep="last")
    candidates["overlap_outcome"] = "discarded_earlier_duplicate"
    candidates.loc[retained & candidates["cex_overlap_versions"].eq(1), "overlap_outcome"] = (
        "retained_unique"
    )
    candidates.loc[retained & candidates["cex_overlap_versions"].gt(1), "overlap_outcome"] = (
        "retained_later"
    )

    audit_group = [
        "cex_quarter_token",
        "interview_year",
        "interview_month",
        "cex_source_package",
        "cex_source_member",
        "overlap_outcome",
    ]
    overlap_audit = (
        candidates.groupby(audit_group, dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(audit_group, kind="mergesort")
    )
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    overlap_audit.to_csv(RECORD_OVERLAP_AUDIT_PATH, index=False)

    reconciled = candidates.loc[retained].drop(columns="overlap_outcome").copy()
    if reconciled.duplicated(record_key).any():
        raise SystemExit("CEX reconciliation did not produce unique interview-record keys.")
    largest_package = candidates.groupby(record_key, dropna=False)[
        "cex_retained_package_year"
    ].max()
    retained_package = reconciled.set_index(record_key)["cex_retained_package_year"]
    if not retained_package.astype(int).equals(largest_package.astype(int)):
        raise SystemExit("CEX reconciliation did not retain the latest annual package.")

    reconciled["cu_id_raw"] = reconciled["newid"].astype("string").str[:-1]
    if not reconciled["cu_id_raw"].str.fullmatch(r"[0-9]{7}").fillna(False).all():
        raise SystemExit("CEX reconciliation produced an invalid raw CU identifier.")
    reconciled["cex_sample_design"] = assign_cex_sample_design(reconciled)
    reconciled["cex_panel_id"] = (
        reconciled["cex_sample_design"].astype("string")
        + ":"
        + reconciled["cu_id_raw"].astype("string")
    )
    reconciled["cu_id"] = reconciled["cu_id_raw"]
    reconciled = reconciled.sort_values(record_key, kind="mergesort").reset_index(drop=True)
    validate_cex_panel_designs(reconciled)

    candidate_rows = len(candidates)
    retained_rows = len(reconciled)
    discarded_rows = int((candidates["overlap_outcome"] == "discarded_earlier_duplicate").sum())
    if candidate_rows != retained_rows + discarded_rows:
        raise SystemExit("CEX overlap decomposition does not reconcile candidate and retained rows.")
    print(
        "cex_record_reconciliation: "
        f"candidate_rows={candidate_rows:,}, retained_rows={retained_rows:,}, "
        f"discarded_earlier_duplicate_rows={discarded_rows:,}"
    )
    return reconciled


def load_cex_release_date_crosswalk():
    required_columns = ["cex_data_year", "release_date", "source", "source_url", "comments"]
    if not RELEASE_DATE_CROSSWALK_PATH.exists():
        raise SystemExit(f"Missing CEX release-date crosswalk: {RELEASE_DATE_CROSSWALK_PATH}")

    crosswalk = pd.read_csv(RELEASE_DATE_CROSSWALK_PATH, dtype="string")
    missing_columns = [column for column in required_columns if column not in crosswalk.columns]
    if missing_columns:
        raise SystemExit(
            f"{RELEASE_DATE_CROSSWALK_PATH} is missing required columns: {missing_columns}"
        )

    crosswalk = crosswalk.loc[:, required_columns].copy()
    crosswalk["cex_data_year"] = pd.to_numeric(crosswalk["cex_data_year"], errors="coerce")
    crosswalk["release_date"] = pd.to_datetime(crosswalk["release_date"], errors="coerce")
    crosswalk["source"] = crosswalk["source"].astype("string").str.strip()
    crosswalk["comments"] = crosswalk["comments"].fillna("").astype("string").str.strip()

    if crosswalk["cex_data_year"].isna().any():
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains missing cex_data_year values.")
    if crosswalk["release_date"].isna().any():
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains missing release_date values.")
    if crosswalk["source"].isna().any() or crosswalk["source"].str.strip().eq("").any():
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains missing source values.")
    if crosswalk["source_url"].isna().any() or crosswalk["source_url"].str.strip().eq("").any():
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains missing source_url values.")
    if crosswalk["comments"].isna().any() or crosswalk["comments"].str.strip().eq("").any():
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains missing comments values.")

    crosswalk["cex_data_year"] = crosswalk["cex_data_year"].astype(int)
    duplicate_years = crosswalk.loc[crosswalk["cex_data_year"].duplicated(), "cex_data_year"]
    if not duplicate_years.empty:
        raise SystemExit(
            f"{RELEASE_DATE_CROSSWALK_PATH} has duplicate CEX data years: "
            f"{sorted(duplicate_years.astype(int).unique())}"
        )

    unexpected_sources = sorted(set(crosswalk["source"]) - CEX_RELEASE_DATE_SOURCE_VALUES)
    if unexpected_sources:
        raise SystemExit(
            f"{RELEASE_DATE_CROSSWALK_PATH} has unexpected release-date sources: "
            f"{unexpected_sources}"
        )

    return crosswalk


def assign_cex_release_dates(df, release_crosswalk):
    required_columns = [
        "cex_first_release_year",
        "cex_retained_package_year",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise SystemExit(f"CEX release-date assignment is missing columns: {missing_columns}")

    first_release_year = pd.to_numeric(df["cex_first_release_year"], errors="coerce")
    retained_package_year = pd.to_numeric(df["cex_retained_package_year"], errors="coerce")

    missing_timing = first_release_year.isna() | retained_package_year.isna()
    if missing_timing.any():
        raise SystemExit(f"Cannot assign CEX release dates for {int(missing_timing.sum())} rows.")

    first_release_year = first_release_year.astype(int)
    retained_package_year = retained_package_year.astype(int)

    if retained_package_year.lt(first_release_year).any():
        raise SystemExit("CEX retained package year is earlier than first release year.")

    release_lookup = release_crosswalk.set_index("cex_data_year", verify_integrity=True)
    missing_years = sorted(set(first_release_year) - set(release_lookup.index))
    if missing_years:
        raise SystemExit(f"CEX release-date crosswalk is missing years: {missing_years}")

    release_date = pd.to_datetime(first_release_year.map(release_lookup["release_date"]), errors="coerce")
    release_date_source = first_release_year.map(release_lookup["source"]).astype("string")

    if release_date.isna().any():
        raise SystemExit("CEX release-date assignment left missing release_date values.")
    if release_date_source.isna().any() or release_date_source.str.strip().eq("").any():
        raise SystemExit("CEX release-date assignment left missing release_date_source values.")

    df["release_date"] = release_date.dt.strftime("%Y-%m-%d").astype("string")
    df["release_date_source"] = release_date_source
    df["cex_first_release_year"] = first_release_year
    df["cex_retained_package_year"] = retained_package_year
    return df


def build_release_date_coverage(df):
    required_columns = ["release_date_source", "release_date"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise SystemExit(f"CEX release-date coverage is missing columns: {missing_columns}")

    coverage = (
        df.groupby(["release_date_source", "release_date"], dropna=False)
        .size()
        .reset_index(name="rows")
        .rename(
            columns={
                "release_date_source": "release_rule_id",
                "release_date": "microdata_release_date",
            }
        )
    )
    coverage.insert(0, "dataset", "CEX")
    coverage["unmatched_rows"] = 0
    return coverage[
        ["dataset", "release_rule_id", "microdata_release_date", "rows", "unmatched_rows"]
    ]


def write_release_date_coverage(coverage):
    RELEASE_DATE_COVERAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = ["dataset", "release_rule_id", "microdata_release_date", "rows", "unmatched_rows"]

    if RELEASE_DATE_COVERAGE_PATH.exists():
        existing = pd.read_csv(RELEASE_DATE_COVERAGE_PATH)
        missing_columns = [column for column in columns if column not in existing.columns]
        if missing_columns:
            raise SystemExit(
                f"{RELEASE_DATE_COVERAGE_PATH} is missing required columns: {missing_columns}"
            )
        existing = existing.loc[existing["dataset"].ne("CEX"), columns].copy()
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


def main():
    validate_source_inputs("cex")
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(RAW_DIR / name for name in REQUIRED_ARCHIVES)
    missing = [str(path) for path in zip_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required CEX Interview archives are missing: {missing}")

    state_crosswalk = STATE_CROSSWALK
    release_crosswalk = load_cex_release_date_crosswalk()

    member_inventory = build_member_inventory(zip_files)
    members_by_zip = defaultdict(list)
    for entry in member_inventory:
        members_by_zip[entry["zip_path"]].append(entry)

    for path in [OUT_PATH, TEMP_CSV_PATH]:
        if path.exists():
            path.unlink()

    write_header = True
    member_inventory_rows = []
    for zip_path in sorted(members_by_zip):
        with zipfile.ZipFile(zip_path) as zf:
            for entry in sorted(members_by_zip[zip_path], key=lambda item: item["member"]):
                member = entry["member"]
                print(f"processing {zip_path.name}:{member}")
                write_header, member_counts = process_member(
                    zf,
                    zip_path.name,
                    entry["zip_year"],
                    member,
                    entry["token"],
                    write_header,
                    state_crosswalk,
                )
                member_inventory_rows.append(member_counts)

    if write_header:
        raise SystemExit("No rows were written; check source files and variable mappings.")

    pd.DataFrame(member_inventory_rows).to_csv(MEMBER_INVENTORY_PATH, index=False)

    candidates = pd.read_csv(
        TEMP_CSV_PATH,
        dtype={
            "newid": "string",
            "cu_id": "string",
            "cu_id_raw": "string",
            "cex_panel_id": "string",
            "cex_sample_design": "string",
            "cex_source_package": "string",
            "cex_source_member": "string",
            "cex_quarter_token": "string",
            "state": "string",
        },
    )
    df = reconcile_cex_records(candidates)

    # Geography block quality checks: coverage and unexpected category levels.
    # These are logged as run-time diagnostics and intended to catch schema breaks.
    validate_geography_blocks(df, state_crosswalk)
    df = assign_cex_release_dates(df, release_crosswalk)
    df = build_household_quality_flags(df)

    category_specs = {
        "region_label": (CEX_REGION_LABELS, False),
        "bls_urbn_label": (CEX_URBAN_LABELS, False),
        "sex_ref_label": (CEX_SEX_LABELS, False),
        "sex2_label": (CEX_SEX_LABELS, False),
        "marital1_label": (CEX_MARITAL_LABELS, False),
        "tenure_harmonized": (CEX_TENURE_LABELS, False),
        "race_ref_harmonized": (CEX_RACE_LABELS, False),
        "race2_harmonized": (CEX_RACE_LABELS, False),
        "educ_ref_harmonized": (CEX_EDUCATION_LABELS, True),
        "educ2_harmonized": (CEX_EDUCATION_LABELS, True),
    }
    for column, (labels, ordered) in category_specs.items():
        df[column] = pd.Categorical(df[column], categories=list(labels.values()), ordered=ordered)

    df.to_parquet(OUT_PATH, index=False)
    TEMP_CSV_PATH.unlink()
    with exclusive_lock(RELEASE_DATE_COVERAGE_PATH):
        write_release_date_coverage(build_release_date_coverage(df))

    print(f"done: {OUT_PATH}")
    print(
        f"rows: {len(df):,}; unique newid: {df['newid'].nunique():,}; "
        f"unique cex_panel_id: {df['cex_panel_id'].nunique():,}; "
        f"unique cu_id_raw: {df['cu_id_raw'].nunique():,}"
    )


if __name__ == "__main__":
    main()
