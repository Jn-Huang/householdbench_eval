#!/usr/bin/env python
"""Build cleaned Decennial Census 1990 and 2000 microdata panel from local IPUMS Stata extract."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.source_inputs import validate_source_inputs

from scripts.utils.io import exclusive_lock, sha256_file

RAW_DTA_PATH = Path("data/raw/micro/census/census.gz")
MAPPING_DIR = Path(__file__).resolve().parent / "mappings"
OUT_DATA_PATH = Path("data/intermediate/census_decennial.parquet")
OUT_DIR = Path("output/preprocessing/census/build")
RELEASE_DATE_CROSSWALK_PATH = Path("data/raw/release_dates/release_census.csv")
RELEASE_DATE_COVERAGE_PATH = Path("output/preprocessing/release_dates/metadata_coverage.csv")

SAMPLE_ORDER = [199002, 200007]
SAMPLE_TO_YEAR = {199002: 1990, 200007: 2000}
# Core subset for downstream modeling and validation.
KEEP_COLS = [
    "YEAR",
    "SAMPLE",
    "SERIAL",
    "PERNUM",
    "NUMPREC",
    "FAMSIZE",
    "HHWT",
    "PERWT",
    "CPI99",
    "STATEFIP",
    "REGION",
    "METRO",
    "METAREA",
    "METAREAD",
    "PUMA",
    "CNTYGP97",
    "CNTYGP98",
    "PUMASUPR",
    "GQ",
    "OWNERSHP",
    "VALUEH",
    "RENTGRS",
    "ROOMS",
    "MOMLOC",
    "POPLOC",
    "SPLOC",
    "RELATE",
    "AGE",
    "SEX",
    "RACE",
    "RACED",
    "HISPAN",
    "HISPAND",
    "EDUC",
    "EDUCD",
    "SCHOOL",
    "GRADEATT",
    "GRADEATTD",
    "EMPSTAT",
    "LABFORCE",
    "CLASSWKR",
    "OCC",
    "OCC1950",
    "OCC1990",
    "IND",
    "IND1950",
    "IND1990",
    "WORKEDYR",
    "WKSWORK1",
    "WKSWORK2",
    "UHRSWORK",
    "INCTOT",
    "INCWAGE",
    "INCBUS",
    "INCFARM",
    "INCSS",
    "INCINVST",
    "INCRETIR",
    "POVERTY",
    "MIGRATE5",
    "MIGPLAC5",
    "MIGMETRO5",
    "MIGCITY5",
]

CORE_BURDEN_VARS = ["AGE", "SEX", "RACE", "EDUC", "EMPSTAT", "OCC", "INCTOT", "OWNERSHP"]
REAL_INCOME_VARS = ["INCTOT", "INCWAGE", "INCBUS", "INCFARM", "INCSS", "INCINVST", "INCRETIR"]
QUALITY_FLAGS_REQUESTED = ["QEDUC", "QINCWAGE", "QINCBUS", "QINCFARM", "QINCSS", "QINCINVS", "QINCRETI", "QMIGRAT5", "QOWNERSH", "QAUGMENT", "QSUBSTIH"]

INCOME_CENSORING_RULES = [
    # variable, year, bottom code, top threshold, flag rule, replacement rule, source URL
    ("INCTOT", 1990, -19998, 400000, "gt", "state_median_replacement", "https://usa.ipums.org/usa-action/variables/INCTOT"),
    ("INCTOT", 2000, -20000, 999998, "ge", "state_mean_replacement", "https://usa.ipums.org/usa-action/variables/INCTOT"),
    ("INCWAGE", 1990, None, 140000, "gt", "state_median_replacement", "https://usa.ipums.org/usa-action/variables/INCWAGE"),
    ("INCWAGE", 2000, None, 175000, "gt", "state_mean_replacement", "https://usa.ipums.org/usa-action/variables/INCWAGE"),
    ("INCBUS", 1990, -9900, 90000, "gt", "state_median_replacement", "https://usa.ipums.org/usa-action/variables/INCBUS"),
    ("INCFARM", 1990, -9999, 54000, "gt", "state_median_replacement", "https://usa.ipums.org/usa-action/variables/INCFARM"),
    ("INCSS", 1990, None, 17000, "gt", "state_median_replacement", "https://usa.ipums.org/usa-action/variables/INCSS"),
    ("INCSS", 2000, None, 18000, "gt", "state_mean_replacement", "https://usa.ipums.org/usa-action/variables/INCSS"),
    ("INCINVST", 1990, -9999, 40000, "gt", "state_median_replacement", "https://usa.ipums.org/usa-action/variables/INCINVST"),
    ("INCINVST", 2000, -10000, 50000, "gt", "state_mean_replacement", "https://usa.ipums.org/usa-action/variables/INCINVST"),
    ("INCRETIR", 1990, None, 30000, "gt", "state_median_replacement", "https://usa.ipums.org/usa-action/variables/INCRETIR"),
    ("INCRETIR", 2000, None, 52000, "gt", "state_mean_replacement", "https://usa.ipums.org/usa-action/variables/INCRETIR"),
]

INCOME_CENSORING_RULE_DF = pd.DataFrame(
    INCOME_CENSORING_RULES,
    columns=[
        "variable",
        "year",
        "bottom_code",
        "top_threshold",
        "flag_rule",
        "replacement_rule",
        "source_url",
    ],
)

# Explicit fallback NIU/missing codes for variables where Stata labels are absent/incomplete.
MANUAL_NIU_CODES = {
    "INCTOT": {9999999},
    "INCBUS": {999999},
    "INCFARM": {999999},
    "INCSS": {99999},
    "INCINVST": {999999},
    "INCRETIR": {999999},
    "POVERTY": {0},
    "OCC": {0},
    "IND": {0},
}
MANUAL_VALUE_LABELS = {
    # Present in the 1980 extract for same-house movers:
    # MIGRATE5=1 and MIGPLAC5=990 are both labelled "same house".
    "MIGMETRO5": {5: "Same house"},
}
CENSUS_LABELLED_CATEGORICAL_VARS = [
    "SEX",
    "RACE",
    "RACED",
    "HISPAN",
    "HISPAND",
    "EDUC",
    "EDUCD",
    "STATEFIP",
    "REGION",
    "METRO",
    "OWNERSHP",
    "MIGRATE5",
    "MIGPLAC5",
    "MIGMETRO5",
    "MIGCITY5",
    "RELATE",
    "GQ",
]


def is_niu_like_label(label: str) -> bool:
    text = " ".join(label.upper().split())
    patterns = (
        r"\bNOT IN UNIVERSE\b",
        r"\bNOT APPLICABLE\b",
        r"(?<![A-Z0-9])N/A(?![A-Z0-9])",
        r"\bMISSING\b",
        r"\bUNKNOWN\b",
        r"\bILLEGIBLE\b",
        r"\bNOT ASCERTAINED\b",
        r"\bBLANK\b",
        r"\bUNCLASSIFIABLE\b",
    )
    return any(re.search(pat, text) for pat in patterns)


def parse_niu_from_stata_labels(
    value_labels: dict[str, dict[object, str]],
    allowed_vars: set[str],
) -> tuple[dict[str, set[float]], pd.DataFrame]:
    niu_map: dict[str, set[float]] = defaultdict(set)
    rows: list[dict[str, object]] = []

    for var in allowed_vars:
        label_map = value_labels.get(var)
        if not label_map:
            continue
        for raw_code, raw_label in label_map.items():
            label = str(raw_label).strip()
            if not label or not is_niu_like_label(label):
                continue
            try:
                numeric_code = float(raw_code)
            except Exception:
                continue
            niu_map[var].add(numeric_code)
            rows.append(
                {
                    "variable": var,
                    "code": str(raw_code),
                    "code_numeric": numeric_code,
                    "label": label,
                    "source": "stata_value_label",
                }
            )
    return niu_map, pd.DataFrame(rows)


def combine_niu_maps(*maps: dict[str, set[float]]) -> dict[str, set[float]]:
    out: dict[str, set[float]] = defaultdict(set)
    for mp in maps:
        for var, codes in mp.items():
            out[var].update(codes)
    return out


def normalized_value_label_map(value_labels: dict[str, dict[object, str]], variable: str) -> dict[int, str]:
    raw_map = value_labels.get(variable, {})
    out: dict[int, str] = {}
    for raw_code, raw_label in raw_map.items():
        try:
            code = float(raw_code)
        except Exception:
            continue
        rounded = int(round(code))
        if abs(code - rounded) > 1e-9:
            continue
        label = str(raw_label).strip()
        if not label:
            continue
        out[rounded] = label
    return out


def categorical_labels_from_stata_codes(
    series: pd.Series,
    label_map: dict[int, str],
    variable: str,
) -> pd.Categorical:
    if not label_map:
        raise RuntimeError(f"{variable} has no Stata value labels in the Census extract.")

    numeric = pd.to_numeric(series, errors="coerce")
    source_nonmissing = series.notna()
    nonnumeric = source_nonmissing & numeric.isna()
    rounded = numeric.round()
    noninteger = numeric.notna() & ((numeric - rounded).abs() > 1e-9)
    codes = rounded.astype("Int64")
    unmapped = numeric.notna() & ~codes.isin(set(label_map))
    bad = nonnumeric | noninteger | unmapped
    if bad.any():
        examples = sorted(series.loc[bad].dropna().astype(str).unique().tolist())[:20]
        raise RuntimeError(f"{variable} has nonmissing codes without Census Stata labels: {examples}")

    categories = list(dict.fromkeys(label_map[key] for key in sorted(label_map)))
    return pd.Categorical(codes.map(label_map), categories=categories)


def add_stata_label_columns(
    chunk: pd.DataFrame,
    value_labels: dict[str, dict[object, str]],
    variables: list[str],
) -> pd.DataFrame:
    out = chunk.copy()
    for variable in variables:
        if variable not in out.columns:
            raise RuntimeError(f"Cannot label missing Census variable: {variable}")
        label_map = normalized_value_label_map(value_labels, variable)
        label_map.update(MANUAL_VALUE_LABELS.get(variable, {}))
        out[f"{variable}_label"] = categorical_labels_from_stata_codes(out[variable], label_map, variable)
    return out


def enforce_structural_missing(
    chunk: pd.DataFrame,
    availability: dict[str, dict[int, bool]],
    counts: Counter,
) -> pd.Series:
    sample = pd.to_numeric(chunk["SAMPLE"], errors="coerce")
    structural_missing_count = pd.Series(0, index=chunk.index, dtype="int16")

    for var in KEEP_COLS:
        if var not in chunk.columns:
            continue
        if var not in availability:
            continue
        available_samples = [s for s, is_avail in availability[var].items() if is_avail]
        mask_unavailable_sample = ~sample.isin(available_samples)
        if not mask_unavailable_sample.any():
            continue
        to_blank = mask_unavailable_sample & chunk[var].notna()
        n = int(to_blank.sum())
        if n > 0:
            chunk.loc[to_blank, var] = pd.NA
            counts[var] += n
        structural_missing_count = structural_missing_count + mask_unavailable_sample.astype("int16")
    return structural_missing_count


def apply_niu_recode(
    chunk: pd.DataFrame,
    niu_codes: dict[str, set[float]],
    recode_counts: Counter,
) -> None:
    for var, codes in niu_codes.items():
        if var not in chunk.columns or not codes:
            continue
        numeric = pd.to_numeric(chunk[var], errors="coerce")
        mask = numeric.isin(codes)
        n = int(mask.sum())
        if n > 0:
            chunk.loc[mask, var] = pd.NA
            recode_counts[var] += n


def add_derived_columns(chunk: pd.DataFrame) -> pd.DataFrame:
    year = pd.to_numeric(chunk["YEAR"], errors="coerce")
    sample = pd.to_numeric(chunk["SAMPLE"], errors="coerce")
    perwt = pd.to_numeric(chunk["PERWT"], errors="coerce")
    hhwt = pd.to_numeric(chunk["HHWT"], errors="coerce")
    cpi99 = pd.to_numeric(chunk["CPI99"], errors="coerce")

    # Retain the existing output schema; neither supported sample is a 1970 form.
    chunk["FORM_1970"] = np.int8(0)
    chunk["IS_1970_FORM1"] = np.int8(0)
    chunk["IS_1970_FORM2"] = np.int8(0)
    chunk["PERWT_ANALYSIS"] = perwt * 1.0
    chunk["HHWT_ANALYSIS"] = hhwt * 1.0

    chunk["WEIGHT_REGIME"] = (year >= 1990).astype("int8")
    chunk["EDUC_DEGREE_ERA"] = (year >= 1990).astype("int8")
    chunk["HISPAN_IMPUTED_ERA"] = np.int8(0)
    chunk["MULTIRACE_ERA"] = (year >= 2000).astype("int8")

    race = pd.to_numeric(chunk.get("RACE"), errors="coerce")
    # In this extract, code 7 exists pre-2000 ("other race"), while 8/9 capture multiracial coding.
    chunk["RACE_MULTIPLE_MARKED"] = race.isin([8, 9]).fillna(False).astype("int8")

    statefip = pd.to_numeric(chunk.get("STATEFIP"), errors="coerce")
    chunk["STATEFIP_UNIDENTIFIED"] = statefip.eq(99).fillna(False).astype("int8")
    chunk["PERWT_ZERO_FLAG"] = (perwt == 0).fillna(False).astype("int8")

    for var in REAL_INCOME_VARS:
        flag_col = f"{var}_TOPCODE_KNOWN"
        chunk[flag_col] = pd.Series(pd.NA, index=chunk.index, dtype="Int8")
        values = pd.to_numeric(chunk.get(var), errors="coerce")
        for rule in INCOME_CENSORING_RULE_DF[INCOME_CENSORING_RULE_DF["variable"] == var].itertuples(index=False):
            year_mask = year == int(rule.year)
            chunk.loc[year_mask & values.notna(), flag_col] = 0
            if rule.flag_rule == "eq":
                hit = values == float(rule.top_threshold)
            elif rule.flag_rule == "gt":
                hit = values > float(rule.top_threshold)
            elif rule.flag_rule == "ge":
                hit = values >= float(rule.top_threshold)
            else:
                raise RuntimeError(f"Unknown income censoring flag rule: {rule.flag_rule}")
            chunk.loc[year_mask & hit, flag_col] = 1

    for var in REAL_INCOME_VARS:
        nom = pd.to_numeric(chunk.get(var), errors="coerce")
        chunk[f"{var}_REAL99"] = nom * cpi99

    core_missing = chunk[CORE_BURDEN_VARS].isna().mean(axis=1)
    chunk["CORE_MISSING_BURDEN"] = core_missing.astype("float32")

    return chunk


def choose_chunk_size(n_cols: int, requested: int) -> int:
    if requested < 25_000:
        raise SystemExit("chunksize is too small; use at least 25,000 rows.")
    if n_cols <= 80:
        return requested
    return min(requested, 200_000)


def load_census_release_date_crosswalk() -> pd.DataFrame:
    required_columns = ["year", "sample", "release_date", "source", "source_url", "comments"]
    if not RELEASE_DATE_CROSSWALK_PATH.exists():
        raise SystemExit(f"Missing Census release-date crosswalk: {RELEASE_DATE_CROSSWALK_PATH}")

    crosswalk = pd.read_csv(
        RELEASE_DATE_CROSSWALK_PATH,
        dtype={"source": "string", "source_url": "string", "comments": "string"},
    )
    missing = [column for column in required_columns if column not in crosswalk.columns]
    if missing:
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} is missing columns: {missing}")

    crosswalk = crosswalk.loc[:, required_columns].copy()
    crosswalk["year"] = pd.to_numeric(crosswalk["year"], errors="coerce").astype("Int64")
    crosswalk["sample"] = pd.to_numeric(crosswalk["sample"], errors="coerce").astype("Int64")
    crosswalk["release_date"] = pd.to_datetime(crosswalk["release_date"], errors="coerce")
    if crosswalk[["year", "sample"]].isna().any().any():
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains missing or nonnumeric keys.")
    if crosswalk.duplicated(["year", "sample"]).any():
        duplicates = crosswalk.loc[crosswalk.duplicated(["year", "sample"]), ["year", "sample"]]
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains duplicate keys:\n{duplicates.to_string(index=False)}")

    crosswalk = crosswalk.loc[crosswalk["sample"].isin(SAMPLE_ORDER)].copy()
    observed_pairs = set(zip(crosswalk["year"].astype(int), crosswalk["sample"].astype(int), strict=True))
    expected_pairs = {(SAMPLE_TO_YEAR[sample], sample) for sample in SAMPLE_ORDER}
    missing_pairs = sorted(expected_pairs - observed_pairs)
    unexpected_pairs = sorted(observed_pairs - expected_pairs)
    if missing_pairs or unexpected_pairs:
        raise SystemExit(
            f"{RELEASE_DATE_CROSSWALK_PATH} does not match the supported Census sample list. "
            f"missing={missing_pairs}; unexpected={unexpected_pairs}"
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


def assign_census_release_dates(df: pd.DataFrame, release_crosswalk: pd.DataFrame) -> pd.DataFrame:
    key = pd.DataFrame(
        {
            "year": pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64"),
            "sample": pd.to_numeric(df["SAMPLE"], errors="coerce").astype("Int64"),
        },
        index=df.index,
    )
    if key.isna().any().any():
        raise SystemExit("Census release-date assignment found missing YEAR or SAMPLE values.")

    mapped = key.merge(release_crosswalk, on=["year", "sample"], how="left", validate="many_to_one")
    mapped.index = df.index
    out = pd.DataFrame(
        {
            "release_date": mapped["release_date"].astype("string"),
            "release_date_source": mapped["source"].astype("string"),
        },
        index=df.index,
    )
    if out["release_date"].isna().any():
        bad = key.loc[out["release_date"].isna()].drop_duplicates().head(10)
        raise SystemExit(f"Census release-date crosswalk does not cover keys:\n{bad.to_string(index=False)}")
    if out["release_date_source"].isna().any() or out["release_date_source"].str.strip().eq("").any():
        raise SystemExit("Census release-date assignment left missing release_date_source values.")
    return out


def add_release_assignment_to_counter(counter: Counter[tuple[str, str]], release_assignment: pd.DataFrame) -> None:
    required = {"release_date_source", "release_date"}
    missing = required - set(release_assignment.columns)
    if missing:
        raise SystemExit(f"Census release-date coverage is missing columns: {sorted(missing)}")
    counts = release_assignment.groupby(["release_date_source", "release_date"], dropna=False).size()
    for key, count in counts.items():
        counter[(str(key[0]), str(key[1]))] += int(count)


def coverage_rows_from_counter(dataset: str, counter: Counter[tuple[str, str]]) -> pd.DataFrame:
    rows = [
        {
            "dataset": dataset,
            "release_rule_id": rule,
            "microdata_release_date": date,
            "rows": count,
            "unmatched_rows": 0,
        }
        for (rule, date), count in sorted(counter.items())
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
        raise SystemExit(f"Census release-date coverage is missing columns: {missing}")

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
    # 1. Resolve paths and check the source files.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dta", type=Path, default=RAW_DTA_PATH)
    parser.add_argument("--out-data", type=Path, default=OUT_DATA_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--max-chunks", type=int, default=None, help="Optional smoke-test cap on chunk count.")
    args = parser.parse_args()
    validate_source_inputs("census", overrides={RAW_DTA_PATH: args.raw_dta})
    if not args.raw_dta.exists():
        raise SystemExit(f"Missing raw Census Stata file: {args.raw_dta}")
    args.out_data.parent.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # pandas' Stata reader seeks back to the file's string table before every chunk. On a gzip
    # stream each backward seek restarts decompression, so read one decompressed copy instead;
    # it holds the same bytes, so the chunks and the output do not change.
    scratch = tempfile.TemporaryDirectory(prefix="census_")
    stata_path = Path(scratch.name) / "census.dta"
    with gzip.open(args.raw_dta, "rb") as source, stata_path.open("wb") as target:
        shutil.copyfileobj(source, target, length=1 << 24)

    # 2. Read the source labels, availability and missing-value definitions.
    reader = pd.read_stata(stata_path, convert_categoricals=False, iterator=True)
    variable_labels = reader.value_labels()
    first_chunk = reader.read(5)
    if first_chunk.empty:
        raise SystemExit("Census Stata file appears empty.")

    available_cols = [c.upper() for c in first_chunk.columns]
    # These legacy geographies are unavailable in both retained samples.
    source_columns = [c for c in KEEP_COLS if c not in {"CNTYGP97", "CNTYGP98"}]
    missing_keep = [c for c in source_columns if c not in available_cols]
    if missing_keep:
        raise SystemExit(
            "Required columns not present in raw Census extract:\n"
            + "\n".join(f"- {col}" for col in missing_keep)
        )

    # Frozen IPUMS codebook mappings; Stata labels remain part of the raw extract.
    availability_rows = pd.read_csv(MAPPING_DIR / "census_availability.csv")
    availability = {
        row["variable"]: {sample: row[str(sample)] for sample in SAMPLE_ORDER}
        for row in availability_rows.to_dict("records")
    }
    allowed_vars = set(KEEP_COLS)
    niu_codebook_audit = pd.read_csv(MAPPING_DIR / "census_missing_codes.csv", dtype={"code": str}, keep_default_na=False)
    niu_from_codebook = {
        variable: set(rows["code_numeric"])
        for variable, rows in niu_codebook_audit.groupby("variable", sort=False)
    }
    niu_from_stata, niu_stata_audit = parse_niu_from_stata_labels(variable_labels, allowed_vars)
    niu_manual = {k: {float(v) for v in vals} for k, vals in MANUAL_NIU_CODES.items() if k in allowed_vars}

    niu_map = combine_niu_maps(niu_from_codebook, niu_from_stata, niu_manual)
    release_crosswalk = load_census_release_date_crosswalk()

    niu_manual_rows = []
    for var, codes in niu_manual.items():
        for code in sorted(codes):
            niu_manual_rows.append(
                {
                    "variable": var,
                    "code": str(int(code) if float(code).is_integer() else code),
                    "code_numeric": code,
                    "label": "manual_fallback",
                    "source": "manual_fallback",
                }
            )
    niu_manual_audit = pd.DataFrame(niu_manual_rows)
    niu_audit = pd.concat([niu_stata_audit, niu_codebook_audit, niu_manual_audit], ignore_index=True)
    if not niu_audit.empty:
        niu_audit = niu_audit.drop_duplicates(["variable", "code_numeric", "source"]).sort_values(
            ["variable", "code_numeric", "source"]
        )

    # 3. Stream source rows through the documented recodes and transformations.
    stream = pd.read_stata(
        stata_path,
        convert_categoricals=False,
        iterator=True,
        columns=[c.lower() for c in source_columns],
    )

    chunksize = choose_chunk_size(len(KEEP_COLS), args.chunksize)
    writer: pq.ParquetWriter | None = None
    if args.out_data.exists():
        args.out_data.unlink()

    row_count = 0
    chunk_idx = 0
    year_counts: Counter = Counter()
    sample_counts: Counter = Counter()
    year_sample_counts: Counter = Counter()
    niu_recode_counts: Counter = Counter()
    structural_recode_counts: Counter = Counter()
    release_counter: Counter[tuple[str, str]] = Counter()

    while True:
        if args.max_chunks is not None and chunk_idx >= args.max_chunks:
            break
        try:
            chunk = stream.read(chunksize)
        except StopIteration:
            break
        if chunk is None or chunk.empty:
            break

        chunk_idx += 1
        chunk.columns = [c.upper() for c in chunk.columns]
        # Retain the cleaned schema without requiring obsolete input columns.
        chunk["CNTYGP97"] = np.nan
        chunk["CNTYGP98"] = np.nan
        chunk = chunk.loc[chunk["SAMPLE"].isin(SAMPLE_ORDER), KEEP_COLS].copy()
        if chunk.empty:
            continue

        structural_missing_count = enforce_structural_missing(chunk, availability, structural_recode_counts)
        apply_niu_recode(chunk, niu_map, niu_recode_counts)
        # Preserve the released schema: missing values in earlier samples made
        # these columns float64 before the extract was restricted to 1990/2000.
        for variable in ["METAREA", "METAREAD", "WKSWORK1", "PUMA"]:
            chunk[variable] = chunk[variable].astype("float64")
        chunk = add_stata_label_columns(chunk, variable_labels, CENSUS_LABELLED_CATEGORICAL_VARS)
        chunk = add_derived_columns(chunk)
        chunk["STRUCTURAL_MISSING_COUNT"] = structural_missing_count
        release_assignment = assign_census_release_dates(chunk, release_crosswalk)
        chunk["release_date"] = release_assignment["release_date"]
        chunk["release_date_source"] = release_assignment["release_date_source"]
        add_release_assignment_to_counter(release_counter, release_assignment)

        label_cols = [f"{variable}_label" for variable in CENSUS_LABELLED_CATEGORICAL_VARS]
        final_cols = KEEP_COLS + label_cols + [
            "release_date",
            "release_date_source",
            "FORM_1970",
            "IS_1970_FORM1",
            "IS_1970_FORM2",
            "PERWT_ANALYSIS",
            "HHWT_ANALYSIS",
            "WEIGHT_REGIME",
            "EDUC_DEGREE_ERA",
            "HISPAN_IMPUTED_ERA",
            "MULTIRACE_ERA",
            "RACE_MULTIPLE_MARKED",
            "STATEFIP_UNIDENTIFIED",
            "PERWT_ZERO_FLAG",
        ] + [f"{v}_TOPCODE_KNOWN" for v in REAL_INCOME_VARS] + [
            "CORE_MISSING_BURDEN",
            "STRUCTURAL_MISSING_COUNT",
        ] + [f"{v}_REAL99" for v in REAL_INCOME_VARS]
        chunk = chunk[final_cols]

        year = pd.to_numeric(chunk["YEAR"], errors="coerce")
        sample = pd.to_numeric(chunk["SAMPLE"], errors="coerce")
        yc = year.dropna().astype(int).value_counts()
        sc = sample.dropna().astype(int).value_counts()
        for y, n in yc.items():
            year_counts[int(y)] += int(n)
        for s, n in sc.items():
            sample_counts[int(s)] += int(n)
        ys = (
            pd.DataFrame({"YEAR": year, "SAMPLE": sample})
            .dropna()
            .assign(YEAR=lambda d: d["YEAR"].astype(int), SAMPLE=lambda d: d["SAMPLE"].astype(int))
            .groupby(["YEAR", "SAMPLE"], dropna=False)
            .size()
        )
        for (y, s), n in ys.items():
            year_sample_counts[(int(y), int(s))] += int(n)

        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(args.out_data, table.schema, compression="snappy")
        else:
            table = table.cast(writer.schema, safe=False)
        writer.write_table(table)

        row_count += len(chunk)
        print(f"chunk={chunk_idx} wrote_rows={len(chunk):,} total_rows={row_count:,}", flush=True)

    if writer is not None:
        writer.close()
    stream.close()
    reader.close()
    scratch.cleanup()

    if row_count == 0:
        raise SystemExit("Census build produced zero rows.")

    # 4. Record recode counts, coverage and source provenance.
    niu_audit.to_csv(args.out_dir / "niu_code_map.csv", index=False)
    pd.DataFrame(
        [{"variable": var, "n_recoded": int(n)} for var, n in sorted(niu_recode_counts.items())]
    ).to_csv(args.out_dir / "niu_recode_counts.csv", index=False)
    pd.DataFrame(
        [{"variable": var, "n_forced_missing": int(n)} for var, n in sorted(structural_recode_counts.items())]
    ).to_csv(args.out_dir / "structural_missing_recode_counts.csv", index=False)

    pd.DataFrame(
        [{"year": int(y), "rows": int(n)} for y, n in sorted(year_counts.items())]
    ).to_csv(args.out_dir / "year_counts.csv", index=False)
    pd.DataFrame(
        [{"sample": int(s), "rows": int(n)} for s, n in sorted(sample_counts.items())]
    ).to_csv(args.out_dir / "sample_counts.csv", index=False)
    pd.DataFrame(
        [
            {"year": int(y), "sample": int(s), "rows": int(n)}
            for (y, s), n in sorted(year_sample_counts.items())
        ]
    ).to_csv(args.out_dir / "year_sample_counts.csv", index=False)

    availability_rows = []
    for var in sorted(set(KEEP_COLS)):
        row = {"variable": var}
        for sample in SAMPLE_ORDER:
            row[f"avail_{sample}"] = int(availability.get(var, {}).get(sample, False))
        availability_rows.append(row)
    pd.DataFrame(availability_rows).to_csv(args.out_dir / "variable_availability.csv", index=False)

    build_summary = pd.DataFrame(
        [
            {
                "raw_dta_path": str(args.raw_dta),
                "out_data_path": str(args.out_data),
                "rows": int(row_count),
                "cols": int(len(final_cols)),
                "chunksize": int(chunksize),
                "n_chunks": int(chunk_idx),
                "build_utc": datetime.now(timezone.utc).isoformat(),
            }
        ]
    )
    build_summary.to_csv(args.out_dir / "build_summary.csv", index=False)
    with exclusive_lock(RELEASE_DATE_COVERAGE_PATH):
        update_release_date_coverage("Census", coverage_rows_from_counter("Census", release_counter))

    censor_meta = INCOME_CENSORING_RULE_DF.copy()
    censor_meta["exact_identification_from_public_value"] = np.where(
        censor_meta["flag_rule"].eq("eq"),
        1,
        0,
    )
    censor_meta["note"] = np.where(
        censor_meta["flag_rule"].eq("eq"),
        "Public-use values at the listed threshold identify literal top-coded values.",
        "Public-use values above the listed threshold are replacement values; the original uncensored amount is not recoverable.",
    )
    censor_meta.to_csv(args.out_dir / "income_censoring_metadata.csv", index=False)

    provenance = {
        "source": "IPUMS USA local Stata extract",
        "extract_id": None,
        "extract_id_note": "Original IPUMS extract request ID is not recoverable from the local raw files.",
        "raw_dta_path": str(args.raw_dta),
        "cleaned_parquet_path": str(args.out_data),
        "file_type": "rectangular",
        "case_selection": "No",
        "samples": SAMPLE_ORDER,
        "sample_labels": {
            "199002": "1990 1%",
            "200007": "2000 1%",
        },
        "kept_variables": KEEP_COLS,
        "requested_quality_flags_for_future_extract": QUALITY_FLAGS_REQUESTED,
        "sha256": {
            str(args.raw_dta): sha256_file(args.raw_dta),
            str(args.out_data): sha256_file(args.out_data),
        },
        "build_utc": build_summary.loc[0, "build_utc"],
    }
    (args.out_dir / "extract_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Census build complete.")
    print(f"rows={row_count:,}")
    print(f"output={args.out_data}")


if __name__ == "__main__":
    main()
