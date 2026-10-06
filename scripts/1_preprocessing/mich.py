#!/usr/bin/env python
"""Build cleaned Michigan Survey of Consumers microdata with panel links and metadata sidecars."""

from __future__ import annotations

import sys
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

RAW_DIR = Path("data/raw/micro/mich")
INTERMEDIATE_DIR = Path("data/intermediate")
OUT_DIR = Path("output/preprocessing/mich/build")
RELEASE_DATE_CROSSWALK_PATH = Path("data/raw/release_dates/release_mich.csv")
RELEASE_DATE_COVERAGE_PATH = Path("output/preprocessing/release_dates/metadata_coverage.csv")

RAW_DATA_PATH = RAW_DIR / "mich.csv"
MAPPING_DIR = Path(__file__).resolve().parent / "mappings"

OUT_PARQUET_PATH = INTERMEDIATE_DIR / "mich.parquet"
OUT_CONCORDANCE_PATH = OUT_DIR / "variable_concordance.csv"
OUT_MISSING_RULE_AUDIT_PATH = OUT_DIR / "missing_rule_application.csv"
OUT_PANEL_LINKAGE_PATH = OUT_DIR / "panel_linkage.csv"
OUT_BUILD_SUMMARY_PATH = OUT_DIR / "build_summary.csv"
OUT_VARIABLE_DICTIONARY_CSV_PATH = OUT_DIR / "variable_dictionary_thematic.csv"

PIPELINE_VERSION = "1.0.0"

SOURCE_LABELS = {
    1: "landline_rdd_interview",
    2: "landline_rdd_reinterview",
    3: "cell_rdd_interview",
    4: "cell_rdd_reinterview",
    5: "cell_rdd_second_reinterview",
    6: "fresh_abs_web_interview",
    7: "first_abs_web_reinterview",
    8: "second_abs_web_reinterview",
}

PANEL_WAVE_FROM_SAMPLE = {
    1: 1,
    2: 2,
    3: 1,
    4: 2,
    5: 3,
    6: 1,
    7: 2,
    8: 3,
}

MODE_BY_METHOD = {
    1: "phone",
    2: "web",
    3: "phone",
    4: "in_person",
}

HEADLINE_COMPONENTS = ["PAGO", "PEXP", "BUS12", "BUS5", "DUR"]

MICHIGAN_LABELLED_CATEGORICAL_VARS = [
    "SEX",
    "REGION",
    "MARRY",
    "EDUC",
    "PAGO",
    "BAGO",
    "GOVT",
    "DUR",
    "HOM",
    "CAR",
    "SHOM",
    "BUS12",
    "PEXP",
    "INEXQ1",
]
MICHIGAN_EXTRA_LABEL_MISSING_CODES: dict[str, set[int]] = {}


def categorical_labels_from_source_codes(
    series: pd.Series,
    label_map: dict[int, str],
    variable: str,
    extra_missing_codes: set[int] | None = None,
) -> pd.Categorical:
    if not label_map:
        raise RuntimeError(f"{variable} has no parsed Stata value labels.")

    numeric = pd.to_numeric(series, errors="coerce")
    source_nonmissing = series.notna()
    nonnumeric = source_nonmissing & numeric.isna()
    rounded = numeric.round()
    noninteger = numeric.notna() & ((numeric - rounded).abs() > 1e-9)
    codes = rounded.astype("Int64")
    missing_codes = set(extra_missing_codes or set())
    unmapped = numeric.notna() & ~codes.isin(set(label_map)) & ~codes.isin(missing_codes)
    bad = nonnumeric | noninteger | unmapped
    if bad.any():
        examples = sorted(series.loc[bad].dropna().astype(str).unique().tolist())[:20]
        raise RuntimeError(f"{variable} has nonmissing codes without parsed Stata labels: {examples}")

    ordered_labels = list(dict.fromkeys(label_map[key] for key in sorted(label_map)))
    labels = codes.map(label_map)
    if missing_codes:
        labels.loc[codes.isin(missing_codes)] = pd.NA
    return pd.Categorical(labels, categories=ordered_labels)


def add_labelled_categorical_columns(
    df: pd.DataFrame,
    var_meta: dict[str, dict[str, object]],
    label_maps: dict[str, dict[int, str]],
    variables: list[str],
) -> pd.DataFrame:
    out = df.copy()
    for variable in variables:
        if variable not in out.columns:
            raise RuntimeError(f"Cannot label missing Michigan variable: {variable}")
        label_set = str(var_meta.get(variable, {}).get("label_set", "")).strip()
        if not label_set:
            raise RuntimeError(f"{variable} has no Stata label set in read_mich.do.")
        label_map = label_maps.get(label_set, {})
        out[f"{variable}_label"] = categorical_labels_from_source_codes(
            out[variable],
            label_map,
            variable,
            MICHIGAN_EXTRA_LABEL_MISSING_CODES.get(variable),
        )
    return out


def apply_missing_rules(df: pd.DataFrame, rules: dict[str, list[tuple[str, str, float]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    audit_rows: list[dict[str, object]] = []

    for target_var, target_rules in rules.items():
        if target_var not in out.columns:
            audit_rows.append(
                {
                    "target_variable": target_var,
                    "rule_lhs": "",
                    "rule_operator": "",
                    "rule_rhs": np.nan,
                    "applied": 0,
                    "rows_set_missing": 0,
                    "missing_before": np.nan,
                    "missing_after": np.nan,
                }
            )
            continue

        missing_before = int(out[target_var].isna().sum())

        for lhs_var, op, rhs in target_rules:
            if lhs_var not in out.columns:
                audit_rows.append(
                    {
                        "target_variable": target_var,
                        "rule_lhs": lhs_var,
                        "rule_operator": op,
                        "rule_rhs": rhs,
                        "applied": 0,
                        "rows_set_missing": 0,
                        "missing_before": missing_before,
                        "missing_after": int(out[target_var].isna().sum()),
                    }
                )
                continue

            lhs = out[lhs_var]
            if op == "==":
                mask = lhs == rhs
            elif op == ">=":
                mask = lhs >= rhs
            elif op == "<=":
                mask = lhs <= rhs
            elif op == ">":
                mask = lhs > rhs
            elif op == "<":
                mask = lhs < rhs
            else:
                mask = pd.Series(False, index=out.index)

            rows_set_missing = int((mask & out[target_var].notna()).sum())
            out.loc[mask, target_var] = np.nan

            audit_rows.append(
                {
                    "target_variable": target_var,
                    "rule_lhs": lhs_var,
                    "rule_operator": op,
                    "rule_rhs": rhs,
                    "applied": 1,
                    "rows_set_missing": rows_set_missing,
                    "missing_before": missing_before,
                    "missing_after": int(out[target_var].isna().sum()),
                }
            )

    return out, pd.DataFrame(audit_rows)


def month_index(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    year = (numeric // 100).astype("Float64")
    month = (numeric % 100).astype("Float64")
    return year * 12 + month


def assign_methodology_regime(yyyymm: pd.Series) -> pd.Series:
    ym = pd.to_numeric(yyyymm, errors="coerce")
    conds = [
        (ym >= 197801) & (ym <= 199309),
        (ym >= 199310) & (ym <= 201206),
        (ym >= 201207) & (ym <= 201412),
        (ym >= 201501) & (ym <= 202403),
        (ym >= 202404) & (ym <= 202406),
        (ym >= 202407),
    ]
    choices = [
        "landline_rdd_waksberg",
        "landline_rdd_list_assisted",
        "dual_frame_cell_plus_landline",
        "cell_phone_rdd_only",
        "transition_phone_web",
        "web_abs_only",
    ]
    return pd.Series(np.select(conds, choices, default="unknown"), index=yyyymm.index, dtype="string")


def derive_panel_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    key = out[["YYYYMM", "ID", "CASEID"]].copy()
    for col in ["YYYYMM", "ID", "CASEID"]:
        key[col] = pd.to_numeric(key[col], errors="coerce").astype("Int64")
    key = key.dropna(subset=["YYYYMM", "ID", "CASEID"]).drop_duplicates(["YYYYMM", "ID"])

    links = out[["CASEID", "YYYYMM", "DATEPR", "IDPREV"]].copy()
    for col in ["CASEID", "YYYYMM", "DATEPR", "IDPREV"]:
        links[col] = pd.to_numeric(links[col], errors="coerce").astype("Int64")

    key_prev = key.rename(columns={"CASEID": "prev_caseid", "YYYYMM": "DATEPR", "ID": "IDPREV"})
    links = links.merge(key_prev, on=["DATEPR", "IDPREV"], how="left")

    links["panel_link_expected"] = (links["DATEPR"].notna() & links["IDPREV"].notna()).astype("Int64")
    links["panel_link_ok"] = (links["panel_link_expected"].eq(1) & links["prev_caseid"].notna()).astype("Int64")
    links["months_since_prev"] = month_index(links["YYYYMM"]) - month_index(links["DATEPR"])

    out = out.merge(
        links[["CASEID", "prev_caseid", "panel_link_expected", "panel_link_ok", "months_since_prev"]],
        on="CASEID",
        how="left",
    )

    # Build root panel id and wave depth from linked predecessor chains.
    case_ids = pd.to_numeric(out["CASEID"], errors="coerce").astype("Int64")
    prev_caseids = pd.to_numeric(out["prev_caseid"], errors="coerce").astype("Int64")

    prev_map = {
        int(case): int(prev)
        for case, prev, ok in zip(case_ids, prev_caseids, out["panel_link_ok"], strict=False)
        if pd.notna(case) and pd.notna(prev) and ok == 1
    }

    memo_root: dict[int, int] = {}
    memo_wave: dict[int, int] = {}
    cycle_cases: set[int] = set()

    def resolve(case: int) -> tuple[int, int, int]:
        if case in memo_root:
            return memo_root[case], memo_wave[case], 0

        chain: list[int] = []
        seen: set[int] = set()
        cur = case

        while True:
            if cur in memo_root:
                root = memo_root[cur]
                depth = memo_wave[cur] + 1
                for node in reversed(chain):
                    memo_root[node] = root
                    memo_wave[node] = depth
                    depth += 1
                return memo_root[case], memo_wave[case], 0

            if cur in seen:
                cycle_cases.update(chain)
                for node in chain:
                    memo_root[node] = node
                    memo_wave[node] = 1
                return memo_root[case], memo_wave[case], 1

            seen.add(cur)
            chain.append(cur)

            prev = prev_map.get(cur)
            if prev is None:
                root = cur
                depth = 1
                for node in reversed(chain):
                    memo_root[node] = root
                    memo_wave[node] = depth
                    depth += 1
                return memo_root[case], memo_wave[case], 0

            cur = prev

    unique_cases = [int(x) for x in case_ids.dropna().unique().tolist()]
    cycle_flag_map: dict[int, int] = {}
    for case in unique_cases:
        _, _, has_cycle = resolve(case)
        cycle_flag_map[case] = has_cycle

    out["panel_id"] = case_ids.map(memo_root).astype("Int64")
    out["panel_wave"] = case_ids.map(memo_wave).astype("Int64")
    out["panel_chain_cycle_flag"] = case_ids.map(cycle_flag_map).fillna(0).astype("Int64")

    sample_num = pd.to_numeric(out["SAMPLE"], errors="coerce").astype("Int64")
    out["panel_wave_from_sample"] = sample_num.map(PANEL_WAVE_FROM_SAMPLE).astype("Int64")
    out["panel_wave_consistent_with_sample"] = (
        out["panel_wave"].notna()
        & out["panel_wave_from_sample"].notna()
        & (out["panel_wave"] == out["panel_wave_from_sample"])
    ).astype("Int64")

    return out


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    yyyymm = pd.to_numeric(out["YYYYMM"], errors="coerce")
    out["year"] = (yyyymm // 100).astype("Int64")
    out["month"] = (yyyymm % 100).astype("Int64")
    out["quarter"] = (((out["month"] - 1) // 3) + 1).astype("Int64")

    out["survey_date"] = pd.to_datetime(
        out["year"].astype("string") + "-" + out["month"].astype("string").str.zfill(2) + "-01",
        errors="coerce",
    )
    out["year_month"] = out["survey_date"].dt.strftime("%Y-%m")

    out["methodology_regime"] = assign_methodology_regime(out["YYYYMM"])

    method_num = pd.to_numeric(out["METHOD"], errors="coerce").astype("Int64")
    mode_by_method = method_num.map(MODE_BY_METHOD).fillna("unknown")

    date_based_mode = np.where(
        yyyymm < 202404,
        "phone",
        np.where(yyyymm > 202406, "web", "transition"),
    )

    out["mode_by_method"] = mode_by_method.astype("string")
    out["mode_effective"] = np.where(
        out["methodology_regime"] == "transition_phone_web",
        out["mode_by_method"],
        date_based_mode,
    )
    out["mode_effective"] = pd.Series(out["mode_effective"], index=out.index, dtype="string")
    out["web_respondent"] = (out["mode_effective"] == "web").astype("Int64")

    sample_num = pd.to_numeric(out["SAMPLE"], errors="coerce").astype("Int64")
    out["sample_type_label"] = sample_num.map(SOURCE_LABELS).astype("string")

    # Index-construction recodes and diagnostics.
    out["BUS12_rep"] = out["BUS12"].replace({2: 1, 4: 5})
    out["BUS5_rep"] = out["BUS5"].replace({2: 1, 4: 5})

    out["PAGO_valid_rep"] = out["PAGO"].isin([1, 3, 5]).astype("Int64")
    out["PEXP_valid_rep"] = out["PEXP"].isin([1, 3, 5]).astype("Int64")
    out["BUS12_valid_rep"] = out["BUS12_rep"].isin([1, 3, 5]).astype("Int64")
    out["BUS5_valid_rep"] = out["BUS5_rep"].isin([1, 3, 5]).astype("Int64")
    out["DUR_valid_rep"] = out["DUR"].isin([1, 3, 5]).astype("Int64")

    for q in ["PAGO", "PEXP", "DUR"]:
        out[f"{q}_dk_na"] = out[f"{q}_raw"].isin([8, 9]).astype("Int64")
    out["BUS12_dk_na"] = out["BUS12_raw"].isin([8, 9]).astype("Int64")
    out["BUS5_dk_na"] = out["BUS5_raw"].isin([98, 99]).astype("Int64")

    out["PX1_outlier_abs_gt25"] = (out["PX1"].abs() > 25).fillna(False).astype("Int64")
    out["PX5_outlier_abs_gt25"] = (out["PX5"].abs() > 25).fillna(False).astype("Int64")

    out["income_topcoded_500k_flag"] = (out["INCOME"] == 500_000).fillna(False).astype("Int64")
    out["income_anomaly_gt_500k_flag"] = (out["INCOME"] > 500_000).fillna(False).astype("Int64")
    out["income_sentinel_999998_999999_flag"] = out["INCOME"].isin([999_998, 999_999]).astype("Int64")

    out["wt_positive"] = (out["WT"] > 0).fillna(False).astype("Int64")

    return out


def load_michigan_release_date_crosswalk() -> pd.DataFrame:
    required_columns = ["yyyymm", "release_date", "source", "source_url", "comments"]
    if not RELEASE_DATE_CROSSWALK_PATH.exists():
        raise SystemExit(f"Missing Michigan release-date crosswalk: {RELEASE_DATE_CROSSWALK_PATH}")

    crosswalk = pd.read_csv(
        RELEASE_DATE_CROSSWALK_PATH,
        dtype={"source": "string", "source_url": "string", "comments": "string"},
    )
    missing = [column for column in required_columns if column not in crosswalk.columns]
    if missing:
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} is missing columns: {missing}")

    crosswalk = crosswalk.loc[:, required_columns].copy()
    crosswalk["yyyymm"] = pd.to_numeric(crosswalk["yyyymm"], errors="coerce").astype("Int64")
    crosswalk["release_date"] = pd.to_datetime(crosswalk["release_date"], errors="coerce")
    if crosswalk["yyyymm"].isna().any():
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains missing or nonnumeric yyyymm values.")
    if crosswalk["yyyymm"].duplicated().any():
        duplicates = sorted(crosswalk.loc[crosswalk["yyyymm"].duplicated(), "yyyymm"].astype(int).unique())[:10]
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains duplicate yyyymm values: {duplicates}")
    if crosswalk["release_date"].isna().any():
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains missing release_date values.")
    for column in ["source", "source_url", "comments"]:
        values = crosswalk[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains missing {column} values.")
        crosswalk[column] = values

    crosswalk["release_date"] = crosswalk["release_date"].dt.strftime("%Y-%m-%d").astype("string")
    return crosswalk


def assign_michigan_release_dates(df: pd.DataFrame, release_crosswalk: pd.DataFrame) -> pd.DataFrame:
    yyyymm = pd.to_numeric(df["YYYYMM"], errors="coerce").astype("Int64")
    if yyyymm.isna().any():
        raise SystemExit("Michigan release-date assignment found missing YYYYMM values.")

    month = (yyyymm % 100).astype(int)
    bad_month = ~month.between(1, 12)
    if bad_month.any():
        examples = sorted(yyyymm.loc[bad_month].dropna().astype(int).unique())[:10]
        raise SystemExit(f"Michigan release-date assignment found invalid YYYYMM values: {examples}")

    lookup = release_crosswalk.set_index("yyyymm")
    release_date = yyyymm.map(lookup["release_date"]).astype("string")
    release_source = yyyymm.map(lookup["source"]).astype("string")

    if release_date.isna().any():
        examples = sorted(yyyymm.loc[release_date.isna()].dropna().astype(int).unique())[:10]
        raise SystemExit(f"Michigan release-date crosswalk does not cover YYYYMM values: {examples}")
    if release_source.isna().any() or release_source.str.strip().eq("").any():
        raise SystemExit("Michigan release-date assignment produced missing values.")

    return pd.DataFrame(
        {
            "release_date": release_date,
            "release_date_source": release_source,
        },
        index=df.index,
    )


def build_release_date_coverage(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["release_date_source", "release_date"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise SystemExit(f"Michigan release-date coverage is missing columns: {missing}")

    counts = df.groupby(["release_date_source", "release_date"], dropna=False).size().reset_index(name="rows")
    coverage = counts.rename(
        columns={
            "release_date_source": "release_rule_id",
            "release_date": "microdata_release_date",
        }
    )
    coverage.insert(0, "dataset", "Michigan")
    coverage["unmatched_rows"] = 0
    return coverage[["dataset", "release_rule_id", "microdata_release_date", "rows", "unmatched_rows"]]


def update_release_date_coverage(dataset: str, coverage: pd.DataFrame) -> None:
    RELEASE_DATE_COVERAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    columns = ["dataset", "release_rule_id", "microdata_release_date", "rows", "unmatched_rows"]
    missing = [column for column in columns if column not in coverage.columns]
    if missing:
        raise SystemExit(f"Michigan release-date coverage is missing columns: {missing}")

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


def write_parquet_with_metadata(df: pd.DataFrame, path: Path, metadata: dict[str, str]) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    md = dict(table.schema.metadata or {})
    for key, value in metadata.items():
        md[str(key).encode("utf-8")] = str(value).encode("utf-8")
    table = table.replace_schema_metadata(md)
    pq.write_table(table, path)


def build_concordance(
    output_columns: list[str],
    var_meta: dict[str, dict[str, object]],
    codebook_toc: dict[str, str],
    label_maps: dict[str, dict[int, str]],
    missing_rules: dict[str, list[tuple[str, str, float]]],
) -> pd.DataFrame:
    derived_descriptions = {
        "year": "Survey year derived from YYYYMM.",
        "month": "Survey month derived from YYYYMM.",
        "quarter": "Survey quarter derived from month.",
        "survey_date": "Canonical first-of-month survey date.",
        "release_date": "Row-level public microdata release date used for model-cutoff filtering.",
        "release_date_source": "Source rule for release_date.",
        "year_month": "Canonical year-month string (YYYY-MM).",
        "methodology_regime": "Date-based methodology regime classification.",
        "mode_by_method": "Interview mode mapped from METHOD field.",
        "mode_effective": "Final mode assignment combining date regime and METHOD.",
        "web_respondent": "Indicator for web interviews.",
        "sample_type_label": "Human-readable sample type from SAMPLE codes.",
        "BUS12_rep": "BUS12 collapsed for index replication (2->1, 4->5).",
        "BUS5_rep": "BUS5 collapsed for index replication (2->1, 4->5).",
        "PAGO_valid_rep": "Indicator that PAGO contributes to index replication.",
        "PEXP_valid_rep": "Indicator that PEXP contributes to index replication.",
        "BUS12_valid_rep": "Indicator that BUS12 contributes to index replication.",
        "BUS5_valid_rep": "Indicator that BUS5 contributes to index replication.",
        "DUR_valid_rep": "Indicator that DUR contributes to index replication.",
        "PAGO_dk_na": "Indicator for DK/NA on PAGO using raw coding.",
        "PEXP_dk_na": "Indicator for DK/NA on PEXP using raw coding.",
        "BUS12_dk_na": "Indicator for DK/NA on BUS12 using raw coding.",
        "BUS5_dk_na": "Indicator for DK/NA on BUS5 using raw coding.",
        "DUR_dk_na": "Indicator for DK/NA on DUR using raw coding.",
        "PX1_outlier_abs_gt25": "Indicator for |PX1| > 25.",
        "PX5_outlier_abs_gt25": "Indicator for |PX5| > 25.",
        "income_topcoded_500k_flag": "Indicator for INCOME topcoded at $500k.",
        "income_anomaly_gt_500k_flag": "Indicator for INCOME above $500k.",
        "income_sentinel_999998_999999_flag": "Indicator for sentinel income values 999998/999999.",
        "wt_positive": "Indicator for positive WT.",
        "prev_caseid": "Linked prior CASEID using (DATEPR, IDPREV).",
        "panel_link_expected": "Indicator where DATEPR and IDPREV suggest a recontact link.",
        "panel_link_ok": "Indicator where expected link successfully matched to prior CASEID.",
        "months_since_prev": "Month gap between current YYYYMM and DATEPR.",
        "panel_id": "Root CASEID for linked panel chain.",
        "panel_wave": "Wave number inferred from linked chain.",
        "panel_chain_cycle_flag": "Indicator if cycle detected in chain traversal.",
        "panel_wave_from_sample": "Wave implied by SAMPLE code.",
        "panel_wave_consistent_with_sample": "Indicator that inferred panel wave matches SAMPLE-implied wave.",
    }
    for variable in MICHIGAN_LABELLED_CATEGORICAL_VARS:
        derived_descriptions[f"{variable}_label"] = f"Value label for {variable} parsed from read_mich.do."

    rows: list[dict[str, object]] = []
    for col in output_columns:
        source_variable = col if col in var_meta else "derived"
        label_set = var_meta.get(col, {}).get("label_set", "")
        value_labels = label_maps.get(str(label_set), {}) if label_set else {}

        rules = missing_rules.get(col, [])
        rule_text = "; ".join(f"{lhs} {op} {rhs:g}" for lhs, op, rhs in rules)

        rows.append(
            {
                "output_variable": col,
                "source_variable": source_variable,
                "do_description": var_meta.get(col, {}).get("do_description", ""),
                "codebook_description": codebook_toc.get(col, ""),
                "derived_description": derived_descriptions.get(col, ""),
                "label_set": label_set,
                "n_value_labels": len(value_labels),
                "value_label_keys": ";".join(str(x) for x in sorted(value_labels.keys())),
                "missing_rules": rule_text,
            }
        )

    return pd.DataFrame(rows)


def mich_variable_group(var: str) -> str:
    if var in {"CASEID", "YYYYMM", "YYYYQ", "YYYY", "ID", "year", "month", "quarter", "survey_date", "year_month"}:
        return "Identifiers and calendar variables"
    if var in {"DATEPR", "IDPREV", "prev_caseid", "months_since_prev"}:
        return "Recontact references and panel timing"
    if var.startswith("panel_"):
        return "Panel linkage and chain reconstruction"
    if var in {
        "SAMPLE",
        "METHOD",
        "WT",
        "sample_type_label",
        "methodology_regime",
        "mode_by_method",
        "mode_effective",
        "web_respondent",
        "wt_positive",
    }:
        return "Sample design, mode, and weights"
    if var.endswith("_raw"):
        return "Raw snapshots preserved before recoding"
    if (
        var in {"ICS", "ICC", "ICE", "PAGO", "PEXP", "PAGOR1", "PAGOR2", "PAGO5", "PEXP5", "BUS12", "BUS5", "DUR", "DURRN1", "DURRN2"}
        or var.endswith(("_rep", "_valid_rep", "_dk_na"))
    ):
        return "Headline sentiment indexes and personal economic assessments"
    if var in {"BAGO", "BEXP", "NEWS1", "NEWS2", "UNEMP", "GOVT", "RATEX"}:
        return "Business conditions and policy expectations"
    if var.startswith("PX") or "infl" in var.lower():
        return "Inflation expectations"
    if var.startswith(("INEX", "PINC", "PJOB", "PSSA", "PCRY", "PSTK")) or var == "RINC":
        return "Income and probabilistic expectations"
    if var.startswith(("HOM", "SHOM", "CAR", "VEH", "GAS", "INV", "STL", "HTL")) or var in {"HOMEOWN", "HOMEAMT", "HOMEQFM", "HOMEVAL"}:
        return "Housing, vehicles, and assets"
    if var in {"AGE", "SEX", "MARRY", "NUMKID", "EDUC", "REGION", "HHINC", "INCOME"}:
        return "Demographics and household economics"
    if var.startswith("POL"):
        return "Politics and partisan detail"
    if "income" in var.lower() or var in {"INCOME_raw", "HHINC_raw"}:
        return "Income recodes and flags"
    return "Other source variables"


def clean_description(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return "No source description available."
    if text.upper() == text:
        text = text.title()
    replacements = {
        "B/W": "better/worse",
        "U/D": "up/down",
        " Id": " ID",
        " Yr": " year",
        " 1Yr": " 1 year",
        " 5Yrs": " 5 years",
        "3Rd": "3rd",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_variable_dictionary(concordance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in concordance.itertuples(index=False):
        description = getattr(row, "derived_description")
        if pd.isna(description) or str(description).strip() == "":
            description = getattr(row, "codebook_description")
        if pd.isna(description) or str(description).strip() == "":
            description = getattr(row, "do_description")
        rows.append(
            {
                "group": mich_variable_group(str(row.output_variable)),
                "variable": str(row.output_variable),
                "description": clean_description(description),
            }
        )

    dictionary = pd.DataFrame(rows)
    dictionary.to_csv(OUT_VARIABLE_DICTIONARY_CSV_PATH, index=False)

    return dictionary


def main() -> None:
    validate_source_inputs("mich")
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    required_paths = [RAW_DATA_PATH]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required raw inputs: {missing}")

    # Freeze the official Stata labels/recode rules and codebook descriptions.
    variables = pd.read_csv(MAPPING_DIR / "mich_variables.csv", keep_default_na=False)
    var_meta = {row["variable"]: {key: row[key] for key in ["do_description", "label_set"]}
                for row in variables.to_dict("records")}
    codebook_toc = dict(zip(variables["variable"], variables["codebook_description"], strict=True))
    labels = pd.read_csv(MAPPING_DIR / "mich_labels.csv", keep_default_na=False)
    label_maps = {label_set: dict(zip(rows["code"], rows["label"], strict=True))
                  for label_set, rows in labels.groupby("label_set", sort=False)}
    rules = pd.read_csv(MAPPING_DIR / "mich_missing_rules.csv", keep_default_na=False)
    missing_rules = {variable: list(rows[["condition_variable", "operator", "value"]].itertuples(index=False, name=None))
                     for variable, rows in rules.groupby("variable", sort=False)}

    raw_df = pd.read_csv(RAW_DATA_PATH, dtype=str, low_memory=False)
    csv_columns = raw_df.columns.tolist()

    # Normalize blanks and convert to numeric where possible.
    for col in raw_df.columns:
        raw_df[col] = raw_df[col].astype("string").str.strip()
        raw_df[col] = raw_df[col].replace("", pd.NA)
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

    clean_df = raw_df.copy()

    # Preserve raw versions for diagnostics before missing recodes.
    for col in [
        "PAGO",
        "PEXP",
        "BUS12",
        "BUS5",
        "DUR",
        "PX1",
        "PX5",
        "INCOME",
        "POLAFF",
        "METHOD",
        "SAMPLE",
        "WT",
    ]:
        if col in clean_df.columns:
            clean_df[f"{col}_raw"] = raw_df[col]

    clean_df, missing_audit = apply_missing_rules(clean_df, missing_rules)
    clean_df = add_labelled_categorical_columns(
        clean_df,
        var_meta,
        label_maps,
        MICHIGAN_LABELLED_CATEGORICAL_VARS,
    )
    clean_df = add_derived_columns(clean_df)
    clean_df = derive_panel_columns(clean_df)
    release_crosswalk = load_michigan_release_date_crosswalk()
    release_assignment = assign_michigan_release_dates(clean_df, release_crosswalk)
    clean_df["release_date"] = release_assignment["release_date"]
    clean_df["release_date_source"] = release_assignment["release_date_source"]
    with exclusive_lock(RELEASE_DATE_COVERAGE_PATH):
        update_release_date_coverage("Michigan", build_release_date_coverage(clean_df))

    # Order output with source columns first, then raw diagnostic snapshots, then derived columns.
    base_cols = csv_columns.copy()
    raw_diag_cols = sorted([c for c in clean_df.columns if c.endswith("_raw")])
    derived_cols = [c for c in clean_df.columns if c not in set(base_cols + raw_diag_cols)]
    clean_df = clean_df[base_cols + raw_diag_cols + derived_cols].copy()

    concordance = build_concordance(clean_df.columns.tolist(), var_meta, codebook_toc, label_maps, missing_rules)

    panel_linkage = clean_df[
        [
            "CASEID",
            "YYYYMM",
            "ID",
            "DATEPR",
            "IDPREV",
            "prev_caseid",
            "months_since_prev",
            "panel_link_expected",
            "panel_link_ok",
            "panel_id",
            "panel_wave",
            "panel_wave_from_sample",
            "panel_wave_consistent_with_sample",
            "panel_chain_cycle_flag",
        ]
    ].copy()

    build_summary = pd.DataFrame(
        [
            {
                "rows": len(clean_df),
                "columns": len(clean_df.columns),
                "min_yyyymm": int(pd.to_numeric(clean_df["YYYYMM"], errors="coerce").min()),
                "max_yyyymm": int(pd.to_numeric(clean_df["YYYYMM"], errors="coerce").max()),
                "unique_caseid": int(pd.to_numeric(clean_df["CASEID"], errors="coerce").nunique()),
                "duplicate_caseid": int(pd.to_numeric(clean_df["CASEID"], errors="coerce").duplicated().sum()),
                "panel_link_expected": int(clean_df["panel_link_expected"].sum()),
                "panel_link_ok": int(clean_df["panel_link_ok"].sum()),
                "panel_link_match_rate": float(
                    clean_df.loc[clean_df["panel_link_expected"].eq(1), "panel_link_ok"].mean()
                )
                if int(clean_df["panel_link_expected"].sum()) > 0
                else np.nan,
                "pipeline_version": PIPELINE_VERSION,
                "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        ]
    )

    source_hashes = {
        "mich_csv_sha256": sha256_file(RAW_DATA_PATH),
    }

    parquet_metadata = {
        "pipeline_name": "scripts/1_preprocessing/mich.py",
        "pipeline_version": PIPELINE_VERSION,
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "validation_status": "not_validated",
        "validation_ics_mae": "",
        **source_hashes,
    }

    write_parquet_with_metadata(clean_df, OUT_PARQUET_PATH, parquet_metadata)

    concordance.to_csv(OUT_CONCORDANCE_PATH, index=False)
    write_variable_dictionary(concordance)
    missing_audit.to_csv(OUT_MISSING_RULE_AUDIT_PATH, index=False)
    panel_linkage.to_csv(OUT_PANEL_LINKAGE_PATH, index=False)
    build_summary.to_csv(OUT_BUILD_SUMMARY_PATH, index=False)

    print(f"Wrote cleaned parquet: {OUT_PARQUET_PATH}")
    print(f"Wrote variable concordance: {OUT_CONCORDANCE_PATH}")
    print(f"Wrote thematic variable dictionary: {OUT_VARIABLE_DICTIONARY_CSV_PATH}")
    print(f"Wrote panel linkage extract: {OUT_PANEL_LINKAGE_PATH}")
    print("Build summary:")
    print(build_summary.to_string(index=False))


if __name__ == "__main__":
    main()
