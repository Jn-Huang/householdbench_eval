"""Shared CPS mechanics for HouseholdBench task table scripts."""

from __future__ import annotations

import hashlib
import math
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from scripts.utils.macro_context import CORE_MACRO_COLUMNS, load_macro_context


ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def baseline_macro_key(obs: dict[str, object]) -> int:
    return int(obs["year"]) * 4 + ((int(obs["month"]) - 1) // 3 + 1)


def load_macro_context_map() -> dict[int, dict[str, float | int]]:
    context = load_macro_context(CORE_MACRO_COLUMNS)
    return {
        int(row["eligible_origin_q_index"]): {
            column: row[column]
            for column in CORE_MACRO_COLUMNS
        }
        for row in context.to_dict("records")
    }


def month_index(year: int, month: int) -> int:
    return int(year) * 12 + int(month)


def clean_release_date_value(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if ISO_DATE_RE.match(stripped):
            return stripped
        if len(stripped) >= 10 and ISO_DATE_RE.match(stripped[:10]):
            return stripped[:10]
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def clean_release_date_source(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def safe_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError, OverflowError):
        return None


def safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if math.isnan(out):
        return None
    return out


def safe_id(value: Any) -> int | None:
    out = safe_float(value)
    if out is None:
        return None
    rounded = int(round(out))
    if abs(out - rounded) > 1e-6:
        return None
    return rounded


def bounded_int(value: Any, *, min_value: int, max_value: int) -> int | None:
    out = safe_int(value)
    if out is None or out < min_value or out > max_value:
        return None
    return out


def bounded_float(value: Any, *, min_value: float, max_value: float) -> float | None:
    out = safe_float(value)
    if out is None or out < min_value or out > max_value:
        return None
    return out


def broad_status_from_empstat(value: Any) -> str | None:
    code = safe_int(value)
    if code in {10, 12}:
        return "employed"
    if code in {20, 21, 22}:
        return "unemployed"
    if code in {30, 31, 32, 33, 34, 35, 36}:
        return "not_in_labor_force"
    return None


def older_status_from_empstat_nilfact(empstat_value: Any, nilfact_value: Any) -> str | None:
    empstat = safe_int(empstat_value)
    nilfact = safe_int(nilfact_value)
    if empstat in {10, 12}:
        return "employed"
    if empstat in {20, 21, 22}:
        return "unemployed"
    if empstat == 36:
        return "nilf_retired"
    if empstat == 32 or nilfact == 1:
        return "nilf_disabled"
    if empstat in {30, 31, 33, 34, 35}:
        return "nilf_other"
    return None


def age_group(age: int | None) -> str | None:
    if age is None:
        return None
    if age < 25:
        return "16_24"
    if age < 35:
        return "25_34"
    if age < 45:
        return "35_44"
    if age < 55:
        return "45_54"
    if age < 65:
        return "55_64"
    return "65_plus"


def stable_row_id(task_id: str, row_id_prefix: str, *parts: Any) -> str:
    digest = hashlib.blake2b(
        "|".join([task_id, *[str(part) for part in parts]]).encode("utf-8"),
        digest_size=8,
    ).hexdigest()
    return f"{row_id_prefix}_{digest}"


def missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "nat", "<na>"}


def clean_label_value(value: Any) -> str | None:
    if missing(value):
        return None
    return str(value).strip()


def validate_table_row(row: dict[str, Any], required_columns: list[str], targets: list[str]) -> None:
    missing_columns = [column for column in required_columns if missing(row.get(column))]
    if missing_columns:
        raise RuntimeError(f"Missing required output values for {row.get('row_id')}: {missing_columns}")
    if row["target"] not in targets:
        raise RuntimeError(f"Unexpected target {row['target']!r} for {row.get('row_id')}.")


def guard_parquet_chronology(path: Path) -> None:
    pf = pq.ParquetFile(path)
    previous_last: tuple[int, int] | None = None
    for row_group in range(pf.num_row_groups):
        table = pf.read_row_group(row_group, columns=["YEAR", "MONTH"])
        years = table.column("YEAR").to_pylist()
        months = table.column("MONTH").to_pylist()
        if not years:
            continue
        first = (int(years[0]), int(months[0]))
        last = (int(years[-1]), int(months[-1]))
        if first > last:
            raise SystemExit(f"{path} row group {row_group} is internally out of chronological order.")
        if previous_last is not None and first < previous_last:
            raise SystemExit(f"{path} is not globally chronological by YEAR,MONTH.")
        previous_last = last


def check_columns(path: Path, required: list[str]) -> list[str]:
    names = set(pq.ParquetFile(path).schema.names)
    missing_columns = sorted(set(required) - names)
    if missing_columns:
        raise SystemExit(f"Missing required columns in {path}: {missing_columns}")
    return required


def prepare_month_dataframe(
    df: pd.DataFrame,
    *,
    preserve_rows: bool = False,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    cpsidp = pd.to_numeric(out["CPSIDP"], errors="coerce")
    cpsidp_round = cpsidp.round()
    valid_id = cpsidp.notna() & ((cpsidp - cpsidp_round).abs() < 1e-6)
    for column in ["YEAR", "MONTH", "MISH", "AGE", "EMPSTAT", "NILFACT"]:
        out[f"_{column.lower()}"] = pd.to_numeric(out[column], errors="coerce")
    valid = valid_id & out["_year"].notna() & out["_month"].notna() & out["_mish"].notna() & out["_empstat"].notna()
    if not preserve_rows:
        out = out.loc[valid].copy()
        if out.empty:
            return out
    out["_cpsidp"] = cpsidp_round.loc[out.index].astype(
        "Int64" if preserve_rows else "int64"
    )
    out["_year"] = out["_year"].round().astype(
        "Int64" if preserve_rows else "int64"
    )
    out["_month"] = out["_month"].round().astype(
        "Int64" if preserve_rows else "int64"
    )
    out["_mish"] = out["_mish"].round().astype(
        "Int64" if preserve_rows else "int64"
    )
    out["_age"] = out["_age"].round()
    out["_empstat"] = out["_empstat"].round().astype(
        "Int64" if preserve_rows else "int64"
    )
    out["_nilfact"] = out["_nilfact"].round()
    out["_month_index"] = out["_year"] * 12 + out["_month"]
    out["_status"] = None
    out.loc[out["_empstat"].isin([10, 12]), "_status"] = "employed"
    out.loc[out["_empstat"].isin([20, 21, 22]), "_status"] = "unemployed"
    out.loc[out["_empstat"].isin([30, 31, 32, 33, 34, 35, 36]), "_status"] = "not_in_labor_force"
    out["_older_status"] = None
    out.loc[out["_empstat"].isin([10, 12]), "_older_status"] = "employed"
    out.loc[out["_empstat"].isin([20, 21, 22]), "_older_status"] = "unemployed"
    out.loc[out["_empstat"].eq(36), "_older_status"] = "nilf_retired"
    out.loc[out["_empstat"].eq(32) | out["_nilfact"].eq(1), "_older_status"] = "nilf_disabled"
    out.loc[out["_empstat"].isin([30, 31, 33, 34, 35]), "_older_status"] = "nilf_other"
    duplicated = out.duplicated(subset=["_cpsidp"], keep=False)
    if preserve_rows:
        out["_valid_link_key"] = valid.loc[out.index]
        out["_unique_cpsidp"] = out["_valid_link_key"] & ~duplicated
        return out
    return out.loc[~duplicated].copy()


# Scanner batches of the forward iterator; a row group up to this size is one batch.
FORWARD_BATCH_ROWS = 500_000


def iter_month_dataframes(path: Path, columns: list[str]):
    dataset = ds.dataset(path, format="parquet")
    scanner = dataset.scanner(columns=columns, batch_size=FORWARD_BATCH_ROWS, use_threads=False)
    current_month_index: int | None = None
    current_pieces: list[pd.DataFrame] = []
    for batch in scanner.to_batches():
        df = batch.to_pandas()
        year = pd.to_numeric(df["YEAR"], errors="coerce")
        month = pd.to_numeric(df["MONTH"], errors="coerce")
        df["_sort_month_index"] = year * 12 + month
        df = df.dropna(subset=["_sort_month_index"])
        for month_idx, part in df.groupby("_sort_month_index", sort=True):
            idx = int(month_idx)
            part = part.drop(columns=["_sort_month_index"])
            if current_month_index is None:
                current_month_index = idx
                current_pieces = [part]
            elif idx != current_month_index:
                yield current_month_index, pd.concat(current_pieces, ignore_index=True)
                current_month_index = idx
                current_pieces = [part]
            else:
                current_pieces.append(part)
    if current_month_index is not None:
        yield current_month_index, pd.concat(current_pieces, ignore_index=True)


def iter_month_dataframes_reverse(path: Path, columns: list[str]):
    pf = pq.ParquetFile(path)
    current_month_index: int | None = None
    current_pieces: list[pd.DataFrame] = []
    for row_group in reversed(range(pf.num_row_groups)):
        df = pf.read_row_group(row_group, columns=columns).to_pandas()
        year = pd.to_numeric(df["YEAR"], errors="coerce")
        month = pd.to_numeric(df["MONTH"], errors="coerce")
        df["_sort_month_index"] = year * 12 + month
        df = df.dropna(subset=["_sort_month_index"])
        for month_idx in sorted(df["_sort_month_index"].unique(), reverse=True):
            idx = int(month_idx)
            part = df.loc[df["_sort_month_index"].eq(month_idx)].drop(columns=["_sort_month_index"])
            if current_month_index is None:
                current_month_index = idx
                current_pieces = [part]
            elif idx != current_month_index:
                yield current_month_index, pd.concat(current_pieces, ignore_index=True)
                current_month_index = idx
                current_pieces = [part]
            else:
                current_pieces.append(part)
    if current_month_index is not None:
        yield current_month_index, pd.concat(current_pieces, ignore_index=True)


def _month_indices(df: pd.DataFrame) -> pd.Series:
    year = pd.to_numeric(df["YEAR"], errors="coerce")
    month = pd.to_numeric(df["MONTH"], errors="coerce")
    return year * 12 + month


def month_row_groups(path: Path) -> dict[int, list[int]]:
    """Map each month index in a chronological parquet file to the row groups holding it.

    Months are those the month iterators yield, in file order; rows without YEAR or
    MONTH belong to none. iter_month_block converts one row group at a time, which matches
    the forward iterator only while no row group is larger than one of its batches.
    """
    pf = pq.ParquetFile(path)
    largest = max((pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)), default=0)
    if largest > FORWARD_BATCH_ROWS:
        raise ValueError(
            f"{path} has a row group of {largest:,} rows; rebuild it with --chunksize at most "
            f"{FORWARD_BATCH_ROWS:,}."
        )
    groups: dict[int, list[int]] = {}
    for row_group in range(pf.num_row_groups):
        indices = _month_indices(pf.read_row_group(row_group, columns=["YEAR", "MONTH"]).to_pandas())
        for month_idx in sorted(indices.dropna().unique()):
            groups.setdefault(int(month_idx), []).append(row_group)
    return groups


def iter_month_block(
    path: Path,
    columns: list[str],
    months: list[int],
    row_groups: dict[int, list[int]],
    *,
    reverse: bool = False,
):
    """Yield (month index, frame) for `months`, the frames the full-file iterators yield.

    With reverse=False each frame equals the one iter_month_dataframes yields for that
    month, and with reverse=True the one iter_month_dataframes_reverse yields. Each row
    group converts to pandas on its own, as in those iterators (a month that spans two
    row groups can differ in dtypes from one that does not), and the pieces join in the
    same order. Pass months in iteration order so that a row group shared by consecutive
    months is read once. `row_groups` comes from month_row_groups(path).
    """
    pf = pq.ParquetFile(path)
    cache: dict[int, tuple[pd.DataFrame, pd.Series]] = {}
    for month_idx in months:
        groups = row_groups[month_idx]
        for row_group in [group for group in cache if group not in groups]:
            del cache[row_group]
        pieces = []
        for row_group in reversed(groups) if reverse else groups:
            if row_group not in cache:
                df = pf.read_row_group(row_group, columns=columns).to_pandas()
                cache[row_group] = (df, _month_indices(df))
            df, indices = cache[row_group]
            pieces.append(df.loc[indices.eq(month_idx)])
        yield month_idx, pd.concat(pieces, ignore_index=True)


def month_blocks(months: list[int], size: int = 12) -> list[list[int]]:
    """Split months into consecutive blocks of at most `size` for worker processes."""
    return [months[start:start + size] for start in range(0, len(months), size)]


def normalize_basic_record(row: dict[str, Any], *, prefix: str = "") -> dict[str, Any] | None:
    if prefix:
        row = {
            key.removeprefix(prefix): value for key, value in row.items() if key.startswith(prefix)
        }
    year = safe_int(row.get("YEAR"))
    month = safe_int(row.get("MONTH"))
    cpsidp = safe_id(row.get("CPSIDP"))
    mish = safe_int(row.get("MISH"))
    if year is None or month is None or cpsidp is None or mish is None:
        return None
    empstat = safe_int(row.get("EMPSTAT"))
    nilfact = safe_int(row.get("NILFACT"))
    sex_code = safe_int(row.get("SEX"))
    race_code = safe_int(row.get("RACE"))
    dwjobsince = bounded_int(row.get("dwjobsince"), min_value=0, max_value=94)
    return {
        "cpsidp": cpsidp,
        "cpsidv": safe_id(row.get("CPSIDV")),
        "year": year,
        "month": month,
        "month_index": month_index(year, month),
        "mish": mish,
        "age": safe_int(row.get("AGE")),
        "empstat_code": empstat,
        "nilfact_code": nilfact,
        "empstat": clean_label_value(row.get("empstat")),
        "nilfact": clean_label_value(row.get("nilfact")),
        "status": broad_status_from_empstat(empstat),
        "older_status": older_status_from_empstat_nilfact(empstat, nilfact),
        "release_date": clean_release_date_value(row.get("release_date")),
        "release_date_source": clean_release_date_source(row.get("release_date_source")),
        "sex_code": sex_code,
        "race_code": race_code,
        "sex": clean_label_value(row.get("sex")),
        "race": clean_label_value(row.get("race")),
        "educ": clean_label_value(row.get("educ")),
        "marst": clean_label_value(row.get("marst")),
        "state": clean_label_value(row.get("state")),
        "region": clean_label_value(row.get("region")),
        "metro": clean_label_value(row.get("metro")),
        "relate": clean_label_value(row.get("relate")),
        "famsize": bounded_int(row.get("FAMSIZE"), min_value=1, max_value=99),
        "nchild": bounded_int(row.get("NCHILD"), min_value=0, max_value=99),
        "hispan": clean_label_value(row.get("hispan")),
        "nativity": clean_label_value(row.get("nativity")),
        "citizen": clean_label_value(row.get("citizen")),
        "vetstat": clean_label_value(row.get("vetstat")),
        "classwkr": clean_label_value(row.get("classwkr")),
        "wkstat": clean_label_value(row.get("wkstat")),
        "durunemp": bounded_int(row.get("DURUNEMP"), min_value=0, max_value=998),
        "wnftlook": clean_label_value(row.get("wnftlook")),
        "ahrsworkt": bounded_float(row.get("AHRSWORKT"), min_value=0, max_value=99),
        "uhrsworkt": bounded_float(row.get("UHRSWORKT"), min_value=0, max_value=99),
        "multjob": clean_label_value(row.get("multjob")),
        "union": clean_label_value(row.get("union")),
        "earnweek": bounded_float(row.get("EARNWEEK"), min_value=0, max_value=9999.0),
        "dwmove": clean_label_value(row.get("dwmove")),
        "dwresp": clean_label_value(row.get("dwresp")),
        "dwstat": clean_label_value(row.get("dwstat")),
        "dwreas": clean_label_value(row.get("dwreas")),
        "dwrecall": clean_label_value(row.get("dwrecall")),
        "dwnotice": clean_label_value(row.get("dwnotice")),
        "dwlastwrk": clean_label_value(row.get("dwlastwrk")),
        "dwfulltime": clean_label_value(row.get("dwfulltime")),
        "dwunion": clean_label_value(row.get("dwunion")),
        "dwben": clean_label_value(row.get("dwben")),
        "dwhi": clean_label_value(row.get("dwhi")),
        "dwhinow": clean_label_value(row.get("dwhinow")),
        "dwclass": clean_label_value(row.get("dwclass")),
        "dwyears": bounded_float(row.get("dwyears"), min_value=0, max_value=99.95),
        "dwweekl": bounded_float(row.get("dwweekl"), min_value=0, max_value=9999.95),
        "dwweekc": bounded_float(row.get("dwweekc"), min_value=0, max_value=9999.97),
        "dwwksun": bounded_int(row.get("dwwksun"), min_value=0, max_value=995),
        "dwjobsince": dwjobsince,
    }


def base_table_row(
    fieldnames: list[str],
    baseline: dict[str, Any],
    target: dict[str, Any],
    macro: dict[str, float | None],
    row_id: str,
    answer: str,
) -> dict[str, Any]:
    release_date = target.get("release_date")
    release_date_source = target.get("release_date_source")
    if release_date is None or release_date_source is None:
        raise RuntimeError(f"Missing CPS release metadata for {row_id}.")
    row = {field: None for field in fieldnames}
    row.update(
        {
            "row_id": row_id,
            "subject_id": str(baseline["cpsidp"]),
            "baseline_year": baseline["year"],
            "baseline_month": baseline["month"],
            "outcome_year": target["year"],
            "outcome_month": target["month"],
            "release_date": release_date,
            "release_date_source": release_date_source,
            "mish_baseline": baseline.get("mish"),
            "mish_target": target.get("mish"),
            "baseline_status": baseline.get("status"),
            "target": answer,
            "age": baseline["age"],
            "age_group": age_group(baseline.get("age")),
            "sex": baseline["sex"],
            "race": baseline["race"],
            "educ": baseline["educ"],
            "marst": baseline["marst"],
            "state": baseline["state"],
            "region": baseline["region"],
            "metro": baseline["metro"],
            "relate": baseline["relate"],
            "famsize": baseline["famsize"],
            "nchild": baseline["nchild"],
            "hispan": baseline["hispan"],
            "nativity": baseline["nativity"],
            "citizen": baseline["citizen"],
            "vetstat": baseline["vetstat"],
            "classwkr": baseline.get("classwkr"),
            **{column: macro.get(column) for column in CORE_MACRO_COLUMNS},
        }
    )
    return row


def validate_12m_link(baseline: dict[str, Any], target_obs: dict[str, Any]) -> str | None:
    if baseline.get("cpsidv") is not None and target_obs.get("cpsidv") is not None:
        if baseline["cpsidv"] != target_obs["cpsidv"]:
            return None
        cpsidv_status = "cpsidv_match"
    else:
        cpsidv_status = "cpsidv_unavailable"
    if baseline.get("sex_code") != target_obs.get("sex_code"):
        return None
    if baseline.get("race_code") != target_obs.get("race_code"):
        return None
    base_age = baseline.get("age")
    target_age = target_obs.get("age")
    if base_age is None or target_age is None or int(target_age) - int(base_age) not in {0, 1, 2}:
        return None
    return f"{cpsidv_status};sex_race_age_plausible"
