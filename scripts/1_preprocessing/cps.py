#!/usr/bin/env python
"""Build the CPS Basic Monthly panel, including Displaced Worker Supplement fields."""

from __future__ import annotations

import argparse
import os
import gc
import gzip
import io
import itertools
import queue
import re
import threading
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.io import exclusive_lock
from scripts.utils.parallel import available_cpus
from scripts.utils.source_inputs import validate_source_inputs

BASIC_RAW_NAME = "cps_basic.csv.gz"
MAPPING_DIR = Path(__file__).resolve().parent / "mappings"
RELEASE_DATE_CROSSWALK_PATH = Path("data/raw/release_dates/release_cps.csv")
RELEASE_DATE_COVERAGE_PATH = Path("output/preprocessing/release_dates/metadata_coverage.csv")

PARTITIONS = ("basic",)
BASIC_OFFICIAL_RELEASE_RULE_ID = "cps_basic_official_exact"
BASIC_FALLBACK_RELEASE_RULE_ID = "cps_basic_collection_completion_plus_30_lower_bound"
CPS_RELEASE_ARCHIVE_URL_RE = re.compile(
    r"^https://www\.census\.gov/data/what-is-data-census-gov/latest-releases/\d{4}\.html$"
)
CPS_FAQ_URL = "https://www.census.gov/programs-surveys/cps/about/faqs.html"
PARTITION_STRING_OVERRIDES = {
    "basic": {"HRSAMPLE"},
}
DWS_NUMERIC_HARMONIZATION = {
    "DWWEEKC": {
        "out": "dwweekc",
        "missing_codes": (9999.98, 9999.99),
        "minimum": 0,
        "maximum_exclusive": 9999.98,
        "integer": False,
    },
    "DWWEEKL": {
        "out": "dwweekl",
        "missing_codes": (9999.96, 9999.97, 9999.98, 9999.99),
        "minimum": 0,
        "maximum_exclusive": 9999.96,
        "integer": False,
    },
    "DWYEARS": {
        "out": "dwyears",
        "missing_codes": (99.96, 99.97, 99.98, 99.99),
        "minimum": 0,
        "maximum_exclusive": 99.96,
        "integer": False,
    },
    "DWJOBSINCE": {
        "out": "dwjobsince",
        "missing_codes": (95, 96, 97, 98, 99),
        "minimum": 0,
        "maximum_exclusive": 95,
        "integer": True,
    },
    "DWWKSUN": {
        "out": "dwwksun",
        "missing_codes": (996, 997, 998, 999),
        "minimum": 0,
        "maximum_exclusive": 996,
        "integer": True,
    },
}
NIU_LABEL_KEYWORDS = (
    "NIU",
    "NOT IN UNIVERSE",
    "MISSING",
    "NO RESPONSE",
    "NOT ASCERTAINED",
    "UNKNOWN",
    "BLANK",
)
NUMERIC_CODE_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
ALPHANUM_CODE_RE = re.compile(r"^[A-Za-z]*\d+[A-Za-z0-9]*$")

STATEFIP_LABELS = {
    1: "Alabama",
    2: "Alaska",
    4: "Arizona",
    5: "Arkansas",
    6: "California",
    8: "Colorado",
    9: "Connecticut",
    10: "Delaware",
    11: "District of Columbia",
    12: "Florida",
    13: "Georgia",
    15: "Hawaii",
    16: "Idaho",
    17: "Illinois",
    18: "Indiana",
    19: "Iowa",
    20: "Kansas",
    21: "Kentucky",
    22: "Louisiana",
    23: "Maine",
    24: "Maryland",
    25: "Massachusetts",
    26: "Michigan",
    27: "Minnesota",
    28: "Mississippi",
    29: "Missouri",
    30: "Montana",
    31: "Nebraska",
    32: "Nevada",
    33: "New Hampshire",
    34: "New Jersey",
    35: "New Mexico",
    36: "New York",
    37: "North Carolina",
    38: "North Dakota",
    39: "Ohio",
    40: "Oklahoma",
    41: "Oregon",
    42: "Pennsylvania",
    44: "Rhode Island",
    45: "South Carolina",
    46: "South Dakota",
    47: "Tennessee",
    48: "Texas",
    49: "Utah",
    50: "Vermont",
    51: "Virginia",
    53: "Washington",
    54: "West Virginia",
    55: "Wisconsin",
    56: "Wyoming",
    61: "Maine-New Hampshire-Vermont",
    65: "Montana-Idaho-Wyoming",
    68: "Alaska-Hawaii",
    69: "Nebraska-North Dakota-South Dakota",
    70: "Maine-Massachusetts-New Hampshire-Rhode Island-Vermont",
    71: "Michigan-Wisconsin",
    72: "Minnesota-Iowa",
    73: "Nebraska-North Dakota-South Dakota-Kansas",
    74: "Delaware-Virginia",
    75: "North Carolina-South Carolina",
    76: "Alabama-Mississippi",
    77: "Arkansas-Oklahoma",
    78: "Arizona-New Mexico-Colorado",
    79: "Idaho-Wyoming-Utah-Montana-Nevada",
    80: "Alaska-Washington-Hawaii",
    81: "New Hampshire-Maine-Vermont-Rhode Island",
    83: "South Carolina-Georgia",
    84: "Kentucky-Tennessee",
    85: "Arkansas-Louisiana-Oklahoma",
    87: "Iowa-N Dakota-S Dakota-Nebraska-Kansas-Minnesota-Missouri",
    88: "Washington-Oregon-Alaska-Hawaii",
    89: "Montana-Wyoming-Colorado-New Mexico-Utah-Nevada-Arizona",
    90: "Delaware-Maryland-Virginia-West Virginia",
    99: "State not identified",
}

CPS_CATEGORICAL_LABEL_SPECS = {
    "SEX": {
        "out": "sex",
        "exact": {1: "man", 2: "woman"},
    },
    "RACE": {
        "out": "race",
        "exact": {
            100: "White",
            200: "Black",
            300: "American Indian",
            650: "Asian or Pacific Islander",
            651: "Asian or Pacific Islander",
            652: "Asian or Pacific Islander",
            700: "other race",
        },
        "ranges": [(801, 899, "multiracial")],
    },
    "EDUC": {
        "out": "educ",
        "exact": {
            73: "high school diploma or GED",
            110: "bachelor's degree",
            111: "bachelor's degree",
        },
        "ranges": [
            (0, 72, "high school or less"),
            (80, 100, "some college or associate degree"),
            (120, 999, "graduate or professional education"),
        ],
    },
    "MARST": {
        "out": "marst",
        "exact": {
            1: "married, spouse present",
            2: "married, spouse absent",
            3: "separated",
            4: "divorced",
            5: "widowed",
            6: "never married",
            7: "widowed or divorced",
        },
        "missing_codes": {9},
    },
    "STATEFIP": {
        "out": "state",
        "exact": STATEFIP_LABELS,
    },
    "REGION": {
        "out": "region",
        "exact": {
            11: "Northeast",
            12: "Northeast",
            21: "Midwest",
            22: "Midwest",
            31: "South",
            32: "South",
            33: "South",
            41: "West",
            42: "West",
            97: "State not identified",
        },
    },
    "METRO": {
        "out": "metro",
        "exact": {
            1: "non-metropolitan area",
            2: "metropolitan area",
            3: "metropolitan area",
            4: "metropolitan area",
        },
        "missing_codes": {0, 9},
    },
    "RELATE": {
        "out": "relate",
        "exact": {
            101: "reference person",
            201: "spouse",
            202: "spouse",
            203: "spouse",
            301: "child",
            303: "child",
            501: "parent",
            701: "sibling",
            901: "grandchild",
            1001: "other relative",
            1113: "partner",
            1114: "partner",
            1115: "housemate or roommate",
            1116: "partner",
            1117: "partner",
            1241: "housemate or roommate",
            1242: "foster child",
            1260: "other nonrelative",
            9900: "relationship unknown",
        },
        "missing_codes": {9999},
    },
    "HISPAN": {
        "out": "hispan",
        "exact": {
            0: "not Hispanic",
            200: "Puerto Rican",
            300: "Cuban",
            400: "Dominican",
        },
        "ranges": [
            (100, 199, "Mexican origin"),
            (500, 699, "other Hispanic"),
        ],
        "missing_codes": {901, 902},
    },
    "NATIVITY": {
        "out": "nativity",
        "exact": {
            1: "native-born",
            2: "native-born",
            3: "native-born",
            4: "native-born",
            5: "foreign-born",
        },
        "missing_codes": {0},
    },
    "CITIZEN": {
        "out": "citizen",
        "exact": {
            1: "U.S.-born citizen",
            2: "U.S.-born citizen",
            3: "U.S.-born citizen",
            4: "naturalized citizen",
            5: "not a U.S. citizen",
        },
    },
    "VETSTAT": {
        "out": "vetstat",
        "exact": {1: "not a veteran", 2: "veteran", 9: "Unknown"},
        "missing_codes": {0},
    },
    "CLASSWKR": {
        "out": "classwkr",
        "exact": {
            10: "self-employed",
            13: "self-employed, not incorporated",
            14: "self-employed, incorporated",
            20: "wage or salary worker",
            21: "private wage and salary worker",
            22: "private for-profit wage and salary worker",
            23: "private nonprofit wage and salary worker",
            24: "government wage and salary worker",
            25: "federal government employee",
            26: "armed forces",
            27: "state government employee",
            28: "local government employee",
            29: "unpaid family worker",
        },
        "missing_codes": {0, 99},
    },
    "EMPSTAT": {
        "out": "empstat",
        "exact": {
            1: "armed forces",
            10: "at work",
            12: "has job, not at work last week",
            20: "unemployed",
            21: "unemployed, experienced worker",
            22: "unemployed, new worker",
            30: "not in labor force",
            31: "not in labor force, housework",
            32: "not in labor force, unable to work",
            33: "not in labor force, school",
            34: "not in labor force, other",
            35: "not in labor force, unpaid fewer than 15 hours",
            36: "not in labor force, retired",
        },
        "missing_codes": {0},
    },
    "LABFORCE": {
        "out": "labforce",
        "exact": {1: "not in the labor force", 2: "in the labor force"},
        "missing_codes": {0},
    },
    "NILFACT": {
        "out": "nilfact",
        "exact": {
            1: "disabled",
            2: "ill",
            3: "in school",
            4: "taking care of house or family",
            6: "something else or other",
        },
        "missing_codes": {0, 99},
    },
    "WKSTAT": {
        "out": "wkstat",
        "exact": {
            10: "full-time schedules",
            11: "full-time hours, usually full-time",
            12: "part-time for non-economic reasons, usually full-time",
            13: "not at work, usually full-time",
            14: "full-time hours, usually part-time for economic reasons",
            15: "full-time hours, usually part-time for non-economic reasons",
            20: "part-time for economic reasons",
            21: "part-time for economic reasons, usually full-time",
            22: "part-time hours, usually part-time for economic reasons",
            40: "part-time for non-economic reasons",
            41: "part-time hours, usually part-time for non-economic reasons",
            42: "not at work, usually part-time",
            50: "unemployed, seeking full-time work",
            60: "unemployed, seeking part-time work",
        },
        "missing_codes": {0, 99},
    },
    "WNFTLOOK": {
        "out": "wnftlook",
        "exact": {
            10: "less than 5 years ago",
            11: "within the last 12 months",
            12: "one to five years ago",
            20: "more than 12 months ago",
            30: "more than 5 years ago",
            40: "never worked",
            41: "never worked full-time for 2 or more weeks",
            42: "never worked at all",
        },
        "missing_codes": {0, 99},
    },
    "MULTJOB": {
        "out": "multjob",
        "exact": {1: "no", 2: "yes"},
        "missing_codes": {0},
    },
    "UNION": {
        "out": "union",
        "exact": {
            1: "no union coverage",
            2: "member of a labor union",
            3: "covered by a union but not a member",
        },
        "missing_codes": {0},
    },
    "DWMOVE": {
        "out": "dwmove",
        "exact": {
            1: "did_not_move",
            2: "moved_not_because_job_loss",
            3: "moved_because_job_loss",
        },
        "missing_codes": {96, 97, 98, 99},
    },
    "DWRESP": {
        "out": "dwresp",
        "exact": {
            0: "not_eligible",
            1: "eligible_not_interviewed",
            2: "eligible_interviewed",
            6: "refused",
            7: "don't know",
        },
    },
    "DWSTAT": {
        "out": "dwstat",
        "exact": {0: "not_displaced_worker", 1: "displaced_worker"},
        "missing_codes": {99},
    },
    "DWREAS": {
        "out": "dwreas",
        "exact": {
            1: "plant or company closed down or moved",
            2: "insufficient work",
            3: "position or shift abolished",
            4: "seasonal job completed",
            5: "self-operated business failed",
            6: "other reason",
        },
        "missing_codes": {96, 97, 98, 99},
    },
    "DWRECALL": {
        "out": "dwrecall",
        "exact": {1: "no", 2: "yes"},
        "missing_codes": {96, 97, 98, 99},
    },
    "DWNOTICE": {
        "out": "dwnotice",
        "exact": {
            1: "no notice",
            2: "less than 1 month notice",
            3: "1 to 2 months notice",
            4: "more than 2 months notice",
            5: "notice given, time period not reported",
        },
        "missing_codes": {96, 97, 98, 99},
    },
    "DWLASTWRK": {
        "out": "dwlastwrk",
        "exact": {
            0: "this year",
            1: "last year",
            2: "two years ago",
            3: "three years ago",
            4: "four years ago",
            5: "five years ago",
            95: "other timing",
        },
        "missing_codes": {96, 97, 98, 99},
    },
    "DWFULLTIME": {
        "out": "dwfulltime",
        "exact": {1: "not full time", 2: "full time", 3: "hours varied"},
        "missing_codes": {96, 97, 98, 99},
    },
    "DWUNION": {
        "out": "dwunion",
        "exact": {1: "no", 2: "yes"},
        "missing_codes": {96, 97, 98, 99},
    },
    "DWBEN": {
        "out": "dwben",
        "exact": {1: "no", 2: "yes"},
        "missing_codes": {96, 97, 98, 99},
    },
    "DWHI": {
        "out": "dwhi",
        "exact": {1: "no", 2: "yes"},
        "missing_codes": {96, 97, 98, 99},
    },
    "DWHINOW": {
        "out": "dwhinow",
        "exact": {1: "no", 2: "yes"},
        "missing_codes": {96, 97, 98, 99},
    },
    "DWCLASS": {
        "out": "dwclass",
        "exact": {
            1: "government",
            2: "private for-profit",
            3: "private nonprofit",
            4: "self-employed",
            5: "without pay or family business",
        },
        "missing_codes": {96, 97, 98, 99},
    },
}


def is_supported_code_token(token: str) -> bool:
    t = token.strip()
    return bool(NUMERIC_CODE_RE.fullmatch(t) or ALPHANUM_CODE_RE.fullmatch(t))


def required_input_paths(raw_dir: Path) -> dict[str, Path]:
    paths = {
        "basic_csv": raw_dir / BASIC_RAW_NAME,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit(
            "Missing required CPS input files. Expected exact local files:\n"
            + "\n".join(f"- {m}" for m in missing)
        )
    return paths


def iter_csv_chunks(path: Path, chunksize: int, string_columns: list[str] | None = None):
    print(f"Streaming {path} with chunksize={chunksize:,} ...")
    dtype_map = {col: "string" for col in (string_columns or [])}
    reader = pd.read_csv(path, low_memory=False, chunksize=chunksize, dtype=(dtype_map or None))
    for i, chunk in enumerate(reader, start=1):
        yield i, chunk


def iter_raw_row_blocks(path: Path, chunksize: int, block_size: int = 1 << 24):
    """Yield (header, rows) byte blocks of exactly `chunksize` CSV rows (fewer in the last block).

    The blocks match the chunks pd.read_csv(chunksize=...) yields, so parsing each block on its
    own gives the same frames. This relies on one row per line. The extract quotes some short
    codes (HRSAMPLE from 1994), which is fine while every line holds an even number of quotes;
    a quoted line break, a carriage return, a blank line or a line that starts with a space
    or tab stops the build instead.
    """
    with gzip.open(path, "rb") as handle:
        header = handle.readline()
        parts: list[bytes] = []
        newlines = 0
        while True:
            block = handle.read(block_size)
            if block:
                parts.append(block)
                newlines += block.count(b"\n")
            while parts and (newlines >= chunksize or not block):
                data = b"".join(parts)
                if newlines >= chunksize:
                    ends = np.flatnonzero(np.frombuffer(data, dtype=np.uint8) == ord("\n"))
                    cut = int(ends[chunksize - 1]) + 1
                    rows, rest = data[:cut], data[cut:]
                    newlines -= chunksize
                else:
                    rows, rest = data, b""
                    newlines = 0
                parts = [rest] if rest else []
                if not _one_row_per_line(rows):
                    raise SystemExit(
                        f"{path.name} has a quoted line break, a carriage return, a blank line "
                        "or a line that starts with whitespace; rerun the CPS build with --jobs 1."
                    )
                if rows:
                    yield header, rows
            if not block:
                return


def _one_row_per_line(rows: bytes) -> bool:
    """True when every line of `rows` is one CSV row: no CR, no blank line, quotes paired per line.

    pandas skips lines of only spaces or tabs, so a line that starts with either fails too.
    """
    if b"\r" in rows or b"\n\n" in rows or rows.startswith((b"\n", b" ", b"\t")):
        return False
    if b"\n " in rows or b"\n\t" in rows:
        return False
    if b'"' not in rows:
        return True
    data = np.frombuffer(rows, dtype=np.uint8)
    line_ends = np.flatnonzero(data == ord("\n"))
    quote_lines = np.searchsorted(line_ends, np.flatnonzero(data == ord('"')))
    return not (np.bincount(quote_lines, minlength=len(line_ends) + 1) % 2).any()


def _read_ahead(items, depth: int):
    """Yield from `items` while a background thread advances it up to `depth` items ahead.

    gzip decompression and numpy release the GIL, so cutting the next row blocks overlaps
    with the parquet writes on the main thread. An exception in the thread re-raises here.
    """
    buffer: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def put(entry) -> bool:
        while not stop.is_set():
            try:
                buffer.put(entry, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        try:
            for item in items:
                if not put(("item", item)):
                    return
            put(("done", None))
        except BaseException as error:
            put(("error", error))

    thread = threading.Thread(target=produce, name="cps-block-reader", daemon=True)
    thread.start()
    try:
        while True:
            kind, value = buffer.get()
            if kind == "done":
                return
            if kind == "error":
                raise value
            yield value
    finally:
        stop.set()


_CHUNK_CONTEXT: dict[str, object] = {}


def _init_chunk_worker(context: dict[str, object]) -> None:
    _CHUNK_CONTEXT.clear()
    _CHUNK_CONTEXT.update(context)


def _parse_and_process_chunk(chunk_idx: int, start_row: int, header: bytes, rows: bytes) -> dict[str, object] | None:
    context = _CHUNK_CONTEXT
    dtype_map = {col: "string" for col in context["string_columns"]}
    chunk = pd.read_csv(io.BytesIO(header + rows), low_memory=False, dtype=(dtype_map or None))
    chunk.index = pd.RangeIndex(start_row, start_row + len(chunk))
    return process_chunk(chunk_idx, chunk, context)


def iter_processed_chunks(csv_path: Path, chunksize: int, context: dict[str, object], jobs: int):
    """Yield (chunk_idx, process_chunk result) in file order, using `jobs` worker processes."""
    if jobs == 1:
        for chunk_idx, chunk in iter_csv_chunks(csv_path, chunksize, string_columns=context["string_columns"]):
            yield chunk_idx, process_chunk(chunk_idx, chunk, context)
        return
    print(f"Streaming {csv_path} with chunksize={chunksize:,} across {jobs} worker processes ...")
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_chunk_worker, initargs=(context,)) as pool:
        in_flight: deque = deque()
        raw_blocks = iter_raw_row_blocks(csv_path, chunksize)
        # The pool forks its workers at the first submit; start the reader thread after that.
        first_block = next(raw_blocks, None)
        blocks = itertools.chain([] if first_block is None else [first_block], _read_ahead(raw_blocks, depth=4))
        for chunk_idx, (header, rows) in enumerate(blocks, start=1):
            start_row = (chunk_idx - 1) * chunksize
            in_flight.append((chunk_idx, pool.submit(_parse_and_process_chunk, chunk_idx, start_row, header, rows)))
            if len(in_flight) >= 2 * jobs:
                done_idx, future = in_flight.popleft()
                yield done_idx, future.result()
        while in_flight:
            done_idx, future = in_flight.popleft()
            yield done_idx, future.result()


def filter_year_window(df: pd.DataFrame, start_year: int | None, end_year: int | None, dataset_name: str) -> pd.DataFrame:
    if "YEAR" not in df.columns:
        raise SystemExit(f"{dataset_name} is missing YEAR, cannot apply year filtering.")

    year = pd.to_numeric(df["YEAR"], errors="coerce")
    mask = year.notna()
    if start_year is not None:
        mask &= year >= start_year
    if end_year is not None:
        mask &= year <= end_year

    return df.loc[mask].copy()


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "YEAR" not in out.columns:
        return out

    year = pd.to_numeric(out["YEAR"], errors="coerce")
    month = pd.to_numeric(out["MONTH"], errors="coerce") if "MONTH" in out.columns else pd.Series(pd.NA, index=out.index)
    out["post_1994_redesign"] = (year >= 1994).astype("Int64")
    out["multi_race_era"] = (year >= 2003).astype("Int64")
    out["covid_period"] = (year == 2020).astype("Int64")
    out["post_2023_disclosure_era"] = (year >= 2023).astype("Int64")
    out["cps_2025_2026_sample_redesign"] = (
        ((year == 2025) & (month >= 4)) | (year == 2026)
    ).astype("Int64")
    out["cps_nov_2025_shutdown_caveat"] = ((year == 2025) & (month == 11)).astype("Int64")
    out["cps_jan_2026_population_control_revision"] = ((year == 2026) & (month == 1)).astype("Int64")

    # Keep the Basic output schema; ASEC experiment flags are inapplicable.
    out["asec_2014_traditional"] = pd.Series(0, index=out.index, dtype="Int64")
    out["asec_2014_redesigned"] = pd.Series(0, index=out.index, dtype="Int64")

    return out


def _ordered_categories(spec: dict[str, object]) -> list[str]:
    labels: list[str] = []
    for label in dict(spec.get("exact", {})).values():
        labels.append(str(label))
    for _lower, _upper, label in list(spec.get("ranges", [])):
        labels.append(str(label))
    return list(dict.fromkeys(labels))


def labelled_categorical_from_codes(
    df: pd.DataFrame,
    source: str,
    spec: dict[str, object],
    partition: str,
    chunk_idx: int,
) -> pd.Categorical:
    numeric = pd.to_numeric(df[source], errors="coerce")
    rounded = numeric.round()
    invalid_fractional = numeric.notna() & ((numeric - rounded).abs() > 1e-6)
    if bool(invalid_fractional.any()):
        examples = sorted(numeric.loc[invalid_fractional].dropna().astype(str).unique().tolist())[:10]
        raise SystemExit(
            f"{partition} chunk {chunk_idx}: {source} has non-integer categorical codes: {examples}"
        )

    codes = rounded.astype("Int64")
    labels = pd.Series(pd.NA, index=df.index, dtype="string")

    exact = dict(spec.get("exact", {}))
    for code, label in exact.items():
        labels.loc[codes.eq(int(code))] = str(label)

    for lower, upper, label in list(spec.get("ranges", [])):
        labels.loc[codes.between(int(lower), int(upper), inclusive="both")] = str(label)

    missing_codes = {int(code) for code in set(spec.get("missing_codes", set()))}
    missing_mask = codes.isna()
    if missing_codes:
        missing_mask |= codes.isin(missing_codes)
    labels.loc[missing_mask] = pd.NA

    unmapped = codes.notna() & ~codes.isin(missing_codes) & labels.isna()
    if bool(unmapped.any()):
        examples = sorted(codes.loc[unmapped].dropna().astype(int).unique().tolist())[:20]
        out = str(spec["out"])
        raise SystemExit(
            f"{partition} chunk {chunk_idx}: {source} has nonmissing codes not mapped to {out}: {examples}"
        )

    return pd.Categorical(labels, categories=_ordered_categories(spec))


def add_labelled_categorical_columns(
    df: pd.DataFrame,
    partition: str,
    chunk_idx: int,
) -> pd.DataFrame:
    out = df.copy()
    for source, spec in CPS_CATEGORICAL_LABEL_SPECS.items():
        if source not in out.columns:
            continue
        out[str(spec["out"])] = labelled_categorical_from_codes(out, source, spec, partition, chunk_idx)
    return out


def add_harmonized_dws_numeric_columns(
    df: pd.DataFrame,
    partition: str,
    chunk_idx: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Recode official DWS missing values and add reusable numeric fields."""
    out = df.copy()
    recoded_counts: dict[str, int] = {}

    for source, spec in DWS_NUMERIC_HARMONIZATION.items():
        if source not in out.columns:
            continue

        numeric = pd.to_numeric(out[source], errors="raise")
        missing_codes = tuple(float(code) for code in spec["missing_codes"])
        missing_mask = numeric.isin(missing_codes)
        recoded_counts[source] = int(missing_mask.sum())
        if bool(missing_mask.any()):
            numeric = numeric.mask(missing_mask)

        nonfinite = numeric.notna() & numeric.abs().eq(float("inf"))
        if bool(nonfinite.any()):
            raise SystemExit(f"{partition} chunk {chunk_idx}: {source} has non-finite values.")

        minimum = float(spec["minimum"])
        maximum_exclusive = float(spec["maximum_exclusive"])
        outside_domain = numeric.notna() & ((numeric < minimum) | (numeric >= maximum_exclusive))
        if bool(outside_domain.any()):
            examples = sorted(numeric.loc[outside_domain].dropna().astype(str).unique().tolist())[:10]
            raise SystemExit(
                f"{partition} chunk {chunk_idx}: {source} has values outside "
                f"[{minimum}, {maximum_exclusive}): {examples}"
            )

        if bool(spec["integer"]):
            fractional = numeric.notna() & ((numeric - numeric.round()).abs() > 1e-6)
            if bool(fractional.any()):
                examples = sorted(numeric.loc[fractional].dropna().astype(str).unique().tolist())[:10]
                raise SystemExit(
                    f"{partition} chunk {chunk_idx}: {source} has non-integer substantive values: {examples}"
                )
            numeric = numeric.round().astype("Int64")

        out[source] = numeric
        out[str(spec["out"])] = numeric.copy()

    return out, recoded_counts


def load_cps_release_date_crosswalk() -> pd.DataFrame:
    required_columns = ["partition", "year", "month", "release_date", "source", "source_url", "comments"]
    if not RELEASE_DATE_CROSSWALK_PATH.exists():
        raise SystemExit(f"Missing CPS release-date crosswalk: {RELEASE_DATE_CROSSWALK_PATH}")

    crosswalk = pd.read_csv(
        RELEASE_DATE_CROSSWALK_PATH,
        dtype={
            "partition": "string",
            "source": "string",
            "source_url": "string",
            "comments": "string",
        },
    )
    missing = [column for column in required_columns if column not in crosswalk.columns]
    if missing:
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} is missing columns: {missing}")

    crosswalk = crosswalk.loc[:, required_columns].copy()
    crosswalk["partition"] = crosswalk["partition"].str.strip()
    crosswalk = crosswalk.loc[crosswalk["partition"].eq("basic")].copy()
    if crosswalk.empty:
        raise ValueError("CPS release crosswalk contains no Basic Monthly rows.")
    crosswalk["year"] = pd.to_numeric(crosswalk["year"], errors="coerce").astype("Int64")
    crosswalk["month"] = pd.to_numeric(crosswalk["month"], errors="coerce").astype("Int64")
    crosswalk["release_date"] = pd.to_datetime(crosswalk["release_date"], errors="coerce")

    if crosswalk["partition"].isna().any() or crosswalk["partition"].eq("").any():
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains missing partition values.")
    unexpected_partitions = sorted(set(crosswalk["partition"].dropna().astype(str)) - set(PARTITIONS))
    if unexpected_partitions:
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains unexpected partitions: {unexpected_partitions}")
    if crosswalk[["year", "month"]].isna().any().any():
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains missing or nonnumeric year/month keys.")
    if crosswalk.duplicated(["partition", "year", "month"]).any():
        duplicates = crosswalk.loc[crosswalk.duplicated(["partition", "year", "month"]), ["partition", "year", "month"]]
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains duplicate keys:\n{duplicates.to_string(index=False)}")
    if crosswalk["release_date"].isna().any():
        raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains missing release_date values.")
    for column in ["source", "source_url", "comments"]:
        values = crosswalk[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise SystemExit(f"{RELEASE_DATE_CROSSWALK_PATH} contains missing {column} values.")
        crosswalk[column] = values

    basic = crosswalk.loc[crosswalk["partition"].eq("basic")].copy()
    basic_official = basic.loc[
        basic["year"].gt(2020) | (basic["year"].eq(2020) & basic["month"].ge(1))
    ]
    basic_fallback = basic.loc[
        basic["year"].lt(2020)
    ]
    if not basic_official["source"].eq("Official release date").all():
        raise SystemExit("CPS Basic rows from January 2020 onward must use Official release date metadata.")
    if not basic_official["source_url"].str.match(CPS_RELEASE_ARCHIVE_URL_RE).all():
        raise SystemExit("CPS Basic rows from January 2020 onward must cite an annual Census release archive.")
    if not basic_fallback["source"].eq("Project conservative proxy").all():
        raise SystemExit("CPS Basic rows before January 2020 must use Project conservative proxy metadata.")
    if not basic_fallback["source_url"].eq(CPS_FAQ_URL).all():
        raise SystemExit("CPS Basic rows before January 2020 must cite the CPS FAQ.")

    crosswalk["release_date"] = crosswalk["release_date"].dt.strftime("%Y-%m-%d").astype("string")
    return crosswalk


def cps_release_date_assignment(
    partition: str,
    year: pd.Series,
    month: pd.Series | None,
    release_crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    years = pd.to_numeric(year, errors="coerce")
    if years.isna().any():
        raise SystemExit(f"{partition}: release-date assignment found missing YEAR values.")
    years = years.astype(int)

    if partition not in PARTITIONS:
        raise SystemExit(f"Unsupported CPS partition for release-date assignment: {partition!r}")

    if month is None:
        raise SystemExit(f"{partition}: release-date assignment requires MONTH.")
    months = pd.to_numeric(month, errors="coerce")
    bad_month = months.isna() | ~months.between(1, 12)
    if bad_month.any():
        examples = sorted(months.loc[bad_month].dropna().astype(int).unique())[:10]
        raise SystemExit(f"{partition}: invalid MONTH values for release dates: {examples}")
    months = months.astype(int)

    key = pd.DataFrame({"partition": partition, "year": years, "month": months}, index=years.index)
    lookup = release_crosswalk.set_index(["partition", "year", "month"])
    mapped = key.join(lookup, on=["partition", "year", "month"], how="left")
    if mapped["release_date"].isna().any():
        examples = (
            mapped.loc[mapped["release_date"].isna(), ["partition", "year", "month"]]
            .drop_duplicates()
            .head(10)
        )
        raise SystemExit(f"CPS release-date crosswalk does not cover keys:\n{examples.to_string(index=False)}")
    if mapped["source"].isna().any() or mapped["source"].astype("string").str.strip().eq("").any():
        raise SystemExit("CPS release-date assignment left missing source values.")

    release_rule_id = pd.Series(BASIC_FALLBACK_RELEASE_RULE_ID, index=years.index, dtype="string")
    official = years.gt(2020) | (years.eq(2020) & months.ge(1))
    release_rule_id.loc[official] = BASIC_OFFICIAL_RELEASE_RULE_ID

    return pd.DataFrame(
        {
            "release_date": mapped["release_date"].astype("string"),
            "release_date_source": mapped["source"].astype("string"),
            "release_rule_id": release_rule_id,
        },
        index=years.index,
    )


def add_release_assignment_to_counter(counter: Counter[tuple[str, str]], release_assignment: pd.DataFrame) -> None:
    required = {"release_rule_id", "release_date"}
    missing = required - set(release_assignment.columns)
    if missing:
        raise SystemExit(f"CPS release assignment is missing columns: {sorted(missing)}")
    counts = release_assignment.groupby(["release_rule_id", "release_date"], dropna=False).size()
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
        raise SystemExit(f"CPS release-date coverage is missing columns: {missing}")

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


def normalize_chunk_types(
    df: pd.DataFrame,
    type_map: dict[str, str],
    partition: str,
    chunk_idx: int,
) -> pd.DataFrame:
    """Enforce per-column types declared in the supplied mapping."""
    out = df.copy()
    for col in out.columns:
        expected = type_map.get(col, "numeric")
        series = out[col]

        if expected == "string":
            if not pd.api.types.is_string_dtype(series):
                out[col] = series.astype("string")
            continue

        if pd.api.types.is_numeric_dtype(series):
            continue

        try:
            out[col] = pd.to_numeric(series, errors="raise")
        except Exception as exc:
            s = series.dropna().astype(str).str.strip()
            if s.empty:
                bad_example = "<all-missing>"
            else:
                bad = s[~s.str.fullmatch(NUMERIC_CODE_RE.pattern)]
                bad_example = bad.iloc[0] if not bad.empty else s.iloc[0]
            raise SystemExit(
                f"{partition} chunk {chunk_idx}: column {col} is declared numeric in cps_types.csv "
                f"but has non-numeric value {bad_example!r}."
            ) from exc

    return out


def apply_niu_recode(
    df: pd.DataFrame,
    niu_map: dict[str, list[str]],
    type_map: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = df.copy()
    recoded_counts: dict[str, int] = {}

    for col, codes in niu_map.items():
        if col not in out.columns or not codes:
            continue

        expected = type_map.get(col, "numeric")
        series = out[col]

        if expected == "string":
            s = series.astype("string").str.strip()
            code_set = {c.strip() for c in codes}
            mask = s.isin(code_set)
        else:
            numeric_codes: list[float] = []
            for code in codes:
                if not NUMERIC_CODE_RE.fullmatch(code):
                    raise SystemExit(
                        f"NIU code {code!r} in numeric column {col} is non-numeric. "
                        "Check the source extract and supplied type mapping."
                    )
                numeric_codes.append(float(code))
            numeric_values = series if pd.api.types.is_numeric_dtype(series) else pd.to_numeric(series, errors="raise")
            mask = numeric_values.isin(numeric_codes)

        n = int(mask.sum())
        if n:
            out.loc[mask, col] = pd.NA
        recoded_counts[col] = n

    return out, recoded_counts


def process_chunk(chunk_idx: int, chunk: pd.DataFrame, context: dict[str, object]) -> dict[str, object] | None:
    """Apply the per-chunk source construction; return the Arrow table and the chunk's tallies.

    Runs in a worker process when the build uses more than one job, so it reads only its
    arguments. Counts are returned as ordered lists and summed by the caller in chunk order.
    """
    partition = str(context["partition"])
    type_map = context["type_map"]
    chunk = filter_year_window(chunk, context["start_year"], context["end_year"], str(context["dataset_name"]))
    if chunk.empty:
        return None

    recode_counts: list[tuple[str, int]] = []
    chunk = normalize_chunk_types(chunk, type_map, partition, chunk_idx)
    chunk, niu_recode_counts = apply_niu_recode(chunk, context["niu_map"], type_map)
    recode_counts.extend((var, int(n)) for var, n in niu_recode_counts.items())

    if partition == "basic":
        rows_before_dws_harmonization = len(chunk)
        index_before_dws_harmonization = chunk.index.copy()
        cpsidp_before_dws_harmonization = chunk["CPSIDP"].copy()
        chunk, dws_recode_counts = add_harmonized_dws_numeric_columns(chunk, partition, chunk_idx)
        if len(chunk) != rows_before_dws_harmonization or not chunk.index.equals(index_before_dws_harmonization):
            raise SystemExit(f"{partition} chunk {chunk_idx}: DWS harmonization changed rows or row order.")
        if not chunk["CPSIDP"].equals(cpsidp_before_dws_harmonization):
            raise SystemExit(f"{partition} chunk {chunk_idx}: DWS harmonization changed CPSIDP values.")
        recode_counts.extend((var, int(n)) for var, n in dws_recode_counts.items())

    chunk = add_labelled_categorical_columns(chunk, partition, chunk_idx)
    chunk = add_flags(chunk)
    release_assignment = cps_release_date_assignment(
        partition,
        chunk["YEAR"],
        chunk["MONTH"] if "MONTH" in chunk.columns else None,
        context["release_crosswalk"],
    )
    chunk["release_date"] = release_assignment["release_date"]
    chunk["release_date_source"] = release_assignment["release_date_source"]
    release_counts: Counter[tuple[str, str]] = Counter()
    add_release_assignment_to_counter(release_counts, release_assignment)

    yc = pd.to_numeric(chunk.get("YEAR"), errors="coerce").dropna().astype(int).value_counts()
    year_counts = [(int(year), int(n)) for year, n in yc.items()]

    year_month_counts: list[tuple[tuple[int, int], int]] = []
    if {"YEAR", "MONTH"}.issubset(chunk.columns):
        ym = chunk[["YEAR", "MONTH"]].copy()
        ym["YEAR"] = pd.to_numeric(ym["YEAR"], errors="coerce")
        ym["MONTH"] = pd.to_numeric(ym["MONTH"], errors="coerce")
        ym = ym.dropna(subset=["YEAR", "MONTH"])
        ym = ym[(ym["MONTH"] >= 1) & (ym["MONTH"] <= 12)]
        if not ym.empty:
            g = (
                ym.assign(YEAR=ym["YEAR"].astype(int), MONTH=ym["MONTH"].astype(int))
                .groupby(["YEAR", "MONTH"], dropna=False)
                .size()
            )
            year_month_counts = [((int(year), int(month)), int(n)) for (year, month), n in g.items()]

    sample_size = int(context["sample_size"])
    per_chunk_n = max(50, min(sample_size // 2, 500))
    sampled = chunk.sample(n=min(per_chunk_n, len(chunk)), random_state=int(context["sample_seed"]) + chunk_idx)

    return {
        "table": pa.Table.from_pandas(chunk, preserve_index=False),
        "rows": int(len(chunk)),
        "columns": int(len(chunk.columns)),
        "column_names": chunk.columns.tolist(),
        "recode_counts": recode_counts,
        "release_counts": release_counts,
        "year_counts": year_counts,
        "year_month_counts": year_month_counts,
        "sample": sampled,
    }


def write_partition_stream(
    csv_path: Path,
    partition: str,
    intermediate_dir: Path,
    release_crosswalk: pd.DataFrame,
    start_year: int | None,
    end_year: int | None,
    chunksize: int,
    sample_size: int,
    sample_seed: int,
    jobs: int = 1,
) -> dict[str, object]:
    out_path = intermediate_dir / f"cps_{partition}.parquet"
    if out_path.exists():
        out_path.unlink()

    header_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    allowed_variables = set(header_columns)
    if partition == "basic":
        missing_dws_variables = sorted(set(DWS_NUMERIC_HARMONIZATION) - allowed_variables)
        if missing_dws_variables:
            raise SystemExit(
                f"{csv_path.name} is missing required DWS numeric variables: {missing_dws_variables}"
            )
    niu_audit = pd.read_csv(MAPPING_DIR / "cps_missing_codes.csv", dtype=str, keep_default_na=False)
    niu_map = {
        variable: sorted(set(rows["code"]), key=lambda code: (len(code), code))
        for variable, rows in niu_audit.groupby("variable", sort=False)
    }
    type_audit = pd.read_csv(MAPPING_DIR / "cps_types.csv", dtype=str)
    type_map = dict(zip(type_audit["variable"], type_audit["declared_type"], strict=True))
    if set(type_map) != allowed_variables:
        raise ValueError("CPS columns differ from the supplied extract type mapping.")
    forced_string_cols = sorted(PARTITION_STRING_OVERRIDES.get(partition, set()).intersection(allowed_variables))
    for col in forced_string_cols:
        type_map[col] = "string"
    if forced_string_cols:
        for col in forced_string_cols:
            mask = type_audit["variable"] == col
            if mask.any():
                type_audit.loc[mask, "declared_type"] = "string"
                type_audit.loc[mask, "trigger_code"] = "<manual_override>"
                type_audit.loc[mask, "trigger_label"] = f"forced string for {partition}"
            else:
                type_audit = pd.concat(
                    [
                        type_audit,
                        pd.DataFrame(
                            [
                                {
                                    "variable": col,
                                    "declared_type": "string",
                                    "trigger_code": "<manual_override>",
                                    "trigger_label": f"forced string for {partition}",
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
    string_columns = [col for col in header_columns if type_map.get(col) == "string"]
    print(
        f"{partition}: effective types -> "
        f"numeric={len(type_map) - len(string_columns):,}, string={len(string_columns):,}"
    )

    writer: pq.ParquetWriter | None = None
    row_count = 0
    col_count = 0
    columns_seen: set[str] = set()
    year_counter: dict[int, int] = defaultdict(int)
    year_month_counter: dict[tuple[int, int], int] = defaultdict(int)
    recode_counter: dict[str, int] = defaultdict(int)
    release_counter: Counter[tuple[str, str]] = Counter()
    sample_buffers: list[pd.DataFrame] = []

    context = {
        "partition": partition,
        "dataset_name": csv_path.name,
        "start_year": start_year,
        "end_year": end_year,
        "type_map": type_map,
        "niu_map": niu_map,
        "release_crosswalk": release_crosswalk,
        "string_columns": string_columns,
        "sample_size": sample_size,
        "sample_seed": sample_seed,
    }
    for chunk_idx, result in iter_processed_chunks(csv_path, chunksize, context, jobs):
        if result is None:
            continue
        for var, n in result["recode_counts"]:
            recode_counter[var] += n
        release_counter.update(result["release_counts"])
        row_count += result["rows"]
        col_count = max(col_count, result["columns"])
        columns_seen.update(result["column_names"])
        for year, n in result["year_counts"]:
            year_counter[year] += n
        for year_month, n in result["year_month_counts"]:
            year_month_counter[year_month] += n

        # Keep bounded in-memory sample buffers.
        sample_buffers.append(result["sample"])
        if len(sample_buffers) > 200:
            sample_buffers = [pd.concat(sample_buffers, ignore_index=True).sample(n=min(20_000, sum(len(x) for x in sample_buffers)), random_state=sample_seed)]

        table = result["table"]
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
        elif table.schema != writer.schema:
            try:
                table = table.cast(writer.schema, safe=False)
            except Exception as exc:
                raise SystemExit(
                    f"{partition} chunk {chunk_idx}: schema mismatch while casting to writer schema.\n"
                    f"Expected: {writer.schema}\nObserved: {table.schema}\nError: {exc}"
                ) from exc
        writer.write_table(table)

        print(f"{partition} chunk {chunk_idx}: wrote {result['rows']:,} rows", flush=True)
        del result, table
        gc.collect()

    if writer is not None:
        writer.close()

    if row_count == 0:
        lower = str(start_year) if start_year is not None else "earliest"
        upper = str(end_year) if end_year is not None else "latest"
        raise SystemExit(f"{csv_path.name} has zero rows after filtering YEAR to [{lower}, {upper}].")

    if sample_buffers:
        sample_cat = pd.concat(sample_buffers, ignore_index=True)
        if len(sample_cat) > sample_size:
            sample_cat = sample_cat.sample(n=sample_size, random_state=sample_seed)
    else:
        sample_cat = pd.DataFrame()

    year_df = pd.DataFrame(
        [{"partition": partition, "year": int(y), "rows": int(n)} for y, n in sorted(year_counter.items())]
    )
    ym_df = pd.DataFrame(
        [
            {"partition": partition, "year": int(y), "month": int(m), "rows": int(n)}
            for (y, m), n in sorted(year_month_counter.items())
        ]
    )
    recode_df = pd.DataFrame(
        [{"partition": partition, "variable": var, "n_recoded": int(n)} for var, n in sorted(recode_counter.items())]
    )

    if not niu_audit.empty:
        niu_audit.insert(0, "partition", partition)
    if not type_audit.empty:
        type_audit.insert(0, "partition", partition)

    return {
        "partition": partition,
        "out_path": out_path,
        "rows": row_count,
        "columns": col_count,
        "columns_seen": columns_seen,
        "year_df": year_df,
        "ym_df": ym_df,
        "recode_df": recode_df,
        "release_coverage": coverage_rows_from_counter("CPS", release_counter),
        "niu_audit": niu_audit,
        "type_audit": type_audit,
        "sample": sample_cat,
    }


def partition_variable_coverage(partitions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    all_vars = sorted({col for df in partitions.values() for col in df.columns})
    rows = []
    for partition, df in partitions.items():
        cols = set(df.columns)
        for var in all_vars:
            rows.append({"partition": partition, "variable": var, "present": int(var in cols)})
    return pd.DataFrame(rows)


def main() -> None:
    # 1. Resolve the source files and construction options.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=None, help="Default: no lower year bound")
    parser.add_argument("--end-year", type=int, default=None, help="Default: latest available year")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/micro/cps"))
    parser.add_argument("--intermediate-dir", type=Path, default=Path("data/intermediate"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/preprocessing/cps/build"))
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(available_cpus(), 16),
        help="Worker processes that parse and transform chunks (default: available CPUs, at "
        "most 16). Each holds about 2 GB; the output is the same for any value.",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1.")

    validate_source_inputs("cps", overrides={Path("data/raw/micro/cps"): args.raw_dir})

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.intermediate_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = required_input_paths(args.raw_dir)
    release_crosswalk = load_cps_release_date_crosswalk()
    # 2. Stream Basic Monthly and its DWS measurements through source construction.
    basic_result = write_partition_stream(
        csv_path=paths["basic_csv"],
        partition="basic",
        intermediate_dir=args.intermediate_dir,
        release_crosswalk=release_crosswalk,
        start_year=args.start_year,
        end_year=args.end_year,
        chunksize=args.chunksize,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        jobs=args.jobs,
    )
    # 3. Record Basic construction summaries and release-date coverage.
    results = [basic_result]
    release_coverage = pd.concat(
        [r["release_coverage"] for r in results],
        ignore_index=True,
    )
    with exclusive_lock(RELEASE_DATE_COVERAGE_PATH):
        update_release_date_coverage("CPS", release_coverage)

    for res in results:
        part = str(res["partition"])
        part_sample = res["sample"]
        if isinstance(part_sample, pd.DataFrame):
            part_sample.to_csv(args.output_dir / f"{part}_sample.csv", index=False)

    row_counts = pd.DataFrame(
        [{"partition": str(r["partition"]), "rows": int(r["rows"])} for r in results]
    ).sort_values("partition")
    row_counts.to_csv(args.output_dir / "partition_row_counts.csv", index=False)

    pd.concat([r["year_df"] for r in results], ignore_index=True).to_csv(
        args.output_dir / "partition_year_counts.csv", index=False
    )
    pd.concat([r["ym_df"] for r in results], ignore_index=True).to_csv(
        args.output_dir / "year_month_counts.csv", index=False
    )

    coverage = partition_variable_coverage(
        {
            "basic": pd.DataFrame(columns=sorted(set(basic_result["columns_seen"]))),
        }
    )
    coverage.to_csv(args.output_dir / "partition_variable_coverage.csv", index=False)

    pd.concat([r["recode_df"] for r in results], ignore_index=True).to_csv(
        args.output_dir / "niu_recode_counts.csv", index=False
    )

    niu_frames = [r["niu_audit"] for r in results if isinstance(r["niu_audit"], pd.DataFrame) and not r["niu_audit"].empty]
    niu_audit = pd.concat(niu_frames, ignore_index=True) if niu_frames else pd.DataFrame()
    if not niu_audit.empty:
        niu_audit = niu_audit.drop_duplicates(["partition", "variable", "code", "label"]).sort_values(
            ["partition", "variable", "code"]
        )
    niu_audit.to_csv(args.output_dir / "niu_code_map.csv", index=False)

    type_frames = [r["type_audit"] for r in results if isinstance(r["type_audit"], pd.DataFrame) and not r["type_audit"].empty]
    type_audit = pd.concat(type_frames, ignore_index=True) if type_frames else pd.DataFrame()
    if not type_audit.empty:
        type_audit = type_audit.sort_values(["partition", "variable"]).reset_index(drop=True)
    type_audit.to_csv(args.output_dir / "codebook_type_map.csv", index=False)

    extract_meta = pd.DataFrame(
        [
            {
                "extract_description": "manual_local_csv_extract",
                "raw_dir": os.path.relpath(args.raw_dir, PROJECT_ROOT),
                "basic_csv": os.path.relpath(paths["basic_csv"], PROJECT_ROOT),
                "start_year": args.start_year,
                "end_year": args.end_year,
                "chunksize": args.chunksize,
                "basic_rows": int(basic_result["rows"]),
                "basic_columns": int(basic_result["columns"]),
                "build_utc": datetime.now(timezone.utc).isoformat(),
            }
        ]
    )
    extract_meta.to_csv(args.output_dir / "extract_metadata.csv", index=False)

    print("CPS build complete.")
    print(row_counts.to_string(index=False))
    print("Wrote partitions:")
    for res in results:
        print(f"  {res['partition']}: {res['out_path']}")


if __name__ == "__main__":
    main()
