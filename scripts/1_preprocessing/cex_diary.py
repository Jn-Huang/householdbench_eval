#!/usr/bin/env python
"""Validate canonical archives and build the full 1982--2011 CEX Diary panel.

Use ICPSR for 1982--1989 and BLS for 1990--2011.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.utils.parallel import available_cpus
from scripts.utils.source_inputs import validate_source_inputs

from scripts.utils.io import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data/raw/micro/cex/diary"
BLS_DIR = RAW_DIR / "bls"
ICPSR_DIR = RAW_DIR / "icpsr"
RELEASE_DATE_PATH = PROJECT_ROOT / "data/raw/release_dates/release_cex.csv"
OUT_DIR = PROJECT_ROOT / "output/preprocessing/cex_diary/build"
INTERMEDIATE_DIR = PROJECT_ROOT / "data/intermediate"
INTERMEDIATE_PATH = INTERMEDIATE_DIR / "cex_diary_daily.parquet"
INTERMEDIATE_TEMP_PATH = INTERMEDIATE_DIR / "cex_diary_daily.parquet.tmp"

ARCHIVE_INVENTORY_PATH = OUT_DIR / "01_archive_inventory.csv"
MEMBER_INVENTORY_PATH = OUT_DIR / "01_member_inventory.csv"
DOCUMENTATION_INVENTORY_PATH = OUT_DIR / "01_documentation_inventory.csv"
PIPELINE_STATUS_PATH = OUT_DIR.parent / "00_pipeline_status.json"

CANONICAL_YEARS = tuple(range(1982, 2012))
ICPSR_YEARS = tuple(range(1982, 1990))
BLS_YEARS = tuple(range(1990, 2012))
RULE_VERSION = "cex_diary_general_calendar_v1"
LEGACY_STRICT_RULE_VERSION = "stephens_1986_1996_strict_v1"


# This is an explicit release map, not a filename-to-year inference.  The
# primary evidence for the BLS CSV releases is the official CEX Dictionary's
# Diary FMLD/MEMD/EXPD coverage together with the released CSV headers.
# The fixed-width positions and CSV aliases below are frozen from that
# documentation; separate guides and annual stubs are not runtime inputs.
BLS_RELEASE_YEARS = BLS_YEARS
BLS_DOCUMENTED_RELEASES = {
    f"diary{year % 100:02d}.zip": {
        "survey_year": year,
        "year_evidence": "explicit local BLS release map; Dictionary coverage and CSV member tokens checked",
        "layout_evidence": "CEX PUMD Dictionary plus released CSV header",
    }
    for year in BLS_RELEASE_YEARS
}


# The ICPSR study-year map comes from each embedded description-citation page.
# The member-by-member fixed-width map below is authoritative for source inventory.  It
# is anchored in each archive's embedded manifest, which lists the DS section,
# filename, record length, and record count.  The supporting Codebook or
# Documentation member remains in the inventory as the logical-width source.
ICPSR_DOCUMENTED_RELEASES = {
    "ICPSR_08599-V1.zip": {
        "study_id": "08599",
        "survey_year": 1982,
        "survey_years": (1982, 1983),
        "release_parts_by_year": {1982: (1, 2, 3, 4), 1983: (1, 2, 3, 4)},
        "year_evidence_member": "ICPSR_08599/08599-descriptioncitation.html",
        "year_evidence_text": "Consumer Expenditure Survey, 1982-1983:  Diary Survey",
        "layout_evidence_member": "ICPSR_08599/08599-Documentation.txt",
        "mapped_file_families": ("FMLY", "MEMB", "EXPD"),
        "logical_widths": {"FMLY": 1240, "MEMB": 245, "EXPD": 40},
    },
    "ICPSR_08628-V1.zip": {
        "study_id": "08628",
        "survey_year": 1984,
        "year_evidence_member": "ICPSR_08628/08628-descriptioncitation.html",
        "year_evidence_text": "Consumer Expenditure Survey, 1984:  Diary Survey",
        "layout_evidence_member": "ICPSR_08628/08628-Documentation.txt",
        "mapped_file_families": ("FMLY", "MEMB", "EXPD"),
        "logical_widths": {"FMLY": 1485, "MEMB": 245, "EXPD": 40},
    },
    "ICPSR_08905-V1.zip": {
        "study_id": "08905",
        "survey_year": 1985,
        "year_evidence_member": "ICPSR_08905/08905-descriptioncitation.html",
        "year_evidence_text": "Consumer Expenditure Survey, 1985:  Diary Survey",
        "layout_evidence_member": "ICPSR_08905/08905-Documentation.txt",
        "mapped_file_families": ("FMLY", "MEMB", "EXPD"),
        "logical_widths": {"FMLY": 1252, "MEMB": 245, "EXPD": 40},
    },
    "ICPSR_09114-V1.zip": {
        "study_id": "09114",
        "survey_year": 1986,
        "year_evidence_member": "ICPSR_09114/09114-descriptioncitation.html",
        "year_evidence_text": "Consumer Expenditure Survey, 1986",
        "layout_evidence_member": "ICPSR_09114/09114-Documentation.txt",
        "logical_widths": {"FMLY": 1526, "MEMB": 243, "EXPD": 38, "DTBD": 28},
    },
    "ICPSR_09333-V1.zip": {
        "study_id": "09333",
        "survey_year": 1987,
        "year_evidence_member": "ICPSR_09333/09333-descriptioncitation.html",
        "year_evidence_text": "Consumer Expenditure Survey, 1987",
        "layout_evidence_member": "ICPSR_09333/09333-Documentation.txt",
        "logical_widths": {"FMLY": 1526, "MEMB": 243, "EXPD": 38, "DTBD": 28},
    },
    "ICPSR_09570-V1.zip": {
        "study_id": "09570",
        "survey_year": 1988,
        "year_evidence_member": "ICPSR_09570/09570-descriptioncitation.html",
        "year_evidence_text": "Consumer Expenditure Survey, 1988",
        "layout_evidence_member": "ICPSR_09570/09570-Documentation.txt",
        "logical_widths": {"FMLY": 1526, "MEMB": 263, "EXPD": 38, "DTBD": 28},
    },
    "ICPSR_09714-V1.zip": {
        "study_id": "09714",
        "survey_year": 1989,
        "year_evidence_member": "ICPSR_09714/09714-descriptioncitation.html",
        "year_evidence_text": "Consumer Expenditure Survey, 1989",
        "layout_evidence_member": "ICPSR_09714/09714-Documentation.txt",
        "logical_widths": {"FMLY": 1533, "MEMB": 263, "EXPD": 38, "DTBD": 28},
    },
}

DICTIONARY_FILE_BY_FAMILY = {"FMLY": "FMLD", "MEMB": "MEMD", "EXPD": "EXPD"}
PRIMARY_FAMILIES = set(DICTIONARY_FILE_BY_FAMILY)


# Each primary member is mapped explicitly from the embedded ICPSR naming table.
# These source maps are intentionally literal: source inventory must not recover a file
# family, quarter, or fixed-width layout by parsing the DS number at runtime.
ICPSR_PRIMARY_MEMBER_MAP = {
    "ICPSR_08599-V1.zip": {
        "anchor": "ICPSR_08599/08599-manifest.txt, DS0001--DS0024 listings (section title, filename, Record Length, Record Count)",
        "members": {
            "ICPSR_08599/DS0001/08599-0001-Data.txt": ("FMLY", 1, 1240), "ICPSR_08599/DS0002/08599-0002-Data.txt": ("MEMB", 1, 245), "ICPSR_08599/DS0003/08599-0003-Data.txt": ("EXPD", 1, 40),
            "ICPSR_08599/DS0004/08599-0004-Data.txt": ("FMLY", 2, 1240), "ICPSR_08599/DS0005/08599-0005-Data.txt": ("MEMB", 2, 245), "ICPSR_08599/DS0006/08599-0006-Data.txt": ("EXPD", 2, 40),
            "ICPSR_08599/DS0007/08599-0007-Data.txt": ("FMLY", 3, 1240), "ICPSR_08599/DS0008/08599-0008-Data.txt": ("MEMB", 3, 245), "ICPSR_08599/DS0009/08599-0009-Data.txt": ("EXPD", 3, 40),
            "ICPSR_08599/DS0010/08599-0010-Data.txt": ("FMLY", 4, 1240), "ICPSR_08599/DS0011/08599-0011-Data.txt": ("MEMB", 4, 245), "ICPSR_08599/DS0012/08599-0012-Data.txt": ("EXPD", 4, 40),
            "ICPSR_08599/DS0013/08599-0013-Data.txt": ("FMLY", 1, 1240), "ICPSR_08599/DS0014/08599-0014-Data.txt": ("MEMB", 1, 245), "ICPSR_08599/DS0015/08599-0015-Data.txt": ("EXPD", 1, 40),
            "ICPSR_08599/DS0016/08599-0016-Data.txt": ("FMLY", 2, 1240), "ICPSR_08599/DS0017/08599-0017-Data.txt": ("MEMB", 2, 245), "ICPSR_08599/DS0018/08599-0018-Data.txt": ("EXPD", 2, 40),
            "ICPSR_08599/DS0019/08599-0019-Data.txt": ("FMLY", 3, 1240), "ICPSR_08599/DS0020/08599-0020-Data.txt": ("MEMB", 3, 245), "ICPSR_08599/DS0021/08599-0021-Data.txt": ("EXPD", 3, 40),
            "ICPSR_08599/DS0022/08599-0022-Data.txt": ("FMLY", 4, 1240), "ICPSR_08599/DS0023/08599-0023-Data.txt": ("MEMB", 4, 245), "ICPSR_08599/DS0024/08599-0024-Data.txt": ("EXPD", 4, 40),
        },
    },
    "ICPSR_08628-V1.zip": {
        "anchor": "ICPSR_08628/08628-manifest.txt, DS0001--DS0012 listings (section title, filename, Record Length, Record Count)",
        "members": {
            "ICPSR_08628/DS0001/08628-0001-Data.txt": ("FMLY", 1, 1485), "ICPSR_08628/DS0002/08628-0002-Data.txt": ("MEMB", 1, 245), "ICPSR_08628/DS0003/08628-0003-Data.txt": ("EXPD", 1, 40),
            "ICPSR_08628/DS0004/08628-0004-Data.txt": ("FMLY", 2, 1485), "ICPSR_08628/DS0005/08628-0005-Data.txt": ("MEMB", 2, 245), "ICPSR_08628/DS0006/08628-0006-Data.txt": ("EXPD", 2, 40),
            "ICPSR_08628/DS0007/08628-0007-Data.txt": ("FMLY", 3, 1485), "ICPSR_08628/DS0008/08628-0008-Data.txt": ("MEMB", 3, 245), "ICPSR_08628/DS0009/08628-0009-Data.txt": ("EXPD", 3, 40),
            "ICPSR_08628/DS0010/08628-0010-Data.txt": ("FMLY", 4, 1485), "ICPSR_08628/DS0011/08628-0011-Data.txt": ("MEMB", 4, 245), "ICPSR_08628/DS0012/08628-0012-Data.txt": ("EXPD", 4, 40),
        },
    },
    "ICPSR_08905-V1.zip": {
        "anchor": "ICPSR_08905/08905-manifest.txt, DS0001--DS0012 listings (section title, filename, Record Length, Record Count)",
        "members": {
            "ICPSR_08905/DS0001/08905-0001-Data.txt": ("FMLY", 1, 1252), "ICPSR_08905/DS0002/08905-0002-Data.txt": ("MEMB", 1, 245), "ICPSR_08905/DS0003/08905-0003-Data.txt": ("EXPD", 1, 40),
            "ICPSR_08905/DS0004/08905-0004-Data.txt": ("FMLY", 2, 1252), "ICPSR_08905/DS0005/08905-0005-Data.txt": ("MEMB", 2, 245), "ICPSR_08905/DS0006/08905-0006-Data.txt": ("EXPD", 2, 40),
            "ICPSR_08905/DS0007/08905-0007-Data.txt": ("FMLY", 3, 1252), "ICPSR_08905/DS0008/08905-0008-Data.txt": ("MEMB", 3, 245), "ICPSR_08905/DS0009/08905-0009-Data.txt": ("EXPD", 3, 40),
            "ICPSR_08905/DS0010/08905-0010-Data.txt": ("FMLY", 4, 1252), "ICPSR_08905/DS0011/08905-0011-Data.txt": ("MEMB", 4, 245), "ICPSR_08905/DS0012/08905-0012-Data.txt": ("EXPD", 4, 40),
        },
    },
    "ICPSR_09114-V1.zip": {
        "anchor": "ICPSR_09114/09114-manifest.txt, DS0001--DS0016 listings (section title, filename, Record Length, Record Count)",
        "members": {
            "ICPSR_09114/DS0001/09114-0001-Data.txt": ("FMLY", 1, 1526), "ICPSR_09114/DS0002/09114-0002-Data.txt": ("MEMB", 1, 243), "ICPSR_09114/DS0003/09114-0003-Data.txt": ("EXPD", 1, 38), "ICPSR_09114/DS0004/09114-0004-Data.txt": ("DTBD", 1, 28),
            "ICPSR_09114/DS0005/09114-0005-Data.txt": ("FMLY", 2, 1526), "ICPSR_09114/DS0006/09114-0006-Data.txt": ("MEMB", 2, 243), "ICPSR_09114/DS0007/09114-0007-Data.txt": ("EXPD", 2, 38), "ICPSR_09114/DS0008/09114-0008-Data.txt": ("DTBD", 2, 28),
            "ICPSR_09114/DS0009/09114-0009-Data.txt": ("FMLY", 3, 1526), "ICPSR_09114/DS0010/09114-0010-Data.txt": ("MEMB", 3, 243), "ICPSR_09114/DS0011/09114-0011-Data.txt": ("EXPD", 3, 38), "ICPSR_09114/DS0012/09114-0012-Data.txt": ("DTBD", 3, 28),
            "ICPSR_09114/DS0013/09114-0013-Data.txt": ("FMLY", 4, 1526), "ICPSR_09114/DS0014/09114-0014-Data.txt": ("MEMB", 4, 243), "ICPSR_09114/DS0015/09114-0015-Data.txt": ("EXPD", 4, 38), "ICPSR_09114/DS0016/09114-0016-Data.txt": ("DTBD", 4, 28),
        },
    },
    "ICPSR_09333-V1.zip": {
        "anchor": "ICPSR_09333/09333-manifest.txt, DS0001--DS0016 listings (section title, filename, Record Length, Record Count)",
        "members": {
            "ICPSR_09333/DS0001/09333-0001-Data.txt": ("FMLY", 1, 1526), "ICPSR_09333/DS0002/09333-0002-Data.txt": ("MEMB", 1, 243), "ICPSR_09333/DS0003/09333-0003-Data.txt": ("EXPD", 1, 38), "ICPSR_09333/DS0004/09333-0004-Data.txt": ("DTBD", 1, 28),
            "ICPSR_09333/DS0005/09333-0005-Data.txt": ("FMLY", 2, 1526), "ICPSR_09333/DS0006/09333-0006-Data.txt": ("MEMB", 2, 243), "ICPSR_09333/DS0007/09333-0007-Data.txt": ("EXPD", 2, 38), "ICPSR_09333/DS0008/09333-0008-Data.txt": ("DTBD", 2, 28),
            "ICPSR_09333/DS0009/09333-0009-Data.txt": ("FMLY", 3, 1526), "ICPSR_09333/DS0010/09333-0010-Data.txt": ("MEMB", 3, 243), "ICPSR_09333/DS0011/09333-0011-Data.txt": ("EXPD", 3, 38), "ICPSR_09333/DS0012/09333-0012-Data.txt": ("DTBD", 3, 28),
            "ICPSR_09333/DS0013/09333-0013-Data.txt": ("FMLY", 4, 1526), "ICPSR_09333/DS0014/09333-0014-Data.txt": ("MEMB", 4, 243), "ICPSR_09333/DS0015/09333-0015-Data.txt": ("EXPD", 4, 38), "ICPSR_09333/DS0016/09333-0016-Data.txt": ("DTBD", 4, 28),
        },
    },
    "ICPSR_09570-V1.zip": {
        "anchor": "ICPSR_09570/09570-manifest.txt, DS0001--DS0016 listings (section title, filename, Record Length, Record Count)",
        "members": {
            "ICPSR_09570/DS0001/09570-0001-Data.txt": ("FMLY", 1, 1526), "ICPSR_09570/DS0002/09570-0002-Data.txt": ("MEMB", 1, 263), "ICPSR_09570/DS0003/09570-0003-Data.txt": ("EXPD", 1, 38), "ICPSR_09570/DS0004/09570-0004-Data.txt": ("DTBD", 1, 28),
            "ICPSR_09570/DS0005/09570-0005-Data.txt": ("FMLY", 2, 1526), "ICPSR_09570/DS0006/09570-0006-Data.txt": ("MEMB", 2, 263), "ICPSR_09570/DS0007/09570-0007-Data.txt": ("EXPD", 2, 38), "ICPSR_09570/DS0008/09570-0008-Data.txt": ("DTBD", 2, 28),
            "ICPSR_09570/DS0009/09570-0009-Data.txt": ("FMLY", 3, 1526), "ICPSR_09570/DS0010/09570-0010-Data.txt": ("MEMB", 3, 263), "ICPSR_09570/DS0011/09570-0011-Data.txt": ("EXPD", 3, 38), "ICPSR_09570/DS0012/09570-0012-Data.txt": ("DTBD", 3, 28),
            "ICPSR_09570/DS0013/09570-0013-Data.txt": ("FMLY", 4, 1526), "ICPSR_09570/DS0014/09570-0014-Data.txt": ("MEMB", 4, 263), "ICPSR_09570/DS0015/09570-0015-Data.txt": ("EXPD", 4, 38), "ICPSR_09570/DS0016/09570-0016-Data.txt": ("DTBD", 4, 28),
        },
    },
    "ICPSR_09714-V1.zip": {
        "anchor": "ICPSR_09714/09714-manifest.txt, DS0001--DS0016 listings (section title, filename, Record Length, Record Count)",
        "members": {
            "ICPSR_09714/DS0001/09714-0001-Data.txt": ("FMLY", 1, 1533), "ICPSR_09714/DS0002/09714-0002-Data.txt": ("MEMB", 1, 263), "ICPSR_09714/DS0003/09714-0003-Data.txt": ("EXPD", 1, 38), "ICPSR_09714/DS0004/09714-0004-Data.txt": ("DTBD", 1, 28),
            "ICPSR_09714/DS0005/09714-0005-Data.txt": ("FMLY", 2, 1533), "ICPSR_09714/DS0006/09714-0006-Data.txt": ("MEMB", 2, 263), "ICPSR_09714/DS0007/09714-0007-Data.txt": ("EXPD", 2, 38), "ICPSR_09714/DS0008/09714-0008-Data.txt": ("DTBD", 2, 28),
            "ICPSR_09714/DS0009/09714-0009-Data.txt": ("FMLY", 3, 1533), "ICPSR_09714/DS0010/09714-0010-Data.txt": ("MEMB", 3, 263), "ICPSR_09714/DS0011/09714-0011-Data.txt": ("EXPD", 3, 38), "ICPSR_09714/DS0012/09714-0012-Data.txt": ("DTBD", 3, 28),
            "ICPSR_09714/DS0013/09714-0013-Data.txt": ("FMLY", 4, 1533), "ICPSR_09714/DS0014/09714-0014-Data.txt": ("MEMB", 4, 263), "ICPSR_09714/DS0015/09714-0015-Data.txt": ("EXPD", 4, 38), "ICPSR_09714/DS0016/09714-0016-Data.txt": ("DTBD", 4, 28),
        },
    },
}

# A combined archive has an explicit member-to-survey-year map.  It is not
# recovered from DS numbers or filenames at runtime.
ICPSR_MEMBER_SURVEY_YEAR_MAP = {
    "ICPSR_08599-V1.zip": {
        "ICPSR_08599/DS0001/08599-0001-Data.txt": 1982, "ICPSR_08599/DS0002/08599-0002-Data.txt": 1982, "ICPSR_08599/DS0003/08599-0003-Data.txt": 1982,
        "ICPSR_08599/DS0004/08599-0004-Data.txt": 1982, "ICPSR_08599/DS0005/08599-0005-Data.txt": 1982, "ICPSR_08599/DS0006/08599-0006-Data.txt": 1982,
        "ICPSR_08599/DS0007/08599-0007-Data.txt": 1982, "ICPSR_08599/DS0008/08599-0008-Data.txt": 1982, "ICPSR_08599/DS0009/08599-0009-Data.txt": 1982,
        "ICPSR_08599/DS0010/08599-0010-Data.txt": 1982, "ICPSR_08599/DS0011/08599-0011-Data.txt": 1982, "ICPSR_08599/DS0012/08599-0012-Data.txt": 1982,
        "ICPSR_08599/DS0013/08599-0013-Data.txt": 1983, "ICPSR_08599/DS0014/08599-0014-Data.txt": 1983, "ICPSR_08599/DS0015/08599-0015-Data.txt": 1983,
        "ICPSR_08599/DS0016/08599-0016-Data.txt": 1983, "ICPSR_08599/DS0017/08599-0017-Data.txt": 1983, "ICPSR_08599/DS0018/08599-0018-Data.txt": 1983,
        "ICPSR_08599/DS0019/08599-0019-Data.txt": 1983, "ICPSR_08599/DS0020/08599-0020-Data.txt": 1983, "ICPSR_08599/DS0021/08599-0021-Data.txt": 1983,
        "ICPSR_08599/DS0022/08599-0022-Data.txt": 1983, "ICPSR_08599/DS0023/08599-0023-Data.txt": 1983, "ICPSR_08599/DS0024/08599-0024-Data.txt": 1983,
    },
}

# The manifest lists an 80-byte physical record for this one file.  The
# embedded 1986 documentation identifies its logical EXPD record as 38 bytes;
# source inventory records both facts and does not parse the file.
ICPSR_MANIFEST_PHYSICAL_WIDTH_EXCEPTIONS = {
    ("ICPSR_09114-V1.zip", "ICPSR_09114/DS0015/09114-0015-Data.txt"): 80,
}


def sha256_zip_member(archive: zipfile.ZipFile, member_name: str) -> str:
    digest = hashlib.sha256()
    with archive.open(member_name) as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_text(raw: bytes) -> str:
    return raw.decode("latin1", errors="replace")


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def release_years(release: dict[str, object]) -> tuple[int, ...]:
    return tuple(int(year) for year in release.get("survey_years", (release["survey_year"],)))


def member_survey_year(archive_name: str, release: dict[str, object], member_name: str) -> int:
    explicit_map = ICPSR_MEMBER_SURVEY_YEAR_MAP.get(archive_name)
    if explicit_map is None:
        if len(release_years(release)) != 1:
            raise RuntimeError(
                f"CEX Diary source inventory: combined ICPSR archive {archive_name} lacks an explicit member-year map."
            )
        return release_years(release)[0]
    if member_name not in explicit_map:
        raise RuntimeError(
            f"CEX Diary source inventory: mapped ICPSR primary member {member_name} has no explicit survey year."
        )
    return int(explicit_map[member_name])


def source_contract(year: int, source_provider: str) -> dict[str, object]:
    """Require the canonical provider for each retained year."""
    if year not in CANONICAL_YEARS:
        raise RuntimeError(f"CEX Diary: unsupported source year {year}.")
    provider = "ICPSR" if year <= 1989 else "BLS"
    if source_provider != provider:
        raise RuntimeError(f"CEX Diary {year} requires {provider}, not {source_provider}.")
    return {
        "canonical_source_provider": provider,
        "source_role": "canonical",
        "intended_intermediate_status": "in_intended_1982_2011_span",
    }


def expected_archive_paths() -> dict[Path, dict[str, object]]:
    expected: dict[Path, dict[str, object]] = {}
    for archive_name, release in BLS_DOCUMENTED_RELEASES.items():
        expected[BLS_DIR / archive_name] = {"source_provider": "BLS", **release}
    for archive_name, release in ICPSR_DOCUMENTED_RELEASES.items():
        expected[ICPSR_DIR / archive_name] = {"source_provider": "ICPSR", **release}
    return expected


def verify_expected_archives(expected: dict[Path, dict[str, object]]) -> None:
    missing = sorted(path for path in expected if not path.is_file())
    if missing:
        raise RuntimeError(
            f"CEX Diary: missing canonical archives: {[str(path) for path in missing]}."
        )


def classify_bls_member(member_name: str) -> tuple[str, str, str]:
    name = Path(member_name).name.lower()
    match = re.fullmatch(r"(fmld|memd|expd|dtbd|dtid|dhhid)(\d+)\.csv", name)
    if match:
        raw_family, token = match.groups()
        family = {"fmld": "FMLY", "memd": "MEMB"}.get(raw_family, raw_family.upper())
        quarter = token[-1] if len(token) >= 3 else "not_quartered"
        return family, quarter, "csv"
    if name.endswith(".csv"):
        return "OTHER_CSV", "not_applicable", "csv"
    if name.endswith(".txt"):
        return "AUXILIARY_TEXT", "not_applicable", "text"
    return "AUXILIARY", "not_applicable", "binary_or_other"


def classify_icpsr_member(archive_name: str, member_name: str) -> tuple[str, str, str, int | None, str]:
    member_map = dict(ICPSR_PRIMARY_MEMBER_MAP[archive_name]["members"])
    if member_name in member_map:
        family, quarter, logical_width = member_map[member_name]
        return family, str(quarter), "fixed_width", int(logical_width), str(
            ICPSR_PRIMARY_MEMBER_MAP[archive_name]["anchor"]
        )
    lower = member_name.lower()
    if "documentation" in lower or "codebook" in lower or "descriptioncitation" in lower:
        return "DOCUMENTATION", "not_applicable", "documentation", None, ""
    if lower.endswith((".txt", ".html", ".doc", ".pdf")):
        return "AUXILIARY_DOCUMENTATION_OR_TEXT", "not_applicable", "documentation_or_text", None, ""
    return "AUXILIARY", "not_applicable", "binary_or_other", None, ""


def inspect_data_member(
    archive: zipfile.ZipFile,
    member_name: str,
    layout_type: str,
) -> tuple[str, int | None, str, str, list[str]]:
    """Return member hash, raw rows, observed layout, header hash, and columns."""
    digest = hashlib.sha256()
    line_count = 0
    widths: Counter[int] = Counter()
    header_line: bytes | None = None

    with archive.open(member_name) as handle:
        if layout_type == "fixed_width":
            for line in handle:
                digest.update(line)
                line_count += 1
                widths[len(line.rstrip(b"\r\n"))] += 1
        elif layout_type == "csv":
            for line in handle:
                digest.update(line)
                line_count += 1
                if header_line is None:
                    header_line = line
        else:
            for block in iter(lambda: handle.read(1_048_576), b""):
                digest.update(block)

    member_sha256 = digest.hexdigest()
    if layout_type == "fixed_width":
        observed_layout = "|".join(
            f"{width}:{count}" for width, count in sorted(widths.items())
        )
        return member_sha256, line_count, observed_layout, "", []

    if layout_type == "csv":
        if header_line is None:
            raise RuntimeError(f"CEX Diary source inventory: empty CSV member: {member_name}")
        header_text = header_line.decode("utf-8-sig", errors="strict").rstrip("\r\n")
        columns = next(csv.reader([header_text]))
        if not columns or any(not column.strip() for column in columns):
            raise RuntimeError(f"CEX Diary source inventory: invalid CSV header in {member_name}")
        header_sha256 = hashlib.sha256(header_text.encode("utf-8")).hexdigest()
        return member_sha256, line_count - 1, f"csv_columns:{len(columns)}", header_sha256, columns

    return member_sha256, None, "not_applicable", "", []


def assert_icpsr_documentation(
    archive: zipfile.ZipFile,
    release: dict[str, object],
    archive_path: Path,
) -> None:
    names = set(archive.namelist())
    year_member = str(release["year_evidence_member"])
    layout_members = tuple(
        str(member_name)
        for member_name in release.get("layout_evidence_members", (release["layout_evidence_member"],))
    )
    for member_name in [year_member, *layout_members]:
        if member_name not in names:
            raise RuntimeError(
                f"CEX Diary source inventory: {relative(archive_path)} lacks required embedded documentation {member_name}."
            )
    evidence = decode_text(archive.read(year_member))
    year_evidence_text = str(release["year_evidence_text"])
    if year_evidence_text.casefold() not in evidence.casefold():
        raise RuntimeError(
            f"CEX Diary source inventory: {relative(archive_path)} does not document its mapped year "
            f"with {year_evidence_text!r} in {year_member}."
        )


def parse_icpsr_manifest_member_map(
    archive: zipfile.ZipFile,
    archive_path: Path,
    release: dict[str, object],
) -> dict[str, tuple[int, int]]:
    """Validate the literal primary-member map against the embedded manifest."""
    archive_name = archive_path.name
    if archive_name not in ICPSR_PRIMARY_MEMBER_MAP:
        raise RuntimeError(f"CEX Diary source inventory: no primary-member map for {relative(archive_path)}.")
    mapped = ICPSR_PRIMARY_MEMBER_MAP[archive_name]
    member_map = dict(mapped["members"])
    manifest_member = f"ICPSR_{release['study_id']}/{release['study_id']}-manifest.txt"
    names = set(archive.namelist())
    if manifest_member not in names:
        raise RuntimeError(
            f"CEX Diary source inventory: {relative(archive_path)} lacks manifest anchor {manifest_member}."
        )
    mapped_families = tuple(release.get("mapped_file_families", ("FMLY", "MEMB", "EXPD", "DTBD")))
    release_parts = {
        year: tuple(release.get("release_parts_by_year", {}).get(year, (1, 2, 3, 4)))
        for year in release_years(release)
    }
    required = {
        (year, family, str(part))
        for year, parts in release_parts.items()
        for family in mapped_families
        for part in parts
    }
    mapped_family_release_parts = Counter(
        (member_survey_year(archive_name, release, member_name), family, str(part))
        for member_name, (family, part, _) in member_map.items()
    )
    if set(mapped_family_release_parts) != required or any(
        count != 1 for count in mapped_family_release_parts.values()
    ):
        raise RuntimeError(
            f"CEX Diary source inventory: {relative(archive_path)} ICPSR primary-member map is not one member for each documented family-release part."
        )
    missing = sorted(set(member_map) - names)
    if missing:
        raise RuntimeError(
            f"CEX Diary source inventory: {relative(archive_path)} lacks mapped primary members {missing}."
        )

    manifest_text = decode_text(archive.read(manifest_member))
    manifest_rows: dict[str, tuple[int, int]] = {}
    for member_name, (_, _, logical_width) in member_map.items():
        filename = Path(member_name).name
        match = re.search(
            rf"^\s*{re.escape(filename)}\s+(\d+)\s+([\d,]+)\s+",
            manifest_text,
            flags=re.MULTILINE,
        )
        if match is None:
            raise RuntimeError(
                f"CEX Diary source inventory: {relative(archive_path)} manifest does not list mapped member {member_name}."
            )
        manifest_width = int(match.group(1))
        manifest_count = int(match.group(2).replace(",", ""))
        expected_physical_width = ICPSR_MANIFEST_PHYSICAL_WIDTH_EXCEPTIONS.get(
            (archive_name, member_name), int(logical_width)
        )
        if manifest_width != expected_physical_width:
            raise RuntimeError(
                "CEX Diary source inventory: manifest record width disagrees with the explicit primary-member map: "
                f"{member_name}; manifest={manifest_width}; expected physical={expected_physical_width}; "
                f"logical={logical_width}."
            )
        manifest_rows[member_name] = (manifest_width, manifest_count)

    return manifest_rows


def layout_review_status(
    source_provider: str,
    family: str,
    member_name: str,
    observed_layout: str,
    release: dict[str, object],
    documented_logical_width: int | None,
) -> tuple[str, int | None]:
    if source_provider == "BLS":
        return "csv_header_observed", None
    if family not in {"FMLY", "MEMB", "EXPD", "DTBD"}:
        return "not_a_primary_fixed_width_member", None

    if documented_logical_width is None:
        raise RuntimeError(
            f"CEX Diary source inventory: no explicit logical width for mapped ICPSR member {member_name}."
        )
    logical_width = documented_logical_width
    observed_widths = {int(part.split(":", 1)[0]) for part in observed_layout.split("|")}
    if observed_widths == {logical_width}:
        return "matches_documented_logical_width", logical_width
    if observed_widths == {logical_width + 1}:
        return "physical_line_is_one_byte_longer_than_documented_logical_width", logical_width
    if (
        str(release["study_id"]) == "09114"
        and family == "EXPD"
        and member_name.endswith("/DS0015/09114-0015-Data.txt")
        and observed_widths == {80}
    ):
        return "documented_expd_logical_width_38_with_observed_padded_80_byte_q4_record", logical_width
    raise RuntimeError(
        "CEX Diary source inventory: fixed-width member does not match its documented layout: "
        f"{member_name}; observed={observed_layout}; documented logical width={logical_width}."
    )


def build_inventory(
    expected: dict[Path, dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[tuple[int, str], set[str]]]:
    archive_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    documentation_rows: list[dict[str, object]] = []
    headers_by_year_source_family: dict[tuple[int, str], set[str]] = defaultdict(set)

    for archive_path, release in sorted(expected.items(), key=lambda item: (int(item[1]["survey_year"]), item[0].name)):
        source_provider = str(release["source_provider"])
        survey_year = int(release["survey_year"])
        archive_years = release_years(release)
        archive_sha256 = sha256_file(archive_path)

        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise RuntimeError(f"CEX Diary source inventory: corrupt ZIP member in {relative(archive_path)}.")
            if source_provider == "ICPSR":
                assert_icpsr_documentation(archive, release, archive_path)
                manifest_rows = parse_icpsr_manifest_member_map(archive, archive_path, release)
                manifest_member = f"ICPSR_{release['study_id']}/{release['study_id']}-manifest.txt"
                member_map_anchor = str(ICPSR_PRIMARY_MEMBER_MAP[archive_path.name]["anchor"])
            else:
                manifest_rows = {}
                manifest_member = ""
                member_map_anchor = ""

            archive_members = [info for info in archive.infolist() if not info.is_dir()]
            family_member_counts: Counter[str] = Counter()
            primary_family_counts: Counter[str] = Counter()
            primary_family_counts_by_year: Counter[tuple[int, str]] = Counter()
            for info in archive_members:
                if source_provider == "BLS":
                    family, quarter, layout_type = classify_bls_member(info.filename)
                    mapped_logical_width = None
                    mapped_member_anchor = ""
                    member_year = survey_year
                else:
                    family, quarter, layout_type, mapped_logical_width, mapped_member_anchor = classify_icpsr_member(
                        archive_path.name, info.filename
                    )
                    member_year = (
                        member_survey_year(archive_path.name, release, info.filename)
                        if info.filename in ICPSR_PRIMARY_MEMBER_MAP[archive_path.name]["members"]
                        else survey_year
                    )

                if layout_type in {"csv", "fixed_width"}:
                    member_sha256, raw_rows, observed_layout, header_sha256, columns = inspect_data_member(
                        archive, info.filename, layout_type
                    )
                else:
                    member_sha256 = sha256_zip_member(archive, info.filename)
                    raw_rows, observed_layout, header_sha256, columns = None, "not_applicable", "", []

                status, documented_logical_width = layout_review_status(
                    source_provider,
                    family,
                    info.filename,
                    observed_layout,
                    release,
                    mapped_logical_width,
                )
                manifest_record_width: int | None = None
                manifest_record_count: int | None = None
                if info.filename in manifest_rows:
                    manifest_record_width, manifest_record_count = manifest_rows[info.filename]
                    if raw_rows != manifest_record_count:
                        raise RuntimeError(
                            "CEX Diary source inventory: raw row count disagrees with the embedded ICPSR manifest: "
                            f"{info.filename}; raw={raw_rows}; manifest={manifest_record_count}."
                        )
                if family in PRIMARY_FAMILIES:
                    primary_family_counts[family] += 1
                    primary_family_counts_by_year[(member_year, family)] += 1
                family_member_counts[family] += 1
                if source_provider == "BLS" and family in PRIMARY_FAMILIES:
                    headers_by_year_source_family[(member_year, family)].update(
                        column.strip().upper() for column in columns
                    )

                member_rows.append(
                    {
                        "source_provider": source_provider,
                        "survey_year": member_year,
                        "archive_path": relative(archive_path),
                        "archive_sha256": archive_sha256,
                        "member_path": info.filename,
                        "member_sha256": member_sha256,
                        "uncompressed_bytes": info.file_size,
                        "compressed_bytes": info.compress_size,
                        "file_family": family,
                        "quarter_or_release_part": quarter,
                        "layout_type": layout_type,
                        "raw_row_count": raw_rows,
                        "observed_layout": observed_layout,
                        "header_sha256": header_sha256,
                        "header_columns": "|".join(columns),
                        "documented_logical_record_width": documented_logical_width,
                        "manifest_record_width": manifest_record_width,
                        "manifest_record_count": manifest_record_count,
                        "primary_member_map_anchor": mapped_member_anchor,
                        "layout_review_status": status,
                    }
                )

                if "documentation" in family.casefold() or "codebook" in info.filename.casefold() or "descriptioncitation" in info.filename.casefold():
                    documentation_rows.append(
                        {
                            "source_provider": source_provider,
                            "survey_year": survey_year,
                            "archive_path": relative(archive_path),
                            "documentation_path": f"{relative(archive_path)}::{info.filename}",
                            "documentation_sha256": member_sha256,
                            "documentation_kind": "embedded_archive_member",
                            "use_in_source_inventory": "archive-year or fixed-width-layout evidence",
                        }
                    )

            required_families = {"FMLY", "MEMB", "EXPD"}
            missing_families = sorted(required_families - set(primary_family_counts))
            if missing_families:
                raise RuntimeError(
                    f"CEX Diary source inventory: {relative(archive_path)} lacks required file families {missing_families}."
                )

            if source_provider == "ICPSR":
                documentation_rows.append(
                    {
                        "source_provider": source_provider,
                        "survey_year": survey_year,
                        "archive_path": relative(archive_path),
                        "documentation_path": f"{relative(archive_path)}::{release['year_evidence_member']}",
                        "documentation_sha256": sha256_zip_member(archive, str(release["year_evidence_member"])),
                        "documentation_kind": "embedded_year_mapping_evidence",
                        "use_in_source_inventory": str(release["year_evidence_text"]),
                    }
                )
                documentation_rows.append(
                    {
                        "source_provider": source_provider,
                        "survey_year": survey_year,
                        "archive_path": relative(archive_path),
                        "documentation_path": f"{relative(archive_path)}::{manifest_member}",
                        "documentation_sha256": sha256_zip_member(archive, manifest_member),
                        "documentation_kind": "embedded_primary_member_map_anchor",
                        "use_in_source_inventory": member_map_anchor,
                    }
                )
                for layout_member in release.get("layout_evidence_members", (release["layout_evidence_member"],)):
                    documentation_rows.append(
                        {
                            "source_provider": source_provider,
                            "survey_year": survey_year,
                            "archive_path": relative(archive_path),
                            "documentation_path": f"{relative(archive_path)}::{layout_member}",
                            "documentation_sha256": sha256_zip_member(archive, str(layout_member)),
                            "documentation_kind": "embedded_fixed_width_layout_evidence",
                            "use_in_source_inventory": "documented FMLY, MEMB, EXPD, and DTBD logical record widths",
                        }
                    )
            for archive_year in archive_years:
                contract = source_contract(archive_year, source_provider)
                archive_rows.append(
                    {
                        "source_provider": source_provider,
                        "survey_year": archive_year,
                        "archive_path": relative(archive_path),
                        "archive_sha256": archive_sha256,
                        "archive_member_count": len(archive_members),
                        "fml_family_member_count": primary_family_counts_by_year[(archive_year, "FMLY")],
                        "memb_family_member_count": primary_family_counts_by_year[(archive_year, "MEMB")],
                        "expd_family_member_count": primary_family_counts_by_year[(archive_year, "EXPD")],
                        "archive_year_evidence": str(
                            release["year_evidence_text"]
                            if "year_evidence_text" in release
                            else release["year_evidence"]
                        ),
                        "layout_evidence": str(
                            release["layout_evidence_member"]
                            if "layout_evidence_member" in release
                            else release["layout_evidence"]
                        ),
                        "primary_member_map_anchor": member_map_anchor,
                        "canonical_source_provider": contract["canonical_source_provider"],
                        "source_role": contract["source_role"],
                        "intended_intermediate_status": contract["intended_intermediate_status"],
                        "metadata_consistency_status": "no_known_archive_metadata_conflict",
                        "metadata_consistency_detail": "",
                        "source_inventory_status": "documented_for_source_inventory_review",
                    }
                )

    return archive_rows, member_rows, documentation_rows, headers_by_year_source_family


# construction deliberately uses literal, documented selected-field maps.  Nothing
# below attempts to recover a fixed-width location from a filename or a data
# value.  Positions are zero-based, end-exclusive equivalents of the embedded
# ICPSR Documentation field starts and formats.
FMLY_FIELDS = [
    "NEWID", "AGE_REF", "BLS_URBN", "CUTENURE", "EDUC_REF", "EMPLTYP1",
    "FAM_SIZE", "FINCBEFX", "FINLWT21", "FSS_RRX", "FSUPPX", "INCLASS",
    "MARITAL1", "PERSLT18", "PICK_UP", "REF_RACE", "REGION", "RESPSTAT",
    "SEX_REF", "SMSASTAT", "STRTDAY", "STRTMNTH", "STRTYEAR", "WEEKI", "WEEKN",
]
MEMB_FIELDS = [
    "NEWID", "MEMBNO", "ANYRAIL", "ANYSSINC", "US_SUPP", "SS_RRX", "SUPPX",
    "CU_CODE1", "WHYNOWRK",
]
EXPD_FIELDS = ["NEWID", "COST", "QREDATE", "UCC"]


def classify_stephens_ucc(source_year: pd.Series, ucc_number: pd.Series) -> pd.DataFrame:
    """Classify UCCs using the exact 1986--1996 Stephens replication rules."""
    year = pd.to_numeric(source_year, errors="raise").astype(int)
    ucc = pd.to_numeric(ucc_number, errors="coerce")
    valid = ucc.notna()
    food_home_base = valid & (ucc.between(10110, 180710) | ucc.eq(200112))
    alcohol_home = valid & (ucc.isin([200110, 200111]) | ucc.between(200210, 200410))
    food_away_base = valid & (ucc.between(190110, 190320) | ucc.isin([190901, 190902]))
    alcohol_away_upper = pd.Series(200530, index=ucc.index).where(year.le(1993), 200536)
    alcohol_away = valid & ucc.ge(200510) & ucc.le(alcohol_away_upper)
    return pd.DataFrame(
        {
            "included_broad_total": valid,
            "included_food_at_home": food_home_base | alcohol_home,
            "included_food_away_from_home": food_away_base | alcohol_away,
            "paper_food_home_base": food_home_base,
            "paper_alcohol_home": alcohol_home,
            "paper_food_away_base": food_away_base,
            "paper_alcohol_away": alcohol_away,
        },
        index=ucc.index,
    )

ICPSR_FMLY_COLSPECS_BY_YEAR = {
    # ICPSR_08599/08599-Documentation.txt: FMLY entries at lines 959--1976.
    1982: [(0, 8), (35, 37), None, (285, 286), (310, 311), (314, 315), (318, 320),
           (379, 387), (261, 272), (573, 581), (591, 599), (628, 629), (699, 700),
           (774, 776), (789, 791), (808, 809), (810, 811), (812, 813), (832, 833),
           (836, 837), (855, 857), (857, 859), (859, 861), (884, 885), (886, 887)],
    1983: [(0, 8), (35, 37), None, (285, 286), (310, 311), (314, 315), (318, 320),
           (379, 387), (261, 272), (573, 581), (591, 599), (628, 629), (699, 700),
           (774, 776), (789, 791), (808, 809), (810, 811), (812, 813), (832, 833),
           (836, 837), (855, 857), (857, 859), (859, 861), (884, 885), (886, 887)],
    # ICPSR_08628/08628-Documentation.txt: FMLY entries at lines 948--2049.
    1984: [(0, 8), (35, 37), (41, 42), (286, 287), (311, 312), (315, 316), (319, 321),
           (380, 388), (609, 620), (805, 813), (823, 831), (870, 871), (941, 942),
           (1016, 1018), (1031, 1033), (1050, 1051), (1052, 1053), (1054, 1055),
           (1074, 1075), (1078, 1079), (1097, 1099), (1099, 1101), (1101, 1103),
           (1126, 1127), (1128, 1129)],
    # ICPSR_08905/08905-Documentation.txt: FMLY entries at lines 916--1934.
    1985: [(0, 8), (35, 37), (41, 42), (55, 56), (80, 81), (84, 85), (88, 90),
           (149, 157), (378, 389), (574, 582), (592, 600), (639, 640), (710, 711),
           (785, 787), (800, 802), (819, 820), (821, 822), (823, 824), (843, 844),
           (847, 848), (866, 868), (868, 870), (870, 872), (895, 896), (897, 898)],
    # ICPSR_09114--09714 embedded Documentation FMLY dictionary.  These are
    # the published selected positions previously used for the frozen 1986--89
    # Stephens construction; they are now explicit source maps.
    1986: [(0, 8), (35, 37), (41, 42), (55, 56), (80, 81), (84, 85), (88, 90),
           (149, 157), (158, 169), (354, 362), (372, 380), (419, 420), (490, 491),
           (565, 567), (580, 582), (599, 600), (601, 602), (603, 604), (623, 624),
           (627, 628), (646, 648), (648, 650), (650, 652), (675, 676), (677, 678)],
    1987: [(0, 8), (35, 37), (41, 42), (55, 56), (80, 81), (84, 85), (88, 90),
           (149, 157), (158, 169), (354, 362), (372, 380), (419, 420), (490, 491),
           (565, 567), (580, 582), (599, 600), (601, 602), (603, 604), (623, 624),
           (627, 628), (646, 648), (648, 650), (650, 652), (675, 676), (677, 678)],
    1988: [(0, 8), (35, 37), (41, 42), (55, 56), (80, 81), (84, 85), (88, 90),
           (149, 157), (158, 169), (354, 362), (372, 380), (419, 420), (490, 491),
           (565, 567), (580, 582), (599, 600), (601, 602), (603, 604), (623, 624),
           (627, 628), (646, 648), (648, 650), (650, 652), (675, 676), (677, 678)],
    1989: [(0, 8), (35, 37), (41, 42), (55, 56), (80, 81), (84, 85), (88, 90),
           (149, 157), (158, 169), (354, 362), (372, 380), (419, 420), (490, 491),
           (565, 567), (580, 582), (599, 600), (601, 602), (603, 604), (623, 624),
           (627, 628), (646, 648), (648, 650), (650, 652), (675, 676), (677, 678)],
}
ICPSR_MEMB_COLSPECS_BY_YEAR = {
    year: [(0, 8), (148, 150), (56, 57), (58, 59), (227, 228), (198, 206), (218, 226), (71, 72), (238, 239)]
    for year in range(1982, 1989)
} | {
    1989: [(0, 8), (148, 150), (56, 57), (58, 59), (225, 226), (196, 204), (216, 224), (71, 72), (236, 237)]
}
ICPSR_EXPD_COLSPECS_BY_YEAR = {
    year: [(0, 8), (9, 21), (23, 31), (32, 38)] for year in ICPSR_YEARS
}
ICPSR_CROSSWALK_ANCHORS = {
    1982: "ICPSR_08599/08599-Documentation.txt: FMLY 959--1976; MEMB 2127--2472; EXPD 2501--2539",
    1983: "ICPSR_08599/08599-Documentation.txt: FMLY 959--1976; MEMB 2127--2472; EXPD 2501--2539",
    1984: "ICPSR_08628/08628-Documentation.txt: FMLY 948--2049; MEMB 2200--2547; EXPD 2576--2613",
    1985: "ICPSR_08905/08905-Documentation.txt: FMLY 916--1934; MEMB 2085--2432; EXPD 2461--2498",
    1986: "ICPSR_09114/09114-Documentation.txt: FMLY, MEMB, and EXPD variable dictionary entries",
    1987: "ICPSR_09333/09333-Documentation.txt: FMLY, MEMB, and EXPD variable dictionary entries",
    1988: "ICPSR_09570/09570-Documentation.txt: FMLY, MEMB, and EXPD variable dictionary entries",
    1989: "ICPSR_09714/09714-Documentation.txt: FMLY, MEMB, and EXPD variable dictionary entries",
}

# These aliases are documented by the 2004--05 released headers and the CEX
# Dictionary.  Record them in each output row rather than silently renaming.
BLS_FMLY_SOURCE_ALIASES = {
    "FINCBEFX": {2004: "FINCBEFM", 2005: "FINCBEFM"},
    "FSS_RRX": {2004: "FSS_RRXM", 2005: "FSS_RRXM"},
    "FSUPPX": {2004: "FSUPPXM", 2005: "FSUPPXM"},
    "PICK_UP": {year: "PICKCODE" for year in range(2004, 2012)},
}
BLS_MEMB_SOURCE_ALIASES = {
    "SS_RRX": {2004: "SS_RRXM", 2005: "SS_RRXM"},
    "SUPPX": {2004: "SUPPXM", 2005: "SUPPXM"},
}
BLS_IMPUTED_OR_COLLECTED_FMLY_FIELDS = ["FINCBEFM", "FSS_RRXM", "FSUPPXM"]

SEX_LABEL = {"1": "male", "2": "female"}
RACE_LABEL = {"1": "White", "2": "Black", "3": "American Indian, Aleut, or Eskimo", "4": "Asian or Pacific Islander", "5": "other race"}
RACE_LABEL_2003_PLUS = {"1": "White", "2": "Black", "3": "American Indian, Aleut, or Eskimo", "4": "Asian", "5": "Native Hawaiian or Pacific Islander", "6": "multiracial"}
EDUC_LABEL = {"1": "elementary school", "2": "some high school", "3": "high school graduate", "4": "some college", "5": "college graduate", "6": "more than four years of college", "7": "no reported schooling"}
EDUC_LABEL_1996 = {"00": "no reported schooling", "10": "elementary school", "11": "some high school", "12": "high school graduate", "13": "some college", "14": "associate degree or some college", "15": "college graduate", "16": "master's degree", "17": "doctoral or professional degree"}
MARITAL_LABEL = {"1": "married", "2": "widowed", "3": "divorced", "4": "separated", "5": "never married"}
REGION_LABEL = {"1": "Northeast", "2": "Midwest", "3": "South", "4": "West"}
URBAN_LABEL = {"1": "urban area", "2": "rural area"}
TENURE_LABEL = {"1": "owns a home with a mortgage", "2": "owns a home without a mortgage", "3": "owns a home", "4": "rents a home", "5": "occupies housing without cash rent", "6": "lives in college housing"}


def source_archive(year: int) -> Path:
    """Return the literal canonical archive for one approved source year."""
    icpsr_by_year = {
        1982: "ICPSR_08599-V1.zip", 1983: "ICPSR_08599-V1.zip",
        1984: "ICPSR_08628-V1.zip", 1985: "ICPSR_08905-V1.zip",
        1986: "ICPSR_09114-V1.zip", 1987: "ICPSR_09333-V1.zip",
        1988: "ICPSR_09570-V1.zip", 1989: "ICPSR_09714-V1.zip",
    }
    if year in icpsr_by_year:
        return ICPSR_DIR / icpsr_by_year[year]
    if year in BLS_YEARS:
        return BLS_DIR / f"diary{year % 100:02d}.zip"
    raise RuntimeError(f"CEX Diary construction: {year} is not an approved canonical source year.")


def raw_text(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        out[column] = out[column].astype("string").str.strip()
    return out


def normalize_newid(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    raw = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    valid = raw.str.fullmatch(r"[0-9]{1,8}").fillna(False)
    return raw.where(valid).str.zfill(8), valid


def read_fixed(
    archive_path: Path,
    member_name: str,
    colspecs: list[tuple[int, int]],
    names: list[str],
    *,
    require_1986_q4_tail_rule: bool = False,
) -> pd.DataFrame:
    """Read one fixed-width member after its documented layout check."""
    with zipfile.ZipFile(archive_path) as archive:
        payload = archive.read(member_name)
    lines = payload.splitlines()
    if not lines:
        raise RuntimeError(f"CEX Diary construction: empty fixed-width member {member_name}.")
    if require_1986_q4_tail_rule:
        widths = Counter(len(line) for line in lines)
        if widths != Counter({80: 200_008}):
            raise RuntimeError(
                "CEX Diary construction: ICPSR 1986 Q4 EXPD physical-width assertion failed: "
                f"{dict(widths)} rather than {{80: 200008}}."
            )
        if any(line[38:] != b" " * 42 for line in lines):
            raise RuntimeError(
                "CEX Diary construction: ICPSR 1986 Q4 EXPD bytes 39--80 are not all blank; "
                "the documented first-38-byte parser may not proceed."
            )
        payload = b"\n".join(line[:38] for line in lines) + b"\n"
    return pd.read_fwf(
        io.StringIO(payload.decode("latin1")), colspecs=colspecs, names=names,
        dtype="string", header=None,
    )


def icpsr_members(year: int, family: str) -> list[tuple[str, str]]:
    """Return literal mapped member paths, never a filename-derived map."""
    archive_path = source_archive(year)
    member_map = ICPSR_PRIMARY_MEMBER_MAP[archive_path.name]["members"]
    found: list[tuple[str, str]] = []
    for member_name, (mapped_family, release_part, _) in member_map.items():
        if mapped_family != family:
            continue
        if member_survey_year(archive_path.name, ICPSR_DOCUMENTED_RELEASES[archive_path.name], member_name) != year:
            continue
        found.append((member_name, f"Q{release_part}"))
    if len(found) != 4:
        raise RuntimeError(
            f"CEX Diary construction: {archive_path.name} has {len(found)} mapped {family} members for {year}, not four."
        )
    return sorted(found, key=lambda item: item[1])


def read_icpsr_year(year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    archive_path = source_archive(year)
    frames: dict[str, list[pd.DataFrame]] = defaultdict(list)
    input_rows: list[dict[str, object]] = []
    specs = {
        "FMLY": (ICPSR_FMLY_COLSPECS_BY_YEAR[year], FMLY_FIELDS),
        "MEMB": (ICPSR_MEMB_COLSPECS_BY_YEAR[year], MEMB_FIELDS),
        "EXPD": (ICPSR_EXPD_COLSPECS_BY_YEAR[year], EXPD_FIELDS),
    }
    for family, (colspecs_raw, names) in specs.items():
        colspecs = [spec if spec is not None else (0, 0) for spec in colspecs_raw]
        for member_name, quarter in icpsr_members(year, family):
            tail_rule = year == 1986 and family == "EXPD" and quarter == "Q4"
            frame = read_fixed(
                archive_path, member_name, colspecs, names,
                require_1986_q4_tail_rule=tail_rule,
            )
            for position, name in enumerate(names):
                if colspecs_raw[position] is None:
                    frame[name] = pd.Series(pd.NA, index=frame.index, dtype="string")
            frame["source_year"] = year
            frame["source_provider"] = "ICPSR"
            frame["source_family"] = "ICPSR_fixed_width"
            frame["source_archive"] = archive_path.name
            frame["source_quarter_file"] = quarter
            frame["source_member"] = member_name
            frames[family].append(frame)
            input_rows.append({
                "source_year": year, "source_provider": "ICPSR", "source_archive": archive_path.name,
                "source_family": family, "source_member": member_name, "source_quarter_file": quarter,
                "raw_rows": len(frame), "member_sha256": hashlib.sha256(zipfile.ZipFile(archive_path).read(member_name)).hexdigest(),
                "crosswalk_anchor": ICPSR_CROSSWALK_ANCHORS[year],
                "parser_rule": "first_38_bytes_only_after_blank_tail_assertion" if tail_rule else "documented_fixed_width_selected_fields",
            })
    return (
        pd.concat(frames["FMLY"], ignore_index=True), pd.concat(frames["MEMB"], ignore_index=True),
        pd.concat(frames["EXPD"], ignore_index=True), input_rows,
    )


def bls_member_name(archive: zipfile.ZipFile, year: int, family: str, quarter: int) -> str:
    prefix = {"FMLY": "fmld", "MEMB": "memd", "EXPD": "expd"}[family]
    expected = f"{prefix}{year % 100:02d}{quarter}.csv"
    matches = [name for name in archive.namelist() if Path(name).name.lower() == expected]
    if len(matches) != 1:
        raise RuntimeError(
            f"CEX Diary construction: {archive.filename} lacks a unique documented {family} member {expected}: {matches}."
        )
    return matches[0]


def bls_field(year: int, standard_name: str) -> str:
    return BLS_FMLY_SOURCE_ALIASES.get(standard_name, {}).get(year, standard_name)


def bls_memb_field(year: int, standard_name: str) -> str:
    return BLS_MEMB_SOURCE_ALIASES.get(standard_name, {}).get(year, standard_name)


def read_bls_year(year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    archive_path = source_archive(year)
    frames: dict[str, list[pd.DataFrame]] = defaultdict(list)
    input_rows: list[dict[str, object]] = []
    with zipfile.ZipFile(archive_path) as archive:
        archive_digest = sha256_file(archive_path)
        for family in ["FMLY", "MEMB", "EXPD"]:
            for quarter in range(1, 5):
                member_name = bls_member_name(archive, year, family, quarter)
                header = next(csv.reader([archive.read(member_name).splitlines()[0].decode("utf-8-sig")]))
                if family == "FMLY":
                    source_names = {name: bls_field(year, name) for name in FMLY_FIELDS}
                    read_columns = list(dict.fromkeys(source_names.values()))
                    if year >= 2004:
                        read_columns.extend(BLS_IMPUTED_OR_COLLECTED_FMLY_FIELDS)
                        read_columns = list(dict.fromkeys(read_columns))
                elif family == "MEMB":
                    source_names = {name: bls_memb_field(year, name) for name in MEMB_FIELDS}
                    read_columns = list(dict.fromkeys(source_names.values()))
                else:
                    source_names = {name: name for name in EXPD_FIELDS}
                    read_columns = EXPD_FIELDS
                missing = sorted(set(read_columns) - set(header))
                if missing:
                    raise RuntimeError(
                        f"CEX Diary construction: {year} {member_name} is missing required released columns {missing}."
                    )
                frame = pd.read_csv(archive.open(member_name), usecols=read_columns, dtype="string")
                for standard, source in source_names.items():
                    if standard != source:
                        frame[standard] = frame[source]
                frame["source_year"] = year
                frame["source_provider"] = "BLS"
                frame["source_family"] = "BLS_csv"
                frame["source_archive"] = archive_path.name
                frame["source_quarter_file"] = f"Q{quarter}"
                frame["source_member"] = member_name
                for standard, source in source_names.items():
                    frame[f"source_field_{standard}"] = source
                frames[family].append(frame)
                input_rows.append({
                    "source_year": year, "source_provider": "BLS", "source_archive": archive_path.name,
                    "source_family": family, "source_member": member_name, "source_quarter_file": f"Q{quarter}",
                    "raw_rows": len(frame), "member_sha256": hashlib.sha256(archive.read(member_name)).hexdigest(),
                    "crosswalk_anchor": "CEX Dictionary released-CSV header; 2004--05 alias explicitly recorded",
                    "parser_rule": "released_csv_header_exact",
                })
    return (
        pd.concat(frames["FMLY"], ignore_index=True), pd.concat(frames["MEMB"], ignore_index=True),
        pd.concat(frames["EXPD"], ignore_index=True), input_rows,
    )


def category_label(values: pd.Series, labels: dict[str, str]) -> pd.Series:
    return values.astype("string").map(labels).fillna("not reported")


def parse_fmly(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    out = raw_text(frame)
    out["source_year"] = year
    for name in BLS_IMPUTED_OR_COLLECTED_FMLY_FIELDS:
        if name not in out:
            out[name] = pd.Series(pd.NA, index=out.index, dtype="string")
    newid, newid_valid = normalize_newid(out["NEWID"])
    if not newid_valid.all():
        raise RuntimeError(f"CEX Diary construction: {year} FMLY has invalid NEWID values.")
    out["newid"] = newid
    out["week_key"] = str(year) + ":" + out["newid"]
    out["cu_id"] = str(year) + ":" + out["newid"].str.slice(0, 7)
    out["source_week_id"] = (
        out["source_provider"] + ":" + str(year) + ":" + out["source_archive"] + ":"
        + out["source_quarter_file"] + ":" + out["newid"]
    )
    out["source_fmly_member"] = out["source_member"].astype("string")
    if out["source_week_id"].duplicated().any() or out["week_key"].duplicated().any():
        raise RuntimeError(f"CEX Diary construction: {year} FMLY has duplicate canonical week keys.")
    for name in ["AGE_REF", "FAM_SIZE", "FINCBEFX", "FINLWT21", "FSS_RRX", "FSUPPX", "PERSLT18", "PICK_UP", "WEEKN"]:
        out[name + "_number"] = pd.to_numeric(out[name], errors="coerce").astype(float)
    for name in BLS_IMPUTED_OR_COLLECTED_FMLY_FIELDS:
        out[name + "_number"] = pd.to_numeric(out[name], errors="coerce").astype(float)
    start_year_raw = pd.to_numeric(out["STRTYEAR"], errors="coerce")
    out["start_year"] = np.select(
        [start_year_raw.between(0, 99), start_year_raw.between(1900, 2099)],
        [1900 + start_year_raw, start_year_raw],
        default=np.nan,
    )
    start_month = pd.to_numeric(out["STRTMNTH"], errors="coerce")
    start_day = pd.to_numeric(out["STRTDAY"], errors="coerce")
    out["diary_week_start"] = pd.to_datetime(
        pd.Series(out["start_year"], index=out.index).round().astype("Int64").astype("string") + "-"
        + start_month.round().astype("Int64").astype("string").str.zfill(2) + "-"
        + start_day.round().astype("Int64").astype("string").str.zfill(2),
        errors="coerce",
    )
    out["start_date_valid"] = out["diary_week_start"].notna()
    out["legacy_strict_start_date_valid"] = out["start_date_valid"] & out["start_year"].eq(year)
    out["diary_week"] = pd.to_numeric(out["newid"].str[-1], errors="coerce").astype("Int64")
    out["source_weight_field"] = "CENWT21" if year in {1982, 1983} else "FINLWT21"
    out["analysis_weight"] = out["FINLWT21_number"].astype(float)
    out["diary_final_weight"] = out["analysis_weight"]
    out["sample_scope"] = "urban_only" if year in {1982, 1983} else "all_areas"
    out["canonical_source_selected"] = True
    out["cross_package_overlap_available"] = year in set(range(1990, 1997))
    out["canonical_source_precedence_rule"] = (
        "ICPSR_1982_1989_canonical" if year <= 1989
        else "BLS_1990_2011_canonical_ICPSR_1990_1996_audit_only"
    )
    out["overlap_resolution_status"] = (
        "ICPSR_1990_1996_available_audit_only_BLS_selected" if year in set(range(1990, 1997))
        else "no_canonical_overlap_decision_required"
    )
    out["source_field_FINLWT21"] = np.where(year in {1982, 1983}, "CENWT21", "FINLWT21")
    for name in FMLY_FIELDS:
        source_name = f"source_field_{name}"
        if source_name not in out:
            out[source_name] = name
    for name in FMLY_FIELDS:
        out[name + "_raw"] = out[name].astype("string")
    for name in BLS_IMPUTED_OR_COLLECTED_FMLY_FIELDS:
        out[name + "_raw"] = out[name].astype("string")
    out["age_ref"] = out["AGE_REF_number"].astype(float)
    out["household_size"] = out["FAM_SIZE_number"].astype(float)
    out["children_under_18"] = out["PERSLT18_number"].astype(float)
    out["number_adults"] = out["household_size"] - out["children_under_18"]
    out["number_adults_valid"] = (
        out["household_size"].notna() & out["children_under_18"].notna() & out["number_adults"].ge(0)
    )
    out.loc[~out["number_adults_valid"], "number_adults"] = np.nan
    collected_fields_are_observed = out["source_field_FINCBEFX"].eq("FINCBEFX")
    out["annual_income_before_tax_collected"] = out["FINCBEFX_number"].where(collected_fields_are_observed)
    out["annual_social_security_railroad_income_collected"] = out["FSS_RRX_number"].where(
        out["source_field_FSS_RRX"].eq("FSS_RRX")
    )
    out["annual_ssi_income_collected"] = out["FSUPPX_number"].where(out["source_field_FSUPPX"].eq("FSUPPX"))
    out["annual_income_before_tax_imputed_or_collected"] = out["FINCBEFM_number"]
    out["annual_social_security_railroad_income_imputed_or_collected"] = out["FSS_RRXM_number"]
    out["annual_ssi_income_imputed_or_collected"] = out["FSUPPXM_number"]
    if year >= 2004:
        out["annual_income_before_tax"] = out["annual_income_before_tax_imputed_or_collected"]
        out["annual_social_security_railroad_income"] = out[
            "annual_social_security_railroad_income_imputed_or_collected"
        ]
        out["annual_ssi_income"] = out["annual_ssi_income_imputed_or_collected"]
        out["annual_resource_measure_definition"] = "imputed_or_collected_2004_2011"
        out["annual_income_before_tax_source_field"] = "FINCBEFM"
        out["annual_social_security_railroad_income_source_field"] = "FSS_RRXM"
        out["annual_ssi_income_source_field"] = "FSUPPXM"
    else:
        out["annual_income_before_tax"] = out["annual_income_before_tax_collected"]
        out["annual_social_security_railroad_income"] = out[
            "annual_social_security_railroad_income_collected"
        ]
        out["annual_ssi_income"] = out["annual_ssi_income_collected"]
        out["annual_resource_measure_definition"] = "collected_1982_2003"
        out["annual_income_before_tax_source_field"] = "FINCBEFX"
        out["annual_social_security_railroad_income_source_field"] = "FSS_RRX"
        out["annual_ssi_income_source_field"] = "FSUPPX"
    pickup_number = out["PICK_UP_number"]
    out["pickup_status_harmonized"] = np.select(
        [pickup_number.isin([1, 201]), pickup_number.isin([3, 217])],
        ["complete_interview", "temporarily_absent"],
        default="not_reported_or_unmapped",
    )
    out["pickup_status_mapping_version"] = (
        "pickcode_2004_2011_v1" if year >= 2004 else "pick_up_1982_2003_v1"
    )
    out["detailed_expenditure_coverage_regime"] = (
        "reduced_observed_ucc_coverage_1982_1985"
        if year <= 1985 else "expanded_observed_ucc_coverage_1986_2011"
    )
    out["sex"] = category_label(out["SEX_REF"], SEX_LABEL)
    out["race"] = category_label(out["REF_RACE"], RACE_LABEL)
    out["education"] = category_label(out["EDUC_REF"], EDUC_LABEL)
    if year >= 1996:
        out["education"] = out["EDUC_REF"].astype("string").map(EDUC_LABEL_1996).fillna("not reported")
    out["education_mapping_version"] = "education_1996_plus_v1" if year >= 1996 else "education_1982_1995_v1"
    out["marital_status"] = category_label(out["MARITAL1"], MARITAL_LABEL)
    out["region"] = category_label(out["REGION"], REGION_LABEL)
    out["race"] = category_label(out["REF_RACE"], RACE_LABEL_2003_PLUS if year >= 2003 else RACE_LABEL)
    out["race_mapping_version"] = "race_2003_plus_v1" if year >= 2003 else "race_1982_2002_v1"
    out["urban_status"] = category_label(out["BLS_URBN"], URBAN_LABEL)
    if year in {1982, 1983}:
        out["urban_status"] = "not_classified_urban_only_source"
    out["home_tenure"] = category_label(out["CUTENURE"], TENURE_LABEL)
    out["reference_person_work_status"] = np.where(
        out["EMPLTYP1"].isin(["1", "2", "3", "4", "5", "6"]), "currently working", "not currently working"
    )
    out["demographic_mapping_version"] = "cex_diary_profile_v1"
    # PICKCODE is the released 2004--11 source for the stable PICK_UP field.
    # Its value is retained in PICK_UP_raw and its name in
    # source_field_PICK_UP; do not let the source-only alias change the
    # intermediate schema at the 2003--04 boundary.
    if "PICKCODE" in out:
        out = out.drop(columns="PICKCODE")
    return out


def parse_memb(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    out = raw_text(frame)
    out["source_year"] = year
    out["newid"], out["newid_valid"] = normalize_newid(out["NEWID"])
    out["week_key"] = str(year) + ":" + out["newid"].astype("string")
    out["cu_id"] = str(year) + ":" + out["newid"].astype("string").str.slice(0, 7)
    out["source_week_id"] = (
        out["source_provider"] + ":" + str(year) + ":" + out["source_archive"] + ":"
        + out["source_quarter_file"] + ":" + out["newid"].astype("string")
    )
    out["member_number"] = out["MEMBNO"].astype("string")
    for name in ["SS_RRX", "SUPPX"]:
        out[name + "_number"] = pd.to_numeric(out[name], errors="coerce")
    for name in MEMB_FIELDS:
        out[name + "_raw"] = out[name].astype("string")
    out["anyssinc_affirmative"] = out["ANYSSINC"].eq("1")
    out["anyrail_affirmative"] = out["ANYRAIL"].eq("1")
    out["us_supp_affirmative"] = out["US_SUPP"].eq("1")
    out["anyssinc_missing"] = out["ANYSSINC"].isna() | out["ANYSSINC"].eq("")
    out["anyrail_missing"] = out["ANYRAIL"].isna() | out["ANYRAIL"].eq("")
    out["us_supp_missing"] = out["US_SUPP"].isna() | out["US_SUPP"].eq("")
    out["reference_or_spouse"] = out["CU_CODE1"].isin(["1", "2"])
    return out


def parse_qredate(values: pd.Series) -> pd.DataFrame:
    raw = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    width8 = raw.str.fullmatch(r"[0-9]{8}").fillna(False)
    width10 = raw.str.fullmatch(r"[0-9]{10}").fillna(False)
    sequence = pd.to_numeric(raw.str.slice(0, 1), errors="coerce")
    weekday = pd.to_numeric(raw.str.slice(1, 2), errors="coerce")
    month = pd.to_numeric(raw.str.slice(2, 4), errors="coerce")
    day = pd.to_numeric(raw.str.slice(4, 6), errors="coerce")
    raw_year = pd.to_numeric(raw.str.slice(6, None), errors="coerce")
    date_year = raw_year.where(width10, 1900 + raw_year)
    date = pd.to_datetime(
        date_year.round().astype("Int64").astype("string") + "-"
        + month.round().astype("Int64").astype("string").str.zfill(2) + "-"
        + day.round().astype("Int64").astype("string").str.zfill(2),
        errors="coerce",
    )
    return pd.DataFrame({
        "qredate_raw": raw, "diary_day_from_qredate": sequence, "weekday_from_qredate": weekday,
        "qredate_calendar_year": date_year, "qredate_date": date,
        "qredate_blank": raw.isna() | raw.eq(""),
        "qredate_width_8": width8, "qredate_width_10": width10,
        "qredate_valid_exact_date": (width8 | width10) & sequence.between(1, 7) & weekday.between(1, 7) & date.notna(),
    })


def parse_expd(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    out = raw_text(frame)
    out["source_year"] = year
    out["newid"], out["newid_valid"] = normalize_newid(out["NEWID"])
    out["week_key"] = str(year) + ":" + out["newid"].astype("string")
    out["cu_id"] = str(year) + ":" + out["newid"].astype("string").str.slice(0, 7)
    out["source_week_id"] = (
        out["source_provider"] + ":" + str(year) + ":" + out["source_archive"] + ":"
        + out["source_quarter_file"] + ":" + out["newid"].astype("string")
    )
    out["cost"] = pd.to_numeric(out["COST"], errors="coerce")
    out["cost_valid"] = out["cost"].notna() & np.isfinite(out["cost"])
    out["ucc_raw"] = out["UCC"].astype("string").str.strip()
    out["ucc"] = out["ucc_raw"].where(out["ucc_raw"].str.fullmatch(r"[0-9]{1,6}").fillna(False)).str.zfill(6)
    out["ucc_valid"] = out["ucc"].str.fullmatch(r"[0-9]{6}").fillna(False)
    out["ucc_number"] = pd.to_numeric(out["ucc"], errors="coerce")
    dates = parse_qredate(out["QREDATE"])
    for name in dates:
        out[name] = dates[name].to_numpy()
    return out


def code_set(values: pd.Series) -> str:
    cleaned = sorted({str(value).strip() for value in values.dropna() if str(value).strip()})
    return "|".join(cleaned)


def summarize_members(memb: pd.DataFrame, fml: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    joined = memb.merge(
        fml[["source_week_id", "cu_id"]], on=["source_week_id", "cu_id"], how="left",
        validate="many_to_one", indicator="_fmly_merge",
    )
    accounting = {
        "raw_member_rows": int(len(joined)),
        "matched_member_rows": int(joined["_fmly_merge"].eq("both").sum()),
        "unmatched_member_rows": int(joined["_fmly_merge"].ne("both").sum()),
        "invalid_member_newid_rows": int((~joined["newid_valid"]).sum()),
    }
    matched = joined.loc[joined["_fmly_merge"].eq("both")].copy()
    if matched.empty:
        return pd.DataFrame(columns=["source_week_id"]), accounting
    summary = matched.groupby("source_week_id", observed=True).agg(
        member_record_count=("source_week_id", "size"),
        member_anyssinc_affirmative_count=("anyssinc_affirmative", "sum"),
        member_anyrail_affirmative_count=("anyrail_affirmative", "sum"),
        member_us_supp_affirmative_count=("us_supp_affirmative", "sum"),
        member_anyssinc_missing_count=("anyssinc_missing", "sum"),
        member_anyrail_missing_count=("anyrail_missing", "sum"),
        member_us_supp_missing_count=("us_supp_missing", "sum"),
        member_anyssinc_raw_code_set=("ANYSSINC_raw", code_set),
        member_anyrail_raw_code_set=("ANYRAIL_raw", code_set),
        member_us_supp_raw_code_set=("US_SUPP_raw", code_set),
        member_cu_code1_raw_code_set=("CU_CODE1_raw", code_set),
        member_number_raw_code_set=("MEMBNO_raw", code_set),
        source_memb_member_set=("source_member", code_set),
    ).reset_index()
    ref = matched.loc[matched["reference_or_spouse"]].copy()
    ref_summary = ref.groupby("source_week_id", observed=True).agg(
        reference_or_spouse_member_count=("source_week_id", "size"),
        reference_or_spouse_oasdi_affirmative=("anyssinc_affirmative", "max"),
        reference_or_spouse_railroad_affirmative=("anyrail_affirmative", "max"),
        reference_or_spouse_ssi_affirmative=("us_supp_affirmative", "max"),
        reference_or_spouse_oasdi_missing=("anyssinc_missing", "max"),
        reference_or_spouse_railroad_missing=("anyrail_missing", "max"),
        reference_or_spouse_ssi_missing=("us_supp_missing", "max"),
    ).reset_index()
    summary = summary.merge(ref_summary, on="source_week_id", how="left", validate="one_to_one")
    indicator_columns = [name for name in summary if name.startswith("reference_or_spouse_") and name not in {"reference_or_spouse_member_count"}]
    for name in indicator_columns:
        summary[name] = summary[name].fillna(False).astype(bool)
    summary["reference_or_spouse_member_count"] = summary["reference_or_spouse_member_count"].fillna(0).astype(int)
    return summary, accounting


def recipient_summary(memb: pd.DataFrame) -> pd.DataFrame:
    """Implement the unfiltered 1986--96 recipient component exactly once per member."""
    ref = memb.loc[memb["reference_or_spouse"] & memb["newid_valid"]].copy()
    if ref.empty:
        return pd.DataFrame(columns=["cu_id"])
    person = ref.groupby(["cu_id", "member_number"], observed=True).agg(
        oasdi=("anyssinc_affirmative", "max"), railroad=("anyrail_affirmative", "max"),
        ssi=("us_supp_affirmative", "max"),
    ).reset_index()
    person["dual_oasdi_ssi"] = person["oasdi"] & person["ssi"]
    person["railroad_only"] = ~person["oasdi"] & person["railroad"] & ~person["ssi"]
    person["ssi_only"] = ~person["oasdi"] & person["ssi"]
    recipients = person.groupby("cu_id", observed=True).agg(
        reference_or_spouse_oasdi=("oasdi", "max"),
        reference_or_spouse_railroad=("railroad", "max"),
        reference_or_spouse_ssi=("ssi", "max"),
        reference_or_spouse_dual_oasdi_ssi=("dual_oasdi_ssi", "max"),
        reference_or_spouse_railroad_only=("railroad_only", "max"),
        reference_or_spouse_ssi_only=("ssi_only", "max"),
        likely_social_security_recipients=("oasdi", "sum"),
    ).reset_index()
    recipients["reference_or_spouse_paper_recipient"] = (
        recipients["reference_or_spouse_oasdi"] | recipients["reference_or_spouse_railroad"]
    )
    reference_reason = ref.loc[ref["CU_CODE1"].eq("1")].groupby("cu_id", observed=True).agg(
        reference_person_reason_not_working=("WHYNOWRK_raw", code_set)
    ).reset_index()
    recipients = recipients.merge(reference_reason, on="cu_id", how="left", validate="one_to_one")
    recipients["reference_person_reason_not_working"] = recipients["reference_person_reason_not_working"].fillna("").astype("string")
    for code, prefix in [("1", "reference_person"), ("2", "spouse")]:
        role = memb.loc[memb["newid_valid"] & memb["CU_CODE1"].eq(code)].copy()
        if role.empty:
            continue
        role_summary = role.groupby("cu_id", observed=True).agg(
            **{
                f"{prefix}_oasdi_affirmative": ("anyssinc_affirmative", "max"),
                f"{prefix}_railroad_affirmative": ("anyrail_affirmative", "max"),
                f"{prefix}_ssi_affirmative": ("us_supp_affirmative", "max"),
                f"{prefix}_oasdi_missing": ("anyssinc_missing", "max"),
                f"{prefix}_railroad_missing": ("anyrail_missing", "max"),
                f"{prefix}_ssi_missing": ("us_supp_missing", "max"),
                f"{prefix}_anyssinc_raw_code_set": ("ANYSSINC_raw", code_set),
                f"{prefix}_anyrail_raw_code_set": ("ANYRAIL_raw", code_set),
                f"{prefix}_us_supp_raw_code_set": ("US_SUPP_raw", code_set),
            }
        ).reset_index()
        recipients = recipients.merge(role_summary, on="cu_id", how="left", validate="one_to_one")
        for suffix in ["oasdi_affirmative", "railroad_affirmative", "ssi_affirmative", "oasdi_missing", "railroad_missing", "ssi_missing"]:
            recipients[f"{prefix}_{suffix}"] = recipients[f"{prefix}_{suffix}"].fillna(False).astype(bool)
        for suffix in ["anyssinc_raw_code_set", "anyrail_raw_code_set", "us_supp_raw_code_set"]:
            recipients[f"{prefix}_{suffix}"] = recipients[f"{prefix}_{suffix}"].fillna("").astype("string")
    return recipients


def prepare_expd(exp: pd.DataFrame, fml: pd.DataFrame, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    week_fields = fml[["source_week_id", "diary_week_start", "start_date_valid", "legacy_strict_start_date_valid"]]
    out = exp.merge(week_fields, on="source_week_id", how="left", validate="many_to_one", indicator="_fmly_merge")
    out["source_week_exists"] = out["_fmly_merge"].eq("both")
    out["sequence_date_consistent_with_fmly_grid"] = (
        out["qredate_valid_exact_date"] & out["start_date_valid"].fillna(False)
        & out["qredate_date"].eq(out["diary_week_start"] + pd.to_timedelta(out["diary_day_from_qredate"] - 1, unit="D"))
    )
    expected_weekday = ((out["qredate_date"].dt.dayofweek + 1) % 7) + 1
    out["qredate_weekday_matches_calendar"] = (
        out["qredate_valid_exact_date"] & expected_weekday.eq(out["weekday_from_qredate"])
    )
    out["qredate_source_year_matches"] = out["qredate_calendar_year"].eq(year)
    out["legacy_strict_qredate_width_matches"] = out["qredate_width_10"] if year == 1996 else out["qredate_width_8"]
    out["general_transaction_allocated"] = (
        out["newid_valid"] & out["source_week_exists"] & out["qredate_valid_exact_date"]
        & out["cost_valid"] & out["ucc_valid"] & out["sequence_date_consistent_with_fmly_grid"]
    )
    out["legacy_strict_transaction_allocated"] = (
        out["general_transaction_allocated"] & out["qredate_source_year_matches"]
        & out["qredate_weekday_matches_calendar"] & out["legacy_strict_qredate_width_matches"]
    )
    status = pd.Series("allocated_general", index=out.index, dtype="string")
    status = status.mask(~out["newid_valid"], "invalid_normalized_key")
    status = status.mask(out["newid_valid"] & ~out["source_week_exists"], "unmatched_fmly_key")
    status = status.mask(out["source_week_exists"] & out["qredate_blank"], "blank_qredate")
    status = status.mask(out["source_week_exists"] & ~out["qredate_blank"] & ~out["qredate_valid_exact_date"], "invalid_qredate")
    status = status.mask(out["source_week_exists"] & out["qredate_valid_exact_date"] & ~out["cost_valid"], "invalid_cost")
    status = status.mask(out["source_week_exists"] & out["qredate_valid_exact_date"] & out["cost_valid"] & ~out["ucc_valid"], "invalid_ucc")
    status = status.mask(
        out["source_week_exists"] & out["qredate_valid_exact_date"] & out["cost_valid"] & out["ucc_valid"]
        & ~out["start_date_valid"].fillna(False), "invalid_fmly_start",
    )
    status = status.mask(
        out["source_week_exists"] & out["qredate_valid_exact_date"] & out["cost_valid"] & out["ucc_valid"]
        & out["start_date_valid"].fillna(False) & ~out["sequence_date_consistent_with_fmly_grid"],
        "qredate_outside_or_sequence_inconsistent_with_fmly_grid",
    )
    out["general_allocation_status"] = status
    if not out.loc[out["general_transaction_allocated"], "general_allocation_status"].eq("allocated_general").all():
        raise RuntimeError(f"CEX Diary construction: {year} general EXPD accounting is not mutually exclusive.")
    out["otherwise_usable_unallocated"] = (
        out["source_week_exists"] & out["cost_valid"] & out["ucc_valid"] & ~out["general_transaction_allocated"]
    )
    unallocatable = out.loc[out["otherwise_usable_unallocated"]].groupby("source_week_id", observed=True).size()
    fml = fml.copy()
    fml["otherwise_usable_unallocatable_expd_rows"] = fml["source_week_id"].map(unallocatable).fillna(0).astype(int)
    fml["expenditure_complete"] = fml["start_date_valid"] & fml["otherwise_usable_unallocatable_expd_rows"].eq(0)
    fml["expenditure_incomplete"] = ~fml["expenditure_complete"]
    strict_unalloc = out.loc[
        out["source_week_exists"] & out["cost_valid"] & out["ucc_valid"] & ~out["legacy_strict_transaction_allocated"]
    ].groupby("source_week_id", observed=True).size()
    fml["legacy_strict_unallocatable_expd_rows"] = fml["source_week_id"].map(strict_unalloc).fillna(0).astype(int)
    fml["legacy_strict_expenditure_complete"] = (
        fml["legacy_strict_start_date_valid"] & fml["legacy_strict_unallocatable_expd_rows"].eq(0)
    )
    out["qredate_malformed"] = ~out["qredate_blank"] & ~out["qredate_valid_exact_date"]
    out["qredate_weekday_disagreement_audit"] = out["qredate_valid_exact_date"] & ~out["qredate_weekday_matches_calendar"]
    out["qredate_source_year_mismatch_audit"] = out["qredate_valid_exact_date"] & ~out["qredate_source_year_matches"]
    date_counts = out.groupby("source_week_id", observed=True).agg(
        expd_valid_qredate_rows=("qredate_valid_exact_date", "sum"),
        expd_blank_qredate_rows=("qredate_blank", "sum"),
        expd_malformed_qredate_rows=("qredate_malformed", "sum"),
        expd_general_allocated_rows=("general_transaction_allocated", "sum"),
        expd_outside_or_sequence_rows=("general_allocation_status", lambda values: int(values.eq("qredate_outside_or_sequence_inconsistent_with_fmly_grid").sum())),
        expd_weekday_mismatch_rows=("qredate_weekday_disagreement_audit", "sum"),
        expd_source_year_mismatch_rows=("qredate_source_year_mismatch_audit", "sum"),
    ).reset_index()
    first_day = out.loc[out["qredate_valid_exact_date"] & out["source_week_exists"]].groupby("source_week_id", observed=True).apply(
        lambda part: bool(part["qredate_date"].eq(part["diary_week_start"]).all()), include_groups=False
    ).rename("general_first_day_only_date_pattern").reset_index()
    fml = fml.merge(date_counts, on="source_week_id", how="left", validate="one_to_one")
    fml = fml.merge(first_day, on="source_week_id", how="left", validate="one_to_one")
    for name in [
        "expd_valid_qredate_rows", "expd_blank_qredate_rows", "expd_malformed_qredate_rows",
        "expd_general_allocated_rows", "expd_outside_or_sequence_rows", "expd_weekday_mismatch_rows",
        "expd_source_year_mismatch_rows",
    ]:
        fml[name] = fml[name].fillna(0).astype(int)
    fml["general_first_day_only_date_pattern"] = (
        fml["general_first_day_only_date_pattern"].astype("boolean").fillna(False).astype(bool)
    )
    return out, fml


def attach_unfiltered_restriction_inputs(fml: pd.DataFrame) -> pd.DataFrame:
    """Retain every task-stage input as a day-level reproducible source flag."""
    out = fml.copy()
    out["pickup_complete_week"] = out["pickup_status_harmonized"].eq("complete_interview")
    out["weekn_equals_two"] = out["WEEKN_number"].eq(2)
    out["response_income_complete_week"] = out["RESPSTAT_raw"].eq("1")
    out["positive_diary_final_weight_week"] = out["diary_final_weight"].notna() & out["diary_final_weight"].gt(0)
    out["positive_combined_ss_railroad_income_week"] = (
        out["annual_social_security_railroad_income"].notna()
        & out["annual_social_security_railroad_income"].gt(0)
    )
    unit = out.groupby("cu_id", observed=True).agg(
        observed_diary_week_records=("source_week_id", "size"),
        observed_diary_week_numbers=("diary_week", lambda values: set(values.dropna().astype(int))),
        valid_week_start_records=("start_date_valid", "sum"),
        pickup_complete_records=("pickup_complete_week", "sum"),
        weekn_equals_two_records=("weekn_equals_two", "sum"),
        response_income_complete_records=("response_income_complete_week", "sum"),
        positive_weight_records=("positive_diary_final_weight_week", "sum"),
        positive_combined_ss_railroad_income_records=("positive_combined_ss_railroad_income_week", "sum"),
        nonoverlapping_diary_week_dates=("diary_week_start", lambda values: bool(
            len(values) == 2 and values.notna().all() and abs((values.iloc[0] - values.iloc[1]).days) >= 7
        )),
    ).reset_index()
    unit["both_diary_week_numbers_available"] = unit["observed_diary_week_numbers"].map(lambda values: values == {1, 2})
    unit["both_diary_week_records_available"] = unit["observed_diary_week_records"].eq(2)
    unit = unit.drop(columns="observed_diary_week_numbers")
    return out.merge(unit, on="cu_id", how="left", validate="many_to_one")


def attach_strict_quality(fml: pd.DataFrame, exp: pd.DataFrame, memb: pd.DataFrame, year: int) -> pd.DataFrame:
    """Carry the old Stephens 1986--96 quality rule without filtering rows."""
    out = fml.copy()
    out["stephens_1986_1996_quality_flag_v1"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out["legacy_strict_rule_version"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["strict_qredate_valid_rows"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["strict_all_dated_transactions_on_week_start"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out["legacy_strict_dated_week_valid"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out["legacy_strict_valid_start_week_records"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["legacy_strict_valid_dated_week_records"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    if year < 1986 or year > 1996:
        return out
    out["legacy_strict_rule_version"] = LEGACY_STRICT_RULE_VERSION
    qredate_rows = exp.loc[
        exp["source_week_exists"] & exp["qredate_valid_exact_date"] & exp["qredate_source_year_matches"]
        & exp["legacy_strict_qredate_width_matches"]
    ].copy()
    counts = qredate_rows.groupby("source_week_id", observed=True).agg(
        strict_qredate_valid_rows=("source_week_id", "size"),
        strict_all_dated_transactions_on_week_start=("qredate_date", lambda dates: bool(dates.eq(dates.iloc[0]).all())),
    ).reset_index()
    # The equality in the second statistic is to the FMLY start, not merely to
    # the first observed EXPD date.
    if not counts.empty:
        dates = qredate_rows[["source_week_id", "qredate_date", "diary_week_start"]].copy()
        all_start = dates.groupby("source_week_id", observed=True).apply(
            lambda part: bool(part["qredate_date"].eq(part["diary_week_start"]).all()), include_groups=False
        ).rename("strict_all_dated_transactions_on_week_start").reset_index()
        counts = counts.drop(columns="strict_all_dated_transactions_on_week_start").merge(all_start, on="source_week_id", how="left", validate="one_to_one")
    count_by_week = counts.set_index("source_week_id")
    if count_by_week.index.duplicated().any():
        raise RuntimeError(f"CEX Diary construction: {year} strict week-quality counts are not unique.")
    out["strict_qredate_valid_rows"] = (
        out["source_week_id"].map(count_by_week["strict_qredate_valid_rows"]).fillna(0).astype("Int64")
    )
    out["strict_all_dated_transactions_on_week_start"] = (
        out["source_week_id"].map(count_by_week["strict_all_dated_transactions_on_week_start"])
        .astype("boolean").fillna(False)
    )
    out["legacy_strict_dated_week_valid"] = (
        out["legacy_strict_start_date_valid"] & out["strict_qredate_valid_rows"].gt(0)
        & ~out["strict_all_dated_transactions_on_week_start"]
    )
    if "likely_social_security_recipients" not in out:
        recipients = recipient_summary(memb)
        out = out.merge(recipients, on="cu_id", how="left", validate="many_to_one")
    for name in [
        "reference_or_spouse_oasdi", "reference_or_spouse_railroad", "reference_or_spouse_ssi",
        "reference_or_spouse_dual_oasdi_ssi", "reference_or_spouse_railroad_only",
        "reference_or_spouse_ssi_only", "reference_or_spouse_paper_recipient",
    ]:
        out[name] = out[name].fillna(False).astype(bool)
    out["likely_social_security_recipients"] = out["likely_social_security_recipients"].fillna(0).astype(int)
    quality = out.groupby("cu_id", observed=True).agg(
        observed_weeks=("source_week_id", "size"),
        diary_week_values=("diary_week", lambda values: set(values.dropna().astype(int))),
        legacy_strict_valid_start_week_records=("legacy_strict_start_date_valid", "sum"),
        legacy_strict_valid_dated_week_records=("legacy_strict_dated_week_valid", "sum"),
        pickup_complete_weeks=("PICK_UP_number", lambda values: int(values.eq(1).sum())),
        two_week_interview_weeks=("WEEKN_number", lambda values: int(values.eq(2).sum())),
        nonoverlap=("diary_week_start", lambda values: bool(
            len(values) == 2 and values.notna().all() and abs((values.iloc[0] - values.iloc[1]).days) >= 7
        )),
    ).reset_index()
    quality["two_diary_weeks"] = quality["observed_weeks"].eq(2) & quality["diary_week_values"].map(lambda item: item == {1, 2})
    quality["stephens_1986_1996_quality_flag_v1"] = (
        quality["two_diary_weeks"]
        & quality["legacy_strict_valid_start_week_records"].eq(2)
        & quality["legacy_strict_valid_dated_week_records"].eq(2)
        & quality["pickup_complete_weeks"].eq(2) & quality["two_week_interview_weeks"].eq(2)
        & quality["nonoverlap"]
    )
    out = out.merge(
        quality[[
            "cu_id", "legacy_strict_valid_start_week_records",
            "legacy_strict_valid_dated_week_records", "stephens_1986_1996_quality_flag_v1",
        ]], on="cu_id", how="left",
        validate="many_to_one", suffixes=("", "_computed"),
    )
    for name in ["legacy_strict_valid_start_week_records", "legacy_strict_valid_dated_week_records"]:
        out[name] = out[f"{name}_computed"].astype("Int64")
        out = out.drop(columns=f"{name}_computed")
    out["stephens_1986_1996_quality_flag_v1"] = out["stephens_1986_1996_quality_flag_v1_computed"].astype("boolean")
    out = out.drop(columns="stephens_1986_1996_quality_flag_v1_computed")
    return out


def make_daily(fml: pd.DataFrame, exp: pd.DataFrame, year: int) -> pd.DataFrame:
    grid = fml.loc[fml.index.repeat(7)].copy().reset_index(drop=True)
    grid["diary_day_sequence"] = np.tile(np.arange(1, 8), len(fml))
    grid["source_day_id"] = grid["source_week_id"] + ":D" + grid["diary_day_sequence"].astype(str)
    grid["diary_date"] = grid["diary_week_start"] + pd.to_timedelta(grid["diary_day_sequence"] - 1, unit="D")
    grid["stephens_food_available"] = 1986 <= year <= 1996
    grid["stephens_food_version"] = LEGACY_STRICT_RULE_VERSION if 1986 <= year <= 1996 else "unavailable_outside_1986_1996"
    grid.loc[~grid["start_date_valid"], "diary_date"] = pd.NaT
    general = exp.loc[exp["general_transaction_allocated"]].groupby(
        ["source_week_id", "qredate_date"], observed=True
    ).agg(
        daily_total_expenditure=("cost", "sum"),
        general_aggregated_transaction_rows=("cost", "size"),
    ).reset_index().rename(columns={"qredate_date": "diary_date"})
    dated_grid_keys = grid.loc[grid["diary_date"].notna(), ["source_week_id", "diary_date"]]
    if dated_grid_keys.duplicated().any():
        raise RuntimeError(f"CEX Diary construction: {year} has duplicate dated source-week/day keys.")
    grid = grid.merge(general, on=["source_week_id", "diary_date"], how="left", validate="many_to_one")
    grid["general_aggregated_transaction_rows"] = grid["general_aggregated_transaction_rows"].fillna(0).astype(int)
    valid_date = grid["diary_date"].notna()
    grid.loc[valid_date, "daily_total_expenditure"] = grid.loc[valid_date, "daily_total_expenditure"].fillna(0.0)
    grid.loc[~valid_date, "daily_total_expenditure"] = np.nan
    grid["observed_zero_day"] = (
        valid_date & grid["expenditure_complete"] & grid["general_aggregated_transaction_rows"].eq(0)
    )
    grid["legacy_strict_aggregated_transaction_rows"] = pd.Series(pd.NA, index=grid.index, dtype="Int64")
    grid["daily_total_expenditure_legacy_strict_v1"] = np.nan
    grid["daily_food_at_home_legacy_strict_v1"] = np.nan
    grid["daily_food_away_from_home_legacy_strict_v1"] = np.nan
    if 1986 <= year <= 1996:
        strict = exp.loc[exp["legacy_strict_transaction_allocated"]].copy()
        if not strict.empty:
            flags = classify_stephens_ucc(strict["source_year"], strict["ucc_number"])
            strict["strict_food_home"] = strict["cost"].where(flags["included_food_at_home"], 0.0)
            strict["strict_food_away"] = strict["cost"].where(flags["included_food_away_from_home"], 0.0)
            strict_daily = strict.groupby(["source_week_id", "qredate_date"], observed=True).agg(
                daily_total_expenditure_legacy_strict_v1=("cost", "sum"),
                daily_food_at_home_legacy_strict_v1=("strict_food_home", "sum"),
                daily_food_away_from_home_legacy_strict_v1=("strict_food_away", "sum"),
                legacy_strict_aggregated_transaction_rows=("cost", "size"),
            ).reset_index().rename(columns={"qredate_date": "diary_date"})
            grid = grid.merge(strict_daily, on=["source_week_id", "diary_date"], how="left", validate="many_to_one", suffixes=("", "_loaded"))
            for name in [
                "daily_total_expenditure_legacy_strict_v1", "daily_food_at_home_legacy_strict_v1",
                "daily_food_away_from_home_legacy_strict_v1", "legacy_strict_aggregated_transaction_rows",
            ]:
                loaded = name + "_loaded"
                if loaded in grid:
                    grid[name] = grid[loaded].combine_first(grid[name])
                    grid = grid.drop(columns=loaded)
        strict_date = grid["legacy_strict_start_date_valid"]
        for name in [
            "daily_total_expenditure_legacy_strict_v1", "daily_food_at_home_legacy_strict_v1",
            "daily_food_away_from_home_legacy_strict_v1",
        ]:
            grid.loc[strict_date, name] = grid.loc[strict_date, name].fillna(0.0)
        grid.loc[~strict_date, [
            "daily_total_expenditure_legacy_strict_v1", "daily_food_at_home_legacy_strict_v1",
            "daily_food_away_from_home_legacy_strict_v1",
        ]] = np.nan
        grid["legacy_strict_aggregated_transaction_rows"] = grid["legacy_strict_aggregated_transaction_rows"].fillna(0).astype("Int64")
    return grid


def crosswalk_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for year in ICPSR_YEARS:
        for family, names, specs in [
            ("FMLY", FMLY_FIELDS, ICPSR_FMLY_COLSPECS_BY_YEAR[year]),
            ("MEMB", MEMB_FIELDS, ICPSR_MEMB_COLSPECS_BY_YEAR[year]),
            ("EXPD", EXPD_FIELDS, ICPSR_EXPD_COLSPECS_BY_YEAR[year]),
        ]:
            for name, spec in zip(names, specs, strict=True):
                rows.append({
                    "source_year": year, "source_provider": "ICPSR", "file_family": family,
                    "standard_field": name,
                    "documented_start_position_1_based": None if spec is None else spec[0] + 1,
                    "documented_end_position_1_based": None if spec is None else spec[1],
                    "parser_action": "fill_missing_not_documented" if spec is None else "read_documented_fixed_width_slice",
                    "documentation_anchor": ICPSR_CROSSWALK_ANCHORS[year],
                })
    for year in BLS_YEARS:
        for standard in ["FINCBEFX", "FSS_RRX", "FSUPPX", "PICK_UP"]:
            source = bls_field(year, standard)
            pickcode_recode = standard == "PICK_UP" and source == "PICKCODE"
            rows.append({
                "source_year": year, "source_provider": "BLS", "file_family": "FMLY",
                "standard_field": standard, "documented_start_position_1_based": pd.NA,
                "documented_end_position_1_based": pd.NA,
                "parser_action": (
                    "released_csv_alias_plus_documented_201_217_to_complete_absent_recode"
                    if pickcode_recode else "released_csv_alias" if source != standard else "released_csv_same_name"
                ),
                "documentation_anchor": (
                    "CEX Dictionary Codes: PICKCODE 201=Interview and 217=Interview--temporarily absent"
                    if pickcode_recode else
                    "CEX Dictionary and released 2004--05 FMLD headers: "
                    "FINCBEFM=>FINCBEFX, FSS_RRXM=>FSS_RRX, FSUPPXM=>FSUPPX"
                    if source != standard else "CEX Dictionary and released CSV header"
                ),
                "source_field": source,
            })
        if year >= 2004:
            for source, standard in [
                ("FINCBEFM", "annual_income_before_tax_imputed_or_collected"),
                ("FSS_RRXM", "annual_social_security_railroad_income_imputed_or_collected"),
                ("FSUPPXM", "annual_ssi_income_imputed_or_collected"),
            ]:
                rows.append({
                    "source_year": year, "source_provider": "BLS", "file_family": "FMLY",
                    "standard_field": standard, "documented_start_position_1_based": pd.NA,
                    "documented_end_position_1_based": pd.NA, "source_field": source,
                    "parser_action": "retain_imputed_or_collected_measure_without_relabelling_as_collected",
                    "documentation_anchor": "CEX Dictionary Variables sheet: M measure is imputed or collected data",
                })
        for standard in ["SS_RRX", "SUPPX"]:
            source = bls_memb_field(year, standard)
            if source != standard:
                rows.append({
                    "source_year": year, "source_provider": "BLS", "file_family": "MEMB",
                    "standard_field": standard, "documented_start_position_1_based": pd.NA,
                    "documented_end_position_1_based": pd.NA, "source_field": source,
                    "parser_action": "released_csv_alias",
                    "documentation_anchor": "CEX Dictionary and released 2004--05 MEMD header: SS_RRXM=>SS_RRX; SUPPXM=>SUPPX",
                })
    return pd.DataFrame(rows).sort_values(["source_year", "source_provider", "file_family", "standard_field"], kind="mergesort").reset_index(drop=True)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    ordered = frame.copy()
    ordered.to_csv(path, index=False)


def write_parquet_chunk(
    frame: pd.DataFrame, temp_path: Path, writer: pq.ParquetWriter | None,
    column_order: list[str] | None,
) -> tuple[pq.ParquetWriter, list[str]]:
    if column_order is None:
        column_order = frame.columns.tolist()
    if frame.columns.tolist() != column_order:
        missing = sorted(set(column_order) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(column_order))
        if missing or extra:
            raise RuntimeError(f"CEX Diary construction: inconsistent intermediate schema; missing={missing}, extra={extra}.")
        frame = frame[column_order]
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(temp_path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer, column_order


def build_year(year: int, release_dates: pd.DataFrame) -> dict[str, object]:
    """Build one survey year's CU-day rows and its audit rows; reads only its own archive.

    Runs in a worker process when the build uses more than one job. The counts and sums it
    returns are per-year deltas that run_build adds up in year order.
    """
    hard_counts: Counter[str] = Counter()
    hard_sums: defaultdict[str, float] = defaultdict(float)
    print(f"CEX Diary construction: building {year}.", flush=True)
    if year in ICPSR_YEARS:
        fml_raw, memb_raw, exp_raw, source_rows = read_icpsr_year(year)
    else:
        fml_raw, memb_raw, exp_raw, source_rows = read_bls_year(year)
    fml = parse_fmly(fml_raw, year)
    fml = fml.merge(release_dates, on="source_year", how="left", validate="many_to_one")
    start_blank = (
        fml[["STRTDAY_raw", "STRTMNTH_raw", "STRTYEAR_raw"]].fillna("").eq("").any(axis=1)
    )
    hard_counts["raw_fmly"] += len(fml)
    hard_counts["start_valid"] += int(fml["start_date_valid"].sum())
    hard_counts["start_blank"] += int((~fml["start_date_valid"] & start_blank).sum())
    hard_counts["start_malformed"] += int((~fml["start_date_valid"] & ~start_blank).sum())
    memb = parse_memb(memb_raw, year)
    member_summary, member_accounting = summarize_members(memb, fml)
    fml = fml.merge(member_summary, on="source_week_id", how="left", validate="one_to_one")
    for name in [
        "member_record_count", "member_anyssinc_affirmative_count", "member_anyrail_affirmative_count",
        "member_us_supp_affirmative_count", "member_anyssinc_missing_count", "member_anyrail_missing_count",
        "member_us_supp_missing_count", "reference_or_spouse_member_count",
    ]:
        fml[name] = fml[name].fillna(0).astype(int)
    for name in [
        "reference_or_spouse_oasdi_affirmative", "reference_or_spouse_railroad_affirmative",
        "reference_or_spouse_ssi_affirmative", "reference_or_spouse_oasdi_missing",
        "reference_or_spouse_railroad_missing", "reference_or_spouse_ssi_missing",
    ]:
        fml[name] = fml[name].fillna(False).astype(bool)
    for name in [
        "member_anyssinc_raw_code_set", "member_anyrail_raw_code_set", "member_us_supp_raw_code_set",
        "member_cu_code1_raw_code_set", "member_number_raw_code_set",
        "source_memb_member_set",
    ]:
        fml[name] = fml[name].fillna("").astype("string")
    recipients = recipient_summary(memb)
    fml = fml.merge(recipients, on="cu_id", how="left", validate="many_to_one")
    for name in [
        "reference_or_spouse_oasdi", "reference_or_spouse_railroad", "reference_or_spouse_ssi",
        "reference_or_spouse_dual_oasdi_ssi", "reference_or_spouse_railroad_only",
        "reference_or_spouse_ssi_only", "reference_or_spouse_paper_recipient",
    ]:
        fml[name] = fml[name].fillna(False).astype(bool)
    for prefix in ["reference_person", "spouse"]:
        for suffix in ["oasdi_affirmative", "railroad_affirmative", "ssi_affirmative", "oasdi_missing", "railroad_missing", "ssi_missing"]:
            name = f"{prefix}_{suffix}"
            fml[name] = fml[name].fillna(False).astype(bool)
        for suffix in ["anyssinc_raw_code_set", "anyrail_raw_code_set", "us_supp_raw_code_set"]:
            name = f"{prefix}_{suffix}"
            fml[name] = fml[name].fillna("").astype("string")
    fml["likely_social_security_recipients"] = fml["likely_social_security_recipients"].fillna(0).astype(int)
    fml["reference_person_reason_not_working"] = fml["reference_person_reason_not_working"].fillna("").astype("string")
    retirement_code = "1" if year >= 1997 else "5"
    fml["reference_person_work_status"] = np.where(
        fml["reference_person_reason_not_working"].eq(retirement_code), "retired",
        np.where(fml["EMPLTYP1_raw"].isin(["1", "2", "3", "4", "5", "6"]), "currently working", "not currently working"),
    )
    exp = parse_expd(exp_raw, year)
    hard_counts["raw_memb"] += len(memb)
    hard_counts["raw_expd"] += len(exp)
    hard_counts["qredate_valid"] += int(exp["qredate_valid_exact_date"].sum())
    hard_counts["qredate_blank"] += int(exp["qredate_blank"].sum())
    hard_counts["qredate_malformed"] += int((~exp["qredate_blank"] & ~exp["qredate_valid_exact_date"]).sum())
    if not exp["cost_valid"].all() or not exp["ucc_valid"].all():
        raise RuntimeError(f"CEX Diary construction: {year} has an invalid COST or UCC despite the approved source contract.")
    exp, fml = prepare_expd(exp, fml, year)
    if year == 1985:
        known_q4_outside = int((
            exp["source_quarter_file"].eq("Q4")
            & exp["general_allocation_status"].eq("qredate_outside_or_sequence_inconsistent_with_fmly_grid")
        ).sum())
        if known_q4_outside != 45:
            raise RuntimeError(
                f"CEX Diary construction: expected 45 known 1985 Q4 outside-grid rows, found {known_q4_outside}."
            )
    hard_counts["weekday_mismatch"] += int((exp["qredate_valid_exact_date"] & ~exp["qredate_weekday_matches_calendar"]).sum())
    hard_counts["outside_or_sequence"] += int(exp["general_allocation_status"].eq("qredate_outside_or_sequence_inconsistent_with_fmly_grid").sum())
    hard_counts["general_allocated"] += int(exp["general_transaction_allocated"].sum())
    hard_sums["general_signed_cost"] += float(exp.loc[exp["general_transaction_allocated"], "cost"].sum())
    fml = attach_strict_quality(fml, exp, memb, year)
    fml = attach_unfiltered_restriction_inputs(fml)
    daily = make_daily(fml, exp, year)
    valid_date = daily["diary_date"].notna()

    expected_rows = 7 * len(fml)
    if len(daily) != expected_rows or daily["source_day_id"].duplicated().any():
        raise RuntimeError(
            f"CEX Diary construction: {year} FMLY-to-seven-day invariant failed: {len(fml)} weeks, {len(daily)} days."
        )
    if daily.loc[~daily["start_date_valid"], ["diary_date", "daily_total_expenditure"]].notna().any().any():
        raise RuntimeError(f"CEX Diary construction: {year} invalid-start weeks have non-null date or general amount.")
    if (daily["observed_zero_day"] & ~(
        daily["diary_date"].notna() & daily["expenditure_complete"] & daily["general_aggregated_transaction_rows"].eq(0)
    )).any():
        raise RuntimeError(f"CEX Diary construction: {year} observed-zero-day rule failed.")
    adult_identity = daily["number_adults_valid"] & ~daily["number_adults"].eq(daily["household_size"] - daily["children_under_18"])
    if adult_identity.any():
        raise RuntimeError(f"CEX Diary construction: {year} adult-count identity failed.")
    allocated_sum = float(exp.loc[exp["general_transaction_allocated"], "cost"].sum())
    daily_sum = float(daily["daily_total_expenditure"].sum(skipna=True))
    if not np.isclose(allocated_sum, daily_sum, rtol=0, atol=1e-8):
        raise RuntimeError(f"CEX Diary construction: {year} signed general reconciliation failed {allocated_sum} != {daily_sum}.")
    if 1986 <= year <= 1996:
        strict_sum = float(exp.loc[exp["legacy_strict_transaction_allocated"], "cost"].sum())
        daily_strict_sum = float(daily["daily_total_expenditure_legacy_strict_v1"].sum(skipna=True))
        if not np.isclose(strict_sum, daily_strict_sum, rtol=0, atol=1e-8):
            raise RuntimeError(f"CEX Diary construction: {year} signed strict reconciliation failed.")
        strict_transactions = exp.loc[exp["legacy_strict_transaction_allocated"]].copy()
        strict_flags = classify_stephens_ucc(strict_transactions["source_year"], strict_transactions["ucc_number"])
        hard_counts["strict_transaction_rows"] += len(strict_transactions)
        hard_counts["strict_food_home_transactions"] += int(strict_flags["included_food_at_home"].sum())
        hard_counts["strict_food_away_transactions"] += int(strict_flags["included_food_away_from_home"].sum())
        hard_sums["strict_total"] += strict_sum
        hard_sums["strict_food_home"] += float(strict_transactions.loc[strict_flags["included_food_at_home"], "cost"].sum())
        hard_sums["strict_food_away"] += float(strict_transactions.loc[strict_flags["included_food_away_from_home"], "cost"].sum())
    elif daily[[
        "daily_total_expenditure_legacy_strict_v1", "daily_food_at_home_legacy_strict_v1",
        "daily_food_away_from_home_legacy_strict_v1",
    ]].notna().any().any():
        raise RuntimeError(f"CEX Diary construction: {year} has unavailable strict fields outside 1986--96.")

    fmly_audit_row = {
        "source_year": year, "source_provider": fml["source_provider"].iloc[0], "canonical_source": fml["source_archive"].iloc[0],
        "raw_fmly_rows": len(fml), "expected_seven_day_rows": expected_rows, "written_day_rows": len(daily),
        "invalid_or_blank_start_rows": int((~fml["start_date_valid"]).sum()),
        "complete_week_rows": int(fml["expenditure_complete"].sum()),
        "incomplete_week_rows": int(fml["expenditure_incomplete"].sum()),
    }
    member_audit_row = {"source_year": year, "source_provider": fml["source_provider"].iloc[0], **member_accounting}
    expd_audit_rows: list[dict[str, object]] = []
    status_counts = exp["general_allocation_status"].value_counts(dropna=False).sort_index()
    for status, count in status_counts.items():
        expd_audit_rows.append({"source_year": year, "source_provider": fml["source_provider"].iloc[0], "allocation_status": status, "raw_expd_rows": int(count)})
    if sum(row["raw_expd_rows"] for row in expd_audit_rows if row["source_year"] == year) != len(exp):
        raise RuntimeError(f"CEX Diary construction: {year} EXPD accounting is not exhaustive.")
    reconciliation_row = {
        "source_year": year, "general_allocated_transaction_rows": int(exp["general_transaction_allocated"].sum()),
        "general_allocated_signed_cost_sum": allocated_sum, "daily_signed_cost_sum": daily_sum,
        "general_reconciles": True,
        "legacy_strict_allocated_transaction_rows": int(exp["legacy_strict_transaction_allocated"].sum()) if 1986 <= year <= 1996 else pd.NA,
        "legacy_strict_allocated_signed_cost_sum": strict_sum if 1986 <= year <= 1996 else np.nan,
        "legacy_strict_daily_signed_cost_sum": daily_strict_sum if 1986 <= year <= 1996 else np.nan,
        "legacy_strict_reconciles": True if 1986 <= year <= 1996 else pd.NA,
    }
    ucc_inventory = None
    measurement_regime_row = None
    valid_ucc = exp.loc[exp["ucc_valid"]].copy()
    if not valid_ucc.empty:
        inventory = valid_ucc.groupby("ucc", observed=True).agg(
            raw_expd_rows=("ucc", "size"), general_allocated_rows=("general_transaction_allocated", "sum"),
            general_allocated_signed_cost_sum=("cost", lambda values: float(values.sum(skipna=True))),
        ).reset_index()
        inventory.insert(0, "source_year", year)
        ucc_inventory = inventory
        measurement_regime_row = {
            "source_year": year,
            "broad_expenditure_coverage_regime": daily["detailed_expenditure_coverage_regime"].iloc[0],
            "observed_distinct_ucc_count": int(inventory["ucc"].nunique()),
            "broad_expenditure_comparability_note": (
                "Reduced observed UCC coverage; not level-comparable to the expanded 1986+ broad total"
                if year <= 1985 else
                "Expanded observed UCC coverage; 1985-to-1986 level break must not be interpreted as spending growth"
                if year == 1986 else "Within expanded 1986-2011 observed-UCC regime"
            ),
            "annual_resource_measure_definition": daily["annual_resource_measure_definition"].iloc[0],
            "annual_income_before_tax_source_field": daily["annual_income_before_tax_source_field"].iloc[0],
            "annual_social_security_railroad_income_source_field": daily[
                "annual_social_security_railroad_income_source_field"
            ].iloc[0],
            "annual_ssi_income_source_field": daily["annual_ssi_income_source_field"].iloc[0],
            "pickup_status_mapping_version": daily["pickup_status_mapping_version"].iloc[0],
        }
    date_audit_row = {
        "source_year": year, "raw_expd_rows": len(exp), "blank_qredate_rows": int(exp["qredate_blank"].sum()),
        "valid_exact_qredate_rows": int(exp["qredate_valid_exact_date"].sum()),
        "qredate_source_year_mismatch_rows": int((exp["qredate_valid_exact_date"] & ~exp["qredate_source_year_matches"]).sum()),
        "parse_valid_weekday_disagreement_rows": int((exp["qredate_valid_exact_date"] & ~exp["qredate_weekday_matches_calendar"]).sum()),
        "allocated_weekday_disagreement_rows": int((exp["general_transaction_allocated"] & ~exp["qredate_weekday_matches_calendar"]).sum()),
        "outside_or_sequence_inconsistent_rows": int(exp["general_allocation_status"].eq("qredate_outside_or_sequence_inconsistent_with_fmly_grid").sum()),
        "otherwise_usable_unallocated_rows": int(exp["otherwise_usable_unallocated"].sum()),
    }
    strict_row = None
    if 1986 <= year <= 1996:
        strict_row = {
            "source_year": year, "rule_version": LEGACY_STRICT_RULE_VERSION,
            "daily_rows_with_strict_total": int(daily["daily_total_expenditure_legacy_strict_v1"].notna().sum()),
            "strict_total_signed_sum": daily_strict_sum,
            "strict_food_home_signed_sum": float(daily["daily_food_at_home_legacy_strict_v1"].sum(skipna=True)),
            "strict_food_away_signed_sum": float(daily["daily_food_away_from_home_legacy_strict_v1"].sum(skipna=True)),
            "unfiltered_stephens_quality_day_rows": int(daily["stephens_1986_1996_quality_flag_v1"].fillna(False).sum()),
            "unfiltered_stephens_quality_cu_ids": int(daily.loc[daily["stephens_1986_1996_quality_flag_v1"].fillna(False), "cu_id"].nunique()),
        }
    return {
        "year": year,
        "source_rows": source_rows,
        "daily": daily,
        "fmly_rows": len(fml),
        "hard_counts": hard_counts,
        "hard_sums": hard_sums,
        "fmly_audit_row": fmly_audit_row,
        "member_audit_row": member_audit_row,
        "expd_audit_rows": expd_audit_rows,
        "reconciliation_row": reconciliation_row,
        "ucc_inventory": ucc_inventory,
        "measurement_regime_row": measurement_regime_row,
        "date_audit_row": date_audit_row,
        "strict_row": strict_row,
    }


def iter_built_years(release_dates: pd.DataFrame, jobs: int):
    """Yield build_year results in CANONICAL_YEARS order, built by `jobs` worker processes."""
    if jobs == 1:
        for year in CANONICAL_YEARS:
            yield build_year(year, release_dates)
        return
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(build_year, year, release_dates) for year in CANONICAL_YEARS]
        try:
            for future in futures:
                yield future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def run_build(jobs: int = 1) -> None:
    """Build the unfiltered canonical 1982--2011 CU-day intermediate."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    if INTERMEDIATE_TEMP_PATH.exists():
        INTERMEDIATE_TEMP_PATH.unlink()

    fixed_crosswalk = crosswalk_rows()
    release_dates = pd.read_csv(RELEASE_DATE_PATH, dtype={"cex_data_year": int, "release_date": "string", "source": "string"})
    release_dates = release_dates.loc[release_dates["cex_data_year"].isin(CANONICAL_YEARS), ["cex_data_year", "release_date", "source"]].rename(
        columns={"cex_data_year": "source_year", "source": "release_date_source"}
    )
    release_dates["release_date"] = pd.to_datetime(release_dates["release_date"], errors="coerce")
    if release_dates["source_year"].duplicated().any() or set(release_dates["source_year"]) != set(CANONICAL_YEARS) or release_dates["release_date"].isna().any():
        raise RuntimeError("CEX Diary construction: release-date source does not uniquely cover all canonical years.")
    input_rows: list[dict[str, object]] = []
    fmly_audit_rows: list[dict[str, object]] = []
    member_audit_rows: list[dict[str, object]] = []
    expd_audit_rows: list[dict[str, object]] = []
    reconciliation_rows: list[dict[str, object]] = []
    ucc_rows: list[pd.DataFrame] = []
    date_audit_rows: list[dict[str, object]] = []
    measurement_regime_rows: list[dict[str, object]] = []
    strict_rows: list[dict[str, object]] = []
    writer: pq.ParquetWriter | None = None
    output_columns: list[str] | None = None
    total_fmly = 0
    total_daily = 0
    hard_counts: Counter[str] = Counter()
    hard_sums: defaultdict[str, float] = defaultdict(float)

    try:
        for result in iter_built_years(release_dates, jobs):
            input_rows.extend(result["source_rows"])
            for name, count in result["hard_counts"].items():
                hard_counts[name] += count
            for name, value in result["hard_sums"].items():
                hard_sums[name] += value
            daily = result["daily"]
            writer, output_columns = write_parquet_chunk(daily, INTERMEDIATE_TEMP_PATH, writer, output_columns)
            total_fmly += result["fmly_rows"]
            total_daily += len(daily)
            fmly_audit_rows.append(result["fmly_audit_row"])
            member_audit_rows.append(result["member_audit_row"])
            expd_audit_rows.extend(result["expd_audit_rows"])
            reconciliation_rows.append(result["reconciliation_row"])
            if result["ucc_inventory"] is not None:
                ucc_rows.append(result["ucc_inventory"])
                measurement_regime_rows.append(result["measurement_regime_row"])
            date_audit_rows.append(result["date_audit_row"])
            if result["strict_row"] is not None:
                strict_rows.append(result["strict_row"])
            del result, daily
    except Exception:
        if writer is not None:
            writer.close()
        raise
    if writer is None:
        raise RuntimeError("CEX Diary construction: no canonical output was written.")
    writer.close()
    if total_daily != 7 * total_fmly:
        raise RuntimeError("CEX Diary construction: global FMLY-to-seven-day invariant failed.")
    expected_hard_counts = {
        "raw_fmly": 388_686, "raw_memb": 988_393, "raw_expd": 15_805_797,
        "start_valid": 385_039, "start_blank": 3_647, "start_malformed": 0,
        "qredate_valid": 15_571_641, "qredate_blank": 233_875, "qredate_malformed": 281,
        "weekday_mismatch": 97_936, "outside_or_sequence": 4_754, "general_allocated": 15_566_887,
        "strict_transaction_rows": 5_489_756, "strict_food_home_transactions": 2_900_852,
        "strict_food_away_transactions": 850_551,
    }
    expected_hard_sums = {
        "general_signed_cost": 204_789_518.53255, "strict_total": 53_843_774.24127,
        "strict_food_home": 6_357_175.00143, "strict_food_away": 3_426_206.96943,
    }
    if total_daily != 2_720_802 or hard_counts != Counter(expected_hard_counts):
        raise RuntimeError(f"CEX Diary construction: raw/oracle count invariant failed: {dict(hard_counts)}.")
    for name, expected_value in expected_hard_sums.items():
        if not np.isclose(hard_sums[name], expected_value, rtol=0, atol=1e-6):
            raise RuntimeError(f"CEX Diary construction: {name} oracle sum failed: {hard_sums[name]} != {expected_value}.")
    INTERMEDIATE_TEMP_PATH.replace(INTERMEDIATE_PATH)

    report_tables = {
        OUT_DIR / "02_canonical_source_rule.csv": pd.DataFrame([
            {"year_range": "1982-1989", "canonical_provider": "ICPSR", "canonical_rule": "ICPSR only"},
            {"year_range": "1990-2011", "canonical_provider": "BLS", "canonical_rule": "BLS only; ICPSR 1990-96 excluded"},
        ]),
        OUT_DIR / "02_fixed_width_crosswalk_and_alias_provenance.csv": fixed_crosswalk,
        OUT_DIR / "02_raw_input_manifest.csv": pd.DataFrame(input_rows).sort_values(["source_year", "source_family", "source_quarter_file"], kind="mergesort"),
        OUT_DIR / "02_fmly_to_seven_day_audit.csv": pd.DataFrame(fmly_audit_rows),
        OUT_DIR / "02_memb_accounting.csv": pd.DataFrame(member_audit_rows),
        OUT_DIR / "02_expd_mutually_exclusive_accounting.csv": pd.DataFrame(expd_audit_rows),
        OUT_DIR / "02_daily_signed_sum_reconciliation.csv": pd.DataFrame(reconciliation_rows),
        OUT_DIR / "02_ucc_inventory.csv": pd.concat(ucc_rows, ignore_index=True).sort_values(["source_year", "ucc"], kind="mergesort"),
        OUT_DIR / "02_key_date_audit.csv": pd.DataFrame(date_audit_rows),
        OUT_DIR / "02_measurement_regime_audit.csv": pd.DataFrame(measurement_regime_rows),
        OUT_DIR / "02_legacy_strict_1986_1996_compatibility.csv": pd.DataFrame(strict_rows),
    }
    parquet_schema = pq.ParquetFile(INTERMEDIATE_PATH).schema_arrow
    report_tables[OUT_DIR / "02_intermediate_schema.csv"] = pd.DataFrame({
        "column": [field.name for field in parquet_schema],
        "arrow_type": [str(field.type) for field in parquet_schema],
        "null_count": "reported_in_02_missingness_audit_for_contract_fields",
    })
    for path, frame in report_tables.items():
        write_csv(frame, path)
    hash_rows = [
        {"path": relative(path), "sha256": sha256_file(path)}
        for path in sorted(report_tables)
    ]
    hash_rows.append({"path": relative(INTERMEDIATE_PATH), "sha256": sha256_file(INTERMEDIATE_PATH)})
    hashes_path = OUT_DIR / "02_output_hashes.csv"
    write_csv(pd.DataFrame(hash_rows), hashes_path)
    report_tables[hashes_path] = pd.read_csv(hashes_path, dtype=str)
    manifest = {
        "status": "BUILT",
        "rule_version": RULE_VERSION,
        "intermediate": relative(INTERMEDIATE_PATH),
        "intermediate_sha256": sha256_file(INTERMEDIATE_PATH),
        "canonical_years": list(CANONICAL_YEARS),
        "canonical_precedence": "ICPSR 1982-89; BLS 1990-2011; ICPSR 1990-96 never admitted",
        "fml_week_count": total_fmly,
        "daily_row_count": total_daily,
        "strict_parity_rule": "1986-96 only: source-year and weekday checks are imposed only on versioned strict total and Stephens food fields",
        "measurement_breaks": [
            "1982-85 reduced observed-UCC broad-total coverage versus expanded 1986-2011 coverage; no rescaling applied",
            "1982-2003 collected annual resources versus 2004-2011 imputed-or-collected annual resources; separate fields retained",
            "PICK_UP 01/03 and PICKCODE 201/217 harmonised with raw values and mapping version retained",
        ],
        "outputs": [relative(path) for path in report_tables],
    }
    (OUT_DIR / "02_build_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    PIPELINE_STATUS_PATH.write_text(json.dumps({
        "status": "BUILT",
        "stage": "general_cu_day_panel_built",
        "intermediate": relative(INTERMEDIATE_PATH),
        "intermediate_sha256": sha256_file(INTERMEDIATE_PATH),
        "intermediate_written": True,
        "canonical_years": list(CANONICAL_YEARS),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"CEX Diary construction build completed: {total_daily:,} daily rows at {relative(INTERMEDIATE_PATH)}.")


def validate_canonical_sources() -> None:
    """Check the source layouts used in construction, without audit-only archives."""
    expected = expected_archive_paths()
    verify_expected_archives(expected)
    archives, members, documentation, _ = build_inventory(expected)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for rows, path in [(archives, ARCHIVE_INVENTORY_PATH), (members, MEMBER_INVENTORY_PATH),
                       (documentation, DOCUMENTATION_INVENTORY_PATH)]:
        pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(available_cpus(), 8),
        help="Worker processes that build survey years (default: available CPUs, at most 8). "
        "Each holds about 3 GB; the output is the same for any value.",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1.")
    validate_source_inputs("cex_diary")
    validate_canonical_sources()
    run_build(args.jobs)


if __name__ == "__main__":
    main()
