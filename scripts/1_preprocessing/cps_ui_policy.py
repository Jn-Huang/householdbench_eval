#!/usr/bin/env python
"""Build the harmonised state-month UI-duration policy intermediate."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from pandas.io.stata import StataReader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.source_inputs import validate_source_inputs

from scripts.utils.io import sha256_file

SOURCE_DIR = PROJECT_ROOT / "data/raw/micro/cps/ui_duration_farber_2015/P2015_1088_data"
POLICY_PATH = SOURCE_DIR / "eui_state_08-14.dta"
INTERMEDIATE_PATH = PROJECT_ROOT / "data/intermediate/cps_ui_policy.parquet"
OUTPUT_DIR = PROJECT_ROOT / "output/preprocessing/cps_ui_policy"

EXPECTED_LABELS = {
    "fips": "FIPS state code",
    "year": "year of obs.",
    "month": "month of obs.",
    "ext_wks": "total weeks of extended UI (13-73)",
    "reg_UI": "regular UI weeks (<26 for some)",
    "ui_weeks": "Total UI weeks (no suspensions)",
}
PUBLIC_COLUMNS = [
    "fips", "year", "month", "policy_effective_date", "regular_ui_weeks",
    "extension_ui_weeks", "maximum_ui_weeks", "source_active_extension_weeks",
]


validate_source_inputs("cps_ui_policy")

for path in (POLICY_PATH,):
    if not path.is_file():
        raise RuntimeError(f"Required CPS UI-policy source is missing: {path}")

with StataReader(POLICY_PATH) as reader:
    labels = reader.variable_labels()
observed_labels = {name: labels.get(name) for name in EXPECTED_LABELS}
if observed_labels != EXPECTED_LABELS:
    raise RuntimeError(f"CPS UI-policy variable labels changed: {observed_labels}")

source = pd.read_stata(POLICY_PATH, convert_categoricals=False)
missing = sorted(set(EXPECTED_LABELS) - set(source.columns))
if missing:
    raise RuntimeError(f"CPS UI-policy source is missing columns: {missing}")

policy = source[list(EXPECTED_LABELS)].copy()
for column in EXPECTED_LABELS:
    policy[column] = pd.to_numeric(policy[column], errors="coerce")
if policy.isna().any().any():
    raise RuntimeError("CPS UI-policy source contains missing required values.")
for column in ("fips", "year", "month"):
    rounded = policy[column].round()
    if not policy[column].eq(rounded).all():
        raise RuntimeError(f"CPS UI-policy key {column} contains noninteger values.")
    policy[column] = rounded.astype(int)
if policy.duplicated(["fips", "year", "month"]).any():
    raise RuntimeError("CPS UI-policy source is not unique by state and month.")
if not policy["month"].between(1, 12).all():
    raise RuntimeError("CPS UI-policy source contains an invalid month.")

policy["regular_ui_weeks"] = policy["reg_UI"].astype(float)
policy["maximum_ui_weeks"] = policy["ui_weeks"].astype(float)
policy["extension_ui_weeks"] = policy["maximum_ui_weeks"] - policy["regular_ui_weeks"]
policy["source_active_extension_weeks"] = policy["ext_wks"].astype(float)
# Farber, Rothstein and Valletta (2015), archive README: policy status on day 5.
policy["policy_effective_date"] = pd.to_datetime(
    policy["year"].astype(str) + "-" + policy["month"].astype(str).str.zfill(2) + "-05"
).dt.strftime("%Y-%m-%d")
policy = policy[PUBLIC_COLUMNS].sort_values(["year", "month", "fips"], kind="mergesort").reset_index(drop=True)

duration = policy[["regular_ui_weeks", "extension_ui_weeks", "maximum_ui_weeks"]]
if (
    duration.lt(0).any().any()
    or not (duration["regular_ui_weeks"] + duration["extension_ui_weeks"]).sub(duration["maximum_ui_weeks"]).abs().lt(1e-6).all()
):
    raise RuntimeError("CPS UI-policy duration levels or decomposition are invalid.")
if len(policy) != 8_976 or policy["fips"].nunique() != 51:
    raise RuntimeError("CPS UI-policy source coverage changed from 8,976 state-month cells and 51 states.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INTERMEDIATE_PATH.parent.mkdir(parents=True, exist_ok=True)
policy.to_parquet(INTERMEDIATE_PATH, index=False)

pd.DataFrame(
    [
        {
            "source_role": role,
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(path),
            "version_or_definition": definition,
        }
        for role, path, definition in (
            ("UI state-month panel", POLICY_PATH, "Farber-Rothstein-Valletta 2015 replication archive"),
        )
    ]
).to_csv(OUTPUT_DIR / "1_source_manifest.csv", index=False)

pd.DataFrame(
    [{
        "rows": len(policy),
        "states": policy["fips"].nunique(),
        "first_policy_month": policy["policy_effective_date"].min()[:7],
        "last_policy_month": policy["policy_effective_date"].max()[:7],
        "intermediate_path": str(INTERMEDIATE_PATH.relative_to(PROJECT_ROOT)),
        "intermediate_sha256": sha256_file(INTERMEDIATE_PATH),
        "maximum_duration_decomposition_gap": float(
            (duration["regular_ui_weeks"] + duration["extension_ui_weeks"] - duration["maximum_ui_weeks"]).abs().max()
        ),
        "status": "PASS",
    }]
).to_csv(OUTPUT_DIR / "1_build_summary.csv", index=False)

print(f"Built {INTERMEDIATE_PATH.relative_to(PROJECT_ROOT)} with {len(policy):,} state-month rows.")
