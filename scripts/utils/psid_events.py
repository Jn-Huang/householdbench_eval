"""Shared in-memory PSID household, displacement-report, and episode construction.

The only PSID microdata input is the enriched canonical person-by-wave panel.
This module constructs task-layer reports and episodes in memory.  None of
those task constructs are required to exist in the canonical panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


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
ELIGIBLE_EPISODE_CLASSES = {"first_observed_episode", "confirmed_distinct_later_episode"}
EARNINGS_TOPCODE = 9_999_999

FOOD_COLUMNS = [
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

PANEL_COLUMNS = [
    "id", "year", "income_year", "fuid", "rel", "panel_current", "response",
    "sample_person", "subsample", "fam_partnered", "demo_sex", "demo_age_gen",
    "edu_year", "edu_year_max", "earn_tot_nd", "earn_tot_ndf", "finc_tot_nd",
    "fam_size", "fam_size_chi", "home_stat", "geo_region", "fw", "release_date",
    "emp_work", "race_eth_maj_col", "demo_sex_label", "race_eth_maj_col_label",
    "geo_region_label", "home_stat_label", "emp_work_label", "reference_person_id",
    "spouse_person_id", "labor_reference_year", "annual_hours_rp", "annual_hours_sp",
    "annual_hours_valid_rp", "annual_hours_valid_sp", "annual_hours_accuracy_code_rp",
    "annual_hours_accuracy_code_sp", "annual_hours_assignment_status_rp",
    "annual_hours_assignment_status_sp", "annual_work_status_rp", "annual_work_status_sp",
    "job_end_reason_ascertained_rp", "job_end_reason_ascertained_sp",
    "job_end_reason_employed_code_rp", "job_end_reason_employed_code_sp",
    "job_end_reason_unemployed_code_rp", "job_end_reason_unemployed_code_sp",
    *FOOD_COLUMNS,
]


@dataclass(frozen=True)
class PsidSourceLayers:
    adults: pd.DataFrame
    households: pd.DataFrame
    income_cutoffs: pd.DataFrame
    ambiguous_family_roles: pd.DataFrame


@dataclass(frozen=True)
class PsidDisplacementLayers:
    reports: pd.DataFrame
    episodes: pd.DataFrame
    shocks: pd.DataFrame


def connected_components(pair_rows: pd.DataFrame, people: pd.Series) -> dict[int, str]:
    parent = {int(person): int(person) for person in people.dropna().astype(int).unique()}

    def find(person: int) -> int:
        while parent[person] != person:
            parent[person] = parent[parent[person]]
            person = parent[person]
        return person

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            small, large = sorted([root_first, root_second])
            parent[large] = small

    for row in pair_rows.dropna(subset=["person_id", "partner_person_id"]).itertuples(index=False):
        union(int(row.person_id), int(row.partner_person_id))
    groups: dict[int, list[int]] = {}
    for person in sorted(parent):
        groups.setdefault(find(person), []).append(person)
    return {
        person: f"psid_component_{min(members)}"
        for members in groups.values()
        for person in members
    }


def require_enriched_schema(panel_path: Path) -> None:
    if not panel_path.is_file():
        raise RuntimeError(f"Missing canonical PSID panel: {panel_path}")
    columns = set(pq.ParquetFile(panel_path).schema_arrow.names)
    missing = sorted(set(PANEL_COLUMNS) - columns)
    if missing:
        raise RuntimeError(
            f"Canonical PSID panel lacks Complete Main Study enrichment fields: {missing}. "
            "Validate the common enriched PSID intermediate before constructing tasks."
        )


def build_reference_adults(panel_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_enriched_schema(panel_path)
    panel = pd.read_parquet(
        panel_path,
        columns=PANEL_COLUMNS,
        filters=[("year", ">=", 1968), ("year", "<=", 1992)],
    )
    adults = panel.loc[
        panel["panel_current"].eq(1)
        & panel["subsample"].eq("src")
        & panel["fuid"].notna()
        & panel["rel"].isin([1, 2])
    ].copy()
    adults = adults.rename(columns={"id": "person_id", "year": "survey_year"})
    adults["person_id"] = pd.to_numeric(adults["person_id"], errors="raise").astype("int64")
    adults["survey_year"] = pd.to_numeric(adults["survey_year"], errors="raise").astype(int)
    adults["fuid"] = pd.to_numeric(adults["fuid"], errors="raise")

    duplicate = adults.duplicated(["survey_year", "fuid", "rel"], keep=False)
    ambiguous = adults.loc[duplicate, ["survey_year", "fuid"]].drop_duplicates()
    if not ambiguous.empty:
        adults = adults.merge(ambiguous.assign(_ambiguous=1), on=["survey_year", "fuid"], how="left")
        adults = adults.loc[adults["_ambiguous"].isna()].drop(columns="_ambiguous")
    if adults.duplicated(["person_id", "survey_year"]).any():
        raise RuntimeError("Reference-adult panel is not unique by person and wave.")

    adults["role"] = adults["rel"].map({1: "reference_person", 2: "spouse_partner"})
    adults["partner_person_id"] = np.where(
        adults["rel"].eq(1),
        adults["spouse_person_id"],
        adults["reference_person_id"],
    )
    adults["household_structure"] = np.where(
        adults["partner_person_id"].notna(), "partnered", "unpartnered"
    )
    adults["stable_household_subject_id"] = np.where(
        adults["partner_person_id"].notna(),
        "psid_pair_"
        + adults[["person_id", "partner_person_id"]].min(axis=1).astype("int64").astype(str)
        + "_"
        + adults[["person_id", "partner_person_id"]].max(axis=1).astype("int64").astype(str),
        "psid_single_" + adults["person_id"].astype(str),
    )
    adults["split_group_id"] = adults["person_id"].map(
        connected_components(adults, adults["person_id"])
    )

    for stem in [
        "annual_hours", "annual_hours_valid", "annual_hours_accuracy_code",
        "annual_hours_assignment_status", "annual_work_status",
    ]:
        adults[stem] = np.where(
            adults["role"].eq("reference_person"),
            adults[f"{stem}_rp"],
            adults[f"{stem}_sp"],
        )
    adults["annual_work_status_label"] = adults["annual_work_status"].map(
        {0.0: "did not work positive annual hours", 1.0: "worked positive annual hours"}
    )
    adults["annual_earnings_source_value"] = pd.to_numeric(adults["earn_tot_nd"], errors="coerce")
    adults["annual_earnings_valid"] = (
        adults["annual_earnings_source_value"].ge(0)
        & adults["annual_earnings_source_value"].ne(EARNINGS_TOPCODE)
    ).astype("int8")
    adults.loc[adults["annual_earnings_valid"].eq(0), "earn_tot_nd"] = np.nan
    if not adults["labor_reference_year"].eq(adults["income_year"]).all():
        raise RuntimeError("Complete Main Study hours and PSID-SHELF earnings reference years differ.")

    for stem in [
        "job_end_reason_ascertained",
        "job_end_reason_employed_code",
        "job_end_reason_unemployed_code",
    ]:
        adults[stem] = np.where(
            adults["role"].eq("reference_person"),
            adults[f"{stem}_rp"],
            adults[f"{stem}_sp"],
        )
    adults = adults.rename(
        columns={
            "job_end_reason_ascertained": "displacement_ascertained",
            "job_end_reason_employed_code": "displacement_employed_code",
            "job_end_reason_unemployed_code": "displacement_unemployed_code",
        }
    )
    adults["displacement_ascertained"] = adults["displacement_ascertained"].astype("int8")
    return (
        adults.sort_values(["survey_year", "fuid", "rel"], kind="mergesort").reset_index(drop=True),
        ambiguous,
    )


def build_households(adults: pd.DataFrame) -> pd.DataFrame:
    reference = adults.loc[adults["role"].eq("reference_person")].copy()
    partner = adults.loc[adults["role"].eq("spouse_partner")].copy()
    keys = ["survey_year", "fuid"]
    reference = reference.rename(
        columns={column: f"reference_{column}" for column in reference.columns if column not in keys}
    )
    partner = partner.rename(
        columns={column: f"partner_{column}" for column in partner.columns if column not in keys}
    )
    household = reference.merge(partner, on=keys, how="left", validate="one_to_one")
    household = household.rename(
        columns={
            "reference_stable_household_subject_id": "stable_household_subject_id",
            "reference_split_group_id": "split_group_id",
            "reference_household_structure": "household_structure",
        }
    )
    household["relevant_adults_ascertained"] = np.where(
        household["household_structure"].eq("partnered"),
        household["reference_displacement_ascertained"].eq(1)
        & household["partner_displacement_ascertained"].eq(1),
        household["reference_displacement_ascertained"].eq(1),
    ).astype("int8")
    for column in FOOD_COLUMNS:
        household[column] = household[f"reference_{column}"]
    if household.duplicated(["survey_year", "fuid"]).any():
        raise RuntimeError("Household panel is not unique by family and wave.")
    return household.sort_values(["survey_year", "fuid"], kind="mergesort").reset_index(drop=True)


def build_reports(adults: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for adult in adults.loc[adults["displacement_ascertained"].eq(1)].to_dict("records"):
        employed = adult["displacement_employed_code"]
        unemployed = adult["displacement_unemployed_code"]
        employed_qualifies = pd.notna(employed) and int(employed) in QUALIFYING_REASON_CODES
        unemployed_qualifies = pd.notna(unemployed) and int(unemployed) in QUALIFYING_REASON_CODES
        combined_1968_route = (
            adult["role"] == "reference_person" and int(adult["survey_year"]) == 1968
        )
        if employed_qualifies and unemployed_qualifies and not combined_1968_route:
            raise RuntimeError(
                f"Both job-ending reason routes qualify for person {adult['person_id']} "
                f"in survey year {adult['survey_year']}."
            )
        if not employed_qualifies and not unemployed_qualifies:
            continue
        if unemployed_qualifies and not combined_1968_route:
            route = "unemployed"
            code = int(unemployed)
        else:
            route = "combined_employed_or_unemployed" if combined_1968_route else "employed"
            code = int(employed)
        canonical_field = (
            "job_end_reason_unemployed_code"
            if route == "unemployed"
            else "job_end_reason_employed_code"
        )
        rows.append(
            {
                "report_id": f"psid_report_{adult['person_id']}_{adult['survey_year']}_{route}",
                "person_id": int(adult["person_id"]),
                "partner_person_id": adult["partner_person_id"],
                "survey_year": int(adult["survey_year"]),
                "assigned_event_year": int(adult["survey_year"] - 1),
                "fuid": adult["fuid"],
                "stable_household_subject_id": adult["stable_household_subject_id"],
                "split_group_id": adult["split_group_id"],
                "person_role": adult["role"],
                "person_sex": adult["demo_sex"],
                "person_sex_label": adult["demo_sex_label"],
                "source_route": route,
                "source_variable": canonical_field,
                "source_reason_code": code,
                "source_reason_label": REASON_LABELS[code],
                "displacement_reason": HARMONISED_REASONS[code],
                "is_1968_prior_marker": int(adult["survey_year"] == 1968),
            }
        )
    reports = pd.DataFrame(rows)
    if reports.empty or reports["report_id"].duplicated().any():
        raise RuntimeError("Qualifying PSID report registry is empty or has duplicate IDs.")
    return reports.sort_values(["person_id", "survey_year", "report_id"], kind="mergesort").reset_index(drop=True)


def adjudicate_person_reports(reports: pd.DataFrame, adults: pd.DataFrame) -> pd.DataFrame:
    panel = adults.copy()
    for column in ["emp_work", "annual_hours", "earn_tot_nd"]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel["positive_work_evidence"] = (
        panel["emp_work"].eq(1)
        | panel["annual_hours"].gt(0)
        | panel["earn_tot_nd"].gt(0)
    )
    evidence_years = {
        int(person): set(group.loc[group["positive_work_evidence"], "survey_year"].astype(int))
        for person, group in panel.groupby("person_id", sort=False)
    }

    classified: list[dict[str, object]] = []
    for person, group in reports.groupby("person_id", sort=False):
        group = group.sort_values(["survey_year", "report_id"], kind="mergesort")
        prior_confirmed_years: list[int] = []
        left_censored = False
        previous_report_year: int | None = None
        previous_assigned_year: int | None = None
        for report in group.to_dict("records"):
            survey_year = int(report["survey_year"])
            assigned_year = int(report["assigned_event_year"])
            years_since = assigned_year - prior_confirmed_years[-1] if prior_confirmed_years else pd.NA
            if bool(report["is_1968_prior_marker"]):
                classification = "prior_history_marker"
                reason = "1968_retrospective_marker_not_scored"
                left_censored = True
                prior_count = 0
                prior_confirmed_years.append(assigned_year)
            elif not prior_confirmed_years:
                classification = "first_observed_episode"
                reason = "first_qualifying_report_without_prior_marker"
                prior_count = 0
                prior_confirmed_years.append(assigned_year)
            else:
                prior_count = len(prior_confirmed_years)
                intervening = (
                    any(
                        previous_report_year < year < survey_year
                        for year in evidence_years.get(int(person), set())
                    )
                    if previous_report_year is not None
                    else False
                )
                if previous_assigned_year is not None and assigned_year <= previous_assigned_year:
                    classification = "probable_duplicate_or_continuation"
                    reason = "same_or_earlier_assigned_event_year"
                elif previous_report_year is not None and survey_year - previous_report_year <= 1:
                    classification = "probable_duplicate_or_continuation"
                    reason = "adjacent_report_without_distinct_job_end_evidence"
                elif intervening:
                    classification = "confirmed_distinct_later_episode"
                    reason = "positive_intervening_work_or_new_job_evidence"
                    prior_confirmed_years.append(assigned_year)
                else:
                    classification = "unresolved_repeat_status"
                    reason = "later_report_without_positive_distinct_job_evidence"
            report.update(
                {
                    "repeat_classification": classification,
                    "repeat_reason_code": reason,
                    "prior_confirmed_episode_count": prior_count,
                    "years_since_previous_confirmed_episode": years_since,
                    "displacement_history_left_censored": int(left_censored),
                    "eligible_episode": int(classification in ELIGIBLE_EPISODE_CLASSES),
                }
            )
            classified.append(report)
            previous_report_year = survey_year
            previous_assigned_year = assigned_year
    return pd.DataFrame(classified)


def build_episodes(reports: pd.DataFrame, adults: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    classified = adjudicate_person_reports(reports, adults)
    episodes = classified.loc[
        classified["eligible_episode"].eq(1) & classified["is_1968_prior_marker"].eq(0)
    ].copy()
    episodes["episode_id"] = (
        "psid_episode_"
        + episodes["person_id"].astype(str)
        + "_"
        + episodes["assigned_event_year"].astype(str)
        + "_"
        + episodes["source_route"].astype(str)
    )
    if episodes["episode_id"].duplicated().any():
        raise RuntimeError("PSID episode IDs are not unique.")
    columns = [
        "episode_id", "report_id", "person_id", "partner_person_id", "survey_year",
        "assigned_event_year", "fuid", "stable_household_subject_id", "split_group_id",
        "person_role", "person_sex", "person_sex_label", "source_route", "source_variable",
        "source_reason_code", "source_reason_label", "displacement_reason",
        "repeat_classification", "repeat_reason_code", "prior_confirmed_episode_count",
        "years_since_previous_confirmed_episode", "displacement_history_left_censored",
    ]
    return classified, episodes.loc[:, columns].reset_index(drop=True)


def build_shocks(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (subject, survey_year), group in episodes.groupby(
        ["stable_household_subject_id", "survey_year"], sort=True
    ):
        episode_ids = sorted(group["episode_id"].astype(str))
        person_ids = sorted(group["person_id"].astype(int).unique())
        rows.append(
            {
                "household_shock_id": f"psid_household_shock_{subject}_{int(survey_year)}",
                "stable_household_subject_id": subject,
                "survey_year": int(survey_year),
                "assigned_event_year": int(survey_year) - 1,
                "fuid": group["fuid"].iloc[0],
                "split_group_id": group["split_group_id"].iloc[0],
                "episode_ids": "|".join(episode_ids),
                "displaced_person_ids": "|".join(map(str, person_ids)),
                "number_displaced_adults": len(person_ids),
                "displacement_reasons": "|".join(sorted(group["displacement_reason"].unique())),
                "max_prior_confirmed_episode_count": int(group["prior_confirmed_episode_count"].max()),
                "any_left_censored_history": int(group["displacement_history_left_censored"].max()),
            }
        )
    shocks = pd.DataFrame(rows)
    if shocks.duplicated(["stable_household_subject_id", "survey_year"]).any():
        raise RuntimeError("Household shocks are not unique by subject and survey wave.")
    return shocks


def family_income_cutoffs(panel_path: Path) -> pd.DataFrame:
    source = pd.read_parquet(
        panel_path,
        columns=["id", "year", "response", "sample_person", "finc_tot_nd"],
        filters=[
            ("response", "=", 0),
            ("sample_person", "=", True),
            ("year", ">=", 1968),
            ("year", "<=", 1992),
        ],
    )
    source["finc_tot_nd"] = pd.to_numeric(source["finc_tot_nd"], errors="coerce")
    rows = []
    for year, group in source.groupby("year", sort=True):
        values = group["finc_tot_nd"].dropna()
        p01, p99 = values.quantile([0.01, 0.99]).tolist()
        if not p01 < p99:
            raise RuntimeError(f"Degenerate family-income cutoffs in {year}.")
        rows.append(
            {
                "survey_year": int(year),
                "source_person_year_rows": len(group),
                "nonmissing_family_income_rows": len(values),
                "p01": float(p01),
                "p99": float(p99),
                "n_at_or_below_p01": int(values.le(p01).sum()),
                "n_at_or_above_p99": int(values.ge(p99).sum()),
                "quantile_method": "pandas Series.quantile linear interpolation",
                "source_population": "response==0 and sample_person==True before task selection",
            }
        )
    return pd.DataFrame(rows)


def build_source_layers(panel_path: Path) -> PsidSourceLayers:
    """Construct the common household-wave source before task-universe selection."""
    adults, ambiguous = build_reference_adults(panel_path)
    households = build_households(adults)
    cutoffs = family_income_cutoffs(panel_path)
    return PsidSourceLayers(
        adults=adults,
        households=households,
        income_cutoffs=cutoffs,
        ambiguous_family_roles=ambiguous,
    )


def build_displacement_layers(adults: pd.DataFrame) -> PsidDisplacementLayers:
    """Select reports, episodes, and household shocks for Step 4 universe restriction."""
    raw_reports = build_reports(adults)
    reports, episodes = build_episodes(raw_reports, adults)
    shocks = build_shocks(episodes)
    return PsidDisplacementLayers(
        reports=reports,
        episodes=episodes,
        shocks=shocks,
    )
