#!/usr/bin/env python
"""Build the PSID job-loss annual-food-spending prediction task."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.sample_accounting import construction_step, filter_count

from scripts.utils.macro_context import CORE_MACRO_COLUMNS, attach_macro_context
from scripts.utils.psid_events import build_displacement_layers, build_source_layers
from scripts.utils.sampling import sample_task_records
from scripts.utils.table_schema import write_public_table


TASK_ID = "cons_psid_jobloss"
SEED = 42018
PANEL_PATH = PROJECT_ROOT / "data/intermediate/psid_main.parquet"
TABLE_PATH = PROJECT_ROOT / "data/householdbench/tabular/cons_psid_jobloss.csv"
OUTPUT_DIR = PROJECT_ROOT / "output/householdbench/tasks/cons_psid_jobloss"
SAMPLING_DIR = PROJECT_ROOT / "data" / "intermediate" / "householdbench" / TASK_ID
CONTROL_LEDGER_PATH = OUTPUT_DIR / "01_control_candidate_ledger.parquet"
STANDARD_STEPS = [
    "Start from source observations",
    "Attach additional sources",
    "Construct leads and lags",
    "Restrict universe",
    "Require observed targets",
    "Require observed predictors",
    "Apply logical and validity filters",
    "Apply outlier trimming",
    "Export sample",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel-path",
        type=Path,
        default=PANEL_PATH,
        help="Enriched canonical PSID person-by-wave panel.",
    )
    return parser.parse_args()


def income_inside_cutoffs(value: object, year: int, cutoffs: pd.DataFrame) -> bool:
    if pd.isna(value) or year not in cutoffs.index:
        return False
    return float(cutoffs.at[year, "p01"]) < float(value) < float(cutoffs.at[year, "p99"])


def valid_age(value: object) -> bool:
    return pd.notna(value) and 25 <= float(value) <= 65


def cash_food_value(row: pd.Series | None) -> float:
    if row is None:
        return np.nan
    home = row.get("food_at_home_nominal_annual")
    away = row.get("food_away_nominal_annual")
    if pd.isna(home) or pd.isna(away):
        return np.nan
    return float(home) + float(away)


def cash_food_valid(row: pd.Series) -> bool:
    value = cash_food_value(row)
    invalid_statuses = {None, "missing_or_topcoded"}
    return (
        np.isfinite(value)
        and value > 0
        and row.get("food_at_home_assignment_status") not in invalid_statuses
        and row.get("food_away_assignment_status") not in invalid_statuses
    )


def cash_assignment_status(row: pd.Series) -> str:
    statuses = {
        str(row.get("food_at_home_assignment_status")),
        str(row.get("food_away_assignment_status")),
    }
    if "major_assignment" in statuses:
        return "contains_major_assignment"
    if "minor_assignment" in statuses:
        return "contains_minor_assignment"
    return "all_direct_or_valid_zero"


def profile_payload(profile: pd.Series) -> dict[str, object]:
    fields = [
        "reference_demo_sex_label",
        "reference_demo_age_gen",
        "reference_edu_year",
        "partner_demo_sex_label",
        "partner_demo_age_gen",
        "partner_edu_year",
        "reference_fam_size",
        "reference_fam_size_chi",
        "reference_home_stat_label",
        "reference_geo_region_label",
        "reference_race_eth_maj_col_label",
    ]
    out = {"current_profile_survey_year": int(profile["survey_year"])}
    for field in fields:
        out[f"current_{field}"] = profile.get(field, np.nan)
    return out


def history_payload(history: list[pd.Series], target_year: int) -> dict[str, object]:
    out: dict[str, object] = {"history_depth": len(history)}
    fields = [
        "survey_year",
        "reference_annual_work_status_label",
        "reference_annual_hours",
        "reference_earn_tot_nd",
        "partner_annual_work_status_label",
        "partner_annual_hours",
        "partner_earn_tot_nd",
        "reference_finc_tot_nd",
    ]
    for lag in range(1, 4):
        for field in fields:
            out[f"history_lag{lag}_{field}"] = np.nan
        out[f"history_lag{lag}_annual_food_spending"] = np.nan
        out[f"history_lag{lag}_elapsed_years"] = np.nan
    for lag, row in enumerate(history, start=1):
        for field in fields:
            out[f"history_lag{lag}_{field}"] = row.get(field, np.nan)
        out[f"history_lag{lag}_annual_food_spending"] = cash_food_value(row)
        out[f"history_lag{lag}_elapsed_years"] = target_year - int(row["survey_year"])
    food_history = next((row for row in history if cash_food_valid(row)), None)
    out["food_baseline_survey_year"] = int(food_history["survey_year"]) if food_history is not None else np.nan
    out["prior_annual_food_spending"] = cash_food_value(food_history) if food_history is not None else np.nan
    return out


def predictor_payload(
    target: pd.Series,
    household_index: pd.DataFrame,
) -> tuple[dict[str, object] | None, str]:
    target_year = int(target["survey_year"])
    prompt_year = target_year - 1
    subject = str(target["stable_household_subject_id"])
    profile_key = (subject, prompt_year)
    lag1_key = (subject, target_year - 2)
    if profile_key not in household_index.index:
        return None, "missing_current_demographic_profile"
    if lag1_key not in household_index.index:
        return None, "missing_mandatory_pre_event_observation"
    profile = household_index.loc[profile_key]
    lag1 = household_index.loc[lag1_key]
    if isinstance(profile, pd.DataFrame) or isinstance(lag1, pd.DataFrame):
        raise RuntimeError(f"{TASK_ID}: duplicate subject-wave in household panel")

    profile_required = [
        "reference_demo_sex_label", "reference_demo_age_gen",
        "reference_fam_size", "reference_fam_size_chi",
    ]
    lag_required = [
        "reference_annual_work_status", "reference_earn_tot_nd", "reference_finc_tot_nd",
    ]
    if profile[profile_required].isna().any() or lag1[lag_required].isna().any():
        return None, "missing_mandatory_profile_or_history_predictor"
    if target["household_structure"] == "partnered":
        partner_profile = ["partner_demo_sex_label", "partner_demo_age_gen"]
        partner_history = ["partner_annual_work_status", "partner_earn_tot_nd"]
        if profile[partner_profile].isna().any() or lag1[partner_history].isna().any():
            return None, "missing_mandatory_partner_profile_or_history_predictor"

    history = [lag1]
    for year in [target_year - 3, target_year - 4]:
        key = (subject, year)
        if key not in household_index.index:
            break
        older = household_index.loc[key]
        if isinstance(older, pd.DataFrame):
            raise RuntimeError(f"{TASK_ID}: duplicate optional subject-wave")
        if older[lag_required].isna().any():
            break
        if target["household_structure"] == "partnered" and older[["partner_annual_work_status", "partner_earn_tot_nd"]].isna().any():
            break
        if older["household_structure"] != target["household_structure"]:
            break
        history.append(older)
    payload = {**profile_payload(profile), **history_payload(history, target_year)}
    if pd.isna(payload["prior_annual_food_spending"]):
        return None, "no_positive_cash_food_in_retained_history"
    return payload, "eligible"


def logical_valid(target: pd.Series, profile: pd.Series, lag1: pd.Series) -> bool:
    structure = target["household_structure"]
    if profile["household_structure"] != structure or lag1["household_structure"] != structure:
        return False
    if not valid_age(profile["reference_demo_age_gen"]):
        return False
    if structure == "partnered" and not valid_age(profile["partner_demo_age_gen"]):
        return False
    return True


def history_income_valid(payload: dict[str, object], cutoffs: pd.DataFrame) -> bool:
    for lag in range(1, int(payload["history_depth"]) + 1):
        year = int(payload[f"history_lag{lag}_survey_year"])
        value = payload[f"history_lag{lag}_reference_finc_tot_nd"]
        if not income_inside_cutoffs(value, year, cutoffs):
            return False
    return True


def prior_episode_fields(
    current_episodes: pd.DataFrame,
    episodes_by_person: dict[int, list[dict[str, object]]],
    prompt_year: int,
) -> dict[str, str]:
    result = {
        "reference_prior_displacement_years_ago": "",
        "reference_prior_displacement_reasons": "",
        "partner_prior_displacement_years_ago": "",
        "partner_prior_displacement_reasons": "",
    }
    for episode in current_episodes.to_dict("records"):
        role = "reference" if episode["person_role"] == "reference_person" else "partner"
        previous = [
            row for row in episodes_by_person.get(int(episode["person_id"]), [])
            if int(row["assigned_event_year"]) < int(episode["assigned_event_year"])
        ]
        result[f"{role}_prior_displacement_years_ago"] = "|".join(
            str(prompt_year - int(row["assigned_event_year"])) for row in previous
        )
        result[f"{role}_prior_displacement_reasons"] = "|".join(
            str(row["displacement_reason"]) for row in previous
        )
    return result


def attach_macro(frame: pd.DataFrame) -> pd.DataFrame:
    macro_keys = pd.read_parquet(
        PROJECT_ROOT / "data/intermediate/householdbench_macro_context.parquet",
        columns=["eligible_origin_q_index", "macro_reference_year", "macro_reference_quarter"],
    )
    macro_keys = macro_keys.loc[macro_keys["macro_reference_quarter"].eq(4)].rename(
        columns={"macro_reference_year": "assigned_event_year"}
    )
    frame = frame.merge(macro_keys, on="assigned_event_year", how="left", validate="many_to_one")
    if frame["eligible_origin_q_index"].isna().any():
        years = sorted(frame.loc[frame["eligible_origin_q_index"].isna(), "assigned_event_year"].unique())
        raise RuntimeError(f"{TASK_ID}: missing Q4 macro keys for {years}")
    frame = attach_macro_context(frame, origin_q_index_col="eligible_origin_q_index", required_columns=CORE_MACRO_COLUMNS)
    frame["macro_reference_year"] = frame["assigned_event_year"]
    frame["macro_reference_quarter"] = 4
    return frame


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    source = build_source_layers(args.panel_path)
    households = source.households.copy()
    cutoffs = source.income_cutoffs.set_index("survey_year")

    # Steps 1--3 operate on the common one-row-per-household-wave PSID
    # intermediate. Attach the event-year macro block and annual family-income
    # cutoffs without changing the row universe, then construct every available
    # demographic profile and one-to-three-wave history before selecting shocks.
    stage_counts = [len(households)]
    households["assigned_event_year"] = households["survey_year"].astype(int) - 1
    attached_households = attach_macro(households)
    attached_households = attached_households.merge(
        source.income_cutoffs.rename(columns={"p01": "family_income_p01", "p99": "family_income_p99"}),
        on="survey_year",
        how="left",
        validate="many_to_one",
    )
    if len(attached_households) != len(households) or attached_households[["family_income_p01", "family_income_p99"]].isna().any().any():
        raise RuntimeError(f"{TASK_ID}: Step 2 macro or income-cutoff attachment changed or incompletely covered the household-wave source.")
    stage_counts.append(len(attached_households))
    household_index = attached_households.set_index(
        ["stable_household_subject_id", "survey_year"], drop=False, verify_integrity=True
    )
    predictor_lookup = {
        (str(target["stable_household_subject_id"]), int(target["survey_year"])):
        predictor_payload(pd.Series(target), household_index)
        for target in attached_households.to_dict("records")
    }
    if len(predictor_lookup) != len(attached_households):
        raise RuntimeError(f"{TASK_ID}: Step 3 history construction changed household-wave key uniqueness.")
    stage_counts.append(len(attached_households))

    # Step 4 uses the common adult displacement reports to form event windows
    # and a deterministic subsample of never-displaced household controls.
    displacement = build_displacement_layers(source.adults)
    episodes = displacement.episodes
    reports = displacement.reports
    shocks = displacement.shocks
    episode_by_id = episodes.set_index("episode_id", verify_integrity=True)
    episodes_by_person = {
        int(person): group.sort_values(["assigned_event_year", "episode_id"]).to_dict("records")
        for person, group in episodes.groupby("person_id")
    }
    shock_by_key = {
        (str(row["stable_household_subject_id"]), int(row["survey_year"])): row
        for row in shocks.to_dict("records")
    }
    prior_report_years = {
        int(person): sorted(group["survey_year"].astype(int).tolist())
        for person, group in reports.groupby("person_id")
    }

    control_ledger = []
    control_candidates = []
    for target in attached_households.to_dict("records"):
        year = int(target["survey_year"])
        subject = str(target["stable_household_subject_id"])
        candidate_id = f"psid_control_{subject}_{year}"
        status = "eligible_universe"
        if int(target["relevant_adults_ascertained"]) != 1:
            status = "incomplete_relevant_adult_ascertainment"
        elif (subject, year) in shock_by_key:
            status = "current_household_shock"
        else:
            people = [int(target["reference_person_id"])]
            if pd.notna(target["partner_person_id"]):
                people.append(int(target["partner_person_id"]))
            if any(any(report_year < year for report_year in prior_report_years.get(person, [])) for person in people):
                status = "prior_displacement"
        row = {
            "candidate_id": candidate_id,
            "stable_household_subject_id": subject,
            "survey_year": year,
            "status": status,
        }
        control_ledger.append(row)
        if status == "eligible_universe":
            control_candidates.append({**row, "target": target})

    controls = pd.DataFrame([
        {
            "candidate_id": row["candidate_id"],
            "stable_household_subject_id": row["stable_household_subject_id"],
            "survey_year": row["survey_year"],
            "control_hash": hashlib.sha256(
                f"{SEED}|{row['stable_household_subject_id']}|{row['survey_year']}".encode()
            ).hexdigest(),
            "target": row["target"],
        }
        for row in control_candidates
    ])
    if not controls.empty:
        controls = controls.sort_values(
            ["stable_household_subject_id", "control_hash"], kind="mergesort"
        ).drop_duplicates("stable_household_subject_id", keep="first")
    selected_control_ids = set(controls["candidate_id"]) if not controls.empty else set()
    for row in control_ledger:
        if row["status"] == "eligible_universe":
            row["status"] = "selected_universe_control" if row["candidate_id"] in selected_control_ids else "not_selected_universe_control"
    control_ledger_frame = pd.DataFrame(control_ledger)
    control_ledger_frame.to_parquet(CONTROL_LEDGER_PATH, index=False)
    control_ledger_frame.groupby("status", as_index=False).size().rename(columns={"size": "candidate_windows"}).to_csv(
        OUTPUT_DIR / "01_control_sample_waterfall.csv", index=False
    )

    candidates: list[dict[str, object]] = []
    event_universe_audit = []
    for shock in shocks.to_dict("records"):
        key = (str(shock["stable_household_subject_id"]), int(shock["survey_year"]))
        if key not in household_index.index:
            event_universe_audit.append({"candidate_id": shock["household_shock_id"], "status": "missing_target_household"})
            continue
        candidates.append({"observation_type": "event", "target": household_index.loc[key], "shock": shock})
        event_universe_audit.append({"candidate_id": shock["household_shock_id"], "status": "eligible_universe"})
    for row in controls.to_dict("records"):
        candidates.append({"observation_type": "control", "target": pd.Series(row["target"]), "candidate_id": row["candidate_id"]})

    stage_counts.append(len(candidates))
    stage5 = []
    stage6 = []
    stage7 = []
    stage8 = []
    candidate_audit = []
    for candidate in candidates:
        target = candidate["target"]
        candidate_id = candidate["shock"]["household_shock_id"] if candidate["observation_type"] == "event" else candidate["candidate_id"]
        audit = {"candidate_id": candidate_id, "observation_type": candidate["observation_type"], "status": "eligible_universe"}
        if not cash_food_valid(target):
            audit["status"] = "target_cash_food_not_positive_complete"
            candidate_audit.append(audit)
            continue
        stage5.append(candidate)
        subject = str(target["stable_household_subject_id"])
        target_year = int(target["survey_year"])
        payload, status = predictor_lookup[(subject, target_year)]
        if payload is None:
            audit["status"] = status
            candidate_audit.append(audit)
            continue
        candidate = {**candidate, "payload": payload}
        stage6.append(candidate)
        profile = household_index.loc[(subject, target_year - 1)]
        lag1 = household_index.loc[(subject, target_year - 2)]
        if not logical_valid(target, profile, lag1):
            audit["status"] = "profile_age_or_household_structure_invalid"
            candidate_audit.append(audit)
            continue
        stage7.append(candidate)
        if not history_income_valid(payload, cutoffs):
            audit["status"] = "displayed_family_income_outside_strict_p01_p99"
            candidate_audit.append(audit)
            continue
        stage8.append(candidate)
        audit["status"] = "exported"
        candidate_audit.append(audit)

    stage_counts.extend([len(stage5), len(stage6), len(stage7), len(stage8), len(stage8)])
    event_audit = pd.DataFrame(event_universe_audit + [
        {"candidate_id": row["candidate_id"], "status": row["status"]}
        for row in candidate_audit if row["observation_type"] == "event"
    ])
    event_audit.groupby("status", as_index=False).size().rename(columns={"size": "candidate_windows"}).to_csv(
        OUTPUT_DIR / "01_event_sample_waterfall.csv", index=False
    )
    pd.DataFrame(candidate_audit).to_csv(OUTPUT_DIR / "01_sample_candidate_lineage.csv", index=False)

    rows = []
    for candidate in stage8:
        target = candidate["target"]
        payload = candidate["payload"]
        year = int(target["survey_year"])
        subject = str(target["stable_household_subject_id"])
        base = {
            **payload,
            "stable_household_subject_id": subject,
            "survey_year": year,
            "assigned_event_year": year - 1,
            "fuid": target["fuid"],
            "split_group_id": target["split_group_id"],
            "target_survey_year": year,
            "household_structure": target["household_structure"],
            "reference_person_id": int(target["reference_person_id"]),
            "partner_person_id": target["partner_person_id"],
            "annual_food_spending": cash_food_value(target),
            "food_at_home_target": float(target["food_at_home_nominal_annual"]),
            "food_away_target": float(target["food_away_nominal_annual"]),
            "target_assignment_status": cash_assignment_status(target),
            "family_weight": target["reference_fw"],
            "release_date": str(target["reference_release_date"])[:10],
            **{column: target[column] for column in CORE_MACRO_COLUMNS},
            "eligible_origin_q_index": target["eligible_origin_q_index"],
            "macro_reference_year": target["macro_reference_year"],
            "macro_reference_quarter": target["macro_reference_quarter"],
        }
        if candidate["observation_type"] == "event":
            shock = candidate["shock"]
            ids = str(shock["episode_ids"]).split("|")
            current_episodes = episode_by_id.loc[ids]
            if isinstance(current_episodes, pd.Series):
                current_episodes = current_episodes.to_frame().T
            base.update({
                **shock,
                "observation_type": "event",
                "displaced_adult_roles": "|".join(sorted(current_episodes["person_role"].astype(str))),
                "displaced_adult_sexes": "|".join(sorted(current_episodes["person_sex_label"].astype(str))),
                "reference_current_displacement_reasons": "|".join(
                    current_episodes.loc[
                        current_episodes["person_role"].eq("reference_person"),
                        "displacement_reason",
                    ].astype(str)
                ),
                "partner_current_displacement_reasons": "|".join(
                    current_episodes.loc[
                        current_episodes["person_role"].eq("spouse_partner"),
                        "displacement_reason",
                    ].astype(str)
                ),
                **prior_episode_fields(current_episodes, episodes_by_person, year - 1),
            })
        else:
            candidate_id = candidate["candidate_id"]
            base.update({
                "household_shock_id": candidate_id,
                "episode_ids": "",
                "displaced_person_ids": "",
                "number_displaced_adults": 0,
                "displacement_reasons": "none",
                "observation_type": "control",
                "displaced_adult_roles": "none",
                "displaced_adult_sexes": "none",
                "reference_current_displacement_reasons": "",
                "partner_current_displacement_reasons": "",
                "reference_prior_displacement_years_ago": "",
                "reference_prior_displacement_reasons": "",
                "partner_prior_displacement_years_ago": "",
                "partner_prior_displacement_reasons": "",
            })
        rows.append(base)

    pool = pd.DataFrame(rows)
    if pool.empty or set(pool["observation_type"]) != {"event", "control"}:
        raise RuntimeError(f"{TASK_ID}: final pool lacks event or control observations")
    pool["naive_annual_food_spending"] = pool["prior_annual_food_spending"]
    pool["row_id"] = np.where(
        pool["observation_type"].eq("event"),
        pool["household_shock_id"],
        "psid_control_" + pool["stable_household_subject_id"].astype(str) + "_" + pool["target_survey_year"].astype(int).astype(str),
    )
    pool["subject_id"] = pool["stable_household_subject_id"]
    pool["group_id"] = pool["split_group_id"]
    pool["selection_period"] = pool["assigned_event_year"].astype(int).astype(str)
    pool["time"] = pool["assigned_event_year"].astype(int).astype(str) + "-12-31"
    if pool["row_id"].duplicated().any():
        raise RuntimeError(f"{TASK_ID}: duplicate private row IDs")
    pool["id"] = pool["row_id"].astype("string")
    # Public keys are fixed before sampling; source keys retain their sort types.
    pool["release_date"] = pd.to_datetime(pool["release_date"], errors="raise").dt.strftime(
        "%Y-%m-%d"
    )
    selected, _, _ = sample_task_records(
        pool,
        task_id=TASK_ID,
        output_dir=SAMPLING_DIR,
        output_sort_cols=["assigned_event_year", "row_id"],
    )
    stage_counts[-1] = len(selected)
    write_public_table(selected, task_id=TASK_ID, public_path=TABLE_PATH)
    events = selected["observation_type"].eq("event")
    controls_selected = selected["observation_type"].eq("control")
    event_reasons = {
        "layoff_or_firing",
        "plant_closure_or_employer_move",
        "layoff_or_firing|plant_closure_or_employer_move",
    }
    final_invariants = pd.DataFrame({
        "id": selected["id"].astype("string"),
        "time": selected["time"].astype("string"),
        "event_has_one_or_two_displaced_adults": (~events | selected["number_displaced_adults"].between(1, 2)).astype(int),
        "control_has_no_displaced_adult": (~controls_selected | selected["number_displaced_adults"].eq(0)).astype(int),
        "event_has_episode_lineage": (~events | selected["episode_ids"].astype(str).str.len().gt(0)).astype(int),
        "control_has_no_episode_lineage": (~controls_selected | selected["episode_ids"].fillna("").astype(str).eq("")).astype(int),
        "eligible_displacement_reason": (
            (events & selected["displacement_reasons"].isin(event_reasons))
            | (controls_selected & selected["displacement_reasons"].eq("none"))
        ).astype(int),
        "target_wave_is_one_after_event_year": selected["target_survey_year"].eq(selected["assigned_event_year"] + 1).astype(int),
        "mandatory_history_is_two_years_before_target": selected["history_lag1_elapsed_years"].eq(2).astype(int),
        "food_target_and_naive_baseline_positive": (
            selected["annual_food_spending"].gt(0) & selected["prior_annual_food_spending"].gt(0)
        ).astype(int),
        "macro_context_complete": selected[CORE_MACRO_COLUMNS].notna().all(axis=1).astype(int),
    })
    if not final_invariants.drop(columns=["id", "time"]).eq(1).all().all():
        raise RuntimeError(f"{TASK_ID}: a selected row fails the final household-event validation checks.")
    final_invariants.to_csv(OUTPUT_DIR / "01_final_event_invariants.csv", index=False)
    cutoffs.reset_index().to_csv(OUTPUT_DIR / "01_outlier_cutoffs.csv", index=False)
    step_rows = []
    for index, (label, after) in enumerate(zip(STANDARD_STEPS, stage_counts, strict=True), start=1):
        before = stage_counts[index - 2] if index > 1 else stage_counts[0]
        step_rows.append(construction_step(
                             TASK_ID, index, label,
                             before, after, before - after,
                         ))
    step_frame = pd.DataFrame(step_rows)
    step_frame.to_csv(OUTPUT_DIR / "01_sample_construction_steps.csv", index=False)
    status_counts = pd.Series([row["status"] for row in candidate_audit]).value_counts()
    detail_specs = [
        (1, "common_psid_household_waves", "start from every unique household-wave record in the validated common PSID intermediate", stage_counts[0] - stage_counts[0]),
        (2, "macro_release_and_income_cutoffs_attached", "attach the event-year macro block, PSID release metadata, and survey-year family-income cutoffs to every household wave", stage_counts[0] - stage_counts[1]),
        (3, "profiles_and_histories_constructed", "construct the available current profile and up to three earlier household histories for every household-wave key before displacement selection", stage_counts[1] - stage_counts[2]),
        (4, "job_loss_event_or_selected_control_window", "retain household waves containing an eligible layoff, firing, plant closure, employer move, or cessation report, plus one deterministic never-displaced control wave per household", stage_counts[2] - stage_counts[3]),
        (5, "positive_complete_cash_food_target", "require positive annual cash food spending with valid food-at-home and food-away source assignments", int(status_counts.get("target_cash_food_not_positive_complete", 0))),
        (6, "current_household_profile_observed", "require the household profile in the event calendar year", int(status_counts.get("missing_current_demographic_profile", 0))),
        (6, "mandatory_pre_event_history_observed", "require the household history two years before the target survey wave", int(status_counts.get("missing_mandatory_pre_event_observation", 0))),
        (6, "reference_predictors_observed", "require the reference person's sex, age, family composition, work status, labor earnings, and total family income", int(status_counts.get("missing_mandatory_profile_or_history_predictor", 0))),
        (6, "partner_predictors_observed_when_partnered", "for partnered households, require the partner's sex, age, work status, and labor earnings", int(status_counts.get("missing_mandatory_partner_profile_or_history_predictor", 0))),
        (6, "positive_food_history_observed", "require positive annual cash food spending in at least one retained pre-event history wave", int(status_counts.get("no_positive_cash_food_in_retained_history", 0))),
        (7, "stable_structure_and_adult_ages", "require household structure to agree across the target, profile, and mandatory history waves and require each current adult to be age 25 through 65", int(status_counts.get("profile_age_or_household_structure_invalid", 0))),
        (8, "displayed_family_income_inside_cutoffs", "require every displayed family-income history value to lie strictly inside its survey-year 1st and 99th percentiles", int(status_counts.get("displayed_family_income_outside_strict_p01_p99", 0))),
        (9, "deterministic_export_cap", "apply the registered deterministic sampling order and export all eligible rows because the sample is below the task cap", stage_counts[7] - stage_counts[8]),
    ]
    detail_rows = []
    for step in range(1, 10):
        for order, (_, filter_id, condition, fail_count) in enumerate(
            [spec for spec in detail_specs if spec[0] == step], start=1
        ):
            detail_rows.append(filter_count(
                                   TASK_ID, step, order,
                                   filter_id, condition, fail_count,
                               ))
    pd.DataFrame(detail_rows).to_csv(OUTPUT_DIR / "01_sample_construction_drop_details.csv", index=False)
    print(
        f"{TASK_ID}: wrote {len(selected):,} rows: "
        f"{int(selected['observation_type'].eq('event').sum()):,} events and "
        f"{int(selected['observation_type'].eq('control').sum()):,} controls"
    )


if __name__ == "__main__":
    main()
