#!/usr/bin/env python
"""Build the four-target PSID added-worker prediction task."""

from __future__ import annotations

import argparse
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


TASK_ID = "labor_psid_addedworker"
SEED = 42019
PANEL_PATH = PROJECT_ROOT / "data/intermediate/psid_main.parquet"
TABLE_PATH = PROJECT_ROOT / "data/householdbench/tabular/labor_psid_addedworker.csv"
OUTPUT_DIR = PROJECT_ROOT / "output/householdbench/tasks/labor_psid_addedworker"
SAMPLING_DIR = PROJECT_ROOT / "data" / "intermediate" / "householdbench" / TASK_ID
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


def income_inside_cutoffs(value: object, survey_year: int, cutoffs: pd.DataFrame) -> bool:
    if pd.isna(value) or survey_year not in cutoffs.index:
        return False
    return float(cutoffs.at[survey_year, "p01"]) < float(value) < float(cutoffs.at[survey_year, "p99"])


def valid_age(value: object) -> bool:
    return pd.notna(value) and 25 <= float(value) <= 65


def labor_observed(row: pd.Series) -> bool:
    return (
        pd.notna(row["annual_work_status"])
        and pd.notna(row["annual_hours"])
        and pd.notna(row["earn_tot_nd"])
        and int(row["annual_hours_valid"]) == 1
        and int(row["annual_earnings_valid"]) == 1
    )


def current_profile_payload(displaced: pd.Series, partner: pd.Series) -> dict[str, object]:
    return {
        "current_you_demo_age_gen": partner["demo_age_gen"],
        "current_you_demo_sex_label": partner["demo_sex_label"],
        "current_you_edu_year": partner["edu_year"],
        "current_partner_demo_age_gen": displaced["demo_age_gen"],
        "current_partner_demo_sex_label": displaced["demo_sex_label"],
        "current_partner_edu_year": displaced["edu_year"],
        "current_fam_size": partner["fam_size"],
        "current_fam_size_chi": partner["fam_size_chi"],
        "current_home_stat_label": partner["home_stat_label"],
        "current_geo_region_label": partner["geo_region_label"],
    }


def history_payload(
    histories: list[tuple[pd.Series, pd.Series]],
    prompt_year: int,
) -> dict[str, object]:
    out: dict[str, object] = {"history_depth": len(histories)}
    fields = ["annual_work_status_label", "annual_hours", "earn_tot_nd"]
    for lag in range(1, 4):
        out[f"history_lag{lag}_survey_year"] = np.nan
        out[f"history_lag{lag}_reference_year"] = np.nan
        out[f"history_lag{lag}_elapsed_years_to_event"] = np.nan
        for subject in ["displaced", "nondisplaced_partner"]:
            for field in fields:
                out[f"history_lag{lag}_{subject}_{field}"] = np.nan
        out[f"history_lag{lag}_finc_tot_nd"] = np.nan
    for lag, (displaced, partner) in enumerate(histories, start=1):
        out[f"history_lag{lag}_survey_year"] = int(displaced["survey_year"])
        out[f"history_lag{lag}_reference_year"] = int(displaced["labor_reference_year"])
        out[f"history_lag{lag}_elapsed_years_to_event"] = prompt_year - int(displaced["labor_reference_year"])
        for field in fields:
            out[f"history_lag{lag}_displaced_{field}"] = displaced.get(field, np.nan)
            out[f"history_lag{lag}_nondisplaced_partner_{field}"] = partner.get(field, np.nan)
        out[f"history_lag{lag}_finc_tot_nd"] = displaced.get("finc_tot_nd", np.nan)
    return out


def prior_episode_payload(
    episode: dict[str, object],
    episodes_by_person: dict[int, list[dict[str, object]]],
    prompt_year: int,
) -> dict[str, str]:
    previous = [
        row for row in episodes_by_person.get(int(episode["person_id"]), [])
        if int(row["assigned_event_year"]) < int(episode["assigned_event_year"])
    ]
    return {
        "prior_displacement_years_ago": "|".join(
            str(prompt_year - int(row["assigned_event_year"])) for row in previous
        ),
        "prior_displacement_reasons": "|".join(
            str(row["displacement_reason"]) for row in previous
        ),
    }


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
        raise RuntimeError(f"{TASK_ID}: missing Q4 macro context")
    frame = attach_macro_context(frame, origin_q_index_col="eligible_origin_q_index", required_columns=CORE_MACRO_COLUMNS)
    frame["macro_reference_year"] = frame["assigned_event_year"]
    frame["macro_reference_quarter"] = 4
    return frame


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    source = build_source_layers(args.panel_path)
    adults = source.adults.copy()
    households = source.households.copy()
    cutoffs = source.income_cutoffs.set_index("survey_year")

    # Steps 1--3 operate on the common one-row-per-household-wave PSID
    # intermediate. Attach event-year macro information and family-income
    # cutoffs, then construct the adult lead/lag lookups for every household
    # wave before selecting displacement shocks.
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
    household_index = attached_households.set_index(
        ["stable_household_subject_id", "survey_year"], drop=False, verify_integrity=True
    )
    adult_index = adults.set_index(["person_id", "survey_year"], drop=False, verify_integrity=True)
    support_lookup: dict[tuple[str, int], dict[str, object]] = {}
    for household in attached_households.to_dict("records"):
        subject = str(household["stable_household_subject_id"])
        survey_year = int(household["survey_year"])
        person_ids = [int(household["reference_person_id"])]
        if pd.notna(household["partner_person_id"]):
            person_ids.append(int(household["partner_person_id"]))
        observations = {
            (person_id, year): adult_index.loc[(person_id, year)]
            for person_id in person_ids
            for year in range(survey_year - 3, survey_year + 2)
            if (person_id, year) in adult_index.index
        }
        support_lookup[(subject, survey_year)] = {
            "person_ids": person_ids,
            "observations": observations,
            "required_pair_complete": len(person_ids) == 2 and all(
                (person_id, year) in observations
                for person_id in person_ids
                for year in [survey_year - 1, survey_year, survey_year + 1]
            ),
        }
    if len(support_lookup) != len(attached_households):
        raise RuntimeError(f"{TASK_ID}: Step 3 adult lead/lag support changed household-wave key uniqueness.")

    # Step 4 performs the common displacement selection and applies the task's
    # single-displaced, partnered, nonoverlapping event-window restrictions.
    displacement = build_displacement_layers(adults)
    episodes = displacement.episodes
    shocks = displacement.shocks
    episode_lookup = episodes.set_index("episode_id", drop=False, verify_integrity=True)
    shock_years = {
        str(subject): sorted(group["survey_year"].astype(int).tolist())
        for subject, group in shocks.groupby("stable_household_subject_id")
    }
    episodes_by_person = {
        int(person): group.sort_values(["assigned_event_year", "episode_id"]).to_dict("records")
        for person, group in episodes.groupby("person_id")
    }

    stage4 = []
    stage5 = []
    stage6 = []
    stage7 = []
    stage8 = []
    ledger = []
    for shock in shocks.to_dict("records"):
        candidate_id = str(shock["household_shock_id"])
        survey_year = int(shock["survey_year"])
        prompt_year = int(shock["assigned_event_year"])
        episode_ids = str(shock["episode_ids"]).split("|")
        if int(shock["number_displaced_adults"]) != 1 or len(episode_ids) != 1:
            ledger.append(
                {
                    "household_shock_id": candidate_id,
                    "episode_id": "|".join(episode_ids),
                    "status": "dual_adult_shock",
                }
            )
            continue
        episode = episode_lookup.loc[episode_ids[0]].to_dict()
        displaced_id = int(episode["person_id"])

        def ledger_row(status: str) -> dict[str, object]:
            return {
                "household_shock_id": candidate_id,
                "episode_id": str(episode["episode_id"]),
                "status": status,
            }

        if pd.isna(episode["partner_person_id"]):
            ledger.append(ledger_row("unpartnered_at_event"))
            continue
        partner_id = int(episode["partner_person_id"])
        subject = str(shock["stable_household_subject_id"])
        if any(other != survey_year and abs(other - survey_year) <= 1 for other in shock_years.get(subject, [])):
            ledger.append(ledger_row("overlapping_displacement_window"))
            continue
        required_keys = [
            (person, year)
            for person in [displaced_id, partner_id]
            for year in [survey_year - 1, survey_year, survey_year + 1]
        ]
        support = support_lookup.get((subject, survey_year))
        if (
            support is None
            or not bool(support["required_pair_complete"])
            or set(support["person_ids"]) != {displaced_id, partner_id}
            or not all(key in support["observations"] for key in required_keys)
        ):
            ledger.append(ledger_row("pair_not_observed_at_all_required_waves"))
            continue
        observations = support["observations"]
        context = {
            "episode": episode,
            "shock": shock,
            "subject": subject,
            "survey_year": survey_year,
            "prompt_year": prompt_year,
            "displaced_id": displaced_id,
            "partner_id": partner_id,
            "observations": observations,
            "candidate_id": candidate_id,
        }
        stage4.append(context)

        displaced_event = observations[(displaced_id, survey_year)]
        partner_event = observations[(partner_id, survey_year)]
        partner_follow = observations[(partner_id, survey_year + 1)]
        event_targets_valid = labor_observed(partner_event)
        following_targets_valid = labor_observed(partner_follow)
        if not (event_targets_valid and following_targets_valid):
            ledger.append(ledger_row("missing_event_or_following_target"))
            continue
        stage5.append(context)

        displaced_lag = observations[(displaced_id, survey_year - 1)]
        partner_lag = observations[(partner_id, survey_year - 1)]
        profile_required = ["demo_age_gen", "demo_sex_label", "fam_size", "fam_size_chi"]
        if displaced_event[profile_required].isna().any() or partner_event[profile_required].isna().any():
            ledger.append(ledger_row("missing_current_demographic_profile"))
            continue
        if not labor_observed(displaced_lag) or not labor_observed(partner_lag):
            ledger.append(ledger_row("missing_mandatory_lag1_labor_fields"))
            continue
        if pd.isna(displaced_lag["finc_tot_nd"]):
            ledger.append(ledger_row("missing_mandatory_lag1_family_income"))
            continue
        histories = [(displaced_lag, partner_lag)]
        for older_year in [survey_year - 2, survey_year - 3]:
            older_keys = [(displaced_id, older_year), (partner_id, older_year)]
            if not all(key in observations for key in older_keys):
                break
            displaced_old, partner_old = [observations[key] for key in older_keys]
            if (
                str(displaced_old["stable_household_subject_id"]) != subject
                or str(partner_old["stable_household_subject_id"]) != subject
                or not labor_observed(displaced_old)
                or not labor_observed(partner_old)
                or pd.isna(displaced_old["finc_tot_nd"])
            ):
                break
            histories.append((displaced_old, partner_old))
        context = {**context, "histories": histories}
        stage6.append(context)

        if any(
            str(observations[(person, year)]["stable_household_subject_id"]) != subject
            for person in [displaced_id, partner_id]
            for year in [survey_year - 1, survey_year, survey_year + 1]
        ):
            ledger.append(ledger_row("pair_identity_not_stable"))
            continue
        if any(
            not valid_age(observations[(person, year)]["demo_age_gen"])
            for person in [displaced_id, partner_id]
            for year in [survey_year - 1, survey_year, survey_year + 1]
        ):
            ledger.append(ledger_row("age_outside_25_65"))
            continue
        if int(displaced_event["displacement_ascertained"]) != 1 or int(partner_event["displacement_ascertained"]) != 1:
            ledger.append(ledger_row("incomplete_both_adult_ascertainment"))
            continue
        stage7.append(context)

        if any(
            not income_inside_cutoffs(displaced["finc_tot_nd"], int(displaced["survey_year"]), cutoffs)
            for displaced, _ in histories
        ):
            ledger.append(ledger_row("displayed_family_income_outside_strict_p01_p99"))
            continue
        stage8.append(context)
        ledger.append(ledger_row("exported"))

    ledger_frame = pd.DataFrame(ledger)
    ledger_frame.to_csv(OUTPUT_DIR / "01_event_candidate_ledger.csv", index=False)
    rows = []
    for context in stage8:
        episode = context["episode"]
        shock = context["shock"]
        observations = context["observations"]
        survey_year = context["survey_year"]
        prompt_year = context["prompt_year"]
        displaced_id = context["displaced_id"]
        partner_id = context["partner_id"]
        displaced_event = observations[(displaced_id, survey_year)]
        partner_event = observations[(partner_id, survey_year)]
        partner_follow = observations[(partner_id, survey_year + 1)]
        displaced_lag = observations[(displaced_id, survey_year - 1)]
        partner_lag = observations[(partner_id, survey_year - 1)]
        row = {
            **episode,
            **current_profile_payload(displaced_event, partner_event),
            **history_payload(context["histories"], prompt_year),
            **prior_episode_payload(episode, episodes_by_person, prompt_year),
            "household_shock_id": shock["household_shock_id"],
            "task_unit": "household_displacement_shock",
            "displaced_person_id": displaced_id,
            "nondisplaced_partner_id": partner_id,
            "focal_person_id": partner_id,
            "reference_person_id": int(household_index.loc[(context["subject"], survey_year), "reference_person_id"]),
            "focal_person_role_at_event": partner_event["role"],
            "focal_person_sex": partner_event["demo_sex_label"],
            "displaced_person_role_at_event": displaced_event["role"],
            "displaced_person_sex": displaced_event["demo_sex_label"],
            "nondisplaced_partner_role_at_event": partner_event["role"],
            "nondisplaced_partner_sex": partner_event["demo_sex_label"],
            "event_year_target_survey_year": survey_year,
            "event_year_target_calendar_year": prompt_year,
            "following_year_target_survey_year": survey_year + 1,
            "following_year_target_calendar_year": prompt_year + 1,
            "partner_annual_hours_event_year": float(partner_event["annual_hours"]),
            "partner_annual_earnings_event_year": float(partner_event["earn_tot_nd"]),
            "partner_annual_hours_following_year": float(partner_follow["annual_hours"]),
            "partner_annual_earnings_following_year": float(partner_follow["earn_tot_nd"]),
            "family_weight": displaced_lag["fw"],
            "release_date": max(str(partner_event["release_date"])[:10], str(partner_follow["release_date"])[:10]),
            "displaced_is_female": int(str(displaced_event["demo_sex_label"]).lower() == "female"),
            "displaced_is_spouse_partner": int(displaced_event["role"] == "spouse_partner"),
            "reason_is_layoff_firing": int(episode["displacement_reason"] == "layoff_or_firing"),
            "is_confirmed_later_episode": int(episode["repeat_classification"] == "confirmed_distinct_later_episode"),
            "baseline_partner_worked": float(partner_lag["annual_work_status"]),
            "baseline_partner_hours": float(partner_lag["annual_hours"]),
            "baseline_partner_earnings": float(partner_lag["earn_tot_nd"]),
            "baseline_displaced_hours": float(displaced_lag["annual_hours"]),
            "baseline_displaced_earnings": float(displaced_lag["earn_tot_nd"]),
            **{column: household_index.loc[(context["subject"], survey_year), column] for column in CORE_MACRO_COLUMNS},
            "eligible_origin_q_index": household_index.loc[(context["subject"], survey_year), "eligible_origin_q_index"],
            "macro_reference_year": household_index.loc[(context["subject"], survey_year), "macro_reference_year"],
            "macro_reference_quarter": household_index.loc[(context["subject"], survey_year), "macro_reference_quarter"],
        }
        rows.append(row)

    pool = pd.DataFrame(rows)
    if pool.empty:
        raise RuntimeError(f"{TASK_ID}: no rows remain after the standard construction")
    pool["naive_partner_annual_hours_event_year"] = pool["baseline_partner_hours"]
    pool["naive_partner_annual_earnings_event_year"] = pool["baseline_partner_earnings"]
    pool["naive_partner_annual_hours_following_year"] = pool["baseline_partner_hours"]
    pool["naive_partner_annual_earnings_following_year"] = pool["baseline_partner_earnings"]
    pool["row_id"] = pool["household_shock_id"]
    pool["subject_id"] = pool["stable_household_subject_id"]
    pool["group_id"] = pool["split_group_id"]
    pool["selection_period"] = pool["assigned_event_year"].astype(int).astype(str)
    pool["time"] = pool["assigned_event_year"].astype(int).astype(str) + "-12-31"
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
    write_public_table(selected, task_id=TASK_ID, public_path=TABLE_PATH)
    final_invariants = pd.DataFrame({
        "id": selected["id"].astype("string"),
        "time": selected["time"].astype("string"),
        "exactly_one_displacement_episode": selected["episode_id"].notna().astype(int),
        "displaced_and_nondisplaced_adults_differ": selected["displaced_person_id"].ne(selected["nondisplaced_partner_id"]).astype(int),
        "focal_adult_is_nondisplaced_partner": selected["focal_person_id"].eq(selected["nondisplaced_partner_id"]).astype(int),
        "event_target_uses_event_wave": selected["event_year_target_survey_year"].eq(selected["survey_year"]).astype(int),
        "following_target_uses_next_wave": selected["following_year_target_survey_year"].eq(selected["survey_year"] + 1).astype(int),
        "mandatory_history_is_one_year_before_event": selected["history_lag1_elapsed_years_to_event"].eq(1).astype(int),
        "eligible_displacement_reason": selected["displacement_reason"].isin(["layoff_or_firing", "plant_closure_or_employer_move"]).astype(int),
        "four_targets_nonnegative": selected[[
            "partner_annual_hours_event_year", "partner_annual_earnings_event_year",
            "partner_annual_hours_following_year", "partner_annual_earnings_following_year",
        ]].ge(0).all(axis=1).astype(int),
        "macro_context_complete": selected[CORE_MACRO_COLUMNS].notna().all(axis=1).astype(int),
    })
    if not final_invariants.drop(columns=["id", "time"]).eq(1).all().all():
        raise RuntimeError(f"{TASK_ID}: a selected row fails the final household-event validation checks.")
    final_invariants.to_csv(OUTPUT_DIR / "01_final_event_invariants.csv", index=False)

    stage_counts = [
        len(households), len(attached_households), len(support_lookup), len(stage4), len(stage5),
        len(stage6), len(stage7), len(stage8), len(selected),
    ]
    step_rows = []
    for index, (label, after) in enumerate(zip(STANDARD_STEPS, stage_counts, strict=True), start=1):
        before = stage_counts[index - 2] if index > 1 else stage_counts[0]
        step_rows.append(construction_step(
                             TASK_ID, index, label,
                             before, after, before - after,
                         ))
    step_frame = pd.DataFrame(step_rows)
    step_frame.to_csv(OUTPUT_DIR / "01_sample_construction_steps.csv", index=False)
    status_counts = ledger_frame["status"].value_counts()
    detail_specs = [
        (1, "common_psid_household_waves", "start from every unique household-wave record in the validated common PSID intermediate", stage_counts[0] - stage_counts[0]),
        (2, "macro_release_and_income_cutoffs_attached", "attach the event-year macro block, PSID release metadata, and survey-year family-income cutoffs to every household wave", stage_counts[0] - stage_counts[1]),
        (3, "adult_leads_and_histories_constructed", "construct each household's adult observations for the event wave, following wave, and up to three earlier waves before displacement selection", stage_counts[1] - stage_counts[2]),
        (4, "single_displaced_partnered_household_event", "retain a nonoverlapping household event with exactly one displaced adult, an identified nondisplaced partner, and both adults observed before, during, and after the event", stage_counts[2] - stage_counts[3]),
        (5, "four_partner_labor_targets_observed", "require the nondisplaced partner's annual hours and labor earnings in both the event wave and the following wave", int(status_counts.get("missing_event_or_following_target", 0))),
        (6, "current_demographic_profile_observed", "require age, sex, and family-composition fields for both adults in the event wave", int(status_counts.get("missing_current_demographic_profile", 0))),
        (6, "pre_event_labor_history_observed", "require valid annual work status, hours, and earnings for both adults in the wave before the event", int(status_counts.get("missing_mandatory_lag1_labor_fields", 0))),
        (6, "pre_event_family_income_observed", "require total family income in the wave before the event", int(status_counts.get("missing_mandatory_lag1_family_income", 0))),
        (7, "same_pair_across_required_waves", "require the same two adults to remain in the same household across the pre-event, event, and following waves", int(status_counts.get("pair_identity_not_stable", 0))),
        (7, "adult_ages_25_to_65", "require both adults to be age 25 through 65 in every required wave", int(status_counts.get("age_outside_25_65", 0))),
        (7, "both_adults_job_loss_questions_ascertained", "require valid job-ending information for both adults in the event wave", int(status_counts.get("incomplete_both_adult_ascertainment", 0))),
        (8, "displayed_family_income_inside_cutoffs", "require every displayed pre-event family-income value to lie strictly inside its survey-year 1st and 99th percentiles", int(status_counts.get("displayed_family_income_outside_strict_p01_p99", 0))),
        (9, "deterministic_export_cap", "apply the registered deterministic sampling order and export all eligible events because the sample is below the task cap", stage_counts[7] - stage_counts[8]),
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
    source.income_cutoffs.to_csv(OUTPUT_DIR / "01_outlier_cutoffs.csv", index=False)
    print(f"{TASK_ID}: wrote {len(selected):,} four-target rows")


if __name__ == "__main__":
    main()
