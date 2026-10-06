"""Task specifications: targets, answer formats and metrics for the 32 HouseholdBench tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FieldKind = Literal["numeric", "categorical"]
OutputShape = Literal["array", "distribution_array"]
TargetType = Literal["numeric", "categorical", "distribution"]


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: FieldKind
    labels: tuple[str | int, ...] = ()


@dataclass(frozen=True)
class DistributionSpec:
    name: str
    labels: tuple[str, ...]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    mode: Literal["baseline", "policy"]
    output_shape: OutputShape
    target_type: TargetType
    fields: tuple[FieldSpec, ...]
    distributions: tuple[DistributionSpec, ...]
    summary: str

    @property
    def numeric_fields(self) -> tuple[FieldSpec, ...]:
        return tuple(field for field in self.fields if field.kind == "numeric")

    @property
    def categorical_fields(self) -> tuple[FieldSpec, ...]:
        return tuple(field for field in self.fields if field.kind == "categorical")

    @property
    def official_metrics(self) -> tuple[str, ...]:
        if self.target_type == "numeric":
            return ("relmae",)
        if self.target_type == "categorical":
            return ("macro_f1", "accuracy")
        if self.target_type == "distribution":
            return ("total_variation",)
        raise ValueError(f"unsupported_target_type:{self.target_type}")

    @property
    def diagnostic_metrics(self) -> tuple[str, ...]:
        return ("nmae",) if self.target_type == "numeric" else ()


IMPACT_LABELS = (
    "very_negative",
    "somewhat_negative",
    "no_impact",
    "somewhat_positive",
    "very_positive",
)

LABOR_FORCE_LABELS = ("employed", "unemployed", "not_in_labor_force")


TASK_SPECS: dict[str, TaskSpec] = {
    "cons_cex_categories": TaskSpec(
        name="cons_cex_categories",
        mode="baseline",
        output_shape="array",
        target_type="numeric",
        fields=tuple(
            FieldSpec(name, "numeric")
            for name in (
                "food",
                "alcoholic_beverages",
                "housing",
                "apparel_and_services",
                "transportation",
                "health_care",
                "entertainment",
                "personal_care",
                "reading",
                "education",
                "tobacco",
                "miscellaneous",
            )
        ),
        distributions=(),
        summary="CEX current-quarter expenditure across twelve detailed categories.",
    ),
    "cons_cex_rebate01": TaskSpec(
        name="cons_cex_rebate01",
        mode="policy",
        output_shape="array",
        target_type="numeric",
        fields=(
            FieldSpec("total_expenditure", "numeric"),
            FieldSpec("non_durable_expenditure", "numeric"),
            FieldSpec("durable_expenditure", "numeric"),
        ),
        distributions=(),
        summary="CEX expenditure after the 2001 federal tax rebate.",
    ),
    "cons_cex_total": TaskSpec(
        name="cons_cex_total",
        mode="baseline",
        output_shape="array",
        target_type="numeric",
        fields=(
            FieldSpec("total_expenditure", "numeric"),
            FieldSpec("non_durable_expenditure", "numeric"),
            FieldSpec("durable_expenditure", "numeric"),
        ),
        distributions=(),
        summary="CEX current-quarter total, non-durable, and durable-related expenditure.",
    ),
    "cons_cex_stimulus08": TaskSpec(
        name="cons_cex_stimulus08",
        mode="policy",
        output_shape="array",
        target_type="numeric",
        fields=(
            FieldSpec("total_expenditure", "numeric"),
            FieldSpec("non_durable_expenditure", "numeric"),
            FieldSpec("durable_expenditure", "numeric"),
        ),
        distributions=(),
        summary="CEX expenditure after the 2008 stimulus-payment policy.",
    ),
    "cons_psid_wealth": TaskSpec(
        name="cons_psid_wealth",
        mode="baseline",
        output_shape="array",
        target_type="numeric",
        fields=(FieldSpec("next_wave_net_worth", "numeric"),),
        distributions=(),
        summary="PSID next-wave household net worth.",
    ),
    "cons_sce_growth": TaskSpec(
        name="cons_sce_growth",
        mode="baseline",
        output_shape="array",
        target_type="numeric",
        fields=(FieldSpec("monthly_spending_growth_12m", "numeric"),),
        distributions=(),
        summary="SCE realized household spending growth over twelve months.",
    ),
    "cons_sce_shock": TaskSpec(
        name="cons_sce_shock",
        mode="policy",
        output_shape="array",
        target_type="numeric",
        fields=(
            FieldSpec("spend_or_donate_income_gain", "numeric"),
            FieldSpec("reduce_spending_income_loss", "numeric"),
        ),
        distributions=(),
        summary="SCE hypothetical spending responses to household-income shocks.",
    ),
    "house_census_move": TaskSpec(
        name="house_census_move",
        mode="baseline",
        output_shape="array",
        target_type="categorical",
        fields=(
            FieldSpec(
                "mobility_destination",
                "categorical",
                ("stay", "move_within_state", "move_to_different_state"),
            ),
        ),
        distributions=(),
        summary="Census five-year mobility destination.",
    ),
    "house_sce_move": TaskSpec(
        name="house_sce_move",
        mode="baseline",
        output_shape="distribution_array",
        target_type="distribution",
        fields=(),
        distributions=(DistributionSpec("move_probability_12m", ("no_move", "move")),),
        summary="SCE twelve-month moving distribution.",
    ),
    "income_cps_displace": TaskSpec(
        name="income_cps_displace",
        mode="policy",
        output_shape="array",
        target_type="numeric",
        fields=(FieldSpec("current_weekly_earnings", "numeric"),),
        distributions=(),
        summary="CPS current weekly earnings after displacement.",
    ),
    "income_mich_finance": TaskSpec(
        name="income_mich_finance",
        mode="baseline",
        output_shape="array",
        target_type="numeric",
        fields=(FieldSpec("expected_family_income_change_1y", "numeric"),),
        distributions=(),
        summary="Michigan survey expected family-income change.",
    ),
    "income_psid_earnings": TaskSpec(
        name="income_psid_earnings",
        mode="baseline",
        output_shape="array",
        target_type="numeric",
        fields=(
            FieldSpec("earn_2y", "numeric"),
            FieldSpec("earn_4y", "numeric"),
            FieldSpec("earn_10y", "numeric"),
        ),
        distributions=(),
        summary="PSID nominal labor earnings two, four, and ten years ahead.",
    ),
    "income_sce_growth": TaskSpec(
        name="income_sce_growth",
        mode="baseline",
        output_shape="array",
        target_type="numeric",
        fields=(FieldSpec("expected_household_income_change_1y", "numeric"),),
        distributions=(),
        summary="SCE expected household-income growth over twelve months.",
    ),
    "income_sce_policy": TaskSpec(
        name="income_sce_policy",
        mode="policy",
        output_shape="array",
        target_type="categorical",
        fields=(
            FieldSpec("welfare_benefits", "categorical", IMPACT_LABELS),
            FieldSpec("unemployment_benefits", "categorical", IMPACT_LABELS),
            FieldSpec("payroll_tax_rate", "categorical", IMPACT_LABELS),
            FieldSpec("average_income_tax_rate", "categorical", IMPACT_LABELS),
        ),
        distributions=(),
        summary="SCE household impact labels for four policy changes.",
    ),
    "labor_cps_jobfind": TaskSpec(
        name="labor_cps_jobfind",
        mode="baseline",
        output_shape="array",
        target_type="categorical",
        fields=(FieldSpec("labor_force_status", "categorical", LABOR_FORCE_LABELS),),
        distributions=(),
        summary="CPS one-month-ahead labor-force status.",
    ),
    "labor_cps_displace": TaskSpec(
        name="labor_cps_displace",
        mode="policy",
        output_shape="array",
        target_type="categorical",
        fields=(FieldSpec("labor_force_status", "categorical", LABOR_FORCE_LABELS),),
        distributions=(),
        summary="CPS current labor-force status after displacement.",
    ),
    "labor_cps_retire": TaskSpec(
        name="labor_cps_retire",
        mode="baseline",
        output_shape="array",
        target_type="categorical",
        fields=(
            FieldSpec(
                "retirement_status",
                "categorical",
                ("retired", "not_retired"),
            ),
        ),
        distributions=(),
        summary="CPS older-worker retirement status twelve months ahead.",
    ),
    "labor_cps_separation": TaskSpec(
        name="labor_cps_separation",
        mode="baseline",
        output_shape="array",
        target_type="categorical",
        fields=(FieldSpec("labor_force_status", "categorical", LABOR_FORCE_LABELS),),
        distributions=(),
        summary="CPS one-month-ahead labor-force status for employed workers.",
    ),
    "labor_sce_offer": TaskSpec(
        name="labor_sce_offer",
        mode="baseline",
        output_shape="array",
        target_type="categorical",
        fields=(
            FieldSpec(
                "first_offer_decision",
                "categorical",
                ("accepted", "rejected"),
            ),
        ),
        distributions=(),
        summary="SCE acceptance of the first recent job offer.",
    ),
    "labor_sce_reswage": TaskSpec(
        name="labor_sce_reswage",
        mode="policy",
        output_shape="array",
        target_type="numeric",
        fields=(
            FieldSpec("hourly_reservation_wage", "numeric"),
            FieldSpec("preferred_weekly_hours", "numeric"),
        ),
        distributions=(),
        summary="SCE reservation wage and preferred weekly work hours.",
    ),
    "labor_sce_risk": TaskSpec(
        name="labor_sce_risk",
        mode="baseline",
        output_shape="distribution_array",
        target_type="distribution",
        fields=(),
        distributions=(
            DistributionSpec("job_loss_12m", ("no_event", "event")),
            DistributionSpec("voluntary_leave_12m", ("no_event", "event")),
            DistributionSpec("find_acceptable_work_3m", ("no_event", "event")),
        ),
        summary="SCE employed-worker labor-risk belief distributions.",
    ),
    "labor_sce_search": TaskSpec(
        name="labor_sce_search",
        mode="baseline",
        output_shape="distribution_array",
        target_type="distribution",
        fields=(),
        distributions=(
            DistributionSpec("find_acceptable_work_12m", ("not_found", "found")),
            DistributionSpec("find_acceptable_work_3m", ("not_found", "found")),
        ),
        summary="SCE active job-seeker job-finding belief distributions.",
    ),
    "macro_mich_outlook": TaskSpec(
        name="macro_mich_outlook",
        mode="baseline",
        output_shape="array",
        target_type="numeric",
        fields=(
            FieldSpec("expected_inflation_1y", "numeric"),
            FieldSpec("expected_inflation_5_to_10y_annual", "numeric"),
        ),
        distributions=(),
        summary="Michigan survey one-year and long-run inflation expectations.",
    ),
    "macro_sce_revision": TaskSpec(
        name="macro_sce_revision",
        mode="policy",
        output_shape="array",
        target_type="numeric",
        fields=(
            FieldSpec("expected_inflation_1y", "numeric"),
            FieldSpec("expected_inflation_2_to_3y", "numeric"),
        ),
        distributions=(),
        summary="SCE revised one-year and two-to-three-year inflation expectations.",
    ),
    "house_psid_owner": TaskSpec(
        name="house_psid_owner",
        mode="baseline",
        output_shape="array",
        target_type="categorical",
        fields=(FieldSpec("owns_home_next_wave", "categorical", ("no", "yes")),),
        distributions=(),
        summary="PSID renter-to-owner transition by the next observed wave.",
    ),
    "house_sce_lockin": TaskSpec(
        name="house_sce_lockin",
        mode="policy",
        output_shape="distribution_array",
        target_type="distribution",
        fields=(),
        distributions=(DistributionSpec("move_probability_3y", ("no_move", "move")),),
        summary="SCE moving distribution under a mortgage-rate portability counterfactual.",
    ),
    "macro_sce_uncertainty": TaskSpec(
        name="macro_sce_uncertainty",
        mode="baseline",
        output_shape="distribution_array",
        target_type="distribution",
        fields=(),
        distributions=(
            DistributionSpec(
                "inflation_1y",
                (
                    "inflation_ge_12",
                    "inflation_8_12",
                    "inflation_4_8",
                    "inflation_2_4",
                    "inflation_0_2",
                    "deflation_0_2",
                    "deflation_2_4",
                    "deflation_4_8",
                    "deflation_8_12",
                    "deflation_ge_12",
                ),
            ),
            DistributionSpec(
                "inflation_2_to_3y",
                (
                    "inflation_ge_12",
                    "inflation_8_12",
                    "inflation_4_8",
                    "inflation_2_4",
                    "inflation_0_2",
                    "deflation_0_2",
                    "deflation_2_4",
                    "deflation_4_8",
                    "deflation_8_12",
                    "deflation_ge_12",
                ),
            ),
            DistributionSpec("unemployment_higher", ("not_higher", "higher")),
            DistributionSpec("savings_rate_higher", ("not_higher", "higher")),
            DistributionSpec("stock_prices_higher", ("not_higher", "higher")),
        ),
        summary="SCE macro-risk and uncertainty belief distributions.",
    ),
    "cons_cex_sspay": TaskSpec(
        name="cons_cex_sspay",
        mode="policy",
        output_shape="array",
        target_type="numeric",
        fields=(
            FieldSpec("total_expenditure", "numeric"),
            FieldSpec("food_at_home", "numeric"),
            FieldSpec("food_away_from_home", "numeric"),
        ),
        distributions=(),
        summary="CEX daily spending around scheduled Social Security payments.",
    ),
    "cons_psid_jobloss": TaskSpec(
        name="cons_psid_jobloss",
        mode="policy",
        output_shape="array",
        target_type="numeric",
        fields=(FieldSpec("annual_food_spending", "numeric"),),
        distributions=(),
        summary="PSID annual food spending after current-year job loss.",
    ),
    "house_sce_financing": TaskSpec(
        name="house_sce_financing",
        mode="policy",
        output_shape="array",
        target_type="numeric",
        fields=(
            FieldSpec("situation_1_maximum_price", "numeric"),
            FieldSpec("situation_1_down_payment", "numeric"),
            FieldSpec("situation_2_maximum_price", "numeric"),
            FieldSpec("situation_2_down_payment", "numeric"),
            FieldSpec("situation_3_maximum_price", "numeric"),
            FieldSpec("situation_3_down_payment", "numeric"),
        ),
        distributions=(),
        summary="SCE home-purchase price and down payment under financing scenarios.",
    ),
    "labor_cps_ui": TaskSpec(
        name="labor_cps_ui",
        mode="policy",
        output_shape="array",
        target_type="categorical",
        fields=(FieldSpec("work_status", "categorical", LABOR_FORCE_LABELS),),
        distributions=(),
        summary="CPS next-month work status under unemployment-insurance duration.",
    ),
    "labor_psid_addedworker": TaskSpec(
        name="labor_psid_addedworker",
        mode="policy",
        output_shape="array",
        target_type="numeric",
        fields=(
            FieldSpec("current_year_work_hours", "numeric"),
            FieldSpec("current_year_labor_earnings", "numeric"),
            FieldSpec("next_year_work_hours", "numeric"),
            FieldSpec("next_year_labor_earnings", "numeric"),
        ),
        distributions=(),
        summary="PSID partner annual hours and earnings after a spouse's job loss.",
    ),
}

DISTRIBUTION_TASKS = tuple(
    sorted(
        task_name
        for task_name, spec in TASK_SPECS.items()
        if spec.target_type == "distribution"
    )
)
ALL_TASKS = tuple(sorted(TASK_SPECS))


def get_task_spec(task_name: str) -> TaskSpec:
    try:
        return TASK_SPECS[task_name]
    except KeyError as exc:
        available = ", ".join(ALL_TASKS)
        raise KeyError(f"Unknown HouseholdBench task {task_name!r}. Available: {available}") from exc
