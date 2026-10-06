# Tasks

Targets, modes and metrics for all 32 HouseholdBench tasks, as built by `scripts/` and scored
by `household_bench_eval`. `household-bench-eval describe --task <task>` prints a task's full
answer format, including target names and label sets.

Prompt rows contain `id`, `time`, `release_date`, `system`, `user`, `assistant`, and
`naive_baseline` (the no-change prediction). Answers and baselines are JSON-formatted strings
using native arrays. Rows are unique on `(id, time)`; when a task's gold file repeats an `id`,
the scorer matches predictions on `id` and `time` together.

## Task summary

| Task | Mode | Targets | Scoring |
|---|---|---|---|
| `cons_cex_categories` | Baseline | 12 numeric expenditure categories | Mean RelMAE. nMAE diagnostic. |
| `cons_cex_rebate01` | Policy | 3 numeric expenditure targets | Mean RelMAE. nMAE diagnostic. |
| `cons_cex_sspay` | Policy | 3 numeric daily-spending targets | Mean RelMAE. nMAE diagnostic. |
| `cons_cex_stimulus08` | Policy | 3 numeric expenditure targets | Mean RelMAE. nMAE diagnostic. |
| `cons_cex_total` | Baseline | 3 numeric expenditure targets | Mean RelMAE. nMAE diagnostic. |
| `cons_psid_jobloss` | Policy | 1 numeric food-spending target | RelMAE. nMAE diagnostic. |
| `cons_psid_wealth` | Baseline | 1 numeric net-worth target | RelMAE. nMAE diagnostic. |
| `cons_sce_growth` | Baseline | 1 numeric spending-growth target | RelMAE. nMAE diagnostic. |
| `cons_sce_shock` | Policy | 2 numeric spending-response targets | Mean RelMAE. nMAE diagnostic. |
| `house_census_move` | Baseline | 1 categorical mobility destination | Macro-F1 primary and accuracy secondary. |
| `house_psid_owner` | Baseline | 1 categorical ownership outcome | Macro-F1 primary and accuracy secondary. |
| `house_sce_financing` | Policy | 6 numeric price and down-payment targets | Mean RelMAE. nMAE diagnostic. |
| `house_sce_lockin` | Policy | 1 Bernoulli distribution ordered `[no_move, move]` | Mean Total Variation distance. |
| `house_sce_move` | Baseline | 1 Bernoulli moving distribution | Mean Total Variation distance. |
| `income_cps_displace` | Policy | 1 numeric weekly-earnings target | RelMAE. nMAE diagnostic. |
| `income_mich_finance` | Baseline | 1 numeric income-change target | RelMAE. nMAE diagnostic. |
| `income_psid_earnings` | Baseline | 3 numeric earnings horizons | Mean RelMAE. nMAE diagnostic. |
| `income_sce_growth` | Baseline | 1 numeric income-growth target | RelMAE. nMAE diagnostic. |
| `income_sce_policy` | Policy | 4 categorical policy impacts | Mean macro-F1 primary and mean accuracy secondary. |
| `labor_cps_displace` | Policy | 1 categorical labor-force status | Macro-F1 primary and accuracy secondary. |
| `labor_cps_jobfind` | Baseline | 1 categorical labor-force status | Macro-F1 primary and accuracy secondary. |
| `labor_cps_retire` | Baseline | 1 categorical retirement status | Macro-F1 primary and accuracy secondary. |
| `labor_cps_separation` | Baseline | 1 categorical labor-force status | Macro-F1 primary and accuracy secondary. |
| `labor_cps_ui` | Policy | 1 categorical labor-force status | Macro-F1 primary and accuracy secondary. |
| `labor_psid_addedworker` | Policy | 4 numeric hours and earnings targets | Mean RelMAE. nMAE diagnostic. |
| `labor_sce_offer` | Baseline | 1 categorical offer decision | Macro-F1 primary and accuracy secondary. |
| `labor_sce_reswage` | Policy | 2 numeric labor-supply targets | Mean RelMAE. nMAE diagnostic. |
| `labor_sce_risk` | Baseline | 3 Bernoulli labor-risk distributions | Mean Total Variation distance. |
| `labor_sce_search` | Baseline | 2 Bernoulli job-finding distributions | Mean Total Variation distance. |
| `macro_mich_outlook` | Baseline | 2 numeric inflation horizons | Mean RelMAE. nMAE diagnostic. |
| `macro_sce_revision` | Policy | 2 numeric inflation horizons | Mean RelMAE. nMAE diagnostic. |
| `macro_sce_uncertainty` | Baseline | 5 macro-risk distributions | Mean Total Variation distance. |

## Scoring Rules

- Numeric RelMAE is computed per target as the model absolute-error sum divided by the row-level
  no-change baseline absolute-error sum. Task RelMAE is the unweighted target mean.
- Numeric nMAE is MAE divided by the target population standard deviation. It is reference-only and
  is not the primary score.
- Macro-F1 includes every allowed class, including classes absent from the scored subset. Accuracy
  is reported separately. Multi-target task scores use unweighted arithmetic means.
- Total Variation is half the L1 distance between predicted and reference probability vectors. It
  is averaged over examples and then distributions.
- Missing or malformed numeric baselines and zero RelMAE denominators are scoring errors.
- Missing or invalid model predictions are excluded from metrics and recorded by ID and reason.
  Coverage rate and valid-response rate are always reported.
- Metric choice is fixed by target type. Unknown configuration fields and attempted metric
  overrides are rejected.
