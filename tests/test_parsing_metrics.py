import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from household_bench_eval.config_lib import load_config_path
from household_bench_eval.experiment import run_experiment
from household_bench_eval.metrics import compute_task_metrics, macro_f1, total_variation
from household_bench_eval.parsing import parse_response
from household_bench_eval.scorer import score_task
from household_bench_eval.tasks import (
    ALL_TASKS,
    DISTRIBUTION_TASKS,
    get_task_spec,
)


class TaskSpecTests(unittest.TestCase):
    def test_all_32_tasks_are_enabled(self):
        self.assertEqual(len(ALL_TASKS), 32)
        self.assertIn("cons_cex_categories", ALL_TASKS)
        self.assertIn("macro_sce_uncertainty", ALL_TASKS)

    def test_selected_policy_task_specs(self):
        tasks = (
            "cons_cex_sspay",
            "cons_psid_jobloss",
            "house_sce_financing",
            "labor_cps_ui",
            "labor_psid_addedworker",
        )
        for task in tasks:
            with self.subTest(task=task):
                spec = get_task_spec(task)
                self.assertIn(task, ALL_TASKS)
                self.assertEqual(spec.mode, "policy")
                self.assertEqual(spec.output_shape, "array")
                self.assertNotIn(task, DISTRIBUTION_TASKS)
        self.assertEqual(get_task_spec("labor_cps_ui").target_type, "categorical")
        for task in set(tasks) - {"labor_cps_ui"}:
            self.assertEqual(get_task_spec(task).target_type, "numeric")

    def test_all_distribution_tasks_are_enabled(self):
        self.assertEqual(len(DISTRIBUTION_TASKS), 5)
        self.assertEqual(
            DISTRIBUTION_TASKS,
            (
                "house_sce_lockin",
                "house_sce_move",
                "labor_sce_risk",
                "labor_sce_search",
                "macro_sce_uncertainty",
            ),
        )

    def test_selected_task_specs(self):
        stimulus = get_task_spec("cons_cex_stimulus08")
        self.assertEqual(stimulus.target_type, "numeric")
        self.assertEqual(len(stimulus.fields), 3)
        self.assertEqual(stimulus.official_metrics, ("relmae",))
        self.assertEqual(stimulus.diagnostic_metrics, ("nmae",))

        displaced = get_task_spec("labor_cps_displace")
        self.assertEqual(
            displaced.fields[0].labels,
            ("employed", "unemployed", "not_in_labor_force"),
        )

        lockin = get_task_spec("house_sce_lockin")
        self.assertEqual(lockin.target_type, "distribution")
        self.assertEqual(lockin.official_metrics, ("total_variation",))


class ParsingTests(unittest.TestCase):
    def test_numeric_array_task(self):
        result = parse_response("[1662, 1067, 595]", get_task_spec("cons_cex_total"))
        self.assertTrue(result.ok)
        self.assertEqual(result.value["total_expenditure"], 1662.0)

    def test_numeric_task_rejects_keyed_object(self):
        result = parse_response(
            (
                '{"total_expenditure": 1662, "non_durable_expenditure": 1067, '
                '"durable_expenditure": 595}'
            ),
            get_task_spec("cons_cex_total"),
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "expected_json_array")

    def test_categorical_array_is_case_sensitive(self):
        valid = parse_response(
            '["not_in_labor_force"]',
            get_task_spec("labor_cps_jobfind"),
        )
        self.assertTrue(valid.ok)

        invalid = parse_response(
            '["Not_in_labor_force"]',
            get_task_spec("labor_cps_jobfind"),
        )
        self.assertFalse(invalid.ok)

    def test_distribution_array(self):
        result = parse_response("[[0.75, 0.25]]", get_task_spec("house_sce_lockin"))
        self.assertTrue(result.ok)
        self.assertEqual(result.value["move_probability_3y"], (0.75, 0.25))

    def test_distribution_rejects_bad_nesting_bounds_and_sum(self):
        spec = get_task_spec("house_sce_lockin")
        self.assertFalse(parse_response("[0.75, 0.25]", spec).ok)
        self.assertFalse(parse_response("[[-0.1, 1.1]]", spec).ok)
        self.assertFalse(parse_response("[[0.75, 0.20]]", spec).ok)

    def test_distribution_sum_tolerance(self):
        result = parse_response(
            "[[0.7000004, 0.3]]",
            get_task_spec("house_sce_lockin"),
        )
        self.assertTrue(result.ok)

    def test_think_wrapper_keeps_json_array(self):
        result = parse_response(
            "<think>reasoning</think>\n[10.0, 13.0]",
            get_task_spec("macro_mich_outlook"),
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.value["expected_inflation_1y"], 10.0)

    def test_think_wrapper_keeps_outer_distribution_array(self):
        result = parse_response(
            "<think>\n\n</think>\n\n[[0.75, 0.25]]",
            get_task_spec("house_sce_lockin"),
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.value["move_probability_3y"], (0.75, 0.25))

    def test_think_wrapper_without_separator_keeps_outer_distribution_array(self):
        result = parse_response(
            "<think>\n\n</think>[[0.75, 0.25]]",
            get_task_spec("house_sce_lockin"),
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.value["move_probability_3y"], (0.75, 0.25))


class NumberFormatRepairTests(unittest.TestCase):
    """Thousands separators and currency symbols observed in DeepSeek-V4 responses.

    JSON reads "6,620" as two elements, so the strict parser rejects the row with an
    arity error indistinguishable from a model that gave the wrong number of values.
    """

    def test_thousands_separators_are_rejected_without_the_flag(self):
        spec = get_task_spec("cons_cex_total")
        result = parse_response("[6,620, 3,800, 2,820]", spec)
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "expected_3_array_elements")

    def test_thousands_separators_are_repaired_with_the_flag(self):
        spec = get_task_spec("cons_cex_total")
        result = parse_response(
            "[6,620, 3,800, 2,820]",
            spec,
            repair_number_formatting=True,
        )
        self.assertTrue(result.ok)
        self.assertTrue(result.repaired)
        self.assertEqual(result.value["total_expenditure"], 6620.0)
        self.assertEqual(result.value["non_durable_expenditure"], 3800.0)
        self.assertEqual(result.value["durable_expenditure"], 2820.0)

    def test_single_target_thousands_separator(self):
        result = parse_response(
            "[2,530]",
            get_task_spec("cons_psid_jobloss"),
            repair_number_formatting=True,
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.value["annual_food_spending"], 2530.0)

    def test_leading_zero_group_is_repaired(self):
        # "[1,024]" is not decodable JSON at all, unlike "[2,530]".
        result = parse_response(
            "[1,024]",
            get_task_spec("cons_psid_jobloss"),
            repair_number_formatting=True,
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.value["annual_food_spending"], 1024.0)

    def test_negative_currency_symbol_is_repaired(self):
        result = parse_response(
            "[-$21,000]",
            get_task_spec("cons_psid_wealth"),
            repair_number_formatting=True,
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.value["next_wave_net_worth"], -21000.0)

    def test_compact_array_is_never_rewritten(self):
        """The regression that matters: "757" is a real element, not a digit group.

        Models that emit compact JSON produce commas that match the repair pattern.
        Because the repair only runs on a response that already failed, these rows
        must be byte-identical with the flag on and off.
        """
        spec = get_task_spec("cons_cex_stimulus08")
        strict = parse_response("[3333,2576,757]", spec)
        lenient = parse_response("[3333,2576,757]", spec, repair_number_formatting=True)
        self.assertTrue(strict.ok)
        self.assertEqual(strict.value, lenient.value)
        self.assertFalse(lenient.repaired)
        self.assertEqual(lenient.value["total_expenditure"], 3333.0)
        self.assertEqual(lenient.value["durable_expenditure"], 757.0)

    def test_repair_does_not_apply_to_distribution_tasks(self):
        spec = get_task_spec("house_sce_lockin")
        self.assertFalse(
            parse_response("[0.75, 0.25]", spec, repair_number_formatting=True).ok
        )

    def test_unrepairable_response_keeps_its_original_error(self):
        spec = get_task_spec("cons_cex_total")
        result = parse_response(
            "[100,200,300,400]",
            spec,
            repair_number_formatting=True,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "expected_3_array_elements")

    def test_scorer_counts_repaired_rows(self):
        gold = [
            {
                "example_id": "a",
                "assistant": "[6620, 3800, 2820]",
                "naive_baseline": "[5000, 3000, 2000]",
            }
        ]
        predictions = [{"example_id": "a", "prediction": "[6,620, 3,800, 2,820]"}]

        strict = score_task("cons_cex_total", gold, predictions)
        self.assertEqual(strict["num_valid"], 0)
        self.assertEqual(strict["num_repaired"], 0)

        lenient = score_task(
            "cons_cex_total",
            gold,
            predictions,
            repair_number_formatting=True,
        )
        self.assertEqual(lenient["num_valid"], 1)
        self.assertEqual(lenient["num_repaired"], 1)
        self.assertTrue(lenient["details"][0]["repaired"])
        self.assertEqual(lenient["metrics"]["primary_score"], 0.0)


class DistributionNestingRepairTests(unittest.TestCase):
    """The dominant invalidity mode for weaker open-weight models: right numbers,
    missing outer list. Accepted as valid, since the task's answer format makes it unambiguous."""

    def test_single_distribution_missing_outer_list(self):
        spec = get_task_spec("house_sce_move")
        raw = "<think>\n\n</think>\n\n[0.985, 0.015]"
        self.assertEqual(parse_response(raw, spec).error, "expected_1_distributions")
        result = parse_response(raw, spec, repair_distribution_nesting=True)
        self.assertTrue(result.ok)
        self.assertTrue(result.repaired)
        self.assertEqual(result.value["move_probability_12m"], (0.985, 0.015))

    def test_two_distributions_missing_outer_list(self):
        spec = get_task_spec("labor_sce_search")
        raw = "<think>\n\n</think>\n\n[0.96, 0.04], [0.97, 0.03]"
        self.assertEqual(
            parse_response(raw, spec).error,
            "find_acceptable_work_12m:expected_probability_vector",
        )
        result = parse_response(raw, spec, repair_distribution_nesting=True)
        self.assertTrue(result.ok)
        self.assertEqual(result.value["find_acceptable_work_12m"], (0.96, 0.04))
        self.assertEqual(result.value["find_acceptable_work_3m"], (0.97, 0.03))

    def test_flattened_vector_is_split_by_spec_lengths(self):
        spec = get_task_spec("labor_sce_search")
        raw = "<think>\n\n</think>\n\n[0.65, 0.35, 0.2, 0.8]"
        self.assertEqual(parse_response(raw, spec).error, "expected_2_distributions")
        result = parse_response(raw, spec, repair_distribution_nesting=True)
        self.assertTrue(result.ok)
        self.assertEqual(result.value["find_acceptable_work_12m"], (0.65, 0.35))
        self.assertEqual(result.value["find_acceptable_work_3m"], (0.2, 0.8))

    def test_repair_still_enforces_sum_to_one(self):
        # Right shape after unwrapping, but the vector does not sum to one, so it
        # stays invalid and keeps its ORIGINAL error rather than the repaired one.
        spec = get_task_spec("house_sce_move")
        result = parse_response(
            "[0.6, 0.6]",
            spec,
            repair_distribution_nesting=True,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "expected_1_distributions")

    def test_repair_still_enforces_probability_range(self):
        spec = get_task_spec("house_sce_move")
        self.assertFalse(
            parse_response("[-0.1, 1.1]", spec, repair_distribution_nesting=True).ok
        )

    def test_wrong_count_is_not_forced_into_the_spec(self):
        # Six distributions where the spec wants five: neither shape matches,
        # so the row stays invalid instead of being silently truncated.
        spec = get_task_spec("macro_sce_uncertainty")
        raw = "[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]"
        self.assertFalse(
            parse_response(raw, spec, repair_distribution_nesting=True).ok
        )

    def test_properly_nested_response_is_never_rewritten(self):
        spec = get_task_spec("house_sce_move")
        strict = parse_response("[[0.985, 0.015]]", spec)
        lenient = parse_response("[[0.985, 0.015]]", spec, repair_distribution_nesting=True)
        self.assertTrue(strict.ok)
        self.assertEqual(strict.value, lenient.value)
        self.assertFalse(lenient.repaired)

    def test_repair_does_not_apply_to_numeric_tasks(self):
        spec = get_task_spec("cons_cex_total")
        self.assertFalse(
            parse_response("[1.0, 2.0]", spec, repair_distribution_nesting=True).ok
        )


class MetricTests(unittest.TestCase):
    def test_numeric_metrics_match_hand_calculation(self):
        spec = get_task_spec("cons_cex_total")
        predictions = [
            {
                "total_expenditure": 2.0,
                "non_durable_expenditure": 4.0,
                "durable_expenditure": 6.0,
            },
            {
                "total_expenditure": 4.0,
                "non_durable_expenditure": 8.0,
                "durable_expenditure": 12.0,
            },
        ]
        references = [
            {
                "total_expenditure": 1.0,
                "non_durable_expenditure": 2.0,
                "durable_expenditure": 3.0,
            },
            {
                "total_expenditure": 3.0,
                "non_durable_expenditure": 6.0,
                "durable_expenditure": 9.0,
            },
        ]
        baselines = [
            {
                "total_expenditure": 0.0,
                "non_durable_expenditure": 0.0,
                "durable_expenditure": 0.0,
            },
            {
                "total_expenditure": 2.0,
                "non_durable_expenditure": 4.0,
                "durable_expenditure": 8.0,
            },
        ]

        metrics = compute_task_metrics(
            predictions,
            references,
            spec,
            naive_baselines=baselines,
        )

        self.assertEqual(metrics["primary_metric"], "mean_relmae")
        self.assertAlmostEqual(metrics["relmae_by_target"]["total_expenditure"], 1.0)
        self.assertAlmostEqual(metrics["relmae_by_target"]["non_durable_expenditure"], 1.0)
        self.assertAlmostEqual(metrics["relmae_by_target"]["durable_expenditure"], 1.5)
        self.assertAlmostEqual(metrics["mean_relmae"], 7 / 6)
        self.assertAlmostEqual(metrics["mean_nmae"], 1.0)
        self.assertIn("mean_nmae", metrics["diagnostic_metrics"])

    def test_numeric_metrics_reject_zero_baseline_denominator(self):
        spec = get_task_spec("macro_mich_outlook")
        with self.assertRaisesRegex(ValueError, "zero_naive_baseline_error"):
            compute_task_metrics(
                [
                    {
                        "expected_inflation_1y": 2.0,
                        "expected_inflation_5_to_10y_annual": 3.0,
                    }
                ],
                [
                    {
                        "expected_inflation_1y": 1.0,
                        "expected_inflation_5_to_10y_annual": 2.0,
                    }
                ],
                spec,
                naive_baselines=[
                    {
                        "expected_inflation_1y": 1.0,
                        "expected_inflation_5_to_10y_annual": 2.0,
                    }
                ],
            )

    def test_macro_f1_uses_all_allowed_labels(self):
        score = macro_f1(["yes", "yes", "no"], ["yes", "no", "no"], ("no", "yes"))
        self.assertAlmostEqual(score, (2 / 3 + 2 / 3) / 2)

    def test_categorical_reports_macro_f1_and_accuracy(self):
        gold = [
            {
                "id": "a",
                "assistant": '["yes"]',
                "naive_baseline": '["no"]',
            },
            {
                "id": "b",
                "assistant": '["no"]',
                "naive_baseline": '["no"]',
            },
        ]
        predictions = [
            {"id": "a", "prediction": '["yes"]'},
            {"id": "b", "prediction": '["yes"]'},
        ]
        result = score_task("house_psid_owner", gold, predictions)
        metrics = result["metrics"]
        self.assertEqual(metrics["accuracy_by_target"]["owns_home_next_wave"], 0.5)
        self.assertIn("macro_f1_by_target", metrics)
        self.assertEqual(metrics["primary_metric"], "mean_macro_f1")

    def test_total_variation(self):
        self.assertAlmostEqual(total_variation((0.6, 0.4), (0.8, 0.2)), 0.2)

    def test_distribution_task_metric(self):
        result = score_task(
            "house_sce_lockin",
            [
                {
                    "id": "a",
                    "assistant": "[[0.8, 0.2]]",
                    "naive_baseline": "[[0.7, 0.3]]",
                },
                {
                    "id": "b",
                    "assistant": "[[0.4, 0.6]]",
                    "naive_baseline": "[[0.5, 0.5]]",
                },
            ],
            [
                {"id": "a", "prediction": "[[0.6, 0.4]]"},
                {"id": "b", "prediction": "[[0.5, 0.5]]"},
            ],
        )
        self.assertAlmostEqual(
            result["metrics"]["tv_by_distribution"]["move_probability_3y"],
            0.15,
        )
        self.assertEqual(result["metrics"]["primary_metric"], "mean_total_variation")

    def test_valid_subset_and_ids_are_always_logged(self):
        gold = [
            {
                "id": "a",
                "assistant": "[1, 2, 3]",
                "naive_baseline": "[0, 1, 2]",
            },
            {
                "id": "b",
                "assistant": "[2, 4, 6]",
                "naive_baseline": "[1, 2, 3]",
            },
            {
                "id": "c",
                "assistant": "[3, 6, 9]",
                "naive_baseline": "[2, 3, 4]",
            },
        ]
        predictions = [
            {"id": "a", "prediction": "[2, 2, 4]"},
            {"id": "b", "prediction": "invalid"},
        ]

        result = score_task("cons_cex_total", gold, predictions)

        self.assertEqual(result["num_valid"], 1)
        self.assertEqual(result["valid_response_rate"], 1 / 3)
        self.assertEqual(result["coverage_rate"], 2 / 3)
        self.assertEqual(result["valid_example_ids"], ["a"])
        self.assertEqual(
            result["invalid_examples"],
            [
                {"id": "b", "error": "expected_json_array"},
                {"id": "c", "error": "missing_prediction"},
            ],
        )
        self.assertTrue(result["details"][0]["valid"])
        self.assertEqual(
            result["details"][0]["naive_baseline"]["total_expenditure"],
            0.0,
        )
        self.assertFalse(result["details"][1]["valid"])

    def test_numeric_task_requires_naive_baseline(self):
        with self.assertRaisesRegex(ValueError, "Invalid naive baseline"):
            score_task(
                "cons_cex_total",
                [{"id": "a", "assistant": "[1, 2, 3]"}],
                [{"id": "a", "prediction": "[1, 2, 3]"}],
            )


class ExperimentConfigTests(unittest.TestCase):
    def test_config_rejects_unknown_metric_field(self):
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "name: invalid",
                        "description: invalid metric override",
                        "data_root: /tmp/data",
                        "prediction_root: /tmp/predictions",
                        "metric: mae",
                    ]
                )
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "Unknown ExperimentConfig field"):
                load_config_path(config_path)

    def test_config_rejects_unknown_nested_field(self):
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "name: invalid",
                        "description: invalid metric override",
                        "data_root: /tmp/data",
                        "prediction_root: /tmp/predictions",
                        "run:",
                        "  metric: mae",
                    ]
                )
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "Unknown RunConfig field"):
                load_config_path(config_path)

    def test_config_backed_experiment_writes_metrics(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            pred_root = root / "pred"
            (data_root / "prompts").mkdir(parents=True)
            pred_root.mkdir()

            (data_root / "prompts" / "cons_cex_total.jsonl").write_text(
                '{"id":"a","assistant":"[1, 2, 3]",'
                '"naive_baseline":"[0, 1, 2]"}\n'
            )
            (pred_root / "cons_cex_total.jsonl").write_text(
                '{"id":"a","prediction":"[2, 2, 4]"}\n'
            )
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "name: tiny_householdbench_scoring",
                        "description: tiny scoring test",
                        f"data_root: {data_root}",
                        f"prediction_root: {pred_root}",
                        "output_dir: results",
                        'gold_path_template: "{data_root}/prompts/{task}.jsonl"',
                        'prediction_path_template: "{prediction_root}/{task}.jsonl"',
                        "model:",
                        "  model_type: external_predictions",
                        "  model_name: tiny-model",
                        "run:",
                        "  seed: 42",
                        "  split: test",
                        "  prediction_field: prediction",
                        "  include_details: false",
                        "task_groups:",
                        "  - name: tiny",
                        "    output_subdir: tiny",
                        "    tasks:",
                        "      - cons_cex_total",
                    ]
                )
                + "\n"
            )

            final = run_experiment(load_config_path(config_path), root / "run")
            result_path = (
                root / "run" / "results" / "tiny" / "tiny-model" / "cons_cex_total.json"
            )
            result_text = result_path.read_text()

            self.assertEqual(final["experiment"], "tiny_householdbench_scoring")
            self.assertIn('"mean_relmae"', result_text)
            self.assertIn('"mean_nmae"', result_text)
            self.assertIn('"valid_example_ids"', result_text)
            self.assertNotIn('"details"', result_text)


if __name__ == "__main__":
    unittest.main()
