"""Spec consistency and parse round-trips across all 32 task specs."""

import json

import pytest

from household_bench_eval.parsing import parse_response
from household_bench_eval.tasks import ALL_TASKS, TASK_SPECS, get_task_spec


@pytest.mark.parametrize("task_name", ALL_TASKS)
def test_spec_is_internally_consistent(task_name):
    spec = TASK_SPECS[task_name]
    assert spec.name == task_name
    assert spec.mode in ("baseline", "policy")
    assert spec.summary

    if spec.output_shape == "array":
        assert spec.fields
        assert not spec.distributions
        assert spec.target_type in ("numeric", "categorical")
    else:
        assert spec.output_shape == "distribution_array"
        assert spec.distributions
        assert not spec.fields
        assert spec.target_type == "distribution"
        assert all(len(distribution.labels) >= 2 for distribution in spec.distributions)

    if spec.target_type == "numeric":
        assert all(field.kind == "numeric" for field in spec.fields)
        assert all(not field.labels for field in spec.fields)
        assert spec.official_metrics == ("relmae",)
        assert spec.diagnostic_metrics == ("nmae",)
    elif spec.target_type == "categorical":
        assert all(field.kind == "categorical" for field in spec.fields)
        assert all(len(field.labels) >= 2 for field in spec.fields)
        assert spec.official_metrics == ("macro_f1", "accuracy")
        assert spec.diagnostic_metrics == ()
    else:
        assert spec.official_metrics == ("total_variation",)
        assert spec.diagnostic_metrics == ()


def _canonical_response(spec) -> str:
    """Build a minimal valid raw response straight from the task spec."""
    if spec.output_shape == "array":
        values = [
            1.5 if field.kind == "numeric" else field.labels[0] for field in spec.fields
        ]
        return json.dumps(values)
    vectors = [
        [1.0 / len(distribution.labels)] * len(distribution.labels)
        for distribution in spec.distributions
    ]
    return json.dumps(vectors)


@pytest.mark.parametrize("task_name", ALL_TASKS)
def test_canonical_response_round_trips(task_name):
    spec = TASK_SPECS[task_name]
    result = parse_response(_canonical_response(spec), spec)
    assert result.ok, result.error

    expected_keys = (
        {field.name for field in spec.fields}
        if spec.output_shape == "array"
        else {distribution.name for distribution in spec.distributions}
    )
    assert set(result.value) == expected_keys


@pytest.mark.parametrize("task_name", ALL_TASKS)
def test_wrong_length_array_is_rejected(task_name):
    spec = TASK_SPECS[task_name]
    payload = json.loads(_canonical_response(spec))
    payload.append(payload[-1])
    result = parse_response(json.dumps(payload), spec)
    assert not result.ok


def test_unknown_task_raises_with_available_list():
    with pytest.raises(KeyError, match="Unknown HouseholdBench task"):
        get_task_spec("not_a_task")


def test_markdown_fenced_json_is_extracted():
    result = parse_response(
        "```json\n[10.0, 13.0]\n```",
        get_task_spec("macro_mich_outlook"),
    )
    assert result.ok
    assert result.value["expected_inflation_1y"] == 10.0


def test_prose_wrapped_json_is_extracted():
    result = parse_response(
        "The household expects [10.0, 13.0].",
        get_task_spec("macro_mich_outlook"),
    )
    assert result.ok


def test_boolean_values_are_not_numbers():
    result = parse_response("[true, false]", get_task_spec("cons_sce_shock"))
    assert not result.ok
    assert "expected_number" in result.error


def test_non_finite_numbers_are_rejected():
    result = parse_response("[NaN, 1.0]", get_task_spec("cons_sce_shock"))
    assert not result.ok
    assert "non_finite_number" in result.error
