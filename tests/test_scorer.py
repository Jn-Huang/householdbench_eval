"""Scorer edge cases: id handling, prediction fields, and coverage accounting."""

import pytest

from household_bench_eval.scorer import score_task

GOLD = [
    {"id": "a", "assistant": "[1, 2, 3]", "naive_baseline": "[0, 1, 2]"},
    {"id": "b", "assistant": "[2, 4, 6]", "naive_baseline": "[1, 2, 3]"},
]


def test_duplicate_prediction_ids_are_rejected():
    predictions = [
        {"id": "a", "prediction": "[1, 2, 3]"},
        {"id": "a", "prediction": "[2, 3, 4]"},
    ]
    with pytest.raises(ValueError, match="Duplicate prediction ids"):
        score_task("cons_cex_total", GOLD, predictions)


def test_prediction_rows_require_an_id():
    with pytest.raises(ValueError, match="missing required field"):
        score_task("cons_cex_total", GOLD, [{"prediction": "[1, 2, 3]"}])


def test_example_id_is_accepted_as_row_key():
    predictions = [
        {"example_id": "a", "prediction": "[1, 2, 3]"},
        {"example_id": "b", "prediction": "[2, 4, 6]"},
    ]
    result = score_task("cons_cex_total", GOLD, predictions)
    assert result["num_valid"] == 2


def test_default_field_priority_prefers_prediction():
    predictions = [
        {"id": "a", "prediction": "[1, 2, 3]", "response": "not json"},
        {"id": "b", "response": "[2, 4, 6]"},
    ]
    result = score_task("cons_cex_total", GOLD, predictions)
    assert result["num_valid"] == 2


def test_prediction_field_override_is_strict():
    predictions = [
        {"id": "a", "raw_output": "[1, 2, 3]"},
        {"id": "b", "raw_output": "[2, 4, 6]"},
    ]
    relaxed = score_task("cons_cex_total", GOLD, predictions)
    assert relaxed["num_valid"] == 2

    strict = score_task(
        "cons_cex_total",
        GOLD,
        predictions,
        prediction_field="prediction",
    )
    assert strict["num_valid"] == 0
    assert all(
        "no configured response field" in example["error"]
        for example in strict["invalid_examples"]
    )


def test_extra_predictions_are_reported():
    predictions = [
        {"id": "a", "prediction": "[1, 2, 3]"},
        {"id": "b", "prediction": "[2, 4, 6]"},
        {"id": "zzz", "prediction": "[9, 9, 9]"},
    ]
    result = score_task("cons_cex_total", GOLD, predictions)
    assert result["num_extra_predictions"] == 1
    assert result["extra_prediction_ids"] == ["zzz"]
    assert result["num_valid"] == 2


def test_empty_prediction_file_yields_null_metrics():
    result = score_task("cons_cex_total", GOLD, [])
    assert result["num_valid"] == 0
    assert result["coverage_rate"] == 0.0
    assert result["metrics"]["primary_score"] is None
    assert result["error_counts"] == {"missing_prediction": 2}


# Released prompt files repeat an id when a household appears in several periods.
PANEL_GOLD = [
    {"id": "h1", "time": "2019-12-01", "assistant": "[1.0]", "naive_baseline": "[0.0]"},
    {"id": "h1", "time": "2020-03-01", "assistant": "[3.0]", "naive_baseline": "[0.0]"},
    {"id": "h2", "time": "2020-03-01", "assistant": "[5.0]", "naive_baseline": "[0.0]"},
]


def test_repeated_ids_join_on_id_and_time():
    predictions = [
        {"id": "h2", "time": "2020-03-01", "prediction": "[5.0]"},
        {"id": "h1", "time": "2020-03-01", "prediction": "[3.0]"},
        {"id": "h1", "time": "2019-12-01", "prediction": "[1.0]"},
    ]
    result = score_task("cons_sce_growth", PANEL_GOLD, predictions)
    assert result["num_valid"] == 3
    assert result["valid_example_ids"] == ["h1@2019-12-01", "h1@2020-03-01", "h2@2020-03-01"]
    assert result["metrics"]["primary_score"] == 0.0


def test_repeated_ids_require_time_in_predictions():
    predictions = [{"id": row["id"], "prediction": row["assistant"]} for row in PANEL_GOLD]
    with pytest.raises(ValueError, match="has no 'time'"):
        score_task("cons_sce_growth", PANEL_GOLD, predictions)


def test_repeated_id_time_pairs_are_rejected():
    gold = PANEL_GOLD + [dict(PANEL_GOLD[0])]
    with pytest.raises(ValueError, match="repeat the key 'h1@2019-12-01'"):
        score_task("cons_sce_growth", gold, [])


def test_repeated_ids_without_time_are_rejected():
    gold = [{k: v for k, v in row.items() if k != "time"} for row in PANEL_GOLD]
    with pytest.raises(ValueError, match="repeat the key 'h1'"):
        score_task("cons_sce_growth", gold, [])


def test_repeated_ids_with_partial_example_ids_are_rejected():
    gold = [dict(PANEL_GOLD[0], example_id="t:0")] + [
        {k: v for k, v in row.items() if k != "time"} for row in PANEL_GOLD
    ]
    with pytest.raises(ValueError, match="repeat the key 'h1'"):
        score_task("cons_sce_growth", gold, [])


def test_example_id_gold_with_repeated_ids_matches_on_example_id():
    gold = [dict(row, example_id=f"cons_sce_growth:{i:09d}") for i, row in enumerate(PANEL_GOLD)]
    predictions = [
        {"example_id": row["example_id"], "id": row["id"], "prediction": row["assistant"]}
        for row in gold
    ]
    result = score_task("cons_sce_growth", gold, predictions)
    assert result["num_valid"] == 3
    assert result["valid_example_ids"] == [row["example_id"] for row in gold]


def test_id_only_predictions_match_when_ids_are_unique():
    gold = [dict(row, time="2020-03-01") for row in GOLD]
    predictions = [
        {"id": "a", "prediction": "[1, 2, 3]"},
        {"id": "b", "example_id": "task:000001", "prediction": "[2, 4, 6]"},
    ]
    result = score_task("cons_cex_total", gold, predictions)
    assert result["num_valid"] == 2
    assert result["valid_example_ids"] == ["a@2020-03-01", "b@2020-03-01"]


def test_wrong_time_is_reported_as_time_mismatch():
    gold = [dict(row, time="2020-03-01") for row in GOLD]
    predictions = [
        {"id": "a", "time": "2020-03-01", "prediction": "[1, 2, 3]"},
        {"id": "b", "time": "2020-03-01T00:00:00", "prediction": "[2, 4, 6]"},
    ]
    result = score_task("cons_cex_total", gold, predictions)
    assert result["num_valid"] == 1
    assert result["invalid_examples"] == [{"id": "b@2020-03-01", "error": "time_mismatch"}]
    assert result["extra_prediction_ids"] == ["b@2020-03-01T00:00:00"]
