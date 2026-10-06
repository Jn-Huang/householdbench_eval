"""Task-level evaluation metrics for HouseholdBench predictions."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from scripts.utils.responses import (
    validate_categorical_array,
    validate_distribution_array,
    validate_numeric_array,
)


def score_numeric(
    targets: Sequence[list[float]],
    predictions: Sequence[list[float]],
    naive_baselines: Sequence[list[float]],
    *,
    validate_inputs: bool = True,
) -> dict[str, Any]:
    """Compute atomic and equal-weight task RelMAE."""
    target_count = _common_outer_length(targets, predictions, naive_baselines)
    if validate_inputs:
        for row_number, (target, prediction, baseline) in enumerate(
            zip(targets, predictions, naive_baselines, strict=True),
            start=1,
        ):
            validate_numeric_array(
                target, length=target_count, field=f"target row {row_number}"
            )
            validate_numeric_array(
                prediction,
                length=target_count,
                field=f"prediction row {row_number}",
            )
            validate_numeric_array(
                baseline,
                length=target_count,
                field=f"naive_baseline row {row_number}",
            )

    target_array = np.asarray(targets, dtype=float)
    prediction_array = np.asarray(predictions, dtype=float)
    baseline_array = np.asarray(naive_baselines, dtype=float)
    numerator = np.abs(target_array - prediction_array).sum(axis=0)
    denominator = np.abs(target_array - baseline_array).sum(axis=0)
    zero_denominators = np.flatnonzero(denominator == 0)
    if len(zero_denominators):
        raise ValueError(
            f"Numeric target {int(zero_denominators[0])} has a zero "
            "naive-baseline MAE denominator."
        )
    atomic_scores = (numerator / denominator).tolist()

    return {
        "atomic_relmae": atomic_scores,
        "task_relmae": sum(atomic_scores) / target_count,
        "naive_task_relmae": 1.0,
    }


def score_categorical(
    targets: Sequence[list[str]],
    predictions: Sequence[list[str]],
    naive_baselines: Sequence[list[str]],
    *,
    category_sets: Sequence[set[str] | frozenset[str]],
    validate_inputs: bool = True,
) -> dict[str, Any]:
    """Compute atomic and equal-weight task macro-F1.

    A class-specific F1 contribution is zero when its denominator is zero.
    """
    target_count = _common_outer_length(targets, predictions, naive_baselines)
    if len(category_sets) != target_count:
        raise ValueError("category_sets length does not match the categorical target count.")
    if validate_inputs:
        for row_number, (target, prediction, baseline) in enumerate(
            zip(targets, predictions, naive_baselines, strict=True),
            start=1,
        ):
            for field, value in [
                ("target", target),
                ("prediction", prediction),
                ("naive_baseline", baseline),
            ]:
                for target_index, categories in enumerate(category_sets):
                    validate_categorical_array(
                        [value[target_index]],
                        length=1,
                        categories=categories,
                        field=f"{field} row {row_number} target {target_index}",
                    )

    target_array = np.asarray(targets, dtype=object)
    prediction_array = np.asarray(predictions, dtype=object)
    baseline_array = np.asarray(naive_baselines, dtype=object)
    model_atomic = [
        macro_f1(
            target_array[:, target_index],
            prediction_array[:, target_index],
            category_sets[target_index],
        )
        for target_index in range(target_count)
    ]
    naive_atomic = [
        macro_f1(
            target_array[:, target_index],
            baseline_array[:, target_index],
            category_sets[target_index],
        )
        for target_index in range(target_count)
    ]
    return {
        "atomic_macro_f1": model_atomic,
        "task_macro_f1": sum(model_atomic) / target_count,
        "naive_atomic_macro_f1": naive_atomic,
        "naive_task_macro_f1": sum(naive_atomic) / target_count,
    }


def score_distribution(
    targets: Sequence[list[list[float]]],
    predictions: Sequence[list[list[float]]],
    naive_baselines: Sequence[list[list[float]]],
    *,
    category_counts: Sequence[int],
    validate_inputs: bool = True,
    include_respondent: bool = True,
) -> dict[str, Any]:
    """Compute respondent TV, atomic mean TV, and equal-weight task mean TV."""
    distribution_count = _common_outer_length(targets, predictions, naive_baselines)
    if len(category_counts) != distribution_count:
        raise ValueError(
            "category_counts length does not match the number of atomic distributions."
        )
    if validate_inputs:
        for row_number, (target, prediction, baseline) in enumerate(
            zip(targets, predictions, naive_baselines, strict=True),
            start=1,
        ):
            validate_distribution_array(
                target,
                category_counts=category_counts,
                field=f"target row {row_number}",
            )
            validate_distribution_array(
                prediction,
                category_counts=category_counts,
                field=f"prediction row {row_number}",
            )
            validate_distribution_array(
                baseline,
                category_counts=category_counts,
                field=f"naive_baseline row {row_number}",
            )

    respondent_columns = []
    naive_respondent_columns = []
    for distribution_index in range(distribution_count):
        target_array = np.asarray(
            [row[distribution_index] for row in targets], dtype=float
        )
        prediction_array = np.asarray(
            [row[distribution_index] for row in predictions], dtype=float
        )
        baseline_array = np.asarray(
            [row[distribution_index] for row in naive_baselines], dtype=float
        )
        respondent_columns.append(
            0.5 * np.abs(target_array - prediction_array).sum(axis=1)
        )
        naive_respondent_columns.append(
            0.5 * np.abs(target_array - baseline_array).sum(axis=1)
        )

    respondent_array = np.column_stack(respondent_columns)
    naive_respondent_array = np.column_stack(naive_respondent_columns)
    if not np.isfinite(respondent_array).all() or not np.isfinite(
        naive_respondent_array
    ).all():
        raise ValueError("Total Variation is not finite.")
    atomic_tv = respondent_array.mean(axis=0).tolist()
    naive_atomic_tv = naive_respondent_array.mean(axis=0).tolist()
    result = {
        "atomic_mean_tv": atomic_tv,
        "task_mean_tv": sum(atomic_tv) / distribution_count,
        "naive_atomic_mean_tv": naive_atomic_tv,
        "naive_task_mean_tv": sum(naive_atomic_tv) / distribution_count,
    }
    if include_respondent:
        result["respondent_tv"] = respondent_array.tolist()
        result["naive_respondent_tv"] = naive_respondent_array.tolist()
    return result


def _common_outer_length(*collections: Sequence[list[Any]]) -> int:
    if not collections or not collections[0]:
        raise ValueError("Evaluation inputs must contain at least one record.")
    row_count = len(collections[0])
    if any(len(collection) != row_count for collection in collections[1:]):
        raise ValueError("Target, prediction, and naive-baseline row counts differ.")
    target_count = len(collections[0][0])
    if target_count == 0:
        raise ValueError("Evaluation arrays must contain at least one target.")
    if any(len(row) != target_count for collection in collections for row in collection):
        raise ValueError("Evaluation array shapes differ across records.")
    return target_count


def macro_f1(
    targets: np.ndarray,
    predictions: np.ndarray,
    categories: Sequence[Any] | set[str] | frozenset[str],
) -> float:
    targets = np.asarray(targets).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    if targets.shape != predictions.shape:
        raise ValueError("Macro-F1 truth and prediction shapes differ.")
    if not categories:
        raise ValueError("Each categorical target must define at least one class.")
    class_scores = []
    for category in sorted(categories):
        target_positive = targets == category
        prediction_positive = predictions == category
        true_positive = np.count_nonzero(target_positive & prediction_positive)
        false_positive = np.count_nonzero(~target_positive & prediction_positive)
        false_negative = np.count_nonzero(target_positive & ~prediction_positive)
        denominator = 2 * true_positive + false_positive + false_negative
        class_scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(class_scores) / len(class_scores)
