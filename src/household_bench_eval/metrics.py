"""Metric implementations for HouseholdBench scoring."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from statistics import fmean, pstdev

from household_bench_eval.parsing import DistributionValue, ParsedRecord
from household_bench_eval.tasks import FieldSpec, TaskSpec


def compute_task_metrics(
    predictions: list[ParsedRecord],
    references: list[ParsedRecord],
    spec: TaskSpec,
    naive_baselines: list[ParsedRecord] | None = None,
) -> dict[str, object]:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")

    if spec.target_type == "numeric":
        if naive_baselines is None or len(naive_baselines) != len(references):
            raise ValueError("numeric tasks require one naive baseline per scored example")
        if not predictions:
            return {
                "metric_category": "numeric",
                "primary_metric": "mean_relmae",
                "primary_score": None,
                "score_direction": "lower_is_better",
                "relmae_by_target": {},
                "mean_relmae": None,
                "nmae_by_target": {},
                "mean_nmae": None,
                "diagnostic_metrics": ["nmae_by_target", "mean_nmae"],
            }
        metrics = _numeric_metrics(
            predictions,
            references,
            naive_baselines,
            spec.numeric_fields,
        )
        return {
            "metric_category": "numeric",
            "primary_metric": "mean_relmae",
            "primary_score": metrics["mean_relmae"],
            "score_direction": "lower_is_better",
            **metrics,
        }

    if naive_baselines is not None:
        raise ValueError("naive baselines are only accepted for numeric task scoring")

    if spec.target_type == "categorical":
        if not predictions:
            return {
                "metric_category": "categorical",
                "primary_metric": "mean_macro_f1",
                "primary_score": None,
                "score_direction": "higher_is_better",
                "macro_f1_by_target": {},
                "mean_macro_f1": None,
                "accuracy_by_target": {},
                "mean_accuracy": None,
            }
        metrics = _categorical_metrics(
            predictions,
            references,
            spec.categorical_fields,
        )
        return {
            "metric_category": "categorical",
            "primary_metric": "mean_macro_f1",
            "primary_score": metrics["mean_macro_f1"],
            "score_direction": "higher_is_better",
            **metrics,
        }

    if spec.target_type == "distribution":
        if not predictions:
            return {
                "metric_category": "distribution",
                "primary_metric": "mean_total_variation",
                "primary_score": None,
                "score_direction": "lower_is_better",
                "tv_by_distribution": {},
                "mean_total_variation": None,
            }
        metrics = _distribution_metrics(predictions, references, spec)
        return {
            "metric_category": "distribution",
            "primary_metric": "mean_total_variation",
            "primary_score": metrics["mean_total_variation"],
            "score_direction": "lower_is_better",
            **metrics,
        }

    raise ValueError(f"unsupported_target_type:{spec.target_type}")


def _numeric_metrics(
    predictions: list[ParsedRecord],
    references: list[ParsedRecord],
    naive_baselines: list[ParsedRecord],
    fields: Iterable[FieldSpec],
) -> dict[str, object]:
    relmae_by_target: dict[str, float] = {}
    nmae_by_target: dict[str, float | None] = {}

    for field in fields:
        pred_values = [float(row[field.name]) for row in predictions]
        ref_values = [float(row[field.name]) for row in references]
        baseline_values = [float(row[field.name]) for row in naive_baselines]

        model_absolute_error = sum(
            abs(pred - ref)
            for pred, ref in zip(pred_values, ref_values, strict=True)
        )
        baseline_absolute_error = sum(
            abs(baseline - ref)
            for baseline, ref in zip(baseline_values, ref_values, strict=True)
        )
        if baseline_absolute_error == 0:
            raise ValueError(f"{field.name}:zero_naive_baseline_error")

        mae = model_absolute_error / len(ref_values)
        reference_sd = pstdev(ref_values)
        relmae_by_target[field.name] = model_absolute_error / baseline_absolute_error
        nmae_by_target[field.name] = mae / reference_sd if reference_sd > 0 else None

    usable_nmae = [value for value in nmae_by_target.values() if value is not None]
    return {
        "relmae_by_target": relmae_by_target,
        "mean_relmae": fmean(relmae_by_target.values()),
        "nmae_by_target": nmae_by_target,
        "mean_nmae": fmean(usable_nmae) if usable_nmae else None,
        "diagnostic_metrics": ["nmae_by_target", "mean_nmae"],
    }


def _categorical_metrics(
    predictions: list[ParsedRecord],
    references: list[ParsedRecord],
    fields: Iterable[FieldSpec],
) -> dict[str, object]:
    macro_f1_by_target: dict[str, float] = {}
    accuracy_by_target: dict[str, float] = {}

    for field in fields:
        pred_values = [row[field.name] for row in predictions]
        ref_values = [row[field.name] for row in references]
        macro_f1_by_target[field.name] = macro_f1(
            pred_values,
            ref_values,
            field.labels,
        )
        accuracy_by_target[field.name] = accuracy(pred_values, ref_values)

    return {
        "macro_f1_by_target": macro_f1_by_target,
        "mean_macro_f1": (
            fmean(macro_f1_by_target.values()) if macro_f1_by_target else None
        ),
        "accuracy_by_target": accuracy_by_target,
        "mean_accuracy": fmean(accuracy_by_target.values()) if accuracy_by_target else None,
    }


def _distribution_metrics(
    predictions: list[ParsedRecord],
    references: list[ParsedRecord],
    spec: TaskSpec,
) -> dict[str, object]:
    tv_by_distribution: dict[str, float] = {}

    for distribution in spec.distributions:
        example_distances = [
            total_variation(
                _as_distribution(prediction[distribution.name], distribution.name),
                _as_distribution(reference[distribution.name], distribution.name),
            )
            for prediction, reference in zip(predictions, references, strict=True)
        ]
        tv_by_distribution[distribution.name] = (
            fmean(example_distances) if example_distances else math.nan
        )

    return {
        "tv_by_distribution": tv_by_distribution,
        "mean_total_variation": (
            fmean(tv_by_distribution.values()) if tv_by_distribution else None
        ),
    }


def total_variation(
    prediction: DistributionValue,
    reference: DistributionValue,
) -> float:
    if len(prediction) != len(reference):
        raise ValueError("probability vectors must have equal length")
    return 0.5 * sum(
        abs(predicted_probability - reference_probability)
        for predicted_probability, reference_probability in zip(
            prediction,
            reference,
            strict=True,
        )
    )


def accuracy(predictions: list[object], references: list[object]) -> float:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    if not predictions:
        return math.nan
    return sum(
        prediction == reference
        for prediction, reference in zip(predictions, references, strict=True)
    ) / len(predictions)


def macro_f1(
    predictions: list[object],
    references: list[object],
    labels: tuple[object, ...],
) -> float:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    if not predictions:
        return math.nan

    pred_counts = Counter(predictions)
    ref_counts = Counter(references)
    true_positive_counts = Counter(
        prediction
        for prediction, reference in zip(predictions, references, strict=True)
        if prediction == reference
    )

    f1s: list[float] = []
    for label in labels:
        true_positives = true_positive_counts[label]
        false_positives = pred_counts[label] - true_positives
        false_negatives = ref_counts[label] - true_positives
        denominator = (2 * true_positives) + false_positives + false_negatives
        f1s.append((2 * true_positives / denominator) if denominator else 0.0)
    return fmean(f1s)


def _as_distribution(value: object, name: str) -> DistributionValue:
    if not isinstance(value, tuple):
        raise TypeError(f"{name}:expected_normalized_distribution")
    return value
