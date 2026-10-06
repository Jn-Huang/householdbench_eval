"""Shared statistical contracts and repeated mechanics for the XGBoost baseline."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scripts.utils.io import sha256_file, sha256_json
from scripts.utils.evaluation import macro_f1
from scripts.utils.prompt_rendering import (
    PROMPT_MONTH_RULES,
    MONTH_NAMES,
)
from scripts.utils.table_schema import load_table_schema
from scripts.utils.task_registry import TASK_REGISTRY_PATH
from scripts.utils.responses import NUMERIC_TASKS, CATEGORICAL_TARGET_DOMAINS, DISTRIBUTION_GROUPS


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASELINE_SPEC_VERSION = "3.0"
PUBLICATION_SPEC_VERSION = "5.0"
MODEL_SEEDS_PATH = PROJECT_ROOT / "scripts" / "3_xgboost" / "model_seeds.csv"
TRAINING_PROFILES = {
    "train_400k": {"train": 400_000, "validation": 50_000},
}
PROFILE_COLUMN_PREFIXES = {
    "train_400k": "xgb_400k",
}
DIRECT_VARIANTS = ("xgb_tuned_direct",)
MULTI_OUTPUT_STRATEGY = "one_output_per_tree"
MISSING_CATEGORY = "__HOUSEHOLDBENCH_MISSING__"
UNSEEN_CATEGORY = "__HOUSEHOLDBENCH_UNSEEN__"

TUNED_CLASSIFICATION_PARAMETERS = {
    "max_depth": 6,
    "learning_rate": 0.08,
    "subsample": 0.65,
    "colsample_bytree": 1.00,
    "colsample_bylevel": 0.90,
    "min_child_weight": 0.000005,
    "reg_lambda": 0.00,
    "reg_alpha": 0.00,
    "gamma": 0.00,
    "tree_method": "hist",
    "max_bin": 256,
    "n_estimators": 1_000,
    "early_stopping_rounds": 300,
}
TUNED_REGRESSION_PARAMETERS = {
    "max_depth": 9,
    "learning_rate": 0.05,
    "subsample": 0.70,
    "colsample_bytree": 1.00,
    "colsample_bylevel": 1.00,
    "min_child_weight": 2.00,
    "reg_lambda": 0.00,
    "reg_alpha": 0.00,
    "gamma": 0.00,
    "tree_method": "hist",
    "max_bin": 256,
    "n_estimators": 1_000,
    "early_stopping_rounds": 300,
}
TUNED_DISTRIBUTION_PARAMETERS = {
    **TUNED_REGRESSION_PARAMETERS,
    "n_estimators": 2_000,
}

XGBOOST_TASKS = NUMERIC_TASKS | set(CATEGORICAL_TARGET_DOMAINS) | set(DISTRIBUTION_GROUPS)


@dataclass(frozen=True)
class XGBoostContract:
    """Explicit task registries for one XGBoost publication.

    The default contract is the released HouseholdBench suite.  Candidate
    suites can pass another contract without changing this module's globals.
    """

    numeric_tasks: frozenset[str]
    categorical_target_domains: dict[str, dict[str, list[str]]]
    distribution_groups: dict[str, dict[str, list[str]]]

    @property
    def task_ids(self) -> frozenset[str]:
        return frozenset(
            set(self.numeric_tasks)
            | set(self.categorical_target_domains)
            | set(self.distribution_groups)
        )


CORE_XGBOOST_CONTRACT = XGBoostContract(
    numeric_tasks=frozenset(NUMERIC_TASKS),
    categorical_target_domains=CATEGORICAL_TARGET_DOMAINS,
    distribution_groups=DISTRIBUTION_GROUPS,
)


def _contract_or_core(contract: XGBoostContract | None) -> XGBoostContract:
    return CORE_XGBOOST_CONTRACT if contract is None else contract


def load_model_seeds(
    path: Path,
    schema: dict[str, dict[str, Any]],
    *,
    contract: XGBoostContract | None = None,
) -> dict[tuple[str, str, str], int]:
    """Require one explicit historical seed for every retained model."""
    active = _contract_or_core(contract)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    keys = ["task_id", "output_name", "variant"]
    if frame.columns.tolist() != [*keys, "seed"]:
        raise ValueError("Model seed table must contain task_id, output_name, variant, seed.")
    if frame.duplicated(keys).any():
        raise ValueError("Model seed table repeats a model.")
    if not frame["seed"].str.fullmatch(r"[0-9]+").all():
        raise ValueError("Model seeds must be unsigned integers.")
    seeds = {
        (row.task_id, row.output_name, row.variant): int(row.seed)
        for row in frame.itertuples(index=False)
    }
    if any(seed >= 2**32 for seed in seeds.values()):
        raise ValueError("Model seeds must be smaller than 2**32.")
    expected = {
        (task_id, output_name, variant)
        for task_id in active.task_ids
        for output_name in (
            active.distribution_groups[task_id]
            if task_id in active.distribution_groups
            else schema[task_id]["targets"]
        )
        for variant in DIRECT_VARIANTS
    }
    if set(seeds) != expected:
        raise ValueError(
            f"Model seed coverage differs: missing={sorted(expected - set(seeds))}; "
            f"unexpected={sorted(set(seeds) - expected)}"
        )
    return seeds


def prediction_column(training_profile: str, variant: str) -> str:
    if training_profile not in TRAINING_PROFILES:
        raise KeyError(f"Unknown XGBoost training profile: {training_profile}")
    suffixes = {
        "xgb_tuned_direct": "tuned_direct",
    }
    if variant not in suffixes:
        raise KeyError(f"Unknown XGBoost variant: {variant}")
    return f"{PROFILE_COLUMN_PREFIXES[training_profile]}_{suffixes[variant]}"


def prediction_columns(
    task_id: str, *, contract: XGBoostContract | None = None
) -> list[str]:
    response_family(task_id, contract=contract)
    columns = []
    for training_profile in TRAINING_PROFILES:
        for variant in DIRECT_VARIANTS:
            columns.append(prediction_column(training_profile, variant))
    return columns


def model_specification_sha256(
    feature_registry_sha256: str,
    *,
    model_seeds_sha256: str,
    contract: XGBoostContract | None = None,
) -> str:
    """Hash only choices that can alter fitted trees or their interpretation."""
    active = _contract_or_core(contract)
    return sha256_json(
        {
            "baseline_spec_version": BASELINE_SPEC_VERSION,
            "model_seeds_sha256": model_seeds_sha256,
            "feature_registry_sha256": feature_registry_sha256,
            "training_profiles": TRAINING_PROFILES,
            "tuned_classification_parameters": TUNED_CLASSIFICATION_PARAMETERS,
            "tuned_regression_parameters": TUNED_REGRESSION_PARAMETERS,
            "tuned_distribution_parameters": TUNED_DISTRIBUTION_PARAMETERS,
            "multi_output_strategy": MULTI_OUTPUT_STRATEGY,
            "numeric_tasks": sorted(active.numeric_tasks),
            "categorical_target_domains": active.categorical_target_domains,
            "distribution_groups": active.distribution_groups,
        }
    )


def select_xgboost_schema(
    schema: dict[str, dict[str, list[str]]],
    *,
    contract: XGBoostContract | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Project a possibly broader HouseholdBench schema to the approved XGBoost suite."""
    active = _contract_or_core(contract)
    missing = sorted(active.task_ids - set(schema))
    if missing:
        raise ValueError(f"Table schema is missing XGBoost tasks: {missing}")
    return {task_id: schema[task_id] for task_id in sorted(active.task_ids)}


def response_family(
    task_id: str, *, contract: XGBoostContract | None = None
) -> str:
    active = _contract_or_core(contract)
    if task_id in active.numeric_tasks:
        return "numeric"
    if task_id in active.categorical_target_domains:
        return "categorical"
    if task_id in active.distribution_groups:
        return "distribution"
    raise KeyError(f"Unknown XGBoost response family: {task_id}")


def load_feature_types(path: Path = TASK_REGISTRY_PATH) -> dict[str, dict[str, list[str]]]:
    """Derive the existing ordered model-feature view from the table contract."""
    registry = {}
    for task_id, spec in load_table_schema(path).items():
        categorical = spec["categorical_predictors"].copy()
        if spec["prompt_month"] is not None:
            categorical.append("prompt_month")
        registry[task_id] = {
            "categorical": categorical,
            "numeric": [
                column for column in spec["prompt_predictors"] if column not in categorical
            ],
        }
    return registry


def feature_registry_sha256(registry: dict[str, dict[str, list[str]]]) -> str:
    """Keep the established feature digest and model seeds after registry consolidation."""
    return hashlib.sha256((json.dumps(registry, indent=2) + "\n").encode("utf-8")).hexdigest()


def validate_baseline_registries(
    schema: dict[str, dict[str, list[str]]],
    feature_types: dict[str, dict[str, list[str]]],
    *,
    contract: XGBoostContract | None = None,
) -> None:
    active = _contract_or_core(contract)
    task_ids = set(schema)
    if task_ids != set(active.task_ids) or task_ids != set(feature_types):
        raise ValueError("XGBoost task registries do not equal the table-schema task set.")
    family_sets = [
        set(active.numeric_tasks),
        set(active.categorical_target_domains),
        set(active.distribution_groups),
    ]
    if set.union(*family_sets) != task_ids or any(
        left & right
        for index, left in enumerate(family_sets)
        for right in family_sets[index + 1 :]
    ):
        raise ValueError("Response-family registries must form a disjoint task partition.")
    for task_id, task_schema in schema.items():
        targets = task_schema["targets"]
        if task_id in active.categorical_target_domains:
            if list(active.categorical_target_domains[task_id]) != targets:
                raise ValueError(f"{task_id}: categorical target order differs from table schema.")
        if task_id in active.distribution_groups:
            flattened = [
                column
                for columns in active.distribution_groups[task_id].values()
                for column in columns
            ]
            if flattened != targets:
                raise ValueError(f"{task_id}: distribution grouping differs from table schema.")
        permitted = [*task_schema["prompt_predictors"]]
        if task_id in PROMPT_MONTH_RULES:
            permitted.append("prompt_month")
        entry = feature_types[task_id]
        combined = [*entry["numeric"], *entry["categorical"]]
        if set(combined) != set(permitted) or len(combined) != len(permitted):
            raise ValueError(f"{task_id}: feature registry does not cover prompt-visible inputs exactly.")
        ordered = [column for column in permitted if column in set(entry["numeric"])] + [
            column for column in permitted if column in set(entry["categorical"])
        ]
        if combined != ordered:
            raise ValueError(f"{task_id}: feature registry does not preserve within-type schema order.")

def model_parameters(
    *,
    variant: str,
    model_kind: str,
    seed: int,
    threads: int,
    class_count: int | None = None,
) -> dict[str, Any]:
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer.")
    if variant != "xgb_tuned_direct":
        raise ValueError(f"Unsupported XGBoost variant: {variant}")
    if model_kind == "classifier":
        parameters = dict(TUNED_CLASSIFICATION_PARAMETERS)
    elif model_kind == "distribution":
        parameters = dict(TUNED_DISTRIBUTION_PARAMETERS)
    elif model_kind == "regressor":
        parameters = dict(TUNED_REGRESSION_PARAMETERS)
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")

    parameters.update(
        {
            "enable_categorical": True,
            "device": "cpu",
            "random_state": seed,
            "n_jobs": threads,
            "verbosity": 0,
        }
    )
    if model_kind in {"regressor", "distribution"}:
        parameters["objective"] = "reg:absoluteerror"
        parameters["eval_metric"] = "mae" if model_kind == "regressor" else "projected_tv"
    else:
        if class_count is None or class_count < 2:
            raise ValueError("class_count must be at least two for a classifier.")
        parameters["eval_metric"] = "macro_f1_cost"
        if class_count == 2:
            parameters["objective"] = "binary:logistic"
        else:
            parameters["objective"] = "multi:softprob"
            parameters["num_class"] = class_count
    if model_kind == "distribution":
        parameters["multi_strategy"] = MULTI_OUTPUT_STRATEGY
    return parameters


def _month_feature(frame: pd.DataFrame, *, task_id: str) -> pd.Series:
    fixed_month = PROMPT_MONTH_RULES[task_id]["fixed_month"]
    if fixed_month is None:
        month = pd.to_numeric(frame["time"].astype("string").str.slice(5, 7), errors="coerce")
    else:
        month = pd.Series(fixed_month, index=frame.index, dtype="int64")
    if month.isna().any() or not month.between(1, 12).all():
        raise ValueError(f"{task_id}: cannot derive a valid prompt month.")
    return month.astype(int).map(lambda value: MONTH_NAMES[value - 1]).astype("string")


def prepare_features(
    frames: dict[str, pd.DataFrame],
    *,
    task_id: str,
    predictors: Sequence[str],
    categorical_predictors: set[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if "train" not in frames or frames["train"].empty:
        raise ValueError(f"{task_id}: feature construction requires nonempty train rows.")
    feature_names = list(predictors)
    if len(feature_names) != len(set(feature_names)):
        raise ValueError(f"{task_id}: duplicate feature names.")
    unknown_categorical = categorical_predictors - set(feature_names)
    if unknown_categorical:
        raise ValueError(f"{task_id}: categorical features absent from predictor order: {unknown_categorical}")

    raw: dict[str, pd.DataFrame] = {}
    for split_name, frame in frames.items():
        missing = [column for column in feature_names if column != "prompt_month" and column not in frame]
        if missing:
            raise KeyError(f"{task_id}: {split_name} is missing features: {missing}")
        raw[split_name] = pd.DataFrame(
            {
                column: (
                    _month_feature(frame, task_id=task_id)
                    if column == "prompt_month"
                    else frame[column]
                )
                for column in feature_names
            },
            index=frame.index,
        )

    prepared_columns: dict[str, dict[str, pd.Series]] = {name: {} for name in raw}
    category_schemas: dict[str, list[str]] = {}
    unseen_counts: dict[str, dict[str, int]] = {name: {} for name in raw}
    missing_rates: dict[str, dict[str, float]] = {name: {} for name in raw}

    for column in feature_names:
        if column not in categorical_predictors:
            for split_name, frame in raw.items():
                source = frame[column]
                numeric = pd.to_numeric(source, errors="coerce")
                invalid = source.notna() & numeric.isna()
                if invalid.any():
                    examples = source.loc[invalid].astype(str).head(5).tolist()
                    raise ValueError(f"{task_id}: nonnumeric values in {column}: {examples}")
                prepared_columns[split_name][column] = numeric.astype("float64")
                missing_rates[split_name][column] = float(numeric.isna().mean())
            continue

        train_source = raw["train"][column].astype("string").str.strip()
        train_source = train_source.mask(train_source.eq(""), pd.NA)
        observed = sorted(train_source.dropna().unique().tolist())
        reserved_collision = {MISSING_CATEGORY, UNSEEN_CATEGORY} & set(observed)
        if reserved_collision:
            raise ValueError(f"{task_id}: reserved categorical token occurs in {column}.")
        categories = [*observed, MISSING_CATEGORY, UNSEEN_CATEGORY]
        dtype = pd.CategoricalDtype(categories=categories, ordered=False)
        category_schemas[column] = categories
        observed_set = set(observed)
        for split_name, frame in raw.items():
            source = frame[column].astype("string").str.strip()
            source = source.mask(source.eq(""), pd.NA)
            unseen = source.notna() & ~source.isin(observed_set)
            encoded = source.mask(unseen, UNSEEN_CATEGORY).fillna(MISSING_CATEGORY)
            prepared_columns[split_name][column] = encoded.astype(dtype)
            unseen_counts[split_name][column] = int(unseen.sum())
            missing_rates[split_name][column] = float(source.isna().mean())

    prepared = {
        split_name: pd.DataFrame(columns, index=raw[split_name].index).loc[:, feature_names]
        for split_name, columns in prepared_columns.items()
    }
    return prepared, {
        "feature_names": feature_names,
        "feature_types": [
            "c" if column in categorical_predictors else "q" for column in feature_names
        ],
        "category_schemas": category_schemas,
        "unseen_counts": unseen_counts,
        "missing_rates": missing_rates,
    }


def make_distribution_matrix(frame: pd.DataFrame, columns: Sequence[str], *, label: str) -> np.ndarray:
    if len(columns) == 1:
        probability = pd.to_numeric(frame[columns[0]], errors="coerce").to_numpy(dtype=float)
        matrix = np.column_stack([1.0 - probability, probability])
    else:
        matrix = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2 or not np.isfinite(matrix).all():
        raise ValueError(f"{label}: distribution labels must be a finite two-dimensional matrix.")
    if ((matrix < 0.0) | (matrix > 1.0)).any():
        raise ValueError(f"{label}: distribution labels lie outside [0, 1].")
    if not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError(f"{label}: distribution labels do not sum to one.")
    return matrix


def project_simplex_fast(values: Any) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    if raw.ndim != 2 or raw.shape[1] < 2 or not np.isfinite(raw).all():
        raise ValueError("Simplex projection requires a finite two-dimensional array.")
    ordered = np.sort(raw, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    divisors = np.arange(1, raw.shape[1] + 1, dtype=float)
    positive = ordered - cumulative / divisors > 0.0
    rho = positive.sum(axis=1) - 1
    if (rho < 0).any():
        raise RuntimeError("Simplex projection did not identify an active coordinate.")
    threshold = cumulative[np.arange(len(raw)), rho] / (rho + 1.0)
    projected = np.maximum(raw - threshold[:, None], 0.0)
    if not np.allclose(projected.sum(axis=1), 1.0, rtol=0.0, atol=1e-10):
        raise RuntimeError("Projected rows do not sum to one.")
    return projected


def project_rows_to_simplex(values: Any) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(values, dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    projected = project_simplex_fast(raw)
    adjustment = np.abs(projected - raw)
    changed = np.any(adjustment > 1e-12, axis=1)
    return projected, {
        "rows": int(len(raw)),
        "raw_coordinates_below_zero": int((raw < 0).sum()),
        "raw_coordinates_above_one": int((raw > 1).sum()),
        "vectors_changed": int(changed.sum()),
        "mean_absolute_adjustment": float(adjustment.mean()),
        "maximum_absolute_adjustment": float(adjustment.max()),
        "raw_sum_mean": float(raw.sum(axis=1).mean()),
        "raw_sum_min": float(raw.sum(axis=1).min()),
        "raw_sum_max": float(raw.sum(axis=1).max()),
    }


def make_macro_f1_cost(class_count: int) -> Any:
    def macro_f1_cost(y_true: Any, y_prediction: Any) -> float:
        truth = np.asarray(y_true, dtype=int).reshape(-1)
        values = np.asarray(y_prediction)
        if class_count == 2 and values.shape == (len(truth),):
            predicted = (values > 0.5).astype(int)
        elif values.shape == (len(truth), class_count):
            predicted = np.argmax(values, axis=1)
        elif values.size == len(truth) * class_count:
            predicted = np.argmax(values.reshape(len(truth), class_count), axis=1)
        else:
            raise ValueError(
                f"Unexpected prediction shape {values.shape} for {class_count} classes."
            )
        return 1.0 - macro_f1(truth, predicted, range(class_count))

    macro_f1_cost.__name__ = "macro_f1_cost"
    return macro_f1_cost


def projected_tv(y_true: Any, y_prediction: Any) -> float:
    truth = np.asarray(y_true, dtype=float)
    projected = project_simplex_fast(y_prediction)
    if truth.shape != projected.shape:
        raise ValueError("Projected-TV truth and prediction shapes differ.")
    return float(np.mean(0.5 * np.abs(truth - projected).sum(axis=1)))


projected_tv.__name__ = "projected_tv"


def fit_scalar_target(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_validation: pd.DataFrame | None,
    y_validation: pd.Series | np.ndarray | None,
    *,
    variant: str,
    seed: int,
    threads: int,
    class_domain: Sequence[str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    import xgboost as xgb

    started = dt.datetime.now(dt.timezone.utc)
    clock = time.perf_counter()
    if class_domain is None:
        model_kind = "regressor"
        train_values = pd.to_numeric(pd.Series(y_train), errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(train_values).all():
            raise ValueError("Numeric training targets must be finite and nonmissing.")
        validation_values = None
        if y_validation is not None:
            validation_values = pd.to_numeric(pd.Series(y_validation), errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(validation_values).all():
                raise ValueError("Numeric validation targets must be finite and nonmissing.")
        parameters = model_parameters(
            variant=variant, model_kind=model_kind, seed=seed, threads=threads
        )
        runtime_parameters = dict(parameters)
        model = xgb.XGBRegressor(**runtime_parameters)
    else:
        model_kind = "classifier"
        domain = list(class_domain)
        if len(domain) != len(set(domain)) or len(domain) < 2:
            raise ValueError("Categorical target domain must contain distinct classes.")
        class_map = {category: index for index, category in enumerate(domain)}
        train_labels = pd.Series(y_train).astype("string")
        if set(train_labels) != set(domain):
            raise ValueError(
                f"Training classes {sorted(set(train_labels))} do not equal {sorted(domain)}."
            )
        train_values = train_labels.map(class_map).to_numpy(dtype=int)
        validation_values = None
        if y_validation is not None:
            validation_labels = pd.Series(y_validation).astype("string")
            unknown = sorted(set(validation_labels) - set(domain))
            if unknown:
                raise ValueError(f"Validation target contains unknown classes: {unknown}")
            validation_values = validation_labels.map(class_map).to_numpy(dtype=int)
        parameters = model_parameters(
            variant=variant,
            model_kind=model_kind,
            seed=seed,
            threads=threads,
            class_count=len(domain),
        )
        runtime_parameters = dict(parameters)
        if parameters.get("eval_metric") == "macro_f1_cost":
            runtime_parameters["eval_metric"] = make_macro_f1_cost(len(domain))
        model = xgb.XGBClassifier(**runtime_parameters)

    fit_arguments: dict[str, Any] = {"verbose": False}
    if X_validation is None or validation_values is None or X_validation.empty:
        raise ValueError("Fitting requires nonempty validation features and labels.")
    fit_arguments["eval_set"] = [(X_validation, validation_values)]
    model.fit(X_train, train_values, **fit_arguments)
    elapsed = time.perf_counter() - clock
    best_iteration = getattr(model, "best_iteration", None)
    best_score = getattr(model, "best_score", None)
    return model, {
        "model_kind": model_kind,
        "parameters": parameters,
        "class_domain": None if class_domain is None else list(class_domain),
        "started_at": started.isoformat(),
        "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "best_iteration": None if best_iteration is None else int(best_iteration),
        "best_score": None if best_score is None else float(best_score),
    }


def fit_distribution_group(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_validation: pd.DataFrame | None,
    y_validation: np.ndarray | None,
    *,
    variant: str,
    seed: int,
    threads: int,
) -> tuple[Any, dict[str, Any]]:
    import xgboost as xgb

    if y_train.ndim != 2 or y_train.shape[1] < 2:
        raise ValueError("Multi-output fitting requires at least two label coordinates.")
    parameters = model_parameters(
        variant=variant,
        model_kind="distribution",
        seed=seed,
        threads=threads,
    )
    started = dt.datetime.now(dt.timezone.utc)
    clock = time.perf_counter()
    runtime_parameters = dict(parameters)
    runtime_parameters["eval_metric"] = projected_tv
    model = xgb.XGBRegressor(**runtime_parameters)
    fit_arguments: dict[str, Any] = {"verbose": False}
    if X_validation is None or y_validation is None or X_validation.empty:
        raise ValueError("Multi-output fitting requires validation data.")
    fit_arguments["eval_set"] = [(X_validation, y_validation)]
    model.fit(X_train, y_train, **fit_arguments)
    elapsed = time.perf_counter() - clock
    best_iteration = getattr(model, "best_iteration", None)
    best_score = getattr(model, "best_score", None)
    return model, {
        "model_kind": "distribution",
        "parameters": parameters,
        "class_domain": None,
        "started_at": started.isoformat(),
        "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "best_iteration": None if best_iteration is None else int(best_iteration),
        "best_score": None if best_score is None else float(best_score),
    }


def save_model_and_metadata(
    model: Any,
    *,
    model_path: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    X_roundtrip: pd.DataFrame,
    expected_prediction: Any,
) -> dict[str, Any]:
    import xgboost as xgb

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    model_tmp = model_path.with_name(f"{model_path.stem}.tmp.{os.getpid()}{model_path.suffix}")
    metadata_tmp = metadata_path.with_name(f"{metadata_path.name}.tmp.{os.getpid()}")
    model.save_model(model_tmp)
    model_tmp.replace(model_path)
    loaded = xgb.XGBClassifier() if metadata["model_kind"] == "classifier" else xgb.XGBRegressor()
    loaded.load_model(model_path)
    roundtrip_prediction = np.asarray(loaded.predict(X_roundtrip))
    expected = np.asarray(expected_prediction)
    if roundtrip_prediction.shape != expected.shape:
        raise RuntimeError(f"Model roundtrip changed prediction shape for {model_path}.")
    if expected.dtype.kind in "OUS":
        if not np.array_equal(roundtrip_prediction, expected):
            raise RuntimeError(f"Model roundtrip changed categorical predictions for {model_path}.")
        maximum_difference = 0.0
    else:
        maximum_difference = float(np.max(np.abs(roundtrip_prediction.astype(float) - expected.astype(float))))
        if maximum_difference > 1e-10:
            raise RuntimeError(f"Model roundtrip changed predictions for {model_path}: {maximum_difference}")
    booster = model.get_booster()
    completed = {
        **metadata,
        "model_sha256": sha256_file(model_path),
        "model_bytes": model_path.stat().st_size,
        "serialization_roundtrip_max_abs_difference": maximum_difference,
        "feature_names": booster.feature_names,
        "feature_types": booster.feature_types,
        "xgboost_version": xgb.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    metadata_tmp.write_text(
        json.dumps(completed, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metadata_tmp.replace(metadata_path)
    return completed


def load_completed_model(
    *, model_path: Path, metadata_path: Path, expected_model_identity: str
) -> tuple[Any, dict[str, Any]] | None:
    if not model_path.is_file() and not metadata_path.is_file():
        return None
    if not model_path.is_file() or not metadata_path.is_file():
        raise RuntimeError(f"Incomplete model checkpoint: {model_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model_identity_sha256") != expected_model_identity:
        raise RuntimeError(f"Checkpoint specification differs for {model_path}.")
    if metadata.get("model_sha256") != sha256_file(model_path):
        raise RuntimeError(f"Checkpoint checksum differs for {model_path}.")
    import xgboost as xgb

    model = xgb.XGBClassifier() if metadata["model_kind"] == "classifier" else xgb.XGBRegressor()
    model.load_model(model_path)
    return model, metadata


def assemble_response_arrays(
    task_id: str,
    outputs: dict[str, Any],
    *,
    target_order: Sequence[str],
    contract: XGBoostContract | None = None,
) -> list[list[Any]]:
    active = _contract_or_core(contract)
    family = response_family(task_id, contract=active)
    if family == "distribution":
        group_names = list(active.distribution_groups[task_id])
        if set(outputs) != set(group_names):
            raise ValueError(f"{task_id}: distribution output groups differ from registry.")
        row_count = len(np.asarray(outputs[group_names[0]]))
        arrays = []
        for row_index in range(row_count):
            arrays.append(
                [
                    [float(value) for value in np.asarray(outputs[group])[row_index]]
                    for group in group_names
                ]
            )
        return arrays

    if set(outputs) != set(target_order):
        raise ValueError(f"{task_id}: scalar outputs differ from target registry.")
    row_count = len(np.asarray(outputs[target_order[0]]))
    arrays: list[list[Any]] = []
    for row_index in range(row_count):
        row: list[Any] = []
        for target in target_order:
            value = np.asarray(outputs[target])[row_index]
            if family == "categorical":
                row.append(str(value))
            elif task_id == "cons_psid_wealth":
                row.append(int(np.rint(float(value))))
            else:
                row.append(float(value))
        arrays.append(row)
    return arrays


def canonical_json_array(value: list[Any]) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def read_selected_prompt_answers(
    prompt_path: Path, *, selected_keys: set[tuple[str, str]]
) -> pd.DataFrame:
    rows = []
    with prompt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = (str(record["id"]), str(record["time"]))
            if key in selected_keys:
                rows.append(
                    {
                        "id": key[0],
                        "time": key[1],
                        "assistant": json.loads(record["assistant"]),
                        "naive_baseline": json.loads(record["naive_baseline"]),
                    }
                )
    answers = pd.DataFrame(rows)
    if len(answers) != len(selected_keys) or answers.duplicated(["id", "time"]).any():
        raise RuntimeError(
            f"{prompt_path}: selected prompt answers do not match the requested keys."
        )
    return answers


def runtime_identity() -> dict[str, Any]:
    try:
        import xgboost as xgb

        xgboost_version = xgb.__version__
    except ImportError:
        xgboost_version = None
    return {
        "baseline_spec_version": BASELINE_SPEC_VERSION,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "xgboost_version": xgboost_version,
    }
