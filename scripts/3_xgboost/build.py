#!/usr/bin/env python
"""Generate the unified HouseholdBench XGBoost statistical-baseline publication."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
import resource
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.evaluation import score_categorical, score_distribution, score_numeric
from scripts.utils.sampling import (
    SPLIT_VALUES,
    load_sampling_config,
    split_prefix,
    validate_sampling_sidecar,
)
from scripts.utils.io import (
    publish_staged_directory,
    sha256_file,
    sha256_json,
)
from scripts.utils.table_schema import load_table_schema
from scripts.utils.task_registry import TASK_REGISTRY_PATH
from scripts.utils.xgboost_baseline import (
    BASELINE_SPEC_VERSION,
    CORE_XGBOOST_CONTRACT,
    feature_registry_sha256,
    MULTI_OUTPUT_STRATEGY,
    MODEL_SEEDS_PATH,
    PUBLICATION_SPEC_VERSION,
    TRAINING_PROFILES,
    XGBoostContract,
    assemble_response_arrays,
    canonical_json_array,
    fit_distribution_group,
    fit_scalar_target,
    load_completed_model,
    load_feature_types,
    load_model_seeds,
    make_distribution_matrix,
    model_parameters,
    model_specification_sha256,
    prediction_column,
    prediction_columns,
    prepare_features,
    project_rows_to_simplex,
    read_selected_prompt_answers,
    response_family,
    runtime_identity,
    save_model_and_metadata,
    select_xgboost_schema,
    validate_baseline_registries,
)


PACKAGE_ROOT = PROJECT_ROOT / "data" / "householdbench"
PROMPTS_DIR = PACKAGE_ROOT / "prompts"
TABULAR_DIR = PACKAGE_ROOT / "tabular"
SPLITS_DIR = PACKAGE_ROOT / "splits"
DATA_ROOT = PACKAGE_ROOT / "baselines"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "householdbench" / "baselines"
DATA_STAGING = DATA_ROOT.with_name("baselines.staging")
OUTPUT_STAGING = OUTPUT_ROOT.with_name("baselines.staging")


@dataclass(frozen=True)
class XGBoostPublicationConfig:
    """Explicit registries and paths for one baseline publication."""

    contract: XGBoostContract
    task_registry_path: Path
    package_root: Path
    prompts_dir: Path
    tabular_dir: Path
    splits_dir: Path
    data_root: Path
    output_root: Path
    data_staging: Path
    output_staging: Path
    model_seeds_path: Path = MODEL_SEEDS_PATH


CORE_PUBLICATION_CONFIG = XGBoostPublicationConfig(
    contract=CORE_XGBOOST_CONTRACT,
    task_registry_path=TASK_REGISTRY_PATH,
    package_root=PACKAGE_ROOT,
    prompts_dir=PROMPTS_DIR,
    tabular_dir=TABULAR_DIR,
    splits_dir=SPLITS_DIR,
    data_root=DATA_ROOT,
    output_root=OUTPUT_ROOT,
    data_staging=DATA_STAGING,
    output_staging=OUTPUT_STAGING,
)


def _write_frame_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_json_atomic(value: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_full_input_preflight(
    *,
    schema: dict[str, dict[str, list[str]]],
    split_manifest: pd.DataFrame,
    config: XGBoostPublicationConfig,
) -> pd.DataFrame:
    """Audit row alignment and target support before fitting any model."""
    rows: list[dict[str, object]] = []
    for task_id in sorted(schema):
        table_path = config.tabular_dir / f"{task_id}.csv"
        sidecar_path = config.splits_dir / f"{task_id}.csv"
        prompt_path = config.prompts_dir / f"{task_id}.jsonl"
        split_row = split_manifest.loc[split_manifest["task_id"].eq(task_id)]
        if len(split_row) != 1:
            raise RuntimeError(f"{task_id}: preflight expected one split-manifest row.")
        if sha256_file(table_path) != str(split_row.iloc[0]["table_sha256"]):
            raise RuntimeError(f"{task_id}: preflight table checksum differs.")
        if sha256_file(sidecar_path) != str(split_row.iloc[0]["sidecar_sha256"]):
            raise RuntimeError(f"{task_id}: preflight sidecar checksum differs.")

        targets = schema[task_id]["targets"]
        predictors = schema[task_id]["prompt_predictors"]
        table = pd.read_csv(
            table_path,
            usecols=["id", "time", "release_date", *targets, *predictors],
            dtype={"id": "string", "time": "string", "release_date": "string"},
            low_memory=False,
        )
        sidecar = pd.read_csv(
            sidecar_path,
            dtype={"id": "string", "time": "string", "release_date": "string"},
        )
        key_columns = ["id", "time", "release_date"]
        if not table[key_columns].equals(sidecar[key_columns]):
            raise RuntimeError(f"{task_id}: preflight table and sidecar row order differs.")

        table_ids = table["id"].astype(str).to_numpy()
        table_times = table["time"].astype(str).to_numpy()
        prompt_rows = 0
        with prompt_path.open("r", encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                if not line.strip():
                    raise ValueError(f"{task_id}: blank prompt line in preflight.")
                if row_index >= len(table):
                    raise RuntimeError(f"{task_id}: prompt has more rows than the table.")
                record = json.loads(line)
                expected_key = (table_ids[row_index], table_times[row_index])
                if (str(record["id"]), str(record["time"])) != expected_key:
                    raise RuntimeError(f"{task_id}: table and prompt row order differs.")
                prompt_rows += 1
        if prompt_rows != len(table):
            raise RuntimeError(f"{task_id}: prompt and table row counts differ.")
        rows.append(
            {
                "task_id": task_id,
                "item_type": "task_alignment",
                "item_name": task_id,
                "rows": len(table),
                "status": "PASS",
            }
        )

        family = response_family(task_id, contract=config.contract)
        if family == "numeric":
            values = table[targets].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"{task_id}: numeric targets must be finite and nonmissing.")
        elif family == "categorical":
            for target, domain in config.contract.categorical_target_domains[task_id].items():
                if not table[target].isin(domain).all():
                    raise ValueError(f"{task_id} {target}: targets lie outside the class domain.")
        else:
            for group_name, columns in config.contract.distribution_groups[task_id].items():
                make_distribution_matrix(table, columns, label=f"{task_id} {group_name}")
        del table, sidecar
        gc.collect()

    audit = pd.DataFrame(rows)
    if len(audit) != len(schema):
        raise RuntimeError("Full-input preflight coverage differs from the fixed contract.")
    return audit


def _resolve_checkpoint(
    *,
    model_path: Path,
    metadata_path: Path,
    model_identity_sha256: str,
    output_root: Path,
    output_staging: Path,
) -> tuple[Any, dict[str, Any], str] | None:
    """Reuse a checkpoint only when its exact fit identity is unchanged."""
    completed = load_completed_model(
        model_path=model_path,
        metadata_path=metadata_path,
        expected_model_identity=model_identity_sha256,
    )
    origin = "resumed_staging"
    if completed is not None:
        model, metadata = completed
        if "fit_identity_sha256" in metadata or "split_revision" in metadata:
            metadata.pop("fit_identity_sha256", None)
            metadata.pop("split_revision", None)
            _write_json_atomic(metadata, metadata_path)
        return model, metadata, origin

    active_model = output_root / model_path.relative_to(output_staging)
    active_metadata = active_model.with_suffix(".metadata.json")
    if active_model.is_file() and active_metadata.is_file():
        active = json.loads(active_metadata.read_text(encoding="utf-8"))
        if active.get("model_identity_sha256") == model_identity_sha256:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(active_model, model_path)
            shutil.copy2(active_metadata, metadata_path)
            completed = load_completed_model(
                model_path=model_path,
                metadata_path=metadata_path,
                expected_model_identity=model_identity_sha256,
            )
            if completed is None:
                raise RuntimeError(f"Failed to copy current checkpoint: {active_model}")
            model, metadata = completed
            metadata.pop("fit_identity_sha256", None)
            metadata.pop("split_revision", None)
            _write_json_atomic(metadata, metadata_path)
            return model, metadata, "reused_current"
    return None


def _check_model_features(
    model: Any, metadata: dict[str, Any], prepared: pd.DataFrame, feature_audit: dict[str, Any]
) -> None:
    if model.get_booster().feature_names != list(prepared.columns):
        raise RuntimeError("Serialized model feature order differs from reconstructed features.")
    if metadata["feature_audit"]["category_schemas"] != feature_audit["category_schemas"]:
        raise RuntimeError("Serialized model categorical schemas differ from reconstructed schemas.")


def _model_record(
    *,
    task_id: str,
    training_profile: str,
    variant: str,
    output_name: str,
    response_family_name: str,
    metadata: dict[str, Any],
    origin: str,
    model_path: Path,
    metadata_path: Path,
    output_root: Path,
    output_staging: Path,
) -> dict[str, object]:
    final_model_path = output_root / model_path.relative_to(output_staging)
    final_metadata_path = output_root / metadata_path.relative_to(output_staging)
    return {
        "task_id": task_id,
        "training_profile": training_profile,
        "variant": variant,
        "output_name": output_name,
        "response_family": response_family_name,
        "model_kind": metadata["model_kind"],
        "checkpoint_origin": origin,
        "best_iteration": metadata.get("best_iteration"),
        "best_score": metadata.get("best_score"),
        "elapsed_seconds": metadata["elapsed_seconds"],
        "model_bytes": metadata["model_bytes"],
        "model_sha256": metadata["model_sha256"],
        "model_identity_sha256": metadata["model_identity_sha256"],
        "fit_data_sha256": metadata["fit_data_sha256"],
        "publication_provenance_sha256": metadata["publication_provenance_sha256"],
        "model_path": str(final_model_path.relative_to(PROJECT_ROOT)),
        "metadata_path": str(final_metadata_path.relative_to(PROJECT_ROOT)),
        "parameters": json.dumps(metadata["parameters"], sort_keys=True),
        "multi_output_strategy": metadata.get("multi_output_strategy") or "",
        "train_rows": metadata["train_rows"],
        "validation_rows": metadata["validation_rows"],
    }


def _record_feature_importance(
    rows: list[dict[str, object]], model: Any, record: dict[str, object]
) -> None:
    scores = model.get_booster().get_score(importance_type="gain")
    total_gain = sum(float(value) for value in scores.values())
    for feature_name, gain in scores.items():
        rows.append(
            {
                "task_id": record["task_id"],
                "training_profile": record["training_profile"],
                "variant": record["variant"],
                "output_name": record["output_name"],
                "feature": feature_name,
                "gain": float(gain),
                "gain_share": float(gain) / total_gain if total_gain else 0.0,
            }
        )


def _profile_selection(
    sidecar: pd.DataFrame, *, task_id: str, training_profile: str
) -> tuple[dict[str, pd.DataFrame], list[pd.DataFrame]]:
    caps = TRAINING_PROFILES[training_profile]
    selected: dict[str, pd.DataFrame] = {}
    audits = []
    for split_name in ("train", "validation"):
        requested = int(caps[split_name])
        selected_split = split_prefix(
            sidecar, split=split_name, requested_rows=requested
        )
        available = (
            sidecar.loc[sidecar["split"].eq(split_name)]
            .groupby(["split", "split_quarter"], sort=True, observed=True)
            .size()
            .rename("available_rows")
            .reset_index()
        )
        realised = (
            selected_split.groupby(
                ["split", "split_quarter"], sort=True, observed=True
            )
            .size()
            .rename("prefix_rows")
            .reset_index()
        )
        audit = available.merge(
            realised,
            on=["split", "split_quarter"],
            how="left",
            validate="one_to_one",
        )
        audit["prefix_rows"] = audit["prefix_rows"].fillna(0).astype(int)
        selected[split_name] = selected_split
        audits.append(
            audit.assign(
                task_id=task_id,
                training_profile=training_profile,
                requested_rows=requested,
            )
        )
    return selected, audits


def _update_frame_digest(
    digest: Any, *, label: str, frame: pd.DataFrame | pd.Series | np.ndarray
) -> None:
    """Hash one ordered fitting object without serializing it as a large JSON value."""
    if isinstance(frame, np.ndarray):
        array = np.asarray(frame)
        if array.ndim == 1:
            frame = pd.Series(array, name="value")
        elif array.ndim == 2:
            frame = pd.DataFrame(
                array, columns=[f"coordinate_{index}" for index in range(array.shape[1])]
            )
        else:
            raise ValueError(f"Cannot hash {label}: expected a one- or two-dimensional array.")
    if isinstance(frame, pd.Series):
        frame = frame.rename(frame.name or "value").to_frame()
    frame = frame.reset_index(drop=True)
    header = {
        "label": label,
        "rows": len(frame),
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
    }
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for start in range(0, len(frame), 100_000):
        chunk = frame.iloc[start : start + 100_000]
        hashes = pd.util.hash_pandas_object(chunk, index=False, categorize=False)
        digest.update(hashes.to_numpy(dtype="uint64", copy=False).tobytes())


def _fit_data_sha256(
    *,
    train_keys: pd.DataFrame,
    validation_keys: pd.DataFrame,
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    train_target: pd.DataFrame | pd.Series | np.ndarray,
    validation_target: pd.DataFrame | pd.Series | np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for label, frame in (
        ("train_keys", train_keys),
        ("validation_keys", validation_keys),
        ("train_features", train_features),
        ("validation_features", validation_features),
        ("train_target", train_target),
        ("validation_target", validation_target),
    ):
        _update_frame_digest(digest, label=label, frame=frame)
    return digest.hexdigest()


def _append_categorical_calibration(
    rows: list[dict[str, object]],
    *,
    task_id: str,
    training_profile: str,
    variant: str,
    target: str,
    sample: pd.DataFrame,
    probabilities: np.ndarray,
    contract: XGBoostContract,
) -> None:
    domain = contract.categorical_target_domains[task_id][target]
    if probabilities.shape != (len(sample), len(domain)):
        raise RuntimeError(f"{task_id} {target}: invalid class-probability shape.")
    actual = sample[target].astype("string").to_numpy()
    for split_name in ("validation", "pre_cutoff_test", "post_cutoff_test"):
        split_mask = sample["split"].eq(split_name).to_numpy()
        if not split_mask.any():
            continue
        for class_index, class_label in enumerate(domain):
            probability = probabilities[split_mask, class_index]
            observed = (actual[split_mask] == class_label).astype(float)
            bins = np.minimum((probability * 10).astype(int), 9)
            for bin_index in range(10):
                bin_mask = bins == bin_index
                if bin_mask.any():
                    rows.append(
                        {
                            "task_id": task_id,
                            "training_profile": training_profile,
                            "variant": variant,
                            "target": target,
                            "split": split_name,
                            "class": class_label,
                            "probability_bin": bin_index,
                            "rows": int(bin_mask.sum()),
                            "mean_predicted_probability": float(
                                probability[bin_mask].mean()
                            ),
                            "observed_frequency": float(observed[bin_mask].mean()),
                        }
                    )


def build_xgboost_publication(
    *,
    threads: int,
    config: XGBoostPublicationConfig = CORE_PUBLICATION_CONFIG,
) -> None:
    """Fit or reuse the 400k literature-informed models and publish predictions."""
    import xgboost as xgb

    training_profile = "train_400k"
    variant = "xgb_tuned_direct"
    caps = TRAINING_PROFILES[training_profile]

    # Validate the publication inputs and preserve the existing run identity.
    if xgb.__version__ != "3.3.0":
        raise RuntimeError(f"Expected XGBoost 3.3.0, found {xgb.__version__}.")
    source_schema = load_table_schema(config.task_registry_path)
    schema = select_xgboost_schema(source_schema, contract=config.contract)
    feature_types = load_feature_types(config.task_registry_path)
    validate_baseline_registries(schema, feature_types, contract=config.contract)
    model_seeds = load_model_seeds(config.model_seeds_path, schema, contract=config.contract)
    model_seeds_sha256 = sha256_file(config.model_seeds_path)
    task_ids = sorted(schema)

    split_ready_path = config.splits_dir / "READY.json"
    split_manifest_path = config.splits_dir / "manifest.csv"
    if not split_ready_path.is_file() or not split_manifest_path.is_file():
        raise FileNotFoundError("The active HouseholdBench split publication is incomplete.")
    split_ready = json.loads(split_ready_path.read_text(encoding="utf-8"))
    split_manifest = pd.read_csv(split_manifest_path, dtype={"task_id": "string"})
    if not set(task_ids) <= set(split_manifest["task_id"].astype(str)):
        raise RuntimeError("Split manifest does not cover the XGBoost task set.")
    split_manifest = split_manifest.loc[
        split_manifest["task_id"].astype(str).isin(task_ids)
    ].reset_index(drop=True)

    feature_registry_sha256_value = feature_registry_sha256(feature_types)
    model_spec_sha256 = model_specification_sha256(
        feature_registry_sha256_value,
        model_seeds_sha256=model_seeds_sha256,
        contract=config.contract,
    )
    run_contract: dict[str, object] = {
        "publication_spec_version": PUBLICATION_SPEC_VERSION,
        "training_profiles": TRAINING_PROFILES,
        "split_ready_sha256": sha256_file(split_ready_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "split_config_sha256": split_ready["split_config_sha256"],
        "table_schema_sha256": sha256_json(schema),
        "task_registry_sha256": sha256_file(config.task_registry_path),
        "feature_registry_sha256": feature_registry_sha256_value,
        "model_specification_sha256": model_spec_sha256,
        "model_seeds_sha256": model_seeds_sha256,
        "baseline_spec_version": BASELINE_SPEC_VERSION,
        "threads": threads,
        "multi_output_strategy": MULTI_OUTPUT_STRATEGY,
        "runtime": runtime_identity(),
    }
    run_contract["run_id"] = sha256_json(run_contract)
    contract_path = config.output_staging / "xgboost_run_contract.json"
    if config.output_staging.exists() or config.data_staging.exists():
        if not config.output_staging.is_dir() or not config.data_staging.is_dir() or not contract_path.is_file():
            raise RuntimeError("Refusing to resume incomplete baseline staging directories.")
        if json.loads(contract_path.read_text(encoding="utf-8")) != run_contract:
            raise RuntimeError(f"Staged settings differ. Review {config.output_staging}.")
        print(f"Resuming XGBoost checkpoints under {config.output_staging}.", flush=True)
    else:
        config.output_staging.mkdir(parents=True)
        config.data_staging.mkdir(parents=True)
        _write_json_atomic(run_contract, contract_path)

    print("Running full-input XGBoost preflight before model fitting.", flush=True)
    preflight = _run_full_input_preflight(
        schema=schema, split_manifest=split_manifest, config=config
    )
    _write_frame_atomic(preflight, config.output_staging / "xgboost_preflight_audit.csv")
    print(f"Full-input XGBoost preflight passed for {len(task_ids)} tasks.", flush=True)

    quarter_audits: list[pd.DataFrame] = []
    task_manifest_rows: list[dict[str, object]] = []
    model_records: list[dict[str, object]] = []
    postprocessing_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    atomic_metric_rows: list[dict[str, object]] = []
    resource_rows: list[dict[str, object]] = []
    feature_quality_rows: list[dict[str, object]] = []
    feature_importance_rows: list[dict[str, object]] = []
    categorical_calibration_rows: list[dict[str, object]] = []
    expected_model_count = 0

    for task_index, task_id in enumerate(task_ids, start=1):
        task_clock = time.perf_counter()
        input_clock = time.perf_counter()
        print(f"[{task_index}/{len(task_ids)}] XGBoost suite: {task_id}", flush=True)
        table_path = config.tabular_dir / f"{task_id}.csv"
        sidecar_path = config.splits_dir / f"{task_id}.csv"
        prompt_path = config.prompts_dir / f"{task_id}.jsonl"
        split_row = split_manifest.loc[split_manifest["task_id"].eq(task_id)]
        if len(split_row) != 1:
            raise RuntimeError(f"{task_id}: expected one split-manifest row.")
        table_sha256 = sha256_file(table_path)
        sidecar_sha256 = sha256_file(sidecar_path)
        publication_provenance_sha256 = sha256_json(
            {"table_sha256": table_sha256, "sidecar_sha256": sidecar_sha256}
        )
        if table_sha256 != str(split_row.iloc[0]["table_sha256"]):
            raise RuntimeError(f"{task_id}: table checksum differs from split manifest.")
        if sidecar_sha256 != str(split_row.iloc[0]["sidecar_sha256"]):
            raise RuntimeError(f"{task_id}: sidecar checksum differs from split manifest.")

        sidecar = pd.read_csv(
            sidecar_path,
            dtype={
                "id": "string",
                "time": "string",
                "release_date": "string",
                "split": "string",
                "split_quarter": "string",
            },
        )
        validate_sampling_sidecar(
            sidecar,
            task_id=task_id,
            config=load_sampling_config(),
        )
        targets = schema[task_id]["targets"]
        predictors = schema[task_id]["prompt_predictors"]
        table = pd.read_csv(
            table_path,
            usecols=["id", "time", "release_date", *targets, *predictors],
            dtype={"id": "string", "time": "string", "release_date": "string"},
            low_memory=False,
        )
        table["_table_order"] = np.arange(len(table), dtype=np.int64)
        sample = table.merge(
            sidecar[
                [
                    "id",
                    "time",
                    "release_date",
                    "split",
                    "split_quarter",
                    "within_period_order",
                    "split_position",
                ]
            ],
            on=["id", "time", "release_date"],
            how="inner",
            validate="one_to_one",
        ).sort_values("_table_order", kind="mergesort").reset_index(drop=True)
        if len(sample) != len(table) or len(sample) != len(sidecar):
            raise RuntimeError(f"{task_id}: table and sidecar membership differs.")
        if any(sample[target].isna().any() for target in targets):
            raise RuntimeError(f"{task_id}: target columns contain missing values.")

        input_seconds = time.perf_counter() - input_clock
        direct_feature_names = [*predictors]
        if "prompt_month" in feature_types[task_id]["categorical"]:
            direct_feature_names.append("prompt_month")
        categorical_predictors = set(feature_types[task_id]["categorical"])
        response_text: dict[str, list[str]] = {}
        profile_counts: dict[str, dict[str, int]] = {}
        task_model_count = 0

        profile_clock = time.perf_counter()
        profile_fit_seconds = 0.0
        profile_prediction_seconds = 0.0
        profile_model_start = len(model_records)
        selected, audits = _profile_selection(
            sidecar, task_id=task_id, training_profile=training_profile
        )
        quarter_audits.extend(audits)
        sample_index = pd.MultiIndex.from_frame(sample[["id", "time"]])
        train = sample.loc[
            sample_index.isin(pd.MultiIndex.from_frame(selected["train"][["id", "time"]]))
        ].copy()
        validation = sample.loc[
            sample_index.isin(
                pd.MultiIndex.from_frame(selected["validation"][["id", "time"]])
            )
        ].copy()
        realised_counts = {"train": len(train), "validation": len(validation)}
        profile_counts[training_profile] = realised_counts
        if not len(train) or not len(validation):
            raise RuntimeError(f"{task_id} {training_profile}: empty fitting sample.")

        feature_clock = time.perf_counter()
        prepared, feature_audit = prepare_features(
            {"train": train, "validation": validation, "prediction": sample},
            task_id=task_id,
            predictors=direct_feature_names,
            categorical_predictors=categorical_predictors,
        )
        feature_seconds = time.perf_counter() - feature_clock
        for split_name, rates in feature_audit["missing_rates"].items():
            for feature_name, missing_rate in rates.items():
                feature_quality_rows.append(
                    {
                        "task_id": task_id,
                        "training_profile": training_profile,
                        "sample": split_name,
                        "feature": feature_name,
                        "feature_type": (
                            "categorical"
                            if feature_name in categorical_predictors
                            else "numeric"
                        ),
                        "missing_rate": missing_rate,
                        "unseen_count": feature_audit["unseen_counts"].get(
                            split_name, {}
                        ).get(feature_name, 0),
                        "rows": len(prepared[split_name]),
                    }
                )

        family = response_family(task_id, contract=config.contract)
        outputs: dict[str, np.ndarray] = {}
        if family == "distribution":
            for group_name, output_columns in config.contract.distribution_groups[task_id].items():
                expected_model_count += 1
                task_model_count += 1
                seed = model_seeds[(task_id, group_name, variant)]
                parameters = model_parameters(
                    variant=variant,
                    model_kind="distribution",
                    seed=seed,
                    threads=threads,
                )
                y_train = make_distribution_matrix(
                    train, output_columns, label=f"{task_id} {group_name} train"
                )
                y_validation = make_distribution_matrix(
                    validation,
                    output_columns,
                    label=f"{task_id} {group_name} validation",
                )
                fit_data_sha256 = _fit_data_sha256(
                    train_keys=train[["id", "time"]],
                    validation_keys=validation[["id", "time"]],
                    train_features=prepared["train"],
                    validation_features=prepared["validation"],
                    train_target=y_train,
                    validation_target=y_validation,
                )
                identity = sha256_json(
                    {
                        "model_specification_sha256": model_spec_sha256,
                        "runtime": run_contract["runtime"],
                        "fit_data_sha256": fit_data_sha256,
                        "task_id": task_id,
                        "output_name": group_name,
                        "variant": variant,
                        "training_profile": training_profile,
                        "caps": caps,
                        "realised_counts": realised_counts,
                        "parameters": parameters,
                    }
                )
                model_path = (
                    config.output_staging
                    / "models"
                    / "xgboost"
                    / training_profile
                    / variant
                    / task_id
                    / f"{group_name}.ubj"
                )
                metadata_path = model_path.with_suffix(".metadata.json")
                resolved = _resolve_checkpoint(
                    model_path=model_path,
                    metadata_path=metadata_path,
                    model_identity_sha256=identity,
                    output_root=config.output_root,
                    output_staging=config.output_staging,
                )
                if resolved is None:
                    model, fit_metadata = fit_distribution_group(
                        prepared["train"],
                        y_train,
                        prepared["validation"],
                        y_validation,
                        variant=variant,
                        seed=seed,
                        threads=threads,
                    )
                    profile_fit_seconds += float(fit_metadata["elapsed_seconds"])
                    roundtrip = prepared["prediction"].head(256)
                    fit_metadata = save_model_and_metadata(
                        model,
                        model_path=model_path,
                        metadata_path=metadata_path,
                        metadata={
                            **fit_metadata,
                            "model_identity_sha256": identity,
                            "fit_data_sha256": fit_data_sha256,
                            "publication_provenance_sha256": publication_provenance_sha256,
                            "task_id": task_id,
                            "output_name": group_name,
                            "output_coordinates": list(output_columns),
                            "response_family": "distribution",
                            "variant": variant,
                            "training_profile": training_profile,
                            "table_sha256": table_sha256,
                            "sidecar_sha256": sidecar_sha256,
                            "table_schema_sha256": run_contract["table_schema_sha256"],
                            "feature_registry_sha256": feature_registry_sha256_value,
                            "model_specification_sha256": model_spec_sha256,
                            "split_caps": caps,
                            "realised_split_counts": realised_counts,
                            "seed": seed,
                            "threads": threads,
                            "train_rows": len(train),
                            "validation_rows": len(validation),
                            "multi_output_strategy": MULTI_OUTPUT_STRATEGY,
                            "feature_audit": feature_audit,
                        },
                        X_roundtrip=roundtrip,
                        expected_prediction=model.predict(roundtrip),
                    )
                    origin = "fitted"
                else:
                    model, fit_metadata, origin = resolved
                _check_model_features(
                    model, fit_metadata, prepared["train"], feature_audit
                )
                prediction_clock = time.perf_counter()
                raw_prediction = np.asarray(
                    model.predict(prepared["prediction"]), dtype=float
                )
                profile_prediction_seconds += time.perf_counter() - prediction_clock
                if raw_prediction.ndim == 1:
                    raw_prediction = raw_prediction.reshape(-1, 1)
                if raw_prediction.shape != (len(sample), y_train.shape[1]):
                    raise RuntimeError(f"{task_id} {group_name}: invalid prediction shape.")
                projected, audit = project_rows_to_simplex(raw_prediction)
                postprocessing_rows.append(
                    {
                        "task_id": task_id,
                        "training_profile": training_profile,
                        "variant": variant,
                        "group_name": group_name,
                        **audit,
                    }
                )
                outputs[group_name] = projected
                record = _model_record(
                    task_id=task_id,
                    training_profile=training_profile,
                    variant=variant,
                    output_name=group_name,
                    response_family_name="distribution",
                    metadata=fit_metadata,
                    origin=origin,
                    model_path=model_path,
                    metadata_path=metadata_path,
                    output_root=config.output_root,
                    output_staging=config.output_staging,
                )
                model_records.append(record)
                _record_feature_importance(feature_importance_rows, model, record)
        else:
            for target in targets:
                expected_model_count += 1
                task_model_count += 1
                class_domain = (
                    config.contract.categorical_target_domains[task_id][target]
                    if family == "categorical"
                    else None
                )
                model_kind = "classifier" if class_domain is not None else "regressor"
                seed = model_seeds[(task_id, target, variant)]
                parameters = model_parameters(
                    variant=variant,
                    model_kind=model_kind,
                    seed=seed,
                    threads=threads,
                    class_count=None if class_domain is None else len(class_domain),
                )
                fit_data_sha256 = _fit_data_sha256(
                    train_keys=train[["id", "time"]],
                    validation_keys=validation[["id", "time"]],
                    train_features=prepared["train"],
                    validation_features=prepared["validation"],
                    train_target=train[target],
                    validation_target=validation[target],
                )
                identity = sha256_json(
                    {
                        "model_specification_sha256": model_spec_sha256,
                        "runtime": run_contract["runtime"],
                        "fit_data_sha256": fit_data_sha256,
                        "task_id": task_id,
                        "output_name": target,
                        "variant": variant,
                        "training_profile": training_profile,
                        "caps": caps,
                        "realised_counts": realised_counts,
                        "parameters": parameters,
                        "class_domain": class_domain,
                    }
                )
                model_path = (
                    config.output_staging
                    / "models"
                    / "xgboost"
                    / training_profile
                    / variant
                    / task_id
                    / f"{target}.ubj"
                )
                metadata_path = model_path.with_suffix(".metadata.json")
                resolved = _resolve_checkpoint(
                    model_path=model_path,
                    metadata_path=metadata_path,
                    model_identity_sha256=identity,
                    output_root=config.output_root,
                    output_staging=config.output_staging,
                )
                if resolved is None:
                    model, fit_metadata = fit_scalar_target(
                        prepared["train"],
                        train[target],
                        prepared["validation"],
                        validation[target],
                        variant=variant,
                        seed=seed,
                        threads=threads,
                        class_domain=class_domain,
                    )
                    profile_fit_seconds += float(fit_metadata["elapsed_seconds"])
                    roundtrip = prepared["prediction"].head(256)
                    fit_metadata = save_model_and_metadata(
                        model,
                        model_path=model_path,
                        metadata_path=metadata_path,
                        metadata={
                            **fit_metadata,
                            "model_identity_sha256": identity,
                            "fit_data_sha256": fit_data_sha256,
                            "publication_provenance_sha256": publication_provenance_sha256,
                            "task_id": task_id,
                            "output_name": target,
                            "output_coordinates": [target],
                            "response_family": family,
                            "variant": variant,
                            "training_profile": training_profile,
                            "table_sha256": table_sha256,
                            "sidecar_sha256": sidecar_sha256,
                            "table_schema_sha256": run_contract["table_schema_sha256"],
                            "feature_registry_sha256": feature_registry_sha256_value,
                            "model_specification_sha256": model_spec_sha256,
                            "split_caps": caps,
                            "realised_split_counts": realised_counts,
                            "seed": seed,
                            "threads": threads,
                            "train_rows": len(train),
                            "validation_rows": len(validation),
                            "multi_output_strategy": None,
                            "feature_audit": feature_audit,
                            "class_map": (
                                None
                                if class_domain is None
                                else {
                                    label: index for index, label in enumerate(class_domain)
                                }
                            ),
                        },
                        X_roundtrip=roundtrip,
                        expected_prediction=model.predict(roundtrip),
                    )
                    origin = "fitted"
                else:
                    model, fit_metadata, origin = resolved
                _check_model_features(
                    model, fit_metadata, prepared["train"], feature_audit
                )
                prediction_clock = time.perf_counter()
                raw_prediction = np.asarray(model.predict(prepared["prediction"]))
                profile_prediction_seconds += time.perf_counter() - prediction_clock
                if raw_prediction.shape != (len(sample),):
                    raise RuntimeError(f"{task_id} {target}: invalid prediction shape.")
                if class_domain is not None:
                    indices = raw_prediction.astype(int)
                    if ((indices < 0) | (indices >= len(class_domain))).any():
                        raise RuntimeError(f"{task_id} {target}: invalid class index.")
                    outputs[target] = np.asarray(class_domain, dtype=object)[indices]
                    _append_categorical_calibration(
                        categorical_calibration_rows,
                        task_id=task_id,
                        training_profile=training_profile,
                        variant=variant,
                        target=target,
                        sample=sample,
                        probabilities=np.asarray(
                            model.predict_proba(prepared["prediction"]), dtype=float
                        ),
                        contract=config.contract,
                    )
                else:
                    outputs[target] = raw_prediction.astype(float)
                record = _model_record(
                    task_id=task_id,
                    training_profile=training_profile,
                    variant=variant,
                    output_name=target,
                    response_family_name=family,
                    metadata=fit_metadata,
                    origin=origin,
                    model_path=model_path,
                    metadata_path=metadata_path,
                    output_root=config.output_root,
                    output_staging=config.output_staging,
                )
                model_records.append(record)
                _record_feature_importance(feature_importance_rows, model, record)

        arrays = assemble_response_arrays(
            task_id, outputs, target_order=targets, contract=config.contract
        )
        response_text[prediction_column(training_profile, variant)] = [
            canonical_json_array(array) for array in arrays
        ]
        del arrays

        resource_rows.append(
            {
                "task_id": task_id,
                "training_profile": training_profile,
                "prediction_rows": len(sample),
                "train_rows": len(train),
                "validation_rows": len(validation),
                "feature_count": len(direct_feature_names),
                "model_count": len(model_records) - profile_model_start,
                "input_seconds": input_seconds,
                "feature_seconds": feature_seconds,
                "fit_seconds_this_run": profile_fit_seconds,
                "prediction_seconds": profile_prediction_seconds,
                "total_profile_seconds": time.perf_counter() - profile_clock,
                "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            }
        )
        del prepared, outputs, train, validation
        gc.collect()

        answer_frame = read_selected_prompt_answers(
            prompt_path,
            selected_keys=set(
                zip(sample["id"].astype(str), sample["time"].astype(str), strict=True)
            ),
        )
        aligned_answers = sample[["id", "time", "split"]].merge(
            answer_frame,
            on=["id", "time"],
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if aligned_answers[["assistant", "naive_baseline"]].isna().any().any():
            raise RuntimeError(f"{task_id}: prompt answers failed to align.")

        prediction_frame = sample[
            [
                "id",
                "time",
                "release_date",
                "split",
                "split_quarter",
                "within_period_order",
                "split_position",
            ]
        ].copy()
        prediction_frame.insert(0, "task_id", task_id)
        prediction_frame["actual_answer"] = aligned_answers["assistant"].map(
            canonical_json_array
        )
        prediction_frame["naive_baseline"] = aligned_answers["naive_baseline"].map(
            canonical_json_array
        )
        for column in prediction_columns(task_id, contract=config.contract):
            prediction_frame[column] = response_text[column]
        if prediction_frame.empty or prediction_frame.isna().any().any():
            raise RuntimeError(f"{task_id}: prediction output is empty or contains missing values.")
        if prediction_frame.duplicated(["id", "time"]).any():
            raise RuntimeError(f"{task_id}: prediction output contains duplicate observation keys.")
        prediction_path = config.data_staging / f"{task_id}.csv"
        _write_frame_atomic(prediction_frame, prediction_path)

        family = response_family(task_id, contract=config.contract)
        for split_name in SPLIT_VALUES:
            positions = np.flatnonzero(sample["split"].eq(split_name).to_numpy())
            if not len(positions):
                continue
            truth = [aligned_answers.iloc[index]["assistant"] for index in positions]
            naive = [aligned_answers.iloc[index]["naive_baseline"] for index in positions]
            column = prediction_column(training_profile, variant)
            predictions = [
                json.loads(response_text[column][index]) for index in positions
            ]
            if family == "numeric":
                score = score_numeric(
                    truth, predictions, naive, validate_inputs=False
                )
                metric_name = "task_relmae"
                model_value = score[metric_name]
                naive_value = score["naive_task_relmae"]
                atomic_values = score["atomic_relmae"]
                naive_atomic = [1.0] * len(atomic_values)
                atomic_names = targets
            elif family == "categorical":
                score = score_categorical(
                    truth,
                    predictions,
                    naive,
                    category_sets=[
                        set(config.contract.categorical_target_domains[task_id][target])
                        for target in targets
                    ],
                    validate_inputs=False,
                )
                metric_name = "task_macro_f1"
                model_value = score[metric_name]
                naive_value = score["naive_task_macro_f1"]
                atomic_values = score["atomic_macro_f1"]
                naive_atomic = score["naive_atomic_macro_f1"]
                atomic_names = targets
            else:
                score = score_distribution(
                    truth,
                    predictions,
                    naive,
                    category_counts=[
                        2 if len(columns) == 1 else len(columns)
                        for columns in config.contract.distribution_groups[task_id].values()
                    ],
                    validate_inputs=False,
                    include_respondent=False,
                )
                metric_name = "task_mean_tv"
                model_value = score[metric_name]
                naive_value = score["naive_task_mean_tv"]
                atomic_values = score["atomic_mean_tv"]
                naive_atomic = score["naive_atomic_mean_tv"]
                atomic_names = list(config.contract.distribution_groups[task_id])
            metric_rows.append(
                {
                    "task_id": task_id,
                    "split": split_name,
                    "training_profile": training_profile,
                    "variant": variant,
                    "prediction_column": column,
                    "rows": len(positions),
                    "metric": metric_name,
                    "model_value": model_value,
                    "naive_value": naive_value,
                }
            )
            atomic_metric_rows.extend(
                {
                    "task_id": task_id,
                    "split": split_name,
                    "training_profile": training_profile,
                    "variant": variant,
                    "prediction_column": column,
                    "rows": len(positions),
                    "target_or_group": atomic_name,
                    "metric": metric_name.removeprefix("task_"),
                    "model_value": atomic_value,
                    "naive_value": baseline_value,
                }
                for atomic_name, atomic_value, baseline_value in zip(
                    atomic_names, atomic_values, naive_atomic, strict=True
                )
            )

        task_manifest_rows.append(
            {
                "task_id": task_id,
                "prediction_file": str(
                    (config.data_root / f"{task_id}.csv").relative_to(PROJECT_ROOT)
                ),
                "prediction_sha256": sha256_file(prediction_path),
                "rows": len(sample),
                "columns": len(prediction_frame.columns),
                "prediction_columns": json.dumps(
                    prediction_columns(task_id, contract=config.contract)
                ),
                "training_profile_counts": json.dumps(profile_counts, sort_keys=True),
                "model_count": task_model_count,
                "table_sha256": table_sha256,
                "sidecar_sha256": sidecar_sha256,
                "build_status": "complete",
            }
        )
        print(
            f"Completed {task_id}: {len(sample):,} rows, {task_model_count} models, "
            f"{time.perf_counter() - task_clock:.1f}s.",
            flush=True,
        )
        del prediction_frame, response_text, aligned_answers, answer_frame, sample, table
        gc.collect()

    if len(model_records) != expected_model_count:
        raise RuntimeError(
            f"Expected {expected_model_count} models but recorded {len(model_records)}."
        )
    _write_frame_atomic(pd.DataFrame(task_manifest_rows), config.data_staging / "manifest.csv")
    _write_frame_atomic(
        pd.concat(quarter_audits, ignore_index=True),
        config.output_staging / "xgboost_sample_allocations.csv",
    )
    model_frame = pd.DataFrame(model_records)
    # A completed publication must identify every checkpoint actually written.
    for record in model_records:
        model_path = config.output_staging / (PROJECT_ROOT / record["model_path"]).relative_to(config.output_root)
        metadata_path = config.output_staging / (PROJECT_ROOT / record["metadata_path"]).relative_to(config.output_root)
        if not model_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Incomplete model checkpoint: {model_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if sha256_file(model_path) != record["model_sha256"] or metadata["model_sha256"] != record["model_sha256"]:
            raise RuntimeError(f"Model bytes and checkpoint metadata disagree: {model_path}")
    _write_frame_atomic(model_frame, config.output_staging / "xgboost_model_manifest.csv")
    _write_frame_atomic(model_frame, config.output_staging / "xgboost_model_diagnostics.csv")
    _write_frame_atomic(
        pd.DataFrame(
            postprocessing_rows,
            columns=[
                "task_id", "training_profile", "variant", "group_name", "rows",
                "raw_coordinates_below_zero", "raw_coordinates_above_one",
                "vectors_changed", "mean_absolute_adjustment",
                "maximum_absolute_adjustment", "raw_sum_mean", "raw_sum_min",
                "raw_sum_max",
            ],
        ),
        config.output_staging / "xgboost_postprocessing_audit.csv",
    )
    _write_frame_atomic(pd.DataFrame(metric_rows), config.output_staging / "xgboost_metrics_task.csv")
    _write_frame_atomic(
        pd.DataFrame(atomic_metric_rows), config.output_staging / "xgboost_metrics_atomic.csv"
    )
    _write_frame_atomic(
        pd.DataFrame(resource_rows), config.output_staging / "xgboost_resource_summary.csv"
    )
    _write_frame_atomic(
        pd.DataFrame(feature_quality_rows), config.output_staging / "xgboost_feature_quality.csv"
    )
    _write_frame_atomic(
        pd.DataFrame(feature_importance_rows),
        config.output_staging / "xgboost_feature_importance.csv",
    )
    _write_frame_atomic(
        pd.DataFrame(categorical_calibration_rows),
        config.output_staging / "xgboost_categorical_probability_calibration.csv",
    )
    output_ready = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "READY",
        "run_id": run_contract["run_id"],
        "task_count": len(task_ids),
        "model_count": len(model_records),
        "model_manifest_sha256": sha256_file(config.output_staging / "xgboost_model_manifest.csv"),
        "table_schema_sha256": run_contract["table_schema_sha256"],
        "feature_registry_sha256": feature_registry_sha256_value,
        "model_specification_sha256": model_spec_sha256,
        "training_profiles": TRAINING_PROFILES,
        "runtime": runtime_identity(),
        "multi_output_strategy": MULTI_OUTPUT_STRATEGY,
    }
    _write_json_atomic(output_ready, config.output_staging / "READY.json")
    data_ready = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "READY",
        "run_id": run_contract["run_id"],
        "task_count": len(task_ids),
        "prediction_rows": sum(int(row["rows"]) for row in task_manifest_rows),
        "manifest_sha256": sha256_file(config.data_staging / "manifest.csv"),
        "output_ready_sha256": sha256_file(config.output_staging / "READY.json"),
    }
    _write_json_atomic(data_ready, config.data_staging / "READY.json")

    publish_staged_directory(config.output_staging, config.output_root)
    publish_staged_directory(config.data_staging, config.data_root)
    print(
        f"Published {len(model_records)} XGBoost models and "
        f"{len(task_ids)} flat baseline CSVs.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4, help="Threads per sequential fit (default: 4).")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be a positive integer")
    build_xgboost_publication(threads=args.threads)


if __name__ == "__main__":
    main()
