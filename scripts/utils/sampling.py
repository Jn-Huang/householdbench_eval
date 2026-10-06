"""Authoritative full-universe sampling for HouseholdBench."""

from __future__ import annotations

import datetime as dt
import hashlib
import math
import numbers
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from scripts.utils.table_schema import canonical_public_metadata
from scripts.utils.task_registry import load_task_registry


# Fixed replication settings. Both quarter and row splits use these shares;
# the same integer weights govern redistribution of unused historical budgets.
CUTOFF_DATE = "2026-01-31"
SAMPLING_SEED = 42
SPLIT_WEIGHTS = {"train": 8, "validation": 1, "pre_cutoff_test": 1}
HISTORICAL_EXPORT_CAP = 500_000
MINIMUM_HOLDOUT_ROWS = 100

SPLIT_VALUES = ("train", "validation", "pre_cutoff_test", "post_cutoff_test")
HISTORICAL_SPLITS = ("train", "validation", "pre_cutoff_test")
SIDECAR_COLUMNS = (
    "id",
    "time",
    "release_date",
    "split",
    "split_quarter",
    "within_period_order",
    "split_position",
)
REGISTRY_COLUMNS = (
    "task_id",
    *SIDECAR_COLUMNS,
    "split_rows",
    "period_rows",
    "exported",
)
SUMMARY_COLUMNS = (
    "task_id",
    "full_eligible_rows",
    "pre_cutoff_eligible_rows",
    "post_cutoff_eligible_rows",
    "pre_cutoff_exported_rows",
    "post_cutoff_exported_rows",
    "exported_rows",
    "fallback_used",
)
QUARTER_PATTERN = re.compile(r"^[0-9]{4}Q[1-4]$")


def _iso_date(date_text: object, *, label: str) -> dt.date:
    if not isinstance(date_text, str) or not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}", date_text
    ):
        raise ValueError(f"{label} must be an ISO YYYY-MM-DD string: {date_text!r}")
    try:
        parsed = dt.date.fromisoformat(date_text)
    except ValueError as error:
        raise ValueError(f"{label} is not a valid calendar date: {date_text!r}") from error
    if parsed.isoformat() != date_text:
        raise ValueError(f"{label} must use canonical ISO formatting: {date_text!r}")
    return parsed


def quarter_from_date(date_text: str) -> str:
    """Return YYYYQn for one canonical observation date."""
    date = _iso_date(date_text, label="observation date")
    return f"{date.year:04d}Q{((date.month - 1) // 3) + 1}"


def stable_seed(seed: int, *parts: object) -> int:
    """Derive the seed from the first 64 SHA-256 bits of pipe-joined inputs."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    text = "|".join([str(seed), *(str(part) for part in parts)])
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def load_sampling_config() -> dict[str, Any]:
    """Combine fixed replication settings with the registry's row-split tasks."""
    registry = load_task_registry()
    total_weight = sum(SPLIT_WEIGHTS.values())
    return {
        "cutoff_date": CUTOFF_DATE,
        "seed": SAMPLING_SEED,
        "train_share": SPLIT_WEIGHTS["train"] / total_weight,
        "validation_share": SPLIT_WEIGHTS["validation"] / total_weight,
        "pre_cutoff_test_share": SPLIT_WEIGHTS["pre_cutoff_test"] / total_weight,
        "minimum_validation_fallback_rows": MINIMUM_HOLDOUT_ROWS,
        "minimum_pre_cutoff_test_fallback_rows": MINIMUM_HOLDOUT_ROWS,
        "historical_export_cap": HISTORICAL_EXPORT_CAP,
        "historical_train_budget": HISTORICAL_EXPORT_CAP * SPLIT_WEIGHTS["train"] // total_weight,
        "historical_validation_budget": HISTORICAL_EXPORT_CAP * SPLIT_WEIGHTS["validation"] // total_weight,
        "historical_pre_cutoff_test_budget": HISTORICAL_EXPORT_CAP * SPLIT_WEIGHTS["pre_cutoff_test"] // total_weight,
        "row_fallback_tasks": sorted(task_id for task_id, row in registry.items() if row["split_unit"] == "row"),
    }


def load_sampling_order() -> dict[str, Any]:
    """Project the task registry to the ordering used by both sampling paths."""
    return {"tasks": {
        task_id: {
            "order": row["sort_order"],
            "typed_id_components": [f"{row['sort_id_column']}:{row['sort_id_type']}"]
                if row["sort_id_column"] else [],
        }
        for task_id, row in load_task_registry().items()
    }}


def assign_pre_cutoff_quarters(
    quarters: list[str], *, task_id: str, seed: int
) -> dict[str, str]:
    """Assign distinct pre-cutoff quarters under the retained deterministic rule."""
    if not task_id:
        raise ValueError("task_id must be nonempty.")
    if len(quarters) != len(set(quarters)):
        raise ValueError("Pre-cutoff quarters must be distinct.")
    if any(not QUARTER_PATTERN.fullmatch(quarter) for quarter in quarters):
        raise ValueError(f"Invalid split quarter list: {quarters}")
    shuffled = sorted(quarters)
    random.Random(stable_seed(seed, task_id, "quarter_split")).shuffle(shuffled)
    n_quarters = len(shuffled)
    if n_quarters == 0:
        return {}
    n_validation = round(SPLIT_WEIGHTS["validation"] / sum(SPLIT_WEIGHTS.values()) * n_quarters)
    n_test = round(SPLIT_WEIGHTS["pre_cutoff_test"] / sum(SPLIT_WEIGHTS.values()) * n_quarters)
    if n_quarters >= 3:
        n_validation = max(1, n_validation)
        n_test = max(1, n_test)
    if n_validation + n_test >= n_quarters:
        if n_quarters < 3:
            raise ValueError("Fewer than three quarters cannot form three nonempty pools.")
        n_validation = max(1, min(n_validation, n_quarters - 2))
        n_test = max(1, min(n_test, n_quarters - n_validation - 1))
    n_train = n_quarters - n_validation - n_test
    if n_train < 1:
        raise ValueError("Quarter assignment must retain a nonempty training pool.")
    return {
        **{quarter: "train" for quarter in shuffled[:n_train]},
        **{
            quarter: "validation"
            for quarter in shuffled[n_train : n_train + n_validation]
        },
        **{
            quarter: "pre_cutoff_test"
            for quarter in shuffled[n_train + n_validation :]
        },
    }


def _task_order_specification(task_id: str, registry: dict[str, Any]) -> dict[str, Any]:
    if task_id not in registry["tasks"]:
        raise KeyError(f"{task_id}: missing explicit sampling-order entry.")
    return registry["tasks"][task_id]


def _canonical_indices(
    source: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    task_id: str,
    order_registry: dict[str, Any],
) -> list[int]:
    specification = _task_order_specification(task_id, order_registry)
    if not source.index.equals(metadata.index):
        raise ValueError(f"{task_id}: source and metadata indices differ.")
    sort_frame = pd.DataFrame(index=source.index)
    if specification["order"] == "id_time":
        sort_frame["__id"] = metadata["id"].astype("string")
        sort_frame["__time"] = metadata["time"].astype("string")
        sort_columns = ["__id", "__time"]
    else:
        sort_frame["__time"] = metadata["time"].astype("string")
        sort_columns = ["__time"]
        for position, declaration in enumerate(specification["typed_id_components"]):
            column, kind = declaration.split(":", maxsplit=1)
            if column not in source:
                raise KeyError(f"{task_id}: canonical ID component is missing: {column}")
            values = source[column]
            key = f"__component_{position}"
            if kind == "integer":
                numeric = pd.to_numeric(values, errors="coerce")
                if numeric.isna().any() or not numeric.mod(1).eq(0).all():
                    raise ValueError(f"{task_id}: {column} is not a complete integer component.")
                sort_frame[key] = numeric.astype("int64")
            else:
                strings = values.astype("string").str.strip()
                if strings.isna().any() or strings.eq("").any():
                    raise ValueError(f"{task_id}: {column} is not a complete string component.")
                sort_frame[key] = strings
            sort_columns.append(key)
        sort_frame["__id"] = metadata["id"].astype("string")
        sort_columns.append("__id")
    if sort_frame.duplicated(sort_columns).any():
        raise ValueError(f"{task_id}: canonical sort keys are not unique.")
    return sort_frame.sort_values(sort_columns, kind="mergesort").index.tolist()


def _fallback_membership(
    canonical_pre_indices: list[int], *, task_id: str, config: dict[str, Any]
) -> dict[int, str]:
    n_rows = len(canonical_pre_indices)
    validation_rows = max(
        round(config["validation_share"] * n_rows),
        config["minimum_validation_fallback_rows"],
    )
    pretest_rows = max(
        round(config["pre_cutoff_test_share"] * n_rows),
        config["minimum_pre_cutoff_test_fallback_rows"],
    )
    train_rows = n_rows - validation_rows - pretest_rows
    if min(train_rows, validation_rows, pretest_rows) < 1:
        raise ValueError(
            f"{task_id}: fallback minima cannot form three nonempty pools from {n_rows} rows."
        )
    shuffled = list(canonical_pre_indices)
    random.Random(stable_seed(config["seed"], task_id, "row_fallback")).shuffle(shuffled)
    membership = {index: "validation" for index in shuffled[:validation_rows]}
    membership.update(
        {
            index: "pre_cutoff_test"
            for index in shuffled[validation_rows : validation_rows + pretest_rows]
        }
    )
    membership.update({index: "train" for index in shuffled[validation_rows + pretest_rows :]})
    return membership


def allocate_historical_budgets(
    historical_counts: Mapping[str, int], *, config: dict[str, Any]
) -> dict[str, int]:
    """Apply the 400k/50k/50k budgets and fixed 8:1:1 redistribution."""
    if set(historical_counts) != set(HISTORICAL_SPLITS):
        raise ValueError("Historical counts must contain exactly the three historical splits.")
    capacity: dict[str, int] = {}
    for split, raw_count in historical_counts.items():
        if isinstance(raw_count, bool) or not isinstance(raw_count, numbers.Integral):
            raise TypeError(f"Historical count for {split} must be an integer.")
        count = int(raw_count)
        if count < 0:
            raise ValueError(f"Historical count for {split} must be nonnegative.")
        capacity[split] = count
    initial = {
        "train": config["historical_train_budget"],
        "validation": config["historical_validation_budget"],
        "pre_cutoff_test": config["historical_pre_cutoff_test_budget"],
    }
    allocation = {split: min(capacity[split], initial[split]) for split in HISTORICAL_SPLITS}
    target = min(sum(capacity.values()), config["historical_export_cap"])
    remaining = target - sum(allocation.values())
    weights = SPLIT_WEIGHTS
    tie_order = {split: position for position, split in enumerate(HISTORICAL_SPLITS)}
    while remaining:
        active = [split for split in HISTORICAL_SPLITS if allocation[split] < capacity[split]]
        if not active:
            raise RuntimeError("Historical budget redistribution exhausted all split capacities.")
        total_weight = sum(weights[split] for split in active)
        exact = {split: remaining * weights[split] / total_weight for split in active}
        added = 0
        for split in active:
            increment = min(
                math.floor(exact[split]), capacity[split] - allocation[split]
            )
            allocation[split] += increment
            added += increment
        remaining -= added
        if remaining == 0:
            break
        candidates = sorted(
            (
                (exact[split] - math.floor(exact[split]), -tie_order[split], split)
                for split in active
                if allocation[split] < capacity[split]
            ),
            reverse=True,
        )
        if not candidates:
            continue
        for _, _, split in candidates:
            if remaining == 0:
                break
            if allocation[split] < capacity[split]:
                allocation[split] += 1
                remaining -= 1
    if sum(allocation.values()) != target:
        raise RuntimeError("Historical budget allocation does not fill its target.")
    return allocation


def interleave_prefix(
    quarter_queues: Mapping[str, np.ndarray | list[int]], *, prefix_size: int
) -> dict[int, int]:
    """Materialize an exact prefix of the scaled-deficit split order."""
    if any(not QUARTER_PATTERN.fullmatch(quarter) for quarter in quarter_queues):
        raise ValueError("Interleaver received an invalid quarter.")
    arrays = interleave_position_arrays(
        {quarter: len(queue) for quarter, queue in quarter_queues.items() if len(queue)},
        prefix_size=prefix_size,
    )
    positions = {
        int(quarter_queues[quarter][rank]): int(position)
        for quarter, array in arrays.items()
        for rank, position in enumerate(array)
        if position > 0
    }
    if len(positions) != min(prefix_size, sum(map(len, quarter_queues.values()))):
        raise RuntimeError("Interleaver did not produce the requested prefix.")
    return positions


def interleave_position_arrays(
    period_sizes: Mapping[str, int], *, prefix_size: int
) -> dict[str, np.ndarray]:
    """Return global split positions indexed by zero-based within-period rank."""
    if isinstance(prefix_size, bool) or not isinstance(prefix_size, int) or prefix_size < 0:
        raise ValueError("prefix_size must be a nonnegative integer.")
    quarters = sorted(period_sizes)
    if any(not QUARTER_PATTERN.fullmatch(quarter) for quarter in quarters):
        raise ValueError("Interleaver received an invalid quarter.")
    sizes = np.asarray([int(period_sizes[quarter]) for quarter in quarters], dtype=np.int64)
    if np.any(sizes <= 0):
        raise ValueError("Interleaving period sizes must be positive.")
    total = int(sizes.sum())
    target = min(prefix_size, total)
    used = np.zeros(len(quarters), dtype=np.int64)
    position_arrays = {
        quarter: np.zeros(min(int(size), target), dtype=np.int64)
        for quarter, size in zip(quarters, sizes, strict=True)
    }
    for position in range(1, target + 1):
        deficits = position * sizes - used * total
        deficits[used >= sizes] = np.iinfo(np.int64).min
        largest = int(deficits.max())
        quarter_index = int(np.flatnonzero(deficits == largest)[-1])
        quarter = quarters[quarter_index]
        within_index = int(used[quarter_index])
        position_arrays[quarter][within_index] = position
        used[quarter_index] += 1
    for quarter, used_rows in zip(quarters, used, strict=True):
        array = position_arrays[quarter]
        if np.any(array[: int(used_rows)] <= 0) or np.any(array[int(used_rows) :] != 0):
            raise RuntimeError(f"Interleaving positions are invalid for {quarter}.")
    return position_arrays


def _assign_membership_and_within_period_order(
    source: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    task_id: str,
    config: dict[str, Any],
    order_registry: dict[str, Any],
) -> tuple[pd.DataFrame, bool]:
    work = metadata.copy()
    work["split_quarter"] = work["time"].map(quarter_from_date).astype("string")
    cutoff = _iso_date(config["cutoff_date"], label="cutoff_date")
    releases = work["release_date"].map(
        lambda value: _iso_date(str(value), label=f"{task_id} release_date")
    )
    pre_indices = work.index[releases.le(cutoff)].tolist()
    post_indices = work.index[releases.gt(cutoff)].tolist()
    fallback_used = task_id in set(config["row_fallback_tasks"])
    split_by_index = {index: "post_cutoff_test" for index in post_indices}
    if fallback_used:
        canonical_pre = _canonical_indices(
            source.loc[pre_indices],
            metadata.loc[pre_indices],
            task_id=task_id,
            order_registry=order_registry,
        )
        split_by_index.update(
            _fallback_membership(canonical_pre, task_id=task_id, config=config)
        )
    else:
        pre_quarters = sorted(work.loc[pre_indices, "split_quarter"].unique().tolist())
        assignment = assign_pre_cutoff_quarters(
            pre_quarters, task_id=task_id, seed=config["seed"]
        )
        split_by_index.update(
            {index: assignment[str(work.at[index, "split_quarter"])] for index in pre_indices}
        )
    if set(split_by_index) != set(work.index):
        raise RuntimeError(f"{task_id}: split assignment did not cover every eligible row.")
    work["split"] = pd.Series(split_by_index).reindex(work.index).astype("string")
    within = np.zeros(len(work), dtype=np.int64)
    for (split, quarter), group in work.groupby(
        ["split", "split_quarter"], sort=True, observed=True
    ):
        group_indices = group.index.tolist()
        canonical = _canonical_indices(
            source.loc[group_indices],
            metadata.loc[group_indices],
            task_id=task_id,
            order_registry=order_registry,
        )
        random.Random(stable_seed(config["seed"], task_id, split, quarter)).shuffle(canonical)
        within[np.asarray(canonical, dtype=np.int64)] = np.arange(1, len(canonical) + 1)
    if not np.all(within >= 1):
        raise RuntimeError(f"{task_id}: within-period order is incomplete.")
    work["within_period_order"] = within
    return work, fallback_used


def build_sampling_registry(
    source: pd.DataFrame,
    *,
    task_id: str,
    metadata: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
    order_registry: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, int]]:
    """Construct and validate the full eligible registry and export flags."""
    if source.empty:
        raise ValueError(f"{task_id}: sampling source must be nonempty.")
    source = source.reset_index(drop=True)
    if metadata is None:
        metadata = canonical_public_metadata(source, task_id=task_id)
    else:
        metadata = metadata.reset_index(drop=True).loc[:, ["id", "time", "release_date"]]
    if len(metadata) != len(source):
        raise ValueError(f"{task_id}: metadata and source row counts differ.")
    if metadata.astype("string").isna().any().any() or metadata.duplicated(["id", "time"]).any():
        raise ValueError(f"{task_id}: full-universe public metadata is invalid.")
    config = load_sampling_config() if config is None else config
    order_registry = load_sampling_order() if order_registry is None else order_registry
    work, fallback_used = _assign_membership_and_within_period_order(
        source,
        metadata,
        task_id=task_id,
        config=config,
        order_registry=order_registry,
    )
    cell_counts = (
        work.groupby(["split", "split_quarter"], observed=True, sort=True)
        .size()
        .astype(int)
    )
    work["period_rows"] = [
        int(cell_counts.loc[(split, quarter)])
        for split, quarter in zip(work["split"], work["split_quarter"], strict=True)
    ]
    split_counts = work["split"].value_counts().astype(int).to_dict()
    work["split_rows"] = work["split"].map(split_counts).astype(int)
    historical_counts = {
        split: int(split_counts.get(split, 0)) for split in HISTORICAL_SPLITS
    }
    historical_budgets = allocate_historical_budgets(historical_counts, config=config)
    prefix_by_split = {
        **historical_budgets,
        "post_cutoff_test": int(split_counts.get("post_cutoff_test", 0)),
    }
    split_position = pd.Series(pd.NA, index=work.index, dtype="Int64")
    for split in SPLIT_VALUES:
        split_rows = work.loc[work["split"].eq(split)]
        if split_rows.empty:
            continue
        queues = {
            str(quarter): group.sort_values("within_period_order", kind="mergesort")
            .index.to_numpy(dtype=np.int64)
            for quarter, group in split_rows.groupby("split_quarter", sort=True, observed=True)
        }
        positions = interleave_prefix(queues, prefix_size=prefix_by_split[split])
        if positions:
            indices = list(positions)
            split_position.loc[indices] = [positions[index] for index in indices]
    work["split_position"] = split_position
    work["exported"] = work["split"].eq("post_cutoff_test") | work["split_position"].notna()
    registry = pd.DataFrame(
        {
            "task_id": task_id,
            "id": work["id"].astype("string"),
            "time": work["time"].astype("string"),
            "release_date": work["release_date"].astype("string"),
            "split": work["split"].astype("string"),
            "split_quarter": work["split_quarter"].astype("string"),
            "within_period_order": work["within_period_order"].astype("int64"),
            "split_position": work["split_position"].astype("Int64"),
            "split_rows": work["split_rows"].astype("int64"),
            "period_rows": work["period_rows"].astype("int64"),
            "exported": work["exported"].astype(bool),
        },
        columns=REGISTRY_COLUMNS,
    )
    cutoff = config["cutoff_date"]
    historical = registry["release_date"].le(cutoff)
    post_cutoff = ~historical
    summary = {
        "task_id": task_id,
        "full_eligible_rows": len(registry),
        "pre_cutoff_eligible_rows": int(historical.sum()),
        "post_cutoff_eligible_rows": int(post_cutoff.sum()),
        "pre_cutoff_exported_rows": int((historical & registry["exported"]).sum()),
        "post_cutoff_exported_rows": int((post_cutoff & registry["exported"]).sum()),
        "exported_rows": int(registry["exported"].sum()),
        "fallback_used": fallback_used,
    }
    validate_sampling_registry(
        registry,
        task_id=task_id,
        config=config,
        historical_budgets=historical_budgets,
    )
    return registry, summary, historical_budgets


def validate_sampling_registry(
    registry: pd.DataFrame,
    *,
    task_id: str,
    config: dict[str, Any],
    historical_budgets: Mapping[str, int] | None = None,
) -> None:
    """Fail if a full registry violates any sampling or export invariant."""
    if list(registry.columns) != list(REGISTRY_COLUMNS):
        raise ValueError(f"{task_id}: invalid full sampling-registry columns.")
    if registry.empty or registry.drop(columns=["split_position"]).isna().any().any():
        raise ValueError(f"{task_id}: sampling registry must be nonempty and complete.")
    if set(registry["task_id"].astype(str)) != {task_id}:
        raise ValueError(f"{task_id}: invalid task identity in sampling registry.")
    if registry.duplicated(["id", "time"]).any():
        raise ValueError(f"{task_id}: duplicate full-universe registry keys.")
    unknown = sorted(set(registry["split"].astype(str)) - set(SPLIT_VALUES))
    if unknown:
        raise ValueError(f"{task_id}: invalid split values: {unknown}")
    cutoff = config["cutoff_date"]
    if not registry.loc[registry["release_date"].gt(cutoff), "split"].eq(
        "post_cutoff_test"
    ).all():
        raise ValueError(f"{task_id}: a post-cutoff row is outside post_cutoff_test.")
    if registry.loc[registry["release_date"].le(cutoff), "split"].eq(
        "post_cutoff_test"
    ).any():
        raise ValueError(f"{task_id}: a historical row appears in post_cutoff_test.")
    expected_quarters = registry["time"].map(quarter_from_date)
    if not registry["split_quarter"].astype(str).eq(expected_quarters).all():
        raise ValueError(f"{task_id}: split_quarter does not match time.")
    for key, group in registry.groupby(["split", "split_quarter"], observed=True):
        orders = sorted(group["within_period_order"].astype(int).tolist())
        if orders != list(range(1, len(group) + 1)):
            raise ValueError(f"{task_id}: noncontiguous within-period order for {key}.")
        if not group["period_rows"].astype(int).eq(len(group)).all():
            raise ValueError(f"{task_id}: invalid period row count for {key}.")
    for split, group in registry.groupby("split", observed=True):
        if not group["split_rows"].astype(int).eq(len(group)).all():
            raise ValueError(f"{task_id}: invalid split row count for {split}.")
        positions = sorted(group["split_position"].dropna().astype(int).tolist())
        if positions != list(range(1, len(positions) + 1)):
            raise ValueError(f"{task_id}: noncontiguous materialized split positions for {split}.")
    post = registry["release_date"].gt(cutoff)
    if not registry.loc[post, "exported"].all():
        raise ValueError(f"{task_id}: not every post-cutoff row is exported.")
    if registry.loc[~post & registry["exported"], "split_position"].isna().any():
        raise ValueError(f"{task_id}: a historical export lacks split_position.")
    expected_historical = min(int((~post).sum()), config["historical_export_cap"])
    actual_historical = int((~post & registry["exported"]).sum())
    if actual_historical != expected_historical:
        raise ValueError(
            f"{task_id}: historical exports={actual_historical:,}, expected={expected_historical:,}."
        )
    if historical_budgets is not None:
        for split in HISTORICAL_SPLITS:
            observed = int(registry.loc[registry["split"].eq(split), "exported"].sum())
            if observed != historical_budgets[split]:
                raise ValueError(f"{task_id}: export budget mismatch for {split}.")


def sampling_sidecar(registry: pd.DataFrame) -> pd.DataFrame:
    """Return the chronologically ordered public sidecar rows."""
    sidecar = registry.loc[registry["exported"], SIDECAR_COLUMNS].copy()
    if sidecar["split_position"].isna().any():
        raise ValueError("Every exported sidecar row must have a materialized split position.")
    sidecar["split_position"] = sidecar["split_position"].astype("int64")
    return sidecar.sort_values(["time", "id"], kind="mergesort").reset_index(drop=True)


def validate_task_sampling_sidecar(
    *,
    table_keys: pd.DataFrame,
    prompt_keys: pd.DataFrame,
    sidecar: pd.DataFrame,
    task_id: str,
    config: dict[str, Any],
) -> None:
    """Require exact table, prompt, and stage-01 sidecar keys and release dates."""
    expected_keys = ["id", "time", "release_date"]
    for label, frame in (("table", table_keys), ("prompt", prompt_keys)):
        if list(frame.columns) != expected_keys:
            raise ValueError(f"{task_id}: {label} keys must contain exactly {expected_keys}.")
        if frame.astype("string").isna().any().any() or frame.duplicated(["id", "time"]).any():
            raise ValueError(f"{task_id}: invalid {label} keys.")
    if list(sidecar.columns) != list(SIDECAR_COLUMNS):
        raise ValueError(f"{task_id}: invalid sidecar schema.")
    if sidecar.isna().any().any() or sidecar.duplicated(["id", "time"]).any():
        raise ValueError(f"{task_id}: invalid public sidecar keys or values.")
    cutoff = config["cutoff_date"]
    if not sidecar.loc[sidecar["release_date"].gt(cutoff), "split"].eq(
        "post_cutoff_test"
    ).all():
        raise ValueError(f"{task_id}: sidecar post-cutoff assignment is invalid.")
    table = table_keys.astype("string").sort_values(["id", "time"]).reset_index(drop=True)
    prompt = prompt_keys.astype("string").sort_values(["id", "time"]).reset_index(drop=True)
    split = (
        sidecar.loc[:, expected_keys]
        .astype("string")
        .sort_values(["id", "time"])
        .reset_index(drop=True)
    )
    if not table.equals(prompt):
        raise ValueError(f"{task_id}: table and prompt keys or release dates differ.")
    if not table.equals(split):
        raise ValueError(f"{task_id}: table and sidecar keys or release dates differ.")


def validate_sampling_sidecar(
    sidecar: pd.DataFrame, *, task_id: str, config: dict[str, Any]
) -> None:
    """Validate one exported sidecar without reading table or prompts."""
    if list(sidecar.columns) != list(SIDECAR_COLUMNS):
        raise ValueError(f"{task_id}: invalid sidecar schema.")
    if sidecar.empty or sidecar.isna().any().any():
        raise ValueError(f"{task_id}: sidecar must be nonempty and complete.")
    if sidecar.duplicated(["id", "time"]).any():
        raise ValueError(f"{task_id}: duplicate sidecar keys.")
    unknown = sorted(set(sidecar["split"].astype(str)) - set(SPLIT_VALUES))
    if unknown:
        raise ValueError(f"{task_id}: invalid split values: {unknown}")
    expected_quarters = sidecar["time"].astype(str).map(quarter_from_date)
    if not sidecar["split_quarter"].astype(str).eq(expected_quarters).all():
        raise ValueError(f"{task_id}: sidecar split quarters do not match time.")
    cutoff = config["cutoff_date"]
    if not sidecar.loc[sidecar["release_date"].astype(str).gt(cutoff), "split"].eq(
        "post_cutoff_test"
    ).all():
        raise ValueError(f"{task_id}: post-cutoff sidecar assignment is invalid.")
    for split, group in sidecar.groupby("split", observed=True):
        positions = sorted(pd.to_numeric(group["split_position"], errors="raise").astype(int))
        if positions != list(range(1, len(group) + 1)):
            raise ValueError(f"{task_id}: noncontiguous exported split positions for {split}.")


def write_sampling_candidate_partition(
    source: pd.DataFrame,
    *,
    task_id: str,
    destination: Path,
) -> int:
    """Write one compact candidate shard for a streaming stage-01 builder."""
    if source.empty:
        raise ValueError(f"{task_id}: cannot write an empty sampling-candidate partition.")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite sampling candidates: {destination}")
    metadata = canonical_public_metadata(source.reset_index(drop=True), task_id=task_id)
    order_registry = load_sampling_order()
    specification = _task_order_specification(task_id, order_registry)
    candidates = metadata.copy()
    for declaration in specification["typed_id_components"]:
        column, kind = declaration.split(":", maxsplit=1)
        if column not in source:
            raise KeyError(f"{task_id}: candidate source is missing {column}.")
        values = source.reset_index(drop=True)[column]
        if kind == "integer":
            numeric = pd.to_numeric(values, errors="coerce")
            if numeric.isna().any() or not numeric.mod(1).eq(0).all():
                raise ValueError(f"{task_id}: candidate component {column} is not integral.")
            candidates[column] = numeric.astype("int64")
        else:
            strings = values.astype("string").str.strip()
            if strings.isna().any() or strings.eq("").any():
                raise ValueError(f"{task_id}: candidate component {column} is incomplete.")
            candidates[column] = strings
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(
        destination,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    return len(candidates)


def build_partitioned_sampling_registry(
    candidate_dir: Path,
    *,
    task_id: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a full registry from compact shards without loading a large CPS pool."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    candidate_dir = Path(candidate_dir)
    candidate_paths = sorted(candidate_dir.glob("*.parquet"))
    if not candidate_paths:
        raise FileNotFoundError(f"{task_id}: no sampling candidate partitions in {candidate_dir}.")
    config = load_sampling_config()
    order_registry = load_sampling_order()
    specification = _task_order_specification(task_id, order_registry)
    candidate_columns = [
        "id",
        "time",
        "release_date",
        *(item.split(":", maxsplit=1)[0] for item in specification["typed_id_components"]),
    ]
    file_quarters: dict[Path, set[str]] = {}
    pre_quarters: set[str] = set()
    total_candidates = 0
    cutoff = config["cutoff_date"]
    for path in candidate_paths:
        frame = pd.read_parquet(path, columns=candidate_columns)
        if frame.empty:
            raise ValueError(f"{task_id}: empty candidate partition: {path}")
        if frame.duplicated(["id", "time"]).any():
            raise ValueError(f"{task_id}: duplicate key within candidate partition {path}.")
        quarters = frame["time"].astype(str).map(quarter_from_date)
        file_quarters[path] = set(quarters)
        pre_quarters.update(quarters.loc[frame["release_date"].astype(str).le(cutoff)])
        total_candidates += len(frame)

    if task_id in set(config["row_fallback_tasks"]):
        candidates = pd.concat(
            [pd.read_parquet(path, columns=candidate_columns) for path in candidate_paths],
            ignore_index=True,
        )
        registry, summary, _ = build_sampling_registry(
            candidates,
            task_id=task_id,
            metadata=candidates.loc[:, ["id", "time", "release_date"]],
            config=config,
            order_registry=order_registry,
        )
        if len(registry) != total_candidates:
            raise RuntimeError(f"{task_id}: partitioned candidate count changed during sampling.")
        write_sampling_artifacts(
            registry=registry,
            summary=summary,
            output_dir=output_dir,
        )
        return sampling_sidecar(registry), summary

    quarter_assignment = assign_pre_cutoff_quarters(
        sorted(pre_quarters), task_id=task_id, seed=config["seed"]
    )
    cell_dir = Path(output_dir) / f".sampling_cells.{os.getpid()}"
    if cell_dir.exists():
        raise FileExistsError(f"Refusing to overwrite temporary sampling cells: {cell_dir}")
    cell_dir.mkdir(parents=True)
    cell_paths: dict[tuple[str, str], Path] = {}
    cell_counts: dict[tuple[str, str], int] = {}
    try:
        all_quarters = sorted(set().union(*file_quarters.values()))
        for quarter in all_quarters:
            relevant = [path for path in candidate_paths if quarter in file_quarters[path]]
            quarter_parts = []
            for path in relevant:
                frame = pd.read_parquet(path, columns=candidate_columns)
                mask = frame["time"].astype(str).map(quarter_from_date).eq(quarter)
                if mask.any():
                    quarter_parts.append(frame.loc[mask].copy())
            quarter_frame = pd.concat(quarter_parts, ignore_index=True)
            if quarter_frame.duplicated(["id", "time"]).any():
                raise ValueError(f"{task_id}: duplicate full-universe keys in {quarter}.")
            historical_mask = quarter_frame["release_date"].astype(str).le(cutoff)
            memberships = []
            if historical_mask.any():
                memberships.append((quarter_assignment[quarter], quarter_frame.loc[historical_mask]))
            if (~historical_mask).any():
                memberships.append(("post_cutoff_test", quarter_frame.loc[~historical_mask]))
            for split, cell in memberships:
                cell = cell.reset_index(drop=True)
                metadata = cell.loc[:, ["id", "time", "release_date"]]
                canonical = _canonical_indices(
                    cell,
                    metadata,
                    task_id=task_id,
                    order_registry=order_registry,
                )
                random.Random(stable_seed(config["seed"], task_id, split, quarter)).shuffle(
                    canonical
                )
                ordered = cell.loc[canonical, ["id", "time", "release_date"]].reset_index(
                    drop=True
                )
                ordered["within_period_order"] = np.arange(1, len(ordered) + 1)
                key = (split, quarter)
                if key in cell_paths:
                    raise RuntimeError(f"{task_id}: duplicate temporary sampling cell {key}.")
                path = cell_dir / f"{split}__{quarter}.parquet"
                ordered.to_parquet(path, index=False, engine="pyarrow", compression="zstd")
                cell_paths[key] = path
                cell_counts[key] = len(ordered)

        split_counts = {
            split: sum(count for (cell_split, _), count in cell_counts.items() if cell_split == split)
            for split in SPLIT_VALUES
        }
        if sum(split_counts.values()) != total_candidates:
            raise RuntimeError(f"{task_id}: candidate rows were lost while building sampling cells.")
        historical_counts = {split: split_counts[split] for split in HISTORICAL_SPLITS}
        historical_budgets = allocate_historical_budgets(historical_counts, config=config)
        prefix_by_split = {
            **historical_budgets,
            "post_cutoff_test": split_counts["post_cutoff_test"],
        }
        position_arrays: dict[tuple[str, str], np.ndarray] = {}
        for split in SPLIT_VALUES:
            period_sizes = {
                quarter: count
                for (cell_split, quarter), count in cell_counts.items()
                if cell_split == split
            }
            if period_sizes:
                arrays = interleave_position_arrays(
                    period_sizes, prefix_size=prefix_by_split[split]
                )
                position_arrays.update(
                    {(split, quarter): array for quarter, array in arrays.items()}
                )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        registry_path = output_dir / "01_sampling_registry.parquet"
        registry_writer: pq.ParquetWriter | None = None
        exported_parts = []
        try:
            for key in sorted(cell_paths):
                split, quarter = key
                frame = pd.read_parquet(cell_paths[key])
                positions = position_arrays[key]
                materialized = int(np.count_nonzero(positions))
                split_position = pd.Series(pd.NA, index=frame.index, dtype="Int64")
                if materialized:
                    split_position.iloc[:materialized] = positions[:materialized]
                exported = (
                    pd.Series(True, index=frame.index)
                    if split == "post_cutoff_test"
                    else split_position.notna()
                )
                final = pd.DataFrame(
                    {
                        "task_id": task_id,
                        "id": frame["id"].astype("string"),
                        "time": frame["time"].astype("string"),
                        "release_date": frame["release_date"].astype("string"),
                        "split": split,
                        "split_quarter": quarter,
                        "within_period_order": frame["within_period_order"].astype("int64"),
                        "split_position": split_position,
                        "split_rows": split_counts[split],
                        "period_rows": len(frame),
                        "exported": exported.astype(bool),
                    },
                    columns=REGISTRY_COLUMNS,
                )
                table = pa.Table.from_pandas(final, preserve_index=False)
                if registry_writer is None:
                    registry_writer = pq.ParquetWriter(
                        registry_path,
                        table.schema,
                        compression="zstd",
                        use_dictionary=["task_id", "split", "split_quarter"],
                    )
                registry_writer.write_table(table)
                exported_parts.append(final.loc[final["exported"], SIDECAR_COLUMNS])
        finally:
            if registry_writer is not None:
                registry_writer.close()
        if not registry_path.is_file():
            raise RuntimeError(f"{task_id}: partitioned registry was not written.")
        sidecar = pd.concat(exported_parts, ignore_index=True)
        for column in ("id", "time", "release_date", "split", "split_quarter"):
            sidecar[column] = sidecar[column].astype("string")
        sidecar["within_period_order"] = sidecar["within_period_order"].astype("int64")
        sidecar["split_position"] = sidecar["split_position"].astype("int64")
        sidecar = sidecar.sort_values(["time", "id"], kind="mergesort").reset_index(drop=True)
        historical_eligible = sum(historical_counts.values())
        post_cutoff_eligible = split_counts["post_cutoff_test"]
        summary = {
            "task_id": task_id,
            "full_eligible_rows": total_candidates,
            "pre_cutoff_eligible_rows": historical_eligible,
            "post_cutoff_eligible_rows": post_cutoff_eligible,
            "pre_cutoff_exported_rows": sum(historical_budgets.values()),
            "post_cutoff_exported_rows": post_cutoff_eligible,
            "exported_rows": len(sidecar),
            "fallback_used": False,
        }
        if summary["pre_cutoff_exported_rows"] != min(
            historical_eligible, config["historical_export_cap"]
        ):
            raise RuntimeError(f"{task_id}: historical export count is invalid.")
        if summary["exported_rows"] != (
            summary["pre_cutoff_exported_rows"] + summary["post_cutoff_exported_rows"]
        ):
            raise RuntimeError(f"{task_id}: exported sidecar count does not reconcile.")
        pd.DataFrame([summary], columns=SUMMARY_COLUMNS).to_csv(
            output_dir / "01_sampling_summary.csv", index=False
        )
        return sidecar, summary
    finally:
        if cell_dir.exists():
            shutil.rmtree(cell_dir)


def write_sampling_artifacts(
    *,
    registry: pd.DataFrame,
    summary: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Write the full sampling registry and count summary as intermediates."""
    if list(registry.columns) != list(REGISTRY_COLUMNS):
        raise ValueError("Sampling registry has an unexpected schema.")
    if list(summary) != list(SUMMARY_COLUMNS):
        raise ValueError("Sampling summary has an unexpected schema.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "registry": output_dir / "01_sampling_registry.parquet",
        "summary": output_dir / "01_sampling_summary.csv",
    }
    registry.to_parquet(
        paths["registry"],
        index=False,
        engine="pyarrow",
        compression="zstd",
        use_dictionary=["task_id", "split", "split_quarter"],
    )
    pd.DataFrame([summary], columns=SUMMARY_COLUMNS).to_csv(paths["summary"], index=False)
    return paths


def sample_task_records(
    source: pd.DataFrame,
    *,
    task_id: str,
    output_dir: Path,
    output_sort_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build the full registry, save intermediates, and return public rows."""
    if not output_sort_cols:
        raise ValueError("output_sort_cols must contain at least one column.")
    missing = [column for column in output_sort_cols if column not in source]
    if missing:
        raise KeyError(f"{task_id}: output sort columns are missing: {missing}")
    source = source.reset_index(drop=True)
    registry, summary, _ = build_sampling_registry(source, task_id=task_id)
    write_sampling_artifacts(
        registry=registry,
        summary=summary,
        output_dir=output_dir,
    )
    selected = source.loc[registry["exported"].to_numpy()].copy()
    selected = selected.sort_values(output_sort_cols, kind="mergesort").reset_index(drop=True)
    if len(selected) != summary["exported_rows"]:
        raise RuntimeError(f"{task_id}: selected rows do not reconcile with the sampling registry.")
    return selected, registry, summary


def split_prefix(
    sidecar: pd.DataFrame, *, split: str, requested_rows: int
) -> pd.DataFrame:
    """Select a fixed downstream sample as one contiguous split-position prefix."""
    if split not in SPLIT_VALUES:
        raise ValueError(f"Unknown split: {split}")
    if isinstance(requested_rows, bool) or not isinstance(requested_rows, int) or requested_rows < 0:
        raise ValueError("requested_rows must be a nonnegative integer.")
    pool = sidecar.loc[sidecar["split"].eq(split)].copy()
    realised = min(requested_rows, len(pool))
    selected = pool.loc[pd.to_numeric(pool["split_position"], errors="raise").le(realised)]
    if len(selected) != realised:
        raise RuntimeError("split_position is not a contiguous prefix.")
    positions = sorted(selected["split_position"].astype(int).tolist())
    if positions != list(range(1, realised + 1)):
        raise RuntimeError("Selected split positions are not exactly the requested prefix.")
    return selected.reset_index(drop=True)
