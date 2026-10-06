"""File serialization, checksums and safe directory publication."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on a sidecar `<path>.lock` file while the block runs.

    Source builders that run at the same time each read, update and rewrite one shared
    file. The lock stops a builder from reading another's half-written copy or dropping
    the rows another builder just wrote.
    """
    import fcntl

    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def publish_staged_directory(staging_dir: Path, active_dir: Path) -> None:
    """Publish one validated directory and restore the previous tree on failure."""
    if not staging_dir.is_dir():
        raise FileNotFoundError(f"Missing validated staging directory: {staging_dir}")
    backup_dir = active_dir.with_name(f"{active_dir.name}.previous.{os.getpid()}")
    if backup_dir.exists():
        raise FileExistsError(f"Refusing to overwrite sampling backup: {backup_dir}")
    moved_active = False
    try:
        if active_dir.exists():
            active_dir.replace(backup_dir)
            moved_active = True
        staging_dir.replace(active_dir)
    except BaseException:
        if moved_active and not active_dir.exists() and backup_dir.exists():
            backup_dir.replace(active_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
