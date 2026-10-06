"""Check the frozen raw-input inventory before production writes outputs."""

import csv
import gzip
import hashlib
import importlib
from pathlib import Path
import zipfile

from scripts.utils.io import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_MANIFEST = PROJECT_ROOT / "scripts/1_preprocessing/input_manifest.csv"
PRODUCTION_MODULES = ("numpy", "pandas", "pyarrow", "openpyxl", "xgboost")


def input_sha256(path: Path) -> str:
    """Hash data bytes; ignore gzip headers and ZIP compression/member order.

    ZIP identity includes every file's UTF-8 name, a null separator, its
    eight-byte big-endian length, and its uncompressed bytes, sorted by name.
    Names and embedded documentation remain part of the accepted release.
    """
    digest = hashlib.sha256()
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = sorted((info for info in archive.infolist() if not info.is_dir()), key=lambda info: info.filename)
            names = [info.filename for info in members]
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate ZIP member names: {path}")
            for info in members:
                digest.update(info.filename.encode("utf-8") + b"\0")
                digest.update(info.file_size.to_bytes(8, "big"))
                with archive.open(info) as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
    elif path.suffix.lower() == ".gz":
        with gzip.open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    else:
        return sha256_file(path)
    return digest.hexdigest()


def check_production_dependencies() -> None:
    for module in PRODUCTION_MODULES:
        importlib.import_module(module)


def validate_source_inputs(
    source: str | None = None,
    *,
    root: Path = PROJECT_ROOT,
    manifest_path: Path = INPUT_MANIFEST,
    overrides: dict[Path, Path] | None = None,
    verify_checksums: bool = True,
) -> int:
    """Require the supplied inventory and, by default, the reference data bytes.

    Overrides map an existing file/directory argument to its supplied location;
    they change locations, never the accepted source version. The master checks
    paths first; each source builder verifies its data before constructing it.
    No hashes persist between calls. New observations need an explicit review.
    An empty fingerprint marks a download that cannot match a stored checksum (a
    stamped file, or a source that serves only its current revision), so only its
    presence is checked.
    """
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    paths = [row["path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate paths in the input manifest.")
    for row in rows:
        fingerprint = row["sha256"]
        if fingerprint and (len(fingerprint) != 64 or any(character not in "0123456789abcdef" for character in fingerprint)):
            raise ValueError(f"Invalid input fingerprint: {row['path']}")
    rows = [row for row in rows if source is None or source in row["source"].split(";")]
    if not rows:
        raise ValueError(f"No frozen input contract for source {source!r} in {manifest_path}")
    replacements = {
        (old if old.is_absolute() else root / old): (new if new.is_absolute() else root / new)
        for old, new in (overrides or {}).items()
    }
    for row in rows:
        path = root / row["path"]
        for old, new in replacements.items():
            if path.is_relative_to(old):
                path = new / path.relative_to(old)
                break
        if not path.is_file():
            raise FileNotFoundError(f"Required {row['source']} input missing: {path}. See scripts/README.md#{row['acquisition_section']}.")
        expected = row["sha256"]
        if verify_checksums and expected:
            observed = input_sha256(path)
            if observed != expected:
                raise ValueError(f"Input fingerprint differs: {path}; expected {expected}, observed {observed}. Obtain the reference vintage; do not replace its checksum to accept changed data.")
    return len(rows)
