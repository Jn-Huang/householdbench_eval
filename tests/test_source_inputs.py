"""Input manifest checks in scripts/utils/source_inputs.py."""

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.utils.source_inputs import validate_source_inputs  # noqa: E402

HEADER = "source,path,source_version,acquisition_section,kind,role,sha256\n"


def _manifest(tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
    path = tmp_path / "manifest.csv"
    path.write_text(HEADER + "".join(f"s,{name},v,section,download,data,{sha}\n" for name, sha in rows))
    return path


def test_an_empty_checksum_checks_presence_only(tmp_path):
    (tmp_path / "fixed.csv").write_text("a\n")
    (tmp_path / "stamped.csv").write_text("downloaded at any time\n")
    fixed_sha = hashlib.sha256(b"a\n").hexdigest()
    manifest = _manifest(tmp_path, [("fixed.csv", fixed_sha), ("stamped.csv", "")])
    assert validate_source_inputs(root=tmp_path, manifest_path=manifest) == 2

    (tmp_path / "fixed.csv").write_text("b\n")
    with pytest.raises(ValueError, match="fingerprint differs"):
        validate_source_inputs(root=tmp_path, manifest_path=manifest)

    (tmp_path / "fixed.csv").write_text("a\n")
    (tmp_path / "stamped.csv").unlink()
    with pytest.raises(FileNotFoundError):
        validate_source_inputs(root=tmp_path, manifest_path=manifest)


def test_a_malformed_checksum_is_rejected(tmp_path):
    (tmp_path / "fixed.csv").write_text("a\n")
    with pytest.raises(ValueError, match="Invalid input fingerprint"):
        validate_source_inputs(root=tmp_path, manifest_path=_manifest(tmp_path, [("fixed.csv", "abc")]))
