"""JSONL loading and JSON dumping helpers."""

import json

import pytest

from household_bench_eval.io import dump_json, load_jsonl


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": "a"}\n\n{"id": "b"}\n')
    assert load_jsonl(path) == [{"id": "a"}, {"id": "b"}]


def test_load_jsonl_reports_line_number_on_invalid_json(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": "a"}\nnot json\n')
    with pytest.raises(ValueError, match=":2: invalid JSONL row"):
        load_jsonl(path)


def test_load_jsonl_rejects_non_object_rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("[1, 2, 3]\n")
    with pytest.raises(ValueError, match="expected JSON object row"):
        load_jsonl(path)


def test_dump_json_creates_parent_directories(tmp_path):
    target = tmp_path / "nested" / "out.json"
    dump_json({"b": 2, "a": 1}, target)
    assert json.loads(target.read_text()) == {"a": 1, "b": 2}
