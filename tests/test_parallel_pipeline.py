"""Parallel step runner and the source-builder waves in run_pipeline.py."""

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.utils.parallel import run_steps  # noqa: E402


def _python(code: str) -> list[str]:
    return [sys.executable, "-c", code]


@pytest.mark.parametrize("jobs", [1, 3])
def test_steps_run_their_commands_in_order(tmp_path, jobs):
    steps = [
        (f"step{i}", [_python(f"open('{tmp_path}/s{i}', 'a').write('a')"),
                      _python(f"open('{tmp_path}/s{i}', 'a').write('b')")])
        for i in range(5)
    ]
    run_steps(steps, jobs=jobs, cwd=tmp_path, log_dir=tmp_path / "logs")
    assert [(tmp_path / f"s{i}").read_text() for i in range(5)] == ["ab"] * 5


def test_parallel_steps_overlap(tmp_path):
    # Each step waits for the other's marker, so the run only finishes if both run at once.
    wait = "import os,time\nfor _ in range(200):\n    if os.path.exists('{}'): break\n    time.sleep(0.05)\nelse: raise SystemExit(3)"
    steps = [
        ("a", [_python(f"open('{tmp_path}/a', 'w')"), _python(wait.format(tmp_path / "b"))]),
        ("b", [_python(f"open('{tmp_path}/b', 'w')"), _python(wait.format(tmp_path / "a"))]),
    ]
    run_steps(steps, jobs=2, cwd=tmp_path, log_dir=tmp_path / "logs")


def test_failure_stops_the_run_and_keeps_its_log(tmp_path, capsys):
    steps = [
        ("bad", [_python("print('boom'); raise SystemExit(7)")]),
        ("slow", [_python("import time; time.sleep(30)")]),
        ("never", [_python(f"open('{tmp_path}/never', 'w')")]),
    ]
    with pytest.raises(SystemExit) as error:
        run_steps(steps, jobs=2, cwd=tmp_path, log_dir=tmp_path / "logs")
    assert error.value.code == 7
    assert "boom" in (tmp_path / "logs" / "bad.log").read_text()
    assert "boom" in capsys.readouterr().out
    assert not (tmp_path / "never").exists()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_failure_also_stops_the_workers_of_the_other_steps(tmp_path):
    import time

    # The worker ignores SIGINT and outlives its step's main process, as a pool worker can.
    worker = "import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(60)"
    spawner = (
        "import subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', {worker!r}])\n"
        f"open('{tmp_path}/pid', 'w').write(str(child.pid))\n"
        "time.sleep(60)"
    )
    wait_then_fail = (
        f"import os, time\nwhile not os.path.exists('{tmp_path}/pid'): time.sleep(0.05)\n"
        "time.sleep(0.5)\nraise SystemExit(7)"
    )
    steps = [("spawner", [_python(spawner)]), ("bad", [_python(wait_then_fail)])]
    started = time.monotonic()
    with pytest.raises(SystemExit):
        run_steps(steps, jobs=2, cwd=tmp_path, log_dir=tmp_path / "logs")
    assert time.monotonic() - started < 20
    assert _process_is_gone(int((tmp_path / "pid").read_text()))


def test_rejects_duplicate_names_and_bad_jobs(tmp_path):
    with pytest.raises(ValueError):
        run_steps([("x", []), ("x", [])], jobs=2, cwd=tmp_path, log_dir=tmp_path)
    with pytest.raises(ValueError):
        run_steps([], jobs=0, cwd=tmp_path, log_dir=tmp_path)
    for jobs in (1, 2):
        with pytest.raises(ValueError):
            run_steps([("empty", [])], jobs=jobs, cwd=tmp_path, log_dir=tmp_path)


def test_shared_file_updates_under_the_lock_keep_every_row(tmp_path):
    # Six builders rewrite one coverage file at once; with the lock none of their rows is lost.
    target = tmp_path / "coverage.csv"
    update = (
        "import sys, time\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from pathlib import Path\n"
        "from scripts.utils.io import exclusive_lock\n"
        f"path = Path({str(target)!r})\n"
        "with exclusive_lock(path):\n"
        "    rows = path.read_text().splitlines() if path.exists() else []\n"
        "    time.sleep(0.2)\n"
        "    path.write_text('\\n'.join(sorted(rows + [sys.argv[1]])) + '\\n')\n"
    )
    steps = [(f"b{i}", [_python(update) + [f"row{i}"]]) for i in range(6)]
    run_steps(steps, jobs=6, cwd=tmp_path, log_dir=tmp_path / "logs")
    assert target.read_text().splitlines() == [f"row{i}" for i in range(6)]


def test_cps_row_blocks_parse_to_the_same_chunks_as_chunked_read_csv(tmp_path):
    import gzip
    import importlib.util
    import io

    import pandas as pd

    spec = importlib.util.spec_from_file_location("cps_build", ROOT / "scripts/1_preprocessing/cps.py")
    cps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cps)
    path = tmp_path / "extract.csv.gz"
    lines = ['"YEAR","MONTH","WAGE","CODE"'] + [
        f"{1990 + i % 30},{1 + i % 12},{'' if i % 7 == 0 else round(i * 1.37, 2)},"
        + (f'"A{i % 5:02d} "' if i > 600 else f"{i % 5:02d}")
        for i in range(1234)
    ]
    path.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode()))
    expected = list(pd.read_csv(path, low_memory=False, chunksize=100, dtype={"CODE": "string"}))
    blocks = list(cps.iter_raw_row_blocks(path, 100, block_size=777))
    assert len(blocks) == len(expected) == 13
    for (header, rows), frame in zip(blocks, expected, strict=True):
        parsed = pd.read_csv(io.BytesIO(header + rows), low_memory=False, dtype={"CODE": "string"})
        parsed.index = frame.index
        pd.testing.assert_frame_equal(parsed, frame, check_exact=True)
    assert cps._one_row_per_line(b'1,"A61 "\n2,"B"\n')
    for bad in (b'1,"A\n61"\n', b"1,2\r\n", b"1,2\n\n3,4\n", b"\n1,2\n", b"1,2\n   \n3,4\n", b"\t\n1,2\n"):
        assert not cps._one_row_per_line(bad)


def _square_unless_seven(value: int) -> int:
    if value == 7:
        raise ValueError("seven")
    return value * value


@pytest.mark.parametrize("jobs", [1, 3])
def test_map_in_order_keeps_the_input_order_and_raises_the_first_error(jobs):
    from scripts.utils.parallel import map_in_order

    assert list(map_in_order(_square_unless_seven, range(7), jobs=jobs)) == [i * i for i in range(7)]
    results = map_in_order(_square_unless_seven, range(10), jobs=jobs)
    assert [next(results) for _ in range(7)] == [i * i for i in range(7)]
    with pytest.raises(ValueError, match="seven"):
        next(results)


def _fail_fast_or_sleep(value: int) -> int:
    import time

    if value == 0:
        time.sleep(0.5)
        raise ValueError("first")
    time.sleep(60)
    return value


def test_map_in_order_terminates_the_running_items_on_an_error():
    import time

    from scripts.utils.parallel import map_in_order

    started = time.monotonic()
    with pytest.raises(ValueError, match="first"):
        list(map_in_order(_fail_fast_or_sleep, range(4), jobs=3))
    assert time.monotonic() - started < 20


def _process_is_gone(pid: int) -> bool:
    import time

    status = Path(f"/proc/{pid}/stat")
    for _ in range(100):
        if not status.exists() or status.read_text().split(") ")[1].startswith(("Z", "X")):
            return True
        time.sleep(0.05)
    return False


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_terminating_the_runner_stops_its_steps(tmp_path):
    import signal
    import subprocess
    import time

    step = f"import os, time\nopen('{tmp_path}/pid', 'w').write(str(os.getpid()))\ntime.sleep(60)"
    runner = (
        f"import sys\nsys.path.insert(0, {str(ROOT)!r})\nfrom pathlib import Path\n"
        "from scripts.utils.parallel import run_steps\n"
        f"run_steps([('a', [[sys.executable, '-c', {step!r}]]), ('b', [[sys.executable, '-c', 'import time; time.sleep(60)']])],"
        f" jobs=2, cwd=Path({str(tmp_path)!r}), log_dir=Path({str(tmp_path / 'logs')!r}))\n"
    )
    process = subprocess.Popen([sys.executable, "-c", runner])
    for _ in range(200):
        if (tmp_path / "pid").exists() and (tmp_path / "pid").read_text():
            break
        time.sleep(0.05)
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=40) == 128 + signal.SIGTERM
    assert _process_is_gone(int((tmp_path / "pid").read_text()))


def test_cps_month_blocks_match_the_full_file_month_iterators(tmp_path):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from scripts.utils import cps

    # Months straddle row groups; the label column's dictionary and the integer
    # column's missing values differ by row group, so dtypes depend on the split.
    path = tmp_path / "basic.parquet"
    rows = [(1990 + (i // 37) // 12, 1 + (i // 37) % 12, i) for i in range(500)]
    schema = pa.schema([
        ("YEAR", pa.int64()), ("MONTH", pa.int64()), ("AGE", pa.int64()),
        ("label", pa.dictionary(pa.int32(), pa.string())),
    ])
    with pq.ParquetWriter(path, schema) as writer:
        for start in range(0, len(rows), 60):
            group = rows[start:start + 60]
            group_number = start // 60
            writer.write_table(pa.table({
                "YEAR": pa.array([None if i == 61 else y for y, _, i in group], pa.int64()),
                "MONTH": [m for _, m, _ in group],
                "AGE": pa.array([None if group_number % 3 == 0 and i % 5 == 0 else i for *_, i in group], pa.int64()),
                "label": pa.array([f"g{group_number % 4}-{i % 3}" for *_, i in group]).dictionary_encode(),
            }, schema=schema))
    columns = ["YEAR", "MONTH", "AGE", "label"]
    groups = cps.month_row_groups(path)
    assert any(len(row_groups) > 1 for row_groups in groups.values())
    for reverse, full_scan in ((False, cps.iter_month_dataframes), (True, cps.iter_month_dataframes_reverse)):
        expected = list(full_scan(path, columns))
        months = sorted(groups, reverse=reverse)
        assert [month_idx for month_idx, _ in expected] == months
        blocks = cps.month_blocks(months, size=4)
        got = [item for block in blocks for item in cps.iter_month_block(path, columns, block, groups, reverse=reverse)]
        for (month_a, frame_a), (month_b, frame_b) in zip(got, expected, strict=True):
            assert month_a == month_b
            pd.testing.assert_frame_equal(frame_a, frame_b, check_exact=True)


def test_prompt_prefilters_never_skip_a_forbidden_match():
    import random
    import re

    from scripts.utils import prompt_rendering

    def reference_validate(text: str) -> str | None:
        note_free = prompt_rendering.CURRENT_MONTH_NOTE_PATTERN.sub("", text)
        for pattern, reason in prompt_rendering.FORBIDDEN_MODEL_TEXT_PATTERNS:
            if pattern.search(note_free if reason == "month-name reference" else text):
                return reason
        return None

    def validate(text: str) -> str | None:
        try:
            prompt_rendering.validate_model_facing_text(text, field="user")
        except ValueError as error:
            return re.match(r"Forbidden (.*) in user prompt", str(error)).group(1)
        return None

    assert all(prefilter is not None for prefilter in prompt_rendering.FORBIDDEN_TEXT_PREFILTERS)
    phrases = [
        "survey", "Surveyed", "interviewed RESPONDENT", "respondents", "panel month", "Panelist",
        "BenchMark", "parker-style", "CPS", "cps", "Census", "PSIDs", "Michigan Survey of Consumers",
        "survey of consumer expectations", "Current Population Survey", "Consumer Expenditures Survey",
        "panel study of income dynamics", "Economic Stimulus Payment", "Federal Income-Tax Rebate",
        "2008-01", "2008-01-15", "1990 to 1995", "2001-02", "1999 through 2003", "In 2008",
        "since 1994", "year 2020", "January", "january", "May", "qoq", "QoQ", "Q-1", "q-1", "Unknown",
        "not reported", "Missing", "day(s)", "Job(s)", "1 people", "1 Children", "You say",
        "reported reason", "retrospective report", "Numeracy classification", "work-status indicators",
    ]
    base = (
        "You are 45 years old and live with 2 other people. Here is your current situation. "
        "Note that the current month is March, and that $ denotes amounts in U.S. dollars. "
        "You work full-time.\n\n"
        "Over the next year, what will happen? Answer with one option."
    )
    rng = random.Random(0)
    cases = [base, base.replace("month is March", "month is in March"), "émigré " + base]
    for phrase in phrases:
        for _ in range(6):
            cut = rng.randrange(len(base) + 1)
            left, right = rng.choice(["", " ", "x", "\n", "("]), rng.choice(["", " ", "x", ".", "\n"])
            cases.append(base[:cut] + left + phrase + right + base[cut:])
    outcomes = [reference_validate(text) for text in cases]
    assert outcomes[0] is None and sum(outcome is not None for outcome in outcomes) > len(phrases)
    assert [validate(text) for text in cases] == outcomes


def test_source_waves_cover_every_builder_once_and_respect_dependencies():
    tree = ast.parse((ROOT / "scripts/run_pipeline.py").read_text())
    waves = next(
        ast.literal_eval(node.value) for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "source_waves"
    )
    flat = [source for wave in waves for source in wave]
    builders = {path.stem for path in (ROOT / "scripts/1_preprocessing").glob("*.py")}
    assert len(flat) == len(set(flat))
    assert set(flat) == builders
    position = {source: index for index, wave in enumerate(waves) for source in wave}
    for upstream, downstream in [("cpi", "macro_context"), ("bea_sainc", "state_macro_panel"),
                                 ("bls_laus", "state_macro_panel"), ("fhfa_hpi", "state_macro_panel")]:
        assert position[upstream] < position[downstream]
    for survey in ["cex", "cex_diary", "cps", "cps_ui_policy", "census", "psid", "mich", "sce", "sce_financing"]:
        assert position[survey] > max(position["macro_context"], position["state_macro_panel"])
