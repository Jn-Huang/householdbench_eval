"""Run independent pipeline steps in parallel subprocesses.

Each step is a named list of commands run in order. Steps run concurrently up to a job
limit. With one job, output streams to the console as before. With more, each step writes
to its own log, which is printed when the step finishes, so the console stays grouped by
step. The first failure stops the remaining steps and exits with that step's log.
Each step runs in its own process group, so stopping a step also stops its workers.
Child Python processes run unbuffered so their logs fill in as they run.

map_in_order spreads the work inside one step over forked worker processes.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TypeVar

Step = tuple[str, Sequence[Sequence[str]]]
Item = TypeVar("Item")
Result = TypeVar("Result")


def available_cpus() -> int:
    """CPUs this process may use (respects Slurm/cgroup affinity where the OS reports it)."""
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def default_worker_count(limit: int = 8) -> int:
    """Worker processes for one step: the available CPUs, at most `limit`."""
    return min(available_cpus(), limit)


def map_in_order(
    function: Callable[[Item], Result], items: Iterable[Item], *, jobs: int
) -> Iterator[Result]:
    """Yield function(item) for each item in order, computed by up to `jobs` processes.

    The workers are forked, so they inherit the caller's module state and `function` may
    be defined in a script run as __main__; fork happens at the first item, so set up any
    state the workers read before iterating. Without fork (Windows), or with one job, the
    items run in this process. The first exception, or an interrupt, stops the remaining
    items and terminates the workers rather than waiting for the items they are running.
    """
    items = list(items)
    if jobs <= 1 or len(items) <= 1 or "fork" not in multiprocessing.get_all_start_methods():
        for item in items:
            yield function(item)
        return
    context = multiprocessing.get_context("fork")
    pool = ProcessPoolExecutor(max_workers=min(jobs, len(items)), mp_context=context)
    try:
        futures = [pool.submit(function, item) for item in items]
        for future in futures:
            yield future.result()
    except BaseException:
        # Otherwise shutdown waits for the items the workers are already running.
        for process in list((getattr(pool, "_processes", None) or {}).values()):
            process.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)


def _signal_step(process: subprocess.Popen, signal_number: int) -> None:
    """Send a signal to a step's process group: its main process and any workers left.

    Without process groups (Windows) the main process is terminated instead.
    """
    if os.name != "posix":
        if process.poll() is None:
            process.kill()
        return
    try:
        os.killpg(process.pid, signal_number)
    except (ProcessLookupError, PermissionError):
        pass  # the group has no live members left


def _exit_on_signal(signal_number: int, frame) -> None:
    raise SystemExit(128 + signal_number)


def _tail(path: Path, characters: int = 6000) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if len(text) <= characters else "...\n" + text[-characters:]


def run_steps(steps: Sequence[Step], *, jobs: int, cwd: Path, log_dir: Path) -> None:
    """Run steps with at most `jobs` at a time and stop at the first failure.

    With one job a failure raises CalledProcessError, as before; with more it prints the
    failed step's log and raises SystemExit with its exit code.
    """
    if jobs < 1:
        raise ValueError("jobs must be at least 1.")
    names = [name for name, _ in steps]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate step names: {names}")
    empty = [name for name, commands in steps if not commands]
    if empty:
        raise ValueError(f"Steps without commands: {empty}")
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if jobs == 1:
        for name, commands in steps:
            print(f"Running: {name}", flush=True)
            for command in commands:
                subprocess.run(list(command), cwd=cwd, env=env, check=True)
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(steps)
    running: dict[str, dict] = {}

    def start_next_command(name: str) -> None:
        state = running[name]
        command = state["commands"].pop(0)
        state["process"] = subprocess.Popen(
            list(command), cwd=cwd, env=env, stdout=state["log"], stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def stop_all() -> None:
        # SIGINT first so Python steps unwind and remove their temporary files; then
        # SIGKILL whatever is left in each group, such as workers of a step that exited.
        for state in running.values():
            if "process" in state:
                _signal_step(state["process"], signal.SIGINT)
        deadline = time.monotonic() + 30
        for state in running.values():
            if "process" in state:
                try:
                    state["process"].wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    pass
                _signal_step(state["process"], getattr(signal, "SIGKILL", signal.SIGTERM))
                state["process"].wait()
            state["log"].close()
        running.clear()

    # The steps have their own sessions, so a closed terminal or a kill reaches only this
    # process; turn those signals into an exit that stops the steps too.
    previous_handlers = {
        number: signal.signal(number, _exit_on_signal)
        for number in (getattr(signal, "SIGHUP", None), signal.SIGTERM) if number is not None
    }
    try:
        while pending or running:
            while pending and len(running) < jobs:
                name, commands = pending.pop(0)
                log_path = log_dir / f"{name}.log"
                running[name] = {
                    "commands": [list(command) for command in commands],
                    "log": log_path.open("w", encoding="utf-8"),
                    "log_path": log_path,
                    "started": time.monotonic(),
                }
                print(f"Started: {name}", flush=True)
                start_next_command(name)
            for name in list(running):
                state = running[name]
                code = state["process"].poll()
                if code is None:
                    continue
                if code != 0:
                    state["log"].flush()
                    print(_tail(state["log_path"]), flush=True)
                    print(f"Failed: {name} (exit {code}); full log: {state['log_path']}", flush=True)
                    if code < 0:
                        print(
                            f"{name} was killed by signal {-code}; if that was SIGKILL, the "
                            "machine may have run out of memory, so retry with a lower --jobs.",
                            flush=True,
                        )
                    raise SystemExit(code if code > 0 else 1)
                if state["commands"]:
                    start_next_command(name)
                    continue
                state["log"].close()
                elapsed = time.monotonic() - state["started"]
                print(f"===== {name} finished in {elapsed:.0f}s =====", flush=True)
                print(_tail(state["log_path"]), end="", flush=True)
                del running[name]
            time.sleep(0.2)
    except BaseException:
        stop_all()
        raise
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)


def python_step(name: str, *scripts: Path) -> Step:
    """A step that runs each script with the current interpreter, in order."""
    return name, [[sys.executable, str(script)] for script in scripts]
