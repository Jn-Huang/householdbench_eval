"""Read the task metadata shared by production and internal stage drivers."""

from pathlib import Path

from scripts.utils.responses import validate_response_structure_registry
from scripts.utils.task_registry import TASK_REGISTRY_PATH, load_task_registry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = PROJECT_ROOT / "scripts/2_tasks"
METADATA_COLUMNS = ["task_id", "dataset", "topic", "mode"]
EXPECTED_STAGE_FILES = ["01_make_table.py", "02_render_prompts.py"]


def load_tasks() -> list[tuple[Path, dict]]:
    metadata_by_id = load_task_registry(TASK_REGISTRY_PATH)
    task_dirs = sorted(
        path for path in TASK_ROOT.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    )
    if not task_dirs:
        raise RuntimeError(f"No task folders found under {TASK_ROOT}.")
    folder_ids = {path.name for path in task_dirs}
    task_ids = set(metadata_by_id)
    if task_ids != folder_ids:
        raise RuntimeError(
            "Task registry and task folders differ. "
            f"Missing folders: {sorted(task_ids - folder_ids)}. "
            f"Unregistered folders: {sorted(folder_ids - task_ids)}."
        )

    tasks = []
    for task_dir in task_dirs:
        metadata = {
            "task_id": task_dir.name,
            **{field: metadata_by_id[task_dir.name][field] for field in METADATA_COLUMNS[1:]},
        }
        for filename in EXPECTED_STAGE_FILES:
            script_path = task_dir / filename
            if not script_path.is_file():
                raise RuntimeError(f"Missing task stage script: {script_path}")

        unexpected_py_files = sorted(path.name for path in task_dir.glob("*.py") if path.name not in EXPECTED_STAGE_FILES)
        if unexpected_py_files:
            raise RuntimeError(f"{task_dir} has unexpected Python files: {unexpected_py_files}")

        tasks.append((task_dir, metadata))

    validate_response_structure_registry(task_ids)
    return tasks
