from __future__ import annotations

import contextlib
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def send(payload: dict[str, Any]) -> None:
    sys.__stdout__.write(json.dumps(payload, ensure_ascii=True, default=repr) + "\n")
    sys.__stdout__.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    request = json.loads(sys.stdin.readline())
    request_type = request.get("type")
    root = Path(request["root"])
    from appworld import AppWorld, __version__, load_task_ids, update_root

    update_root(str(root))
    if request_type == "inspect":
        dataset_name = str(request.get("dataset_name", "dev"))
        dataset_path = root / "data" / "datasets" / f"{dataset_name}.txt"
        data_version_path = root / "data" / "version.txt"
        send(
            {
                "type": "inspection",
                "appworld_version": __version__,
                "data_version": data_version_path.read_text(encoding="utf-8").strip(),
                "dataset_name": dataset_name,
                "dataset_hash": file_hash(dataset_path),
                "task_ids": load_task_ids(dataset_name),
            }
        )
        return 0
    if request_type == "evaluate_tasks":
        from appworld.evaluator import evaluate_tasks

        data_version = (root / "data" / "version.txt").read_text(encoding="utf-8").strip()
        with contextlib.redirect_stdout(sys.stderr):
            evaluation = evaluate_tasks(
                list(request["task_ids"]),
                experiment_name=str(request["experiment_name"]),
                suppress_errors=True,
                include_details=True,
                save_reports=True,
            )
        send(
            {
                "type": "aggregate_evaluation",
                "evaluation": evaluation,
                "appworld_version": __version__,
                "data_version": data_version,
            }
        )
        return 0
    if request_type != "initialize":
        send({"type": "error", "error": f"unknown initial request: {request_type}"})
        return 2

    task_id = str(request["task_id"])
    experiment_name = str(request["experiment_name"])
    with contextlib.redirect_stdout(sys.stderr):
        world = AppWorld(
            task_id=task_id,
            experiment_name=experiment_name,
            timeout_seconds=float(request.get("timeout_seconds", 100)),
            load_ground_truth=True,
            ground_truth_mode="minimal",
        )
    api_log_path = Path(world.output_logs_directory) / "api_calls.jsonl"
    api_cursor = 0
    send(
        {
            "type": "ready",
            "task_id": task_id,
            "instruction": world.task.instruction,
            "db_version": world.task.db_version,
            "appworld_version": __version__,
            "data_version": (root / "data" / "version.txt").read_text(encoding="utf-8").strip(),
            "output_directory": world.output_directory,
            "api_calls_path": str(api_log_path),
            "environment_io_path": str(Path(world.output_logs_directory) / "environment_io.md"),
            "dbs_directory": world.output_db_home_path_on_disk,
        }
    )
    try:
        for line in sys.stdin:
            request = json.loads(line)
            request_type = request.get("type")
            if request_type == "execute":
                with contextlib.redirect_stdout(sys.stderr):
                    output = world.execute(str(request["code"]))
                    completed = world.task_completed()
                calls = read_jsonl(api_log_path)
                new_calls = calls[api_cursor:]
                api_cursor = len(calls)
                success = not output.lstrip().startswith("Execution failed. Traceback:")
                send(
                    {
                        "type": "execution",
                        "stdout": output,
                        "success": success,
                        "completed": completed,
                        "api_calls": new_calls,
                    }
                )
            elif request_type == "evaluate":
                with contextlib.redirect_stdout(sys.stderr):
                    tracker = world.evaluate(suppress_errors=True)
                send({"type": "evaluation", "evaluation": tracker.to_dict()})
            elif request_type == "close":
                with contextlib.redirect_stdout(sys.stderr):
                    world.close()
                send({"type": "closed"})
                return 0
            else:
                send({"type": "error", "error": f"unknown request: {request_type}"})
    finally:
        with contextlib.suppress(Exception), contextlib.redirect_stdout(sys.stderr):
            world.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
