from __future__ import annotations

import json
from pathlib import Path

from graphptc.benchmarks.intercode.runtime import InterCodeProgramRuntime


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = str(REPO_ROOT / "external" / "intercode")


def worker_command() -> tuple[str, ...]:
    worker = REPO_ROOT / "src" / "graphptc" / "benchmarks" / "intercode" / "worker.py"
    return (
        str(REPO_ROOT / "external" / "intercode" / ".venv" / "bin" / "python"),
        str(worker),
    )


def check(
    spec: dict[str, object], code: str, *, worker: tuple[str, ...]
) -> dict[str, object]:
    runtime = InterCodeProgramRuntime(
        worker_command=worker,
        root=ROOT,
        task_id=str(spec["task_id"]),
        environment=str(spec["environment"]),
        data_path=str(spec["data_path"]),
        data_index=0,
        image_name=str(spec["image_name"]),
        container_prefix=f"ic-gold-sanity-{spec['environment']}",
        max_actions=10,
        timeout_seconds=180,
    )
    try:
        execution = runtime.execute(code)
        return {
            "task_id": spec["task_id"],
            "return_code": execution.return_code,
            "evaluation": runtime.evaluate(),
        }
    finally:
        runtime.close()


def main() -> None:
    worker = worker_command()
    checks = [
        check(
            {
                "task_id": "bash:fs1:0",
                "environment": "bash",
                "data_path": "data/nl2bash/nl2bash_fs_1.json",
                "image_name": "intercode-nl2bash-fs1",
            },
            "print(bash(command=\"md5sum /testbed/*.java | awk '{print $1}' | sort | uniq -d\"))",
            worker=worker,
        ),
        check(
            {
                "task_id": "sql:0",
                "environment": "sql",
                "data_path": "data/sql/spider/ic_spider_dev.json",
                "image_name": "docker-env-sql",
            },
            'print(sql(query="SELECT T1.Name FROM people AS T1 JOIN poker_player AS T2 ON T1.People_ID = T2.People_ID ORDER BY T2.Final_Table_Made"))',
            worker=worker,
        ),
    ]
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if any(
        item["return_code"] != 0 or item["evaluation"]["max_reward"] != 1.0
        for item in checks
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
