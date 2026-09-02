from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from graphptc.benchmarks.agent_diff.worker import Session, _http_actions, _normalize_evaluation


TASK = {
    "test_id": "box_001",
    "question": "Rename one folder.",
    "answer": {"assertions": [{"diff_type": "changed", "entity": "box_folders"}]},
    "info": {
        "service": "box",
        "seed_template": "box_default",
        "impersonate_user_id": "user",
    },
}


def test_execute_does_not_consume_official_change_journal(monkeypatch, tmp_path: Path) -> None:
    class Client:
        base_url = "http://localhost:8000"
        api_key = None

        def init_env(self, **kwargs):
            return SimpleNamespace(environmentId="env")

        def start_run(self, **kwargs):
            return SimpleNamespace(runId="run")

        def diff_run(self, **kwargs):
            raise AssertionError("per-block diff_run consumes the official journal")

        def delete_env(self, **kwargs):
            return None

    class Executor:
        def __init__(self, *args, **kwargs):
            self.workspace = SimpleNamespace(path=str(tmp_path), destroy=lambda: None)

        def execute(self, code):
            return {"status": "success", "stdout": "200\n", "stderr": "", "exit_code": 0}

    monkeypatch.setitem(
        sys.modules,
        "agent_diff",
        SimpleNamespace(AgentDiff=Client, PythonExecutorProxy=Executor),
    )
    original = Path.cwd()
    session = Session({"task": TASK, "timeout_seconds": 480})
    try:
        assert Path.cwd() == tmp_path
        result = session.execute("import requests\nrequests.put('https://api.box.com/2.0/folders/1')")
    finally:
        session.close()
    assert Path.cwd() == original

    assert result["state_effects"] == []
    assert result["external_actions"] == [
        {
            "name": "PUT https://api.box.com/2.0/folders/1",
            "arguments": {"method": "PUT", "url": "https://api.box.com/2.0/folders/1"},
            "effect": "write",
            "success": None,
            "outcome_unknown": True,
            "effect_basis": "program_execution_only",
        }
    ]


def test_official_score_dict_is_preserved_and_normalized() -> None:
    normalized = _normalize_evaluation(
        {
            "status": "failed",
            "passed": False,
            "score": {"passed": 2, "total": 3, "percent": 66.6667},
            "failures": ["assertion#2 failed"],
        },
        {},
        3,
    )

    assert normalized["score"] == 2 / 3
    assert normalized["score_details"] == {"passed": 2, "total": 3, "percent": 66.6667}
    assert normalized["satisfied_assertions"] == 2
    assert normalized["total_assertions"] == 3
    assert normalized["failures"] == ["assertion#2 failed"]


def test_successful_program_does_not_claim_http_success_or_dict_get_as_api() -> None:
    actions = _http_actions(
        "import requests\nresponse = requests.get('https://example.test')\nprint(response.json().get('ok'))",
        block_success=True,
    )
    assert len(actions) == 1
    assert actions[0]["success"] is None
    assert actions[0]["outcome_unknown"] is True
