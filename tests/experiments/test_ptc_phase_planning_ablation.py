from __future__ import annotations

from scripts.experiments.ablate_ptc_phase_planning import (
    PHASE_PLANNING_SUFFIX,
    VARIANTS,
    _system_prompt,
    summarize,
)


def test_ablation_is_strict_two_by_two() -> None:
    assert {
        (variant.thinking, variant.phase_planning) for variant in VARIANTS
    } == {
        ("disabled", False),
        ("enabled", False),
        ("disabled", True),
        ("enabled", True),
    }
    assert _system_prompt(phase_planning=False) not in PHASE_PLANNING_SUFFIX
    prompt = _system_prompt(phase_planning=True)
    assert "stage_goal:" in prompt
    assert "parallel_subgoals:" in prompt
    assert "return_condition:" in prompt
    assert "same assistant response" in prompt
    assert "step-by-step reasoning" in prompt
    assert "at least" not in PHASE_PLANNING_SUFFIX


def test_summary_separates_first_block_from_later_observation_effects() -> None:
    records = [
        _record(
            blocks=[
                _block(4, loop=True, filter_=True, aggregation=True),
                _block(1),
            ],
            calls=[
                _call("search", query="alpha"),
                _call("search", query="beta"),
                _call("fetch", docid="doc-1"),
                _call("fetch", docid="doc-1"),
                _call("search", query="alpha"),
            ],
            turns=3,
            phase_planning=True,
        ),
        _record(
            blocks=[_block(1)],
            calls=[_call("search", query="gamma")],
            turns=2,
            phase_planning=True,
        ),
    ]

    summary = summarize(records)

    assert summary["first_block"]["multi_call_rate"] == 0.5
    assert summary["first_block"]["loop_rate"] == 0.5
    assert summary["first_block"]["tool_loop_rate"] == 0.5
    assert summary["first_block"]["filter_rate"] == 0.5
    assert summary["first_block"]["aggregation_rate"] == 0.5
    assert summary["first_block"]["coherent_program_rate"] == 0.5
    assert summary["first_block"]["calls_mean"] == 2.5
    assert summary["first_block"]["repeat_search_rate"] == 0.0
    assert summary["first_block"]["repeat_fetch_rate"] == 0.5
    assert summary["first_block"]["repeat_retrieval_rate"] == 0.2
    assert summary["all_blocks"]["repeat_search_rate"] == 0.25
    assert summary["all_blocks"]["repeat_retrieval_rate"] == 2 / 6
    assert summary["turns"]["mean"] == 2.5
    assert summary["phase_plan"]["same_response_compliance_rate"] == 1.0


def _record(
    *,
    blocks: list[dict],
    calls: list[dict],
    turns: int,
    phase_planning: bool,
) -> dict:
    return {
        "phase_planning": phase_planning,
        "correct": True,
        "agent": {
            "status": "success",
            "blocks": blocks,
            "search_calls": calls,
            "model_requests": turns,
        },
        "model_turns": [
            {
                "text": (
                    "<phase_plan>\nstage_goal: x\nparallel_subgoals: y\n"
                    "return_condition: z\n</phase_plan>"
                ),
                "tool_calls": 1,
            }
        ],
    }


def _block(
    calls: int,
    *,
    loop: bool = False,
    filter_: bool = False,
    aggregation: bool = False,
) -> dict:
    return {
        "runtime_calls": calls,
        "program_analysis": {
            "has_loop": loop,
            "tool_calls_in_loops": int(loop),
            "has_filter": filter_,
            "has_aggregation": aggregation,
        },
    }


def _call(
    operation: str, *, query: str | None = None, docid: str | None = None
) -> dict:
    return {"operation": operation, "query": query, "docid": docid}
