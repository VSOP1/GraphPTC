from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from .benchmark import BenchmarkRunSummary, ProgressCallback
from .browsecomp_plus_benchmark import run_browsecomp_plus_benchmark
from .local_search import OfficialCorpusSearchTools
from .model import OpenAIChatModel
from .config import ExperimentConfig
from .observability import (
    ExecutionObserver,
    FanoutEventSink,
    InMemoryEventSink,
    JsonlEventSink,
)
from .stage6_shadow import analyze_shadow_episode
from .stage6_active import repair_active_block


def run_stage1_browsecomp_plus(
    config: ExperimentConfig,
    *,
    events_path: str | Path | None = None,
    limit: int | None = None,
    example_ids: Iterable[str] | None = None,
    resume: bool = True,
    progress: ProgressCallback | None = None,
    shadow_output_path: str | Path | None = None,
    active_repair_output_path: str | Path | None = None,
    checkpoint_archive_dir: str | Path | None = None,
) -> BenchmarkRunSummary:
    """Run the frozen few-shot PTC variant with append-only Stage 1 events."""
    if config.browsecomp_plus.prompt_variant != "fewshot-ptc-v1":
        raise ValueError("GraphPTC Stage 1 requires prompt_variant='fewshot-ptc-v1'")

    destination = Path(events_path or config.benchmark.responses_path.parent / "events.jsonl")
    if not resume:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("", encoding="utf-8")
    sink = JsonlEventSink(destination)
    if shadow_output_path is not None and active_repair_output_path is not None:
        raise ValueError("shadow and active repair modes are mutually exclusive")
    shadow_destination = Path(shadow_output_path) if shadow_output_path else None
    active_destination = (
        Path(active_repair_output_path) if active_repair_output_path else None
    )
    analysis_destination = shadow_destination or active_destination
    shadow_sink = None
    captures: dict[str, InMemoryEventSink] = {}
    capture_lock = threading.Lock()
    if analysis_destination is not None:
        if analysis_destination.resolve() in {
            destination.resolve(),
            config.benchmark.responses_path.resolve(),
        }:
            raise ValueError("repair output must be separate from events and responses")
        if not resume:
            analysis_destination.parent.mkdir(parents=True, exist_ok=True)
            analysis_destination.write_text("", encoding="utf-8")
        shadow_sink = JsonlEventSink(analysis_destination)

    def observer_factory(example_id: str, run_signature: str) -> ExecutionObserver:
        event_sink = sink
        if analysis_destination is not None:
            capture = InMemoryEventSink()
            with capture_lock:
                captures[example_id] = capture
            event_sink = FanoutEventSink(sink, capture)
        return ExecutionObserver(
            event_sink,
            episode_id=f"{run_signature}:{example_id}",
            task_id=example_id,
        )

    active_results: dict[str, dict[str, object]] = {}

    def active_repair_callback_factory(
        example_id: str,
        run_signature: str,
        tools: OfficialCorpusSearchTools,
    ):
        def callback(block_id: str, runtime):  # type: ignore[no-untyped-def]
            with capture_lock:
                capture = captures.get(example_id)
            if capture is None:
                raise ValueError("episode event capture is unavailable")
            result = repair_active_block(
                capture.events,
                block_id=block_id,
                repair_model=_LazyRepairModel(config),
                live_tools={"search": tools.search, "fetch": tools.fetch},
                runtime=runtime,
                timeout_seconds=config.runtime.code_timeout_seconds,
            )
            with capture_lock:
                active_results[example_id] = result
            return result

        return callback

    def post_episode_callback(
        example_id: str,
        run_signature: str,
        record: dict[str, object],
    ) -> None:
        assert shadow_sink is not None
        with capture_lock:
            capture = captures.pop(example_id, None)
        mode = "stage6.2-active" if active_destination is not None else "stage6.1-shadow"
        base = {
            "schema_version": 1,
            "mode": mode,
            "example_id": example_id,
            "run_signature": run_signature,
            "primary_status": record.get("status"),
            "primary_prediction": record.get("prediction"),
            "primary_record_unchanged": True,
        }
        try:
            if capture is None:
                raise ValueError("episode event capture is unavailable")
            if active_destination is not None:
                with capture_lock:
                    active = active_results.pop(
                        example_id,
                        {"status": "no_repairable_failure", "model_request_count": 0},
                    )
                shadow_sink.append(
                    {
                        **base,
                        "active": active,
                    }
                )
                return
            tools = OfficialCorpusSearchTools(
                config.browsecomp_plus.retriever_url,
                max_tool_calls=config.browsecomp_plus.max_tool_calls,
                timeout_seconds=config.browsecomp_plus.retriever_timeout_seconds,
            )
            shadow = analyze_shadow_episode(
                capture.events,
                repair_model=_LazyRepairModel(config),
                live_tools={"search": tools.search, "fetch": tools.fetch},
                timeout_seconds=config.runtime.code_timeout_seconds,
            )
            shadow_sink.append({**base, "shadow": shadow})
        except Exception as exc:
            result_key = "active" if active_destination is not None else "shadow"
            shadow_sink.append(
                {
                    **base,
                    result_key: {
                        "schema_version": 1,
                        "status": (
                            "active_repair_error"
                            if result_key == "active"
                            else "shadow_error"
                        ),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "model_request_count": 0,
                        **({"commit": None} if result_key == "shadow" else {}),
                    },
                }
            )

    return run_browsecomp_plus_benchmark(
        config,
        limit=limit,
        example_ids=example_ids,
        resume=resume,
        progress=progress,
        observer_factory=observer_factory,
        post_episode_callback=(
            post_episode_callback if shadow_sink is not None else None
        ),
        active_repair_callback_factory=(
            active_repair_callback_factory
            if active_destination is not None
            else None
        ),
        checkpoint_archive_dir=(
            Path(checkpoint_archive_dir) if checkpoint_archive_dir is not None else None
        ),
    )


class _LazyRepairModel:
    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config

    def create_turn(self, **kwargs: object):  # type: ignore[no-untyped-def]
        api_key = self._config.require_api_key(self._config.model.api_key_env)
        model = OpenAIChatModel(
            replace(self._config.model, max_retries=0),
            api_key,
        )
        return model.create_turn(**kwargs)
