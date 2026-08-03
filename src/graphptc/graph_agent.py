from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .config import RuntimeConfig
from .observability import (
    EventSink,
    ExecutionEvent,
    GraphPTCRuntime,
    ProgramAnalysis,
    ProgramAnalyzer,
)
from .ptc import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    AgentResult,
    MessagesModel,
    OriginalPTCAgent,
    PTCBlockTrace,
)
from .search import TavilySearchTools


@dataclass(frozen=True)
class GraphPTCResult:
    episode_id: str
    agent: AgentResult
    events: tuple[ExecutionEvent, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "agent": self.agent.to_dict(),
            "events": [event.to_dict() for event in self.events],
        }


class GraphPTCAgent:
    """Stage 1 adapter that observes Original PTC without changing its policy."""

    def __init__(
        self,
        model: MessagesModel,
        search_tools: TavilySearchTools,
        runtime: RuntimeConfig,
        *,
        system_prompt: str = SYSTEM_PROMPT,
        user_prompt_template: str = USER_PROMPT_TEMPLATE,
        runtime_functions: Iterable[Callable[..., Any]] | None = None,
        event_sink: EventSink | None = None,
        analyzer: ProgramAnalyzer | None = None,
    ) -> None:
        functions = tuple(runtime_functions or _default_functions(search_tools))
        graph_runtime = GraphPTCRuntime(event_sink)
        self._runtime = graph_runtime
        self._delegate = _ObservableOriginalPTCAgent(
            model=model,
            search_tools=search_tools,
            runtime=runtime,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            runtime_functions=functions,
            graph_runtime=graph_runtime,
            analyzer=analyzer or ProgramAnalyzer(),
            tool_names={function.__name__ for function in functions},
        )

    def run(self, task: str) -> GraphPTCResult:
        first_event = len(self._runtime.events)
        episode_id = self._runtime.start_episode(task)
        result = self._delegate.run(task)
        self._runtime.finish_episode(
            status=result.status,
            error=result.error,
            duration_ms=result.duration_ms,
        )
        return GraphPTCResult(
            episode_id=episode_id,
            agent=result,
            events=self._runtime.events[first_event:],
        )


class _ObservableOriginalPTCAgent(OriginalPTCAgent):
    def __init__(
        self,
        *args: Any,
        graph_runtime: GraphPTCRuntime,
        analyzer: ProgramAnalyzer,
        tool_names: set[str],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._graph_runtime = graph_runtime
        self._analyzer = analyzer
        self._tool_names = tool_names
        self._execution_log = self._registry.enable_logging(max_entries=10_000)

    def _execute_block(
        self, turn_number: int, code: Any
    ) -> tuple[str, bool, PTCBlockTrace]:
        source = code if isinstance(code, str) else repr(code)
        block_id, started = self._graph_runtime.start_block(turn_number, source)
        analysis = self._analyzer.analyze(source, self._tool_names)
        self._graph_runtime.record_analysis(block_id, started.event_id, analysis)
        output, is_error, trace = super()._execute_block(turn_number, code)
        self._record_tool_events(block_id, started.event_id, trace, analysis)
        self._graph_runtime.finish_block(
            block_id=block_id,
            block_event_id=started.event_id,
            success=trace.success,
            duration_ms=trace.duration_ms,
            stdout=trace.stdout,
            invocation_id=trace.invocation_id,
        )
        return output, is_error, trace

    def _record_tool_events(
        self,
        block_id: str,
        block_event_id: str,
        trace: PTCBlockTrace,
        analysis: ProgramAnalysis,
    ) -> None:
        if trace.invocation_id is None:
            return
        entries = self._execution_log.get_entries(
            invocation_id=trace.invocation_id
        )
        for entry in reversed(entries):
            self._graph_runtime.record_tool_entry(
                block_id=block_id,
                block_event_id=block_event_id,
                entry=entry,
                analysis=analysis,
            )


def _default_functions(
    tools: TavilySearchTools,
) -> tuple[Callable[..., Any], ...]:
    return (
        tools.search_web,
        tools.search_web_batch,
        tools.fetch_url,
        tools.fetch_urls,
    )
