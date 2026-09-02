from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .episode import EpisodeGraph


ArgumentNormalizer = Callable[[Mapping[str, Any]], Mapping[str, Any]]
NoveltyKey = Callable[[Any], Any]


def _identity_arguments(arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    return arguments


def _identity_value(value: Any) -> Any:
    return value


@dataclass(frozen=True)
class ToolEffectContract:
    """Domain-neutral execution properties needed for dependency management."""

    name: str
    effect: str = "read"
    deterministic: bool = False
    cacheable: bool = False
    artifact_kind: str = "tool_result"
    normalize_arguments: ArgumentNormalizer = _identity_arguments
    normalize_artifact: NoveltyKey = _identity_value
    novelty_key: NoveltyKey = _identity_value

    def __post_init__(self) -> None:
        if self.effect not in {"pure", "read", "write"}:
            raise ValueError("tool effect must be pure, read, or write")
        if self.cacheable and (not self.deterministic or self.effect == "write"):
            raise ValueError("only deterministic non-writing tools may be cached")


@dataclass(frozen=True)
class ToolInvocation:
    tool_name: str
    action_id: str
    observation_id: str
    artifact_id: str | None
    value: Any
    success: bool
    reused: bool
    novel: bool
    state_before: str | None = None
    state_after: str | None = None


class ToolGraphRuntime:
    """Execute arbitrary tools while materializing their dependencies."""

    def __init__(self, graph: EpisodeGraph) -> None:
        self.graph = graph
        self._functions: dict[str, Callable[..., Any]] = {}
        self._contracts: dict[str, ToolEffectContract] = {}
        self._counts: dict[str, int] = {}
        self._artifact_counts: dict[str, int] = {}
        self._cache: dict[tuple[str, str], str] = {}
        self._artifact_by_value: dict[str, str] = {}
        self._state_versions: dict[str, int] = {}
        self._reuse_hits = 0

    def register(
        self,
        function: Callable[..., Any],
        contract: ToolEffectContract | None = None,
    ) -> None:
        selected = contract or ToolEffectContract(name=function.__name__)
        if selected.name in self._functions:
            raise ValueError(f"tool {selected.name!r} is already registered")
        self._functions[selected.name] = function
        self._contracts[selected.name] = selected

    def invoke(
        self,
        tool_name: str,
        *,
        graph_target: str | None = None,
        consumes: tuple[str, ...] = (),
        **arguments: Any,
    ) -> ToolInvocation:
        try:
            function = self._functions[tool_name]
            contract = self._contracts[tool_name]
        except KeyError as exc:
            raise ValueError(f"unknown graph tool {tool_name!r}") from exc
        normalized = dict(contract.normalize_arguments(arguments))
        cache_key = _canonical_arguments(normalized)
        index = self._counts.get(tool_name, 0) + 1
        self._counts[tool_name] = index
        action_id = f"tool:{tool_name}:{index}"
        self.graph.add_node(
            action_id,
            "TOOL_ACTION",
            {
                "tool": tool_name,
                "arguments": normalized,
                "effect": contract.effect,
            },
        )
        if graph_target and graph_target in self.graph.nodes:
            self.graph.add_edge("targets", action_id, graph_target)
        inferred_inputs = {
            self._artifact_by_value[key]
            for value in arguments.values()
            for key in _value_keys(value)
            if key in self._artifact_by_value
        }
        for artifact_id in dict.fromkeys((*consumes, *inferred_inputs)):
            if artifact_id in self.graph.nodes:
                self.graph.add_edge("consumes", artifact_id, action_id)

        state_before = self._state_node(tool_name) if contract.effect == "write" else None
        if state_before is not None:
            self.graph.add_edge("reads", state_before, action_id)

        cached_artifact = (
            self._cache.get((tool_name, cache_key)) if contract.cacheable else None
        )
        reused = cached_artifact is not None
        success = True
        error: Exception | None = None
        if reused:
            self._reuse_hits += 1
            value = self.graph.load_artifact(cached_artifact)
        else:
            try:
                value = function(**arguments)
            except Exception as exc:
                value = None
                success = False
                error = exc

        observation_id = f"observation:{tool_name}:{index}"
        self.graph.add_node(
            observation_id,
            "OBSERVATION",
            {"tool": tool_name, "success": success, "reused": reused},
        )
        self.graph.add_edge("observes", action_id, observation_id)

        artifact_id: str | None = cached_artifact
        novel = False
        if success and not reused:
            artifact_value = contract.normalize_artifact(value)
            artifact_index = self._artifact_counts.get(tool_name, 0) + 1
            self._artifact_counts[tool_name] = artifact_index
            artifact_id = f"artifact:{tool_name}:{artifact_index}"
            self.graph.put_artifact(
                artifact_id,
                artifact_value,
                kind=contract.artifact_kind,
                data={"operation": tool_name},
            )
            self.graph.add_edge("produces", action_id, artifact_id)
            value_key = _canonical_value(contract.novelty_key(value))
            equivalent_artifact = self._artifact_by_value.get(value_key)
            if equivalent_artifact is None:
                novel = True
            else:
                self.graph.add_edge("equivalent_to", equivalent_artifact, artifact_id)
            for key in _value_keys(artifact_value):
                self._artifact_by_value[key] = artifact_id
            if contract.cacheable:
                self._cache[(tool_name, cache_key)] = artifact_id
        elif success and artifact_id is not None:
            self.graph.add_edge("reuses", artifact_id, action_id)

        state_after = None
        if success and contract.effect == "write":
            version = self._state_versions.get(tool_name, 0) + 1
            self._state_versions[tool_name] = version
            state_after = f"state:{tool_name}:{version}"
            self.graph.add_node(state_after, "STATE_VERSION", {"tool": tool_name, "version": version})
            self.graph.add_edge("mutates", action_id, state_after)
            if state_before is not None:
                self.graph.add_edge("supersedes", state_before, state_after)

        invocation = ToolInvocation(
            tool_name=tool_name,
            action_id=action_id,
            observation_id=observation_id,
            artifact_id=artifact_id,
            value=copy.deepcopy(value),
            success=success,
            reused=reused,
            novel=novel,
            state_before=state_before,
            state_after=state_after,
        )
        if error is not None:
            self.graph.nodes[observation_id]["data"].update(
                {"error_type": type(error).__name__, "error": str(error)[:500]}
            )
            raise error
        return invocation

    @property
    def reuse_hits(self) -> int:
        return self._reuse_hits

    def _state_node(self, tool_name: str) -> str:
        version = self._state_versions.get(tool_name, 0)
        node_id = f"state:{tool_name}:{version}"
        self.graph.add_node(node_id, "STATE_VERSION", {"tool": tool_name, "version": version})
        return node_id


def _canonical_arguments(arguments: Mapping[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=repr)


def _canonical_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=repr)


def _value_keys(value: Any, *, depth: int = 0) -> set[str]:
    keys = {_canonical_value(value)}
    if depth >= 4:
        return keys
    if isinstance(value, Mapping):
        for child in value.values():
            keys.update(_value_keys(child, depth=depth + 1))
    elif isinstance(value, (list, tuple)):
        for child in value:
            keys.update(_value_keys(child, depth=depth + 1))
    if value is None or isinstance(value, bool):
        keys.discard(_canonical_value(value))
    elif isinstance(value, (int, float)) and depth > 0:
        keys.discard(_canonical_value(value))
    elif isinstance(value, str) and depth > 0 and len(value.strip()) < 2:
        keys.discard(_canonical_value(value))
    return keys
