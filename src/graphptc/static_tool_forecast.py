from __future__ import annotations

import ast
from typing import Any


def forecast_tool_calls(code: str) -> dict[str, Any]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _result(False, 0, 0, 0, 0)
    lengths: dict[str, int] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            length = _known_length(statement.value, lengths)
            if isinstance(target, ast.Name) and length is not None:
                lengths[target.id] = length
    counts = [0, 0, 0, 0]
    for statement in tree.body:
        _visit(statement, multiplier=1, lengths=lengths, counts=counts)
    return _result(True, *counts)


def _visit(
    node: ast.AST,
    *,
    multiplier: int | None,
    lengths: dict[str, int],
    counts: list[int],
) -> None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and _tool_name(child) in {"search", "fetch"}:
                counts[2 if _tool_name(child) == "search" else 3] += 1
        return
    if isinstance(node, (ast.For, ast.AsyncFor)):
        _visit(node.iter, multiplier=multiplier, lengths=lengths, counts=counts)
        iterations = _known_length(node.iter, lengths)
        nested_multiplier = (
            None if multiplier is None or iterations is None else multiplier * iterations
        )
        for child in node.body:
            _visit(child, multiplier=nested_multiplier, lengths=lengths, counts=counts)
        for child in node.orelse:
            _visit(child, multiplier=multiplier, lengths=lengths, counts=counts)
        return
    if isinstance(node, ast.Call):
        tool = _tool_name(node)
        if tool in {"search", "fetch"}:
            if multiplier is None:
                counts[2 if tool == "search" else 3] += 1
            else:
                counts[0 if tool == "search" else 1] += multiplier
        for child in [*node.args, *(keyword.value for keyword in node.keywords)]:
            _visit(child, multiplier=multiplier, lengths=lengths, counts=counts)
        return
    for child in ast.iter_child_nodes(node):
        _visit(child, multiplier=multiplier, lengths=lengths, counts=counts)


def _known_length(node: ast.AST, lengths: dict[str, int]) -> int | None:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return len(node.elts)
    if isinstance(node, ast.Name):
        return lengths.get(node.id)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "range" and all(
            isinstance(arg, ast.Constant) and isinstance(arg.value, int)
            for arg in node.args
        ):
            values = [int(arg.value) for arg in node.args]
            return len(range(*values))
    return None


def _tool_name(node: ast.Call) -> str | None:
    return node.func.id if isinstance(node.func, ast.Name) else None


def _result(
    syntax_valid: bool,
    search_calls: int,
    fetch_calls: int,
    unknown_search_sites: int,
    unknown_fetch_sites: int,
) -> dict[str, Any]:
    return {
        "syntax_valid": syntax_valid,
        "known_search_calls": search_calls,
        "known_fetch_calls": fetch_calls,
        "unknown_search_sites": unknown_search_sites,
        "unknown_fetch_sites": unknown_fetch_sites,
        "fully_determined": syntax_valid
        and unknown_search_sites == 0
        and unknown_fetch_sites == 0,
    }
