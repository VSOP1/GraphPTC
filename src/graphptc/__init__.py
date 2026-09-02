"""GraphPTC experiment package."""

from typing import Any

__all__ = ["OriginalPTCAgent"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name == "OriginalPTCAgent":
        from .agents.original_ptc import OriginalPTCAgent

        return OriginalPTCAgent
    raise AttributeError(name)
