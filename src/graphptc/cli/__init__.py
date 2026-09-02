"""Command-line entrypoint and parser exports."""

from .main import main
from .parser import (
    AGENT_DIFF_CONFIG,
    ALFWORLD_CONFIG,
    APIFLOW_CONFIG,
    APPWORLD_CONFIG,
    BROWSECOMP_PLUS_CONFIG,
    DEEPPLANNING_CONFIG,
    FANOUTQA_CONFIG,
    FRAMES_CONFIG,
    INTERCODE_CONFIG,
    TOOLHOP_CONFIG,
    TOOL_SANDBOX_CONFIG,
    _build_parser,
)

__all__ = ["main"]
