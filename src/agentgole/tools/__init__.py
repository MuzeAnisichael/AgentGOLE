"""Function tools and built-in tools."""

from collections.abc import Iterable

from ..protocols import Tool
from .builtins import calculator, current_datetime
from .function import tool

__all__ = ["resolve_tools", "tool"]


def resolve_tools(tools: Iterable[str | Tool]) -> dict[str, Tool]:
    """Resolve built-in names and index structural tools by their public name."""
    builtins = {"calculator": calculator, "datetime": current_datetime}
    resolved: dict[str, Tool] = {}
    for item in tools:
        if isinstance(item, str):
            if item not in builtins:
                raise ValueError(f"Unknown built-in tool: {item}")
            item = builtins[item]
        if item.spec.name in resolved:
            raise ValueError(f"Duplicate tool name: {item.spec.name}")
        resolved[item.spec.name] = item
    return resolved
