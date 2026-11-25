"""Function tools and built-in tools."""

import asyncio
from collections.abc import Iterable
from typing import Any

from ..protocols import Tool
from ._schema import validate_arguments
from .builtins import calculator, current_datetime
from .function import FunctionTool, tool

__all__ = ["resolve_tools", "tool"]

_DEFAULT_TIMEOUT = 30


class _ExternalTool:
    """Apply the standard argument and timeout policy to a structural tool."""

    def __init__(self, wrapped: Tool) -> None:
        self._wrapped = wrapped
        self.spec = wrapped.spec
        self.confirm = bool(getattr(wrapped, "confirm", False))

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        validate_arguments(self._wrapped, arguments)

    check_arguments = validate_arguments

    async def call(self, **arguments: Any) -> Any:
        self.validate_arguments(arguments)
        async with asyncio.timeout(_DEFAULT_TIMEOUT):
            return await self._wrapped.call(**arguments)


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
        resolved[item.spec.name] = item if isinstance(item, FunctionTool) else _ExternalTool(item)
    return resolved
