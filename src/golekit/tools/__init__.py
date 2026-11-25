"""Function tools and built-in tools."""

import asyncio
from collections.abc import Callable, Iterable
from typing import Any

from ..protocols import Tool
from ._schema import validate_arguments
from .builtins import calculator, current_datetime
from .files import files
from .function import FunctionTool, tool
from .system import python, shell

__all__ = ["files", "python", "resolve_tools", "shell", "tool"]

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


def resolve_tools(tools: Iterable[str | Tool | Iterable[Tool]]) -> dict[str, Tool]:
    """Resolve built-in names, flatten tool lists and index tools by their public name."""
    builtins: dict[str, Tool] = {"calculator": calculator, "datetime": current_datetime}
    # Configurable tools are built on demand so a missing key only matters when they are used.
    factories: dict[str, Callable[[], Tool]] = {"python": python, "shell": shell}
    resolved: dict[str, Tool] = {}
    for entry in tools:
        group = [entry] if isinstance(entry, (str, Tool)) else list(entry)
        for item in group:
            if isinstance(item, str):
                if item == "files":
                    raise ValueError(
                        "The file tools need a root directory. Pass gole.tools.files(root=...)."
                    )
                if item in factories:
                    item = factories[item]()
                elif item in builtins:
                    item = builtins[item]
                else:
                    raise ValueError(f"Unknown built-in tool: {item}")
            if item.spec.name in resolved:
                raise ValueError(f"Duplicate tool name: {item.spec.name}")
            resolved[item.spec.name] = (
                item if isinstance(item, FunctionTool) else _ExternalTool(item)
            )
    return resolved
