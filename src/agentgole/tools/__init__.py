"""Function tools and built-in tools."""

from collections.abc import Callable
from typing import Any

from ..protocols import Tool


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    timeout: float = 30,
    retries: int = 0,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Decorate a typed Python function as a tool."""
    raise NotImplementedError
