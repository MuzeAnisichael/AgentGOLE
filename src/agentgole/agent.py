"""Public agent facade."""

from collections.abc import AsyncIterator, Callable, Iterable, Iterator

from .model import Model
from .protocols import EventCallback, Memory, ModelClient, Paradigm, Tool
from .protocols.types import Event, Result, ToolMode


class Agent:
    """Run a model with tools and optional session memory."""

    def __init__(
        self,
        model: str | Model | ModelClient | None = None,
        *,
        instructions: str | None = None,
        tools: Iterable[str | Tool] = (),
        memory: str | Memory | None = "session",
        paradigm: str | Paradigm = "auto",
        name: str | None = None,
        max_steps: int = 20,
        timeout: float | None = None,
        tool_mode: ToolMode | None = None,
        verbose: bool = False,
    ) -> None:
        raise NotImplementedError

    def run(self, input: str, *, session: str = "default") -> Result:
        """Run to completion and return the result."""
        raise NotImplementedError

    async def arun(self, input: str, *, session: str = "default") -> Result:
        """Run asynchronously to completion."""
        raise NotImplementedError

    def stream(
        self, input: str, *, session: str = "default", events: list[str] | None = None
    ) -> Iterator[Event]:
        """Iterate over events as they happen."""
        raise NotImplementedError

    def astream(
        self, input: str, *, session: str = "default", events: list[str] | None = None
    ) -> AsyncIterator[Event]:
        """Iterate asynchronously over events as they happen."""
        raise NotImplementedError

    def on(self, event_type: str) -> Callable[[EventCallback], EventCallback]:
        """Register a callback for an event type or '*'."""
        raise NotImplementedError
