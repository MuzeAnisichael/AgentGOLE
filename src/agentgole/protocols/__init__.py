"""Structural interfaces shared by native components."""

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from .types import (
    Action,
    Capabilities,
    Event,
    Message,
    ModelChunk,
    ModelResponse,
    ResolvedMode,
    RunState,
    ToolSpec,
)


@runtime_checkable
class ModelClient(Protocol):
    """Asynchronous model interface accepted by Agent."""

    name: str

    async def generate(
        self, messages: list[Message], *, tools: list[ToolSpec] | None = None, **params: Any
    ) -> ModelResponse: ...

    def stream(
        self, messages: list[Message], *, tools: list[ToolSpec] | None = None, **params: Any
    ) -> AsyncIterator[ModelChunk]: ...

    async def probe(self) -> Capabilities: ...


@runtime_checkable
class Tool(Protocol):
    """Callable component with a model-readable schema."""

    spec: ToolSpec

    async def call(self, **arguments: Any) -> Any: ...


class Memory(Protocol):
    """Load and append conversation turns for a session."""

    async def load(self, session: str, query: str | None = None) -> list[Message]: ...

    async def save(self, session: str, messages: list[Message]) -> None: ...


class StepContext(BaseModel):
    """Resolved components and event hooks available during an action."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: ModelClient
    tools: dict[str, Tool]
    tool_mode: ResolvedMode
    run_id: str
    report_error: Callable[[Exception], Awaitable[None]]
    downgrade: Callable[[], None]


class Paradigm(Protocol):
    """Choose the next action from the current state."""

    async def step(self, state: RunState, ctx: StepContext) -> Action: ...


EventSink = Callable[[Event], Awaitable[None]]
EventCallback = Callable[[Event], Any]
