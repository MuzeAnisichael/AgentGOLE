"""Data exchanged by models, tools, paradigms and the run loop."""

from __future__ import annotations

from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from . import ModelClient

ToolMode = Literal["auto", "native", "prompt"]
ResolvedMode = Literal["native", "prompt"]


class ToolCall(BaseModel):
    """A model's request to call one named tool."""

    id: str
    name: str
    arguments: dict[str, Any]
    invalid_arguments: bool = Field(default=False, exclude=True)


class Message(BaseModel):
    """A chat message with optional tool calls or private reasoning."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list[ToolCall])
    tool_call_id: str | None = None
    name: str | None = None
    reasoning: str | None = None


class ToolSpec(BaseModel):
    """The name, description and JSON Schema sent to a model."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolResult(BaseModel):
    """The serialized observation for a single tool call."""

    call_id: str
    name: str
    output: str
    error: str | None = None


class Usage(BaseModel):
    """Token counts reported by a model."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class Capabilities(BaseModel):
    """Features and context size reported by a provider."""

    tools: bool = False
    json_mode: bool = False
    vision: bool = False
    json_schema: bool = False
    context: int | None = None


class ModelResponse(BaseModel):
    """A completed model reply, including usage and provider metadata."""

    message: Message
    usage: Usage = Field(default_factory=Usage)
    model: str
    raw: Any = None


class ModelChunk(BaseModel):
    """A visible text delta, or the completed response at end of stream."""

    text: str = ""
    response: ModelResponse | None = None


class Event(BaseModel):
    """A timestamped run event with additional fields specific to its type."""

    model_config = ConfigDict(extra="allow")
    type: str
    run_id: str
    ts: float = Field(default_factory=time)
    step: int | None = None

    # These fields exist only on the event kinds that carry them.
    if TYPE_CHECKING:

        def __init__(
            self, *, type: str, run_id: str, ts: float = ..., step: int | None = None, **data: Any
        ) -> None: ...

        def __getattr__(self, name: str) -> Any: ...


class Step(BaseModel):
    """One executed action and its result."""

    turn: int
    action: Literal["model", "tool", "finish"]
    output: str | None = None
    calls: list[ToolCall] = Field(default_factory=list[ToolCall])
    results: list[ToolResult] = Field(default_factory=list[ToolResult])
    duration: float = 0.0


class Trace(BaseModel):
    """The events of a run up to its terminal result."""

    events: list[Event] = Field(default_factory=list[Event])

    def show(self) -> None:
        """Print the recorded run as a plain-text tree."""
        from ..trace import show

        show(self)

    def to_json(self, path: str | Path | None = None) -> str:
        """Return the events as a JSON document, also writing it to ``path`` when given.

        The document holds the complete conversation, tool arguments and tool results, but
        no credentials: request headers never enter events. Check its content before
        sending it to someone else.

        Example:
            result.trace.to_json("traces/run.json")
        """
        from ..trace import to_json

        return to_json(self, path)


class Document(BaseModel):
    """Loaded source text and optional one-based page number."""

    text: str
    source: str
    page: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """An indexed passage with stable identity and retrieval score."""

    id: str
    text: str
    source: str
    page: int | None = None
    score: float = 0.0


class Citation(BaseModel):
    """A short source excerpt collected during retrieval."""

    source: str
    page: int | None = None
    text: str

    @field_validator("text")
    @classmethod
    def _excerpt(cls, value: str) -> str:
        return value[:300]


class Result(BaseModel):
    """The final text, actions, token usage and trace of one agent run."""

    text: str
    steps: list[Step]
    usage: Usage
    citations: list[Citation] = Field(default_factory=list[Citation])
    output: Any = None
    trace: Trace
    run_id: str
    session: str

    def __str__(self) -> str:
        return self.text


class RunState(BaseModel):
    """The conversation and action history available to the paradigm."""

    input: str
    messages: list[Message]
    turn: int = 0
    scratch: dict[str, Any] = Field(default_factory=dict)
    history: list[Step] = Field(default_factory=list[Step])
    last_response: Message | None = None


class CallModel(BaseModel):
    """Generate a reply, optionally with another model, hidden from the token stream.

    ``model`` selects the client for this call and defaults to the run's model.
    ``visible=False`` marks intermediate text that must not be emitted as tokens.
    ``response_schema`` is a JSON Schema the reply should satisfy; the engine passes it
    to backends that support constrained decoding.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    messages: list[Message] | None = None
    use_tools: bool = True
    emit_tokens_after: str | None = None
    model: ModelClient | None = None
    visible: bool = True
    response_schema: dict[str, Any] | None = None


class CallTool(BaseModel):
    """Execute tool requests and append their observations to the conversation."""

    calls: list[ToolCall]
    as_observation: bool = False


class Finish(BaseModel):
    """End a run with the supplied text."""

    output: str


Action = CallModel | CallTool | Finish
