"""Scripted offline models for tests of public agent behavior."""

import hashlib
from collections.abc import AsyncIterator, Iterable
from typing import Any, TypedDict
from uuid import uuid4

from ..protocols.types import (
    Capabilities,
    Message,
    ModelChunk,
    ModelResponse,
    ToolCall,
    ToolSpec,
)


class Request(TypedDict):
    """An independent snapshot of one model request."""

    messages: list[Message]
    tools: list[ToolSpec] | None


Reply = str | ToolCall | list[ToolCall]


def fake_tool_call(name: str, args: dict[str, Any]) -> ToolCall:
    """Create a scripted tool request with a unique call ID."""
    return ToolCall(id=uuid4().hex, name=name, arguments=args)


class FakeModel:
    """Return one scripted reply per call and retain request snapshots."""

    def __init__(
        self, script: Iterable[Reply], *, tools_supported: bool = True, name: str = "fake"
    ):
        self.name = name
        self.requests: list[Request] = []
        self._script = iter(script)
        self._tools_supported = tools_supported

    async def generate(
        self, messages: list[Message], *, tools: list[ToolSpec] | None = None, **params: Any
    ) -> ModelResponse:
        """Consume the next scripted text or tool call."""
        self.requests.append(
            {
                "messages": [message.model_copy(deep=True) for message in messages],
                "tools": None if tools is None else [spec.model_copy(deep=True) for spec in tools],
            }
        )
        reply = next(self._script, None)
        if reply is None:
            raise AssertionError("FakeModel script exhausted. Add a reply for this call.")
        message = (
            Message(role="assistant", content=reply)
            if isinstance(reply, str)
            else Message(
                role="assistant", tool_calls=[reply] if isinstance(reply, ToolCall) else reply
            )
        )
        return ModelResponse(message=message, model=self.name)

    async def stream(
        self, messages: list[Message], *, tools: list[ToolSpec] | None = None, **params: Any
    ) -> AsyncIterator[ModelChunk]:
        """Stream small text fragments followed by the complete response."""
        response = await self.generate(messages, tools=tools, **params)
        content = response.message.content or ""
        for index in range(0, len(content), 3):
            yield ModelChunk(text=content[index : index + 3])
        yield ModelChunk(response=response)

    async def probe(self) -> Capabilities:
        """Return the configured native tool capability."""
        return Capabilities(tools=self._tools_supported)


class FakeEmbedding:
    """Deterministically embed text while recording nonempty batch requests."""

    def __init__(self, dim: int = 16):
        if dim < 1:
            raise ValueError("Choose a positive embedding dimension.")
        self.dim = dim
        self.name = f"fake-{dim}"
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return reproducible vectors, including across processes."""
        if texts:
            self.calls.append(list(texts))
        return [
            [(byte - 127.5) / 127.5 for byte in hashlib.shake_256(text.encode()).digest(self.dim)]
            for text in texts
        ]
