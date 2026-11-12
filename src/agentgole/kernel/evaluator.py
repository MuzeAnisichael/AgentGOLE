"""Execute model and tool actions chosen by the paradigm."""

import json
from time import monotonic
from typing import Any

from pydantic import BaseModel

from ..errors import ModelError, ToolError
from ..protocols import StepContext
from ..protocols.types import CallModel, CallTool, Message, ModelResponse, ToolResult, ToolSpec
from .events import RunEvents


async def call_model(
    action: CallModel,
    conversation: list[Message],
    ctx: StepContext,
    streaming: bool,
    events: RunEvents,
) -> ModelResponse:
    """Generate a reply and emit visible content deltas when streaming."""
    messages = conversation if action.messages is None else action.messages
    specs = (
        [tool.spec for tool in ctx.tools.values()]
        if action.use_tools and ctx.tool_mode == "native"
        else []
    )
    started = monotonic()
    await events.send("model_start", messages=list(messages), model=ctx.model.name)
    if streaming:
        response = await _stream_model(
            messages, specs or None, ctx, action.emit_tokens_after, events
        )
    else:
        response = await ctx.model.generate(messages, tools=specs or None)
    await events.send(
        "model_end",
        model=response.model,
        usage=response.usage,
        tool_calls=response.message.tool_calls,
        duration=monotonic() - started,
    )
    return response


async def _stream_model(
    messages: list[Message],
    specs: list[ToolSpec] | None,
    ctx: StepContext,
    marker: str | None,
    events: RunEvents,
) -> ModelResponse:
    response = None
    pending = ""
    chunks = ctx.model.stream(messages, tools=specs)
    try:
        async for chunk in chunks:
            if chunk.response is not None:
                response = chunk.response
            text = chunk.text
            if marker and text:
                pending += text
                index = pending.find(marker)
                if index == -1:
                    # Only this suffix can complete the marker in the next chunk.
                    pending = pending[-(len(marker) - 1) :] if len(marker) > 1 else ""
                    continue
                text = pending[index + len(marker) :]
                marker = None
                pending = ""
            if text:
                await events.send("token", text=text)
    finally:
        close = getattr(chunks, "aclose", None)
        if close is not None:
            await close()
    if response is None:
        raise ModelError("Model stream ended without a completed response.")
    return response


def _serialize(value: Any) -> str:
    if isinstance(value, str):
        text = value
    elif isinstance(value, BaseModel):
        text = value.model_dump_json()
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    marker = "...[truncated]"
    return text[: 8000 - len(marker)] + marker if len(text) > 8000 else text


async def call_tools(action: CallTool, ctx: StepContext, events: RunEvents) -> list[ToolResult]:
    """Execute calls in order, returning errors as observations for the model."""
    results: list[ToolResult] = []
    for call in action.calls:
        started = monotonic()
        await events.send("tool_call", tool=call.name, args=call.arguments)
        error = None
        try:
            tool = ctx.tools.get(call.name)
            if tool is None:
                raise ToolError(f"Unknown tool '{call.name}'. Choose a registered tool.")
            output = _serialize(await tool.call(**call.arguments))
        except Exception as exc:  # noqa: BLE001 - tool failures become observations
            error = f"{type(exc).__name__}: {exc}"
            output = _serialize(f"Error: {error}")
        results.append(ToolResult(call_id=call.id, name=call.name, output=output, error=error))
        await events.send(
            "tool_result",
            tool=call.name,
            args=call.arguments,
            output=output,
            error=error,
            duration=monotonic() - started,
        )
    return results
