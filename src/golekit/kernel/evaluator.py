"""Execute model and tool actions chosen by the paradigm."""

import json
from time import monotonic
from typing import Any

from pydantic import BaseModel

from ..errors import ModelError, UnsupportedSchemaError
from ..protocols import ModelClient, StepContext
from ..protocols.types import (
    CallModel,
    CallTool,
    Capabilities,
    Message,
    ModelResponse,
    ToolResult,
    ToolSpec,
)
from .chain import Refusal, run_tool
from .events import RunEvents


class ModelCalls:
    """What one run has learned about the models it talks to."""

    def __init__(self, ctx: StepContext) -> None:
        self._ctx = ctx
        self._known: dict[int, Capabilities] = {}

    async def capabilities(self, client: ModelClient) -> Capabilities:
        """Resolve a client's features once per run."""
        key = id(client)
        if key not in self._known:
            shared = self._ctx.capabilities if client is self._ctx.model else None
            self._known[key] = await (shared() if shared is not None else client.probe())
        return self._known[key]

    def reject_schema(self, client: ModelClient) -> None:
        """Stop sending JSON Schema response formats to a client that refused one."""
        key = id(client)
        self._known[key] = self._known[key].model_copy(update={"json_schema": False})
        if client is self._ctx.model:
            self._ctx.constrained = False
            self._ctx.downgrade("json_schema")


async def call_model(
    action: CallModel,
    conversation: list[Message],
    ctx: StepContext,
    streaming: bool,
    events: RunEvents,
    pending_tokens: list[str] | None,
    calls: ModelCalls,
) -> ModelResponse:
    """Generate a reply; a streamed call emits its deltas or collects them in pending_tokens."""
    client = action.model or ctx.model
    messages = conversation if action.messages is None else action.messages
    specs = (
        [tool.spec for tool in ctx.tools.values()]
        if action.use_tools and ctx.tool_mode == "native"
        else []
    )
    # Constrained decoding and native tools are never combined in one request (D-307).
    constrain = (
        action.response_schema is not None
        and not specs
        and (await calls.capabilities(client)).json_schema
    )
    started = monotonic()
    await events.send("model_start", messages=list(messages), model=client.name)

    async def request(params: dict[str, Any]) -> ModelResponse:
        if streaming:
            return await _stream_model(
                client,
                messages,
                specs or None,
                action.emit_tokens_after,
                events,
                pending_tokens,
                params,
            )
        return await client.generate(messages, tools=specs or None, **params)

    if constrain:
        schema = {"name": "response", "schema": action.response_schema}
        try:
            response = await request(
                {"response_format": {"type": "json_schema", "json_schema": schema}}
            )
        except UnsupportedSchemaError as exc:
            # Only a successful plain request proves that the response format was the problem.
            response = await request({})
            calls.reject_schema(client)
            await events.recover(exc)
    else:
        response = await request({})
    await events.send(
        "model_end",
        model=response.model,
        usage=response.usage,
        tool_calls=response.message.tool_calls,
        duration=monotonic() - started,
    )
    return response


async def _stream_model(
    client: ModelClient,
    messages: list[Message],
    specs: list[ToolSpec] | None,
    marker: str | None,
    events: RunEvents,
    pending_tokens: list[str] | None,
    params: dict[str, Any],
) -> ModelResponse:
    response = None
    pending = ""
    trim_answer = bool(marker) and pending_tokens is None
    started_answer = False
    trailing = ""
    chunks = client.stream(messages, tools=specs, **params)
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
            if trim_answer and text:
                # ReAct strips the answer. Hold whitespace until another word confirms it is internal.
                text = trailing + text
                if not started_answer:
                    text = text.lstrip()
                visible = text.rstrip()
                trailing = text[len(visible) :]
                text = visible
                started_answer = started_answer or bool(text)
            if text:
                if pending_tokens is None:
                    await events.send("token", text=text)
                else:
                    pending_tokens.append(text)
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
            output = _serialize(await run_tool(call, ctx, events))
        except Refusal as exc:
            error = str(exc)
            output = _serialize(f"Error: {error}")
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
