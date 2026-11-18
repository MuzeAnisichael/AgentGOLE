"""Translate OpenAI-compatible messages and SSE deltas into kernel types."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..protocols.types import Message, ModelChunk, ModelResponse, ToolCall, Usage


def wire_message(message: Message) -> dict[str, Any]:
    """Serialize conversation history, excluding private reasoning."""
    result = message.model_dump(
        include={"role", "content", "name", "tool_call_id"}, exclude_none=True
    )
    if message.tool_calls:
        result["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in message.tool_calls
        ]
    return result


class _Thinking:
    """Buffer tag boundaries so hidden reasoning never becomes a visible token."""

    def __init__(self) -> None:
        self.phase = "prefix"
        self.pending = ""
        self.reasoning: list[str] = []

    def feed(self, text: str) -> str:
        if self.phase == "visible":
            return text
        text = self.pending + text
        self.pending = ""
        if self.phase == "prefix":
            prefix = text.lstrip()
            if prefix.startswith("<think>"):
                self.phase = "hidden"
                text = prefix[len("<think>") :]
            elif "<think>".startswith(prefix):
                self.pending = text
                return ""
            else:
                self.phase = "visible"
                return text
        if self.phase == "hidden":
            hidden, end, rest = text.partition("</think>")
            if not end:
                # Only a closing tag split between deltas still needs inspection.
                self.reasoning.append(text[:-7])
                self.pending = text[-7:]
                return ""
            self.reasoning.append(hidden)
            self.phase = "separator"
            text = rest
        text = text.lstrip()
        if text:
            self.phase = "visible"
        return text

    def finish(self) -> str:
        if self.phase == "hidden":
            self.reasoning.append(self.pending)
        return self.pending if self.phase == "prefix" else ""


def _call(record: dict[str, Any]) -> ToolCall:
    function = record["function"]
    return ToolCall(
        id=record["id"], name=function["name"], arguments=json.loads(function["arguments"])
    )


def completed_response(data: dict[str, Any], model: str) -> ModelResponse:
    """Parse a complete response, separating private reasoning from the answer."""
    msg = data["choices"][0]["message"]
    thinking = _Thinking()
    content = msg.get("content")
    if content is not None:
        content = thinking.feed(content) + thinking.finish()
    reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "") + "".join(
        thinking.reasoning
    )
    return ModelResponse(
        message=Message(
            role="assistant",
            content=content,
            reasoning=reasoning or None,
            tool_calls=[_call(call) for call in msg.get("tool_calls") or []],
        ),
        model=data.get("model") or model,
        usage=Usage.model_validate(data.get("usage") or {}),
        raw=data,
    )


async def _events(response: httpx.Response) -> AsyncIterator[str]:
    lines: list[str] = []
    async for line in response.aiter_lines():
        if line.startswith("data:"):
            lines.append(line[5:].removeprefix(" "))
        elif not line and lines:
            yield "\n".join(lines)
            lines = []
    if lines:
        yield "\n".join(lines)


async def stream_response(response: httpx.Response, model: str) -> AsyncIterator[ModelChunk]:
    """Yield visible tokens as received and one reconstructed response at the end."""
    thinking = _Thinking()
    text: list[str] = []
    reasoning: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    usage = Usage()
    finished = False
    has_choice = False
    async for payload in _events(response):
        if payload.strip() == "[DONE]":
            finished = True
            break
        frame = json.loads(payload)
        if frame.get("usage"):
            usage = Usage.model_validate(frame["usage"])
        model = frame.get("model") or model
        for choice in frame["choices"]:
            if choice.get("index", 0) != 0:
                continue
            has_choice = True
            finished = finished or bool(choice.get("finish_reason"))
            delta = choice.get("delta") or {}
            reasoning.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
            for part in delta.get("tool_calls") or []:
                record = calls.setdefault(
                    part["index"], {"id": "", "function": {"name": "", "arguments": ""}}
                )
                record["id"] += part.get("id") or ""
                function = part.get("function") or {}
                record["function"]["name"] += function.get("name") or ""
                record["function"]["arguments"] += function.get("arguments") or ""
            visible = thinking.feed(delta.get("content") or "")
            if visible:
                text.append(visible)
                yield ModelChunk(text=visible)
    if not (finished and has_choice):
        raise ValueError("incomplete completion stream")
    if tail := thinking.finish():
        text.append(tail)
        yield ModelChunk(text=tail)
    reasoning.extend(thinking.reasoning)
    message = Message(
        role="assistant",
        content="".join(text) if text or not calls else None,
        reasoning="".join(reasoning) or None,
        tool_calls=[_call(calls[i]) for i in sorted(calls)],
    )
    yield ModelChunk(response=ModelResponse(message=message, model=model, usage=usage))
