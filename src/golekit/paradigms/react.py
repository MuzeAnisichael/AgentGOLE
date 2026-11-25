"""Prompt-based reasoning and tool dispatch."""

import json
import re
from typing import Any
from uuid import uuid4

from .._json import parse_object
from ..protocols import StepContext
from ..protocols.types import Action, CallModel, CallTool, Finish, Message, RunState, ToolCall
from ..tools._schema import validate_arguments

_FINAL = "Final Answer:"
_PROTOCOL = """Available tools (name, description, and JSON argument schema):
{tools}

To request a tool, reply in this format:
Thought: <brief thought>
Action:
```json
{{"tool": "<tool name>", "arguments": {{}}}}
```

To finish, reply in this format:
Thought: <brief thought>
Final Answer: <final answer>"""


_JSON_PROTOCOL = """Available tools (name, description, and JSON argument schema):
{tools}

Reply with one JSON object and nothing else.
To use a tool: {{"thought": "<brief thought>", "tool": "<tool name>", "arguments": {{}}}}
When you have everything needed to answer: {{"thought": "<brief thought>", "tool": "final_answer"}}"""
_ANSWER = "final_answer"
_WRITE_ANSWER = "Now write the final answer to the user's request. Do not use the JSON format."
_WRITE_OUTPUT = "Now write the final answer to the user's request."


def _request() -> CallModel:
    return CallModel(use_tools=False, emit_tokens_after=_FINAL)


def _selection_schema(ctx: StepContext) -> dict[str, Any]:
    """Describe one reply of the constrained protocol: a tool call or the answer signal."""
    thought = {"type": "string"}
    definitions: dict[str, Any] = {}
    options: list[dict[str, Any]] = []
    for name, tool in ctx.tools.items():
        arguments = dict(tool.spec.parameters)
        # References inside a tool's schema resolve from the root, so its definitions move there.
        definitions.update(arguments.pop("$defs", {}))
        options.append(
            {
                "type": "object",
                "properties": {
                    "thought": thought,
                    "tool": {"type": "string", "enum": [name]},
                    "arguments": arguments,
                },
                "required": ["thought", "tool", "arguments"],
                "additionalProperties": False,
            }
        )
    options.append(
        {
            "type": "object",
            "properties": {"thought": thought, "tool": {"type": "string", "enum": [_ANSWER]}},
            "required": ["thought", "tool"],
            "additionalProperties": False,
        }
    )
    schema: dict[str, Any] = {"anyOf": options}
    if definitions:
        schema["$defs"] = definitions
    return schema


class React:
    """Drive tools through a prompt protocol with up to two corrective replies per action.

    When the model accepts a JSON Schema response format, each reply is a constrained JSON
    object that selects a tool or signals the answer, and the answer itself is written by
    one more unconstrained call. Otherwise the Thought / Action / Final Answer text format
    is used.
    """

    async def step(self, state: RunState, ctx: StepContext) -> Action:
        if not ctx.tools:
            if state.last_response is None:
                return CallModel(use_tools=False)
            return Finish(output=state.last_response.content or "")
        if state.scratch.pop("react_answering", False) and state.last_response is not None:
            return Finish(output=state.last_response.content or "")
        if not state.scratch.get("react_prompt"):
            constrained = ctx.constrained and _ANSWER not in ctx.tools
            schema = [tool.spec.model_dump() for tool in ctx.tools.values()]
            template = _JSON_PROTOCOL if constrained else _PROTOCOL
            prefix = template.format(tools=json.dumps(schema, ensure_ascii=False))
            # Only the instructions may carry the protocol; later system messages hold
            # memory or reference material.
            first = state.messages[0] if state.messages else None
            if first and first.role == "system" and state.scratch.get("instructions", True):
                state.scratch["react_plain"] = first.content
                first.content = f"{first.content}\n\n{prefix}" if first.content else prefix
            else:
                state.scratch["react_plain"] = None
                state.messages.insert(0, Message(role="system", content=prefix))
            state.scratch["react_prompt"] = "json" if constrained else "text"
            fresh = True
        else:
            fresh = False
        constrained = state.scratch["react_prompt"] == "json"
        request = (
            CallModel(use_tools=False, visible=False, response_schema=_selection_schema(ctx))
            if constrained
            else _request()
        )
        # A reply that predates the protocol, such as a rejected native call, is not parsed.
        if (
            fresh
            or not state.history
            or state.history[-1].action != "model"
            or state.last_response is None
        ):
            return request
        raw = state.last_response.content or ""
        if constrained:
            return await self._select(state, ctx, raw, request)
        if state.scratch.get("structured_output"):
            # JSON data may itself contain protocol markers or a "tool" field.
            control = re.match(r"(?:\s*Thought:[^\n]*\n)?\s*(Action:|Final Answer:)", raw)
            if control is None:
                return Finish(output=raw)
            if control[1] == _FINAL:
                return Finish(output=raw[control.end() :].strip())
            raw = raw[control.end() :]
        elif _FINAL in raw:
            return Finish(output=raw.split(_FINAL, 1)[1].strip())
        if "{" not in raw:
            return Finish(output=re.sub(r"^\s*Thought:[^\n]*(?:\n|$)", "", raw).strip())
        try:
            call = _tool_call(parse_object(raw), ctx)
        except _ERRORS as exc:
            return await self._retry(
                state, ctx, exc, raw, "the specified Action or Final Answer format", request
            )
        state.scratch["react_retries"] = 0
        return CallTool(calls=[call], as_observation=True)

    async def _select(
        self, state: RunState, ctx: StepContext, raw: str, request: CallModel
    ) -> Action:
        try:
            choice = parse_object(raw)
            thought = choice.get("thought")
            if isinstance(thought, str):
                state.scratch["thought"] = thought
            call = None if choice.get("tool") == _ANSWER else _tool_call(choice, ctx)
        except _ERRORS as exc:
            return await self._retry(
                state, ctx, exc, raw, "one JSON object in the specified format", request
            )
        state.scratch["react_retries"] = 0
        if call is not None:
            return CallTool(calls=[call], as_observation=True)
        # The answer is written outside the JSON protocol so it can stream as plain text.
        plain = state.scratch.get("react_plain")
        messages = state.messages[1:-1]
        if plain:
            messages.insert(0, Message(role="system", content=plain))
        prompt = _WRITE_OUTPUT if state.scratch.get("structured_output") else _WRITE_ANSWER
        messages.append(Message(role="user", content=prompt))
        state.messages[:] = messages
        state.scratch["react_answering"] = True
        # The protocol left the conversation; a later tool loop in this run adds it again.
        state.scratch["react_prompt"] = None
        return CallModel(use_tools=False)

    async def _retry(
        self,
        state: RunState,
        ctx: StepContext,
        exc: Exception,
        raw: str,
        expected: str,
        request: CallModel,
    ) -> Action:
        attempts = state.scratch.get("react_retries", 0)
        if attempts >= 2:
            await ctx.report_error(exc)
            return Finish(output=raw)
        state.scratch["react_retries"] = attempts + 1
        state.messages.append(
            Message(role="user", content=f"Error: {exc}. Reply again using {expected}.")
        )
        return request


_ERRORS = (ValueError, TypeError, KeyError, SyntaxError, ArithmeticError, re.error)


def _tool_call(action: dict[str, Any], ctx: StepContext) -> ToolCall:
    name = action.get("tool")
    if not isinstance(name, str) or name not in ctx.tools:
        raise ValueError(f"Unknown tool: {name!r}")
    arguments = action.get("arguments")
    if not isinstance(arguments, dict):
        raise TypeError("Tool arguments must be a JSON object")
    validate_arguments(ctx.tools[name], arguments)
    return ToolCall(id=uuid4().hex, name=name, arguments=arguments)
