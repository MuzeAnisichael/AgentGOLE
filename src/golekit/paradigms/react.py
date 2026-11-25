"""Prompt-based reasoning and tool dispatch."""

import json
import re
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


def _request() -> CallModel:
    return CallModel(use_tools=False, emit_tokens_after=_FINAL)


class React:
    """Use a text protocol with up to two corrective model replies per action."""

    async def step(self, state: RunState, ctx: StepContext) -> Action:
        if not ctx.tools:
            if state.last_response is None:
                return CallModel(use_tools=False)
            return Finish(output=state.last_response.content or "")
        if not state.scratch.get("react_prompt"):
            schema = [tool.spec.model_dump() for tool in ctx.tools.values()]
            prefix = _PROTOCOL.format(tools=json.dumps(schema, ensure_ascii=False))
            system = next((msg for msg in state.messages if msg.role == "system"), None)
            if system is None:
                state.messages.insert(0, Message(role="system", content=prefix))
            else:
                system.content = f"{system.content}\n\n{prefix}" if system.content else prefix
            state.scratch["react_prompt"] = True
            return _request()
        if not state.history or state.history[-1].action != "model" or state.last_response is None:
            return _request()
        raw = state.last_response.content or ""
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
            action = parse_object(raw)
            name = action.get("tool")
            if not isinstance(name, str) or name not in ctx.tools:
                raise ValueError(f"Unknown tool: {name!r}")
            arguments = action.get("arguments")
            if not isinstance(arguments, dict):
                raise TypeError("Tool arguments must be a JSON object")
            validate_arguments(ctx.tools[name], arguments)
        except (ValueError, TypeError, KeyError, SyntaxError, ArithmeticError, re.error) as exc:
            attempts = state.scratch.get("react_retries", 0)
            if attempts >= 2:
                await ctx.report_error(exc)
                return Finish(output=raw)
            state.scratch["react_retries"] = attempts + 1
            state.messages.append(
                Message(
                    role="user",
                    content=f"Error: {exc}. Reply again using the specified Action or Final Answer format.",
                )
            )
            return _request()
        state.scratch["react_retries"] = 0
        call = ToolCall(id=uuid4().hex, name=name, arguments=arguments)
        return CallTool(calls=[call], as_observation=True)
