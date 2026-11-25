"""Replay recorded model replies through the tool-call parsers."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from golekit.models.wire import completed_response
from golekit.paradigms import React
from golekit.protocols import StepContext
from golekit.protocols.types import CallModel, CallTool, Message, RunState, Step, ToolSpec

CORPUS = Path(__file__).resolve().parents[1] / "corpus" / "tool_calling"


class Offered:
    def __init__(self, spec: dict[str, Any]):
        self.spec = ToolSpec(**spec)

    async def call(self, **arguments: Any) -> Any:
        raise AssertionError("Interpreting a reply must never execute a tool")


def samples():
    for path in sorted(CORPUS.glob("*.json")):
        sample = json.loads(path.read_text(encoding="utf-8"))
        marks = (
            [pytest.mark.xfail(reason=sample["xfail"], strict=True)] if "xfail" in sample else []
        )
        yield pytest.param(sample, id=path.stem, marks=marks)


async def interpret(sample: dict[str, Any]) -> dict[str, Any]:
    if sample["protocol"] == "native":
        function = {"name": sample["tools"][0]["name"], "arguments": sample["reply"]}
        body = {"choices": [{"message": {"tool_calls": [{"function": function}]}}]}
        call = completed_response(body, "corpus").message.tool_calls[0]
        return {"invalid": True} if call.invalid_arguments else {"arguments": call.arguments}
    ctx = StepContext.model_construct(
        tool_mode="prompt",
        tools={spec["name"]: Offered(spec) for spec in sample["tools"]},
        report_error=AsyncMock(),
        run_id="corpus",
        constrained=sample["protocol"] == "constrained",
    )
    state = RunState(input="", messages=[])
    react = React()
    await react.step(state, ctx)
    reply = Message(role="assistant", content=sample["reply"])
    state.history.append(Step(turn=1, action="model"))
    state.messages.append(reply)
    state.last_response = reply
    action = await react.step(state, ctx)
    if isinstance(action, CallTool):
        return {"tool": action.calls[0].name, "arguments": action.calls[0].arguments}
    if isinstance(action, CallModel):
        return {"answer": True} if state.scratch.get("react_answering") else {"retry": True}
    return {"final": action.output}


@pytest.mark.parametrize("sample", list(samples()))
async def test_reply_is_interpreted_as_recorded(sample):
    assert await interpret(sample) == sample["expected"]
