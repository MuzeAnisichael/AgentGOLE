"""Structured final answers through the public synchronous and streaming facades."""

import pytest
from pydantic import BaseModel

import agentgole as gole
from agentgole.errors import MaxStepsError, OutputValidationError
from agentgole.testing import FakeModel


class Paper(BaseModel):
    title: str
    year: int
    keywords: list[str]


@pytest.mark.parametrize("mode", ["native", "prompt"])
def test_paper_output_preserves_raw_json(mode):
    raw = "{'title':'A','year':2025,'keywords':['agents',],}"
    model = FakeModel([raw])
    result = gole.Agent(model, tool_mode=mode).run("extract", output=Paper)
    assert result.output == Paper(title="A", year=2025, keywords=["agents"])
    assert result.text == raw
    assert "JSON Schema" in (model.requests[0]["messages"][0].content or "")


async def test_invalid_answer_then_valid_array_streams_only_accepted_text():
    model = FakeModel(["{}", '["fixed"]'])
    events = [ev async for ev in gole.Agent(model).astream("extract", output=list[str])]
    assert events[-1].result.output == ["fixed"]
    assert "".join(ev.text for ev in events if ev.type == "token") == events[-1].result.text
    assert "Invalid output:" in (model.requests[1]["messages"][-1].content or "")


def test_three_failures_raise_validation_error_and_actions_consume_budget():
    model = FakeModel(["[]", "[]", "[]"])
    with pytest.raises(OutputValidationError):
        list(gole.Agent(model).stream("extract", output=Paper))
    assert len(model.requests) == 3
    with pytest.raises(MaxStepsError):
        gole.Agent(FakeModel(["[]"]), max_steps=2).run("extract", output=Paper)
