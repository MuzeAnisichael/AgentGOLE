"""Draft, review and revision cycles driven by scripted models."""

import pytest

import golekit as gole
from golekit.paradigms import Reflect
from golekit.testing import FakeModel

PASS = '{"pass": true, "feedback": ""}'
FAIL = '{"pass": false, "feedback": "更具体一些"}'


@pytest.mark.parametrize("verdict", [PASS, "looks fine to me"])
async def test_accepted_draft_is_returned_and_emitted_once(verdict):
    model = FakeModel(["draft", verdict])
    agent = gole.Agent(model, paradigm=Reflect())
    events = [event async for event in agent.astream("question")]
    result = events[-1].result
    assert result.text == "draft" and len(model.requests) == 2
    assert [e.text for e in events if e.type == "token"] == ["draft"]
    review = model.requests[1]["messages"][-1].content or ""
    assert "question" in review and "draft" in review and "回答准确、完整、切题" in review
    # An unreadable verdict counts as a pass and is reported.
    assert len([e for e in events if e.type == "error"]) == (0 if verdict == PASS else 1)


async def test_one_revision_then_pass_with_a_separate_critic():
    critic = FakeModel([FAIL, PASS], name="critic")
    model = FakeModel(["draft", "better"])
    agent = gole.Agent(model, paradigm=Reflect(critic=critic, criteria="要有数字"))
    events = [event async for event in agent.astream("question")]
    assert events[-1].result.text == "better"
    assert len(critic.requests) == 2 and len(model.requests) == 2
    assert all("要有数字" in (r["messages"][-1].content or "") for r in critic.requests)
    revision = model.requests[1]["messages"]
    assert [m.role for m in revision] == ["user", "assistant", "user"]
    assert revision[1].content == "draft" and "更具体一些" in (revision[2].content or "")
    assert "better" in (critic.requests[1]["messages"][-1].content or "")
    assert "".join(e.text for e in events if e.type == "token") == "better"


async def test_last_revision_is_streamed_and_not_reviewed_again():
    critic = FakeModel([FAIL])
    model = FakeModel(["draft", "final version"])
    agent = gole.Agent(model, paradigm=Reflect(critic=critic, rounds=1))
    events = [event async for event in agent.astream("question")]
    assert events[-1].result.text == "final version" and len(critic.requests) == 1
    tokens = [e.text for e in events if e.type == "token"]
    assert len(tokens) > 1 and "".join(tokens) == "final version"


async def test_zero_rounds_skips_the_review():
    model = FakeModel(["draft"])
    result = await gole.Agent(model, paradigm=Reflect(rounds=0)).arun("question")
    assert result.text == "draft" and len(model.requests) == 1
