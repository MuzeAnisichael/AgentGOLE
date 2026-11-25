"""Plan-and-execute runs driven by scripted planner and executor models."""

import golekit as gole
from golekit.paradigms import Direct, PlanExecute
from golekit.testing import FakeModel, fake_tool_call


@gole.tool
def weather(city: str) -> str:
    """Look up a city's weather."""
    return f"{city} 18°C"


def last_user(request) -> str:
    return request["messages"][-1].content or ""


async def test_two_steps_with_a_tool_then_streamed_synthesis():
    planner = FakeModel(['["查巴黎天气", "给出穿衣建议"]', "巴黎 18°C，带件外套。"], name="planner")
    executor = FakeModel(
        [fake_tool_call("weather", {"city": "巴黎"}), "巴黎 18°C", "建议带外套"], name="executor"
    )
    agent = gole.Agent(
        executor,
        instructions="简短回答。",
        tools=[weather],
        paradigm=PlanExecute(planner=planner, max_replans=0),
    )
    events = [event async for event in agent.astream("去巴黎穿什么？")]
    result = events[-1].result
    assert result.text == "巴黎 18°C，带件外套。"
    assert "".join(e.text for e in events if e.type == "token") == result.text
    # Each model only receives its own calls; without replans there is no check call.
    assert len(planner.requests) == 2 and len(executor.requests) == 3
    assert all(request["tools"] is None for request in planner.requests)
    first, _, second = executor.requests
    assert [m.role for m in first["messages"]] == ["system", "user"]
    assert "当前步骤：查巴黎天气" in last_user(first) and "（无）" in last_user(first)
    assert "结果：巴黎 18°C" in last_user(second) and "当前步骤：给出穿衣建议" in last_user(second)
    assert "建议带外套" in last_user(planner.requests[1])
    thoughts = [e.thought for e in events if e.type == "step"]
    assert thoughts[0] == "1. 查巴黎天气\n2. 给出穿衣建议" and not any(thoughts[1:])


async def test_check_replans_once_and_keeps_earlier_results():
    planner = FakeModel(['["a"]', '{"done": false, "plan": ["b"]}', "answer"])
    executor = FakeModel(["result a", "result b"])
    agent = gole.Agent(executor, paradigm=PlanExecute(planner=planner, max_replans=1))
    result = await agent.arun("task")
    assert result.text == "answer"
    # The replan budget is spent, so the second batch is not checked again.
    assert len(planner.requests) == 3 and len(executor.requests) == 2
    assert "result a" in last_user(executor.requests[1])
    assert "result a" in last_user(planner.requests[2])
    assert "result b" in last_user(planner.requests[2])
    thoughts = [e.thought for e in result.trace.events if e.type == "step" and e.thought]
    assert thoughts == ["1. a", "1. b"]
    assert result.usage.total_tokens == 0 and len(result.steps) == 6


async def test_finished_check_goes_straight_to_the_answer():
    planner = FakeModel(['["a"]', "not json", "answer"])
    executor = FakeModel(["result a"])
    agent = gole.Agent(executor, paradigm=PlanExecute(planner=planner))
    assert (await agent.arun("task")).text == "answer"
    assert len(planner.requests) == 3 and len(executor.requests) == 1


async def test_unparseable_plan_is_retried_then_runs_as_one_step():
    planner = FakeModel(["I would first look around.", "Step one, step two.", "answer"])
    executor = FakeModel(["done"])
    agent = gole.Agent(executor, paradigm=PlanExecute(planner=planner, max_replans=0))
    result = await agent.arun("find the cause")
    assert result.text == "answer"
    retry = planner.requests[1]["messages"]
    assert retry[-2].content == "I would first look around." and "无法解析" in (
        retry[-1].content or ""
    )
    errors = [e for e in result.trace.events if e.type == "error"]
    assert len(errors) == 1 and errors[0].recoverable
    assert "当前步骤：find the cause" in last_user(executor.requests[0])


async def test_direct_makes_one_call_without_tools():
    model = FakeModel(["42"])
    result = await gole.Agent(model, paradigm=Direct()).arun("answer", output=int)
    assert result.output == 42 and len(model.requests) == 1
    assert model.requests[0]["tools"] is None
    assert [step.action for step in result.steps] == ["model", "finish"]
