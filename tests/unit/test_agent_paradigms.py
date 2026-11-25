"""Paradigm selection and model binding on the Agent facade."""

import pytest

import golekit as gole
from golekit.errors import ConfigError
from golekit.testing import FakeModel


@gole.tool
def weather(city: str) -> str:
    """Look up a city's weather."""
    return f"{city} 18°C"


async def test_direct_rejects_tools_and_reads_knowledge_as_context(tmp_path):
    source = tmp_path / "policy.md"
    source.write_text("审批流程由经理负责。", encoding="utf-8")
    kb = gole.Knowledge(source, store=tmp_path / "kb", retriever="keyword", name="policy")
    model = FakeModel(["经理"])
    result = await gole.Agent(model, knowledge=kb, paradigm="direct").arun("审批")
    assert result.text == "经理" and len(model.requests) == 1
    request = model.requests[0]
    assert request["tools"] is None
    assert any("参考资料" in (m.content or "") for m in request["messages"])
    conflicts = [
        {"tools": [weather]},
        {"knowledge": kb, "knowledge_mode": "tool"},
        {"memory": gole.memory.LongTerm(tmp_path / "facts.db", write="tool")},
    ]
    for options in conflicts:
        with pytest.raises(ConfigError, match="direct"):
            gole.Agent(model, paradigm="direct", **options)
    with pytest.raises(ConfigError, match="'plan', 'reflect'"):
        gole.Agent(model, paradigm="tree")


async def test_plan_and_reflect_names_use_the_agents_model():
    planner = FakeModel(['["a"]', "result a", '{"done": true, "plan": []}', "answer"])
    assert (await gole.Agent(planner, paradigm="plan").arun("task")).text == "answer"
    assert len(planner.requests) == 4
    writer = FakeModel(["draft", '{"pass": true, "feedback": ""}'])
    assert (await gole.Agent(writer, paradigm="reflect").arun("task")).text == "draft"


async def test_executor_becomes_the_runs_model_and_sets_the_tool_mode():
    planner = FakeModel(['["查天气"]', "巴黎 18°C。"], name="planner")
    executor = FakeModel(
        [
            'Thought: look it up\nAction:\n```json\n{"tool": "weather", "arguments": {"city": "巴黎"}}\n```',
            "Final Answer: 巴黎 18°C",
        ],
        tools_supported=False,
        name="executor",
    )
    paradigm = gole.paradigms.PlanExecute(executor=executor, max_replans=0)
    agent = gole.Agent(planner, instructions="简短。", tools=[weather], paradigm=paradigm)
    result = await agent.arun("巴黎天气？")
    assert result.text == "巴黎 18°C。"
    # The planner supports native tools, but the executor does not: the loop uses the prompt protocol.
    assert len(planner.requests) == 2 and len(executor.requests) == 2
    assert all(request["tools"] is None for request in executor.requests)
    assert "Available tools" in (executor.requests[0]["messages"][0].content or "")
    assert "Available tools" not in (planner.requests[1]["messages"][0].content or "")
    assert "结果：巴黎 18°C" in (planner.requests[1]["messages"][-1].content or "")
    started = [e.model for e in result.trace.events if e.type == "model_start"]
    assert started == ["planner", "executor", "executor", "planner"]
