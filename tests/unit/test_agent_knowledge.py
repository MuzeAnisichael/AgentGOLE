"""Integration of per-run retrieval and memory tools."""

import pytest

import agentgole as gole
from agentgole.errors import ConfigError
from agentgole.testing import FakeModel, fake_tool_call


@pytest.mark.parametrize(
    "mode,native", [("auto", True), ("auto", False), ("tool", True), ("context", True)]
)
async def test_retrieve_and_cite(tmp_path, mode, native):
    source = tmp_path / "policy.md"
    source.write_text("审批流程由经理负责。", encoding="utf-8")
    kb = gole.Knowledge(source, store=tmp_path, retriever="keyword", name="policy")
    via_tool = mode == "tool" or (mode == "auto" and native)
    call = fake_tool_call("search_policy", {"query": "审批"})
    model = FakeModel([call, call, "经理"] if via_tool else ["经理"], tools_supported=native)
    agent = gole.Agent(model, knowledge=kb, knowledge_mode=mode)
    result = await agent.arun("审批")
    assert result.text == "经理"
    assert len(result.citations) == 1 and result.citations[0].source.endswith("policy.md")
    retrieved = [ev for ev in result.trace.events if ev.type == "retrieve"]
    assert len(retrieved) == (2 if via_tool else 1)
    assert all(e.query == "审批" and e.chunks[0].id for e in retrieved)
    sent = model.requests[-1]["messages"]
    assert any("[1]" in (m.content or "") and "经理" in (m.content or "") for m in sent)
    with pytest.raises(ConfigError, match="name"):
        gole.Agent(model, knowledge=[kb, kb])


async def test_session_bound_memory_tool(tmp_path):
    memory = gole.memory.LongTerm(tmp_path / "facts.db", write="tool")
    model = FakeModel([fake_tool_call("remember", {"fact": "喜欢蓝色"}), "ok"])
    agent = gole.Agent(model, memory=memory)
    await agent.arun("记住", session="a")
    assert agent.memory.search("蓝色", session="a") == ["喜欢蓝色"]
    assert agent.memory.search("蓝色", session="b") == []


def test_ask_uses_context_and_default_model(tmp_path, monkeypatch):
    source = tmp_path / "agenda.txt"
    source.write_text("议程：周三组会。", encoding="utf-8")
    model = FakeModel(["周三"])
    monkeypatch.setenv("GOLE_MODEL", "ollama/test")
    monkeypatch.setattr("agentgole.agent.create_client", lambda *args, **kw: model)
    kb = gole.Knowledge(source, retriever="keyword", store=tmp_path)
    result = kb.ask("组会")
    assert result.text == "周三" and result.citations[0].source.endswith("agenda.txt")
    assert any("参考资料：" in (m.content or "") for m in model.requests[0]["messages"])
