"""Integration of per-run retrieval and memory tools."""

import pytest

import golekit as gole
from golekit.errors import ConfigError
from golekit.testing import FakeModel, fake_tool_call


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
    monkeypatch.setattr("golekit.agent.create_client", lambda *args, **kw: model)
    kb = gole.Knowledge(source, retriever="keyword", store=tmp_path)
    result = kb.ask("组会")
    assert result.text == "周三" and result.citations[0].source.endswith("agenda.txt")
    assert any("参考资料：" in (m.content or "") for m in model.requests[0]["messages"])


def test_memory_resolves_default_embeddings_only_if_needed(tmp_path, monkeypatch):
    from golekit.memory import LongTerm
    from golekit.testing import FakeEmbedding

    monkeypatch.setenv("GOLE_EMBEDDING", "openai/text-embedding-3-small")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    model = FakeModel(["hi"])
    assert gole.Agent(model, memory="summary").run("hello").text == "hi"
    embedding = FakeEmbedding()
    memory = LongTerm(tmp_path / "notes.db", embedding=embedding, write="tool")
    gole.Agent(model, memory=memory).memory.add("blue")
    assert embedding.calls == [["blue"]]


@pytest.mark.parametrize("instructions", ["Be brief.", None])
async def test_context_references_follow_instructions_and_skip_empty(tmp_path, instructions):
    source = tmp_path / "policy.md"
    source.write_text("审批流程由经理负责。", encoding="utf-8")
    kb = gole.Knowledge(source, store=tmp_path, retriever="keyword", name="policy")

    @gole.tool
    def ping() -> str:
        """Check the service."""
        return "pong"

    model = FakeModel(["Final Answer:经理", "Final Answer:none"], tools_supported=False)
    agent = gole.Agent(
        model, instructions=instructions, tools=[ping], knowledge=kb, knowledge_mode="context"
    )
    await agent.arun("审批", session="hit")
    protocol, references, question = model.requests[0]["messages"]
    assert protocol.role == "system" and "Available tools" in (protocol.content or "")
    assert (protocol.content or "").startswith("Be brief." if instructions else "Available tools")
    assert (references.content or "").startswith("参考资料：\n[1]") and question.role == "user"
    await agent.arun("zebra", session="miss")
    assert not any("参考资料" in (m.content or "") for m in model.requests[1]["messages"])


async def test_search_tool_reports_no_match(tmp_path):
    source = tmp_path / "policy.md"
    source.write_text("审批流程由经理负责。", encoding="utf-8")
    kb = gole.Knowledge(source, store=tmp_path, retriever="keyword", name="policy")
    model = FakeModel([fake_tool_call("search_policy", {"query": "zebra"}), "unknown"])
    result = await gole.Agent(model, knowledge=kb, knowledge_mode="tool").arun("zebra")
    assert result.steps[1].results[0].output == "未找到相关内容。"
