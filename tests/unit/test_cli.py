"""CLI setup, connectivity reports and streamed conversations."""

import os

import httpx

from golekit import cli


def test_init_preserves_existing_files_until_forced(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    example = tmp_path / "agent_example.py"
    example.write_text("original")
    assert cli.main(["init"]) == 1
    assert example.read_text() == "original" and not (tmp_path / "gole.yaml").exists()
    assert cli.main(["init", "--force"]) == 0
    assert "gole.Agent()" in example.read_text(encoding="utf-8")
    assert "${DEEPSEEK_API_KEY}" in (tmp_path / "gole.yaml").read_text(encoding="utf-8")


def test_doctor_reports_capabilities_and_missing_key(tmp_path, monkeypatch, capsys):
    config = tmp_path / "gole.yaml"
    config.write_text(
        "default_model: ollama/demo\nmodels:\n  cloud: openai/demo\n", encoding="utf-8"
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def respond(request):
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": []})
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )

    factory = cli.create_client
    monkeypatch.setattr(
        cli,
        "create_client",
        lambda *a, **kw: factory(*a, **kw, transport=httpx.MockTransport(respond)),
    )
    assert cli.main(["doctor"]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("[ok]   ollama/demo") and "不支持 → 将使用提示词模式" in lines[0]
    assert "约束解码: 支持" in lines[0] and "上下文: 未知" in lines[0]
    assert "ollama ps" in lines[1]
    assert lines[2].startswith("[fail] openai/demo") and "缺少 OPENAI_API_KEY" in lines[2]
    monkeypatch.setenv("GOLE_CONTEXT", "4096")
    cli.main(["doctor"])
    lines = capsys.readouterr().out.splitlines()
    assert "上下文: 4096" in lines[0] and lines[1].startswith("[fail]")


def test_chat_loads_target_streams_and_resets(tmp_path, monkeypatch, capsys):
    target = tmp_path / "assistant.py"
    target.write_text(
        "from golekit import Agent\nfrom golekit.testing import FakeModel\n"
        "class ChatModel(FakeModel):\n"
        "    async def generate(self, messages, **kw):\n"
        "        assert len(messages) == 1\n"
        "        return await super().generate(messages, **kw)\n"
        "agent = Agent(ChatModel(['hello', 'fresh']))\n",
        encoding="utf-8",
    )
    replies = iter(["hi", "/reset", "again", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(replies))
    assert cli.main(["chat", f"{target}:agent"]) == 0
    assert capsys.readouterr().out.splitlines() == ["hello", "已开始新会话。", "fresh"]


def test_chat_model_available_while_importing_target(tmp_path, monkeypatch):
    target = tmp_path / "default_agent.py"
    target.write_text("from golekit import Agent\nagent = Agent()\n")
    monkeypatch.setattr("builtins.input", lambda _: "/exit")
    assert cli.main(["chat", "--model", "ollama/demo", f"{target}:agent"]) == 0
    assert "GOLE_MODEL" not in os.environ


def test_chat_recovers_after_failed_turn(monkeypatch, capsys):
    from golekit import Agent
    from golekit.errors import ModelError
    from golekit.testing import FakeModel

    class Flaky(FakeModel):
        async def generate(self, messages, **kwargs):
            if messages[-1].content == "fail":
                raise ModelError("first turn failed")
            return await super().generate(messages, **kwargs)

    monkeypatch.setattr(cli, "_load_agent", lambda *args: Agent(Flaky(["recovered"])))
    inputs = iter(["fail", "again", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    assert cli.main(["chat", "agent.py:agent"]) == 0
    output = capsys.readouterr()
    assert "first turn failed" in output.err and "recovered" in output.out


def test_chat_separates_text_before_a_tool_call(monkeypatch, capsys):
    from golekit import Agent, tool
    from golekit.testing import FakeModel, fake_tool_call

    class Narrating(FakeModel):
        async def generate(self, messages, **kwargs):
            response = await super().generate(messages, **kwargs)
            if response.message.tool_calls:
                response.message.content = "Checking."
            return response

    @tool
    def clock() -> str:
        """Tell the time."""
        return "noon"

    model = Narrating([fake_tool_call("clock", {}), "It is noon."])
    monkeypatch.setattr(cli, "_load_agent", lambda *args: Agent(model, tools=[clock]))
    inputs = iter(["time?", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    assert cli.main(["chat", "agent.py:agent"]) == 0
    assert capsys.readouterr().out.splitlines() == ["Checking.", "It is noon."]


def test_kb_build_prints_incremental_counts(tmp_path, monkeypatch, capsys):
    from golekit.testing import FakeEmbedding

    embedding = FakeEmbedding()
    monkeypatch.setattr("golekit.models.embedding.resolve_embedding", lambda *a, **kw: embedding)
    source = tmp_path / "notes.md"
    source.write_text("Team policy", encoding="utf-8")
    args = ["kb", "build", str(source), "--store", str(tmp_path / "store"), "--name", "test"]
    assert cli.main(args) == 0
    assert "文件 1，新增块 1，跳过块 0，删除块 0，耗时" in capsys.readouterr().out
    assert cli.main(args) == 0
    assert "新增块 0，跳过块 1" in capsys.readouterr().out
    assert len(embedding.calls) == 1


def test_chat_model_override_also_switches_fact_extraction(tmp_path, monkeypatch):
    from golekit import Agent
    from golekit.memory import LongTerm
    from golekit.testing import FakeModel

    original, local = FakeModel([]), FakeModel(["ok", '["local fact"]'])
    memory = LongTerm(tmp_path / "facts.db")
    agent = Agent(original, memory=memory)
    monkeypatch.setattr(cli, "_load_agent", lambda *_: agent)
    monkeypatch.setattr("golekit.model.create_client", lambda *args, **kwargs: local)
    replies = iter(["remember", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(replies))
    assert cli.main(["chat", "agent.py:agent", "--model", "ollama/local"]) == 0
    assert not original.requests and memory.export()["facts"] == ["local fact"]


def test_doctor_probes_embed_alias_with_embeddings(tmp_path, monkeypatch, capsys):
    (tmp_path / "gole.yaml").write_text("models:\n  embed: ollama/bge-m3\n", encoding="utf-8")

    def respond(request):
        assert request.url.path == "/v1/embeddings"
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    factory = cli.create_client
    monkeypatch.setattr(
        "golekit.models.embedding.create_client",
        lambda *a, **kw: factory(*a, **kw, transport=httpx.MockTransport(respond)),
    )
    assert cli.main(["doctor"]) == 0
    assert "[ok]" in (out := capsys.readouterr().out) and "文本嵌入" in out
