import asyncio
import json
import sqlite3
from contextlib import closing

import pytest

from golekit.errors import ConfigError
from golekit.memory import Layered, LongTerm, Summary, Window
from golekit.protocols.types import Message
from golekit.testing import FakeEmbedding, FakeModel


def turn(user: str, reply: str = "好") -> list[Message]:
    return [Message(role="user", content=user), Message(role="assistant", content=reply)]


async def test_summary_uses_explicit_model_and_only_compacts_older_turns():
    model = FakeModel(["first summary", "updated summary"])
    memory = Summary(max_tokens=6, keep_turns=1, model=model)
    memory.bind(FakeModel([]), None)
    await memory.save("me", turn("甲"))
    await memory.save("me", turn("乙"))
    assert not model.requests
    assert await memory.load("me") == turn("乙")
    await memory.save("me", turn("丙丙丙"))
    assert await memory.load("me") == [
        Message(role="system", content="first summary"),
        *turn("丙丙丙"),
    ]
    assert "甲" in str(model.requests[0]["messages"]) and "丙" not in str(
        model.requests[0]["messages"]
    )
    await memory.save("me", turn("丁"))
    assert "first summary" in str(model.requests[1]["messages"])
    assert memory.export(session="me")["summary"] == "updated summary"
    assert await memory.load("other") == []


async def test_failed_summary_keeps_originals_and_retries_next_turn(caplog):
    memory = Summary(max_tokens=1, keep_turns=1)
    memory.bind(FakeModel(["", "recovered"]), None)
    await memory.save("s", turn("one"))
    await memory.save("s", turn("two"))
    assert await memory.load("s") == [*turn("one"), *turn("two")]
    assert "Retaining original turns" in caplog.text
    await memory.save("s", turn("three"))
    assert await memory.load("s") == [Message(role="system", content="recovered"), *turn("three")]
    await memory.save("other", turn("unrelated"))
    memory.clear(session="s")
    assert await memory.load("s") == []
    assert await memory.load("other") == turn("unrelated")


async def test_facts_survive_reopen_and_are_normalized_capped_and_session_scoped(tmp_path):
    path = tmp_path / "work" / "facts.db"
    memory = LongTerm(path)
    model = FakeModel(
        [
            "```json\n['  Ｔea   time  ', 'tea time', '周三下午开组会', 'blue', 'piano', 'extra',]\n```",
            '["TEA TIME"]',
        ]
    )
    memory.bind(model, None)
    await memory.save("me", turn("facts"))
    await memory.save("me", turn("more facts"))
    assert len(model.requests) == 2
    assert memory.tools() == []
    reopened = LongTerm(path)
    reopened.add("周三上课", session="other")
    assert reopened.export(session="me")["facts"] == ["Tea time", "周三下午开组会", "blue", "piano"]
    assert await reopened.load("me", query="周三下午") == [
        Message(role="system", content="已知信息：\n- 周三下午开组会")
    ]
    assert reopened.search("周三", session="other") == ["周三上课"]
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("PRAGMA journal_mode").fetchone() == ("wal",)


async def test_cosine_retrieval_includes_preexisting_facts_and_preserves_explicit_embedding(
    tmp_path,
):
    path = tmp_path / "facts.db"
    LongTerm(path).add("existing fact")
    embedding = FakeEmbedding(8)
    memory = LongTerm(path, top_k=1, embedding=embedding)
    other = FakeEmbedding(2)
    memory.bind(FakeModel([]), other)
    memory.add("new fact")
    assert memory.search("existing fact", k=1) == ["existing fact"]
    assert await memory.load("default", "new fact") == [
        Message(role="system", content="已知信息：\n- new fact")
    ]
    assert not other.calls
    reopened = LongTerm(path)
    reopened.bind(FakeModel([]), other)
    assert reopened.search("new fact", k=1) == ["new fact"]


async def test_tools_capture_sessions_and_skip_automatic_extraction(tmp_path):
    memory = Layered(Window(), LongTerm(tmp_path / "db", write="tool"))
    model = FakeModel([])
    memory.bind(model, None)
    remember_a, recall_a = memory.tools("a")
    remember_b, recall_b = memory.tools("b")
    assert [remember_a.spec.name, recall_a.spec.name] == ["remember", "recall"]
    assert set(remember_a.spec.parameters["properties"]) == {"fact"}
    await asyncio.gather(remember_a.call(fact="tea green"), remember_b.call(fact="tea black"))
    await memory.save("a", turn("hello"))
    assert not model.requests
    assert json.loads(await recall_a.call(query="tea")) == ["tea green"]
    assert json.loads(await recall_b.call(query="tea")) == ["tea black"]


async def test_layered_management_and_load_order(tmp_path):
    model = FakeModel(['["I prefer tea"]'])
    memory = Layered(short=Window(), long=LongTerm(tmp_path / "db"))
    memory.bind(model, None)
    await memory.save("alice", turn("tea"))
    memory.add("tea time", session="bob")
    assert memory.search("tea", session="alice") == ["I prefer tea"]
    assert await memory.load("alice", "tea") == [
        Message(role="system", content="已知信息：\n- I prefer tea"),
        *turn("tea"),
    ]
    data = memory.export(session="alice")
    assert json.loads(json.dumps(data)) == data
    assert [m["content"] for m in data["short"]["messages"]] == ["tea", "好"]
    assert data["long"]["facts"] == ["I prefer tea"]
    memory.clear(session="alice")
    assert await memory.load("alice", "tea") == []
    assert memory.export(session="bob")["long"]["facts"] == ["tea time"]


def test_external_memory_backend_reports_version():
    with pytest.raises(ConfigError, match="v0.4"):
        LongTerm(backend="mem0")


async def test_layered_save_exposes_model_errors_to_callers(tmp_path):
    from golekit import Agent
    from golekit.errors import ModelError

    class FailedExtraction(FakeModel):
        async def generate(self, messages, **kwargs):
            if messages[0].role == "system" and "durable facts" in messages[0].content:
                raise ModelError("extraction unavailable")
            return await super().generate(messages, **kwargs)

    memory = Layered(short=Window(), long=LongTerm(tmp_path / "facts.db"))
    with pytest.raises(ModelError, match="extraction unavailable"):
        await Agent(FailedExtraction(["ok"]), memory=memory).arun("remember")
    assert memory.export()["short"]["messages"][-1]["content"] == "ok"
