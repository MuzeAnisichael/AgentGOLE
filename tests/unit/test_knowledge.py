import asyncio
import sqlite3
from contextlib import closing
from itertools import pairwise

import pytest

from agentgole.errors import ConfigError, ModelError
from agentgole.knowledge import Knowledge
from agentgole.knowledge.chunking import split_document
from agentgole.protocols.types import Document
from agentgole.testing import FakeEmbedding


async def test_incremental_build_across_instances(tmp_path):
    docs = tmp_path / "notes"
    docs.mkdir()
    first, second = docs / "one.md", docs / "two.md"
    first.write_text("alpha")
    second.write_text("beta")
    fake = FakeEmbedding()
    kb = Knowledge(docs, embedding=fake, store=tmp_path)
    assert await kb.abuild() == {"files": 2, "added": 2, "skipped": 0, "deleted": 0}
    fake.calls.clear()
    reopened = Knowledge(docs, embedding=fake, store=tmp_path)
    assert (await reopened.abuild())["skipped"] == 2 and fake.calls == []
    first.write_text("gamma")
    assert (await reopened.abuild())["added"] == 1 and fake.calls == [["gamma"]]
    second.unlink()
    assert (await reopened.abuild())["deleted"] == 1 and fake.calls == [["gamma"]]
    assert [chunk.text for chunk in await reopened.asearch("beta")] == ["gamma"]


def test_keyword_search_finds_cjk_and_persists_added_sources(tmp_path):
    base, extra = tmp_path / "Manual.md", tmp_path / "more.md"
    base.write_text("差旅报销需要经理批准。Ticket ABC123", encoding="utf-8")
    extra.write_text("猫", encoding="utf-8")
    fake = FakeEmbedding()
    kb = Knowledge(base, embedding=fake, store=tmp_path, retriever="keyword")
    assert kb.name == "manual" and str(base) in kb.description
    first = kb.search("报销审批")
    assert len(first) == 1 and first[0].source == str(base) and first[0].score > 0
    assert kb.search("abc123")[0].id == first[0].id and kb.search("无关") == []
    kb.add(extra)
    other = Knowledge(base, store=tmp_path, retriever="keyword")
    assert other.search("猫")[0].source == str(extra) and fake.calls == []


def test_chunk_boundaries_overlap_and_page_identity():
    text = "Header\n\n# Next\nFirst. Second sentence! 01234567890123456789"
    doc = Document(text=text, source="manual.pdf", page=2)
    chunks = split_document(doc, 19, 3)
    assert chunks[0].text == "Header\n\n"
    assert chunks[1].text.endswith("First.")
    assert all(
        len(chunk.text) <= 19 and chunk.page == 2 and chunk.source == doc.source for chunk in chunks
    )
    assert all(left.text[-3:] == right.text[:3] for left, right in pairwise(chunks))
    assert chunks[0].text + "".join(chunk.text[3:] for chunk in chunks[1:]) == text
    assert len({chunk.id for chunk in chunks}) == len(chunks)


async def test_model_and_dimension_changes_rebuild_all_sources(tmp_path, caplog):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("one")
    (docs / "b.md").write_text("two")
    fake = FakeEmbedding(dim=3)
    kb = Knowledge(docs, store=tmp_path, embedding=fake)
    await kb.abuild()
    fake.name = "replacement"
    assert (await kb.abuild())["added"] == 2
    fake.dim = 5
    assert (await kb.abuild())["added"] == 2
    assert "Rebuilding" in caplog.text
    with closing(sqlite3.connect(next(tmp_path.glob("docs-*.db")))) as db:
        assert set(db.execute("SELECT length(embedding) FROM chunks")) == {(5 * 4,)}


def test_vector_and_hybrid_scores(tmp_path):
    docs = tmp_path / "ranking"
    docs.mkdir()
    for i, text in enumerate(["alpha", "alpha note", "other"]):
        (docs / f"{i}.txt").write_text(text)
    fake = FakeEmbedding()
    vector = Knowledge(docs, store=tmp_path, embedding=fake, retriever="vector")
    results = vector.search("alpha")
    assert results[0].text == "alpha" and results[0].score == pytest.approx(1.0)
    hybrid = Knowledge(docs, store=tmp_path, embedding=fake, retriever="hybrid")
    winner = hybrid.search("alpha", k=1)[0]
    assert winner.id == results[0].id and winner.score == pytest.approx(2 / 61)


async def test_failed_rebuild_leaves_previous_index_intact(tmp_path):
    class FailingEmbedding(FakeEmbedding):
        async def embed(self, texts):
            if "broken" in texts:
                raise ModelError("unavailable")
            return await super().embed(texts)

    docs = tmp_path / "docs"
    docs.mkdir()
    for name in ["a", "b"]:
        (docs / f"{name}.md").write_text(name)
    kb = Knowledge(docs, embedding=FailingEmbedding(), store=tmp_path)
    await kb.abuild()
    (docs / "a.md").write_text("updated")
    (docs / "b.md").write_text("broken")
    with pytest.raises(ModelError, match="unavailable"):
        await kb.abuild()
    assert {c.text for c in await kb.asearch("a")} == {"a", "b"}


async def test_concurrent_additions_are_preserved(tmp_path):
    files = [tmp_path / f"{term}.md" for term in ("first", "second", "third")]
    for path in files:
        path.write_text(path.stem)
    one = Knowledge(files[0], store=tmp_path, retriever="keyword")
    two = Knowledge(files[0], store=tmp_path, retriever="keyword")
    await asyncio.gather(one.aadd(files[1]), two.aadd(files[2]))
    assert len(await one.asearch("first second third")) == 3


def test_name_collision_and_unsupported_settings(tmp_path):
    for subdir in ["one", "two"]:
        root = tmp_path / subdir / "Manuals"
        root.mkdir(parents=True)
        (root / "only.txt").write_text(subdir)
        kb = Knowledge(root, store=tmp_path, retriever="keyword")
        assert kb.search(subdir) and kb.name == "manuals"
    assert len(list(tmp_path.glob("manuals-????????.db"))) == 2
    with pytest.raises(ConfigError, match="v0.4"):
        Knowledge(tmp_path, rerank="other")
    with pytest.raises(ConfigError, match="v0.4"):
        Knowledge(tmp_path, backend="other")
