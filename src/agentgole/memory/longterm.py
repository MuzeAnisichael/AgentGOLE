"""Session-scoped durable facts with keyword or cosine retrieval."""

from __future__ import annotations

import asyncio
import logging
import math
import sqlite3
import unicodedata
from array import array
from collections.abc import Iterator
from contextlib import closing, contextmanager, nullcontext
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal

from .._json import parse_json
from .._sync import run_sync
from ..errors import ConfigError
from ..models.embedding import resolve_embedding
from ..protocols import EmbeddingClient, ModelClient, Tool
from ..protocols.types import Message
from ..tools.function import tool

if TYPE_CHECKING:
    from ..models.embedding import Embedding

_log = logging.getLogger("agentgole.memory.longterm")


def _unit(vector: list[float]) -> array[float]:
    norm = math.hypot(*vector)
    return array("f", (value / norm if norm else 0.0 for value in vector))


class LongTerm:
    """Persist durable facts in SQLite and recall them within the same session."""

    def __init__(
        self,
        path: str | Path = "./.gole/memory.db",
        embedding: str | Embedding | EmbeddingClient | None = None,
        top_k: int = 5,
        write: Literal["after_run", "tool"] = "after_run",
        backend: str = "builtin",
    ) -> None:
        if backend != "builtin":
            raise ConfigError(
                "Choose backend='builtin'. Additional backends are supported from v0.4."
            )
        if write not in ("after_run", "tool") or top_k < 0:
            raise ConfigError("Choose write='after_run' or 'tool' and a nonnegative top_k.")
        self.path = Path(path).expanduser()
        self.top_k = top_k
        self.write = write
        self._model: ModelClient | None = None
        self._explicit: EmbeddingClient | None = None
        if embedding is not None:
            self._explicit = resolve_embedding(embedding)
        self._embedding = self._explicit
        self._lock = Lock()
        self._ready = False
        self._in_memory: sqlite3.Connection | None = None

    def bind(self, model: ModelClient, embedding: EmbeddingClient | None) -> None:
        """Attach the extraction model and optional default embedding client."""
        self._model = model
        self._embedding = self._explicit if self._explicit is not None else embedding
        if self._embedding is None:
            self._embedding = resolve_embedding(required=False)

    @contextmanager
    def _database(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self.path == Path(":memory:"):
                if self._in_memory is None:
                    self._in_memory = sqlite3.connect(":memory:", check_same_thread=False)
                connection = nullcontext(self._in_memory)
            else:
                if not self._ready:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                connection = closing(sqlite3.connect(self.path))
            with connection as db, db:
                if not self._ready:
                    db.execute("PRAGMA journal_mode=WAL")
                    db.execute("""
                        CREATE TABLE IF NOT EXISTS facts (
                            session TEXT NOT NULL, text TEXT NOT NULL, fact_key TEXT NOT NULL,
                            embedding BLOB, encoder TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(session, fact_key)
                        )
                    """)
                    self._ready = True
                yield db

    def _rows(self, session: str) -> list[tuple[str, bytes | None, str | None]]:
        with self._database() as db:
            return db.execute(
                "SELECT text, embedding, encoder FROM facts WHERE session = ? ORDER BY rowid",
                (session,),
            ).fetchall()

    def _insert(
        self, session: str, facts: dict[str, str], vectors: list[bytes | None], encoder: str | None
    ) -> None:
        with self._database() as db:
            db.executemany(
                "INSERT INTO facts(session, fact_key, text, embedding, encoder) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(session, fact_key) DO NOTHING",
                (
                    (session, key, text, vec, encoder)
                    for (key, text), vec in zip(facts.items(), vectors, strict=True)
                ),
            )

    async def _add(self, session: str, texts: list[str]) -> None:
        facts: dict[str, str] = {}
        for text in texts:
            normalized = " ".join(unicodedata.normalize("NFKC", text).split())
            if normalized:
                facts.setdefault(normalized.casefold(), normalized)
        if not facts:
            return
        embedding = self._embedding
        vectors: list[bytes | None]
        if embedding is None:
            vectors = [None] * len(facts)
        else:
            vectors = [_unit(v).tobytes() for v in await embedding.embed(list(facts.values()))]
        await asyncio.to_thread(
            self._insert, session, facts, vectors, embedding.name if embedding else None
        )

    def add(self, text: str, *, session: str = "default") -> None:
        """Add a fact, ignoring normalized duplicates in the same session."""
        run_sync(self._add(session, [text]))

    def _update_vectors(self, session: str, name: str, updated: list[tuple[bytes, str]]) -> None:
        with self._database() as db:
            db.executemany(
                "UPDATE facts SET embedding=?, encoder=? WHERE session=? AND text=?",
                ((vector, name, session, text) for vector, text in updated),
            )

    async def _search(self, session: str, query: str, k: int) -> list[str]:
        if not query.strip() or k <= 0:
            return []
        rows = await asyncio.to_thread(self._rows, session)
        if not rows:
            return []
        texts = [row[0] for row in rows]
        embedding = self._embedding
        if embedding is None:
            from ..knowledge.retrieval import BM25

            return [texts[i] for i, _ in BM25(texts).rank(query, k)]
        [target] = await embedding.embed([query])
        query_vector = _unit(target)
        vectors = [array("f", blob or b"") for _, blob, _ in rows]
        stale = [
            i
            for i, (_, _, name) in enumerate(rows)
            if name != embedding.name or len(vectors[i]) != len(query_vector)
        ]
        if stale:
            replacements = await embedding.embed([texts[i] for i in stale])
            updated = []
            for i, values in zip(stale, replacements, strict=True):
                vectors[i] = _unit(values)
                updated.append((vectors[i].tobytes(), texts[i]))
            await asyncio.to_thread(self._update_vectors, session, embedding.name, updated)
        scores = [sum(a * b for a, b in zip(query_vector, vec, strict=True)) for vec in vectors]
        ranked = sorted(range(len(texts)), key=lambda i: scores[i], reverse=True)
        return [texts[i] for i in ranked[:k]]

    def search(self, query: str, *, session: str = "default", k: int = 5) -> list[str]:
        """Retrieve up to k matching facts from one session."""
        return run_sync(self._search(session, query, k))

    async def load(self, session: str, query: str | None = None) -> list[Message]:
        """Present relevant facts as a single system message."""
        facts = await self._search(session, query or "", self.top_k)
        if not facts:
            return []
        return [
            Message(
                role="system", content="已知信息：\n" + "\n".join(f"- {fact}" for fact in facts)
            )
        ]

    async def save(self, session: str, messages: list[Message]) -> None:
        """Extract at most five durable facts from the completed turn."""
        if self.write == "tool":
            return
        if self._model is None:
            raise ConfigError("Bind LongTerm to an Agent or choose write='tool'.")
        history = "\n".join(f"{m.role}: {m.content or ''}" for m in messages)
        response = await self._model.generate(
            [
                Message(
                    role="system",
                    content=(
                        "Extract up to five durable facts or preferences about the user from the turn. "
                        "Use the user's language, preserve details such as days and times, and ignore guesses. "
                        "Reply only with a JSON array of strings, or [] if the turn contains no new facts."
                    ),
                ),
                Message(role="user", content=history),
            ]
        )
        try:
            facts = parse_json(response.message.content or "")
            if not isinstance(facts, list) or not all(isinstance(f, str) for f in facts):
                raise ValueError("Expected a JSON array of fact strings.")
        except (ValueError, SyntaxError):
            _log.warning(
                "The extraction model did not return an array of facts. Skipping this turn."
            )
            return
        await self._add(session, facts[:5])

    def tools(self, session: str = "default") -> list[Tool]:
        """Create remember and recall tools bound to the current run's session."""
        if self.write != "tool":
            return []

        @tool
        async def remember(fact: str) -> str:
            """Remember a durable fact or preference for future conversations."""
            await self._add(session, [fact])
            return "Remembered."

        @tool
        async def recall(query: str) -> list[str]:
            """Retrieve remembered facts relevant to the query."""
            return await self._search(session, query, self.top_k)

        return [remember, recall]

    def export(self, *, session: str = "default") -> dict[str, Any]:
        """Export this session's facts in insertion order."""
        with self._database() as db:
            rows = db.execute("SELECT text FROM facts WHERE session=? ORDER BY rowid", (session,))
            return {"facts": [text for (text,) in rows]}

    def clear(self, *, session: str = "default") -> None:
        """Delete the durable facts in the given session."""
        with self._database() as db:
            db.execute("DELETE FROM facts WHERE session=?", (session,))
