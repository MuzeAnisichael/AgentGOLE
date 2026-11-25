"""SQLite persistence for incremental knowledge indexes."""

import hashlib
import json
import re
import sqlite3
from array import array
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from ..protocols.types import Chunk
from .retrieval import SearchIndex


def normalize(source: str | Path) -> str:
    """Return a repeatable source identifier, preserving URL paths and queries."""
    value = str(source)
    if value.lower().startswith(("http://", "https://")):
        return value
    return str(Path(value).expanduser().resolve(strict=False))


def index_name(source: str | Path, name: str | None) -> str:
    """Derive a tool-safe name from an explicit name or the first input."""
    if name is None:
        value = str(source)
        if value.lower().startswith(("http://", "https://")):
            name = urlsplit(value).hostname or ""
        else:
            path = Path(normalize(source))
            if any(char in str(path) for char in "*?["):
                path = next((p for p in path.parents if not any(c in str(p) for c in "*?[")), path)
            name = path.name if path.is_dir() else path.stem
    return re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "kb"


@dataclass
class State:
    meta: dict[str, str]
    documents: dict[str, tuple[str, int]]
    extra_sources: list[str]

    @property
    def revision(self) -> int:
        return int(self.meta.get("revision", "0"))


@dataclass
class IndexedDocument:
    digest: str
    pages: int
    chunks: list[Chunk]
    vectors: list[bytes | None] = field(default_factory=list)


class Store:
    """Read snapshots and atomically commit a complete set of changed sources."""

    def __init__(self, root: str | Path, name: str, sources: list[str]) -> None:
        key = json.dumps(sorted(set(sources)), ensure_ascii=False).encode()
        suffix = hashlib.sha256(key).hexdigest()[:8]
        self.path = Path(root).expanduser().resolve() / f"{name}-{suffix}.db"

    def state(self) -> State:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    source TEXT PRIMARY KEY, sha256 TEXT NOT NULL, page_count INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, source TEXT NOT NULL, page INTEGER,
                    text TEXT NOT NULL, embedding BLOB
                );
                CREATE INDEX IF NOT EXISTS chunks_source ON chunks(source);
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS extra_sources (source TEXT PRIMARY KEY);
            """)
            db.execute("BEGIN")
            documents = {
                source: (digest, count)
                for source, digest, count in db.execute("""
                    SELECT d.source, d.sha256, count(c.id) FROM documents d
                    LEFT JOIN chunks c ON c.source=d.source GROUP BY d.source
                """)
            }
            extra = [
                row[0] for row in db.execute("SELECT source FROM extra_sources ORDER BY source")
            ]
            return State(dict(db.execute("SELECT key,value FROM meta")), documents, extra)

    def update(
        self,
        prior: State,
        removed: set[str],
        changed: dict[str, IndexedDocument],
        meta: dict[str, str],
        extra: str | None,
    ) -> bool:
        """Apply a prepared build only if its starting revision is still current."""
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT value FROM meta WHERE key='revision'").fetchone()
            if int(row[0] if row else 0) != prior.revision:
                return False
            for source in removed | changed.keys():
                db.execute("DELETE FROM chunks WHERE source=?", (source,))
                db.execute("DELETE FROM documents WHERE source=?", (source,))
            for source, doc in changed.items():
                db.execute("INSERT INTO documents VALUES (?,?,?)", (source, doc.digest, doc.pages))
                db.executemany(
                    "INSERT INTO chunks VALUES (?,?,?,?,?)",
                    (
                        (chunk.id, source, chunk.page, chunk.text, vector)
                        for chunk, vector in zip(doc.chunks, doc.vectors, strict=True)
                    ),
                )
            if extra:
                db.execute("INSERT OR IGNORE INTO extra_sources VALUES (?)", (extra,))
            if (
                removed
                or changed
                or meta.items() - prior.meta.items()
                or (extra and extra not in prior.extra_sources)
            ):
                db.executemany(
                    "INSERT OR REPLACE INTO meta VALUES (?,?)",
                    ({**meta, "revision": str(prior.revision + 1)}.items()),
                )
            return True

    def snapshot(self, known_revision: int) -> tuple[int, dict[str, str], SearchIndex] | None:
        """Load all chunks together only when the database revision has changed."""
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("BEGIN")
            meta = dict(db.execute("SELECT key,value FROM meta"))
            revision = int(meta["revision"])
            if revision == known_revision:
                return None
            chunks, vectors = [], []
            for identity, source, page, text, blob in db.execute(
                "SELECT id,source,page,text,embedding FROM chunks ORDER BY source,rowid"
            ):
                chunks.append(Chunk(id=identity, source=source, page=page, text=text))
                if blob is not None:
                    values = array("f")
                    values.frombytes(blob)
                    vectors.append(values)
            return revision, meta, SearchIndex(chunks, vectors)
