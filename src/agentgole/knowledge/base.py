"""Local document indexing and retrieval."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .._sync import run_sync
from ..errors import ConfigError, ModelError
from ..models import embedding as embedding_models
from ..models.embedding import Embedding
from ..protocols import EmbeddingClient, ModelClient
from ..protocols.types import Chunk, Document, Result
from .chunking import split_document
from .loaders import discover_sources, load_source
from .retrieval import SearchIndex, unit_vector
from .store import IndexedDocument, State, Store, index_name, normalize

if TYPE_CHECKING:
    from ..model import Model

_logger = logging.getLogger("agentgole.knowledge")


def _file_digest(source: str) -> str:
    with open(source, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Knowledge:
    """Index local documents or HTTP sources and retrieve passages with citations."""

    def __init__(
        self,
        sources: str | Path | list[str | Path],
        *,
        embedding: str | Embedding | EmbeddingClient | None = None,
        store: str | Path = "./.gole/kb",
        name: str | None = None,
        description: str | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        retriever: str = "hybrid",
        rerank: str | None = None,
        backend: str = "builtin",
    ) -> None:
        if rerank is not None or backend != "builtin":
            raise ConfigError("Custom rerank and backend settings are supported starting in v0.4.")
        if not 0 <= chunk_overlap < chunk_size:
            raise ConfigError("Set chunk_size > 0 and 0 <= chunk_overlap < chunk_size.")
        if retriever not in {"vector", "hybrid", "keyword"}:
            raise ConfigError("Choose the vector, keyword or hybrid retriever.")
        items = [sources] if isinstance(sources, (str, Path)) else sources
        if not items:
            raise ConfigError("Provide at least one knowledge source.")
        self.name = index_name(items[0], name)
        self.description = (
            description
            if description is not None
            else (f"在知识库 {self.name} 中检索，来源：{', '.join(map(str, items[:3]))}")
        )
        self._sources = [normalize(item) for item in items]
        self._store = Store(store, self.name, self._sources)
        self._embedding = (
            embedding_models.resolve_embedding(embedding) if retriever != "keyword" else None
        )
        self._retriever = retriever
        self._size = chunk_size
        self._overlap = chunk_overlap
        self._built = False
        self._cache: tuple[int, dict[str, str], SearchIndex] | None = None

    def _settings(self) -> dict[str, str]:
        return {
            "model": self._embedding.name if self._embedding else "",
            "chunk_size": str(self._size),
            "chunk_overlap": str(self._overlap),
        }

    async def _encode(
        self, source: str, digest: str, docs: list[Document] | None
    ) -> tuple[IndexedDocument, int]:
        if docs is None:
            docs = await load_source(source)
        chunks = [chunk for doc in docs for chunk in split_document(doc, self._size, self._overlap)]
        result = IndexedDocument(digest, len(docs), chunks)
        if not self._embedding or not chunks:
            result.vectors = [None] * len(chunks)
            return result, 0
        vectors = await self._embedding.embed([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks) or any(len(v) != len(vectors[0]) for v in vectors):
            raise ModelError("The embedding model returned inconsistent vectors.")
        result.vectors = [unit_vector(v).tobytes() for v in vectors]
        return result, len(vectors[0])

    async def _build(
        self, prior: State, extra: str | None, expected_dim: int | None
    ) -> dict[str, int] | None:
        specs = list(
            dict.fromkeys([*self._sources, *prior.extra_sources, *([extra] if extra else [])])
        )
        sources = await discover_sources(specs)
        removed = prior.documents.keys() - set(sources)
        meta = self._settings()
        old_dim = int(prior.meta.get("dimension", 0))
        rebuild = bool(prior.revision and meta.items() - prior.meta.items())
        rebuild |= bool(expected_dim and old_dim and expected_dim != old_dim)
        digests: dict[str, str] = {}
        previews: dict[str, list[Document]] = {}
        pending: list[str] = []
        for source in sources:
            if source.lower().startswith(("http://", "https://")):
                docs = previews[source] = await load_source(source)
                digest = hashlib.sha256("\0".join(doc.text for doc in docs).encode()).hexdigest()
            else:
                digest = await asyncio.to_thread(_file_digest, source)
            digests[source] = digest
            if rebuild or prior.documents.get(source, ("", 0))[0] != digest:
                pending.append(source)

        changed: dict[str, IndexedDocument] = {}
        dim = expected_dim or 0
        position = 0
        while position < len(pending):
            source = pending[position]
            position += 1
            result, observed = await self._encode(source, digests[source], previews.get(source))
            changed[source] = result
            if not observed:
                continue
            if dim and observed != dim:
                raise ModelError(
                    "Embedding dimensions changed during indexing. Check the provider."
                )
            dim = observed
            if old_dim and dim != old_dim and not rebuild:
                rebuild = True
                scheduled = set(pending)
                pending.extend(item for item in sources if item not in scheduled)

        if rebuild:
            _logger.warning(
                "Index settings or embedding dimensions changed. Rebuilding %s.", self.name
            )
        meta["dimension"] = str(dim or (0 if rebuild else old_dim))
        committed = await asyncio.to_thread(
            self._store.update, prior, removed, changed, meta, extra
        )
        if not committed:
            return None
        self._built = True
        return {
            "files": len(sources),
            "added": sum(len(doc.chunks) for doc in changed.values()),
            "skipped": sum(prior.documents[source][1] for source in set(sources) - changed.keys()),
            "deleted": sum(
                prior.documents.get(source, ("", 0))[1] for source in removed | changed.keys()
            ),
        }

    async def _update(
        self, extra: str | None = None, expected_dim: int | None = None
    ) -> dict[str, int]:
        if expected_dim is None:
            # The protocol need not advertise dimensions. Observe the next new vector if absent.
            advertised = getattr(self._embedding, "dim", None)
            expected_dim = advertised if isinstance(advertised, int) and advertised > 0 else None
        while True:
            prior = await asyncio.to_thread(self._store.state)
            stats = await self._build(prior, extra, expected_dim)
            if stats is not None:
                return stats

    async def abuild(self) -> dict[str, int]:
        """Refresh the index and return file and chunk counts."""
        return await self._update()

    def add(self, source: str | Path) -> None:
        """Add and index another persistent source."""
        run_sync(self.aadd(source))

    async def aadd(self, source: str | Path) -> None:
        """Asynchronously add and index another persistent source."""
        await self._update(extra=normalize(source))

    async def _snapshot(self) -> tuple[dict[str, str], SearchIndex]:
        previous = self._cache
        fresh = await asyncio.to_thread(self._store.snapshot, previous[0] if previous else -1)
        if fresh is not None:
            self._cache = fresh
            return fresh[1], fresh[2]
        assert previous is not None
        return previous[1], previous[2]

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        """Retrieve the top k passages for a query, building the index on first use."""
        return run_sync(self.asearch(query, k))

    async def asearch(self, query: str, k: int = 5) -> list[Chunk]:
        """Asynchronously retrieve the top k passages for a query."""
        if k <= 0 or not query.strip():
            return []
        if not self._built:
            await self.abuild()
        meta, index = await self._snapshot()
        if self._settings().items() - meta.items():
            await self.abuild()
            meta, index = await self._snapshot()
        if not index.chunks:
            return []
        vector = None
        if self._embedding:
            encoded = await self._embedding.embed([query])
            if len(encoded) != 1:
                raise ModelError("The embedding model must return one vector per query.")
            vector = unit_vector(encoded[0])
            if len(vector) != int(meta["dimension"]):
                await self._update(expected_dim=len(vector))
                _, index = await self._snapshot()
        return await asyncio.to_thread(index.search, query, k, self._retriever, vector)

    def ask(self, question: str, *, model: str | Model | ModelClient | None = None) -> Result:
        """Answer with retrieved context and return the cited passages in the result."""
        from ..agent import Agent

        return Agent(model, knowledge=self, knowledge_mode="context").run(question)
