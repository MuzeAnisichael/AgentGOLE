"""Batch embeddings from OpenAI-compatible providers."""

import math
import os

from .._sync import run_sync
from ..config import load_config
from ..errors import ConfigError, ModelError
from ..protocols import EmbeddingClient
from . import create_client
from .client import CompatibleClient


class _BatchEmbedding:
    def __init__(self, client: CompatibleClient, batch_size: int) -> None:
        if batch_size < 1:
            raise ConfigError("Choose a positive embedding batch_size.")
        self.name = client.name
        self._client = client
        self._batch_size = batch_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            data = await self._client.embedding_data(batch)
            try:
                if not isinstance(data, list) or len(data) != len(batch):
                    raise ValueError
                for index, row in enumerate(sorted(data, key=lambda item: item["index"])):
                    raw = row["embedding"]
                    if row["index"] != index or not isinstance(raw, list) or not raw:
                        raise ValueError
                    vector = [float(value) for value in raw]
                    if not all(map(math.isfinite, vector)) or (
                        vectors and len(vector) != len(vectors[0])
                    ):
                        raise ValueError
                    vectors.append(vector)
            except (TypeError, KeyError, ValueError, OverflowError):
                raise ModelError(
                    "Invalid embedding response. Check the provider and model."
                ) from None
        return vectors


class Embedding:
    """Configure an embedding model by provider/model or a gole.yaml alias."""

    def __init__(
        self,
        spec: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        batch_size: int = 32,
        timeout: float = 60.0,
        retries: int = 2,
    ) -> None:
        client = create_client(
            spec,
            config=load_config(),
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            retries=retries,
            embedding=True,
        )
        self._client = _BatchEmbedding(client, batch_size)
        self.name = client.name

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode texts in input order."""
        return run_sync(self.aembed(texts))

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously encode texts in input order."""
        return await self._client.embed(texts)


def resolve_embedding(
    embedding: str | Embedding | EmbeddingClient | None = None, *, required: bool = True
) -> EmbeddingClient | None:
    """Resolve an explicit embedding, the configured embed alias, or GOLE_EMBEDDING."""
    if isinstance(embedding, Embedding):
        return embedding._client
    if embedding is not None and not isinstance(embedding, str):
        return embedding
    config = load_config()
    spec = (
        embedding
        if embedding is not None
        else (config.get("models", {}).get("embed") or os.environ.get("GOLE_EMBEDDING"))
    )
    if spec:
        return _BatchEmbedding(create_client(spec, config=config, embedding=True), 32)
    if required:
        raise ConfigError("Set models.embed in gole.yaml or GOLE_EMBEDDING to an embedding model.")
    return None
