import json

import httpx
import pytest

from agentgole.errors import ConfigError, ModelError
from agentgole.models import create_client, embedding
from agentgole.models.embedding import Embedding, resolve_embedding
from agentgole.testing import FakeEmbedding


def use_transport(monkeypatch, callback):
    transport = httpx.MockTransport(callback)
    monkeypatch.setattr(
        embedding,
        "create_client",
        lambda spec, **params: create_client(spec, **params, transport=transport),
    )


def test_batch_embeddings_preserve_input_order(monkeypatch):
    batches = []

    def serve(req):
        assert req.method == "POST" and str(req.url) == "http://localhost:11434/v1/embeddings"
        body = json.loads(req.content)
        texts = body["input"]
        assert body == {"input": texts, "model": "bge-m3"}
        batches.append(texts)
        rows = [{"index": i, "embedding": [len(s), i]} for i, s in enumerate(texts)]
        return httpx.Response(200, json={"data": list(reversed(rows))})

    use_transport(monkeypatch, serve)
    model = Embedding("ollama/bge-m3", batch_size=2)
    assert model.embed(["a", "bb", "ccc"]) == [[1, 0], [2, 1], [3, 0]]
    assert model.embed([]) == [] and batches == [["a", "bb"], ["ccc"]]


async def test_embedding_retry_and_error_redaction(monkeypatch):
    replies = iter(
        [
            httpx.Response(429, headers={"Retry-After": "0"}),
            httpx.Response(200, json={"data": [{"index": 0, "embedding": [1]}]}),
            httpx.Response(403, json={"error": {"message": "Bad test-secret"}}),
        ]
    )

    def serve(request):
        assert request.headers["Authorization"] == "Bearer test-secret"
        return next(replies)

    use_transport(monkeypatch, serve)
    model = Embedding("openai/m", api_key="test-secret", retries=1)
    client = resolve_embedding(model)
    assert client is not None and await client.embed(["ok"]) == [[1.0]]
    with pytest.raises(ModelError, match="403.*redacted") as failure:
        await model.aembed(["denied"])
    assert "test-secret" not in str(failure.value)


def test_embedding_resolution_precedence_and_provider_errors(tmp_path, monkeypatch):
    config = tmp_path / "gole.yaml"
    config.write_text("models:\n  embed: lmstudio/local\n  other: ollama/explicit\n")
    monkeypatch.setenv("GOLE_CONFIG", str(config))
    monkeypatch.setenv("GOLE_EMBEDDING", "ollama/env")
    fake = FakeEmbedding()
    assert resolve_embedding(fake) is fake
    explicit = resolve_embedding("other")
    default = resolve_embedding()
    assert explicit and explicit.name == "ollama/explicit"
    assert default and default.name == "lmstudio/local"
    config.write_text("models:\n  unsupported: deepseek/chat\n")
    default = resolve_embedding()
    assert default and default.name == "ollama/env"
    with pytest.raises(ConfigError, match="another provider"):
        resolve_embedding("unsupported")
    monkeypatch.delenv("GOLE_EMBEDDING")
    assert resolve_embedding(required=False) is None
    with pytest.raises(ConfigError, match="models.embed.*GOLE_EMBEDDING"):
        resolve_embedding()


async def test_incomplete_embedding_response_rejected(monkeypatch):
    use_transport(monkeypatch, lambda req: httpx.Response(200, json={"data": []}))
    with pytest.raises(ModelError, match="embedding response"):
        await Embedding("lmstudio/local").aembed(["one"])
