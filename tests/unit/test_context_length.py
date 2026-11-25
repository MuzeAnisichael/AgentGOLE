"""Where the context length comes from and how a nearly full prompt is reported."""

import json
import logging
from typing import Any

import httpx
import pytest

import golekit as gole
from golekit.errors import ConfigError
from golekit.models import create_client


def server(*, num_ctx: int | None = 2048, prompt_tokens: int | None = None):
    def respond(request):
        if request.url.path.endswith("/api/show"):
            parameters = "temperature 0.7" + (f"\nnum_ctx {num_ctx}" if num_ctx else "")
            info = {"llama.context_length": 131072}
            return httpx.Response(
                200, json={"capabilities": [], "parameters": parameters, "model_info": info}
            )
        body: dict[str, Any] = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        if prompt_tokens is not None:
            body["usage"] = {"prompt_tokens": prompt_tokens, "completion_tokens": 1}
        assert json.loads(request.content)["messages"]
        return httpx.Response(200, json=body)

    return httpx.MockTransport(respond)


async def test_sources_apply_in_priority_order(monkeypatch):
    config = {"providers": {"ollama": {"context": 300}}}

    async def resolved(config=config, num_ctx: int | None = 2048, **options):
        client = create_client(
            "ollama/demo", config=config, transport=server(num_ctx=num_ctx), **options
        )
        return (await client.probe()).context

    monkeypatch.setenv("GOLE_CONTEXT", "200")
    assert await resolved(context=100) == 100
    assert await resolved() == 200
    monkeypatch.delenv("GOLE_CONTEXT")
    assert await resolved() == 300
    assert await resolved(config={}) == 2048
    # The model's maximum context is not what the server applies, so it is never used.
    assert await resolved(config={}, num_ctx=None) is None
    monkeypatch.setenv("GOLE_CONTEXT", "large")
    with pytest.raises(ConfigError, match="GOLE_CONTEXT"):
        await resolved()


async def test_nearly_full_prompt_warns_once_per_run(caplog):
    @gole.tool
    def ping() -> str:
        """Check the service."""
        return "pong"

    client = create_client(
        "ollama/demo", config={}, context=100, transport=server(prompt_tokens=95)
    )
    # The scripted server never follows the tool protocol, so one run makes three calls.
    agent = gole.Agent(client, tools=[ping], memory=None)
    with caplog.at_level(logging.WARNING):
        result = await agent.arun("hello")
    ends = [e for e in result.trace.events if e.type == "model_end"]
    assert [e.context_used for e in ends] == [0.95, 0.95, 0.95]
    warnings = [r for r in caplog.records if "context tokens" in r.getMessage()]
    assert len(warnings) == 1 and "95 of 100" in warnings[0].getMessage()
    assert result.text == "ok"


async def test_unknown_length_is_silent_and_missing_usage_is_estimated(caplog):
    unknown = create_client("ollama/demo", config={}, transport=server(num_ctx=None))
    with caplog.at_level(logging.WARNING):
        result = await gole.Agent(unknown, memory=None).arun("hello")
        end = next(e for e in result.trace.events if e.type == "model_end")
        assert end.context_used is None and "context tokens" not in caplog.text
        estimated = create_client("ollama/demo", config={}, context=10, transport=server())
        result = await gole.Agent(estimated, memory=None).arun("word " * 20)
    end = next(e for e in result.trace.events if e.type == "model_end")
    assert end.context_used == 2.5 and "context tokens" in caplog.text
