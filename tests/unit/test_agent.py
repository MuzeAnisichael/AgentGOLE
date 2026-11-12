"""Acceptance tests for the public agent and model entry points."""

import json
import os
import subprocess
import sys

import httpx

import agentgole as gole
from agentgole import model as model_module
from agentgole.testing import FakeModel


def test_session_history_and_stateless_runs():
    fake = FakeModel(["hello", "Ada", "unknown", "one", "two"])
    agent = gole.Agent(fake, instructions="Remember names.")
    agent.run("I am Ada.")
    agent.run("Name?")
    agent.run("Name?", session="other")
    assert [m.content for m in fake.requests[1]["messages"]] == [
        "Remember names.",
        "I am Ada.",
        "hello",
        "Name?",
    ]
    assert [m.content for m in fake.requests[2]["messages"]] == ["Remember names.", "Name?"]
    stateless = gole.Agent(fake, memory=None)
    stateless.run("first")
    stateless.run("second")
    assert [m.content for m in fake.requests[-1]["messages"]] == ["second"]


async def test_sync_call_inside_loop_and_filtered_streams():
    agent = gole.Agent(FakeModel(["nested", "first stream", "second stream"]))
    assert str(agent.run("hi")) == "nested"
    sync_events = list(agent.stream("hi", events=["token"]))
    assert all(ev.type == "token" for ev in sync_events)
    assert "".join(ev.text for ev in sync_events) == "first stream"
    events = [ev async for ev in agent.astream("hi", events=["token", "run_end"])]
    assert "".join(ev.text for ev in events[:-1]) == "second stream"
    assert events[-1].result.text == "second stream"


async def test_callbacks_isolate_failure_and_trace_shows_actions(capsys, caplog):
    agent = gole.Agent(FakeModel(["hello"]), verbose=True)
    seen = []

    @agent.on("model_start")
    def broken(event):
        raise ValueError("broken hook")

    @agent.on("*")
    async def record(event):
        seen.append(event.type)

    result = await agent.arun("hi")
    assert seen[0] == "run_start" and seen[-1] == "run_end"
    assert "broken hook" in caplog.text
    result.trace.show()
    out, err = capsys.readouterr()
    for rendered in (out, err):
        assert result.run_id[:4] in rendered
        assert "model  fake" in rendered and "最终回答" in rendered


async def test_model_public_generation_and_cached_capabilities(monkeypatch):
    paths = []

    def respond(request):
        paths.append(request.url.path)
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["tools"]})
        data = json.loads(request.content)
        assert data["temperature"] == 0.4 and data["messages"][-1]["content"] == "hi"
        return httpx.Response(
            200,
            json={
                "model": "demo",
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
            },
        )

    factory = model_module.create_client
    monkeypatch.setattr(
        model_module,
        "create_client",
        lambda *a, **kw: factory(*a, **kw, transport=httpx.MockTransport(respond)),
    )
    model = gole.Model("ollama/demo", temperature=0.4)
    assert model.generate("hi").message.content == "hello"
    assert (await model.agenerate("hi")).message.content == "hello"
    assert model.capabilities.tools and model.capabilities.tools
    assert paths.count("/api/show") == 1


def test_import_without_configuration_or_network(tmp_path):
    code = """
import os, sys
def audit(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo'):
        raise AssertionError(event)
    if event == 'open' and str(args[0]).endswith('gole.yaml'):
        raise AssertionError('config read during import')
sys.addaudithook(audit)
import agentgole
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        cwd=tmp_path,
        env={**os.environ, "GOLE_CONFIG": str(tmp_path / "gole.yaml")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
