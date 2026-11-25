"""JSON Schema constrained decoding on OpenAI-compatible chat requests."""

import json

import httpx

import golekit as gole
from golekit.models import create_client
from golekit.protocols.types import Capabilities
from golekit.testing import FakeModel


@gole.tool
def weather(city: str) -> str:
    """Look up a city's weather."""
    return f"{city} 18°C"


class Server:
    """Answer chat requests from a script and keep the request bodies."""

    def __init__(self, replies, *, native_tools=False, reject_format=False):
        self.replies = iter(replies)
        self.bodies: list[dict] = []
        self.native_tools = native_tools
        self.reject_format = reject_format

    def __call__(self, request):
        if request.url.path.endswith("/api/show"):
            capabilities = ["completion", "tools"] if self.native_tools else ["completion"]
            return httpx.Response(200, json={"capabilities": capabilities})
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.reject_format and "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "invalid request"}})
        message = {"role": "assistant", "content": next(self.replies)}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    def client(self, spec="ollama/demo", **options):
        return create_client(spec, config={}, transport=httpx.MockTransport(self), **options)


async def test_response_format_follows_provider_override_and_native_tools(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
    local = Server(["7"])
    assert (await gole.Agent(local.client()).arun("n", output=int)).output == 7
    assert local.bodies[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": {"type": "integer"}},
    }
    hosted = Server(["7"])
    await gole.Agent(hosted.client("deepseek/demo")).arun("n", output=int)
    assert "response_format" not in hosted.bodies[0]
    forced = Server(["7"])
    await gole.Agent(forced.client("deepseek/demo", constrained=True)).arun("n", output=int)
    assert "response_format" in forced.bodies[0]
    native = Server(["7"], native_tools=True)
    await gole.Agent(native.client(), tools=[weather]).arun("n", output=int)
    assert "tools" in native.bodies[0] and "response_format" not in native.bodies[0]


async def test_rejected_response_format_is_resent_and_remembered():
    server = Server(["7", "8"], reject_format=True)
    agent = gole.Agent(server.client(), memory=None)
    first = await agent.arun("n", output=int)
    assert first.output == 7
    assert ["response_format" in body for body in server.bodies] == [True, False]
    errors = [e for e in first.trace.events if e.type == "error"]
    assert len(errors) == 1 and errors[0].recoverable
    assert [e.type for e in first.trace.events].count("model_end") == 1
    second = await agent.arun("n", output=int)
    assert second.output == 8 and "response_format" not in server.bodies[2]
    assert not [e for e in second.trace.events if e.type == "error"]


async def test_constrained_prompt_protocol_selects_a_tool_then_answers():
    server = Server(
        [
            '{"thought": "need the weather", "tool": "weather", "arguments": {"city": "Paris"}}',
            '{"thought": "enough", "tool": "final_answer"}',
            "Paris is 18°C.",
        ]
    )
    agent = gole.Agent(server.client(), instructions="Be brief.", tools=[weather])
    result = await agent.arun("Weather in Paris?")
    assert result.text == "Paris is 18°C."
    select, again, answer = server.bodies
    options = select["response_format"]["json_schema"]["schema"]["anyOf"]
    assert [option["properties"]["tool"]["enum"] for option in options] == [
        ["weather"],
        ["final_answer"],
    ]
    assert options[0]["properties"]["arguments"]["required"] == ["city"]
    assert (
        "tools" not in select and "Reply with one JSON object" in select["messages"][0]["content"]
    )
    assert again["messages"][-1]["content"] == "Observation: weather: Paris 18°C"
    # The answer is written by a plain call that no longer carries the protocol.
    assert "response_format" not in answer and answer["messages"][0]["content"] == "Be brief."
    assert "Do not use the JSON format" in answer["messages"][-1]["content"]
    thoughts = [e.thought for e in result.trace.events if e.type == "step" and e.thought]
    assert thoughts == ["need the weather", "enough"]
    assert [step.action for step in result.steps] == ["model", "tool", "model", "model", "finish"]


async def test_constrained_protocol_retries_bad_replies_and_streams_only_the_answer():
    class Constrained(FakeModel):
        async def probe(self):
            return Capabilities(json_schema=True)

    model = Constrained(
        [
            '{"thought": "x", "tool": "forecast", "arguments": {}}',
            '{"tool": "final_answer"}',
            "Mild.",
        ]
    )
    agent = gole.Agent(model, tools=[weather])
    events = [event async for event in agent.astream("Weather?")]
    assert events[-1].result.text == "Mild."
    assert "".join(e.text for e in events if e.type == "token") == "Mild."
    retry = model.requests[1]["messages"][-1].content or ""
    assert "Unknown tool: 'forecast'" in retry and "one JSON object" in retry
